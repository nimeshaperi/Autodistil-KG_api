"""
FastAPI app: REST endpoints and WebSocket for Autodistil-KG pipelines.
WebSocket runs use a Redis queue + pub/sub when Redis is available (same dependency as graph traverser).
"""
# =============================================================================
# CRITICAL: Disable unsloth/torch.compile BEFORE any other imports
# This must be at the very top to prevent PyTorch compatibility issues
# (cuda.cutlass_epilogue_fusion_enabled was removed in newer PyTorch versions)
# =============================================================================
import os
import shutil
from pathlib import Path

# Force disable torch.compile in unsloth (use = not setdefault to override)
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Clear any stale unsloth compiled cache that may contain incompatible torch.compile options
_cache_dirs_to_clear = [
    Path("/tmp/unsloth_compiled_cache"),
    Path.cwd() / "unsloth_compiled_cache",
    Path(__file__).parent.parent.parent / "unsloth_compiled_cache",
]
for _cache_dir in _cache_dirs_to_clear:
    if _cache_dir.exists():
        try:
            shutil.rmtree(_cache_dir)
        except Exception:
            pass

# Patch torch's _TorchCompileInductorWrapper to silently ignore unknown options
# This is needed because unsloth's cached compiled files may reference options
# that were removed in newer PyTorch versions (e.g., cuda.cutlass_epilogue_fusion_enabled)
def _patch_torch_compile_wrapper():
    """Patch _TorchCompileInductorWrapper.apply_options to skip unknown options instead of raising."""
    try:
        import torch
        wrapper_class = getattr(torch, "_TorchCompileInductorWrapper", None)
        if wrapper_class is None:
            return
        
        _original_apply_options = wrapper_class.apply_options
        
        def _patched_apply_options(self, options):
            if not options:
                return
            # Get known options from current config
            from torch._inductor import config as inductor_config
            
            def get_known_keys(cfg, prefix=""):
                keys = set()
                for key in dir(cfg):
                    if key.startswith("_"):
                        continue
                    full_key = f"{prefix}{key}" if prefix else key
                    val = getattr(cfg, key, None)
                    if hasattr(val, "__dict__") and not callable(val) and not isinstance(val, (str, int, float, bool, list, dict, type(None))):
                        # It's a nested config section
                        keys.update(get_known_keys(val, f"{full_key}."))
                    else:
                        keys.add(full_key)
                return keys
            
            try:
                known_keys = get_known_keys(inductor_config)
            except Exception:
                known_keys = set()
            
            # Filter options to only include known keys
            filtered_options = {k: v for k, v in options.items() if k in known_keys}
            
            if filtered_options:
                try:
                    _original_apply_options(self, filtered_options)
                except RuntimeError:
                    # If it still fails, just skip
                    pass
        
        wrapper_class.apply_options = _patched_apply_options
    except Exception:
        pass

_patch_torch_compile_wrapper()

import asyncio
import concurrent.futures
import json
import logging
import queue as import_queue
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import redis.asyncio as aioredis
import redis

from autodistil_kg.pipeline import Pipeline
from autodistil_kg.pipeline.interfaces import PipelineContext, StageResult

from .config_loader import config_from_dict, context_from_config
from .redis_client import REDIS_QUEUE_KEY, get_redis_url, pipeline_run_channel


def _prepare_run_dir(run_id: str, config_dict: Dict[str, Any]) -> Path:
    """Create a per-run workspace directory and inject run_id into config for isolation.

    Returns the run-specific base directory.  All relative output paths will
    resolve under ``WORKSPACE/runs/<run_id>/`` so concurrent runs never collide.
    The Redis key_prefix for graph traversal is also made run-specific.
    """
    run_dir = WORKSPACE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Inject run-scoped Redis key prefix so visited-node sets don't collide
    gt = config_dict.get("graph_traverser")
    if gt and isinstance(gt, dict):
        redis_cfg = gt.setdefault("redis", {})
        if isinstance(redis_cfg, dict) and not redis_cfg.get("key_prefix"):
            redis_cfg["key_prefix"] = f"run:{run_id}:gt:"

    return run_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

async def _lifespan(app: FastAPI):
    global _redis_available
    try:
        await asyncio.to_thread(
            lambda: redis.from_url(get_redis_url(), decode_responses=True).ping()
        )
        _redis_available = True
        t = threading.Thread(target=_pipeline_worker_loop, daemon=True)
        t.start()
        logger.info("Redis pipeline queue enabled")
    except Exception as e:
        logger.info("Redis not used for pipeline queue: %s. WebSocket runs in-process.", e)
    yield
    _worker_stop.set()


app = FastAPI(
    title="KG Pipeline API",
    description="Create and run Autodistil-KG pipelines via REST and WebSocket",
    version="0.1.0",
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory for resolving config paths. Override with KG_PIPELINE_WORKSPACE (e.g. path to Autodistil-KG).
WORKSPACE = Path(__file__).resolve().parents[2] / "workspace"
if os.environ.get("KG_PIPELINE_WORKSPACE"):
    WORKSPACE = Path(os.environ["KG_PIPELINE_WORKSPACE"]).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Thread pool for running blocking pipeline stages
_executor: Optional[concurrent.futures.Executor] = None


def get_executor() -> concurrent.futures.Executor:
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    return _executor

# In-memory store for async run status (run_id -> { status, context, results, error })
_run_store: Dict[str, Dict[str, Any]] = {}
# Ordered stage names for sequential run
STAGE_ORDER = ("graph_traverser", "chatml_converter", "finetuner", "evaluator")

# Redis: when available, WebSocket runs are enqueued and a worker publishes progress to pub/sub
_redis_available = False
_worker_stop = threading.Event()


class QueueLogHandler(logging.Handler):
    """Log handler that pushes structured events to a thread-safe queue for in-process WebSocket runs."""

    def __init__(self, queue: "import_queue.Queue", level=logging.INFO):
        super().__init__(level)
        self.queue = queue
        self.allowed_loggers = {
            "autodistil_kg", "unsloth", "trl", "transformers", "datasets",
            "unsloth.trainer", "autodistil_kg.finetuner", "autodistil_kg.pipeline",
        }

    def emit(self, record: logging.LogRecord):
        try:
            logger_name = record.name
            is_allowed = any(logger_name.startswith(prefix) for prefix in self.allowed_loggers)
            if not is_allowed:
                return

            traversal_event = getattr(record, "traversal_event", None)
            if traversal_event and isinstance(traversal_event, dict):
                self.queue.put({"event": "traversal_progress", **traversal_event})
                return

            msg = self.format(record)
            if len(msg) > 500:
                msg = msg[:497] + "..."
            self.queue.put({
                "event": "log",
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
            })
        except Exception:
            pass


class RedisLogHandler(logging.Handler):
    """Custom logging handler that publishes log messages to a Redis channel."""
    
    def __init__(self, redis_client, channel: str, level=logging.INFO):
        super().__init__(level)
        self.redis_client = redis_client
        self.channel = channel
        # Only capture logs from pipeline-related modules
        self.allowed_loggers = {
            "autodistil_kg", "unsloth", "trl", "transformers", "datasets",
            "unsloth.trainer", "autodistil_kg.finetuner", "autodistil_kg.pipeline",
        }
    
    def emit(self, record: logging.LogRecord):
        try:
            # Filter to only relevant loggers
            logger_name = record.name
            is_allowed = any(logger_name.startswith(prefix) for prefix in self.allowed_loggers)
            if not is_allowed:
                return

            # Check for structured traversal events (emitted by GraphTraverserAgent)
            traversal_event = getattr(record, "traversal_event", None)
            if traversal_event and isinstance(traversal_event, dict):
                self.redis_client.publish(self.channel, json.dumps({
                    "event": "traversal_progress",
                    **traversal_event,
                }))
                return

            msg = self.format(record)
            # Skip very long messages (e.g., progress bars with many characters)
            if len(msg) > 500:
                msg = msg[:497] + "..."

            self.redis_client.publish(self.channel, json.dumps({
                "event": "log",
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
            }))
        except Exception:
            pass  # Don't let logging errors break the pipeline


def _pipeline_worker_loop() -> None:
    """Background thread: BRPOP from Redis queue, run pipeline, publish events to pipeline:run:{run_id}."""
    global _run_store
    url = get_redis_url()
    try:
        r = redis.from_url(url, decode_responses=True)
        r.ping()
    except Exception as e:
        logger.warning("Redis not available for pipeline queue: %s. WebSocket runs will be in-process.", e)
        return
    logger.info("Pipeline queue worker started (Redis queue: %s)", REDIS_QUEUE_KEY)
    while not _worker_stop.is_set():
        try:
            result = r.brpop(REDIS_QUEUE_KEY, timeout=2)
            if result is None:
                continue
            _, raw = result
            job = json.loads(raw)
            run_id = job["run_id"]
            config_dict = job["config"]
            run_dir = _prepare_run_dir(run_id, config_dict)
            logger.info("Pipeline worker picked up job run_id=%s stages=%s dir=%s", run_id, config_dict.get("run_stages"), run_dir)
            _run_store[run_id] = {"status": "running", "context": None, "results": None, "error": None, "stages": [], "current_stage": None, "events": []}
            channel = pipeline_run_channel(run_id)

            # Set up log streaming to Redis for this run
            log_handler = RedisLogHandler(r, channel)
            log_handler.setFormatter(logging.Formatter("%(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)

            def _store_event(evt: Dict[str, Any]) -> None:
                """Store event in run store for replay and publish to Redis."""
                _run_store[run_id].setdefault("events", []).append(evt)
                r.publish(channel, json.dumps(evt))

            try:
                config = config_from_dict(config_dict, run_dir)
                pipeline = Pipeline(config)
                context = context_from_config(config)
                run_order = config.run_stages or list(pipeline.available_stages)
                ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
                _run_store[run_id]["stages"] = ordered
                _store_event({"event": "pipeline_start", "stages": ordered})
                results: List[Dict[str, Any]] = []
                for name in ordered:
                    _run_store[run_id]["current_stage"] = name
                    _store_event({"event": "stage_start", "stage": name})
                    _store_event({"event": "log", "level": "INFO", "logger": "pipeline", "message": f"Starting stage: {name}"})
                    try:
                        result = pipeline.run_stage(name, context)
                        results.append({"success": result.success, "error": result.error, "metadata": result.metadata or {}})
                        _store_event({
                            "event": "stage_end",
                            "stage": name,
                            "success": result.success,
                            "error": result.error,
                            "metadata": result.metadata or {},
                        })
                        if result.success:
                            _store_event({"event": "log", "level": "INFO", "logger": "pipeline", "message": f"Stage {name} completed successfully"})
                        else:
                            _store_event({"event": "log", "level": "ERROR", "logger": "pipeline", "message": f"Stage {name} failed: {result.error}"})
                        if not result.success:
                            break
                    except Exception as e:
                        logger.exception("Stage %s failed", name)
                        results.append({"success": False, "error": str(e), "metadata": {}})
                        _store_event({
                            "event": "stage_end",
                            "stage": name,
                            "success": False,
                            "error": str(e),
                            "metadata": {},
                        })
                        _store_event({"event": "log", "level": "ERROR", "logger": "pipeline", "message": f"Stage {name} exception: {str(e)}"})
                        break
                success = all(x["success"] for x in results)
                _store_event({
                    "event": "done",
                    "success": success,
                    "context": context.to_dict(),
                    "results": results,
                })
                _run_store[run_id]["status"] = "completed" if success else "failed"
                _run_store[run_id]["context"] = context.to_dict()
                _run_store[run_id]["results"] = results
                _run_store[run_id]["error"] = next((x.get("error") for x in results if not x.get("success")), None)
            except Exception as e:
                logger.exception("Pipeline run failed for run_id=%s", run_id)
                _run_store[run_id]["status"] = "failed"
                _run_store[run_id]["error"] = str(e)
                r.publish(channel, json.dumps({"event": "error", "message": str(e)}))
            finally:
                # Remove the log handler to avoid duplicate logs in future runs
                root_logger.removeHandler(log_handler)
        except redis.ConnectionError:
            if not _worker_stop.is_set():
                logger.warning("Pipeline worker Redis connection lost; reconnecting on next iteration")
        except Exception as e:
            if not _worker_stop.is_set():
                logger.exception("Pipeline worker error: %s", e)
    logger.info("Pipeline queue worker stopped")


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    message: Optional[str] = None


class PipelineRunResultResponse(BaseModel):
    run_id: str
    status: str
    success: bool
    context: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    stages: Optional[List[str]] = None  # ordered stage names for UI
    current_stage: Optional[str] = None  # e.g. "graph_traverser" while running


def _run_pipeline_sync(
    config_dict: Dict[str, Any],
    base_dir: Path,
    run_id: Optional[str] = None,
) -> tuple[PipelineContext, list[StageResult]]:
    """Build config, pipeline, and run to completion. Returns (context, results)."""
    config = config_from_dict(config_dict, base_dir)
    pipeline = Pipeline(config)
    context = context_from_config(config)
    run_order = config.run_stages or list(pipeline.available_stages)
    ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
    results: List[StageResult] = []
    for name in ordered:
        if run_id and run_id in _run_store:
            _run_store[run_id]["current_stage"] = name
            _run_store[run_id]["events"].append({"event": "stage_start", "stage": name})
        result = pipeline.run_stage(name, context)
        results.append(result)
        if run_id and run_id in _run_store:
            _run_store[run_id]["events"].append({
                "event": "stage_end",
                "stage": name,
                "success": result.success,
                "error": result.error,
                "metadata": result.metadata or {},
            })
        if not result.success:
            break
    return context, results


@app.get("/health")
def health():
    return {"status": "ok", "service": "kg-pipeline-api"}


@app.get("/stages")
def get_stages():
    """Return pipeline stage definitions, config schemas, and available options.

    The UI uses this to render forms dynamically instead of hardcoding
    strategies, providers, metrics, and model lists.
    """
    return {
        "stage_order": list(STAGE_ORDER),
        "stages": {
            "graph_traverser": {
                "label": "Graph Traverser",
                "description": "Traverse a Neo4j knowledge graph and generate ChatML conversation datasets.",
                "config": {
                    "traversal": {
                        "strategies": [
                            {"value": "bfs", "label": "Breadth-First Search", "description": "Explore graph layer by layer from seed nodes."},
                            {"value": "dfs", "label": "Depth-First Search", "description": "Explore graph depth-first along each branch."},
                            {"value": "random", "label": "Random Walk", "description": "Randomly select neighbours at each step."},
                            {"value": "semantic", "label": "Semantic (LLM-guided)", "description": "LLM selects the most relevant neighbour at each step based on context."},
                            {"value": "reasoning", "label": "Reasoning (multi-hop)", "description": "Deep multi-hop reasoning with subgraph exploration and path analysis."},
                        ],
                        "fields": {
                            "max_nodes": {"type": "number", "default": 500, "label": "Max Nodes"},
                            "max_depth": {"type": "number", "default": 5, "label": "Max Depth"},
                            "reasoning_depth": {"type": "number", "default": 2, "label": "Reasoning Depth", "description": "Subgraph depth for REASONING strategy.", "show_when_strategy": ["reasoning"]},
                            "max_paths_per_node": {"type": "number", "default": 15, "label": "Max Paths per Node", "description": "Max paths to reason over per node.", "show_when_strategy": ["reasoning"]},
                            "relationship_types": {"type": "string[]", "default": None, "label": "Relationship Types", "description": "Filter by relationship types (comma-separated)."},
                            "node_labels": {"type": "string[]", "default": None, "label": "Node Labels", "description": "Filter by node labels (comma-separated)."},
                            "seed_node_ids": {"type": "string[]", "default": None, "label": "Seed Node IDs", "description": "Starting node IDs (comma-separated)."},
                        },
                    },
                    "llm_providers": [
                        {"value": "openai", "label": "OpenAI", "fields": ["api_key", "model", "base_url"]},
                        {"value": "claude", "label": "Claude (Anthropic)", "fields": ["api_key", "model"]},
                        {"value": "gemini", "label": "Gemini (Google)", "fields": ["project_id", "location", "model", "credentials_path"]},
                        {"value": "ollama", "label": "Ollama (local)", "fields": ["base_url", "model"]},
                        {"value": "vllm", "label": "vLLM (local)", "fields": ["base_url", "model"]},
                    ],
                },
            },
            "chatml_converter": {
                "label": "ChatML Converter",
                "description": "Normalize and prepare ChatML datasets for fine-tuning.",
                "config": {
                    "fields": {
                        "input_path": {"type": "string", "default": "output/dataset.jsonl", "label": "Input Path"},
                        "output_path": {"type": "string", "default": "output/prepared.jsonl", "label": "Output Path"},
                        "prepare_for_finetuning": {"type": "boolean", "default": True, "label": "Prepare for Fine-tuning"},
                        "chat_template": {"type": "string", "default": "auto", "label": "Chat Template"},
                    },
                },
            },
            "finetuner": {
                "label": "FineTuner",
                "description": "Fine-tune language models using Unsloth with LoRA adapters.",
                "config": {
                    "model_types": [
                        {"value": "gemma3", "label": "Gemma 3"},
                        {"value": "llama3", "label": "Llama 3"},
                        {"value": "qwen2", "label": "Qwen 2"},
                        {"value": "qwen3", "label": "Qwen 3"},
                        {"value": "mistral", "label": "Mistral"},
                        {"value": "phi3", "label": "Phi 3"},
                        {"value": "phi4", "label": "Phi 4"},
                    ],
                    "suggested_models": [
                        {"value": "unsloth/gemma-3-270m-it", "label": "Gemma 3 270M (instruction-tuned)", "type": "gemma3"},
                        {"value": "unsloth/gemma-3-1b-it", "label": "Gemma 3 1B (instruction-tuned)", "type": "gemma3"},
                        {"value": "unsloth/gemma-3-4b-it", "label": "Gemma 3 4B (instruction-tuned)", "type": "gemma3"},
                    ],
                    "fields": {
                        "model_name": {"type": "string", "label": "Model Name"},
                        "model_type": {"type": "string", "label": "Model Type"},
                        "train_data_path": {"type": "string", "default": "output/prepared.jsonl", "label": "Train Data Path"},
                        "output_dir": {"type": "string", "default": "output/finetuned", "label": "Output Directory"},
                        "max_seq_length": {"type": "number", "default": 2048, "label": "Max Sequence Length"},
                        "num_train_epochs": {"type": "number", "default": 1, "label": "Epochs"},
                        "per_device_train_batch_size": {"type": "number", "default": 2, "label": "Batch Size"},
                        "learning_rate": {"type": "number", "default": 2e-4, "label": "Learning Rate"},
                    },
                },
            },
            "evaluator": {
                "label": "Evaluator",
                "description": "Compare finetuned model against base models and Graph RAG using configurable metrics.",
                "config": {
                    "modes": [
                        {"value": "internal", "label": "Internal (in-process)", "description": "Run evaluation in the API process using ROUGE and LLM judge."},
                        {"value": "cli", "label": "CLI (external command)", "description": "Invoke an external EvalG CLI command."},
                        {"value": "noop", "label": "No-op (stub report)", "description": "Emit a stub report without running actual evaluation."},
                    ],
                    "metrics": [
                        {"value": "rouge", "label": "ROUGE (1/2/L)", "description": "Lexical overlap metrics comparing prediction against reference."},
                        {"value": "llm_judge", "label": "LLM Judge", "description": "LLM rates predictions on accuracy, completeness, and relevance (1-5)."},
                    ],
                    "system_kinds": [
                        {"value": "distilled", "label": "Finetuned Model", "description": "The LoRA-adapted model from the finetuner stage."},
                        {"value": "base", "label": "Base Model", "description": "A non-finetuned model via LLM provider for comparison."},
                        {"value": "graph_rag", "label": "Graph RAG", "description": "A Graph RAG pipeline querying the knowledge graph directly."},
                        {"value": "external", "label": "External Model", "description": "Any external LLM API for additional comparison."},
                    ],
                    "llm_providers": [
                        {"value": "openai", "label": "OpenAI"},
                        {"value": "claude", "label": "Claude"},
                        {"value": "gemini", "label": "Gemini"},
                        {"value": "ollama", "label": "Ollama"},
                        {"value": "vllm", "label": "vLLM"},
                    ],
                    "fields": {
                        "eval_dataset_path": {"type": "string", "default": "output/prepared.jsonl", "label": "Eval Dataset Path"},
                        "output_report_path": {"type": "string", "default": "output/eval_report.json", "label": "Output Report Path"},
                        "model_path": {"type": "string", "label": "Finetuned Model Path (optional)"},
                        "max_eval_samples": {"type": "number", "label": "Max Eval Samples (optional)"},
                    },
                },
            },
        },
        "artifact_keys": list(_ARTIFACT_KEYS.keys()),
    }


@app.post("/pipelines/run", response_model=PipelineRunResultResponse)
def run_pipeline(
    body: Dict[str, Any],
    async_run: bool = Query(False, alias="async"),
):
    """
    Run a pipeline with the given JSON config (same shape as Autodistil-KG config files).
    Paths in config are relative to the API workspace directory.
    Use ?async=true to run in background and get run_id; then GET /pipelines/runs/{run_id} for result.
    """
    if async_run:
        run_id = str(uuid.uuid4())
        run_dir = _prepare_run_dir(run_id, body)
        try:
            config = config_from_dict(body, run_dir)
            pipeline = Pipeline(config)
            run_order = config.run_stages or list(pipeline.available_stages)
            ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
        except Exception:
            ordered = []
        _run_store[run_id] = {
            "status": "running",
            "context": None,
            "results": None,
            "error": None,
            "stages": ordered,
            "current_stage": ordered[0] if ordered else None,
            "events": [],
        }

        def task():
            try:
                _run_store[run_id]["events"].append({"event": "pipeline_start", "stages": ordered})
                context, results = _run_pipeline_sync(body, run_dir, run_id=run_id)
                _run_store[run_id]["status"] = "completed"
                _run_store[run_id]["current_stage"] = None
                _run_store[run_id]["context"] = context.to_dict()
                _run_store[run_id]["results"] = [
                    {"success": r.success, "error": r.error, "metadata": r.metadata} for r in results
                ]
                _run_store[run_id]["events"].append({"event": "done", "success": all(r.success for r in results)})
            except Exception as e:
                logger.exception("Async pipeline run failed")
                _run_store[run_id]["status"] = "failed"
                _run_store[run_id]["error"] = str(e)
                _run_store[run_id]["events"].append({"event": "error", "message": str(e)})

        threading.Thread(target=task, daemon=True).start()
        return PipelineRunResultResponse(run_id=run_id, status="running", success=True)

    try:
        sync_run_id = str(uuid.uuid4())
        sync_run_dir = _prepare_run_dir(sync_run_id, body)
        context, results = _run_pipeline_sync(body, sync_run_dir)
        success = all(r.success for r in results)
        return PipelineRunResultResponse(
            run_id="",
            status="completed",
            success=success,
            context=context.to_dict(),
            results=[{"success": r.success, "error": r.error, "metadata": r.metadata} for r in results],
            error=next((r.error for r in results if not r.success), None),
        )
    except Exception as e:
        logger.exception("Pipeline run failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipelines/runs")
def list_runs():
    """List all known pipeline runs (newest first)."""
    runs = []
    for run_id, rec in _run_store.items():
        runs.append({
            "run_id": run_id,
            "status": rec.get("status", "unknown"),
            "error": rec.get("error"),
            "stages": rec.get("stages"),
        })
    # Newest first (dict insertion order in Python 3.7+)
    runs.reverse()
    return {"runs": runs}


def _make_json_safe(obj: Any) -> Any:
    """Ensure dict is JSON-serializable (e.g. for context.extra)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


@app.get("/pipelines/runs/{run_id}", response_model=PipelineRunResultResponse)
def get_run_status(run_id: str):
    """Get status and result of an async pipeline run."""
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    rec = _run_store[run_id]
    try:
        context = rec.get("context")
        if context is not None:
            context = _make_json_safe(context)
        results = rec.get("results")
        stages = rec.get("stages")
        current = rec.get("current_stage")
        if not current and stages and rec["status"] == "running":
            current = stages[0] if stages else None
        return PipelineRunResultResponse(
            run_id=run_id,
            status=rec["status"],
            success=rec["status"] == "completed" and not any(
                r.get("success") is False for r in (results or [])
            ),
            context=context,
            results=results,
            error=rec.get("error"),
            stages=stages,
            current_stage=current,
        )
    except Exception as e:
        logger.exception("get_run_status failed for run_id=%s", run_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipelines/runs/{run_id}/events")
def get_run_events(run_id: str, since: int = Query(0, ge=0)):
    """Return stored events for a run (for replaying progress on historical runs).

    Use ``?since=N`` to get only events after index N (for incremental polling).
    """
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    events = _run_store[run_id].get("events", [])
    return {"run_id": run_id, "events": events[since:], "total": len(events)}


_ARTIFACT_KEYS = {
    "chatml": "chatml_dataset_path",
    "prepared": "prepared_dataset_path",
    "eval_report": "eval_report_path",
}


@app.get("/pipelines/runs/{run_id}/artifacts/{artifact_key}")
def get_run_artifact(run_id: str, artifact_key: str):
    """Download an output file from a run (e.g. chatml -> chatml_dataset_path, prepared -> prepared_dataset_path)."""
    if artifact_key not in _ARTIFACT_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown artifact: {artifact_key}. Use one of: {list(_ARTIFACT_KEYS)}")
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    rec = _run_store[run_id]
    context = rec.get("context") or {}
    path_key = _ARTIFACT_KEYS[artifact_key]
    raw_path = context.get(path_key)
    if not raw_path or not isinstance(raw_path, str):
        raise HTTPException(status_code=404, detail=f"No {path_key} for this run")
    file_path = Path(raw_path).resolve()
    workspace_resolved = WORKSPACE.resolve()
    try:
        file_path.relative_to(workspace_resolved)
    except ValueError:
        raise HTTPException(status_code=403, detail="Artifact path is outside workspace")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    filename = file_path.name
    return FileResponse(path=str(file_path), filename=filename, media_type="application/json")


async def _ws_run_via_redis(websocket: WebSocket, config_dict: Dict[str, Any]) -> None:
    """Enqueue job to Redis, subscribe to run_id channel, forward events to client until done/error."""
    run_id = str(uuid.uuid4())
    # Pre-compute stages so they're available immediately via REST polling
    try:
        _tmp_cfg = config_from_dict(config_dict, WORKSPACE)
        _tmp_pipe = Pipeline(_tmp_cfg)
        _tmp_order = _tmp_cfg.run_stages or list(_tmp_pipe.available_stages)
        pre_stages = [s for s in STAGE_ORDER if s in _tmp_order and s in _tmp_pipe.available_stages]
    except Exception:
        pre_stages = config_dict.get("run_stages", [])

    # Register in _run_store immediately so REST polling works before worker picks up the job
    _run_store[run_id] = {
        "status": "running",
        "context": None,
        "results": None,
        "error": None,
        "stages": pre_stages,
        "current_stage": pre_stages[0] if pre_stages else None,
        "events": [],
    }

    job = {
        "run_id": run_id,
        "config": config_dict,
    }
    red = aioredis.from_url(get_redis_url(), decode_responses=True)
    try:
        await red.lpush(REDIS_QUEUE_KEY, json.dumps(job))
        await websocket.send_json({"event": "run_start", "run_id": run_id})
        channel = pipeline_run_channel(run_id)
        pubsub = red.pubsub()
        await pubsub.subscribe(channel)
        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                if msg is None:
                    continue
                if msg["type"] != "message":
                    continue
                payload = json.loads(msg["data"])
                # Also store events in _run_store for REST replay
                if run_id in _run_store:
                    _run_store[run_id].setdefault("events", []).append(payload)
                await websocket.send_json(payload)
                if payload.get("event") in ("done", "error"):
                    break
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    finally:
        await red.aclose()


async def _ws_run_in_process(websocket: WebSocket, config_dict: Dict[str, Any]) -> None:
    """Run pipeline in-process and send events directly (no Redis)."""
    run_id = str(uuid.uuid4())
    _run_store[run_id] = {"status": "running", "context": None, "results": None, "error": None, "stages": [], "current_stage": None, "events": []}
    await websocket.send_json({"event": "run_start", "run_id": run_id})
    run_dir = _prepare_run_dir(run_id, config_dict)
    config = config_from_dict(config_dict, run_dir)
    pipeline = Pipeline(config)
    context = context_from_config(config)
    run_order = config.run_stages or list(pipeline.available_stages)
    ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
    _run_store[run_id]["stages"] = ordered
    results: List[StageResult] = []
    await websocket.send_json({"event": "pipeline_start", "stages": ordered})
    _run_store[run_id]["events"].append({"event": "pipeline_start", "stages": ordered})

    # Set up in-process log streaming via a thread-safe queue
    log_queue: import_queue.Queue = import_queue.Queue()
    log_handler = QueueLogHandler(log_queue)
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    async def _drain_log_queue():
        """Drain queued log/traversal_progress events and send to WebSocket."""
        while True:
            try:
                payload = log_queue.get_nowait()
                await websocket.send_json(payload)
            except import_queue.Empty:
                break

    loop = asyncio.get_event_loop()
    try:
        for name in ordered:
            _run_store[run_id]["current_stage"] = name
            await websocket.send_json({"event": "stage_start", "stage": name})
            _run_store[run_id]["events"].append({"event": "stage_start", "stage": name})
            try:
                result = await loop.run_in_executor(
                    get_executor(),
                    lambda n=name: pipeline.run_stage(n, context),
                )
                await _drain_log_queue()
                results.append(result)
                stage_end_evt = {
                    "event": "stage_end",
                    "stage": name,
                    "success": result.success,
                    "error": result.error,
                    "metadata": result.metadata or {},
                }
                await websocket.send_json(stage_end_evt)
                _run_store[run_id]["events"].append(stage_end_evt)
                if not result.success:
                    break
            except Exception as e:
                logger.exception("Stage %s failed", name)
                await _drain_log_queue()
                stage_end_evt = {
                    "event": "stage_end",
                    "stage": name,
                    "success": False,
                    "error": str(e),
                    "metadata": {},
                }
                await websocket.send_json(stage_end_evt)
                _run_store[run_id]["events"].append(stage_end_evt)
                results.append(StageResult(success=False, error=str(e)))
                break
    finally:
        root_logger.removeHandler(log_handler)
        await _drain_log_queue()

    success = all(r.success for r in results)
    _run_store[run_id]["status"] = "completed" if success else "failed"
    _run_store[run_id]["context"] = context.to_dict()
    _run_store[run_id]["results"] = [
        {"success": r.success, "error": r.error, "metadata": r.metadata or {}}
        for r in results
    ]
    _run_store[run_id]["error"] = next((r.error for r in results if not r.success), None)
    await websocket.send_json({
        "event": "done",
        "success": success,
        "context": context.to_dict(),
        "results": [
            {"success": r.success, "error": r.error, "metadata": r.metadata or {}}
            for r in results
        ],
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket: send {"action": "run", "config": {...}} to run a pipeline.
    When Redis is available, the run is enqueued and progress is streamed via Redis pub/sub.
    Otherwise the pipeline runs in-process and events are sent directly.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            if action != "run":
                await websocket.send_json({"event": "error", "message": f"Unknown action: {action}"})
                continue

            config_dict = data.get("config") or {}
            run_stages = config_dict.get("run_stages") or []
            logger.info("Pipeline run requested via WebSocket, stages=%s", run_stages)
            try:
                config_from_dict(config_dict, WORKSPACE)
            except Exception as e:
                logger.exception("Pipeline config validation failed")
                await websocket.send_json({"event": "error", "message": str(e)})
                continue

            if _redis_available:
                await _ws_run_via_redis(websocket, config_dict)
            else:
                await _ws_run_in_process(websocket, config_dict)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
