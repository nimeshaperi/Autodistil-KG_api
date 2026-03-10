"""
FastAPI app: REST endpoints and WebSocket for Autodistil-KG pipelines.
WebSocket runs use a Redis queue + pub/sub when Redis is available (same dependency as graph traverser).
"""
import asyncio
import concurrent.futures
import json
import logging
import os
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
            base_dir = Path(job["base_dir"])
            logger.info("Pipeline worker picked up job run_id=%s stages=%s", run_id, config_dict.get("run_stages"))
            _run_store[run_id] = {"status": "running", "context": None, "results": None, "error": None}
            channel = pipeline_run_channel(run_id)
            try:
                config = config_from_dict(config_dict, base_dir)
                pipeline = Pipeline(config)
                context = context_from_config(config)
                run_order = config.run_stages or list(pipeline.available_stages)
                ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
                r.publish(channel, json.dumps({"event": "pipeline_start", "stages": ordered}))
                results: List[Dict[str, Any]] = []
                for name in ordered:
                    r.publish(channel, json.dumps({"event": "stage_start", "stage": name}))
                    try:
                        result = pipeline.run_stage(name, context)
                        results.append({"success": result.success, "error": result.error, "metadata": result.metadata or {}})
                        r.publish(channel, json.dumps({
                            "event": "stage_end",
                            "stage": name,
                            "success": result.success,
                            "error": result.error,
                            "metadata": result.metadata or {},
                        }))
                        if not result.success:
                            break
                    except Exception as e:
                        logger.exception("Stage %s failed", name)
                        results.append({"success": False, "error": str(e), "metadata": {}})
                        r.publish(channel, json.dumps({
                            "event": "stage_end",
                            "stage": name,
                            "success": False,
                            "error": str(e),
                            "metadata": {},
                        }))
                        break
                success = all(x["success"] for x in results)
                r.publish(channel, json.dumps({
                    "event": "done",
                    "success": success,
                    "context": context.to_dict(),
                    "results": results,
                }))
                _run_store[run_id]["status"] = "completed" if success else "failed"
                _run_store[run_id]["context"] = context.to_dict()
                _run_store[run_id]["results"] = results
                _run_store[run_id]["error"] = next((x.get("error") for x in results if not x.get("success")), None)
            except Exception as e:
                logger.exception("Pipeline run failed for run_id=%s", run_id)
                _run_store[run_id]["status"] = "failed"
                _run_store[run_id]["error"] = str(e)
                r.publish(channel, json.dumps({"event": "error", "message": str(e)}))
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


def _run_pipeline_sync(config_dict: Dict[str, Any], base_dir: Path) -> tuple[PipelineContext, list[StageResult]]:
    """Build config, pipeline, and run to completion. Returns (context, results)."""
    config = config_from_dict(config_dict, base_dir)
    pipeline = Pipeline(config)
    context = context_from_config(config)
    run_order = config.run_stages or list(pipeline.available_stages)
    ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
    results: List[StageResult] = []
    for name in ordered:
        result = pipeline.run_stage(name, context)
        results.append(result)
        if not result.success:
            break
    return context, results


@app.get("/health")
def health():
    return {"status": "ok", "service": "kg-pipeline-api"}


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
    base_dir = WORKSPACE
    if async_run:
        run_id = str(uuid.uuid4())
        try:
            config = config_from_dict(body, base_dir)
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
        }

        def task():
            try:
                context, results = _run_pipeline_sync(body, base_dir)
                _run_store[run_id]["status"] = "completed"
                _run_store[run_id]["context"] = context.to_dict()
                _run_store[run_id]["results"] = [
                    {"success": r.success, "error": r.error, "metadata": r.metadata} for r in results
                ]
            except Exception as e:
                logger.exception("Async pipeline run failed")
                _run_store[run_id]["status"] = "failed"
                _run_store[run_id]["error"] = str(e)

        threading.Thread(target=task, daemon=True).start()
        return PipelineRunResultResponse(run_id=run_id, status="running", success=True)

    try:
        context, results = _run_pipeline_sync(body, base_dir)
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


_ARTIFACT_KEYS = {
    "chatml": "chatml_dataset_path",
    "prepared": "prepared_dataset_path",
    "eval_report": "eval_report_path",
}


@app.get("/pipelines/runs/{run_id}/artifacts/{artifact_key}")
def get_run_artifact(run_id: str, artifact_key: str):
    """Download an output file from a run (e.g. chatml -> chatml_dataset_path, prepared -> prepared_dataset_path, eval_report -> eval_report_path)."""
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
    job = {
        "run_id": run_id,
        "config": config_dict,
        "base_dir": str(WORKSPACE),
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
    _run_store[run_id] = {"status": "running", "context": None, "results": None, "error": None}
    await websocket.send_json({"event": "run_start", "run_id": run_id})
    base_dir = WORKSPACE
    config = config_from_dict(config_dict, base_dir)
    pipeline = Pipeline(config)
    context = context_from_config(config)
    run_order = config.run_stages or list(pipeline.available_stages)
    ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
    results: List[StageResult] = []
    await websocket.send_json({"event": "pipeline_start", "stages": ordered})
    loop = asyncio.get_event_loop()
    for name in ordered:
        await websocket.send_json({"event": "stage_start", "stage": name})
        try:
            result = await loop.run_in_executor(
                get_executor(),
                lambda n=name: pipeline.run_stage(n, context),
            )
            results.append(result)
            await websocket.send_json({
                "event": "stage_end",
                "stage": name,
                "success": result.success,
                "error": result.error,
                "metadata": result.metadata or {},
            })
            if not result.success:
                break
        except Exception as e:
            logger.exception("Stage %s failed", name)
            await websocket.send_json({
                "event": "stage_end",
                "stage": name,
                "success": False,
                "error": str(e),
                "metadata": {},
            })
            results.append(StageResult(success=False, error=str(e)))
            break
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
