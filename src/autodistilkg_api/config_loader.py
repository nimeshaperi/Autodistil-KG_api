"""
Build PipelineConfig from a JSON-like dict (e.g. request body).
Paths are resolved relative to base_dir (e.g. API workspace or temp dir).
"""
from pathlib import Path
from typing import Any, Dict, Optional

from autodistil_kg.pipeline.config import (
    PipelineConfig,
    ChatMLConverterStageConfig,
    FineTunerStageConfig,
    EvaluatorStageConfig,
    GraphTraverserStageConfig,
)
from autodistil_kg.pipeline.interfaces import PipelineContext


def _resolve_path(path: Optional[str], base: Path) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def _graph_db_from_dict(data: dict) -> "GraphDatabaseConfig":
    """Build GraphDatabaseConfig from request payload (e.g. graph_traverser.neo4j)."""
    from autodistil_kg.graph_traverser.graph_db.config import GraphDatabaseConfig
    return GraphDatabaseConfig(
        provider="neo4j",
        uri=data.get("uri") or "bolt://localhost:7687",
        user=data.get("username") or data.get("user") or "neo4j",
        password=data.get("password") or "",
        database=data.get("database") or None,
    )


def _state_storage_from_dict(data: dict) -> "StateStorageConfig":
    """Build StateStorageConfig from request payload (e.g. graph_traverser.redis)."""
    from autodistil_kg.graph_traverser.state_storage.config import StateStorageConfig
    port = data.get("port", 6379)
    if isinstance(port, str):
        port = int(port, 10)
    db = data.get("db", 0)
    if isinstance(db, str):
        db = int(db, 10)
    return StateStorageConfig(
        provider="redis",
        host=data.get("host", "localhost"),
        port=port,
        db=db,
        password=data.get("password") or None,
        key_prefix=data.get("key_prefix", "graph_traverser:"),
    )


def _llm_config_from_dict(data: dict) -> "LLMConfig":
    """Build LLMConfig from request payload (e.g. graph_traverser.llm)."""
    from autodistil_kg.graph_traverser.llm.config import LLMConfig
    return LLMConfig(
        provider=(data.get("provider") or "openai").lower(),
        api_key=data.get("api_key") or None,
        model=data.get("model") or None,
        base_url=data.get("base_url") or None,
        project_id=data.get("project_id") or None,
        location=data.get("location") or None,
        credentials_path=data.get("credentials_path") or None,
        additional_params=data.get("additional_params") or {},
    )


def _parse_graph_traverser_config(gt_data: dict, base: Path) -> GraphTraverserStageConfig:
    """Build GraphTraverserStageConfig from dict. Uses credentials from request when provided, else .env."""
    from pathlib import Path
    try:
        from autodistil_kg.graph_traverser.env_config import (
            load_env_file,
            get_graph_db_config_from_env,
            get_llm_config_from_env,
            get_state_storage_config_from_env,
        )
        from autodistil_kg.graph_traverser.config import TraversalConfig, DatasetGenerationConfig, TraversalStrategy
    except ImportError as e:
        raise ImportError(
            "Graph traverser requires python-dotenv and LLM SDKs. "
            "Install in the Autodistil-KG environment."
        ) from e

    # Optional: load .env for fallback when request does not provide credentials
    env_loaded = False
    env_path = base / ".env"
    if env_path.exists():
        load_env_file(str(env_path))
        env_loaded = True
    else:
        for parent in [base, Path.cwd(), Path(__file__).resolve().parents[2]]:
            ep = parent / ".env"
            if ep.exists():
                load_env_file(str(ep))
                env_loaded = True
                break

    # Graph DB: from request neo4j dict or env
    if gt_data.get("neo4j") and isinstance(gt_data["neo4j"], dict):
        graph_db = _graph_db_from_dict(gt_data["neo4j"])
    elif env_loaded:
        graph_db = get_graph_db_config_from_env()
    else:
        raise ValueError(
            "Graph traverser needs Neo4j credentials. Either set graph_traverser.neo4j in the request "
            "(uri, username, password, database) or provide a .env with NEO4J_*."
        )

    # State storage (Redis): from request redis dict or env
    if gt_data.get("redis") and isinstance(gt_data["redis"], dict):
        state_storage = _state_storage_from_dict(gt_data["redis"])
    elif env_loaded:
        state_storage = get_state_storage_config_from_env()
    else:
        raise ValueError(
            "Graph traverser needs Redis credentials. Either set graph_traverser.redis in the request "
            "(host, port, db, password, key_prefix) or provide a .env with REDIS_*."
        )

    # LLM: from request llm dict or env
    if gt_data.get("llm") and gt_data["llm"].get("provider"):
        llm_config = _llm_config_from_dict(gt_data["llm"])
    elif env_loaded:
        llm_config = get_llm_config_from_env(provider=gt_data.get("llm_provider"))
    else:
        raise ValueError(
            "Graph traverser needs LLM credentials. Either set graph_traverser.llm in the request "
            "(provider, api_key, model, etc.) or provide a .env with OPENAI_API_KEY / GEMINI_* / CLAUDE_* / etc."
        )

    t = gt_data.get("traversal") or {}
    strategy_str = (t.get("strategy") or "bfs").lower()
    try:
        strategy = TraversalStrategy(strategy_str)
    except ValueError:
        strategy = TraversalStrategy.BFS
    traversal = TraversalConfig(
        strategy=strategy,
        max_nodes=t.get("max_nodes", 500),
        max_depth=t.get("max_depth", 5),
        relationship_types=t.get("relationship_types"),
        node_labels=t.get("node_labels"),
        seed_node_ids=t.get("seed_node_ids"),
    )

    d = gt_data.get("dataset") or {}
    output_path = _resolve_path(gt_data.get("output_path") or d.get("output_path"), base)
    dataset = DatasetGenerationConfig(
        seed_prompts=d.get("seed_prompts", ["What can you tell me about this node? Describe: {properties}"]),
        system_message=d.get("system_message"),
        prompt_template=d.get("prompt_template"),
        include_metadata=d.get("include_metadata", True),
        output_format=d.get("output_format", "jsonl"),
        output_path=output_path,
    )

    return GraphTraverserStageConfig(
        graph_db=graph_db,
        llm_config=llm_config,
        state_storage=state_storage,
        traversal=traversal,
        dataset=dataset,
        output_path=output_path,
    )


def config_from_dict(data: Dict[str, Any], base_dir: Path) -> PipelineConfig:
    """Build PipelineConfig from a dict (e.g. POST body). Paths resolved relative to base_dir."""
    def _cc(d):
        if not d:
            return None
        return ChatMLConverterStageConfig(
            input_path=_resolve_path(d.get("input_path"), base_dir),
            output_path=_resolve_path(d.get("output_path"), base_dir),
            prepare_for_finetuning=d.get("prepare_for_finetuning", True),
            chat_template=d.get("chat_template"),
        )

    def _ft(d):
        if not d:
            return None
        return FineTunerStageConfig(
            model_name=d.get("model_name", "unsloth/gemma-3-270m-it"),
            model_type=d.get("model_type"),
            train_data_path=_resolve_path(d.get("train_data_path"), base_dir),
            eval_data_path=_resolve_path(d.get("eval_data_path"), base_dir),
            output_dir=_resolve_path(d.get("output_dir"), base_dir),
            max_seq_length=d.get("max_seq_length", 2048),
            num_train_epochs=d.get("num_train_epochs", 1),
            per_device_train_batch_size=d.get("per_device_train_batch_size", 2),
            learning_rate=d.get("learning_rate", 2e-4),
        )

    ev = data.get("evaluator")
    evaluator = None
    if ev:
        evaluator = EvaluatorStageConfig(
            model_path=_resolve_path(ev.get("model_path"), base_dir),
            eval_dataset_path=_resolve_path(ev.get("eval_dataset_path"), base_dir),
            output_report_path=_resolve_path(ev.get("output_report_path"), base_dir),
            metrics=ev.get("metrics"),
            additional_params=ev.get("additional_params", {}),
        )

    gt_data = data.get("graph_traverser")
    graph_traverser = None
    if gt_data:
        graph_traverser = _parse_graph_traverser_config(gt_data, base_dir)

    return PipelineConfig(
        graph_traverser=graph_traverser,
        chatml_converter=_cc(data.get("chatml_converter")),
        finetuner=_ft(data.get("finetuner")),
        evaluator=evaluator,
        output_dir=_resolve_path(data.get("output_dir"), base_dir) if data.get("output_dir") else None,
        run_stages=data.get("run_stages"),
    )


def context_from_config(config: PipelineConfig) -> PipelineContext:
    """Build initial PipelineContext from config."""
    chatml_path = None
    prepared_path = None
    if config.chatml_converter:
        chatml_path = config.chatml_converter.input_path
        prepared_path = config.chatml_converter.output_path
    if config.finetuner:
        prepared_path = config.finetuner.train_data_path or prepared_path
    return PipelineContext(
        chatml_dataset_path=chatml_path,
        prepared_dataset_path=prepared_path,
    )
