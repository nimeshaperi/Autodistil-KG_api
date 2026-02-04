# KG Pipeline API

FastAPI server that exposes **Autodistil-KG** pipelines over REST and **WebSocket**, so you can create and run pipelines (graph_traverser → chatml_converter → finetuner → evaluator) via HTTP and stream progress in real time.

## Setup

From the repo root, install with Poetry (Python 3.13 required; autodistil-kg is a path dependency):

```bash
cd Autodistil-KG_api
poetry install
```

This installs `autodistil-kg` with the **finetune** extra (unsloth, trl, transformers, datasets). For finetuning to work, you also need Python development headers:

- **Ubuntu/Debian**: `sudo apt install python3-dev` or `python3.13-dev`
- **Fedora**: `sudo dnf install python3-devel`

## Run

```bash
poetry run uvicorn autodistilkg_api.main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude "unsloth_compiled_cache/*"
```

> **Note**: The `--reload-exclude` flag is important! Without it, Unsloth creates cache files during training that trigger server reloads and break WebSocket connections.

Config paths in the pipeline JSON are resolved relative to a **workspace** directory (default: `Autodistil-KG_api/workspace`). To use data from the Autodistil-KG repo, set:

```bash
export KG_PIPELINE_WORKSPACE=/path/to/Code/Autodistil-KG
```

- **REST**: `http://localhost:8000/docs`
- **WebSocket**: `ws://localhost:8000/ws`

## Redis queue (WebSocket)

When Redis is available, WebSocket pipeline runs are **enqueued** and a background worker runs the pipeline and publishes progress over Redis pub/sub. This keeps long runs off the request thread and reuses the same Redis already used by the graph traverser.

- **Env**: `REDIS_URL` or `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` / `REDIS_PASSWORD` (same as Autodistil-KG). Default: `localhost:6379`.
- If Redis is not reachable at startup, WebSocket runs fall back to **in-process** (same behaviour as before).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/pipelines/run` | Run a pipeline with JSON config (same shape as Autodistil-KG `config/*.json`). Returns run result or `run_id` if `?async=1`. |
| GET | `/pipelines/runs/{run_id}` | Get status/result of an async run |
| WebSocket | `/ws` | Send `{"action": "run", "config": {...}}` to run a pipeline and receive `stage_start` / `stage_end` / `done` events |

## WebSocket events (server → client)

- `{"event": "run_start", "run_id": "uuid"}` — Pipeline run started
- `{"event": "pipeline_start", "stages": ["stage1", "stage2"]}` — Stages list
- `{"event": "stage_start", "stage": "chatml_converter"}` — Stage started
- `{"event": "stage_end", "stage": "chatml_converter", "success": true, "metadata": {...}}` — Stage completed
- `{"event": "log", "level": "INFO", "logger": "pipeline", "message": "..."}` — Log message from pipeline (training progress, etc.)
- `{"event": "done", "success": true, "context": {...}, "results": [...]}` — Pipeline completed
- `{"event": "error", "message": "..."}` — Error occurred
