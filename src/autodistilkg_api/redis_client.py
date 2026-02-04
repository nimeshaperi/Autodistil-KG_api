"""
Redis connection and queue/channel names for pipeline WebSocket flow.
Uses the same Redis as the graph traverser when possible (REDIS_* env).
"""
import os
from typing import Optional

REDIS_QUEUE_KEY = "pipeline:queue"


def pipeline_run_channel(run_id: str) -> str:
    """Pub/sub channel for progress events for a given run_id."""
    return f"pipeline:run:{run_id}"


def get_redis_url() -> str:
    """Build Redis URL from REDIS_URL or REDIS_HOST/PORT/DB/PASSWORD."""
    url = os.environ.get("REDIS_URL")
    if url:
        return url
    host = os.environ.get("REDIS_HOST", "localhost")
    port = os.environ.get("REDIS_PORT", "6379")
    db = os.environ.get("REDIS_DB", "0")
    password = os.environ.get("REDIS_PASSWORD", "")
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"
