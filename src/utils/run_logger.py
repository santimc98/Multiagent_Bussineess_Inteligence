import json
import os
from datetime import datetime
from typing import Any, Dict


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _log_path(run_id: str, log_dir: str = "logs") -> str:
    return os.path.join(log_dir, f"run_{run_id}.jsonl")


def init_run_log(run_id: str, metadata: Dict[str, Any], log_dir: str = "logs") -> str:
    _ensure_dir(log_dir)
    path = _log_path(run_id, log_dir)
    event = {
        "event": "run_start",
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def log_run_event(run_id: str, event: str, payload: Dict[str, Any] | None = None, log_dir: str = "logs") -> None:
    _ensure_dir(log_dir)
    path = _log_path(run_id, log_dir)
    record = {
        "event": event,
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "payload": payload or {},
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def finalize_run_log(run_id: str, summary: Dict[str, Any], log_dir: str = "logs") -> None:
    log_run_event(run_id, "run_end", summary, log_dir=log_dir)
