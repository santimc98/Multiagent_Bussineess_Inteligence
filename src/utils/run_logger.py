# src/utils/run_logger.py
import os
import json
from datetime import datetime, UTC
from typing import Dict, Any, Optional

LOG_PATHS: Dict[str, str] = {}


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def register_run_log(run_id: str, path: str) -> None:
    # CLAVE: guardar SIEMPRE en absoluto para que no dependa del cwd
    LOG_PATHS[run_id] = os.path.abspath(path)


def get_log_path(run_id: str, log_dir: str = "logs") -> str:
    if run_id in LOG_PATHS:
        return LOG_PATHS[run_id]
    # default también absoluto
    return os.path.abspath(os.path.join(log_dir, f"run_{run_id}.jsonl"))


def init_run_log(run_id: str, metadata: Optional[Dict[str, Any]] = None, log_dir: str = "logs") -> str:
    path = get_log_path(run_id, log_dir=log_dir)
    _ensure_parent_dir(path)  # CLAVE

    # Crea el fichero si no existe
    with open(path, "a", encoding="utf-8") as _:
        pass

    # Si quieres escribir evento init:
    if metadata is not None:
        log_run_event(run_id, "run_init", metadata, log_dir=log_dir)

    return path


def log_run_event(run_id: str, event_type: str, payload: Dict[str, Any], log_dir: str = "logs") -> None:
    path = get_log_path(run_id, log_dir=log_dir)
    _ensure_parent_dir(path)  # CLAVE

    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event_type,
        "payload": payload,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def finalize_run_log(
    run_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    log_dir: str = "logs",
) -> str:
    """
    Compat: algunos módulos esperan esta función.
    Asegura que el log existe y registra un evento de cierre.
    """
    path = init_run_log(run_id, metadata=None, log_dir=log_dir)

    if metadata is None:
        metadata = {}

    # Evento final (no cierra nada “real”, pero deja trazabilidad)
    log_run_event(run_id, "run_finalize", metadata, log_dir=log_dir)
    return path
