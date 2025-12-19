import json
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List


def fingerprint_dataset(path: str, max_bytes: int = 1024 * 1024) -> str:
    try:
        stat = os.stat(path)
        h = hashlib.sha256()
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
        with open(path, "rb") as f:
            chunk = f.read(max_bytes)
            h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _memory_path(path: str = "data/dataset_memory.json") -> str:
    return path


def load_dataset_memory(path: str = "data/dataset_memory.json") -> List[Dict[str, Any]]:
    try:
        with open(_memory_path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def record_dataset_memory(entry: Dict[str, Any], path: str = "data/dataset_memory.json", max_entries: int = 50) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entries = load_dataset_memory(path)
    entry = dict(entry)
    entry.setdefault("timestamp", datetime.utcnow().isoformat())
    entries.append(entry)
    if len(entries) > max_entries:
        entries = entries[-max_entries:]
    try:
        with open(_memory_path(path), "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def summarize_memory(entries: List[Dict[str, Any]], fingerprint: str) -> str:
    matches = [e for e in entries if e.get("fingerprint") == fingerprint]
    if not matches:
        return ""
    tail = matches[-3:]
    lines = []
    for item in tail:
        status = item.get("status", "unknown")
        gates = item.get("failed_gates", [])
        gates_text = ", ".join(gates) if gates else "none"
        lines.append(f"- {item.get('timestamp')}: status={status}, failed_gates={gates_text}")
    return "MEMORY_CONTEXT:\n" + "\n".join(lines)
