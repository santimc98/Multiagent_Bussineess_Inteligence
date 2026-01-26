import csv
import json
import os
from typing import Any, Dict, List, Optional

from src.utils.dataset_size import estimate_rows_fast
from src.utils.column_sets import summarize_column_sets

COLUMN_LIST_POINTER = "Full list in data/column_inventory.json; use selectors in data/column_sets.json"


def summarize_long_list(items: List[Any], head: int = 12, tail: int = 6) -> Dict[str, Any]:
    items = [str(item) for item in items if item is not None]
    if len(items) <= head + tail:
        return {"count": len(items), "head": items, "tail": []}
    return {"count": len(items), "head": items[:head], "tail": items[-tail:]}


def compress_long_lists(
    payload: Any,
    threshold: int = 80,
    head: int = 12,
    tail: int = 6,
) -> tuple[Any, bool]:
    compressed = False

    def _should_compress(value: list[Any]) -> bool:
        if len(value) <= threshold:
            return False
        for item in value:
            if item is None:
                continue
            if not isinstance(item, (str, int, float)):
                return False
        return True

    def _compress(value: Any) -> Any:
        nonlocal compressed
        if isinstance(value, dict):
            return {key: _compress(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            items = list(value)
            if _should_compress(items):
                summary = summarize_long_list(items, head=head, tail=tail)
                summary["note"] = COLUMN_LIST_POINTER
                compressed = True
                return summary
            return [_compress(item) for item in items]
        return value

    compressed_payload = _compress(payload)
    if compressed and isinstance(compressed_payload, dict):
        compressed_payload.setdefault("column_list_reference", COLUMN_LIST_POINTER)
    return compressed_payload, compressed


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _limit_list(items: List[Any], max_items: int = 25) -> List[Any]:
    if len(items) <= max_items:
        return items
    remainder = len(items) - max_items
    trimmed = items[:max_items]
    trimmed.append(f"...({remainder} more)")
    return trimmed


def _summarize_if_long(items: List[Any], threshold: int = 80) -> Any:
    items = [str(item) for item in items if item is not None]
    if len(items) > threshold:
        summary = summarize_long_list(items)
        summary["note"] = COLUMN_LIST_POINTER
        return summary
    return items


def _extract_columns(value: Any) -> List[str]:
    if isinstance(value, dict):
        columns = value.get("columns")
        if isinstance(columns, list):
            return [str(col) for col in columns if col]
        return []
    if isinstance(value, list):
        return [str(col) for col in value if col]
    return []


def _resolve_work_dir(state: Dict[str, Any]) -> Optional[str]:
    work_dir = state.get("work_dir_abs") or state.get("work_dir")
    if work_dir and isinstance(work_dir, str):
        return os.path.abspath(work_dir)
    return None


def _candidate_paths(state: Dict[str, Any], rel_path: str) -> List[str]:
    candidates: List[str] = []
    work_dir = _resolve_work_dir(state)
    if work_dir:
        candidates.append(os.path.join(work_dir, rel_path))
    candidates.append(rel_path)
    return list(dict.fromkeys(candidates))


def _first_existing_path(state: Dict[str, Any], rel_path: str) -> Optional[str]:
    for path in _candidate_paths(state, rel_path):
        if path and os.path.exists(path):
            return path
    return None


def _read_csv_header(path: str, sep: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle, delimiter=sep)
            return next(reader, [])
    except Exception:
        return []


def _artifact_file_info(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"exists": False}
    try:
        size_bytes = os.path.getsize(path)
    except Exception:
        size_bytes = None
    return {
        "exists": True,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 4) if isinstance(size_bytes, (int, float)) else None,
    }


def _safe_load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _extract_decisioning_columns(contract_like: Dict[str, Any]) -> List[str]:
    if not isinstance(contract_like, dict):
        return []
    decisioning = contract_like.get("decisioning_requirements")
    if not isinstance(decisioning, dict):
        return []
    output = decisioning.get("output")
    if not isinstance(output, dict):
        return []
    required = output.get("required_columns")
    if not isinstance(required, list):
        return []
    names: List[str] = []
    for item in required:
        name = None
        if isinstance(item, dict):
            name = item.get("name") or item.get("column")
        elif isinstance(item, str):
            name = item
        if name:
            names.append(str(name))
    return list(dict.fromkeys(names))


def build_context_pack(stage: str, state: Dict[str, Any]) -> str:
    """
    Build a compact context pack for agent prompts.
    Keeps output short and deterministic.
    """
    state = state if isinstance(state, dict) else {}
    contract_min = state.get("execution_contract_min") or state.get("contract_min") or {}
    contract = state.get("execution_contract") or {}

    dataset_scale_hints = state.get("dataset_scale_hints") if isinstance(state.get("dataset_scale_hints"), dict) else {}
    column_inventory = _extract_columns(
        state.get("column_inventory")
        or state.get("cleaned_column_inventory")
        or (state.get("ml_context_snapshot") or {}).get("cleaned_column_inventory")
        or []
    )

    rows_est = None
    if isinstance(dataset_scale_hints, dict):
        rows_est = dataset_scale_hints.get("est_rows") or dataset_scale_hints.get("rows")

    dataset_scale = {
        "scale": dataset_scale_hints.get("scale") if isinstance(dataset_scale_hints, dict) else None,
        "rows": rows_est,
        "cols": len(column_inventory) if column_inventory else None,
        "file_mb": dataset_scale_hints.get("file_mb") if isinstance(dataset_scale_hints, dict) else None,
    }

    dialect = {
        "sep": state.get("csv_sep"),
        "decimal": state.get("csv_decimal"),
        "encoding": state.get("csv_encoding"),
    }

    required_outputs = []
    if isinstance(contract_min, dict) and isinstance(contract_min.get("required_outputs"), list):
        required_outputs = [str(item) for item in contract_min.get("required_outputs") if item]
    elif isinstance(contract, dict) and isinstance(contract.get("required_outputs"), list):
        required_outputs = [str(item) for item in contract.get("required_outputs") if item]

    decisioning_columns = _extract_decisioning_columns(contract_min if isinstance(contract_min, dict) else contract)
    if not decisioning_columns:
        decisioning_columns = _extract_decisioning_columns(contract if isinstance(contract, dict) else {})

    column_sets = state.get("column_sets")
    if not isinstance(column_sets, dict) or not column_sets:
        column_sets = _safe_load_json(_first_existing_path(state, os.path.join("data", "column_sets.json")) or "")
    column_sets_summary = summarize_column_sets(column_sets) if isinstance(column_sets, dict) else ""

    artifacts: Dict[str, Any] = {}
    metrics_path = _first_existing_path(state, os.path.join("data", "metrics.json"))
    alignment_path = _first_existing_path(state, os.path.join("data", "alignment_check.json"))
    scored_path = _first_existing_path(state, os.path.join("data", "scored_rows.csv"))
    cleaned_path = _first_existing_path(state, os.path.join("data", "cleaned_data.csv"))
    output_contract_path = _first_existing_path(state, os.path.join("data", "output_contract_report.json"))

    artifacts["data/metrics.json"] = _artifact_file_info(metrics_path) if metrics_path else {"exists": False}
    artifacts["data/alignment_check.json"] = _artifact_file_info(alignment_path) if alignment_path else {"exists": False}

    if scored_path:
        scored_info = _artifact_file_info(scored_path)
        sep = state.get("csv_sep") or ","
        scored_header = _read_csv_header(scored_path, sep)
        if scored_header:
            scored_columns = [str(col) for col in scored_header if col]
            scored_info["header"] = _summarize_if_long(scored_columns)
            scored_info["header_cols"] = len(scored_header)
        artifacts["data/scored_rows.csv"] = scored_info
    else:
        artifacts["data/scored_rows.csv"] = {"exists": False}

    if cleaned_path:
        cleaned_info = _artifact_file_info(cleaned_path)
        sep = state.get("csv_sep") or ","
        cleaned_header = _read_csv_header(cleaned_path, sep)
        if cleaned_header:
            cleaned_info["cols"] = len(cleaned_header)
        est_rows = estimate_rows_fast(cleaned_path, encoding=str(state.get("csv_encoding") or "utf-8"))
        if est_rows is not None and cleaned_info.get("cols") is not None:
            cleaned_info["shape"] = [int(est_rows), int(cleaned_info.get("cols"))]
        artifacts["data/cleaned_data.csv"] = cleaned_info
    else:
        artifacts["data/cleaned_data.csv"] = {"exists": False}

    if output_contract_path:
        report = _safe_load_json(output_contract_path)
        missing = report.get("missing") if isinstance(report.get("missing"), list) else []
        present = report.get("present") if isinstance(report.get("present"), list) else []
        artifacts["data/output_contract_report.json"] = {
            **_artifact_file_info(output_contract_path),
            "missing_count": len(missing),
            "present_count": len(present),
            "missing": _limit_list([str(item) for item in missing if item], max_items=25),
            "present": _limit_list([str(item) for item in present if item], max_items=25),
        }
    else:
        artifacts["data/output_contract_report.json"] = {"exists": False}

    evidence_index = []
    for path in [
        "data/metrics.json",
        "data/alignment_check.json",
        "data/scored_rows.csv",
        "data/cleaned_data.csv",
        "data/output_contract_report.json",
        "data/dataset_semantics.json",
        "data/dataset_training_mask.json",
        "data/column_sets.json",
        "data/column_inventory.json",
        "data/insights.json",
    ]:
        if len(evidence_index) >= 8:
            break
        exists = _first_existing_path(state, path)
        if exists or path not in evidence_index:
            evidence_index.append(path)

    guard_warnings = state.get("data_engineer_guard_warnings")
    if not isinstance(guard_warnings, list):
        guard_warnings = []

    payload = {
        "stage": stage,
        "run_id": state.get("run_id"),
        "dataset_scale": dataset_scale,
        "dialect": dialect,
        "required_outputs": _summarize_if_long(required_outputs),
        "decisioning_required_columns": _summarize_if_long(decisioning_columns),
        "column_sets_summary": column_sets_summary,
        "guard_warnings": _limit_list([str(item) for item in guard_warnings if item], max_items=10),
        "artifacts": artifacts,
        "evidence_index": evidence_index,
    }
    payload, _ = compress_long_lists(payload)
    return "CONTEXT_PACK_JSON:\n" + json.dumps(payload, indent=2, ensure_ascii=True)
