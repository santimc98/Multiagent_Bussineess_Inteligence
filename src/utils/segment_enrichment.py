import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


_EXPLICIT_SEGMENT_LABELS = {
    "segment_id",
    "segment",
    "segment_label",
    "segment_key",
    "cluster",
    "cluster_id",
    "cluster_label",
    "typology",
    "typology_id",
    "group",
    "group_id",
}


def _column_lookup(columns: Iterable[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for col in columns:
        if not col:
            continue
        key = str(col).lower()
        if key not in lookup:
            lookup[key] = str(col)
    return lookup


def _resolve_label_column(
    columns: Iterable[str],
    segment_label_column: Optional[str],
    segment_id_key: str,
) -> Optional[str]:
    lookup = _column_lookup(columns)
    if segment_label_column:
        return lookup.get(str(segment_label_column).lower())
    if segment_id_key and str(segment_id_key).lower() in lookup:
        return lookup.get(str(segment_id_key).lower())
    for name in sorted(_EXPLICIT_SEGMENT_LABELS):
        if name in lookup:
            return lookup.get(name)
    return None


def _safe_load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f_obj:
            return json.load(f_obj)
    except Exception:
        return None


def _safe_write_json(path: str, payload: Dict[str, Any]) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f_obj:
            json.dump(payload, f_obj, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def _mark_unavailable(payload: Dict[str, Any], reason: str) -> Dict[str, Any]:
    payload["segment_metadata_unavailable"] = True
    payload["segment_metadata_reason"] = reason
    return payload


def _append_caveat(payload: Dict[str, Any], message: str) -> None:
    caveats = payload.get("caveats")
    if not isinstance(caveats, list):
        caveats = []
    if message not in caveats:
        caveats.append(message)
    payload["caveats"] = caveats


def enrich_segmented_summary_json(
    summary_path: str,
    scored_rows_path: Optional[str] = None,
    segment_label_column: Optional[str] = None,
    segment_id_key: str = "segment_id",
) -> bool:
    if not summary_path or not os.path.exists(summary_path):
        return False
    payload = _safe_load_json(summary_path)
    if not isinstance(payload, dict):
        return False

    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        _mark_unavailable(payload, "segments_missing_or_empty")
        return _safe_write_json(summary_path, payload)

    missing_ids = [idx for idx, item in enumerate(segments) if not isinstance(item, dict) or segment_id_key not in item]
    if missing_ids:
        _mark_unavailable(payload, f"segment_id_missing_in_summary_items:{missing_ids}")
        return _safe_write_json(summary_path, payload)

    if not scored_rows_path or not os.path.exists(scored_rows_path):
        _mark_unavailable(payload, "scored_rows_missing")
        return _safe_write_json(summary_path, payload)

    try:
        import pandas as pd
        df = pd.read_csv(scored_rows_path)
    except Exception:
        _mark_unavailable(payload, "scored_rows_read_failed")
        return _safe_write_json(summary_path, payload)

    label_col = _resolve_label_column(df.columns, segment_label_column, segment_id_key)
    if not label_col:
        _mark_unavailable(payload, "segment_label_column_unavailable")
        return _safe_write_json(summary_path, payload)

    label_series = df[label_col].astype(str)
    label_values = {str(val) for val in label_series.dropna().unique()}
    summary_ids = {str(item.get(segment_id_key)) for item in segments if isinstance(item, dict)}

    overlap = summary_ids.intersection(label_values)
    if not overlap:
        _append_caveat(
            payload,
            f"Segment ids in summary do not match scored_rows.{label_col}. No segment metadata added.",
        )
        _mark_unavailable(payload, "segment_id_mismatch")
        return _safe_write_json(summary_path, payload)

    missing_in_scored = sorted(summary_ids - label_values)
    if missing_in_scored:
        _append_caveat(
            payload,
            f"Some segment ids missing in scored_rows.{label_col}: {missing_in_scored[:10]}",
        )

    counts = label_series.value_counts(dropna=False)
    sizes = {str(k): int(v) for k, v in counts.items()}

    for item in segments:
        if not isinstance(item, dict):
            continue
        seg_id = str(item.get(segment_id_key))
        if seg_id in sizes and "segment_size" not in item:
            item["segment_size"] = sizes.get(seg_id)
        if label_col != segment_id_key and "segment_definition" not in item and seg_id in label_values:
            item["segment_definition"] = {label_col: seg_id}
        if label_col != segment_id_key and "segment_description" not in item and seg_id in label_values:
            item["segment_description"] = f"{label_col}={seg_id}"

    payload.setdefault("segment_metadata_unavailable", False)
    if "segment_sizes" not in payload:
        payload["segment_sizes"] = {
            "total": int(len(df)),
            "by_segment_id": sizes,
            "label_column": label_col,
        }
    return _safe_write_json(summary_path, payload)
