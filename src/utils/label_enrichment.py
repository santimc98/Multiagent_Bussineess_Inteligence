import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple


_LABEL_TOKENS = (
    "label",
    "segment",
    "cluster",
    "cohort",
    "group",
    "partition",
)

_EXPLICIT_LABEL_COLUMNS = (
    "label_id",
    "label",
    "segment_id",
    "segment_label",
    "segment",
    "cluster_id",
    "cluster_label",
    "cluster",
    "cohort_id",
    "cohort_label",
    "cohort",
    "group_id",
    "group_label",
    "group",
    "partition_id",
    "partition_label",
    "partition",
)


def _column_lookup(columns: Iterable[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for col in columns:
        if not col:
            continue
        key = str(col).lower()
        if key not in lookup:
            lookup[key] = str(col)
    return lookup


def detect_label_column(df, preferred: Optional[str] = None) -> Optional[str]:
    if df is None:
        return None
    columns = getattr(df, "columns", None)
    if columns is None:
        return None
    lookup = _column_lookup(columns)
    if preferred:
        return lookup.get(str(preferred).lower())
    for name in _EXPLICIT_LABEL_COLUMNS:
        if name in lookup:
            return lookup.get(name)
    for raw in lookup.values():
        lowered = raw.lower()
        if lowered.endswith("_id") and any(token in lowered for token in _LABEL_TOKENS):
            return raw
        if lowered.endswith("_label") and any(token in lowered for token in _LABEL_TOKENS):
            return raw
    return None


def _derive_id_col_name(label_col: str) -> str:
    if not label_col:
        return "label_id"
    lowered = label_col.lower()
    if lowered.endswith("_id") or lowered.endswith("id"):
        return label_col
    if lowered.endswith("_label"):
        return f"{label_col[:-6]}_id"
    if lowered.endswith("_name"):
        return f"{label_col[:-5]}_id"
    if lowered in _LABEL_TOKENS:
        return f"{label_col}_id"
    return "label_id"


def add_stable_label_id(
    df,
    label_col: str,
    id_col_name: str = "label_id",
) -> Tuple[Any, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "label_col": label_col,
        "id_col_name": id_col_name,
        "created": False,
        "status": "skipped",
    }
    if df is None or label_col not in getattr(df, "columns", []):
        meta["status"] = "label_column_missing"
        return df, meta
    if id_col_name in df.columns:
        meta["status"] = "id_column_exists"
        return df, meta
    series = df[label_col]
    try:
        values = series.dropna().astype(str).tolist()
    except Exception:
        meta["status"] = "label_series_unreadable"
        return df, meta
    unique_labels = sorted(set(values))
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    try:
        df[id_col_name] = series.astype(str).map(mapping)
    except Exception:
        meta["status"] = "label_id_assignment_failed"
        return df, meta
    meta.update(
        {
            "created": True,
            "status": "ok",
            "mapping": mapping,
            "label_count": len(unique_labels),
            "null_count": int(series.isna().sum()) if hasattr(series, "isna") else None,
            "method": "sorted_unique",
        }
    )
    return df, meta


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


def _find_summary_list(payload: Dict[str, Any], label_keys: Iterable[str]) -> Tuple[Optional[str], Optional[list]]:
    for key in ("groups", "partitions", "segments", "items", "labels"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and any(label_key in item for label_key in label_keys):
                return key, items
        if items and all(isinstance(item, dict) for item in items):
            return key, items
    return None, None


def enrich_outputs(
    scored_rows_path: str,
    summary_json_path: Optional[str] = None,
    label_col_hint: Optional[str] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "status": "skipped",
        "scored_rows_path": scored_rows_path,
        "summary_json_path": summary_json_path,
        "label_col_hint": label_col_hint,
        "illustrative_only": True,
        "not_production": True,
    }
    if not scored_rows_path or not os.path.exists(scored_rows_path):
        meta["status"] = "scored_rows_missing"
        return _write_meta(meta)
    try:
        import pandas as pd
        df = pd.read_csv(scored_rows_path)
    except Exception:
        meta["status"] = "scored_rows_read_failed"
        return _write_meta(meta)

    label_col = detect_label_column(df, preferred=label_col_hint)
    if not label_col:
        meta["status"] = "label_column_unavailable"
        return _write_meta(meta)

    id_col_name = _derive_id_col_name(label_col)
    df, mapping_meta = add_stable_label_id(df, label_col, id_col_name=id_col_name)
    meta["label_col"] = label_col
    meta["id_col_name"] = id_col_name
    meta["mapping_meta"] = mapping_meta

    if mapping_meta.get("created"):
        try:
            df.to_csv(scored_rows_path, index=False)
        except Exception:
            meta["status"] = "scored_rows_write_failed"
            return _write_meta(meta)

    if summary_json_path and os.path.exists(summary_json_path):
        payload = _safe_load_json(summary_json_path)
        if isinstance(payload, dict):
            counts = df[label_col].astype(str).value_counts(dropna=False)
            sizes = {str(k): int(v) for k, v in counts.items()}
            label_keys = {label_col, id_col_name, "label_id"}
            _, items = _find_summary_list(payload, label_keys)
            if items:
                unmatched = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    value = None
                    for key in label_keys:
                        if key in item:
                            value = str(item.get(key))
                            break
                    if value is None:
                        continue
                    if value in sizes and "label_size" not in item:
                        item["label_size"] = sizes.get(value)
                    else:
                        unmatched.append(value)
                if unmatched:
                    payload.setdefault("caveats", [])
                    payload["caveats"].append(
                        f"Label ids missing in scored_rows.{label_col}: {sorted(set(unmatched))[:10]}"
                    )
            payload.setdefault("label_metadata", {})
            payload["label_metadata"].update(
                {
                    "label_column": label_col,
                    "id_column": id_col_name,
                    "label_sizes": sizes,
                }
            )
            if _safe_write_json(summary_json_path, payload):
                meta["summary_enriched"] = True
    meta["status"] = "ok"
    return _write_meta(meta)


def _write_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    meta_path = os.path.join("report", "governance", "enrichment_meta.json")
    try:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f_obj:
            json.dump(meta, f_obj, indent=2, ensure_ascii=False)
        meta["meta_path"] = meta_path.replace("\\", "/")
    except Exception:
        meta["meta_path"] = None
    return meta
