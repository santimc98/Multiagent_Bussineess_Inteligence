import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.missing import is_effectively_missing_series
from src.utils.type_inference import safe_convert_numeric_currency


def _canonicalize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name))
    cleaned = cleaned.strip("_").lower()
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _dedup_columns(cols: List[str]) -> Tuple[List[str], Dict[str, int]]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for col in cols:
        base = _canonicalize_name(col) or "unknown_col"
        if base not in seen:
            seen[base] = 0
            out.append(base)
            continue
        seen[base] += 1
        out.append(f"{base}_{seen[base]}")
    return out, seen


def _ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _json_default(o):
    import numpy as _np
    import pandas as _pd
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, (_pd.Timestamp,)):
        return o.isoformat()
    if hasattr(_pd, "Timedelta") and isinstance(o, _pd.Timedelta):
        return o.total_seconds()
    if isinstance(o, (set, tuple)):
        return list(o)
    if o is None:
        return None
    try:
        if isinstance(o, float) and (o != o):
            return None
    except Exception:
        pass
    try:
        if getattr(_pd, "isna", None) and _pd.isna(o):
            return None
    except Exception:
        pass
    return str(o)


def _series_for_compare(series: pd.Series, value: Any) -> pd.Series:
    if isinstance(value, (int, float)):
        return pd.to_numeric(series, errors="coerce")
    return series


def _eval_condition(df: pd.DataFrame, cond: Dict[str, Any]) -> pd.Series:
    col = cond.get("col")
    op = cond.get("op")
    value = cond.get("value")
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    series = df[col]
    if op in {"is_null", "isna"}:
        return series.isna()
    if op in {"not_null", "notna"}:
        return series.notna()
    series_cmp = _series_for_compare(series, value)
    if op == "eq":
        return series_cmp == value
    if op == "neq":
        return series_cmp != value
    if op == "lt":
        return series_cmp < value
    if op == "lte":
        return series_cmp <= value
    if op == "gt":
        return series_cmp > value
    if op == "gte":
        return series_cmp >= value
    if op == "in":
        return series_cmp.isin(value if isinstance(value, list) else [value])
    if op == "not_in":
        return ~series_cmp.isin(value if isinstance(value, list) else [value])
    if op == "between":
        if isinstance(value, list) and len(value) == 2:
            lo, hi = value
            return series_cmp.between(lo, hi, inclusive="both")
    if op == "not_between":
        if isinstance(value, list) and len(value) == 2:
            lo, hi = value
            return ~series_cmp.between(lo, hi, inclusive="both")
    return pd.Series([False] * len(df), index=df.index)


def _apply_case_when(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.Series:
    cases = spec.get("cases", [])
    default = spec.get("default")
    result = pd.Series([default] * len(df), index=df.index)
    for case in cases:
        when = case.get("when", [])
        value = case.get("value")
        if not isinstance(when, list) or not when:
            continue
        mask = pd.Series([True] * len(df), index=df.index)
        for cond in when:
            if not isinstance(cond, dict):
                continue
            mask &= _eval_condition(df, cond)
        result = result.mask(mask, value)
    return result


def _apply_clip_rule(series: pd.Series, rule: Dict[str, Any]) -> pd.Series:
    min_val = rule.get("min")
    max_val = rule.get("max")
    overflow_to = rule.get("overflow_to")
    numeric = pd.to_numeric(series, errors="coerce")
    if min_val is not None:
        numeric = numeric.mask(numeric < min_val, overflow_to if overflow_to is not None else min_val)
    if max_val is not None:
        numeric = numeric.mask(numeric > max_val, overflow_to if overflow_to is not None else max_val)
    return numeric


def _normalize_percentage(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        return numeric
    median = float(non_null.median())
    if median > 1.0 and median <= 100.0:
        return numeric / 100.0
    return numeric


def _merge_conversions(plan_conversions: List[Dict[str, Any]], contract: Dict[str, Any], dialect: Dict[str, str]) -> List[Dict[str, Any]]:
    existing = {c.get("column"): c for c in plan_conversions if isinstance(c, dict)}
    reqs = contract.get("data_requirements", []) or []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        if req.get("source", "input") != "input":
            continue
        name = req.get("canonical_name") or req.get("name")
        if not name or name in existing:
            continue
        expected_kind = (req.get("expected_kind") or "").lower()
        if expected_kind == "numeric":
            role = (req.get("role") or "").lower()
            conversion = {
                "column": name,
                "kind": "numeric",
                "role": role,
                "decimal_hint": dialect.get("decimal"),
                "thousands_hint": "." if dialect.get("decimal") == "," else ",",
                "normalize_0_1": role in {"percentage", "probability", "risk_score", "ratio"},
            }
            existing[name] = conversion
        elif expected_kind == "categorical":
            existing[name] = {"column": name, "kind": "categorical"}
        elif expected_kind == "datetime":
            existing[name] = {"column": name, "kind": "datetime"}
    return list(existing.values())


def execute_cleaning_plan(plan: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, Any]:
    input_cfg = plan.get("input") or {}
    dialect = plan.get("dialect") or {}
    output_cfg = plan.get("output") or {}
    input_path = input_cfg.get("path", "data/raw.csv")
    sep = dialect.get("sep", ",")
    decimal = dialect.get("decimal", ".")
    encoding = dialect.get("encoding", "utf-8")

    df = pd.read_csv(input_path, sep=sep, decimal=decimal, encoding=encoding, dtype=str, low_memory=False)
    rows_before = len(df)
    original_columns = list(df.columns)

    new_cols, _ = _dedup_columns(original_columns)
    column_mapping = {orig: new for orig, new in zip(original_columns, new_cols)}
    df.columns = new_cols

    conversions_meta: Dict[str, Any] = {}
    conversions: Dict[str, str] = {}
    warnings: List[str] = []

    plan_conversions = plan.get("type_conversions") or []
    merged_conversions = _merge_conversions(plan_conversions, contract, dialect)
    for conv in merged_conversions:
        if not isinstance(conv, dict):
            continue
        col = conv.get("column")
        if not col or col not in df.columns:
            continue
        kind = (conv.get("kind") or "").lower()
        role = (conv.get("role") or "").lower()
        if kind == "numeric":
            converted, meta = safe_convert_numeric_currency(
                df[col],
                decimal_hint=conv.get("decimal_hint"),
                thousands_hint=conv.get("thousands_hint"),
            )
            df[col] = converted
            conversions[col] = "clean_numeric"
            conversions_meta[col] = meta
            if conv.get("normalize_0_1") or role in {"percentage", "probability", "risk_score", "ratio"}:
                df[col] = _normalize_percentage(df[col])
                conversions[col] = conversions[col] + "_normalized_0_1"
        elif kind == "categorical":
            df[col] = df[col].astype(str).str.strip()
            conversions[col] = "categorical"
        elif kind == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
            conversions[col] = "datetime"

    # Derived columns
    derived = plan.get("derived_columns") or []
    for deriv in derived:
        if not isinstance(deriv, dict):
            continue
        name = deriv.get("name")
        op = deriv.get("op")
        if not name:
            continue
        if op == "copy":
            source = deriv.get("source")
            if source in df.columns:
                df[name] = df[source]
            continue
        if op == "clip":
            source = deriv.get("source")
            if source in df.columns:
                df[name] = _apply_clip_rule(df[source], deriv)
            continue
        if op == "case_when":
            df[name] = _apply_case_when(df, deriv)
            continue

    # Case assignment
    case_cfg = plan.get("case_assignment") or {}
    if case_cfg:
        case_id_col = case_cfg.get("case_id_col", "case_id")
        refscore_col = case_cfg.get("refscore_col", "refscore")
        rules = case_cfg.get("rules", [])
        default_case = case_cfg.get("default") or {}
        default_case_id = default_case.get("case_id")
        default_ref = default_case.get("ref_score")
        df[case_id_col] = default_case_id
        df[refscore_col] = default_ref
        for rule in rules:
            when = rule.get("when", [])
            if not isinstance(when, list) or not when:
                continue
            mask = pd.Series([True] * len(df), index=df.index)
            for cond in when:
                if not isinstance(cond, dict):
                    continue
                mask &= _eval_condition(df, cond)
            if "case_id" in rule:
                df.loc[mask, case_id_col] = rule.get("case_id")
            if "ref_score" in rule:
                df.loc[mask, refscore_col] = rule.get("ref_score")

    # Constant columns warning
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            warnings.append(f"Constant column kept: {col}")

    output_path = output_cfg.get("cleaned_path", "data/cleaned_data.csv")
    manifest_path = output_cfg.get("manifest_path", "data/cleaning_manifest.json")
    _ensure_dir(output_path)
    _ensure_dir(manifest_path)

    df.to_csv(output_path, index=False, sep=",", decimal=".", encoding="utf-8")

    manifest = {
        "input_dialect": {"sep": sep, "decimal": decimal, "encoding": encoding},
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "original_columns": original_columns,
        "column_mapping": column_mapping,
        "dropped_columns": [],
        "conversions": conversions,
        "conversions_meta": conversions_meta,
        "type_checks": [],
        "rows_before": rows_before,
        "rows_after": len(df),
        "warnings": warnings,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)

    return {
        "cleaned_path": output_path,
        "manifest_path": manifest_path,
        "rows_before": rows_before,
        "rows_after": len(df),
        "warnings": warnings,
    }
