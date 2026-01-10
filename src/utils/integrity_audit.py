"""
Integrity Audit - V4.1 Compatible.

This module provides integrity auditing functionality for cleaned datasets
using the V4.1 contract schema (canonical_columns, column_roles, validation_requirements).
"""

import difflib
from typing import Dict, List, Tuple, Any

import pandas as pd

from src.utils.contract_v41 import (
    get_canonical_columns,
    get_column_roles,
    get_validation_requirements,
    get_preprocessing_requirements,
    get_artifact_requirements,
)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _find_column(df: pd.DataFrame, name: str) -> Tuple[str | None, bool]:
    """
    Returns (column_name, is_exact). Uses exact normalized match first, then cautious fuzzy.
    """
    norm = _normalize_name(name)
    candidates = {_normalize_name(col): col for col in df.columns}
    if norm in candidates:
        return candidates[norm], True
    close = difflib.get_close_matches(norm, candidates.keys(), n=1, cutoff=0.9)
    if close:
        return candidates[close[0]], False
    return None, False


def _numeric_stats(series: pd.Series) -> Dict[str, float | None]:
    try:
        if pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series.astype(float), errors="coerce")
        else:
            numeric = pd.to_numeric(series, errors="coerce")
    except Exception:
        try:
            numeric = pd.to_numeric(series.astype(float), errors="coerce")
        except Exception:
            numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return {"min": None, "p50": None, "p95": None, "max": None}
    return {
        "min": float(numeric.min()),
        "p50": float(numeric.quantile(0.5)),
        "p95": float(numeric.quantile(0.95)),
        "max": float(numeric.max()),
    }


def _column_stats(series: pd.Series) -> Dict[str, Any]:
    null_frac = float(series.isna().mean()) if len(series) else 0.0
    stats = {
        "dtype": str(series.dtype),
        "nunique": int(series.nunique(dropna=True)),
        "null_frac": null_frac,
        "count": int(len(series)),
    }
    if pd.api.types.is_numeric_dtype(series):
        stats.update(_numeric_stats(series))
    else:
        # try best-effort numeric stats
        stats.update(_numeric_stats(series))
    return stats


def _build_requirements_from_v41(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a list of column requirements from V4.1 contract keys:
    - canonical_columns (required input columns)
    - column_roles (role -> columns mapping)
    - validation_requirements.column_validations
    - preprocessing_requirements.expected_kinds
    """
    requirements: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _coerce_list(value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item]
        if isinstance(value, str):
            return [value]
        return []

    def _resolve_required_columns(contract_obj: Dict[str, Any]) -> List[str]:
        artifacts = get_artifact_requirements(contract_obj)
        clean_dataset = artifacts.get("clean_dataset") if isinstance(artifacts, dict) else None
        required_cols = _coerce_list(clean_dataset.get("required_columns")) if isinstance(clean_dataset, dict) else []
        if required_cols:
            return required_cols
        canonical_cols = get_canonical_columns(contract_obj)
        if canonical_cols:
            return canonical_cols
        strategy = {}
        for key in ("strategy", "strategy_spec", "selected_strategy"):
            candidate = contract_obj.get(key)
            if isinstance(candidate, dict):
                strategy = candidate
                break
        return _coerce_list(strategy.get("required_columns"))

    # Required columns for clean_dataset (no audit-only/unknown enforcement)
    required_columns = _resolve_required_columns(contract)
    for col in required_columns:
        if col not in seen:
            seen.add(col)
            requirements.append(
                {
                    "name": col,
                    "canonical_name": col,
                    "source": "clean_dataset_required",
                }
            )

    # Add role info for columns already required (do not enforce new requirements)
    roles = get_column_roles(contract)
    for role, columns in roles.items():
        if role == "unknown":
            continue
        for col in _coerce_list(columns):
            if col in seen:
                for req in requirements:
                    if req.get("canonical_name") == col or req.get("name") == col:
                        req["role"] = role
                        break
    
    # From validation_requirements.column_validations
    val_reqs = get_validation_requirements(contract)
    column_validations = val_reqs.get("column_validations")
    if isinstance(column_validations, list):
        for item in column_validations:
            if isinstance(item, dict):
                col = item.get("column") or item.get("name")
                if col and col in seen:
                    # Update existing only; do not add new requirements
                    for req in requirements:
                        if req.get("canonical_name") == col or req.get("name") == col:
                            if item.get("expected_range"):
                                req["expected_range"] = item.get("expected_range")
                            if item.get("allowed_null_frac") is not None:
                                req["allowed_null_frac"] = item.get("allowed_null_frac")
                            break
    
    # From preprocessing_requirements.expected_kinds
    prep_reqs = get_preprocessing_requirements(contract)
    expected_kinds = prep_reqs.get("expected_kinds")
    if isinstance(expected_kinds, dict):
        for col, kind in expected_kinds.items():
            for req in requirements:
                if req.get("canonical_name") == col or req.get("name") == col:
                    req["expected_kind"] = kind
                    break
    
    return requirements


def run_integrity_audit(df: pd.DataFrame, contract: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Generic integrity audit against an optional execution contract.
    Returns (issues, stats_by_column).
    Issues are descriptive only; no mutations happen here.
    
    V4.1: Uses canonical_columns, column_roles, validation_requirements instead of
    the legacy data_requirements key.
    """
    if not isinstance(contract, dict):
        contract = {}
    
    # V4.1: Build requirements from V4.1 keys
    requirements = _build_requirements_from_v41(contract)
    validations = contract.get("validations", []) or []
    artifacts = get_artifact_requirements(contract)
    schema_binding = artifacts.get("schema_binding") if isinstance(artifacts, dict) else {}
    optional_cols = schema_binding.get("optional_passthrough_columns", []) if isinstance(schema_binding, dict) else []
    if not isinstance(optional_cols, list):
        optional_cols = []

    stats: Dict[str, Dict[str, Any]] = {}
    issues: List[Dict[str, Any]] = []

    # Basic stats for all columns
    for col in df.columns:
        stats[col] = _column_stats(df[col])

    # Map requirements
    requirement_to_actual: Dict[str, Tuple[str, bool]] = {}
    used_actuals: Dict[str, List[Tuple[str, bool]]] = {}
    for req in requirements:
        name = req.get("name")
        canonical = req.get("canonical_name")
        req_label = canonical or name
        if not req_label:
            continue
        actual, is_exact = _find_column(df, req_label)
        if not actual and canonical and name:
            actual, is_exact = _find_column(df, name)
        if actual:
            requirement_to_actual[req_label] = (actual, is_exact)
            used_actuals.setdefault(actual, []).append((req_label, is_exact))
        else:
            missing_label = req_label
            detail_suffix = ""
            if canonical and name and canonical != name:
                detail_suffix = f" (canonical_name={canonical}, name={name})"
            issues.append(
                {
                    "type": "MISSING_COLUMN",
                    "severity": "critical",
                    "column": missing_label,
                    "detail": "Column required by contract not found in cleaned dataset." + detail_suffix,
                }
            )

    # Optional passthrough columns missing => warning only
    for opt_col in [str(col) for col in optional_cols if col]:
        actual, _is_exact = _find_column(df, opt_col)
        if not actual:
            issues.append(
                {
                    "type": "OPTIONAL_COLUMN_MISSING",
                    "code": "OPTIONAL_COLUMN_MISSING",
                    "severity": "warning",
                    "column": opt_col,
                    "detail": "Optional passthrough column not found in cleaned dataset.",
                }
            )

    # Aliasing risk: one actual column matched to multiple requirements
    for actual, reqs in used_actuals.items():
        if len(reqs) > 1:
            severity = "critical" if all(flag for _, flag in reqs) else "warning"
            issues.append(
                {
                    "type": "ALIASING_RISK",
                    "severity": severity,
                    "column": actual,
                    "detail": f"Actual column reused for multiple requirements: {reqs}",
                }
            )

    # Checks per requirement
    for req in requirements:
        name = req.get("name")
        canonical = req.get("canonical_name")
        req_label = canonical or name
        if not req_label:
            continue
        actual_entry = requirement_to_actual.get(req_label)
        if not actual_entry:
            continue
        actual, is_exact = actual_entry
        if actual not in df.columns:
            continue
        series = df[actual]
        col_stats = stats.get(actual, _column_stats(series))
        null_frac = col_stats.get("null_frac", 0.0)
        nunique = col_stats.get("nunique", 0)
        role = (req.get("role") or "").lower()
        expected_range = req.get("expected_range")
        allowed_null = req.get("allowed_null_frac")

        # High nulls vs allowed
        if allowed_null is not None and null_frac > allowed_null:
            issues.append(
                {
                    "type": "HIGH_NULLS",
                    "severity": "warning",
                    "column": actual,
                    "detail": f"Null fraction {null_frac:.3f} exceeds allowed {allowed_null}",
                }
            )

        # Low variance target
        if role in {"target", "outcome"} and nunique <= 1:
            issues.append(
                {
                    "type": "LOW_VARIANCE_TARGET",
                    "severity": "critical",
                    "column": actual,
                    "detail": "Target has no variance.",
                }
            )

        # Out-of-range numeric
        if expected_range and isinstance(expected_range, (list, tuple)) and len(expected_range) == 2:
            lo, hi = expected_range
            num = pd.to_numeric(series, errors="coerce")
            if num.dropna().empty:
                issues.append(
                    {
                        "type": "COERCION_FAILED",
                        "severity": "critical",
                        "column": actual,
                        "detail": "Numeric coercion produced all NaN; cannot validate range.",
                    }
                )
                continue
            if not num.dropna().empty:
                p95 = float(num.quantile(0.95))
                p05 = float(num.quantile(0.05))
                p50 = float(num.quantile(0.50))
                max_val = float(num.max())
                frac_gt_one = float((num > 1).mean())
                tolerance = 0.05 * (hi - lo) if hi is not None and lo is not None else 0.0
                if (hi is not None and p95 > hi + tolerance) or (lo is not None and p05 < lo - tolerance):
                    issues.append(
                        {
                            "type": "OUT_OF_RANGE",
                            "severity": "warning",
                            "column": actual,
                            "detail": f"Values fall outside expected_range {expected_range}; p05={p05:.3f}, p95={p95:.3f}",
                        }
                    )
                # Percentage scaling suspicion
                if lo == 0 and hi == 1:
                    if p95 > 1.5 or (max_val > 1 and frac_gt_one > 0.2):
                        severity = "critical" if (p95 > 1.5 or frac_gt_one > 0.5) else "warning"
                        issues.append(
                            {
                                "type": "PERCENT_SCALE_SUSPECTED",
                                "severity": severity,
                                "column": actual,
                                "detail": f"Expected ~[0,1] but observed p50={p50:.3f}, p95={p95:.3f}, max={max_val:.3f}, frac_gt_one={frac_gt_one:.2f}.",
                            }
                        )

        # Categorical destroyed by parsing
        if role == "categorical":
            if nunique <= 50 and null_frac > 0.9:
                issues.append(
                    {
                        "type": "CATEGORICAL_DESTROYED_BY_PARSING",
                        "severity": "warning",
                        "column": actual,
                        "detail": f"Likely categorical but null_frac={null_frac:.3f} with low nunique={nunique}.",
                    }
                )

    # Additional validations
    for val in validations:
        if isinstance(val, str):
            metric = "spearman" if "spearman" in val.lower() else ("kendall" if "kendall" in val.lower() else None)
            detail = f"Contract validation requested: {val}"
            if metric:
                detail += f" (metric={metric})"
            issues.append({"type": "VALIDATION_REQUIRED", "severity": "info", "detail": detail})
            continue
        if not isinstance(val, dict):
            issues.append({"type": "INVALID_VALIDATION_SCHEMA", "severity": "warning", "detail": str(val)})
            continue
        valtype = val.get("type") or ""
        valtype_lower = valtype.lower() if isinstance(valtype, str) else ""
        if valtype_lower == "ranking_coherence":
            metric = val.get("metric", "spearman")
            min_value = val.get("min_value")
            issues.append(
                {
                    "type": "VALIDATION_REQUIRED",
                    "severity": "info",
                    "detail": f"Validate ranking coherence using {metric} with min_value={min_value}.",
                }
            )

    return issues, stats
