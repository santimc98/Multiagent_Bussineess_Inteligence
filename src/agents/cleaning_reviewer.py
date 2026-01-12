import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.code_extract import extract_code_block

_CLEANING_FALLBACK_WARNING = (
    "CONTRACT_BROKEN_FALLBACK: cleaning_gates missing; please fix contract generation"
)

_DEFAULT_ID_REGEX = r"(?i)(^id$|id$|entity|cod|code|key|partida|invoice|account)"
_DEFAULT_PERCENT_REGEX = r"(?i)%|pct|percent|plazo"

_FALLBACK_CLEANING_GATES = [
    {
        "name": "required_columns_present",
        "severity": "HARD",
        "params": {},
    },
    {
        "name": "id_integrity",
        "severity": "HARD",
        "params": {
            "identifier_name_regex": _DEFAULT_ID_REGEX,
            "detect_scientific_notation": True,
        },
    },
    {
        "name": "no_semantic_rescale",
        "severity": "HARD",
        "params": {
            "allow_percent_like_only": True,
            "percent_like_name_regex": _DEFAULT_PERCENT_REGEX,
        },
    },
    {
        "name": "no_synthetic_data",
        "severity": "HARD",
        "params": {},
    },
    {
        "name": "row_count_sanity",
        "severity": "SOFT",
        "params": {
            "max_drop_pct": 5.0,
            "max_dup_increase_pct": 1.0,
        },
    },
]


class CleaningReviewerAgent:
    """
    Contract-driven cleaning reviewer. Evaluates ONLY the cleaning_gates specified
    by the Execution Planner (via CLEANING_VIEW_CONTEXT).
    """

    def __init__(self):
        self.last_prompt = None
        self.last_response = None
        self.model_name = "contract-driven"

    def review_cleaning(
        self,
        cleaning_view: Dict[str, Any],
        cleaned_csv_path: str,
        cleaning_manifest_path: str,
        raw_csv_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            result = _review_cleaning_impl(
                cleaning_view=cleaning_view,
                cleaned_csv_path=cleaned_csv_path,
                cleaning_manifest_path=cleaning_manifest_path,
                raw_csv_path=raw_csv_path,
            )
        except Exception as exc:
            result = {
                "status": "REJECTED",
                "feedback": f"Cleaning reviewer exception: {exc}",
                "failed_checks": ["CLEANING_REVIEWER_EXCEPTION"],
                "required_fixes": ["Investigate cleaning reviewer failure."],
                "warnings": [str(exc)],
                "cleaning_gates_evaluated": [],
                "hard_failures": ["CLEANING_REVIEWER_EXCEPTION"],
                "soft_failures": [],
                "contract_source_used": "error",
            }

        self.last_prompt = "contract-driven-cleaning-review"
        self.last_response = json.dumps(result, ensure_ascii=True)
        return result


def _review_cleaning_impl(
    cleaning_view: Dict[str, Any],
    cleaned_csv_path: str,
    cleaning_manifest_path: str,
    raw_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    view = cleaning_view if isinstance(cleaning_view, dict) else {}
    gates, contract_source_used, warnings = _resolve_cleaning_gates(view)
    gates_evaluated = [gate["name"] for gate in gates]

    dialect = _resolve_dialect(view.get("dialect"))
    required_columns = _list_str(view.get("required_columns"))
    column_roles = _coerce_roles(view.get("column_roles"))

    manifest = _load_json(cleaning_manifest_path)
    cleaned_header = _read_csv_header(cleaned_csv_path, dialect)
    sample_str = _read_csv_sample(cleaned_csv_path, dialect, cleaned_header, dtype=str)
    sample_infer = _read_csv_sample(cleaned_csv_path, dialect, cleaned_header, dtype=None)
    raw_sample = None
    if raw_csv_path:
        raw_sample = _read_csv_sample(raw_csv_path, dialect, None, dtype=str)

    hard_failures: List[str] = []
    soft_failures: List[str] = []
    failed_checks: List[str] = []
    required_fixes: List[str] = []
    soft_warnings: List[str] = list(warnings)
    failure_summaries: List[str] = []
    warning_summaries: List[str] = []

    for gate in gates:
        name = gate["name"]
        severity = gate["severity"]
        params = gate["params"]
        issues: List[str] = []

        if name == "required_columns_present":
            issues = _check_required_columns(required_columns, cleaned_header, cleaned_csv_path)
        elif name == "id_integrity":
            issues = _check_id_integrity(
                cleaned_header,
                sample_str,
                sample_infer,
                params,
                column_roles,
            )
        elif name == "no_semantic_rescale":
            issues = _check_no_semantic_rescale(
                manifest,
                params,
                cleaned_header,
                column_roles,
                raw_sample,
                sample_infer or sample_str,
            )
        elif name == "no_synthetic_data":
            issues = _check_no_synthetic_data(manifest)
        elif name == "row_count_sanity":
            issues = _check_row_count_sanity(manifest, params)
        else:
            soft_warnings.append(f"UNKNOWN_GATE_SKIPPED: {name}")

        if issues:
            _record_gate_failure(
                name,
                severity,
                issues,
                hard_failures,
                soft_failures,
                failed_checks,
                required_fixes,
                failure_summaries,
                warning_summaries,
            )

    status = "APPROVED"
    if hard_failures:
        status = "REJECTED"
    elif soft_failures:
        status = "APPROVE_WITH_WARNINGS"

    if hard_failures:
        feedback = "Cleaning reviewer rejected: " + " | ".join(failure_summaries)
    elif soft_failures:
        feedback = "Cleaning reviewer approved with warnings: " + " | ".join(warning_summaries)
    else:
        feedback = "Cleaning reviewer approved: all gates passed."

    if warning_summaries:
        soft_warnings.extend(warning_summaries)

    result = {
        "status": status,
        "feedback": feedback,
        "failed_checks": failed_checks,
        "required_fixes": required_fixes,
        "warnings": soft_warnings,
        "cleaning_gates_evaluated": gates_evaluated,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "contract_source_used": contract_source_used,
    }
    return normalize_cleaning_reviewer_result(result)


def _resolve_cleaning_gates(view: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    raw = view.get("cleaning_gates") if isinstance(view, dict) else None
    if isinstance(raw, list) and raw:
        return _normalize_cleaning_gates(raw), "cleaning_view", []
    return _normalize_cleaning_gates(_FALLBACK_CLEANING_GATES), "fallback", [_CLEANING_FALLBACK_WARNING]


def _normalize_cleaning_gates(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for gate in raw:
        if isinstance(gate, dict):
            name = gate.get("name") or gate.get("id") or gate.get("gate")
            if not name:
                continue
            severity = _normalize_severity(gate.get("severity"), gate.get("required"))
            params = gate.get("params")
            if not isinstance(params, dict):
                params = {}
            key = str(name).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"name": str(name), "severity": severity, "params": params})
        elif isinstance(gate, str):
            key = gate.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append({"name": gate.strip(), "severity": "HARD", "params": {}})
    return normalized


def _normalize_severity(severity: Any, required: Any = None) -> str:
    if severity is None and required is not None:
        severity = "HARD" if bool(required) else "SOFT"
    sev = str(severity).strip().upper() if severity else "HARD"
    return sev if sev in {"HARD", "SOFT"} else "HARD"


def _resolve_dialect(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"sep": ",", "decimal": ".", "encoding": "utf-8"}
    return {
        "sep": raw.get("sep", ","),
        "decimal": raw.get("decimal", "."),
        "encoding": raw.get("encoding", "utf-8"),
    }


def _load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_csv_header(path: str, dialect: Dict[str, Any]) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(
            path,
            nrows=0,
            sep=dialect.get("sep", ","),
            decimal=dialect.get("decimal", "."),
            encoding=dialect.get("encoding", "utf-8"),
        )
        return [str(col) for col in df.columns if col]
    except Exception:
        return []


def _read_csv_sample(
    path: str,
    dialect: Dict[str, Any],
    columns: Optional[List[str]],
    dtype: Optional[Any],
    nrows: int = 1000,
) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    kwargs = {
        "nrows": nrows,
        "sep": dialect.get("sep", ","),
        "decimal": dialect.get("decimal", "."),
        "encoding": dialect.get("encoding", "utf-8"),
        "low_memory": False,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if columns:
        kwargs["usecols"] = columns
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def _list_str(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_roles(raw: Any) -> Dict[str, List[str]]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for key, val in raw.items():
        if isinstance(val, list):
            out[str(key)] = [str(item) for item in val if item]
    return out


def _record_gate_failure(
    gate_name: str,
    severity: str,
    issues: List[str],
    hard_failures: List[str],
    soft_failures: List[str],
    failed_checks: List[str],
    required_fixes: List[str],
    failure_summaries: List[str],
    warning_summaries: List[str],
) -> None:
    summary = f"{gate_name}: " + "; ".join(issues)
    if gate_name not in failed_checks:
        failed_checks.append(gate_name)
    if severity == "HARD":
        if gate_name not in hard_failures:
            hard_failures.append(gate_name)
        required_fixes.append(summary)
        failure_summaries.append(summary)
    else:
        if gate_name not in soft_failures:
            soft_failures.append(gate_name)
        warning_summaries.append(summary)


def _check_required_columns(
    required_columns: List[str],
    cleaned_header: List[str],
    cleaned_csv_path: str,
) -> List[str]:
    if not required_columns:
        return []
    if not cleaned_header:
        return [f"Unable to read cleaned CSV header: {cleaned_csv_path}"]
    missing = [col for col in required_columns if col not in cleaned_header]
    if missing:
        return [f"Missing required columns: {', '.join(missing)}"]
    return []


def _check_id_integrity(
    cleaned_header: List[str],
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> List[str]:
    if not cleaned_header:
        return []
    regex = params.get("identifier_name_regex") or _DEFAULT_ID_REGEX
    try:
        pattern = re.compile(regex)
    except re.error:
        pattern = re.compile(_DEFAULT_ID_REGEX)

    candidates = [col for col in cleaned_header if pattern.search(col)]
    role_candidates = _columns_with_role_tokens(column_roles, {"id", "identifier"})
    for col in role_candidates:
        if col in cleaned_header and col not in candidates:
            candidates.append(col)
    if not candidates:
        return []

    detect_sci = bool(params.get("detect_scientific_notation", True))
    sci_threshold = float(params.get("scientific_notation_ratio_threshold", 0.02))
    dot0_threshold = float(params.get("dot_zero_ratio_threshold", 0.1))
    min_samples = int(params.get("min_samples", 20))

    issues: List[str] = []
    for col in candidates:
        values = _string_values(sample_str, col)
        if len(values) < min_samples:
            continue
        sci_count = 0
        dot0_count = 0
        for val in values:
            lowered = val.lower()
            if detect_sci and ("e+" in lowered or "e-" in lowered):
                sci_count += 1
            if re.search(r"\.0+$", val):
                dot0_count += 1
        total = len(values)
        if detect_sci and sci_count / total >= sci_threshold:
            issues.append(f"{col} contains scientific notation ({sci_count}/{total})")
        if dot0_count / total >= dot0_threshold:
            issues.append(f"{col} coerced to float-like values ({dot0_count}/{total} end with .0)")
        if sample_infer is not None and col in sample_infer.columns:
            if pd.api.types.is_float_dtype(sample_infer[col]):
                issues.append(f"{col} inferred as float dtype in cleaned data")
    return issues


def _check_no_semantic_rescale(
    manifest: Dict[str, Any],
    params: Dict[str, Any],
    cleaned_header: List[str],
    column_roles: Dict[str, List[str]],
    raw_sample: Optional[pd.DataFrame],
    cleaned_sample: Optional[pd.DataFrame],
) -> List[str]:
    allow_percent_only = bool(params.get("allow_percent_like_only", True))
    regex = params.get("percent_like_name_regex") or _DEFAULT_PERCENT_REGEX
    try:
        percent_pattern = re.compile(regex)
    except re.error:
        percent_pattern = re.compile(_DEFAULT_PERCENT_REGEX)

    percent_like = {col for col in cleaned_header if percent_pattern.search(col)}
    percent_like.update(_columns_with_role_tokens(column_roles, {"percent", "percentage", "ratio", "probability"}))

    conversions = manifest.get("conversions") if isinstance(manifest, dict) else {}
    if not isinstance(conversions, dict):
        conversions = {}
    conversions_meta = manifest.get("conversions_meta") if isinstance(manifest, dict) else {}
    if not isinstance(conversions_meta, dict):
        conversions_meta = {}

    rescaled_cols: List[str] = []
    for col, conv in conversions.items():
        if not isinstance(col, str):
            continue
        if isinstance(conv, str) and "normalized_0_1" in conv:
            rescaled_cols.append(col)
    for col, meta in conversions_meta.items():
        if not isinstance(col, str) or not isinstance(meta, dict):
            continue
        if meta.get("normalized") or meta.get("scale_factor") or meta.get("scaled_by"):
            rescaled_cols.append(col)

    rescaled_cols.extend(_scan_code_for_rescale_ops())
    rescaled_cols = [col for col in rescaled_cols if col]
    if not rescaled_cols:
        return _check_semantic_rescale_from_raw(
            raw_sample,
            cleaned_sample,
            cleaned_header,
            percent_like,
            allow_percent_only,
        )

    issues: List[str] = []
    for col in rescaled_cols:
        if col == "__MINMAX__":
            issues.append("MinMaxScaler detected in cleaning script")
            continue
        if allow_percent_only and col not in percent_like:
            issues.append(f"{col} appears rescaled but is not percent-like")
    return issues


def _check_semantic_rescale_from_raw(
    raw_sample: Optional[pd.DataFrame],
    cleaned_sample: Optional[pd.DataFrame],
    cleaned_header: List[str],
    percent_like: set[str],
    allow_percent_only: bool,
) -> List[str]:
    if raw_sample is None or raw_sample.empty or cleaned_sample is None or cleaned_sample.empty:
        return []
    issues: List[str] = []
    for col in cleaned_header:
        if col not in raw_sample.columns or col in percent_like:
            continue
        if col not in cleaned_sample.columns:
            continue
        raw_vals = pd.to_numeric(raw_sample[col], errors="coerce")
        cleaned_vals = pd.to_numeric(cleaned_sample[col], errors="coerce")
        if raw_vals.dropna().empty or cleaned_vals.dropna().empty:
            continue
        raw_max = float(raw_vals.max())
        cleaned_max = float(cleaned_vals.max())
        if raw_max >= 80 and cleaned_max <= 1.5 and allow_percent_only:
            issues.append(f"{col} appears scaled from 0-100 to 0-1 but is not percent-like")
    return issues


def _scan_code_for_rescale_ops() -> List[str]:
    path = os.path.join("artifacts", "data_engineer_last.py")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            code = handle.read()
    except Exception:
        return []
    stripped = extract_code_block(code)
    text = stripped if stripped.strip() else code
    matches: List[str] = []
    patterns = [
        r"df\[['\"](?P<col>[^'\"]+)['\"]\]\s*=\s*df\[['\"](?P=col)['\"]\]\s*/\s*100",
        r"df\[['\"](?P<col>[^'\"]+)['\"]\]\s*=\s*df\[['\"](?P=col)['\"]\]\s*\*\s*0\.01",
        r"df\[['\"](?P<col>[^'\"]+)['\"]\]\s*=\s*df\[['\"](?P=col)['\"]\]\.div\(\s*100",
        r"df\[['\"](?P<col>[^'\"]+)['\"]\]\s*=\s*df\[['\"](?P=col)['\"]\]\.mul\(\s*100",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            col = match.group("col")
            if col:
                matches.append(col)
    if "MinMaxScaler" in text:
        matches.append("__MINMAX__")
    return matches


def _check_no_synthetic_data(manifest: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    warnings = manifest.get("warnings") if isinstance(manifest, dict) else []
    if isinstance(warnings, list):
        for warning in warnings:
            if "synthetic" in str(warning).lower():
                issues.append("Manifest reports synthetic data usage")
                break

    path = os.path.join("artifacts", "data_engineer_last.py")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                code = handle.read()
        except Exception:
            code = ""
        stripped = extract_code_block(code)
        text = stripped if stripped.strip() else code
        if _detect_synthetic_patterns(text):
            issues.append("Cleaning script appears to generate synthetic data")
    return issues


def _detect_synthetic_patterns(code: str) -> bool:
    lowered = code.lower()
    if "faker" in lowered or "make_classification" in lowered or "make_regression" in lowered:
        return True
    if "sklearn.datasets.make_" in lowered:
        return True
    if re.search(r"pd\.dataframe\([^)]*np\.random", lowered, re.DOTALL):
        return True
    if re.search(r"df\[['\"][^'\"]+['\"]\]\s*=\s*.*np\.random", lowered):
        return True
    return False


def _check_row_count_sanity(manifest: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    if not isinstance(manifest, dict):
        return []
    rows_before = manifest.get("rows_before")
    rows_after = manifest.get("rows_after")
    row_counts = manifest.get("row_counts") or {}
    if rows_before is None:
        rows_before = row_counts.get("initial")
    if rows_after is None:
        rows_after = row_counts.get("final")
    if not isinstance(rows_before, (int, float)) or not isinstance(rows_after, (int, float)):
        return []
    if rows_before <= 0:
        return []
    max_drop_pct = float(params.get("max_drop_pct", 5.0))
    max_dup_increase_pct = float(params.get("max_dup_increase_pct", 1.0))
    issues: List[str] = []
    if rows_after <= rows_before:
        drop_pct = (rows_before - rows_after) / rows_before * 100.0
        if drop_pct > max_drop_pct:
            issues.append(f"Row drop {drop_pct:.2f}% exceeds {max_drop_pct:.2f}%")
    else:
        increase_pct = (rows_after - rows_before) / rows_before * 100.0
        if increase_pct > max_dup_increase_pct:
            issues.append(f"Row increase {increase_pct:.2f}% exceeds {max_dup_increase_pct:.2f}%")
    return issues


def _string_values(sample: Optional[pd.DataFrame], col: str) -> List[str]:
    if sample is None or col not in sample.columns:
        return []
    values: List[str] = []
    for val in sample[col].tolist():
        if val is None:
            continue
        text = str(val).strip()
        if not text or text.lower() == "nan":
            continue
        values.append(text)
    return values


def _columns_with_role_tokens(column_roles: Dict[str, List[str]], tokens: set[str]) -> List[str]:
    cols: List[str] = []
    for role, names in column_roles.items():
        if any(token in role.lower() for token in tokens):
            cols.extend(names)
    return cols


def _map_status_value(status: Any) -> str | None:
    if status is None:
        return None
    raw = str(status).strip()
    if not raw:
        return None
    if raw in {"APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"}:
        return raw
    normalized = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    normalized = re.sub(r"_+", "_", normalized)
    if normalized in {"approved", "approve"}:
        return "APPROVED"
    if normalized in {"rejected", "reject", "failed", "fail"}:
        return "REJECTED"
    if "warn" in normalized and "approve" in normalized:
        return "APPROVE_WITH_WARNINGS"
    if normalized in {"approved_with_warning", "approved_with_warnings", "approve_with_warning", "approve_with_warnings"}:
        return "APPROVE_WITH_WARNINGS"
    return None


def normalize_cleaning_reviewer_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "status": "REJECTED",
            "feedback": "Cleaning reviewer returned invalid status.",
            "failed_checks": [],
            "required_fixes": [],
        }

    status_raw = result.get("status")
    status_exact = str(status_raw).strip() if status_raw is not None else ""
    mapped = _map_status_value(status_raw)
    normalized_applied = False

    if status_exact in {"APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"}:
        result["status"] = status_exact
    elif mapped:
        result["status"] = mapped
        normalized_applied = True
    else:
        result["status"] = "REJECTED"

    for field in ["failed_checks", "required_fixes"]:
        val = result.get(field, [])
        if isinstance(val, str):
            result[field] = [val]
        elif not isinstance(val, list):
            result[field] = []
    if "feedback" not in result:
        result["feedback"] = ""

    if normalized_applied:
        if "STATUS_ENUM_NORMALIZED" not in result["failed_checks"]:
            result["failed_checks"].append("STATUS_ENUM_NORMALIZED")

    if not mapped and status_exact not in {"APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"}:
        essential_missing = not result.get("feedback") and not result["failed_checks"] and not result["required_fixes"]
        if essential_missing:
            result["feedback"] = "Cleaning reviewer returned invalid status."

    if result.get("required_fixes"):
        if result.get("status") in {"APPROVED", "APPROVE_WITH_WARNINGS"}:
            result["status"] = "REJECTED"
            if result["feedback"]:
                result["feedback"] = result["feedback"] + " Status corrected due to required fixes."
            else:
                result["feedback"] = "Status corrected due to required fixes."

    return result
