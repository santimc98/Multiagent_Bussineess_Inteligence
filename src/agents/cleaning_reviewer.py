import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.code_extract import extract_code_block

load_dotenv()

_CLEANING_FALLBACK_WARNING = (
    "CONTRACT_BROKEN_FALLBACK: cleaning_gates missing; please fix contract generation"
)
_LLM_DISABLED_WARNING = "LLM_DISABLED_NO_API_KEY"
_LLM_PARSE_WARNING = "LLM_PARSE_FAILED"
_LLM_CALL_WARNING = "LLM_CALL_FAILED"

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
    LLM-driven cleaning reviewer with deterministic evidence checks.
    Falls back to deterministic mode if MIMO_API_KEY is unavailable.
    """

    def __init__(self, api_key: Any = None):
        self.api_key = api_key or os.getenv("MIMO_API_KEY")
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.xiaomimimo.com/v1",
            )
        else:
            self.client = None
        self.model_name = "mimo-v2-flash"
        self.last_prompt = None
        self.last_response = None

    def review_cleaning(self, *args, **kwargs) -> Dict[str, Any]:
        try:
            request = _parse_review_inputs(args, kwargs)
            result, prompt, response = _review_cleaning_impl(
                cleaning_view=request["cleaning_view"],
                cleaned_csv_path=request["cleaned_csv_path"],
                cleaning_manifest_path=request["cleaning_manifest_path"],
                raw_csv_path=request.get("raw_csv_path"),
                client=self.client,
                model_name=self.model_name,
            )
        except Exception as exc:
            result = _exception_result(exc)
            prompt = "cleaning_reviewer_exception"
            response = str(exc)

        self.last_prompt = prompt
        self.last_response = response
        return result


def _parse_review_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs.get("cleaned_csv_path"):
        return _parse_legacy_context(args[0])

    cleaning_view = args[0] if args else kwargs.get("cleaning_view") or {}
    if not isinstance(cleaning_view, dict):
        cleaning_view = {}
    input_dialect = kwargs.get("input_dialect")
    output_dialect = kwargs.get("output_dialect")
    if isinstance(input_dialect, dict) and not cleaning_view.get("input_dialect"):
        cleaning_view["input_dialect"] = input_dialect
    if isinstance(output_dialect, dict) and not cleaning_view.get("output_dialect"):
        cleaning_view["output_dialect"] = output_dialect
    if isinstance(kwargs.get("dialect"), dict) and not cleaning_view.get("dialect"):
        cleaning_view["dialect"] = kwargs.get("dialect")
    cleaned_csv_path = (
        (args[1] if len(args) > 1 else None)
        or kwargs.get("cleaned_csv_path")
        or kwargs.get("cleaned_path")
        or "data/cleaned_data.csv"
    )
    cleaning_manifest_path = (
        (args[2] if len(args) > 2 else None)
        or kwargs.get("cleaning_manifest_path")
        or kwargs.get("manifest_path")
        or "data/cleaning_manifest.json"
    )
    raw_csv_path = (
        (args[3] if len(args) > 3 else None)
        or kwargs.get("raw_csv_path")
        or kwargs.get("raw_path")
    )
    return {
        "cleaning_view": cleaning_view,
        "cleaned_csv_path": str(cleaned_csv_path),
        "cleaning_manifest_path": str(cleaning_manifest_path),
        "raw_csv_path": str(raw_csv_path) if raw_csv_path else None,
    }


def _parse_legacy_context(context: Dict[str, Any]) -> Dict[str, Any]:
    context = context if isinstance(context, dict) else {}
    cleaning_view = context.get("cleaning_view") if isinstance(context.get("cleaning_view"), dict) else {}

    if not cleaning_view.get("cleaning_gates") and isinstance(context.get("cleaning_gates"), list):
        cleaning_view["cleaning_gates"] = context.get("cleaning_gates")
    if not cleaning_view.get("required_columns") and context.get("required_columns"):
        cleaning_view["required_columns"] = context.get("required_columns")
    if not cleaning_view.get("input_dialect"):
        input_dialect = context.get("input_dialect")
        if isinstance(input_dialect, dict):
            cleaning_view["input_dialect"] = input_dialect
    if not cleaning_view.get("output_dialect"):
        output_dialect = context.get("output_dialect")
        if isinstance(output_dialect, dict):
            cleaning_view["output_dialect"] = output_dialect
    if not cleaning_view.get("dialect"):
        legacy_dialect = context.get("dialect")
        if isinstance(legacy_dialect, dict):
            cleaning_view["dialect"] = legacy_dialect
            if not cleaning_view.get("input_dialect"):
                cleaning_view["input_dialect"] = legacy_dialect
    if not cleaning_view.get("column_roles") and isinstance(context.get("column_roles"), dict):
        cleaning_view["column_roles"] = context.get("column_roles")

    cleaned_csv_path = (
        context.get("cleaned_csv_path")
        or context.get("cleaned_path")
        or context.get("cleaned_csv")
        or "data/cleaned_data.csv"
    )
    cleaning_manifest_path = (
        context.get("cleaning_manifest_path")
        or context.get("manifest_path")
        or "data/cleaning_manifest.json"
    )
    raw_csv_path = context.get("raw_csv_path") or context.get("raw_path")
    return {
        "cleaning_view": cleaning_view,
        "cleaned_csv_path": str(cleaned_csv_path),
        "cleaning_manifest_path": str(cleaning_manifest_path),
        "raw_csv_path": str(raw_csv_path) if raw_csv_path else None,
    }


def _review_cleaning_impl(
    cleaning_view: Dict[str, Any],
    cleaned_csv_path: str,
    cleaning_manifest_path: str,
    raw_csv_path: Optional[str],
    client: Optional[OpenAI],
    model_name: str,
) -> Tuple[Dict[str, Any], str, str]:
    view = cleaning_view if isinstance(cleaning_view, dict) else {}
    gates, contract_source_used, warnings = _merge_cleaning_gates(view)
    gate_names = [gate["name"] for gate in gates]

    required_columns = _list_str(view.get("required_columns"))
    column_roles = _coerce_roles(view.get("column_roles"))

    manifest = _load_json(cleaning_manifest_path)
    dialect_in_context = _resolve_dialect(view.get("dialect"))
    dialect_raw = _resolve_dialect(view.get("input_dialect") or view.get("dialect") or dialect_in_context)
    dialect_cleaned = _resolve_dialect(
        view.get("output_dialect")
        or _extract_output_dialect_from_manifest(manifest)
        or view.get("dialect")
    )
    dialect_warnings: List[str] = []

    cleaned_header = _read_csv_header(cleaned_csv_path, dialect_cleaned)
    if cleaned_header and len(cleaned_header) == 1:
        header_text = str(cleaned_header[0])
        if any(token in header_text for token in [",", ";", "\t", "|"]):
            inferred_sep = _infer_delimiter_from_file(cleaned_csv_path)
            if inferred_sep and inferred_sep != dialect_cleaned.get("sep"):
                encoding = dialect_cleaned.get("encoding") or _infer_encoding(cleaned_csv_path)
                sample_text = _read_text_sample(cleaned_csv_path, encoding, 50000)
                inferred_decimal = _infer_decimal_from_sample(sample_text, inferred_sep)
                dialect_cleaned = _resolve_dialect(
                    {"sep": inferred_sep, "decimal": inferred_decimal, "encoding": encoding}
                )
                dialect_warnings.append(
                    "DIALECT_AUTO_INFERRED_FOR_CLEANED: input dialect mismatched; "
                    f"inferred sep={inferred_sep}, decimal={inferred_decimal}"
                )
                cleaned_header = _read_csv_header(cleaned_csv_path, dialect_cleaned)

    sample_str = _read_csv_sample(cleaned_csv_path, dialect_cleaned, cleaned_header, dtype=str, nrows=400)
    sample_infer = _read_csv_sample(cleaned_csv_path, dialect_cleaned, cleaned_header, dtype=None, nrows=400)
    raw_sample = None
    if raw_csv_path:
        raw_sample = _read_csv_sample(raw_csv_path, dialect_raw, None, dtype=str, nrows=200)

    facts = _build_facts(
        cleaned_header=cleaned_header,
        required_columns=required_columns,
        manifest=manifest,
        sample_str=sample_str,
        sample_infer=sample_infer,
        raw_sample=raw_sample,
    )

    deterministic = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=required_columns,
        cleaned_header=cleaned_header,
        cleaned_csv_path=cleaned_csv_path,
        sample_str=sample_str,
        sample_infer=sample_infer,
        manifest=manifest,
        raw_sample=raw_sample,
        column_roles=column_roles,
        allowed_feature_sets=view.get("allowed_feature_sets") or {},
    )
    warnings.extend(dialect_warnings)
    deterministic_result = _assemble_result(
        deterministic,
        gate_names,
        warnings,
        contract_source_used,
    )

    if not client:
        deterministic_result["warnings"].append(_LLM_DISABLED_WARNING)
        return normalize_cleaning_reviewer_result(deterministic_result), "LLM_DISABLED", "deterministic"

    prompt, payload = _build_llm_prompt(
        gates=gates,
        required_columns=required_columns,
        dialect=dialect_cleaned,
        column_roles=column_roles,
        facts=facts,
        deterministic_gate_results=deterministic["gate_results"],
        contract_source_used=contract_source_used,
    )
    print(f"DEBUG: Cleaning Reviewer calling MIMO ({model_name})...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content
    except Exception as exc:
        deterministic_result["warnings"].append(f"{_LLM_CALL_WARNING}: {exc}")
        return normalize_cleaning_reviewer_result(deterministic_result), prompt, str(exc)

    parsed = _parse_llm_json(content)
    if not parsed:
        deterministic_result["warnings"].append(_LLM_PARSE_WARNING)
        return normalize_cleaning_reviewer_result(deterministic_result), prompt, content

    merged = _merge_llm_with_deterministic(
        llm_result=parsed,
        deterministic=deterministic,
        gate_names=gate_names,
        contract_source_used=contract_source_used,
        warnings=warnings,
    )
    return normalize_cleaning_reviewer_result(merged), prompt, content


def _exception_result(exc: Exception) -> Dict[str, Any]:
    return {
        "status": "REJECTED",
        "feedback": f"Cleaning reviewer exception: {exc}",
        "failed_checks": ["CLEANING_REVIEWER_EXCEPTION"],
        "required_fixes": ["Investigate cleaning reviewer failure."],
        "warnings": [str(exc)],
        "cleaning_gates_evaluated": [],
        "hard_failures": ["CLEANING_REVIEWER_EXCEPTION"],
        "soft_failures": [],
        "gate_results": [],
        "contract_source_used": "error",
    }


def _merge_cleaning_gates(view: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    contract_gates = _normalize_cleaning_gates(view.get("cleaning_gates"))
    universal_gates = _normalize_cleaning_gates(_FALLBACK_CLEANING_GATES)
    warnings: List[str] = []
    if contract_gates:
        merged = _dedupe_gates(contract_gates)
        source = "cleaning_view"
    else:
        merged = universal_gates
        source = "fallback"
        warnings.append(_CLEANING_FALLBACK_WARNING)
    return merged, source, warnings


def _normalize_cleaning_gates(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for gate in raw:
        if isinstance(gate, dict):
            name = _normalize_gate_name(gate.get("name") or gate.get("id") or gate.get("gate"))
            if not name:
                continue
            severity = _normalize_severity(gate.get("severity"), gate.get("required"))
            params = gate.get("params")
            if not isinstance(params, dict):
                params = {}
            if name in seen:
                continue
            seen.add(name)
            normalized.append({"name": name, "severity": severity, "params": params})
        elif isinstance(gate, str):
            name = _normalize_gate_name(gate)
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append({"name": name, "severity": "HARD", "params": {}})
    return normalized


def _dedupe_gates(gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for gate in gates:
        key = _normalize_gate_name(gate.get("name", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(gate)
    return deduped


def _normalize_severity(severity: Any, required: Any = None) -> str:
    if severity is None and required is not None:
        severity = "HARD" if bool(required) else "SOFT"
    sev = str(severity).strip().upper() if severity else "HARD"
    return sev if sev in {"HARD", "SOFT"} else "HARD"


def normalize_gate_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


def _normalize_gate_name(name: Any) -> str:
    if name is None:
        return ""
    key = normalize_gate_name(str(name))
    if not key:
        return ""
    alias_map = {
        "numericparsingvalidation": "numeric_parsing_validation",
        "numeric_parsing_validation": "numeric_parsing_validation",
        "numeric_parsing_verification": "numeric_parsing_validation",
        "numericparsingverification": "numeric_parsing_validation",
        "numeric_parsing_check": "numeric_parsing_validation",
    }
    return alias_map.get(key, key)


def _resolve_dialect(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"sep": ",", "decimal": ".", "encoding": "utf-8"}
    return {
        "sep": raw.get("sep", ","),
        "decimal": raw.get("decimal", "."),
        "encoding": raw.get("encoding", "utf-8"),
    }


def _extract_output_dialect_from_manifest(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(manifest, dict):
        return None
    output = manifest.get("output_dialect")
    return output if isinstance(output, dict) else None


def _infer_encoding(path: str) -> str:
    encodings = ("utf-8", "latin-1", "cp1252")
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as handle:
                handle.read(2048)
            return enc
        except Exception:
            continue
    return "utf-8"


def _read_text_sample(path: str, encoding: str, max_bytes: int) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding=encoding, errors="replace") as handle:
            return handle.read(max_bytes)
    except Exception:
        return ""


def _infer_delimiter_from_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    encoding = _infer_encoding(path)
    sample = _read_text_sample(path, encoding, 50000)
    if not sample:
        return None
    delimiters = [",", ";", "\t", "|"]
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=delimiters)
        if getattr(sniffed, "delimiter", None):
            return sniffed.delimiter
    except Exception:
        pass
    counts = {delim: sample.count(delim) for delim in delimiters}
    best = max(counts, key=counts.get)
    return best if counts.get(best, 0) > 0 else None


def _infer_decimal_from_sample(sample: str, sep: str) -> str:
    if not sample:
        return "."
    comma_hits = len(re.findall(r"\d+,\d+", sample))
    dot_hits = len(re.findall(r"\d+\.\d+", sample))
    if comma_hits > dot_hits:
        return ","
    if dot_hits > comma_hits:
        return "."
    return "."


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
    nrows: int,
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


def _clean_numeric_strings(values: pd.Series) -> pd.Series:
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    cleaned = cleaned[~cleaned.str.lower().isin({"nan", "none", "null"})]
    cleaned = cleaned.str.replace(" ", "", regex=False)
    return cleaned


def _best_numeric_parse_ratio(values: pd.Series) -> float:
    cleaned = _clean_numeric_strings(values)
    if cleaned.empty:
        return 1.0
    candidates = [
        cleaned,
        cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        cleaned.str.replace(",", "", regex=False),
    ]
    best_ratio = 0.0
    total = float(len(cleaned))
    for candidate in candidates:
        parsed = pd.to_numeric(candidate, errors="coerce")
        ratio = float(parsed.notna().sum()) / total if total else 1.0
        if ratio > best_ratio:
            best_ratio = ratio
    return best_ratio


def _check_numeric_parsing_validation(
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    params: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    columns = _list_str(params.get("columns"))
    threshold = float(params.get("min_parse_ratio", 0.9))
    check_mode = str(params.get("check") or "no_string_remainders").strip().lower()
    evidence: Dict[str, Any] = {"check": check_mode, "threshold": threshold, "ratios": {}}
    if not columns:
        evidence["note"] = "no_columns_configured"
        return [], evidence

    issues: List[str] = []
    for col in columns:
        if sample_infer is not None and col in sample_infer.columns:
            if pd.api.types.is_numeric_dtype(sample_infer[col]):
                evidence["ratios"][col] = 1.0
                continue
        if sample_str is None or col not in sample_str.columns:
            evidence["ratios"][col] = None
            continue
        ratio = _best_numeric_parse_ratio(sample_str[col])
        evidence["ratios"][col] = round(ratio, 4)
        if ratio < threshold:
            issues.append(f"{col} parseable_ratio={ratio:.2f} < {threshold:.2f}")
    return issues, evidence


def _compute_null_fraction(
    sample_infer: Optional[pd.DataFrame],
    sample_str: Optional[pd.DataFrame],
    column: str,
) -> Optional[float]:
    series = None
    if sample_infer is not None and column in sample_infer.columns:
        series = sample_infer[column]
    elif sample_str is not None and column in sample_str.columns:
        series = sample_str[column]
    if series is None:
        return None
    total = len(series)
    if total == 0:
        return None
    return float(series.isna().sum() / total)


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


def _build_facts(
    cleaned_header: List[str],
    required_columns: List[str],
    manifest: Dict[str, Any],
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    raw_sample: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    facts["cleaned_header"] = cleaned_header[:200]
    facts["required_columns"] = required_columns
    facts["missing_required_columns"] = [c for c in required_columns if c not in cleaned_header]

    row_counts = manifest.get("row_counts") or {}
    facts["row_counts"] = {
        "rows_before": manifest.get("rows_before") or row_counts.get("initial"),
        "rows_after": manifest.get("rows_after") or row_counts.get("final"),
    }
    conversions = manifest.get("conversions") if isinstance(manifest, dict) else {}
    dropped = manifest.get("dropped_columns") if isinstance(manifest, dict) else {}
    warnings = manifest.get("warnings") if isinstance(manifest, dict) else []
    facts["manifest_summary"] = {
        "conversions_count": len(conversions) if isinstance(conversions, dict) else 0,
        "dropped_columns_count": len(dropped) if isinstance(dropped, (list, dict)) else 0,
        "warnings": [str(w) for w in warnings[:5]] if isinstance(warnings, list) else [],
    }
    facts["column_stats_sample"] = _build_column_stats(sample_str, sample_infer, max_cols=40)
    facts["cleaned_sample_rows"] = _sample_rows(sample_str, max_rows=5)
    facts["raw_sample_rows"] = _sample_rows(raw_sample, max_rows=5)
    return facts


def _build_column_stats(
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    max_cols: int,
) -> List[Dict[str, Any]]:
    if sample_str is None or sample_str.empty:
        return []
    columns = list(sample_str.columns)[:max_cols]
    stats: List[Dict[str, Any]] = []
    for col in columns:
        series = sample_str[col]
        total = len(series)
        null_frac = float(series.isna().sum() / total) if total else 0.0
        unique_count = int(series.nunique(dropna=True)) if total else 0
        examples = [str(v) for v in series.dropna().head(3).tolist()]
        dtype = "unknown"
        if sample_infer is not None and col in sample_infer.columns:
            dtype = str(sample_infer[col].dtype)
        stats.append(
            {
                "column": col,
                "dtype": dtype,
                "null_frac": round(null_frac, 4),
                "unique_count": unique_count,
                "examples": examples,
            }
        )
    return stats


def _sample_rows(sample_df: Optional[pd.DataFrame], max_rows: int) -> List[Dict[str, Any]]:
    if sample_df is None or sample_df.empty:
        return []
    rows = sample_df.head(max_rows).to_dict(orient="records")
    return rows if isinstance(rows, list) else []


def _evaluate_gates_deterministic(
    gates: List[Dict[str, Any]],
    required_columns: List[str],
    cleaned_header: List[str],
    cleaned_csv_path: str,
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    manifest: Dict[str, Any],
    raw_sample: Optional[pd.DataFrame],
    column_roles: Dict[str, List[str]],
    allowed_feature_sets: Dict[str, Any],
) -> Dict[str, Any]:
    hard_failures: List[str] = []
    soft_failures: List[str] = []
    failed_checks: List[str] = []
    required_fixes: List[str] = []
    warnings: List[str] = []
    failure_summaries: List[str] = []
    warning_summaries: List[str] = []
    gate_results: List[Dict[str, Any]] = []
    model_features = _list_str(allowed_feature_sets.get("model_features")) if isinstance(allowed_feature_sets, dict) else []

    for gate in gates:
        name = gate["name"]
        gate_key = _normalize_gate_name(name)
        severity = gate["severity"]
        params = gate["params"]
        issues: List[str] = []
        evidence: Dict[str, Any] = {}
        evaluated = True
        severity_used = severity

        if gate_key == "required_columns_present":
            issues = _check_required_columns(required_columns, cleaned_header, cleaned_csv_path)
            evidence["missing"] = [c for c in required_columns if c not in cleaned_header]
        elif gate_key == "id_integrity":
            issues = _check_id_integrity(
                cleaned_header,
                sample_str,
                sample_infer,
                params,
                column_roles,
            )
        elif gate_key == "no_semantic_rescale":
            issues = _check_no_semantic_rescale(
                manifest,
                params,
                cleaned_header,
                column_roles,
                raw_sample,
                sample_infer if sample_infer is not None else sample_str,
            )
        elif gate_key == "no_synthetic_data":
            issues = _check_no_synthetic_data(manifest)
        elif gate_key == "row_count_sanity":
            issues = _check_row_count_sanity(manifest, params)
            evidence["row_counts"] = (manifest.get("row_counts") or {})
        elif gate_key == "numeric_parsing_validation":
            issues, evidence = _check_numeric_parsing_validation(sample_str, sample_infer, params)
        elif gate_key == "null_handling_verification":
            columns = _list_str(params.get("columns"))
            allow_nulls = params.get("allow_nulls", True)
            tolerance = params.get("tolerance")
            if tolerance is None:
                tolerance = params.get("null_tolerance", 0.0)
            try:
                tolerance_val = float(tolerance)
            except Exception:
                tolerance_val = 0.0
            evidence = {
                "columns": columns,
                "allow_nulls": bool(allow_nulls),
                "tolerance": tolerance_val,
                "null_frac": {},
            }
            if not columns:
                evidence["note"] = "no_columns_configured"
            for col in columns:
                if cleaned_header and col not in cleaned_header:
                    issues.append(f"Missing column: {col}")
                    severity_used = "HARD"
                    continue
                null_frac = _compute_null_fraction(sample_infer, sample_str, col)
                evidence["null_frac"][col] = None if null_frac is None else round(null_frac, 4)
                if null_frac is None:
                    warnings.append(f"NULL_HANDLING_EVIDENCE_MISSING: {col}")
                    continue
                if allow_nulls is False and null_frac > tolerance_val:
                    issues.append(f"{col} null_frac={null_frac:.4f} > {tolerance_val:.4f}")
            warn_threshold = params.get("warn_null_frac_threshold", 0.5)
            try:
                warn_threshold_val = float(warn_threshold)
            except Exception:
                warn_threshold_val = 0.5
            for col in model_features:
                if col in columns:
                    continue
                if cleaned_header and col not in cleaned_header:
                    continue
                null_frac = _compute_null_fraction(sample_infer, sample_str, col)
                if null_frac is None or null_frac < warn_threshold_val:
                    continue
                warnings.append(f"NULLS_OUTSIDE_GATE: {col} null_frac={null_frac:.4f}")
        else:
            evaluated = False
            warnings.append(f"UNKNOWN_GATE_SKIPPED: {name}")

        passed = None
        if evaluated:
            passed = not issues
        gate_results.append(
            {
                "name": name,
                "severity": severity_used,
                "passed": passed,
                "issues": issues,
                "evidence": evidence or {"source": "deterministic"},
            }
        )

        if issues:
            _record_gate_failure(
                _normalize_gate_name(name),
                severity_used,
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
        warnings.extend(warning_summaries)

    return {
        "status": status,
        "feedback": feedback,
        "failed_checks": failed_checks,
        "required_fixes": required_fixes,
        "warnings": warnings,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "gate_results": gate_results,
    }


def _assemble_result(
    deterministic: Dict[str, Any],
    gate_names: List[str],
    warnings: List[str],
    contract_source_used: str,
) -> Dict[str, Any]:
    result = dict(deterministic)
    result["warnings"] = _dedupe_list(list(warnings) + result.get("warnings", []))
    result["cleaning_gates_evaluated"] = gate_names
    result["contract_source_used"] = contract_source_used
    result.setdefault("gate_results", [])
    result.setdefault("hard_failures", [])
    result.setdefault("soft_failures", [])
    return result


def _build_llm_prompt(
    gates: List[Dict[str, Any]],
    required_columns: List[str],
    dialect: Dict[str, Any],
    column_roles: Dict[str, List[str]],
    facts: Dict[str, Any],
    deterministic_gate_results: List[Dict[str, Any]],
    contract_source_used: str,
) -> Tuple[str, Dict[str, Any]]:
    system_prompt = (
        "You are a Senior Data Cleaning Reviewer.\n"
        "Evaluate ONLY the gates listed in cleaning_gates.\n"
        "Use the provided evidence and samples; do not invent data.\n"
        "If a HARD gate fails, status must be REJECTED.\n"
        "If only SOFT gates fail, status must be APPROVE_WITH_WARNINGS.\n"
        "If no gates fail, status must be APPROVED.\n"
        "When evidence is insufficient, mark the gate as not failed and add a warning.\n"
        "Return JSON only with the specified schema.\n"
        "\n"
        "Required JSON schema:\n"
        "{\n"
        '  "status": "APPROVED" | "APPROVE_WITH_WARNINGS" | "REJECTED",\n'
        '  "feedback": "string",\n'
        '  "failed_checks": ["gate_name", ...],\n'
        '  "required_fixes": ["actionable fix", ...],\n'
        '  "warnings": ["warning", ...],\n'
        '  "hard_failures": ["gate_name", ...],\n'
        '  "soft_failures": ["gate_name", ...],\n'
        '  "gate_results": [\n'
        "     {\n"
        '       "name": "gate_name",\n'
        '       "severity": "HARD|SOFT",\n'
        '       "passed": true|false,\n'
        '       "issues": ["issue", ...],\n'
        '       "evidence": "short evidence string"\n'
        "     }\n"
        "  ],\n"
        '  "contract_source_used": "cleaning_view|fallback|merged"\n'
        "}\n"
    )

    payload = {
        "cleaning_gates": gates,
        "required_columns": required_columns,
        "dialect": dialect,
        "column_roles": column_roles,
        "facts": facts,
        "deterministic_gate_results": deterministic_gate_results,
        "contract_source_used": contract_source_used,
    }
    return system_prompt, payload


def _parse_llm_json(content: str) -> Optional[Dict[str, Any]]:
    if not isinstance(content, str):
        return None
    cleaned = extract_code_block(content)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _merge_llm_with_deterministic(
    llm_result: Dict[str, Any],
    deterministic: Dict[str, Any],
    gate_names: List[str],
    contract_source_used: str,
    warnings: List[str],
) -> Dict[str, Any]:
    llm = _normalize_llm_result(llm_result)
    det = deterministic

    gate_results = _merge_gate_results(
        llm.get("gate_results", []),
        det.get("gate_results", []),
        gate_names,
    )
    summary = _summarize_gate_results(gate_results)
    merged_warnings = _dedupe_list(
        warnings + det.get("warnings", []) + llm.get("warnings", []) + summary["warning_summaries"]
    )
    required_fixes = _dedupe_list(summary["required_fixes"] + det.get("required_fixes", []))

    return {
        "status": summary["status"],
        "feedback": summary["feedback"],
        "failed_checks": summary["failed_checks"],
        "required_fixes": required_fixes,
        "warnings": merged_warnings,
        "hard_failures": summary["hard_failures"],
        "soft_failures": summary["soft_failures"],
        "gate_results": gate_results,
        "cleaning_gates_evaluated": gate_names,
        "contract_source_used": contract_source_used,
    }


def _normalize_llm_result(result: Dict[str, Any]) -> Dict[str, Any]:
    out = result if isinstance(result, dict) else {}
    out.setdefault("failed_checks", [])
    out.setdefault("required_fixes", [])
    out.setdefault("warnings", [])
    out.setdefault("hard_failures", [])
    out.setdefault("soft_failures", [])
    out.setdefault("gate_results", [])
    return out


def _merge_gate_lists(primary: List[Any], secondary: List[Any], allowed: List[str]) -> List[str]:
    allowed_norm = {_normalize_gate_name(name) for name in allowed if name}
    merged = _dedupe_list([str(item) for item in primary + secondary if item])
    filtered = [item for item in merged if _normalize_gate_name(item) in allowed_norm]
    return filtered


def _merge_gate_results(
    llm_results: List[Any],
    det_results: List[Any],
    gate_names: List[str],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in det_results:
        if not isinstance(entry, dict):
            continue
        name = _normalize_gate_name(entry.get("name", ""))
        if not name:
            continue
        normalized = dict(entry)
        normalized["name"] = name
        merged[name] = normalized
    for entry in llm_results:
        if not isinstance(entry, dict):
            continue
        name = _normalize_gate_name(entry.get("name", ""))
        if not name:
            continue
        existing = merged.get(name)
        if existing and existing.get("passed") is not None:
            if not existing.get("evidence") and entry.get("evidence"):
                existing["evidence"] = entry.get("evidence")
            continue
        if existing:
            updated = dict(existing)
            updated.update(entry)
        else:
            updated = dict(entry)
        updated["name"] = name
        merged[name] = updated
    ordered = []
    for gate in gate_names:
        key = _normalize_gate_name(gate)
        if key in merged:
            entry = merged[key]
            entry["name"] = gate
            ordered.append(entry)
        else:
            ordered.append(
                {
                    "name": gate,
                    "severity": "HARD",
                    "passed": None,
                    "issues": [],
                    "evidence": "no_result",
                }
            )
    return ordered


def _summarize_gate_results(gate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    hard_failures: List[str] = []
    soft_failures: List[str] = []
    failed_checks: List[str] = []
    required_fixes: List[str] = []
    failure_summaries: List[str] = []
    warning_summaries: List[str] = []

    for gate in gate_results:
        name = str(gate.get("name", "")).strip()
        passed = gate.get("passed")
        if passed is not False:
            continue
        severity = str(gate.get("severity", "HARD")).strip().upper()
        if severity not in {"HARD", "SOFT"}:
            severity = "HARD"
        issues = gate.get("issues") or []
        issues_text = "; ".join(str(issue) for issue in issues if issue)
        summary = f"{name}: {issues_text}" if issues_text else f"{name}: failed"
        if name and name not in failed_checks:
            failed_checks.append(name)
        if severity == "HARD":
            if name and name not in hard_failures:
                hard_failures.append(name)
            required_fixes.append(summary)
            failure_summaries.append(summary)
        else:
            if name and name not in soft_failures:
                soft_failures.append(name)
            warning_summaries.append(summary)

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

    return {
        "status": status,
        "feedback": feedback,
        "failed_checks": failed_checks,
        "required_fixes": required_fixes,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "warning_summaries": warning_summaries,
    }


def _dedupe_list(items: List[Any]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


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
        for key in ("initial", "original", "input", "rows_before", "total"):
            if rows_before is None:
                rows_before = row_counts.get(key)
    if rows_after is None:
        for key in ("final", "after_cleaning", "output", "rows_after"):
            if rows_after is None:
                rows_after = row_counts.get(key)
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
