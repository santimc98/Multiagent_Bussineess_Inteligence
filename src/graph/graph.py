import sys
import os
import subprocess
import re
import json
import hashlib
import threading
from datetime import datetime
from typing import TypedDict, Dict, Any, List, Literal
from langgraph.graph import StateGraph, END
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
import base64

# Add src to path to allow imports if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.steward import StewardAgent
from src.agents.strategist import StrategistAgent
from src.agents.ml_engineer import MLEngineerAgent
from src.agents.business_translator import BusinessTranslatorAgent
# from src.agents.selector import SelectorAgent # DEPRECATED
from src.agents.data_engineer import DataEngineerAgent
from src.agents.cleaning_reviewer import CleaningReviewerAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.qa_reviewer import QAReviewerAgent # New QA Gate
from src.agents.execution_planner import ExecutionPlannerAgent
from src.agents.failure_explainer import FailureExplainerAgent
from src.agents.results_advisor import ResultsAdvisorAgent
from src.utils.pdf_generator import convert_report_to_pdf
from src.utils.static_safety_scan import scan_code_safety
from src.utils.dialect_guardrails import (
    assert_not_single_column_delimiter_mismatch,
    get_output_dialect_from_manifest,
)
from src.utils.leakage_sanity_audit import run_unsupervised_numeric_relation_audit
from src.utils.cleaning_validation import (
    normalize_manifest,
    sample_raw_columns,
    detect_destructive_conversions,
    format_issue_report,
)
from src.utils.integrity_audit import run_integrity_audit
from src.utils.output_contract import check_required_outputs
from src.utils.sandbox_deps import (
    BASE_ALLOWLIST,
    EXTENDED_ALLOWLIST,
    BANNED_ALLOWLIST,
    check_dependency_precheck,
    get_sandbox_install_packages,
)
from src.utils.case_alignment import build_case_alignment_report
from src.utils.contract_validation import ensure_role_runbooks
from src.utils.data_engineer_preflight import data_engineer_preflight
from src.utils.cleaning_plan import parse_cleaning_plan, validate_cleaning_plan
from src.utils.cleaning_executor import execute_cleaning_plan
from src.utils.run_logger import init_run_log, log_run_event, finalize_run_log
from src.utils.dataset_memory import (
    fingerprint_dataset,
    load_dataset_memory,
    record_dataset_memory,
    summarize_memory,
)
from src.utils.governance import write_governance_report, build_run_summary
from src.utils.data_adequacy import build_data_adequacy_report, write_data_adequacy_report
from src.utils.code_extract import is_syntax_valid

def _norm_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

def _truncate_text(text: str, max_len: int = 18000, head_len: int = 10000, tail_len: int = 6000) -> str:
    if not text:
        return text
    if len(text) <= max_len:
        return text
    safe_head = max(0, min(head_len, max_len))
    safe_tail = max(0, min(tail_len, max_len - safe_head))
    if safe_head + safe_tail == 0:
        return text[:max_len]
    if safe_head + safe_tail < max_len:
        safe_head = max_len - safe_tail
    return text[:safe_head] + "\n...[TRUNCATED]...\n" + text[-safe_tail:]

_ABORT_EVENT = threading.Event()

def request_abort(reason: str | None = None) -> None:
    _ABORT_EVENT.set()
    if reason:
        print(f"ABORT_REQUESTED: {reason}")

def clear_abort() -> None:
    _ABORT_EVENT.clear()

def abort_requested() -> bool:
    return _ABORT_EVENT.is_set()

def _abort_if_requested(state: Dict[str, Any], stage: str) -> Dict[str, Any] | None:
    if not _ABORT_EVENT.is_set():
        return None
    run_id = state.get("run_id") if isinstance(state, dict) else None
    if run_id:
        log_run_event(run_id, "abort_requested", {"stage": stage})
    return {
        "error_message": "ABORTED_BY_USER",
        "cleaned_data_preview": "Error: Aborted",
        "budget_counters": state.get("budget_counters", {}) if isinstance(state, dict) else {},
    }

def _extract_manifest_alerts(manifest: Dict[str, Any], derived_columns: List[str]) -> List[str]:
    if not isinstance(manifest, dict):
        return []

    row_counts = manifest.get("row_counts") or {}
    summary_stats = manifest.get("summary_statistics") or {}
    warnings = manifest.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = [warnings]

    cleaned_shape = summary_stats.get("cleaned_shape")
    cleaned_rows = None
    if isinstance(cleaned_shape, (list, tuple)) and cleaned_shape:
        try:
            cleaned_rows = int(cleaned_shape[0])
        except Exception:
            cleaned_rows = None

    def _is_zero(val: Any) -> bool:
        return isinstance(val, (int, float)) and val == 0

    row_candidates = [
        row_counts.get("final"),
        row_counts.get("rows_after"),
        row_counts.get("after"),
        summary_stats.get("rows_processed"),
        cleaned_rows,
    ]

    alerts: List[str] = []
    if any(_is_zero(val) for val in row_candidates):
        alerts.append(f"EMPTY_OUTPUT: row_counts={row_counts} summary_statistics={summary_stats}")

    missing_derived: List[str] = []
    if derived_columns and warnings:
        derived_norm = {_norm_name(col) for col in derived_columns if col}
        for warning in warnings:
            if not isinstance(warning, str):
                continue
            warning_text = warning
            if "Missing columns in raw data" in warning_text or "Missing required columns" in warning_text:
                parsed_cols = _parse_manifest_warning_list(warning_text)
                if parsed_cols:
                    for col in parsed_cols:
                        if _norm_name(col) in derived_norm:
                            missing_derived.append(col)
                else:
                    warning_norm = _norm_name(warning_text)
                    for col in derived_columns:
                        col_norm = _norm_name(col)
                        if col_norm and col_norm in warning_norm:
                            missing_derived.append(col)

    if missing_derived:
        alerts.append(f"MISSING_DERIVED_IN_WARNINGS: {sorted(set(missing_derived))}")

    if alerts and warnings:
        preview = warnings if len(warnings) <= 3 else warnings[:3] + ["..."]
        alerts.append(f"WARNINGS_PREVIEW: {preview}")

    return alerts

def _summarize_row_drop(
    manifest: Dict[str, Any],
    required_cols: List[str],
    min_drop_frac: float = 0.5,
    min_null_increase_frac: float = 0.2,
    initial_override: int | None = None,
    after_override: int | None = None,
) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        return {}
    row_counts = manifest.get("row_counts") or {}
    try:
        initial = int(
            row_counts.get("initial")
            or row_counts.get("rows_before")
            or row_counts.get("input")
            or row_counts.get("total")
        )
    except Exception:
        initial = None
    try:
        after = int(
            row_counts.get("after_cleaning")
            or row_counts.get("final")
            or row_counts.get("rows_after")
            or row_counts.get("after")
            or row_counts.get("output")
        )
    except Exception:
        after = None

    if initial_override is not None:
        initial = initial_override
    if after_override is not None:
        after = after_override

    if not initial or after is None or initial <= 0:
        return {}
    drop = max(initial - after, 0)
    drop_frac = drop / initial if initial else 0.0
    if drop_frac < min_drop_frac:
        return {}

    norm = normalize_manifest(manifest)
    conversions = norm.get("conversions", {}) or {}
    suspects: List[Dict[str, Any]] = []
    for col in required_cols or []:
        key = str(col).lower()
        conv = conversions.get(key)
        if not isinstance(conv, dict):
            continue
        null_inc = conv.get("null_increase")
        try:
            null_inc_val = float(null_inc)
        except Exception:
            continue
        if null_inc_val <= 0:
            continue
        if initial and null_inc_val / initial < min_null_increase_frac:
            continue
        suspects.append(
            {
                "column": col,
                "null_increase": int(null_inc_val),
                "from_dtype": conv.get("from_dtype"),
                "to_dtype": conv.get("to_dtype"),
                "method": conv.get("method"),
            }
        )

    return {
        "initial": initial,
        "after": after,
        "drop": drop,
        "drop_frac": round(drop_frac, 4),
        "suspects": suspects,
    }

def _count_raw_rows(csv_path: str, encoding: str, sep: str | None = None, decimal: str | None = None) -> int | None:
    try:
        import pandas as pd

        df = pd.read_csv(
            csv_path,
            encoding=encoding,
            sep=sep or ",",
            decimal=decimal or ".",
            usecols=[0],
            low_memory=False,
        )
        return len(df)
    except Exception:
        try:
            with open(csv_path, "r", encoding=encoding, errors="ignore") as f:
                count = sum(1 for _ in f)
            return max(count - 1, 0)
        except Exception:
            return None

def _parse_manifest_warning_list(warning: str) -> List[str]:
    import ast
    if ":" not in warning:
        return []
    list_text = warning.split(":", 1)[1].strip()
    try:
        parsed = ast.literal_eval(list_text)
    except Exception:
        return []
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return []

def _find_empty_required_columns(df, required_cols: List[str], threshold: float = 0.98) -> List[Dict[str, Any]]:
    issues = []
    if df is None or not required_cols:
        return issues
    for col in required_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if series.empty:
            continue
        try:
            if series.dtype == object:
                stripped = series.astype(str).str.strip()
                null_like = series.isna() | (stripped == "")
            else:
                null_like = series.isna()
            null_frac = float(null_like.mean()) if len(series) else 0.0
            non_null_count = int((~null_like).sum())
        except Exception:
            null_frac = 0.0
            non_null_count = 0
        if null_frac >= threshold or non_null_count == 0:
            issues.append(
                {
                    "column": col,
                    "null_frac": null_frac,
                    "non_null_count": non_null_count,
                }
            )
    return issues

def _build_required_sample_context(
    csv_path: str,
    dialect: Dict[str, Any],
    required_cols: List[str],
    norm_map: Dict[str, str],
    max_rows: int = 50,
    max_examples: int = 6,
) -> str:
    if not csv_path or not required_cols:
        return ""
    raw_cols = []
    canon_to_raw: Dict[str, str] = {}
    for col in required_cols:
        normed = _norm_name(col)
        raw = norm_map.get(normed)
        if raw:
            canon_to_raw[col] = raw
            raw_cols.append(raw)
    if not raw_cols:
        return ""
    sample_df = sample_raw_columns(csv_path, dialect, raw_cols, nrows=max_rows, dtype=str)
    if sample_df is None or getattr(sample_df, "empty", False):
        return ""

    def _pattern_stats(series) -> Dict[str, float]:
        try:
            series = series.dropna().astype(str)
        except Exception:
            return {}
        if series.empty:
            return {}
        sample = series
        percent_like = float(sample.str.contains("%").mean())
        comma_decimal = float(sample.str.contains(r"\d+,\d+").mean())
        dot_decimal = float(sample.str.contains(r"\d+\.\d+").mean())
        numeric_like = float(sample.str.contains(r"^[\s\-\+]*[\d,.\s%]+$").mean())
        whitespace = float(sample.str.contains(r"^\s+|\s+$").mean())
        return {
            "numeric_like_ratio": round(numeric_like, 4),
            "percent_like_ratio": round(percent_like, 4),
            "comma_decimal_ratio": round(comma_decimal, 4),
            "dot_decimal_ratio": round(dot_decimal, 4),
            "whitespace_ratio": round(whitespace, 4),
        }

    samples: Dict[str, Dict[str, Any]] = {}
    for canon, raw in canon_to_raw.items():
        if raw not in sample_df.columns:
            continue
        series = sample_df[raw]
        try:
            if series.dtype == object:
                series = series.astype(str).str.strip()
            values = [v for v in series.dropna().tolist() if str(v).strip() != ""]
        except Exception:
            values = []
        uniq: List[str] = []
        for val in values:
            sval = str(val)
            if sval not in uniq:
                uniq.append(sval)
            if len(uniq) >= max_examples:
                break
        samples[canon] = {
            "raw_column": raw,
            "examples": uniq,
            "pattern_stats": _pattern_stats(series),
        }

    if not samples:
        return ""
    payload = {"sample_rows": int(len(sample_df)), "columns": samples}
    return "RAW_REQUIRED_COLUMN_SAMPLES:\n" + json.dumps(payload, ensure_ascii=True)

def _extract_manifest_row_count(manifest: Dict[str, Any]) -> int | None:
    if not isinstance(manifest, dict):
        return None
    row_counts = manifest.get("row_counts")
    if isinstance(row_counts, dict):
        for key in ("after_cleaning", "final", "output", "cleaned", "rows"):
            value = row_counts.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                return int(value)
    for key in ("row_count", "rows", "n_rows", "cleaned_row_count", "output_row_count"):
        value = manifest.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None

def _estimate_row_count(csv_path: str, encoding: str = "utf-8") -> int | None:
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", encoding=encoding, errors="ignore") as f:
            total = sum(1 for _ in f)
        return max(total - 1, 0)
    except Exception:
        return None

def _build_signal_summary_context(
    csv_path: str,
    dialect: Dict[str, Any],
    required_cols: List[str],
    norm_map: Dict[str, str],
    header_cols: List[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    manifest = _load_json_safe("data/cleaning_manifest.json")
    row_count = _extract_manifest_row_count(manifest) or _estimate_row_count(csv_path, dialect.get("encoding", "utf-8"))
    if row_count is not None:
        summary["row_count"] = row_count
    if header_cols:
        summary["column_count"] = len(header_cols)
    if required_cols:
        summary["required_columns"] = required_cols
        summary["required_feature_count"] = len(required_cols)
        if row_count:
            summary["rows_per_required_feature"] = round(row_count / max(len(required_cols), 1), 2)
        raw_cols = []
        canon_to_raw: Dict[str, str] = {}
        for col in required_cols:
            normed = _norm_name(col)
            raw = norm_map.get(normed)
            if raw:
                canon_to_raw[col] = raw
                raw_cols.append(raw)
        sample_df = sample_raw_columns(csv_path, dialect, raw_cols, nrows=200, dtype=str)
        if sample_df is not None and not sample_df.empty:
            stats: Dict[str, Any] = {}
            for canon, raw in canon_to_raw.items():
                if raw not in sample_df.columns:
                    continue
                series = sample_df[raw]
                try:
                    values = series.astype(str)
                    non_null_ratio = float(values.replace("", None).notna().mean())
                    nunique = int(values.dropna().nunique())
                    examples = [str(v) for v in values.dropna().head(3).tolist()]
                except Exception:
                    non_null_ratio = None
                    nunique = None
                    examples = []
                stats[canon] = {
                    "raw_column": raw,
                    "sample_non_null_ratio": non_null_ratio,
                    "sample_nunique": nunique,
                    "sample_examples": examples,
                }
            if stats:
                summary["required_column_sample_stats"] = stats
        required_raw = set(canon_to_raw.values())
        candidate_cols = [col for col in header_cols if col not in required_raw]
        if candidate_cols:
            summary["candidate_column_count"] = len(candidate_cols)
            sample_candidates = candidate_cols[:40]
            candidate_df = sample_raw_columns(csv_path, dialect, sample_candidates, nrows=200, dtype=str)
            if candidate_df is not None and not candidate_df.empty:
                candidate_stats: List[Dict[str, Any]] = []
                for col in sample_candidates:
                    if col not in candidate_df.columns:
                        continue
                    series = candidate_df[col]
                    try:
                        values = series.astype(str)
                        non_null_ratio = float(values.replace("", None).notna().mean())
                        nunique = int(values.dropna().nunique())
                        examples = [str(v) for v in values.dropna().head(3).tolist()]
                    except Exception:
                        non_null_ratio = None
                        nunique = None
                        examples = []
                    likely_id = False
                    try:
                        sample_len = len(values.dropna())
                        if sample_len > 0 and nunique is not None:
                            likely_id = nunique >= max(int(sample_len * 0.9), 25)
                    except Exception:
                        likely_id = False
                    candidate_stats.append(
                        {
                            "column": col,
                            "sample_non_null_ratio": non_null_ratio,
                            "sample_nunique": nunique,
                            "sample_examples": examples,
                            "likely_id": likely_id,
                        }
                    )
                if candidate_stats:
                    summary["candidate_columns"] = candidate_stats
    return summary

def _build_required_raw_map(
    required_cols: List[str],
    norm_map: Dict[str, str],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col in required_cols or []:
        normed = _norm_name(col)
        raw = norm_map.get(normed)
        if raw:
            mapping[col] = raw
    return mapping

def _filter_input_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of contract with data_requirements limited to source=='input'."""
    if not isinstance(contract, dict):
        return {}
    reqs = contract.get("data_requirements", []) or []
    filtered = [r for r in reqs if isinstance(r, dict) and r.get("source", "input") == "input"]
    new_contract = dict(contract)
    new_contract["data_requirements"] = filtered
    return new_contract

def _filter_contract_for_data_engineer(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of contract without model output-only requirements."""
    if not isinstance(contract, dict):
        return {}
    reqs = contract.get("data_requirements", []) or []
    filtered = []
    canonical_cols: List[str] = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        role = (req.get("role") or "").lower()
        owner = (req.get("derived_owner") or "").lower()
        if role == "output":
            continue
        if req.get("source") == "derived" and owner == "ml_engineer":
            continue
        filtered.append(req)
        name = req.get("canonical_name") or req.get("name")
        if name:
            canonical_cols.append(name)
    new_contract = dict(contract)
    new_contract["data_requirements"] = filtered
    if canonical_cols:
        new_contract["canonical_columns"] = canonical_cols
    return new_contract

def _resolve_required_input_columns(contract: Dict[str, Any], strategy: Dict[str, Any]) -> List[str]:
    if contract and isinstance(contract, dict):
        contract_reqs = contract.get("data_requirements", []) or []
        if contract_reqs:
            resolved = []
            for req in contract_reqs:
                if not isinstance(req, dict):
                    continue
                if req.get("source", "input") != "input":
                    continue
                name = req.get("canonical_name") or req.get("name")
                if name:
                    resolved.append(name)
            return resolved
    return strategy.get("required_columns", []) if strategy else []

def _resolve_contract_columns(contract: Dict[str, Any], sources: set[str] | None = None) -> List[str]:
    if not contract or not isinstance(contract, dict):
        return []
    reqs = contract.get("data_requirements", []) or []
    out: List[str] = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        name = req.get("canonical_name") or req.get("name")
        if not name:
            continue
        source = req.get("source", "input") or "input"
        if sources is None or source in sources:
            out.append(name)
    return out

def _resolve_contract_columns_for_cleaning(contract: Dict[str, Any], sources: set[str] | None = None) -> List[str]:
    if not contract or not isinstance(contract, dict):
        return []
    reqs = contract.get("data_requirements", []) or []
    out: List[str] = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        name = req.get("canonical_name") or req.get("name")
        if not name:
            continue
        source = req.get("source", "input") or "input"
        owner = (req.get("derived_owner") or "").lower()
        if source == "derived" and owner == "ml_engineer":
            continue
        if sources is None or source in sources:
            out.append(name)
    return out

def _ensure_scored_rows_output(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    reqs = contract.get("data_requirements", []) or []
    needs_scored = any(isinstance(r, dict) and r.get("source") == "derived" for r in reqs)
    if needs_scored:
        outputs = contract.get("required_outputs", []) or []
        if "data/scored_rows.csv" not in outputs:
            outputs.append("data/scored_rows.csv")
        contract["required_outputs"] = outputs
    return contract

def _ensure_alignment_check_output(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    reqs = contract.get("alignment_requirements", [])
    if not isinstance(reqs, list) or not reqs:
        return contract
    outputs = contract.get("required_outputs", []) or []
    if "data/alignment_check.json" not in outputs:
        outputs.append("data/alignment_check.json")
    contract["required_outputs"] = outputs
    return contract

def _normalize_execution_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    normalized = dict(contract)

    quality_gates = normalized.get("quality_gates")
    if isinstance(quality_gates, list):
        normalized["quality_gates_raw"] = quality_gates
        normalized["quality_gates"] = {}
    elif not isinstance(quality_gates, dict):
        normalized["quality_gates"] = {}

    spec = normalized.get("spec_extraction")
    if spec is None or not isinstance(spec, dict):
        normalized["spec_extraction"] = {}
        spec = normalized["spec_extraction"]

    derived = spec.get("derived_columns")
    if isinstance(derived, list) and derived:
        if all(isinstance(item, str) for item in derived):
            spec["derived_columns"] = [
                {"name": name, "formula": None, "depends_on": [], "constraints": []}
                for name in derived
                if name
            ]

    if not spec.get("scoring_formula"):
        formulas = spec.get("formulas")
        if isinstance(formulas, list) and formulas:
            chosen = None
            for formula in formulas:
                if isinstance(formula, str) and "score_nuevo" in formula.lower():
                    chosen = formula
                    break
            if not chosen:
                first = formulas[0]
                chosen = first if isinstance(first, str) else None
            if chosen:
                spec["scoring_formula"] = chosen

    ba = normalized.get("business_alignment")
    if not isinstance(ba, dict):
        ba = {}
    normalized["business_alignment"] = ba

    iteration_policy = normalized.get("iteration_policy")
    if not isinstance(iteration_policy, dict):
        normalized["iteration_policy"] = {}
    if not isinstance(normalized.get("compliance_checklist"), list):
        normalized["compliance_checklist"] = []
    if not isinstance(normalized.get("alignment_requirements"), list):
        normalized["alignment_requirements"] = []

    return normalized

def _normalize_required_fixes(items: List[Any] | None) -> List[str]:
    fixes: List[str] = []
    for item in items or []:
        if item is None:
            continue
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                fixes.append(cleaned)
        else:
            fixes.append(str(item))
    return fixes

def _expand_required_fixes(required_fixes: List[Any] | None, failed_gates: List[Any] | None) -> List[str]:
    fixes = _normalize_required_fixes(required_fixes)
    tokens = [str(item) for item in (failed_gates or []) if item]
    tokens.extend(fixes)

    guidance_map = {
        "MAPPING_SUMMARY": [
            "Print `Mapping Summary: Target=<...>, Features=<...>, SegFeatures=<...>` immediately after column mapping.",
        ],
        "TARGET_VARIANCE_GUARD": [
            "Add `if y.nunique() <= 1: raise ValueError(...)` before any training or optimization.",
        ],
        "HIGH_CARDINALITY_HANDLING": [
            "Build X strictly from contract feature_cols or drop high-cardinality ID columns using nunique ratio.",
        ],
        "TARGET_NOT_IN_X": [
            "Ensure target column is excluded from features when building X.",
        ],
        "CROSS_VALIDATION_REQUIRED": [
            "Use cross-validation (StratifiedKFold/KFold) and report mean/std; avoid only train_test_split.",
        ],
        "DATA_LOAD_MISSING": [
            "Load the cleaned dataset using pd.read_csv with the detected dialect before processing.",
        ],
        "DATA_PATH_NOT_USED": [
            "Read from data/cleaned_data.csv or data/cleaned_full.csv (or data.csv in sandbox) using pd.read_csv.",
        ],
        "REQUIRED_OUTPUTS_MISSING": [
            "Write all required outputs to the exact contract paths before exiting.",
        ],
        "REQUIRED_COLUMNS_NOT_USED": [
            "Use the required business columns from the contract (e.g., Size, Debtors, Sector) in mapping/processing.",
        ],
        "SYNTHETIC_DATA_DETECTED": [
            "Remove synthetic data generation; load and use the provided cleaned dataset only.",
        ],
        "ALIGNMENT_REQUIREMENTS_MISSING": [
            "Populate alignment_check.json with per-requirement status + evidence list.",
        ],
        "AST_PARSE_FAILED": [
            "Return valid Python syntax; do not output partial code or truncated blocks.",
        ],
        "ALIGNMENT_CHECK_OUTPUT": [
            "Write data/alignment_check.json with per-requirement status and evidence (metrics, artifacts, or logs).",
        ],
        "alignment_check_missing": [
            "Create data/alignment_check.json and include it in required outputs.",
        ],
        "alignment_method_choice": [
            "Revise methodology to align with contract requirements (segmentation, decision variables, validation).",
        ],
    }

    for token in tokens:
        for key, additions in guidance_map.items():
            if key in token:
                fixes.extend(additions)

    deduped: List[str] = []
    seen = set()
    for fix in fixes:
        if fix not in seen:
            seen.add(fix)
            deduped.append(fix)
    return deduped

def _build_fix_instructions(required_fixes: List[str]) -> str:
    if not required_fixes:
        return ""
    lines = ["PATCH TARGETS (apply minimal edits):"]
    for fix in required_fixes:
        lines.append(f"- {fix}")
    return "\n".join(lines)

def _get_iteration_policy(state: Dict[str, Any]) -> Dict[str, int] | None:
    contract = state.get("execution_contract") or {}
    policy = contract.get("iteration_policy")
    if not isinstance(policy, dict):
        return None
    compliance_max = policy.get("compliance_bootstrap_max")
    metric_max = policy.get("metric_improvement_max")
    runtime_max = policy.get("runtime_fix_max")
    out: Dict[str, int] = {}
    for key, val in {
        "compliance_bootstrap_max": compliance_max,
        "metric_improvement_max": metric_max,
        "runtime_fix_max": runtime_max,
    }.items():
        if val is None:
            continue
        try:
            out[key] = max(1, int(val))
        except Exception:
            continue
    return out or None

def _classify_iteration_type(
    status: str,
    audit_rejected: bool,
    oc_report: Dict[str, Any] | None,
    feedback: str | None,
) -> str | None:
    if status != "NEEDS_IMPROVEMENT":
        return None
    if audit_rejected:
        return "compliance"
    if oc_report and oc_report.get("missing"):
        return "compliance"
    if feedback and any(token in feedback for token in ["CODE_AUDIT_REJECTED", "OUTPUT_CONTRACT_MISSING"]):
        return "compliance"
    return "metric"

def _detect_refscore_alias(execution_output: str, contract: Dict[str, Any]) -> bool:
    if not execution_output or not isinstance(contract, dict):
        return False
    derived_target = False
    for req in contract.get("data_requirements", []) or []:
        if not isinstance(req, dict):
            continue
        name = (req.get("canonical_name") or req.get("name") or "").lower()
        role = (req.get("role") or "").lower()
        source = (req.get("source") or "input").lower()
        if role == "target" and "refscore" in name and source == "derived":
            derived_target = True
            break
    if not derived_target:
        return False
    alias_patterns = [
        '"RefScore": "Score"',
        "'RefScore': 'Score'",
        "RefScore -> Score",
    ]
    return any(pat in execution_output for pat in alias_patterns)

DEFAULT_RUN_BUDGET = {
    "max_de_calls": 4,
    "max_ml_calls": 6,
    "max_reviewer_calls": 5,
    "max_qa_calls": 5,
    "max_execution_calls": 6,
}

def _ensure_budget_state(state: Dict[str, Any]) -> Dict[str, Any]:
    budget = dict(state.get("run_budget") or {})
    if not budget:
        budget = dict(DEFAULT_RUN_BUDGET)
    counters = dict(state.get("budget_counters") or {})
    for key in ["de_calls", "ml_calls", "reviewer_calls", "qa_calls", "execution_calls"]:
        counters.setdefault(key, 0)
    return {"run_budget": budget, "budget_counters": counters}

def _consume_budget(state: Dict[str, Any], counter_key: str, limit_key: str, label: str):
    budget = state.get("run_budget") or DEFAULT_RUN_BUDGET
    counters = dict(state.get("budget_counters") or {})
    limit = budget.get(limit_key, DEFAULT_RUN_BUDGET.get(limit_key))
    used = counters.get(counter_key, 0) + 1
    counters[counter_key] = used
    if limit is not None and used > limit:
        return False, counters, f"BUDGET_EXCEEDED: {label} exceeded {used}/{limit}"
    return True, counters, ""

def detect_undefined_names(code: str) -> List[str]:
    """
    AST-based preflight to detect names that are used but never defined/imported.
    Avoids wasting sandbox runs on obvious NameError.
    """
    import ast
    import builtins as _builtins

    try:
        tree = ast.parse(code)
    except Exception:
        return []

    defined: set[str] = set(dir(_builtins)) | {"__name__", "__file__", "__package__", "__spec__", "__doc__", "__cached__"}

    def add_target(target):
        if isinstance(target, ast.Name):
            defined.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                add_target(elt)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname:
                    defined.add(alias.asname)
                else:
                    # top-level module name
                    defined.add(alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            for arg in node.args.args + node.args.kwonlyargs:
                defined.add(arg.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Lambda):
            for arg in node.args.args + node.args.kwonlyargs:
                defined.add(arg.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            add_target(node.target if hasattr(node, "target") else None)
            if hasattr(node, "targets"):
                for tgt in node.targets:
                    add_target(tgt)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            add_target(node.target)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    add_target(item.optional_vars)
        elif isinstance(node, ast.comprehension):
            add_target(node.target)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)

    undefined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined:
                undefined.add(node.id)
    return sorted(undefined)

def manifest_dump_missing_default(code: str) -> bool:
    """
    Detects json.dump calls without default= to guard manifest serialization.
    Returns True if a risky call is found.
    """
    import ast
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_json_dump = False
            if isinstance(func, ast.Attribute) and func.attr == "dump":
                if isinstance(func.value, ast.Name) and func.value.id == "json":
                    is_json_dump = True
            if isinstance(func, ast.Name) and func.id == "dump":
                is_json_dump = True
            if not is_json_dump:
                continue
            kw_args = {kw.arg for kw in node.keywords if kw.arg}
            if "default" not in kw_args:
                return True
    return False

def contains_json_null_literal(code: str) -> bool:
    """
    Detects use of JSON literal `null` in Python code (likely from copied JSON).
    """
    import io
    import tokenize
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    except Exception:
        return False
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string == "null":
            return True
    return False

def ml_quality_preflight(code: str) -> List[str]:
    """
    Static ML quality checks to prevent QA loops before reviewer/sandbox.
    Returns list of missing gates.
    """
    import ast

    issues: List[str] = []
    code_lower = code.lower()
    if "mapping summary" not in code_lower:
        issues.append("MAPPING_SUMMARY")
    if "alignment_check" not in code_lower:
        issues.append("ALIGNMENT_CHECK_OUTPUT")

    try:
        tree = ast.parse(code)
    except Exception:
        return ["AST_PARSE_FAILED"]

    has_read_csv = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "read_csv":
                if isinstance(func.value, ast.Name) and func.value.id in {"pd", "pandas"}:
                    has_read_csv = True
                    break
            if isinstance(func, ast.Name) and func.id == "read_csv":
                has_read_csv = True
                break
    if not has_read_csv:
        issues.append("DATA_LOAD_MISSING")

    def _condition_checks_nunique_guard(test_node: ast.AST) -> bool:
        if not isinstance(test_node, ast.Compare):
            return False
        left = test_node.left
        if not isinstance(left, ast.Call):
            return False
        func = left.func
        if not (isinstance(func, ast.Attribute) and func.attr == "nunique"):
            return False
        comparator_consts = [
            comp
            for comp in test_node.comparators
            if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float))
        ]
        if not comparator_consts:
            return False
        for op, comp in zip(test_node.ops, comparator_consts):
            if isinstance(op, ast.Eq) and comp.value == 1:
                return True
            if isinstance(op, ast.LtE) and comp.value <= 1:
                return True
            if isinstance(op, ast.Lt) and comp.value <= 2:
                return True
        return False

    def _if_has_value_error_raise(body_nodes) -> bool:
        for node in body_nodes:
            if isinstance(node, ast.Raise):
                exc = node.exc
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "ValueError":
                    return True
            for child in ast.walk(node):
                if isinstance(child, ast.Raise):
                    exc = child.exc
                    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "ValueError":
                        return True
        return False

    has_variance_guard = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            if _condition_checks_nunique_guard(node.test) and _if_has_value_error_raise(node.body):
                has_variance_guard = True
                break
    if not has_variance_guard:
        issues.append("TARGET_VARIANCE_GUARD")

    def _is_df_feature_cols_subscript(node: ast.AST) -> bool:
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "df":
                sl = node.slice
                if isinstance(sl, ast.Name) and sl.id == "feature_cols":
                    return True
                if isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
                    return any(isinstance(e, ast.Name) and e.id == "feature_cols" for e in sl.elts)
            if isinstance(node.value, ast.Attribute):
                # df.loc[:, feature_cols]
                if isinstance(node.value.value, ast.Name) and node.value.value.id == "df" and node.value.attr == "loc":
                    sl = node.slice
                    if isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
                        return any(isinstance(e, ast.Name) and e.id == "feature_cols" for e in sl.elts)
        return False

    def _drop_excludes_target(call: ast.Call, target_names: set[str]) -> bool:
        if not isinstance(call.func, ast.Attribute):
            return False
        if call.func.attr != "drop":
            return False
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "df":
            return False
        targets = set(target_names) | {"target", "y", "target_col"}
        # keywords
        for kw in call.keywords:
            if kw.arg != "columns":
                continue
            if isinstance(kw.value, ast.List):
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Name) and elt.id in targets:
                        return True
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and elt.value.lower() in targets:
                        return True
        # positional args
        for arg in call.args:
            if isinstance(arg, ast.List):
                for elt in arg.elts:
                    if isinstance(elt, ast.Name) and elt.id in targets:
                        return True
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and elt.value.lower() in targets:
                        return True
        return False

    target_names: set[str] = set()
    has_target_assignment = False
    has_x_from_feature_cols = False
    has_drop_target = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # target assignment
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in {"y", "target", "y_train", "target_series"}:
                    has_target_assignment = True
                    target_names.add(tgt.id)
            # X assignment
            if any(isinstance(t, ast.Name) and t.id in {"X", "x"} for t in node.targets):
                if _is_df_feature_cols_subscript(node.value):
                    has_x_from_feature_cols = True
                if isinstance(node.value, ast.Call) and _drop_excludes_target(node.value, target_names):
                    has_drop_target = True
        elif isinstance(node, ast.Call):
            if _drop_excludes_target(node, target_names):
                has_drop_target = True

    has_target_exclusion = has_target_assignment and (has_x_from_feature_cols or has_drop_target)
    if not has_target_exclusion:
        issues.append("TARGET_NOT_IN_X")

    has_high_card_filter = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            left = node.left
            comps = [left] + list(node.comparators)
            if any(isinstance(op, (ast.Gt, ast.GtE)) for op in node.ops):
                for comp in comps:
                    if isinstance(comp, ast.Call):
                        func = comp.func
                        if isinstance(func, ast.Attribute) and func.attr == "nunique":
                            has_high_card_filter = True
                            break
        if has_high_card_filter:
            break

    if not (has_x_from_feature_cols or has_high_card_filter):
        issues.append("HIGH_CARDINALITY_HANDLING")

    cv_markers = [
        "cross_val_score",
        "cross_validate",
        "kfold",
        "stratifiedkfold",
        "groupkfold",
        "timeseriessplit",
        "stratifiedgroupkfold",
    ]
    if "train_test_split" in code_lower and not any(marker in code_lower for marker in cv_markers):
        issues.append("CROSS_VALIDATION_REQUIRED")

    return issues

def _analyze_read_csv_usage(code: str) -> Dict[str, Any]:
    import ast

    result = {"has_read_csv": False, "paths": []}
    if not code:
        return result
    try:
        tree = ast.parse(code)
    except Exception:
        return result
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_read = False
            if isinstance(func, ast.Attribute) and func.attr == "read_csv":
                if isinstance(func.value, ast.Name) and func.value.id in {"pd", "pandas"}:
                    is_read = True
            if isinstance(func, ast.Name) and func.id == "read_csv":
                is_read = True
            if not is_read:
                continue
            result["has_read_csv"] = True
            if node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    result["paths"].append(arg0.value)
    return result

def _required_columns_coverage(code: str, required_columns: List[str]) -> Dict[str, Any]:
    if not code or not required_columns:
        return {"hits": [], "missing": []}
    code_lower = code.lower()
    hits = []
    for col in required_columns:
        if not col:
            continue
        if str(col).lower() in code_lower:
            hits.append(col)
    missing = [col for col in required_columns if col and col not in hits]
    return {"hits": hits, "missing": missing}

def _detect_synthetic_data(code: str) -> bool:
    if not code:
        return False
    code_lower = code.lower()
    synthetic_markers = [
        "np.random",
        "random.",
        "make_classification",
        "make_regression",
        "make_blobs",
        "faker",
        "synthetic data",
        "generate synthetic",
        "load_iris",
        "load_breast_cancer",
        "load_wine",
    ]
    return any(marker in code_lower for marker in synthetic_markers)

def _missing_required_output_refs(code: str, outputs: List[str]) -> List[str]:
    if not code or not outputs:
        return []
    missing: List[str] = []
    for output in outputs:
        if output and output not in code:
            missing.append(output)
    return missing

def dialect_guard_violations(code: str, csv_sep: str, csv_decimal: str, csv_encoding: str, expected_path: str | None = None) -> List[str]:
    """
    AST-based guard to ensure pd.read_csv (first call or the one reading expected_path) sets sep/decimal/encoding.
    Only rejects when keywords are missing or literal strings mismatch the provided dialect.
    """
    import ast

    violations: List[str] = []
    try:
        tree = ast.parse(code)
    except Exception as parse_err:
        return [f"Could not parse code for dialect guard: {parse_err}"]

    calls: List[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "read_csv":
                if isinstance(func.value, ast.Name) and func.value.id == "pd":
                    calls.append(node)
            elif isinstance(func, ast.Name) and func.id == "read_csv":
                calls.append(node)

    if not calls:
        return violations

    def _is_expected_path(arg: ast.AST) -> bool:
        if expected_path is None:
            return False
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value == expected_path
        return False

    target_call = None
    for call in calls:
        if call.args and _is_expected_path(call.args[0]):
            target_call = call
            break
    if target_call is None:
        target_call = calls[0]

    expected = {
        "sep": csv_sep,
        "decimal": csv_decimal,
        "encoding": csv_encoding,
    }
    kw_map = {kw.arg: kw.value for kw in target_call.keywords if kw.arg}

    def _normalize_encoding(value: str) -> str:
        return str(value).strip().lower().replace("_", "-")

    def _encoding_matches(expected_value: str, actual_value: str) -> bool:
        exp = _normalize_encoding(expected_value)
        act = _normalize_encoding(actual_value)
        if exp in {"utf-8", "utf8"}:
            return act in {"utf-8", "utf8", "utf-8-sig", "utf8-sig"}
        if exp in {"utf-8-sig", "utf8-sig"}:
            return act in {"utf-8", "utf8", "utf-8-sig", "utf8-sig"}
        return exp == act

    for param, expected_value in expected.items():
        if param not in kw_map:
            violations.append(f"pd.read_csv missing {param}= for dialect")
            continue
        val_node = kw_map[param]
        if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
            if param == "encoding":
                if not _encoding_matches(expected_value, val_node.value):
                    violations.append(
                        f"pd.read_csv {param} literal '{val_node.value}' does not match expected '{expected_value}'"
                    )
                continue
            if val_node.value != expected_value:
                violations.append(f"pd.read_csv {param} literal '{val_node.value}' does not match expected '{expected_value}'")
        else:
            # Non-literal (variable/expression) accepted
            continue

    seen = set()
    unique = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique

def _infer_de_failure_cause(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    if "missing required columns" in lower or "mapping failed" in lower:
        return "Required columns not found after canonicalization or alias mapping."
    if "cleaning_plan_not_allowed" in lower or "plan output" in lower:
        return "Model returned a JSON plan instead of executable Python code."
    if "dialect" in lower or "read_csv" in lower:
        return "CSV dialect mismatch (sep/decimal/encoding) or missing dialect params."
    if "parsererror" in lower or "tokenizing data" in lower:
        return "CSV dialect/quoting mismatch; enforce detected sep/decimal/encoding."
    if "nameerror" in lower:
        return "Undefined name referenced in the generated code."
    if "keyerror" in lower and "not in index" in lower:
        return "Column name mismatch after normalization; mapping resolution failed."
    if "list of cases must be same length as list of conditions" in lower:
        return "np.select called with mismatched conditions/choices list lengths."
    if "typeerror" in lower or "ufunc" in lower:
        return "Type conversion missing; numeric ops executed on string/object data."
    if "numpy.bool_" in lower and "not serializable" in lower:
        return "JSON serialization failed due to numpy.bool_ values not handled in _json_default."
    if "json" in lower and "default" in lower:
        return "Manifest json.dump missing default for numpy/pandas types."
    if "syntaxerror" in lower:
        return "Generated code is not valid Python syntax."
    return ""

def _build_de_runtime_diagnosis(error_details: str) -> List[str]:
    if not error_details:
        return []
    lower = error_details.lower()
    lines: List[str] = []
    if "list of cases must be same length as list of conditions" in lower:
        lines.append(
            "np.select raised a length mismatch: number of conditions does not match number of choices (see assign_fec_window)."
        )
    if "numpy.bool_" in lower and "not serializable" in lower:
        lines.append(
            "json.dumps failed because validation_results contains numpy.bool_, which _json_default does not handle."
        )
    if "keyerror" in lower and "not in index" in lower:
        lines.append(
            "KeyError indicates a column name mismatch after renaming/normalization; selection list includes a missing column."
        )
    if "not supported between instances of 'str' and 'int'" in lower or "not supported between instances of 'str' and 'float'" in lower:
        lines.append(
            "Type comparison on object/string data: convert %plazoConsumido (or any percentage string) to numeric before >/< checks; use pd.to_numeric/clean_plazo_consumido and compare the numeric series."
        )
    if "keyerror" in lower and "'score'" in lower:
        lines.append(
            "KeyError on 'Score' suggests conversion metadata was updated before creating a 'Score' entry or the column mapping for Score did not run."
        )
    return lines

def _merge_de_audit_override(base: str, payload: str) -> str:
    base = base or ""
    payload = payload or ""
    if not payload:
        return base
    if payload in base:
        return base
    if base:
        return f"{base}\n\n{payload}"
    return payload

def _read_csv_header(csv_path: str, encoding: str, sep: str) -> List[str]:
    try:
        import csv
        with open(csv_path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f, delimiter=sep)
            return next(reader, [])
    except Exception:
        return []

def _cleanup_run_artifacts() -> None:
    paths_to_remove = [
        os.path.join("artifacts", "ml_engineer_last.py"),
        os.path.join("artifacts", "ml_engineer_context.json"),
        os.path.join("artifacts", "ml_engineer_context_ops.txt"),
        os.path.join("artifacts", "ml_results_advisor.txt"),
        os.path.join("artifacts", "ml_engineer_failure_explainer.txt"),
        os.path.join("artifacts", "data_engineer_last.py"),
        os.path.join("artifacts", "data_engineer_sandbox_last.log"),
    ]
    for path in paths_to_remove:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    # Clear iteration artifacts
    iter_dir = os.path.join("artifacts", "iterations")
    if os.path.isdir(iter_dir):
        for filename in os.listdir(iter_dir):
            file_path = os.path.join(iter_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception:
                pass

    # Clear previous outputs that can contaminate new runs
    data_files = [
        os.path.join("data", "weights.json"),
        os.path.join("data", "case_summary.csv"),
        os.path.join("data", "metrics.json"),
        os.path.join("data", "scored_rows.csv"),
        os.path.join("data", "case_alignment_report.json"),
        os.path.join("data", "output_contract_report.json"),
        os.path.join("data", "execution_contract.json"),
    ]
    for path in data_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    # Clear plots from previous runs
    plots_dir = os.path.join("static", "plots")
    if os.path.isdir(plots_dir):
        for filename in os.listdir(plots_dir):
            file_path = os.path.join(plots_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception:
                pass

def _load_json_safe(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _normalize_alignment_check(
    alignment_check: Dict[str, Any],
    alignment_requirements: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    normalized = dict(alignment_check or {})
    status = str(normalized.get("status") or "").upper()
    if status not in {"PASS", "WARN", "FAIL"}:
        status = "WARN"
        issues.append("alignment_status_invalid")

    requirements_payload = normalized.get("requirements")
    per_req = normalized.get("per_requirement")
    evidence_map = normalized.get("evidence")

    normalized_reqs: List[Dict[str, Any]] = []
    missing_status = 0
    missing_evidence = 0
    for req in alignment_requirements or []:
        req_id = req.get("id")
        if not req_id:
            continue
        req_status = None
        req_evidence: List[str] = []
        if isinstance(requirements_payload, list):
            for item in requirements_payload:
                if not isinstance(item, dict):
                    continue
                if item.get("id") == req_id:
                    req_status = str(item.get("status") or item.get("result") or "").upper()
                    evidence_val = item.get("evidence") or item.get("notes") or []
                    if isinstance(evidence_val, list):
                        req_evidence = [str(v) for v in evidence_val if v]
                    elif isinstance(evidence_val, str) and evidence_val.strip():
                        req_evidence = [evidence_val.strip()]
                    break
        if req_status is None and isinstance(per_req, dict):
            raw = per_req.get(req_id)
            if raw is not None:
                req_status = str(raw).upper()
        if not req_evidence and isinstance(evidence_map, dict):
            raw_ev = evidence_map.get(req_id)
            if isinstance(raw_ev, list):
                req_evidence = [str(v) for v in raw_ev if v]
            elif isinstance(raw_ev, str) and raw_ev.strip():
                req_evidence = [raw_ev.strip()]
        if not req_status:
            missing_status += 1
            req_status = "MISSING"
        if not req_evidence:
            missing_evidence += 1
        normalized_reqs.append(
            {"id": req_id, "status": req_status, "evidence": req_evidence}
        )

    if missing_status:
        issues.append("alignment_missing_requirement_status")
    if missing_evidence:
        issues.append("alignment_missing_evidence")

    failure_mode = normalized.get("failure_mode")
    if not failure_mode and issues:
        failure_mode = "method_choice"

    normalized["status"] = status
    normalized["failure_mode"] = failure_mode
    normalized["requirements"] = normalized_reqs
    if issues:
        summary = normalized.get("summary") or ""
        issue_text = ", ".join(sorted(set(issues)))
        normalized["summary"] = f"{summary} Alignment issues: {issue_text}".strip()
    return normalized, issues

def _hash_json(payload: Any) -> str | None:
    if not payload:
        return None
    try:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return None

def _extract_weights_from_obj(obj: Any) -> Dict[str, float]:
    if not isinstance(obj, dict):
        return {}
    if isinstance(obj.get("weights"), dict):
        return {str(k): float(v) for k, v in obj["weights"].items() if _is_number(v)}
    if isinstance(obj.get("feature_weights"), dict):
        return {str(k): float(v) for k, v in obj["feature_weights"].items() if _is_number(v)}
    weights = obj.get("weights")
    features = obj.get("features")
    if isinstance(weights, list) and isinstance(features, list) and len(weights) == len(features):
        out = {}
        for idx, feat in enumerate(features):
            val = weights[idx]
            if _is_number(val):
                out[str(feat)] = float(val)
        if out:
            return out
    weights = obj.get("optimized_weights")
    features = obj.get("feature_cols") or obj.get("feature_columns")
    if isinstance(weights, list) and isinstance(features, list) and len(weights) == len(features):
        out = {}
        for idx, feat in enumerate(features):
            val = weights[idx]
            if _is_number(val):
                out[str(feat)] = float(val)
        if out:
            return out
    return {}

def _summarize_weight_uniformity(weights_obj: Dict[str, Any]) -> Dict[str, Any]:
    weights = _extract_weights_from_obj(weights_obj)
    if not weights:
        return {"uniform": None}
    vals = list(weights.values())
    try:
        max_val = max(vals)
        min_val = min(vals)
        mean_val = sum(vals) / len(vals)
        spread = max_val - min_val
    except Exception:
        return {"uniform": None}
    return {
        "uniform": spread < 1e-6,
        "min": float(min_val),
        "max": float(max_val),
        "mean": float(mean_val),
        "spread": float(spread),
    }

def _normalize_metrics_from_weights(weights_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(weights_obj, dict):
        return {}
    metrics = weights_obj.get("metrics") if isinstance(weights_obj.get("metrics"), dict) else {}
    spearman = (
        metrics.get("global_spearman_correlation")
        or metrics.get("spearman_correlation")
        or metrics.get("spearman")
    )
    kendall = metrics.get("kendall_correlation") or metrics.get("kendall_tau")
    case_spearman = (
        metrics.get("case_level_spearman_correlation")
        or metrics.get("case_order_spearman")
    )
    ranking_violations = metrics.get("ranking_violations")
    max_weight = metrics.get("max_weight")
    hhi = metrics.get("weight_hhi")
    near_zero = metrics.get("near_zero_weights") or metrics.get("near_zero_weights_count")
    if spearman is None and kendall is None and case_spearman is None and max_weight is None:
        return {}
    report = {
        "global_ranking": {
            "spearman": float(spearman) if _is_number(spearman) else None,
            "spearman_p": metrics.get("global_spearman_p_value") if metrics else None,
            "kendall_tau": float(kendall) if _is_number(kendall) else None,
        },
        "case_level": {
            "spearman": float(case_spearman) if _is_number(case_spearman) else None,
            "adjacent_violations_pct": None,
            "adjacent_violations_count": int(ranking_violations) if _is_number(ranking_violations) else None,
        },
        "weight_concentration": {
            "max_weight": float(max_weight) if _is_number(max_weight) else None,
            "hhi": float(hhi) if _is_number(hhi) else None,
            "weights_near_zero": int(near_zero) if _is_number(near_zero) else None,
        },
        "baseline_comparison": {},
    }
    return report

def _load_output_dialect_local(manifest_path: str = "data/cleaning_manifest.json") -> Dict[str, str]:
    defaults = {"sep": ",", "decimal": ".", "encoding": "utf-8"}
    manifest = _load_json_safe(manifest_path) or {}
    if isinstance(manifest, dict):
        output_dialect = manifest.get("output_dialect") or {}
        if isinstance(output_dialect, dict):
            return {
                "sep": output_dialect.get("sep", defaults["sep"]),
                "decimal": output_dialect.get("decimal", defaults["decimal"]),
                "encoding": output_dialect.get("encoding", defaults["encoding"]),
            }
    return defaults

def _compute_adjacent_violations(ref_series, score_series) -> int:
    import pandas as pd
    df = pd.DataFrame({"ref": ref_series, "score": score_series}).dropna()
    if df.empty:
        return 0
    df = df.sort_values("ref")
    refs = df["ref"].values
    scores = df["score"].values
    violations = 0
    for idx in range(len(scores) - 1):
        if refs[idx + 1] > refs[idx] and scores[idx + 1] < scores[idx]:
            violations += 1
    return int(violations)

def _compute_metrics_from_scored_rows(
    scored_rows_path: str = "data/scored_rows.csv",
    case_summary_path: str = "data/case_summary.csv",
    weights_obj: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not os.path.exists(scored_rows_path):
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}
    dialect = _load_output_dialect_local()
    try:
        df = pd.read_csv(
            scored_rows_path,
            sep=dialect["sep"],
            decimal=dialect["decimal"],
            encoding=dialect["encoding"],
        )
    except Exception:
        return {}
    score_candidates = ["Score_nuevo", "ScoreNuevo", "score_nuevo", "Score_Nuevo"]
    ref_candidates = ["RefScore", "refscore", "Ref_Score"]
    case_candidates = ["Case_ID", "Case", "case_id", "caso", "caso_id"]

    score_col = next((c for c in score_candidates if c in df.columns), None)
    ref_col = next((c for c in ref_candidates if c in df.columns), None)
    case_col = next((c for c in case_candidates if c in df.columns), None)
    if not score_col or not ref_col:
        return {}

    ref_series = pd.to_numeric(df[ref_col], errors="coerce")
    score_series = pd.to_numeric(df[score_col], errors="coerce")
    spearman = None
    kendall = None
    try:
        spearman = float(ref_series.corr(score_series, method="spearman"))
        kendall = float(ref_series.corr(score_series, method="kendall"))
    except Exception:
        pass

    case_spearman = None
    adjacent_violations = None
    if case_col and case_col in df.columns:
        try:
            case_means = (
                df[[case_col, score_col, ref_col]]
                .copy()
                .groupby(case_col)
                .agg({score_col: "mean", ref_col: "first"})
                .reset_index()
            )
            ref_case = pd.to_numeric(case_means[ref_col], errors="coerce")
            score_case = pd.to_numeric(case_means[score_col], errors="coerce")
            case_spearman = float(ref_case.corr(score_case, method="spearman"))
            adjacent_violations = _compute_adjacent_violations(ref_case, score_case)
        except Exception:
            pass

    weights = _extract_weights_from_obj(weights_obj or {})
    weight_metrics = {
        "max_weight": None,
        "hhi": None,
        "weights_near_zero": None,
    }
    if weights:
        vals = list(weights.values())
        try:
            max_weight = max(vals)
            hhi = sum(v ** 2 for v in vals)
            near_zero = sum(1 for v in vals if v < 0.01)
            weight_metrics = {
                "max_weight": float(max_weight),
                "hhi": float(hhi),
                "weights_near_zero": int(near_zero),
            }
        except Exception:
            pass

    return {
        "global_ranking": {
            "spearman": spearman,
            "spearman_p": None,
            "kendall_tau": kendall,
        },
        "case_level": {
            "spearman": case_spearman,
            "adjacent_violations_pct": None,
            "adjacent_violations_count": int(adjacent_violations) if adjacent_violations is not None else None,
        },
        "weight_concentration": weight_metrics,
        "baseline_comparison": {},
    }

def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False

def _infer_objective_signature(code: str) -> Dict[str, Any]:
    if not code:
        return {}
    lower = code.lower()
    optimizer = None
    for opt in ("linprog", "slsqp", "minimize", "cobyla", "trust-constr", "differential_evolution"):
        if opt in lower:
            optimizer = opt
            break
    terms = []
    for token in (
        "case_order_violations",
        "adjacent_refscore_violations",
        "case_spearman",
        "case_level",
        "case_means",
        "kendall",
        "spearman",
        "rank_correlation",
        "violation",
        "penalty",
        "regularization",
    ):
        if token in lower and token not in terms:
            terms.append(token)
    code_hash = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()[:10]
    return {"optimizer": optimizer or "unknown", "terms": terms[:6], "code_hash": code_hash}

def _format_metric_delta(label: str, prev: Any, curr: Any) -> str:
    if prev is None or curr is None:
        return f"{label}: {curr}"
    try:
        delta = float(curr) - float(prev)
        return f"{label}: {prev} -> {curr} (delta {delta:+.4f})"
    except Exception:
        return f"{label}: {prev} -> {curr}"

def _extract_metrics_snapshot(metrics_report: Dict[str, Any], weights_report: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    if isinstance(metrics_report, dict):
        global_ranking = metrics_report.get("global_ranking", {})
        if isinstance(global_ranking, dict) and _is_number(global_ranking.get("spearman")):
            snapshot["global_spearman"] = float(global_ranking["spearman"])
    if not isinstance(weights_report, dict):
        return snapshot
    propensity = weights_report.get("propensity_model")
    if isinstance(propensity, dict):
        for key in ("auc_cv_mean", "auc_cv_std", "baseline_auc", "improvement_pct"):
            value = propensity.get(key)
            if _is_number(value):
                snapshot[f"propensity_{key}"] = float(value)
        if "calibrated" in propensity:
            snapshot["propensity_calibrated"] = bool(propensity.get("calibrated"))
    price = weights_report.get("price_model")
    if isinstance(price, dict):
        for key in ("mae_cv_mean", "mape_cv_mean", "baseline_mae", "improvement_pct"):
            value = price.get(key)
            if _is_number(value):
                snapshot[f"price_{key}"] = float(value)
        model_type = price.get("model_type")
        if isinstance(model_type, str):
            snapshot["price_model_type"] = model_type
    optimization = weights_report.get("optimization")
    if isinstance(optimization, dict):
        for key in ("total_typologies", "high_value_threshold", "low_value_threshold"):
            value = optimization.get(key)
            if _is_number(value):
                snapshot[f"optimization_{key}"] = float(value)
    return snapshot

def _build_iteration_memory(
    iter_id: int,
    metrics_report: Dict[str, Any],
    case_report: Dict[str, Any],
    weights_report: Dict[str, Any],
    code: str,
    prev_summary: Dict[str, Any] | None,
    advisor_note: str | None,
) -> Dict[str, Any]:
    objective_sig = _infer_objective_signature(code)
    weights = _extract_weights_from_obj(weights_report)
    top_weights = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
    metrics_global = (metrics_report or {}).get("global_ranking", {})
    metrics_case = (metrics_report or {}).get("case_level", {})
    case_metrics = (case_report or {}).get("metrics", {})
    metrics_snapshot = _extract_metrics_snapshot(metrics_report, weights_report)
    summary = {
        "iteration_id": iter_id,
        "objective_signature": objective_sig,
        "weights_top": top_weights,
        "metrics_global": metrics_global,
        "metrics_case_level": metrics_case,
        "case_alignment_metrics": case_metrics,
        "failures": case_report.get("failures", []) if isinstance(case_report, dict) else [],
        "metrics_snapshot": metrics_snapshot,
    }
    if advisor_note:
        summary["advisor_note"] = advisor_note.strip()
    if prev_summary:
        delta_snapshot: Dict[str, Any] = {}
        prev_snapshot = prev_summary.get("metrics_snapshot", {}) if isinstance(prev_summary, dict) else {}
        if isinstance(prev_snapshot, dict):
            for key, curr_val in metrics_snapshot.items():
                prev_val = prev_snapshot.get(key)
                if prev_val is None and curr_val is None:
                    continue
                delta_snapshot[key] = {"prev": prev_val, "curr": curr_val}
        summary["delta"] = {
            "spearman_case_means": {
                "prev": prev_summary.get("case_alignment_metrics", {}).get("spearman_case_means"),
                "curr": case_metrics.get("spearman_case_means"),
            },
            "adjacent_refscore_violations": {
                "prev": prev_summary.get("case_alignment_metrics", {}).get("adjacent_refscore_violations"),
                "curr": case_metrics.get("adjacent_refscore_violations"),
            },
            "global_spearman": {
                "prev": prev_summary.get("metrics_global", {}).get("spearman"),
                "curr": metrics_global.get("spearman"),
            },
        }
        if delta_snapshot:
            summary["delta"]["metrics_snapshot"] = delta_snapshot
    return summary

def _build_edit_instructions(summary: Dict[str, Any]) -> str:
    if not summary:
        return ""
    objective = summary.get("objective_signature", {})
    failures = summary.get("failures", [])
    delta = summary.get("delta", {})
    weights_top = summary.get("weights_top", [])
    lines = ["EDIT THESE BLOCKS (modify previous code; do not rewrite):"]
    if objective:
        terms = ", ".join(objective.get("terms") or [])
        lines.append(
            f"- Objective: optimizer={objective.get('optimizer')} terms=[{terms}] code_hash={objective.get('code_hash')}"
        )
    if weights_top:
        weights_text = ", ".join([f"{k}={v:.4f}" for k, v in weights_top])
        lines.append(f"- Weights(top): {weights_text}")
    metrics_snapshot = summary.get("metrics_snapshot", {})
    if isinstance(metrics_snapshot, dict) and metrics_snapshot:
        snap_items = []
        for key in sorted(metrics_snapshot.keys())[:6]:
            snap_items.append(f"{key}={metrics_snapshot[key]}")
        lines.append(f"- Metrics snapshot: {', '.join(snap_items)}")
    if delta:
        metrics_delta = delta.get("metrics_snapshot", {}) if isinstance(delta, dict) else {}
        extra_metrics = []
        if isinstance(metrics_delta, dict):
            for key in sorted(metrics_delta.keys())[:4]:
                extra_metrics.append(
                    _format_metric_delta(
                        key,
                        metrics_delta.get(key, {}).get("prev"),
                        metrics_delta.get(key, {}).get("curr"),
                    )
                )
        lines.append(
            "- Metric deltas: "
            + "; ".join(
                [
                    _format_metric_delta("spearman_case_means", delta.get("spearman_case_means", {}).get("prev"), delta.get("spearman_case_means", {}).get("curr")),
                    _format_metric_delta("adjacent_refscore_violations", delta.get("adjacent_refscore_violations", {}).get("prev"), delta.get("adjacent_refscore_violations", {}).get("curr")),
                    _format_metric_delta("global_spearman", delta.get("global_spearman", {}).get("prev"), delta.get("global_spearman", {}).get("curr")),
                    *extra_metrics,
                ]
            )
        )
    if failures:
        lines.append(f"- Current failures: {failures}")
    advisor_note = summary.get("advisor_note")
    if advisor_note:
        lines.append(f"- Advisor: {advisor_note}")
    return "\n".join(lines)
def _summarize_case_summary(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        return {}
    case_col = "Case" if "Case" in df.columns else ("Caso" if "Caso" in df.columns else None)
    if not case_col:
        return {"rows": int(len(df))}
    try:
        cases = [int(x) for x in df[case_col].dropna().unique().tolist()]
    except Exception:
        cases = []
    missing_cases = [c for c in range(1, 21) if c not in cases] if cases else []
    respects_col = "respects_order" if "respects_order" in df.columns else None
    false_count = 0
    if respects_col:
        try:
            false_count = int((df[respects_col] == False).sum())
        except Exception:
            false_count = 0
    return {
        "rows": int(len(df)),
        "case_count": int(len(cases)),
        "missing_cases": missing_cases,
        "respects_order_false": false_count,
        "case_col": case_col,
    }

def _persist_iteration_artifacts(iter_id: int) -> Dict[str, str]:
    if not iter_id or iter_id < 1:
        return {}
    import shutil
    os.makedirs(os.path.join("artifacts", "iterations"), exist_ok=True)
    mappings = {
        "case_alignment_report": ("data/case_alignment_report.json", f"case_alignment_report_iter_{iter_id}.json"),
        "weights": ("data/weights.json", f"weights_iter_{iter_id}.json"),
        "metrics": ("data/metrics.json", f"metrics_iter_{iter_id}.json"),
        "case_summary": ("data/case_summary.csv", f"case_summary_iter_{iter_id}.csv"),
        "scored_rows": ("data/scored_rows.csv", f"scored_rows_iter_{iter_id}.csv"),
    }
    saved: Dict[str, str] = {}
    for key, (src, dst_name) in mappings.items():
        if not os.path.exists(src):
            continue
        dst = os.path.join("artifacts", "iterations", dst_name)
        try:
            shutil.copyfile(src, dst)
            saved[key] = dst
        except Exception:
            continue
    return saved
# 1. Define State
class AgentState(TypedDict):
    csv_path: str
    business_objective: str
    data_summary: str
    strategies: Dict[str, Any]
    selected_strategy: Dict[str, Any]
    selection_reason: str # Added for UI visibility
    cleaning_code: str    # Added for DE visibility
    cleaned_data_preview: str # Added for DE visibility
    generated_code: str
    execution_output: str
    final_report: str
    # Feedback Loop Fields
    iteration_count: int
    ml_iteration_memory: List[Dict[str, Any]]
    ml_iteration_memory_block: str
    metrics_signature: str
    weights_signature: str
    execution_output_stale: bool
    sandbox_failed: bool
    sandbox_retry_count: int
    max_sandbox_retries: int
    ml_call_refund_pending: bool
    execution_call_refund_pending: bool
    last_successful_execution_output: str
    last_successful_plots: List[str]
    last_successful_output_contract_report: Dict[str, Any]
    model_performance: float
    feedback_history: List[str]
    # PDF
    pdf_path: str
    # Encoding & CSV Format
    csv_encoding: str
    csv_sep: str
    csv_decimal: str
    # Reviewer State
    review_verdict: str
    review_feedback: str
    reviewer_iteration: int
    # Failure State
    error_message: str
    # Domain Expert
    domain_expert_reviews: List[Dict[str, Any]]
    # Visuals State
    has_partial_visuals: bool
    plots_local: List[str]
    # Patch Mode State
    last_generated_code: str
    last_gate_context: Dict[str, Any]
    review_reject_streak: int
    qa_reject_streak: int
    execution_attempt: int # Track runtime retries
    runtime_fix_count: int
    last_runtime_error_tail: str # Added for Runtime Error Visibility
    data_engineer_audit_override: str
    ml_engineer_audit_override: str
    leakage_audit_summary: str
    ml_skipped_reason: str
    execution_contract: Dict[str, Any]
    restrategize_count: int
    strategist_context_override: str
    missing_repeat_count: int
    run_id: str
    run_start_ts: str
    dataset_fingerprint: str
    dataset_memory_context: str
    run_budget: Dict[str, Any]
    budget_counters: Dict[str, int]

from src.agents.domain_expert import DomainExpertAgent

# Initialize Agents
steward = StewardAgent()
strategist = StrategistAgent()
domain_expert = DomainExpertAgent() # New Agent
data_engineer = DataEngineerAgent()
cleaning_reviewer = CleaningReviewerAgent()
ml_engineer = MLEngineerAgent()
translator = BusinessTranslatorAgent()
reviewer = ReviewerAgent()
qa_reviewer = QAReviewerAgent()
execution_planner = ExecutionPlannerAgent()
failure_explainer = FailureExplainerAgent()
results_advisor = ResultsAdvisorAgent()


# 2. Define Nodes

def run_steward(state: AgentState) -> AgentState:
    print("--- [1] Steward: Analyzing Data ---")
    abort_state = _abort_if_requested(state, "steward")
    if abort_state:
        return abort_state
    _cleanup_run_artifacts()
    run_id = state.get("run_id") if state else None
    if not run_id:
        run_id = uuid.uuid4().hex[:8]
    run_start_ts = state.get("run_start_ts") if state else None
    if not run_start_ts:
        run_start_ts = datetime.utcnow().isoformat()
    csv_path = state.get("csv_path") if state else ""
    dataset_fingerprint = fingerprint_dataset(csv_path)
    memory_entries = load_dataset_memory()
    memory_context = summarize_memory(memory_entries, dataset_fingerprint)
    init_run_log(
        run_id,
        {
            "csv_path": csv_path,
            "business_objective": state.get("business_objective") if state else "",
            "dataset_fingerprint": dataset_fingerprint,
        },
    )
    log_run_event(run_id, "steward_start", {"csv_path": csv_path})
    
    # Pre-emptive cleanup
    import glob
    # Delete all potential report files including unique ones to avoid clutter
    for pdf_file in glob.glob("final_report*.pdf"):
        try:
            os.remove(pdf_file)
            print(f"🗑️ Cleaned up old report: {pdf_file}")
        except PermissionError:
            print(f"Note: '{pdf_file}' is locked. Ignoring.")
        except Exception as e:
            print(f"Debug: Cleanup failed for {pdf_file}: {e}")

    result = steward.analyze_data(state['csv_path'], state['business_objective'])
    
    summary = result.get('summary', 'Error')
    encoding = result.get('encoding', 'utf-8')
    sep = result.get('sep', ',')
    decimal = result.get('decimal', '.')

    try:
        os.makedirs("data", exist_ok=True)
        with open("data/steward_summary.txt", "w", encoding="utf-8") as f_summary:
            f_summary.write(summary or "")
        with open("data/steward_summary.json", "w", encoding="utf-8") as f_summary_json:
            json.dump(
                {
                    "run_id": run_id,
                    "csv_path": csv_path,
                    "encoding": encoding,
                    "sep": sep,
                    "decimal": decimal,
                    "summary": summary,
                },
                f_summary_json,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as summary_err:
        print(f"Warning: failed to persist steward summary: {summary_err}")
    
    print(f"Steward Detected: Encoding={encoding}, Sep='{sep}', Decimal='{decimal}'")
    
    log_run_event(
        run_id,
        "steward_complete",
        {"summary_len": len(summary or ""), "encoding": encoding, "sep": sep, "decimal": decimal},
    )

    # Initialize loop variables
    budget_state = _ensure_budget_state(state or {})
    return {
        "data_summary": summary,
        "csv_encoding": encoding,
        "csv_sep": sep,
        "csv_decimal": decimal,
        "csv_decimal": decimal,
        "iteration_count": 0,
        "compliance_iterations": 0,
        "metric_iterations": 0,
        "compliance_passed": False,
        "last_iteration_type": None,
        "feedback_history": [],
        "has_partial_visuals": False,
        "plots_local": [],
        "last_generated_code": None,
        "last_gate_context": None,
        "review_reject_streak": 0,
        "qa_reject_streak": 0,
        "leakage_audit_summary": "",
        "restrategize_count": state.get("restrategize_count", 0) if state else 0,
        "strategist_context_override": state.get("strategist_context_override", "") if state else "",
        "missing_repeat_count": 0,
        "run_id": run_id,
        "run_start_ts": run_start_ts,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_memory_context": memory_context,
        "run_budget": budget_state.get("run_budget", {}),
        "budget_counters": budget_state.get("budget_counters", {}),
    }

def run_strategist(state: AgentState) -> AgentState:
    print("--- [2] Strategist: Formulating 3 Strategies (MIMO v2 Flash) ---")
    abort_state = _abort_if_requested(state, "strategist")
    if abort_state:
        return abort_state
    
    # Strategist now returns a dict with "strategies": [list of 3]
    user_context = state.get("strategist_context_override") or state.get("business_objective", "")
    result = strategist.generate_strategies(state['data_summary'], user_context)
    strategies_list = result.get('strategies', [])
    
    # Fallback if list is empty or malformed
    if not strategies_list:
        print("Warning: Strategist returned no strategies. Using fallback.")
        strategies_list = [{
                "title": "Fallback Analysis",
                "analysis_type": "statistical",
                "hypothesis": "Basic analysis due to generator failure.",
                "required_columns": [], # Dangerous but treated by DE
                "estimated_difficulty": "Low",
                "reasoning": "Fallback"
        }]

    return {
        "strategies": {"strategies": strategies_list} 
    }

def run_domain_expert(state: AgentState) -> AgentState:
    print("--- [2.5] Domain Expert: Evaluating & Selecting Strategy ---")
    abort_state = _abort_if_requested(state, "domain_expert")
    if abort_state:
        return abort_state
    
    strategies_wrapper = state.get('strategies', {})
    strategies_list = strategies_wrapper.get('strategies', [])
    business_objective = state.get('business_objective', '')
    data_summary = state.get('data_summary', '')
    
    # Deliberation Step
    evaluation = domain_expert.evaluate_strategies(data_summary, business_objective, strategies_list)
    reviews = evaluation.get('reviews', [])
    
    # Selection Logic
    best_strategy = None
    best_score = -1.0
    selection_reason = "Default Selection"
    
    # Map reviews to strategies (assuming order consistency or title matching)
    # We use Title matching for robustness
    
    print("\n🧐 EXPERT DELIBERATION:")
    for strat in strategies_list:
        # Find matching review
        match = next((r for r in reviews if r.get('title') == strat.get('title')), None)
        score = match.get('score', 0) if match else 0
        
        print(f"  • Strategy: {strat.get('title')} | Score: {score}/10")
        if match:
             print(f"    - Critique: {match.get('reasoning')[:100]}...")
        
        if score > best_score:
            best_score = score
            best_strategy = strat
            selection_reason = match.get('reasoning') if match else "Highest Score"
            
    # Fallback to first if no valid scores
    if not best_strategy and strategies_list:
        best_strategy = strategies_list[0]
        selection_reason = "Fallback: First strategy selected (No scores available)."
        
    print(f"\n🏆 WINNER: {best_strategy.get('title')} (Score: {best_score})")
    print(f"   Reason: {selection_reason}\n")

    return {
        "selected_strategy": best_strategy,
        "selection_reason": selection_reason,
        "domain_expert_reviews": reviews
    }


def run_execution_planner(state: AgentState) -> AgentState:
    print("--- [2.7] Execution Planner: Building Contract ---")
    abort_state = _abort_if_requested(state, "execution_planner")
    if abort_state:
        return abort_state
    strategy = state.get("selected_strategy", {})
    data_summary = state.get("data_summary", "")
    memory_context = state.get("dataset_memory_context")
    if memory_context:
        data_summary = f"{data_summary}\n\n{memory_context}"
    business_objective = state.get("business_objective", "")
    run_id = state.get("run_id")
    if run_id:
        log_run_event(run_id, "execution_planner_start", {"strategy": strategy.get("title", "")})
    csv_path = state.get("csv_path", "")
    csv_sep = state.get("csv_sep", ",")
    csv_decimal = state.get("csv_decimal", ".")
    csv_encoding = state.get("csv_encoding", "utf-8")
    column_inventory = []
    try:
        import pandas as pd
        header_df = pd.read_csv(csv_path, nrows=0, sep=csv_sep, decimal=csv_decimal, encoding=csv_encoding)
        column_inventory = header_df.columns.tolist()
    except Exception as inv_err:
        print(f"Warning: failed to read column inventory: {inv_err}")
    try:
        contract = execution_planner.generate_contract(
            strategy=strategy,
            data_summary=data_summary,
            business_objective=business_objective,
            column_inventory=column_inventory,
        )
    except Exception as e:
        print(f"Warning: execution planner failed ({e}); using fallback contract.")
        contract = {
            "contract_version": 1,
            "data_requirements": [],
            "validations": [],
            "required_outputs": ["data/cleaned_data.csv"],
            "notes_for_engineers": ["Planner failed; use strategy + data_summary."],
        }
    contract = _normalize_execution_contract(contract)
    contract = ensure_role_runbooks(contract)
    contract = _ensure_scored_rows_output(contract)
    contract = _ensure_alignment_check_output(contract)
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/execution_contract.json", "w", encoding="utf-8") as f:
            import json
            json.dump(contract, f, indent=2)
    except Exception as save_err:
        print(f"Warning: failed to persist execution_contract.json: {save_err}")
    if run_id:
        log_run_event(
            run_id,
            "execution_planner_complete",
            {"required_outputs": contract.get("required_outputs", [])},
        )
    policy = contract.get("iteration_policy") if isinstance(contract, dict) else {}
    result = {"execution_contract": contract}
    if isinstance(policy, dict) and policy:
        result["iteration_policy"] = policy
        runtime_fix_max = policy.get("runtime_fix_max")
        if runtime_fix_max is not None:
            try:
                result["max_runtime_fix_attempts"] = max(1, int(runtime_fix_max))
            except Exception:
                pass
    return result




def run_data_engineer(state: AgentState) -> AgentState:
    print("--- [3] Data Engineer: Cleaning Data (E2B Sandbox) ---")
    abort_state = _abort_if_requested(state, "data_engineer")
    if abort_state:
        return abort_state
    run_id = state.get("run_id")
    ok, counters, err_msg = _consume_budget(state, "de_calls", "max_de_calls", "Data Engineer")
    state["budget_counters"] = counters
    if not ok:
        if run_id:
            log_run_event(run_id, "budget_exceeded", {"label": "data_engineer", "error": err_msg})
        return {
            "cleaning_code": "",
            "cleaned_data_preview": "Error: Budget Exceeded",
            "error_message": err_msg,
            "budget_counters": counters,
        }
    if run_id:
        log_run_event(run_id, "data_engineer_start", {})
    
    selected = state.get('selected_strategy')
    if not selected:
        raise ValueError("No strategy selected for data cleaning.")
        
    business_objective = state.get('business_objective', '')
    csv_path = state['csv_path']
    csv_encoding = state.get('csv_encoding', 'utf-8')
    csv_decimal = state.get('csv_decimal', '.')
    csv_sep = state.get('csv_sep', ',')
    input_dialect = {"encoding": csv_encoding, "sep": csv_sep, "decimal": csv_decimal}
    leakage_audit_summary = state.get("leakage_audit_summary", "")
    data_engineer_audit_override = state.get("data_engineer_audit_override", state.get("data_summary", ""))
    header_cols = _read_csv_header(csv_path, csv_encoding, csv_sep)
    if header_cols:
        norm_map = {}
        for col in header_cols:
            normed = _norm_name(col)
            if normed and normed not in norm_map:
                norm_map[normed] = col
        header_context = (
            "COLUMN_INVENTORY_RAW: "
            + json.dumps(header_cols, ensure_ascii=False)
            + "\nNORMALIZED_HEADER_MAP: "
            + json.dumps(norm_map, ensure_ascii=False)
        )
        data_engineer_audit_override = _merge_de_audit_override(data_engineer_audit_override, header_context)
        required_cols = _resolve_required_input_columns(state.get("execution_contract", {}), selected)
        required_raw_map = _build_required_raw_map(required_cols, norm_map)
        if required_raw_map:
            raw_map_payload = "REQUIRED_RAW_HEADER_MAP:\n" + json.dumps(required_raw_map, ensure_ascii=True)
            data_engineer_audit_override = _merge_de_audit_override(data_engineer_audit_override, raw_map_payload)
        sample_context = _build_required_sample_context(csv_path, input_dialect, required_cols, norm_map)
        if sample_context:
            data_engineer_audit_override = _merge_de_audit_override(data_engineer_audit_override, sample_context)
    state["data_engineer_audit_override"] = data_engineer_audit_override
    de_contract = _filter_contract_for_data_engineer(state.get("execution_contract", {}))
    try:
        os.makedirs("artifacts", exist_ok=True)
        context_payload = {
            "csv_path": csv_path,
            "csv_encoding": csv_encoding,
            "csv_sep": csv_sep,
            "csv_decimal": csv_decimal,
            "header_cols": header_cols,
            "required_input_columns": _resolve_required_input_columns(state.get("execution_contract", {}), selected),
            "required_all_columns": _resolve_contract_columns_for_cleaning(de_contract),
            "required_raw_header_map": required_raw_map,
            "raw_required_sample_context": sample_context,
            "data_engineer_audit_override": data_engineer_audit_override,
        }
        with open(os.path.join("artifacts", "data_engineer_context.json"), "w", encoding="utf-8") as f_ctx:
            json.dump(context_payload, f_ctx, indent=2, ensure_ascii=False)
    except Exception as ctx_err:
        print(f"Warning: failed to persist data_engineer_context.json: {ctx_err}")

    # Generate cleaning script (targeting REMOTE path)
    code = data_engineer.generate_cleaning_script(
        data_engineer_audit_override,
        selected,
        input_path="data/raw.csv", # Remote Sandbox Path
        business_objective=business_objective,
        csv_encoding=csv_encoding,
        csv_sep=csv_sep,
        csv_decimal=csv_decimal,
        execution_contract=de_contract,
    )
    try:
        os.makedirs("artifacts", exist_ok=True)
        with open(os.path.join("artifacts", "data_engineer_last.py"), "w", encoding="utf-8") as f_art:
            f_art.write(code)
    except Exception as art_err:
        print(f"Warning: failed to persist data_engineer_last.py: {art_err}")

    plan_payload = None
    is_plan = False
    plan_issues: List[str] = []
    force_code_mode = state.get("force_code_mode", True)
    plan, _ = parse_cleaning_plan(code)
    if plan:
        if force_code_mode:
            msg = "CLEANING_PLAN_NOT_ALLOWED: expected Python cleaning code."
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Error: Plan Output",
                "error_message": msg,
                "budget_counters": counters,
            }
        is_plan = True
        plan_payload = plan
        plan_issues = validate_cleaning_plan(plan)
        try:
            os.makedirs("artifacts", exist_ok=True)
            with open(os.path.join("artifacts", "data_engineer_last_plan.json"), "w", encoding="utf-8") as f_plan:
                json.dump(plan_payload, f_plan, indent=2, ensure_ascii=False)
        except Exception as art_err:
            print(f"Warning: failed to persist data_engineer_last_plan.json: {art_err}")
        if plan_issues:
            msg = "CLEANING_PLAN_INVALID: " + "; ".join(plan_issues)
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Error: Invalid Plan",
                "error_message": msg,
                "budget_counters": counters,
            }
    
    # Check if generation failed
    if code.strip().startswith("# Error"):
        print(f"Correction: Data Engineer failed to generate code. Error: {code}")
        return {
            "cleaning_code": code,
            "cleaned_data_preview": "Error: Generation Failed",
            "error_message": code,
            "budget_counters": counters,
        }

    if "GENERATED CODE BLOCKED BY STATIC SCAN" in code:
        if not state.get("de_static_guard_retry_done"):
            new_state = dict(state)
            new_state["de_static_guard_retry_done"] = True
            base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
            payload = "SECURITY_VIOLATION_CONTEXT:\n" + code.strip()
            new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, payload)
            print("Static scan guard: retrying Data Engineer with violation context.")
            return run_data_engineer(new_state)
        return {
            "cleaning_code": code,
            "cleaned_data_preview": "Error: Security Blocked",
            "error_message": "CRITICAL: cleaning code blocked by static scan.",
            "budget_counters": counters,
        }

    if not is_plan:
        preflight_issues = data_engineer_preflight(code)
        if preflight_issues:
            msg = "DATA_ENGINEER_PREFLIGHT: " + "; ".join(preflight_issues)
            fh = list(state.get("feedback_history", []))
            fh.append(msg)
            lgc = {
                "source": "data_engineer_preflight",
                "status": "REJECTED",
                "feedback": msg,
                "required_fixes": preflight_issues,
                "failed_gates": preflight_issues,
            }
            try:
                print(msg)
            except Exception:
                pass
            if not state.get("de_preflight_retry_done"):
                new_state = dict(state)
                new_state["de_preflight_retry_done"] = True
                override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                try:
                    override += "\n\nPREFLIGHT_ERROR_CONTEXT:\n" + msg
                    explainer_text = ""
                    try:
                        required_input = _resolve_required_input_columns(state.get("execution_contract", {}), selected)
                        header_cols = _read_csv_header(csv_path, csv_encoding, csv_sep)
                        norm_map = {}
                        for col in header_cols:
                            normed = _norm_name(col)
                            if normed and normed not in norm_map:
                                norm_map[normed] = col
                        required_raw_map = _build_required_raw_map(required_input, norm_map)
                        explainer_ctx = {
                            "strategy_title": selected.get("title", "") if selected else "",
                            "csv_dialect": input_dialect,
                            "required_input_columns": required_input,
                            "required_raw_header_map": required_raw_map,
                            "preflight_issues": preflight_issues,
                        }
                        explainer_text = failure_explainer.explain_data_engineer_failure(
                            code=code,
                            error_details=msg,
                            context=explainer_ctx,
                        )
                    except Exception as explainer_err:
                        print(f"Warning: failure explainer failed: {explainer_err}")
                        explainer_text = ""
                    if explainer_text:
                        override += "\nLLM_FAILURE_EXPLANATION:\n" + explainer_text.strip()
                        try:
                            os.makedirs("artifacts", exist_ok=True)
                            with open(
                                os.path.join("artifacts", "data_engineer_preflight_explainer.txt"),
                                "w",
                                encoding="utf-8",
                            ) as f_exp:
                                f_exp.write(explainer_text.strip())
                        except Exception as exp_err:
                            print(f"Warning: failed to persist data_engineer_preflight_explainer.txt: {exp_err}")
                except Exception:
                    pass
                new_state["data_engineer_audit_override"] = override
                print("Preflight guard: retrying Data Engineer with explainer context.")
                return run_data_engineer(new_state)
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Preflight Failed",
                "error_message": msg,
                "feedback_history": fh,
                "last_gate_context": lgc,
                "budget_counters": counters,
            }

    # 0a. Undefined name preflight for Data Engineer code
    if not is_plan:
        undefined = detect_undefined_names(code)
        if undefined:
            msg = f"STATIC_PRECHECK_UNDEFINED: Undefined names detected preflight: {', '.join(undefined)}"
            if not state.get("de_undefined_retry_done"):
                new_state = dict(state)
                new_state["de_undefined_retry_done"] = True
                override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                try:
                    override += (
                        "\n\nUNDEFINED_NAME_GUARD: Ensure every referenced name is defined in the same scope. "
                        "Avoid using variables created inside helper functions in outer scopes. "
                        "Do not inline JSON literals (null/true/false) into Python code."
                    )
                except Exception:
                    pass
                new_state["data_engineer_audit_override"] = override
                print("Undefined name guard triggered: retrying Data Engineer with safety instructions.")
                return run_data_engineer(new_state)
            fh = list(state.get("feedback_history", []))
            fh.append(msg)
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Preflight Failed",
                "error_message": msg,
                "feedback_history": fh,
                "budget_counters": counters,
            }

    if not is_plan and contains_json_null_literal(code):
        if not state.get("de_json_literal_retry_done"):
            new_state = dict(state)
            new_state["de_json_literal_retry_done"] = True
            override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
            try:
                override += (
                    "\n\nCONTRACT_LITERAL_GUARD: Do not inline raw JSON into Python code. "
                    "JSON literals like null/true/false are invalid identifiers in Python. "
                    "If you need the execution contract, read it from data/execution_contract.json "
                    "or use Python-native values."
                )
            except Exception:
                pass
            new_state["data_engineer_audit_override"] = override
            print("Contract literal guard triggered: retrying Data Engineer with safety instructions.")
            return run_data_engineer(new_state)
        return {
            "cleaning_code": code,
            "cleaned_data_preview": "Error: Contract Literal Guard",
            "error_message": "CRITICAL: JSON literals (null/true/false) found in Python code.",
            "budget_counters": counters,
        }

    # Manifest serialization guard: ensure json.dump uses default=
    if not is_plan and manifest_dump_missing_default(code):
        if not state.get("de_manifest_retry_done"):
            new_state = dict(state)
            new_state["de_manifest_retry_done"] = True
            override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
            try:
                override += "\n\nMANIFEST_JSON_SAFETY: ensure json.dump(manifest, ..., default=_json_default) to handle numpy/pandas types."
            except Exception:
                pass
            new_state["data_engineer_audit_override"] = override
            print("Manifest JSON guard triggered: retrying Data Engineer with safety instructions.")
            return run_data_engineer(new_state)
        else:
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Error: Manifest JSON Guard",
                "error_message": "CRITICAL: Manifest serialization must use json.dump(..., default=_json_default).",
                "budget_counters": counters,
            }

    # Dialect enforcement guard: ensure pd.read_csv uses provided dialect
    dialect_issues = []
    if not is_plan:
        dialect_issues = dialect_guard_violations(code, csv_sep, csv_decimal, csv_encoding, expected_path="data/raw.csv")
    if dialect_issues:
        if not state.get("de_dialect_retry_done"):
            new_state = dict(state)
            new_state["de_dialect_retry_done"] = True
            override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
            try:
                override += "\n\nDIALECT_GUARD:\n" + "\n".join(dialect_issues)
            except Exception:
                pass
            new_state["data_engineer_audit_override"] = override
            print("⚠️ DIALECT_GUARD: retrying Data Engineer with enforced dialect instructions.")
            return run_data_engineer(new_state)
        else:
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Error: Dialect Guard",
                "error_message": "CRITICAL: pd.read_csv must use provided dialect parameters.",
                "budget_counters": counters,
            }

    # STATIC SAFETY SCAN FOR CLEANING CODE (HARDENING)
    is_safe, violations = (True, [])
    if not is_plan:
        is_safe, violations = scan_code_safety(code)
    if not is_safe:
        failure_reason = "CRITICAL: Security Violations:\n" + "\n".join(violations)
        print(f"🚫 Security Block (Data Engineer): {failure_reason}")
        return {
            "cleaning_code": code,
            "cleaned_data_preview": "Security Block",
            "error_message": f"Cleaning Code Blocked: {failure_reason}",
            "budget_counters": counters,
        }

    # Execute in E2B Sandbox
    try:
        local_cleaned_path = "data/cleaned_data.csv"
        local_manifest_path = "data/cleaning_manifest.json"
        output_log = ""
        if is_plan:
            try:
                plan_exec = dict(plan_payload or {})
                input_cfg = plan_exec.get("input")
                if not isinstance(input_cfg, dict):
                    input_cfg = {}
                input_cfg["path"] = csv_path
                plan_exec["input"] = input_cfg
                dialect_cfg = plan_exec.get("dialect")
                if not isinstance(dialect_cfg, dict):
                    dialect_cfg = {}
                dialect_cfg.setdefault("sep", csv_sep)
                dialect_cfg.setdefault("decimal", csv_decimal)
                dialect_cfg.setdefault("encoding", csv_encoding)
                plan_exec["dialect"] = dialect_cfg
                result = execute_cleaning_plan(plan_exec, state.get("execution_contract", {}) or {})
                local_cleaned_path = result.get("cleaned_path", local_cleaned_path)
                local_manifest_path = result.get("manifest_path", local_manifest_path)
                csv_sep, csv_decimal, csv_encoding, dialect_updated = get_output_dialect_from_manifest(
                    local_manifest_path, csv_sep, csv_decimal, csv_encoding
                )
                if dialect_updated:
                    print(f"Downstream dialect updated from output_dialect: sep={csv_sep}, decimal={csv_decimal}, encoding={csv_encoding}")
            except Exception as plan_err:
                msg = f"Host Cleaning Plan Failed: {plan_err}"
                return {"cleaning_code": code, "cleaned_data_preview": "Error: Plan Failed", "error_message": msg, "budget_counters": counters}
        else:
            # load_dotenv() is called at module level or main
            api_key = os.getenv("E2B_API_KEY")
            if not api_key:
                msg = "CRITICAL: E2B_API_KEY missing in .env file."
                return {"error_message": msg, "cleaned_data_preview": "Error: Missing E2B Key", "budget_counters": counters}
            
            os.environ["E2B_API_KEY"] = api_key
    
            with Sandbox.create() as sandbox:
                # 1. Setup Environment
                print("Installing dependencies in Sandbox...")
                sandbox.commands.run("pip install pandas numpy")
                sandbox.commands.run("mkdir -p data")
                
                # 2. Upload Raw Data
                print(f"Uploading {csv_path} to sandbox...")
                with open(csv_path, "rb") as f:
                    sandbox.files.write("data/raw.csv", f)
    
                # 2.5 Upload Execution Contract (best-effort)
                local_contract_path = "data/execution_contract.json"
                if os.path.exists(local_contract_path):
                    try:
                        with open(local_contract_path, "rb") as f:
                            sandbox.files.write("data/execution_contract.json", f)
                    except Exception as contract_err:
                        print(f"Warning: failed to upload execution_contract.json: {contract_err}")
    
                # 3. Execute Cleaning
                print("Executing Cleaning Script in Sandbox...")
                execution = sandbox.run_code(code)
                
                # Capture Output
                output_log = ""
                if execution.logs.stdout: output_log += "\n".join(execution.logs.stdout)
                if execution.logs.stderr: output_log += "\n".join(execution.logs.stderr)
                try:
                    os.makedirs("artifacts", exist_ok=True)
                    with open(os.path.join("artifacts", "data_engineer_sandbox_last.log"), "w", encoding="utf-8") as f_log:
                        f_log.write(output_log or "")
                except Exception as log_err:
                    print(f"Warning: failed to persist data_engineer_sandbox_last.log: {log_err}")
                
                if execution.error:
                    error_details = f"{execution.error.name}: {execution.error.value}\n{execution.error.traceback}"
                    print(f"Cleaning Failed in Sandbox: {error_details}")
                    contract = state.get("execution_contract", {}) or {}
                    derived_cols = _resolve_contract_columns(contract, sources={"derived", "output"})
                    if "Missing required columns" in error_details and derived_cols and not state.get("de_missing_derived_retry_done"):
                        new_state = dict(state)
                        new_state["de_missing_derived_retry_done"] = True
                        override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                        try:
                            override += (
                                "\n\nDERIVED_COLUMN_MISSING_GUARD: "
                                f"{derived_cols} are derived. Do not treat them as required input columns. "
                                "Only enforce source='input' columns after canonicalization; derive the rest."
                            )
                        except Exception:
                            pass
                        new_state["data_engineer_audit_override"] = override
                        print("Derived column missing guard: retrying Data Engineer with derived-column guidance.")
                        return run_data_engineer(new_state)
                    if "actual_column" in error_details and "NoneType" in error_details and not state.get("de_actual_column_guard_retry_done"):
                        new_state = dict(state)
                        new_state["de_actual_column_guard_retry_done"] = True
                        override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                        try:
                            override += (
                                "\n\nVALIDATION_PRINT_GUARD: Use actual = str(result.get('actual_column') or 'MISSING') "
                                "before slicing; do not subscript None."
                            )
                        except Exception:
                            pass
                        new_state["data_engineer_audit_override"] = override
                        print("Validation print guard: retrying Data Engineer with None-safe formatting.")
                        return run_data_engineer(new_state)
                    if "AttributeError" in error_details and "DataFrame" in error_details and "dtype" in error_details:
                        if not state.get("de_dtype_guard_retry_done"):
                            new_state = dict(state)
                            new_state["de_dtype_guard_retry_done"] = True
                            override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                            try:
                                override += (
                                    "\n\nDUPLICATE_COLUMN_GUARD: When reading dtype, handle duplicate column labels. "
                                    "Use series = df[actual_col]; if isinstance(series, pd.DataFrame), "
                                    "use series = series.iloc[:, 0] and log a warning."
                                )
                            except Exception:
                                pass
                            new_state["data_engineer_audit_override"] = override
                            print("Duplicate column dtype guard: retrying Data Engineer.")
                            return run_data_engineer(new_state)
                    if not state.get("de_runtime_retry_done"):
                        new_state = dict(state)
                        new_state["de_runtime_retry_done"] = True
                        override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                        try:
                            override += "\n\nRUNTIME_ERROR_CONTEXT:\n" + error_details[-2000:]
                            failure_cause = _infer_de_failure_cause(error_details)
                            if failure_cause:
                                override += "\nWHY_IT_HAPPENED: " + failure_cause
                            diagnosis_lines = _build_de_runtime_diagnosis(error_details)
                            if diagnosis_lines:
                                override += "\nERROR_DIAGNOSIS:\n- " + "\n- ".join(diagnosis_lines)
                            explainer_text = ""
                            try:
                                required_input = _resolve_required_input_columns(state.get("execution_contract", {}), selected)
                                header_cols = _read_csv_header(csv_path, csv_encoding, csv_sep)
                                norm_map = {}
                                for col in header_cols:
                                    normed = _norm_name(col)
                                    if normed and normed not in norm_map:
                                        norm_map[normed] = col
                                required_raw_map = _build_required_raw_map(required_input, norm_map)
                                explainer_ctx = {
                                    "strategy_title": selected.get("title", "") if selected else "",
                                    "csv_dialect": input_dialect,
                                    "required_input_columns": required_input,
                                    "required_raw_header_map": required_raw_map,
                                }
                                explainer_text = failure_explainer.explain_data_engineer_failure(
                                    code=code,
                                    error_details=error_details,
                                    context=explainer_ctx,
                                )
                            except Exception as explainer_err:
                                print(f"Warning: failure explainer failed: {explainer_err}")
                                explainer_text = ""
                            if explainer_text:
                                override += "\nLLM_FAILURE_EXPLANATION:\n" + explainer_text.strip()
                                try:
                                    os.makedirs("artifacts", exist_ok=True)
                                    with open(
                                        os.path.join("artifacts", "data_engineer_failure_explainer.txt"),
                                        "w",
                                        encoding="utf-8",
                                    ) as f_exp:
                                        f_exp.write(explainer_text.strip())
                                except Exception as exp_err:
                                    print(f"Warning: failed to persist data_engineer_failure_explainer.txt: {exp_err}")
                        except Exception:
                            pass
                        new_state["data_engineer_audit_override"] = override
                        print("Runtime error guard: retrying Data Engineer with error context.")
                        return run_data_engineer(new_state)
                    return {
                        "cleaning_code": code,
                        "cleaned_data_preview": "Error: Cleaning Failed",
                        "error_message": f"Sandbox Cleaning Failed: {error_details}",
                        "budget_counters": counters,
                    }
                
                # 4. Verification & Download
                # Check if cleaned file exists
                ls_check = sandbox.commands.run("ls data/cleaned_data.csv")
                if ls_check.exit_code != 0:
                      return {
                         "cleaning_code": code,
                         "cleaned_data_preview": "Error: File Not Created",
                         "error_message": f"Cleaning script finished but data/cleaned_data.csv was not found.\nLogs:\n{output_log}",
                         "budget_counters": counters,
                    }
    
                # Persist DE artifact
                try:
                    os.makedirs("artifacts", exist_ok=True)
                    with open(os.path.join("artifacts", "data_engineer_last.py"), "w", encoding="utf-8") as f_art:
                        f_art.write(code)
                except Exception as art_err:
                    print(f"Warning: failed to persist data_engineer_last.py: {art_err}")
    
                # Download Result (CSV)
                print("Downloading cleaned data...")
                local_cleaned_path = "data/cleaned_data.csv"
                os.makedirs("data", exist_ok=True)
                
                try:
                    # Try standard download
                    with open(local_cleaned_path, "wb") as f_local:
                        remote_bytes = sandbox.files.read("data/cleaned_data.csv")
                        f_local.write(remote_bytes)
                except Exception as dl_err:
                    print(f"Standard download failed ({dl_err}), trying Base64 fallback...")
                    proc = sandbox.commands.run("base64 -w 0 data/cleaned_data.csv")
                    if proc.exit_code == 0:
                        b64_content = proc.stdout.strip()
                        decoded = base64.b64decode(b64_content)
                        with open(local_cleaned_path, "wb") as f_local:
                            f_local.write(decoded)
                    else:
                          return {
                            "cleaning_code": code,
                            "cleaned_data_preview": "Error: Download Failed",
                            "error_message": f"Failed to download cleaned data: {proc.stderr}",
                            "budget_counters": counters,
                        }
    
                # Download Manifest (JSON) - Roundtrip Support
                print("Downloading cleaning manifest...")
                local_manifest_path = "data/cleaning_manifest.json"
                try:
                    # Try standard download
                    with open(local_manifest_path, "wb") as f_local:
                        remote_bytes = sandbox.files.read("data/cleaning_manifest.json")
                        f_local.write(remote_bytes)
                except Exception:
                    # Fallback Base64
                    proc = sandbox.commands.run("base64 -w 0 data/cleaning_manifest.json")
                    if proc.exit_code == 0:
                        b64_content = proc.stdout.strip()
                        decoded = base64.b64decode(b64_content)
                        with open(local_manifest_path, "wb") as f_local:
                            f_local.write(decoded)
                    else:
                        print("Warning: Manifest download failed (not found?). Creating default.")
                        # Create default manifest if missing
                        default_manifest = {
                            "input_dialect": {"encoding": csv_encoding, "sep": csv_sep, "decimal": csv_decimal},
                            "output_dialect": {"encoding": "utf-8", "sep": ",", "decimal": "."},
                            "generated_by": "Host Fallback"
                        }
                        with open(local_manifest_path, "w") as f_def:
                            json.dump(default_manifest, f_def)
                try:
                    os.makedirs("artifacts", exist_ok=True)
                    with open(os.path.join("artifacts", "cleaning_manifest_last.json"), "wb") as f_copy:
                        with open(local_manifest_path, "rb") as f_src:
                            f_copy.write(f_src.read())
                except Exception as copy_err:
                    print(f"Warning: failed to persist cleaning_manifest_last.json: {copy_err}")

                manifest_data = {}
                try:
                    with open(local_manifest_path, "r", encoding="utf-8") as f_manifest:
                        manifest_data = json.load(f_manifest)
                except Exception:
                    manifest_data = {}
                manifest_alerts = _extract_manifest_alerts(
                    manifest_data,
                    _resolve_contract_columns(state.get("execution_contract", {}), sources={"derived", "output"})
                )
                if manifest_alerts and not state.get("de_manifest_guard_retry_done"):
                    new_state = dict(state)
                    new_state["de_manifest_guard_retry_done"] = True
                    base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                    payload = "MANIFEST_ALERTS:\n" + "\n".join(f"- {alert}" for alert in manifest_alerts)
                    new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, payload)
                    print("MANIFEST_GUARD: retrying Data Engineer once with manifest alerts.")
                    return run_data_engineer(new_state)

                required_input = _resolve_required_input_columns(state.get("execution_contract", {}), selected)
                raw_rows = _count_raw_rows(csv_path, csv_encoding, csv_sep, csv_decimal)
                cleaned_encoding = None
                try:
                    cleaned_encoding = (
                        manifest_data.get("output_dialect", {}).get("encoding")
                        if isinstance(manifest_data.get("output_dialect"), dict)
                        else None
                    )
                except Exception:
                    cleaned_encoding = None
                cleaned_encoding = cleaned_encoding or "utf-8"
                cleaned_sep = None
                cleaned_decimal = None
                try:
                    if isinstance(manifest_data.get("output_dialect"), dict):
                        cleaned_sep = manifest_data["output_dialect"].get("sep")
                        cleaned_decimal = manifest_data["output_dialect"].get("decimal")
                except Exception:
                    cleaned_sep = None
                    cleaned_decimal = None
                cleaned_rows = _count_raw_rows(local_cleaned_path, cleaned_encoding, cleaned_sep, cleaned_decimal)
                row_drop_summary = _summarize_row_drop(
                    manifest_data,
                    required_input,
                    initial_override=raw_rows,
                    after_override=cleaned_rows,
                )
                if row_drop_summary and not state.get("de_row_drop_retry_done"):
                    new_state = dict(state)
                    new_state["de_row_drop_retry_done"] = True
                    base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                    payload = (
                        "CLEANING_RECOVERY_ALERT:\n"
                        f"- rows_initial={row_drop_summary.get('initial')}\n"
                        f"- rows_after={row_drop_summary.get('after')}\n"
                        f"- drop_frac={row_drop_summary.get('drop_frac')}\n"
                    )
                    suspects = row_drop_summary.get("suspects") or []
                    if suspects:
                        payload += "SUSPECT_CONVERSIONS:\n" + json.dumps(suspects, ensure_ascii=True)
                        try:
                            header_cols = _read_csv_header(csv_path, csv_encoding, csv_sep)
                            norm_map = {}
                            for col in header_cols:
                                normed = _norm_name(col)
                                if normed and normed not in norm_map:
                                    norm_map[normed] = col
                            suspect_cols = [item.get("column") for item in suspects if item.get("column")]
                            sample_context = _build_required_sample_context(
                                csv_path, input_dialect, suspect_cols, norm_map, max_rows=200
                            )
                            if sample_context:
                                payload = f"{payload}\n\n{sample_context}"
                        except Exception:
                            pass
                    try:
                        explainer_text = failure_explainer.explain_data_engineer_failure(
                            code=code,
                            error_details=payload,
                            context={
                                "strategy_title": selected.get("title", "") if selected else "",
                                "csv_dialect": input_dialect,
                                "required_input_columns": required_input,
                                "row_drop_summary": row_drop_summary,
                            },
                        )
                    except Exception as explainer_err:
                        print(f"Warning: failure explainer failed: {explainer_err}")
                        explainer_text = ""
                    if explainer_text:
                        payload = payload + "\nLLM_FAILURE_EXPLANATION:\n" + explainer_text.strip()
                        try:
                            os.makedirs("artifacts", exist_ok=True)
                            with open(
                                os.path.join("artifacts", "data_engineer_failure_explainer.txt"),
                                "w",
                                encoding="utf-8",
                            ) as f_exp:
                                f_exp.write(explainer_text.strip())
                        except Exception as exp_err:
                            print(f"Warning: failed to persist data_engineer_failure_explainer.txt: {exp_err}")
                    new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, payload)
                    print("ROW_DROP_GUARD: retrying Data Engineer with recovery context.")
                    return run_data_engineer(new_state)

                print("Cleaning Success (Artifacts Downloaded).")
    
                # Apply output_dialect for downstream reads
                csv_sep, csv_decimal, csv_encoding, dialect_updated = get_output_dialect_from_manifest(
                    local_manifest_path, csv_sep, csv_decimal, csv_encoding
                )
                if dialect_updated:
                    print(f"Downstream dialect updated from output_dialect: sep={csv_sep}, decimal={csv_decimal}, encoding={csv_encoding}")
    
        if not os.path.exists(local_cleaned_path):
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Error: File Not Created",
                "error_message": f"Cleaning finished but {local_cleaned_path} was not found.\nLogs:\n{output_log}",
                "budget_counters": counters,
            }
        if not os.path.exists(local_manifest_path):
            default_manifest = {
                "input_dialect": {"encoding": csv_encoding, "sep": csv_sep, "decimal": csv_decimal},
                "output_dialect": {"encoding": "utf-8", "sep": ",", "decimal": "."},
                "generated_by": "Host Fallback",
            }
            os.makedirs(os.path.dirname(local_manifest_path) or ".", exist_ok=True)
            with open(local_manifest_path, "w", encoding="utf-8") as f_def:
                json.dump(default_manifest, f_def, indent=2)
        try:
            os.makedirs("artifacts", exist_ok=True)
            with open(os.path.join("artifacts", "cleaning_manifest_last.json"), "wb") as f_copy:
                with open(local_manifest_path, "rb") as f_src:
                    f_copy.write(f_src.read())
        except Exception as copy_err:
            print(f"Warning: failed to persist cleaning_manifest_last.json: {copy_err}")
        # --- HOST-SIDE COLUMN MAPPING PROTOCOL v2 ---
        import pandas as pd
        from src.utils.column_mapping import build_mapping
        from src.utils.cleaning_validation import (
            normalize_manifest,
            load_json,
            detect_destructive_drop,
            format_patch_instructions,
            sample_raw_columns,
        )
        
        try:
            df = pd.read_csv(
                local_cleaned_path,
                sep=csv_sep,
                decimal=csv_decimal,
                encoding=csv_encoding
            )
            assert_not_single_column_delimiter_mismatch(df, csv_sep, csv_decimal, csv_encoding)
            if df.empty:
                raise ValueError(
                    f"Delimiter/Dialect mismatch: cleaned dataset loaded empty with sep='{csv_sep}', decimal='{csv_decimal}', encoding='{csv_encoding}'."
                )
            contract = state.get("execution_contract", {}) or {}
            required_cols = _resolve_required_input_columns(contract, selected)
            contract_all_cols = _resolve_contract_columns(contract)
            contract_derived_cols = _resolve_contract_columns_for_cleaning(contract, sources={"derived", "output"})
            cleaned_columns_set = set(df.columns.str.lower())
            
            print(f"Applying Column Mapping v2 for strategy: {selected.get('title')}")
            print(f"Required: {required_cols}")
            
            mapping_result = build_mapping(required_cols, df.columns.tolist())
            
            # Check for Missing Critical Columns (input only)
            if mapping_result['missing']:
                missing_cols = mapping_result['missing']
                missing_input = [m for m in missing_cols]
                if missing_input:
                    alias_conflicts = [req for req, meta in mapping_result.get("summary", {}).items() if meta.get("method") == "alias_conflict"]
                    manifest_raw = normalize_manifest(load_json(local_manifest_path))
                    issues = detect_destructive_drop(
                        manifest_raw,
                        missing_input,
                        csv_path,
                        input_dialect
                    )
                    if issues and not state.get("de_destructive_retry_done"):
                        patch = format_patch_instructions(issues)
                        new_state = dict(state)
                        new_state["de_destructive_retry_done"] = True
                        base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                        new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, patch)
                        print("⚠️ DESTRUCTIVE_CONVERSION_GUARD: retrying Data Engineer once with patch instructions.")
                        return run_data_engineer(new_state)
                    conflict_msg = f" Alias conflicts: {alias_conflicts}" if alias_conflicts else ""
                    failure_msg = f"CRITICAL: Missing required columns after mapping: {missing_input}.{conflict_msg} Cleaning failed to provide necessary features."
                    return {
                        "cleaning_code": code,
                        "cleaned_data_preview": "Error: Missing Columns",
                        "error_message": failure_msg,
                        "csv_sep": csv_sep,
                        "csv_decimal": csv_decimal,
                        "csv_encoding": csv_encoding,
                        "leakage_audit_summary": leakage_audit_summary,
                        "budget_counters": counters,
                    }
            
            # Apply Renaming to Canonical Names
            df_mapped = df.rename(columns=mapping_result['rename_mapping'])

            # Best-effort rename for derived/output columns already present
            if contract_all_cols:
                existing_norm = {}
                for col in df_mapped.columns:
                    normed = _norm_name(col)
                    if normed and normed not in existing_norm:
                        existing_norm[normed] = col
                derived_rename = {}
                for req_name in contract_all_cols:
                    if req_name in df_mapped.columns:
                        continue
                    req_norm = _norm_name(req_name)
                    match = existing_norm.get(req_norm)
                    if match and match not in derived_rename:
                        derived_rename[match] = req_name
                if derived_rename:
                    df_mapped = df_mapped.rename(columns=derived_rename)

            empty_required = _find_empty_required_columns(df_mapped, required_cols)
            if empty_required:
                if not state.get("de_empty_required_retry_done"):
                    new_state = dict(state)
                    new_state["de_empty_required_retry_done"] = True
                    base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                    payload = "EMPTY_REQUIRED_COLUMNS:\n" + "\n".join(
                        f"- {item['column']}: null_frac={item['null_frac']:.2%}, non_null_count={item['non_null_count']}"
                        for item in empty_required
                    )
                    try:
                        header_cols = _read_csv_header(csv_path, csv_encoding, csv_sep)
                        norm_map = {}
                        for col in header_cols:
                            normed = _norm_name(col)
                            if normed and normed not in norm_map:
                                norm_map[normed] = col
                        empty_cols = [item["column"] for item in empty_required]
                        sample_context = _build_required_sample_context(csv_path, input_dialect, empty_cols, norm_map, max_rows=200)
                        if sample_context:
                            payload = f"{payload}\n\n{sample_context}"
                    except Exception:
                        pass
                    new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, payload)
                    print("Empty required columns guard: retrying Data Engineer with evidence.")
                    return run_data_engineer(new_state)
                return {
                    "cleaning_code": code,
                    "cleaned_data_preview": "Error: Empty Required Columns",
                    "error_message": "CRITICAL: Required input columns are empty after cleaning.",
                }

            # --- CLEANING REVIEWER (LLM) ---
            if cleaning_reviewer:
                try:
                    manifest_obj = load_json(local_manifest_path)
                    conversions = manifest_obj.get("conversions", []) if isinstance(manifest_obj, dict) else []
                    type_checks = manifest_obj.get("type_checks", []) if isinstance(manifest_obj, dict) else []
                    leakage_check = manifest_obj.get("leakage_check", {}) if isinstance(manifest_obj, dict) else {}
                    steward_summary = _load_json_safe("data/steward_summary.json")

                    cleaned_stats = {}
                    cleaned_samples = {}
                    cleaned_value_counts = {}
                    cleaned_range_stats = {}
                    cleaned_preview_rows = []
                    review_columns = list(dict.fromkeys(required_cols + list(contract_derived_cols or [])))
                    for col in review_columns:
                        if col not in df_mapped.columns:
                            continue
                        series = df_mapped[col]
                        if series.dtype == object:
                            stripped = series.astype(str).str.strip()
                            null_like = series.isna() | (stripped == "")
                        else:
                            null_like = series.isna()
                        cleaned_stats[col] = {
                            "dtype": str(series.dtype),
                            "null_frac": float(null_like.mean()) if len(series) else 0.0,
                            "non_null_count": int((~null_like).sum()),
                            "unique_count": int(series.nunique(dropna=True)),
                        }
                        if pd.api.types.is_numeric_dtype(series):
                            try:
                                cleaned_range_stats[col] = {
                                    "min": float(series.min()),
                                    "max": float(series.max()),
                                }
                            except Exception:
                                cleaned_range_stats[col] = {}
                        try:
                            cleaned_samples[col] = series.dropna().astype(str).head(6).tolist()
                        except Exception:
                            cleaned_samples[col] = []
                        try:
                            vc = series.dropna().astype(str).value_counts().head(6)
                            cleaned_value_counts[col] = {str(k): int(v) for k, v in vc.items()}
                        except Exception:
                            cleaned_value_counts[col] = {}
                    try:
                        preview_df = df_mapped[review_columns].head(6).copy()
                        cleaned_preview_rows = preview_df.astype(str).to_dict(orient="records")
                    except Exception:
                        cleaned_preview_rows = []

                    raw_samples = {}
                    raw_pattern_stats = {}
                    raw_value_counts = {}
                    raw_null_stats = {}
                    raw_vs_clean_null_delta = {}
                    try:
                        sample_df = sample_raw_columns(csv_path, input_dialect, required_cols, nrows=200, dtype=str)
                        if not sample_df.empty:
                            for col in sample_df.columns:
                                raw_series = sample_df[col].dropna().astype(str)
                                raw_samples[col] = raw_series.head(6).tolist()
                                try:
                                    raw_str = sample_df[col].astype(str).str.strip()
                                    null_like = sample_df[col].isna() | raw_str.eq("") | raw_str.str.lower().isin(["nan", "null", "none"])
                                    raw_null_stats[col] = {
                                        "null_frac": float(null_like.mean()) if len(sample_df) else 0.0,
                                        "sample_rows": int(len(sample_df)),
                                    }
                                    if col in cleaned_stats:
                                        raw_vs_clean_null_delta[col] = float(cleaned_stats[col]["null_frac"] - raw_null_stats[col]["null_frac"])
                                except Exception:
                                    raw_null_stats[col] = {}
                                if not raw_series.empty:
                                    total = len(raw_series)
                                    try:
                                        vc = raw_series.value_counts().head(6)
                                        raw_value_counts[col] = {str(k): int(v) for k, v in vc.items()}
                                    except Exception:
                                        raw_value_counts[col] = {}
                                    raw_pattern_stats[col] = {
                                        "numeric_like_ratio": float(sum(any(ch.isdigit() for ch in val) for val in raw_series) / total),
                                        "dot_ratio": float(sum("." in val for val in raw_series) / total),
                                        "comma_ratio": float(sum("," in val for val in raw_series) / total),
                                        "percent_ratio": float(sum("%" in val for val in raw_series) / total),
                                        "currency_ratio": float(sum(("€" in val) or ("â‚¬" in val) or ("$" in val) or ("£" in val) for val in raw_series) / total),
                                    }
                    except Exception:
                        raw_samples = {}
                        raw_pattern_stats = {}
                        raw_value_counts = {}

                    review_context = {
                        "business_objective": business_objective,
                        "strategy_title": selected.get("title", ""),
                        "required_columns": required_cols,
                        "input_dialect": input_dialect,
                        "raw_samples": raw_samples,
                        "raw_pattern_stats": raw_pattern_stats,
                        "raw_value_counts": raw_value_counts,
                        "raw_null_stats": raw_null_stats,
                        "raw_vs_clean_null_delta": raw_vs_clean_null_delta,
                        "cleaned_stats": cleaned_stats,
                        "cleaned_samples": cleaned_samples,
                        "cleaned_value_counts": cleaned_value_counts,
                        "cleaned_range_stats": cleaned_range_stats,
                        "cleaned_preview_rows": cleaned_preview_rows,
                        "cleaning_manifest": manifest_obj,
                        "manifest_conversions": conversions,
                        "manifest_type_checks": type_checks,
                        "leakage_check": leakage_check,
                        "steward_summary": steward_summary,
                        "execution_contract": {
                            "data_requirements": contract.get("data_requirements", []),
                            "validations": contract.get("validations", []),
                            "feature_availability": contract.get("feature_availability", []),
                            "availability_summary": contract.get("availability_summary", ""),
                            "decision_variables": contract.get("decision_variables", []),
                            "missing_sentinels": contract.get("missing_sentinels", []),
                            "spec_extraction": contract.get("spec_extraction", {}),
                            "business_alignment": contract.get("business_alignment", {}),
                            "notes_for_engineers": contract.get("notes_for_engineers", []),
                        },
                        "cleaning_code_excerpt": code[:4000],
                    }
                    review_result = cleaning_reviewer.review_cleaning(review_context)
                    try:
                        os.makedirs("artifacts", exist_ok=True)
                        with open(os.path.join("artifacts", "cleaning_reviewer_report.json"), "w", encoding="utf-8") as f_rep:
                            json.dump(review_result, f_rep, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

                    if isinstance(review_result, dict) and review_result.get("status") == "REJECTED":
                        if not state.get("cleaning_reviewer_retry_done"):
                            base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                            fixes = review_result.get("required_fixes", [])
                            fixes_text = ""
                            if isinstance(fixes, list) and fixes:
                                fixes_text = "\nREQUIRED_FIXES:\n- " + "\n- ".join(str(item) for item in fixes)
                            payload = "CLEANING_REVIEWER_ALERT:\n" + str(review_result.get("feedback", "")).strip() + fixes_text
                            new_state = dict(state)
                            new_state["cleaning_reviewer_retry_done"] = True
                            new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, payload)
                            print("Cleaning reviewer rejected output: retrying Data Engineer with guidance.")
                            return run_data_engineer(new_state)
                        return {
                            "cleaning_code": code,
                            "cleaned_data_preview": "Error: Cleaning Reviewer Rejected",
                            "error_message": "CRITICAL: Cleaning reviewer rejected cleaned data.",
                        }
                except Exception as review_err:
                    print(f"Warning: cleaning reviewer failed: {review_err}")

            # Guard: derived columns should not be constant if present
            derived_issues = []
            derived_evidence = {}
            if contract_derived_cols:
                col_by_norm = {_norm_name(c): c for c in df_mapped.columns}
                spec = contract.get("spec_extraction", {}) if isinstance(contract, dict) else {}
                spec_derived = spec.get("derived_columns", []) if isinstance(spec, dict) else []
                for derived_name in contract_derived_cols:
                    norm_name = _norm_name(derived_name)
                    actual_name = col_by_norm.get(norm_name)
                    if not actual_name:
                        continue
                    nunique = df_mapped[actual_name].nunique(dropna=False)
                    if nunique <= 1:
                        derived_issues.append(f"{derived_name} has no variance (nunique={nunique})")
                    deps = []
                    for entry in spec_derived:
                        if not isinstance(entry, dict):
                            continue
                        entry_name = entry.get("name")
                        if entry_name and _norm_name(entry_name) == norm_name:
                            deps = entry.get("depends_on", []) or []
                            break
                    if deps:
                        dep_stats = {}
                        for dep in deps:
                            dep_norm = _norm_name(dep)
                            dep_actual = col_by_norm.get(dep_norm)
                            if not dep_actual:
                                dep_stats[dep] = {"status": "missing"}
                                continue
                            series = df_mapped[dep_actual]
                            non_null = int(series.notna().sum())
                            sample_vals = []
                            try:
                                for val in series.dropna().astype(str).head(3).tolist():
                                    if val not in sample_vals:
                                        sample_vals.append(val)
                            except Exception:
                                sample_vals = []
                            dep_stats[dep] = {
                                "actual": dep_actual,
                                "non_null": non_null,
                                "sample": sample_vals,
                            }
                        derived_evidence[derived_name] = dep_stats
            if derived_issues:
                if not state.get("de_derived_guard_retry_done"):
                    new_state = dict(state)
                    new_state["de_derived_guard_retry_done"] = True
                    override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                    try:
                        override += (
                            "\n\nDERIVED_COLUMN_GUARD: "
                            + "; ".join(derived_issues)
                            + ". Ensure derived columns use normalized column mapping "
                            "to access source columns and do not default all rows."
                        )
                        if derived_evidence:
                            override += "\nDERIVED_SOURCE_EVIDENCE:\n" + json.dumps(
                                derived_evidence,
                                ensure_ascii=True,
                            )
                    except Exception:
                        pass
                    new_state["data_engineer_audit_override"] = override
                    print("Derived column guard: retrying Data Engineer with mapping guidance.")
                    return run_data_engineer(new_state)
                else:
                    return {
                        "cleaning_code": code,
                        "cleaned_data_preview": "Error: Derived Columns Constant",
                        "error_message": "CRITICAL: Derived columns present but constant; verify mapping and derivation.",
                    }
            
            # Create Synthetic Columns if needed
            for synth in mapping_result['synthetic']:
                print(f"Creating Synthetic Column: {synth} (0.0)")
                df_mapped[synth] = 0.0

            # Integrity audit only (no auto-fix)
            feature_flags = {}
            feature_stats = {}
            contract = state.get("execution_contract", {})
            try:
                contract_for_audit = _filter_input_contract(contract)
                issues, stats = run_integrity_audit(df_mapped, contract_for_audit)
                feature_stats = stats
                os.makedirs("data", exist_ok=True)
                with open("data/integrity_audit_report.json", "w", encoding="utf-8") as f:
                    json.dump({"issues": issues, "stats": stats}, f, indent=2)
                critical = [i for i in issues if i.get("severity") == "critical"]
                if critical and not state.get("integrity_retry_done"):
                    new_state = dict(state)
                    new_state["integrity_retry_done"] = True
                    try:
                        issue_text = json.dumps(issues, indent=2)
                    except Exception:
                        issue_text = str(issues)
                    base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                    payload = "INTEGRITY_AUDIT_ISSUES:\n" + issue_text
                    new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, payload)
                    print("⚠️ INTEGRITY_AUDIT: triggering Data Engineer retry with issues context.")
                    return run_data_engineer(new_state)
            except Exception as audit_err:
                print(f"Warning: integrity audit failed: {audit_err}")
                try:
                    os.makedirs("data", exist_ok=True)
                    with open("data/integrity_audit_report.json", "w", encoding="utf-8") as f:
                        json.dump({"error": str(audit_err)}, f, indent=2)
                except Exception:
                    pass
        
            try:
                os.makedirs("data", exist_ok=True)
                df_mapped.to_csv(
                    "data/cleaned_full.csv",
                    index=False,
                    sep=csv_sep,
                    decimal=csv_decimal,
                    encoding=csv_encoding
                )
            except Exception as full_save_err:
                print(f"Warning: failed to save cleaned_full.csv: {full_save_err}")

            try:
                audit = run_unsupervised_numeric_relation_audit(df_mapped)
                with open("data/leakage_audit.json", "w", encoding="utf-8") as f:
                    json.dump(audit, f, indent=2)
                findings = audit.get("relations", [])
                if findings:
                    top = findings[:3]
                    leakage_audit_summary = "; ".join(
                        [
                            f"{rel.get('type')}:{','.join(rel.get('columns', []))} (support={rel.get('support_frac', 0):.3f})"
                            for rel in top
                        ]
                    )
                else:
                    leakage_audit_summary = "No deterministic numeric relations found."
            except Exception as audit_err:
                print(f"Warning: leakage audit failed: {audit_err}")
                leakage_audit_summary = "Leakage audit failed; proceed with caution."

            # Filter to only required columns (Strict output)
            # Ensure we strictly have what we asked for, aligned by name
            final_cols = [c for c in required_cols if c in df_mapped.columns]
            if contract_derived_cols:
                for col in contract_derived_cols:
                    if col in df_mapped.columns and col not in final_cols:
                        final_cols.append(col)
            if not final_cols:
                final_cols = df_mapped.columns.tolist()
            df_final = df_mapped[final_cols]
            
            # Save Mapped Data back to disk
            df_final.to_csv(
                local_cleaned_path,
                index=False,
                sep=csv_sep,
                decimal=csv_decimal,
                encoding=csv_encoding
            )
            print("Mapped data saved to 'data/cleaned_data.csv'")
            
            # Save Summary
            with open("data/column_mapping_summary.json", "w") as f:
                json.dump(mapping_result, f, indent=2)
                
            preview = df_final.head(5).to_json(orient='split')
            
        except Exception as e:
            print(f"Column Mapping Failed: {e}")
            return {
                "cleaning_code": code,
                "cleaned_data_preview": f"Error: Mapping Failed {e}",
                "error_message": f"Host-side Column Mapping Failed: {str(e)}",
                "csv_sep": csv_sep,
                "csv_decimal": csv_decimal,
                "csv_encoding": csv_encoding,
                "leakage_audit_summary": leakage_audit_summary,
                "budget_counters": counters,
            }

        if run_id:
            log_run_event(
                run_id,
                "data_engineer_complete",
                {"rows": len(df_final), "columns": len(df_final.columns)},
            )
        return {
            "cleaning_code": code,
            "cleaned_data_preview": preview,
            "csv_sep": csv_sep,
            "csv_decimal": csv_decimal,
            "csv_encoding": csv_encoding,
            "leakage_audit_summary": leakage_audit_summary,
            "budget_counters": counters,
        }

    except Exception as e:
        print(f"Cleaning Execution Error (System): {e}")
        return {
            "cleaning_code": code,
            "cleaned_data_preview": f"Error: {e}",
            "error_message": f"System Error during Data Cleaning: {str(e)}",
            "budget_counters": counters,
        }

def check_data_success(state: AgentState):
    err = state.get("error_message")
    if err:
        print(f"❌ Data Engineer Failed: {err}")
        return "failed"
    
    if not os.path.exists("data/cleaned_data.csv"):
        print("❌ Critical: cleaned_data.csv missing locally.")
        return "failed"
        
    preview = state.get("cleaned_data_preview", "")
    if str(preview).startswith("Error") or "Error reading preview" in str(preview):
        print(f"❌ Data Engineer Failed (Preview Error): {preview}")
        return "failed"

    return "success"

def run_engineer(state: AgentState) -> AgentState:
    print(f"--- [4] ML Engineer: Generating Code (Iteration {state.get('iteration_count', 0) + 1}) ---")
    abort_state = _abort_if_requested(state, "ml_engineer")
    if abort_state:
        return abort_state
    run_id = state.get("run_id")
    ok, counters, err_msg = _consume_budget(state, "ml_calls", "max_ml_calls", "ML Engineer")
    state["budget_counters"] = counters
    if not ok:
        if run_id:
            log_run_event(run_id, "budget_exceeded", {"label": "ml_engineer", "error": err_msg})
        last_code = state.get("last_generated_code")
        last_success_output = state.get("last_successful_execution_output")
        return {
            "error_message": err_msg,
            "generated_code": last_code or "# Generation Failed",
            "last_generated_code": last_code,
            "execution_output": last_success_output or err_msg,
            "execution_output_stale": bool(last_success_output),
            "budget_counters": counters,
        }
    if run_id:
        log_run_event(run_id, "ml_engineer_start", {"iteration": state.get("iteration_count", 0) + 1})
    
    strategy = state.get('selected_strategy')
    
    # Pass input context
    data_path = "data/cleaned_full.csv" if os.path.exists("data/cleaned_full.csv") else "data/cleaned_data.csv"
    feedback_history = state.get('feedback_history', [])
    leakage_summary = state.get("leakage_audit_summary", "")
    data_audit_context = state.get('data_summary', '')
    if leakage_summary:
        data_audit_context = f"{data_audit_context}\nLEAKAGE_AUDIT: {leakage_summary}"
    ml_audit_override = state.get("ml_engineer_audit_override", "")
    if ml_audit_override:
        data_audit_context = _merge_de_audit_override(data_audit_context, ml_audit_override)
    business_objective = state.get('business_objective', '')
    csv_encoding = state.get('csv_encoding', 'utf-8') # Pass real encoding
    csv_sep = state.get('csv_sep', ',')
    csv_decimal = state.get('csv_decimal', '.')
    execution_contract = state.get("execution_contract", {})
    feature_availability = (execution_contract or {}).get("feature_availability", [])
    availability_summary = (execution_contract or {}).get("availability_summary", "")
    iteration_memory = list(state.get("ml_iteration_memory", []) or [])
    iteration_memory_block = state.get("ml_iteration_memory_block", "")
    compliance_checklist = (execution_contract or {}).get("compliance_checklist", [])
    if compliance_checklist and not state.get("compliance_passed", False):
        checklist_payload = "COMPLIANCE_BOOTSTRAP_CHECKLIST:\n" + json.dumps(compliance_checklist, ensure_ascii=True)
        data_audit_context = _merge_de_audit_override(data_audit_context, checklist_payload)

    # Patch Mode Inputs
    previous_code = state.get('last_generated_code')
    gate_context = state.get('last_gate_context')
    if state.get("reset_ml_patch_context"):
        previous_code = None
        gate_context = None

    print(f"DEBUG: Generating code for strategy: {strategy['title']}")
    
    try:
        import inspect
        kwargs = dict(
            strategy=strategy,
            data_path=data_path,
            feedback_history=feedback_history,
            previous_code=previous_code,
            gate_context=gate_context,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
            data_audit_context=data_audit_context,
            business_objective=business_objective,
            feature_availability=feature_availability,
            availability_summary=availability_summary,
            signal_summary={},
            iteration_memory=iteration_memory,
            iteration_memory_block=iteration_memory_block,
        )
        sig = inspect.signature(ml_engineer.generate_code)
        if "execution_contract" in sig.parameters:
            kwargs["execution_contract"] = execution_contract
        header_cols = _read_csv_header(data_path, csv_encoding, csv_sep)
        aliasing = {}
        derived_present = []
        sample_context = ""
        context_ops_blocks = []
        if feature_availability or availability_summary:
            availability_payload = {
                "availability_summary": availability_summary,
                "feature_availability": feature_availability,
            }
            context_ops_blocks.append(
                "FEATURE_AVAILABILITY_CONTEXT:\n" + json.dumps(availability_payload, ensure_ascii=True)
            )
        if iteration_memory:
            memory_slice = iteration_memory[-2:]
            context_ops_blocks.append(
                "ITERATION_MEMORY_CONTEXT:\n" + json.dumps(memory_slice, ensure_ascii=True)
            )
        if header_cols:
            norm_map = {}
            norm_buckets = {}
            for col in header_cols:
                normed = _norm_name(col)
                if normed and normed not in norm_map:
                    norm_map[normed] = col
                if normed:
                    norm_buckets.setdefault(normed, []).append(col)
            aliasing = {k: v for k, v in norm_buckets.items() if len(v) > 1}
            try:
                derived_cols = _resolve_contract_columns(execution_contract, sources={"derived"})
                if derived_cols:
                    derived_present = [c for c in derived_cols if c in set(header_cols)]
            except Exception:
                derived_present = []
            header_context = (
                "CLEANED_COLUMN_INVENTORY_RAW: "
                + json.dumps(header_cols, ensure_ascii=False)
                + "\nNORMALIZED_CLEANED_HEADER_MAP: "
                + json.dumps(norm_map, ensure_ascii=False)
                + "\nCLEANED_ALIASING_COLLISIONS: "
                + json.dumps(aliasing, ensure_ascii=False)
                + "\nDERIVED_COLUMNS_PRESENT: "
                + json.dumps(derived_present, ensure_ascii=False)
            )
            data_audit_context = _merge_de_audit_override(data_audit_context, header_context)
            kwargs["data_audit_context"] = data_audit_context
            required_input_cols = _resolve_required_input_columns(execution_contract, strategy)
            sample_context = _build_required_sample_context(
                data_path,
                {"sep": csv_sep, "encoding": csv_encoding, "decimal": csv_decimal},
                required_input_cols,
                norm_map,
                max_rows=80,
            )
            if sample_context:
                data_audit_context = _merge_de_audit_override(data_audit_context, sample_context)
                kwargs["data_audit_context"] = data_audit_context
            signal_summary = _build_signal_summary_context(
                data_path,
                {"sep": csv_sep, "encoding": csv_encoding, "decimal": csv_decimal},
                required_input_cols,
                norm_map,
                header_cols,
            )
            if signal_summary:
                context_ops_blocks.append(
                    "SIGNAL_SUMMARY_CONTEXT:\n" + json.dumps(signal_summary, ensure_ascii=True)
                )
                kwargs["signal_summary"] = signal_summary
            try:
                derived_cols = _resolve_contract_columns(execution_contract, sources={"derived"}) or []
                output_cols = _resolve_contract_columns(execution_contract, sources={"output"}) or []
                target_cols = [c for c in derived_cols + output_cols if c]
                if target_cols:
                    summary = {}
                    for tgt in target_cols:
                        norm = _norm_name(tgt)
                        actual = norm_map.get(norm, tgt if tgt in header_cols else None)
                        if not actual:
                            continue
                        series_stats = {"present": True}
                        try:
                            import pandas as pd
                            df_preview = pd.read_csv(
                                data_path,
                                sep=csv_sep,
                                decimal=csv_decimal,
                                encoding=csv_encoding,
                                usecols=[actual],
                                nrows=200,
                                low_memory=False,
                            )
                            series = df_preview[actual]
                            series_stats["non_null"] = int(series.notna().sum())
                            series_stats["nunique"] = int(series.nunique(dropna=False))
                            series_stats["sample"] = [str(v) for v in series.dropna().head(3).tolist()]
                        except Exception:
                            pass
                        summary[tgt] = series_stats
                    if summary:
                        context_ops_blocks.append(
                            "DERIVED_OUTPUT_OBSERVATIONS:\n" + json.dumps(summary, ensure_ascii=True)
                        )
            except Exception:
                pass
            try:
                business_alignment = (execution_contract or {}).get("business_alignment", {})
                if business_alignment:
                    context_ops_blocks.append(
                        "BUSINESS_ALIGNMENT_OPERATIVE:\n" + json.dumps(business_alignment, ensure_ascii=True)
                    )
            except Exception:
                pass
        try:
            os.makedirs("artifacts", exist_ok=True)
            ctx_payload = {
                "data_path": data_path,
                "csv_encoding": csv_encoding,
                "csv_sep": csv_sep,
                "csv_decimal": csv_decimal,
                "header_cols": header_cols,
                "cleaned_aliasing_collisions": aliasing if header_cols else {},
                "derived_columns_present": derived_present if header_cols else [],
                "raw_required_sample_context": sample_context,
                "context_ops_blocks": context_ops_blocks,
                "required_features": strategy.get("required_columns", []),
                "execution_contract": execution_contract,
                "data_audit_context": data_audit_context,
                "ml_engineer_audit_override": ml_audit_override,
                "feature_availability": feature_availability,
                "availability_summary": availability_summary,
                "signal_summary": kwargs.get("signal_summary", {}),
                "iteration_memory": iteration_memory,
                "iteration_memory_block": iteration_memory_block,
            }
            with open(os.path.join("artifacts", "ml_engineer_context.json"), "w", encoding="utf-8") as f_ctx:
                json.dump(ctx_payload, f_ctx, indent=2, ensure_ascii=False)
        except Exception as ctx_err:
            print(f"Warning: failed to persist ml_engineer_context.json: {ctx_err}")
        if context_ops_blocks:
            ops_payload = "\n\n".join(context_ops_blocks)
            data_audit_context = _merge_de_audit_override(data_audit_context, ops_payload)
            kwargs["data_audit_context"] = data_audit_context
            try:
                ops_preview = ops_payload
                if len(ops_preview) > 1200:
                    ops_preview = ops_preview[:1200] + "...(truncated)"
                print("ML_CONTEXT_OPERATIVE_PREVIEW:\n" + ops_preview)
                os.makedirs("artifacts", exist_ok=True)
                with open(os.path.join("artifacts", "ml_engineer_context_ops.txt"), "w", encoding="utf-8") as f_ops:
                    f_ops.write(ops_payload)
                if run_id:
                    log_run_event(run_id, "ml_context_ops_preview", {"preview": ops_preview})
            except Exception as ops_err:
                print(f"Warning: failed to persist ml_engineer_context_ops.txt: {ops_err}")
        data_audit_context = _truncate_text(data_audit_context)
        iteration_memory_block = _truncate_text(iteration_memory_block, max_len=6000, head_len=3500, tail_len=2000)
        kwargs["data_audit_context"] = data_audit_context
        kwargs["iteration_memory_block"] = iteration_memory_block
        code = ml_engineer.generate_code(**kwargs)
        try:
            os.makedirs("artifacts", exist_ok=True)
            with open(os.path.join("artifacts", "ml_engineer_last.py"), "w", encoding="utf-8") as f_art:
                f_art.write(code)
        except Exception as artifact_err:
            print(f"Warning: failed to persist ml_engineer_last.py: {artifact_err}")
        try:
            iter_id = int(state.get("iteration_count", 0)) + 1
            os.makedirs(os.path.join("artifacts", "iterations"), exist_ok=True)
            iter_path = os.path.join("artifacts", "iterations", f"ml_code_iter_{iter_id}.py")
            with open(iter_path, "w", encoding="utf-8") as f_iter:
                f_iter.write(code)
        except Exception as iter_err:
            print(f"Warning: failed to persist ml_code_iter_{iter_id}.py: {iter_err}")

        if run_id:
            log_run_event(run_id, "ml_engineer_complete", {"code_len": len(code or "")})
        if str(code).strip().startswith("{") or str(code).strip().startswith("["):
            msg = "ML_CODE_REQUIRED: model returned JSON plan; expected Python code."
            if not state.get("ml_code_retry_done"):
                new_state = dict(state)
                new_state["ml_code_retry_done"] = True
                override = new_state.get("ml_engineer_audit_override") or state.get("data_summary", "")
                override += "\n\nML_CODE_REQUIRED: Return executable Python code only. Do not output JSON plans."
                new_state["ml_engineer_audit_override"] = override
                print("ML code-only guard triggered: retrying ML Engineer with code-only instructions.")
                return run_engineer(new_state)
            return {
                "error_message": msg,
                "generated_code": code,
                "execution_output": msg,
                "budget_counters": counters,
            }
        if str(code).strip().startswith("# Error:"):
            return {
                "error_message": code,
                "generated_code": code,
                "execution_output": code,
                "budget_counters": counters,
            }
        return {
            "selected_strategy": strategy,
            "generated_code": code,
            "last_generated_code": code, # Update for next patch
            "ml_data_path": data_path,
            "error_message": "",
            "ml_call_refund_pending": True,
            "execution_call_refund_pending": True,
            "ml_context_snapshot": {
                "cleaned_column_inventory": header_cols,
                "cleaned_aliasing_collisions": aliasing if header_cols else {},
                "derived_columns_present": derived_present if header_cols else [],
                "raw_required_sample_context": sample_context,
                "feature_semantics": (execution_contract or {}).get("feature_semantics", []),
                "business_sanity_checks": (execution_contract or {}).get("business_sanity_checks", []),
            },
            "budget_counters": counters,
        }
    except Exception as e:
        msg = f"CRITICAL: ML Engineer crashed in host: {str(e)}"
        print(msg)
        return {
            "error_message": msg,
            "generated_code": "# Generation Failed",
            "execution_output": msg,
            "budget_counters": counters,
        }

def check_engineer_success(state: AgentState):
    if state.get("error_message"):
        return "failed"
    return "success"


def run_reviewer(state: AgentState) -> AgentState:
    print("--- REVIEWER AGENT ---")
    abort_state = _abort_if_requested(state, "reviewer")
    if abort_state:
        return abort_state
    run_id = state.get("run_id")
    ok, counters, err_msg = _consume_budget(state, "reviewer_calls", "max_reviewer_calls", "Reviewer")
    state["budget_counters"] = counters
    if not ok:
        if run_id:
            log_run_event(run_id, "budget_exceeded", {"label": "reviewer", "error": err_msg})
        return {
            "review_verdict": "REJECTED",
            "review_feedback": err_msg,
            "error_message": err_msg,
            "budget_counters": counters,
        }
    if run_id:
        log_run_event(run_id, "reviewer_start", {})
    
    code = state['generated_code']
    current_iter = state.get('reviewer_iteration', 0)
    new_history = list(state.get('feedback_history', []))
    # Context Construction
    strategy = state.get('selected_strategy', {})
    analysis_type = strategy.get('analysis_type', 'predictive')
    strategy_context = f"Strategy: {strategy.get('title')}\nType: {strategy.get('analysis_type')}\nRules: {strategy.get('reasoning')}"
    business_objective = state.get('business_objective', 'Analyze data.')
    
    try:
        review = reviewer.review_code(code, analysis_type, business_objective, strategy_context)
        print(f"Verdict: {review['status']}")
        
        # Update Streak
        streak = state.get('review_reject_streak', 0)
        if review['status'] == "REJECTED":
            streak += 1
            print(f"Feedback: {review['feedback']}")
        else:
            streak = 0
            
        # Pass Structured Context for Patching
        gate_context = {
            "source": "reviewer",
            "status": review['status'],
            "feedback": review['feedback'],
            "failed_gates": review.get('failed_gates', []),
            "required_fixes": _expand_required_fixes(review.get('required_fixes', []), review.get('failed_gates', []))
        }
        fix_block = _build_fix_instructions(gate_context["required_fixes"])
        if fix_block:
            gate_context["edit_instructions"] = fix_block

        if review['status'] == "REJECTED":
            new_history.append(f"REVIEWER FEEDBACK (Attempt {current_iter+1}): {review['feedback']}")
            
            # Check Fail-Safe Condition (Streak Based)
            # 1. Critical API Error
            if "Reviewer unavailable" in review['feedback']:
                msg = f"CRITICAL SYSTEM FAILURE: {review['feedback']}"
                return {
                    "review_verdict": "REJECTED",
                    "review_feedback": review['feedback'],
                    "feedback_history": new_history,
                    "reviewer_iteration": current_iter + 1,
                    "review_abort_reason": msg,
                    "budget_counters": counters,
                }

            # 2. Reject Streak Exceeded (Avoid infinite loop on bad code)
            if streak >= 3:
                msg = f"CRITICAL: Code review failed {streak} times in a row. Aborting to avoid loops."
                print(msg)
                return {
                    "review_verdict": "REJECTED",
                    "review_feedback": review['feedback'],
                    "feedback_history": new_history,
                    "reviewer_iteration": current_iter + 1,
                    "review_abort_reason": msg,
                    "budget_counters": counters,
                }

        if run_id:
            log_run_event(run_id, "reviewer_complete", {"status": review.get("status")})
        return {
            "review_verdict": review['status'],
            "review_feedback": review['feedback'],
            "feedback_history": new_history,
            "reviewer_iteration": current_iter + 1,
            "last_gate_context": gate_context,
            "review_reject_streak": streak,
            "budget_counters": counters,
        }

    except Exception as e:
        msg = f"CRITICAL: Reviewer Agent crashed: {str(e)}"
        print(msg)
        return {
            "review_verdict": "REJECTED",
            "review_feedback": msg,
            "feedback_history": new_history,
            "reviewer_iteration": current_iter + 1,
            "error_message": msg,
            "budget_counters": counters,
        }

def run_qa_reviewer(state: AgentState) -> AgentState:
    print("--- QA REVIEWER AGENT ---")
    abort_state = _abort_if_requested(state, "qa_reviewer")
    if abort_state:
        return abort_state
    run_id = state.get("run_id")
    ok, counters, err_msg = _consume_budget(state, "qa_calls", "max_qa_calls", "QA Reviewer")
    state["budget_counters"] = counters
    if not ok:
        if run_id:
            log_run_event(run_id, "budget_exceeded", {"label": "qa_reviewer", "error": err_msg})
        return {
            "review_verdict": "REJECTED",
            "review_feedback": err_msg,
            "error_message": err_msg,
            "budget_counters": counters,
        }
    if run_id:
        log_run_event(run_id, "qa_reviewer_start", {})
    code = state['generated_code']
    strategy = state.get('selected_strategy', {})
    business_objective = state.get('business_objective', '')
    
    current_history = list(state.get('feedback_history', []))
    try:
        # Run QA Audit
        qa_result = qa_reviewer.review_code(code, strategy, business_objective)
        preflight_issues = ml_quality_preflight(code)
        has_variance_guard = "TARGET_VARIANCE_GUARD" not in preflight_issues

        status = qa_result['status']
        feedback = qa_result['feedback']
        failed_gates = qa_result.get('failed_gates', [])
        required_fixes = qa_result.get('required_fixes', [])

        # Override obvious false positives from QA LLM when deterministic facts disagree
        if status == "REJECTED" and "TARGET_VARIANCE" in failed_gates and has_variance_guard:
            status = "APPROVE_WITH_WARNINGS"
            feedback = f"QA_LLM_FALSE_POSITIVE_OVERRIDDEN (variance guard detected statically). Original: {feedback}"
            failed_gates = []
            required_fixes = []
            print(feedback)
            current_history.append(f"QA_LLM_FALSE_POSITIVE_OVERRIDDEN: variance guard present; overriding rejection.")
        
        print(f"QA Verdict: {status}")
        
        # Update QA Streak
        streak = state.get('qa_reject_streak', 0)
        if status == "REJECTED":
            streak += 1
            print(f"QA Feedback: {feedback}")
        else:
            streak = 0
        
        # Structured context for Patching
        gate_context = {
            "source": "qa_reviewer",
            "status": status,
            "feedback": feedback,
            "failed_gates": failed_gates,
            "required_fixes": _expand_required_fixes(required_fixes, failed_gates)
        }
        fix_block = _build_fix_instructions(gate_context["required_fixes"])
        if fix_block:
            gate_context["edit_instructions"] = fix_block
        
        if status == "REJECTED":
            current_history.append(f"QA TEAM FEEDBACK (Critical): {feedback}")
            
            # QA Fail Safe
            if streak >= 5:
                msg = f"CRITICAL: QA Rejected code {streak} times consecutively. Quality Standard not met."
                return {
                    "review_verdict": "REJECTED",
                    "review_feedback": feedback,
                    "feedback_history": current_history,
                    "error_message": msg,
                    "last_gate_context": gate_context,
                    "qa_reject_streak": streak,
                    "budget_counters": counters,
                }
            
            return {
                "review_verdict": "REJECTED", # Mark as rejected to trigger retry
                "review_feedback": feedback,
                "feedback_history": current_history,
                "last_gate_context": gate_context,
                "qa_reject_streak": streak,
                "budget_counters": counters,
            }
            
        if run_id:
            log_run_event(run_id, "qa_reviewer_complete", {"status": status})
        return {
            "review_verdict": "APPROVED",
            "feedback_history": current_history,
            "last_gate_context": gate_context,
            "qa_reject_streak": streak,
            "budget_counters": counters,
        }

    except Exception as e:
        msg = f"CRITICAL: QA Reviewer crashed: {str(e)}"
        print(msg)
        return {
            "review_verdict": "REJECTED",
            "review_feedback": msg,
            "feedback_history": current_history,
            "error_message": msg,
            "budget_counters": counters,
        }

def run_ml_preflight(state: AgentState) -> AgentState:
    print("--- ML PREFLIGHT ---")
    abort_state = _abort_if_requested(state, "ml_preflight")
    if abort_state:
        return abort_state
    code = state.get("generated_code", "")
    if code and not is_syntax_valid(code):
        feedback = "ML_PREFLIGHT_SYNTAX_ERROR: Generated code is not valid Python syntax."
        history = list(state.get("feedback_history", []))
        history.append(feedback)
        gate_context = {
            "source": "ml_preflight",
            "status": "REJECTED",
            "feedback": feedback,
            "failed_gates": ["AST_PARSE_FAILED"],
            "required_fixes": ["AST_PARSE_FAILED"],
        }
        return {
            "ml_preflight_failed": True,
            "feedback_history": history,
            "last_gate_context": gate_context,
            "review_verdict": "REJECTED",
            "review_feedback": feedback,
        }
    contract = state.get("execution_contract", {}) or {}
    required_deps = contract.get("required_dependencies", []) or []
    dep_result = check_dependency_precheck(code, required_deps)
    if dep_result.get("banned") or dep_result.get("blocked"):
        blocked = dep_result.get("blocked", [])
        banned = dep_result.get("banned", [])
        suggestions = dep_result.get("suggestions", {})
        parts = []
        if banned:
            parts.append(f"banned={', '.join(banned)}")
        if blocked:
            parts.append(f"blocked={', '.join(blocked)}")
        hint_parts = []
        for key in banned + blocked:
            hint = suggestions.get(key)
            if hint:
                hint_parts.append(f"{key}: {hint}")
        hint_text = f" Suggestions: {' | '.join(hint_parts)}" if hint_parts else ""
        feedback = f"DEPENDENCY_BLOCKED: {'; '.join(parts)}.{hint_text}"
        history = list(state.get("feedback_history", []))
        history.append(feedback)
        gate_context = {
            "source": "ml_preflight",
            "status": "REJECTED",
            "feedback": feedback,
            "failed_gates": ["DEPENDENCY_BLOCKED"],
            "required_fixes": ["DEPENDENCY_BLOCKED"],
        }
        return {
            "ml_preflight_failed": True,
            "feedback_history": history,
            "last_gate_context": gate_context,
            "review_verdict": "REJECTED",
            "review_feedback": feedback,
        }

    issues = ml_quality_preflight(code)
    required_columns = contract.get("required_columns") or strategy.get("required_columns", []) or []
    required_outputs = contract.get("required_outputs", []) or []
    missing_outputs = _missing_required_output_refs(code, required_outputs)
    if missing_outputs:
        issues.append("REQUIRED_OUTPUTS_MISSING")
    read_csv_info = _analyze_read_csv_usage(code)
    has_read_csv = read_csv_info.get("has_read_csv")
    if not has_read_csv:
        issues.append("DATA_LOAD_MISSING")
    else:
        allowed_paths = ("data/cleaned_data.csv", "data/cleaned_full.csv", "data.csv")
        path_hits = [p for p in read_csv_info.get("paths", []) if any(a in p for a in allowed_paths)]
        if not path_hits and "data_path" not in (code or "").lower():
            issues.append("DATA_PATH_NOT_USED")
    col_coverage = _required_columns_coverage(code, required_columns)
    if required_columns:
        min_abs = 2 if len(required_columns) >= 4 else 1
        min_ratio = int(len(required_columns) * 0.5 + 0.5)
        min_required = max(min_abs, min_ratio)
        if len(col_coverage.get("hits", [])) < min_required:
            issues.append("REQUIRED_COLUMNS_NOT_USED")
    if _detect_synthetic_data(code) and not has_read_csv:
        issues.append("SYNTHETIC_DATA_DETECTED")
    if "alignment_check.json" in (code or "") and "\"requirements\"" not in (code or "") and "'requirements'" not in (code or ""):
        issues.append("ALIGNMENT_REQUIREMENTS_MISSING")
    if issues:
        expanded = _expand_required_fixes(issues, issues)
        feedback = f"ML_PREFLIGHT_MISSING: {', '.join(issues)}"
        if missing_outputs:
            feedback += f" | Missing output refs: {missing_outputs}"
        if col_coverage.get("hits") is not None and required_columns:
            feedback += f" | Required columns hits: {col_coverage.get('hits')}"
        history = list(state.get("feedback_history", []))
        history.append(feedback)
        gate_context = {
            "source": "ml_preflight",
            "status": "REJECTED",
            "feedback": feedback,
            "failed_gates": issues,
            "required_fixes": expanded,
        }
        fix_block = _build_fix_instructions(expanded)
        if fix_block:
            gate_context["edit_instructions"] = fix_block
        return {
            "ml_preflight_failed": True,
            "feedback_history": history,
            "last_gate_context": gate_context,
            "review_verdict": "REJECTED",
            "review_feedback": feedback,
        }
    if manifest_dump_missing_default(code):
        feedback = "ML_JSON_SERIALIZATION_GUARD: json.dump must use default=_json_default to serialize numpy/pandas types."
        history = list(state.get("feedback_history", []))
        history.append(feedback)
        gate_context = {
            "source": "ml_preflight",
            "status": "REJECTED",
            "feedback": feedback,
            "failed_gates": ["JSON_SERIALIZATION_GUARD"],
            "required_fixes": ["Add json.dump(..., default=_json_default) and define _json_default helper."],
        }
        return {
            "ml_preflight_failed": True,
            "feedback_history": history,
            "last_gate_context": gate_context,
            "review_verdict": "REJECTED",
            "review_feedback": feedback,
        }
    return {"ml_preflight_failed": False}

def check_qa_review(state: AgentState):
    if state.get("error_message") or state.get("review_abort_reason"):
        return "failed"
    if state.get('review_verdict') == "REJECTED":
        return "rejected"
    # APPROVED or APPROVE_WITH_WARNINGS -> proceeds
    return "approved"

def check_ml_preflight(state: AgentState):
    if state.get("ml_preflight_failed"):
        return "failed"
    return "passed"


def check_review(state: AgentState):
    # Fail-Safe Logic
    if state.get("error_message") or state.get("review_abort_reason"):
        return "failed"

    if state['review_verdict'] in ["APPROVED", "APPROVE_WITH_WARNINGS"]:
        return "approved"
    else:
        return "rejected"

def execute_code(state: AgentState) -> AgentState:
    print("--- [5] System: Executing Code (E2B Sandbox) ---")
    run_id = state.get("run_id")
    ok, counters, err_msg = _consume_budget(state, "execution_calls", "max_execution_calls", "Execution")
    state["budget_counters"] = counters
    if not ok:
        if run_id:
            log_run_event(run_id, "budget_exceeded", {"label": "execution", "error": err_msg})
        return {"error_message": err_msg, "execution_output": err_msg, "budget_counters": counters}
    if run_id:
        log_run_event(run_id, "execution_start", {})
    code = state['generated_code']

    # Prevent stale metrics from previous iterations
    try:
        if os.path.exists("data/metrics.json"):
            os.remove("data/metrics.json")
    except Exception:
        pass
    
    # 0. Static Safety Scan
    is_safe, violations = scan_code_safety(code)
    if not is_safe:
        failure_reason = "CRITICAL: Security Violations:\n" + "\n".join(violations)
        print(f"🚫 Security Block: {failure_reason}")
        return {"error_message": failure_reason, "execution_output": failure_reason, "budget_counters": counters}

    # Dependency allowlist precheck
    contract = state.get("execution_contract", {}) or {}
    required_deps = contract.get("required_dependencies", []) or []
    dep_result = check_dependency_precheck(code, required_deps)
    if dep_result.get("banned"):
        banned = ", ".join(dep_result["banned"])
        msg = f"EXECUTION ERROR: DEPENDENCY_BLOCKED banned imports: {banned}"
        fh = list(state.get("feedback_history", []))
        fh.append(f"DEPENDENCY_BLOCKED: {banned}")
        return {"error_message": msg, "execution_output": msg, "feedback_history": fh, "budget_counters": counters}
    if dep_result.get("blocked"):
        blocked = ", ".join(dep_result["blocked"])
        msg = f"EXECUTION ERROR: DEPENDENCY_BLOCKED imports not in allowlist: {blocked}"
        fh = list(state.get("feedback_history", []))
        fh.append(f"DEPENDENCY_BLOCKED: {blocked}")
        return {"error_message": msg, "execution_output": msg, "feedback_history": fh, "budget_counters": counters}

    # 0b. Undefined name preflight (avoid sandbox NameError)
    undefined = detect_undefined_names(code)
    if undefined:
        msg = f"STATIC_PRECHECK_UNDEFINED: Undefined names detected preflight: {', '.join(undefined)}"
        fh = list(state.get("feedback_history", []))
        fh.append(f"STATIC_PRECHECK_UNDEFINED: {msg}")
        try:
            print(msg)
        except Exception:
            pass
        return {"error_message": msg, "execution_output": msg, "feedback_history": fh}

    # Secure execution using E2B
    try:
        # load_dotenv() removed (module level)
        api_key = os.getenv("E2B_API_KEY")
        if not api_key:
            msg = "CRITICAL: E2B_API_KEY missing in .env file."
            return {"error_message": msg, "execution_output": msg, "budget_counters": counters}
        os.environ["E2B_API_KEY"] = api_key 
        
        with Sandbox.create() as sandbox:
            print("Installing dependencies in Sandbox...")
            pkg_sets = get_sandbox_install_packages(required_deps)
            base_cmd = "pip install -q " + " ".join(pkg_sets["base"])
            sandbox.commands.run(base_cmd)
            if pkg_sets["extra"]:
                extra_cmd = "pip install -q " + " ".join(pkg_sets["extra"])
                sandbox.commands.run(extra_cmd)
            sandbox.commands.run("mkdir -p static/plots") # Ensure plots dir exists
            sandbox.commands.run("mkdir -p data") # Ensure data dir exists for outputs
            
            local_csv = state.get("ml_data_path") or "data/cleaned_data.csv"
            if not os.path.exists(local_csv) and local_csv != "data/cleaned_data.csv":
                local_csv = "data/cleaned_data.csv"
            remote_csv = "/home/user/data.csv"
            
            if os.path.exists(local_csv):
                with open(local_csv, "rb") as f:
                    sandbox.files.write(remote_csv, f)
                print(f"Data uploaded to {remote_csv}")
                
                # Robust path patching
                if local_csv:
                    code = code.replace(local_csv, remote_csv)
                    if not local_csv.startswith("./"):
                        code = code.replace(f"./{local_csv}", remote_csv)
                code = code.replace("data/cleaned_data.csv", remote_csv)
                code = code.replace("./data/cleaned_data.csv", remote_csv)
                code = code.replace("data/cleaned_full.csv", remote_csv)
                code = code.replace("./data/cleaned_full.csv", remote_csv)
                
                # Manifest Round-trip (Upload & Patch)
                local_manifest = "data/cleaning_manifest.json"
                remote_manifest = "/home/user/cleaning_manifest.json"
                
                if os.path.exists(local_manifest):
                    with open(local_manifest, "rb") as f:
                        sandbox.files.write(remote_manifest, f)
                    print(f"Manifest uploaded to {remote_manifest}")
                    
                    # Patch Manifest Path
                    code = code.replace("data/cleaning_manifest.json", remote_manifest)
                    code = code.replace("./data/cleaning_manifest.json", remote_manifest)
                    
                    # Idempotent Injection of Manifest Path (Contract)
                    # Ensures the remote path is available even if not explicitly used in original script
                    if remote_manifest not in code:
                        injection = f'\n# Cleaning manifest available at {remote_manifest}\nCLEANING_MANIFEST_PATH = "{remote_manifest}"\n'
                        # Import injection if needed? No, just variable definition at top
                        # We append it to import section or top of file. 
                        # Simplest: Prepend to code
                        code = injection + code
                else:
                    print("Warning: Manifest not found locally. Code patching skipped.")
            else:
                print("Warning: Local cleaned data not found. Upload skipped.")

            # Persist executed ML script for traceability
            try:
                os.makedirs("artifacts", exist_ok=True)
                with open(os.path.join("artifacts", "ml_engineer_last.py"), "w", encoding="utf-8") as f_art:
                    f_art.write(code)
            except Exception as artifact_err:
                print(f"Warning: failed to persist ml_engineer_last.py: {artifact_err}")

            print("Running code in Sandbox...")
            execution = sandbox.run_code(code)
            
            output = ""
            if execution.logs.stdout: output += "\nSTDOUT:\n" + "\n".join(execution.logs.stdout)
            if execution.logs.stderr: output += "\nSTDERR:\n" + "\n".join(execution.logs.stderr)
            
            if execution.error:
                output += f"\n\nEXECUTION ERROR:\n{execution.error.name}: {execution.error.value}\n{execution.error.traceback}"

            # Robust Artifact Download (P0 Fix)
            # Use shell command that always succeeds even if no files found
            ls_proc = sandbox.commands.run("sh -lc 'ls -1 static/plots/*.png 2>/dev/null || true'")
            
            if ls_proc.exit_code == 0:
                os.makedirs("static/plots", exist_ok=True)
                # Filter empty strings from split
                plot_files = [p for p in ls_proc.stdout.strip().split('\n') if p]
                
                if not plot_files:
                    print("Info: No plots generated by the script.")
                else:
                    for remote_plot in plot_files:
                        if remote_plot.endswith('.png'):
                            try:
                                # Base64 robust download
                                proc = sandbox.commands.run(f"base64 -w 0 {remote_plot}")
                                if proc.exit_code == 0:
                                    b64_content = proc.stdout.strip()
                                    content = base64.b64decode(b64_content)
                                    if len(content) > 0:
                                        local_name = os.path.basename(remote_plot)
                                        with open(os.path.join("static/plots", local_name), "wb") as f_local:
                                            f_local.write(content)
                                        print(f"Downloaded plot: {local_name}")
                            except Exception as e:
                                 print(f"Failed to download {remote_plot}: {e}")
            else:
                print(f"Warning: Plot listing failed (Exit Code {ls_proc.exit_code})")

            # Download required outputs per contract (beyond plots)
            req_outputs = state.get("execution_contract", {}).get("required_outputs", []) or []
            for pattern in req_outputs:
                if not pattern:
                    continue
                list_cmd = f"sh -lc 'ls -1 {pattern} 2>/dev/null || true'"
                lst = sandbox.commands.run(list_cmd)
                if lst.exit_code != 0:
                    continue
                files = [p for p in lst.stdout.strip().split("\n") if p]
                for remote_path in files:
                    if not remote_path:
                        continue
                    try:
                        proc = sandbox.commands.run(f"base64 -w 0 {remote_path}")
                        if proc.exit_code == 0:
                            b64_content = proc.stdout.strip()
                            content = base64.b64decode(b64_content)
                            if len(content) >= 0:
                                if remote_path.startswith("/home/user/"):
                                    local_path = remote_path[len("/home/user/"):].lstrip("/")
                                else:
                                    local_path = remote_path.lstrip("/")
                                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                                with open(local_path, "wb") as f_local:
                                    f_local.write(content)
                                print(f"Downloaded required output: {local_path}")
                    except Exception as dl_err:
                        print(f"Warning: failed to download required output {remote_path}: {dl_err}")

    except Exception as e:
        output = f"Sandbox Execution Failed: {e}"
        print(output)
        
    # Calculate Visuals State locally
    import glob
    plots_local = glob.glob("static/plots/*.png")
    
    fallback_plots_local = []
             
    has_partial_visuals = len(plots_local) > 0
    
    print(f"Execution finished. Plots generated: {len(plots_local)}")
    
    # Check for Runtime Errors in Output (Traceback)
    sandbox_failed = (
        "Sandbox Execution Failed" in output
        or "peer closed connection" in output
        or "incomplete chunked read" in output
        or "Response 404" in output
    )
    error_in_output = (
        "Traceback (most recent call last)" in output
        or "EXECUTION ERROR" in output
        or sandbox_failed
    )
    
    runtime_tail = None
    ml_skipped_reason = state.get("ml_skipped_reason", None)
    if error_in_output:
        print("runtime error detected in output.")
        tail = output[-4000:] if isinstance(output, str) else str(output)[-4000:]
        print("----- SANDBOX EXECUTION_OUTPUT TAIL (last 4000 chars) -----")
        print(tail)
        print("-----------------------------------------------------------")
        runtime_tail = tail
        if "DETERMINISTIC_TARGET_RELATION" in tail:
            ml_skipped_reason = "DETERMINISTIC_TARGET_RELATION"
        if state.get("ml_call_refund_pending"):
            refund_counters = dict(state.get("budget_counters") or {})
            if refund_counters.get("ml_calls", 0) > 0:
                refund_counters["ml_calls"] = max(0, refund_counters.get("ml_calls", 0) - 1)
            counters = refund_counters
            state["budget_counters"] = refund_counters
        if state.get("execution_call_refund_pending"):
            refund_counters = dict(state.get("budget_counters") or {})
            if refund_counters.get("execution_calls", 0) > 0:
                refund_counters["execution_calls"] = max(0, refund_counters.get("execution_calls", 0) - 1)
            counters = refund_counters
            state["budget_counters"] = refund_counters
        
    # Suppress fallback plots in successful executions (avoid reporting placeholders)
    if not error_in_output and not sandbox_failed and plots_local:
        plots_local = [
            plot for plot in plots_local
            if not os.path.basename(plot).startswith("fallback_")
        ]
        fallback_plots_local = [
            plot for plot in fallback_plots_local
            if not os.path.basename(plot).startswith("fallback_")
        ]
    # Only flag partial visuals when there was an execution error or sandbox failure
    if not error_in_output and not sandbox_failed:
        has_partial_visuals = False
    else:
        has_partial_visuals = len(plots_local) > 0

    # Validate required outputs early
    output_contract = state.get("execution_contract", {}).get("required_outputs", [])
    oc_report = check_required_outputs(output_contract)
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
            json.dump(oc_report, f, indent=2)
    except Exception as oc_err:
        print(f"Warning: failed to persist output_contract_report.json: {oc_err}")

    if run_id:
        log_run_event(
            run_id,
            "execution_complete",
            {
                "plots": len(plots_local),
                "has_error": bool(error_in_output),
                "output_missing": len(oc_report.get("missing", [])) if isinstance(oc_report, dict) else None,
            },
        )
    artifact_index = []
    if isinstance(oc_report, dict):
        artifact_index.extend(oc_report.get("present", []))
    if plots_local:
        artifact_index.extend(plots_local)
    # De-duplicate while preserving order
    deduped = []
    seen = set()
    for item in artifact_index:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    artifact_index = deduped

    result = {
        "execution_output": output,
        "plots_local": plots_local,
        "fallback_plots": fallback_plots_local,
        "has_partial_visuals": has_partial_visuals,
        "execution_error": error_in_output,
        "execution_attempt": state.get('execution_attempt', 0) + 1,
        "last_runtime_error_tail": runtime_tail,
        "ml_skipped_reason": ml_skipped_reason,
        "output_contract_report": oc_report,
        "artifact_index": artifact_index,
        "sandbox_failed": sandbox_failed,
        "sandbox_retry_count": 0 if not sandbox_failed else state.get("sandbox_retry_count", 0),
        "ml_call_refund_pending": False,
        "execution_call_refund_pending": False,
        "budget_counters": counters,
    }
    if not error_in_output:
        result["runtime_fix_count"] = 0
        result["last_successful_execution_output"] = output
        result["last_successful_plots"] = plots_local
        result["last_successful_output_contract_report"] = oc_report
    return result

def retry_handler(state: AgentState) -> AgentState:
    print("--- [!] Performance Low. Retrying... ---")
    
    # Fix: Ensure feedback_history is a list
    current_history = list(state.get('feedback_history', []))
    
    # Append execution summary
    last_output = state.get('execution_output', '')
    if last_output and ("Traceback" in last_output or "EXECUTION ERROR" in last_output):
        summary = f"EXECUTION OUTPUT (last run): {last_output[-3000:]}" # Truncate to save context
        current_history.append(summary)
    
    # Cap history size
    if len(current_history) > 20:
        current_history = current_history[-20:]

    return {
        "feedback_history": current_history
    }

def retry_sandbox_execution(state: AgentState) -> AgentState:
    retry_count = int(state.get("sandbox_retry_count", 0)) + 1
    max_retries = int(state.get("max_sandbox_retries", 2))
    print(f"--- [!] Sandbox failure detected. Retrying execution ({retry_count}/{max_retries}) ---")
    history = list(state.get("feedback_history", []))
    last_output = state.get("execution_output", "")
    if last_output:
        history.append(f"SANDBOX_RETRY {retry_count}: {last_output[-500:]}")
    return {
        "sandbox_retry_count": retry_count,
        "feedback_history": history,
    }
    
def run_result_evaluator(state: AgentState) -> AgentState:
    print("--- [5.5] Reviewer: Evaluating Results ---")
    abort_state = _abort_if_requested(state, "result_evaluator")
    if abort_state:
        return abort_state
    run_id = state.get("run_id")
    if run_id:
        log_run_event(run_id, "result_evaluator_start", {})
    
    execution_output = state.get('execution_output', '')
    if "Traceback" in execution_output:
         print("Reviewer: Critical Execution Error detected. Requesting fix.")
         return {
             "review_verdict": "NEEDS_IMPROVEMENT",
             "review_feedback": f"Execution failed with traceback: {execution_output[-500:]}",
             "feedback_history": state.get("feedback_history", []) # Maintain history
         }

    strategy = state.get('selected_strategy', {}) or {}
    strategy_context = f"Strategy: {strategy.get('title')}\nType: {strategy.get('analysis_type')}\nRules: {strategy.get('reasoning')}"
    business_objective = state.get('business_objective', '')
    
    eval_result = reviewer.evaluate_results(execution_output, business_objective, strategy_context)
    
    status = eval_result.get('status', 'APPROVED')
    feedback = eval_result.get('feedback', '')
    
    new_history = list(state.get('feedback_history', []))
    if status == "NEEDS_IMPROVEMENT":
        new_history.append(f"RESULT EVALUATION FEEDBACK: {feedback}")

    # Case alignment QA gate (optional if artifacts exist)
    contract = state.get("execution_contract", {}) or {}
    data_paths = []
    if os.path.exists("data/cleaned_full.csv"):
        data_paths.append("data/cleaned_full.csv")
    if os.path.exists("data/cleaned_data.csv"):
        data_paths.append("data/cleaned_data.csv")
    case_report = build_case_alignment_report(
        contract=contract,
        case_summary_path="data/case_summary.csv",
        weights_path="data/weights.json",
        data_paths=data_paths,
    )
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/case_alignment_report.json", "w", encoding="utf-8") as f:
            json.dump(case_report, f, indent=2)
    except Exception as err:
        print(f"Warning: failed to persist case_alignment_report.json: {err}")

    # Track case alignment history for adaptive stopping
    case_history = list(state.get("case_alignment_history", []) or [])
    try:
        metrics = case_report.get("metrics", {}) if isinstance(case_report, dict) else {}
        violations = metrics.get("adjacent_refscore_violations")
        if violations is not None:
            case_history.append(float(violations))
    except Exception:
        pass

    # Detect stale metrics file across iterations (diagnostic for ML)
    metrics_report = _load_json_safe("data/metrics.json")
    weights_report = _load_json_safe("data/weights.json")
    metrics_signature = _hash_json(metrics_report)
    weights_signature = _hash_json(weights_report)
    prev_metrics_signature = state.get("metrics_signature")
    prev_weights_signature = state.get("weights_signature")
    metrics_stale = False
    if metrics_signature and metrics_signature == prev_metrics_signature:
        if weights_signature and weights_signature != prev_weights_signature:
            metrics_stale = True
            new_history.append(
                "METRICS_STALE: data/metrics.json unchanged while weights changed; recompute metrics from current outputs."
            )
        else:
            new_history.append(
                "METRICS_UNCHANGED: data/metrics.json is identical to the prior iteration; ensure metrics are recomputed and saved per run."
            )
    if not metrics_report:
        new_history.append(
            "METRICS_MISSING: data/metrics.json not found or empty; downstream evaluation may be using stale metrics."
        )
    if not metrics_report or metrics_stale:
        synthesized = _normalize_metrics_from_weights(weights_report)
        if not synthesized:
            synthesized = _compute_metrics_from_scored_rows(
                scored_rows_path="data/scored_rows.csv",
                case_summary_path="data/case_summary.csv",
                weights_obj=weights_report,
            )
        if synthesized:
            metrics_report = dict(synthesized)
            metrics_report["source"] = "computed_fallback"
            metrics_signature = _hash_json(metrics_report)
            new_history.append(
                "METRICS_FALLBACK: metrics synthesized from weights/scored_rows due to missing or stale metrics.json."
            )
            try:
                os.makedirs("data", exist_ok=True)
                with open("data/metrics.json", "w", encoding="utf-8") as f_metrics:
                    json.dump(metrics_report, f_metrics, indent=2)
            except Exception as metrics_err:
                print(f"Warning: failed to persist fallback metrics.json: {metrics_err}")
    iter_id = int(state.get("iteration_count", 0)) + 1
    saved_iter_artifacts = _persist_iteration_artifacts(iter_id)

    if case_report.get("status") == "FAIL":
        feedback = f"CASE_ALIGNMENT_GATE_FAILED: {case_report.get('explanation')}"
        status = "NEEDS_IMPROVEMENT"
        new_history.append(f"CASE_ALIGNMENT_GATE_FAILED: {case_report.get('failures')}")

    # Output contract validation
    output_contract = state.get("execution_contract", {}).get("required_outputs", [])
    oc_report = check_required_outputs(output_contract)
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
            json.dump(oc_report, f, indent=2)
    except Exception as oc_err:
        print(f"Warning: failed to persist output_contract_report.json: {oc_err}")
    if oc_report.get("missing"):
        status = "NEEDS_IMPROVEMENT"
        miss_text = json.dumps(oc_report, indent=2)
        feedback_missing = f"Missing required outputs per contract: {miss_text}"
        feedback = f"{feedback}\n{feedback_missing}" if feedback else feedback_missing
        new_history.append(f"OUTPUT_CONTRACT_MISSING: {miss_text}")

    alignment_failed_gates: List[str] = []
    alignment_check = _load_json_safe("data/alignment_check.json")
    alignment_requirements = contract.get("alignment_requirements", []) if isinstance(contract, dict) else []
    if isinstance(alignment_requirements, list) and alignment_requirements:
        if not alignment_check:
            status = "NEEDS_IMPROVEMENT"
            alignment_failed_gates.append("alignment_check_missing")
            msg = "ALIGNMENT_CHECK_MISSING: data/alignment_check.json not found."
            feedback = f"{feedback}\n{msg}" if feedback else msg
            new_history.append(msg)
        else:
            normalized_alignment, alignment_issues = _normalize_alignment_check(
                alignment_check, alignment_requirements
            )
            if normalized_alignment != alignment_check:
                alignment_check = normalized_alignment
                try:
                    with open("data/alignment_check.json", "w", encoding="utf-8") as f_align:
                        json.dump(alignment_check, f_align, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            raw_status = str(alignment_check.get("status", "")).upper()
            failure_mode = str(alignment_check.get("failure_mode", "")).lower()
            summary = alignment_check.get("summary") or alignment_check.get("notes") or ""
            if alignment_issues and raw_status == "PASS":
                raw_status = "WARN"
                summary = f"{summary} Alignment evidence missing." if summary else "Alignment evidence missing."
                failure_mode = failure_mode or "method_choice"
                alignment_check["status"] = raw_status
                alignment_check["summary"] = summary
                alignment_check["failure_mode"] = failure_mode
            data_modes = {"data_limited", "data", "insufficient_data", "data_limitations"}
            method_modes = {"method_choice", "method", "strategy", "approach"}
            msg = f"ALIGNMENT_CHECK_{raw_status}: failure_mode={failure_mode}; summary={summary}"
            if raw_status in {"WARN", "FAIL"}:
                if failure_mode in data_modes:
                    if status != "NEEDS_IMPROVEMENT":
                        status = "APPROVE_WITH_WARNINGS"
                    new_history.append(msg)
                    feedback = f"{feedback}\n{msg}" if feedback else msg
                elif failure_mode in method_modes or raw_status == "FAIL":
                    status = "NEEDS_IMPROVEMENT"
                    alignment_failed_gates.append("alignment_method_choice")
                    new_history.append(msg)
                    feedback = f"{feedback}\n{msg}" if feedback else msg
                else:
                    if raw_status == "FAIL":
                        status = "NEEDS_IMPROVEMENT"
                        alignment_failed_gates.append("alignment_unknown")
                    elif status != "NEEDS_IMPROVEMENT":
                        status = "APPROVE_WITH_WARNINGS"
                    new_history.append(msg)
                    feedback = f"{feedback}\n{msg}" if feedback else msg

    data_adequacy_report = {}
    adequacy_status = None
    adequacy_threshold = int(state.get("data_adequacy_threshold", 3) or 3)
    adequacy_consecutive = int(state.get("data_adequacy_consecutive", 0) or 0)
    try:
        data_adequacy_report = build_data_adequacy_report(state)
        adequacy_status = data_adequacy_report.get("status")
        if adequacy_status == "data_limited":
            adequacy_consecutive += 1
        else:
            adequacy_consecutive = 0
        data_adequacy_report["consecutive_data_limited"] = adequacy_consecutive
        data_adequacy_report["data_limited_threshold"] = adequacy_threshold
        data_adequacy_report["threshold_reached"] = adequacy_consecutive >= adequacy_threshold
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/data_adequacy_report.json", "w", encoding="utf-8") as f_adequacy:
                json.dump(data_adequacy_report, f_adequacy, indent=2, ensure_ascii=False)
        except Exception as adequacy_err:
            print(f"Warning: failed to persist data_adequacy_report.json: {adequacy_err}")
    except Exception as adequacy_err:
        print(f"Warning: data adequacy evaluation failed: {adequacy_err}")

    if _detect_refscore_alias(execution_output, contract):
        status = "NEEDS_IMPROVEMENT"
        alias_msg = (
            "TARGET_MAPPING_ERROR: RefScore is derived per contract but was mapped to Score. "
            "Derive RefScore from case taxonomy instead of aliasing Score."
        )
        feedback = f"{feedback}\n{alias_msg}" if feedback else alias_msg
        new_history.append(alias_msg)

    # Post-exec code audit (non-blocking warnings)
    code = state.get("generated_code") or state.get("last_generated_code") or ""
    counters = dict(state.get("budget_counters") or {})
    review_counters = counters
    audit_rejected = False
    if code:
        analysis_type = strategy.get("analysis_type", "predictive")
        review_warnings: List[str] = []
        try:
            ok, counters, err_msg = _consume_budget(state, "reviewer_calls", "max_reviewer_calls", "Reviewer")
            review_counters = counters
            if ok:
                review_result = reviewer.review_code(code, analysis_type, business_objective, strategy_context)
                if review_result and review_result.get("status") != "APPROVED":
                    review_warnings.append(
                        f"REVIEWER_CODE_AUDIT[{review_result.get('status')}]: {review_result.get('feedback')}"
                    )
            else:
                review_warnings.append(f"REVIEWER_CODE_AUDIT_SKIPPED: {err_msg}")
        except Exception as review_err:
            review_warnings.append(f"REVIEWER_CODE_AUDIT_ERROR: {review_err}")
        try:
            ok, counters, err_msg = _consume_budget(state, "qa_calls", "max_qa_calls", "QA Reviewer")
            review_counters = counters
            qa_result = None
            if ok:
                qa_result = qa_reviewer.review_code(code, strategy, business_objective)
                if qa_result and qa_result.get("status") != "APPROVED":
                    review_warnings.append(
                        f"QA_CODE_AUDIT[{qa_result.get('status')}]: {qa_result.get('feedback')}"
                    )
            else:
                review_warnings.append(f"QA_CODE_AUDIT_SKIPPED: {err_msg}")
        except Exception as qa_err:
            review_warnings.append(f"QA_CODE_AUDIT_ERROR: {qa_err}")
        if review_warnings:
            warn_text = "\n".join(review_warnings)
            feedback = f"{feedback}\n{warn_text}" if feedback else warn_text
            new_history.append(warn_text)
            audit_rejected = any(
                "REVIEWER_CODE_AUDIT[REJECTED]" in warn_text
                or "QA_CODE_AUDIT[REJECTED]" in warn_text
                for warn_text in review_warnings
            )

    if audit_rejected:
        status = "NEEDS_IMPROVEMENT"
        if feedback:
            feedback = f"{feedback}\nCODE_AUDIT_REJECTED: reviewer/QA rejection requires fixes."
        else:
            feedback = "CODE_AUDIT_REJECTED: reviewer/QA rejection requires fixes."
        new_history.append("CODE_AUDIT_REJECTED: reviewer/QA rejection requires fixes.")
        if alignment_requirements:
            if "alignment_method_choice" not in alignment_failed_gates:
                alignment_failed_gates.append("alignment_method_choice")
        if alignment_check:
            try:
                alignment_check["status"] = "FAIL"
                alignment_check["failure_mode"] = "method_choice"
                summary = alignment_check.get("summary") or ""
                note = "Reviewer/QA rejection indicates method choice misalignment."
                alignment_check["summary"] = f"{summary} {note}".strip()
                with open("data/alignment_check.json", "w", encoding="utf-8") as f_align:
                    json.dump(alignment_check, f_align, indent=2, ensure_ascii=False)
            except Exception:
                pass

    failed_gates = case_report.get("failures", []) if case_report.get("status") == "FAIL" else []
    required_fixes = _expand_required_fixes(failed_gates, failed_gates)
    if alignment_failed_gates:
        for item in alignment_failed_gates:
            if item not in failed_gates:
                failed_gates.append(item)
                required_fixes.append(item)
    required_fixes = _expand_required_fixes(required_fixes, failed_gates)
    if case_report.get("status") == "FAIL":
        gate_source = "case_alignment_gate"
    elif alignment_failed_gates:
        gate_source = "alignment_check"
    else:
        gate_source = "result_evaluator"
    gate_context = {
        "source": gate_source,
        "status": status,
        "feedback": feedback,
        "failed_gates": failed_gates,
        "required_fixes": required_fixes,
    }
    fix_block = _build_fix_instructions(required_fixes)
    if fix_block:
        gate_context["edit_instructions"] = fix_block

    print(f"Reviewer Verdict: {status}")
    if status == "NEEDS_IMPROVEMENT":
        print(f"Advice: {feedback}")

    iteration_type = _classify_iteration_type(status, audit_rejected, oc_report, feedback)
    if iteration_type:
        gate_context["iteration_type"] = iteration_type
    compliance_iterations = int(state.get("compliance_iterations", 0))
    metric_iterations = int(state.get("metric_iterations", 0))
    compliance_passed = bool(state.get("compliance_passed", False))
    if status == "NEEDS_IMPROVEMENT":
        if iteration_type == "compliance":
            compliance_iterations += 1
            compliance_passed = False
        elif iteration_type == "metric":
            metric_iterations += 1
            compliance_passed = True

    review_feedback = feedback or state.get("review_feedback", "")
    result_state = {
        "review_verdict": status,
        "review_feedback": review_feedback,
        "execution_feedback": feedback,
        "feedback_history": new_history,
        "output_contract_report": oc_report,
        "case_alignment_report": case_report,
        "case_alignment_history": case_history,
        "metrics_signature": metrics_signature,
        "weights_signature": weights_signature,
        "last_gate_context": gate_context,
        "compliance_iterations": compliance_iterations,
        "metric_iterations": metric_iterations,
        "compliance_passed": compliance_passed,
        "last_iteration_type": iteration_type,
        "data_adequacy_consecutive": adequacy_consecutive,
        "data_adequacy_threshold": adequacy_threshold,
        "data_adequacy_status": adequacy_status,
    }
    if status in ["APPROVED", "APPROVE_WITH_WARNINGS"]:
        result_state["last_successful_review_verdict"] = status
        result_state["last_successful_gate_context"] = gate_context
    if review_counters:
        result_state["budget_counters"] = review_counters
    if status == "NEEDS_IMPROVEMENT" and iteration_type == "metric":
        try:
            prev_case_report = _load_json_safe(
                os.path.join("artifacts", "iterations", f"case_alignment_report_iter_{iter_id - 1}.json")
            ) if iter_id > 1 else {}
            advisor_ctx = {
                "iteration_id": iter_id,
                "previous_case_alignment_report": prev_case_report,
                "case_alignment_report": case_report,
                "output_contract_report": oc_report,
                "case_summary_stats": _summarize_case_summary("data/case_summary.csv"),
                "weights": _load_json_safe("data/weights.json"),
                "weights_uniformity": _summarize_weight_uniformity(_load_json_safe("data/weights.json")),
                "metrics": _load_json_safe("data/metrics.json"),
                "review_feedback": feedback,
                "business_alignment": contract.get("business_alignment", {}),
                "spec_extraction": contract.get("spec_extraction", {}),
                "saved_iteration_artifacts": saved_iter_artifacts,
            }
            advice = results_advisor.generate_ml_advice(advisor_ctx)
            if advice:
                base_override = state.get("ml_engineer_audit_override") or state.get("data_summary", "")
                result_state["ml_engineer_audit_override"] = _merge_de_audit_override(
                    base_override,
                    "RESULTS_ADVISOR:\n" + advice.strip(),
                )
                result_state["ml_results_advice"] = advice.strip()
                try:
                    os.makedirs("artifacts", exist_ok=True)
                    with open(os.path.join("artifacts", "ml_results_advisor.txt"), "w", encoding="utf-8") as f_adv:
                        f_adv.write(advice.strip())
                except Exception as adv_err:
                    print(f"Warning: failed to persist ml_results_advisor.txt: {adv_err}")
        except Exception as adv_err:
            print(f"Warning: results advisor failed: {adv_err}")

    # Iteration memory (delta + objective + weights) for patch-mode guidance
    iteration_memory = list(state.get("ml_iteration_memory", []) or [])
    prev_summary = iteration_memory[-1] if iteration_memory else None
    summary = _build_iteration_memory(
        iter_id=iter_id,
        metrics_report=metrics_report,
        case_report=case_report,
        weights_report=_load_json_safe("data/weights.json"),
        code=code,
        prev_summary=prev_summary,
        advisor_note=result_state.get("ml_results_advice"),
    )
    if summary:
        iteration_memory.append(summary)
        iteration_memory = iteration_memory[-3:]
        try:
            os.makedirs(os.path.join("artifacts", "iterations"), exist_ok=True)
            summary_path = os.path.join("artifacts", "iterations", f"ml_iteration_summary_iter_{iter_id}.json")
            with open(summary_path, "w", encoding="utf-8") as f_sum:
                json.dump(summary, f_sum, indent=2, ensure_ascii=False)
        except Exception as mem_err:
            print(f"Warning: failed to persist ml_iteration_summary_iter_{iter_id}.json: {mem_err}")
    edit_block = _build_edit_instructions(summary)
    if edit_block:
        gate_context["edit_instructions"] = edit_block
    result_state["ml_iteration_memory"] = iteration_memory
    result_state["ml_iteration_memory_block"] = edit_block

    if status == "NEEDS_IMPROVEMENT":
        result_state["iteration_count"] = state.get("iteration_count", 0) + 1
    if run_id:
        log_run_event(run_id, "result_evaluator_complete", {"status": status})
    return result_state

def check_execution_status(state: AgentState):
    # Check for critical failures first
    if state.get("error_message") and "Security Block" in state["error_message"]:
        return "failed"

    output = state.get("execution_output", "")
    attempt = state.get("execution_attempt", 1)
    runtime_fix_count = int(state.get("runtime_fix_count", 0))
    skipped_reason = state.get("ml_skipped_reason")
    
    # Traceback check (Runtime Error)
    has_error = (
        "Traceback (most recent call last)" in output
        or "EXECUTION ERROR" in output
        or "Sandbox Execution Failed" in output
        or "peer closed connection" in output
        or "incomplete chunked read" in output
        or "Response 404" in output
        or state.get("sandbox_failed")
    )
    determinism_error = "DETERMINISTIC_TARGET_RELATION" in output
    undefined_precheck = "STATIC_PRECHECK_UNDEFINED" in output
    sandbox_error = (
        state.get("sandbox_failed")
        or "Sandbox Execution Failed" in output
        or "peer closed connection" in output
        or "incomplete chunked read" in output
        or "Response 404" in output
    )

    if determinism_error or skipped_reason == "DETERMINISTIC_TARGET_RELATION":
        return "evaluate"
    
    if undefined_precheck:
        return "retry_fix"

    if sandbox_error:
        retry_count = int(state.get("sandbox_retry_count", 0))
        max_retries = int(state.get("max_sandbox_retries", 2))
        if retry_count < max_retries:
            return "retry_sandbox"
        print(f"Sandbox failure detected. Max retries reached ({retry_count}/{max_retries}).")
        return "failed"

    if has_error:
        max_runtime_fixes = int(state.get("max_runtime_fix_attempts", 3))
        next_attempt = runtime_fix_count + 1
        if runtime_fix_count < max_runtime_fixes:
            print(f"Runtime Error detected (Attempt {next_attempt}/{max_runtime_fixes}). Preparing runtime fix.")
            return "retry_fix"
        print(f"Runtime Error detected (Attempt {runtime_fix_count}/{max_runtime_fixes}). Max runtime fixes reached.")
        return "failed_runtime"
             
    return "evaluate"

def prepare_runtime_fix(state: AgentState) -> AgentState:
    print("--- [!] Preparing Runtime Fix Context ---")
    output = state.get("execution_output", "")
    base_fix_count = int(state.get("runtime_fix_count", 0))
    terminal_fix = bool(state.get("runtime_fix_terminal"))
    fix_attempt = base_fix_count if terminal_fix else base_fix_count + 1
    max_runtime_fixes = int(state.get("max_runtime_fix_attempts", 3))
    
    error_context = {
        "source": "Execution Runtime",
        "status": "REJECTED",
        "feedback": f"RUNTIME ERROR trace:\n{output[-2000:]}\n\nFIX THIS CRASH.",
        "failed_gates": ["Runtime Stability"],
        "required_fixes": ["Fix the exception."]
    }

    ml_override = state.get("ml_engineer_audit_override", "")
    try:
        error_details = state.get("last_runtime_error_tail") or output
        code = state.get("generated_code") or state.get("last_generated_code") or ""
        expl_ctx = state.get("ml_context_snapshot") or {}
        expl_text = failure_explainer.explain_ml_failure(
            code=code,
            error_details=str(error_details),
            context=expl_ctx,
        )
        if expl_text:
            payload = "ML_FAILURE_EXPLANATION:\n" + expl_text.strip()
            payload += "\nRUNTIME_ERROR_TAIL:\n" + str(error_details)[-2000:]
            ml_override = _merge_de_audit_override(ml_override, payload)
            expl_lines = [line.strip() for line in expl_text.splitlines() if line.strip()]
            if expl_lines:
                fix_summary = "ML_FAILURE_EXPLANATION: " + " | ".join(expl_lines)
                error_context["required_fixes"].append(fix_summary)
            try:
                os.makedirs("artifacts", exist_ok=True)
                with open(os.path.join("artifacts", "ml_engineer_failure_explainer.txt"), "w", encoding="utf-8") as f_exp:
                    f_exp.write(expl_text.strip())
            except Exception as exp_err:
                print(f"Warning: failed to persist ml_engineer_failure_explainer.txt: {exp_err}")
    except Exception as expl_err:
        print(f"Warning: ML failure explainer failed during runtime fix: {expl_err}")
    
    return {
        "last_gate_context": error_context,
         # We add error to history so it persists
        "feedback_history": state.get("feedback_history", []) + [f"RUNTIME ERROR (Attempt {fix_attempt}/{max_runtime_fixes}):\n{output[-500:]}"],
        "ml_engineer_audit_override": ml_override,
        "runtime_fix_count": base_fix_count if terminal_fix else fix_attempt,
    }

def finalize_runtime_failure(state: AgentState) -> AgentState:
    print("--- [!] Final runtime failure: capturing failure explanation ---")
    state_with_terminal = dict(state)
    state_with_terminal["runtime_fix_terminal"] = True
    result = prepare_runtime_fix(state_with_terminal)
    result["runtime_fix_terminal"] = True
    return result

def check_evaluation(state: AgentState):
    policy = _get_iteration_policy(state)
    last_iter_type = state.get("last_iteration_type")
    if policy:
        compliance_max = policy.get("compliance_bootstrap_max")
        metric_max = policy.get("metric_improvement_max")
        if last_iter_type == "compliance" and compliance_max:
            if state.get("compliance_iterations", 0) >= compliance_max:
                print("WARNING: Compliance bootstrap limit reached. Proceeding with current results.")
                return "approved"
        if last_iter_type != "compliance" and metric_max:
            if state.get("metric_iterations", 0) >= metric_max:
                print("WARNING: Metric-iteration limit reached. Proceeding with current results.")
                return "approved"
    else:
        if state.get('iteration_count', 0) >= 6:
            print("WARNING: Max iterations reached. Proceeding with current results.")
            return "approved"

    # Adaptive stop: if case alignment degrades or stagnates, stop early.
    if last_iter_type == "compliance":
        if state.get('review_verdict') == "NEEDS_IMPROVEMENT":
            return "retry"
        return "approved"
    case_report = state.get("case_alignment_report", {}) or {}
    metrics = case_report.get("metrics", {}) if isinstance(case_report, dict) else {}
    try:
        curr_violation_rate = float(metrics.get("adjacent_refscore_violations", metrics.get("case_order_violations", 1.0)))
    except Exception:
        curr_violation_rate = None
    prev_rates = state.get("case_alignment_history", []) or []
    if curr_violation_rate is not None:
        new_history = prev_rates + [curr_violation_rate]
        # stop if last two iterations didn't improve by at least 5%
        if len(new_history) >= 3:
            last = new_history[-1]
            prev = new_history[-2]
            prev2 = new_history[-3]
            if last >= prev * 0.95 and prev >= prev2 * 0.95:
                print("WARNING: Case alignment not improving across iterations. Stopping early.")
                return "approved"
        # stop if regression >10% vs best so far
        best = min(new_history) if new_history else curr_violation_rate
        if curr_violation_rate > best * 1.10:
            print("WARNING: Case alignment regressed vs best. Stopping early.")
            return "approved"

    if state.get('review_verdict') == "NEEDS_IMPROVEMENT":
        return "retry"
    else:
        return "approved"

def run_translator(state: AgentState) -> AgentState:
    print("--- [6] Translator: Generating Report ---")
    abort_state = _abort_if_requested(state, "translator")
    if abort_state:
        return abort_state
    run_id = state.get("run_id")
    if run_id:
        log_run_event(run_id, "translator_start", {})
    
    error_msg = state.get("error_message")
    
    # Extract visuals context
    has_partial_visuals = state.get("has_partial_visuals", False)
    plots_local = state.get("plots_local", [])
    fallback_plots = state.get("fallback_plots", [])
    
    report_state = dict(state)
    report_error = error_msg
    if error_msg and "BUDGET_EXCEEDED" in str(error_msg) and state.get("last_successful_execution_output"):
        report_state["execution_output"] = state.get("last_successful_execution_output")
        if state.get("last_successful_plots"):
            report_state["plots_local"] = state.get("last_successful_plots")
            report_state["has_partial_visuals"] = True
        report_error = None
    report_state.setdefault("execution_error", state.get("execution_error", False))
    report_state.setdefault("sandbox_failed", state.get("sandbox_failed", False))
    report_plots = report_state.get("plots_local", plots_local)
    report_artifacts = list(report_state.get("artifact_index") or [])
    error_flag = bool(report_error) or report_state.get("execution_error") or report_state.get("sandbox_failed")
    if not error_flag:
        if fallback_plots:
            report_plots = [plot for plot in report_plots if plot not in fallback_plots]
            report_state["plots_local"] = report_plots
            if report_artifacts:
                report_artifacts = [path for path in report_artifacts if path not in fallback_plots]
                report_state["artifact_index"] = report_artifacts
    report_state["has_partial_visuals"] = bool(report_plots) and error_flag
    report_has_partial = report_state["has_partial_visuals"]
    try:
        report = translator.generate_report(
            report_state,
            error_message=report_error,
            has_partial_visuals=report_has_partial,
            plots=report_plots
        )
    except Exception as e:
        print(f"CRITICAL: Translator crashed in host: {e}")
        # Fallback Report
        report = f"""
        # System Recovery Report
        
        **CRITICAL FAILURE IN REPORT GENERATION**
        
        The system encountered an internal error while synthesizing the final report.
        
        ### Error Details:
        {str(e)}
        
        ### Original Error (if any):
        {error_msg if error_msg else "None"}
        
        ### Partial Results:
        - Visuals Generated: {has_partial_visuals}
        - Plots Available: {len(plots_local)}
        
        Please check the logs for more details.
        """
        
    try:
        write_data_adequacy_report(state)
        write_governance_report(state)
        summary = build_run_summary(state)
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/run_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass
        fingerprint = state.get("dataset_fingerprint")
        if fingerprint:
            record_dataset_memory(
                {
                    "fingerprint": fingerprint,
                    "run_id": state.get("run_id"),
                    "status": summary.get("status"),
                    "failed_gates": summary.get("failed_gates", []),
                }
            )
        if run_id:
            finalize_run_log(run_id, summary)
            log_run_event(run_id, "translator_complete", {"report_len": len(report or "")})
    except Exception:
        pass
    return {"final_report": report}
# Generate Unique PDF Path to avoid file locks
import uuid

def generate_pdf_artifact(state: AgentState) -> AgentState:
    print("--- [7] System: Generating PDF Report ---")
    report = state['final_report']
    
    # Check for visualizations
    if "static/plots" not in report:
        plots = state.get("plots_local", []) or []
        fallback_plots = state.get("fallback_plots", []) or []
        has_exec_error = bool(state.get("execution_error") or state.get("sandbox_failed"))
        if not has_exec_error and fallback_plots:
            plots = [plot for plot in plots if plot not in fallback_plots]
        if plots:
            report += "\n\n## Visualizations\n"
            for plot in plots:
                report += f"![{os.path.basename(plot)}]({plot})\n"
    
    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]
    pdf_filename = f"final_report_{unique_id}.pdf"
    
    # Absolute path for clarity
    abs_pdf_path = os.path.abspath(pdf_filename)
    
    # Convert
    success = convert_report_to_pdf(report, pdf_filename)
    
    if success:
        print(f"PDF generated at: {abs_pdf_path}")
        return {"final_report": state["final_report"], "pdf_path": pdf_filename}
    else:
        print("PDF Generation Failed")
        return {"final_report": state["final_report"], "pdf_path": None}

# 3. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("steward", run_steward)
workflow.add_node("strategist", run_strategist)
workflow.add_node("domain_expert", run_domain_expert) # Add Node
workflow.add_node("execution_planner", run_execution_planner)

workflow.add_node("data_engineer", run_data_engineer)
workflow.add_node("engineer", run_engineer)
workflow.add_node("reviewer", run_reviewer)
workflow.add_node("qa_reviewer", run_qa_reviewer) # QA Node
workflow.add_node("final_runtime_fix", finalize_runtime_failure)
workflow.add_node("ml_preflight", run_ml_preflight)
workflow.add_node("execute_code", execute_code)
workflow.add_node("evaluate_results", run_result_evaluator) # New Node
workflow.add_node("retry_handler", retry_handler)
workflow.add_node("retry_sandbox", retry_sandbox_execution)
workflow.add_node("prepare_runtime_fix", prepare_runtime_fix) # New Node

workflow.add_node("translator", run_translator)
workflow.add_node("generate_pdf", generate_pdf_artifact)

workflow.set_entry_point("steward")

workflow.add_edge("steward", "strategist")
workflow.add_edge("strategist", "domain_expert") # Rewire
workflow.add_edge("domain_expert", "execution_planner")
workflow.add_edge("execution_planner", "data_engineer")

# workflow.add_edge("data_engineer", "engineer") -> Replaced by Conditional Edge
# workflow.add_edge("engineer", "reviewer") -> Replaced by Conditional Edge

workflow.add_conditional_edges(
    "engineer",
    check_engineer_success,
    {
        "success": "ml_preflight",
        "failed": "translator"
    }
)

workflow.add_conditional_edges(
    "ml_preflight",
    check_ml_preflight,
    {
        "passed": "execute_code",
        "failed": "retry_handler",
    }
)

# Conditional Edge for Data Engineer Failure
workflow.add_conditional_edges(
    "data_engineer",
    check_data_success,
    {
        "success": "engineer",
        "failed": "translator"
    }
)

"""
# Reviewer/QA pre-exec gates bypassed in favor of post-exec advisory checks.
# Nodes remain registered for optional future use.
"""

# New Flow: Execution -> Loop
workflow.add_conditional_edges(
    "execute_code",
    check_execution_status,
    {
        "evaluate": "evaluate_results",
        "retry_fix": "prepare_runtime_fix",
        "retry_sandbox": "retry_sandbox",
        "failed": "translator",
        "failed_runtime": "final_runtime_fix",
    }
)

workflow.add_edge("prepare_runtime_fix", "engineer")
workflow.add_edge("final_runtime_fix", "translator")
workflow.add_edge("retry_sandbox", "execute_code")

# Conditional Edge for Result Evaluation
workflow.add_conditional_edges(
    "evaluate_results",
    check_evaluation,
    {
        "retry": "retry_handler",
        "approved": "translator"
    }
)

workflow.add_edge("retry_handler", "engineer")
workflow.add_edge("translator", "generate_pdf")
workflow.add_edge("generate_pdf", END)

# 4. Compile
app_graph = workflow.compile()
