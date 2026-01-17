import sys
import os
import shutil
import subprocess
import re
import json
import hashlib
import threading
import time
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Dict, Any, List, Literal, Optional
from langgraph.graph import StateGraph, END
from e2b_code_interpreter import Sandbox as CodeSandbox
try:
    from e2b import Sandbox as BaseSandbox
except Exception:
    BaseSandbox = None
Sandbox = CodeSandbox
from dotenv import load_dotenv
import base64

# Add src to path to allow imports if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.steward import StewardAgent
from src.agents.strategist import StrategistAgent
from src.agents.ml_engineer import MLEngineerAgent
from src.agents.business_translator import BusinessTranslatorAgent

from src.agents.data_engineer import DataEngineerAgent
from src.agents.cleaning_reviewer import CleaningReviewerAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.qa_reviewer import QAReviewerAgent, collect_static_qa_facts, run_static_qa_checks # New QA Gate
from src.agents.execution_planner import (
    ExecutionPlannerAgent,
    build_execution_plan,
    build_dataset_profile,
    build_reporting_policy,
    build_plot_spec,
)
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
from src.utils.cleaning_guards import (
    detect_identifier_scientific_notation,
    is_identifier_like,
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
# REMOVED: from src.utils.contract_validation import ensure_role_runbooks  # V4.1 cutover
from src.utils.data_engineer_preflight import data_engineer_preflight
from src.utils.contract_v41 import (
    get_canonical_columns,
    get_artifact_requirements,
    get_derived_column_names,
    get_outcome_columns,
    get_required_outputs,
    get_column_roles,
    get_validation_requirements,
    get_qa_gates,
    get_reviewer_gates,
)
from src.utils.contract_views import (
    build_de_view,
    build_ml_view,
    build_cleaning_view,
    build_qa_view,
    build_reviewer_view,
    build_translator_view,
    build_results_advisor_view,
    persist_views,
    sanitize_contract_min_for_de,
)
from src.utils.cleaning_plan import parse_cleaning_plan, validate_cleaning_plan
from src.utils.cleaning_executor import execute_cleaning_plan
from src.utils.run_logger import init_run_log, log_run_event, finalize_run_log
from src.utils.run_bundle import (
    init_run_bundle,
    log_agent_snapshot,
    log_sandbox_attempt,
    update_sandbox_attempt,
    copy_run_artifacts,
    copy_run_contracts,
    copy_run_reports,
    write_run_manifest,
    get_run_dir,
)
from src.utils.run_storage import (
    init_run_dir,
    write_manifest_partial,
    finalize_run,
    clean_workspace_outputs,
    normalize_status,
)
from src.utils.run_workspace import enter_run_workspace, exit_run_workspace
from src.utils.path_resolution import _resolve_csv_path_with_base, _add_workspace_metadata
from src.utils.artifact_resolver import (
    load_json_scoped,
    exists_scoped,
    get_artifact_from_state_or_scoped,
)
from src.utils.dataset_memory import (
    fingerprint_dataset,
    load_dataset_memory,
    record_dataset_memory,
    summarize_memory,
)
from src.utils.governance import write_governance_report, build_run_summary
from src.utils.data_adequacy import (
    build_data_adequacy_report,
    write_data_adequacy_report,
    _calc_lift,
    _metric_higher_is_better,
)
from src.utils.code_extract import extract_code_block, is_syntax_valid
from src.utils.visuals import generate_fallback_plots
from src.utils.recommendations_preview import build_recommendations_preview
from src.utils.label_enrichment import enrich_outputs
from src.utils.ml_validation import validate_model_metrics_consistency, validate_metrics_ci_consistency
from src.utils.json_sanitize import dump_json
from src.utils.sandbox_paths import (
    CANONICAL_RAW_REL,
    CANONICAL_CLEANED_REL,
    CANONICAL_MANIFEST_REL,
    COMMON_RAW_ALIASES,
    COMMON_CLEANED_ALIASES,
    patch_placeholders,
    build_symlink_or_copy_commands,
    canonical_abs,
)
from src.utils.sandbox_resilience import (
    run_code_with_optional_timeout,
    run_cmd_with_retry,
    safe_download_file,
    safe_download_bytes,
    is_transient_sandbox_error,
    create_sandbox_with_retry,
    DE_TIMEOUT_S,
    ML_TIMEOUT_SMALL_S,
    ML_TIMEOUT_MEDIUM_S,
    ML_TIMEOUT_LARGE_S,
)
from src.utils.dataset_size import (
    file_size_mb,
    estimate_rows_fast,
    classify_dataset_scale,
    get_dataset_scale_hints,
)


def _retry_sandbox_operations(sandbox_func, max_attempts: int = 2, run_id: Optional[str] = None, step: Optional[str] = None) -> Any:
    """
    Retry sandbox operations on transient errors.

    Wraps sandbox_func (which must return a sandbox instance) with retry logic.
    Returns sandbox instance and result.

    Args:
        sandbox_func: Function that creates and returns a sandbox instance
        max_attempts: Number of attempts (default: 2)
        run_id: Optional run ID for logging
        step: Optional step name for logging

    Returns:
        (sandbox, result) where result is the result of sandbox_func
    """
    from src.utils.sandbox_resilience import is_transient_sandbox_error, run_cmd_with_retry

    last_error = None

    for attempt in range(max_attempts):
        try:
            # Call the sandbox creation function
            sandbox, result = sandbox_func()
            return sandbox, result
        except Exception as e:
            last_error = e
            error_msg = str(e)

            print(f"SANDBOX_ATTEMPT_{attempt + 1}/{max_attempts}: {error_msg}")

            # Only retry if error is transient
            if attempt < max_attempts - 1 and is_transient_sandbox_error(e):
                print(f"RETRYING_SANDBOX step={step} attempt={attempt + 1}/{max_attempts}")
                time.sleep(2 ** attempt)
            else:
                # Non-transient error or last attempt
                raise

    raise last_error

def _norm_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())


def _extract_qa_gate_names(raw_gates: Any) -> List[str]:
    names: List[str] = []
    if isinstance(raw_gates, list):
        for gate in raw_gates:
            if isinstance(gate, dict):
                name = gate.get("name") or gate.get("id") or gate.get("gate")
                if name:
                    names.append(str(name))
            elif isinstance(gate, str) and gate.strip():
                names.append(gate)
    return names


def _is_percent_like(column_name: str, raw_values: List[str]) -> bool:
    if column_name and "%" in str(column_name):
        return True
    for val in raw_values or []:
        if "%" in str(val):
            return True
    return False


def _coerce_raw_numeric(value: str) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    raw = raw.replace("%", "")
    raw = re.sub(r"[^0-9,\\.()+-]", "", raw)
    if not raw:
        return None
    neg = False
    if raw.startswith("(") and raw.endswith(")"):
        neg = True
        raw = raw[1:-1]
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "")
            raw = raw.replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw and "." not in raw:
        raw = raw.replace(".", "")
        raw = raw.replace(",", ".")
    try:
        val = float(raw)
    except Exception:
        return None
    return -val if neg else val


def _median(values: List[float]) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]

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

def _normalize_path_posix(path: str) -> str:
    try:
        return Path(path).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def _hash_text(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _apply_static_autofixes(code: str) -> tuple[str, List[Dict[str, Any]]]:
    if not code:
        return code, []
    fixes: List[Dict[str, Any]] = []
    updated = code

    def _autofix_nested_dict_assignments(text: str) -> tuple[str, int]:
        bases = ["stats", "cleaning_stats", "manifest_stats", "quality_stats", "validation_stats"]
        base_pattern = "|".join(re.escape(base) for base in bases)
        pattern = re.compile(
            rf"(?P<base>\b(?:{base_pattern})\b)\s*\[\s*(?P<key>'[^']+'|\"[^\"]+\")\s*\]\s*"
            rf"\[\s*(?P<inner>'[^']+'|\"[^\"]+\")\s*\]\s*="
        )
        base_inits = {
            base
            for base in bases
            if re.search(rf"\b{re.escape(base)}\s*=\s*(\{{\s*\}}|dict\(\)|defaultdict\(\s*dict\s*\))", text)
        }
        lines = text.splitlines()
        changed = False
        count = 0
        for idx, line in enumerate(lines):
            match = pattern.search(line)
            if not match:
                continue
            base = match.group("base")
            if base not in base_inits:
                continue
            key = match.group("key")
            if "setdefault" in line:
                continue
            if re.search(rf"{base}\.setdefault\(\s*{re.escape(key)}\s*,", text):
                continue
            if re.search(rf"{base}\s*\[\s*{re.escape(key)}\s*\]\s*=", text):
                continue
            replacement = f"{base}.setdefault({key}, {{}})[{match.group('inner')}] ="
            lines[idx] = pattern.sub(replacement, line, count=1)
            changed = True
            count += 1
        if not changed:
            return text, 0
        return "\n".join(lines), count

    for pattern, replacement, label in [
        (r"\bnp\.bool\b", "np.bool_", "np.bool->np.bool_"),
    ]:
        before_hash = _hash_text(updated)
        updated, count = re.subn(pattern, replacement, updated)
        if count:
            fixes.append(
                {
                    "rule": label,
                    "count": int(count),
                    "before_hash": before_hash,
                    "after_hash": _hash_text(updated),
                  }
              )
    before_hash = _hash_text(updated)
    updated, nested_count = _autofix_nested_dict_assignments(updated)
    if nested_count:
        fixes.append(
            {
                "rule": "nested_dict_setdefault",
                "count": int(nested_count),
                "before_hash": before_hash,
                "after_hash": _hash_text(updated),
            }
        )
    return updated, fixes


def _collect_violation_snippets(code: str, violations: List[str], max_lines: int = 12) -> List[str]:
    if not code or not violations:
        return []
    tokens: List[str] = []
    for violation in violations:
        if "np.bool" in violation:
            tokens.append("np.bool")
        matches = re.findall(r"'([^']+)'", str(violation))
        for match in matches:
            if match and match not in tokens:
                tokens.append(match)
    if not tokens:
        return []
    lines = code.splitlines()
    snippets: List[str] = []
    for idx, line in enumerate(lines, start=1):
        if any(tok in line for tok in tokens):
            snippets.append(f"L{idx}: {line.strip()}")
            if len(snippets) >= max_lines:
                break
    return snippets


def _persist_output_contract_report(
    state: Dict[str, Any],
    reason: str | None = None,
    path: str = "data/output_contract_report.json",
) -> Dict[str, Any]:
    contract = state.get("execution_contract", {}) if isinstance(state, dict) else {}
    # V4.1: Use get_required_outputs accessor
    output_contract = get_required_outputs(contract)
    report = check_required_outputs(output_contract)
    if reason:
        report["reason"] = reason
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump_json(path, report)
    except Exception:
        pass
    return report

def _stage_illustrative_assets(
    source_root: str,
    report_root: str = "report",
    label_col_hint: str | None = None,
    contract: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    staged = {"plots": [], "jsons": [], "csvs": [], "metadata_path": None}
    if not source_root or not os.path.isdir(source_root):
        return staged
    report_root = report_root or "report"
    os.makedirs(report_root, exist_ok=True)
    scored_rows_dest = None
    summary_dest = None
    candidate_jsons: List[Dict[str, str]] = []
    deliverable_lookup: Dict[str, Dict[str, Any]] = {}
    schema_paths: set[str] = set()
    if isinstance(contract, dict):
        # V4.1: Use artifact_requirements instead of spec_extraction
        artifacts = get_artifact_requirements(contract)
        required_files = artifacts.get("required_files", [])
        if isinstance(required_files, list):
            for path in required_files:
                if path:
                    deliverable_lookup[str(path)] = {"path": str(path), "required": True}
        schema = contract.get("artifact_schemas")
        file_schemas = artifacts.get("file_schemas")
        if not isinstance(schema, dict) and isinstance(file_schemas, dict):
            schema = file_schemas
        if isinstance(schema, dict):
            schema_paths = {str(path) for path in schema.keys() if path}

    def _score_summary_candidate(rel_path: str, dest_path: str) -> float:
        deliverable = deliverable_lookup.get(rel_path, {})
        kind = str(deliverable.get("kind") or "").lower()
        score = 0.0
        if kind in {"report", "summary", "metrics"}:
            score += 5.0
        if deliverable and deliverable.get("required"):
            score += 3.0
        if rel_path in schema_paths:
            score += 1.0
        try:
            size = os.path.getsize(dest_path)
            if size > 0:
                score += min(size / 2048.0, 4.0)
        except Exception:
            pass
        payload = _load_json_any(dest_path)
        if isinstance(payload, dict):
            keys = list(payload.keys())
            score += min(len(keys), 20) * 0.1
            for tok in ("metrics", "recommendations", "limits", "summary", "results"):
                if tok in payload:
                    score += 0.5
        return score
    for base, _, names in os.walk(source_root):
        for name in names:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, source_root)
            rel_posix = rel.replace("\\", "/")
            lower = rel_posix.lower()
            is_plot = lower.startswith("static/plots/") and lower.endswith((".png", ".jpg", ".jpeg"))
            is_json = lower.startswith(("data/", "reports/")) and lower.endswith(".json")
            is_scored_rows = rel_posix == "data/scored_rows.csv"
            if not (is_plot or is_json or is_scored_rows):
                continue
            dest = os.path.join(report_root, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                shutil.copy2(full, dest)
            except Exception:
                continue
            dest_posix = _normalize_path_posix(dest)
            if is_plot:
                staged["plots"].append(dest_posix)
            elif is_json:
                staged["jsons"].append(dest_posix)
                candidate_jsons.append({"rel": rel_posix, "dest": dest})
            elif is_scored_rows:
                staged["csvs"].append(dest_posix)
                scored_rows_dest = dest

    if candidate_jsons:
        ranked = sorted(
            candidate_jsons,
            key=lambda item: _score_summary_candidate(item["rel"], item["dest"]),
            reverse=True,
        )
        summary_dest = ranked[0]["dest"]

    if summary_dest and scored_rows_dest and label_col_hint:
        try:
            meta = enrich_outputs(
                scored_rows_dest,
                summary_json_path=summary_dest,
                label_col_hint=label_col_hint,
            )
            staged["label_enrichment_meta"] = meta
        except Exception as err:
            print(f"Warning: illustrative label enrichment failed: {err}")

    metadata = {
        "status": "illustrative_only",
        "not_production": True,
        "source_root": _normalize_path_posix(source_root),
        "staged_at": datetime.utcnow().isoformat(),
        "assets": staged,
    }
    try:
        meta_path = os.path.join(report_root, "illustrative_assets.json")
        with open(meta_path, "w", encoding="utf-8") as f_meta:
            json.dump(metadata, f_meta, indent=2, ensure_ascii=False)
        staged["metadata_path"] = _normalize_path_posix(meta_path)
    except Exception:
        staged["metadata_path"] = None
    return staged

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
        try:
            finalize_run(run_id, status_final="CRASH", state=state)
        except Exception:
            pass
        finally:
            exit_run_workspace(state)
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

def _resolve_requirement_meta(contract: Dict[str, Any], col: str) -> Dict[str, Any]:
    """
    V4.1: Resolve metadata for a column used for determining if it is optional.
    Constructs a synthetic requirement dict from V4.1 sources.
    """
    if not isinstance(contract, dict):
        return {}

    norm_col = _norm_name(col)

    # Check V4.1 canonical columns (Strictly required usually)
    from src.utils.contract_v41 import get_canonical_columns, get_artifact_requirements
    canonical = get_canonical_columns(contract)
    for c in canonical:
        if _norm_name(c) == norm_col:
            return {"name": c, "required": True, "nullable": False}

    # Check optional passthrough
    artifact_reqs = get_artifact_requirements(contract)
    schema = artifact_reqs.get("schema_binding", {})
    optional = schema.get("optional_passthrough_columns", [])
    for c in optional:
        if _norm_name(c) == norm_col:
            return {"name": c, "required": False, "nullable": True}

    # Fallback to legacy data_requirements for backwards compatibility
    reqs = contract.get("data_requirements", [])
    if isinstance(reqs, list):
        for req in reqs:
            if not isinstance(req, dict):
                continue
            name = req.get("canonical_name") or req.get("name") or req.get("column")
            if name and _norm_name(name) == norm_col:
                return req

    # Default: if not found in canonical, assume it might be optional or extra
    return {}

def _is_optional_requirement(req: Dict[str, Any]) -> bool:
    if not isinstance(req, dict):
        return True # If semantics not found, don't block
    if req.get("required") is False:
        return True
    if req.get("nullable") is True:
        return True
    return False
    if req.get("optional") is True:
        return True
    return False

def _sample_raw_null_stats(
    csv_path: str,
    dialect: Dict[str, Any],
    raw_cols: List[str],
    nrows: int = 500,
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    sample_df = sample_raw_columns(csv_path, dialect, raw_cols, nrows=nrows, dtype=str)
    if sample_df is None or getattr(sample_df, "empty", False):
        return stats
    for col in raw_cols:
        if col not in sample_df.columns:
            continue
        series = sample_df[col]
        try:
            raw_str = series.astype(str).str.strip()
            null_like = series.isna() | raw_str.eq("") | raw_str.str.lower().isin(["nan", "null", "none"])
            null_frac = float(null_like.mean()) if len(series) else 0.0
            stats[col] = {
                "null_frac": null_frac,
                "non_null_frac": float(1.0 - null_frac),
                "sample_rows": int(len(series)),
            }
        except Exception:
            stats[col] = {}
    return stats

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


def _infer_parsing_hints_from_sample_context(sample_context: str) -> str:
    """
    Builds compact, universal parsing guidance from RAW_REQUIRED_COLUMN_SAMPLES.
    This is intentionally generic: it does not hardcode dataset-specific fixes, it derives hints from patterns.
    """
    if not sample_context:
        return ""
    if "RAW_REQUIRED_COLUMN_SAMPLES:" not in sample_context:
        return ""
    try:
        _, raw_json = sample_context.split("\n", 1)
    except ValueError:
        return ""
    try:
        payload = json.loads(raw_json.strip())
    except Exception:
        return ""
    columns = payload.get("columns") if isinstance(payload, dict) else None
    if not isinstance(columns, dict) or not columns:
        return ""

    hints: List[str] = []
    saw_symbols = False
    saw_multi_dot = False
    saw_multi_comma = False
    saw_percent = False

    for meta in columns.values():
        if not isinstance(meta, dict):
            continue
        examples = meta.get("examples") or []
        for ex in examples:
            s = str(ex)
            if "%" in s:
                saw_percent = True
            if s.count(".") >= 2:
                saw_multi_dot = True
            if s.count(",") >= 2:
                saw_multi_comma = True
            # Anything other than digits, whitespace, separators, sign, parentheses, and percent.
            if re.search(r"[^\d\s,.\-+()%]", s):
                saw_symbols = True

    if saw_symbols:
        hints.append("Strip currency symbols/letters before numeric conversion (keep digits, sign, separators, parentheses, and %).")
    if saw_multi_dot:
        hints.append("Values with multiple '.' are usually thousands separators; remove all '.' (unless you also detect a decimal separator elsewhere).")
    if saw_multi_comma:
        hints.append("Values with multiple ',' are usually thousands separators; remove all ',' and infer decimal by the last separator.")
    if saw_percent:
        hints.append("For percentages: strip '%' and normalize 1–100 to 0–1.")
    hints.append("If raw values are mostly non-null but conversion yields mostly NaN, treat it as a parsing failure and switch to a more permissive sanitizer.")
    hints.append("Add a small parser self-check: parse the provided examples and confirm you do not get all-NaN for required numeric columns.")

    return "DE_PARSING_HINTS:\n- " + "\n- ".join(hints)

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
    """
    V4.1: Canonical columns ARE inputs. Returns contract ready for input processing.
    Legacy: Filter data_requirements to source='input'.
    """
    if not isinstance(contract, dict):
        return {}
    new_contract = dict(contract)

    # Legacy cleanup
    reqs = contract.get("data_requirements", []) or []
    if reqs:
        filtered = [r for r in reqs if isinstance(r, dict) and r.get("source", "input") == "input"]
        new_contract["data_requirements"] = filtered

    return new_contract

def _filter_contract_for_data_engineer(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    V4.1: Data Engineer uses canonical_columns (inputs) and dialect.
    """
    if not isinstance(contract, dict):
        return {}

    # In V4.1, contract is already structured by role ownership (DataEng owns loading/cleaning)
    # We just return the contract; the agent runbook handles the rest.
    new_contract = dict(contract)

    # Legacy legacy cleanup
    reqs = contract.get("data_requirements", []) or []
    if reqs:
        filtered = [r for r in reqs if isinstance(r, dict) and r.get("source", "input") == "input"]
        new_contract["data_requirements"] = filtered

    return new_contract


def build_de_objective(contract: Dict[str, Any]) -> str:
    objective = (
        "Clean dataset to satisfy contract canonical_columns; ensure numeric/date parsing; "
        "produce cleaned_data + manifest."
    )
    if not isinstance(contract, dict):
        return objective
    canonical = get_canonical_columns(contract) or []
    if canonical:
        return f"{objective} Canonical columns count: {len(canonical)}."
    return objective



def _resolve_required_input_columns(contract: Dict[str, Any], strategy: Dict[str, Any]) -> List[str]:
    """V4.1: Prefer canonical_columns, fallback to data_requirements for backwards compatibility."""
    if contract and isinstance(contract, dict):
        # V4.1: Try canonical_columns first
        canonical = get_canonical_columns(contract)
        if canonical:
            return canonical

        # Legacy fallback: data_requirements with source='input'
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
            if resolved:
                return resolved
    return strategy.get("required_columns", []) if strategy else []

def _resolve_contract_deliverables(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    V4.1 compatible deliverables resolver.
    Returns a list of dicts [{'path': str, 'required': bool, ...}]
    """
    from src.utils.contract_v41 import get_required_outputs

    # V4.1: get_required_outputs returns List[str]
    outputs = get_required_outputs(contract)
    if outputs:
        # Convert to list of dicts for compatibility with existing graph logic
        return [{"path": p, "required": True} for p in outputs]

    # Legacy Fallback
    spec = contract.get("spec_extraction") or {}
    deliverables = spec.get("deliverables")
    if isinstance(deliverables, list):
        return deliverables
    return []

def _resolve_contract_columns(contract: Dict[str, Any], sources: set[str] | None = None) -> List[str]:
    """V4.1: Use available_columns/canonical_columns instead of data_requirements source filtering."""
    if not contract or not isinstance(contract, dict):
        return []

    # If sources contains 'input', valid V4.1 sources are canonical_columns
    if sources and 'input' in sources:
        from src.utils.contract_v41 import get_canonical_columns
        return get_canonical_columns(contract)

    # If asking for 'derived', check derived_columns
    if sources and 'derived' in sources:
        from src.utils.contract_v41 import get_derived_column_names
        return get_derived_column_names(contract)

    # Legacy Fallback
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
    return out

def _resolve_allowed_columns_for_gate(
    state: Dict[str, Any],
    contract: Dict[str, Any],
    evaluation_spec: Dict[str, Any] | None = None,
) -> List[str]:
    allowed: List[str] = []
    csv_path = state.get("ml_data_path") or "data/cleaned_data.csv"
    if not os.path.exists(csv_path) and os.path.exists("data/cleaned_full.csv"):
        csv_path = "data/cleaned_full.csv"
    if os.path.exists(csv_path):
        try:
            import pandas as pd

            csv_sep = state.get("csv_sep", ",")
            csv_decimal = state.get("csv_decimal", ".")
            csv_encoding = state.get("csv_encoding", "utf-8")
            header_df = pd.read_csv(csv_path, nrows=0, sep=csv_sep, decimal=csv_decimal, encoding=csv_encoding)
            allowed.extend([str(col) for col in header_df.columns.tolist() if col])
        except Exception:
            pass

    if isinstance(contract, dict):
        from src.utils.contract_v41 import get_derived_column_names
        derived_cols = get_derived_column_names(contract)
        allowed.extend([str(c) for c in derived_cols if c])

        # Minimal legacy fallback for data_requirements
        data_reqs = contract.get("data_requirements", []) or []
        if isinstance(data_reqs, list):
            for req in data_reqs:
                if isinstance(req, dict) and req.get("source") == "derived":
                     name = req.get("canonical_name") or req.get("name")
                     if name: allowed.append(str(name))

    if not allowed:
        profile = state.get("profile") or state.get("dataset_profile")
        if isinstance(profile, dict):
            cols = profile.get("columns")
            if isinstance(cols, list):
                allowed.extend([str(col) for col in cols if col])

    seen: set[str] = set()
    deduped: List[str] = []
    for col in allowed:
        norm = _norm_name(col)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(col)
    return deduped

def _resolve_allowed_patterns_for_gate(contract: Any) -> List[str]:
    patterns: List[str] = []
    if not isinstance(contract, dict):
        return patterns
    artifact_requirements = contract.get("artifact_requirements", {})
    if not isinstance(artifact_requirements, dict):
        artifact_requirements = {}
    schema = artifact_requirements.get("file_schemas")
    if not isinstance(schema, dict):
        # Legacy Fallback
        spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
        schema = spec.get("artifact_schemas") if isinstance(spec, dict) else None
        if not isinstance(schema, dict):
            schema = contract.get("artifact_schemas") if isinstance(contract.get("artifact_schemas"), dict) else None
    if isinstance(schema, dict):
        scored_schema = schema.get("data/scored_rows.csv")
        if isinstance(scored_schema, dict):
            allowed_patterns = scored_schema.get("allowed_name_patterns")
            if isinstance(allowed_patterns, list):
                patterns.extend([str(pat) for pat in allowed_patterns if isinstance(pat, str) and pat.strip()])
    return patterns

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

def _is_glob_pattern(path: str) -> bool:
    if not path:
        return False
    return any(ch in path for ch in ["*", "?", "["]) or path.endswith(("/", "\\"))

def _normalize_output_path(path: str) -> str:
    """Normalize output paths to data/ prefix for consistency."""
    if not path:
        return path
    # Known files that should be in data/
    known_files = ["metrics.json", "alignment_check.json", "scored_rows.csv", "cleaned_data.csv"]
    basename = os.path.basename(path)
    if basename in known_files and not path.startswith("data/"):
        return f"data/{basename}"
    return path


def _resolve_required_outputs(contract: Dict[str, Any]) -> List[str]:
    """
    Resolve required outputs from V4.1 contract structure.
    Priority:
      1. evaluation_spec.required_outputs
      2. contract.required_outputs
      3. artifact_requirements.required_files
      4. Sensible fallback
    NO LONGER uses spec_extraction.deliverables (legacy).
    """
    if not isinstance(contract, dict):
        return []

    # Priority 1: evaluation_spec.required_outputs
    eval_spec = contract.get("evaluation_spec")
    if isinstance(eval_spec, dict):
        eval_outputs = eval_spec.get("required_outputs")
        if isinstance(eval_outputs, list) and eval_outputs:
            return [_normalize_output_path(str(p)) for p in eval_outputs if p]

    # Priority 2: contract.required_outputs
    req_outputs = contract.get("required_outputs")
    if isinstance(req_outputs, list) and req_outputs:
        return [_normalize_output_path(str(p)) for p in req_outputs if p]

    # Priority 3: artifact_requirements.required_files
    artifact_reqs = contract.get("artifact_requirements")
    if isinstance(artifact_reqs, dict):
        req_files = artifact_reqs.get("required_files")
        if isinstance(req_files, list) and req_files:
            resolved: List[str] = []
            for entry in req_files:
                if not entry:
                    continue
                if isinstance(entry, dict):
                    path = entry.get("path") or entry.get("output") or entry.get("artifact")
                else:
                    path = entry
                path = str(path) if path else ""
                if path:
                    resolved.append(_normalize_output_path(path))
            if resolved:
                return resolved

    # Fallback: minimal required outputs for a valid ML run
    return ["data/scored_rows.csv", "data/metrics.json", "data/alignment_check.json"]

def _resolve_expected_output_paths(contract: Dict[str, Any]) -> List[str]:
    """V4.1-only: delegates to _resolve_required_outputs."""
    return _resolve_required_outputs(contract)


def _purge_execution_outputs(required_outputs: List[str], keep_outputs: List[str] | None = None) -> None:
    protected = {
        "data/cleaned_data.csv",
        "data/cleaned_full.csv",
        "data/cleaning_manifest.json",
        "data/dataset_profile.json",
        "data/steward_summary.json",
        "data/steward_summary.txt",
        "data/column_mapping_summary.json",
        "data/strategy_spec.json",
        "data/plan.json",
        "data/evaluation_spec.json",
        "data/execution_contract.json",
    }
    keep_exact = {
        _normalize_path_posix(path)
        for path in (keep_outputs or [])
        if path and not _is_glob_pattern(path)
    }
    keep_globs = [str(path) for path in (keep_outputs or []) if path and _is_glob_pattern(path)]

    def _should_keep(path: str) -> bool:
        norm = _normalize_path_posix(path)
        if norm in keep_exact:
            return True
        for pattern in keep_globs:
            if fnmatch.fnmatch(norm, pattern):
                return True
        return False
    for path in required_outputs or []:
        if not path or _is_glob_pattern(path):
            continue
        if path in protected:
            continue
        if path.startswith("data/cleaned_"):
            continue
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    for extra in [
        "data/metrics.json",
        "data/scored_rows.csv",
        "data/alignment_check.json",
        "data/output_contract_report.json",
        "analysis/leakage_report.json",
    ]:
        if extra in protected:
            continue
        if _should_keep(extra):
            continue
        try:
            if os.path.exists(extra):
                os.remove(extra)
        except Exception:
            pass
    for folder in ["analysis", "models", os.path.join("static", "plots")]:
        try:
            if not os.path.isdir(folder):
                continue
            for base, _, names in os.walk(folder):
                for name in names:
                    full = os.path.join(base, name)
                    rel = _normalize_path_posix(os.path.relpath(full, "."))
                    if rel in protected or rel.startswith("data/cleaned_"):
                        continue
                    if _should_keep(rel):
                        continue
                    try:
                        os.remove(full)
                    except Exception:
                        pass
            for base, dirs, _ in os.walk(folder, topdown=False):
                for dname in dirs:
                    path = os.path.join(base, dname)
                    try:
                        if os.path.isdir(path) and not os.listdir(path):
                            os.rmdir(path)
                    except Exception:
                        pass
        except Exception:
            pass

def _find_stale_outputs(required_outputs: List[str], start_ts: float) -> List[str]:
    stale: List[str] = []
    if not required_outputs:
        return stale
    threshold = start_ts - 1.0
    protected = {
        "data/cleaned_data.csv",
        "data/cleaned_full.csv",
        "data/cleaning_manifest.json",
        "data/dataset_profile.json",
    }
    for path in required_outputs:
        if not path or _is_glob_pattern(path):
            continue
        if path in protected or path.startswith("data/cleaned_"):
            continue
        if not os.path.exists(path):
            continue
        try:
            if os.path.getmtime(path) < threshold:
                stale.append(path)
        except Exception:
            continue
    return stale

def _build_contract_min(contract: Dict[str, Any], evaluation_spec: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Build a minimal contract for ML Engineer prompt (V4.1-only).
    Excludes legacy spec_extraction; uses normalized required_outputs.
    """
    if not isinstance(contract, dict):
        contract = {}

    eval_spec = evaluation_spec if isinstance(evaluation_spec, dict) else contract.get("evaluation_spec") or {}
    alignment = contract.get("alignment_requirements") or (eval_spec.get("alignment_requirements") if isinstance(eval_spec, dict) else []) or []

    # Get derived_columns from V4.1 location (feature_engineering_plan or direct)
    fep = contract.get("feature_engineering_plan")
    derived_cols = []
    if isinstance(fep, dict):
        derived_cols = fep.get("derived_columns", [])
    if not derived_cols:
        derived_cols = contract.get("derived_columns", [])

    column_roles = contract.get("column_roles", {})
    decision_variables = contract.get("decision_variables")
    if not isinstance(decision_variables, list) or not decision_variables:
        if isinstance(column_roles, dict):
            inferred = [
                str(col)
                for col, role in column_roles.items()
                if col and str(role).strip().lower() == "decision"
            ]
            decision_variables = inferred

    allowed_sets_full = contract.get("allowed_feature_sets")
    if not isinstance(allowed_sets_full, dict):
        allowed_sets_full = {}

    def _as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return []

    roles = get_column_roles(contract)
    pre_decision = _as_list(roles.get("pre_decision"))
    decision = _as_list(roles.get("decision"))
    outcome = _as_list(roles.get("outcome"))
    audit_only = _as_list(roles.get("post_decision_audit_only") or roles.get("audit_only"))

    fallback_model = list(dict.fromkeys(pre_decision + decision))
    fallback_seg = list(pre_decision)
    fallback_forbidden = list(dict.fromkeys(outcome + audit_only))
    fallback_audit = list(audit_only)

    missing_sets: List[str] = []
    model_features = _as_list(allowed_sets_full.get("model_features"))
    if not model_features and "model_features" not in allowed_sets_full:
        model_features = fallback_model
        missing_sets.append("model_features")
    segmentation_features = _as_list(allowed_sets_full.get("segmentation_features"))
    if not segmentation_features and "segmentation_features" not in allowed_sets_full:
        segmentation_features = fallback_seg
        missing_sets.append("segmentation_features")
    forbidden_features = _as_list(allowed_sets_full.get("forbidden_for_modeling"))
    if not forbidden_features and "forbidden_for_modeling" not in allowed_sets_full:
        if "forbidden_features" in allowed_sets_full:
            forbidden_features = _as_list(allowed_sets_full.get("forbidden_features"))
        else:
            forbidden_features = fallback_forbidden
            missing_sets.append("forbidden_for_modeling")
    audit_only_features = _as_list(allowed_sets_full.get("audit_only_features"))
    if not audit_only_features and "audit_only_features" not in allowed_sets_full:
        audit_only_features = fallback_audit
        missing_sets.append("audit_only_features")
    if missing_sets:
        print(f"FALLBACK_FEATURE_SETS: {', '.join(sorted(set(missing_sets)))}")

    return {
        "contract_version": contract.get("contract_version"),
        "strategy_title": contract.get("strategy_title"),
        "business_objective": contract.get("business_objective"),
        # Use normalized required_outputs
        "required_outputs": _resolve_required_outputs(contract),
        "data_requirements": contract.get("data_requirements", []) or [],
        "alignment_requirements": alignment,
        "canonical_columns": contract.get("canonical_columns", []) or [],
        "column_roles": column_roles if isinstance(column_roles, dict) else {},
        "decision_variables": decision_variables if isinstance(decision_variables, list) else [],
        "derived_columns": derived_cols if isinstance(derived_cols, list) else [],
        "feature_availability": contract.get("feature_availability", []) or [],
        "availability_summary": contract.get("availability_summary", ""),
        "evaluation_spec": eval_spec,
        # V4.1 fields
        "artifact_requirements": contract.get("artifact_requirements", {}),
        "qa_gates": eval_spec.get("qa_gates", []) if isinstance(eval_spec, dict) else [],
        "reviewer_gates": eval_spec.get("reviewer_gates", []) if isinstance(eval_spec, dict) else [],
        "allowed_feature_sets": {
            "segmentation_features": segmentation_features,
            "model_features": model_features,
            "forbidden_features": forbidden_features,
            "audit_only_features": audit_only_features,
        },
        "leakage_execution_plan": contract.get("leakage_execution_plan", {}),
        "data_limited_mode": contract.get("data_limited_mode", False),
        "ml_engineer_runbook": contract.get("ml_engineer_runbook", {}),
    }

def _infer_artifact_type(path: str, deliverable_kind: str | None = None) -> str:
    if deliverable_kind:
        return str(deliverable_kind)
    lower = str(path or "").lower()
    if "plot" in lower or lower.endswith((".png", ".jpg", ".jpeg", ".svg")):
        return "plot"
    if lower.endswith(".csv"):
        if "pred" in lower or "score" in lower:
            return "predictions"
        if "summary" in lower:
            return "summary"
        return "dataset"
    if lower.endswith(".json"):
        if "weight" in lower:
            return "weights"
        if "metrics" in lower:
            return "metrics"
        if "forecast" in lower:
            return "forecast"
        if "importance" in lower:
            return "feature_importances"
        if "error" in lower or "residual" in lower:
            return "error_analysis"
        if "alignment" in lower or "report" in lower:
            return "report"
        if "insights" in lower:
            return "insights"
        if "strategy_spec" in lower:
            return "strategy_spec"
        if "plan" in lower:
            return "execution_plan"
        return "json"
    if lower.endswith(".md"):
        return "executive_summary"
    return "artifact"

def _infer_produced_by(path: str) -> str:
    lower = str(path or "").lower()
    if "clean" in lower or "cleaning" in lower:
        return "data_engineer"
    if "strategy_spec" in lower:
        return "strategist"
    if "plan" in lower:
        return "execution_planner"
    if "insights" in lower:
        return "results_advisor"
    if "executive_summary" in lower:
        return "business_translator"
    if "alignment" in lower or "report" in lower:
        return "system"
    return "ml_engineer"

def _build_artifact_index(
    paths: List[str],
    deliverables: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    by_path = {}
    deliverable_lookup: Dict[str, Dict[str, Any]] = {}
    for item in deliverables or []:
        if isinstance(item, dict) and item.get("path"):
            deliverable_lookup[item["path"]] = item
    for path in paths or []:
        if not path:
            continue
        deliverable = deliverable_lookup.get(path) or {}
        artifact_type = _infer_artifact_type(path, deliverable.get("kind"))
        produced_by = _infer_produced_by(path)
        by_path[path] = {
            "artifact_type": artifact_type,
            "path": path,
            "schema_version": "1",
            "produced_by": produced_by,
        }
    return list(by_path.values())

def _merge_artifact_index_entries(
    base: List[Dict[str, Any]],
    additions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in base or []:
        path = item.get("path") if isinstance(item, dict) else None
        if path:
            merged[path] = item
    for item in additions or []:
        path = item.get("path") if isinstance(item, dict) else None
        if path:
            merged[path] = item
    return list(merged.values())


def _merge_non_empty_policy(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and len(value) == 0:
            continue
        merged[key] = value
    base_constraints = base.get("constraints") if isinstance(base, dict) else {}
    override_constraints = override.get("constraints") if isinstance(override, dict) else {}
    if isinstance(base_constraints, dict) or isinstance(override_constraints, dict):
        merged["constraints"] = {
            **(base_constraints if isinstance(base_constraints, dict) else {}),
            **(override_constraints if isinstance(override_constraints, dict) else {}),
        }
    if isinstance(override, dict) and isinstance(override.get("plot_spec"), dict) and override.get("plot_spec"):
        merged["plot_spec"] = override["plot_spec"]
    return merged


def _ensure_plot_spec_in_policy(policy: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        policy = {}
    plot_spec = policy.get("plot_spec")
    needs_plot_spec = True
    if isinstance(plot_spec, dict):
        enabled = plot_spec.get("enabled", True)
        plots = plot_spec.get("plots")
        if enabled is False:
            needs_plot_spec = False
        elif isinstance(plots, list) and plots:
            needs_plot_spec = False
    if needs_plot_spec:
        policy = dict(policy)
        policy["plot_spec"] = build_plot_spec(contract)
    return policy


def _compact_reporting_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    plot_spec = policy.get("plot_spec")
    if not isinstance(plot_spec, dict) or not plot_spec:
        return {}
    plots = plot_spec.get("plots") if isinstance(plot_spec.get("plots"), list) else []
    max_plots = plot_spec.get("max_plots")
    if isinstance(max_plots, (int, float)):
        max_plots_val = int(max_plots)
    elif isinstance(max_plots, str) and max_plots.strip().isdigit():
        max_plots_val = int(max_plots.strip())
    else:
        max_plots_val = len(plots)
    return {
        "plot_spec": {
            "enabled": bool(plot_spec.get("enabled", True)),
            "max_plots": max_plots_val,
        }
    }


def _maybe_set_contract_min_policy(contract_min: Dict[str, Any] | None, policy: Dict[str, Any]) -> None:
    if not isinstance(contract_min, dict):
        return
    existing = contract_min.get("reporting_policy")
    if isinstance(existing, dict) and existing:
        return
    compact = _compact_reporting_policy(policy)
    if compact:
        contract_min["reporting_policy"] = compact


def _build_contract_views(
    state: Dict[str, Any],
    contract: Dict[str, Any],
    contract_min: Dict[str, Any] | None,
) -> Dict[str, Any]:
    # P0 FIX: Do NOT read from disk fallback - prevents cross-run contamination.
    # If artifact_index is empty, code below will build it fresh from contract.
    artifact_index = state.get("artifact_index") or []
    if not artifact_index:
        required_outputs = []
        if isinstance(contract_min, dict):
            required_outputs = contract_min.get("required_outputs") or []
        if not required_outputs:
            required_outputs = _resolve_required_outputs(contract)
        deliverables = _resolve_contract_deliverables(contract)
        artifact_index = _build_artifact_index(required_outputs, deliverables)
    de_view = build_de_view(contract, contract_min or {}, artifact_index)
    ml_view = build_ml_view(contract, contract_min or {}, artifact_index)
    cleaning_view = build_cleaning_view(contract, contract_min or {}, artifact_index)
    qa_view = build_qa_view(contract, contract_min or {}, artifact_index)
    reviewer_view = build_reviewer_view(contract, contract_min or {}, artifact_index)
    translator_view = build_translator_view(contract, contract_min or {}, artifact_index)
    results_advisor_view = build_results_advisor_view(contract, contract_min or {}, artifact_index)
    views = {
        "de_view": de_view,
        "ml_view": ml_view,
        "cleaning_view": cleaning_view,
        "qa_view": qa_view,
        "reviewer_view": reviewer_view,
        "translator_view": translator_view,
        "results_advisor_view": results_advisor_view,
    }
    view_paths = persist_views(
        views,
        base_dir="data",
        run_bundle_dir=state.get("run_bundle_dir"),
    )
    return {
        "contract_views": views,
        "contract_view_paths": view_paths,
        "artifact_index": artifact_index,
        "de_view": de_view,
        "ml_view": ml_view,
        "cleaning_view": cleaning_view,
        "qa_view": qa_view,
        "reviewer_view": reviewer_view,
        "translator_view": translator_view,
        "results_advisor_view": results_advisor_view,
    }

def _deliverable_id_from_path(path: str) -> str:
    base = os.path.basename(str(path)) or str(path)
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_").lower()
    return cleaned or "deliverable"

def _infer_deliverable_kind(path: str) -> str:
    lower = str(path).lower()
    if "plots" in lower or lower.endswith((".png", ".jpg", ".jpeg", ".svg")):
        return "plot"
    if lower.endswith(".csv"):
        return "dataset"
    if lower.endswith(".json"):
        if "metrics" in lower:
            return "metrics"
        if "weights" in lower:
            return "weights"
        if "alignment" in lower or "report" in lower:
            return "report"
        return "json"
    return "artifact"

def _infer_scored_rows_schema(
    contract: Dict[str, Any],
    evaluation_spec: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """
    UNIVERSAL: Infer schema for scored_rows.csv based on strategy type and contract.

    Works for ANY objective type (prediction, optimization, clustering, descriptive)
    with NO hardcoded column names or business logic.

    Returns dict with:
      - required_columns: list of canonical input columns (always included)
      - derived_columns: list of derived output columns (based on strategy type)
      - row_count: "must_match_input" (always)
    """
    if not isinstance(contract, dict):
        return None

    # Base schema: always include input columns
    canonical_columns = contract.get("canonical_columns") or []
    if not isinstance(canonical_columns, list):
        canonical_columns = []

    schema = {
        "required_columns": canonical_columns.copy(),
        "derived_columns": [],
        "row_count": "must_match_input",
        "description": "One row per input record with canonical columns plus derived outputs (predictions, scores, segments, optimal values as applicable)"
    }

    # Infer strategy type from contract
    strategy_context = contract.get("strategy_context") or {}
    analysis_type = str(strategy_context.get("analysis_type", "")).lower()

    # Check if evaluation spec requires specific outputs
    eval_spec = evaluation_spec or contract.get("evaluation_spec") or {}
    objective_type = str(eval_spec.get("objective_type", "")).lower()
    requires_target = bool(eval_spec.get("requires_target"))

    # Derive output columns based on objective type (UNIVERSAL)
    derived_cols = schema["derived_columns"]

    # 1. Segmentation/Clustering outputs
    segmentation = eval_spec.get("segmentation") or {}
    if segmentation.get("required") or "segment" in analysis_type or "cluster" in analysis_type:
        # Generic segment identifier (not hardcoded to specific name)
        derived_cols.append("segment_id")  # or cluster_id

    # 2. Predictive modeling outputs  
    if "predict" in analysis_type or "predict" in objective_type or requires_target:
        # Generic prediction columns (adapt to target type)
        target_info = eval_spec.get("target") or {}
        target_name = target_info.get("name", "target")

        # Add prediction column (name based on target if available)
        if target_name and target_name != "target":
            derived_cols.append(f"predicted_{target_name}")
        else:
            derived_cols.append("predicted_value")  # generic

        # Add probability for classification tasks
        if requires_target:  # Likely binary/multi-class
            derived_cols.append("predicted_probability")

    # 3. Optimization/Prescriptive outputs
    decision_var = eval_spec.get("decision_variable") or {}
    decision_var_name = decision_var.get("name")

    if decision_var_name or "optim" in analysis_type or "prescript" in objective_type:
        # Add optimal value for decision variable (generic name)
        if decision_var_name:
            derived_cols.append(f"optimal_{decision_var_name}")
        else:
            derived_cols.append("optimal_value")  # fallback generic

        # Optionally add expected outcome at optimal
        if requires_target:
            derived_cols.append("expected_outcome_at_optimal")

    # 4. Ranking/Scoring outputs
    if "rank" in analysis_type or "scor" in analysis_type:
        derived_cols.append("score")
        derived_cols.append("rank")

    # Remove duplicates while preserving order
    seen = set()
    unique_derived = []
    for col in derived_cols:
        if col not in seen:
            seen.add(col)
            unique_derived.append(col)
    schema["derived_columns"] = unique_derived

    return schema


def _ensure_contract_deliverable(
    contract: Dict[str, Any],
    path: str,
    required: bool = True,
    kind: str | None = None,
    description: str | None = None,
    schema: Dict[str, Any] | None = None,  # ← NEW: optional schema
) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    if not path:
        return contract
    spec = contract.get("spec_extraction")
    if not isinstance(spec, dict):
        spec = {}
    deliverables = spec.get("deliverables")
    if not isinstance(deliverables, list):
        deliverables = []

    normalized: List[Dict[str, Any]] = []
    for item in deliverables:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(
                {
                    "id": _deliverable_id_from_path(item),
                    "path": item,
                    "required": True,
                    "kind": _infer_deliverable_kind(item),
                    "description": "Requested deliverable.",
                }
            )
    deliverables = normalized

    existing = None
    for item in deliverables:
        if item.get("path") == path:
            existing = item
            break
    if existing:
        existing["required"] = bool(required)
        if kind:
            existing["kind"] = kind
        if description:
            existing["description"] = description
        if schema:  # ← NEW: attach schema if provided
            existing["schema"] = schema
        if not existing.get("id"):
            existing["id"] = _deliverable_id_from_path(path)
    else:
        new_deliverable = {
            "id": _deliverable_id_from_path(path),
            "path": path,
            "required": bool(required),
            "kind": kind or _infer_deliverable_kind(path),
            "description": description or "Requested deliverable.",
        }
        if schema:  # ← NEW: attach schema if provided
            new_deliverable["schema"] = schema
        deliverables.append(new_deliverable)


    spec["deliverables"] = deliverables
    contract["spec_extraction"] = spec

    # --- FIX: DO NOT OVERWRITE required_outputs ---
    # Treat existing required_outputs as source-of-truth.
    # Only ADD new required paths (dedupe).
    existing_outputs = contract.get("required_outputs")
    if not isinstance(existing_outputs, list):
        existing_outputs = []

    seen: set[str] = set(_normalize_output_path(p) for p in existing_outputs if p)
    merged_outputs = [_normalize_output_path(p) for p in existing_outputs if p]

    # Add new required deliverables not already in required_outputs
    if required:
        norm_path = _normalize_output_path(path)
        if norm_path not in seen:
            merged_outputs.append(norm_path)

    contract["required_outputs"] = merged_outputs
    return contract

def _ensure_scored_rows_output(
    contract: Dict[str, Any],
    evaluation_spec: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    spec = contract.get("spec_extraction") or {}
    deliverables = spec.get("deliverables")
    required_outputs = contract.get("required_outputs", []) or []
    requires_row_scoring = False
    if isinstance(evaluation_spec, dict):
        requires_row_scoring = bool(evaluation_spec.get("requires_row_scoring"))
    explicit_required = False
    if isinstance(deliverables, list):
        for item in deliverables:
            if isinstance(item, dict) and item.get("path") == "data/scored_rows.csv":
                if bool(item.get("required")):
                    explicit_required = True
                break
            if isinstance(item, str) and item == "data/scored_rows.csv":
                explicit_required = True
                break
    if "data/scored_rows.csv" in required_outputs:
        explicit_required = True
    if explicit_required or requires_row_scoring:
        # ← NEW: Generate universal schema for scored_rows.csv
        scored_rows_schema = _infer_scored_rows_schema(contract, evaluation_spec)

        contract = _ensure_contract_deliverable(
            contract,
            "data/scored_rows.csv",
            required=True,
            kind="dataset",
            description="Row-level scores and key features.",
            schema=scored_rows_schema,  # ← NEW: attach inferred schema
        )
        return contract
    if isinstance(deliverables, list):
        for item in deliverables:
            if isinstance(item, dict) and item.get("path") == "data/scored_rows.csv":
                item["required"] = False
        required_outputs = []
        seen: set[str] = set()
        for item in deliverables:
            if isinstance(item, dict) and item.get("required"):
                path = item.get("path")
            elif isinstance(item, str):
                path = item
            else:
                path = None
            if not path or path in seen:
                continue
            seen.add(path)
            required_outputs.append(path)
        contract["required_outputs"] = required_outputs
        spec["deliverables"] = deliverables
        contract["spec_extraction"] = spec
    else:
        if "data/scored_rows.csv" in required_outputs:
            contract["required_outputs"] = [
                path for path in required_outputs if path != "data/scored_rows.csv"
            ]
    return contract


def _should_run_case_alignment(
    contract: Dict[str, Any] | None,
    evaluation_spec: Dict[str, Any] | None = None,
) -> bool:
    if not isinstance(contract, dict):
        contract = {}
    skip_reason = _case_alignment_skip_reason(contract, evaluation_spec)
    if skip_reason:
        return False
    required_outputs = _resolve_required_outputs(contract)
    required_set = {str(p) for p in (required_outputs or []) if p}
    requires_outputs = bool(
        required_set.intersection({"data/scored_rows.csv", "data/weights.json", "data/case_summary.csv"})
    )
    spec_requires = False
    if isinstance(evaluation_spec, dict):
        spec_requires = bool(
            evaluation_spec.get("requires_case_alignment")
            or evaluation_spec.get("case_alignment_required")
            or evaluation_spec.get("requires_row_scoring")
        )
    flags = _resolve_eval_flags(evaluation_spec) if isinstance(evaluation_spec, dict) else {}
    return requires_outputs or spec_requires or bool(flags.get("requires_row_scoring"))

def _case_alignment_skip_reason(
    contract: Dict[str, Any] | None,
    evaluation_spec: Dict[str, Any] | None = None,
) -> str:
    if not isinstance(contract, dict):
        contract = {}
    spec = contract.get("spec_extraction") or {}
    if not isinstance(spec, dict):
        spec = {}
    case_taxonomy = spec.get("case_taxonomy") if isinstance(spec.get("case_taxonomy"), list) else []
    case_key = spec.get("case_key")
    case_columns = spec.get("case_columns")
    if isinstance(evaluation_spec, dict):
        case_key = case_key or evaluation_spec.get("case_key")
        case_columns = case_columns or evaluation_spec.get("case_columns")

    def _scored_rows_has_group_signals(scored_path: str = "data/scored_rows.csv") -> bool:
        if not os.path.exists(scored_path):
            return False
        dialect = _load_output_dialect_local()
        header = _read_csv_header(scored_path, dialect.get("encoding", "utf-8"), dialect.get("sep", ","))
        if not header:
            return False
        normed = [_norm_name(col) for col in header]
        has_group = any(any(tok in name for tok in ["case", "group", "segment", "bucket", "cluster"]) for name in normed)
        has_score = any(any(tok in name for tok in ["score", "pred", "prob", "rank"]) for name in normed)
        return bool(has_group and has_score)

    if not case_taxonomy:
        if _scored_rows_has_group_signals():
            return ""
        return "case_taxonomy missing or empty"
    if not case_key and not case_columns:
        if _scored_rows_has_group_signals():
            return ""
        return "case_key/case_columns missing"
    return ""

def _ensure_alignment_check_output(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    reqs = contract.get("alignment_requirements", [])
    if not isinstance(reqs, list) or not reqs:
        return contract
    contract = _ensure_contract_deliverable(
        contract,
        "data/alignment_check.json",
        required=True,
        kind="report",
        description="Alignment check results for contract requirements.",
    )
    return contract

def _normalize_execution_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    V4.1 CUTOVER: Minimal normalization only.

    Contract is now IMMUTABLE after Execution Planner generates it.
    We only normalize quality_gates structure for backward compatibility.
    All other fields are preserved as-is from V4.1 schema.
    """
    if not isinstance(contract, dict):
        return {}
    normalized = dict(contract)

    # Only normalize quality_gates structure (for backward compat)
    quality_gates = normalized.get("quality_gates")
    if isinstance(quality_gates, list):
        normalized["quality_gates_raw"] = quality_gates
        normalized["quality_gates"] = {}
    elif not isinstance(quality_gates, dict):
        normalized["quality_gates"] = {}

    # Ensure basic required structures exist (non-invasive)
    if not isinstance(normalized.get("business_alignment"), dict):
        normalized["business_alignment"] = {}
    if not isinstance(normalized.get("iteration_policy"), dict):
        normalized["iteration_policy"] = {}
    if not isinstance(normalized.get("compliance_checklist"), list):
        normalized["compliance_checklist"] = []
    if not isinstance(normalized.get("alignment_requirements"), list):
        normalized["alignment_requirements"] = []
    if not isinstance(normalized.get("evaluation_spec"), dict):
        normalized["evaluation_spec"] = {}

    # V4.1: DO NOT touch spec_extraction, data_requirements, or role_runbooks
    # The contract comes from Execution Planner with V4.1 schema already populated

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
        "TIME_SERIES_SPLIT_REQUIRED": [
            "Use a time-based split (TimeSeriesSplit or chronological holdout with shuffle=False) for forecasting.",
        ],
        "DATA_LOAD_MISSING": [
            "Load the cleaned dataset using pd.read_csv with the detected dialect before processing.",
        ],
        "DATA_PATH_NOT_USED": [
            "CRITICAL: Use the EXACT path provided in context via $data_path variable (e.g., INPUT_FILE = '$data_path').",
            "DO NOT hardcode arbitrary paths like 'input.csv', 'data.csv', 'raw_data.csv', 'cleaned.csv', etc.",
            "The system will substitute $data_path with the correct path (usually 'data/cleaned_data.csv').",
        ],
        "REQUIRED_OUTPUTS_MISSING": [
            "Write all required outputs to the exact contract paths before exiting.",
        ],
        "REQUIRED_COLUMNS_NOT_USED": [
            "Use the required business columns from the contract (e.g., Size, Debtors, Sector) in mapping/processing.",
        ],
        "SYNTHETIC_DATA_DETECTED": [
            "CRITICAL: Remove ALL synthetic data generation logic. Common violations:",
            "Remove any input existence checks and any fallback branches that generate data; the orchestrator guarantees the input exists.",
            "- Fallback patterns: 'if not os.path.exists(filepath): generate dummy data'",
            "- Random generation: np.random, random.randint, faker, pd.DataFrame with literals",
            "- Dummy datasets: pd.DataFrame({'col': [val1, val2, ...]}) when input is missing",
            "The cleaned dataset WILL exist at $data_path. Trust it. If missing, let pd.read_csv() raise FileNotFoundError.",
        ],
        "DATAFRAME_LITERAL_OVERWRITE": [
            "Do not overwrite df/data with pd.DataFrame literals; always load from the cleaned CSV.",
        ],
        "SCORED_ROWS_SCHEMA_VIOLATION": [
            "Keep scored_rows.csv limited to row-level scores/segments; write *_delta columns to a separate artifact.",
        ],
        "ALIGNMENT_REQUIREMENTS_MISSING": [
            "Populate alignment_check.json with per-requirement status + evidence list.",
        ],
        "SEGMENT_FEATURES_MISSING": [
            "Define SEGMENT_FEATURES list with pre-decision columns and use it for segmentation.",
        ],
        "SEGMENT_FEATURES_INCOMPLETE": [
            "SEGMENT_FEATURES must include all pre-decision columns from the contract (e.g., Size, Debtors, Sector).",
        ],
        "MODEL_FEATURES_MISSING": [
            "Define MODEL_FEATURES list for the success model and use it to build X.",
        ],
        "MODEL_FEATURES_INCOMPLETE": [
            "MODEL_FEATURES must include the decision variable (e.g., 1stYearAmount).",
        ],
        "CANONICAL_MAPPING_REQUIRED": [
            "Use canonical column names in SEGMENT_FEATURES/MODEL_FEATURES and rename columns to canonical names after mapping.",
        ],
        "DERIVED_TARGET_REQUIRED": [
            "If target is derived, add a guard to derive it when missing and log `DERIVED_TARGET:<name>` to stdout.",
        ],
        "UNKNOWN_COLUMNS_REFERENCED": [
            "Remove invented columns and only reference columns from the cleaned dataset / contract mapping.",
        ],
        "DF_COLUMN_ASSIGNMENT_FORBIDDEN": [
            "Do not assign new columns into df; use Pipeline/ColumnTransformer or write derived columns to separate artifacts.",
        ],
        "DIALECT_LOADING_MISSING": [
            "CRITICAL: You MUST load output_dialect from 'data/cleaning_manifest.json' BEFORE loading any CSV data.",
            "Define a load_dialect() function that reads cleaning_manifest.json and extracts {sep, decimal, encoding}.",
            "Then use: sep, decimal, encoding = load_dialect() and apply them to ALL pd.read_csv() and .to_csv() calls.",
            "NEVER hardcode sep=',', decimal='.', or other dialect values. Always read from manifest first.",
            "Example pattern:",
            "  def load_dialect():",
            "      manifest_path = 'data/cleaning_manifest.json'",
            "      if os.path.exists(manifest_path):",
            "          with open(manifest_path, 'r') as f:",
            "              manifest = json.load(f)",
            "          dialect = manifest.get('output_dialect', {})",
            "          return dialect.get('sep', ';'), dialect.get('decimal', ','), dialect.get('encoding', 'utf-8')",
            "      return ';', ',', 'utf-8'",
            "",
            "  sep, decimal, encoding = load_dialect()",
            "  df = pd.read_csv(INPUT_FILE, sep=sep, decimal=decimal, encoding=encoding)",
        ],
        "IMPUTER_REQUIRED": [
            "Include SimpleImputer in the preprocessing pipeline before modeling.",
        ],
        "BASELINE_REQUIRED": [
            "Add a DummyClassifier/DummyRegressor baseline and report its metric for lift.",
        ],
        "CALIBRATED_IMPORTANCE_UNSUPPORTED": [
            "Avoid feature_importances_/base_estimator on CalibratedClassifierCV; use coef_ or skip importances.",
        ],
        "segmentation_predecision": [
            "Ensure segmentation uses only pre-decision features defined in SEGMENT_FEATURES.",
        ],
        "decision_variable_handling": [
            "Use decision variables only in the optimization/modeling step, not for segmentation.",
        ],
        "decision_optimization_required": [
            "Provide decision-optimization evidence aligned to the contract (decision variable, constraints, and scoring formula).",
        ],
        "validation_required": [
            "Add cross-validation or an appropriate validation strategy per evaluation_spec.",
        ],
        "methodology_alignment": [
            "Align model choice and outputs with the objective type in evaluation_spec.",
        ],
        "business_value": [
            "Deliver artifacts/insights that directly answer the business objective.",
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

def _get_iteration_policy(state: Dict[str, Any]) -> Dict[str, Any] | None:
    contract = state.get("execution_contract") or {}
    policy = contract.get("iteration_policy")
    if not isinstance(policy, dict):
        return None
    compliance_max = policy.get("compliance_bootstrap_max")
    metric_max = policy.get("metric_improvement_max")
    runtime_max = policy.get("runtime_fix_max")
    if metric_max is None:
        legacy_max = policy.get("max_iterations")
        if legacy_max is not None:
            metric_max = legacy_max
    plateau_window = policy.get("plateau_window")
    plateau_epsilon = policy.get("plateau_epsilon")
    out: Dict[str, Any] = {}
    for key, val in {
        "compliance_bootstrap_max": compliance_max,
        "metric_improvement_max": metric_max,
        "runtime_fix_max": runtime_max,
        "plateau_window": plateau_window,
        "plateau_epsilon": plateau_epsilon,
    }.items():
        if val is None:
            continue
        try:
            if key == "plateau_epsilon":
                out[key] = float(val)
            else:
                out[key] = max(1, int(val))
        except Exception:
            continue
    if out:
        print(f"ITER_POLICY metric_improvement_max={out.get('metric_improvement_max')} runtime_fix_max={out.get('runtime_fix_max')} plateau_window={out.get('plateau_window')} epsilon={out.get('plateau_epsilon')}")
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
    if feedback and any(token in feedback for token in [
        "CODE_AUDIT_REJECTED",
        "OUTPUT_CONTRACT_MISSING",
        "METRICS_SCHEMA_INCONSISTENT",  # Malformed metrics data = code error, not low metrics
    ]):
        return "compliance"
    return "metric"

def _normalize_allowed_feature_sets(feature_sets: Any) -> Any:
    """Normalize allowed_feature_sets for consistent comparison.

    If dict: sort each list value and return normalized dict.
    If list: sort and return.
    Otherwise: return as-is.
    """
    if isinstance(feature_sets, dict):
        normalized = {}
        for key, val in feature_sets.items():
            if isinstance(val, list):
                normalized[key] = sorted(val)
            else:
                normalized[key] = val
        return normalized
    elif isinstance(feature_sets, list):
        return sorted(feature_sets)
    return feature_sets

def _derive_forbidden_from_allowed(allowed_feature_sets: Any, explicit_forbidden: List[str]) -> List[str]:
    """Derive forbidden features from allowed_feature_sets if explicit list is empty.

    If allowed_feature_sets is a dict with 'forbidden' key, use that.
    Otherwise return explicit_forbidden.
    """
    if explicit_forbidden:
        return sorted(explicit_forbidden)
    if isinstance(allowed_feature_sets, dict):
        forbidden = allowed_feature_sets.get("forbidden", [])
        if isinstance(forbidden, list):
            return sorted(forbidden)
    return []

def _capture_strategy_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """Capture snapshot of strategy and contract for drift detection."""
    strategy = state.get("selected_strategy") or {}
    contract = state.get("execution_contract") or {}

    # Use contract_version with fallback to version
    contract_version = contract.get("contract_version") or contract.get("version")

    # Normalize allowed_feature_sets (handles both dict and list formats)
    raw_allowed = contract.get("allowed_feature_sets", []) or []
    allowed_normalized = _normalize_allowed_feature_sets(raw_allowed)

    # Derive forbidden_features from allowed_feature_sets if not explicitly set
    raw_forbidden = contract.get("forbidden_features", []) or []
    forbidden_normalized = _derive_forbidden_from_allowed(raw_allowed, raw_forbidden)

    return {
        "strategy_title": strategy.get("title"),
        "strategy_id": strategy.get("id") or strategy.get("strategy_id"),
        "contract_version": contract_version,
        "canonical_columns": sorted(contract.get("canonical_columns", []) or []),
        "decision_columns": sorted(contract.get("decision_columns", []) or []),
        "outcome_columns": sorted(contract.get("outcome_columns", []) or []),
        "allowed_feature_sets": allowed_normalized,
        "forbidden_features": forbidden_normalized,
    }

def _validate_strategy_lock(state: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    """
    Validate that strategy/contract has not drifted from initial snapshot.
    Returns (ok, details) where ok=True means no drift detected.
    """
    snapshot = state.get("strategy_lock_snapshot")
    if not snapshot:
        # No snapshot to compare - first iteration, capture it
        return True, {"reason": "no_snapshot_yet"}

    current = _capture_strategy_snapshot(state)
    drifts: List[str] = []

    # Compare key fields
    if snapshot.get("strategy_title") and current.get("strategy_title"):
        if snapshot["strategy_title"] != current["strategy_title"]:
            drifts.append(f"strategy_title: {snapshot['strategy_title']} -> {current['strategy_title']}")

    if snapshot.get("strategy_id") and current.get("strategy_id"):
        if snapshot["strategy_id"] != current["strategy_id"]:
            drifts.append(f"strategy_id: {snapshot['strategy_id']} -> {current['strategy_id']}")

    if snapshot.get("contract_version") and current.get("contract_version"):
        if snapshot["contract_version"] != current["contract_version"]:
            drifts.append(f"contract_version: {snapshot['contract_version']} -> {current['contract_version']}")

    if snapshot.get("canonical_columns") and current.get("canonical_columns"):
        if snapshot["canonical_columns"] != current["canonical_columns"]:
            added = set(current["canonical_columns"]) - set(snapshot["canonical_columns"])
            removed = set(snapshot["canonical_columns"]) - set(current["canonical_columns"])
            if added or removed:
                drifts.append(f"canonical_columns changed: +{list(added)}, -{list(removed)}")

    if snapshot.get("decision_columns") and current.get("decision_columns"):
        if snapshot["decision_columns"] != current["decision_columns"]:
            drifts.append(f"decision_columns changed")

    if snapshot.get("outcome_columns") and current.get("outcome_columns"):
        if snapshot["outcome_columns"] != current["outcome_columns"]:
            drifts.append(f"outcome_columns changed")

    # Compare allowed_feature_sets (normalized - can be dict or list)
    snap_allowed = snapshot.get("allowed_feature_sets")
    curr_allowed = current.get("allowed_feature_sets")
    if snap_allowed and curr_allowed:
        if snap_allowed != curr_allowed:
            drifts.append(f"allowed_feature_sets changed")

    if snapshot.get("forbidden_features") and current.get("forbidden_features"):
        if snapshot["forbidden_features"] != current["forbidden_features"]:
            drifts.append(f"forbidden_features changed")

    if drifts:
        return False, {"drifts": drifts, "snapshot": snapshot, "current": current}

    return True, {"reason": "no_drift"}

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
    limit_map = {
        "de_calls": "max_de_calls",
        "ml_calls": "max_ml_calls",
        "reviewer_calls": "max_reviewer_calls",
        "qa_calls": "max_qa_calls",
        "execution_calls": "max_execution_calls",
    }
    for counter_key, limit_key in limit_map.items():
        limit = budget.get(limit_key, DEFAULT_RUN_BUDGET.get(limit_key))
        if limit is None:
            continue
        try:
            if counters.get(counter_key, 0) > int(limit):
                counters[counter_key] = int(limit)
        except Exception:
            continue
    return {"run_budget": budget, "budget_counters": counters}

def _consume_budget(state: Dict[str, Any], counter_key: str, limit_key: str, label: str):
    budget = state.get("run_budget") or DEFAULT_RUN_BUDGET
    counters = dict(state.get("budget_counters") or {})
    limit = budget.get(limit_key, DEFAULT_RUN_BUDGET.get(limit_key))
    used = counters.get(counter_key, 0)
    attempted = used + 1
    if limit is not None and attempted > limit:
        return False, counters, f"BUDGET_EXCEEDED: {label} exceeded {used}/{limit}"
    counters[counter_key] = attempted
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
    if "_safe_dump_json" in code and "json.dump = _safe_dump_json" in code:
        return False
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

def _resolve_eval_flags(evaluation_spec: Dict[str, Any] | None) -> Dict[str, bool]:
    if not isinstance(evaluation_spec, dict) or not evaluation_spec:
        return {
            "requires_target": True,
            "requires_supervised_split": True,
            "requires_time_series_split": False,
            "requires_row_scoring": False,
        }
    objective_type = str(evaluation_spec.get("objective_type") or "").lower()
    requires_target = None
    requires_supervised_split = None
    requires_time_series_split = None
    requires_row_scoring = None
    if isinstance(evaluation_spec, dict):
        requires_target = evaluation_spec.get("requires_target")
        requires_supervised_split = evaluation_spec.get("requires_supervised_split")
        requires_time_series_split = evaluation_spec.get("requires_time_series_split")
        requires_row_scoring = evaluation_spec.get("requires_row_scoring")
    if requires_target is None:
        requires_target = objective_type in {"predictive", "prescriptive", "forecasting"}
    if requires_time_series_split is None:
        requires_time_series_split = objective_type == "forecasting"
    if requires_supervised_split is None:
        requires_supervised_split = bool(requires_target)
    if requires_row_scoring is None:
        requires_row_scoring = False
    return {
        "requires_target": bool(requires_target),
        "requires_supervised_split": bool(requires_supervised_split),
        "requires_time_series_split": bool(requires_time_series_split),
        "requires_row_scoring": bool(requires_row_scoring),
    }

def ml_quality_preflight(
    code: str,
    evaluation_spec: Dict[str, Any] | None = None,
    allowed_columns: List[str] | None = None,
    allowed_patterns: List[str] | None = None,
) -> List[str]:
    """
    Static ML quality checks to prevent QA loops before reviewer/sandbox.
    Returns list of missing gates.
    """
    import ast

    issues: List[str] = []
    code_lower = code.lower()
    if "mapping summary" not in code_lower:
        issues.append("MAPPING_SUMMARY")
    if "calibratedclassifiercv" in code_lower and (
        "feature_importances_" in code_lower or "base_estimator" in code_lower
    ):
        issues.append("CALIBRATED_IMPORTANCE_UNSUPPORTED")

    try:
        tree = ast.parse(code)
    except Exception:
        return ["AST_PARSE_FAILED"]

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

    flags = _resolve_eval_flags(evaluation_spec)
    if not flags["requires_target"]:
        issues = [
            issue
            for issue in issues
            if issue not in {"TARGET_NOT_IN_X", "TARGET_VARIANCE_GUARD", "CROSS_VALIDATION_REQUIRED", "TIME_SERIES_SPLIT_REQUIRED"}
        ]
    if not flags["requires_supervised_split"]:
        issues = [
            issue
            for issue in issues
            if issue not in {"CROSS_VALIDATION_REQUIRED", "TIME_SERIES_SPLIT_REQUIRED"}
        ]
    if flags["requires_time_series_split"]:
        has_time_series_split = (
            "timeseriessplit" in code_lower
            or ("train_test_split" in code_lower and "shuffle=false" in code_lower)
            or "rolling" in code_lower
            or "expanding" in code_lower
        )
        if not has_time_series_split:
            issues.append("TIME_SERIES_SPLIT_REQUIRED")
        issues = [issue for issue in issues if issue != "CROSS_VALIDATION_REQUIRED"]

    def _has_symbol(names: set[str]) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in names:
                return True
            if isinstance(node, ast.Attribute) and node.attr in names:
                return True
        return False

    def _scored_rows_has_delta() -> bool:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            target = node.value
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            else:
                name = None
            if name not in {"scored_rows", "scored_df", "scores_df"}:
                continue
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                if "delta" in slice_node.value.lower():
                    return True
        return False

    baseline_required = True
    if flags["requires_target"] and baseline_required:
        if not _has_symbol({"DummyClassifier", "DummyRegressor"}):
            issues.append("BASELINE_REQUIRED")

    if flags["requires_target"] or flags["requires_supervised_split"]:
        # Enhanced NaN hygiene check: Detect sklearn models without imputation
        has_imputer = _has_symbol({"SimpleImputer", "KNNImputer", "IterativeImputer"})
        has_nan_robust_model = _has_symbol({"HistGradientBoostingClassifier", "HistGradientBoostingRegressor"})
        has_dropna = ".dropna(" in code

        # If using models that DON'T handle NaN natively and NO imputation/dropna found
        if not (has_imputer or has_nan_robust_model or has_dropna):
            issues.append("IMPUTER_REQUIRED")

    # CRITICAL: Check for dialect loading from cleaning_manifest.json
    # ML Engineer MUST read output_dialect before loading data
    has_load_dialect_function = "def load_dialect" in code
    has_manifest_read = "cleaning_manifest.json" in code and ("json.load" in code or "pd.read_json" in code)
    has_dialect_extraction = "output_dialect" in code and "manifest" in code_lower

    # Check if pd.read_csv is called with dialect parameters
    has_read_csv_with_dialect = False
    if "pd.read_csv" in code or "pandas.read_csv" in code:
        # Check if sep= and decimal= are used (not just hardcoded defaults)
        read_csv_calls = code.split("pd.read_csv")
        for call in read_csv_calls[1:]:  # Skip the first split (before first read_csv)
            # Look for sep= and decimal= in the next 200 characters
            snippet = call[:200]
            has_sep = "sep=" in snippet and "sep=sep" in snippet.replace(" ", "")
            has_decimal = "decimal=" in snippet and "decimal=decimal" in snippet.replace(" ", "")
            if has_sep or has_decimal:
                has_read_csv_with_dialect = True
                break

    # If data loading exists but no dialect loading, flag it
    if ("pd.read_csv" in code or "pandas.read_csv" in code):
        if not (has_load_dialect_function or (has_manifest_read and has_dialect_extraction) or has_read_csv_with_dialect):
            issues.append("DIALECT_LOADING_MISSING")

    if allowed_columns:
        unknown_cols = _detect_unknown_columns(code, allowed_columns, allowed_patterns)
        if unknown_cols:
            issues.append("UNKNOWN_COLUMNS_REFERENCED")
        forbidden_assignments = _detect_forbidden_df_assignments(code, allowed_columns, allowed_patterns)
        if forbidden_assignments:
            issues.append("DF_COLUMN_ASSIGNMENT_FORBIDDEN")

    if _detect_synthetic_data(code):
        issues.append("SYNTHETIC_DATA_DETECTED")
    if _detect_dataframe_literal_overwrite(code):
        issues.append("DATAFRAME_LITERAL_OVERWRITE")
    if _scored_rows_has_delta():
        issues.append("SCORED_ROWS_SCHEMA_VIOLATION")

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
    import ast

    if not code:
        return False
    try:
        tree = ast.parse(code)
    except Exception:
        return False

    if _detect_dataframe_literal_overwrite(code):
        return True

    def _call_name(call_node: ast.Call) -> str:
        try:
            return ast.unparse(call_node.func)
        except Exception:
            if isinstance(call_node.func, ast.Name):
                return call_node.func.id
            if isinstance(call_node.func, ast.Attribute):
                return call_node.func.attr
        return ""

    def _is_random_call(call_node: ast.Call) -> bool:
        name = _call_name(call_node).lower()
        if name.startswith("np.random.") or name.startswith("numpy.random."):
            # Resampling indices for bootstrap/CV is NOT synthetic data generation.
            allowed_suffixes = (
                ".choice",
                ".randint",
                ".permutation",
                ".shuffle",
                ".default_rng",
                ".seed",
            )
            if any(name.endswith(suffix) for suffix in allowed_suffixes):
                return False
            return True
        if name.startswith("random.") or name == "random":
            return True
        return False

    def _is_faker_call(call_node: ast.Call) -> bool:
        name = _call_name(call_node).lower()
        return name.endswith("faker") or ".faker" in name

    def _is_sklearn_make_call(call_node: ast.Call) -> bool:
        name = _call_name(call_node).lower()
        make_names = {
            "make_classification",
            "make_regression",
            "make_blobs",
            "make_moons",
            "make_circles",
        }
        if any(name.endswith(item) for item in make_names):
            return True
        if "sklearn.datasets.make_" in name or ".datasets.make_" in name:
            return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _is_random_call(node) or _is_faker_call(node) or _is_sklearn_make_call(node):
                return True
    return False

def _detect_dataframe_literal_overwrite(code: str) -> bool:
    import ast

    if not code:
        return False
    try:
        tree = ast.parse(code)
    except Exception:
        return False

    target_names = {"df", "data", "dataset", "cleaned_df", "scored_df"}

    def _is_dataframe_literal(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        is_df = False
        if isinstance(func, ast.Attribute) and func.attr == "DataFrame":
            if isinstance(func.value, ast.Name) and func.value.id in {"pd", "pandas"}:
                is_df = True
        if isinstance(func, ast.Name) and func.id == "DataFrame":
            is_df = True
        if not is_df:
            return False
        if node.args:
            arg0 = node.args[0]
            if isinstance(arg0, (ast.Dict, ast.List, ast.Tuple)):
                return True
        for kw in node.keywords:
            if kw.arg == "data" and isinstance(kw.value, (ast.Dict, ast.List, ast.Tuple)):
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if hasattr(node, "targets") else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and tgt.id in target_names:
                    if _is_dataframe_literal(node.value):
                        return True
    return False

def _extract_named_string_list(code: str, names: List[str]) -> List[str]:
    import ast

    if not code or not names:
        return []
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                continue
            if isinstance(node.value, ast.List):
                values: List[str] = []
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        values.append(elt.value)
                return values
    return []

def _extract_dataframe_literal_columns(code: str) -> List[str]:
    import ast

    if not code:
        return []
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    cols: set[str] = set()

    def _collect_dict(node: ast.Dict) -> None:
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                cols.add(key.value)

    def _is_dataframe_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "DataFrame":
            if isinstance(func.value, ast.Name) and func.value.id in {"pd", "pandas"}:
                return True
        if isinstance(func, ast.Name) and func.id == "DataFrame":
            return True
        return False

    for node in ast.walk(tree):
        if not _is_dataframe_call(node):
            continue
        if node.args:
            arg0 = node.args[0]
            if isinstance(arg0, ast.Dict):
                _collect_dict(arg0)
            elif isinstance(arg0, (ast.List, ast.Tuple)):
                for elt in arg0.elts:
                    if isinstance(elt, ast.Dict):
                        _collect_dict(elt)
        for kw in node.keywords:
            if kw.arg != "data":
                continue
            value = kw.value
            if isinstance(value, ast.Dict):
                _collect_dict(value)
            elif isinstance(value, (ast.List, ast.Tuple)):
                for elt in value.elts:
                    if isinstance(elt, ast.Dict):
                        _collect_dict(elt)

    return sorted(cols)

def _extract_column_references(code: str) -> List[str]:
    import ast

    if not code:
        return []
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    list_vars: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if isinstance(node.value, (ast.List, ast.Tuple)):
                values: List[str] = []
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        values.append(elt.value)
                if values:
                    list_vars[target.id] = values

    refs: set[str] = set()

    def _collect_from_slice(slice_node: ast.AST) -> None:
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            refs.add(slice_node.value)
            return
        if isinstance(slice_node, ast.Name) and slice_node.id in list_vars:
            refs.update(list_vars[slice_node.id])
            return
        if isinstance(slice_node, (ast.List, ast.Tuple)):
            for elt in slice_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    refs.add(elt.value)
                elif isinstance(elt, ast.Name) and elt.id in list_vars:
                    refs.update(list_vars[elt.id])

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        target = node.value
        if isinstance(target, ast.Attribute) and target.attr == "loc":
            if not (isinstance(target.value, ast.Name) and target.value.id == "df"):
                continue
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) >= 2:
                _collect_from_slice(slice_node.elts[1])
            continue
        if isinstance(target, ast.Attribute) and target.attr == "iloc":
            continue
        if isinstance(target, ast.Name) and target.id == "df":
            _collect_from_slice(node.slice)

    return sorted(refs)

def _detect_unknown_columns(
    code: str,
    allowed_columns: List[str],
    allowed_patterns: List[str] | None = None,
) -> List[str]:
    if not code or not allowed_columns:
        return []
    import re

    allowed_pattern_list = [str(pat) for pat in (allowed_patterns or []) if isinstance(pat, str) and pat.strip()]

    def _pattern_name(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).lower()).strip("_")

    allowed_norm = {_norm_name(col) for col in allowed_columns if col}
    refs = _extract_column_references(code)
    unknown: set[str] = set()
    for col in refs:
        if not col:
            continue
        norm = _norm_name(col)
        if norm in allowed_norm:
            continue
        if allowed_pattern_list:
            target = _pattern_name(col)
            matched = False
            for pattern in allowed_pattern_list:
                try:
                    if re.search(pattern, target):
                        matched = True
                        break
                except re.error:
                    continue
            if matched:
                continue
        unknown.add(col)
    return sorted(unknown)

def _detect_forbidden_df_assignments(
    code: str,
    allowed_columns: List[str],
    allowed_patterns: List[str] | None = None,
) -> List[str]:
    if not code or not allowed_columns:
        return []
    import ast

    try:
        tree = ast.parse(code)
    except Exception:
        return []

    allowed_norm = {_norm_name(col) for col in allowed_columns if col}
    list_vars: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if isinstance(node.value, (ast.List, ast.Tuple)):
                values: List[str] = []
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        values.append(elt.value)
                if values:
                    list_vars[target.id] = values

    def _collect_from_slice(slice_node: ast.AST) -> List[str]:
        cols: List[str] = []
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            cols.append(slice_node.value)
            return cols
        if isinstance(slice_node, ast.Name) and slice_node.id in list_vars:
            cols.extend(list_vars[slice_node.id])
            return cols
        if isinstance(slice_node, (ast.List, ast.Tuple)):
            for elt in slice_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    cols.append(elt.value)
                elif isinstance(elt, ast.Name) and elt.id in list_vars:
                    cols.extend(list_vars[elt.id])
        return cols

    assigned_cols: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if not isinstance(tgt, ast.Subscript):
                    continue
                target = tgt.value
                if isinstance(target, ast.Name) and target.id == "df":
                    assigned_cols.update(_collect_from_slice(tgt.slice))
                elif isinstance(target, ast.Attribute) and target.attr == "loc":
                    if isinstance(target.value, ast.Name) and target.value.id == "df":
                        slice_node = tgt.slice
                        if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) >= 2:
                            assigned_cols.update(_collect_from_slice(slice_node.elts[1]))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "assign":
                if isinstance(func.value, ast.Name) and func.value.id == "df":
                    for kw in node.keywords:
                        if kw.arg:
                            assigned_cols.add(kw.arg)

    forbidden: set[str] = set()
    for col in assigned_cols:
        if not col:
            continue
        norm = _norm_name(col)
        if norm in allowed_norm:
            continue
        forbidden.add(col)

    return sorted(forbidden)

def _find_canonical_mismatches(features: List[str], required_columns: List[str]) -> List[Dict[str, str]]:
    mismatches: List[Dict[str, str]] = []
    if not features or not required_columns:
        return mismatches
    canonical_map = {_norm_name(col): col for col in required_columns if col}
    for feat in features:
        canonical = canonical_map.get(_norm_name(feat))
        if canonical and feat != canonical:
            mismatches.append({"feature": feat, "canonical": canonical})
    return mismatches

def _code_assigns_df_column(code: str, column_name: str) -> bool:
    import ast

    if not code or not column_name:
        return False
    try:
        tree = ast.parse(code)
    except Exception:
        return False

    def _slice_has_column(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and node.value == column_name:
            return True
        if isinstance(node, ast.Tuple):
            return any(_slice_has_column(elt) for elt in node.elts)
        return False

    def _is_df_target(node: ast.Subscript) -> bool:
        value = node.value
        if isinstance(value, ast.Name) and value.id == "df":
            return True
        if isinstance(value, ast.Attribute):
            if isinstance(value.value, ast.Name) and value.value.id == "df" and value.attr in {"loc", "iloc"}:
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Subscript) and _is_df_target(tgt):
                    if _slice_has_column(tgt.slice):
                        return True
    return False

def _has_derived_target_guard(code_lower: str, column_name: str) -> bool:
    if not code_lower or not column_name:
        return False
    target_lower = str(column_name).lower()
    if target_lower not in code_lower:
        return False
    return "not in df.columns" in code_lower

def _collect_derived_targets(contract: Dict[str, Any]) -> List[str]:
    derived_targets: List[str] = []
    if not isinstance(contract, dict):
        return derived_targets
    reqs = contract.get("data_requirements", []) or []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        if str(req.get("role", "")).lower() != "target":
            continue
        if str(req.get("source", "input")).lower() != "derived":
            continue
        name = req.get("canonical_name") or req.get("name")
        if name and name not in derived_targets:
            derived_targets.append(name)
    # V4.1 Priority
    # V4.1 Priority
    fep = contract.get("feature_engineering_plan")
    if not isinstance(fep, dict):
        fep = {}
    derived_cols = fep.get("derived_columns") or []

    spec = contract.get("spec_extraction")
    if not isinstance(spec, dict):
        spec = {}

    if not derived_cols:
        derived_cols = spec.get("derived_columns") or []
    derived_names: List[str] = []
    for col in derived_cols:
        if isinstance(col, dict):
            name = col.get("name") or col.get("canonical_name")
        elif isinstance(col, str):
            name = col
        else:
            name = None
        if name and name not in derived_names:
            derived_names.append(name)
    target_name = spec.get("target_column")
    if target_name and target_name in derived_names and target_name not in derived_targets:
        derived_targets.append(target_name)
    return derived_targets

def _missing_required_output_refs(code: str, outputs: List[str]) -> List[str]:
    if not code or not outputs:
        return []
    missing: List[str] = []
    for output in outputs:
        if not output:
            continue
        if any(ch in output for ch in ["*", "?", "["]):
            continue
        if output.endswith(("/", "\\")):
            continue
        if output not in code:
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
    if "cannot convert the series to <class 'int'>" in lower or "cannot convert the series to <class 'float'>" in lower:
        return "Attempted int()/float() cast on a pandas Series; compute scalar first (e.g., mask.sum())."
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
    if "cannot convert the series to <class 'int'>" in lower or "cannot convert the series to <class 'float'>" in lower:
        lines.append(
            "You are calling int()/float() on a pandas Series. For boolean masks use int(mask.sum())."
        )
        lines.append(
            "Avoid int(...).sum(); it casts before summing. Correct: int((...).sum())."
        )
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

def _detect_alternate_csv_sep(csv_path: str, encoding: str, current_sep: str) -> str | None:
    try:
        import csv
        with open(csv_path, "r", encoding=encoding, errors="replace") as f:
            header_line = f.readline()
        if not header_line:
            return None
        candidates = [",", ";", "\t", "|"]
        best_sep = None
        best_cols = 1
        for sep in candidates:
            if sep == current_sep:
                continue
            try:
                row = next(csv.reader([header_line], delimiter=sep), [])
            except Exception:
                continue
            if len(row) > best_cols:
                best_cols = len(row)
                best_sep = sep
        if best_cols >= 2:
            return best_sep
    except Exception:
        return None
    return None

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

def _load_json_any(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _resolve_artifact_gate_dialect(state: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, str]:
    csv_sep = state.get("csv_sep") or None
    csv_decimal = state.get("csv_decimal") or None
    csv_encoding = state.get("csv_encoding") or None
    if (not csv_sep or not csv_decimal or not csv_encoding) and os.path.exists("data/cleaning_manifest.json"):
        manifest = _load_json_safe("data/cleaning_manifest.json")
        output_dialect = manifest.get("output_dialect") if isinstance(manifest, dict) else None
        if isinstance(output_dialect, dict):
            if not csv_sep:
                csv_sep = output_dialect.get("sep")
            if not csv_decimal:
                csv_decimal = output_dialect.get("decimal")
            if not csv_encoding:
                csv_encoding = output_dialect.get("encoding")
            if csv_sep and not state.get("csv_sep"):
                state["csv_sep"] = csv_sep
            if csv_decimal and not state.get("csv_decimal"):
                state["csv_decimal"] = csv_decimal
            if csv_encoding and not state.get("csv_encoding"):
                state["csv_encoding"] = csv_encoding
    contract_output_dialect = contract.get("output_dialect", {}) if isinstance(contract, dict) else {}
    return {
        "sep": csv_sep or contract_output_dialect.get("sep") or ",",
        "decimal": csv_decimal or contract_output_dialect.get("decimal") or ".",
        "encoding": csv_encoding or contract_output_dialect.get("encoding") or "utf-8",
    }

def _select_segment_column(columns: List[str], contract: Dict[str, Any] | None = None) -> str | None:
    if not columns:
        return None
    label_hint = None
    if isinstance(contract, dict):
        label_hint = contract.get("segment_label_column")
    if label_hint and label_hint in columns:
        return label_hint
    for candidate in [
        "cluster_id",
        "segment_id",
        "segment",
        "cluster",
        "group_id",
        "group",
        "client_segment",
    ]:
        if candidate in columns:
            return candidate
    for col in columns:
        norm = _norm_name(col)
        if any(tok in norm for tok in ["segment", "cluster", "group"]):
            return col
    return None

def _summarize_segmentation_stats(
    scored_path: str,
    csv_sep: str,
    csv_decimal: str,
    csv_encoding: str,
    contract: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    stats = {
        "n_rows": None,
        "n_segments": None,
        "min_segment_size": None,
        "median_segment_size": None,
        "segment_column": None,
    }
    if not scored_path or not os.path.exists(scored_path):
        return stats
    detected_sep = csv_sep
    columns = _read_csv_header(scored_path, csv_encoding, detected_sep)
    if len(columns) <= 1:
        alt_sep = _detect_alternate_csv_sep(scored_path, csv_encoding, detected_sep)
        if alt_sep:
            detected_sep = alt_sep
            columns = _read_csv_header(scored_path, csv_encoding, detected_sep)
            stats["scored_rows_sep_used"] = detected_sep
    segment_col = _select_segment_column(columns, contract)
    stats["segment_column"] = segment_col
    try:
        stats["n_rows"] = _count_raw_rows(scored_path, csv_encoding, detected_sep, csv_decimal)
    except Exception:
        stats["n_rows"] = None
    if not segment_col:
        return stats
    try:
        import pandas as pd
        df = pd.read_csv(
            scored_path,
            sep=detected_sep,
            decimal=csv_decimal,
            encoding=csv_encoding,
            usecols=[segment_col],
            low_memory=False,
        )
        series = df[segment_col].dropna()
        if not series.empty:
            stats["n_segments"] = int(series.nunique())
            sizes = series.value_counts()
            if not sizes.empty:
                stats["min_segment_size"] = int(sizes.min())
                stats["median_segment_size"] = float(sizes.median())
    except Exception:
        return stats
    return stats

def _count_curve_points(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            return len(payload)
        return sum(_count_curve_points(item) for item in payload)
    if isinstance(payload, dict):
        if "points" in payload and isinstance(payload.get("points"), list):
            return len(payload.get("points"))
        total = 0
        for value in payload.values():
            total += _count_curve_points(value)
        return total
    return 0

def _count_curves(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, dict):
        container = payload.get("curves") if isinstance(payload.get("curves"), (dict, list)) else payload
        if isinstance(container, dict):
            return len(container)
        if isinstance(container, list):
            return len(container)
    if isinstance(payload, list):
        return len(payload)
    return 0

def _summarize_pricing_artifacts(
    curves_path: str,
    guide_path: str,
    csv_sep: str,
    csv_decimal: str,
    csv_encoding: str,
) -> Dict[str, Any]:
    summary = {
        "curves_present": None,
        "curves_count": 0,
        "curves_points": 0,
        "guide_rows": None,
        "pct_nan_optimal_price": None,
        "optimal_price_columns": [],
    }
    if curves_path and os.path.exists(curves_path):
        payload = _load_json_any(curves_path)
        summary["curves_count"] = _count_curves(payload)
        summary["curves_points"] = _count_curve_points(payload)
        summary["curves_present"] = bool(summary["curves_count"] or summary["curves_points"])
    if guide_path and os.path.exists(guide_path):
        try:
            import pandas as pd
            header = _read_csv_header(guide_path, csv_encoding, csv_sep)
            target_cols = [c for c in header if "optimal_price" in _norm_name(c)]
            summary["optimal_price_columns"] = target_cols
            df = pd.read_csv(guide_path, sep=csv_sep, decimal=csv_decimal, encoding=csv_encoding)
            summary["guide_rows"] = int(len(df))
            if target_cols:
                sub = df[target_cols]
                nan_frac = float(sub.isna().mean().mean()) if not sub.empty else 1.0
                summary["pct_nan_optimal_price"] = round(nan_frac, 4)
        except Exception:
            return summary
    return summary

def _collect_iteration_diagnostics(state: Dict[str, Any]) -> Dict[str, Any]:
    csv_sep = state.get("csv_sep", ",")
    csv_decimal = state.get("csv_decimal", ".")
    csv_encoding = state.get("csv_encoding", "utf-8")
    contract = state.get("execution_contract", {}) if isinstance(state, dict) else {}
    scored_path = "data/scored_rows.csv"
    seg_stats = _summarize_segmentation_stats(scored_path, csv_sep, csv_decimal, csv_encoding, contract)
    n_rows = seg_stats.get("n_rows")
    if n_rows is None:
        fallback_path = "data/cleaned_full.csv" if os.path.exists("data/cleaned_full.csv") else "data/cleaned_data.csv"
        if os.path.exists(fallback_path):
            try:
                n_rows = _count_raw_rows(fallback_path, csv_encoding, csv_sep, csv_decimal)
                seg_stats["n_rows"] = n_rows
            except Exception:
                pass
    pricing_stats = _summarize_pricing_artifacts(
        curves_path="data/price_sensitivity_curves.json",
        guide_path="data/optimal_pricing_guide.csv",
        csv_sep=csv_sep,
        csv_decimal=csv_decimal,
        csv_encoding=csv_encoding,
    )
    diagnostics = {}
    diagnostics.update(seg_stats)
    diagnostics.update(pricing_stats)
    diagnostics["curves_generated"] = bool(pricing_stats.get("curves_points") or 0)
    runtime_tail = state.get("last_runtime_error_tail")
    if runtime_tail:
        diagnostics["root_cause"] = str(runtime_tail).strip().splitlines()[-1][:300]
    return diagnostics

def _validate_artifact_content(state: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
    diagnostics = _collect_iteration_diagnostics(state)
    issues: List[str] = []
    curves_path = "data/price_sensitivity_curves.json"
    guide_path = "data/optimal_pricing_guide.csv"
    if curves_path and os.path.exists(curves_path):
        if diagnostics.get("curves_points", 0) == 0:
            issues.append("price_sensitivity_curves_empty")
    if guide_path and os.path.exists(guide_path):
        if diagnostics.get("guide_rows") == 0:
            issues.append("optimal_pricing_guide_empty")
    pct_nan_opt = diagnostics.get("pct_nan_optimal_price")
    if isinstance(pct_nan_opt, (int, float)) and pct_nan_opt >= 0.99:
        issues.append("optimal_price_all_nan")
    n_rows = diagnostics.get("n_rows")
    n_segments = diagnostics.get("n_segments")
    if isinstance(n_rows, (int, float)) and isinstance(n_segments, (int, float)) and n_rows > 0:
        if n_segments > 0.5 * n_rows:
            issues.append("segmentation_degenerate")
    return issues, diagnostics

def _score_attempt(
    outputs_valid: bool,
    output_contract_report: Dict[str, Any],
    content_issues: List[str],
    artifact_paths: List[str],
) -> float:
    score = 0.0
    present = output_contract_report.get("present", []) if isinstance(output_contract_report, dict) else []
    missing = output_contract_report.get("missing", []) if isinstance(output_contract_report, dict) else []
    if outputs_valid:
        score += 10.0
    score += float(len(present))
    score += float(len(artifact_paths)) * 0.25
    score -= float(len(missing)) * 2.0
    score -= float(len(content_issues)) * 3.0
    return score

def _snapshot_best_attempt(
    attempt_id: int,
    artifact_paths: List[str],
    output_contract_report: Dict[str, Any],
    artifact_index: List[Dict[str, Any]],
    execution_output: str,
    plots_local: List[str],
    diagnostics: Dict[str, Any] | None = None,
    dest_root: str = os.path.join("artifacts", "best_attempt"),
) -> str | None:
    if attempt_id < 1:
        return None
    try:
        if os.path.isdir(dest_root):
            shutil.rmtree(dest_root)
        os.makedirs(dest_root, exist_ok=True)
        for path in artifact_paths or []:
            if not path or not os.path.exists(path):
                continue
            rel = path.lstrip("./").replace("\\", "/")
            dest = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(path, dest)
        meta = {
            "attempt_id": attempt_id,
            "artifact_index": artifact_index,
            "output_contract_report": output_contract_report,
            "execution_output": execution_output,
            "plots_local": plots_local,
            "diagnostics": diagnostics or {},
        }
        with open(os.path.join(dest_root, "best_attempt.json"), "w", encoding="utf-8") as f_meta:
            json.dump(meta, f_meta, indent=2, ensure_ascii=False)
        return dest_root
    except Exception:
        return None

def _promote_best_attempt(state: Dict[str, Any]) -> Dict[str, Any]:
    best_dir = state.get("best_attempt_dir")
    if not best_dir or not os.path.isdir(best_dir):
        return {}
    updated: Dict[str, Any] = {}
    try:
        for root, _, files in os.walk(best_dir):
            for name in files:
                if name == "best_attempt.json":
                    continue
                src = os.path.join(root, name)
                rel = os.path.relpath(src, best_dir)
                dest = os.path.join(".", rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
        meta_path = os.path.join(best_dir, "best_attempt.json")
        meta = _load_json_any(meta_path)
        if isinstance(meta, dict):
            if meta.get("artifact_index") is not None:
                updated["artifact_index"] = meta.get("artifact_index")
                updated["produced_artifact_index"] = meta.get("artifact_index")
                try:
                    os.makedirs("data", exist_ok=True)
                    dump_json("data/produced_artifact_index.json", meta.get("artifact_index"))
                except Exception:
                    pass
            if meta.get("output_contract_report") is not None:
                updated["output_contract_report"] = meta.get("output_contract_report")
                try:
                    os.makedirs("data", exist_ok=True)
                    dump_json("data/output_contract_report.json", meta.get("output_contract_report"))
                except Exception:
                    pass
            if meta.get("execution_output"):
                updated["execution_output"] = meta.get("execution_output")
                updated["execution_output_stale"] = True
                updated["execution_error"] = False
                updated["sandbox_failed"] = False
            if meta.get("plots_local") is not None:
                updated["plots_local"] = meta.get("plots_local")
    except Exception:
        return updated
    return updated

def _normalize_alignment_check(
    alignment_check: Dict[str, Any],
    alignment_requirements: List[Any],
) -> tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    normalized = dict(alignment_check or {})
    raw_status = normalized.get("status")
    if not raw_status:
        raw_status = normalized.get("overall_status") or normalized.get("overallStatus")
    status = str(raw_status or "").upper()

    requirements_payload = normalized.get("requirements")
    per_req = normalized.get("per_requirement")
    evidence_map = normalized.get("evidence")
    checks_payload = normalized.get("checks") or normalized.get("checklist") or []
    map_schema = False

    def _requirements_from_checks(checks: Any) -> List[Dict[str, Any]]:
        derived: List[Dict[str, Any]] = []
        if not isinstance(checks, list):
            return derived
        for idx, item in enumerate(checks):
            if not isinstance(item, dict):
                continue
            req_id = item.get("id") or item.get("name") or item.get("key") or f"check_{idx}"
            status_val = item.get("status") or item.get("result") or item.get("outcome")
            evidence_val = item.get("evidence") or item.get("notes") or []
            evidence_list: List[str] = []
            if isinstance(evidence_val, list):
                evidence_list = [str(v) for v in evidence_val if v]
            elif isinstance(evidence_val, str) and evidence_val.strip():
                evidence_list = [evidence_val.strip()]
            derived.append(
                {
                    "id": str(req_id),
                    "status": str(status_val or "").upper(),
                    "evidence": evidence_list,
                }
            )
        return derived

    def _requirements_from_map(payload: Dict[str, Any], req_ids: set[str]) -> List[Dict[str, Any]]:
        derived: List[Dict[str, Any]] = []
        for req_id in req_ids:
            entry = payload.get(req_id)
            if not isinstance(entry, dict):
                continue
            status_val = entry.get("status") or entry.get("result") or entry.get("outcome")
            evidence_val = entry.get("evidence") or entry.get("notes") or []
            evidence_list: List[str] = []
            if isinstance(evidence_val, list):
                evidence_list = [str(v) for v in evidence_val if v]
            elif isinstance(evidence_val, str) and evidence_val.strip():
                evidence_list = [evidence_val.strip()]
            derived.append(
                {
                    "id": str(req_id),
                    "status": str(status_val or "").upper(),
                    "evidence": evidence_list,
                }
            )
        return derived

    def _derive_status_from_requirements(reqs: List[Dict[str, Any]]) -> str:
        if not isinstance(reqs, list) or not reqs:
            return "WARN"
        statuses = [str(item.get("status") or "").upper() for item in reqs if isinstance(item, dict)]
        if any(st == "FAIL" for st in statuses):
            return "FAIL"
        if any(st not in {"PASS"} for st in statuses):
            return "WARN"
        return "PASS"

    if not isinstance(requirements_payload, list) and checks_payload:
        requirements_payload = _requirements_from_checks(checks_payload)
        normalized["requirements"] = requirements_payload
    if not isinstance(requirements_payload, list) and alignment_requirements:
        req_ids = {
            str(req.get("id") or req.get("name") or req.get("key") or req.get("requirement_id"))
            for req in alignment_requirements
            if isinstance(req, dict) and (req.get("id") or req.get("name") or req.get("key") or req.get("requirement_id"))
        }
        matching = [key for key in normalized.keys() if key in req_ids and isinstance(normalized.get(key), dict)]
        if matching:
            map_schema = True
            requirements_payload = _requirements_from_map(normalized, req_ids)
            normalized["requirements"] = requirements_payload
            if status not in {"PASS", "WARN", "FAIL"}:
                status = _derive_status_from_requirements(requirements_payload)
    if not alignment_requirements and isinstance(requirements_payload, list):
        alignment_requirements = [{"id": item.get("id"), "required": True} for item in requirements_payload if item.get("id")]

    normalized_reqs: List[Dict[str, Any]] = []
    missing_status = 0
    missing_evidence = 0
    for req in alignment_requirements or []:
        if isinstance(req, str):
            req_id = req.strip()
            req_obj = {"id": req_id, "required": True}
        elif isinstance(req, dict):
            req_id = req.get("id") or req.get("name") or req.get("key") or req.get("requirement_id")
            req_obj = req
        else:
            continue
        if not req_id:
            continue
        if isinstance(req_obj, dict) and req_obj.get("required") is False:
            normalized_reqs.append({"id": req_id, "status": "SKIP", "evidence": []})
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
            if map_schema:
                req_status = "WARN"
            else:
                missing_status += 1
                req_status = "MISSING"
        if not req_evidence:
            if not map_schema:
                missing_evidence += 1
        normalized_reqs.append(
            {"id": req_id, "status": req_status, "evidence": req_evidence}
        )

    if missing_status:
        issues.append("alignment_missing_requirement_status")
    if missing_evidence:
        issues.append("alignment_missing_evidence")

    if status not in {"PASS", "WARN", "FAIL"}:
        status = "WARN"
        issues.append("alignment_status_invalid")

    failure_mode = normalized.get("failure_mode")
    if not failure_mode and issues:
        failure_mode = "format"

    normalized["status"] = status
    normalized["failure_mode"] = failure_mode
    normalized["requirements"] = normalized_reqs
    if issues:
        summary = normalized.get("summary") or ""
        issue_text = ", ".join(sorted(set(issues)))
        normalized["summary"] = f"{summary} Alignment issues: {issue_text}".strip()
    return normalized, issues


def _resolve_forbidden_features(
    contract: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
) -> List[str]:
    forbidden: set[str] = set()

    def _coerce_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value if item]
        if isinstance(value, str):
            return [value]
        return []

    for source in (contract_min, contract):
        if not isinstance(source, dict):
            continue
        allowed = source.get("allowed_feature_sets") or {}
        if isinstance(allowed, dict):
            forbidden |= set(_coerce_list(allowed.get("forbidden_features")))
            forbidden |= set(_coerce_list(allowed.get("forbidden_for_modeling")))
            forbidden |= set(_coerce_list(allowed.get("audit_only_features")))
        if forbidden:
            break

    if not forbidden:
        roles = get_column_roles(contract_min or contract or {})
        if isinstance(roles, dict):
            forbidden |= set(_coerce_list(roles.get("outcome")))
            forbidden |= set(_coerce_list(roles.get("post_decision_audit_only")))

    return sorted({item for item in forbidden if item})


def _ensure_feature_usage(
    alignment_check: Dict[str, Any],
    code: str,
    contract: Dict[str, Any] | None,
) -> Dict[str, Any]:
    alignment_check = dict(alignment_check or {})
    feature_usage = alignment_check.get("feature_usage")
    if not isinstance(feature_usage, dict):
        feature_usage = {}
    used_features = feature_usage.get("used_features")
    if not isinstance(used_features, list) or not used_features:
        model_features = _extract_named_string_list(code or "", ["MODEL_FEATURES", "model_features"])
        segment_features = _extract_named_string_list(code or "", ["SEGMENT_FEATURES", "segment_features"])
        used_features = list(dict.fromkeys((model_features or []) + (segment_features or [])))
    def _coerce_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value if item]
        if isinstance(value, str):
            return [value]
        return []

    target_columns: List[str] = []
    if isinstance(contract, dict):
        target_columns.extend(_coerce_list(contract.get("outcome_columns") or []))
        target_columns.extend(_coerce_list(contract.get("decision_columns") or []))
    feature_usage.setdefault("excluded_features", [])
    feature_usage.setdefault("reason_exclusions", {})
    feature_usage["used_features"] = used_features or []
    feature_usage["target_columns"] = [col for col in target_columns if col]
    alignment_check["feature_usage"] = feature_usage
    return alignment_check


def _apply_forbidden_feature_gate(
    alignment_check: Dict[str, Any],
    contract: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    code: str,
) -> tuple[Dict[str, Any], List[str]]:
    alignment_check = _ensure_feature_usage(alignment_check, code, contract)
    forbidden = _resolve_forbidden_features(contract, contract_min)
    usage = alignment_check.get("feature_usage") if isinstance(alignment_check.get("feature_usage"), dict) else {}
    used_features = usage.get("used_features") if isinstance(usage, dict) else []
    used_set = {str(item) for item in used_features if item}
    violations = sorted({item for item in forbidden if item in used_set})
    requirements = alignment_check.get("requirements")
    if not isinstance(requirements, list):
        requirements = []
    if violations:
        alignment_check["status"] = "FAIL"
        alignment_check["failure_mode"] = alignment_check.get("failure_mode") or "leakage"
        alignment_check["summary"] = (
            alignment_check.get("summary")
            or "Forbidden features detected in model usage."
        )
        alignment_check["forbidden_features_used"] = violations
        requirements.append(
            {
                "id": "forbidden_features",
                "status": "FAIL",
                "evidence": [f"forbidden_used={violations}"],
            }
        )
    elif used_set:
        requirements.append(
            {
                "id": "forbidden_features",
                "status": "PASS",
                "evidence": ["no_forbidden_features_used"],
            }
        )
    else:
        requirements.append(
            {
                "id": "feature_usage",
                "status": "WARN",
                "evidence": ["feature_usage_missing_or_empty"],
            }
        )
    alignment_check["requirements"] = requirements
    return alignment_check, violations

def _coerce_alignment_requirements(reqs: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(reqs, list):
        return out
    for idx, req in enumerate(reqs):
        if isinstance(req, str) and req.strip():
            out.append({"id": req.strip(), "required": True})
            continue
        if isinstance(req, dict):
            req_id = req.get("id") or req.get("name") or req.get("key") or f"custom_{idx}"
            req_obj = dict(req)
            req_obj["id"] = req_id
            if "required" not in req_obj:
                req_obj["required"] = True
            out.append(req_obj)
    return out

def _extract_leakage_columns(payload: Any) -> List[str]:
    cols: List[str] = []
    if isinstance(payload, dict):
        for key in ["columns", "features", "numeric_columns", "leaky_columns"]:
            vals = payload.get(key)
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str):
                        cols.append(v)
        pairs = payload.get("pairs") or payload.get("relations") or payload.get("top_pairs")
        if isinstance(pairs, list):
            for item in pairs:
                if not isinstance(item, dict):
                    continue
                for key in ["column", "feature", "name", "col_a", "col_b", "x", "y"]:
                    val = item.get(key)
                    if isinstance(val, str):
                        cols.append(val)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                cols.append(item)
            elif isinstance(item, dict):
                for key in ["column", "feature", "name", "col_a", "col_b", "x", "y"]:
                    val = item.get(key)
                    if isinstance(val, str):
                        cols.append(val)
    return cols

def _artifact_alignment_gate(
    cleaned_path: str,
    scored_path: str,
    contract: Dict[str, Any],
    evaluation_spec: Dict[str, Any] | None,
    csv_sep: str,
    csv_decimal: str,
    csv_encoding: str,
) -> List[str]:
    issues: List[str] = []
    if not cleaned_path or not os.path.exists(cleaned_path):
        issues.append(f"cleaned_data_missing:{cleaned_path or 'data/cleaned_data.csv'}")
        return issues

    try:
        import pandas as pd
        cleaned_df = pd.read_csv(cleaned_path, sep=csv_sep, decimal=csv_decimal, encoding=csv_encoding)
    except Exception as err:
        issues.append(f"cleaned_data_read_failed:{err}")
        return issues

    flags = _resolve_eval_flags(evaluation_spec)
    requires_row_scoring = bool(flags.get("requires_row_scoring"))
    required_outputs = contract.get("required_outputs", []) if isinstance(contract, dict) else []
    if "data/scored_rows.csv" in (required_outputs or []):
        requires_row_scoring = True
    deliverables = _resolve_contract_deliverables(contract) if isinstance(contract, dict) else []
    if isinstance(deliverables, list):
        for item in deliverables:
            if isinstance(item, dict) and item.get("path") == "data/scored_rows.csv":
                if bool(item.get("required")):
                    requires_row_scoring = True
                break
            if isinstance(item, str) and item == "data/scored_rows.csv":
                requires_row_scoring = True
                break

    if not requires_row_scoring:
        return issues

    if not scored_path or not os.path.exists(scored_path):
        issues.append("scored_rows_missing")
        return issues

    try:
        import pandas as pd
        scored_sep = csv_sep
        scored_df = pd.read_csv(scored_path, sep=scored_sep, decimal=csv_decimal, encoding=csv_encoding)
        if scored_df.shape[1] == 1:
            alt_sep = _detect_alternate_csv_sep(scored_path, csv_encoding, scored_sep)
            if alt_sep:
                scored_sep = alt_sep
                scored_df = pd.read_csv(scored_path, sep=scored_sep, decimal=csv_decimal, encoding=csv_encoding)
    except Exception as err:
        issues.append(f"scored_rows_read_failed:{err}")
        return issues

    if scored_df.shape[1] == 1:
        issues.append("scored_rows_dialect_mismatch")
        return issues

    if len(scored_df) != len(cleaned_df):
        issues.append("scored_rows_rowcount_mismatch")

    cleaned_cols = [str(c) for c in cleaned_df.columns]
    scored_cols = [str(c) for c in scored_df.columns]
    cleaned_norm = {_norm_name(c) for c in cleaned_cols}
    scored_norm = {_norm_name(c) for c in scored_cols}
    row_id_candidates = {"row_id", "case_id", "caseid", "id"}
    has_row_id = any(c in row_id_candidates for c in scored_norm)
    has_overlap = bool(cleaned_norm.intersection(scored_norm))
    if not has_row_id and not has_overlap:
        issues.append("scored_rows_missing_row_id_or_overlap")

    def _normalize_schema(raw: Any) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if not key:
                    continue
                normalized[str(key)] = value if isinstance(value, dict) else {}
            return normalized
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                path = item.get("path") or item.get("artifact") or item.get("output")
                if not path:
                    continue
                normalized[str(path)] = item
        return normalized

    schema = {}
    if isinstance(contract, dict):
        # V4.1 Priority first
        reqs = contract.get("artifact_requirements") or {}
        schema = _normalize_schema(reqs.get("file_schemas"))
        if not schema:
             schema = _normalize_schema(contract.get("artifact_schemas"))
        if not schema:
            spec = contract.get("spec_extraction") or {}
            schema = _normalize_schema(spec.get("artifact_schemas"))
    scored_schema = schema.get("data/scored_rows.csv") if isinstance(schema, dict) else {}
    allowed_extra = scored_schema.get("allowed_extra_columns") if isinstance(scored_schema, dict) else None
    allowed_patterns = scored_schema.get("allowed_name_patterns") if isinstance(scored_schema, dict) else None

    allowed_cols: List[str] = list(cleaned_cols)
    allowed_cols.extend(_resolve_contract_columns(contract, sources={"derived", "output"}))
    # Legacy spec_extraction removed (covered by _resolve_contract_columns V4.1 + data_requirements)

    if isinstance(evaluation_spec, dict):
        target = evaluation_spec.get("target")
        if isinstance(target, dict) and target.get("name"):
            allowed_cols.append(str(target.get("name")))

    allowed_cols.extend(
        [
            "prediction",
            "pred",
            "predicted",
            "predicted_value",
            "predicted_label",
            "predicted_prob",
            "probability",
            "prob",
            "score",
            "risk_score",
            "rank",
            "ranking",
            "segment",
            "segment_id",
            "cluster",
            "cluster_id",
            "group",
            "group_id",
            "row_id",
            "case_id",
            "caseid",
            "id",
            "expected_value",
            "actual",
            "label",
        ]
    )
    allowed_norm = {_norm_name(col) for col in allowed_cols if col}
    allowed_extra_norm = {_norm_name(col) for col in allowed_extra or [] if col}
    allowed_pattern_list = [str(pat) for pat in (allowed_patterns or []) if isinstance(pat, str) and pat.strip()]

    def _pattern_name(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).lower()).strip("_")

    scored_extras = []
    for col in scored_cols:
        norm = _norm_name(col)
        if not norm:
            continue
        if norm in allowed_norm:
            continue
        if norm in allowed_extra_norm:
            continue
        if allowed_pattern_list:
            pattern_target = _pattern_name(col)
            matched = False
            for pattern in allowed_pattern_list:
                try:
                    if re.search(pattern, pattern_target):
                        matched = True
                        break
                except re.error:
                    continue
            if matched:
                continue
        if any(tok in norm for tok in ["pred", "score", "prob", "rank", "segment", "cluster", "group", "optimal", "recommended", "expected"]):
            continue
        scored_extras.append(col)
    if scored_extras:
        sample = ", ".join(scored_extras[:5])
        issues.append(f"scored_rows_unknown_columns:{sample}")

    leakage_path = os.path.join("analysis", "leakage_report.json")
    if os.path.exists(leakage_path):
        try:
            with open(leakage_path, "r", encoding="utf-8") as f_leak:
                leakage_payload = json.load(f_leak)
            leak_cols = _extract_leakage_columns(leakage_payload)
            if leak_cols:
                numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns.tolist()
                numeric_norm = {_norm_name(c) for c in numeric_cols}
                leak_unknown = [
                    c for c in leak_cols if _norm_name(c) not in numeric_norm
                ]
                if leak_unknown:
                    sample = ", ".join(leak_unknown[:5])
                    issues.append(f"leakage_report_unknown_columns:{sample}")
        except Exception as leak_err:
            issues.append(f"leakage_report_read_failed:{leak_err}")

    align_path = os.path.join("data", "alignment_check.json")
    if os.path.exists(align_path):
        try:
            align_payload = _load_json_safe(align_path)
            if isinstance(align_payload, dict):
                for key in ["row_count", "input_rows", "n_rows", "dataset_rows", "total_rows"]:
                    val = align_payload.get(key)
                    if isinstance(val, int) and val != len(cleaned_df):
                        issues.append("alignment_check_rowcount_mismatch")
                        break
        except Exception:
            pass

    return issues

def _hash_json(payload: Any) -> str | None:
    if not payload:
        return None
    try:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return None

def _hash_file(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
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
    def _pick_column(cols: List[str], tokens: List[str], exclude: List[str] | None = None) -> str | None:
        for col in cols:
            norm = _norm_name(col)
            if exclude and any(tok in norm for tok in exclude):
                continue
            if any(tok in norm for tok in tokens):
                return col
        return None

    cols = list(df.columns)
    score_col = _pick_column(cols, ["score", "pred", "prediction", "prob", "probability", "rank"], ["ref", "target"])
    ref_col = _pick_column(cols, ["ref", "reference", "target", "label", "actual", "rank"], [])
    case_col = _pick_column(cols, ["case", "group", "segment", "bucket", "cluster"], [])
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

def _normalize_metric_key(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

def _is_baseline_metric_name(name: str) -> bool:
    key = _normalize_metric_key(name)
    return any(token in key for token in ["baseline", "dummy", "naive", "null", "default"])

def _objective_metric_priority(objective_type: str) -> List[str]:
    objective = str(objective_type or "unknown").lower()
    if "classif" in objective:
        return ["roc_auc", "auc", "f1", "precision", "recall", "accuracy", "pr_auc"]
    if "regress" in objective or "forecast" in objective:
        return ["rmse", "mae", "mape", "smape", "r2", "mse"]
    if "rank" in objective:
        return ["spearman", "kendall", "ndcg", "map", "mrr", "gini"]
    return ["roc_auc", "f1", "rmse", "mae", "r2", "spearman"]

def _coerce_float(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    return None

def _collect_metric_candidates(metrics_report: Dict[str, Any], weights_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    model_perf = metrics_report.get("model_performance") if isinstance(metrics_report, dict) else {}
    if isinstance(model_perf, dict):
        for key, value in model_perf.items():
            if isinstance(value, dict) and "mean" in value:
                mean = _coerce_float(value.get("mean"))
                if mean is None:
                    continue
                candidates.append(
                    {
                        "name": str(key),
                        "value": mean,
                        "ci_lower": _coerce_float(value.get("ci_lower")),
                        "ci_upper": _coerce_float(value.get("ci_upper")),
                        "is_baseline": _is_baseline_metric_name(str(key)),
                    }
                )
            else:
                num = _coerce_float(value)
                if num is not None:
                    candidates.append(
                        {
                            "name": str(key),
                            "value": num,
                            "ci_lower": None,
                            "ci_upper": None,
                            "is_baseline": _is_baseline_metric_name(str(key)),
                        }
                    )
    if isinstance(weights_report, dict):
        for container in ("metrics", "classification", "regression", "propensity_model", "price_model", "optimization"):
            block = weights_report.get(container)
            if not isinstance(block, dict):
                continue
            for key, value in block.items():
                num = _coerce_float(value)
                if num is None:
                    continue
                candidates.append(
                    {
                        "name": str(key),
                        "value": num,
                        "ci_lower": None,
                        "ci_upper": None,
                        "is_baseline": _is_baseline_metric_name(str(key)),
                    }
                )
    return candidates

def _pick_primary_metric_candidate(
    candidates: List[Dict[str, Any]],
    objective_type: str,
) -> Dict[str, Any] | None:
    if not candidates:
        return None
    priority = _objective_metric_priority(objective_type)
    with_ci = [c for c in candidates if c.get("ci_lower") is not None and c.get("ci_upper") is not None]
    search_pool = with_ci if with_ci else candidates
    for token in priority:
        for cand in search_pool:
            if token in _normalize_metric_key(cand.get("name", "")):
                return cand
    return search_pool[0]

def _resolve_validation_requirements(
    evaluation_spec: Dict[str, Any] | None,
    contract: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if isinstance(evaluation_spec, dict):
        val = evaluation_spec.get("validation_requirements")
        if isinstance(val, dict):
            return val
    if isinstance(contract, dict):
        val = get_validation_requirements(contract)
        if isinstance(val, dict):
            return val
    return {}

def _build_eval_signature(
    objective_type: str | None,
    evaluation_spec: Dict[str, Any] | None,
    contract: Dict[str, Any] | None,
) -> str:
    validation = _resolve_validation_requirements(evaluation_spec, contract)
    method = str(validation.get("method") or "unknown").strip().lower()
    n_folds = validation.get("n_folds") or validation.get("folds")
    n_boot = validation.get("n_bootstrap") or validation.get("n_boot") or validation.get("bootstrap_iterations")
    seed = validation.get("seed") or validation.get("random_state")
    parts = [
        f"objective_type={objective_type or 'unknown'}",
        f"method={method or 'unknown'}",
    ]
    if n_folds is not None:
        parts.append(f"n_folds={n_folds}")
    if n_boot is not None:
        parts.append(f"n_boot={n_boot}")
    if seed is not None:
        parts.append(f"seed={seed}")
    return ";".join(parts)

def _extract_primary_metric_snapshot(
    metrics_report: Dict[str, Any] | None,
    weights_report: Dict[str, Any] | None,
    objective_type: str | None,
    evaluation_spec: Dict[str, Any] | None,
    contract: Dict[str, Any] | None,
) -> Dict[str, Any]:
    metrics_report = metrics_report or {}
    weights_report = weights_report or {}
    candidates = _collect_metric_candidates(metrics_report, weights_report)
    primary = _pick_primary_metric_candidate(candidates, objective_type or "unknown")
    if not primary:
        return {}
    baseline = None
    primary_norm = _normalize_metric_key(primary.get("name", ""))
    for cand in candidates:
        if not cand.get("is_baseline"):
            continue
        cand_norm = _normalize_metric_key(cand.get("name", ""))
        if primary_norm and primary_norm in cand_norm:
            baseline = cand
            break
    if baseline is None:
        baseline = next((cand for cand in candidates if cand.get("is_baseline")), None)
    baseline_value = baseline.get("value") if baseline else None
    primary_value = primary.get("value")
    metric_name = primary.get("name")
    higher_is_better = True
    if metric_name:
        higher_is_better = _metric_higher_is_better(metric_name)
    lift = _calc_lift(baseline_value, primary_value, higher_is_better) if baseline_value is not None else None
    return {
        "primary_metric_name": metric_name,
        "primary_metric_value": primary_value,
        "baseline_metric_name": baseline.get("name") if baseline else None,
        "baseline_value": baseline_value,
        "lift": lift,
        "ci_lower": primary.get("ci_lower"),
        "ci_upper": primary.get("ci_upper"),
        "eval_signature": _build_eval_signature(objective_type, evaluation_spec, contract),
    }

def _detect_metric_plateau(
    metric_history: List[Dict[str, Any]] | None,
    window: int = 2,
    epsilon: float = 0.01,
) -> tuple[bool, str]:
    history = [item for item in (metric_history or []) if isinstance(item, dict)]
    if len(history) < window:
        return False, ""
    last = history[-1]
    signature = last.get("eval_signature") or "unknown"
    metric_name = last.get("primary_metric_name")
    comparable = []
    for item in history:
        if metric_name and item.get("primary_metric_name") != metric_name:
            continue
        if signature != "unknown" and item.get("eval_signature") != signature:
            continue
        comparable.append(item)
    if len(comparable) < window:
        return False, ""
    recent = comparable[-window:]
    lifts = [item.get("lift") for item in recent]
    if all(isinstance(val, (int, float)) for val in lifts):
        if all(val < epsilon for val in lifts):
            return True, f"lift<{epsilon} for {window} iterations"
    values = []
    ci_pairs = []
    for item in recent:
        val = item.get("primary_metric_value")
        if not isinstance(val, (int, float)):
            return False, ""
        values.append(float(val))
        ci_pairs.append((item.get("ci_lower"), item.get("ci_upper")))
    higher_is_better = True
    if metric_name:
        higher_is_better = _metric_higher_is_better(metric_name)
    any_ci = False
    all_ci_overlap = True
    improvements: List[float] = []
    for idx in range(1, len(values)):
        if higher_is_better:
            delta = values[idx] - values[idx - 1]
        else:
            delta = values[idx - 1] - values[idx]
        improvements.append(delta)
        low1, up1 = ci_pairs[idx - 1]
        low2, up2 = ci_pairs[idx]
        if _is_number(low1) and _is_number(up1) and _is_number(low2) and _is_number(up2):
            any_ci = True
            if max(float(low1), float(low2)) > min(float(up1), float(up2)):
                all_ci_overlap = False
    if improvements and all(delta <= epsilon for delta in improvements):
        if not any_ci or all_ci_overlap:
            return True, f"metric_delta<= {epsilon} across {window} iterations"
    return False, ""

def _append_feedback_history(state: Dict[str, Any], message: str) -> None:
    if not message:
        return
    history = list(state.get("feedback_history", []) or [])
    if message not in history:
        history.append(message)
    state["feedback_history"] = history

def _normalize_reason_tags(text: str, failed_gates: List[str] | None = None) -> List[str]:
    tags: List[str] = []
    lower = str(text or "").lower()
    gate_tokens = [str(item).lower() for item in (failed_gates or []) if item]
    combined = " ".join([lower] + gate_tokens)
    mapping = [
        ("synthetic", "synthetic"),
        ("df_column_assignment_forbidden", "df_mutation"),
        ("df_column", "df_mutation"),
        ("unknown_columns_referenced", "unknown_columns"),
        ("unknown column", "unknown_columns"),
        ("baseline", "baseline_missing"),
        ("imputer", "imputer_missing"),
        ("leakage", "leakage"),
        ("output_contract_missing", "contract_missing_outputs"),
        ("required_outputs_missing", "contract_missing_outputs"),
        ("missing output", "contract_missing_outputs"),
        ("time_series", "time_series_split"),
        ("cross_validation_required", "validation_missing"),
        ("validation_required", "validation_missing"),
        ("alignment", "alignment"),
        ("method_choice", "method_choice"),
        ("data_limited", "data_limited"),
        ("variance guard", "target_variance_guard"),
    ]
    for token, tag in mapping:
        if token in combined and tag not in tags:
            tags.append(tag)
    return tags

def _summarize_runtime_error(output: str | None) -> Dict[str, str] | None:
    if not output:
        return None
    text = str(output)
    line = text.strip().splitlines()[-1] if text.strip() else ""
    if "Traceback" in text:
        return {"type": "runtime_error", "message": line[:300]}
    if "EXECUTION ERROR" in text or "Sandbox Execution Failed" in text:
        return {"type": "execution_error", "message": line[:300]}
    return {"type": "error", "message": line[:300]}

def _collect_outputs_state(required_outputs: List[str]) -> Dict[str, List[str]]:
    present: List[str] = []
    missing: List[str] = []
    for path in required_outputs or []:
        if not path or _is_glob_pattern(path):
            continue
        if os.path.exists(path):
            present.append(path)
        else:
            missing.append(path)
    return {"present": present, "missing": missing}

def _suggest_next_actions(
    preflight_issues: List[str],
    outputs_missing: List[str],
    reviewer_reasons: List[str],
    qa_reasons: List[str],
) -> List[str]:
    actions: List[str] = []
    issues = [str(item) for item in (preflight_issues or []) if item]
    reasons = set((reviewer_reasons or []) + (qa_reasons or []))
    if outputs_missing:
        actions.append(f"Write required outputs: {outputs_missing[:5]}")
    if "SYNTHETIC_DATA_DETECTED" in issues or "synthetic" in reasons:
        actions.append("Remove synthetic data generation; load only the provided dataset.")
    if "DF_COLUMN_ASSIGNMENT_FORBIDDEN" in issues or "df_mutation" in reasons:
        actions.append("Avoid df column assignments; use Pipeline/ColumnTransformer or separate artifacts.")
    if "UNKNOWN_COLUMNS_REFERENCED" in issues or "unknown_columns" in reasons:
        actions.append("Use only contract/canonical columns; avoid invented names.")
    if "DIALECT_LOADING_MISSING" in issues:
        actions.append("CRITICAL: Define load_dialect() function and read output_dialect from 'data/cleaning_manifest.json' before ANY data loading.")
    if "BASELINE_REQUIRED" in issues or "baseline_missing" in reasons:
        actions.append("Add a DummyClassifier/DummyRegressor baseline with metrics.")
    if "IMPUTER_REQUIRED" in issues or "imputer_missing" in reasons:
        actions.append("Include SimpleImputer in preprocessing before modeling.")
    if "validation_missing" in reasons:
        actions.append("Add appropriate validation (CV or time-based split) and report metrics.")
    if "leakage" in reasons:
        actions.append("Exclude post-outcome features and document leakage prevention.")
    if "contract_missing_outputs" in reasons:
        actions.append("Verify all required artifacts are written to contract paths.")
    if not actions:
        actions.append("Apply reviewer feedback and align outputs to the execution contract.")
    return actions[:3]

def _ml_iteration_journal_path(run_id: str, base_dir: str = "runs") -> str:
    return os.path.join(base_dir, run_id, "report", "governance", "ml_iteration_journal.jsonl")

def _append_ml_iteration_journal(
    run_id: str,
    entry: Dict[str, Any],
    written_ids: List[str] | None = None,
    base_dir: str = "runs",
) -> List[str]:
    if not run_id or not isinstance(entry, dict):
        return written_ids or []
    iter_id = entry.get("iteration_id")
    stage = entry.get("stage") or "unknown"
    if iter_id is None:
        return written_ids or []
    known: set[str] = set()
    for item in written_ids or []:
        if isinstance(item, str):
            known.add(item)
        elif isinstance(item, int) or str(item).isdigit():
            known.add(f"{int(item)}:unknown")
    entry_key = f"{int(iter_id)}:{stage}"
    if entry_key in known:
        return sorted(known)
    path = _ml_iteration_journal_path(run_id, base_dir=base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except Exception:
        return sorted(known)
    known.add(entry_key)
    return sorted(known)

def _load_ml_iteration_journal(run_id: str, base_dir: str = "runs") -> List[Dict[str, Any]]:
    if not run_id:
        return []
    path = _ml_iteration_journal_path(run_id, base_dir=base_dir)
    if not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return entries

def _build_ml_iteration_memory_block(entries: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    if not entries:
        return ""
    last = entries[-1]
    diagnostics = last.get("iteration_diagnostics") if isinstance(last, dict) else {}
    last_summary = {
        "iteration_id": last.get("iteration_id"),
        "preflight_issues": last.get("preflight_issues"),
        "runtime_error": last.get("runtime_error"),
        "outputs_missing": last.get("outputs_missing"),
        "reviewer_verdict": last.get("reviewer_verdict"),
        "qa_verdict": last.get("qa_verdict"),
        "n_rows": diagnostics.get("n_rows") if isinstance(diagnostics, dict) else None,
        "n_segments": diagnostics.get("n_segments") if isinstance(diagnostics, dict) else None,
        "min_segment_size": diagnostics.get("min_segment_size") if isinstance(diagnostics, dict) else None,
        "median_segment_size": diagnostics.get("median_segment_size") if isinstance(diagnostics, dict) else None,
        "curves_generated": diagnostics.get("curves_generated") if isinstance(diagnostics, dict) else None,
        "curves_points": diagnostics.get("curves_points") if isinstance(diagnostics, dict) else None,
        "guide_rows": diagnostics.get("guide_rows") if isinstance(diagnostics, dict) else None,
        "pct_nan_optimal_price": diagnostics.get("pct_nan_optimal_price") if isinstance(diagnostics, dict) else None,
        "root_cause": diagnostics.get("root_cause") if isinstance(diagnostics, dict) else None,
    }
    counts: Dict[str, int] = {}
    for entry in entries:
        for tag in (entry.get("reviewer_reasons") or []) + (entry.get("qa_reasons") or []) + (entry.get("preflight_issues") or []):
            if not tag:
                continue
            key = str(tag)
            counts[key] = counts.get(key, 0) + 1
    top_failures = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:3]
    top_failures_list = [f"{name} (x{count})" for name, count in top_failures]
    do_lines = []
    for item in last.get("next_actions") or []:
        if not item:
            continue
        do_lines.append(f"DO: {item}")
        if len(do_lines) >= 3:
            break
    dont_lines = []
    dont_map = {
        "synthetic": "Don't generate synthetic data.",
        "df_mutation": "Don't assign new df columns outside allowed patterns.",
        "unknown_columns": "Don't reference columns not in the contract.",
        "contract_missing_outputs": "Don't skip required artifacts.",
        "leakage": "Don't use post-outcome features.",
    }
    for key in counts.keys():
        tag = str(key)
        if tag in dont_map and len(dont_lines) < 3:
            dont_lines.append(dont_map[tag])
    lines = [
        "Last attempt summary: " + json.dumps(last_summary, ensure_ascii=True),
        "Top recurring failures: " + json.dumps(top_failures_list, ensure_ascii=True),
    ]
    lines.extend(do_lines + dont_lines)
    block = "\n".join(lines)
    if len(block) <= max_chars:
        return block
    base_lines = lines[:2]
    trimmed_lines = list(base_lines)
    for extra in lines[2:]:
        candidate = "\n".join(trimmed_lines + [extra])
        if len(candidate) > max_chars:
            break
        trimmed_lines.append(extra)
    block = "\n".join(trimmed_lines)
    if len(block) <= max_chars:
        return block
    summary_line, failures_line = base_lines
    overhead = len(failures_line) + 1
    allowed = max_chars - overhead
    if allowed <= 0:
        return failures_line[:max_chars]
    summary_line = summary_line[:allowed]
    return summary_line + "\n" + failures_line

def _build_ml_iteration_journal_entry(
    state: Dict[str, Any],
    preflight_issues: List[str] | None,
    runtime_error: Dict[str, str] | None,
    outputs_present: List[str],
    outputs_missing: List[str],
    reviewer_verdict: str | None,
    reviewer_reasons: List[str],
    qa_verdict: str | None,
    qa_reasons: List[str],
    next_actions: List[str],
    stage: str,
) -> Dict[str, Any]:
    code = state.get("generated_code") or ""
    code_hash = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()[:12] if code else ""
    iter_id = int(state.get("iteration_count", 0)) + 1
    diagnostics = _collect_iteration_diagnostics(state)
    if runtime_error and isinstance(runtime_error, dict) and runtime_error.get("message"):
        diagnostics.setdefault("root_cause", runtime_error.get("message"))
    stage_value = stage if stage in {"preflight", "runtime_fix", "review_complete"} else "unknown"
    return {
        "iteration_id": iter_id,
        "stage": stage_value,
        "code_hash": code_hash,
        "preflight_issues": preflight_issues or [],
        "runtime_error": runtime_error,
        "outputs_present": outputs_present,
        "outputs_missing": outputs_missing,
        "reviewer_verdict": reviewer_verdict or "UNKNOWN",
        "reviewer_reasons": reviewer_reasons or [],
        "qa_verdict": qa_verdict or "UNKNOWN",
        "qa_reasons": qa_reasons or [],
        "next_actions": next_actions or [],
        "iteration_diagnostics": diagnostics,
    }

def _build_iteration_memory(
    iter_id: int,
    metrics_report: Dict[str, Any],
    case_report: Dict[str, Any],
    weights_report: Dict[str, Any],
    code: str,
    prev_summary: Dict[str, Any] | None,
    advisor_note: str | None,
    diagnostics: Dict[str, Any] | None = None,
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
    if diagnostics:
        summary["iteration_diagnostics"] = diagnostics
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
    ml_journal_written_ids: List[str]
    metrics_signature: str
    weights_signature: str
    execution_output_stale: bool
    sandbox_failed: bool
    sandbox_retry_count: int
    max_sandbox_retries: int
    ml_call_refund_pending: bool
    execution_call_refund_pending: bool
    artifact_content_issues: List[str]
    artifact_content_diagnostics: Dict[str, Any]
    last_successful_execution_output: str
    last_successful_plots: List[str]
    last_successful_output_contract_report: Dict[str, Any]
    last_attempt_score: float
    last_attempt_valid: bool
    best_attempt_score: float
    best_attempt_id: int
    best_attempt_dir: str
    best_attempt_artifact_index: List[Dict[str, Any]]
    best_attempt_output_contract_report: Dict[str, Any]
    best_attempt_execution_output: str
    best_attempt_plots: List[str]
    model_performance: float
    feedback_history: List[str]
    # PDF
    pdf_path: str
    # Encoding & CSV Format
    csv_encoding: str
    csv_sep: str
    csv_decimal: str
    dataset_scale_hints: Dict[str, Any]
    dataset_scale: str
    # Reviewer State
    review_verdict: str
    review_feedback: str
    reviewer_iteration: int
    review_abort_reason: str
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
    data_engineer_attempt: int
    execution_attempt: int # Track runtime retries
    runtime_fix_count: int
    max_runtime_fix_attempts: int
    runtime_fix_terminal: bool
    last_runtime_error_tail: str # Added for Runtime Error Visibility
    data_engineer_audit_override: str
    ml_engineer_audit_override: str
    leakage_audit_summary: str
    ml_skipped_reason: str
    execution_contract: Dict[str, Any]
    execution_contract_min: Dict[str, Any]
    contract_views: Dict[str, Any]
    contract_view_paths: Dict[str, str]
    de_view: Dict[str, Any]
    ml_view: Dict[str, Any]
    reviewer_view: Dict[str, Any]
    translator_view: Dict[str, Any]
    results_advisor_view: Dict[str, Any]
    restrategize_count: int
    strategist_context_override: str
    missing_repeat_count: int
    run_id: str
    run_start_ts: str
    run_start_epoch: float
    dataset_fingerprint: str
    dataset_memory_context: str
    run_budget: Dict[str, Any]
    budget_counters: Dict[str, int]
    qa_budget_exceeded: bool

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
    # NOTE: Cleanup moved AFTER enter_run_workspace to clean workspace, not repo root
    run_id = state.get("run_id") if state else None
    if not run_id:
        run_id = uuid.uuid4().hex[:8]
    run_start_ts = state.get("run_start_ts") if state else None
    if not run_start_ts:
        run_start_ts = datetime.utcnow().isoformat()
    run_start_epoch = state.get("run_start_epoch") if state else None
    if run_start_epoch is None:
        run_start_epoch = time.time()
    orig_cwd_pre = os.getcwd()
    csv_path = state.get("csv_path") if state else ""
    if csv_path:
        resolved_csv_path = _resolve_csv_path_with_base(orig_cwd_pre, csv_path)
        if resolved_csv_path != csv_path:
            state["csv_path"] = resolved_csv_path
            csv_path = resolved_csv_path
    dataset_fingerprint = fingerprint_dataset(csv_path)
    memory_entries = load_dataset_memory()
    memory_context = summarize_memory(memory_entries, dataset_fingerprint)
    run_dir = init_run_dir(run_id, started_at=run_start_ts)
    init_run_bundle(run_id, state, run_dir=run_dir)

    # P0 FIX: Enter isolated run workspace to prevent cross-run contamination
    state = enter_run_workspace(state, run_dir)

    # P0 FIX: Clean workspace AFTER entering (cleans work_dir, not repo root)
    clean_workspace_outputs()
    _cleanup_run_artifacts()

    # Initialize empty artifact index for this run (prevent stale data)
    try:
        os.makedirs("data", exist_ok=True)
        dump_json("data/produced_artifact_index.json", [])
    except Exception:
        pass

    agent_models = {
        "steward": getattr(getattr(steward, "model", None), "model_name", None),
        "strategist": getattr(getattr(strategist, "model", None), "model_name", None),
        "domain_expert": getattr(domain_expert, "model_name", None),
        "execution_planner": getattr(execution_planner, "model_name", None),
        "data_engineer": getattr(data_engineer, "model_name", None),
        "cleaning_reviewer": getattr(cleaning_reviewer, "model_name", None),
        "ml_engineer": getattr(ml_engineer, "model_name", None),
        "reviewer": getattr(reviewer, "model_name", None),
        "qa_reviewer": getattr(qa_reviewer, "model_name", None),
        "results_advisor": getattr(results_advisor, "model_name", None),
        "translator": getattr(translator, "model_name", None),
    }
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
    write_manifest_partial(
        run_id=run_id,
        manifest_path=os.path.join(run_dir, "run_manifest.json"),
        input_info={
            "path": csv_path,
            "sha256": _hash_file(csv_path),
            "dialect": {"encoding": encoding, "sep": sep, "decimal": decimal},
        },
        agent_models=agent_models,
    )

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
    log_agent_snapshot(
        run_id,
        "steward",
        prompt=getattr(steward, "last_prompt", None),
        response=getattr(steward, "last_response", None) or result,
        context={"csv_path": csv_path, "business_objective": state.get("business_objective") if state else ""},
    )

    # Initialize loop variables
    budget_state = _ensure_budget_state(state or {})
    steward_payload = {
        "data_summary": summary,
        "csv_encoding": encoding,
        "csv_sep": sep,
        "csv_decimal": decimal,
        "iteration_count": 0,
        "compliance_iterations": 0,
        "metric_iterations": 0,
        "compliance_passed": False,
        "last_iteration_type": None,
        "ml_journal_written_ids": [],
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
        "run_start_epoch": run_start_epoch,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_memory_context": memory_context,
        "run_budget": budget_state.get("run_budget", {}),
        "budget_counters": budget_state.get("budget_counters", {}),
        "agent_models": agent_models,
    }
    if isinstance(result, dict):
        if "profile" in result:
            steward_payload["profile"] = result.get("profile")
        if "dataset_profile" in result:
            steward_payload["dataset_profile"] = result.get("dataset_profile")
    steward_payload = _add_workspace_metadata(steward_payload, state, orig_cwd_pre, run_dir)
    return steward_payload

def run_strategist(state: AgentState) -> AgentState:
    print("--- [2] Strategist: Formulating 3 Strategies (MIMO v2 Flash) ---")
    abort_state = _abort_if_requested(state, "strategist")
    if abort_state:
        return abort_state

    # Strategist now returns a dict with "strategies": [list of 3]
    user_context = state.get("strategist_context_override") or state.get("business_objective", "")
    result = strategist.generate_strategies(state['data_summary'], user_context)
    run_id = state.get("run_id")
    if run_id:
        log_agent_snapshot(
            run_id,
            "strategist",
            prompt=getattr(strategist, "last_prompt", None),
            response=getattr(strategist, "last_response", None) or result,
            context={"data_summary": state.get("data_summary", ""), "user_context": user_context},
        )
    strategies_list = result.get('strategies', [])
    strategy_spec = result.get("strategy_spec", {})

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

    if strategy_spec:
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/strategy_spec.json", "w", encoding="utf-8") as f_spec:
                json.dump(strategy_spec, f_spec, indent=2)
        except Exception as spec_err:
            print(f"Warning: failed to persist strategy_spec.json: {spec_err}")

    return {
        "strategies": {"strategies": strategies_list},
        "strategy_spec": strategy_spec,
    }

def run_domain_expert(state: AgentState) -> AgentState:
    print("--- [2.5] Domain Expert: Evaluating & Selecting Strategy ---")
    abort_state = _abort_if_requested(state, "domain_expert")
    if abort_state:
        return abort_state

    strategies_wrapper = state.get('strategies', {})
    strategies_list = strategies_wrapper.get('strategies', [])
    business_objective = state.get("business_objective", "")
    data_summary = state.get('data_summary', '')

    # Deliberation Step
    evaluation = domain_expert.evaluate_strategies(data_summary, business_objective, strategies_list)
    reviews = evaluation.get('reviews', [])
    run_id = state.get("run_id")
    if run_id:
        log_agent_snapshot(
            run_id,
            "domain_expert",
            prompt=getattr(domain_expert, "last_prompt", None),
            response=getattr(domain_expert, "last_response", None) or evaluation,
            context={
                "data_summary": data_summary,
                "business_objective": business_objective,
                "strategy_count": len(strategies_list) if isinstance(strategies_list, list) else 0,
            },
        )

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
    orig_cwd = state.get("_orig_cwd")
    if csv_path and not os.path.isabs(csv_path) and orig_cwd:
        resolved_csv_path = _resolve_csv_path_with_base(orig_cwd, csv_path)
        if resolved_csv_path != csv_path:
            state["csv_path"] = resolved_csv_path
            csv_path = resolved_csv_path
    if not csv_path or not os.path.exists(csv_path):
        error_message = f"Warning: input CSV not found: {csv_path}"
        print(error_message)
        if run_id:
            log_run_event(
                run_id,
                "pipeline_aborted_reason",
                {"reason": "input_csv_missing", "csv_path": csv_path},
            )
        oc_report = _persist_output_contract_report(state, reason="input_csv_missing")
        return {
            "error_message": error_message,
            "pipeline_aborted_reason": "input_csv_missing",
            "execution_planner_failed": True,
            "output_contract_report": oc_report,
        }
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
        # Prepare output_dialect from state (csv dialect detected by steward)
        output_dialect = {
            "sep": csv_sep,
            "decimal": csv_decimal,
            "encoding": csv_encoding
        }

        # Prepare env_constraints (conservative default)
        env_constraints = {"forbid_inplace_column_creation": True}

        domain_expert_critique = state.get("selection_reason", "")

        contract = execution_planner.generate_contract(
            strategy=strategy,
            data_summary=data_summary,
            business_objective=business_objective,
            column_inventory=column_inventory,
            output_dialect=output_dialect,
            env_constraints=env_constraints,
            domain_expert_critique=domain_expert_critique,
            run_id=run_id,
        )
    except Exception as e:
        print(f"Warning: execution planner failed ({e}); using fallback contract.")
        # Use V4.1 skeleton fallback instead of legacy
        from src.agents.execution_planner import _create_v41_skeleton
        contract = _create_v41_skeleton(
            strategy=strategy,
            business_objective=business_objective,
            column_inventory=column_inventory,
            output_dialect=output_dialect,
            reason=f"Execution planner exception: {e}"
        )
    contract = _normalize_execution_contract(contract)
    # REMOVED: contract = ensure_role_runbooks(contract)  # V4.1 cutover: contract is immutable
    contract = _ensure_alignment_check_output(contract)
    strategy_spec = state.get("strategy_spec") or _load_json_safe("data/strategy_spec.json")
    objective_type = None
    if isinstance(strategy_spec, dict):
        objective_type = strategy_spec.get("objective_type")
    if not objective_type:
        eval_spec = contract.get("evaluation_spec") if isinstance(contract, dict) else {}
        if isinstance(eval_spec, dict):
            objective_type = eval_spec.get("objective_type")
    dataset_profile = build_dataset_profile(data_summary, column_inventory)
    execution_plan = build_execution_plan(str(objective_type or "unknown"), dataset_profile)
    if isinstance(contract, dict):
        contract["execution_plan"] = execution_plan
    reporting_policy = {}
    try:
        reporting_policy = build_reporting_policy(execution_plan, strategy)
    except Exception:
        reporting_policy = {}
    existing_policy = contract.get("reporting_policy") if isinstance(contract, dict) else {}
    merged_policy = _merge_non_empty_policy(
        reporting_policy if isinstance(reporting_policy, dict) else {},
        existing_policy if isinstance(existing_policy, dict) else {},
    )
    if isinstance(contract, dict):
        merged_policy = _ensure_plot_spec_in_policy(merged_policy, contract)
        if merged_policy:
            contract["reporting_policy"] = merged_policy
    evaluation_spec = {}
    try:
        evaluation_spec = execution_planner.generate_evaluation_spec(
            strategy=strategy,
            contract=contract,
            data_summary=data_summary,
            business_objective=business_objective,
            column_inventory=column_inventory,
        )
        if not isinstance(evaluation_spec, dict):
            evaluation_spec = {}
    except Exception as spec_err:
        print(f"Warning: evaluation spec generation failed: {spec_err}")
        evaluation_spec = {}
    if evaluation_spec:
        contract["evaluation_spec"] = evaluation_spec
    contract = _ensure_scored_rows_output(contract, evaluation_spec if evaluation_spec else None)
    contract_min = getattr(execution_planner, "last_contract_min", None)
    if not contract_min and isinstance(contract, dict):
        try:
            from src.agents.execution_planner import build_contract_min
            contract_min = build_contract_min(
                contract,
                strategy,
                column_inventory,
                contract.get("canonical_columns", []) if isinstance(contract, dict) else [],
            )
        except Exception:
            contract_min = None
    _maybe_set_contract_min_policy(contract_min, merged_policy if isinstance(merged_policy, dict) else {})
    try:
        os.makedirs("data", exist_ok=True)
        dump_json("data/execution_contract.json", contract)
        dump_json("data/plan.json", execution_plan)
        if evaluation_spec:
            dump_json("data/evaluation_spec.json", evaluation_spec)
        if contract_min:
            dump_json("data/contract_min.json", contract_min)
    except Exception as save_err:
        print(f"Warning: failed to persist execution_contract.json: {save_err}")
    if run_id:
        copy_run_contracts(
            run_id,
            [
                "data/execution_contract.json",
                "data/evaluation_spec.json",
                "data/plan.json",
                "data/contract_min.json",
            ],
        )
    view_payload = _build_contract_views(state if isinstance(state, dict) else {}, contract, contract_min)
    if view_payload.get("contract_views"):
        try:
            de_len = len(json.dumps(view_payload["contract_views"].get("de_view", {}), ensure_ascii=True))
            ml_len = len(json.dumps(view_payload["contract_views"].get("ml_view", {}), ensure_ascii=True))
            cleaning_len = len(json.dumps(view_payload["contract_views"].get("cleaning_view", {}), ensure_ascii=True))
            qa_len = len(json.dumps(view_payload["contract_views"].get("qa_view", {}), ensure_ascii=True))
            print(f"Using DE_VIEW_CONTEXT length={de_len}")
            print(f"Using ML_VIEW_CONTEXT length={ml_len}")
            print(f"Using CLEANING_VIEW_CONTEXT length={cleaning_len}")
            print(f"Using QA_VIEW_CONTEXT length={qa_len}")
            if run_id:
                log_run_event(run_id, "de_view_context", {"length": de_len})
                log_run_event(run_id, "ml_view_context", {"length": ml_len})
                log_run_event(run_id, "cleaning_view_context", {"length": cleaning_len})
                log_run_event(run_id, "qa_view_context", {"length": qa_len})
        except Exception:
            pass
    if run_id:
        log_run_event(
            run_id,
            "execution_planner_complete",
            {"required_outputs": contract.get("required_outputs", [])},
        )
    policy = contract.get("iteration_policy") if isinstance(contract, dict) else {}
    result = {"execution_contract": contract}
    if execution_plan:
        result["execution_plan"] = execution_plan
    if evaluation_spec:
        result["evaluation_spec"] = evaluation_spec
    if isinstance(contract_min, dict) and contract_min:
        result["execution_contract_min"] = contract_min
    if view_payload:
        result.update(view_payload)
    if isinstance(policy, dict) and policy:
        result["iteration_policy"] = policy
        runtime_fix_max = policy.get("runtime_fix_max")
        if runtime_fix_max is not None:
            try:
                result["max_runtime_fix_attempts"] = max(1, int(runtime_fix_max))
            except Exception:
                pass
    if run_id:
        log_agent_snapshot(
            run_id,
            "execution_planner",
            prompt=getattr(execution_planner, "last_prompt", None),
            response=getattr(execution_planner, "last_response", None) or result,
            context={
                "strategy": strategy,
                "business_objective": business_objective,
                "planner_diag": getattr(execution_planner, "last_planner_diag", None),
            },
        )
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
        if run_id:
            log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_budget_exceeded"})
        oc_report = _persist_output_contract_report(state, reason="data_engineer_budget_exceeded")
        return {
            "cleaning_code": "",
            "cleaned_data_preview": "Error: Budget Exceeded",
            "error_message": err_msg,
            "output_contract_report": oc_report,
            "pipeline_aborted_reason": "data_engineer_budget_exceeded",
            "data_engineer_failed": True,
            "budget_counters": counters,
        }
    attempt_id = int(state.get("data_engineer_attempt", 0)) + 1
    state["data_engineer_attempt"] = attempt_id
    if run_id:
        log_run_event(run_id, "data_engineer_start", {})

    selected = state.get('selected_strategy')
    if not selected:
        raise ValueError("No strategy selected for data cleaning.")

    business_objective = state.get('business_objective', '')
    csv_path = state['csv_path']
    print(f"DE_INPUT_CSV: {csv_path}")
    if csv_path and not os.path.isabs(csv_path):
        print(f"DE_INPUT_RELATIVE_WARNING: {csv_path}")
    csv_encoding = state.get('csv_encoding', 'utf-8')
    csv_decimal = state.get('csv_decimal', '.')
    csv_sep = state.get('csv_sep', ',')
    input_dialect = {"encoding": csv_encoding, "sep": csv_sep, "decimal": csv_decimal}
    leakage_audit_summary = state.get("leakage_audit_summary", "")
    data_engineer_audit_override = state.get("data_engineer_audit_override", state.get("data_summary", ""))
    required_cols = []
    required_raw_map = {}
    sample_context = ""
    hints = ""
    context_payload = {}
    header_cols = []
    if csv_path and not os.path.exists(csv_path):
        candidate = None
        if not os.path.isabs(csv_path):
            orig_cwd = state.get("_orig_cwd") or ""
            if orig_cwd:
                candidate = os.path.normpath(os.path.abspath(os.path.join(orig_cwd, csv_path)))
        if candidate and os.path.exists(candidate):
            csv_path = candidate
            state["csv_path"] = csv_path
        if not os.path.exists(csv_path):
            error_message = f"Input CSV not found: {csv_path}"
            print(error_message)
            if run_id:
                log_run_event(
                    run_id,
                    "pipeline_aborted_reason",
                    {"reason": "input_csv_missing", "csv_path": csv_path},
                )
            oc_report = _persist_output_contract_report(state, reason="input_csv_missing")
            return {
                "cleaning_code": "",
                "cleaned_data_preview": "Error: Input CSV Missing",
                "error_message": error_message,
                "output_contract_report": oc_report,
                "pipeline_aborted_reason": "input_csv_missing",
                "data_engineer_failed": True,
                "budget_counters": counters,
            }
    if csv_path:
        csv_path = os.path.normpath(csv_path)
        state["csv_path"] = csv_path
    required_cols = _resolve_required_input_columns(state.get("execution_contract", {}), selected)
    header_cols = _read_csv_header(csv_path, csv_encoding, csv_sep) or []
    norm_map: Dict[str, str] = {}
    if header_cols:
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
        required_raw_map = _build_required_raw_map(required_cols, norm_map)
        if required_raw_map:
            raw_map_payload = "REQUIRED_RAW_HEADER_MAP:\n" + json.dumps(required_raw_map, ensure_ascii=True)
            data_engineer_audit_override = _merge_de_audit_override(data_engineer_audit_override, raw_map_payload)
        sample_context = _build_required_sample_context(csv_path, input_dialect, required_cols, norm_map)
        if sample_context:
            data_engineer_audit_override = _merge_de_audit_override(data_engineer_audit_override, sample_context)
            hints = _infer_parsing_hints_from_sample_context(sample_context)
            if hints:
                data_engineer_audit_override = _merge_de_audit_override(data_engineer_audit_override, hints)
    # Inject Raw Snippet for Dialect Grounding (DISABLED: Causing GLM-4.7 Empty Response / Safety Trigger)
    # try:
    #     raw_lines = []
    #     with open(csv_path, "r", encoding=csv_encoding, errors="replace") as f:
    #         for _ in range(5):
    #             line = f.readline()
    #             if not line: break
    #             # Sanitize: keeps only printable characters to avoid breaking LLM JSON prompts
    #             clean_line = "".join(c for c in line.rstrip() if c.isprintable())
    #             raw_lines.append(clean_line)
    #     raw_snippet = "\n".join(raw_lines)
    #     raw_context = f"\n*** RAW FILE SNIPPET (First 5 lines) ***\n{raw_snippet}\n"
    #     data_engineer_audit_override = _merge_de_audit_override(data_engineer_audit_override, raw_context)
    # except Exception as e:
    #     print(f"Warning: could not read raw snippet for DE: {e}")

    state["data_engineer_audit_override"] = data_engineer_audit_override
    contract_min = state.get("execution_contract_min") or _load_json_safe("data/contract_min.json") or {}
    de_view = state.get("de_view") or (state.get("contract_views") or {}).get("de_view")
    if not isinstance(de_view, dict) or not de_view:
        de_view = build_de_view(state.get("execution_contract", {}) or {}, contract_min or {}, state.get("artifact_index") or [])
    de_contract_min = sanitize_contract_min_for_de(contract_min if isinstance(contract_min, dict) else {})
    de_objective = build_de_objective(de_contract_min or {})
    try:
        de_view_len = len(json.dumps(de_view, ensure_ascii=True))
        print(f"Using DE_VIEW_CONTEXT length={de_view_len}")
        if run_id:
            log_run_event(run_id, "de_view_context", {"length": de_view_len})
    except Exception:
        pass
    required_all_columns = _resolve_contract_columns_for_cleaning(state.get("execution_contract", {}))
    context_payload = {
        "csv_path": csv_path,
        "csv_encoding": csv_encoding,
        "csv_sep": csv_sep,
        "csv_decimal": csv_decimal,
        "header_cols": header_cols,
        "required_input_columns": required_cols,
        "required_all_columns": required_all_columns,
        "required_raw_header_map": required_raw_map,
        "raw_required_sample_context": sample_context,
        "data_engineer_audit_override": data_engineer_audit_override,
        "de_view": de_view,
        "execution_contract_min": de_contract_min,
    }
    try:
        os.makedirs("artifacts", exist_ok=True)
        with open(os.path.join("artifacts", "data_engineer_context.json"), "w", encoding="utf-8") as f_ctx:
            json.dump(context_payload, f_ctx, indent=2, ensure_ascii=False)
    except Exception as ctx_err:
        print(f"Warning: failed to persist data_engineer_context.json: {ctx_err}")

    # Generate cleaning script (targeting REMOTE path)
    import inspect

    kwargs = {
        "data_audit": data_engineer_audit_override,
        "strategy": selected,
        "input_path": "data/raw.csv",  # Remote Sandbox Path
        "business_objective": de_objective,
        "csv_encoding": csv_encoding,
        "csv_sep": csv_sep,
        "csv_decimal": csv_decimal,
    }
    sig = inspect.signature(data_engineer.generate_cleaning_script)
    if "execution_contract" in sig.parameters:
        kwargs["execution_contract"] = de_contract_min
    if "contract_min" in sig.parameters:
        kwargs["contract_min"] = de_contract_min
    if "de_view" in sig.parameters:
        kwargs["de_view"] = de_view
    code = data_engineer.generate_cleaning_script(**kwargs)
    try:
        os.makedirs("artifacts", exist_ok=True)
        with open(os.path.join("artifacts", "data_engineer_last.py"), "w", encoding="utf-8") as f_art:
            f_art.write(code)
    except Exception as art_err:
        print(f"Warning: failed to persist data_engineer_last.py: {art_err}")
    if run_id:
        log_agent_snapshot(
            run_id,
            "data_engineer",
            prompt=getattr(data_engineer, "last_prompt", None),
            response=getattr(data_engineer, "last_response", None) or code,
            context=context_payload,
            script=code,
            attempt=attempt_id,
        )

    raw_response = getattr(data_engineer, "last_response", "") or ""
    if "```" in raw_response:
        if run_id:
            log_run_event(run_id, "data_engineer_code_fence_detected", {"attempt": attempt_id})
        stripped = extract_code_block(raw_response)
        if (not stripped or not stripped.strip()) and not code.strip():
            msg = "CRITICAL: Data Engineer returned only fenced/empty code."
            print(msg)
            if run_id:
                log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_code_fence_empty"})
            oc_report = _persist_output_contract_report(state, reason="data_engineer_code_fence_empty")
            return {
                "cleaning_code": "",
                "cleaned_data_preview": "Error: Empty Code After Fence Strip",
                "error_message": msg,
                "output_contract_report": oc_report,
                "pipeline_aborted_reason": "data_engineer_code_fence_empty",
                "data_engineer_failed": True,
                "budget_counters": counters,
            }
        print("Warning: code fences detected in raw response; continuing without retry.")

    plan_payload = None
    is_plan = False
    plan_issues: List[str] = []
    force_code_mode = state.get("force_code_mode", True)
    plan, _ = parse_cleaning_plan(code)
    if plan:
        if force_code_mode:
            msg = "CLEANING_PLAN_NOT_ALLOWED: expected Python cleaning code."
            if run_id:
                log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_plan_not_allowed"})
            oc_report = _persist_output_contract_report(state, reason="data_engineer_plan_not_allowed")
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Error: Plan Output",
                "error_message": msg,
                "output_contract_report": oc_report,
                "pipeline_aborted_reason": "data_engineer_plan_not_allowed",
                "data_engineer_failed": True,
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
            if run_id:
                log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_plan_invalid"})
            oc_report = _persist_output_contract_report(state, reason="data_engineer_plan_invalid")
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Error: Invalid Plan",
                "error_message": msg,
                "output_contract_report": oc_report,
                "pipeline_aborted_reason": "data_engineer_plan_invalid",
                "data_engineer_failed": True,
                "budget_counters": counters,
            }

    # Check if generation failed
    if code.strip().startswith("# Error"):
        print(f"Correction: Data Engineer failed to generate code. Error: {code}")
        if run_id:
            log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_generation_failed"})
        oc_report = _persist_output_contract_report(state, reason="data_engineer_generation_failed")
        return {
            "cleaning_code": code,
            "cleaned_data_preview": "Error: Generation Failed",
            "error_message": code,
            "output_contract_report": oc_report,
            "pipeline_aborted_reason": "data_engineer_generation_failed",
            "data_engineer_failed": True,
            "budget_counters": counters,
        }

    original_code = code
    code, auto_fixes = _apply_static_autofixes(code)
    if auto_fixes and run_id:
        log_run_event(run_id, "auto_fix_applied", {"attempt": attempt_id, "fixes": auto_fixes})
        run_dir = get_run_dir(run_id)
        if run_dir:
            try:
                base = os.path.join(run_dir, "agents", "data_engineer", f"iteration_{attempt_id}")
                os.makedirs(base, exist_ok=True)
                with open(os.path.join(base, "script.py"), "w", encoding="utf-8") as f_script:
                    f_script.write(code)
            except Exception:
                pass
    is_safe, violations = scan_code_safety(code)
    if not is_safe:
        snippets = _collect_violation_snippets(original_code, violations)
        if run_id:
            log_run_event(
                run_id,
                "static_scan_failed",
                {"attempt": attempt_id, "violations": violations, "snippets": snippets},
            )
        run_dir = get_run_dir(run_id) if run_id else None
        if run_dir:
            try:
                base = os.path.join(run_dir, "agents", "data_engineer", f"iteration_{attempt_id}")
                os.makedirs(base, exist_ok=True)
                with open(os.path.join(base, "script_blocked.py"), "w", encoding="utf-8") as f_blocked:
                    f_blocked.write(original_code)
                blocked_reason = {
                    "violations": violations,
                    "snippets": snippets,
                    "auto_fixes_applied": auto_fixes,
                }
                with open(os.path.join(base, "blocked_reason.json"), "w", encoding="utf-8") as f_reason:
                    json.dump(blocked_reason, f_reason, indent=2)
                guard_code = f"raise ValueError('GENERATED CODE BLOCKED BY STATIC SCAN: {violations}')"
                with open(os.path.join(base, "script.py"), "w", encoding="utf-8") as f_guard:
                    f_guard.write(guard_code)
            except Exception:
                pass
        if not state.get("de_static_guard_retry_done"):
            new_state = dict(state)
            new_state["de_static_guard_retry_done"] = True
            base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
            context_payload = {
                "violations": violations,
                "snippets": snippets,
                "auto_fixes_applied": auto_fixes,
            }
            payload = (
                "STATIC_SCAN_VIOLATIONS:\n"
                "ENVIRONMENT_FEEDBACK (Please Adjust Code):\n"
                + json.dumps(context_payload, indent=2, ensure_ascii=False)
            )
            new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, payload)
            if run_id:
                log_run_event(run_id, "static_scan_retry", {"attempt": attempt_id, "violations": violations})
            print("Static scan guard: retrying Data Engineer with violation context.")
            return run_data_engineer(new_state)
        if run_id:
            log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_static_scan_failed"})
        oc_report = _persist_output_contract_report(state, reason="data_engineer_static_scan_failed")
        guard_code = f"raise ValueError('GENERATED CODE BLOCKED BY STATIC SCAN: {violations}')"
        return {
            "cleaning_code": guard_code,
            "cleaned_data_preview": "Error: Security Blocked",
            "error_message": "CRITICAL: cleaning code blocked by static scan.",
            "output_contract_report": oc_report,
            "pipeline_aborted_reason": "data_engineer_static_scan_failed",
            "data_engineer_failed": True,
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
            if run_id:
                log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_preflight_failed"})
            oc_report = _persist_output_contract_report(state, reason="data_engineer_preflight_failed")
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Preflight Failed",
                "error_message": msg,
                "feedback_history": fh,
                "last_gate_context": lgc,
                "output_contract_report": oc_report,
                "pipeline_aborted_reason": "data_engineer_preflight_failed",
                "data_engineer_failed": True,
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
            if run_id:
                log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_undefined_names"})
            oc_report = _persist_output_contract_report(state, reason="data_engineer_undefined_names")
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Preflight Failed",
                "error_message": msg,
                "feedback_history": fh,
                "output_contract_report": oc_report,
                "pipeline_aborted_reason": "data_engineer_undefined_names",
                "data_engineer_failed": True,
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
        if run_id:
            log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_json_literals"})
        oc_report = _persist_output_contract_report(state, reason="data_engineer_json_literals")
        return {
            "cleaning_code": code,
            "cleaned_data_preview": "Error: Contract Literal Guard",
            "error_message": "CRITICAL: JSON literals (null/true/false) found in Python code.",
            "output_contract_report": oc_report,
            "pipeline_aborted_reason": "data_engineer_json_literals",
            "data_engineer_failed": True,
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
            if run_id:
                log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_manifest_guard"})
            oc_report = _persist_output_contract_report(state, reason="data_engineer_manifest_guard")
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Error: Manifest JSON Guard",
                "error_message": "CRITICAL: Manifest serialization must use json.dump(..., default=_json_default).",
                "output_contract_report": oc_report,
                "pipeline_aborted_reason": "data_engineer_manifest_guard",
                "data_engineer_failed": True,
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
            if run_id:
                log_run_event(run_id, "pipeline_aborted_reason", {"reason": "data_engineer_dialect_guard"})
            oc_report = _persist_output_contract_report(state, reason="data_engineer_dialect_guard")
            return {
                "cleaning_code": code,
                "cleaned_data_preview": "Error: Dialect Guard",
                "error_message": "CRITICAL: pd.read_csv must use provided dialect parameters.",
                "output_contract_report": oc_report,
                "pipeline_aborted_reason": "data_engineer_dialect_guard",
                "data_engineer_failed": True,
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

            for sb_attempt in range(2):
              try:
                with create_sandbox_with_retry(Sandbox, max_attempts=2, run_id=run_id, step="data_engineer") as sandbox:
                    if not hasattr(sandbox, "run_code"):
                        raise RuntimeError(
                            "E2B Sandbox missing run_code. Ensure we are using e2b_code_interpreter.Sandbox (Sandbox)."
                        )
                    step_name = "data_engineer"
                    run_root = f"/home/user/run/{run_id}/{step_name}/attempt_{attempt_id}"
                    # 1. Setup Environment
                    print("Installing dependencies in Sandbox...")
                    run_cmd_with_retry(sandbox, "pip install pandas numpy", retries=2)
                    run_cmd_with_retry(sandbox, f"rm -rf {run_root}", retries=2)
                    run_cmd_with_retry(sandbox, f"mkdir -p {run_root}/data", retries=2)

                    # 2. Upload Raw Data (P2.1: Use canonical path + aliases)
                    remote_raw_abs = canonical_abs(run_root, CANONICAL_RAW_REL)
                    print(f"Uploading {csv_path} to sandbox at {CANONICAL_RAW_REL}...")
                    with open(csv_path, "rb") as f:
                        sandbox.files.write(remote_raw_abs, f)

                    # 2.5 Create aliases for raw data (P2.1)
                    alias_commands = build_symlink_or_copy_commands(
                        run_root,
                        canonical_rel=CANONICAL_RAW_REL,
                        aliases=COMMON_RAW_ALIASES
                    )
                    if alias_commands:
                        print(f"Creating {len(alias_commands)} raw data aliases...")
                        full_cmd = " && ".join(alias_commands)
                        run_cmd_with_retry(sandbox, f"sh -lc 'cd {run_root} && {full_cmd}'")
                        print(f"SANDBOX_INPUT_CANONICAL: {CANONICAL_RAW_REL}")
                        print(f"SANDBOX_INPUT_ALIASES: {COMMON_RAW_ALIASES}")

                    # 2.5 Upload Execution Contract (best-effort)
                    local_contract_path = "data/execution_contract.json"
                    if os.path.exists(local_contract_path):
                        try:
                            with open(local_contract_path, "rb") as f:
                                sandbox.files.write(f"{run_root}/data/execution_contract.json", f)
                        except Exception as contract_err:
                            print(f"Warning: failed to upload execution_contract.json: {contract_err}")

                    # 3. Execute Cleaning (P2.1: Patch placeholders + minimal backward compat)
                    code = patch_placeholders(code, data_rel=CANONICAL_RAW_REL)
                    working_dir_injection = (
                        "import os\n"
                        f"os.makedirs(r\"{run_root}\", exist_ok=True)\n"
                        f"os.chdir(r\"{run_root}\")\n"
                    )
                    code = working_dir_injection + code
                    print("Executing Cleaning Script in Sandbox...")
                    execution = run_code_with_optional_timeout(sandbox, code, timeout_s=DE_TIMEOUT_S)

                    # Capture Output
                    stdout_text = "\n".join(execution.logs.stdout or [])
                    stderr_text = "\n".join(execution.logs.stderr or [])
                    output_log = ""
                    if stdout_text:
                        output_log += stdout_text
                    if stderr_text:
                        output_log += stderr_text
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
                    ls_check = run_cmd_with_retry(sandbox, f"ls {run_root}/data/cleaned_data.csv", retries=2)
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

                    downloaded_paths = []

                    # Download Result (CSV) using centralized helper
                    print("Downloading cleaned data...")
                    local_cleaned_path = "data/cleaned_data.csv"
                    os.makedirs("data", exist_ok=True)

                    csv_content = safe_download_bytes(sandbox, f"{run_root}/data/cleaned_data.csv")
                    if csv_content is None:
                        return {
                            "cleaning_code": code,
                            "cleaned_data_preview": "Error: Download Failed",
                            "error_message": "Failed to download cleaned data from sandbox",
                            "budget_counters": counters,
                        }
                    with open(local_cleaned_path, "wb") as f_local:
                        f_local.write(csv_content)
                    if os.path.exists(local_cleaned_path):
                        downloaded_paths.append(local_cleaned_path)

                    # Download Manifest (JSON) - Roundtrip Support using centralized helper
                    print("Downloading cleaning manifest...")
                    local_manifest_path = "data/cleaning_manifest.json"
                    manifest_content = safe_download_bytes(sandbox, f"{run_root}/data/cleaning_manifest.json")
                    if manifest_content is not None:
                        with open(local_manifest_path, "wb") as f_local:
                            f_local.write(manifest_content)
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
                    if os.path.exists(local_manifest_path):
                        downloaded_paths.append(local_manifest_path)
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

                    outputs_listing = []
                    try:
                        listing_proc = sandbox.commands.run(
                            f"sh -lc 'cd {run_root} && find . -maxdepth 4 -type f -printf \"%p\\t%s\\n\" 2>/dev/null'"
                        )
                        if listing_proc.exit_code == 0:
                            outputs_listing = [line for line in listing_proc.stdout.splitlines() if line.strip()]
                    except Exception:
                        outputs_listing = []
                    if run_id:
                        log_sandbox_attempt(
                            run_id,
                            step_name,
                            attempt_id,
                            code=code,
                            stdout=stdout_text,
                            stderr=stderr_text,
                            outputs_listing=outputs_listing,
                            downloaded_paths=downloaded_paths,
                            exit_code=getattr(execution, "exit_code", None),
                            error_tail=(execution.error.traceback if execution.error else None),
                        )

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

                break  # Sandbox execution successful, exit retry loop
              except Exception as e:
                if sb_attempt == 0 and is_transient_sandbox_error(e):
                    print(f"SANDBOX_RETRY step=data_engineer attempt={sb_attempt+1} err={e}")
                    continue
                raise


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

            # --- ID/SCALE GUARDS (Deterministic) ---
            id_issues = []
            scale_issues = []
            identifier_cols = []
            roles = get_column_roles(contract) if isinstance(contract, dict) else {}
            role_ids = roles.get("identifiers") if isinstance(roles, dict) else None
            if isinstance(role_ids, list):
                identifier_cols = [c for c in role_ids if c in df_mapped.columns]
            if not identifier_cols:
                identifier_cols = [c for c in df_mapped.columns if is_identifier_like(c)]

            guard_columns = list(dict.fromkeys(identifier_cols + required_cols))
            raw_map = _build_required_raw_map(guard_columns, norm_map)
            raw_usecols = list(dict.fromkeys(raw_map.values()))
            raw_sample_df = (
                sample_raw_columns(csv_path, input_dialect, raw_usecols, nrows=200, dtype=str)
                if raw_usecols
                else pd.DataFrame()
            )

            if identifier_cols and not state.get("de_id_guard_retry_done"):
                for col in identifier_cols:
                    if col not in df_mapped.columns:
                        continue
                    series = df_mapped[col]
                    detection = detect_identifier_scientific_notation(series)
                    if not detection.get("flag"):
                        continue
                    raw_col = raw_map.get(col)
                    raw_examples = []
                    if raw_col and raw_col in raw_sample_df.columns:
                        raw_examples = raw_sample_df[raw_col].dropna().astype(str).head(6).tolist()
                    cleaned_examples = series.dropna().astype(str).head(6).tolist()
                    id_issues.append(
                        {
                            "column": col,
                            "raw_column": raw_col,
                            "raw_examples": raw_examples,
                            "cleaned_examples": cleaned_examples,
                            "details": detection,
                        }
                    )

            if required_cols and not state.get("de_scale_guard_retry_done"):
                for col in required_cols:
                    if col not in df_mapped.columns:
                        continue
                    if col in identifier_cols:
                        continue
                    series = df_mapped[col]
                    if not pd.api.types.is_numeric_dtype(series):
                        continue
                    raw_col = raw_map.get(col)
                    if not raw_col or raw_col not in raw_sample_df.columns:
                        continue
                    raw_strings = raw_sample_df[raw_col].dropna().astype(str).head(200).tolist()
                    if not raw_strings:
                        continue
                    if _is_percent_like(col, raw_strings):
                        continue
                    raw_values = [v for v in (_coerce_raw_numeric(v) for v in raw_strings) if v is not None]
                    if len(raw_values) < 5:
                        continue
                    try:
                        cleaned_max = float(series.max())
                    except Exception:
                        continue
                    if cleaned_max > 1.5:
                        continue
                    raw_median = _median(raw_values)
                    raw_max = max(raw_values) if raw_values else None
                    raw_high_ratio = (
                        sum(val > 5 for val in raw_values) / len(raw_values) if raw_values else 0.0
                    )
                    if raw_median is None:
                        continue
                    if raw_median > 5 or (raw_high_ratio >= 0.6 and raw_max and raw_max > 5):
                        cleaned_examples = series.dropna().astype(str).head(6).tolist()
                        scale_issues.append(
                            {
                                "column": col,
                                "raw_column": raw_col,
                                "raw_examples": raw_strings[:6],
                                "cleaned_examples": cleaned_examples,
                                "cleaned_max": cleaned_max,
                                "raw_median": raw_median,
                                "raw_max": raw_max,
                                "raw_high_ratio": raw_high_ratio,
                            }
                        )

            if id_issues or scale_issues:
                payload_lines = []
                if id_issues:
                    payload_lines.append("ID_SCI_NOTATION_GUARD:")
                    payload_lines.append(
                        "- Identifier-like columns must remain string-like; avoid scientific notation or .0 suffixes."
                    )
                    for issue in id_issues:
                        payload_lines.append(
                            f"* col={issue.get('column')} raw={issue.get('raw_column')} "
                            f"raw_examples={issue.get('raw_examples')} cleaned_examples={issue.get('cleaned_examples')} "
                            f"details={issue.get('details')}"
                        )
                if scale_issues:
                    payload_lines.append("SCALE_SHIFT_GUARD:")
                    payload_lines.append(
                        "- Do NOT rescale numeric columns unless percent-like evidence exists (name or raw '%')."
                    )
                    for issue in scale_issues:
                        payload_lines.append(
                            f"* col={issue.get('column')} raw={issue.get('raw_column')} "
                            f"raw_examples={issue.get('raw_examples')} cleaned_examples={issue.get('cleaned_examples')} "
                            f"cleaned_max={issue.get('cleaned_max')} raw_median={issue.get('raw_median')} "
                            f"raw_max={issue.get('raw_max')} raw_high_ratio={issue.get('raw_high_ratio')}"
                        )
                if payload_lines:
                    new_state = dict(state)
                    if id_issues:
                        new_state["de_id_guard_retry_done"] = True
                    if scale_issues:
                        new_state["de_scale_guard_retry_done"] = True
                    base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                    new_state["data_engineer_audit_override"] = _merge_de_audit_override(
                        base_override, "\n".join(payload_lines)
                    )
                    print("ID/SCALE guard: retrying Data Engineer with evidence.")
                    return run_data_engineer(new_state)

            empty_required = _find_empty_required_columns(df_mapped, required_cols)
            if empty_required:
                required_issues = []
                optional_issues = []
                for item in empty_required:
                    req_meta = _resolve_requirement_meta(contract, item.get("column", ""))
                    is_optional = _is_optional_requirement(req_meta)
                    item["required"] = not is_optional
                    if is_optional:
                        optional_issues.append(item)
                    else:
                        required_issues.append(item)
                if optional_issues and not required_issues:
                    warn_cols = [item.get("column") for item in optional_issues if item.get("column")]
                    print(f"Warning: optional columns empty after cleaning: {warn_cols}")
                else:
                    required_cols_only = [item.get("column") for item in required_issues if item.get("column")]
                    raw_map = _build_required_raw_map(required_cols_only, norm_map)
                    raw_stats = _sample_raw_null_stats(csv_path, input_dialect, list(raw_map.values()))
                    parse_lines = []
                    has_raw_values = False
                    for item in required_issues:
                        col = item.get("column")
                        raw_col = raw_map.get(col) if col else None
                        stats = raw_stats.get(raw_col, {}) if raw_col else {}
                        non_null_frac = stats.get("non_null_frac")
                        if isinstance(non_null_frac, (int, float)) and non_null_frac > 0.01:
                            has_raw_values = True
                        parse_lines.append(
                            f"- {col}: null_frac={item.get('null_frac'):.2%}, non_null_count={item.get('non_null_count')}, "
                            f"raw_col={raw_col}, raw_non_null_frac={non_null_frac}"
                        )
                    payload = "EMPTY_REQUIRED_COLUMNS:\n" + "\n".join(parse_lines)
                    try:
                        sample_context = _build_required_sample_context(
                            csv_path, input_dialect, required_cols_only, norm_map, max_rows=200
                        )
                        if sample_context:
                            payload = f"{payload}\n\n{sample_context}"
                            hints = _infer_parsing_hints_from_sample_context(sample_context)
                            if hints:
                                payload = f"{payload}\n\n{hints}"
                    except Exception:
                        pass
                    if has_raw_values and not state.get("de_empty_required_retry_done"):
                        new_state = dict(state)
                        new_state["de_empty_required_retry_done"] = True
                        base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
                        new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, payload)
                        print("Empty required columns guard: retrying Data Engineer with evidence.")
                        return run_data_engineer(new_state)
                    return {
                        "cleaning_code": code,
                        "cleaned_data_preview": "Error: Empty Required Columns",
                        "error_message": "CRITICAL: Required input columns are empty after cleaning.\n" + payload,
                    }

            # --- CLEANING REVIEWER (contract-driven) ---
            if cleaning_reviewer:
                try:
                    manifest_for_review = _load_json_safe(local_manifest_path)
                    output_dialect = None
                    output_payload = manifest_for_review.get("output_dialect")
                    if isinstance(output_payload, dict):
                        output_dialect = {
                            "sep": output_payload.get("sep") or csv_sep,
                            "decimal": output_payload.get("decimal") or csv_decimal,
                            "encoding": output_payload.get("encoding") or csv_encoding,
                        }
                    cleaning_view = state.get("cleaning_view") or (state.get("contract_views") or {}).get("cleaning_view")
                    if cleaning_view:
                        cleaning_view_copy = dict(cleaning_view)
                        if output_dialect:
                            cleaning_view_copy["output_dialect"] = output_dialect
                        if isinstance(input_dialect, dict):
                            cleaning_view_copy["input_dialect"] = input_dialect
                        review_result = cleaning_reviewer.review_cleaning(
                            cleaning_view_copy,
                            cleaned_csv_path=local_cleaned_path,
                            cleaning_manifest_path=local_manifest_path,
                            raw_csv_path=csv_path,
                        )
                    else:
                        contract = state.get("execution_contract", {}) or {}
                        contract_min = state.get("execution_contract_min", {}) or {}
                        review_context = {
                            "cleaning_view": {},
                            "cleaning_gates": contract.get("cleaning_gates")
                            or contract_min.get("cleaning_gates")
                            or [],
                            "required_columns": _resolve_required_input_columns(contract, selected),
                            "dialect": input_dialect,
                            "column_roles": contract.get("column_roles") if isinstance(contract, dict) else {},
                            "cleaned_csv_path": local_cleaned_path,
                            "cleaning_manifest_path": local_manifest_path,
                            "raw_csv_path": csv_path,
                        }
                        if output_dialect:
                            review_context["output_dialect"] = output_dialect
                        if isinstance(input_dialect, dict):
                            review_context["input_dialect"] = input_dialect
                        review_result = cleaning_reviewer.review_cleaning(review_context)
                    try:
                        os.makedirs("artifacts", exist_ok=True)
                        with open(os.path.join("artifacts", "cleaning_reviewer_report.json"), "w", encoding="utf-8") as f_rep:
                            json.dump(review_result, f_rep, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                    if run_id:
                        log_agent_snapshot(
                            run_id,
                            "cleaning_reviewer",
                            prompt=getattr(cleaning_reviewer, "last_prompt", None),
                            response=getattr(cleaning_reviewer, "last_response", None) or review_result,
                            context=cleaning_view,
                            verdicts=review_result,
                        )

                    if isinstance(review_result, dict):
                        status = review_result.get("status")
                        if status == "REJECTED":
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
                            if run_id:
                                log_run_event(run_id, "pipeline_aborted_reason", {"reason": "cleaning_reviewer_rejected"})
                            return {
                                "cleaning_code": code,
                                "cleaned_data_preview": "Error: Cleaning Reviewer Rejected",
                                "error_message": "CRITICAL: Cleaning reviewer rejected cleaned data.",
                                "pipeline_aborted_reason": "cleaning_reviewer_rejected",
                                "data_engineer_failed": True,
                                "budget_counters": counters,
                            }
                        if status == "APPROVE_WITH_WARNINGS":
                            warnings = review_result.get("warnings") or []
                            if not isinstance(warnings, list):
                                warnings = [str(warnings)]
                            state["cleaning_reviewer_warnings"] = warnings
                except Exception as review_err:
                    msg = f"CRITICAL: Cleaning reviewer failed: {review_err}"
                    print(msg)
                    if run_id:
                        log_run_event(run_id, "pipeline_aborted_reason", {"reason": "cleaning_reviewer_failed"})
                    return {
                        "cleaning_code": code,
                        "cleaned_data_preview": "Error: Cleaning Reviewer Failed",
                        "error_message": msg,
                        "pipeline_aborted_reason": "cleaning_reviewer_failed",
                        "data_engineer_failed": True,
                        "budget_counters": counters,
                    }
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

        # P2.3: Calculate dataset scale hints after Data Engineer
        work_dir = state.get("work_dir", ".")
        dataset_scale_hints = get_dataset_scale_hints(work_dir, "data/cleaned_data.csv")

        if run_id:
            log_run_event(
                run_id,
                "data_engineer_complete",
                {
                    "rows": len(df_final),
                    "columns": len(df_final.columns),
                    "dataset_scale": dataset_scale_hints.get("scale"),
                    "dataset_file_mb": dataset_scale_hints.get("file_mb"),
                },
            )

        return {
             "cleaning_code": code,
             "cleaned_data_preview": preview,
             "csv_sep": csv_sep,
             "csv_decimal": csv_decimal,
             "csv_encoding": csv_encoding,
             "leakage_audit_summary": leakage_audit_summary,
             "budget_counters": counters,
             "dataset_scale_hints": dataset_scale_hints,
             "dataset_scale": dataset_scale_hints.get("scale") if isinstance(dataset_scale_hints, dict) else None,
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

    run_start_epoch = state.get("run_start_epoch")
    if run_start_epoch is not None:
        try:
            cleaned_mtime = os.path.getmtime("data/cleaned_data.csv")
            if cleaned_mtime < float(run_start_epoch) - 1.0:
                print("Data Engineer Failed: stale cleaned_data.csv detected.")
                return "failed"
        except Exception:
            pass

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

    # Strategy Lock: Capture snapshot on first iteration, validate on subsequent iterations
    strategy_lock_snapshot = state.get("strategy_lock_snapshot")
    if not strategy_lock_snapshot:
        strategy_lock_snapshot = _capture_strategy_snapshot(state)
        state["strategy_lock_snapshot"] = strategy_lock_snapshot
        print(f"STRATEGY_LOCK: Captured initial snapshot: {strategy_lock_snapshot.get('strategy_title')}")
    else:
        lock_ok, lock_details = _validate_strategy_lock(state)
        if not lock_ok:
            drifts = lock_details.get("drifts", [])
            print(f"STRATEGY_LOCK_FAILED: Drift detected: {drifts}")
            if run_id:
                log_run_event(run_id, "strategy_lock_failed", {"drifts": drifts})
            error_msg = f"STRATEGY_LOCK_FAILED: Strategy drift detected: {drifts}"
            placeholder_code = f"# {error_msg}\n# Pipeline halted due to strategy/contract drift."
            return {
                "hard_fail_reason": "STRATEGY_LOCK_FAILED",
                "review_verdict": "NEEDS_IMPROVEMENT",
                "review_retry_worth_it": False,
                "stop_reason": "STRATEGY_LOCK",
                "error_message": error_msg,
                "strategy_lock_failed": True,
                "strategy_lock_details": lock_details,
                "budget_counters": counters,
                # Terminal flags to halt pipeline before execution
                "generated_code": placeholder_code,
                "last_generated_code": placeholder_code,
                "review_abort_reason": "strategy_lock_failed",
            }

    if run_id:
        log_run_event(run_id, "ml_engineer_start", {"iteration": state.get("iteration_count", 0) + 1})

    strategy = state.get('selected_strategy')

    # Pass input context
    data_path = "data/cleaned_data.csv"
    de_view = state.get("de_view") or (state.get("contract_views") or {}).get("de_view")
    if isinstance(de_view, dict) and de_view.get("output_path"):
        data_path = str(de_view.get("output_path"))
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
    execution_contract = state.get("execution_contract", {}) or _load_json_safe("data/execution_contract.json")
    evaluation_spec = state.get("evaluation_spec") or (execution_contract or {}).get("evaluation_spec") or {}
    if isinstance(execution_contract, dict) and evaluation_spec and not execution_contract.get("evaluation_spec"):
        execution_contract["evaluation_spec"] = evaluation_spec
    contract_min = _load_json_safe("data/contract_min.json")
    if not isinstance(contract_min, dict) or not contract_min:
        contract_min = _build_contract_min(execution_contract, evaluation_spec)
        try:
            os.makedirs("data", exist_ok=True)
            dump_json("data/contract_min.json", contract_min)
        except Exception:
            pass
    if not execution_contract:
        data_audit_context = _merge_de_audit_override(
            data_audit_context,
            "WARNING: execution_contract missing; using minimal contract context only.",
        )
    manifest_path = "data/cleaning_manifest.json"
    if os.path.exists(manifest_path):
        use_manifest = bool(state.get("use_output_dialect") or state.get("dialect_from_manifest"))
        if not use_manifest and csv_sep == "," and csv_decimal == "." and str(csv_encoding).lower() in {"utf-8", "utf8"}:
            use_manifest = True
        if use_manifest:
            try:
                csv_sep, csv_decimal, csv_encoding, dialect_updated = get_output_dialect_from_manifest(
                    manifest_path, csv_sep, csv_decimal, csv_encoding
                )
                if dialect_updated:
                    print(f"ML dialect updated from output_dialect: sep={csv_sep}, decimal={csv_decimal}, encoding={csv_encoding}")
            except Exception:
                pass
    if not os.path.exists(data_path) and os.path.exists("data/cleaned_full.csv"):
        data_path = "data/cleaned_full.csv"
    required_input_cols = _resolve_required_input_columns(execution_contract, strategy)
    header_cols = _read_csv_header(data_path, csv_encoding, csv_sep)
    if data_path == "data/cleaned_data.csv" and required_input_cols:
        header_norm = {_norm_name(c) for c in header_cols}
        missing_required = [c for c in required_input_cols if _norm_name(c) not in header_norm]
        if missing_required and os.path.exists("data/cleaned_full.csv"):
            data_path = "data/cleaned_full.csv"
            header_cols = _read_csv_header(data_path, csv_encoding, csv_sep)
            note = f"ML_DATA_PATH_FALLBACK: using cleaned_full.csv due to missing required columns {missing_required[:5]}"
            data_audit_context = _merge_de_audit_override(data_audit_context, note)
    feature_availability = (execution_contract or {}).get("feature_availability", [])
    availability_summary = (execution_contract or {}).get("availability_summary", "")
    iteration_memory = list(state.get("ml_iteration_memory", []) or [])
    iteration_memory_block = state.get("ml_iteration_memory_block", "")
    iter_id = int(state.get("iteration_count", 0)) + 1
    if run_id and iter_id >= 2:
        journal_entries = _load_ml_iteration_journal(run_id)
        journal_block = _build_ml_iteration_memory_block(journal_entries)
        if journal_block:
            iteration_memory_block = journal_block
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
            kwargs["execution_contract"] = contract_min or execution_contract
        if "dataset_scale" in sig.parameters:
            dataset_scale_hints = state.get("dataset_scale_hints") or {}
            kwargs["dataset_scale"] = dataset_scale_hints
        if "dataset_scale_str" in sig.parameters:
            dataset_scale_hints = state.get("dataset_scale_hints") or {}
            kwargs["dataset_scale_str"] = dataset_scale_hints.get("scale") if isinstance(dataset_scale_hints, dict) else None
        if "ml_view" in sig.parameters:
            ml_view = state.get("ml_view") or (state.get("contract_views") or {}).get("ml_view")
            if not isinstance(ml_view, dict) or not ml_view:
                ml_view = build_ml_view(execution_contract or {}, contract_min or {}, state.get("artifact_index") or [])
            kwargs["ml_view"] = ml_view
            try:
                ml_view_len = len(json.dumps(ml_view, ensure_ascii=True))
                print(f"Using ML_VIEW_CONTEXT length={ml_view_len}")
                if run_id:
                    log_run_event(run_id, "ml_view_context", {"length": ml_view_len})
            except Exception:
                pass
        aliasing = {}
        derived_present = []
        sample_context = ""
        context_ops_blocks = []
        dataset_scale_hints = state.get("dataset_scale_hints") or {}
        scale_line = (
            "POST-CLEAN DATASET SCALE: "
            f"scale={dataset_scale_hints.get('scale') or 'unknown'} "
            f"file_mb={dataset_scale_hints.get('file_mb') or 'unknown'} "
            f"est_rows={dataset_scale_hints.get('est_rows') or 'unknown'} "
            f"max_train_rows={dataset_scale_hints.get('max_train_rows') or 'unknown'} "
            f"chunk_size={dataset_scale_hints.get('chunk_size') or 'unknown'}"
        )
        context_ops_blocks.append(scale_line)
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
        if contract_min:
            context_ops_blocks.append(
                "CONTRACT_MIN_CONTEXT:\n" + json.dumps(contract_min, ensure_ascii=True)
            )
        ml_view = state.get("ml_view") or (state.get("contract_views") or {}).get("ml_view")
        if isinstance(ml_view, dict) and ml_view:
            context_ops_blocks.append(
                "ML_VIEW_CONTEXT:\n" + json.dumps(ml_view, ensure_ascii=True)
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
                "execution_contract_min": contract_min,
                "ml_view": state.get("ml_view") or (state.get("contract_views") or {}).get("ml_view"),
                "data_audit_context": data_audit_context,
                "ml_engineer_audit_override": ml_audit_override,
                "feature_availability": feature_availability,
                "availability_summary": availability_summary,
                "dataset_scale_hints": dataset_scale_hints,
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
        if run_id:
            log_agent_snapshot(
                run_id,
                "ml_engineer",
                prompt=getattr(ml_engineer, "last_prompt", None),
                response=getattr(ml_engineer, "last_response", None) or code,
                context=ctx_payload,
                script=code,
                attempt=int(state.get("iteration_count", 0)) + 1,
            )
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
            "csv_sep": csv_sep,
            "csv_decimal": csv_decimal,
            "csv_encoding": csv_encoding,
            "error_message": "",
            "ml_call_refund_pending": False,
            "execution_call_refund_pending": False,
            "ml_context_snapshot": {
                "cleaned_column_inventory": header_cols,
                "cleaned_aliasing_collisions": aliasing if header_cols else {},
                "derived_columns_present": derived_present if header_cols else [],
                "raw_required_sample_context": sample_context,
                "feature_semantics": (execution_contract or {}).get("feature_semantics", []),
                "business_sanity_checks": (execution_contract or {}).get("business_sanity_checks", []),
                "execution_contract_min": contract_min,
            },
            "budget_counters": counters,
            "strategy_lock_snapshot": strategy_lock_snapshot,
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
    evaluation_spec = state.get("evaluation_spec") or (state.get("execution_contract", {}) or {}).get("evaluation_spec")
    reviewer_view = state.get("reviewer_view") or (state.get("contract_views") or {}).get("reviewer_view")
    if not isinstance(reviewer_view, dict) or not reviewer_view:
        reviewer_view = build_reviewer_view(
            state.get("execution_contract", {}) or {},
            _load_json_safe("data/contract_min.json") or {},
            state.get("artifact_index") or [],
        )
    analysis_type = reviewer_view.get("objective_type") or strategy.get('analysis_type', 'predictive')
    strategy_context = reviewer_view.get("strategy_summary") or ""
    business_objective = ""
    try:
        reviewer_view_len = len(json.dumps(reviewer_view, ensure_ascii=True))
        print(f"Using REVIEWER_VIEW_CONTEXT length={reviewer_view_len}")
        if run_id:
            log_run_event(run_id, "reviewer_view_context", {"length": reviewer_view_len})
    except Exception:
        pass

    try:
        review = reviewer.review_code(
            code,
            analysis_type,
            business_objective,
            strategy_context,
            evaluation_spec,
            reviewer_view=reviewer_view,
        )
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
        if run_id:
            log_agent_snapshot(
                run_id,
                "reviewer",
                prompt=getattr(reviewer, "last_prompt", None),
                response=getattr(reviewer, "last_response", None) or review,
                context={"analysis_type": analysis_type, "business_objective": business_objective},
                verdicts=review,
            )
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
            "qa_budget_exceeded": True,
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
        contract = state.get("execution_contract", {}) or {}
        qa_view = state.get("qa_view") or (state.get("contract_views") or {}).get("qa_view")
        contract_min = state.get("execution_contract_min") or state.get("contract_min") or _load_json_safe("data/contract_min.json") or {}
        if isinstance(qa_view, dict) and qa_view:
            qa_context = dict(qa_view)
            contract_source = "qa_view"
        elif isinstance(contract_min, dict) and contract_min:
            qa_context = dict(contract_min)
            contract_source = "contract_min"
        else:
            qa_context = {}
            contract_source = "fallback"
        qa_context["_contract_source"] = contract_source
        ml_data_path = state.get("ml_data_path")
        if not ml_data_path and isinstance(qa_context, dict):
            artifact_reqs = qa_context.get("artifact_requirements")
            if isinstance(artifact_reqs, dict):
                outputs = artifact_reqs.get("required_outputs") or artifact_reqs.get("required_files") or []
                if isinstance(outputs, list):
                    for path in outputs:
                        if isinstance(path, str) and path.endswith(".csv") and "clean" in path.lower():
                            ml_data_path = path
                            break
        qa_context["ml_data_path"] = ml_data_path or "data/cleaned_data.csv"
        if contract_source == "qa_view":
            try:
                qa_len = len(json.dumps(qa_context, ensure_ascii=True))
                print(f"Using QA_VIEW_CONTEXT length={qa_len}")
                if run_id:
                    log_run_event(run_id, "qa_view_context", {"length": qa_len})
            except Exception:
                pass
        allowed_columns = _resolve_allowed_columns_for_gate(state, contract, qa_context)
        allowed_patterns = _resolve_allowed_patterns_for_gate(contract)
        static_facts = collect_static_qa_facts(code)
        static_result = run_static_qa_checks(code, qa_context, static_facts)
        if static_result and static_result.get("status") == "REJECTED":
            status = "REJECTED"
            feedback = static_result.get("feedback") or "Static QA gate failures detected."
            failed_gates = static_result.get("failed_gates", [])
            required_fixes = static_result.get("required_fixes", [])
            current_history.append(f"QA_STATIC_REJECTED: {feedback}")
            print(f"QA Verdict: {status}")
            streak = int(state.get("qa_reject_streak", 0) or 0) + 1
            gate_context = {
                "source": "qa_reviewer",
                "status": status,
                "feedback": feedback,
                "failed_gates": failed_gates,
                "required_fixes": _expand_required_fixes(required_fixes, failed_gates),
            }
            fix_block = _build_fix_instructions(gate_context["required_fixes"])
            if fix_block:
                gate_context["edit_instructions"] = fix_block
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
                "review_verdict": "REJECTED",
                "review_feedback": feedback,
                "feedback_history": current_history,
                "last_gate_context": gate_context,
                "qa_reject_streak": streak,
                "budget_counters": counters,
            }

        try:
            qa_result = qa_reviewer.review_code(code, strategy, business_objective, qa_context)
        except TypeError:
            qa_result = qa_reviewer.review_code(code, strategy, business_objective)

        status = qa_result['status']
        feedback = qa_result['feedback']
        failed_gates = qa_result.get('failed_gates', [])
        required_fixes = qa_result.get('required_fixes', [])

        hard_failures = qa_result.get("hard_failures") or []
        if status == "REJECTED" and not hard_failures:
            feedback = f"QA_LLM_NONBLOCKING_WARNING: {feedback}"
            current_history.append(feedback)
            status = "APPROVE_WITH_WARNINGS"
            failed_gates = []
            required_fixes = []

        print(f"QA Verdict: {status}")

        # Update QA Streak (only deterministic rejects)
        streak = state.get('qa_reject_streak', 0)
        if status in {"APPROVED", "APPROVE_WITH_WARNINGS"}:
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

        review_verdict = status if status in {"APPROVED", "APPROVE_WITH_WARNINGS"} else "APPROVED"
        if run_id:
            log_run_event(run_id, "qa_reviewer_complete", {"status": status})
        if run_id:
            log_agent_snapshot(
                run_id,
                "qa_reviewer",
                prompt=getattr(qa_reviewer, "last_prompt", None),
                response=getattr(qa_reviewer, "last_response", None) or qa_result,
                context={"business_objective": business_objective},
                verdicts=qa_result,
            )
        return {
            "review_verdict": review_verdict,
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
    run_id = state.get("run_id")
    code = state.get("generated_code", "")
    evaluation_spec = state.get("evaluation_spec") or (state.get("execution_contract", {}) or {}).get("evaluation_spec") or {}
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
    strategy = state.get("selected_strategy", {}) or {}
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

    flags = _resolve_eval_flags(evaluation_spec)
    allowed_columns = _resolve_allowed_columns_for_gate(state, contract, evaluation_spec)
    allowed_patterns = _resolve_allowed_patterns_for_gate(contract)
    issues = ml_quality_preflight(code, evaluation_spec, allowed_columns, allowed_patterns)
    if isinstance(evaluation_spec, dict):
        obj_type = str(evaluation_spec.get("objective_type") or "").lower()
        validation_policy = evaluation_spec.get("validation_policy") or {}
        require_cv = bool(validation_policy.get("require_cv")) if isinstance(validation_policy, dict) else False
        if (not require_cv or not flags["requires_supervised_split"]) and "CROSS_VALIDATION_REQUIRED" in issues:
            issues = [i for i in issues if i != "CROSS_VALIDATION_REQUIRED"]
        if flags["requires_time_series_split"] and "TIME_SERIES_SPLIT_REQUIRED" not in issues:
            pass
        qa_gates = _extract_qa_gate_names(evaluation_spec.get("qa_gates") or evaluation_spec.get("gates") or [])
        require_variance = "target_variance_guard" in qa_gates or obj_type in {"predictive", "prescriptive"}
        if not require_variance and "TARGET_VARIANCE_GUARD" in issues:
            issues = [i for i in issues if i != "TARGET_VARIANCE_GUARD"]
    if not flags["requires_target"]:
        issues = [
            i
            for i in issues
            if i not in {"TARGET_NOT_IN_X", "TARGET_VARIANCE_GUARD", "CROSS_VALIDATION_REQUIRED", "TIME_SERIES_SPLIT_REQUIRED"}
        ]
    required_columns = contract.get("required_columns") or strategy.get("required_columns", []) or []
    feature_availability = contract.get("feature_availability", []) or []
    evaluation_spec = state.get("evaluation_spec") or contract.get("evaluation_spec") or {}
    pre_decision_cols = [
        item.get("column")
        for item in feature_availability
        if isinstance(item, dict) and str(item.get("availability", "")).lower() == "pre-decision"
    ]
    required_outputs = contract.get("required_outputs", []) or []
    if not flags["requires_row_scoring"]:
        required_outputs = [
            path for path in required_outputs if path != "data/scored_rows.csv"
        ]
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
    if "alignment_check.json" in (code or "") and "\"requirements\"" not in (code or "") and "'requirements'" not in (code or ""):
        issues.append("ALIGNMENT_REQUIREMENTS_MISSING")
    segment_features = _extract_named_string_list(code, ["SEGMENT_FEATURES", "segment_features"])
    segmentation_required = False
    if isinstance(evaluation_spec, dict):
        segmentation_required = bool((evaluation_spec.get("segmentation") or {}).get("required"))
    if segmentation_required:
        if not segment_features:
            issues.append("SEGMENT_FEATURES_MISSING")
        else:
            expected_seg = pre_decision_cols or (evaluation_spec.get("segmentation") or {}).get("features") or []
            missing_seg = [col for col in expected_seg if col and col not in segment_features]
            if missing_seg:
                issues.append("SEGMENT_FEATURES_INCOMPLETE")
    model_features = _extract_named_string_list(code, ["MODEL_FEATURES", "model_features"])
    decision_vars = [
        item.get("column")
        for item in feature_availability
        if isinstance(item, dict) and str(item.get("availability", "")).lower() == "decision"
    ]
    model_required = False
    if isinstance(evaluation_spec, dict):
        obj_type = str(evaluation_spec.get("objective_type") or "").lower()
        model_required = obj_type in {"predictive", "prescriptive"}
    if decision_vars and model_required:
        if not model_features:
            issues.append("MODEL_FEATURES_MISSING")
        else:
            if not any(var in model_features for var in decision_vars if var):
                issues.append("MODEL_FEATURES_INCOMPLETE")
    canonical_mismatches: List[Dict[str, str]] = []
    if segment_features:
        canonical_mismatches.extend(_find_canonical_mismatches(segment_features, required_columns))
    if model_features:
        canonical_mismatches.extend(_find_canonical_mismatches(model_features, required_columns + decision_vars))
    if canonical_mismatches:
        issues.append("CANONICAL_MAPPING_REQUIRED")
    derived_targets = _collect_derived_targets(contract)
    if derived_targets:
        code_lower = code.lower()
        has_derived_log = "derived_target:" in code_lower or "derived_target" in code_lower
        has_guard = any(_has_derived_target_guard(code_lower, name) for name in derived_targets)
        has_assignment = any(_code_assigns_df_column(code, name) for name in derived_targets)
        if not (has_derived_log or has_guard or has_assignment):
            issues.append("DERIVED_TARGET_REQUIRED")
    unknown_cols: List[str] = []
    if "UNKNOWN_COLUMNS_REFERENCED" in issues and allowed_columns:
        unknown_cols = _detect_unknown_columns(code, allowed_columns, allowed_patterns)
    forbidden_assignments: List[str] = []
    if "DF_COLUMN_ASSIGNMENT_FORBIDDEN" in issues and allowed_columns:
        forbidden_assignments = _detect_forbidden_df_assignments(code, allowed_columns, allowed_patterns)
    if run_id and issues:
        payload = {"issues": issues}
        if unknown_cols:
            payload["unknown_columns"] = unknown_cols
        if forbidden_assignments:
            payload["forbidden_df_assignments"] = forbidden_assignments
        log_run_event(run_id, "ml_preflight_issues", payload)
    if issues:
        expanded = _expand_required_fixes(issues, issues)
        feedback = f"ML_PREFLIGHT_MISSING: {', '.join(issues)}"
        if missing_outputs:
            feedback += f" | Missing output refs: {missing_outputs}"
        if col_coverage.get("hits") is not None and required_columns:
            feedback += f" | Required columns hits: {col_coverage.get('hits')}"
        if canonical_mismatches:
            feedback += f" | Canonical mismatches: {canonical_mismatches}"
        if unknown_cols:
            preview = unknown_cols[:10]
            feedback += f" | Unknown columns ({len(unknown_cols)}): {preview}"
        if forbidden_assignments:
            preview = forbidden_assignments[:10]
            feedback += f" | Forbidden df column assignments ({len(forbidden_assignments)}): {preview}. SOLUTION: Use df.assign() instead of df['col']=value. Pattern: df = df.assign(new_col=expression). Example: df = df.assign(derived_target=(df['source_col']==condition).astype(int))"
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
        if run_id:
            outputs_state = _collect_outputs_state(_resolve_required_outputs(contract))
            reviewer_reasons = _normalize_reason_tags(feedback, issues)
            next_actions = _suggest_next_actions(issues, outputs_state["missing"], reviewer_reasons, [])
            entry = _build_ml_iteration_journal_entry(
                state,
                preflight_issues=issues,
                runtime_error=None,
                outputs_present=outputs_state["present"],
                outputs_missing=outputs_state["missing"],
                reviewer_verdict="REJECTED",
                reviewer_reasons=reviewer_reasons,
                qa_verdict=None,
                qa_reasons=[],
                next_actions=next_actions,
                stage="preflight",
            )
            written_ids = _append_ml_iteration_journal(
                run_id,
                entry,
                state.get("ml_journal_written_ids"),
            )
        else:
            written_ids = state.get("ml_journal_written_ids")
        return {
            "ml_preflight_failed": True,
            "feedback_history": history,
            "last_gate_context": gate_context,
            "review_verdict": "REJECTED",
            "review_feedback": feedback,
            "ml_journal_written_ids": written_ids,
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
        if run_id:
            outputs_state = _collect_outputs_state(_resolve_required_outputs(contract))
            reviewer_reasons = _normalize_reason_tags(feedback, ["JSON_SERIALIZATION_GUARD"])
            next_actions = _suggest_next_actions(["JSON_SERIALIZATION_GUARD"], outputs_state["missing"], reviewer_reasons, [])
            entry = _build_ml_iteration_journal_entry(
                state,
                preflight_issues=["JSON_SERIALIZATION_GUARD"],
                runtime_error=None,
                outputs_present=outputs_state["present"],
                outputs_missing=outputs_state["missing"],
                reviewer_verdict="REJECTED",
                reviewer_reasons=reviewer_reasons,
                qa_verdict=None,
                qa_reasons=[],
                next_actions=next_actions,
                stage="preflight",
            )
            written_ids = _append_ml_iteration_journal(
                run_id,
                entry,
                state.get("ml_journal_written_ids"),
            )
        else:
            written_ids = state.get("ml_journal_written_ids")
        return {
            "ml_preflight_failed": True,
            "feedback_history": history,
            "last_gate_context": gate_context,
            "review_verdict": "REJECTED",
            "review_feedback": feedback,
            "ml_journal_written_ids": written_ids,
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
    if (
        state.get("review_verdict") == "REJECTED"
        and (state.get("last_gate_context") or {}).get("source") == "ml_preflight"
    ):
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
    exec_start_ts = time.time()
    code = state['generated_code']
    attempt_id = int(state.get("execution_attempt", 0)) + 1

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

    required_outputs = _resolve_required_outputs(contract)
    expected_outputs = _resolve_expected_output_paths(contract)
    _purge_execution_outputs(required_outputs, expected_outputs)

    eval_spec = state.get("evaluation_spec") or (contract.get("evaluation_spec") if isinstance(contract, dict) else {})
    allowed_columns = _resolve_allowed_columns_for_gate(state, contract, eval_spec)
    allowed_patterns = _resolve_allowed_patterns_for_gate(contract)
    preflight_issues = ml_quality_preflight(code, eval_spec, allowed_columns, allowed_patterns)
    unknown_cols: List[str] = []
    if "UNKNOWN_COLUMNS_REFERENCED" in preflight_issues and allowed_columns:
        unknown_cols = _detect_unknown_columns(code, allowed_columns, allowed_patterns)
    forbidden_assignments: List[str] = []
    if "DF_COLUMN_ASSIGNMENT_FORBIDDEN" in preflight_issues and allowed_columns:
        forbidden_assignments = _detect_forbidden_df_assignments(code, allowed_columns, allowed_patterns)
    fatal_preflight = {
        "SYNTHETIC_DATA_DETECTED",
        "DATAFRAME_LITERAL_OVERWRITE",
        "UNKNOWN_COLUMNS_REFERENCED",
        "DF_COLUMN_ASSIGNMENT_FORBIDDEN",
    }
    fatal_hits = [issue for issue in preflight_issues if issue in fatal_preflight]
    if fatal_hits:
        msg = f"ML_PREFLIGHT_BLOCK: {', '.join(fatal_hits)}"
        if unknown_cols:
            preview = unknown_cols[:10]
            msg += f" | Unknown columns ({len(unknown_cols)}): {preview}"
        if forbidden_assignments:
            preview = forbidden_assignments[:10]
            msg += f" | Forbidden df column assignments ({len(forbidden_assignments)}): {preview}. SOLUTION: Use df.assign() instead of df['col']=value. Pattern: df = df.assign(new_col=expression). Example: df = df.assign(derived_target=(df['source_col']==condition).astype(int))"
        snippet_tokens = [f"'{col}'" for col in (unknown_cols + forbidden_assignments)[:10]]
        snippets = _collect_violation_snippets(code, snippet_tokens)
        if snippets:
            msg += f" | Code refs: {snippets[:5]}"
        fh = list(state.get("feedback_history", []))
        fh.append(msg)
        if run_id:
            payload = {"issues": fatal_hits}
            if unknown_cols:
                payload["unknown_columns"] = unknown_cols
            if forbidden_assignments:
                payload["forbidden_df_assignments"] = forbidden_assignments
            if snippets:
                payload["snippets"] = snippets
            log_run_event(run_id, "ml_preflight_blocked", payload)
        return {
            "error_message": msg,
            "execution_output": msg,
            "feedback_history": fh,
            "ml_skipped_reason": "ML_PREFLIGHT_BLOCKED",
            "budget_counters": counters,
        }

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

        # Determine ML timeout based on dataset scale (P2.2)
        dataset_scale = state.get("dataset_scale") or (state.get("dataset_scale_hints") or {}).get("scale")
        if dataset_scale == "small":
            ml_timeout = ML_TIMEOUT_SMALL_S
        elif dataset_scale == "large":
            ml_timeout = ML_TIMEOUT_LARGE_S
        else:
            ml_timeout = ML_TIMEOUT_MEDIUM_S  # Default to medium

        for sb_attempt in range(2):
          try:
            with create_sandbox_with_retry(Sandbox, max_attempts=2, run_id=run_id, step="ml_engineer") as sandbox:
                if not hasattr(sandbox, "run_code"):
                    raise RuntimeError(
                        "E2B Sandbox missing run_code. Ensure we are using e2b_code_interpreter.Sandbox (CodeSandbox)."
                    )
                step_name = "ml_engineer"
                run_root = f"/home/user/run/{run_id}/{step_name}/attempt_{attempt_id}"
                print("Installing dependencies in Sandbox...")
                pkg_sets = get_sandbox_install_packages(required_deps)
                base_cmd = "pip install -q " + " ".join(pkg_sets["base"])
                run_cmd_with_retry(sandbox, base_cmd, retries=2)
                if pkg_sets["extra"]:
                    extra_cmd = "pip install -q " + " ".join(pkg_sets["extra"])
                    run_cmd_with_retry(sandbox, extra_cmd, retries=2)
                run_cmd_with_retry(sandbox, f"rm -rf {run_root}", retries=2)
                run_cmd_with_retry(sandbox, f"mkdir -p {run_root}/static/plots", retries=2)
                run_cmd_with_retry(sandbox, f"mkdir -p {run_root}/data", retries=2)

                # P2.1: Use canonical cleaned path + aliases (P2.2: run_cmd_with_retry)
                local_csv = state.get("ml_data_path") or "data/cleaned_data.csv"
                if not os.path.exists(local_csv) and local_csv != "data/cleaned_data.csv":
                    local_csv = "data/cleaned_data.csv"

                # Upload cleaned data (P2.1: Canonical path)
                remote_clean_abs = canonical_abs(run_root, CANONICAL_CLEANED_REL)

                if os.path.exists(local_csv):
                    with open(local_csv, "rb") as f:
                        sandbox.files.write(remote_clean_abs, f)
                    print(f"Data uploaded to {CANONICAL_CLEANED_REL}")

                # P2.1: Create aliases for cleaned data (run_cmd_with_retry for resilience)
                cleaned_alias_commands = build_symlink_or_copy_commands(
                    run_root,
                    canonical_rel=CANONICAL_CLEANED_REL,
                    aliases=COMMON_CLEANED_ALIASES
                )
                if cleaned_alias_commands:
                    print(f"Creating {len(cleaned_alias_commands)} cleaned data aliases...")
                    full_cmd = " && ".join(cleaned_alias_commands)
                    run_cmd_with_retry(sandbox, f"sh -lc 'cd {run_root} && {full_cmd}'")
                    print(f"SANDBOX_INPUT_CANONICAL: {CANONICAL_CLEANED_REL}")
                    print(f"SANDBOX_INPUT_ALIASES: {COMMON_CLEANED_ALIASES}")

                # P2.1: Upload manifest to canonical path and root
                local_manifest = "data/cleaning_manifest.json"
                remote_manifest_abs = canonical_abs(run_root, CANONICAL_MANIFEST_REL)
                remote_manifest_root_abs = os.path.join(run_root, "cleaning_manifest.json").replace("\\", "/")

                if os.path.exists(local_manifest):
                    with open(local_manifest, "rb") as f:
                        manifest_content = f.read()
                        sandbox.files.write(remote_manifest_abs, manifest_content)
                        sandbox.files.write(remote_manifest_root_abs, manifest_content)
                    print(f"Manifest uploaded to {CANONICAL_MANIFEST_REL} and root")

                # P2.1: Patch code with placeholders and explicit absolute entries
                code = patch_placeholders(code, data_rel=CANONICAL_CLEANED_REL, manifest_rel=CANONICAL_MANIFEST_REL)

                # Explicit substitutions to ensure absolute paths and avoid .//
                if local_csv:
                     # Replace generic data paths with remote_clean_abs
                     # Use pattern matching or strict order to avoid double replacement
                     code = code.replace("./data/cleaned_data.csv", remote_clean_abs)
                     code = code.replace("'data/cleaned_data.csv'", f"'{remote_clean_abs}'")
                     code = code.replace('"data/cleaned_data.csv"', f'"{remote_clean_abs}"')

                     # Only if local_csv is something else like 'data/input.csv'
                     if local_csv not in ["data/cleaned_data.csv", "./data/cleaned_data.csv"]:
                         code = code.replace(local_csv, remote_clean_abs)
                         if not local_csv.startswith("./"):
                             code = code.replace(f"./{local_csv}", remote_clean_abs)

                # Manifest patching -> Point to ROOT manifest as requested
                code = code.replace("./data/cleaning_manifest.json", remote_manifest_root_abs)
                code = code.replace("'data/cleaning_manifest.json'", f"'{remote_manifest_root_abs}'")
                code = code.replace('"data/cleaning_manifest.json"', f'"{remote_manifest_root_abs}"')

                # Inject robust prelude
                working_dir_injection = (
                    "import os\n"
                    f"os.makedirs(r\"{run_root}\", exist_ok=True)\n"
                    f"os.chdir(r\"{run_root}\")\n"
                    f"MANIFEST_PATH = r\"{remote_manifest_root_abs}\"\n"
                    f"CLEANED_CSV_PATH = r\"{remote_clean_abs}\"\n"
                )
                code = working_dir_injection + code

                # Persist executed ML script for traceability
                try:
                    os.makedirs("artifacts", exist_ok=True)
                    with open(os.path.join("artifacts", "ml_engineer_last.py"), "w", encoding="utf-8") as f_art:
                        f_art.write(code)
                except Exception as artifact_err:
                    print(f"Warning: failed to persist ml_engineer_last.py: {artifact_err}")
                if run_id:
                    run_dir = get_run_dir(run_id)
                    if run_dir:
                        try:
                            base = os.path.join(run_dir, "agents", "ml_engineer", f"iteration_{attempt_id}")
                            os.makedirs(base, exist_ok=True)
                            with open(os.path.join(base, "script_executed.py"), "w", encoding="utf-8") as f_exec:
                                f_exec.write(code)
                        except Exception:
                            pass

                print("Running code in Sandbox...")
                execution = run_code_with_optional_timeout(sandbox, code, timeout_s=ml_timeout)

                stdout_text = "\n".join(execution.logs.stdout or [])
                stderr_text = "\n".join(execution.logs.stderr or [])
                output = ""
                if stdout_text:
                    output += "\nSTDOUT:\n" + stdout_text
                if stderr_text:
                    output += "\nSTDERR:\n" + stderr_text

                if execution.error:
                    output += f"\n\nEXECUTION ERROR:\n{execution.error.name}: {execution.error.value}\n{execution.error.traceback}"

                downloaded_paths = []

                # Robust Artifact Download (P0 Fix) using centralized helper
                # Use shell command that always succeeds even if no files found
                ls_proc = run_cmd_with_retry(sandbox, f"sh -lc 'ls -1 {run_root}/static/plots/*.png 2>/dev/null || true'", retries=2)

                if ls_proc.exit_code == 0:
                    os.makedirs("static/plots", exist_ok=True)
                    # Filter empty strings from split
                    plot_files = [p for p in ls_proc.stdout.strip().split('\n') if p]

                    if not plot_files:
                        print("Info: No plots generated by the script.")
                    else:
                        for remote_plot in plot_files:
                            if remote_plot.endswith('.png'):
                                # Use centralized download helper for binary files
                                content = safe_download_bytes(sandbox, remote_plot)
                                if content and len(content) > 0:
                                    local_name = os.path.basename(remote_plot)
                                    local_path = os.path.join("static/plots", local_name)
                                    with open(local_path, "wb") as f_local:
                                        f_local.write(content)
                                    downloaded_paths.append(local_path)
                                    print(f"Downloaded plot: {local_name}")
                                else:
                                    print(f"Warning: Failed to download {remote_plot}")
                else:
                    print(f"Warning: Plot listing failed (Exit Code {ls_proc.exit_code})")

                # Download required outputs per contract (beyond plots) using centralized helper
                req_outputs = required_outputs or state.get("execution_contract", {}).get("required_outputs", []) or []
                for pattern in req_outputs:
                    if not pattern:
                        continue
                    if pattern.startswith("/"):
                        remote_pattern = pattern
                    else:
                        remote_pattern = f"{run_root}/{pattern}"
                    list_cmd = f"sh -lc 'ls -1 {remote_pattern} 2>/dev/null || true'"
                    lst = sandbox.commands.run(list_cmd)
                    if lst.exit_code != 0:
                        continue
                    files = [p for p in lst.stdout.strip().split("\n") if p]
                    for remote_path in files:
                        if not remote_path:
                            continue
                        content = safe_download_bytes(sandbox, remote_path)
                        if content is not None:
                            if remote_path.startswith(run_root):
                                local_path = remote_path[len(run_root):].lstrip("/")
                            elif remote_path.startswith("/home/user/"):
                                local_path = remote_path[len("/home/user/"):].lstrip("/")
                            else:
                                local_path = remote_path.lstrip("/")
                            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
                            with open(local_path, "wb") as f_local:
                                f_local.write(content)
                            downloaded_paths.append(local_path)
                            print(f"Downloaded required output: {local_path}")
                        else:
                            print(f"Warning: failed to download required output {remote_path}")

                # Download optional plot insights if present using centralized helper
                for rel_path in ["data/plot_insights.json"]:
                    if rel_path.startswith("/"):
                        remote_path = rel_path
                    else:
                        remote_path = f"{run_root}/{rel_path}"
                    list_cmd = f"sh -lc 'ls -1 {remote_path} 2>/dev/null || true'"
                    lst = sandbox.commands.run(list_cmd)
                    if lst.exit_code != 0:
                        continue
                    files = [p for p in lst.stdout.strip().split("\n") if p]
                    for remote_opt in files:
                        if not remote_opt:
                            continue
                        content = safe_download_bytes(sandbox, remote_opt)
                        if content is not None:
                            if remote_opt.startswith(run_root):
                                local_path = remote_opt[len(run_root):].lstrip("/")
                            elif remote_opt.startswith("/home/user/"):
                                local_path = remote_opt[len("/home/user/"):].lstrip("/")
                            else:
                                local_path = remote_opt.lstrip("/")
                            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
                            with open(local_path, "wb") as f_local:
                                f_local.write(content)
                            downloaded_paths.append(local_path)
                            print(f"Downloaded optional output: {local_path}")
                        else:
                            print(f"Warning: failed to download optional output {remote_opt}")
                list_cmd = f"sh -lc 'cd {run_root} && find . -maxdepth 5 -type f -printf \"%p\\t%s\\n\" 2>/dev/null'"
                outputs_listing = []
                try:
                    listing_proc = sandbox.commands.run(list_cmd)
                    if listing_proc.exit_code == 0:
                        outputs_listing = [line for line in listing_proc.stdout.splitlines() if line.strip()]
                except Exception:
                    outputs_listing = []
                if run_id:
                    log_sandbox_attempt(
                        run_id,
                        step_name,
                        attempt_id,
                        code=code,
                        stdout=stdout_text,
                        stderr=stderr_text,
                        outputs_listing=outputs_listing,
                        downloaded_paths=downloaded_paths,
                        exit_code=getattr(execution, "exit_code", None),
                        error_tail=(execution.error.traceback if execution.error else None),
                    )

            break  # Sandbox execution successful, exit retry loop
          except Exception as e:
            if sb_attempt == 0 and is_transient_sandbox_error(e):
                print(f"SANDBOX_RETRY step=ml_engineer attempt={sb_attempt+1} err={e}")
                continue
            raise

    except Exception as e:
        output = f"Sandbox Execution Failed: {e}"
        print(output)

    # Calculate Visuals State locally
    import glob
    plots_local = glob.glob("static/plots/*.png")

    fallback_plots_local = []
    # Fallback plots removed per user request (User prefers no plots over generic ones)

    has_partial_visuals = len(plots_local) > 0

    print(f"Execution finished. Plots generated: {len(plots_local)}")

    eval_spec = state.get("evaluation_spec") or (contract.get("evaluation_spec") if isinstance(contract, dict) else {})

    # CRITICAL: Use runtime dialect from state first (cleaning_manifest), fallback to contract
    dialect = _resolve_artifact_gate_dialect(state, contract)
    csv_sep = dialect["sep"]
    csv_decimal = dialect["decimal"]
    csv_encoding = dialect["encoding"]

    artifact_issues = _artifact_alignment_gate(
        cleaned_path="data/cleaned_full.csv" if os.path.exists("data/cleaned_full.csv") else "data/cleaned_data.csv",
        scored_path="data/scored_rows.csv",
        contract=contract,
        evaluation_spec=eval_spec,
        csv_sep=csv_sep,
        csv_decimal=csv_decimal,
        csv_encoding=csv_encoding,
    )

    # ← NEW: UNIVERSAL soft-warn logic for artifact alignment issues
    # Works for ANY artifact type (scored_rows, metrics, alignment_check, etc.)
    # No hardcoding of specific issues or objectives
    if artifact_issues:
        # Track issue history across iterations to enable learning
        issue_history = state.get("artifact_alignment_issue_history", [])

        # Count occurrences of each specific issue across all previous iterations
        issue_occurrence_counts = {}
        for past_issues in issue_history:
            for issue in past_issues:
                issue_occurrence_counts[issue] = issue_occurrence_counts.get(issue, 0) + 1

        # Determine max occurrence count for current issues (to decide soft vs hard block)
        max_occurrence = 0
        for issue in artifact_issues:
            count = issue_occurrence_counts.get(issue, 0)
            if count > max_occurrence:
                max_occurrence = count

        # UNIVERSAL POLICY: Soft warn for first 2 occurrences, hard block on 3rd+
        # This gives ML Engineer 2 chances to learn and fix before failing
        SOFT_WARN_THRESHOLD = 2  # Can be made configurable per strategy if needed

        if max_occurrence < SOFT_WARN_THRESHOLD:
            # Soft warning: Add to feedback but DON'T block execution
            issue_text = "; ".join(artifact_issues)
            iteration_num = max_occurrence + 1
            warning_msg = (
                f"⚠️ ARTIFACT_ALIGNMENT_WARNING (iteration {iteration_num}/{SOFT_WARN_THRESHOLD + 1}): {issue_text}\n"
                f"   Note: These artifacts will be required for final validation. "
                f"Please generate them in the next iteration."
            )

            # Add to feedback history so ML Engineer receives this in next prompt
            if "feedback_history" not in state:
                state["feedback_history"] = []
            state["feedback_history"].append(warning_msg)

            # Print to logs for visibility
            print(warning_msg)

        else:
            # Hard block: This is the 3rd+ occurrence, fail the execution
            issue_text = "; ".join(artifact_issues)
            output = f"{output}\nEXECUTION ERROR: ARTIFACT_ALIGNMENT_GUARD: {issue_text}"

        # Track current issues for next iteration (universal tracking)
        if "artifact_alignment_issue_history" not in state:
            state["artifact_alignment_issue_history"] = []
        state["artifact_alignment_issue_history"].append(artifact_issues)

    stale_outputs = _find_stale_outputs(required_outputs, exec_start_ts)
    if stale_outputs:
        output = f"{output}\nEXECUTION ERROR: STALE_OUTPUTS: {stale_outputs}"

    content_issues, content_diagnostics = _validate_artifact_content(state)
    if content_issues:
        issue_text = ", ".join(content_issues)
        output = f"{output}\nEXECUTION ERROR: ARTIFACT_CONTENT_INVALID: {issue_text}"

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

    outputs_valid = not bool(artifact_issues or stale_outputs or content_issues or error_in_output)
    if not outputs_valid:
        # Preserve produced outputs for auditing and contract validation.
        # Outputs are purged at the start of the next execution attempt (and at run start) to prevent contamination.
        has_partial_visuals = False
    if run_id:
        update_sandbox_attempt(
            run_id,
            "ml_engineer",
            attempt_id,
            artifacts_valid=outputs_valid,
            artifact_issues=artifact_issues,
            stale_outputs=stale_outputs,
            content_issues=content_issues,
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
    contract = state.get("execution_contract", {}) or {}
    deliverables = _resolve_contract_deliverables(contract)
    output_contract = deliverables if isinstance(deliverables, list) and deliverables else contract.get("required_outputs", [])
    oc_report = check_required_outputs(output_contract)
    try:
        os.makedirs("data", exist_ok=True)
        dump_json("data/output_contract_report.json", oc_report)
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
    artifact_paths = []
    if isinstance(oc_report, dict):
        artifact_paths.extend(oc_report.get("present", []))
    if plots_local:
        artifact_paths.extend(plots_local)
    for extra_path in [
        "data/strategy_spec.json",
        "data/plan.json",
        "data/evaluation_spec.json",
        "data/plot_insights.json",
    ]:
        if os.path.exists(extra_path):
            artifact_paths.append(extra_path)
    # De-duplicate while preserving order
    deduped = []
    seen = set()
    for item in artifact_paths:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    artifact_paths = deduped
    deliverables = _resolve_contract_deliverables(contract)
    artifact_index = _build_artifact_index(artifact_paths, deliverables if isinstance(deliverables, list) else None)
    try:
        os.makedirs("data", exist_ok=True)
        dump_json("data/produced_artifact_index.json", artifact_index)
    except Exception as idx_err:
        print(f"Warning: failed to persist produced_artifact_index.json: {idx_err}")

    attempt_score = _score_attempt(outputs_valid, oc_report, content_issues, artifact_paths)
    attempt_valid = bool(outputs_valid and not oc_report.get("missing") and not content_issues)

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
        "produced_artifact_index": artifact_index,
        "artifact_paths": artifact_paths,
        "artifact_content_issues": content_issues,
        "artifact_content_diagnostics": content_diagnostics,
        "last_attempt_score": attempt_score,
        "last_attempt_valid": attempt_valid,
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
    best_score = state.get("best_attempt_score")
    if attempt_valid and (best_score is None or attempt_score > float(best_score)):
        dest = _snapshot_best_attempt(
            attempt_id=attempt_id,
            artifact_paths=artifact_paths,
            output_contract_report=oc_report,
            artifact_index=artifact_index,
            execution_output=output,
            plots_local=plots_local,
            diagnostics=content_diagnostics,
        )
        result["best_attempt_score"] = attempt_score
        result["best_attempt_id"] = attempt_id
        result["best_attempt_dir"] = dest
        result["best_attempt_artifact_index"] = artifact_index
        result["best_attempt_output_contract_report"] = oc_report
        result["best_attempt_execution_output"] = output
        result["best_attempt_plots"] = plots_local
    return result

def retry_handler(state: AgentState) -> AgentState:
    print("--- [!] Iteration Retry Triggered ---")

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
    if "Traceback" in execution_output or "EXECUTION ERROR" in execution_output:
         print("Reviewer: Critical Execution Error detected. Requesting fix.")
         return {
             "review_verdict": "NEEDS_IMPROVEMENT",
             "review_feedback": f"Execution failed with traceback: {execution_output[-500:]}",
             "feedback_history": state.get("feedback_history", []) # Maintain history
         }

    strategy = state.get('selected_strategy', {}) or {}
    strategy_context = f"Strategy: {strategy.get('title')}\nType: {strategy.get('analysis_type')}\nRules: {strategy.get('reasoning')}"
    business_objective = state.get('business_objective', '')
    evaluation_spec = state.get("evaluation_spec") or (state.get("execution_contract", {}) or {}).get("evaluation_spec")

    eval_result = reviewer.evaluate_results(execution_output, business_objective, strategy_context, evaluation_spec)

    status = eval_result.get('status', 'APPROVED')
    feedback = eval_result.get('feedback', '')
    retry_worth_it = eval_result.get("retry_worth_it") if isinstance(eval_result, dict) else None

    new_history = list(state.get('feedback_history', []))
    has_deterministic_error = "Traceback" in execution_output or "EXECUTION ERROR" in execution_output
    downgraded = False
    if status == "NEEDS_IMPROVEMENT":
        if has_deterministic_error:
            new_history.append(f"RESULT EVALUATION FEEDBACK: {feedback}")
        else:
            # Only downgrade to APPROVE_WITH_WARNINGS if retry is explicitly NOT worth it.
            # Note: Metric iteration is DISABLED. Metric-only issues are downgraded later via _classify_iteration_type.
            warning = f"REVIEWER_LLM_NONBLOCKING_WARNING: {feedback}"
            new_history.append(warning)
            feedback = warning
            if retry_worth_it is False:
                status = "APPROVE_WITH_WARNINGS"
                downgraded = True
            # else: keep NEEDS_IMPROVEMENT - will be classified as compliance or downgraded if metric-only
    print(f"ITER_EVAL status={status} retry_worth_it={retry_worth_it} downgraded={downgraded} reason={'no_traceback_retry_not_worth' if downgraded else 'none'}")

    # Case alignment QA gate (optional if required by contract/spec)
    contract = state.get("execution_contract", {}) or {}
    skip_reason = _case_alignment_skip_reason(contract, evaluation_spec)
    case_alignment_required = _should_run_case_alignment(contract, evaluation_spec)
    case_report: Dict[str, Any] = {
        "status": "SKIP",
        "failures": [],
        "metrics": {},
        "reason": "case alignment not required",
    }
    case_history = list(state.get("case_alignment_history", []) or [])
    if skip_reason:
        case_alignment_required = False
        case_report = {
            "status": "SKIPPED",
            "mode": "case_level",
            "metrics": {"case_count": None},
            "thresholds": {},
            "failures": [],
            "explanation": f"Case alignment skipped: {skip_reason}.",
            "skip_reason": skip_reason,
        }
    elif case_alignment_required:
        data_paths = []
        if os.path.exists("data/cleaned_full.csv"):
            data_paths.append("data/cleaned_full.csv")
        if os.path.exists("data/cleaned_data.csv"):
            data_paths.append("data/cleaned_data.csv")
        if os.path.exists("data/scored_rows.csv"):
            data_paths.append("data/scored_rows.csv")
        case_report = build_case_alignment_report(
            contract=contract,
            case_summary_path="data/case_summary.csv",
            weights_path="data/weights.json",
            data_paths=data_paths,
            scored_rows_path="data/scored_rows.csv",
        )
        try:
            metrics = case_report.get("metrics", {}) if isinstance(case_report, dict) else {}
            violations = metrics.get("adjacent_refscore_violations")
            if violations is not None:
                case_history.append(float(violations))
        except Exception:
            pass
    try:
        os.makedirs("data", exist_ok=True)
        dump_json("data/case_alignment_report.json", case_report)
    except Exception as err:
        print(f"Warning: failed to persist case_alignment_report.json: {err}")

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
                dump_json("data/metrics.json", metrics_report)
            except Exception as metrics_err:
                print(f"Warning: failed to persist fallback metrics.json: {metrics_err}")

    # 5.5.1) Metrics schema consistency (mean inside CI)
    if metrics_report:
        ci_issues = validate_metrics_ci_consistency(metrics_report)
        if ci_issues:
            status = "NEEDS_IMPROVEMENT"
            issue_text = ", ".join(ci_issues)
            msg = f"METRICS_SCHEMA_INCONSISTENT: {issue_text}"
            new_history.append(msg)
            feedback = f"{feedback}\n{msg}" if feedback else msg

    # 5.6) Model Metrics Consistency Validation (Informative helper)
    if metrics_report:
        try:
            consistency_result = validate_model_metrics_consistency(metrics_report)
            if not consistency_result["passed"]:
                consistency_msg = f"METRIC_CONSISTENCY_WARNING: {consistency_result['error_message']}"
                print(f"Advice: {consistency_msg}")
                new_history.append(consistency_msg)
                if not feedback:
                    feedback = consistency_msg
                else:
                    feedback = f"{feedback}\n{consistency_msg}"
        except Exception as consistency_err:
            print(f"Warning: model metrics consistency validation failed: {consistency_err}")
    iter_id = int(state.get("iteration_count", 0)) + 1
    saved_iter_artifacts = _persist_iteration_artifacts(iter_id)

    if case_alignment_required and case_report.get("status") == "FAIL":
        feedback = f"CASE_ALIGNMENT_GATE_FAILED: {case_report.get('explanation')}"
        status = "NEEDS_IMPROVEMENT"
        new_history.append(f"CASE_ALIGNMENT_GATE_FAILED: {case_report.get('failures')}")

    # Output contract validation
    contract = state.get("execution_contract", {}) or {}
    deliverables = _resolve_contract_deliverables(contract)
    output_contract = deliverables if isinstance(deliverables, list) and deliverables else contract.get("required_outputs", [])
    oc_report = check_required_outputs(output_contract)
    try:
        os.makedirs("data", exist_ok=True)
        dump_json("data/output_contract_report.json", oc_report)
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
    alignment_requirements = []
    if isinstance(evaluation_spec, dict):
        alignment_requirements = evaluation_spec.get("alignment_requirements") or []
    if not alignment_requirements and isinstance(contract, dict):
        alignment_requirements = contract.get("alignment_requirements", []) or []
    alignment_requirements = _coerce_alignment_requirements(alignment_requirements)
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
                    dump_json("data/alignment_check.json", alignment_check)
                except Exception:
                    pass
            raw_status = str(alignment_check.get("status", "")).upper()
            failure_mode = str(alignment_check.get("failure_mode", "")).lower()
            summary = alignment_check.get("summary") or alignment_check.get("notes") or ""
            if alignment_issues and raw_status == "PASS":
                raw_status = "WARN"
                summary = f"{summary} Alignment evidence missing." if summary else "Alignment evidence missing."
                failure_mode = failure_mode or "format"
                alignment_check["status"] = raw_status
                alignment_check["summary"] = summary
                alignment_check["failure_mode"] = failure_mode
            contract_min = _load_json_safe("data/contract_min.json")
            alignment_check, forbidden_violations = _apply_forbidden_feature_gate(
                alignment_check,
                contract,
                contract_min,
                state.get("generated_code") or state.get("last_generated_code") or "",
            )
            if forbidden_violations:
                alignment_failed_gates.append("forbidden_features_used")
                msg = f"FORBIDDEN_FEATURES_USED: {forbidden_violations}"
                if status != "NEEDS_IMPROVEMENT":
                    status = "NEEDS_IMPROVEMENT"
                new_history.append(msg)
                feedback = f"{feedback}\n{msg}" if feedback else msg
                try:
                    dump_json("data/alignment_check.json", alignment_check)
                except Exception:
                    pass
            data_modes = {"data_limited", "data", "insufficient_data", "data_limitations"}
            method_modes = {"method_choice", "method", "strategy", "approach"}
            msg = f"ALIGNMENT_CHECK_{raw_status}: failure_mode={failure_mode}; summary={summary}"
            if raw_status in {"WARN", "FAIL"}:
                if failure_mode in data_modes:
                    if status != "NEEDS_IMPROVEMENT":
                        status = "APPROVE_WITH_WARNINGS"
                    new_history.append(msg)
                    feedback = f"{feedback}\n{msg}" if feedback else msg
                elif failure_mode in method_modes:
                    if status != "NEEDS_IMPROVEMENT":
                        status = "APPROVE_WITH_WARNINGS"
                    new_history.append(msg)
                    feedback = f"{feedback}\n{msg}" if feedback else msg
                elif raw_status == "FAIL":
                    status = "NEEDS_IMPROVEMENT"
                    alignment_failed_gates.append("alignment_unknown")
                    new_history.append(msg)
                    feedback = f"{feedback}\n{msg}" if feedback else msg
                else:
                    if status != "NEEDS_IMPROVEMENT":
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
            dump_json("data/data_adequacy_report.json", data_adequacy_report)
        except Exception as adequacy_err:
            print(f"Warning: failed to persist data_adequacy_report.json: {adequacy_err}")
    except Exception as adequacy_err:
        print(f"Warning: data adequacy evaluation failed: {adequacy_err}")

    objective_type = None
    if isinstance(evaluation_spec, dict):
        objective_type = evaluation_spec.get("objective_type")
    if not objective_type and isinstance(contract, dict):
        eval_spec_contract = contract.get("evaluation_spec")
        if isinstance(eval_spec_contract, dict):
            objective_type = eval_spec_contract.get("objective_type")
    if not objective_type:
        objective_type = strategy.get("analysis_type")

    primary_metric_snapshot = _extract_primary_metric_snapshot(
        metrics_report=metrics_report,
        weights_report=weights_report,
        objective_type=objective_type,
        evaluation_spec=evaluation_spec if isinstance(evaluation_spec, dict) else None,
        contract=contract if isinstance(contract, dict) else None,
    )
    metric_history = list(state.get("metric_history", []) or [])
    if primary_metric_snapshot:
        primary_metric_snapshot = dict(primary_metric_snapshot)
        primary_metric_snapshot["iteration_id"] = int(state.get("iteration_count", 0)) + 1
        metric_history.append(primary_metric_snapshot)
        metric_history = metric_history[-20:]

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
    hard_qa_gate_reject = False
    qa_view = state.get("qa_view") or (state.get("contract_views") or {}).get("qa_view")
    contract_min = state.get("execution_contract_min") or state.get("contract_min") or _load_json_safe("data/contract_min.json") or {}
    if isinstance(qa_view, dict) and qa_view:
        qa_context = dict(qa_view)
        qa_context["_contract_source"] = "qa_view"
    elif isinstance(contract_min, dict) and contract_min:
        qa_context = dict(contract_min)
        qa_context["_contract_source"] = "contract_min"
    else:
        qa_context = {"_contract_source": "fallback"}
    qa_context["ml_data_path"] = state.get("ml_data_path") or "data/cleaned_data.csv"
    hard_qa_gates: set[str] = set()
    for gate in qa_context.get("qa_gates", []) if isinstance(qa_context, dict) else []:
        if isinstance(gate, dict):
            name = gate.get("name") or gate.get("id")
            severity = str(gate.get("severity") or "").upper()
            if name and severity == "HARD":
                hard_qa_gates.add(str(name).lower())
        elif isinstance(gate, str):
            hard_qa_gates.add(gate.lower())
    if code:
        analysis_type = strategy.get("analysis_type", "predictive")
        review_warnings: List[str] = []
        try:
            ok, counters, err_msg = _consume_budget(state, "reviewer_calls", "max_reviewer_calls", "Reviewer")
            review_counters = counters
            if ok:
                reviewer_view = state.get("reviewer_view") or (state.get("contract_views") or {}).get("reviewer_view")
                if not isinstance(reviewer_view, dict) or not reviewer_view:
                    reviewer_view = build_reviewer_view(
                        contract,
                        _load_json_safe("data/contract_min.json") or {},
                        state.get("artifact_index") or [],
                    )
                review_result = reviewer.review_code(
                    code,
                    analysis_type,
                    "",
                    reviewer_view.get("strategy_summary") if isinstance(reviewer_view, dict) else "",
                    evaluation_spec,
                    reviewer_view=reviewer_view,
                )
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
                qa_result = qa_reviewer.review_code(code, strategy, business_objective, qa_context)
                if qa_result and qa_result.get("status") != "APPROVED":
                    review_warnings.append(
                        f"QA_CODE_AUDIT[{qa_result.get('status')}]: {qa_result.get('feedback')}"
                    )
                if qa_result and qa_result.get("status") == "REJECTED":
                    failed = qa_result.get("failed_gates", []) or []
                    is_static = (
                        qa_result.get("feedback") == "Static QA gate failures detected."
                        or qa_result.get("facts") is not None
                    )
                    if is_static and any(str(g).lower() in hard_qa_gates for g in failed):
                        hard_qa_gate_reject = True
                        audit_rejected = True
            else:
                review_warnings.append(f"QA_CODE_AUDIT_SKIPPED: {err_msg}")
                state["qa_budget_exceeded"] = True
        except Exception as qa_err:
            review_warnings.append(f"QA_CODE_AUDIT_ERROR: {qa_err}")
        if review_warnings:
            warn_text = "\n".join(review_warnings)
            feedback = f"{feedback}\n{warn_text}" if feedback else warn_text
            new_history.append(warn_text)
        if state.get("qa_budget_exceeded"):
            warn_text = "QA_INCOMPLETE: QA budget exceeded; QA audit skipped."
            feedback = f"{feedback}\n{warn_text}" if feedback else warn_text
            new_history.append(warn_text)

    if audit_rejected:
        status = "NEEDS_IMPROVEMENT"
        if feedback:
            feedback = f"{feedback}\nCODE_AUDIT_REJECTED: reviewer/QA rejection requires fixes."
        else:
            feedback = "CODE_AUDIT_REJECTED: reviewer/QA rejection requires fixes."
        new_history.append("CODE_AUDIT_REJECTED: reviewer/QA rejection requires fixes.")

    hard_qa_retry_count = int(state.get("hard_qa_retry_count", 0))
    if status == "NEEDS_IMPROVEMENT" and hard_qa_gate_reject:
        hard_qa_retry_count += 1
        hard_note = "HARD_QA_GATE: synthetic data or contract-column violation detected; limit retries to 1."
        new_history.append(hard_note)
        feedback = f"{feedback}\n{hard_note}" if feedback else hard_note

    failed_gates = (
        case_report.get("failures", [])
        if case_alignment_required and case_report.get("status") == "FAIL"
        else []
    )
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

    # METRIC ITERATION DISABLED: Force downgrade to APPROVE_WITH_WARNINGS for metric-only issues.
    # This stops the metric retry loop while preserving compliance/runtime retries.
    metric_iteration_disabled = False
    if status == "NEEDS_IMPROVEMENT" and iteration_type == "metric":
        status = "APPROVE_WITH_WARNINGS"
        metric_disable_msg = "(Metric iteration disabled: reporting limitations & next steps only.)"
        feedback = f"{feedback}\n{metric_disable_msg}" if feedback else metric_disable_msg
        new_history.append(metric_disable_msg)
        iteration_type = None  # Clear iteration_type to prevent any metric loop logic
        metric_iteration_disabled = True
        print(f"METRIC_ITERATION_DISABLED: Downgraded to APPROVE_WITH_WARNINGS for metric-only issue.")

    if iteration_type:
        gate_context["iteration_type"] = iteration_type
    compliance_iterations = int(state.get("compliance_iterations", 0))
    metric_iterations = int(state.get("metric_iterations", 0))
    compliance_passed = bool(state.get("compliance_passed", False))
    if status == "NEEDS_IMPROVEMENT":
        if iteration_type == "compliance":
            compliance_iterations += 1
            compliance_passed = False
        # Note: metric branch removed since iteration_type="metric" is now downgraded above

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
        "data_adequacy_report": data_adequacy_report,
        "primary_metric_snapshot": primary_metric_snapshot,
        "metric_history": metric_history,
        "hard_qa_gate_reject": hard_qa_gate_reject,
        "hard_qa_retry_count": hard_qa_retry_count,
    }
    if retry_worth_it is not None:
        result_state["review_retry_worth_it"] = bool(retry_worth_it)
    if state.get("qa_budget_exceeded"):
        result_state["qa_budget_exceeded"] = True
    if status in ["APPROVED", "APPROVE_WITH_WARNINGS"]:
        result_state["last_successful_review_verdict"] = status
        result_state["last_successful_gate_context"] = gate_context
    if review_counters:
        result_state["budget_counters"] = review_counters
    if metric_iteration_disabled:
        result_state["stop_reason"] = "METRIC_ITERATION_DISABLED"

    # NOTE: results_advisor.generate_ml_advice block REMOVED.
    # Metric iteration is disabled, so this block was dead code.
    # The condition (status=="NEEDS_IMPROVEMENT" and iteration_type=="metric") can never
    # be true because we force downgrade to APPROVE_WITH_WARNINGS above.

    try:
        strategy_spec = state.get("strategy_spec") or _load_json_safe("data/strategy_spec.json")
        objective_type = objective_type or (strategy_spec or {}).get("objective_type")
        if not objective_type:
            plan = state.get("execution_plan") or _load_json_safe("data/plan.json")
            if isinstance(plan, dict):
                objective_type = plan.get("objective_type")
        if not objective_type:
            evaluation_spec = state.get("evaluation_spec") or (state.get("execution_contract", {}) or {}).get("evaluation_spec")
            if isinstance(evaluation_spec, dict):
                objective_type = evaluation_spec.get("objective_type")
        insights_context = {
            "objective_type": objective_type,
            "artifact_index": state.get("artifact_index") or _load_json_any("data/produced_artifact_index.json"),
            "output_contract_report": oc_report,
            "review_feedback": feedback,
            "review_verdict": status,
            "metrics": metrics_report,
            "metric_history": metric_history,
            "data_adequacy_report": data_adequacy_report,
            "iteration_policy": (contract or {}).get("iteration_policy", {}) if isinstance(contract, dict) else {},
            "strategy_spec": strategy_spec,
            "reporting_policy": contract.get("reporting_policy", {}) if isinstance(contract, dict) else {},
        }
        insights = results_advisor.generate_insights(insights_context)
        if insights:
            run_id = state.get("run_id")
            if run_id:
                log_agent_snapshot(
                    run_id,
                    "results_advisor",
                    prompt=getattr(results_advisor, "last_prompt", None),
                    response=getattr(results_advisor, "last_response", None) or insights,
                    context=insights_context,
                )
            try:
                os.makedirs("data", exist_ok=True)
                dump_json("data/insights.json", insights)
                existing_index = _load_json_any("data/produced_artifact_index.json")
                normalized_existing = existing_index if isinstance(existing_index, list) else []
                additions = _build_artifact_index(["data/insights.json"], None)
                merged_index = _merge_artifact_index_entries(normalized_existing, additions)
                dump_json("data/produced_artifact_index.json", merged_index)
                result_state["artifact_index"] = merged_index
                result_state["produced_artifact_index"] = merged_index
            except Exception as ins_err:
                print(f"Warning: failed to persist insights.json: {ins_err}")
    except Exception as ins_err:
        print(f"Warning: results advisor insights failed: {ins_err}")

    # Iteration memory (delta + objective + weights) for patch-mode guidance
    iteration_memory = list(state.get("ml_iteration_memory", []) or [])
    prev_summary = iteration_memory[-1] if iteration_memory else None
    diagnostics = _collect_iteration_diagnostics(state)
    summary = _build_iteration_memory(
        iter_id=iter_id,
        metrics_report=metrics_report,
        case_report=case_report,
        weights_report=_load_json_safe("data/weights.json"),
        code=code,
        prev_summary=prev_summary,
        advisor_note=result_state.get("ml_results_advice"),
        diagnostics=diagnostics,
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

    if run_id:
        outputs_state = _collect_outputs_state(_resolve_required_outputs(contract))
        reviewer_reasons = _normalize_reason_tags(review_feedback, failed_gates)
        next_actions = _suggest_next_actions([], outputs_state["missing"], reviewer_reasons, [])
        entry = _build_ml_iteration_journal_entry(
            state,
            preflight_issues=[],
            runtime_error=None,
            outputs_present=outputs_state["present"],
            outputs_missing=outputs_state["missing"],
            reviewer_verdict=status,
            reviewer_reasons=reviewer_reasons,
            qa_verdict=None,
            qa_reasons=[],
            next_actions=next_actions,
            stage="review_complete",
        )
        written_ids = _append_ml_iteration_journal(
            run_id,
            entry,
            state.get("ml_journal_written_ids"),
        )
        result_state["ml_journal_written_ids"] = written_ids

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
    run_id = state.get("run_id")
    if run_id:
        contract = state.get("execution_contract", {}) or {}
        outputs_state = _collect_outputs_state(_resolve_required_outputs(contract))
        reviewer_feedback = state.get("review_feedback") or ""
        reviewer_reasons = _normalize_reason_tags(reviewer_feedback, [])
        next_actions = _suggest_next_actions([], outputs_state["missing"], reviewer_reasons, [])
        entry = _build_ml_iteration_journal_entry(
            state,
            preflight_issues=[],
            runtime_error=_summarize_runtime_error(output),
            outputs_present=outputs_state["present"],
            outputs_missing=outputs_state["missing"],
            reviewer_verdict=state.get("review_verdict"),
            reviewer_reasons=reviewer_reasons,
            qa_verdict=None,
            qa_reasons=[],
            next_actions=next_actions,
            stage="runtime_fix",
        )
        written_ids = _append_ml_iteration_journal(
            run_id,
            entry,
            state.get("ml_journal_written_ids"),
        )
    else:
        written_ids = state.get("ml_journal_written_ids")

    return {
        "last_gate_context": error_context,
         # We add error to history so it persists
        "feedback_history": state.get("feedback_history", []) + [f"RUNTIME ERROR (Attempt {fix_attempt}/{max_runtime_fixes}):\n{output[-500:]}"],
        "ml_engineer_audit_override": ml_override,
        "runtime_fix_count": base_fix_count if terminal_fix else fix_attempt,
        "ml_journal_written_ids": written_ids,
    }

def finalize_runtime_failure(state: AgentState) -> AgentState:
    print("--- [!] Final runtime failure: capturing failure explanation ---")
    state_with_terminal = dict(state)
    state_with_terminal["runtime_fix_terminal"] = True
    result = prepare_runtime_fix(state_with_terminal)
    result["runtime_fix_terminal"] = True
    return result

def check_evaluation(state: AgentState):
    # Helper to get metric info for logging
    def _get_metric_info():
        snapshot = state.get("primary_metric_snapshot") or {}
        metric_name = snapshot.get("primary_metric_name", "unknown")
        metric_value = snapshot.get("primary_metric_value", "N/A")
        best_value = snapshot.get("baseline_value", "N/A")
        return metric_name, metric_value, best_value

    policy = _get_iteration_policy(state)
    last_iter_type = state.get("last_iteration_type")
    metric_iters = state.get("metric_iterations", 0)
    metric_max = policy.get("metric_improvement_max") if policy else None
    budget_left = (metric_max - metric_iters) if metric_max else "unlimited"
    metric_name, metric_value, best_value = _get_metric_info()

    if state.get("hard_qa_gate_reject"):
        hard_count = int(state.get("hard_qa_retry_count", 0))
        if hard_count > 1:
            # Hard QA gate failed and retry limit reached - NEVER approve, just stop iterating
            print("WARNING: Hard QA gate retry limit reached (max 1). Stopping iterations - NOT approving.")
            print(f"ITER_DECISION type=hard_qa action=stop reason=HARD_GATE_FAIL metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
            # Ensure the state reflects the hard gate failure for final reporting
            state["stop_reason"] = "HARD_GATE_FAIL"
            state["hard_gate_terminal"] = True
            # NOTE: Returning "approved" here means "stop iterating", NOT "final approval"
            # The review_verdict in state should still be NEEDS_IMPROVEMENT
            return "approved"

    if policy:
        compliance_max = policy.get("compliance_bootstrap_max")
        if last_iter_type == "compliance" and compliance_max:
            if state.get("compliance_iterations", 0) >= compliance_max:
                print("WARNING: Compliance bootstrap limit reached. Proceeding with current results.")
                print(f"ITER_DECISION type=compliance action=stop reason=BUDGET metric={metric_name}:{metric_value} best={best_value} budget_left=0")
                return "approved"
        if last_iter_type != "compliance" and metric_max:
            if metric_iters >= metric_max:
                print("WARNING: Metric-iteration limit reached. Proceeding with current results.")
                print(f"ITER_DECISION type=metric action=stop reason=BUDGET metric={metric_name}:{metric_value} best={best_value} budget_left=0")
                return "approved"
    else:
        if state.get('iteration_count', 0) >= 6:
            print("WARNING: Max iterations reached. Proceeding with current results.")
            print(f"ITER_DECISION type=metric action=stop reason=BUDGET metric={metric_name}:{metric_value} best={best_value} budget_left=0")
            return "approved"

    # Adaptive stop: if case alignment degrades or stagnates, stop early.
    if last_iter_type == "compliance":
        if state.get('review_verdict') == "NEEDS_IMPROVEMENT":
            print(f"ITER_DECISION type=compliance action=retry reason=COMPLIANCE_FIX metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
            return "retry"
        print(f"ITER_DECISION type=compliance action=stop reason=SUCCESS metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
        return "approved"

    if last_iter_type == "metric":
        data_report = state.get("data_adequacy_report") or _load_json_safe("data/data_adequacy_report.json") or {}
        reasons = data_report.get("reasons", []) if isinstance(data_report, dict) else []
        threshold_reached = bool(data_report.get("threshold_reached")) if isinstance(data_report, dict) else False
        if threshold_reached or "signal_ceiling_reached" in reasons:
            msg = "DATA_LIMITED_STOP: data adequacy indicates signal ceiling reached; stopping metric iterations."
            print(f"WARNING: {msg}")
            _append_feedback_history(state, msg)
            state["stop_reason"] = "CEILING"
            print(f"ITER_DECISION type=metric action=stop reason=CEILING metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
            return "approved"
        plateau_window = 2
        plateau_epsilon = 0.01
        if policy:
            plateau_window = int(policy.get("plateau_window", plateau_window) or plateau_window)
            plateau_epsilon = float(policy.get("plateau_epsilon", plateau_epsilon) or plateau_epsilon)
        plateau, reason = _detect_metric_plateau(state.get("metric_history"), plateau_window, plateau_epsilon)
        if plateau:
            msg = f"PLATEAU_STOP: {reason or 'metric plateau detected'}"
            print(f"WARNING: {msg}")
            _append_feedback_history(state, msg)
            state["stop_reason"] = "PLATEAU"
            print(f"ITER_DECISION type=metric action=stop reason=PLATEAU metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
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
                state["stop_reason"] = "CASE_ALIGNMENT_STAGNANT"
                print(f"ITER_DECISION type=metric action=stop reason=CASE_ALIGNMENT_STAGNANT metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
                return "approved"
        # stop if regression >10% vs best so far
        best = min(new_history) if new_history else curr_violation_rate
        if curr_violation_rate > best * 1.10:
            print("WARNING: Case alignment regressed vs best. Stopping early.")
            state["stop_reason"] = "CASE_ALIGNMENT_REGRESSED"
            print(f"ITER_DECISION type=metric action=stop reason=CASE_ALIGNMENT_REGRESSED metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
            return "approved"

    # AIRBAG: Metric iteration is DISABLED. Never retry for metric-only issues.
    # Even if somehow NEEDS_IMPROVEMENT with last_iteration_type="metric" reaches here,
    # we force stop and proceed to translator.
    if last_iter_type == "metric":
        print(f"AIRBAG_METRIC_STOP: Metric iteration disabled. Proceeding to report generation.")
        print(f"ITER_DECISION type=metric action=stop reason=METRIC_ITERATION_DISABLED metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
        state["stop_reason"] = "METRIC_ITERATION_DISABLED"
        return "approved"

    if state.get('review_verdict') == "NEEDS_IMPROVEMENT":
        # This branch should only be reached for compliance/runtime retries now
        print(f"ITER_DECISION type=other action=retry reason=COMPLIANCE_OR_RUNTIME metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
        return "retry"
    else:
        print(f"ITER_DECISION type=other action=stop reason=SUCCESS metric={metric_name}:{metric_value} best={best_value} budget_left={budget_left}")
        return "approved"

def run_translator(state: AgentState) -> AgentState:
    print("--- [6] Translator: Generating Report ---")
    abort_state = _abort_if_requested(state, "translator")
    if abort_state:
        return abort_state
    run_id = state.get("run_id")
    if run_id:
        log_run_event(run_id, "translator_start", {})

    best_score = state.get("best_attempt_score")
    last_score = state.get("last_attempt_score")
    promote_best = False
    if best_score is not None and state.get("best_attempt_dir"):
        if state.get("execution_error") or state.get("sandbox_failed"):
            promote_best = True
        if state.get("artifact_content_issues"):
            promote_best = True
        missing_outputs = (state.get("output_contract_report") or {}).get("missing") if isinstance(state.get("output_contract_report"), dict) else []
        if missing_outputs:
            promote_best = True
        try:
            if last_score is not None and float(best_score) > float(last_score):
                promote_best = True
        except Exception:
            pass
    if promote_best:
        updates = _promote_best_attempt(state)
        if updates:
            state = {**state, **updates}

    error_msg = state.get("error_message")

    # Extract visuals context
    has_partial_visuals = state.get("has_partial_visuals", False)
    plots_local = state.get("plots_local", [])
    fallback_plots = state.get("fallback_plots", [])

    report_state = dict(state)
    summary = None
    try:
        if not os.path.exists("data/output_contract_report.json"):
            reason = state.get("pipeline_aborted_reason") or "pipeline_aborted"
            _persist_output_contract_report(state, reason=reason)
        summary = build_run_summary(state)
        os.makedirs("data", exist_ok=True)
        dump_json("data/run_summary.json", summary)
    except Exception:
        summary = None
    report_error = error_msg
    if error_msg and "BUDGET_EXCEEDED" in str(error_msg) and state.get("last_successful_execution_output"):
        report_state["execution_output"] = state.get("last_successful_execution_output")
        if state.get("last_successful_plots"):
            report_state["plots_local"] = state.get("last_successful_plots")
            report_state["has_partial_visuals"] = True
        report_error = None
    report_state.setdefault("execution_error", state.get("execution_error", False))
    report_state.setdefault("sandbox_failed", state.get("sandbox_failed", False))
    translator_view = state.get("translator_view") or (state.get("contract_views") or {}).get("translator_view")
    if not isinstance(translator_view, dict) or not translator_view:
        translator_view = build_translator_view(
            report_state.get("execution_contract", {}) or {},
            _load_json_safe("data/contract_min.json") or {},
            report_state.get("artifact_index") or _load_json_any("data/produced_artifact_index.json") or [],
        )
    report_state["translator_view"] = translator_view
    try:
        translator_view_len = len(json.dumps(translator_view, ensure_ascii=True))
        print(f"Using TRANSLATOR_VIEW_CONTEXT length={translator_view_len}")
        if run_id:
            log_run_event(run_id, "translator_view_context", {"length": translator_view_len})
    except Exception:
        pass
    contract = report_state.get("execution_contract", {}) or {}
    deliverables = _resolve_contract_deliverables(contract)
    has_preview_deliverable = False
    if isinstance(deliverables, list):
        for item in deliverables:
            if isinstance(item, dict) and item.get("path") == "reports/recommendations_preview.json":
                has_preview_deliverable = True
                break
            if isinstance(item, str) and item == "reports/recommendations_preview.json":
                has_preview_deliverable = True
                break
    reporting_policy = contract.get("reporting_policy") if isinstance(contract, dict) else {}
    preview_enabled = True
    if isinstance(reporting_policy, dict) and reporting_policy.get("demonstrative_examples_enabled") is False:
        preview_enabled = False
    if has_preview_deliverable and preview_enabled:
        output_text = report_state.get("execution_output", "") or ""
        artifact_guard = "ARTIFACT_ALIGNMENT_GUARD" in output_text
        stale_guard = "STALE_OUTPUTS" in output_text
        artifacts_valid = not (
            report_state.get("execution_error")
            or report_state.get("sandbox_failed")
            or artifact_guard
            or stale_guard
        )
        cleaned_path = "data/cleaned_full.csv" if os.path.exists("data/cleaned_full.csv") else "data/cleaned_data.csv"
        preview_root = None
        run_dir = get_run_dir(run_id) if run_id else None
        if run_id and not run_dir:
            run_dir = os.path.join("runs", run_id)
        if run_dir:
            preview_root = os.path.join(run_dir, "artifacts")
            try:
                copy_run_artifacts(
                    run_id,
                    ["data", "reports", "static"],
                    since_epoch=state.get("run_start_epoch"),
                )
            except Exception:
                pass
        produced_index = (
            state.get("produced_artifact_index")
            or state.get("artifact_index")
            or _load_json_any("data/produced_artifact_index.json")
        )
        preview_cleaned_path = None
        if preview_root:
            candidate = os.path.join(preview_root, "data", "cleaned_data.csv")
            if os.path.exists(candidate):
                preview_cleaned_path = candidate
        if not preview_cleaned_path and cleaned_path and os.path.exists(cleaned_path):
            preview_cleaned_path = cleaned_path
        preview_payload = build_recommendations_preview(
            contract=contract,
            governance_summary=summary or {},
            artifacts_dir=preview_root or ".",
            cleaned_data_path=preview_cleaned_path,
            produced_artifact_index=produced_index,
            run_scoped_root=run_dir,
        )
        chosen_source = preview_payload.get("chosen_source") if isinstance(preview_payload, dict) else {}
        if not artifacts_valid:
            if isinstance(chosen_source, dict) and chosen_source.get("kind") == "sandbox_attempt":
                preview_payload["status"] = "illustrative_only"
                preview_payload["risk_level"] = "high"
                preview_payload.setdefault("caveats", []).append(
                    "Artifacts were invalid or blocked; illustrative examples sourced from sandbox attempt outputs."
                )
            else:
                preview_payload["items"] = []
                preview_payload["reason"] = preview_payload.get("reason") or "artifacts_invalid"
                preview_payload["status"] = "illustrative_only"
                preview_payload["risk_level"] = "high"
                preview_payload.setdefault("caveats", []).append(
                    "Artifacts were invalid or blocked; examples withheld."
                )
        try:
            os.makedirs("reports", exist_ok=True)
            with open("reports/recommendations_preview.json", "w", encoding="utf-8") as f_prev:
                json.dump(preview_payload, f_prev, indent=2, ensure_ascii=False)
            existing_index = _load_json_any("data/produced_artifact_index.json")
            normalized_existing = existing_index if isinstance(existing_index, list) else []
            additions = _build_artifact_index(["reports/recommendations_preview.json"], deliverables)
            merged_index = _merge_artifact_index_entries(normalized_existing, additions)
            dump_json("data/produced_artifact_index.json", merged_index)
            report_state["artifact_index"] = merged_index
            report_state["produced_artifact_index"] = merged_index
        except Exception as prev_err:
            print(f"Warning: failed to persist recommendations_preview.json: {prev_err}")

        if isinstance(chosen_source, dict) and chosen_source.get("kind") == "sandbox_attempt":
            label_col_hint = None
            if isinstance(contract, dict):
                label_col_hint = contract.get("segment_label_column")
            staged = _stage_illustrative_assets(
                chosen_source.get("root"),
                report_root="report",
                label_col_hint=label_col_hint,
                contract=contract,
            )
            if staged.get("plots") and not report_state.get("plots_local"):
                report_state["plots_local"] = staged.get("plots")
            report_state["illustrative_assets"] = staged
    report_plots = report_state.get("plots_local", plots_local)
    artifact_entries = report_state.get("artifact_index") or []
    report_artifacts = [
        item.get("path") if isinstance(item, dict) else item
        for item in artifact_entries
        if item
    ]
    error_flag = bool(report_error) or report_state.get("execution_error") or report_state.get("sandbox_failed")
    if not error_flag:
        if fallback_plots:
            report_plots = [plot for plot in report_plots if plot not in fallback_plots]
            report_state["plots_local"] = report_plots
            if artifact_entries:
                filtered_entries = []
                for entry in artifact_entries:
                    path = entry.get("path") if isinstance(entry, dict) else entry
                    if path in fallback_plots:
                        continue
                    filtered_entries.append(entry)
                report_state["artifact_index"] = filtered_entries
                report_state["produced_artifact_index"] = filtered_entries
    report_state["has_partial_visuals"] = bool(report_plots) and error_flag
    report_has_partial = report_state["has_partial_visuals"]
    try:
        report = translator.generate_report(
            report_state,
            error_message=report_error,
            has_partial_visuals=report_has_partial,
            plots=report_plots,
            translator_view=translator_view,
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
    if run_id:
        log_agent_snapshot(
            run_id,
            "translator",
            prompt=getattr(translator, "last_prompt", None),
            response=getattr(translator, "last_response", None) or report,
            context={
                "error_message": report_error,
                "has_partial_visuals": report_has_partial,
                "plot_count": len(report_plots) if isinstance(report_plots, list) else 0,
            },
        )

    try:
        os.makedirs("data", exist_ok=True)
        with open("data/executive_summary.md", "w", encoding="utf-8") as f_exec:
            f_exec.write(report or "")
        existing_index = _load_json_any("data/produced_artifact_index.json")
        normalized_existing = existing_index if isinstance(existing_index, list) else []
        additions = _build_artifact_index(["data/executive_summary.md"], None)
        merged_index = _merge_artifact_index_entries(normalized_existing, additions)
        dump_json("data/produced_artifact_index.json", merged_index)
        report_state["artifact_index"] = merged_index
        report_state["produced_artifact_index"] = merged_index
    except Exception as exec_err:
        print(f"Warning: failed to persist executive_summary.md: {exec_err}")

    try:
        pdf_payload = generate_pdf_artifact({**report_state, "final_report": report})
        pdf_path = pdf_payload.get("pdf_path") if isinstance(pdf_payload, dict) else None
        if pdf_path:
            report_state["pdf_path"] = pdf_path
    except Exception as pdf_err:
        print(f"Warning: PDF generation failed in translator: {pdf_err}")

    try:
        write_data_adequacy_report(state)
        write_governance_report(state)
        summary = summary or build_run_summary(state)
        try:
            os.makedirs("data", exist_ok=True)
            dump_json("data/run_summary.json", summary)
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
            copy_run_contracts(
                run_id,
                [
                    "data/execution_contract.json",
                    "data/evaluation_spec.json",
                    "data/produced_artifact_index.json",
                    "data/contract_min.json",
                ],
            )
            since_epoch = state.get("run_start_epoch")
            copy_run_artifacts(
                run_id,
                [
                    "data",
                    "analysis",
                    "models",
                    "plots",
                    os.path.join("static", "plots"),
                ],
                since_epoch=since_epoch,
            )
            report_sources = ["report", "reports"]
            pdf_path = report_state.get("pdf_path") or state.get("pdf_path")
            if pdf_path:
                report_sources.append(pdf_path)
            copy_run_reports(run_id, report_sources, since_epoch=since_epoch)
            if pdf_path and state.get("run_bundle_dir"):
                try:
                    dest_path = os.path.join(state.get("run_bundle_dir"), "report", "final_report.pdf")
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(pdf_path, dest_path)
                except Exception:
                    pass
            status_final = normalize_status(summary.get("status") if isinstance(summary, dict) else None)
        finalize_run(run_id, status_final=status_final, state=state)
    except Exception:
        pass
    finally:
        # P0 FIX: Always restore cwd on exit
        exit_run_workspace(state)
    return {"final_report": report, "pdf_path": report_state.get("pdf_path")}
# Generate Unique PDF Path to avoid file locks
import uuid

def generate_pdf_artifact(state: AgentState) -> AgentState:
    print("--- [7] System: Generating PDF Report ---")
    import glob

    work_dir = state.get("work_dir") if isinstance(state, dict) else None
    if not work_dir:
        work_dir = "."
    work_dir = os.path.abspath(work_dir)

    def _copy_pdf_artifact(pdf_path: str) -> None:
        run_id = state.get("run_id")
        if run_id:
            copy_run_reports(run_id, [pdf_path], since_epoch=None)
            try:
                dest_root = os.path.join("runs", run_id, "report")
                os.makedirs(dest_root, exist_ok=True)
                shutil.copy2(pdf_path, os.path.join(dest_root, "final_report.pdf"))
            except Exception:
                pass
        run_bundle_dir = state.get("run_bundle_dir")
        if run_bundle_dir:
            try:
                dest_path = os.path.join(run_bundle_dir, "report", "final_report.pdf")
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(pdf_path, dest_path)
            except Exception:
                pass

    report = state.get("final_report")
    existing_pdf = state.get("pdf_path")
    if existing_pdf:
        candidate = existing_pdf
        if not os.path.isabs(candidate):
            candidate = os.path.join(work_dir, candidate)
        if os.path.exists(candidate):
            _copy_pdf_artifact(candidate)
            return {"final_report": report, "pdf_path": os.path.abspath(candidate)}
    if not report:
        try:
            summary_path = os.path.join(work_dir, "data", "executive_summary.md")
            with open(summary_path, "r", encoding="utf-8") as f_exec:
                report = f_exec.read()
        except Exception:
            report = ""
    report = str(report or "")
    if not report.strip():
        print("PDF Generation Skipped: no report content available.")
        return {"final_report": state.get("final_report"), "pdf_path": None}

    # Check for visualizations
    if "static/plots" not in report:
        plots = state.get("plots_local", []) or []
        fallback_plots = state.get("fallback_plots", []) or []
        has_exec_error = bool(state.get("execution_error") or state.get("sandbox_failed"))
        if not has_exec_error and fallback_plots:
            plots = [plot for plot in plots if plot not in fallback_plots]
        if not plots:
            report_plots_dir = os.path.join(work_dir, "report", "static", "plots")
            if os.path.isdir(report_plots_dir):
                plots = glob.glob(os.path.join(report_plots_dir, "*.png"))
        if plots:
            report += "\n\n## Visualizations\n"
            for plot in plots:
                if os.path.isabs(plot) and plot.startswith(work_dir):
                    rel_plot = os.path.relpath(plot, work_dir)
                else:
                    rel_plot = plot
                plot_ref = _normalize_path_posix(rel_plot)
                report += f"![{os.path.basename(plot)}]({plot_ref})\n"

    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]
    pdf_filename = os.path.join(work_dir, f"final_report_{unique_id}.pdf")

    # Absolute path for clarity
    abs_pdf_path = os.path.abspath(pdf_filename)

    # Convert - ensure CWD is work_dir for relative asset resolution
    cwd0 = os.getcwd()
    success = False
    try:
        os.chdir(work_dir)
        success = convert_report_to_pdf(report, abs_pdf_path)
    finally:
        os.chdir(cwd0)

    if success:
        print(f"PDF generated at: {abs_pdf_path}")
        run_id = state.get("run_id")
        since_epoch = state.get("run_start_epoch")

        latest_pdf = pdf_filename
        try:
            candidates = []
            for path in glob.glob(os.path.join(work_dir, "final_report*.pdf")):
                try:
                    mtime = os.path.getmtime(path)
                except Exception:
                    continue
                if since_epoch is None or mtime >= float(since_epoch) - 1.0:
                    candidates.append((mtime, path))
            if candidates:
                candidates.sort(reverse=True)
                latest_pdf = candidates[0][1]
        except Exception:
            latest_pdf = pdf_filename

        latest_abs = os.path.abspath(latest_pdf)
        _copy_pdf_artifact(latest_abs)
        return {"final_report": report, "pdf_path": latest_abs}
    else:
        print("PDF Generation Failed")
        return {"final_report": report, "pdf_path": None}

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
        "success": "execute_code",
        "failed": "translator"
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
