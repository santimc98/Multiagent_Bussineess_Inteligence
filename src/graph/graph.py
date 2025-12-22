import sys
import os
import subprocess
import re
import json
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
from src.agents.reviewer import ReviewerAgent
from src.agents.qa_reviewer import QAReviewerAgent # New QA Gate
from src.agents.execution_planner import ExecutionPlannerAgent
from src.agents.postmortem import PostmortemAgent
from src.utils.pdf_generator import convert_report_to_pdf
from src.utils.static_safety_scan import scan_code_safety
from src.utils.visuals import generate_fallback_plots
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

def _norm_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

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
    sample_df = sample_raw_columns(csv_path, dialect, raw_cols, nrows=max_rows)
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
        numeric_like = float(sample.str.contains(r"^[\\s\\-\\+]*[\\d,.\\s%]+$").mean())
        whitespace = float(sample.str.contains(r"^\\s+|\\s+$").mean())
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

def _filter_input_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of contract with data_requirements limited to source=='input'."""
    if not isinstance(contract, dict):
        return {}
    reqs = contract.get("data_requirements", []) or []
    filtered = [r for r in reqs if isinstance(r, dict) and r.get("source", "input") == "input"]
    new_contract = dict(contract)
    new_contract["data_requirements"] = filtered
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

DEFAULT_RUN_BUDGET = {
    "max_de_calls": 4,
    "max_ml_calls": 5,
    "max_reviewer_calls": 5,
    "max_qa_calls": 5,
    "max_execution_calls": 3,
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

    return issues

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

    for param, expected_value in expected.items():
        if param not in kw_map:
            violations.append(f"pd.read_csv missing {param}= for dialect")
            continue
        val_node = kw_map[param]
        if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
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
    if "typeerror" in lower or "ufunc" in lower:
        return "Type conversion missing; numeric ops executed on string/object data."
    if "json" in lower and "default" in lower:
        return "Manifest json.dump missing default for numpy/pandas types."
    if "syntaxerror" in lower:
        return "Generated code is not valid Python syntax."
    return ""

def _build_de_postmortem_context(state: Dict[str, Any], decision: Dict[str, Any]) -> str:
    last_gate = state.get("last_gate_context") or {}
    required_fixes = last_gate.get("required_fixes", [])
    gate_feedback = last_gate.get("feedback", "")
    err_msg = state.get("error_message", "") or ""
    exec_out = state.get("execution_output", "") or ""
    reason = decision.get("reason", "") or ""
    why = ""
    if required_fixes:
        why = f"Gate required fixes: {required_fixes}"
    else:
        why = _infer_de_failure_cause(gate_feedback or err_msg or exec_out)
    if not reason:
        reason = err_msg or gate_feedback or "Data Engineer failure."
    if not why:
        why = "Unknown root cause. Inspect header mapping, canonicalization, and required input checks."
    lines = ["POSTMORTEM_CONTEXT_FOR_DE:"]
    if reason:
        lines.append(f"FAILURE_SUMMARY: {reason}")
    if why:
        lines.append(f"WHY_IT_HAPPENED: {why}")
    if gate_feedback:
        lines.append(f"LAST_GATE_FEEDBACK: {gate_feedback}")
    if err_msg:
        lines.append(f"ERROR_MESSAGE: {err_msg}")
    if exec_out:
        lines.append(f"EXECUTION_OUTPUT_TAIL: {exec_out[-1200:]}")
    lines.append("FIX_GUIDANCE: Fix the root cause and regenerate the full cleaning script.")
    return "\n".join(lines)

def _build_ml_postmortem_context(state: Dict[str, Any], decision: Dict[str, Any]) -> str:
    last_gate = state.get("last_gate_context") or {}
    required_fixes = last_gate.get("required_fixes", [])
    gate_feedback = last_gate.get("feedback", "")
    err_msg = state.get("error_message", "") or ""
    exec_out = state.get("execution_output", "") or ""
    reason = decision.get("reason", "") or ""
    why = ""
    if required_fixes:
        why = f"Gate required fixes: {required_fixes}"
    else:
        why = _infer_de_failure_cause(gate_feedback or err_msg or exec_out)
    if not reason:
        reason = err_msg or gate_feedback or "ML Engineer failure."
    if not why:
        why = "Unknown root cause. Inspect feature mapping, target variance, and dialect usage."
    lines = ["POSTMORTEM_CONTEXT_FOR_ML:"]
    if reason:
        lines.append(f"FAILURE_SUMMARY: {reason}")
    if why:
        lines.append(f"WHY_IT_HAPPENED: {why}")
    if gate_feedback:
        lines.append(f"LAST_GATE_FEEDBACK: {gate_feedback}")
    if err_msg:
        lines.append(f"ERROR_MESSAGE: {err_msg}")
    if exec_out:
        lines.append(f"EXECUTION_OUTPUT_TAIL: {exec_out[-1200:]}")
    lines.append("FIX_GUIDANCE: Fix the root cause and regenerate the full ML script.")
    return "\n".join(lines)

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
    last_runtime_error_tail: str # Added for Runtime Error Visibility
    data_engineer_audit_override: str
    ml_engineer_audit_override: str
    leakage_audit_summary: str
    ml_skipped_reason: str
    execution_contract: Dict[str, Any]
    postmortem_decision: Dict[str, Any]
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
ml_engineer = MLEngineerAgent()
translator = BusinessTranslatorAgent()
reviewer = ReviewerAgent()
qa_reviewer = QAReviewerAgent()
execution_planner = ExecutionPlannerAgent()
postmortem_agent = PostmortemAgent()


# 2. Define Nodes

def run_steward(state: AgentState) -> AgentState:
    print("--- [1] Steward: Analyzing Data ---")
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
    print("--- [2] Strategist: Formulating 3 Strategies (DeepSeek Reasoner) ---")
    
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
    contract = ensure_role_runbooks(contract)
    contract = _ensure_scored_rows_output(contract)
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
    return {"execution_contract": contract}




def run_data_engineer(state: AgentState) -> AgentState:
    print("--- [3] Data Engineer: Cleaning Data (E2B Sandbox) ---")
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
        sample_context = _build_required_sample_context(csv_path, input_dialect, required_cols, norm_map)
        if sample_context:
            data_engineer_audit_override = _merge_de_audit_override(data_engineer_audit_override, sample_context)
    state["data_engineer_audit_override"] = data_engineer_audit_override
    try:
        os.makedirs("artifacts", exist_ok=True)
        context_payload = {
            "csv_path": csv_path,
            "csv_encoding": csv_encoding,
            "csv_sep": csv_sep,
            "csv_decimal": csv_decimal,
            "header_cols": header_cols,
            "required_input_columns": _resolve_required_input_columns(state.get("execution_contract", {}), selected),
            "required_all_columns": _resolve_contract_columns(state.get("execution_contract", {})),
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
        execution_contract=state.get("execution_contract", {}),
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
            contract_derived_cols = _resolve_contract_columns(contract, sources={"derived", "output"})
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

            # Guard: derived columns should not be constant if present
            derived_issues = []
            if contract_derived_cols:
                col_by_norm = {_norm_name(c): c for c in df_mapped.columns}
                for derived_name in contract_derived_cols:
                    norm_name = _norm_name(derived_name)
                    actual_name = col_by_norm.get(norm_name)
                    if not actual_name:
                        continue
                    nunique = df_mapped[actual_name].nunique(dropna=False)
                    if nunique <= 1:
                        derived_issues.append(f"{derived_name} has no variance (nunique={nunique})")
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
    run_id = state.get("run_id")
    ok, counters, err_msg = _consume_budget(state, "ml_calls", "max_ml_calls", "ML Engineer")
    state["budget_counters"] = counters
    if not ok:
        if run_id:
            log_run_event(run_id, "budget_exceeded", {"label": "ml_engineer", "error": err_msg})
        return {
            "error_message": err_msg,
            "generated_code": "# Generation Failed",
            "execution_output": err_msg,
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
        )
        sig = inspect.signature(ml_engineer.generate_code)
        if "execution_contract" in sig.parameters:
            kwargs["execution_contract"] = execution_contract
        header_cols = _read_csv_header(data_path, csv_encoding, csv_sep)
        if header_cols:
            norm_map = {}
            for col in header_cols:
                normed = _norm_name(col)
                if normed and normed not in norm_map:
                    norm_map[normed] = col
            header_context = (
                "CLEANED_COLUMN_INVENTORY_RAW: "
                + json.dumps(header_cols, ensure_ascii=False)
                + "\nNORMALIZED_CLEANED_HEADER_MAP: "
                + json.dumps(norm_map, ensure_ascii=False)
            )
            data_audit_context = _merge_de_audit_override(data_audit_context, header_context)
            kwargs["data_audit_context"] = data_audit_context
        try:
            os.makedirs("artifacts", exist_ok=True)
            ctx_payload = {
                "data_path": data_path,
                "csv_encoding": csv_encoding,
                "csv_sep": csv_sep,
                "csv_decimal": csv_decimal,
                "header_cols": header_cols,
                "required_features": strategy.get("required_columns", []),
                "execution_contract": execution_contract,
                "data_audit_context": data_audit_context,
                "ml_engineer_audit_override": ml_audit_override,
            }
            with open(os.path.join("artifacts", "ml_engineer_context.json"), "w", encoding="utf-8") as f_ctx:
                json.dump(ctx_payload, f_ctx, indent=2, ensure_ascii=False)
        except Exception as ctx_err:
            print(f"Warning: failed to persist ml_engineer_context.json: {ctx_err}")
        code = ml_engineer.generate_code(**kwargs)
        try:
            os.makedirs("artifacts", exist_ok=True)
            with open(os.path.join("artifacts", "ml_engineer_last.py"), "w", encoding="utf-8") as f_art:
                f_art.write(code)
        except Exception as artifact_err:
            print(f"Warning: failed to persist ml_engineer_last.py: {artifact_err}")

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


def check_postmortem_action(state: AgentState):
    decision = state.get("postmortem_decision", {}) or {}
    action = decision.get("action")
    if action == "retry_data_engineer":
        return "retry_de"
    if action == "retry_ml_engineer":
        return "retry_ml"
    if action == "re_strategize":
        return "restrat"
    if action == "stop":
        return "stop"
    return "stop"

def run_reviewer(state: AgentState) -> AgentState:
    print("--- REVIEWER AGENT ---")
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
            "required_fixes": review.get('required_fixes', [])
        }

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
            "required_fixes": required_fixes
        }
        
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
    code = state.get("generated_code", "")
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
    if issues:
        feedback = f"ML_PREFLIGHT_MISSING: {', '.join(issues)}"
        history = list(state.get("feedback_history", []))
        history.append(feedback)
        gate_context = {
            "source": "ml_preflight",
            "status": "REJECTED",
            "feedback": feedback,
            "failed_gates": issues,
            "required_fixes": issues,
        }
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
    
    # Fallback Plotting (Robustness)
    if not plots_local:
        print("Warning: No plots found from sandbox. Attempting fallback generation on host.")
        try:
             # Ensure data exists locally
             if os.path.exists("data/cleaned_data.csv"):
                 fallback_plots = generate_fallback_plots(
                     "data/cleaned_data.csv",
                     sep=state.get("csv_sep", ","),
                     decimal=state.get("csv_decimal", "."),
                     encoding=state.get("csv_encoding", "utf-8")
                 )
                 if fallback_plots:
                     plots_local.extend(fallback_plots)
                     print(f"Fallback successful. Added {len(fallback_plots)} plots.")
             else:
                 print("Critical: Cannot run fallback plots, 'data/cleaned_data.csv' missing.")
        except Exception as fb_err:
             print(f"Fallback generation failed: {fb_err}")
             
    has_partial_visuals = len(plots_local) > 0
    
    print(f"Execution finished. Plots generated: {len(plots_local)}")
    
    # Check for Runtime Errors in Output (Traceback)
    error_in_output = "Traceback (most recent call last)" in output or "EXECUTION ERROR" in output
    
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
    return {
        "execution_output": output,
        "plots_local": plots_local,
        "has_partial_visuals": has_partial_visuals,
        "execution_attempt": state.get('execution_attempt', 0) + 1,
        "last_runtime_error_tail": runtime_tail,
        "ml_skipped_reason": ml_skipped_reason,
        "output_contract_report": oc_report,
        "budget_counters": counters,
    }

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
        "iteration_count": state['iteration_count'] + 1,
        "feedback_history": current_history
    }
    
def run_result_evaluator(state: AgentState) -> AgentState:
    print("--- [5.5] Reviewer: Evaluating Results ---")
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
    
    print(f"Reviewer Verdict: {status}")
    if status == "NEEDS_IMPROVEMENT":
        print(f"Advice: {feedback}")
    
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

    gate_context = {
        "source": "case_alignment_gate" if case_report.get("status") == "FAIL" else "result_evaluator",
        "status": status,
        "feedback": feedback,
        "failed_gates": case_report.get("failures", []) if case_report.get("status") == "FAIL" else [],
        "required_fixes": case_report.get("failures", []) if case_report.get("status") == "FAIL" else [],
    }

    review_feedback = feedback or state.get("review_feedback", "")
    result_state = {
        "review_verdict": status,
        "review_feedback": review_feedback,
        "execution_feedback": feedback,
        "feedback_history": new_history,
        "output_contract_report": oc_report,
        "last_gate_context": gate_context,
    }
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
    skipped_reason = state.get("ml_skipped_reason")
    
    # Traceback check (Runtime Error)
    has_error = "Traceback (most recent call last)" in output or "EXECUTION ERROR" in output
    determinism_error = "DETERMINISTIC_TARGET_RELATION" in output
    undefined_precheck = "STATIC_PRECHECK_UNDEFINED" in output

    if determinism_error or skipped_reason == "DETERMINISTIC_TARGET_RELATION":
        return "evaluate"
    
    if undefined_precheck:
        return "retry_fix"

    if has_error:
        print(f"Runtime Error detected (Attempt {attempt}). Delegating to Postmortem.")
        return "failed"
            
    return "evaluate"

def prepare_runtime_fix(state: AgentState) -> AgentState:
    print("--- [!] Preparing Runtime Fix Context ---")
    output = state.get("execution_output", "")
    
    error_context = {
        "source": "Execution Runtime",
        "status": "REJECTED",
        "feedback": f"RUNTIME ERROR trace:\n{output[-2000:]}\n\nFIX THIS CRASH.",
        "failed_gates": ["Runtime Stability"],
        "required_fixes": ["Fix the exception."]
    }
    
    return {
        "last_gate_context": error_context,
         # We add error to history so it persists
        "feedback_history": state.get("feedback_history", []) + [f"RUNTIME ERROR (Attempt {state.get('execution_attempt')}):\n{output[-500:]}"]
    }

def check_evaluation(state: AgentState):
    if state.get('iteration_count', 0) >= 3:
        print("WARNING: Max iterations reached. Proceeding with current results.")
        return "approved"

    if state.get('review_verdict') == "NEEDS_IMPROVEMENT":
        return "retry"
    else:
        return "approved"

def run_translator(state: AgentState) -> AgentState:
    print("--- [6] Translator: Generating Report ---")
    run_id = state.get("run_id")
    if run_id:
        log_run_event(run_id, "translator_start", {})
    
    # Handle Failure Case
    # Handle Failure Case
    error_msg = state.get("error_message")
    
    # Extract visuals context
    has_partial_visuals = state.get("has_partial_visuals", False)
    plots_local = state.get("plots_local", [])
    
    try:
        report = translator.generate_report(
            state, 
            error_message=error_msg,
            has_partial_visuals=has_partial_visuals,
            plots=plots_local
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


def run_postmortem(state: AgentState) -> AgentState:
    print("--- [POSTMORTEM] Tech Lead Decision ---")
    run_id = state.get("run_id")
    if run_id:
        log_run_event(run_id, "postmortem_start", {})
    # Load integrity audit issues if available
    integrity_issues = []
    try:
        with open("data/integrity_audit_report.json", "r", encoding="utf-8") as f:
            report = json.load(f)
            integrity_issues = report.get("issues", [])
    except Exception:
        integrity_issues = []

    context = {
        "business_objective": state.get("business_objective", ""),
        "selected_strategy": state.get("selected_strategy", {}),
        "execution_contract": state.get("execution_contract", {}),
        "integrity_issues": integrity_issues,
        "output_contract_report": {},
        "execution_output": state.get("execution_output", ""),
        "review_feedback": state.get("review_feedback", ""),
        "feedback_history": state.get("feedback_history", []),
        "iteration_count": state.get("iteration_count", 0),
        "restrategize_count": state.get("restrategize_count", 0),
        "error_message": state.get("error_message", ""),
        "missing_repeat_count": state.get("missing_repeat_count", 0),
        "last_gate_context": state.get("last_gate_context", {}),
    }
    try:
        with open("data/output_contract_report.json", "r", encoding="utf-8") as f:
            context["output_contract_report"] = json.load(f)
    except Exception:
        context["output_contract_report"] = {}
    decision = postmortem_agent.decide(context)
    if not isinstance(decision, dict):
        decision = postmortem_agent._fallback(context)
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/postmortem_decision.json", "w", encoding="utf-8") as f:
            json.dump(decision, f, indent=2)
    except Exception as err:
        print(f"Warning: failed to persist postmortem_decision.json: {err}")
    if run_id:
        log_run_event(run_id, "postmortem_complete", {"action": decision.get("action")})

    # Build state patch
    new_state = {}
    new_state["postmortem_decision"] = decision
    action = decision.get("action")
    context_patch_raw = decision.get("context_patch")
    if isinstance(context_patch_raw, str):
        try:
            context_patch_raw = json.loads(context_patch_raw)
        except Exception:
            context_patch_raw = {}
    context_patch = context_patch_raw if isinstance(context_patch_raw, dict) else {}
    if decision.get("context_patch") is not None and not isinstance(context_patch_raw, dict):
        warn_msg = "POSTMORTEM_BAD_SHAPE: context_patch not dict"
        fh = list(state.get("feedback_history", []))
        fh.append(warn_msg)
        new_state["feedback_history"] = fh
    if context_patch and not context_patch.get("target"):
        # If patch has no explicit target, route to strategist context override to avoid dropping it.
        try:
            new_state["strategist_context_override"] = json.dumps(context_patch)
        except Exception:
            new_state["strategist_context_override"] = str(context_patch)
        context_patch = {}
    if context_patch:
        target = context_patch.get("target")
        payload = context_patch.get("payload", "")
        if target == "data_engineer_audit_override":
            base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
            new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, str(payload))
        elif target == "ml_engineer_audit_override":
            base_override = state.get("ml_engineer_audit_override") or state.get("data_summary", "")
            new_state["ml_engineer_audit_override"] = _merge_de_audit_override(base_override, str(payload))
        elif target == "feedback_history":
            fh = list(new_state.get("feedback_history", state.get("feedback_history", [])))
            fh.append(payload)
            new_state["feedback_history"] = fh
        elif target == "strategist_context_override":
            new_state["strategist_context_override"] = payload

    if action == "retry_data_engineer":
        base_override = new_state.get("data_engineer_audit_override")
        if not base_override:
            base_override = state.get("data_engineer_audit_override") or state.get("data_summary", "")
        auto_payload = _build_de_postmortem_context(state, decision)
        new_state["data_engineer_audit_override"] = _merge_de_audit_override(base_override, auto_payload)
    if action == "retry_ml_engineer":
        base_override = new_state.get("ml_engineer_audit_override")
        if not base_override:
            base_override = state.get("ml_engineer_audit_override") or state.get("data_summary", "")
        auto_payload = _build_ml_postmortem_context(state, decision)
        new_state["ml_engineer_audit_override"] = _merge_de_audit_override(base_override, auto_payload)

    sr_raw = decision.get("should_reset")
    should_reset = sr_raw if isinstance(sr_raw, dict) else {}
    if decision.get("should_reset") is not None and not isinstance(sr_raw, dict):
        fh = list(new_state.get("feedback_history", state.get("feedback_history", [])))
        fh.append("POSTMORTEM_BAD_SHAPE: should_reset not dict")
        new_state["feedback_history"] = fh
    if should_reset.get("reset_ml_patch_context"):
        new_state["reset_ml_patch_context"] = True
        new_state["last_generated_code"] = None
        new_state["last_gate_context"] = None
    if should_reset.get("reset_review_streaks"):
        new_state["review_reject_streak"] = 0
        new_state["qa_reject_streak"] = 0

    err_msg_lower = str(state.get("error_message", "")).lower()
    missing_pattern = ("missing required columns" in err_msg_lower) or ("mapping failed" in err_msg_lower) or ("dialect" in err_msg_lower) or ("pd.read_csv" in err_msg_lower)
    if action == "re_strategize":
        new_state["restrategize_count"] = state.get("restrategize_count", 0) + 1
        # reset iteration loop
        new_state["iteration_count"] = 0
        new_state["last_generated_code"] = None
        new_state["last_gate_context"] = None
        new_state["review_reject_streak"] = 0
        new_state["qa_reject_streak"] = 0
        new_state["missing_repeat_count"] = 0
    else:
        prev_missing = state.get("missing_repeat_count", 0) or 0
        new_state["missing_repeat_count"] = prev_missing + 1 if missing_pattern else 0
    return new_state

# Generate Unique PDF Path to avoid file locks
import uuid

def generate_pdf_artifact(state: AgentState) -> AgentState:
    print("--- [7] System: Generating PDF Report ---")
    report = state['final_report']
    
    # Check for visualizations
    if "static/plots" not in report:
        import glob
        plots = glob.glob("static/plots/*.png")
        if plots:
            report += "\n\n## Visualizations\n"
            for plot in plots:
                # Use relative path for markdown
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
workflow.add_node("ml_preflight", run_ml_preflight)
workflow.add_node("execute_code", execute_code)
workflow.add_node("evaluate_results", run_result_evaluator) # New Node
workflow.add_node("retry_handler", retry_handler)
workflow.add_node("prepare_runtime_fix", prepare_runtime_fix) # New Node
workflow.add_node("postmortem", run_postmortem)

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
        "passed": "reviewer",
        "failed": "retry_handler",
    }
)

# Conditional Edge for Data Engineer Failure
workflow.add_conditional_edges(
    "data_engineer",
    check_data_success,
    {
        "success": "engineer",
        "failed": "postmortem" # Let postmortem decide retry/stop
    }
)

# Conditional Edge for Reviewer
workflow.add_conditional_edges(
    "reviewer",
    check_review,
    {
        "approved": "qa_reviewer", # Pass to QA instead of execute
        "rejected": "retry_handler", 
        "failed": "postmortem"
    }
)

# Conditional Edge for QA Reviewer
workflow.add_conditional_edges(
    "qa_reviewer",
    check_qa_review,
    {
        "approved": "execute_code", # Pass QA -> Execute
        "rejected": "retry_handler", # Back to engineer
        "failed": "postmortem"
    }
)

# New Flow: Execution -> Loop
workflow.add_conditional_edges(
    "execute_code",
    check_execution_status,
    {
        "evaluate": "evaluate_results",
        "retry_fix": "prepare_runtime_fix",
        "failed": "postmortem"
    }
)

workflow.add_edge("prepare_runtime_fix", "engineer")

# Conditional Edge for Result Evaluation
workflow.add_conditional_edges(
    "evaluate_results",
    check_evaluation,
    {
        "retry": "postmortem",
        "approved": "translator"
    }
)

workflow.add_edge("retry_handler", "engineer")
workflow.add_conditional_edges(
    "postmortem",
    check_postmortem_action,
    {
        "retry_de": "data_engineer",
        "retry_ml": "engineer",
        "restrat": "strategist",
        "stop": "translator",
    }
)
workflow.add_edge("translator", "generate_pdf")
workflow.add_edge("generate_pdf", END)

# 4. Compile
app_graph = workflow.compile()
