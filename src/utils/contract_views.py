from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from src.utils.contract_v41 import (
    get_canonical_columns,
    get_column_roles,
    get_cleaning_gates,
    get_derived_column_names,
    get_qa_gates,
    get_required_outputs,
    get_validation_requirements,
)

# Shared helper for decisioning requirements extraction
def _get_decisioning_requirements(contract_full: Dict[str, Any], contract_min: Dict[str, Any]) -> Dict[str, Any]:
    for source in (contract_min, contract_full):
        if not isinstance(source, dict):
            continue
        decisioning = source.get("decisioning_requirements")
        if isinstance(decisioning, dict) and decisioning:
            return {
                "enabled": bool(decisioning.get("enabled")),
                "required": bool(decisioning.get("required")),
                "output": decisioning.get("output", {}),
                "policy_notes": decisioning.get("policy_notes", ""),
            }
    return {"enabled": False, "required": False, "output": {}, "policy_notes": ""}

# contract_full: traceability/strategy; contract_min: binding; views: prompt context.


class DEView(TypedDict, total=False):
    role: str
    required_columns: List[str]
    optional_passthrough_columns: List[str]
    output_path: str
    output_manifest_path: str
    output_dialect: Dict[str, Any]
    constraints: Dict[str, Any]


class MLView(TypedDict, total=False):
    role: str
    objective_type: str
    canonical_columns: List[str]
    derived_features: List[str]
    column_roles: Dict[str, List[str]]
    decision_columns: List[str]
    outcome_columns: List[str]
    audit_only_columns: List[str]
    identifier_columns: List[str]
    allowed_feature_sets: Dict[str, Any]
    forbidden_features: List[str]
    required_outputs: List[str]
    validation_requirements: Dict[str, Any]
    case_rules: Any
    plot_spec: Dict[str, Any]


class ReviewerView(TypedDict, total=False):
    role: str
    objective_type: str
    reviewer_gates: List[Any]
    required_outputs: List[str]
    expected_metrics: List[str]
    strategy_summary: str
    verification: Dict[str, Any]


class TranslatorView(TypedDict, total=False):
    role: str
    reporting_policy: Dict[str, Any]
    plot_spec: Dict[str, Any]
    evidence_inventory: List[Dict[str, Any]]
    key_decisions: List[str]
    limitations: List[str]
    constraints: Dict[str, Any]


class ResultsAdvisorView(TypedDict, total=False):
    role: str
    objective_type: str
    reporting_policy: Dict[str, Any]
    evidence_inventory: List[Dict[str, Any]]


class CleaningView(TypedDict, total=False):
    role: str
    strategy_title: str
    business_objective: str
    required_columns: List[str]
    dialect: Dict[str, Any]
    cleaning_gates: List[Dict[str, Any]]
    column_roles: Dict[str, List[str]]
    allowed_feature_sets: Dict[str, Any]


class QAView(TypedDict, total=False):
    role: str
    qa_gates: List[Dict[str, Any]]
    artifact_requirements: Dict[str, Any]
    allowed_feature_sets: Dict[str, Any]
    column_roles: Dict[str, List[str]]
    canonical_columns: List[str]
    objective_summary: Dict[str, str]
    reporting_policy: Dict[str, Any]


_PRESERVE_KEYS = {
    "required_columns",
    "required_outputs",
    "forbidden_features",
    "reviewer_gates",
    "qa_gates",
    "cleaning_gates",
    "gates",
    "plot_spec",
    "plots",
}

_IDENTIFIER_TOKENS = {
    "id",
    "uuid",
    "guid",
    "key",
    "codigo",
    "code",
    "cod",
    "identifier",
    "reference",
    "ref",
    "account",
    "entity",
}
_SHORT_IDENTIFIER_TOKENS = {"id", "cod", "ref", "key"}
_LONG_IDENTIFIER_TOKENS = sorted(_IDENTIFIER_TOKENS - _SHORT_IDENTIFIER_TOKENS)
_STRICT_IDENTIFIER_EXACT = {
    "id",
    "uuid",
    "guid",
    "rowid",
    "recordid",
    "row_id",
    "record_id",
    "index",
    "idx",
    "record",
}
_STRICT_IDENTIFIER_PATTERN = re.compile(r"^(row|record)[ _\-]?id$", re.IGNORECASE)
_CANDIDATE_IDENTIFIER_TOKENS = {"key", "ref", "code", "cod"}
_CANDIDATE_SUFFIXES = ("_id", "-id", " id")


def _coerce_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def is_identifier_column(col_name: str) -> bool:
    if not col_name:
        return False
    if is_strict_identifier_column(col_name):
        return True
    if is_candidate_identifier_column(col_name):
        return True
    return False


def is_strict_identifier_column(col_name: str) -> bool:
    if not col_name:
        return False
    raw = str(col_name)
    lowered = raw.lower().replace("-", "_")
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "", lowered)
    if normalized in _STRICT_IDENTIFIER_EXACT:
        return True
    if _STRICT_IDENTIFIER_PATTERN.match(raw):
        return True
    tokens = [t for t in re.split(r"[^0-9a-zA-Z]+", lowered) if t]
    camel_tokens = [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\\d+", col_name) if t]
    if any(token in {"uuid", "guid"} for token in (tokens + camel_tokens)):
        return True
    return False


def is_candidate_identifier_column(col_name: str) -> bool:
    if not col_name:
        return False
    if is_strict_identifier_column(col_name):
        return False
    raw = str(col_name)
    lowered = raw.lower()
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "", lowered)
    if normalized.endswith("_id") or normalized.endswith("-id") or normalized.endswith(" id"):
        return True
    tokens = [t for t in re.split(r"[^0-9a-zA-Z]+", lowered) if t]
    camel_tokens = [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\\d+", col_name) if t]
    if any(token in _CANDIDATE_IDENTIFIER_TOKENS for token in (tokens + camel_tokens)):
        return True
    if re.search(r"[A-Za-z]+Id$", raw):
        return True
    return False


def _resolve_required_outputs(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    outputs = contract_min.get("required_outputs")
    if isinstance(outputs, list) and outputs:
        return [str(p) for p in outputs if p]
    outputs = get_required_outputs(contract_full)
    if outputs:
        return [str(p) for p in outputs if p]
    return []


def _resolve_required_columns(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    clean_cfg = _coerce_dict(artifact_reqs.get("clean_dataset"))
    required = clean_cfg.get("required_columns")
    if isinstance(required, list) and required:
        return [str(c) for c in required if c]
    canonical = contract_min.get("canonical_columns")
    if isinstance(canonical, list) and canonical:
        return [str(c) for c in canonical if c]
    canonical = get_canonical_columns(contract_full)
    if canonical:
        return [str(c) for c in canonical if c]
    return []


def _resolve_passthrough_columns(
    contract_min: Dict[str, Any], contract_full: Dict[str, Any], required_columns: List[str]
) -> List[str]:
    allowed_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    audit_only = allowed_sets.get("audit_only_features")
    if not isinstance(audit_only, list) or not audit_only:
        return []
    required_set = {str(c) for c in required_columns if c}
    passthrough = [str(c) for c in audit_only if c]
    return [c for c in passthrough if c not in required_set]


def _resolve_output_path(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    required_outputs: List[str],
) -> Optional[str]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    clean_cfg = _coerce_dict(artifact_reqs.get("clean_dataset"))
    output_path = _first_value(clean_cfg.get("output_path"), clean_cfg.get("output"), clean_cfg.get("path"))
    if output_path:
        return str(output_path)
    for path in required_outputs:
        lower = str(path).lower()
        if "cleaned" in lower and lower.endswith(".csv"):
            return str(path)
    return "data/cleaned_data.csv"


def _resolve_manifest_path(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    required_outputs: List[str],
) -> Optional[str]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    clean_cfg = _coerce_dict(artifact_reqs.get("clean_dataset"))
    manifest_path = _first_value(clean_cfg.get("manifest_path"), clean_cfg.get("output_manifest_path"))
    if manifest_path:
        return str(manifest_path)
    for path in required_outputs:
        if "cleaning_manifest" in str(path).lower():
            return str(path)
    return None


def _resolve_output_dialect(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    dialect = contract_min.get("output_dialect")
    if isinstance(dialect, dict) and dialect:
        return dialect
    dialect = contract_full.get("output_dialect")
    if isinstance(dialect, dict) and dialect:
        return dialect
    return {}


def _normalize_artifact_index(entries: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in entries or []:
        if isinstance(item, dict) and item.get("path"):
            normalized.append(
                {
                    "path": str(item.get("path")),
                    "artifact_type": item.get("artifact_type") or item.get("type"),
                }
            )
        elif isinstance(item, str):
            normalized.append({"path": item, "artifact_type": "artifact"})
    return normalized


def _resolve_objective_type(contract_min: Dict[str, Any], contract_full: Dict[str, Any], required_outputs: List[str]) -> str:
    for source in (contract_min, contract_full):
        eval_spec = source.get("evaluation_spec") if isinstance(source, dict) else None
        if isinstance(eval_spec, dict):
            obj = eval_spec.get("objective_type")
            if obj:
                return str(obj)
    plan = contract_full.get("execution_plan") if isinstance(contract_full, dict) else None
    if isinstance(plan, dict) and plan.get("objective_type"):
        return str(plan.get("objective_type"))
    obj_analysis = contract_full.get("objective_analysis") if isinstance(contract_full, dict) else None
    if isinstance(obj_analysis, dict) and obj_analysis.get("problem_type"):
        return str(obj_analysis.get("problem_type"))
    return _infer_objective_from_outputs(required_outputs)


def _infer_objective_from_outputs(required_outputs: List[str]) -> str:
    tokens = " ".join([str(p).lower() for p in required_outputs or []])
    if "forecast" in tokens:
        return "forecasting"
    if "segment" in tokens or "cluster" in tokens:
        return "segmentation"
    if "weight" in tokens or "optimization" in tokens:
        return "optimization"
    if "score" in tokens or "predict" in tokens:
        return "prediction"
    return "unknown"


def _resolve_column_roles(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, List[str]]:
    roles_min = get_column_roles(contract_min)
    if roles_min:
        return roles_min
    return get_column_roles(contract_full)


def _extract_roles(roles: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        "decision": _coerce_list(roles.get("decision")),
        "outcome": _coerce_list(roles.get("outcome")),
        "audit_only": _coerce_list(roles.get("post_decision_audit_only") or roles.get("audit_only")),
    }


def _resolve_derived_columns(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    derived: List[str] = []
    for source in (contract_min, contract_full):
        if not isinstance(source, dict):
            continue
        derived.extend(get_derived_column_names(source))
    return list(dict.fromkeys([str(c) for c in derived if c]))


def _resolve_allowed_feature_sets(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    allowed = contract_min.get("allowed_feature_sets")
    if isinstance(allowed, dict) and allowed:
        return allowed
    allowed = contract_full.get("allowed_feature_sets")
    if isinstance(allowed, dict) and allowed:
        return allowed
    return {"segmentation_features": [], "model_features": [], "forbidden_features": []}


def _resolve_forbidden_features(allowed_feature_sets: Dict[str, Any]) -> List[str]:
    forbidden = allowed_feature_sets.get("forbidden_features")
    if isinstance(forbidden, list):
        return [str(c) for c in forbidden if c]
    forbidden = allowed_feature_sets.get("forbidden_for_modeling")
    if isinstance(forbidden, list):
        return [str(c) for c in forbidden if c]
    return []


def _resolve_validation_requirements(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    validation = contract_min.get("validation_requirements")
    if isinstance(validation, dict) and validation:
        return validation
    eval_spec = contract_min.get("evaluation_spec") if isinstance(contract_min, dict) else None
    if isinstance(eval_spec, dict):
        validation = eval_spec.get("validation_requirements")
        if isinstance(validation, dict) and validation:
            return validation
    validation = get_validation_requirements(contract_full)
    if validation:
        return validation
    eval_spec = contract_full.get("evaluation_spec") if isinstance(contract_full, dict) else None
    if isinstance(eval_spec, dict):
        validation = eval_spec.get("validation_requirements")
        if isinstance(validation, dict) and validation:
            return validation
    return {}


def _resolve_qa_gates(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[Dict[str, Any]]:
    gates = get_qa_gates(contract_full)
    if gates:
        return gates
    gates = get_qa_gates(contract_min)
    if gates:
        return gates
    def _normalize(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for gate in raw:
            if isinstance(gate, dict):
                name = gate.get("name") or gate.get("id") or gate.get("gate")
                if not name:
                    continue
                severity = str(gate.get("severity") or "HARD").upper()
                if severity not in {"HARD", "SOFT"}:
                    severity = "HARD"
                params = gate.get("params")
                if not isinstance(params, dict):
                    params = {}
                key = str(name).lower()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append({"name": str(name), "severity": severity, "params": params})
            elif isinstance(gate, str) and gate.strip():
                key = gate.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append({"name": gate.strip(), "severity": "HARD", "params": {}})
        return normalized
    eval_spec = contract_full.get("evaluation_spec") if isinstance(contract_full, dict) else None
    if isinstance(eval_spec, dict):
        gates = _normalize(eval_spec.get("qa_gates"))
        if gates:
            return gates
    eval_spec = contract_min.get("evaluation_spec") if isinstance(contract_min, dict) else None
    if isinstance(eval_spec, dict):
        gates = _normalize(eval_spec.get("qa_gates"))
        if gates:
            return gates
    return []


def _resolve_cleaning_gates(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[Dict[str, Any]]:
    gates = get_cleaning_gates(contract_full)
    if gates:
        return gates
    gates = get_cleaning_gates(contract_min)
    if gates:
        return gates
    eval_spec = contract_full.get("evaluation_spec") if isinstance(contract_full, dict) else None
    if isinstance(eval_spec, dict):
        raw = eval_spec.get("cleaning_gates")
        if isinstance(raw, list):
            return [g for g in raw if isinstance(g, dict)]
    eval_spec = contract_min.get("evaluation_spec") if isinstance(contract_min, dict) else None
    if isinstance(eval_spec, dict):
        raw = eval_spec.get("cleaning_gates")
        if isinstance(raw, list):
            return [g for g in raw if isinstance(g, dict)]
    return []


def _resolve_optional_outputs(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    optional_outputs: List[str] = []
    for key in ("optional_files", "optional_outputs"):
        values = artifact_reqs.get(key)
        if isinstance(values, list):
            optional_outputs.extend([str(v) for v in values if v])
    for value in artifact_reqs.values():
        if not isinstance(value, dict):
            continue
        if value.get("optional") is True:
            path = _first_value(value.get("path"), value.get("output_path"), value.get("output"))
            if path:
                optional_outputs.append(str(path))
            expected = value.get("expected")
            if isinstance(expected, list):
                optional_outputs.extend([str(v) for v in expected if v])
    return list(dict.fromkeys([str(v) for v in optional_outputs if v]))


def _resolve_case_rules(contract_full: Dict[str, Any]) -> Any:
    for path in (
        ("case_rules",),
        ("case_taxonomy",),
        ("spec_extraction", "case_taxonomy"),
        ("evaluation_spec", "case_taxonomy"),
        ("spec_extraction", "case_rules"),
    ):
        cursor: Any = contract_full
        for key in path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(key)
        if cursor:
            return cursor
    return None


def _resolve_reviewer_gates(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[Any]:
    gates = contract_min.get("reviewer_gates")
    if isinstance(gates, list) and gates:
        return gates
    eval_spec = contract_min.get("evaluation_spec")
    if isinstance(eval_spec, dict) and isinstance(eval_spec.get("reviewer_gates"), list):
        return eval_spec.get("reviewer_gates") or []
    eval_spec = contract_full.get("evaluation_spec") if isinstance(contract_full, dict) else None
    if isinstance(eval_spec, dict) and isinstance(eval_spec.get("reviewer_gates"), list):
        return eval_spec.get("reviewer_gates") or []
    gates = contract_full.get("reviewer_gates")
    if isinstance(gates, list):
        return gates
    return []


def _summarize_strategy(contract_full: Dict[str, Any], contract_min: Dict[str, Any], max_chars: int = 180) -> str:
    title = _first_value(contract_full.get("strategy_title"), contract_min.get("strategy_title"))
    objective_type = _resolve_objective_type(contract_min, contract_full, [])
    summary = f"{title or 'Strategy'} | objective={objective_type}"
    return summary[:max_chars]


def _expected_metrics_from_objective(objective_type: str, reviewer_gates: List[Any]) -> List[str]:
    obj = str(objective_type or "").lower()
    if "classif" in obj:
        return ["auc", "f1", "precision", "recall"]
    if "regress" in obj:
        return ["rmse", "mae", "r2"]
    if "forecast" in obj:
        return ["mae", "rmse"]
    if "rank" in obj:
        return ["ndcg", "map"]
    if "segment" in obj or "cluster" in obj:
        return ["silhouette"]
    if reviewer_gates:
        return ["metric_required_by_gate"]
    return []


def _build_min_reporting_policy(artifact_index: List[Dict[str, Any]]) -> Dict[str, Any]:
    artifacts = _normalize_artifact_index(artifact_index)
    artifact_types = {str(item.get("artifact_type") or "").lower() for item in artifacts if isinstance(item, dict)}
    slots = []
    sections = ["Executive Decision", "Evidence & Metrics", "Risks & Limitations", "Next Actions"]
    if "metrics" in artifact_types:
        slots.append({"id": "model_metrics", "mode": "required", "sources": ["data/metrics.json"]})
    if "predictions" in artifact_types:
        slots.append({"id": "predictions_overview", "mode": "optional", "sources": ["data/scored_rows.csv"]})
    if "insights" in artifact_types:
        slots.append({"id": "insights", "mode": "optional", "sources": ["data/insights.json"]})
    if "report" in artifact_types:
        slots.append({"id": "alignment_check", "mode": "optional", "sources": ["data/alignment_check.json"]})
    return {"sections": sections, "slots": slots}


def _truncate_text(value: str, max_len: int) -> str:
    if not isinstance(value, str) or len(value) <= max_len:
        return value
    return value[: max_len - 14] + "...[TRUNCATED]"


def _cap_plot_spec(plot_spec: Dict[str, Any] | None, max_caption_chars: int = 400) -> Dict[str, Any] | None:
    if not isinstance(plot_spec, dict) or not plot_spec:
        return None
    capped = dict(plot_spec)
    max_plots = plot_spec.get("max_plots")
    max_plots_int: int | None = None
    if isinstance(max_plots, (int, float)):
        max_plots_int = int(max_plots)
    elif isinstance(max_plots, str) and max_plots.strip().isdigit():
        max_plots_int = int(max_plots.strip())
    if max_plots_int is not None:
        capped["max_plots"] = max_plots_int
    plots = plot_spec.get("plots")
    if isinstance(plots, list):
        limit = max_plots_int if max_plots_int is not None and max_plots_int >= 0 else len(plots)
        trimmed = []
        for item in plots[:limit]:
            if not isinstance(item, dict):
                continue
            plot = dict(item)
            caption = plot.get("caption_template")
            if isinstance(caption, str):
                plot["caption_template"] = _truncate_text(caption, max_caption_chars)
            trimmed.append(plot)
        capped["plots"] = trimmed
    return capped


def _trim_value(
    obj: Any,
    max_str_len: int,
    max_list_items: int,
    preserve_keys: set[str],
    path: List[str],
) -> Any:
    if isinstance(obj, str):
        return _truncate_text(obj, max_str_len)
    if isinstance(obj, list):
        key = path[-1] if path else ""
        if key not in preserve_keys and len(obj) > max_list_items:
            trimmed = obj[:max_list_items]
            trimmed.append(f"...({len(obj)} total)")
            obj = trimmed
        return [
            _trim_value(item, max_str_len, max_list_items, preserve_keys, path + ["[]"])
            for item in obj
        ]
    if isinstance(obj, dict):
        trimmed: Dict[str, Any] = {}
        for key in sorted(obj.keys()):
            trimmed[key] = _trim_value(obj[key], max_str_len, max_list_items, preserve_keys, path + [key])
        return trimmed
    return obj


def trim_to_budget(obj: Any, max_chars: int) -> Any:
    if obj is None:
        return obj
    max_str_len = 1200
    max_list_items = 25
    for _ in range(4):
        trimmed = _trim_value(obj, max_str_len, max_list_items, _PRESERVE_KEYS, [])
        payload = json.dumps(trimmed, ensure_ascii=True, sort_keys=True)
        if len(payload) <= max_chars:
            return trimmed
        max_str_len = max(200, int(max_str_len * 0.7))
        max_list_items = max(8, int(max_list_items * 0.7))
    return trimmed


def build_de_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    required_columns = _resolve_required_columns(contract_min, contract_full)
    passthrough_columns = _resolve_passthrough_columns(contract_min, contract_full, required_columns)
    output_path = _resolve_output_path(contract_min, contract_full, required_outputs)
    manifest_path = _resolve_manifest_path(contract_min, contract_full, required_outputs)
    view: DEView = {
        "role": "data_engineer",
        "required_columns": required_columns,
        "optional_passthrough_columns": passthrough_columns,
        "output_path": output_path or "data/cleaned_data.csv",
        "constraints": {
            "scope": "cleaning_only",
            "hard_constraints": [
                "no_modeling",
                "no_score_fitting",
                "no_prescriptive_tuning",
                "no_analytics",
            ],
        },
    }
    if manifest_path:
        view["output_manifest_path"] = manifest_path
    output_dialect = _resolve_output_dialect(contract_min, contract_full)
    if output_dialect:
        view["output_dialect"] = output_dialect
    return trim_to_budget(view, 8000)


def build_ml_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    objective_type = _resolve_objective_type(contract_min, contract_full, required_outputs)
    canonical_columns = contract_min.get("canonical_columns")
    if not isinstance(canonical_columns, list):
        canonical_columns = get_canonical_columns(contract_full)
    canonical_columns = [str(c) for c in canonical_columns if c]
    roles_min = get_column_roles(contract_min)
    roles_full = get_column_roles(contract_full)
    allowed_sets_min = contract_min.get("allowed_feature_sets")
    if not isinstance(allowed_sets_min, dict):
        allowed_sets_min = {}
    allowed_sets_full = contract_full.get("allowed_feature_sets")
    if not isinstance(allowed_sets_full, dict):
        allowed_sets_full = {}
    forbidden_min = _resolve_forbidden_features(allowed_sets_min)
    role_sets_min = _extract_roles(roles_min)
    pre_decision_min = _coerce_list(roles_min.get("pre_decision"))
    decision_min = _coerce_list(roles_min.get("decision"))
    outcome_min = _coerce_list(roles_min.get("outcome"))
    canonical_set = set(canonical_columns)
    lax_roles = False
    if not roles_min:
        lax_roles = True
    elif not role_sets_min.get("audit_only") and not forbidden_min and not decision_min and not outcome_min:
        if canonical_set:
            overlap = len(set(pre_decision_min) & canonical_set)
            coverage = overlap / max(1, len(canonical_set))
            if coverage >= 0.9:
                lax_roles = True
        else:
            lax_roles = True
    use_full_roles = bool(roles_full) and lax_roles
    column_roles = roles_full if use_full_roles else roles_min
    if not column_roles:
        column_roles = roles_full or roles_min
    role_sets = _extract_roles(column_roles)
    audit_only_cols = [str(c) for c in role_sets.get("audit_only", []) if c]
    decision_cols = [str(c) for c in role_sets.get("decision", []) if c]
    outcome_cols = [str(c) for c in role_sets.get("outcome", []) if c]
    pre_decision_cols = _coerce_list(column_roles.get("pre_decision"))
    if not pre_decision_cols:
        assigned = set(decision_cols + outcome_cols + audit_only_cols)
        pre_decision_cols = [c for c in canonical_columns if c and c not in assigned]

    def _list_or_none(source: Dict[str, Any], key: str) -> List[str] | None:
        val = source.get(key)
        if isinstance(val, list):
            return [str(c) for c in val if c]
        return None

    full_model = _list_or_none(allowed_sets_full, "model_features")
    full_seg = _list_or_none(allowed_sets_full, "segmentation_features")
    full_forbidden = _list_or_none(allowed_sets_full, "forbidden_for_modeling")
    if full_forbidden is None:
        full_forbidden = _list_or_none(allowed_sets_full, "forbidden_features")
    full_audit = _list_or_none(allowed_sets_full, "audit_only_features")

    min_model = _list_or_none(allowed_sets_min, "model_features")
    min_seg = _list_or_none(allowed_sets_min, "segmentation_features")
    min_forbidden = _list_or_none(allowed_sets_min, "forbidden_features")
    if min_forbidden is None:
        min_forbidden = _list_or_none(allowed_sets_min, "forbidden_for_modeling")
    min_audit = _list_or_none(allowed_sets_min, "audit_only_features")

    model_features = (
        full_model
        if full_model is not None
        else (min_model if min_model is not None else list(dict.fromkeys(pre_decision_cols + decision_cols)))
    )
    segmentation_features = (
        full_seg
        if full_seg is not None
        else (min_seg if min_seg is not None else list(pre_decision_cols))
    )
    forbidden = (
        full_forbidden
        if full_forbidden is not None
        else (min_forbidden if min_forbidden is not None else [])
    )
    audit_only_features = full_audit if full_audit is not None else min_audit
    if audit_only_features is not None:
        audit_only_cols = list(dict.fromkeys([str(c) for c in audit_only_features if c]))

    explicit_allowed = set()
    for candidate_list in (full_model, min_model, full_seg, min_seg):
        if isinstance(candidate_list, list):
            explicit_allowed.update(str(c) for c in candidate_list if c)

    derived_columns = _resolve_derived_columns(contract_min, contract_full)
    derived_set = {str(c) for c in derived_columns if c}
    allowed_noncanonical = set(explicit_allowed) | set(derived_set)

    strict_ids = []
    candidate_ids = []
    seen = set()
    for col in canonical_columns:
        if not col:
            continue
        if col in seen:
            continue
        seen.add(col)
        if is_strict_identifier_column(col):
            strict_ids.append(col)
        elif is_candidate_identifier_column(col):
            candidate_ids.append(col)

    strict_allowed_by_contract = [c for c in strict_ids if c in explicit_allowed]
    candidate_allowed_by_contract = [c for c in candidate_ids if c in explicit_allowed]
    strict_forbidden = [c for c in strict_ids if c not in explicit_allowed]

    forbidden = list(dict.fromkeys([str(c) for c in forbidden if c]))
    forbidden = sorted(dict.fromkeys(forbidden + strict_forbidden + outcome_cols + audit_only_cols))

    def _filter_noncanonical(items: List[str]) -> tuple[List[str], List[str]]:
        filtered = []
        dropped = []
        for val in items:
            if not val:
                continue
            if val in canonical_set or val in allowed_noncanonical:
                filtered.append(str(val))
            else:
                dropped.append(str(val))
        return filtered, dropped

    forbidden_filtered, dropped_forbidden = _filter_noncanonical(forbidden)
    forbidden_set = set(forbidden_filtered)
    model_features_filtered, dropped_model = _filter_noncanonical(model_features)
    segmentation_features_filtered, dropped_segmentation = _filter_noncanonical(segmentation_features)

    view_warnings: Dict[str, Any] = {}
    dropped_noncanonical = {}
    if dropped_model:
        dropped_noncanonical["model_features"] = dropped_model
    if dropped_segmentation:
        dropped_noncanonical["segmentation_features"] = dropped_segmentation
    if dropped_forbidden:
        dropped_noncanonical["forbidden_features"] = dropped_forbidden
    if dropped_noncanonical:
        view_warnings["dropped_noncanonical"] = dropped_noncanonical

    model_features = [c for c in model_features_filtered if c not in forbidden_set]
    segmentation_features = [c for c in segmentation_features_filtered if c not in forbidden_set]
    derived_features = list(
        dict.fromkeys([c for c in derived_columns if c in set(model_features + segmentation_features)])
    )
    final_forbidden = forbidden_filtered
    allowed_sets = {
        "segmentation_features": segmentation_features,
        "model_features": model_features,
        "forbidden_features": final_forbidden,
    }
    if audit_only_features is not None:
        allowed_sets["audit_only_features"] = [str(c) for c in audit_only_features if c]
    validation = _resolve_validation_requirements(contract_min, contract_full)
    case_rules = _resolve_case_rules(contract_full)

    view: MLView = {
        "role": "ml_engineer",
        "objective_type": objective_type,
        "canonical_columns": canonical_columns,
        "derived_features": derived_features,
        "column_roles": column_roles,
        "decision_columns": decision_cols,
        "outcome_columns": outcome_cols,
        "audit_only_columns": audit_only_cols,
        "identifier_columns": strict_ids + candidate_ids,
        "allowed_feature_sets": allowed_sets,
        "forbidden_features": final_forbidden,
        "required_outputs": required_outputs,
        "validation_requirements": validation,
    }
    training_rows_rule = contract_min.get("training_rows_rule") or contract_full.get("training_rows_rule")
    scoring_rows_rule = contract_min.get("scoring_rows_rule") or contract_full.get("scoring_rows_rule")
    data_partitioning_notes = contract_min.get("data_partitioning_notes") or contract_full.get("data_partitioning_notes")
    if training_rows_rule:
        view["training_rows_rule"] = training_rows_rule
    if scoring_rows_rule:
        view["scoring_rows_rule"] = scoring_rows_rule
    if isinstance(data_partitioning_notes, list) and data_partitioning_notes:
        view["data_partitioning_notes"] = data_partitioning_notes
    identifier_policy = {
        "strict_forbidden": strict_forbidden,
        "candidates": candidate_ids,
        "guidance": [
            "Candidate identifiers may be useful categorical features if low-cardinality.",
            "If a candidate identifier is high-cardinality or near-unique, drop it.",
            "Perform a quick uniqueness/cardinality check on a sample before using.",
        ],
    }
    view["identifier_policy"] = identifier_policy
    view["identifier_overrides"] = {
        "strict_allowed_by_contract": strict_allowed_by_contract,
        "candidate_allowed_by_contract": candidate_allowed_by_contract,
    }
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    visual_reqs = (
        artifact_reqs.get("visual_requirements") if isinstance(artifact_reqs.get("visual_requirements"), dict) else {}
    )
    visual_payload = {
        "enabled": bool(visual_reqs.get("enabled")),
        "required": bool(visual_reqs.get("required")),
        "outputs_dir": visual_reqs.get("outputs_dir") or "static/plots",
        "items": visual_reqs.get("items") if isinstance(visual_reqs.get("items"), list) else [],
        "notes": visual_reqs.get("notes") or "",
    }
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    if view_warnings:
        view["view_warnings"] = view_warnings
    if case_rules is not None:
        view["case_rules"] = case_rules
    policy = contract_full.get("reporting_policy")
    if not isinstance(policy, dict) or not policy:
        policy = contract_min.get("reporting_policy")
    plot_spec = _cap_plot_spec(policy.get("plot_spec") if isinstance(policy, dict) else None)
    if plot_spec is not None:
        visual_payload["enabled"] = bool(plot_spec.get("enabled", True))
        view["plot_spec"] = plot_spec
    view["visual_requirements"] = visual_payload
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    return trim_to_budget(view, 16000)


def build_reviewer_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    objective_type = _resolve_objective_type(contract_min, contract_full, required_outputs)
    reviewer_gates = _resolve_reviewer_gates(contract_min, contract_full)
    expected_metrics = _expected_metrics_from_objective(objective_type, reviewer_gates)
    strategy_summary = _summarize_strategy(contract_full, contract_min)
    view: ReviewerView = {
        "role": "reviewer",
        "objective_type": objective_type,
        "reviewer_gates": reviewer_gates,
        "required_outputs": required_outputs,
        "expected_metrics": expected_metrics,
        "strategy_summary": strategy_summary,
        "verification": {
            "required_outputs": required_outputs,
            "artifact_index_expected": bool(artifact_index),
        },
    }
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    return trim_to_budget(view, 12000)


def build_qa_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    optional_outputs = _resolve_optional_outputs(contract_min, contract_full)
    qa_gates = _resolve_qa_gates(contract_min, contract_full)
    allowed_feature_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    column_roles = _resolve_column_roles(contract_min, contract_full)
    canonical_columns = contract_min.get("canonical_columns")
    if not isinstance(canonical_columns, list):
        canonical_columns = get_canonical_columns(contract_full)
    canonical_columns = [str(c) for c in canonical_columns if c]
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    artifact_payload: Dict[str, Any] = {
        "required_outputs": required_outputs,
        "optional_outputs": optional_outputs,
    }
    file_schemas = artifact_reqs.get("file_schemas")
    if isinstance(file_schemas, dict) and file_schemas:
        artifact_payload["file_schemas"] = file_schemas
    objective_summary = {
        "strategy_title": _first_value(contract_full.get("strategy_title"), contract_min.get("strategy_title")) or "",
        "business_objective": _first_value(contract_full.get("business_objective"), contract_min.get("business_objective"))
        or "",
    }
    reporting_policy = contract_full.get("reporting_policy")
    if not isinstance(reporting_policy, dict) or not reporting_policy:
        reporting_policy = contract_min.get("reporting_policy")
    view: QAView = {
        "role": "qa_reviewer",
        "qa_gates": qa_gates,
        "artifact_requirements": artifact_payload,
        "allowed_feature_sets": allowed_feature_sets,
        "column_roles": column_roles,
        "canonical_columns": canonical_columns,
        "objective_summary": objective_summary,
    }
    if isinstance(reporting_policy, dict) and reporting_policy:
        view["reporting_policy"] = reporting_policy
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    return trim_to_budget(view, 12000)


def build_translator_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
    insights: Any = None,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    policy = contract_full.get("reporting_policy")
    if not isinstance(policy, dict) or not policy:
        policy = contract_min.get("reporting_policy")
    if not isinstance(policy, dict) or not policy:
        policy = _build_min_reporting_policy(_normalize_artifact_index(artifact_index))
    evidence = _normalize_artifact_index(artifact_index)
    objective_type = _resolve_objective_type(contract_min, contract_full, [])
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    key_decisions = []
    if objective_type:
        key_decisions.append(f"objective_type:{objective_type}")
    if required_outputs:
        key_decisions.append(f"required_outputs:{len(required_outputs)}")
    limitations = []
    risks = contract_full.get("data_risks") if isinstance(contract_full, dict) else None
    if isinstance(risks, list):
        limitations.extend([str(item) for item in risks if item])
    view: TranslatorView = {
        "role": "translator",
        "reporting_policy": policy,
        "evidence_inventory": evidence,
        "key_decisions": key_decisions,
        "limitations": limitations,
        "constraints": {"no_markdown_tables": True, "cite_sources": True},
    }
    plot_spec = _cap_plot_spec(policy.get("plot_spec") if isinstance(policy, dict) else None)
    if plot_spec is not None:
        view["plot_spec"] = plot_spec
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    visual_reqs = (
        artifact_reqs.get("visual_requirements") if isinstance(artifact_reqs.get("visual_requirements"), dict) else {}
    )
    visual_payload = {
        "enabled": bool(visual_reqs.get("enabled")),
        "required": bool(visual_reqs.get("required")),
        "outputs_dir": visual_reqs.get("outputs_dir") or "static/plots",
        "items": visual_reqs.get("items") if isinstance(visual_reqs.get("items"), list) else [],
        "notes": visual_reqs.get("notes") or "",
    }
    if plot_spec is not None:
        visual_payload["enabled"] = bool(plot_spec.get("enabled", True))
    view["visual_requirements"] = visual_payload
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    return trim_to_budget(view, 16000)


def build_results_advisor_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    objective_type = _resolve_objective_type(contract_min, contract_full, required_outputs)
    policy = contract_full.get("reporting_policy")
    if not isinstance(policy, dict) or not policy:
        policy = contract_min.get("reporting_policy")
    if not isinstance(policy, dict) or not policy:
        policy = {}
    view: ResultsAdvisorView = {
        "role": "results_advisor",
        "objective_type": objective_type,
        "reporting_policy": policy,
        "evidence_inventory": _normalize_artifact_index(artifact_index),
    }
    return trim_to_budget(view, 12000)


def build_cleaning_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    required_columns = _resolve_required_columns(contract_min, contract_full)
    column_roles = _resolve_column_roles(contract_min, contract_full)
    allowed_feature_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    dialect = _resolve_output_dialect(contract_min, contract_full)
    cleaning_gates = _resolve_cleaning_gates(contract_min, contract_full)
    view: CleaningView = {
        "role": "cleaning_reviewer",
        "strategy_title": _first_value(contract_full.get("strategy_title"), contract_min.get("strategy_title")) or "",
        "business_objective": _first_value(
            contract_full.get("business_objective"), contract_min.get("business_objective")
        ) or "",
        "required_columns": required_columns,
        "dialect": dialect,
        "cleaning_gates": cleaning_gates,
        "column_roles": column_roles,
        "allowed_feature_sets": allowed_feature_sets,
    }
    return trim_to_budget(view, 12000)


def sanitize_contract_min_for_de(contract_min: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(contract_min, dict):
        return {}
    sanitized = dict(contract_min)
    sanitized.pop("business_objective", None)
    return sanitized


def persist_views(
    views: Dict[str, Dict[str, Any]],
    base_dir: str = "data",
    run_bundle_dir: Optional[str] = None,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if not views:
        return paths
    rel_dir = os.path.join("contracts", "views")
    out_dir = os.path.join(base_dir, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, payload in views.items():
        if not payload:
            continue
        filename = f"{name}.json"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        paths[name] = path
        if run_bundle_dir:
            bundle_path = os.path.join(run_bundle_dir, rel_dir, filename)
            os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
            with open(bundle_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
    return paths
