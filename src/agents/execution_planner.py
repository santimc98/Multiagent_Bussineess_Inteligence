import json
import os
import ast
from typing import Dict, Any, List, Optional, Tuple
from string import Template
import re
import difflib

from dotenv import load_dotenv
from src.agents.prompts import SENIOR_PLANNER_PROMPT
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from src.utils.contract_validation import (
    DEFAULT_DATA_ENGINEER_RUNBOOK,
    DEFAULT_ML_ENGINEER_RUNBOOK,
)
from src.utils.contract_v41 import (
    get_canonical_columns,
    get_column_roles,
    get_derived_column_names,
    get_required_outputs,
    strip_legacy_keys,
    assert_no_legacy_keys,
    assert_only_allowed_v41_keys,
    LEGACY_KEYS,
    CONTRACT_VERSION_V41,
)
from src.utils.run_bundle import get_run_dir
from src.utils.feature_selectors import infer_feature_selectors, compact_column_representation
from src.utils.contract_validator import validate_contract, normalize_artifact_requirements, is_probably_path

load_dotenv()


_QA_SEVERITIES = {"HARD", "SOFT"}
_CLEANING_SEVERITIES = {"HARD", "SOFT"}
_RESAMPLING_TOKENS = {
    "bootstrap",
    "resample",
    "resampling",
    "cross validation",
    "cross-validation",
    "cv",
    "kfold",
    "k-fold",
    "shuffle split",
    "stratified",
    "fold",
}

_DECISIONING_ENABLED_TOKENS = {
    "ranking",
    "priority",
    "prioritization",
    "top",
    "decision",
    "action",
    "triage",
    "moderation",
    "scorecard",
    "targeting",
    "outlier",
    "review",
    "segmentation",
    "operational",
    "rule",
    "uncertainty",
    "policy",
}

_DECISIONING_REQUIRED_PHRASES = {
    "ranking",
    "prioridad",
    "ranking prioritario",
    "decision policy",
    "política de decisión",
    "segmentación",
    "acción recomendada",
    "marcar casos",
    "baja confianza",
    "decision rule",
    "flag de revisión",
    "moderación humana",
    "prioritize",
    "review flag",
}

_EXPLANATION_REQUIRED_TOKENS = {
    "explain",
    "explanation",
    "explainability",
    "interpret",
    "interpretability",
    "justify",
    "justification",
    "driver",
    "drivers",
    "factor",
    "factors",
    "reason",
    "reasons",
    "explicar",
    "explicacion",
    "justificar",
    "determinantes",
}

_EXPLANATION_REQUIRED_PHRASES = {
    "factores determinantes",
    "explicacion por fila",
    "explicacion por registro",
    "explain per row",
    "explain each row",
    "row explanation",
    "per-row explanation",
    "per record explanation",
}

_VISUAL_ENABLED_TOKENS = {
    "segment",
    "segmentation",
    "outlier",
    "outliers",
    "elasticidad",
    "elasticity",
    "incertidumbre",
    "uncertainty",
    "fairness",
    "bias",
    "explicabilidad",
    "explain",
    "calibration",
    "calibración",
    "geography",
    "geographical",
    "geo",
    "series",
    "temporal",
    "timeseries",
    "error",
    "analysis",
    "explainability",
}

_VISUAL_REQUIRED_PHRASES = {
    "tablas y gráficos",
    "tablas y graficos",
    "tablas graficos",
    "visual comparison",
    "comparación visual",
    "comparacion visual",
    "comparativa visual",
    "segmentación por zonas",
    "segmentacion por zonas",
    "segmentación por horas",
    "segmentacion por horas",
    "detección de outliers",
    "deteccion de outliers",
    "explicar drivers",
    "explain drivers",
}


def _normalize_text(*values: Any) -> str:
    tokens: List[str] = []
    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, str):
            text = raw
        else:
            text = json.dumps(raw, ensure_ascii=False)
        cleaned = re.sub(r"[^0-9a-zA-ZÁÉÍÓÚáéíóúüÜñÑ]+", " ", text.lower())
        tokens.extend(cleaned.split())
    return " ".join(token for token in tokens if token)


def _extract_required_paths(artifact_requirements: Dict[str, Any]) -> List[str]:
    if not isinstance(artifact_requirements, dict):
        return []
    required_files = artifact_requirements.get("required_files")
    if not isinstance(required_files, list):
        return []
    paths: List[str] = []
    for entry in required_files:
        if not entry:
            continue
        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("output") or entry.get("artifact")
        else:
            path = entry
        if path and is_probably_path(str(path)):
            paths.append(str(path))
    return paths


def _merge_scored_rows_schema(
    base_schema: Dict[str, Any] | None,
    incoming_schema: Dict[str, Any] | None,
) -> Dict[str, Any]:
    base = base_schema if isinstance(base_schema, dict) else {}
    incoming = incoming_schema if isinstance(incoming_schema, dict) else {}
    if not base and not incoming:
        return {}
    if not base:
        return dict(incoming)
    if not incoming:
        return dict(base)

    def _merge_list(key: str) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for item in (base.get(key, []) or []) + (incoming.get(key, []) or []):
            if item is None:
                continue
            val = str(item)
            if not val or val in seen:
                continue
            seen.add(val)
            merged.append(val)
        return merged

    def _merge_groups(key: str) -> List[List[str]]:
        merged: List[List[str]] = []
        seen: set[tuple[str, ...]] = set()
        for group in (base.get(key, []) or []) + (incoming.get(key, []) or []):
            if not isinstance(group, list) or not group:
                continue
            normalized = tuple(sorted({str(item).lower() for item in group if item}))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append([str(item) for item in group if item])
        return merged

    merged = dict(base)
    merged["required_columns"] = _merge_list("required_columns")
    merged["recommended_columns"] = _merge_list("recommended_columns")
    merged["required_any_of_groups"] = _merge_groups("required_any_of_groups")
    if base.get("required_any_of_group_severity"):
        merged["required_any_of_group_severity"] = list(base.get("required_any_of_group_severity") or [])
    elif incoming.get("required_any_of_group_severity"):
        merged["required_any_of_group_severity"] = list(incoming.get("required_any_of_group_severity") or [])
    return merged


def _merge_unique_values(values: List[str], extras: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in values + extras:
        if not item:
            continue
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_column_token(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(name or "").lower())


def _extract_decisioning_required_column_names(decisioning: Dict[str, Any] | None) -> List[str]:
    if not isinstance(decisioning, dict):
        return []
    if decisioning.get("required") is not True:
        return []
    output = decisioning.get("output")
    if not isinstance(output, dict):
        return []
    required_columns = output.get("required_columns")
    if not isinstance(required_columns, list):
        return []
    names: List[str] = []
    for entry in required_columns:
        if not entry:
            continue
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("column")
        else:
            name = entry
        if name:
            names.append(str(name))
    return names


def _is_prediction_like_column(name: str) -> bool:
    token = _normalize_column_token(name)
    if not token:
        return False
    return any(
        key in token
        for key in (
            "pred",
            "prob",
            "score",
            "risk",
            "likelihood",
            "chance",
        )
    )


def _align_decisioning_requirements_with_schema(
    decisioning: Dict[str, Any] | None,
    scored_rows_schema: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(decisioning, dict):
        return {}
    if not isinstance(scored_rows_schema, dict):
        return decisioning
    required_cols = scored_rows_schema.get("required_columns")
    if not isinstance(required_cols, list) or not required_cols:
        return decisioning

    explanation_name = None
    for col in required_cols:
        if _normalize_column_token(col) == "explanation":
            explanation_name = str(col)
            break
    if explanation_name is None:
        for col in required_cols:
            if "driver" in _normalize_column_token(col):
                explanation_name = str(col)
                break
    if explanation_name is None:
        return decisioning

    output = decisioning.get("output")
    if not isinstance(output, dict):
        return decisioning
    required = output.get("required_columns")
    if not isinstance(required, list) or not required:
        return decisioning

    updated: List[Any] = []
    touched = False
    for entry in required:
        if isinstance(entry, dict):
            name = entry.get("name")
            role = str(entry.get("role") or "").lower()
            if role == "explanation" or _normalize_column_token(name) in {"explanation", "topdrivers", "topdriver"}:
                updated_entry = dict(entry)
                updated_entry["name"] = explanation_name
                updated.append(updated_entry)
                touched = True
            else:
                updated.append(entry)
        else:
            name = str(entry)
            if _normalize_column_token(name) in {"explanation", "topdrivers", "topdriver"}:
                updated.append(explanation_name)
                touched = True
            else:
                updated.append(entry)
    if not touched:
        return decisioning
    aligned = dict(decisioning)
    new_output = dict(output)
    new_output["required_columns"] = updated
    aligned["output"] = new_output
    return aligned


def _sync_execution_contract_outputs(contract: Dict[str, Any], contract_min: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict) or not isinstance(contract_min, dict):
        return contract

    required_outputs = contract.get("required_outputs")
    required_outputs_list = [str(item) for item in required_outputs if item] if isinstance(required_outputs, list) else []
    has_conceptual = any(item and not is_probably_path(item) for item in required_outputs_list)

    min_required_outputs = contract_min.get("required_outputs")
    min_required_outputs_list = (
        [str(item) for item in min_required_outputs if item and is_probably_path(str(item))]
        if isinstance(min_required_outputs, list)
        else []
    )

    if has_conceptual and min_required_outputs_list:
        contract["required_outputs"] = min_required_outputs_list
    elif not required_outputs_list and min_required_outputs_list:
        contract["required_outputs"] = min_required_outputs_list

    contract_artifacts = contract.get("artifact_requirements")
    if not isinstance(contract_artifacts, dict):
        contract_artifacts = {}
    min_artifacts = contract_min.get("artifact_requirements")
    if not isinstance(min_artifacts, dict):
        min_artifacts = {}

    min_required_files = _extract_required_paths(min_artifacts)
    contract_required_files = _extract_required_paths(contract_artifacts)
    if min_required_files:
        merged_files: List[Dict[str, Any]] = []
        seen = {path.lower() for path in contract_required_files if path}
        for path in contract_required_files:
            merged_files.append({"path": path, "description": ""})
        for path in min_required_files:
            key = path.lower()
            if key in seen:
                continue
            seen.add(key)
            merged_files.append({"path": path, "description": ""})
        if merged_files:
            contract_artifacts["required_files"] = merged_files
            contract["artifact_requirements"] = contract_artifacts

    contract_scored_schema = contract_artifacts.get("scored_rows_schema")
    min_scored_schema = min_artifacts.get("scored_rows_schema")
    merged_scored_schema = _merge_scored_rows_schema(contract_scored_schema, min_scored_schema)
    if merged_scored_schema:
        contract_artifacts["scored_rows_schema"] = merged_scored_schema
        contract["artifact_requirements"] = contract_artifacts

    merged_outputs: List[str] = []
    seen_outputs: set[str] = set()
    for path in (contract.get("required_outputs") or []) + min_required_files:
        if not path:
            continue
        if not is_probably_path(str(path)):
            continue
        normalized = str(path)
        if normalized in seen_outputs:
            continue
        seen_outputs.add(normalized)
        merged_outputs.append(normalized)
    if merged_outputs:
        contract["required_outputs"] = merged_outputs

    return contract


def _compress_text_preserve_ends(
    text: str,
    max_chars: int = 1800,
    head: int = 900,
    tail: int = 900,
) -> str:
    if not isinstance(text, str):
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    head_len = max(0, min(head, max_chars))
    tail_len = max(0, min(tail, max_chars - head_len))
    if head_len + tail_len == 0:
        return text[:max_chars]
    if head_len + tail_len < max_chars:
        head_len = max_chars - tail_len
    return text[:head_len] + "\n...\n" + text[-tail_len:]


def _matches_any_phrase(text: str, phrases: set[str]) -> bool:
    normalized = text.lower()
    return any(phrase in normalized for phrase in phrases)


def _contains_decisioning_token(text: str, tokens: set[str]) -> bool:
    if not text:
        return False
    words = set(text.split())
    return any(tok in words for tok in tokens)


def _build_decision_column_entry(
    name: str,
    role: str,
    type_name: str,
    inputs: List[str],
    logic_hint: str,
    allowed_values: List[str] | None = None,
    constraints: Dict[str, Any] | None = None,
    derivation_source: str = "postprocess",
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "name": name,
        "role": role,
        "type": type_name,
        "derivation": {
            "source": derivation_source,
            "inputs": inputs[:],
            "logic_hint": logic_hint,
        },
    }
    if allowed_values:
        entry["allowed_values"] = allowed_values
    if constraints:
        entry["constraints"] = constraints
    return entry


def _normalize_qa_gate_spec(item: Any) -> Dict[str, Any] | None:
    if isinstance(item, dict):
        name = item.get("name") or item.get("id") or item.get("gate")
        if not name:
            return None
        severity = item.get("severity")
        required = item.get("required")
        if severity is None and required is not None:
            severity = "HARD" if bool(required) else "SOFT"
        severity = str(severity).upper() if severity else "HARD"
        if severity not in _QA_SEVERITIES:
            severity = "HARD"
        params = item.get("params")
        if not isinstance(params, dict):
            params = {}
        return {"name": str(name), "severity": severity, "params": params}
    if isinstance(item, str):
        name = item.strip()
        if not name:
            return None
        return {"name": name, "severity": "HARD", "params": {}}
    return None


def _normalize_qa_gates(raw_gates: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_gates, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_gates:
        spec = _normalize_qa_gate_spec(item)
        if not spec:
            continue
        key = spec["name"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(spec)
    return normalized


def _normalize_cleaning_gate_spec(item: Any) -> Dict[str, Any] | None:
    if isinstance(item, dict):
        name = item.get("name") or item.get("id") or item.get("gate")
        if not name:
            return None
        severity = item.get("severity")
        required = item.get("required")
        if severity is None and required is not None:
            severity = "HARD" if bool(required) else "SOFT"
        severity = str(severity).upper() if severity else "HARD"
        if severity not in _CLEANING_SEVERITIES:
            severity = "HARD"
        params = item.get("params")
        if not isinstance(params, dict):
            params = {}
        return {"name": str(name), "severity": severity, "params": params}
    if isinstance(item, str):
        name = item.strip()
        if not name:
            return None
        return {"name": name, "severity": "HARD", "params": {}}
    return None


def _normalize_cleaning_gates(raw_gates: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_gates, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_gates:
        spec = _normalize_cleaning_gate_spec(item)
        if not spec:
            continue
        key = spec["name"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(spec)
    return normalized


def _build_default_cleaning_gates() -> List[Dict[str, Any]]:
    return [
        {"name": "required_columns_present", "severity": "HARD", "params": {}},
        {
            "name": "id_integrity",
            "severity": "HARD",
            "params": {
                "identifier_name_regex": r"(?i)(^id$|id$|entity|cod|code|key|partida|invoice|account)",
                "detect_scientific_notation": True,
            },
        },
        {
            "name": "no_semantic_rescale",
            "severity": "HARD",
            "params": {
                "allow_percent_like_only": True,
                "percent_like_name_regex": r"(?i)%|pct|percent|plazo",
            },
        },
        {"name": "no_synthetic_data", "severity": "HARD", "params": {}},
        {
            "name": "row_count_sanity",
            "severity": "SOFT",
            "params": {"max_drop_pct": 5.0, "max_dup_increase_pct": 1.0},
        },
    ]


def _apply_cleaning_gate_policy(raw_gates: Any) -> List[Dict[str, Any]]:
    gates = _normalize_cleaning_gates(raw_gates)
    default_gates = _build_default_cleaning_gates()
    if not gates:
        return default_gates
    existing = {_normalize_gate_name(gate.get("name")) for gate in gates if isinstance(gate, dict)}
    merged = list(gates)
    for gate in default_gates:
        if not isinstance(gate, dict):
            continue
        name = _normalize_gate_name(gate.get("name"))
        if name and name not in existing:
            merged.append(gate)
            existing.add(name)
    return merged


_DEFAULT_MISSING_CATEGORY_VALUE = "__MISSING__"
_UNSAFE_MISSING_CATEGORY_VALUES = {
    "",
    "none",
    "null",
    "nan",
    "na",
    "n/a",
    "nil",
}


def _normalize_missing_category_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.strip().lower() in _UNSAFE_MISSING_CATEGORY_VALUES:
        return None
    return cleaned


def _normalize_gate_name(name: Any) -> str:
    if not name:
        return ""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _ensure_missing_category_values(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure preprocessing_requirements.nan_strategies missing_category entries
    have a safe missing_category_value and propagate it into null_handling_gate
    params for reviewer visibility.
    """
    if not isinstance(contract, dict):
        return contract
    prep = contract.get("preprocessing_requirements")
    if not isinstance(prep, dict):
        return contract
    nan_strategies = prep.get("nan_strategies")
    if not isinstance(nan_strategies, list):
        return contract

    missing_category_map: Dict[str, str] = {}
    updated = False
    for strategy in nan_strategies:
        if not isinstance(strategy, dict):
            continue
        strat_name = str(strategy.get("strategy") or "").strip().lower()
        if strat_name != "missing_category":
            continue
        col = strategy.get("column") or strategy.get("name")
        if not col:
            continue
        col_name = str(col)
        value = _normalize_missing_category_value(strategy.get("missing_category_value"))
        if not value:
            value = _DEFAULT_MISSING_CATEGORY_VALUE
            strategy["missing_category_value"] = value
            updated = True
        missing_category_map[col_name] = value

    if missing_category_map:
        gates = contract.get("cleaning_gates")
        if isinstance(gates, list):
            for gate in gates:
                if not isinstance(gate, dict):
                    continue
                gate_name = gate.get("name") or gate.get("id") or gate.get("gate")
                if _normalize_gate_name(gate_name) != "null_handling_gate":
                    continue
                params = gate.get("params")
                if not isinstance(params, dict):
                    params = {}
                missing_values = params.get("missing_category_values")
                if not isinstance(missing_values, dict):
                    missing_values = {}
                for col, value in missing_category_map.items():
                    if col not in missing_values:
                        missing_values[col] = value
                        updated = True
                params["missing_category_values"] = missing_values
                gate["params"] = params

    if updated:
        notes = contract.get("notes_for_engineers")
        if not isinstance(notes, list):
            notes = []
        msg = "Injected missing_category_value for missing_category strategies to avoid NA sentinel collisions."
        if msg not in notes:
            notes.append(msg)
        contract["notes_for_engineers"] = notes
    return contract


def _apply_sparse_optional_columns(
    contract: Dict[str, Any],
    data_profile: Dict[str, Any] | None,
    threshold: float = 0.98,
) -> Dict[str, Any]:
    """
    Mark ultra-sparse columns as optional passthrough based on data_profile missingness.
    This prevents required-column guards from blocking on legitimately sparse features.
    """
    if not isinstance(contract, dict) or not isinstance(data_profile, dict):
        return contract
    missingness = data_profile.get("missingness_top30")
    if not isinstance(missingness, dict) or not missingness:
        return contract

    try:
        from src.utils.contract_v41 import get_outcome_columns, get_decision_columns, get_column_roles
    except Exception:
        return contract

    outcomes = {str(c) for c in get_outcome_columns(contract)}
    decisions = {str(c) for c in get_decision_columns(contract)}
    roles = get_column_roles(contract)
    identifiers = set()
    if isinstance(roles, dict):
        raw_ids = roles.get("identifiers") or roles.get("identifier") or []
        if isinstance(raw_ids, list):
            identifiers = {str(c) for c in raw_ids if c}
        elif isinstance(raw_ids, str):
            identifiers = {raw_ids}

    canonical = contract.get("canonical_columns")
    canonical_set = {str(c) for c in canonical} if isinstance(canonical, list) else set()
    available = contract.get("available_columns")
    available_set = {str(c) for c in available} if isinstance(available, list) else set()

    allowed_set = available_set or canonical_set

    def _norm_name(name: Any) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

    optional_candidates: List[str] = []
    for col, frac in missingness.items():
        try:
            frac_val = float(frac)
        except Exception:
            continue
        if frac_val < threshold:
            continue
        if allowed_set and str(col) not in allowed_set:
            continue
        if str(col) in outcomes or str(col) in decisions or str(col) in identifiers:
            continue
        optional_candidates.append(str(col))

    if not optional_candidates:
        return contract

    artifact_reqs = contract.get("artifact_requirements")
    if not isinstance(artifact_reqs, dict):
        artifact_reqs = {}
    schema_binding = artifact_reqs.get("schema_binding")
    if not isinstance(schema_binding, dict):
        schema_binding = {}
    optional_cols = schema_binding.get("optional_passthrough_columns")
    if not isinstance(optional_cols, list):
        optional_cols = []

    existing = {_norm_name(c): c for c in optional_cols if c}
    for col in optional_candidates:
        norm = _norm_name(col)
        if norm in existing:
            continue
        optional_cols.append(col)
        existing[norm] = col

    schema_binding["optional_passthrough_columns"] = optional_cols
    artifact_reqs["schema_binding"] = schema_binding
    contract["artifact_requirements"] = artifact_reqs

    notes = contract.get("notes_for_engineers")
    if not isinstance(notes, list):
        notes = []
    notes.append({
        "item": "Sparse columns treated as optional passthrough based on data_profile missingness.",
        "columns": optional_candidates,
        "null_frac_threshold": threshold,
    })
    contract["notes_for_engineers"] = notes
    return contract


def _strategy_mentions_resampling(strategy: Dict[str, Any], business_objective: str) -> bool:
    if not isinstance(strategy, dict):
        strategy = {}
    parts: List[str] = []
    for key in ("title", "description", "approach", "plan", "notes", "analysis_type", "problem_type"):
        value = strategy.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif isinstance(value, list):
            parts.extend([str(v) for v in value if isinstance(v, str)])
    if isinstance(business_objective, str) and business_objective.strip():
        parts.append(business_objective)
    haystack = " ".join(parts).lower()
    return any(token in haystack for token in _RESAMPLING_TOKENS)


def _infer_requires_target(strategy: Dict[str, Any], contract: Dict[str, Any]) -> bool:
    if not isinstance(strategy, dict):
        strategy = {}
    if not isinstance(contract, dict):
        contract = {}
    for source in (strategy, contract):
        for key in ("target_column", "target_columns", "outcome_columns", "decision_variable", "decision_variables"):
            val = source.get(key)
            if isinstance(val, list) and val:
                return True
            if isinstance(val, str) and val.strip() and val.lower() != "unknown":
                return True
    obj_analysis = contract.get("objective_analysis")
    if isinstance(obj_analysis, dict):
        problem_type = str(obj_analysis.get("problem_type") or "").lower()
        if problem_type:
            if any(token in problem_type for token in ("predict", "prescript", "regress", "classif", "forecast", "rank")):
                return True
    analysis_type = str(strategy.get("analysis_type") or strategy.get("problem_type") or "").lower()
    if analysis_type:
        if any(token in analysis_type for token in ("predict", "prescript", "regress", "classif", "forecast", "rank")):
            return True
    return False


def _allow_resampling_random(requires_target: bool, contract: Dict[str, Any]) -> bool:
    if requires_target:
        return True
    if not isinstance(contract, dict):
        return False
    validation = contract.get("validation_requirements", {})
    if not isinstance(validation, dict):
        return False
    method = str(validation.get("method") or "").strip().lower()
    return method in {"cross_validation", "bootstrap"}


def _build_default_qa_gates(
    strategy: Dict[str, Any],
    business_objective: str,
    contract: Dict[str, Any],
) -> List[Dict[str, Any]]:
    requires_target = _infer_requires_target(strategy, contract)
    allow_resampling = _allow_resampling_random(requires_target, contract)
    gates: List[Dict[str, Any]] = [
        {"name": "security_sandbox", "severity": "HARD", "params": {}},
        {"name": "must_read_input_csv", "severity": "HARD", "params": {}},
        {"name": "must_reference_contract_columns", "severity": "HARD", "params": {}},
        {"name": "no_synthetic_data", "severity": "HARD", "params": {"allow_resampling_random": allow_resampling}},
        {"name": "dialect_mismatch_handling", "severity": "SOFT", "params": {}},
        {"name": "group_split_required", "severity": "SOFT", "params": {}},
    ]
    if requires_target:
        gates.extend(
            [
                {"name": "target_variance_guard", "severity": "HARD", "params": {}},
                {"name": "leakage_prevention", "severity": "HARD", "params": {}},
                {"name": "train_eval_split", "severity": "SOFT", "params": {}},
            ]
        )
    return gates


def _apply_qa_gate_policy(
    raw_gates: Any,
    strategy: Dict[str, Any],
    business_objective: str,
    contract: Dict[str, Any],
) -> List[Dict[str, Any]]:
    gates = _normalize_qa_gates(raw_gates)
    if not gates:
        gates = _build_default_qa_gates(strategy, business_objective, contract)
    requires_target = _infer_requires_target(strategy, contract)
    allow_resampling = _allow_resampling_random(requires_target, contract)
    for gate in gates:
        if gate.get("name") == "no_synthetic_data":
            params = gate.get("params")
            if not isinstance(params, dict):
                params = {}
            params.setdefault("allow_resampling_random", allow_resampling)
            gate["params"] = params
    return gates


_KPI_ALIASES = {
    "accuracy": ["accuracy", "acc", "balanced accuracy", "balanced_accuracy"],
    "auc": ["auc", "auroc", "roc auc", "roc_auc"],
    "f1": ["f1", "f1-score", "f1_score"],
    "precision": ["precision", "prec"],
    "recall": ["recall", "sensitivity", "tpr"],
    "rmse": ["rmse", "root mean squared error"],
    "mae": ["mae", "mean absolute error"],
    "mse": ["mse", "mean squared error"],
    "r2": ["r2", "r^2", "r-squared", "rsquared", "r_squared"],
    "logloss": ["logloss", "log loss", "log_loss", "cross entropy", "cross_entropy"],
    "mape": ["mape", "mean absolute percentage error"],
    "pr_auc": ["pr_auc", "pr auc", "average precision", "average_precision"],
}


def _normalize_kpi_metric(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"[^a-z0-9]+", " ", raw).strip()
    for canonical, aliases in _KPI_ALIASES.items():
        for alias in aliases:
            alias_norm = re.sub(r"[^a-z0-9]+", " ", alias).strip()
            if raw == alias_norm:
                return canonical
    return ""


def _extract_kpi_from_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    for item in values:
        if isinstance(item, dict):
            metric = item.get("metric") or item.get("name") or item.get("id")
            normalized = _normalize_kpi_metric(metric)
        else:
            normalized = _normalize_kpi_metric(item)
        if normalized:
            return normalized
    return ""


def _extract_kpi_from_text(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    for canonical, aliases in _KPI_ALIASES.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, lower):
                return canonical
    return ""


def _ensure_benchmark_kpi_gate(
    contract: Dict[str, Any],
    strategy: Dict[str, Any],
    business_objective: str,
) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    strategy = strategy if isinstance(strategy, dict) else {}

    kpi = _normalize_kpi_metric(strategy.get("success_metric"))
    if not kpi:
        kpi = _extract_kpi_from_list(strategy.get("recommended_evaluation_metrics"))
    if not kpi:
        kpi = _extract_kpi_from_text(business_objective or "")
    if not kpi:
        return contract

    qa_gates = contract.get("qa_gates")
    if not isinstance(qa_gates, list):
        qa_gates = []

    gate_exists = False
    for gate in qa_gates:
        if not isinstance(gate, dict):
            continue
        if gate.get("name") == "benchmark_kpi_report":
            params = gate.get("params")
            if not isinstance(params, dict):
                params = {}
            params.setdefault("metric", kpi)
            params.setdefault("validation", "cross_validation_or_holdout")
            gate["type"] = gate.get("type") or "metric_report"
            gate["params"] = params
            gate.setdefault("severity", "warning")
            gate_exists = True
            break
        if gate.get("type") == "metric_report":
            params = gate.get("params")
            if isinstance(params, dict) and params.get("metric") == kpi:
                gate_exists = True
                break

    if not gate_exists:
        qa_gates.append(
            {
                "name": "benchmark_kpi_report",
                "type": "metric_report",
                "params": {"metric": kpi, "validation": "cross_validation_or_holdout"},
                "severity": "warning",
            }
        )

    contract["qa_gates"] = qa_gates

    validation = contract.get("validation_requirements")
    if not isinstance(validation, dict):
        validation = {}
    validation.setdefault("primary_metric", kpi)
    metrics_list = validation.get("metrics_to_report")
    if not isinstance(metrics_list, list):
        metrics_list = []
    if kpi not in metrics_list:
        metrics_list.append(kpi)
    validation["metrics_to_report"] = metrics_list
    contract["validation_requirements"] = validation
    return contract


def _create_v41_skeleton(
    strategy: Dict[str, Any],
    business_objective: str,
    column_inventory: List[str] | None = None,
    output_dialect: Dict[str, str] | None = None,
    reason: str = "LLM failure",
    data_summary: str = ""
) -> Dict[str, Any]:
    """
    Returns a complete, safe V4.1 schema skeleton with all required fields.
    Used as fallback when LLM fails or for validation.
    Now respects basic strategy inputs (target, problem_type) to support testing.
    """
    strategy_title = strategy.get("title", "Unknown") if isinstance(strategy, dict) else "Unknown"
    required_cols = strategy.get("required_columns", []) if isinstance(strategy, dict) else []
    available_cols = column_inventory or []

    # Canonical columns should represent full available_columns (no truncation).
    canonical_cols = [str(c) for c in available_cols if c]
    missing_cols = []
    fuzzy_matches = {}

    inventory_map = {re.sub(r"[^0-9a-zA-Z]+", "", str(c).lower()): c for c in available_cols}

    def _find_in_inventory(name: str) -> str | None:
        return inventory_map.get(re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower()))

    if required_cols and available_cols:
        for req in required_cols:
            found = _find_in_inventory(req)
            if found:
                if found not in canonical_cols:
                    canonical_cols.append(found)
            else:
                missing_cols.append(str(req))
                # Use difflib to find close matches
                close_matches = difflib.get_close_matches(
                    str(req).lower(),
                    [str(c).lower() for c in available_cols],
                    n=3,
                    cutoff=0.6,
                )
                if close_matches:
                    matched_originals = [
                        c for c in available_cols
                        if str(c).lower() in close_matches
                    ]
                    fuzzy_matches[str(req)] = matched_originals[:3]

    if not canonical_cols and required_cols:
        canonical_cols = [str(col) for col in required_cols if col]

    # Smart Fallback: Infer roles from strategy if present
    target_col = strategy.get("target_column") if isinstance(strategy, dict) else None
    col_roles_outcome = []
    derived_cols_list = []
    
    if target_col:
        real_target = _find_in_inventory(target_col)
        if real_target:
            col_roles_outcome.append(real_target)
        else:
            # Assume derived target
            col_roles_outcome.append(target_col)
            derived_cols_list.append(target_col)

    # Everything else in unknown for now (inventory metadata only - don't force requirements on them)
    available_set = set(available_cols)
    outcome_set = set(col_roles_outcome)
    unknown_cols = [c for c in available_cols if c not in outcome_set]  # canonical might be in unknown
    unknown_summary = {
        "count": len(unknown_cols),
        "sample": unknown_cols[:25],
    }

    # Parse types from summary for tests/fallback utility
    type_distribution = {}
    if data_summary:
        for line in data_summary.splitlines():
            clean_line = line.strip().strip("- ")
            if ":" not in clean_line: continue
            lbl, cols_txt = clean_line.split(":", 1)
            lbl = lbl.lower()
            kind = "unknown"
            if "date" in lbl: kind = "datetime"
            elif "num" in lbl: kind = "numeric"
            elif "cat" in lbl or "bool" in lbl: kind = "categorical"
            elif "text" in lbl: kind = "text"
            
            if kind != "unknown":
                if kind not in type_distribution: type_distribution[kind] = []
                for c_raw in re.split(r"[;,]", cols_txt):
                    if not c_raw.strip(): continue
                    real = _find_in_inventory(c_raw.strip()) or c_raw.strip()
                    type_distribution[kind].append(real)

    # Alias for artifact requirements
    outcome_cols = col_roles_outcome
    
    # Infer identifiers for artifact requirements fallback
    identifiers = []
    for col in available_cols:
        # Simple heuristic: exact 'id', or ends in '_id'/'Id' etc
        if re.search(r"(?i)\b(id|uuid|key)\b", col):
            identifiers.append(col)

    n_rows = 0

    problem_type = strategy.get("problem_type", "unknown") if isinstance(strategy, dict) else "unknown"
    feature_selectors = []
    if len(available_cols) > 200:
        feature_selectors, _remaining = infer_feature_selectors(
            available_cols, max_list_size=200, min_group_size=50
        )

    return {
        "contract_version": CONTRACT_VERSION_V41,
        "strategy_title": strategy_title,
        "business_objective": business_objective or "",
        
        "missing_columns_handling": {
            "missing_from_inventory": missing_cols,
            "attempted_fuzzy_matches": fuzzy_matches,
            "resolution": "unknown" if not missing_cols else "require_verification",
            "impact": "none" if not missing_cols else f"{len(missing_cols)} required columns not found",
            "contract_updates": {
                "canonical_columns_update": "Fuzzy matches suggested" if fuzzy_matches else "",
                "artifact_schema_update": "",
                "derived_plan_update": "",
                "gates_update": ""
            }
        },
        
        "execution_constraints": {
            "inplace_column_creation_policy": "unknown_or_forbidden",
            "preferred_patterns": ["df = df.assign(...)", "derived_arrays_then_concat", "build_new_df_from_columns"],
            "rationale": "Fallback: prefer safe patterns when uncertain."
        },
        
        "objective_analysis": {
            "problem_type": problem_type,
            "decision_variable": None,
            "business_decision": "unknown",
            "success_criteria": "unknown",
            "complexity": "unknown"
        },
        
        "data_analysis": {
            "dataset_size": n_rows,
            "features_with_nulls": [],
            "type_distribution": type_distribution,
            "risk_features": [],
            "data_sufficiency": "unknown"
        },
        
        "column_roles": {
            "pre_decision": [],
            "decision": [],
            "outcome": col_roles_outcome,
            "post_decision_audit_only": [],
            "unknown": []
        },
        "column_roles_unknown_summary": unknown_summary,
        
        "preprocessing_requirements": {},
        
        "feature_engineering_plan": {
            "derived_columns": derived_cols_list
        },
        
        "validation_requirements": {
            "method": "cross_validation",
            "stratification": False,
            "min_samples_required": 100
        },
        
        "leakage_execution_plan": {
            "audit_features": [],
            "method": "correlation_with_target",
            "threshold": 0.9,
            "action_if_exceeds": "exclude_from_features"
        },
        
        "optimization_specification": None,
        "segmentation_constraints": None,
        
        "data_limited_mode": {
            "is_active": False,
            "activation_reasons": [],
            "fallback_methodology": "unknown",
            "minimum_outputs": ["data/metrics.json", "data/scored_rows.csv", "data/alignment_check.json"],
            "artifact_reductions_allowed": True
        },
        
        "allowed_feature_sets": {
            "segmentation_features": [],
            "model_features": [],
            "audit_only_features": [],
            "forbidden_for_modeling": [],
            "rationale": "Fallback: empty feature sets pending analysis."
        },
        
        "artifact_requirements": {
            "required_files": ["data/cleaned_data.csv", "data/metrics.json", "data/scored_rows.csv"],
            "file_schemas": {},
            "scored_rows_schema": {
                "required_columns": identifiers if identifiers else [],  # Dynamic IDs, no hardcoded "id"
                "recommended_columns": ["prediction"] + outcome_cols
            }
        },
        
        "qa_gates": _apply_qa_gate_policy([], strategy, business_objective or "", {}),
        "cleaning_gates": _apply_cleaning_gate_policy([]),
        
        "reviewer_gates": [
            {"id": "methodology_alignment", "required": True, "description": "Methodology aligns with objective"},
            {"id": "business_value", "required": True, "description": "Business value demonstrated"}
        ],
        
        "data_engineer_runbook": DEFAULT_DATA_ENGINEER_RUNBOOK,
        "ml_engineer_runbook": DEFAULT_ML_ENGINEER_RUNBOOK,
        
        "available_columns": available_cols,
          "canonical_columns": canonical_cols,
          "derived_columns": derived_cols_list,
          "feature_selectors": feature_selectors,
          "canonical_columns_compact": compact_column_representation(available_cols, max_display=40)
          if feature_selectors
          else {},
        "required_outputs": ["data/cleaned_data.csv", "data/metrics.json", "data/alignment_check.json"],
        
        "iteration_policy": {
            "max_iterations": 3,
            "early_stop_on_success": True
        },
        
        "unknowns": [
            {
                "item": reason,
                "impact": "Using skeletal V4.1 fallback",
                "mitigation": "Manual review required",
                "requires_verification": True
            }
        ],
        
        "assumptions": [
            "Minimal safe defaults used due to planner unavailability"
        ],
        
        "notes_for_engineers": [
            "This is a fallback contract. Proceed conservatively.",
            "Verify column inventory and semantics manually if possible."
        ]
    }


def _tokenize_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", " ", str(value).lower()).strip()


def _coerce_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        return [value]
    return []


def select_relevant_columns(
    strategy: Dict[str, Any] | None,
    business_objective: str,
    domain_expert_critique: str,
    column_inventory: List[str] | None,
    data_profile_summary: str | None = None,
) -> Dict[str, Any]:
    """
    Deterministically select a compact set of relevant columns for planning.
    """
    inventory = [str(col) for col in (column_inventory or []) if col is not None]
    inventory_set = set(inventory)
    inventory_norm_map: Dict[str, str] = {}
    for col in inventory:
        norm = _normalize_column_identifier(col)
        if norm and norm not in inventory_norm_map:
            inventory_norm_map[norm] = col

    def _resolve_inventory(name: str) -> Optional[str]:
        if not name:
            return None
        if name in inventory_set:
            return name
        norm = _normalize_column_identifier(name)
        return inventory_norm_map.get(norm)

    sources: Dict[str, List[str]] = {
        "strategy_required_columns": [],
        "strategy_decision_columns": [],
        "strategy_outcome_columns": [],
        "strategy_audit_only_columns": [],
        "text_mentions": [],
        "heuristic": [],
    }

    def _add_unique(target: List[str], col: str) -> None:
        if not col or col in target:
            return
        target.append(col)

    def _add_source(col: Optional[str], key: str, collector: List[str]) -> None:
        if not col:
            return
        _add_unique(collector, col)
        _add_unique(sources[key], col)

    strategy_dict = strategy if isinstance(strategy, dict) else {}
    required_cols_raw = _coerce_list(strategy_dict.get("required_columns"))
    decision_cols_raw = _coerce_list(
        strategy_dict.get("decision_columns")
        or strategy_dict.get("decision_variables")
        or strategy_dict.get("decision_column")
    )
    outcome_cols_raw = _coerce_list(
        strategy_dict.get("outcome_columns")
        or strategy_dict.get("target_column")
        or strategy_dict.get("target_columns")
        or strategy_dict.get("outcome_column")
    )
    audit_only_raw = _coerce_list(strategy_dict.get("audit_only_columns"))

    required_cols: List[str] = []
    decision_cols: List[str] = []
    outcome_cols: List[str] = []
    audit_only_cols: List[str] = []

    for col in required_cols_raw:
        _add_source(_resolve_inventory(col), "strategy_required_columns", required_cols)
    for col in decision_cols_raw:
        _add_source(_resolve_inventory(col), "strategy_decision_columns", decision_cols)
    for col in outcome_cols_raw:
        _add_source(_resolve_inventory(col), "strategy_outcome_columns", outcome_cols)
    for col in audit_only_raw:
        _add_source(_resolve_inventory(col), "strategy_audit_only_columns", audit_only_cols)

    text_matches: List[str] = []
    text_blob = "\n".join([business_objective or "", domain_expert_critique or ""])
    if text_blob:
        quote_chars = "\"'`" + "\u2018\u2019\u201c\u201d"
        pattern = f"[{re.escape(quote_chars)}](.+?)[{re.escape(quote_chars)}]"
        for match in re.findall(pattern, text_blob):
            candidate = match.strip()
            if candidate in inventory_set:
                _add_source(candidate, "text_mentions", text_matches)

    heuristic_cols: List[str] = []
    if len(set(required_cols + decision_cols + outcome_cols + audit_only_cols + text_matches)) < 4:
        patterns = [
            (re.compile(r"\b(target|label|outcome|success|converted)\b"), "target_like"),
            (re.compile(r"\b(price|amount|offer|quote|cost)\b"), "decision_like"),
            (re.compile(r"\b(id|uuid|key)\b"), "id_like"),
            (re.compile(r"\b(date|time|timestamp)\b"), "time_like"),
        ]
        for col in inventory:
            tokenized = _tokenize_name(col)
            if not tokenized:
                continue
            for regex, _label in patterns:
                if regex.search(tokenized):
                    _add_source(col, "heuristic", heuristic_cols)
                    break

    ordered: List[str] = []
    for col in required_cols:
        _add_unique(ordered, col)
    for col in outcome_cols + decision_cols:
        _add_unique(ordered, col)
    for col in audit_only_cols + text_matches + heuristic_cols:
        _add_unique(ordered, col)

    # REMOVED: max_columns = 30 limit
    # High-dimensional datasets (e.g., MNIST with 784 pixels) must pass through.
    # The Steward's column_sets.json handles grouping for wide datasets.
    # Downstream agents (ML Engineer) use column_sets.expand_column_sets() at runtime.
    ordered = list(dict.fromkeys(ordered))

    # For high-dimensional datasets, note that column_sets.json provides grouped access
    is_high_dimensional = len(inventory) > 100
    omitted_policy = (
        "High-dimensional dataset: use column_sets.json for grouped feature access."
        if is_high_dimensional
        else "Ignored by default unless promoted by strategy/explicit mention; available via column_inventory."
    )

    return {
        "relevant_columns": [col for col in ordered if col in inventory_set],
        "relevant_sources": sources,
        "omitted_columns_policy": omitted_policy,
    }


def build_contract_min(
    full_contract_or_partial: Dict[str, Any] | None,
    strategy: Dict[str, Any] | None,
    column_inventory: List[str] | None,
    relevant_columns: List[str] | None,
    target_candidates: List[Dict[str, Any]] | None = None,
    data_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a compact contract_min that aligns agents on relevant columns and gates.

    Args:
        data_profile: Optional data profile containing constant_columns to exclude from
                     clean_dataset.required_columns. Columns marked as constant are
                     excluded from required_columns since they provide no information.
    """
    contract = full_contract_or_partial if isinstance(full_contract_or_partial, dict) else {}
    strategy_dict = strategy if isinstance(strategy, dict) else {}
    inventory = [str(col) for col in (column_inventory or []) if col is not None]
    inventory_norms = {_normalize_column_identifier(col): col for col in inventory}

    canonical_columns: List[str] = []
    contract_canonical = contract.get("canonical_columns")
    if isinstance(contract_canonical, list) and contract_canonical:
        canonical_columns = [str(col) for col in contract_canonical if col]
    else:
        for col in (relevant_columns or []):
            if not col:
                continue
            if col in inventory:
                canonical_columns.append(col)
                continue
            norm = _normalize_column_identifier(col)
            resolved = inventory_norms.get(norm)
            if resolved:
                canonical_columns.append(resolved)

        if not canonical_columns:
            for col in _coerce_list(strategy_dict.get("required_columns")):
                norm = _normalize_column_identifier(col)
                resolved = inventory_norms.get(norm)
                if resolved:
                    canonical_columns.append(resolved)
        canonical_columns = list(dict.fromkeys(canonical_columns))

    def _filter_to_canonical(cols: List[str]) -> List[str]:
        canon_set = set(canonical_columns)
        return [col for col in cols if col in canon_set]

    roles_raw = contract.get("column_roles", {}) if isinstance(contract.get("column_roles"), dict) else {}
    roles = roles_raw
    if isinstance(roles_raw, dict) and roles_raw:
        role_keys = {
            "pre_decision",
            "decision",
            "outcome",
            "post_decision_audit_only",
            "unknown",
            "identifiers",
            "time_columns",
        }
        if all(key in role_keys for key in roles_raw.keys()):
            if all(isinstance(val, str) for val in roles_raw.values()):
                roles = {}
    if roles and all(isinstance(val, dict) and "role" in val for val in roles.values()):
        role_lists = {
            "pre_decision": [],
            "decision": [],
            "outcome": [],
            "post_decision_audit_only": [],
            "unknown": [],
        }
        role_aliases = {
            "pre_decision": "pre_decision",
            "predecision": "pre_decision",
            "pre_decision_features": "pre_decision",
            "decision": "decision",
            "decisions": "decision",
            "outcome": "outcome",
            "target": "outcome",
            "label": "outcome",
            "post_decision_audit_only": "post_decision_audit_only",
            "post_decision": "post_decision_audit_only",
            "audit_only": "post_decision_audit_only",
            "audit": "post_decision_audit_only",
            "unknown": "unknown",
        }
        for col, meta in roles.items():
            role_raw = str(meta.get("role") or "").strip().lower()
            role_key = re.sub(r"[^a-z0-9]+", "_", role_raw).strip("_")
            role_bucket = role_aliases.get(role_key, "unknown")
            norm = _normalize_column_identifier(col)
            resolved = inventory_norms.get(norm) or col
            role_lists[role_bucket].append(resolved)
        roles = role_lists
    roles_present = any(
        _coerce_list(roles.get(key))
        for key in ("pre_decision", "decision", "outcome", "post_decision_audit_only")
    )
    role_pre = _filter_to_canonical(_coerce_list(roles.get("pre_decision"))) if roles else []
    role_decision = _filter_to_canonical(_coerce_list(roles.get("decision"))) if roles else []
    role_outcome = _filter_to_canonical(_coerce_list(roles.get("outcome"))) if roles else []
    role_audit = _filter_to_canonical(_coerce_list(roles.get("post_decision_audit_only"))) if roles else []

    outcome_cols: List[str] = []
    decision_cols: List[str] = []
    audit_only_cols: List[str] = []
    identifiers: List[str] = []
    time_columns: List[str] = []

    def _resolve_candidate_targets() -> List[str]:
        resolved_targets: List[str] = []
        if not target_candidates:
            return resolved_targets
        for item in target_candidates:
            if not isinstance(item, dict):
                continue
            raw = item.get("column") or item.get("name") or item.get("candidate")
            if not raw:
                continue
            if raw in canonical_columns:
                resolved_targets.append(raw)
                continue
            norm = _normalize_column_identifier(raw)
            resolved = inventory_norms.get(norm)
            if resolved:
                if resolved not in canonical_columns:
                    canonical_columns.append(resolved)
                resolved_targets.append(resolved)
        return list(dict.fromkeys(resolved_targets))

    if roles_present:
        outcome_cols = list(role_outcome)
        decision_cols = list(role_decision)
        audit_only_cols = list(role_audit)
        if not outcome_cols:
            outcome_cols = _filter_to_canonical(_resolve_candidate_targets())
    else:
        outcome_candidates = []
        decision_candidates = []
        audit_candidates = []
        outcome_candidates.extend(_coerce_list(strategy_dict.get("outcome_columns")))
        outcome_candidates.extend(_coerce_list(strategy_dict.get("target_column")))
        outcome_candidates.extend(_coerce_list(strategy_dict.get("target_columns")))
        outcome_candidates.extend(_coerce_list(contract.get("outcome_columns")))
        decision_candidates.extend(_coerce_list(strategy_dict.get("decision_columns")))
        decision_candidates.extend(_coerce_list(strategy_dict.get("decision_variables")))
        decision_candidates.extend(_coerce_list(contract.get("decision_columns")))
        decision_candidates.extend(_coerce_list(contract.get("decision_variables")))
        audit_candidates.extend(_coerce_list(strategy_dict.get("audit_only_columns")))
        if not outcome_candidates:
            outcome_candidates.extend(_resolve_candidate_targets())
        outcome_cols = _filter_to_canonical([col for col in outcome_candidates if col])
        decision_cols = _filter_to_canonical([col for col in decision_candidates if col])
        audit_only_cols = _filter_to_canonical([col for col in audit_candidates if col])

    # STEWARD-FIRST ROLE INFERENCE
    # Trust the Steward's dataset_semantics.json over regex heuristics.
    # Only fall back to regex if Steward didn't provide the information.
    steward_semantics = {}
    if isinstance(data_profile, dict):
        steward_semantics = data_profile.get("dataset_semantics", {})
        if not isinstance(steward_semantics, dict):
            steward_semantics = {}

    # Extract Steward-identified roles
    steward_identifiers = _coerce_list(steward_semantics.get("identifier_columns"))
    steward_time_cols = _coerce_list(steward_semantics.get("time_columns"))
    steward_categorical = _coerce_list(steward_semantics.get("categorical_columns"))

    # Also check data_profile top-level for backward compatibility
    if not steward_identifiers:
        steward_identifiers = _coerce_list(data_profile.get("identifier_columns")) if data_profile else []
    if not steward_time_cols:
        steward_time_cols = _coerce_list(data_profile.get("time_columns")) if data_profile else []

    # Use Steward's analysis if available
    if steward_identifiers:
        for col in steward_identifiers:
            if col in canonical_columns and col not in identifiers:
                identifiers.append(col)
        print(f"STEWARD_IDENTIFIERS: Using Steward-provided identifiers: {identifiers}")

    if steward_time_cols:
        for col in steward_time_cols:
            if col in canonical_columns and col not in time_columns:
                time_columns.append(col)
        print(f"STEWARD_TIME_COLUMNS: Using Steward-provided time columns: {time_columns}")

    # FALLBACK: Only use regex heuristics if Steward didn't provide role information
    if not steward_identifiers or not steward_time_cols:
        token_patterns = {
            "id_like": re.compile(r"\b(id|uuid|key)\b"),
            "time_like": re.compile(r"\b(date|time|timestamp)\b"),
        }
        for col in canonical_columns:
            tokenized = _tokenize_name(col)
            if not steward_identifiers and token_patterns["id_like"].search(tokenized):
                if col not in identifiers:
                    identifiers.append(col)
            if not steward_time_cols and token_patterns["time_like"].search(tokenized):
                if col not in time_columns:
                    time_columns.append(col)
        if not steward_identifiers and identifiers:
            print(f"REGEX_FALLBACK_IDENTIFIERS: {identifiers}")
        if not steward_time_cols and time_columns:
            print(f"REGEX_FALLBACK_TIME_COLUMNS: {time_columns}")

    outcome_cols = list(dict.fromkeys(outcome_cols))
    decision_cols = list(dict.fromkeys(decision_cols))
    audit_only_cols = list(dict.fromkeys(audit_only_cols))
    identifiers = list(dict.fromkeys(identifiers))
    time_columns = list(dict.fromkeys(time_columns))

    assigned = set(outcome_cols + decision_cols + audit_only_cols + identifiers + time_columns)
    if roles_present:
        pre_decision = list(role_pre)
    else:
        pre_decision = [col for col in canonical_columns if col not in assigned]

    column_roles: Dict[str, List[str]] = {
        "pre_decision": pre_decision,
        "decision": decision_cols,
        "outcome": outcome_cols,
        "post_decision_audit_only": audit_only_cols,
        "unknown": [],
    }
    if identifiers:
        column_roles["identifiers"] = identifiers
    if time_columns:
        column_roles["time_columns"] = time_columns

    segmentation_features = list(pre_decision)
    model_features = list(pre_decision)
    if decision_cols:
        for col in decision_cols:
            if col not in model_features:
                model_features.append(col)
    forbidden_features = list(dict.fromkeys(outcome_cols + audit_only_cols))

    allowed_sets_full = contract.get("allowed_feature_sets")
    if not isinstance(allowed_sets_full, dict):
        allowed_sets_full = {}
    missing_sets: List[str] = []

    def _full_list(key: str) -> tuple[list[str] | None, bool]:
        val = allowed_sets_full.get(key)
        if isinstance(val, list):
            return [str(c) for c in val if c is not None], True
        return None, False

    full_model, has_model = _full_list("model_features")
    full_seg, has_seg = _full_list("segmentation_features")
    full_forbidden, has_forbidden = _full_list("forbidden_for_modeling")
    if not has_forbidden:
        full_forbidden, has_forbidden = _full_list("forbidden_features")
    full_audit, has_audit = _full_list("audit_only_features")

    if has_model:
        model_features = full_model
    else:
        missing_sets.append("model_features")
    if has_seg:
        segmentation_features = full_seg
    else:
        missing_sets.append("segmentation_features")
    if has_forbidden:
        forbidden_features = full_forbidden
    else:
        missing_sets.append("forbidden_for_modeling")
    if not has_audit:
        missing_sets.append("audit_only_features")

    audit_only_features = full_audit if has_audit else list(audit_only_cols)
    if missing_sets:
        print(f"FALLBACK_FEATURE_SETS: {', '.join(sorted(set(missing_sets)))}")

    identifier_candidates = _resolve_identifier_candidates(contract, canonical_columns)
    if identifier_candidates:
        filtered_model = [col for col in model_features if col not in identifier_candidates]
        removed_ids = [col for col in model_features if col not in filtered_model]
        if removed_ids:
            model_features = filtered_model
            for col in removed_ids:
                if col not in audit_only_features:
                    audit_only_features.append(col)

    full_artifact_requirements = contract.get("artifact_requirements")
    if not isinstance(full_artifact_requirements, dict):
        full_artifact_requirements = {}

    default_outputs = [
        "data/cleaned_data.csv",
        "data/scored_rows.csv",
        "data/metrics.json",
        "data/alignment_check.json",
    ]
    full_required_files = _extract_required_paths(full_artifact_requirements)
    if full_required_files:
        required_outputs = list(dict.fromkeys(full_required_files + default_outputs))
    else:
        required_outputs = list(default_outputs)

    # P1.5: Infer feature selectors for wide datasets
    feature_selectors = []
    if len(canonical_columns) > 200:
        feature_selectors, remaining_cols = infer_feature_selectors(
            canonical_columns, max_list_size=200, min_group_size=50
        )
        if feature_selectors:
            print(f"FEATURE_SELECTORS: Inferred {len(feature_selectors)} selectors for {len(canonical_columns)} columns")

    # P1.1: Determine scored_rows required columns based on objective type
    objective_type = contract.get("objective_type") or strategy_dict.get("objective_type") or ""

    # P1.5: Infer identifier column from canonical_columns instead of hardcoding "id"
    # Pattern matches: id, ID, _id, Id, row_id, etc.
    id_pattern = re.compile(r"^id$|^ID$|_id$|Id$|^row_id$|^index$", re.IGNORECASE)
    id_column = None
    for col in canonical_columns:
        if id_pattern.search(col):
            id_column = col
            break

    # P1.6: Keep required_columns minimal (only id if detected)
    scored_rows_required_columns = [id_column] if id_column else []

    # P1.6: Build universal any-of groups (no dataset hardcodes)
    required_any_of_groups = []
    required_any_of_group_severity = []

    # Group 1 (identificador): incluir id detectado + sinónimos genéricos
    group1 = ["id", "row_id", "index", "record_id", "case_id"]
    if id_column and id_column not in group1:
        group1.insert(0, id_column)
    required_any_of_groups.append(group1)
    required_any_of_group_severity.append("warning")  # Identifier is optional (warning)

    # Group 2 (predicción/score): sinónimos universales
    required_any_of_groups.append([
        "prediction", "pred", "probability", "prob", "score",
        "risk_score", "predicted_prob", "predicted_value", "y_pred"
    ])
    required_any_of_group_severity.append("fail")  # Prediction/score is critical (fail)

    # Group 3 (ranking/prioridad) SOLO si objective_type sugiere ranking/triage/targeting
    obj_lower = str(objective_type).lower()
    if any(kw in obj_lower for kw in ["ranking", "triage", "targeting", "priorit", "segment"]):
        required_any_of_groups.append(["priority", "rank", "ranking", "triage_priority"])
        required_any_of_group_severity.append("fail")  # Ranking is critical when required

    scored_rows_schema = {
        "required_columns": scored_rows_required_columns,
        "required_any_of_groups": required_any_of_groups,
        "required_any_of_group_severity": required_any_of_group_severity,
        "recommended_columns": [],
    }
    scored_rows_schema = _merge_scored_rows_schema(
        scored_rows_schema,
        full_artifact_requirements.get("scored_rows_schema"),
    )
    decisioning_requirements = _align_decisioning_requirements_with_schema(
        contract.get("decisioning_requirements", {}),
        scored_rows_schema,
    )
    decisioning_required = _extract_decisioning_required_column_names(decisioning_requirements)
    if decisioning_required:
        scored_rows_schema["required_columns"] = _merge_unique_values(
            scored_rows_schema.get("required_columns", []) or [],
            decisioning_required,
        )
    required_cols_for_anyof = scored_rows_schema.get("required_columns")
    anyof_groups = scored_rows_schema.get("required_any_of_groups")
    if isinstance(required_cols_for_anyof, list) and isinstance(anyof_groups, list):
        prediction_group = None
        for group in anyof_groups:
            if not isinstance(group, list):
                continue
            if any(_is_prediction_like_column(item) for item in group):
                prediction_group = group
                break
        if prediction_group is not None:
            seen = {_normalize_column_token(item) for item in prediction_group if item}
            for col in required_cols_for_anyof:
                if not col or not _is_prediction_like_column(col):
                    continue
                norm = _normalize_column_token(col)
                if norm in seen:
                    continue
                prediction_group.append(col)
                seen.add(norm)

    required_files: List[Dict[str, Any]] = []
    if isinstance(full_artifact_requirements.get("required_files"), list):
        for entry in full_artifact_requirements.get("required_files") or []:
            if not entry:
                continue
            if isinstance(entry, dict):
                path = entry.get("path") or entry.get("output") or entry.get("artifact")
                if path and is_probably_path(str(path)):
                    required_files.append(
                        {"path": str(path), "description": str(entry.get("description") or "")}
                    )
            else:
                path = str(entry)
                if path and is_probably_path(path):
                    required_files.append({"path": path, "description": ""})

    for path in required_outputs:
        if not path:
            continue
        if not is_probably_path(str(path)):
            continue
        if any(str(path).lower() == str(item.get("path", "")).lower() for item in required_files):
            continue
        required_files.append({"path": str(path), "description": ""})

    # SYNC FIX: Filter out constant columns from clean_dataset.required_columns
    # Constant columns provide no information and should be excluded from the final schema
    constant_columns_set: set[str] = set()
    dropped_constant_columns: List[str] = []
    if isinstance(data_profile, dict):
        constant_cols_raw = data_profile.get("constant_columns")
        if isinstance(constant_cols_raw, list):
            constant_columns_set = {str(c) for c in constant_cols_raw if c}
            # Normalize to match canonical_columns
            constant_norms = {_normalize_column_identifier(c): c for c in constant_columns_set}
            for col in canonical_columns:
                col_norm = _normalize_column_identifier(col)
                if col_norm in constant_norms or col in constant_columns_set:
                    dropped_constant_columns.append(col)

    # Compute clean_dataset_required_columns excluding constant columns
    clean_dataset_required_columns = [
        col for col in canonical_columns
        if col not in dropped_constant_columns
    ]

    artifact_requirements = {
        "clean_dataset": {
            "required_columns": clean_dataset_required_columns,
            "output_path": "data/cleaned_data.csv",
            "excluded_constant_columns": dropped_constant_columns if dropped_constant_columns else [],
        },
        "metrics": {"required": True, "path": "data/metrics.json"},
        "alignment_check": {"required": True, "path": "data/alignment_check.json"},
        "plots": {"optional": True, "expected": ["*.png"]},
        # P1.1: Formal file vs column separation
        "required_files": required_files,
        "scored_rows_schema": scored_rows_schema,
        "file_schemas": full_artifact_requirements.get("file_schemas", {}) if isinstance(full_artifact_requirements.get("file_schemas"), dict) else {},
        "schema_binding": {
            "required_columns": clean_dataset_required_columns,
            "optional_passthrough_columns": [],
        },
    }

    data_engineer_runbook = "\n".join(
        [
            "Produce data/cleaned_data.csv containing ONLY the columns listed in required_columns.",
            "Your output CSV must match EXACTLY the required_columns list - no more, no less.",
            "If a column exists in raw data but is NOT in required_columns, DISCARD it (do not include in output).",
            "Constant columns (single unique value) have been pre-excluded from required_columns.",
            "Preserve column names; do not invent or rename columns.",
            "Load using output_dialect from cleaning_manifest.json when available.",
            "Parse numeric/date fields conservatively; document conversions.",
            "If a required column is missing from input, report and stop (no fabrication).",
            "Do not derive targets or train models.",
            "Avoid advanced validation metrics (MAE/correlation); report only dtype and null counts.",
            "Write cleaning_manifest.json with input/output dialect details.",
        ]
    )
    ml_engineer_runbook = "\n".join(
        [
            "Use allowed_feature_sets for modeling/segmentation.",
            "Never use forbidden_features in training or optimization.",
            "Produce data/scored_rows.csv, data/metrics.json, data/alignment_check.json.",
            "Respect output_dialect from cleaning_manifest.json.",
            "Document leakage checks and any data_limited_mode fallback.",
            "Include feature_usage in alignment_check.json (used_features, target_columns, excluded_features).",
        ]
    )

    business_objective = (
        contract.get("business_objective")
        or strategy_dict.get("business_objective")
        or ""
    )
    if business_objective and len(business_objective) > 2000:
        business_objective = business_objective[:2000]

    omitted_columns_policy = contract.get("omitted_columns_policy") or (
        "Ignored by default unless promoted by strategy/explicit mention; available via column_inventory."
    )

    def _extract_preprocessing_requirements_min(source: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(source, dict):
            return {}
        prep = source.get("preprocessing_requirements")
        if not isinstance(prep, dict):
            return {}
        out: Dict[str, Any] = {}
        nan_strategies = prep.get("nan_strategies")
        if isinstance(nan_strategies, list):
            trimmed = []
            for item in nan_strategies:
                if not isinstance(item, dict):
                    continue
                entry: Dict[str, Any] = {}
                for key in ("column", "strategy", "missing_category_value", "owner", "group_by"):
                    if key in item and item.get(key) not in (None, ""):
                        entry[key] = item.get(key)
                if entry:
                    trimmed.append(entry)
            if trimmed:
                out["nan_strategies"] = trimmed
        default_val = prep.get("missing_category_default")
        if isinstance(default_val, str) and default_val.strip():
            out["missing_category_default"] = default_val.strip()
        return out

    prep_reqs_min = _extract_preprocessing_requirements_min(contract)

    reporting_policy = contract.get("reporting_policy")
    if not isinstance(reporting_policy, dict) or not reporting_policy:
        execution_plan = contract.get("execution_plan")
        if isinstance(execution_plan, dict):
            reporting_policy = build_reporting_policy(execution_plan, strategy_dict)
    plot_spec = reporting_policy.get("plot_spec") if isinstance(reporting_policy, dict) else None
    if not isinstance(plot_spec, dict) or not plot_spec:
        plot_spec = build_plot_spec(contract)
    if isinstance(plot_spec, dict) and plot_spec:
        if not isinstance(reporting_policy, dict):
            reporting_policy = {}
        reporting_policy = dict(reporting_policy)
        reporting_policy["plot_spec"] = {
            "enabled": bool(plot_spec.get("enabled", True)),
            "max_plots": int(plot_spec.get("max_plots", len(plot_spec.get("plots") or []))),
        }

    qa_gates = _apply_qa_gate_policy(
        contract.get("qa_gates"),
        strategy_dict,
        business_objective or "",
        contract,
    )
    cleaning_gates = _apply_cleaning_gate_policy(contract.get("cleaning_gates"))
    training_rows_rule = contract.get("training_rows_rule")
    scoring_rows_rule = contract.get("scoring_rows_rule")
    secondary_scoring_subset = contract.get("secondary_scoring_subset")
    data_partitioning_notes = contract.get("data_partitioning_notes")
    if not isinstance(data_partitioning_notes, list):
        data_partitioning_notes = []

    from src.utils.contract_v41 import CONTRACT_VERSION_V41, normalize_contract_version
    contract_min = {
        "contract_version": normalize_contract_version(contract.get("contract_version")),
        "strategy_title": contract.get("strategy_title") or strategy_dict.get("title", ""),
        "business_objective": business_objective,
        "canonical_columns": canonical_columns,
        "outcome_columns": outcome_cols,
        "decision_columns": decision_cols,
        "column_roles": column_roles,
        "allowed_feature_sets": {
            "segmentation_features": segmentation_features,
            "model_features": model_features,
            "forbidden_features": forbidden_features,
            "audit_only_features": audit_only_features,
        },
        "artifact_requirements": artifact_requirements,
        "required_outputs": required_outputs,
        "feature_selectors": feature_selectors,  # P1.5: For wide datasets
        "qa_gates": qa_gates,
        "cleaning_gates": cleaning_gates,
        "reviewer_gates": [
            "strategy_followed",
            "metrics_present",
            "interpretability_ok",
        ],
        "data_engineer_runbook": data_engineer_runbook,
        "ml_engineer_runbook": ml_engineer_runbook,
        "omitted_columns_policy": omitted_columns_policy,
        "reporting_policy": reporting_policy or {},
        "decisioning_requirements": decisioning_requirements,
    }
    if prep_reqs_min:
        contract_min["preprocessing_requirements"] = prep_reqs_min
    if training_rows_rule:
        contract_min["training_rows_rule"] = training_rows_rule
    if scoring_rows_rule:
        contract_min["scoring_rows_rule"] = scoring_rows_rule
    if secondary_scoring_subset:
        contract_min["secondary_scoring_subset"] = secondary_scoring_subset
    if data_partitioning_notes:
        contract_min["data_partitioning_notes"] = data_partitioning_notes
    return contract_min


def ensure_v41_schema(contract: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """
    Validates and fills missing V4.1 schema keys.
    Adds to 'unknowns' array when filling defaults.

    Args:
        contract: Contract dict from LLM
        strict: If True, raise error on missing keys (for tests)

    Returns:
        Contract with all V4.1 keys present
    """
    from src.utils.contract_v41 import CONTRACT_VERSION_V41, normalize_contract_version

    if not isinstance(contract, dict):
        return contract

    required_keys = [
        "contract_version", "strategy_title", "business_objective",
        "missing_columns_handling", "execution_constraints",
        "objective_analysis", "data_analysis", "column_roles",
        "preprocessing_requirements", "feature_engineering_plan",
        "validation_requirements", "leakage_execution_plan",
        "optimization_specification", "segmentation_constraints",
        "data_limited_mode", "allowed_feature_sets",
        "artifact_requirements", "qa_gates", "cleaning_gates", "reviewer_gates",
        "data_engineer_runbook", "ml_engineer_runbook",
        "available_columns", "canonical_columns", "derived_columns",
        "required_outputs", "iteration_policy", "unknowns",
        "assumptions", "notes_for_engineers"
    ]

    repairs = []

    for key in required_keys:
        if key not in contract:
            if strict:
                raise ValueError(f"Missing required V4.1 key: {key}")

            # Fill with safe default
            if key == "contract_version":
                contract[key] = CONTRACT_VERSION_V41
            elif key in ("optimization_specification", "segmentation_constraints"):
                contract[key] = None
            elif key in ("unknowns", "assumptions", "notes_for_engineers", "available_columns",
                         "canonical_columns", "derived_columns", "required_outputs"):
                contract[key] = []
            elif key in ("qa_gates", "cleaning_gates", "reviewer_gates"):
                contract[key] = []
            elif key in ("strategy_title", "business_objective"):
                contract[key] = ""
            else:
                contract[key] = {}

            repairs.append(f"Added missing key: {key}")

    # Ensure unknowns is a list
    unknowns = contract.get("unknowns")
    if not isinstance(unknowns, list):
        unknowns = []
        contract["unknowns"] = unknowns

    # Normalize contract version to V4.1
    version = contract.get("contract_version")
    normalized_version = normalize_contract_version(version)
    if version != normalized_version:
        old_version = version
        contract["contract_version"] = normalized_version
        unknowns.append({
            "item": f"Normalized contract_version from {old_version} to {normalized_version}",
            "impact": "Schema validation enforced V4.1 version",
            "mitigation": "Review LLM output quality",
            "requires_verification": False
        })
    elif version is None:
        contract["contract_version"] = CONTRACT_VERSION_V41

    # Add repair notes to unknowns
    for repair in repairs:
        unknowns.append({
            "item": repair,
            "impact": "Schema validation filled missing field",
            "mitigation": "Review LLM output quality",
            "requires_verification": False
        })

    return contract


def validate_artifact_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates artifact_requirements to ensure required_columns are subset of canonical/derived columns.
    Moves non-canonical columns to optional_passthrough_columns.

    Args:
        contract: Contract dict with V4.1 schema

    Returns:
        Contract with validated artifact_requirements
    """
    if not isinstance(contract, dict):
        return contract

    # Get canonical and derived columns
    canonical_columns = contract.get("canonical_columns", [])
    derived_columns = contract.get("derived_columns", [])
    available_columns = contract.get("available_columns", [])

    if not isinstance(canonical_columns, list):
        canonical_columns = []
    if not isinstance(derived_columns, list):
        derived_columns = []
    if not isinstance(available_columns, list):
        available_columns = []

    # Build allowed column set (canonical + derived)
    allowed_columns_set = set(canonical_columns) | set(derived_columns)

    # Normalize for comparison (case-insensitive)
    allowed_norms = {_normalize_column_identifier(col): col for col in allowed_columns_set}
    available_norms = {_normalize_column_identifier(col): col for col in available_columns}

    # Get artifact_requirements
    artifact_requirements = contract.get("artifact_requirements", {})
    if not isinstance(artifact_requirements, dict):
        return contract

    schema_binding = artifact_requirements.get("schema_binding")
    if not isinstance(schema_binding, dict):
        schema_binding = {}
        artifact_requirements["schema_binding"] = schema_binding

    clean_dataset = artifact_requirements.get("clean_dataset")
    if not isinstance(clean_dataset, dict):
        clean_dataset = None

    if "required_columns" in schema_binding:
        required_columns = schema_binding.get("required_columns")
        if not isinstance(required_columns, list):
            # Preserve invalid type: do not mutate contract
            return contract
    else:
        required_columns = []

    if not required_columns:
        if clean_dataset and isinstance(clean_dataset.get("required_columns"), list):
            schema_binding["required_columns"] = [
                str(col) for col in clean_dataset.get("required_columns") if col
            ]
            required_columns = schema_binding["required_columns"]
        else:
            # V4.1: Use ONLY canonical_columns as fallback, NO legacy required_columns
            if isinstance(canonical_columns, list) and canonical_columns:
                schema_binding["required_columns"] = [str(col) for col in canonical_columns if col]
                required_columns = schema_binding["required_columns"]
            else:
                # No canonical columns available - record as unknown
                schema_binding["required_columns"] = []
                required_columns = []
                unknowns = contract.setdefault("unknowns", [])
                if isinstance(unknowns, list):
                    unknowns.append({
                        "item": "artifact_requirements.schema_binding.required_columns is empty",
                        "impact": "No columns specified for clean dataset validation",
                        "mitigation": "Ensure canonical_columns is populated in contract",
                        "requires_verification": True
                    })

    # Validate required_columns
    valid_required = []
    moved_to_optional = []

    for col in required_columns:
        if not col:
            continue
        col_norm = _normalize_column_identifier(col)

        # Check if column is in canonical or derived columns
        if col_norm in allowed_norms:
            valid_required.append(col)
        # If it's in available_columns but not canonical, move to optional
        elif col_norm in available_norms:
            moved_to_optional.append(col)
        # If it's not even in available_columns, it's invalid - skip it
        else:
            moved_to_optional.append(col)

    # Update schema_binding
    if moved_to_optional:
        schema_binding["required_columns"] = valid_required
        if clean_dataset is not None:
            clean_dataset["required_columns"] = list(valid_required)

        # Add to optional_passthrough_columns
        optional_passthrough = schema_binding.get("optional_passthrough_columns", [])
        if not isinstance(optional_passthrough, list):
            optional_passthrough = []

        # Add moved columns to optional_passthrough
        for col in moved_to_optional:
            if col not in optional_passthrough:
                optional_passthrough.append(col)

        schema_binding["optional_passthrough_columns"] = optional_passthrough

        # Document in unknowns
        unknowns = contract.get("unknowns", [])
        if not isinstance(unknowns, list):
            unknowns = []
            contract["unknowns"] = unknowns

        unknowns.append({
            "item": f"artifact_requirements.required_columns contained non-canonical columns: {moved_to_optional}",
            "impact": "Moved to optional_passthrough_columns to preserve them without enforcement",
            "mitigation": "These columns will be included in outputs if present in data, but are not required",
            "requires_verification": True,
            "auto_corrected": True
        })

    return contract


def _normalize_column_identifier(value: Any) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "", str(value).lower())
    return cleaned


def _is_identifier_like_name(name: str) -> bool:
    if not name:
        return False
    normalized = _tokenize_name(name)
    if not normalized:
        return False
    tokens = normalized.split()
    joined = normalized.replace(" ", "")
    strict = {"id", "uuid", "guid", "rowid", "recordid", "index", "idx"}
    if joined in strict:
        return True
    if any(tok in {"uuid", "guid"} for tok in tokens):
        return True
    if "id" in tokens or "key" in tokens:
        return True
    for tok in tokens:
        if len(tok) > 3 and tok.endswith("id"):
            return True
    return False


def _load_profile_identifier_candidates() -> set[str]:
    return set()


def _resolve_identifier_candidates(
    contract: Dict[str, Any],
    canonical_columns: List[str] | None,
) -> set[str]:
    candidates: set[str] = set()
    roles = contract.get("column_roles") if isinstance(contract, dict) else None
    if isinstance(roles, dict):
        role_ids = roles.get("identifiers")
        if isinstance(role_ids, list):
            candidates.update([str(col) for col in role_ids if col])
    return candidates


def _prune_identifier_model_features(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    allowed_sets = contract.get("allowed_feature_sets")
    if not isinstance(allowed_sets, dict):
        return contract
    model_features = allowed_sets.get("model_features")
    if not isinstance(model_features, list) or not model_features:
        return contract
    canonical_columns = contract.get("canonical_columns")
    if not isinstance(canonical_columns, list):
        canonical_columns = []
    identifier_candidates = _resolve_identifier_candidates(contract, canonical_columns)
    if not identifier_candidates:
        return contract
    filtered = [col for col in model_features if col not in identifier_candidates]
    removed = [col for col in model_features if col not in filtered]
    if not removed:
        return contract
    allowed_sets["model_features"] = filtered
    audit_only = allowed_sets.get("audit_only_features")
    if not isinstance(audit_only, list):
        audit_only = []
    for col in removed:
        if col not in audit_only:
            audit_only.append(col)
    allowed_sets["audit_only_features"] = audit_only
    contract["allowed_feature_sets"] = allowed_sets
    return contract


def _build_allowed_column_norms(column_sets: List[str] | None, *more_sets: List[str] | None) -> set[str]:
    norms: set[str] = set()
    for collection in (column_sets, *more_sets):
        if not isinstance(collection, list):
            continue
        for col in collection:
            normed = _normalize_column_identifier(col)
            if normed:
                norms.add(normed)
    return norms


def _filter_leakage_audit_features(
    spec: Dict[str, Any],
    canonical_columns: List[str] | None,
    column_inventory: List[str] | None,
) -> List[str]:
    policy = spec.get("leakage_policy")
    if not isinstance(policy, dict):
        return []
    features = policy.get("audit_features")
    if not isinstance(features, list):
        return []
    allowed_norms = _build_allowed_column_norms(canonical_columns, column_inventory)
    filtered_out: List[str] = []

    if not allowed_norms:
        filtered_out = [str(item) for item in features if item]
        policy["audit_features"] = []
    else:
        kept: List[str] = []
        for item in features:
            if not item:
                continue
            normed = _normalize_column_identifier(item)
            if normed not in allowed_norms:
                filtered_out.append(str(item))
                continue
            kept.append(item)
        policy["audit_features"] = kept

    if filtered_out:
        detail = spec.get("leakage_policy_detail")
        if not isinstance(detail, dict):
            detail = {}
            spec["leakage_policy_detail"] = detail
        detail.setdefault("filtered_audit_features", [])
        existing = detail["filtered_audit_features"]
        existing.extend(filtered_out)
    return filtered_out

def parse_derive_from_expression(expr: str) -> Dict[str, Any]:
    if not expr or not isinstance(expr, str):
        return {}
    text = expr.strip()
    if not text:
        return {}

    def _coerce_values(raw: str) -> List[str]:
        if not raw:
            return []
        cleaned = raw.strip()
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
            if isinstance(parsed, str):
                return [parsed]
            return [str(parsed)]
        except Exception:
            pass
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", "\""}:
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()
        if "," in cleaned:
            parts = [part.strip(" \"'") for part in cleaned.split(",") if part.strip()]
            return parts
        return [cleaned] if cleaned else []

    match = re.match(r"^\s*([A-Za-z0-9_][A-Za-z0-9_ %\.\-]*)\s*==\s*(.+?)\s*$", text)
    if match:
        column = match.group(1).strip()
        values = _coerce_values(match.group(2))
        return {"column": column, "positive_values": values}
    match = re.match(r"^\s*([A-Za-z0-9_][A-Za-z0-9_ %\.\-]*)\s+in\s+(.+?)\s*$", text, flags=re.IGNORECASE)
    if match:
        column = match.group(1).strip()
        values = _coerce_values(match.group(2))
        return {"column": column, "positive_values": values}
    token_match = re.search(r"[A-Za-z0-9_][A-Za-z0-9_ %\.\-]*", text)
    if token_match:
        column = token_match.group(0).strip()
        return {"column": column, "positive_values": []}
    return {}

def enforce_percentage_ranges(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    reqs = contract.get("data_requirements", []) or []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        role = (req.get("role") or "").lower()
        expected = req.get("expected_range")
        if role == "percentage" and not expected:
            req["expected_range"] = [0, 1]
    notes = contract.get("notes_for_engineers")
    if not isinstance(notes, list):
        notes = []
    note = "Percentages must be normalized to 0-1; if values look like 0-100 scale, divide by 100."
    if note not in notes:
        notes.append(note)
    contract["notes_for_engineers"] = notes
    contract["data_requirements"] = reqs
    return contract


def build_dataset_profile(data_summary: str, column_inventory: List[str] | None = None) -> Dict[str, Any]:
    profile: Dict[str, Any] = {"column_count": len(column_inventory or [])}
    summary = (data_summary or "").strip()
    if summary:
        profile["summary_excerpt"] = summary[:400]
    return profile


def build_execution_plan(objective_type: str, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    objective = (objective_type or "unknown").lower()
    gates = [
        {"id": "data_ok", "description": "Data availability and basic quality checks pass.", "required": True},
        {"id": "target_ok", "description": "Target is valid with sufficient variation.", "required": True},
        {"id": "leakage_ok", "description": "No post-outcome leakage in features.", "required": True},
        {"id": "runtime_ok", "description": "Pipeline executes without runtime failures.", "required": True},
        {"id": "eval_ok", "description": "Evaluation meets objective-specific thresholds.", "required": True},
    ]

    base_outputs = [
        {"artifact_type": "clean_dataset", "required": True, "description": "Cleaned dataset for downstream use."},
        {"artifact_type": "artifact_index", "required": True, "description": "Typed inventory of produced artifacts."},
        {"artifact_type": "insights", "required": True, "description": "Unified insights for downstream reporting."},
        {"artifact_type": "executive_summary", "required": True, "description": "Business-facing summary."},
    ]

    objective_outputs: Dict[str, List[Dict[str, Any]]] = {
        "classification": [
            {"artifact_type": "metrics", "required": True, "description": "Classification metrics."},
            {"artifact_type": "predictions", "required": True, "description": "Predicted labels/probabilities."},
            {"artifact_type": "confusion_matrix", "required": False, "description": "Error breakdown by class."},
        ],
        "regression": [
            {"artifact_type": "metrics", "required": True, "description": "Regression metrics."},
            {"artifact_type": "predictions", "required": True, "description": "Predicted numeric outputs."},
            {"artifact_type": "residuals", "required": False, "description": "Residual diagnostics."},
        ],
        "forecasting": [
            {"artifact_type": "metrics", "required": True, "description": "Forecasting metrics."},
            {"artifact_type": "forecast", "required": True, "description": "Forecast outputs."},
            {"artifact_type": "backtest", "required": False, "description": "Historical forecast evaluation."},
        ],
        "ranking": [
            {"artifact_type": "metrics", "required": True, "description": "Ranking metrics."},
            {"artifact_type": "ranking_scores", "required": True, "description": "Ranked scores output."},
            {"artifact_type": "ranking_report", "required": False, "description": "Ranking diagnostics."},
        ],
    }
    optional_common = [
        {"artifact_type": "feature_importances", "required": False, "description": "Explainability artifact."},
        {"artifact_type": "error_analysis", "required": False, "description": "Failure mode analysis."},
        {"artifact_type": "plots", "required": False, "description": "Diagnostic plots."},
    ]

    outputs = list(base_outputs)
    outputs.extend(objective_outputs.get(objective, [{"artifact_type": "metrics", "required": True, "description": "Evaluation metrics."}]))
    outputs.extend(optional_common)

    return {
        "schema_version": "1",
        "objective_type": objective,
        "dataset_profile": dataset_profile or {},
        "gates": gates,
        "outputs": outputs,
    }


def build_reporting_policy(
    execution_plan: Dict[str, Any] | None,
    strategy: Dict[str, Any] | None = None,
    run_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    outputs = execution_plan.get("outputs", []) if isinstance(execution_plan, dict) else []
    output_types = {
        str(item.get("artifact_type"))
        for item in outputs
        if isinstance(item, dict) and item.get("artifact_type")
    }

    slots: List[Dict[str, Any]] = []

    def _add_slot(slot_id: str, mode: str, insights_key: str, sources: List[str] | None = None) -> None:
        if not slot_id or any(slot.get("id") == slot_id for slot in slots):
            return
        slot = {
            "id": slot_id,
            "mode": mode,
            "insights_key": insights_key,
            "sources": sources or [],
        }
        slots.append(slot)

    if "metrics" in output_types:
        _add_slot("model_metrics", "required", "metrics_summary", ["data/metrics.json"])
    if "predictions" in output_types:
        _add_slot("predictions_overview", "conditional", "predictions_summary", ["data/scored_rows.csv"])
    if "feature_importances" in output_types:
        _add_slot("explainability", "optional", "feature_importances_summary", [])
    if "error_analysis" in output_types:
        _add_slot("error_analysis", "optional", "error_summary", [])
    if "forecast" in output_types:
        _add_slot("forecast_summary", "required", "forecast_summary", [])
    if "ranking_scores" in output_types:
        _add_slot("ranking_top", "required", "ranking_summary", [])

    _add_slot("alignment_risks", "conditional", "leakage_audit", ["data/alignment_check.json"])
    _add_slot("segment_pricing", "conditional", "segment_pricing_summary", ["data/scored_rows.csv"])

    policy = {
        "audience": "executive",
        "language": "auto",
        "sections": [
            "decision",
            "objective_approach",
            "evidence_metrics",
            "business_impact",
            "risks_limitations",
            "next_actions",
            "visual_insights",
        ],
        "slots": slots,
        "constraints": {"no_markdown_tables": True},
    }

    policy.setdefault("demonstrative_examples_enabled", True)
    policy.setdefault("demonstrative_examples_when_outcome_in", ["NO_GO", "GO_WITH_LIMITATIONS"])
    policy.setdefault("max_examples", 5)
    policy.setdefault("require_strong_disclaimer", True)

    return policy


def build_plot_spec(contract_full: Dict[str, Any] | None) -> Dict[str, Any]:
    contract = contract_full if isinstance(contract_full, dict) else {}
    required_outputs = get_required_outputs(contract)
    outputs_lower = [str(path).lower() for path in required_outputs if path]

    def _has_output(token: str) -> bool:
        return any(token in path for path in outputs_lower)

    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in values:
            if not item:
                continue
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _sample(values: List[str], limit: int) -> List[str]:
        return _dedupe([str(v) for v in values if v])[:limit]

    def _safe_column_name(name: str) -> str:
        if not name:
            return ""
        return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()

    def _has_token(name: str, tokens: List[str]) -> bool:
        if not name:
            return False
        normalized = re.sub(r"[^0-9a-zA-Z]+", " ", str(name).lower())
        return any(tok in normalized.split() for tok in tokens)

    def _infer_objective_type() -> str:
        eval_spec = contract.get("evaluation_spec") if isinstance(contract, dict) else None
        if isinstance(eval_spec, dict) and eval_spec.get("objective_type"):
            return str(eval_spec.get("objective_type"))
        plan = contract.get("execution_plan") if isinstance(contract, dict) else None
        if isinstance(plan, dict) and plan.get("objective_type"):
            return str(plan.get("objective_type"))
        obj_analysis = contract.get("objective_analysis") if isinstance(contract, dict) else None
        if isinstance(obj_analysis, dict) and obj_analysis.get("problem_type"):
            return str(obj_analysis.get("problem_type"))
        return "unknown"

    objective_type = _infer_objective_type().lower()
    eval_spec = contract.get("evaluation_spec") if isinstance(contract, dict) else None
    target_type = str(eval_spec.get("target_type") or "").lower() if isinstance(eval_spec, dict) else ""

    is_ranking = any(tok in objective_type for tok in ["rank", "scor", "priorit"])
    is_forecast = "forecast" in objective_type
    is_segmentation = any(tok in objective_type for tok in ["segment", "cluster"])
    is_classification = any(tok in target_type for tok in ["class", "binary", "multiclass"]) or "classif" in objective_type
    is_regression = any(tok in target_type for tok in ["regress", "continuous", "numeric"]) or "regress" in objective_type

    canonical_columns = get_canonical_columns(contract)
    canonical_set = set(canonical_columns)

    def _filter_to_canonical(values: List[str]) -> List[str]:
        if not canonical_set:
            return [str(v) for v in values if v]
        return [str(v) for v in values if v in canonical_set]

    roles = get_column_roles(contract)
    pre_decision = _filter_to_canonical(_coerce_list(roles.get("pre_decision")))
    decision_cols = _filter_to_canonical(_coerce_list(roles.get("decision")))
    outcome_cols = _filter_to_canonical(_coerce_list(roles.get("outcome")))
    audit_only = _filter_to_canonical(
        _coerce_list(roles.get("post_decision_audit_only") or roles.get("audit_only"))
    )

    allowed_sets = contract.get("allowed_feature_sets")
    if not isinstance(allowed_sets, dict):
        allowed_sets = {}
    model_features = _filter_to_canonical(_coerce_list(allowed_sets.get("model_features")))
    segmentation_features = _filter_to_canonical(_coerce_list(allowed_sets.get("segmentation_features")))

    data_analysis = contract.get("data_analysis") if isinstance(contract, dict) else None
    type_dist = data_analysis.get("type_distribution") if isinstance(data_analysis, dict) else None
    numeric_cols = _filter_to_canonical(_coerce_list(type_dist.get("numeric"))) if isinstance(type_dist, dict) else []
    datetime_cols = _filter_to_canonical(_coerce_list(type_dist.get("datetime"))) if isinstance(type_dist, dict) else []
    categorical_cols = _filter_to_canonical(_coerce_list(type_dist.get("categorical"))) if isinstance(type_dist, dict) else []

    if not datetime_cols:
        time_tokens = ["date", "time", "timestamp", "period"]
        datetime_cols = [col for col in canonical_columns if _has_token(col, time_tokens)]

    derived_columns = get_derived_column_names(contract)
    score_tokens = ["score", "pred", "prob", "prediction"]
    pred_name_candidates = [col for col in derived_columns if _has_token(col, score_tokens)]

    pred_candidates: List[str] = ["prediction"]
    for outcome in outcome_cols[:3]:
        safe = _safe_column_name(outcome)
        if safe:
            pred_candidates.extend([f"pred_{safe}", f"predicted_{safe}", f"pred_prob_{safe}"])
    pred_candidates.extend(pred_name_candidates)
    pred_candidates = _sample(pred_candidates, 12)

    segment_tokens = ["segment", "segmentation", "cluster", "cohort", "group", "segmento", "cluster_id"]
    segment_candidates = [col for col in derived_columns if _has_token(col, segment_tokens)]
    if not segment_candidates:
        segment_candidates = [col for col in canonical_columns if _has_token(col, segment_tokens)]
    segment_candidates = _sample(segment_candidates, 8)

    has_scored_rows = _has_output("scored_rows.csv")
    has_metrics = _has_output("metrics.json")
    has_alignment = _has_output("alignment_check.json") or _has_output("case_alignment")
    has_weights = _has_output("weights.json")

    plots: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add_plot(
        plot_id: str,
        title: str,
        goal: str,
        plot_type: str,
        preferred_sources: List[str],
        required_any: List[str] | None = None,
        required_all: List[str] | None = None,
        optional_cols: List[str] | None = None,
        compute: Dict[str, Any] | None = None,
        caption_template: str | None = None,
    ) -> None:
        if not plot_id or plot_id in seen_ids:
            return
        seen_ids.add(plot_id)
        plot = {
            "plot_id": plot_id,
            "title": title,
            "goal": goal,
            "type": plot_type,
            "inputs": {
                "preferred_sources": preferred_sources,
                "required_columns_any_of": [required_any] if required_any else [],
                "required_columns_all_of": required_all or [],
                "optional_columns": optional_cols or [],
            },
            "compute": compute or {},
            "caption_template": caption_template or "",
        }
        plots.append(plot)

    if len(canonical_columns) >= 4:
        _add_plot(
            "missingness_overview",
            "Missingness by column",
            "Quantify missing data to focus cleaning and feature engineering.",
            "bar",
            ["data/cleaned_data.csv"],
            optional_cols=_sample(canonical_columns, 24),
            compute={"metric": "missing_fraction", "top_k": 20},
            caption_template="Top missingness rates across columns (top {top_k}).",
        )

    if numeric_cols:
        _add_plot(
            "numeric_distributions",
            "Numeric feature distributions",
            "Show the distribution of key numeric features.",
            "histogram",
            ["data/cleaned_data.csv"],
            required_any=_sample(numeric_cols, 12),
            optional_cols=_sample(numeric_cols, 12),
            compute={"x": "AUTO_NUMERIC", "max_columns": min(6, len(numeric_cols))},
            caption_template="Distributions for selected numeric features.",
        )

    if datetime_cols and (is_forecast or has_scored_rows):
        _add_plot(
            "trend_over_time",
            "Trend over time",
            "Highlight temporal trends for the primary target or prediction.",
            "timeseries",
            ["data/cleaned_data.csv", "data/scored_rows.csv"],
            required_any=_sample(datetime_cols, 6),
            optional_cols=_sample(datetime_cols, 6),
            compute={"x": "AUTO_TIME", "y": "AUTO_TARGET_OR_NUMERIC"},
            caption_template="Trend over time using available temporal columns.",
        )

    if has_scored_rows and pred_candidates:
        _add_plot(
            "score_distribution",
            "Prediction/score distribution",
            "Summarize the distribution of model outputs.",
            "histogram",
            ["data/scored_rows.csv", "data/cleaned_data.csv"],
            required_any=pred_candidates,
            compute={"x": "PREDICTION", "bins": 30},
            caption_template="Distribution of predicted scores.",
        )

    if has_scored_rows and outcome_cols and (is_classification or is_ranking):
        _add_plot(
            "topk_lift",
            "Top-k outcome lift",
            "Show outcome rate across score buckets.",
            "bar",
            ["data/scored_rows.csv"],
            required_any=pred_candidates,
            required_all=[outcome_cols[0]],
            compute={"x": "PREDICTION", "y": outcome_cols[0], "group_by": "decile", "metric": "mean"},
            caption_template="Outcome rate by score decile.",
        )

    if has_scored_rows and outcome_cols and is_regression:
        _add_plot(
            "residuals_scatter",
            "Prediction vs actual",
            "Assess residuals and bias in regression outputs.",
            "scatter",
            ["data/scored_rows.csv"],
            required_any=pred_candidates,
            required_all=[outcome_cols[0]],
            compute={"x": "PREDICTION", "y": outcome_cols[0]},
            caption_template="Predicted vs actual values.",
        )

    if has_weights or has_metrics:
        _add_plot(
            "feature_weights",
            "Feature weights/importance",
            "Highlight the strongest feature contributions or weights.",
            "bar",
            ["data/weights.json", "data/metrics.json"],
            compute={"metric": "weights", "top_k": 20},
            caption_template="Top contributing features (if weights available).",
        )

    if has_alignment:
        _add_plot(
            "alignment_check_summary",
            "Alignment check summary",
            "Visualize alignment requirements or validation outcomes.",
            "bar",
            ["data/alignment_check.json"],
            compute={"metric": "alignment_requirements", "top_k": 12},
            caption_template="Alignment check results by requirement.",
        )

    if is_segmentation or segmentation_features or segment_candidates:
        _add_plot(
            "segment_sizes",
            "Segment distribution",
            "Show sizes or performance by segment where available.",
            "bar",
            ["data/scored_rows.csv", "data/cleaned_data.csv"],
            required_any=segment_candidates,
            compute={"x": "SEGMENT_COLUMN", "metric": "count", "top_k": 20},
            caption_template="Segment sizes based on available segment identifiers.",
        )

    max_plots = 8
    trimmed_plots = plots[:max_plots]

    return {
        "enabled": bool(trimmed_plots),
        "max_plots": max_plots,
        "plots": trimmed_plots,
    }


def _contains_visual_token(text: str, tokens: set[str]) -> bool:
    if not text:
        return False
    words = set(text.split())
    return any(tok in words for tok in tokens)


def _map_plot_type(plot_type: str | None) -> str:
    if not plot_type:
        return "other"
    normalized = str(plot_type).lower()
    mapping = {
        "histogram": "distribution",
        "bar": "comparison",
        "line": "timeseries",
        "timeseries": "timeseries",
        "scatter": "comparison",
        "heatmap": "comparison",
        "box": "distribution",
        "pie": "comparison",
        "area": "timeseries",
    }
    return mapping.get(normalized, "other")


def _extract_columns_from_inputs(plot: Dict[str, Any]) -> List[str]:
    inputs = plot.get("inputs") if isinstance(plot.get("inputs"), dict) else {}
    columns: List[str] = []
    for key in ("required_columns_any_of", "required_columns_all_of", "optional_columns"):
        value = inputs.get(key)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, list):
                    columns.extend([str(item) for item in entry if item])
                elif entry:
                    columns.append(str(entry))
    return columns


def _build_visual_requirements(
    contract: Dict[str, Any],
    strategy: Dict[str, Any],
    business_objective: str,
) -> Dict[str, Any]:
    vision_text = _normalize_text(
        strategy.get("analysis_type"),
        strategy.get("techniques"),
        strategy.get("notes"),
        strategy.get("description"),
        strategy.get("objective_type"),
        business_objective,
        contract.get("business_objective"),
        contract.get("strategy_title"),
    )
    enabled = _contains_visual_token(vision_text, _VISUAL_ENABLED_TOKENS)
    required = _matches_any_phrase(vision_text, _VISUAL_REQUIRED_PHRASES)

    dataset_profile = (
        contract.get("dataset_profile") if isinstance(contract.get("dataset_profile"), dict) else {}
    )
    row_count = 0
    if dataset_profile:
        for key in ("row_count", "rows", "estimated_rows"):
            val = dataset_profile.get(key)
            if isinstance(val, (int, float)) and val > 0:
                row_count = int(val)
                break
    sampling_strategy = "random" if row_count > 50000 else "none"
    max_rows_for_plot = 5000
    if row_count and row_count < max_rows_for_plot:
        max_rows_for_plot = max(row_count, 1000)

    outputs_dir = "static/plots"
    artifact_reqs = contract.get("artifact_requirements")
    if isinstance(artifact_reqs, dict):
        outputs_dir = artifact_reqs.get("visual_outputs_dir") or artifact_reqs.get("outputs_dir") or outputs_dir

    plot_spec = build_plot_spec(contract) if enabled else {"enabled": False, "plots": [], "max_plots": 0}
    plots = plot_spec.get("plots") if isinstance(plot_spec.get("plots"), list) else []
    column_roles = get_column_roles(contract)
    outcome_cols = [str(c) for c in (column_roles.get("outcome") or []) if c]
    items: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, plot in enumerate(plots):
        if not isinstance(plot, dict):
            continue
        plot_id = str(plot.get("plot_id") or plot.get("id") or f"visual_{idx}")
        safe_id = re.sub(r"[^0-9a-zA-Z]+", "_", plot_id).strip("_").lower() or f"visual_{idx}"
        if safe_id in seen_ids:
            safe_id = f"{safe_id}_{idx}"
        seen_ids.add(safe_id)
        goal = str(plot.get("goal") or plot.get("title") or "Visual insight")
        inputs = plot.get("inputs") if isinstance(plot.get("inputs"), dict) else {}
        preferred_sources = [str(src) for src in (inputs.get("preferred_sources") or []) if src]
        columns_from_plot = _extract_columns_from_inputs(plot)
        requires_target = any(col in outcome_cols for col in columns_from_plot)
        requires_predictions = any("scored_rows.csv" in src for src in preferred_sources)
        requires_segments = "segment" in safe_id or "segment" in goal.lower()
        items.append(
            {
                "id": safe_id,
                "purpose": goal,
                "type": _map_plot_type(plot.get("type") or plot.get("plot_type")),
                "inputs": {
                    "requires_target": requires_target,
                    "requires_predictions": requires_predictions,
                    "requires_segments": requires_segments,
                    "columns_hint": columns_from_plot[:6],
                },
                "constraints": {
                    "max_rows_for_plot": max_rows_for_plot,
                    "sampling_strategy": sampling_strategy,
                },
                "expected_filename": f"{safe_id}.png",
            }
        )
    notes = (
        "Visual requirements are contract-driven. If items are listed, produce each exactly and store status in data/visuals_status.json."
        if items
        else "Visual requirements are disabled for this strategy."
    )
    return {
        "enabled": enabled,
        "required": required,
        "outputs_dir": outputs_dir,
        "items": items,
        "notes": notes,
        "plot_spec": plot_spec,
    }


def _build_decisioning_requirements(
    contract: Dict[str, Any],
    strategy: Dict[str, Any],
    business_objective: str,
) -> Dict[str, Any]:
    strategy_text = _normalize_text(
        strategy.get("analysis_type"),
        strategy.get("techniques"),
        strategy.get("notes"),
        strategy.get("description"),
        contract.get("strategy_title"),
        contract.get("business_objective"),
        business_objective,
    )
    objective_type = str(
        contract.get("objective_type")
        or strategy.get("analysis_type")
        or ""
    ).lower()
    enabled = _contains_decisioning_token(strategy_text, _DECISIONING_ENABLED_TOKENS) or any(
        kw in objective_type for kw in ["rank", "priority", "decision", "segment", "triage", "outlier", "action"]
    )
    required = _matches_any_phrase(strategy_text, _DECISIONING_REQUIRED_PHRASES)
    explanation_needed = _contains_decisioning_token(strategy_text, _EXPLANATION_REQUIRED_TOKENS) or _matches_any_phrase(
        strategy_text, _EXPLANATION_REQUIRED_PHRASES
    )
    if explanation_needed:
        enabled = True
        required = True

    canonical_columns = contract.get("canonical_columns") if isinstance(contract.get("canonical_columns"), list) else []
    canonical_columns = [str(col) for col in canonical_columns if col]
    id_column = next((col for col in canonical_columns if re.search(r"^id$|_id$|row_id", col, flags=re.IGNORECASE)), None)
    key_columns = [id_column] if id_column else canonical_columns[:1]

    def _resolve_explanation_column_name() -> str:
        artifact_reqs = contract.get("artifact_requirements")
        if not isinstance(artifact_reqs, dict):
            artifact_reqs = {}
        scored_schema = artifact_reqs.get("scored_rows_schema")
        required_cols: List[str] = []
        if isinstance(scored_schema, dict):
            required = scored_schema.get("required_columns")
            if isinstance(required, list):
                required_cols = [str(c) for c in required if c]
        if required_cols:
            for col in required_cols:
                if _normalize_column_token(col) == "explanation":
                    return col
            for col in required_cols:
                if "driver" in _normalize_column_token(col):
                    return col
        if any(tok in strategy_text for tok in ["driver", "drivers", "factor", "factors"]):
            return "top_drivers"
        return "explanation"

    def _has_objective(tok_list: List[str]) -> bool:
        return any(tok in objective_type for tok in tok_list) or _contains_decisioning_token(strategy_text, set(tok_list))

    is_classification = _has_objective(["class", "classification", "binary", "propensity", "moderation"])
    is_regression = _has_objective(["regress", "price", "eta", "forecast", "numeric", "value"])
    is_ranking = _has_objective(["rank", "priority", "top", "targeting", "triage"])
    is_segmentation = _has_objective(["segment", "cohort", "cluster"])
    is_outlier = _has_objective(["outlier", "anomaly"])
    needs_action = _has_objective(["action", "decision", "policy", "review", "moderation"])

    columns: List[Dict[str, Any]] = []
    inputs_base = ["prediction"]
    if is_classification:
        inputs_base.append("probability")
    if is_regression:
        inputs_base.append("prediction")

    if is_ranking or is_classification:
        columns.append(
            _build_decision_column_entry(
                name="priority_score",
                role="score",
                type_name="numeric",
                inputs=["prediction", "probability"],
                logic_hint="Use normalized model score/probability to represent priority from 0 to 1.",
                constraints={"non_null_rate_min": 0.98, "range": {"min": 0.0, "max": 1.0}},
            )
        )
        columns.append(
            _build_decision_column_entry(
                name="priority_rank",
                role="priority",
                type_name="integer",
                inputs=["priority_score"],
                logic_hint="Rank records by priority_score into integer ranks (1=head, higher=lower priority).",
                constraints={"non_null_rate_min": 0.95, "range": {"min": 1, "max": None}},
            )
        )
    if is_regression:
        columns.append(
            _build_decision_column_entry(
                name="prediction_value",
                role="prediction",
                type_name="numeric",
                inputs=["prediction"],
                logic_hint="Use the regression model output as the primary prediction value.",
                constraints={"non_null_rate_min": 0.98},
            )
        )
        columns.append(
            _build_decision_column_entry(
                name="uncertainty_flag",
                role="flag",
                type_name="bool",
                inputs=["prediction", "residual"],
                logic_hint="Flag high uncertainty when residuals exceed thresholds or standard deviation is high.",
                allowed_values=["True", "False"],
                constraints={"non_null_rate_min": 0.95},
            )
        )
    if explanation_needed:
        name_hint = _resolve_explanation_column_name()
        if not any(str(entry.get("name")) == name_hint for entry in columns if isinstance(entry, dict)):
            columns.append(
                _build_decision_column_entry(
                    name=name_hint,
                    role="explanation",
                    type_name="string",
                    inputs=["prediction", "probability", "segment", "rule"],
                    logic_hint="Provide a concise per-row explanation or top drivers; keep it short and auditable.",
                    constraints={"non_null_rate_min": 0.9},
                )
            )
    if is_outlier:
        columns.append(
            _build_decision_column_entry(
                name="outlier_flag",
                role="flag",
                type_name="bool",
                inputs=["residual", "prediction"],
                logic_hint="Mark rows as outliers when residuals or scoring errors exceed predefined thresholds.",
                allowed_values=["True", "False"],
                constraints={"non_null_rate_min": 0.95},
            )
        )
        columns.append(
            _build_decision_column_entry(
                name="outlier_severity",
                role="score",
                type_name="numeric",
                inputs=["residual"],
                logic_hint="Use absolute residual magnitude to quantify severity.",
                constraints={"non_null_rate_min": 0.9, "range": {"min": 0.0}},
            )
        )
    if is_segmentation:
        columns.append(
            _build_decision_column_entry(
                name="segment_assignment",
                role="segment",
                type_name="category",
                inputs=["segment_id", "cluster_label"],
                logic_hint="Assign rows to the declared segment identifiers detected in the contract or derived clusters.",
                constraints={"non_null_rate_min": 0.9},
            )
        )
    if needs_action:
        columns.append(
            _build_decision_column_entry(
                name="action_recommendation",
                role="action",
                type_name="category",
                inputs=["priority_score", "uncertainty_flag"],
                logic_hint="Map scores and flags to actions (review/contact/escalate) using simple thresholds.",
                allowed_values=["review", "contact", "escalate", "ignore"],
                constraints={"non_null_rate_min": 0.9},
            )
        )

    if not columns and enabled:
        # Provide a default minimal decision pipeline
        columns.append(
            _build_decision_column_entry(
                name="decision_flag",
                role="flag",
                type_name="bool",
                inputs=["priority_score"],
                logic_hint="Flag rows for follow-up when priority_score exceeds a business threshold.",
                allowed_values=["True", "False"],
                constraints={"non_null_rate_min": 0.9},
            )
        )

    unique = []
    seen_names = set()
    for col in columns:
        if col["name"] in seen_names:
            continue
        seen_names.add(col["name"])
        unique.append(col)

    policy_notes = (
        "These decision columns capture business priority, action recommendations, and uncertainty flags requested by the strategy."
        if unique
        else "Decision policy is disabled for this strategy."
    )

    return {
        "enabled": bool(enabled and unique),
        "required": required and bool(unique),
        "output": {
            "file": "data/scored_rows.csv",
            "key_columns": [col for col in key_columns if col],
            "required_columns": unique,
        },
        "policy_notes": policy_notes,
    }


class ExecutionPlannerAgent:
    """
    LLM-driven planner that emits an execution contract (JSON) to guide downstream agents.
    Falls back to heuristic contract if the model call fails.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            self.client = None
        else:
            genai.configure(api_key=self.api_key)
            generation_config = {
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            self.client = genai.GenerativeModel(
                model_name="gemini-3-flash-preview",
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
        self.model_name = "gemini-3-flash-preview"
        self.last_prompt = None
        self.last_response = None

    def generate_contract(
        self,
        strategy: Dict[str, Any],
        data_summary: str = "",
        business_objective: str = "",
        column_inventory: list[str] | None = None,
        output_dialect: Dict[str, str] | None = None,
        env_constraints: Dict[str, Any] | None = None,
        domain_expert_critique: str = "",
        data_profile: Dict[str, Any] | None = None,
        run_id: str | None = None
    ) -> Dict[str, Any]:
        def _norm(name: str) -> str:
            return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

        def _canonicalize_name(name: str) -> str:
            return str(name)

        def _resolve_exact_header(name: str) -> str | None:
            if not name or not column_inventory:
                return None
            norm_name = _norm(name)
            if not norm_name:
                return None
            best_match = None
            best_score = 0.0
            for raw in column_inventory:
                if raw is None:
                    continue
                raw_str = str(raw)
                raw_norm = _norm(raw_str)
                if raw_norm == norm_name:
                    return raw_str
                score = difflib.SequenceMatcher(None, norm_name, raw_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_match = raw_str
            if best_score >= 0.9:
                return best_match
            return None

        def _contract_column_norms(contract_obj: Dict[str, Any] | None = None) -> set[str]:
            norms: set[str] = set()
            canonical = contract_obj.get("canonical_columns") if isinstance(contract_obj, dict) else []
            if isinstance(canonical, list):
                for col in canonical:
                    if col:
                        norm_col = _norm(col)
                        if norm_col:
                            norms.add(norm_col)
            for col in column_inventory or []:
                if not col:
                    continue
                norm_col = _norm(col)
                if norm_col:
                    norms.add(norm_col)
            return norms

        def _filter_columns_against_contract(columns: List[str] | None, contract_obj: Dict[str, Any] | None = None) -> List[str]:
            if not columns:
                return []
            allowed = _contract_column_norms(contract_obj)
            if not allowed:
                return [col for col in columns if col]
            filtered: List[str] = []
            for col in columns:
                if not col:
                    continue
                if _norm(col) in allowed:
                    filtered.append(col)
            return filtered

        def _normalize_artifact_schema_payload(raw: Any) -> Dict[str, Dict[str, Any]]:
            normalized: Dict[str, Dict[str, Any]] = {}
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if not key:
                        continue
                    normalized[str(key)] = dict(value) if isinstance(value, dict) else {}
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

        def _parse_summary_kinds(summary_text: str) -> Dict[str, str]:
            kind_map: Dict[str, str] = {}
            if not summary_text:
                return kind_map
            for raw_line in summary_text.splitlines():
                line = raw_line.strip()
                if not line.startswith("-"):
                    continue
                content = line.lstrip("-").strip()
                if ":" not in content:
                    continue
                label, cols = content.split(":", 1)
                label_lower = label.strip().lower()
                cols_list = [c.strip() for c in re.split(r"[;,]", cols) if c.strip()]
                kind = None
                if "date" in label_lower:
                    kind = "datetime"
                elif "numerical" in label_lower or "numeric" in label_lower:
                    kind = "numeric"
                elif "categor" in label_lower or "boolean" in label_lower or "identifier" in label_lower:
                    kind = "categorical"
                if kind:
                    for col in cols_list:
                        kind_map[_norm(col)] = kind
            return kind_map

        def _guess_kind_from_name(name: str) -> str | None:
            if not name:
                return None
            raw = str(name)
            norm_name = _norm(raw)
            if not norm_name:
                return None
            if any(tok in norm_name for tok in ["date", "time", "fecha", "day", "month", "year"]):
                return "datetime"
            if any(tok in norm_name for tok in ["salesrep", "owner", "channel", "sector", "category", "type", "status", "phase", "segment", "email", "phone", "country", "city", "region", "industry", "name"]):
                return "categorical"
            if any(tok in norm_name for tok in ["id", "uuid", "code", "ref"]):
                return "categorical"
            if "%" in raw or any(tok in norm_name for tok in ["pct", "percent", "ratio", "rate", "prob", "score", "amount", "price", "size", "debt", "count", "number", "num", "qty"]):
                return "numeric"
            return None

        def _extract_formula_tokens(formula: str) -> List[str]:
            if not formula:
                return []
            tokens = re.findall(r"[A-Za-z%_][A-Za-z0-9_%]*", formula)
            ignore = {
                "score_nuevo",
                "scorenuevo",
                "score",
                "w",
            }
            cleaned: List[str] = []
            for tok in tokens:
                tok_norm = _norm(tok)
                if not tok_norm:
                    continue
                if tok_norm in ignore:
                    continue
                if tok_norm.startswith("w") and tok_norm[1:].isdigit():
                    continue
                cleaned.append(tok)
            # preserve order, unique
            seen = set()
            unique = []
            for tok in cleaned:
                tok_norm = _norm(tok)
                if tok_norm in seen:
                    continue
                seen.add(tok_norm)
                unique.append(tok)
            return unique

        def _guess_expected_range(name: str) -> List[float] | None:
            if not name:
                return None
            lower = name.lower()
            if "%" in name or "norm" in lower or "score" in lower or "impact" in lower:
                return [0, 1]
            return None

        def _ensure_formula_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            spec = contract.get("spec_extraction")
            if not isinstance(spec, dict):
                return contract
            formula = spec.get("scoring_formula") or ""
            if not isinstance(formula, str) or not formula.strip():
                return contract
            reqs = contract.get("data_requirements", []) or []
            existing = {_norm(r.get("canonical_name") or r.get("name")) for r in reqs if isinstance(r, dict)}
            derived_cols = spec.get("derived_columns") or []
            derived_names = {
                _norm(dc.get("name"))
                for dc in derived_cols
                if isinstance(dc, dict) and dc.get("name")
            }
            tokens = _extract_formula_tokens(formula)
            for tok in tokens:
                tok_norm = _norm(tok)
                if not tok_norm or tok_norm in existing:
                    continue
                source = "derived" if tok_norm in derived_names else "input"
                raw_match = _resolve_exact_header(tok)
                canonical = raw_match or tok
                reqs.append(
                    {
                        "name": tok,
                        "role": "feature",
                        "expected_range": _guess_expected_range(tok),
                        "allowed_null_frac": None,
                        "source": source,
                        "expected_kind": "numeric",
                        "canonical_name": canonical,
                    }
                )
                existing.add(tok_norm)
            contract["data_requirements"] = reqs
            return contract

        def _apply_expected_kind(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            kind_map = _parse_summary_kinds(data_summary)
            reqs = contract.get("data_requirements", []) or []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                if req.get("expected_kind"):
                    continue
                name = req.get("name")
                norm_name = _norm(name) if name else ""
                if norm_name in kind_map:
                    req["expected_kind"] = kind_map[norm_name]
                    continue
                inferred = _guess_kind_from_name(name or "")
                if inferred:
                    req["expected_kind"] = inferred
                    continue
                role = (req.get("role") or "").lower()
                if role in {"percentage", "risk_score", "probability", "ratio"}:
                    req["expected_kind"] = "numeric"
                elif role == "categorical":
                    req["expected_kind"] = "categorical"
                elif role == "date":
                    req["expected_kind"] = "datetime"
                else:
                    req["expected_kind"] = "unknown"
            contract["data_requirements"] = reqs
            return contract

        def _apply_inventory_source(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            inv_norm = {_norm(c) for c in (column_inventory or []) if c is not None}
            reqs = contract.get("data_requirements", []) or []
            updated_reqs = []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("name")
                if not name:
                    continue
                norm_name = _norm(name)
                best_match = None
                best_score = 0.0
                if inv_norm and norm_name:
                    for candidate in inv_norm:
                        score = difflib.SequenceMatcher(None, norm_name, candidate).ratio()
                        if score > best_score:
                            best_score = score
                            best_match = candidate
                if not inv_norm:
                    req["source"] = req.get("source", "input") or "input"
                elif norm_name and (norm_name in inv_norm or best_score >= 0.9):
                    req["source"] = req.get("source", "input") or "input"
                else:
                    req["source"] = "derived"
                updated_reqs.append(req)
            contract["data_requirements"] = updated_reqs
            return contract

        def _ensure_strategy_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            required = strategy.get("required_columns", []) if isinstance(strategy, dict) else []
            if not required:
                return contract
            reqs = contract.get("data_requirements", []) or []
            existing = {_norm(r.get("canonical_name") or r.get("name")) for r in reqs if isinstance(r, dict)}
            for col in required:
                if not col:
                    continue
                norm = _norm(col)
                if not norm or norm in existing:
                    continue
                raw_match = _resolve_exact_header(col)
                canonical = raw_match or _canonicalize_name(col)
                reqs.append(
                    {
                        "name": col,
                        "role": "feature",
                        "expected_range": _guess_expected_range(col),
                        "allowed_null_frac": None,
                        "source": "input",
                        "expected_kind": None,
                        "canonical_name": canonical,
                    }
                )
                existing.add(norm)
            contract["data_requirements"] = reqs
            return contract

        def _attach_canonical_names(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            reqs = contract.get("data_requirements", []) or []
            canonical_cols: List[str] = []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("name")
                if not name:
                    continue
                source = (req.get("source") or "input").lower()
                canonical = req.get("canonical_name")
                if source == "derived":
                    if not canonical:
                        canonical = _canonicalize_name(name)
                        req["canonical_name"] = canonical
                    continue
                raw_match = _resolve_exact_header(canonical or name)
                if raw_match:
                    canonical = raw_match
                    req["canonical_name"] = canonical
                elif not canonical:
                    canonical = _canonicalize_name(name)
                    req["canonical_name"] = canonical
                if canonical:
                    canonical_cols.append(canonical)
            contract["data_requirements"] = reqs
            if canonical_cols:
                contract["canonical_columns"] = canonical_cols
                notes = contract.get("notes_for_engineers")
                if not isinstance(notes, list):
                    notes = []
                note = "Use data_requirements.canonical_name for consistent column references across agents."
                if note not in notes:
                    notes.append(note)
                contract["notes_for_engineers"] = notes
            return contract

        def _propagate_business_alignment(contract: Dict[str, Any]) -> Dict[str, Any]:
            """V4.1: Propagate business_alignment to direct runbook keys, not legacy role_runbooks."""
            if not isinstance(contract, dict):
                return contract
            ba = contract.get("business_alignment")
            if not isinstance(ba, dict):
                return contract
            # V4.1: Use direct ml_engineer_runbook, not role_runbooks
            ml_runbook = contract.get("ml_engineer_runbook")
            if isinstance(ml_runbook, dict):
                ml_runbook["business_alignment"] = ba
                contract["ml_engineer_runbook"] = ml_runbook
            # Also propagate to data_engineer_runbook if needed
            de_runbook = contract.get("data_engineer_runbook")
            if isinstance(de_runbook, dict):
                de_runbook["business_alignment"] = ba
                contract["data_engineer_runbook"] = de_runbook
            return contract

        def _ensure_case_id_requirement(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") or {}
            case_taxonomy = spec.get("case_taxonomy")
            if not isinstance(case_taxonomy, list) or not case_taxonomy:
                return contract
            reqs = contract.get("data_requirements", []) or []
            case_names = {"case_id", "case", "caso", "caseid", "case_id"}
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = (req.get("name") or "").strip()
                if _norm(name) in case_names:
                    return contract
                canonical = (req.get("canonical_name") or "").strip()
                if _norm(canonical) in case_names:
                    return contract
            reqs.append(
                {
                    "name": "Case_ID",
                    "role": "case_id",
                    "expected_range": None,
                    "allowed_null_frac": 0.0,
                    "source": "derived",
                    "expected_kind": "categorical",
                    "canonical_name": "Case_ID",
                }
            )
            contract["data_requirements"] = reqs
            notes = contract.get("notes_for_engineers")
            if not isinstance(notes, list):
                notes = []
            note = "Case taxonomy present; include a case identifier column (e.g., Case_ID) in cleaned outputs."
            if note not in notes:
                notes.append(note)
            contract["notes_for_engineers"] = notes
            return contract

        def _attach_strategy_context(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            strategy_ctx = {
                "title": strategy.get("title") if isinstance(strategy, dict) else None,
                "analysis_type": strategy.get("analysis_type") if isinstance(strategy, dict) else None,
                "hypothesis": strategy.get("hypothesis") if isinstance(strategy, dict) else None,
                "techniques": strategy.get("techniques", []) if isinstance(strategy, dict) else [],
                "required_columns": strategy.get("required_columns", []) if isinstance(strategy, dict) else [],
                "estimated_difficulty": strategy.get("estimated_difficulty") if isinstance(strategy, dict) else None,
            }
            contract["strategy_context"] = strategy_ctx
            runbooks = contract.get("role_runbooks")
            if isinstance(runbooks, dict):
                ml_runbook = runbooks.get("ml_engineer")
                if isinstance(ml_runbook, dict):
                    ml_runbook["strategy_context"] = strategy_ctx
                    runbooks["ml_engineer"] = ml_runbook
                    contract["role_runbooks"] = runbooks
            return contract

        def _detect_canonical_collisions() -> List[tuple[str, List[str]]]:
            names = [c for c in (column_inventory or []) if c is not None]
            buckets: Dict[str, List[str]] = {}
            for name in names:
                canon = _canonicalize_name(name)
                if not canon:
                    continue
                buckets.setdefault(canon, []).append(str(name))
            return [(canon, vals) for canon, vals in buckets.items() if len(vals) > 1]

        def _has_numeric_conversion_risk(risk_items: List[str]) -> bool:
            marker = "Ensure numeric conversion before comparisons/normalization"
            return any(marker in risk for risk in risk_items)

        def _has_canonical_collision_risk(risk_items: List[str]) -> bool:
            marker = "Potential normalization collisions in column names"
            return any(marker in risk for risk in risk_items)

        def _extract_data_risks(contract: Dict[str, Any]) -> List[str]:
            risks: List[str] = []
            summary_text = data_summary or ""
            summary_lower = summary_text.lower()
            numeric_name_tokens = {
                "pct",
                "percent",
                "ratio",
                "rate",
                "prob",
                "probability",
                "score",
                "norm",
                "amount",
                "value",
                "price",
                "cost",
                "revenue",
                "income",
                "importe",
                "monto",
                "saldo",
                "age",
                "days",
                "term",
            }

            def _looks_numeric_name(col_name: str) -> bool:
                if "%" in col_name:
                    return True
                norm_name = _norm(col_name)
                return any(tok in norm_name for tok in numeric_name_tokens)

            def _parse_column_dtypes(text: str) -> List[tuple[str, str]]:
                cols: List[tuple[str, str]] = []
                in_section = False
                for raw_line in text.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    lower = line.lower()
                    if lower.startswith("key columns"):
                        in_section = True
                        continue
                    if in_section and lower.startswith("potential "):
                        break
                    if in_section and lower.startswith("example rows"):
                        break
                    if not in_section:
                        continue
                    if not line.startswith("-"):
                        continue
                    match = re.match(r"-\s*(.+?):\s*([^,]+),", line)
                    if match:
                        col = match.group(1).strip()
                        dtype = match.group(2).strip()
                        cols.append((col, dtype))
                return cols

            # Surface explicit alert/critical lines from steward summary
            for raw_line in summary_text.splitlines():
                line = raw_line.strip()
                lower = line.lower()
                if not line:
                    continue
                if "alert" in lower or "critical" in lower or "warning" in lower:
                    risks.append(line)
            # Sampling warning
            if "sample" in summary_lower and ("5000" in summary_lower or "sampled" in summary_lower):
                risks.append("Summary indicates sampling; verify results on full dataset.")
            # Dialect/encoding hints
            if "delimiter" in summary_lower or "dialect" in summary_lower or "encoding" in summary_lower:
                risks.append("Potential dialect/encoding sensitivity; enforce manifest output_dialect on load.")
            # Variance/constant hints
            if "no variation" in summary_lower or "no variance" in summary_lower or "constant" in summary_lower:
                risks.append("Potential low-variance/constant columns; guard for target/feature variance.")

            # Object dtypes on numeric-looking columns (type conversion risks)
            for col, dtype in _parse_column_dtypes(summary_text):
                dtype_lower = dtype.lower()
                if "object" in dtype_lower or "string" in dtype_lower:
                    if _looks_numeric_name(col):
                        risks.append(
                            f"Column '{col}' appears numeric/percentage but dtype is '{dtype}'. "
                            "Ensure numeric conversion before comparisons/normalization to avoid type errors."
                        )

            collisions = _detect_canonical_collisions()
            if collisions:
                examples = []
                for canon, originals in collisions[:3]:
                    sample = ", ".join(originals[:3])
                    examples.append(f"{canon}: {sample}")
                suffix = "; ".join(examples)
                risks.append(
                    "Potential normalization collisions in column names; ensure column selection is unambiguous "
                    f"after canonicalization (examples: {suffix})."
                )

            inv_norm = {_norm(c) for c in (column_inventory or []) if c is not None}
            missing_inputs: List[str] = []
            derived_needed: List[str] = []
            for req in contract.get("data_requirements", []) or []:
                if not isinstance(req, dict):
                    continue
                name = req.get("name")
                if not name:
                    continue
                source = req.get("source", "input") or "input"
                if source == "input" and inv_norm:
                    if _norm(name) not in inv_norm:
                        missing_inputs.append(name)
                if source == "derived":
                    derived_needed.append(name)
            if missing_inputs:
                risks.append(
                    f"Input requirements not found in header inventory: {missing_inputs}. "
                    "Use normalized mapping after canonicalization; do not fail before mapping."
                )
            if derived_needed:
                risks.append(
                    f"Derived columns required: {derived_needed}. "
                    "Derive after mapping; do not expect in raw input."
                )

            # Deduplicate and cap
            seen = set()
            unique = []
            for r in risks:
                if r not in seen:
                    seen.add(r)
                    unique.append(r)
            return unique[:8]

        def _attach_data_risks(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            risks = _extract_data_risks(contract)
            if risks:
                existing = contract.get("data_risks")
                if not isinstance(existing, list):
                    existing = []
                combined = existing + [r for r in risks if r not in existing]
                contract["data_risks"] = combined
                notes = contract.get("notes_for_engineers")
                if not isinstance(notes, list):
                    notes = []
                for r in risks:
                    note = f"DATA_RISK: {r}"
                    if note not in notes:
                        notes.append(note)
                contract["notes_for_engineers"] = notes
                if _has_numeric_conversion_risk(risks):
                    planner_self_check = contract.get("planner_self_check")
                    if not isinstance(planner_self_check, list):
                        planner_self_check = []
                    msg = (
                        "Flagged numeric-looking object columns; warn DE to convert to numeric before "
                        "comparisons/normalization."
                    )
                    if msg not in planner_self_check:
                        planner_self_check.append(msg)
                    contract["planner_self_check"] = planner_self_check
                if _has_canonical_collision_risk(risks):
                    planner_self_check = contract.get("planner_self_check")
                    if not isinstance(planner_self_check, list):
                        planner_self_check = []
                    msg = (
                        "Detected potential column-name collisions after canonicalization; ensure unambiguous "
                        "selection before validation."
                    )
                    if msg not in planner_self_check:
                        planner_self_check.append(msg)
                    contract["planner_self_check"] = planner_self_check
            return contract

        def _infer_objective_type() -> str:
            objective_text = (business_objective or "").lower()
            strategy_obj = strategy if isinstance(strategy, dict) else {}
            analysis_type = str(strategy_obj.get("analysis_type") or "").lower()
            techniques_text = " ".join([str(t) for t in (strategy_obj.get("techniques") or [])]).lower()
            signal_text = " ".join([objective_text, analysis_type, techniques_text])
            prescriptive_tokens = [
                "optimiz",
                "maximize",
                "minimize",
                "pricing",
                "precio",
                "optimal",
                "optimo",
                "revenue",
                "expected value",
                "recommend",
                "allocation",
                "decision",
                "prescriptive",
                "ranking",
                "scoring",
            ]
            predictive_tokens = [
                "predict",
                "classification",
                "regression",
                "forecast",
                "probability",
                "probabilidad",
                "clasific",
                "conversion",
                "convert",
                "churn",
                "contract",
                "propensity",
                "predictive",
            ]
            causal_tokens = [
                "causal",
                "uplift",
                "impact",
                "intervention",
                "treatment",
            ]
            if any(tok in signal_text for tok in prescriptive_tokens):
                return "prescriptive"
            if any(tok in signal_text for tok in predictive_tokens):
                return "predictive"
            if any(tok in signal_text for tok in causal_tokens):
                return "causal"
            return "descriptive"

        def _safe_column_name(name: str) -> str:
            if not name:
                return ""
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()

        def _recommended_column_name(name: str) -> str:
            if not name:
                return ""
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")

        def _collect_text_blob() -> str:
            parts = []
            if business_objective:
                parts.append(business_objective)
            if data_summary:
                parts.append(data_summary)
            if isinstance(strategy, dict):
                parts.append(json.dumps(strategy, ensure_ascii=True))
            return "\n".join(parts)

        def _infer_segmentation_required() -> bool:
            text = _collect_text_blob().lower()
            tokens = ["segment", "segmentation", "cluster", "clustering", "typology", "tipolog"]
            return any(tok in text for tok in tokens)

        def _estimate_n_rows(summary_text: str) -> int | None:
            profile_path = os.path.join("data", "dataset_profile.json")
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, "r", encoding="utf-8") as f_profile:
                        profile = json.load(f_profile)
                    if isinstance(profile, dict):
                        rows = profile.get("rows") or profile.get("row_count") or profile.get("n_rows")
                        if isinstance(rows, int) and rows >= 0:
                            return rows
                except Exception:
                    pass
            if summary_text:
                match = re.search(r"\brows\s*[:=]\s*(\d+)", summary_text, flags=re.IGNORECASE)
                if match:
                    try:
                        return int(match.group(1))
                    except Exception:
                        return None
            return None

        def _compute_segmentation_constraints(n_rows: int | None) -> Dict[str, Any]:
            n_val = int(n_rows) if isinstance(n_rows, int) and n_rows >= 0 else 0
            max_segments = min(15, max(3, n_val // 20))
            min_segment_size = max(10, n_val // 100)
            if n_val > 0:
                max_segments = min(max_segments, max(2, n_val // max(min_segment_size, 1)))
            preferred_max = min(10, max_segments)
            preferred_min = 2 if preferred_max >= 2 else 1
            return {
                "n_rows_estimate": n_val if n_rows is not None else None,
                "max_segments": int(max_segments),
                "min_segment_size": int(min_segment_size),
                "preferred_k_range": [int(preferred_min), int(preferred_max)],
            }

        def _attach_segmentation_constraints(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            constraints = contract.get("segmentation_constraints")
            if not isinstance(constraints, dict):
                constraints = {}
            n_rows = _estimate_n_rows(data_summary)
            computed = _compute_segmentation_constraints(n_rows)
            for key, value in computed.items():
                if value is None:
                    continue
                constraints[key] = value
            contract["segmentation_constraints"] = constraints
            return contract

        def _extract_summary_candidates(summary_text: str) -> List[str]:
            if not summary_text:
                return []
            candidates: List[str] = []
            label_tokens = ["status", "phase", "stage", "outcome", "result"]
            for raw_line in summary_text.splitlines():
                line = raw_line.strip()
                line_lower = line.lower()
                if not any(tok in line_lower for tok in label_tokens):
                    continue
                if ":" not in line:
                    continue
                _, cols = line.split(":", 1)
                cols_list = [c.strip() for c in re.split(r"[;,]", cols) if c.strip()]
                candidates.extend(cols_list)
            return candidates

        def _status_candidate_score(name: str) -> int:
            if not name:
                return -1
            norm = _norm(name)
            if not norm:
                return -1
            if any(tok in norm for tok in ["prob", "score", "pred"]):
                return -1
            tokens = [
                "status",
                "phase",
                "stage",
                "outcome",
                "result",
                "success",
                "won",
                "win",
                "closed",
                "churn",
                "conversion",
                "convert",
                "approval",
                "approved",
            ]
            score = 0
            for idx, tok in enumerate(tokens):
                if tok in norm:
                    score += (len(tokens) - idx)
            return score

        def _find_status_candidate() -> str | None:
            candidates: List[str] = []
            for raw in column_inventory or []:
                if raw is None:
                    continue
                raw_str = str(raw)
                if _status_candidate_score(raw_str) > 0:
                    candidates.append(raw_str)
            for candidate in _extract_summary_candidates(data_summary):
                resolved = _resolve_exact_header(candidate) or candidate
                if _status_candidate_score(resolved) > 0:
                    candidates.append(resolved)
            best = None
            best_score = -1
            for cand in candidates:
                score = _status_candidate_score(cand)
                if score > best_score:
                    best_score = score
                    best = cand
            return best

        def _parse_label_list(text: str) -> List[str]:
            if not text:
                return []
            parts = [p.strip(" \"'") for p in re.split(r"[;,/|]", text) if p.strip()]
            return [p for p in parts if p]

        def _extract_positive_labels(summary_text: str) -> List[str]:
            if not summary_text:
                return []
            for pattern in [
                r"positive labels?\s*[:=]\s*([^\n]+)",
                r"positive_values?\s*[:=]\s*([^\n]+)",
                r"success labels?\s*[:=]\s*([^\n]+)",
            ]:
                match = re.search(pattern, summary_text, flags=re.IGNORECASE)
                if match:
                    labels = _parse_label_list(match.group(1))
                    if labels:
                        return labels
            return []

        def _normalize_quotes(text: str) -> str:
            if not text:
                return ""
            return (
                text.replace("\u201c", "\"")
                .replace("\u201d", "\"")
                .replace("\u2018", "'")
                .replace("\u2019", "'")
            )

        def _column_mentioned(text: str, column: str) -> bool:
            if not text or not column:
                return False
            return _norm(column) in _norm(text)

        def _column_name_pattern(column: str) -> str:
            if not column:
                return ""
            tokens = re.findall(r"[A-Za-z0-9]+", column)
            if not tokens:
                return re.escape(column)
            return r"[\\s_\\-\\.]*".join([re.escape(tok) for tok in tokens])

        def _extract_positive_labels_from_objective(
            objective_text: str,
            status_col: str | None,
        ) -> tuple[List[str], str | None]:
            if not objective_text or not status_col:
                return [], None
            normalized_text = _normalize_quotes(objective_text)
            resolved_col = _resolve_exact_header(status_col) or status_col
            column_referenced = _column_mentioned(normalized_text, resolved_col)
            col_pattern = _column_name_pattern(resolved_col)
            if not col_pattern:
                return [], None

            label_pattern = r"(?P<label>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_%\\.-]+)"
            contains_patterns = []
            equals_patterns = []
            if column_referenced:
                contains_patterns.extend(
                    [
                        rf"{col_pattern}[^\n\r]{{0,200}}?(contiene|contains)[^\n\r]{{0,80}}?{label_pattern}",
                        rf"(contiene|contains)[^\n\r]{{0,80}}?{label_pattern}[^\n\r]{{0,200}}?{col_pattern}",
                    ]
                )
                equals_patterns.extend(
                    [
                        rf"{col_pattern}[^\n\r]{{0,200}}?(==|=|equals|equal to|es|igual)[^\n\r]{{0,80}}?{label_pattern}",
                        rf"(==|=|equals|equal to|es|igual)[^\n\r]{{0,80}}?{label_pattern}[^\n\r]{{0,200}}?{col_pattern}",
                    ]
                )

            phase_tokens = r"(status|phase|stage|estado|fase)"
            contains_patterns.append(
                rf"{phase_tokens}[^\n\r]{{0,60}}?(contiene|contains)[^\n\r]{{0,80}}?{label_pattern}"
            )
            equals_patterns.append(
                rf"{phase_tokens}[^\n\r]{{0,60}}?(==|=|equals|equal to|es|igual)[^\n\r]{{0,80}}?{label_pattern}"
            )
            equals_patterns.append(
                rf"{label_pattern}(?:\s+|\s*[-_/:,]+\s*){phase_tokens}"
            )

            for pattern in contains_patterns:
                match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
                if match:
                    labels = _parse_label_list(match.group("label"))
                    if labels:
                        return labels, "contains"

            for pattern in equals_patterns:
                match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
                if match:
                    labels = _parse_label_list(match.group("label"))
                    if labels:
                        return labels, "equals"

            labels = _extract_positive_labels(normalized_text)
            if labels:
                return labels, None
            return [], None

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

        def _default_deliverable_description(path: str, kind: str) -> str:
            known = {
                "data/cleaned_data.csv": "Cleaned dataset used for downstream modeling.",
                "data/metrics.json": "Model metrics and validation summary.",
                "data/weights.json": "Feature weights or scoring coefficients.",
                "data/case_summary.csv": "Per-case scoring summary.",
                "data/case_alignment_report.json": "Case alignment QA metrics.",
                "data/scored_rows.csv": "Row-level scores and key features.",
                "data/alignment_check.json": "Alignment check results for contract requirements.",
                "static/plots/*.png": "Required diagnostic plots.",
                "reports/recommendations_preview.json": "Illustrative recommendation examples for the executive report.",
            }
            if path in known:
                return known[path]
            if kind == "plot":
                return "Diagnostic plots required by the contract."
            if kind == "metrics":
                return "Metrics artifact required by the contract."
            if kind == "weights":
                return "Weights or scoring artifact required by the contract."
            return "Requested deliverable."

        def _build_deliverable(
            path: str,
            required: bool = True,
            kind: str | None = None,
            description: str | None = None,
            deliverable_id: str | None = None,
        ) -> Dict[str, Any]:
            if not path:
                return {}
            kind_val = kind or _infer_deliverable_kind(path)
            desc_val = description or _default_deliverable_description(path, kind_val)
            deliverable_id = deliverable_id or _deliverable_id_from_path(path)
            return {
                "id": deliverable_id,
                "path": path,
                "required": bool(required),
                "kind": kind_val,
                "description": desc_val,
            }

        def _normalize_deliverables(
            raw: Any,
            default_required: bool = True,
            required_paths: set[str] | None = None,
        ) -> List[Dict[str, Any]]:
            if not raw or not isinstance(raw, list):
                return []
            required_paths = {str(p) for p in (required_paths or set()) if p}
            normalized: List[Dict[str, Any]] = []
            for item in raw:
                if isinstance(item, str):
                    path = item
                    required = path in required_paths if required_paths else default_required
                    deliverable = _build_deliverable(path, required=required)
                    if deliverable:
                        normalized.append(deliverable)
                    continue
                if not isinstance(item, dict):
                    continue
                path = item.get("path") or item.get("output") or item.get("artifact")
                if not path:
                    continue
                required = item.get("required")
                if required is None:
                    required = path in required_paths if required_paths else default_required
                deliverable = _build_deliverable(
                    path=path,
                    required=bool(required),
                    kind=item.get("kind"),
                    description=item.get("description"),
                    deliverable_id=item.get("id"),
                )
                if deliverable:
                    normalized.append(deliverable)
            return normalized

        def _merge_deliverables(
            base: List[Dict[str, Any]],
            overrides: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            merged = list(base)
            by_path = {item.get("path"): idx for idx, item in enumerate(merged) if item.get("path")}
            for item in overrides:
                path = item.get("path")
                if not path:
                    continue
                if path in by_path:
                    existing = merged[by_path[path]]
                    for key in ("id", "kind", "description"):
                        if item.get(key):
                            existing[key] = item.get(key)
                    if "required" in item and item.get("required") is not None:
                        existing["required"] = bool(item.get("required"))
                    merged[by_path[path]] = existing
                else:
                    merged.append(item)
                    by_path[path] = len(merged) - 1
            return merged

        def _ensure_unique_deliverable_ids(deliverables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen: set[str] = set()
            for item in deliverables:
                base_id = item.get("id") or _deliverable_id_from_path(item.get("path") or "")
                candidate = base_id
                suffix = 2
                while candidate in seen:
                    candidate = f"{base_id}_{suffix}"
                    suffix += 1
                item["id"] = candidate
                seen.add(candidate)
            return deliverables

        def _derive_deliverables(
            objective_type: str,
            strategy_obj: Dict[str, Any],
            spec_obj: Dict[str, Any],
        ) -> List[Dict[str, Any]]:
            """
            Context-aware deliverable derivation based on objective_type.

            DYNAMIC DELIVERABLES POLICY:
            - descriptive: metrics.json and scored_rows.csv are OPTIONAL (no model training)
            - predictive/causal: metrics.json REQUIRED, scored_rows.csv REQUIRED
            - prescriptive: metrics.json REQUIRED, scored_rows.csv REQUIRED, plus optimization artifacts
            """
            deliverables: List[Dict[str, Any]] = []

            def _add(path: str, required: bool = True, kind: str | None = None, description: str | None = None) -> None:
                item = _build_deliverable(path, required=required, kind=kind, description=description)
                if item:
                    deliverables.append(item)

            # Determine if this objective involves model training
            involves_model_training = objective_type in ("predictive", "prescriptive", "causal")

            # Core deliverable: cleaned_data.csv is always required
            _add("data/cleaned_data.csv", True, "dataset", "Cleaned dataset used for downstream analysis.")

            # CONTEXT-AWARE: metrics.json only required if model training is involved
            if involves_model_training:
                _add("data/metrics.json", True, "metrics", "Model metrics and validation summary.")
            else:
                _add("data/metrics.json", False, "metrics", "Optional metrics for descriptive analysis.")

            _add("static/plots/*.png", False, "plot", "Optional diagnostic plots.")
            _add("data/predictions.csv", False, "predictions", "Optional predictions output.")
            _add("data/feature_importances.json", False, "feature_importances", "Optional feature importance output.")
            _add("data/error_analysis.json", False, "error_analysis", "Optional error analysis output.")
            _add(
                "reports/recommendations_preview.json",
                False,
                "report",
                "Optional illustrative recommendation preview for executive reporting.",
            )

            target_type = str(spec_obj.get("target_type") or "").lower()
            scoring_formula = spec_obj.get("scoring_formula")
            analysis_type = str(strategy_obj.get("analysis_type") or "").lower()
            techniques_text = " ".join([str(t) for t in (strategy_obj.get("techniques") or [])]).lower()
            signal_text = " ".join([analysis_type, techniques_text, target_type, str(scoring_formula or "").lower()])

            # CONTEXT-AWARE: scored_rows.csv only required for scoring/optimization objectives
            if any(tok in signal_text for tok in ["ranking", "scoring", "weight", "weights", "optimization", "optimiz", "priorit"]):
                _add("data/weights.json", False, "weights", "Optional weights artifact for legacy consumers.")
                _add("data/case_summary.csv", False, "dataset", "Optional legacy case summary output.")
                # scored_rows required for prescriptive, optional for descriptive
                _add("data/scored_rows.csv", involves_model_training, "predictions", "Scored rows output.")
                _add("data/case_alignment_report.json", False, "report", "Optional legacy alignment report.")
            elif involves_model_training:
                # Predictive/causal without explicit scoring: scored_rows still required
                _add("data/scored_rows.csv", True, "predictions", "Model predictions output.")
            else:
                # Descriptive: scored_rows is optional
                _add("data/scored_rows.csv", False, "predictions", "Optional scored rows for descriptive analysis.")

            return deliverables

        def _apply_deliverables(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            spec = contract.get("spec_extraction")
            if not isinstance(spec, dict):
                spec = {}
            legacy_required = contract.get("required_outputs", []) or []
            derived = _derive_deliverables(_infer_objective_type(), strategy or {}, spec)
            legacy = _normalize_deliverables(legacy_required, default_required=True)
            existing = _normalize_deliverables(spec.get("deliverables"), default_required=True, required_paths=set(legacy_required))
            deliverables = _merge_deliverables(derived, legacy)
            deliverables = _merge_deliverables(deliverables, existing)
            deliverables = _ensure_unique_deliverable_ids(deliverables)
            spec["deliverables"] = deliverables
            contract["spec_extraction"] = spec
            
            # Build required_outputs from deliverables
            req_outputs = [item["path"] for item in deliverables if item.get("required")]

            # CONTEXT-AWARE: Only force core ML outputs for objectives that involve model training
            # For descriptive objectives, scored_rows.csv and metrics.json are optional
            objective_type = _infer_objective_type()
            involves_model_training = objective_type in ("predictive", "prescriptive", "causal")

            # alignment_check.json is always required (documents what was done)
            if "data/alignment_check.json" not in req_outputs:
                req_outputs.append("data/alignment_check.json")

            # scored_rows.csv and metrics.json only forced for model-training objectives
            if involves_model_training:
                if "data/scored_rows.csv" not in req_outputs:
                    req_outputs.append("data/scored_rows.csv")
                if "data/metrics.json" not in req_outputs:
                    req_outputs.append("data/metrics.json")
            else:
                print(f"DYNAMIC_DELIVERABLES: objective_type={objective_type}, scored_rows.csv and metrics.json are OPTIONAL")
            
            # Normalize paths: ensure data/ prefix
            def _normalize_path(p: str) -> str:
                known = ["metrics.json", "alignment_check.json", "scored_rows.csv", "cleaned_data.csv"]
                import os
                base = os.path.basename(p)
                if base in known and not p.startswith("data/"):
                    return f"data/{base}"
                return p
            
            contract["required_outputs"] = [_normalize_path(p) for p in req_outputs]

            artifact_reqs = contract.get("artifact_requirements", {})
            if isinstance(artifact_reqs, dict):
                visual_reqs = artifact_reqs.get("visual_requirements")
                if isinstance(visual_reqs, dict):
                    outputs_dir = visual_reqs.get("outputs_dir") or "static/plots"
                    items = visual_reqs.get("items") if isinstance(visual_reqs.get("items"), list) else []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        expected = item.get("expected_filename")
                        if not expected:
                            continue
                        plot_path = (
                            expected
                            if os.path.isabs(expected)
                            else os.path.normpath(os.path.join(outputs_dir, expected))
                        )
                        if plot_path not in contract["required_outputs"]:
                            contract["required_outputs"].append(plot_path)

            return contract

        def _merge_unique(values: List[str], extras: List[str]) -> List[str]:
            seen: set[str] = set()
            out: List[str] = []
            for item in values + extras:
                if not item:
                    continue
                text = str(item)
                if text in seen:
                    continue
                seen.add(text)
                out.append(text)
            return out

        def _has_deliverable(contract: Dict[str, Any], path: str) -> bool:
            if not isinstance(contract, dict):
                return False
            if path in (contract.get("required_outputs") or []):
                return True
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            deliverables = spec.get("deliverables")
            if isinstance(deliverables, list):
                for item in deliverables:
                    if isinstance(item, dict) and item.get("path") == path:
                        return True
                    if isinstance(item, str) and item == path:
                        return True
            return False

        def _build_scored_rows_schema(contract: Dict[str, Any]) -> Dict[str, Any] | None:
            if not _has_deliverable(contract, "data/scored_rows.csv"):
                return None
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            derived_cols: List[str] = []
            required_cols: List[str] = []
            derived = spec.get("derived_columns")
            if isinstance(derived, list):
                for entry in derived:
                    if isinstance(entry, dict):
                        name = entry.get("name") or entry.get("canonical_name")
                    elif isinstance(entry, str):
                        name = entry
                    else:
                        name = None
                    if name:
                        derived_cols.append(str(name))

            canonical_cols = contract.get("canonical_columns") if isinstance(contract.get("canonical_columns"), list) else []
            required_cols.extend([str(c) for c in canonical_cols if c])

            target_name = _target_name_from_contract(contract, derived if isinstance(derived, list) else [])
            if target_name:
                required_cols.append(str(target_name))
                safe_target = _safe_column_name(target_name)
                if safe_target:
                    required_cols.append(f"pred_{safe_target}")
                else:
                    required_cols.append("prediction")
                if _norm(target_name) in {"issuccess", "success"}:
                    required_cols.append("pred_prob_success")
            else:
                required_cols.append("prediction")

            if _infer_segmentation_required():
                required_cols.append("cluster_id")

            decision_vars = contract.get("decision_variables") or []
            if isinstance(decision_vars, list):
                for var in decision_vars:
                    if not var:
                        continue
                    safe_var = _recommended_column_name(var)
                    if safe_var:
                        required_cols.append(f"recommended_{safe_var}")
                if decision_vars:
                    required_cols.append("expected_value_at_recommendation")

            objective_type = _infer_objective_type()
            allowed_patterns = [
                r".*_probability$",
                r".*_score$",
                r".*_rank$",
                r".*_segment$",
                r"^segment_.*",
                r"^typology_.*",
                r".*_cluster$",
                r".*_group$",
            ]
            if objective_type in {"predictive", "prescriptive", "forecasting", "ranking"}:
                allowed_patterns.append(r"^pred(icted)?_.*")

            decision_vars = contract.get("decision_variables") or []
            decision_context = bool(decision_vars)
            if not decision_context:
                combined_text = "\n".join(
                    [
                        contract.get("business_objective") or business_objective or "",
                        data_summary or "",
                        json.dumps(strategy, ensure_ascii=True) if isinstance(strategy, dict) else "",
                    ]
                )
                decision_context = _detect_decision_context(combined_text)

            if decision_context:
                allowed_patterns.extend(
                    [
                        r"^(expected|optimal|recommended)_.*(revenue|value|price|profit|margin|cost).*",
                        r"^recommended_.*",
                        r"^expected_value_at_recommendation$",
                    ]
                )
            extra_patterns = ["^recommended_.*", "^expected_.*"]
            for pattern in extra_patterns:
                if pattern not in allowed_patterns:
                    allowed_patterns.append(pattern)

            return {
                "rowcount": "match_cleaned",
                "min_overlap": 1,
                "required_columns": _merge_unique(required_cols, []),
                "allowed_extra_columns": derived_cols,
                "allowed_name_patterns": allowed_patterns,
            }

        def _pattern_name(name: str) -> str:
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).lower()).strip("_")

        def _sanitize_allowed_extras(
            extras: List[str],
            patterns: List[str],
            allowlist: set[str],
            max_count: int = 30,
        ) -> List[str]:
            if not extras:
                return []
            allowed = []
            seen = set()
            norm_allowlist = {_norm(item) for item in allowlist if item}
            pattern_list = [str(pat) for pat in patterns if isinstance(pat, str) and pat.strip()]
            for item in extras:
                if not item:
                    continue
                raw = str(item)
                norm = _norm(raw)
                if not norm or norm in seen:
                    continue
                if norm in norm_allowlist:
                    allowed.append(raw)
                    seen.add(norm)
                else:
                    target = _pattern_name(raw)
                    matched = False
                    for pattern in pattern_list:
                        try:
                            if re.search(pattern, target):
                                matched = True
                                break
                        except re.error:
                            continue
                    if matched:
                        allowed.append(raw)
                        seen.add(norm)
                if len(allowed) >= max_count:
                    break
            return allowed

        def _apply_artifact_schemas(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            schemas = _normalize_artifact_schema_payload(contract.get("artifact_schemas"))
            spec_schemas = _normalize_artifact_schema_payload(spec.get("artifact_schemas"))
            for path, payload in spec_schemas.items():
                if path not in schemas:
                    schemas[path] = payload
            scored_schema = _build_scored_rows_schema(contract)
            if scored_schema:
                existing = schemas.get("data/scored_rows.csv")
                if not isinstance(existing, dict):
                    existing = {}
                merged = dict(existing)
                for key in ("rowcount", "min_overlap"):
                    if key not in merged and key in scored_schema:
                        merged[key] = scored_schema[key]
                if scored_schema.get("required_columns") and not merged.get("required_columns"):
                    merged["required_columns"] = scored_schema.get("required_columns")
                merged["allowed_extra_columns"] = _merge_unique(
                    scored_schema.get("allowed_extra_columns", []),
                    merged.get("allowed_extra_columns", []) or [],
                )
                merged["allowed_name_patterns"] = _merge_unique(
                    scored_schema.get("allowed_name_patterns", []),
                    merged.get("allowed_name_patterns", []) or [],
                )
                extra_patterns = ["^recommended_.*", "^expected_.*"]
                for pattern in extra_patterns:
                    if pattern not in merged.get("allowed_name_patterns", []):
                        merged.setdefault("allowed_name_patterns", []).append(pattern)
                base_allowlist = {
                    "is_success",
                    "success_probability",
                    "client_segment",
                    "cluster_id",
                    "expected_value_at_recommendation",
                }
                base_allowlist.update({str(c) for c in (merged.get("allowed_extra_columns") or []) if c})
                merged["allowed_extra_columns"] = _sanitize_allowed_extras(
                    merged.get("allowed_extra_columns", []) or [],
                    merged.get("allowed_name_patterns", []) or [],
                    base_allowlist,
                    max_count=30,
                )
                if merged.get("allowed_name_patterns"):
                    merged["allowed_name_patterns"] = _merge_unique(
                        merged["allowed_name_patterns"],
                        [],
                    )
                schemas["data/scored_rows.csv"] = merged
            contract["artifact_schemas"] = schemas
            return contract

        def _ensure_spec_extraction(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            spec = contract.get("spec_extraction")
            if not isinstance(spec, dict):
                spec = {}
            spec["derived_columns"] = spec.get("derived_columns") if isinstance(spec.get("derived_columns"), list) else []
            spec["case_taxonomy"] = spec.get("case_taxonomy") if isinstance(spec.get("case_taxonomy"), list) else []
            spec["constraints"] = spec.get("constraints") if isinstance(spec.get("constraints"), list) else []
            spec["deliverables"] = spec.get("deliverables") if isinstance(spec.get("deliverables"), list) else []
            spec["scoring_formula"] = spec.get("scoring_formula") if isinstance(spec.get("scoring_formula"), str) else None
            spec["target_type"] = spec.get("target_type") if isinstance(spec.get("target_type"), str) else None
            spec["leakage_policy"] = spec.get("leakage_policy") if isinstance(spec.get("leakage_policy"), str) else None
            if not isinstance(spec.get("leakage_policy_detail"), dict):
                spec["leakage_policy_detail"] = {}
            contract["spec_extraction"] = spec
            planner_self_check = contract.get("planner_self_check")
            if not isinstance(planner_self_check, list):
                contract["planner_self_check"] = []
            return contract

        def _normalize_derived_columns(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
            derived = spec.get("derived_columns")
            normalized: List[Dict[str, Any]] = []
            if isinstance(derived, list):
                for entry in derived:
                    if isinstance(entry, dict):
                        normalized.append(entry)
                    elif isinstance(entry, str):
                        normalized.append({"name": entry})
            spec["derived_columns"] = normalized
            return normalized

        def _ensure_requirement(
            reqs: List[Dict[str, Any]],
            name: str,
            role: str,
            expected_kind: str | None = None,
            source: str = "derived",
            derived_owner: str = "ml_engineer",
        ) -> None:
            if not name:
                return
            norm_name = _norm(name)
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                existing = req.get("canonical_name") or req.get("name")
                if existing and _norm(existing) == norm_name:
                    if not req.get("source"):
                        req["source"] = source
                    if role and not req.get("role"):
                        req["role"] = role
                    if expected_kind and not req.get("expected_kind"):
                        req["expected_kind"] = expected_kind
                    if derived_owner and not req.get("derived_owner"):
                        req["derived_owner"] = derived_owner
                    return
            payload = {
                "name": name,
                "role": role,
                "expected_range": None,
                "allowed_null_frac": None,
                "source": source,
                "expected_kind": expected_kind or "unknown",
                "canonical_name": name,
                "derived_owner": derived_owner,
            }
            reqs.append(payload)

        def _column_candidate_from_value(value: Any) -> str | None:
            if isinstance(value, str):
                parsed = parse_derive_from_expression(value)
                candidate = parsed.get("column") if parsed else None
                return candidate or value
            if isinstance(value, dict):
                for key in ("column", "source_column", "base_column", "from_column"):
                    val = value.get(key)
                    if isinstance(val, str):
                        return val
                return None
            return None

        def _validate_column(column: str | None, allowed_norms: set[str]) -> str | None:
            if not column:
                return None
            resolved = _resolve_exact_header(column) or column
            normed = _norm(resolved)
            if not normed:
                return None
            if allowed_norms and normed not in allowed_norms:
                return None
            return resolved

        def _extract_explicit_is_success(derived_entries: List[Dict[str, Any]], allowed_norms: set[str]) -> tuple[str | None, Dict[str, Any] | None]:
            for entry in derived_entries:
                if not isinstance(entry, dict):
                    continue
                role = str(entry.get("role") or "").lower()
                if "target" not in role and "label" not in role:
                    continue
                name = entry.get("canonical_name") or entry.get("name")
                if not name or _norm(name) not in {"issuccess", "success"}:
                    continue
                candidates: List[str | None] = []
                for key in ("column", "derived_from", "source_column", "base_column", "from_column"):
                    candidates.append(_column_candidate_from_value(entry.get(key)))
                depends = entry.get("depends_on")
                if isinstance(depends, list):
                    for item in depends:
                        if isinstance(item, str):
                            candidates.append(item)
                for candidate in candidates:
                    resolved = _validate_column(candidate, allowed_norms)
                    if resolved:
                        return resolved, entry
            return None, None

        def _target_name_from_contract(contract: Dict[str, Any], derived: List[Dict[str, Any]]) -> str | None:
            reqs = contract.get("data_requirements", []) or []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                role = str(req.get("role") or "").lower()
                if "target_source" in role:
                    continue
                if "target" in role or "label" in role:
                    return req.get("canonical_name") or req.get("name")
            for entry in derived:
                role = str(entry.get("role") or "").lower()
                if "target" in role or "label" in role:
                    return entry.get("name") or entry.get("canonical_name")
            return None

        def _ensure_derived_columns(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            derived = _normalize_derived_columns(spec)
            derived_keys = {_norm(entry.get("name") or entry.get("canonical_name") or "") for entry in derived}
            reqs = contract.get("data_requirements", []) or []

            objective_type = _infer_objective_type()
            target_req = None
            target_req_name = None
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                role = str(req.get("role") or "").lower()
                if "target" in role or "label" in role:
                    target_req = req
                    target_req_name = req.get("canonical_name") or req.get("name")
                    break
            target_in_inventory = bool(_resolve_exact_header(target_req_name)) if target_req_name else False
            target_role = str(target_req.get("role") or "").lower() if target_req else ""
            target_source_like = any(tok in target_role for tok in ["target_source", "outcome", "status", "phase"])
            target_is_input = (
                target_req
                and str(target_req.get("source") or "input").lower() == "input"
                and target_in_inventory
                and not target_source_like
            )
            allowed_norms = _contract_column_norms(contract)
            explicit_status_column, _ = _extract_explicit_is_success(derived, allowed_norms)
            status_source_col = explicit_status_column
            target_explicit = bool(explicit_status_column) or target_is_input
            if objective_type in {"predictive", "prescriptive", "forecasting"} and not target_explicit:
                status_col = _find_status_candidate()
                status_source_col = status_col
                if status_col:
                    resolved_status = _resolve_exact_header(status_col) or status_col
                    allowed_norms = _contract_column_norms(contract)
                    status_norm = _norm(resolved_status) if resolved_status else ""
                    if not resolved_status or (allowed_norms and status_norm not in allowed_norms):
                        resolved_status = None
                    if resolved_status:
                        status_source_col = resolved_status
                        positive_labels, rule_hint = _extract_positive_labels_from_objective(business_objective, resolved_status)
                        if target_req and str(target_req.get("source") or "").lower() == "derived":
                            target_req["name"] = "is_success"
                            target_req["canonical_name"] = "is_success"
                            target_req["role"] = "target"
                            if not target_req.get("expected_kind"):
                                target_req["expected_kind"] = "categorical"
                            if not target_req.get("derived_owner"):
                                target_req["derived_owner"] = "ml_engineer"
                            target_req_name = "is_success"
                        rule = "1 if status in positive_labels else 0"
                        if rule_hint == "contains":
                            rule = "1 if status contains positive_labels else 0"
                        entry = {
                            "name": "is_success",
                            "canonical_name": "is_success",
                            "role": "target",
                            "dtype": "boolean",
                            "derived_from": resolved_status,
                            "column": resolved_status,
                            "rule": rule,
                            "positive_values": positive_labels,
                        }
                        derived = [
                            item
                            for item in derived
                            if _norm(item.get("name") or item.get("canonical_name") or "") != _norm("is_success")
                        ]
                        derived.append(entry)
                        derived_keys.add(_norm("is_success"))
                        _ensure_requirement(reqs, "is_success", "target", expected_kind="categorical")
                        if not positive_labels:
                            checklist = contract.get("compliance_checklist")
                            if not isinstance(checklist, list):
                                checklist = []
                            note = (
                                "Infer positive_labels for is_success from unique values of the status column "
                                f"('{resolved_status}') before training."
                            )
                            if note not in checklist:
                                checklist.append(note)
                            contract["compliance_checklist"] = checklist

            if _infer_segmentation_required() and _norm("cluster_id") not in derived_keys:
                derived.append(
                    {
                        "name": "cluster_id",
                        "canonical_name": "cluster_id",
                        "role": "segment",
                        "dtype": "integer",
                        "derived_from": "pre_decision_features",
                        "rule": "Cluster rows using only pre-decision features.",
                    }
                )
                derived_keys.add(_norm("cluster_id"))
                _ensure_requirement(reqs, "cluster_id", "segment", expected_kind="categorical")

            status_col = status_source_col or _find_status_candidate()
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                role = str(req.get("role") or "").lower()
                if "target" not in role and "label" not in role:
                    continue
                if str(req.get("source") or "input").lower() != "derived":
                    continue
                name = req.get("canonical_name") or req.get("name")
                if not name:
                    continue
                norm_name = _norm(name)
                if norm_name in derived_keys:
                    continue
                dtype = "boolean" if name.lower().startswith(("is_", "has_")) or "success" in name.lower() else "numeric"
                entry = {
                    "name": name,
                    "canonical_name": name,
                    "role": "target",
                    "dtype": dtype,
                    "derived_from": status_col or "source_status",
                    "rule": "Derive target per contract instructions.",
                }
                if status_col and name.lower() == "is_success":
                    entry["rule"] = "1 if status in positive_labels else 0"
                derived.append(entry)
                derived_keys.add(norm_name)

            decision_vars = contract.get("decision_variables") or []
            if isinstance(decision_vars, list):
                for var in decision_vars:
                    if not var:
                        continue
                    safe_var = _safe_column_name(var)
                    if not safe_var:
                        continue
                    rec_name = f"recommended_{safe_var}"
                    if _norm(rec_name) not in derived_keys:
                        derived.append(
                            {
                                "name": rec_name,
                                "canonical_name": rec_name,
                                "role": "recommendation",
                                "dtype": "float",
                                "derived_from": str(var),
                                "rule": "Optimize expected value over decision variable.",
                            }
                        )
                        derived_keys.add(_norm(rec_name))
                        _ensure_requirement(reqs, rec_name, "recommendation", expected_kind="numeric")
                if decision_vars:
                    ev_name = "expected_value_at_recommendation"
                    if _norm(ev_name) not in derived_keys:
                        derived.append(
                            {
                                "name": ev_name,
                                "canonical_name": ev_name,
                                "role": "expected_value",
                                "dtype": "float",
                                "derived_from": "recommended_decision",
                                "rule": "Expected value at the recommended decision.",
                            }
                        )
                        derived_keys.add(_norm(ev_name))
                        _ensure_requirement(reqs, ev_name, "expected_value", expected_kind="numeric")

            spec["derived_columns"] = derived
            contract["spec_extraction"] = spec
            contract["data_requirements"] = reqs
            return contract

        def _refresh_canonical_columns(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            reqs = contract.get("data_requirements", []) or []
            derived = []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                source = (req.get("source") or "input").lower()
                if source == "derived":
                    continue
                name = req.get("canonical_name") or req.get("name")
                if name:
                    derived.append(str(name))
            existing = contract.get("canonical_columns") if isinstance(contract.get("canonical_columns"), list) else []
            input_norms = {
                _norm(req.get("canonical_name") or req.get("name") or "")
                for req in reqs
                if isinstance(req, dict) and str(req.get("source") or "input").lower() == "input"
            }
            existing = [col for col in existing if _norm(col) in input_norms]
            combined = []
            seen = set()
            for col in list(existing) + derived:
                norm = _norm(col)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                combined.append(col)
            if combined:
                contract["canonical_columns"] = combined
            return contract

        def _infer_availability(name: str, role: str | None, decision_vars: List[str]) -> str:
            norm = _norm(name)
            role_lower = (role or "").lower()
            decision_norms = {_norm(v) for v in decision_vars if v}
            if norm in decision_norms or "decision" in role_lower:
                return "decision"
            if any(tok in norm for tok in ["pred", "prob", "score", "recommend", "optimal", "expectedvalue"]):
                return "post_decision-audit_only"
            if any(tok in norm for tok in ["post", "after"]):
                return "post_decision-audit_only"
            if "target" in role_lower or "label" in role_lower:
                return "outcome"
            if any(tok in norm for tok in ["status", "phase", "stage", "outcome", "result", "success", "churn", "conversion"]):
                return "outcome"
            if any(tok in norm for tok in ["segment", "cluster", "group"]):
                return "pre_decision"
            return "pre_decision"

        def _ensure_feature_availability(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            canonical = contract.get("canonical_columns") if isinstance(contract.get("canonical_columns"), list) else []
            reqs = contract.get("data_requirements", []) or []
            availability = contract.get("feature_availability")
            if not isinstance(availability, list):
                availability = []
            by_norm = { _norm(item.get("column")): item for item in availability if isinstance(item, dict) and item.get("column") }
            decision_vars = contract.get("decision_variables") or []
            role_map = {}
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("canonical_name") or req.get("name")
                if name:
                    role_map[_norm(name)] = req.get("role")
            for col in canonical:
                norm = _norm(col)
                if not norm:
                    continue
                entry = by_norm.get(norm)
                role = role_map.get(norm)
                availability_label = _infer_availability(col, role, decision_vars if isinstance(decision_vars, list) else [])
                if entry:
                    if not entry.get("availability"):
                        entry["availability"] = availability_label
                    if not entry.get("rationale"):
                        entry["rationale"] = "Availability inferred from contract context."
                else:
                    availability.append(
                        {
                            "column": col,
                            "availability": availability_label,
                            "rationale": "Availability inferred from contract context.",
                        }
                    )
            contract["feature_availability"] = availability
            # V4.1: availability_summary removed - no longer generated
            return contract

        def _attach_leakage_policy_detail(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            feature_availability = contract.get("feature_availability") or []
            audit_only: List[str] = []
            forbidden: List[str] = []
            for item in feature_availability:
                if not isinstance(item, dict):
                    continue
                col = item.get("column")
                availability = str(item.get("availability") or "").lower()
                if not col:
                    continue
                if "post" in availability:
                    audit_only.append(col)
                    forbidden.append(col)
                if availability == "outcome":
                    forbidden.append(col)
            if isinstance(strategy, dict):
                for key in ("leakage_risk", "leakage_features", "leakage_columns", "leakage_risk_columns"):
                    vals = strategy.get(key)
                    if isinstance(vals, list):
                        for col in vals:
                            if col:
                                audit_only.append(str(col))
                                forbidden.append(str(col))
            reqs = contract.get("data_requirements", []) or []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("canonical_name") or req.get("name")
                role = str(req.get("role") or "").lower()
                if name and ("target" in role or "label" in role):
                    forbidden.append(name)
            canonical = contract.get("canonical_columns") if isinstance(contract.get("canonical_columns"), list) else []
            for col in canonical:
                norm = _norm(col)
                if any(tok in norm for tok in ["prob", "score", "pred"]):
                    audit_only.append(col)
                    forbidden.append(col)
            detail = {
                "audit_only": list(dict.fromkeys([c for c in audit_only if c])),
                "forbidden_as_feature": list(dict.fromkeys([c for c in forbidden if c])),
            }
            if detail["audit_only"] or detail["forbidden_as_feature"]:
                spec["leakage_policy_detail"] = detail
                contract["spec_extraction"] = spec
            return contract

        def _complete_contract_inference(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            contract = _ensure_derived_columns(contract)
            contract = _refresh_canonical_columns(contract)
            contract = _ensure_feature_availability(contract)
            contract = _attach_leakage_policy_detail(contract)
            return contract

        def _attach_spec_extraction_issues(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            issues = validate_spec_extraction_structure(contract)
            if not issues:
                return contract
            contract["spec_extraction_issues"] = issues
            notes = contract.get("notes_for_engineers")
            if not isinstance(notes, list):
                notes = []
            for issue in issues:
                note = f"SPEC_EXTRACTION_ISSUE: {issue}"
                if note not in notes:
                    notes.append(note)
            contract["notes_for_engineers"] = notes
            return contract

        def _build_feature_semantics(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
            if not isinstance(contract, dict):
                return []
            reqs = contract.get("data_requirements", []) or []
            semantics: List[Dict[str, Any]] = []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("canonical_name") or req.get("name")
                if not name:
                    continue
                role = req.get("role")
                kind = req.get("expected_kind")
                norm_name = _norm(name)
                meaning = None
                expectation = None
                risk = None

                if any(tok in norm_name for tok in ["amount", "price", "value", "revenue", "importe", "monto"]):
                    meaning = "monetary value of the contract or deal size"
                    expectation = "larger values typically indicate higher revenue; treat as outcome or pricing target"
                elif any(tok in norm_name for tok in ["size", "turnover", "facturacion", "ingresos"]):
                    meaning = "client scale or capacity (turnover / size)"
                    expectation = "larger clients may support higher contract values; conversion impact may be non-linear"
                elif any(tok in norm_name for tok in ["debt", "debtor", "risk", "riim"]):
                    meaning = "risk or exposure proxy tied to client behavior"
                    expectation = "higher risk may reduce conversion or price tolerance"
                elif any(tok in norm_name for tok in ["sector", "industry"]):
                    meaning = "industry segment describing client context"
                    expectation = "segment effects are categorical; compare within sector"
                elif any(tok in norm_name for tok in ["phase", "status", "contract"]):
                    meaning = "deal outcome or stage indicator"
                    expectation = "use to derive success labels, not as a predictive feature for conversion"
                    risk = "post-outcome fields can leak target information"
                elif any(tok in norm_name for tok in ["probability", "score"]):
                    meaning = "prior probability or scoring output"
                    expectation = "validate whether it is input signal or model output before use"
                elif any(tok in norm_name for tok in ["date", "time", "day", "month", "year"]):
                    meaning = "temporal marker for event timing"
                    expectation = "use to derive cycle duration or ordering; avoid leaking post-outcome dates"
                    risk = "post-event dates can leak conversion outcome"
                elif any(tok in norm_name for tok in ["channel", "salesrep", "owner", "agent"]):
                    meaning = "operational or assignment attribute"
                    expectation = "high-cardinality categorical; may require grouping to avoid overfitting"
                elif any(tok in norm_name for tok in ["reason", "comment", "note", "desc"]):
                    meaning = "free-text context or explanation"
                    expectation = "high-cardinality text; use with caution or exclude in small samples"

                if meaning is None:
                    meaning = "feature or identifier relevant to the business context"

                semantics.append(
                    {
                        "column": name,
                        "role": role,
                        "expected_kind": kind,
                        "business_meaning": meaning,
                        "directional_expectation": expectation,
                        "risk_notes": risk,
                    }
                )
            return semantics

        def _build_business_sanity_checks(
            contract: Dict[str, Any],
            feature_semantics: List[Dict[str, Any]],
        ) -> List[str]:
            if not isinstance(contract, dict):
                return []
            checks: List[str] = []
            roles = [str(req.get("role") or "").lower() for req in contract.get("data_requirements", []) if isinstance(req, dict)]
            has_reg_target = any("target_regression" in role or role == "target" for role in roles)
            has_cls_target = any("target_classification" in role for role in roles)
            if has_reg_target and has_cls_target:
                checks.append(
                    "If a conversion model is trained, ensure contract-value fields are not used as predictors when they only exist after success."
                )
            for item in feature_semantics:
                if not isinstance(item, dict):
                    continue
                if item.get("risk_notes"):
                    checks.append(f"Review potential leakage for column '{item.get('column')}'.")
            checks.append("If predicted conversion rises with higher price in most segments, re-check leakage or label leakage.")
            checks.append("If recommendations exceed historical max by a large margin, treat as a hypothesis and justify with evidence.")
            checks.append("If a segment has very few samples, aggregate or downweight before recommending prices.")
            return checks

        def _attach_semantic_guidance(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            semantics = contract.get("feature_semantics")
            if not isinstance(semantics, list) or not semantics:
                semantics = _build_feature_semantics(contract)
                contract["feature_semantics"] = semantics
            sanity = contract.get("business_sanity_checks")
            if not isinstance(sanity, list) or not sanity:
                contract["business_sanity_checks"] = _build_business_sanity_checks(contract, semantics)
            return contract

        def _attach_probability_audit_note(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            notes = contract.get("notes_for_engineers")
            if not isinstance(notes, list):
                notes = []
            note = (
                "Probability columns (e.g., *prob*, *probability*, *score*) are post-decision audit only; "
                "do not use for segmentation or modeling."
            )
            if note not in notes:
                notes.append(note)
            contract["notes_for_engineers"] = notes
            return contract

        def _detect_decision_context(text: str) -> bool:
            if not text:
                return False
            norm_text = _norm(text)
            if not norm_text:
                return False
            tokens = [
                "price",
                "pricing",
                "precio",
                "tarifa",
                "quote",
                "offer",
                "cotizacion",
                "importe",
                "amount",
                "valor",
                "premium",
                "rate",
                "fee",
                "cost",
            ]
            token_norms = [_norm(tok) for tok in tokens if tok]
            return any(tok and tok in norm_text for tok in token_norms)

        def _is_price_like(name: str) -> bool:
            if not name:
                return False
            norm_name = _norm(name)
            tokens = [
                "price",
                "precio",
                "amount",
                "importe",
                "valor",
                "fee",
                "tarifa",
                "premium",
                "rate",
                "cost",
            ]
            token_norms = [_norm(tok) for tok in tokens if tok]
            return any(tok and tok in norm_name for tok in token_norms)

        def _extract_missing_sentinel(text: str) -> float | int | None:
            if not text:
                return None
            phrases = [
                "no offer",
                "not offered",
                "sin oferta",
                "sin propuesta",
                "no quote",
                "no proposal",
                "no bid",
                "sin precio",
            ]
            lower = text.lower()
            for sentence in re.split(r"[\n\.]", lower):
                if not any(p in sentence for p in phrases):
                    continue
                match = re.search(r"(?:=|->|:)\s*([0-9]+(?:[.,][0-9]+)?)", sentence)
                if not match:
                    match = re.search(r"\b([0-9]+(?:[.,][0-9]+)?)\b", sentence)
                if not match:
                    continue
                raw_val = match.group(1).replace(",", ".")
                try:
                    return float(raw_val) if "." in raw_val else int(raw_val)
                except ValueError:
                    continue
            return None

        def _assign_derived_owners(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") or {}
            formulas = spec.get("formulas") if isinstance(spec, dict) else {}
            formula_keys = set()
            if isinstance(formulas, dict):
                formula_keys = {_norm(k) for k in formulas.keys()}
            derived_spec = spec.get("derived_columns") if isinstance(spec, dict) else []
            derived_keys = set()
            if isinstance(derived_spec, list):
                for entry in derived_spec:
                    if isinstance(entry, dict):
                        name = entry.get("name")
                        if name:
                            derived_keys.add(_norm(name))
            reqs = contract.get("data_requirements", []) or []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                if req.get("source", "input") != "derived":
                    continue
                if req.get("derived_owner"):
                    continue
                name = req.get("canonical_name") or req.get("name")
                norm_name = _norm(name) if name else ""
                role = (req.get("role") or "").lower()
                if norm_name in derived_keys or norm_name in formula_keys:
                    req["derived_owner"] = "data_engineer"
                elif "segment" in role or "group" in role or "cluster" in role:
                    req["derived_owner"] = "ml_engineer"
                else:
                    req["derived_owner"] = "ml_engineer"
            contract["data_requirements"] = reqs
            return contract

        def _attach_variable_semantics(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            combined_text = "\n".join(
                [
                    business_objective or "",
                    data_summary or "",
                    json.dumps(strategy, ensure_ascii=True) if isinstance(strategy, dict) else "",
                ]
            )
            decision_context = _detect_decision_context(combined_text)
            decision_vars: List[str] = []
            if decision_context:
                for req in contract.get("data_requirements", []) or []:
                    if not isinstance(req, dict):
                        continue
                    name = req.get("canonical_name") or req.get("name")
                    if not name:
                        continue
                    if _is_price_like(name) or (req.get("role") or "").lower() == "target_regression":
                        req["decision_variable"] = True
                        decision_vars.append(name)

            # V4.1 fallback: derive decision variables from column_roles map when data_requirements is absent.
            if not decision_vars:
                col_roles = contract.get("column_roles")
                if isinstance(col_roles, dict):
                    inferred = [
                        str(col)
                        for col, role in col_roles.items()
                        if col and str(role).strip().lower() == "decision"
                    ]
                    if inferred:
                        decision_vars = inferred

            if decision_vars:
                unique_vars: List[str] = []
                seen = set()
                for item in decision_vars:
                    key = _norm(item)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    unique_vars.append(item)
                contract["decision_variables"] = unique_vars

                # Ensure allowed_feature_sets rationale is consistent with decision-variable usage.
                allowed_feature_sets = contract.get("allowed_feature_sets")
                if isinstance(allowed_feature_sets, dict):
                    seg_features = allowed_feature_sets.get("segmentation_features") or []
                    model_features = allowed_feature_sets.get("model_features") or []
                    uses_decision = any(var in model_features for var in unique_vars)
                    if uses_decision:
                        allowed_feature_sets["rationale"] = (
                            "Segmentation must use only pre-decision variables. Modeling may include pre-decision "
                            "variables plus decision variables (controlled by the business) for optimization/elasticity. "
                            "Outcome/post-outcome and audit-only features must not be used for training."
                        )
                        contract["allowed_feature_sets"] = allowed_feature_sets

                feature_availability = contract.get("feature_availability")
                if not isinstance(feature_availability, list):
                    feature_availability = []
                avail_map = {
                    _norm(item.get("column")): item
                    for item in feature_availability
                    if isinstance(item, dict) and item.get("column")
                }
                for col in unique_vars:
                    key = _norm(col)
                    entry = avail_map.get(key)
                    if not entry:
                        entry = {"column": col}
                        feature_availability.append(entry)
                    entry["availability"] = "decision"
                    entry.setdefault(
                        "rationale",
                        "Decision variable controlled by the business; usable for optimization or elasticity modeling.",
                    )
                contract["feature_availability"] = feature_availability
                # V4.1: availability_summary removed - no longer generated

                sentinel_value = _extract_missing_sentinel(combined_text)
                if sentinel_value is not None:
                    missing_sentinels = []
                    for col in unique_vars:
                        missing_sentinels.append(
                            {
                                "column": col,
                                "sentinel": sentinel_value,
                                "meaning": "not_observed",
                                "action": "treat_as_missing",
                            }
                        )
                    contract["missing_sentinels"] = missing_sentinels
                    notes = contract.get("notes_for_engineers")
                    if not isinstance(notes, list):
                        notes = []
                    note = (
                        "Missing sentinel detected for decision variables; treat sentinel values as missing "
                        "when modeling elasticity and document any derived observed flags."
                    )
                    if note not in notes:
                        notes.append(note)
                    contract["notes_for_engineers"] = notes

            return contract

        def _ensure_availability_reasoning(contract: Dict[str, Any]) -> Dict[str, Any]:
            # V4.1: availability_summary removed - only ensure feature_availability if needed
            if not isinstance(contract, dict):
                return contract
            if "feature_availability" not in contract:
                contract["feature_availability"] = []
            # V4.1: availability_summary removed - do NOT set it
            return contract

        def _attach_counterfactual_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            if contract.get("counterfactual_policy"):
                return contract
            decision_vars = contract.get("decision_variables") or []
            if not isinstance(decision_vars, list) or not decision_vars:
                return contract
            spec = contract.get("spec_extraction") or {}
            case_taxonomy = spec.get("case_taxonomy") if isinstance(spec.get("case_taxonomy"), list) else []
            # V4.1: availability_summary and data_requirements removed
            feature_availability = contract.get("feature_availability") or []
            combined_text = " ".join(
                [
                    json.dumps(feature_availability, ensure_ascii=True) if isinstance(feature_availability, list) else "",
                    data_summary or "",
                ]
            ).lower()
            evidence_tokens = [
                "random", "randomized", "experiment", "a/b", "ab test", "treatment",
                "control", "holdout", "policy change", "uplift", "causal", "instrument",
            ]
            has_counterfactual = any(tok in combined_text for tok in evidence_tokens)
            if not case_taxonomy:
                has_counterfactual = False
            if not has_counterfactual:
                contract["counterfactual_policy"] = "observational_only"
                contract["recommendation_scope"] = "within_observed_support_only"
                contract["required_limitations_section"] = True
                contract["required_next_steps"] = True
            else:
                contract["counterfactual_policy"] = "counterfactual_supported"
                contract["recommendation_scope"] = "supported_by_experiment"
            return contract

        def _ensure_iteration_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            if not isinstance(contract.get("compliance_checklist"), list):
                contract["compliance_checklist"] = []
            if not isinstance(contract.get("iteration_policy"), dict):
                contract["iteration_policy"] = {}
            return contract

        def _attach_reporting_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            policy = contract.get("reporting_policy")
            if not isinstance(policy, dict):
                policy = {}
            if "demonstrative_examples_enabled" not in policy:
                policy["demonstrative_examples_enabled"] = True
            if "demonstrative_examples_when_outcome_in" not in policy:
                policy["demonstrative_examples_when_outcome_in"] = ["NO_GO", "GO_WITH_LIMITATIONS"]
            if "max_examples" not in policy:
                policy["max_examples"] = 5
            if "require_strong_disclaimer" not in policy:
                policy["require_strong_disclaimer"] = True
            plot_spec = policy.get("plot_spec")
            if not isinstance(plot_spec, dict) or not plot_spec:
                policy["plot_spec"] = build_plot_spec(contract)
            contract["reporting_policy"] = policy
            return contract

        def _normalize_alignment_requirements(items: Any) -> List[Dict[str, Any]]:
            if not isinstance(items, list):
                return []
            normalized: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for idx, item in enumerate(items):
                if isinstance(item, str) and item.strip():
                    req_id = item.strip()
                    item = {"id": req_id, "requirement": req_id}
                if not isinstance(item, dict):
                    continue
                req_id = str(item.get("id") or item.get("name") or item.get("key") or f"custom_{idx}")
                if not req_id or req_id in seen:
                    continue
                seen.add(req_id)
                req = {
                    "id": req_id,
                    "requirement": str(item.get("requirement") or item.get("description") or "").strip(),
                    "rationale": str(item.get("rationale") or "").strip(),
                    "success_criteria": item.get("success_criteria") if isinstance(item.get("success_criteria"), list) else [],
                    "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
                    "applies_when": str(item.get("applies_when") or "always"),
                    "failure_mode_on_miss": str(item.get("failure_mode_on_miss") or "method_choice"),
                }
                normalized.append(req)
            return normalized

        def _detect_segment_context(text: str) -> bool:
            if not text:
                return False
            norm_text = _norm(text)
            tokens = [
                "segment",
                "segmentation",
                "segmented",
                "cluster",
                "cohort",
                "case",
                "bucket",
                "grupo",
                "segmento",
                "clase",
                "caso",
            ]
            token_norms = [_norm(tok) for tok in tokens if tok]
            return any(tok and tok in norm_text for tok in token_norms)

        def _build_alignment_requirements(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
            requirements: List[Dict[str, Any]] = []

            def _add(req_id: str, requirement: str, rationale: str, success: List[str], applies_when: str, failure_mode: str):
                requirements.append(
                    {
                        "id": req_id,
                        "requirement": requirement,
                        "rationale": rationale,
                        "success_criteria": success,
                        "evidence": [],
                        "applies_when": applies_when,
                        "failure_mode_on_miss": failure_mode,
                    }
                )

            _add(
                "objective_alignment",
                "Methodology directly answers the business objective and the strategy analysis_type.",
                "Prevents optimizing an easier proxy that misses the business goal.",
                [
                    "Approach matches analysis_type/techniques or a justified alternative is documented.",
                    "Outputs support the business decision stated in the objective.",
                ],
                "always",
                "method_choice",
            )

            decision_vars = contract.get("decision_variables") or []
            if isinstance(decision_vars, list) and decision_vars:
                preview = [str(v) for v in decision_vars[:5]]
                if len(decision_vars) > 5:
                    preview.append("...")
                _add(
                    "decision_variable_handling",
                    f"Decision variables are modeled as controllable inputs ({', '.join(preview)}).",
                    "Pricing/decision inputs must drive elasticity or optimization rather than be ignored.",
                    [
                        "Decision variables are used in modeling or optimization.",
                        "If excluded, explain why and quantify the impact on recommendations.",
                    ],
                    "decision_variables_present",
                    "method_choice",
                )

            segment_required = False

        def _fallback(reason: str = "Planner LLM Failed") -> Dict[str, Any]:
            d_sum = data_summary
            if isinstance(d_sum, dict):
                 d_sum = json.dumps(d_sum)
            return _create_v41_skeleton(
                strategy=strategy,
                business_objective=business_objective,
                column_inventory=column_inventory,
                output_dialect=output_dialect,
                reason=reason,
                data_summary=str(d_sum)
            )


        # Ensure data_summary is a string
        data_summary_str = ""
        if isinstance(data_summary, dict):
            data_summary_str = json.dumps(data_summary, indent=2)
        else:
            data_summary_str = str(data_summary)

        relevant_payload = select_relevant_columns(
            strategy=strategy,
            business_objective=business_objective,
            domain_expert_critique=domain_expert_critique,
            column_inventory=column_inventory or [],
            data_profile_summary=data_summary_str,
        )
        relevant_columns = relevant_payload.get("relevant_columns", [])
        relevant_sources = relevant_payload.get("relevant_sources", {})
        omitted_columns_policy = relevant_payload.get("omitted_columns_policy", "")

        planner_dir = None
        if run_id:
            run_dir = get_run_dir(run_id)
            if run_dir:
                planner_dir = os.path.join(run_dir, "agents", "execution_planner")
                os.makedirs(planner_dir, exist_ok=True)

        planner_diag: List[Dict[str, Any]] = []
        self.last_planner_diag = planner_diag
        self.last_contract_min = None

        def _write_text(path: str, content: str) -> None:
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content or "")
            except Exception:
                pass

        def _write_json(path: str, payload: Any) -> None:
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, ensure_ascii=True)
            except Exception:
                pass

        def _persist_attempt(prompt_name: str, response_name: str, prompt_text: str, response_text: str | None) -> None:
            if not planner_dir:
                return
            if prompt_text is not None:
                _write_text(os.path.join(planner_dir, prompt_name), prompt_text)
            if response_text is not None:
                _write_text(os.path.join(planner_dir, response_name), response_text)

        def _persist_contracts(full_contract: Dict[str, Any] | None, contract_min: Dict[str, Any] | None) -> None:
            if not planner_dir:
                return
            if contract_min:
                _write_json(os.path.join(planner_dir, "contract_min.json"), contract_min)
            if full_contract:
                _write_json(os.path.join(planner_dir, "contract_full.json"), full_contract)
            if planner_diag:
                _write_json(os.path.join(planner_dir, "planner_diag.json"), {"attempts": planner_diag})

        def _parse_json_response(raw_text: str) -> Tuple[Optional[Any], Optional[Exception]]:
            if not raw_text:
                return None, ValueError("Empty response text")
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(cleaned), None
            except json.JSONDecodeError as first_err:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    snippet = cleaned[start:end + 1]
                    try:
                        return json.loads(snippet), None
                    except Exception as second_err:
                        return None, second_err
                return None, first_err
            except Exception as err:
                return None, err

        def _normalize_usage_metadata(raw_usage: Any) -> Optional[Dict[str, Any]]:
            if raw_usage is None:
                return None
            if isinstance(raw_usage, dict):
                return raw_usage
            if hasattr(raw_usage, "to_dict"):
                try:
                    return raw_usage.to_dict()
                except Exception:
                    pass
            usage_payload = {}
            for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
                try:
                    value = getattr(raw_usage, key)
                except Exception:
                    value = None
                if value is not None:
                    usage_payload[key] = value
            return usage_payload or {"value": str(raw_usage)}

        def _finalize_and_persist(contract, contract_min, where):
            from src.utils.contract_v41 import strip_legacy_keys, assert_no_legacy_keys, assert_only_allowed_v41_keys
            contract = ensure_v41_schema(contract, strict=False)
            contract = strip_legacy_keys(contract)
            if contract_min:
                contract_min = strip_legacy_keys(contract_min)
            assert_no_legacy_keys(contract, where=where)
            unknown = assert_only_allowed_v41_keys(contract, strict=False)
            if unknown:
                print(f"WARNING: Unknown keys in contract: {unknown}")
                u = contract.setdefault("unknowns", [])
                if isinstance(u, list):
                    u.append({"item": f"Unknown top-level keys detected: {unknown}", "impact":"May not be recognized", "mitigation":"Update allowed keys if needed", "requires_verification": False})
            _persist_contracts(contract, contract_min)
            return contract

        target_candidates: List[Dict[str, Any]] = []
        if not self.client:
            contract = _fallback()
            contract = _attach_reporting_policy(contract)
            if isinstance(data_profile, dict):
                try:
                    from src.utils.data_profile_compact import compact_data_profile_for_llm
                    compact = compact_data_profile_for_llm(data_profile, contract=contract)
                    target_candidates = compact.get("target_candidates") if isinstance(compact, dict) else []
                except Exception:
                    target_candidates = []
            if isinstance(target_candidates, list) and target_candidates:
                contract["target_candidates"] = target_candidates
            explicit_outcomes = contract.get("outcome_columns")
            has_outcomes = False
            if isinstance(explicit_outcomes, list):
                has_outcomes = any(str(v).strip().lower() != "unknown" for v in explicit_outcomes if v is not None)
            elif isinstance(explicit_outcomes, str):
                has_outcomes = explicit_outcomes.strip().lower() != "unknown"
            if not has_outcomes:
                inferred = []
                for item in target_candidates or []:
                    if isinstance(item, dict):
                        col = item.get("column") or item.get("name") or item.get("candidate")
                        if col:
                            inferred.append(col)
                            break
                if inferred:
                    contract["outcome_columns"] = inferred
            contract_min = build_contract_min(contract, strategy, column_inventory, relevant_columns, target_candidates=target_candidates, data_profile=data_profile)
            contract = _sync_execution_contract_outputs(contract, contract_min)
            self.last_contract_min = contract_min
            return _finalize_and_persist(contract, contract_min, where="execution_planner:no_client")

        strategy_json = json.dumps(strategy, indent=2)
        column_inventory_count = len(column_inventory or [])
        column_inventory_sample = (column_inventory or [])[:25]
        inventory_truncated = column_inventory_count > 50
        column_inventory_payload = column_inventory_sample if inventory_truncated else (column_inventory or [])

        user_input = f"""
strategy:
{strategy_json}

business_objective:
{business_objective}

relevant_columns:
{json.dumps(relevant_columns, indent=2)}

relevant_sources:
{json.dumps(relevant_sources, indent=2)}

omitted_columns_policy:
{omitted_columns_policy}

column_inventory_count:
{column_inventory_count}

column_inventory_sample:
{json.dumps(column_inventory_sample, indent=2)}

column_inventory_truncated:
{json.dumps(inventory_truncated)}

column_inventory:
{json.dumps(column_inventory_payload, indent=2)}

data_profile_summary:
{data_summary_str}

output_dialect:
{json.dumps(output_dialect or "unknown")}

env_constraints:
{json.dumps(env_constraints or {"forbid_inplace_column_creation": True})}

domain_expert_critique:
{domain_expert_critique or "None"}
"""

        full_prompt = SENIOR_PLANNER_PROMPT + "\n\nINPUTS:\n" + user_input
        repair_keys = [
            "contract_version",
            "strategy_title",
            "business_objective",
            "canonical_columns",
            "outcome_columns",
            "decision_columns",
            "column_roles",
            "allowed_feature_sets",
            "artifact_requirements",
            "required_outputs",
            "qa_gates",
            "reviewer_gates",
            "data_engineer_runbook",
            "ml_engineer_runbook",
            "omitted_columns_policy",
        ]
        compressed_objective = _compress_text_preserve_ends(business_objective or "")
        success_metric = strategy.get("success_metric") if isinstance(strategy, dict) else None
        recommended_metrics = (
            strategy.get("recommended_evaluation_metrics") if isinstance(strategy, dict) else None
        )
        validation_strategy = strategy.get("validation_strategy") if isinstance(strategy, dict) else None
        repair_prompt = (
            "Return ONLY valid JSON with EXACT keys: "
            + json.dumps(repair_keys)
            + ". Do not include any other keys or commentary.\n"
            + "Use RELEVANT_COLUMNS as focus_columns only; do NOT truncate canonical_columns.\n"
            + "RELEVANT_COLUMNS: "
            + json.dumps(relevant_columns)
            + "\n"
            + "STRATEGY_TITLE: "
            + json.dumps(strategy.get("title", "") if isinstance(strategy, dict) else "")
            + "\n"
            + "BUSINESS_OBJECTIVE: "
            + json.dumps(compressed_objective)
            + "\n"
            + "SUCCESS_METRIC: "
            + json.dumps(success_metric or "")
            + "\n"
            + "RECOMMENDED_EVALUATION_METRICS: "
            + json.dumps(recommended_metrics or [])
            + "\n"
            + "VALIDATION_STRATEGY: "
            + json.dumps(validation_strategy or "")
        )

        attempt_prompts = [
            ("prompt_attempt_1.txt", "response_attempt_1.txt", full_prompt),
            ("prompt_attempt_2_repair.txt", "response_attempt_2.txt", repair_prompt),
        ]

        contract: Dict[str, Any] | None = None
        llm_success = False

        for attempt_index, (prompt_name, response_name, prompt_text) in enumerate(attempt_prompts, start=1):
            self.last_prompt = prompt_text
            response_text = ""
            response = None
            parse_error: Optional[Exception] = None
            finish_reason = None
            usage_metadata = None

            try:
                response = self.client.generate_content(prompt_text)
                response_text = getattr(response, "text", "") or ""
                self.last_response = response_text
            except Exception as err:
                parse_error = err

            if response is not None:
                try:
                    candidates = getattr(response, "candidates", None)
                    if candidates:
                        finish_reason = getattr(candidates[0], "finish_reason", None)
                except Exception:
                    finish_reason = None
                usage_metadata = _normalize_usage_metadata(getattr(response, "usage_metadata", None))

            _persist_attempt(prompt_name, response_name, prompt_text, response_text)

            parsed, parse_exc = _parse_json_response(response_text) if response_text else (None, parse_error)
            if parse_exc:
                parse_error = parse_exc

            had_json_parse_error = parsed is None or not isinstance(parsed, dict)
            if parsed is not None and not isinstance(parsed, dict):
                parse_error = ValueError("Parsed JSON is not an object")

            planner_diag.append(
                {
                    "model_name": self.model_name,
                    "attempt_index": attempt_index,
                    "prompt_char_len": len(prompt_text or ""),
                    "response_char_len": len(response_text or ""),
                    "finish_reason": str(finish_reason) if finish_reason is not None else None,
                    "usage_metadata": usage_metadata,
                    "had_json_parse_error": bool(had_json_parse_error),
                    "parse_error_type": type(parse_error).__name__ if parse_error else None,
                    "parse_error_message": str(parse_error) if parse_error else None,
                }
            )

            if parsed is None or not isinstance(parsed, dict):
                print(f"WARNING: Planner parse failed on attempt {attempt_index}.")
                continue

            contract = parsed
            llm_success = True
            break

        if contract is None:
            contract = _fallback("JSON Parse Error after retries")

        contract = ensure_v41_schema(contract)
        contract["qa_gates"] = _apply_qa_gate_policy(
            contract.get("qa_gates"),
            strategy,
            business_objective or "",
            contract,
        )
        contract = _ensure_benchmark_kpi_gate(contract, strategy, business_objective or "")
        contract["cleaning_gates"] = _apply_cleaning_gate_policy(contract.get("cleaning_gates"))
        contract = _ensure_missing_category_values(contract)

        if isinstance(column_inventory, list) and column_inventory:
            # Always preserve full inventory as available_columns; use focus_columns for relevance.
            contract["available_columns"] = [str(c) for c in column_inventory if c]
            contract["available_columns_source"] = "column_inventory"

        available_columns = contract.get("available_columns")
        if isinstance(available_columns, list) and available_columns:
            contract["canonical_columns"] = [str(c) for c in available_columns if c]
            if len(available_columns) > 200:
                feature_selectors, _remaining = infer_feature_selectors(
                    available_columns, max_list_size=200, min_group_size=50
                )
                if feature_selectors:
                    contract["feature_selectors"] = feature_selectors
                    contract["canonical_columns_compact"] = compact_column_representation(
                        available_columns, max_display=40
                    )

        if relevant_columns:
            contract["focus_columns"] = list(relevant_columns)
            contract["omitted_columns_policy"] = omitted_columns_policy

        contract = _prune_identifier_model_features(contract)
        contract = _apply_sparse_optional_columns(contract, data_profile)

        contract = validate_artifact_requirements(contract)

        # P1.2: Contract Self-Consistency Gate
        validation_result = validate_contract(contract)
        contract["_contract_validation"] = validation_result
        if validation_result["status"] == "error":
            print(f"CONTRACT_VALIDATION_ERROR: {len(validation_result['issues'])} issues found")
            for issue in validation_result["issues"]:
                print(f"  - [{issue['severity']}] {issue['rule']}: {issue['message']}")
        elif validation_result["status"] == "warning":
            print(f"CONTRACT_VALIDATION_WARNING: {len(validation_result['issues'])} issues found")
            for issue in validation_result["issues"]:
                print(f"  - [{issue['severity']}] {issue['rule']}: {issue['message']}")
        issues = validation_result.get("issues") if isinstance(validation_result, dict) else []
        if isinstance(issues, list):
            if any(issue.get("rule") == "output_ambiguity" for issue in issues if isinstance(issue, dict)):
                planner_self_check = contract.get("planner_self_check")
                if not isinstance(planner_self_check, list):
                    planner_self_check = []
                msg = (
                    "OUTPUT_AMBIGUITY: required_outputs contained non-file entries. "
                    "Ensure required_outputs are file paths only; move columns to scored_rows_schema "
                    "and conceptual deliverables to reporting_requirements."
                )
                if msg not in planner_self_check:
                    planner_self_check.append(msg)
                contract["planner_self_check"] = planner_self_check
        # Store normalized artifact_requirements
        if validation_result.get("normalized_artifact_requirements"):
            contract["artifact_requirements"] = validation_result["normalized_artifact_requirements"]

        artifact_reqs = contract.get("artifact_requirements")
        if not isinstance(artifact_reqs, dict):
            artifact_reqs = {}
        artifact_reqs["visual_requirements"] = _build_visual_requirements(
            contract,
            strategy,
            business_objective or "",
        )
        contract["artifact_requirements"] = artifact_reqs
        contract["decisioning_requirements"] = _build_decisioning_requirements(
            contract,
            strategy,
            business_objective or "",
        )

        contract = _attach_reporting_policy(contract)

        def _sanitize_runbook_text(text: str) -> str:
            """Replace hardcoded dialect instructions with dynamic manifest reference."""
            if not isinstance(text, str):
                return text
            import re
            patterns = [
                (r'sep\s*=\s*[\'"]?[;,\t][\'"]?', "sep from cleaning_manifest.json"),
                (r'decimal\s*=\s*[\'"]?[.,][\'"]?', "decimal from cleaning_manifest.json"),
                (r"Load.*CSV.*using.*sep.*=.*[;,]", "Load CSV using dialect from data/cleaning_manifest.json"),
            ]
            for pattern, replacement in patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            return text

        def _sanitize_runbook(runbook: Any) -> Any:
            if isinstance(runbook, str):
                return _sanitize_runbook_text(runbook)
            if isinstance(runbook, dict):
                return {k: _sanitize_runbook(v) for k, v in runbook.items()}
            if isinstance(runbook, list):
                return [_sanitize_runbook(item) for item in runbook]
            return runbook

        if contract.get("data_engineer_runbook"):
            contract["data_engineer_runbook"] = _sanitize_runbook(contract["data_engineer_runbook"])
        if contract.get("ml_engineer_runbook"):
            contract["ml_engineer_runbook"] = _sanitize_runbook(contract["ml_engineer_runbook"])

        # V4.1: Do NOT write role_runbooks - use direct data_engineer_runbook/ml_engineer_runbook keys

        target_candidates = []
        if isinstance(data_profile, dict):
            try:
                from src.utils.data_profile_compact import compact_data_profile_for_llm
                compact = compact_data_profile_for_llm(data_profile, contract=contract)
                target_candidates = compact.get("target_candidates") if isinstance(compact, dict) else []
            except Exception:
                target_candidates = []
        if isinstance(target_candidates, list) and target_candidates:
            contract["target_candidates"] = target_candidates
        explicit_outcomes = contract.get("outcome_columns")
        has_outcomes = False
        if isinstance(explicit_outcomes, list):
            has_outcomes = any(str(v).strip().lower() != "unknown" for v in explicit_outcomes if v is not None)
        elif isinstance(explicit_outcomes, str):
            has_outcomes = explicit_outcomes.strip().lower() != "unknown"
        if not has_outcomes:
            inferred = []
            for item in target_candidates or []:
                if isinstance(item, dict):
                    col = item.get("column") or item.get("name") or item.get("candidate")
                    if col:
                        inferred.append(col)
                        break
            if inferred:
                contract["outcome_columns"] = inferred
        contract_min = build_contract_min(contract, strategy, column_inventory, relevant_columns, target_candidates=target_candidates, data_profile=data_profile)
        contract = _sync_execution_contract_outputs(contract, contract_min)
        self.last_contract_min = contract_min

        if llm_success and planner_diag:
            print(f"SUCCESS: Execution Planner succeeded on attempt {planner_diag[-1]['attempt_index']}")
        return _finalize_and_persist(contract, contract_min, where="execution_planner:final_contract")

    def generate_evaluation_spec(
        self,
        strategy: Dict[str, Any],
        contract: Dict[str, Any],
        data_summary: str = "",
        business_objective: str = "",
        column_inventory: list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate evaluation spec by extracting ONLY from V4.1 contract fields.
        NO legacy fields (data_requirements, spec_extraction) allowed.
        """
        if not isinstance(contract, dict):
            return {
                "confidence": 0.1,
                "qa_gates": [],
                "reviewer_gates": [],
                "artifact_requirements": {},
                "notes": ["Invalid contract structure"],
                "source": "error_fallback",
                "unknowns": ["Contract is not a valid dictionary"]
            }
        
        # Extract ONLY from V4.1 fields
        qa_gates = contract.get("qa_gates", [])
        cleaning_gates = contract.get("cleaning_gates", [])
        reviewer_gates = contract.get("reviewer_gates", [])
        artifact_requirements = contract.get("artifact_requirements", {})
        validation_requirements = contract.get("validation_requirements", {})
        leakage_execution_plan = contract.get("leakage_execution_plan", {})
        allowed_feature_sets = contract.get("allowed_feature_sets", {})
        canonical_columns = contract.get("canonical_columns", [])
        derived_columns = contract.get("derived_columns", [])
        required_outputs = contract.get("required_outputs", [])
        data_limited_mode = contract.get("data_limited_mode", {})
        
        # Build evaluation spec from V4.1 contract
        spec = {
            "qa_gates": qa_gates if isinstance(qa_gates, list) else [],
            "cleaning_gates": cleaning_gates if isinstance(cleaning_gates, list) else [],
            "reviewer_gates": reviewer_gates if isinstance(reviewer_gates, list) else [],
            "artifact_requirements": artifact_requirements if isinstance(artifact_requirements, dict) else {},
            "validation_requirements": validation_requirements if isinstance(validation_requirements, dict) else {},
            "leakage_execution_plan": leakage_execution_plan if isinstance(leakage_execution_plan, dict) else {},
            "allowed_feature_sets": allowed_feature_sets if isinstance(allowed_feature_sets, dict) else {},
            "canonical_columns": canonical_columns if isinstance(canonical_columns, list) else [],
            "derived_columns": derived_columns if isinstance(derived_columns, list) else [],
            "required_outputs": required_outputs if isinstance(required_outputs, list) else [],
            "data_limited_mode": data_limited_mode if isinstance(data_limited_mode, dict) else {},
            "confidence": 0.9,
            "source": "contract_driven_v41",
            "notes": ["Extracted directly from V4.1 contract fields"]
        }
        
        # Add unknowns if critical fields are missing
        unknowns = []
        if not qa_gates:
            unknowns.append("qa_gates missing from contract")
        if not cleaning_gates:
            unknowns.append("cleaning_gates missing from contract")
        if not reviewer_gates:
            unknowns.append("reviewer_gates missing from contract")
        if not required_outputs:
            unknowns.append("required_outputs missing from contract")
        
        if unknowns:
            spec["unknowns"] = unknowns
            spec["confidence"] = 0.6  # Lower confidence if fields missing
        
        return spec
