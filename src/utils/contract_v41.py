#!/usr/bin/env python3
"""
V4.1 Contract Accessors – Unified, safe read-only helpers for Execution Contract V4.1.

These functions provide a stable, typed API for all agents and utilities to query
the contract using V4.1 schema keys only.

All accessors:
  - Return safe, typed defaults (empty lists/dicts) when keys are missing or malformed.
  - Never mutate the contract.
  - Gracefully handle null/unknown values in outcome/decision columns.
"""

from typing import Dict, Any, List, Set
from src.utils.contract_validator import is_probably_path


# ============================================================================
# V4.1 CONTRACT GUARD – Enforcement of legacy field removal
# ============================================================================

# V4.1 Contract Version - Single source of truth
CONTRACT_VERSION_V41: str = "4.1"

# Legacy keys that MUST NOT appear in V4.1 contracts
LEGACY_KEYS: Set[str] = {
    "spec_extraction",
    "data_requirements",
    "role_runbooks",
    "validations",
    "quality_gates",
    "execution_plan",
    "artifact_schemas",
    "required_columns",
    "feature_availability",
    "decision_variables",
    "availability_summary",  # V4.1: Removed - no longer generated or consumed
}


def normalize_contract_version(version: Any) -> str:
    """
    Normalize contract version to V4.1 format.

    Accepts: "4.1", 4.1, 41, "41", 2, "2" -> returns "4.1"
    Returns CONTRACT_VERSION_V41 for any recognized version or None.
    """
    if version is None:
        return CONTRACT_VERSION_V41

    # Handle string versions
    if isinstance(version, str):
        v = version.strip().lower()
        if v in ("4.1", "v4.1", "41", "v41"):
            return CONTRACT_VERSION_V41
        # Legacy version 2 -> upgrade to 4.1
        if v in ("2", "2.0", "v2"):
            return CONTRACT_VERSION_V41

    # Handle numeric versions
    if isinstance(version, (int, float)):
        if version in (4.1, 41, 2, 2.0):
            return CONTRACT_VERSION_V41

    # Default: return V4.1
    return CONTRACT_VERSION_V41

# Allowed top-level keys in V4.1 contracts (for strict validation)
ALLOWED_TOP_LEVEL_KEYS_V41: Set[str] = {
    # Core identification
    "contract_version",
    "strategy_title",
    "business_objective",
    # Column specifications
    "available_columns",
    "canonical_columns",
    "derived_columns",
    "column_roles",
    "outcome_columns",
    "decision_columns",
    # Feature management
    "allowed_feature_sets",
    "feature_selectors",
    "feature_engineering_plan",
    # Requirements & validation
    "preprocessing_requirements",
    "validation_requirements",
    "artifact_requirements",
    "required_outputs",
    # Gates (V4.1 style)
    "qa_gates",
    "cleaning_gates",
    "reviewer_gates",
    # Runbooks (V4.1 style - direct, not nested under role_runbooks)
    "data_engineer_runbook",
    "ml_engineer_runbook",
    # Analysis & constraints
    "objective_analysis",
    "data_analysis",
    "execution_constraints",
    "segmentation_constraints",
    "optimization_specification",
    "leakage_execution_plan",
    # Policies & handling
    "reporting_policy",
    "decisioning_requirements",
    "omitted_columns_policy",
    "missing_columns_handling",
    "output_dialect",
    "visualization_requirements",
    # Data modes
    "data_limited_mode",
    "training_rows_rule",
    "scoring_rows_rule",
    "secondary_scoring_subset",
    "data_partitioning_notes",
    # Metadata
    "iteration_policy",
    "unknowns",
    "assumptions",
    "notes_for_engineers",
    # Ancillary (allowed but not required)
    "business_alignment",
    "compliance_checklist",
    "alignment_requirements",
    "evaluation_spec",  # V4.1 evaluation spec (not legacy)
}


def assert_no_legacy_keys(contract: Dict[str, Any], where: str = "") -> None:
    """
    Assert that the contract contains no legacy keys.

    Raises ValueError if any legacy key is found.

    Args:
        contract: The contract dictionary to validate.
        where: Optional context string for error messages (e.g., "execution_planner:final_contract").
    """
    if not isinstance(contract, dict):
        return

    found_legacy = LEGACY_KEYS & set(contract.keys())
    if found_legacy:
        location = f" at {where}" if where else ""
        raise ValueError(
            f"V4.1 Contract Violation{location}: Legacy keys found: {sorted(found_legacy)}. "
            f"These keys must be removed for V4.1 compliance."
        )


def warn_legacy_keys(contract: Dict[str, Any], where: str = "") -> List[str]:
    """
    Check for legacy keys and return a list of found keys (non-raising version).

    Args:
        contract: The contract dictionary to validate.
        where: Optional context string for logging.

    Returns:
        List of legacy keys found (empty if none).
    """
    if not isinstance(contract, dict):
        return []

    found_legacy = sorted(LEGACY_KEYS & set(contract.keys()))
    if found_legacy and where:
        print(f"WARNING: Legacy keys found at {where}: {found_legacy}")
    return found_legacy


def assert_only_allowed_v41_keys(contract: Dict[str, Any], strict: bool = False) -> List[str]:
    """
    Check that contract only contains allowed V4.1 keys.

    Args:
        contract: The contract dictionary to validate.
        strict: If True, raises ValueError on unknown keys. If False, just returns the list.

    Returns:
        List of unknown/unexpected keys found.
    """
    if not isinstance(contract, dict):
        return []

    contract_keys = set(contract.keys())
    unknown_keys = sorted(contract_keys - ALLOWED_TOP_LEVEL_KEYS_V41 - LEGACY_KEYS)

    if strict and unknown_keys:
        raise ValueError(
            f"V4.1 Contract Violation: Unknown keys found: {unknown_keys}. "
            f"Only allowed V4.1 keys are permitted."
        )

    return unknown_keys


def strip_legacy_keys(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new contract dict with all legacy keys removed.

    Does NOT mutate the original contract.

    Args:
        contract: The contract dictionary to clean.

    Returns:
        New dictionary with legacy keys removed.
    """
    if not isinstance(contract, dict):
        return {}

    return {k: v for k, v in contract.items() if k not in LEGACY_KEYS}


def get_available_columns(contract: Dict[str, Any]) -> List[str]:
    """
    Return the list of column names present in the dataset (before any transformations).
    
    Returns:
        List of column names available in the raw dataset.
    """
    if not isinstance(contract, dict):
        return []
    cols = contract.get("available_columns")
    if not isinstance(cols, list):
        return []
    return [str(c) for c in cols if c is not None]


def get_canonical_columns(contract: Dict[str, Any]) -> List[str]:
    """
    Return the canonical (required input) column names that agents must preserve.
    
    This is the source of truth for which columns are mandatory inputs.
    
    Returns:
        List of canonical column names.
    """
    if not isinstance(contract, dict):
        return []
    cols = contract.get("canonical_columns")
    if not isinstance(cols, list):
        return []
    return [str(c) for c in cols if c is not None]


def get_column_roles(contract: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Return the column_roles dictionary normalized to role -> list[str].
    
    Supports three input formats for robustness:
      A) dict role -> list[str]   (preferred V4.1 format)
      B) list[{"role": ..., "column": ...}]  (alternate list format)
      C) dict column -> role      (legacy V4.1 inverted format)
    
    Always returns: Dict[str, List[str]] mapping role -> columns.
    
    Returns:
        Dictionary mapping role names to lists of column names.
    """
    if not isinstance(contract, dict):
        return {}
    roles_raw = contract.get("column_roles")
    if not roles_raw:
        return {}
    
    # Format A: dict role -> list[str]
    if isinstance(roles_raw, dict):
        first_val = next(iter(roles_raw.values()), None) if roles_raw else None
        if isinstance(first_val, list):
            # Already in preferred format
            result: Dict[str, List[str]] = {}
            for role, cols in roles_raw.items():
                if isinstance(cols, list):
                    result[str(role)] = [str(c) for c in cols if c is not None]
            return result
        elif isinstance(first_val, dict):
            # Format C with metadata: column -> {role: ..., ...}
            result = {}
            for col, meta in roles_raw.items():
                if isinstance(meta, dict):
                    role = meta.get("role")
                    if role:
                        result.setdefault(str(role), []).append(str(col))
            return result
        elif isinstance(first_val, str) or first_val is None:
            # Format C: column -> role (inverted)
            result = {}
            for col, role in roles_raw.items():
                if isinstance(role, str):
                    result.setdefault(role, []).append(str(col))
            return result
    
    # Format B: list[{"role": ..., "column": ...}]
    if isinstance(roles_raw, list):
        result = {}
        for item in roles_raw:
            if isinstance(item, dict):
                role = item.get("role")
                col = item.get("column") or item.get("name")
                if role and col:
                    result.setdefault(str(role), []).append(str(col))
        return result
    
    return {}


def get_outcome_columns(contract: Dict[str, Any]) -> List[str]:
    """
    Return the list of outcome column names.

    Priority:
      1. contract["outcome_columns"] if provided
      2. column_roles["outcome"] if exists
      3. validation_requirements.params (target, label_column, outcome_column)
      4. objective_analysis if contains target-related info
      5. Empty list if nothing found

    Returns:
        List of outcome column names. Empty if not specified or 'unknown'.
    """
    if not isinstance(contract, dict):
        return []

    explicit = contract.get("outcome_columns")
    if isinstance(explicit, list) and explicit:
        return [str(v) for v in explicit if v and str(v).lower() != "unknown"]
    if isinstance(explicit, str) and explicit.lower() != "unknown":
        return [explicit]

    # First: try column_roles["outcome"]
    roles = get_column_roles(contract)
    outcome_cols = roles.get("outcome", [])
    if outcome_cols:
        return [c for c in outcome_cols if c and c.lower() != "unknown"]

    # Fallback: validation_requirements.params
    val_req = contract.get("validation_requirements")
    if isinstance(val_req, dict):
        params = val_req.get("params", {})
        if isinstance(params, dict):
            for key in ("target", "target_column", "label_column", "outcome_column", "y_column"):
                val = params.get(key)
                if isinstance(val, str) and val.strip() and val.lower() != "unknown":
                    return [val.strip()]
                elif isinstance(val, list):
                    result = [str(v).strip() for v in val if v and str(v).lower() != "unknown"]
                    if result:
                        return result

    # Fallback: objective_analysis
    obj_analysis = contract.get("objective_analysis")
    if isinstance(obj_analysis, dict):
        # Check for target/outcome fields
        for key in ("target_column", "outcome_column", "target", "outcome"):
            val = obj_analysis.get(key)
            if val:
                if isinstance(val, list):
                    return [str(v) for v in val if v and str(v).lower() != "unknown"]
                elif isinstance(val, str) and val.lower() != "unknown":
                    return [val]

    return []


def get_decision_columns(contract: Dict[str, Any]) -> List[str]:
    """
    Return the list of decision column names.
    
    Returns:
        List of decision column names. Empty if not specified.
    """
    if not isinstance(contract, dict):
        return []

    explicit = contract.get("decision_columns")
    if isinstance(explicit, list) and explicit:
        return [str(v) for v in explicit if v and str(v).lower() != "unknown"]
    if isinstance(explicit, str) and explicit.lower() != "unknown":
        return [explicit]
    
    # Use column_roles["decision"]
    roles = get_column_roles(contract)
    decision_cols = roles.get("decision", [])
    return [c for c in decision_cols if c and c.lower() != "unknown"]


def get_derived_column_names(contract: Dict[str, Any]) -> List[str]:
    """
    Return names of derived columns (columns created during preprocessing/feature engineering).
    
    Combines:
      - contract.get("derived_columns")
      - contract.get("feature_engineering_plan", {}).get("derived_columns")
    
    Accepts list[str] or list[dict{name/...}] and normalizes to unique list[str].
    
    Returns:
        List of derived column names (unique, order preserved).
    """
    if not isinstance(contract, dict):
        return []
    
    def _extract_names(source: Any) -> List[str]:
        if not source:
            return []
        if isinstance(source, list):
            names = []
            for item in source:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("column")
                    if name:
                        names.append(str(name))
            return names
        return []
    
    derived = []
    
    # From top-level derived_columns
    derived.extend(_extract_names(contract.get("derived_columns")))
    
    # From feature_engineering_plan.derived_columns
    feat_plan = contract.get("feature_engineering_plan")
    if isinstance(feat_plan, dict):
        derived.extend(_extract_names(feat_plan.get("derived_columns")))
        derived.extend(_extract_names(feat_plan.get("features")))
    
    # From preprocessing_requirements transformations
    prep_reqs = contract.get("preprocessing_requirements")
    if isinstance(prep_reqs, dict):
        transformations = prep_reqs.get("transformations")
        if isinstance(transformations, list):
            for t in transformations:
                if isinstance(t, dict):
                    output_col = t.get("output_column")
                    if output_col and isinstance(output_col, str):
                        derived.append(output_col)
    
    # Deduplicate preserving order
    seen = set()
    result = []
    for name in derived:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def get_artifact_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return artifact_requirements section defining required outputs.
    
    Example structure:
    {
      "required_files": ["scored_rows.csv", "model_summary.json"],
      "required_plots": ["confusion_matrix.png"],
      "file_schemas": {...}
    }
    
    Returns:
        Dictionary with artifact requirements.
    """
    if not isinstance(contract, dict):
        return {"required_files": [], "required_plots": []}
    artifacts = contract.get("artifact_requirements")
    if not isinstance(artifacts, dict):
        return {"required_files": [], "required_plots": []}
    return artifacts


def get_qa_gates(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return the list of QA gate specifications.
    
    Returns:
        List of QA gate objects.
    """
    if not isinstance(contract, dict):
        return []
    gates = contract.get("qa_gates")
    if not isinstance(gates, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for gate in gates:
        if isinstance(gate, dict):
            name = gate.get("name") or gate.get("id") or gate.get("gate")
            if not name:
                continue
            severity = gate.get("severity")
            required = gate.get("required")
            if severity is None and required is not None:
                severity = "HARD" if bool(required) else "SOFT"
            severity = str(severity).upper() if severity else "HARD"
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
        elif isinstance(gate, str):
            key = gate.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append({"name": gate.strip(), "severity": "HARD", "params": {}})
    return normalized


def get_cleaning_gates(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return the list of cleaning gate specifications.

    Returns:
        List of cleaning gate objects.
    """
    if not isinstance(contract, dict):
        return []
    gates = contract.get("cleaning_gates")
    if not isinstance(gates, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for gate in gates:
        if isinstance(gate, dict):
            name = gate.get("name") or gate.get("id") or gate.get("gate")
            if not name:
                continue
            severity = gate.get("severity")
            required = gate.get("required")
            if severity is None and required is not None:
                severity = "HARD" if bool(required) else "SOFT"
            severity = str(severity).upper() if severity else "HARD"
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
        elif isinstance(gate, str):
            key = gate.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append({"name": gate.strip(), "severity": "HARD", "params": {}})
    return normalized


def get_reviewer_gates(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return the list of reviewer gate specifications.
    
    Returns:
        List of reviewer gate objects.
    """
    if not isinstance(contract, dict):
        return []
    gates = contract.get("reviewer_gates")
    if not isinstance(gates, list):
        return []
    return [g for g in gates if isinstance(g, dict)]


def get_data_engineer_runbook(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the data_engineer_runbook section.
    
    Returns:
        Dictionary containing data engineer instructions.
    """
    if not isinstance(contract, dict):
        return {}
    runbook = contract.get("data_engineer_runbook")
    if not isinstance(runbook, dict):
        return {}
    return runbook


def get_ml_engineer_runbook(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the ml_engineer_runbook section.
    
    Returns:
        Dictionary containing ML engineer instructions.
    """
    if not isinstance(contract, dict):
        return {}
    runbook = contract.get("ml_engineer_runbook")
    if not isinstance(runbook, dict):
        return {}
    return runbook


def get_preprocessing_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return preprocessing_requirements section.
    
    Returns:
        Dictionary with preprocessing specifications.
    """
    if not isinstance(contract, dict):
        return {}
    prep = contract.get("preprocessing_requirements")
    if not isinstance(prep, dict):
        return {}
    return prep


def get_validation_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return validation_requirements section.
    
    Returns:
        Dictionary with validation specifications.
    """
    if not isinstance(contract, dict):
        return {}
    validation = contract.get("validation_requirements")
    if not isinstance(validation, dict):
        return {}
    return validation


def get_optimization_specification(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return optimization_specification section.
    
    Returns:
        Dictionary with optimization specifications.
    """
    if not isinstance(contract, dict):
        return {}
    opt = contract.get("optimization_specification")
    if not isinstance(opt, dict):
        return {}
    return opt


def get_segmentation_constraints(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return segmentation_constraints section.
    
    Returns:
        Dictionary with segmentation specifications.
    """
    if not isinstance(contract, dict):
        return {}
    seg = contract.get("segmentation_constraints")
    if not isinstance(seg, dict):
        return {}
    return seg


def get_execution_constraints(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return execution_constraints section.
    
    Returns:
        Dictionary with execution constraints.
    """
    if not isinstance(contract, dict):
        return {}
    constraints = contract.get("execution_constraints")
    if not isinstance(constraints, dict):
        return {}
    return constraints


def get_objective_analysis(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return objective_analysis section.
    
    Returns:
        Dictionary with objective analysis.
    """
    if not isinstance(contract, dict):
        return {}
    obj = contract.get("objective_analysis")
    if not isinstance(obj, dict):
        return {}
    return obj


def get_feature_engineering_plan(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return feature_engineering_plan section.
    
    Returns:
        Dictionary with feature engineering plan.
    """
    if not isinstance(contract, dict):
        return {}
    plan = contract.get("feature_engineering_plan")
    if not isinstance(plan, dict):
        return {}
    return plan


def get_leakage_execution_plan(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return leakage_execution_plan section.
    
    Returns:
        Dictionary with leakage audit plan.
    """
    if not isinstance(contract, dict):
        return {}
    plan = contract.get("leakage_execution_plan")
    if not isinstance(plan, dict):
        return {}
    return plan


def get_required_outputs(contract: Dict[str, Any]) -> List[str]:
    """
    Return list of required output files from artifact_requirements and required_outputs.
    
    Combines:
      - artifact_requirements.required_files
      - required_outputs (top-level)
    
    Normalizes paths:
      - Backslash to forward slash
      - Known basenames (metrics.json, etc.) get data/ prefix
    
    Returns:
        Unique list of required output paths (normalized).
    """
    import os
    
    def _normalize_path(path: str) -> str:
        """Normalize output path for consistency."""
        if not path:
            return path
        # Backslash to forward slash
        path = path.replace("\\", "/")
        # Known files that should be in data/
        known_files = ["metrics.json", "alignment_check.json", "scored_rows.csv", "cleaned_data.csv", "cleaning_manifest.json"]
        basename = os.path.basename(path)
        if basename in known_files and not path.startswith("data/"):
            return f"data/{basename}"
        return path
    
    if not isinstance(contract, dict):
        return []
    
    outputs = []

    def _extract_path(item: Any) -> str:
        if not item:
            return ""
        if isinstance(item, dict):
            path = item.get("path") or item.get("output") or item.get("artifact")
            return str(path) if path else ""
        return str(item)
    
    # From artifact_requirements
    artifacts = get_artifact_requirements(contract)
    required_files = artifacts.get("required_files")
    if isinstance(required_files, list):
        for entry in required_files:
            path = _extract_path(entry)
            if path and is_probably_path(path):
                outputs.append(path)
    required_plots = artifacts.get("required_plots")
    if isinstance(required_plots, list):
        for entry in required_plots:
            path = _extract_path(entry)
            if path and is_probably_path(path):
                outputs.append(path)
    
    # From top-level required_outputs
    top_level = contract.get("required_outputs")
    if isinstance(top_level, list):
        for entry in top_level:
            path = _extract_path(entry)
            if path and is_probably_path(path):
                outputs.append(path)
    
    # Normalize and deduplicate
    seen = set()
    result = []
    for o in outputs:
        norm = _normalize_path(o)
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result
