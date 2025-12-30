import copy
from typing import Any, Dict, List


DEFAULT_DATA_ENGINEER_RUNBOOK: Dict[str, Any] = {
    "version": "1",
    "goals": [
        "Deliver a cleaned dataset and manifest that respect the execution contract dialect, roles, and expected kinds.",
        "Preserve required and potentially useful columns; avoid destructive conversions.",
    ],
    "must": [
        "First pd.read_csv must use dialect variables (sep/decimal/encoding) from contract/manifest; do not hardcode literals.",
        "Respect expected_kind: numeric -> pd.to_numeric; datetime -> pd.to_datetime; categorical -> keep as string.",
        "Canonical columns must contain the cleaned values; do not leave raw strings in canonical columns while writing cleaned_* shadows.",
        "If you create cleaned_* helper columns, overwrite the canonical column with cleaned values before saving.",
        "Derive required columns from spec_extraction and ensure they exist in the output.",
        "Perform a post-cleaning self-audit: for each required column report dtype, null_frac, and basic range checks vs expected_range.",
        "Do not import sys.",
        "Preserve exact canonical_name strings (including spaces/symbols); do not normalize away punctuation.",
        "Do not drop required/derived columns solely for being constant; record as constant in manifest.",
        "When deriving contract columns, use the provided canonical_name mapping and preserve the exact header names.",
        "If a required source column for derivations is missing, raise a clear ValueError (do not default all rows).",
        "Do not validate required columns before canonicalization; match after normalization mapping.",
        "Only enforce existence for source='input' columns; source='derived' must be created after mapping.",
        "Ensure _json_default handles numpy scalar types (np.bool_, np.integer, np.floating) before json.dump.",
        "If a derived column has derived_owner='ml_engineer', do not create a placeholder; leave it absent for downstream derivation.",
    ],
    "must_not": [
        "Do not blindly strip '.'; infer thousands/decimal from patterns.",
        "Do not leave numeric columns as object when expected_kind is numeric; fix or abort with clear error.",
        "Do not create downstream ML artifacts (weights/metrics); only cleaned_data.csv + cleaning_manifest.json.",
        "Do not fabricate constant placeholders for derived grouping/segment columns without a formula or depends_on.",
    ],
    "safe_idioms": [
        "For ratios of boolean patterns use mask.mean(); avoid sum(mask.sum()) or sum(<scalar>).",
        "Avoid double sum: never call sum(x.sum()) on a scalar; aggregate with .mean() or divide by len(mask).",
    ],
    "reasoning_checklist": [
        "Use canonical_name (if provided) for consistent references across mapping, validation, and derivations.",
        "If canonical_name includes spaces or symbols, keep it exact when selecting columns.",
        "Verify required columns after normalization/mapping; do not treat pre-mapped absence as missing.",
        "If a numeric-looking column is typed as object/string, treat conversion as a risk before comparisons/normalization.",
        "If the dialect indicates decimal=',' and raw samples show dots, treat dots as thousands unless evidence suggests decimals.",
        "If data_risks mention canonicalization collisions, ensure column selection remains unambiguous.",
        "If normalization causes name collisions, choose deterministically and log a warning for traceability.",
        "If conversion yields too many NaN, revert and log instead of dropping required columns.",
        "If derived columns are required, confirm source inputs exist and document any NA handling assumptions.",
        "If derived_owner indicates ML ownership, defer derivation and document that it will be created later.",
        "When checking dtype on a selected column, handle duplicate labels consistently and log the choice.",
        "If referencing contract/config content in code, ensure it is valid Python (JSON null/true/false must be handled).",
    ],
    "validation_checklist": [
        "Print CLEANING_VALIDATION with null_frac and range/type checks; validation must not raise.",
        "Use norm() helper to match required columns case/spacing insensitive.",
        "Ensure manifest json.dump uses default=_json_default.",
    ],
    "manifest_requirements": {
        "include": ["input_dialect", "output_dialect", "column_mapping", "dropped_columns", "conversions", "conversions_meta", "type_checks"],
    },
}


DEFAULT_ML_ENGINEER_RUNBOOK: Dict[str, Any] = {
    "version": "1",
    "goals": [
        "Train/evaluate models aligned to the strategy and execution contract.",
        "Produce interpretable outputs and required artifacts without leakage.",
    ],
    "spec_extraction": {},
    "must": [
        "Use dialect/output_dialect from manifest when loading data.",
        "Honor allowlist dependencies; do not import banned packages (pulp/cvxpy/fuzzywuzzy/torch/tensorflow/etc.).",
        "If contract.spec_extraction is present, treat it as source-of-truth for formulas, constraints, cases, and deliverables.",
        "Include variance guard: if y.nunique() <= 1 raise ValueError.",
        "Print Mapping Summary and build X only from contract feature columns.",
        "Ensure output dirs exist (data/, static/plots/) before saving artifacts.",
        "If derived columns exist in cleaned data, do not recompute or overwrite; only derive if missing and preserve NaNs unless the contract explicitly sets a default.",
        "If a baseline_metric exists, compare it to the computed score (e.g., correlation/MAE) and report the result.",
        "If derived outputs are present in the contract, save per-row scored output to data/scored_rows.csv.",
        "When writing JSON artifacts, use json.dump(..., default=_json_default) to handle numpy/pandas types.",
        "If contract includes decision_variables, treat them as decision inputs (not leakage by default) and document any selection-bias risks.",
        "If contract includes missing_sentinels, treat sentinel values as missing during modeling and consider adding an observed-flag feature.",
        "If contract includes alignment_requirements, write data/alignment_check.json with PASS/WARN/FAIL and failure_mode.",
        "Include per-requirement evidence in alignment_check.json (metrics, artifacts, or log excerpts).",
    ],
    "must_not": [
        "Do not import sys.",
        "Do not add noise/jitter to target.",
        "Do not treat decision_variables as forbidden leakage unless the contract explicitly marks them post-outcome with no decision context.",
    ],
    "safe_idioms": [
        "For optimization/LP prefer scipy.optimize.linprog/minimize; avoid pulp/cvxpy.",
        "For fuzzy matching prefer difflib; use rapidfuzz only if contract requests dependency.",
    ],
    "reasoning_checklist": [
        "Use canonical_name (if provided) for consistent references across agents.",
        "Treat spec_extraction as source-of-truth; do not invent formulas, cases, or constraints.",
        "If target_type is ordinal/ranking, avoid predictive regression as the primary objective.",
        "Validate weight constraints and explain any regularization choices.",
        "Ensure outputs satisfy the explicitly requested deliverables.",
        "If decision_variables exist, explain how elasticity/optimization uses them and whether they are observed for all rows.",
        "If missing_sentinels exist, ensure sentinel handling does not bias training or scoring.",
    ],
    "methodology": {
        "ranking_loss": "Use ranking-aware loss for ordinal scoring when applicable.",
        "regularization": "Add L2/concentration penalty to avoid degenerate weights.",
    },
    "validation_checklist": [
        "Run leakage/variance checks before training.",
        "Report HHI/max weight/near-zero weights for scoring weights.",
        "Print QA_SELF_CHECK with satisfied checklist items.",
        "Print ALIGNMENT_CHECK with status and ensure alignment_check.json exists when required.",
    ],
    "outputs": {
        "required": ["data/cleaned_data.csv"],
        "optional": ["data/weights.json", "static/plots/*.png", "data/scored_rows.csv", "data/alignment_check.json"],
    },
}


def _ensure_list_of_str(val: Any, default: List[str]) -> List[str]:
    if isinstance(val, list) and all(isinstance(x, str) for x in val) and val:
        return val
    return list(default)


def ensure_role_runbooks(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the contract contains role_runbooks with minimal schema.
    If missing or malformed, inject sensible defaults.
    """
    if not isinstance(contract, dict):
        return {"role_runbooks": {"data_engineer": copy.deepcopy(DEFAULT_DATA_ENGINEER_RUNBOOK), "ml_engineer": copy.deepcopy(DEFAULT_ML_ENGINEER_RUNBOOK)}}

    runbooks = contract.get("role_runbooks")
    if not isinstance(runbooks, dict):
        runbooks = {}

    def _normalize(rb: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(default)
        if not isinstance(rb, dict):
            return out
        out["version"] = str(rb.get("version", default.get("version", "1")))
        if "spec_extraction" in default:
            spec = rb.get("spec_extraction", default.get("spec_extraction", {}))
            out["spec_extraction"] = spec if isinstance(spec, dict) else copy.deepcopy(default.get("spec_extraction", {}))
        out["goals"] = _ensure_list_of_str(rb.get("goals"), default.get("goals", []))
        out["must"] = _ensure_list_of_str(rb.get("must"), default.get("must", []))
        out["must_not"] = _ensure_list_of_str(rb.get("must_not"), default.get("must_not", []))
        out["safe_idioms"] = _ensure_list_of_str(rb.get("safe_idioms"), default.get("safe_idioms", []))
        out["reasoning_checklist"] = _ensure_list_of_str(rb.get("reasoning_checklist"), default.get("reasoning_checklist", []))
        if default.get("reasoning_checklist") and out.get("reasoning_checklist") is not None:
            merged = list(out["reasoning_checklist"])
            for item in default.get("reasoning_checklist", []):
                if item not in merged:
                    merged.append(item)
            out["reasoning_checklist"] = merged
        out["validation_checklist"] = _ensure_list_of_str(rb.get("validation_checklist"), default.get("validation_checklist", []))
        mr = rb.get("manifest_requirements", default.get("manifest_requirements", {}))
        out["manifest_requirements"] = mr if isinstance(mr, dict) else copy.deepcopy(default.get("manifest_requirements", {}))
        meth = rb.get("methodology", default.get("methodology", {}))
        out["methodology"] = meth if isinstance(meth, dict) else copy.deepcopy(default.get("methodology", {}))
        outs = rb.get("outputs", default.get("outputs", {}))
        out["outputs"] = outs if isinstance(outs, dict) else copy.deepcopy(default.get("outputs", {}))
        return out

    runbooks["data_engineer"] = _normalize(runbooks.get("data_engineer", {}), DEFAULT_DATA_ENGINEER_RUNBOOK)
    runbooks["ml_engineer"] = _normalize(runbooks.get("ml_engineer", {}), DEFAULT_ML_ENGINEER_RUNBOOK)
    contract["role_runbooks"] = runbooks
    return contract


def validate_spec_extraction_structure(contract: Dict[str, Any]) -> List[str]:
    """
    Deterministic, structure-only validation for spec_extraction.
    Returns a list of issues; empty list means structure is acceptable.
    """
    issues: List[str] = []
    if not isinstance(contract, dict):
        return ["contract_not_dict"]
    spec = contract.get("spec_extraction")
    if not isinstance(spec, dict):
        return ["spec_extraction_missing_or_not_dict"]

    list_fields = ["derived_columns", "case_taxonomy", "constraints", "deliverables"]
    for field in list_fields:
        val = spec.get(field)
        if val is None:
            continue
        if not isinstance(val, list):
            issues.append(f"spec_extraction.{field}_not_list")
            continue
        if field in {"derived_columns", "case_taxonomy"}:
            for idx, item in enumerate(val):
                if not isinstance(item, dict):
                    issues.append(f"spec_extraction.{field}[{idx}]_not_object")
        else:
            for idx, item in enumerate(val):
                if not isinstance(item, (str, dict)):
                    issues.append(f"spec_extraction.{field}[{idx}]_invalid_type")

    for field in ["scoring_formula", "target_type", "leakage_policy"]:
        val = spec.get(field)
        if val is None:
            continue
        if not isinstance(val, str):
            issues.append(f"spec_extraction.{field}_not_string")

    return issues
