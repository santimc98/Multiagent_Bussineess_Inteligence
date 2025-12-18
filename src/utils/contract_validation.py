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
        "Do not import sys.",
        "Do not drop required/derived columns solely for being constant; record as constant in manifest.",
    ],
    "must_not": [
        "Do not blindly strip '.'; infer thousands/decimal from patterns.",
        "Do not create downstream ML artifacts (weights/metrics); only cleaned_data.csv + cleaning_manifest.json.",
    ],
    "safe_idioms": [
        "For ratios of boolean patterns use mask.mean(); avoid sum(mask.sum()) or sum(<scalar>).",
        "Avoid double sum: never call sum(x.sum()) on a scalar; aggregate with .mean() or divide by len(mask).",
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
    "must": [
        "Use dialect/output_dialect from manifest when loading data.",
        "Honor allowlist dependencies; do not import banned packages (pulp/cvxpy/fuzzywuzzy/torch/tensorflow/etc.).",
        "Include variance guard: if y.nunique() <= 1 raise ValueError.",
        "Print Mapping Summary and build X only from contract feature columns.",
        "Ensure output dirs exist (data/, static/plots/) before saving artifacts.",
    ],
    "must_not": [
        "Do not import sys.",
        "Do not add noise/jitter to target.",
    ],
    "safe_idioms": [
        "For optimization/LP prefer scipy.optimize.linprog/minimize; avoid pulp/cvxpy.",
        "For fuzzy matching prefer difflib; use rapidfuzz only if contract requests dependency.",
    ],
    "methodology": {
        "ranking_loss": "Use ranking-aware loss for ordinal scoring when applicable.",
        "regularization": "Add L2/concentration penalty to avoid degenerate weights.",
    },
    "validation_checklist": [
        "Run leakage/variance checks before training.",
        "Report HHI/max weight/near-zero weights for scoring weights.",
        "Print QA_SELF_CHECK with satisfied checklist items.",
    ],
    "outputs": {
        "required": ["data/cleaned_data.csv"],
        "optional": ["data/weights.json", "static/plots/*.png"],
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
        out["goals"] = _ensure_list_of_str(rb.get("goals"), default.get("goals", []))
        out["must"] = _ensure_list_of_str(rb.get("must"), default.get("must", []))
        out["must_not"] = _ensure_list_of_str(rb.get("must_not"), default.get("must_not", []))
        out["safe_idioms"] = _ensure_list_of_str(rb.get("safe_idioms"), default.get("safe_idioms", []))
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
