"""
Contract Validation Utilities - V4.1 Compatible.

This module provides default runbook configurations for data engineer and ML engineer agents.
These are used as defaults when contracts don't specify custom runbooks.
"""

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
        "Derive required columns from preprocessing_requirements and ensure they exist in the output.",
        "Perform a post-cleaning self-audit: for each required column report dtype and null_frac only.",
        "Do not import sys.",
        "Preserve exact canonical_name strings (including spaces/symbols); do not normalize away punctuation.",
        "Do not drop required/derived columns solely for being constant; record as constant in manifest.",
        "When deriving contract columns, use the provided canonical_name mapping and preserve the exact header names.",
        "If a required source column for derivations is missing, raise a clear ValueError (do not default all rows).",
        "Do not validate required columns before canonicalization; match after normalization mapping.",
        "Only enforce existence for source='input' columns; source='derived' must be created after mapping.",
        "Ensure manifest JSON serialization handles numpy/pandas scalar types safely.",
        "If a derived column has derived_owner='ml_engineer', do not create a placeholder; leave it absent for downstream derivation.",
        "For numeric parsing, sanitize symbols first: strip currency/letters and keep only digits, sign, separators, parentheses, and % before conversion.",
        "If a required numeric column becomes mostly NaN after conversion while raw_non_null_frac is high, treat it as a parsing failure and adjust parsing (do not silently proceed).",
    ],
    "must_not": [
        "Do not blindly strip '.'; infer thousands/decimal from patterns.",
        "Do not assume 'parse_numeric' means 'no currency'—required numeric columns may include currency symbols or text prefixes; sanitize before conversion.",
        "Do not leave numeric columns as object when expected_kind is numeric; fix or abort with clear error.",
        "Do not create downstream ML artifacts (weights/metrics); only cleaned_data.csv + cleaning_manifest.json.",
        "Do not compute advanced validation metrics (MAE/correlation/score) in cleaning; leave analytics to ML/QA.",
        "Do not fabricate constant placeholders for derived grouping/segment columns without a formula or depends_on.",
    ],
    "safe_idioms": [
        "For ratios of boolean patterns use mask.mean(); avoid sum(mask.sum()) or sum(<scalar>).",
        "Avoid double sum: never call sum(x.sum()) on a scalar; aggregate with .mean() or divide by len(mask).",
        "Robust number parsing idiom: s=str(x); s=re.sub(r\"[^0-9,\\.\\-+()%\\s]\",\"\",s).strip(); handle parentheses negatives; infer decimal by last separator; remove thousands separators; then float(s).",
    ],
    "reasoning_checklist": [
        "Use canonical_name (if provided) for consistent references across mapping, validation, and derivations.",
        "If canonical_name includes spaces or symbols, keep it exact when selecting columns.",
        "Verify required columns after normalization/mapping; do not treat pre-mapped absence as missing.",
        "If a numeric-looking column is typed as object/string, treat conversion as a risk before comparisons/normalization.",
        "If the dialect indicates decimal=',' and raw samples show dots, treat dots as thousands unless evidence suggests decimals.",
        "If raw samples contain currency symbols or prefixes, ensure numeric parsing removes them before float conversion.",
        "If data_risks mention canonicalization collisions, ensure column selection remains unambiguous.",
        "If normalization causes name collisions, choose deterministically and log a warning for traceability.",
        "If conversion yields too many NaN, revert and log instead of dropping required columns.",
        "If derived columns are required, confirm source inputs exist and document any NA handling assumptions.",
        "If derived_owner indicates ML ownership, defer derivation and document that it will be created later.",
        "When checking dtype on a selected column, handle duplicate labels consistently and log the choice.",
        "If referencing contract/config content in code, ensure it is valid Python (JSON null/true/false must be handled).",
    ],
    "validation_checklist": [
        "Print CLEANING_VALIDATION with dtype and null_frac only; validation must not raise.",
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
        "If contract contains column_roles/artifact_requirements, treat them as source-of-truth for outputs.",
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
        "Include feature_usage in alignment_check.json (used_features, target_columns, excluded_features, reason_exclusions).",
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
        "Treat column_roles and artifact_requirements as source-of-truth; do not invent outputs.",
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
        "optional": [
            "data/weights.json",
            "static/plots/*.png",
            "data/scored_rows.csv",
            "data/alignment_check.json",
            "data/plot_insights.json",
        ],
    },
}
