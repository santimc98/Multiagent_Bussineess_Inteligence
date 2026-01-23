
from typing import Dict, Any, List
from datetime import datetime, timezone

# Tokens that indicate a column might be a split/fold indicator
SPLIT_CANDIDATE_TOKENS = {"split", "set", "fold", "train", "test", "partition", "is_train", "is_test"}


def convert_dataset_profile_to_data_profile(
    dataset_profile: Dict[str, Any],
    contract: Dict[str, Any],
    analysis_type: str | None = None,
) -> Dict[str, Any]:
    """
    Convert a dataset_profile.json (Steward output) to data_profile schema.

    This is the CANONICAL evidence conversion: dataset_profile is the source of truth,
    and we derive data_profile from it using the contract for context.

    Args:
        dataset_profile: The Steward-generated dataset profile (rows, cols, missing_frac, cardinality, etc.)
        contract: Execution contract with outcome_columns, column_roles, validation_requirements
        analysis_type: Optional analysis type (classification, regression)

    Returns:
        data_profile dict with the standard schema:
        - basic_stats, dtypes, missingness_top30, outcome_analysis, split_candidates,
        - constant_columns, high_cardinality_columns, leakage_flags, schema_version, generated_at
    """
    contract = contract or {}
    dataset_profile = dataset_profile or {}

    n_rows = int(dataset_profile.get("rows", 0))
    n_cols = int(dataset_profile.get("cols", 0))
    columns = list(dataset_profile.get("columns", []))

    # 1. basic_stats
    basic_stats = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
    }

    # 2. dtypes from type_hints (map to dtype-like strings)
    type_hints = dataset_profile.get("type_hints", {})
    dtypes = {}
    for col in columns:
        hint = type_hints.get(col, "object")
        # Map type_hints to pandas-like dtypes for consistency
        if hint == "numeric":
            dtypes[col] = "float64"
        elif hint == "categorical":
            dtypes[col] = "object"
        elif hint == "datetime":
            dtypes[col] = "datetime64"
        elif hint == "boolean":
            dtypes[col] = "bool"
        else:
            dtypes[col] = "object"

    # 3. missingness_top30 from missing_frac
    missing_frac = dataset_profile.get("missing_frac", {})
    sorted_miss = sorted(missing_frac.items(), key=lambda x: x[1], reverse=True)
    missingness_top30 = {k: round(v, 4) for k, v in sorted_miss[:30]}

    # 4. outcome_analysis from contract + missing_frac + cardinality
    outcome_analysis = {}
    outcome_cols = _extract_outcome_columns(contract)
    cardinality = dataset_profile.get("cardinality", {})

    for outcome_col in outcome_cols:
        if outcome_col not in columns:
            outcome_analysis[outcome_col] = {"present": False, "error": "column_not_found"}
            continue

        null_frac = missing_frac.get(outcome_col, 0.0)
        non_null_count = int(n_rows * (1 - null_frac)) if n_rows > 0 else 0

        analysis_entry = {
            "present": True,
            "non_null_count": non_null_count,
            "total_count": n_rows,
            "null_frac": round(null_frac, 4),
        }

        # Determine inferred_type from cardinality or analysis_type
        card_info = cardinality.get(outcome_col, {})
        n_unique = card_info.get("unique", 0)
        inferred = analysis_type or ""
        if not inferred:
            inferred = "classification" if n_unique <= 20 else "regression"
        analysis_entry["inferred_type"] = inferred
        analysis_entry["n_unique"] = n_unique

        # Add class_counts for classification
        if inferred == "classification":
            top_values = card_info.get("top_values", [])
            class_counts = {}
            for tv in top_values:
                val = str(tv.get("value", ""))
                # Skip nan values in class counts
                if val.lower() != "nan":
                    class_counts[val] = int(tv.get("count", 0))
            analysis_entry["n_classes"] = len(class_counts) if class_counts else n_unique
            analysis_entry["class_counts"] = class_counts

        outcome_analysis[outcome_col] = analysis_entry

    # 5. split_candidates: detect columns with split-related names and values
    split_candidates = []
    for col in columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        tokens = set(col_lower.split())
        if tokens & SPLIT_CANDIDATE_TOKENS:
            card_info = cardinality.get(col, {})
            top_values = card_info.get("top_values", [])
            unique_values_sample = [str(tv.get("value", "")) for tv in top_values[:20]]
            split_candidates.append({
                "column": col,
                "unique_values_sample": unique_values_sample,
            })

    # 6. constant_columns: columns with unique <= 1
    constant_columns = []
    for col in columns:
        card_info = cardinality.get(col, {})
        n_unique = card_info.get("unique", 0)
        if n_unique <= 1:
            constant_columns.append(col)

    # 7. high_cardinality_columns: unique ratio > 0.95 and > 50 uniques
    high_cardinality_columns = []
    for col in columns:
        card_info = cardinality.get(col, {})
        n_unique = card_info.get("unique", 0)
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0
        if unique_ratio > 0.95 and n_unique > 50:
            high_cardinality_columns.append({
                "column": col,
                "n_unique": n_unique,
                "unique_ratio": round(unique_ratio, 4),
            })

    # 8. leakage_flags: outcome column name appears in other columns
    leakage_flags = []
    outcome_names_lower = {c.lower() for c in outcome_cols}
    for col in columns:
        col_lower = col.lower()
        for outcome in outcome_names_lower:
            if outcome in col_lower and col not in outcome_cols:
                leakage_flags.append({
                    "column": col,
                    "reason": f"name_contains_outcome:{outcome}",
                    "severity": "SOFT",
                })

    # Build the data_profile
    data_profile = {
        "basic_stats": basic_stats,
        "dtypes": dtypes,
        "missingness_top30": missingness_top30,
        "outcome_analysis": outcome_analysis,
        "split_candidates": split_candidates,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "leakage_flags": leakage_flags,
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "converted_from_dataset_profile",
    }

    return data_profile


def _extract_outcome_columns(contract: Dict[str, Any]) -> List[str]:
    """
    Extract outcome columns from contract using V4.1 accessor.

    Priority (handled by get_outcome_columns):
      1. contract["outcome_columns"]
      2. column_roles["outcome"] (supports all V4.1 formats: role->list, column->role, list of dicts)
      3. objective_analysis target fields
      4. Empty list if nothing found

    This is NOT dataset-specific: it's schema-aware extraction.
    """
    try:
        from src.utils.contract_v41 import get_outcome_columns
        outcomes = get_outcome_columns(contract)
        if outcomes:
            return outcomes
    except ImportError:
        pass

    # Legacy fallback if contract_v41 not available
    outcome_cols = []

    # Try outcome_columns first
    raw_outcomes = contract.get("outcome_columns")
    if raw_outcomes:
        if isinstance(raw_outcomes, list):
            outcome_cols = [str(c) for c in raw_outcomes if c and str(c).lower() != "unknown"]
        elif isinstance(raw_outcomes, str) and raw_outcomes.lower() != "unknown":
            outcome_cols = [raw_outcomes]

    # Fallback: column_roles["outcome"]
    if not outcome_cols:
        roles = contract.get("column_roles", {})
        if isinstance(roles, dict):
            # V4.1 format A: role -> list[str]
            outcome_from_roles = roles.get("outcome", [])
            if isinstance(outcome_from_roles, list):
                outcome_cols = [str(c) for c in outcome_from_roles if c]
            elif isinstance(outcome_from_roles, str):
                outcome_cols = [outcome_from_roles]

            # V4.1 format C: column -> role (inverted)
            if not outcome_cols:
                for col, role in roles.items():
                    if isinstance(role, str) and role.lower() == "outcome":
                        outcome_cols.append(str(col))
                    elif isinstance(role, dict) and role.get("role", "").lower() == "outcome":
                        outcome_cols.append(str(col))

    return outcome_cols


def _is_dataset_profile_schema(profile: Dict[str, Any]) -> bool:
    """Detect if profile is in dataset_profile schema (has rows/missing_frac) vs data_profile schema."""
    # dataset_profile has: rows, cols, missing_frac, cardinality
    # data_profile has: basic_stats, missingness_top30, outcome_analysis
    has_ds_keys = "rows" in profile and "missing_frac" in profile
    has_dp_keys = "basic_stats" in profile and "outcome_analysis" in profile
    return has_ds_keys and not has_dp_keys


def compact_data_profile_for_llm(
    profile: Dict[str, Any],
    max_cols: int = 60,
    contract: Dict[str, Any] | None = None,
    analysis_type: str | None = None,
) -> Dict[str, Any]:
    """
    Compact the data profile for LLM consumption.
    Retains critical decision-making facts while reducing token usage.

    Accepts either:
    - data_profile schema (basic_stats, outcome_analysis, etc.)
    - dataset_profile schema (rows, missing_frac, cardinality) - auto-converts

    Args:
        profile: Data profile or dataset profile dict
        max_cols: Max columns before summarizing dtypes
        contract: Optional contract for conversion (needed if profile is dataset_profile schema)
        analysis_type: Optional analysis type for conversion

    Returns:
        Compacted profile suitable for LLM prompts
    """
    if not isinstance(profile, dict):
        return {}

    # Auto-detect and convert dataset_profile schema if needed
    if _is_dataset_profile_schema(profile):
        if contract is None:
            # Can't convert without contract, return minimal
            return {
                "basic_stats": {"n_rows": profile.get("rows", 0), "n_cols": profile.get("cols", 0)},
                "outcome_analysis": {},
                "split_candidates": [],
                "leakage_flags": [],
                "missingness_top30": dict(list(profile.get("missing_frac", {}).items())[:30]),
                "_warning": "dataset_profile detected but no contract provided for conversion",
            }
        profile = convert_dataset_profile_to_data_profile(profile, contract, analysis_type)

    compact = {}

    # 1. Basic Stats (Critical)
    compact["basic_stats"] = profile.get("basic_stats", {})

    # 2. Outcome Analysis (Critical)
    compact["outcome_analysis"] = profile.get("outcome_analysis", {})

    # 3. Split Candidates (Critical for training rows policy)
    compact["split_candidates"] = profile.get("split_candidates", [])

    # 4. Leakage Flags (Critical for leakage policy)
    compact["leakage_flags"] = profile.get("leakage_flags", [])

    # 5. Missingness (Top 30 only)
    compact["missingness_top30"] = profile.get("missingness_top30", {})

    # 6. Constant columns (useful for feature exclusion)
    compact["constant_columns"] = profile.get("constant_columns", [])

    # 7. High cardinality columns (useful for ID detection)
    compact["high_cardinality_columns"] = profile.get("high_cardinality_columns", [])

    # 8. Column DTypes - Simplify
    dtypes = profile.get("dtypes", {})
    if len(dtypes) > max_cols:
        # Too many columns, summarize
        type_counts = {}
        for col, dtype in dtypes.items():
            t = str(dtype)
            type_counts[t] = type_counts.get(t, 0) + 1
        compact["dtypes_summary"] = type_counts
        compact["dtypes_note"] = f"Total {len(dtypes)} columns. Showing only summary."
    else:
        compact["dtypes"] = dtypes

    return compact
