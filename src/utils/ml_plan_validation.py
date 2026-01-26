
from typing import Dict, Any, List


def _extract_contract_primary_metric(contract: Dict[str, Any]) -> str | None:
    """Extract primary_metric from contract (validation_requirements or evaluation_spec)."""
    # Priority 1: validation_requirements.primary_metric
    val_req = contract.get("validation_requirements", {})
    if isinstance(val_req, dict) and val_req.get("primary_metric"):
        return str(val_req["primary_metric"]).lower()

    # Priority 2: evaluation_spec.primary_metric
    eval_spec = contract.get("evaluation_spec", {})
    if isinstance(eval_spec, dict) and eval_spec.get("primary_metric"):
        return str(eval_spec["primary_metric"]).lower()

    return None


def validate_ml_plan_constraints(
    plan: Dict[str, Any],
    data_profile: Dict[str, Any],
    contract: Dict[str, Any],
    strategy: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate ML Plan against universal constraints.

    This validates:
    1. Plan source validity
    2. Outcome missingness vs training_rows_policy
    3. Metric vs analysis_type consistency
    4. Contract primary_metric vs plan primary_metric
    5. Split candidates evaluation (warning if ignored without justification)
    6. evidence_used coherence with data_profile/contract
    """
    violations = []
    warnings = []

    # constraint 1: Check plan source
    source = str(plan.get("plan_source", "")).lower()
    if source.startswith("missing_") or source == "fallback":
        violations.append(f"ML_PLAN_INVALID_SOURCE: Plan source is '{source}', which indicates valid LLM generation failed.")

    # constraint 2: Outcome missingness vs training_rows_policy
    outcome_analysis = data_profile.get("outcome_analysis", {})
    outcome_null_fracs = {}
    has_null_labels = False
    for col, analysis in outcome_analysis.items():
        if isinstance(analysis, dict) and analysis.get("present"):
            null_frac = analysis.get("null_frac", 0)
            outcome_null_fracs[col] = null_frac
            if null_frac > 0:
                has_null_labels = True

    policy = plan.get("training_rows_policy", "unspecified")
    train_filter = plan.get("train_filter") if isinstance(plan.get("train_filter"), dict) else {}
    tf_type = str(train_filter.get("type") or "").strip().lower()
    if has_null_labels and (policy == "use_all_rows" or tf_type == "none"):
        violations.append(
            f"ML_PLAN_CONSTRAINT_VIOLATION: Outcome has missing values "
            f"(null_fracs={outcome_null_fracs}), but policy is 'use_all_rows'. Must filter rows."
        )

    # constraint 3: Metric vs Analysis Type
    analysis_type = str(strategy.get("analysis_type", "")).lower()
    metric_policy = plan.get("metric_policy", {})
    primary_metric = str(metric_policy.get("primary_metric", "unspecified")).lower()

    classification_metrics = {"roc_auc", "accuracy", "f1", "precision", "recall", "log_loss", "balanced_accuracy"}
    regression_metrics = {"rmse", "mae", "r2", "mse", "mape", "rmsle"}

    if "classif" in analysis_type:
        if primary_metric not in classification_metrics and primary_metric != "unspecified":
            violations.append(f"ML_PLAN_METRIC_MISMATCH: Analysis is classification, but metric '{primary_metric}' is not a valid classification metric.")
    elif "regres" in analysis_type:
        if primary_metric not in regression_metrics and primary_metric != "unspecified":
            violations.append(f"ML_PLAN_METRIC_MISMATCH: Analysis is regression, but metric '{primary_metric}' is not a valid regression metric.")

    # constraint 4: Contract primary_metric vs plan primary_metric
    contract_metric = _extract_contract_primary_metric(contract)
    if contract_metric and primary_metric != "unspecified":
        if contract_metric != primary_metric:
            violations.append(
                f"ML_PLAN_METRIC_CONTRACT_MISMATCH: Contract specifies primary_metric='{contract_metric}', "
                f"but plan uses '{primary_metric}'. Plan must use contract metric."
            )

    # constraint 5: Split candidates evaluation
    split_candidates = data_profile.get("split_candidates", [])
    if split_candidates:
        split_col_names = [sc.get("column", "") for sc in split_candidates if isinstance(sc, dict)]
        evidence_used = plan.get("evidence_used", {})
        split_evaluation = evidence_used.get("split_evaluation", "")

        # If there are split candidates but plan doesn't use split and doesn't explain why
        if policy != "use_split_column" and not split_evaluation:
            warnings.append(
                f"ML_PLAN_SPLIT_NOT_EVALUATED: Data has split_candidates {split_col_names}, "
                f"but plan policy is '{policy}' without explanation. "
                f"Set evidence_used.split_evaluation to explain why split was not used."
            )

    # constraint 6: evidence_used coherence
    evidence_used = plan.get("evidence_used", {})
    if evidence_used:
        # Check outcome_null_frac coherence
        eu_outcome = evidence_used.get("outcome_null_frac", {})
        if isinstance(eu_outcome, dict):
            eu_col = eu_outcome.get("column")
            eu_null_frac = eu_outcome.get("null_frac")
            if eu_col and eu_col in outcome_null_fracs:
                actual_null_frac = outcome_null_fracs[eu_col]
                if eu_null_frac is not None and abs(float(eu_null_frac) - actual_null_frac) > 0.01:
                    violations.append(
                        f"ML_PLAN_EVIDENCE_CONTRADICTION: evidence_used.outcome_null_frac={eu_null_frac} "
                        f"but data_profile shows {eu_col}.null_frac={actual_null_frac}"
                    )

        # Check contract_primary_metric coherence
        eu_contract_metric = evidence_used.get("contract_primary_metric")
        if eu_contract_metric and contract_metric:
            if str(eu_contract_metric).lower() != contract_metric:
                violations.append(
                    f"ML_PLAN_EVIDENCE_CONTRADICTION: evidence_used.contract_primary_metric='{eu_contract_metric}' "
                    f"but contract actually specifies '{contract_metric}'"
                )

    ok = len(violations) == 0
    return {
        "ok": ok,
        "violations": violations,
        "warnings": warnings,
        "corrected_plan": plan
    }

def validate_plan_code_coherence(ml_plan: Dict[str, Any], code: str, data_profile: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Verify if the generated code respects the ML Plan policies.
    """
    violations = []
    policy = ml_plan.get("training_rows_policy", "")
    code_lower = code.lower()
    train_filter = ml_plan.get("train_filter") if isinstance(ml_plan.get("train_filter"), dict) else {}

    # 1. Label Filtering
    tf_type = str(train_filter.get("type") or "").strip().lower()
    if tf_type == "label_not_null" or policy == "only_rows_with_label":
        # Code must filter for notna() or dropna() on the target
        # Heuristic check
        if "notna()" not in code and "dropna(" not in code and "isnull()" not in code_lower:
             # Basic heuristic: if policy requires filter, code must contain filter logic
             violations.append("training_rows_policy mismatch: Plan requires 'only_rows_with_label' but code doesn't seem to filter missings (no notna/dropna)")
    
    # 2. Split Column
    if tf_type == "split_equals" or policy == "use_split_column":
        split_col = ml_plan.get("split_column", "unknown_col")
        if split_col and split_col not in code:
             violations.append(f"split_column mismatch: Plan requires using '{split_col}' but it is not referenced in code.")
             
    passed = len(violations) == 0
    return {
        "passed": passed,
        "status": "APPROVED" if passed else "REJECTED",
        "violations": violations
    }

def validate_plan_data_coherence(
    ml_plan: Dict[str, Any],
    data_profile: Dict[str, Any],
    contract: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Verify if ML Plan is coherent with Data Profile facts and contract.

    Universal checks (not dataset-specific):
    1. Label filter policy vs actual missingness
    2. Split column vs available candidates
    3. Contract metric vs plan metric (if contract provided)
    4. evidence_used vs actual data_profile facts
    """
    inconsistencies = []
    warnings = []
    contract = contract or {}

    # 1. Label Filter vs Missingness
    policy = ml_plan.get("training_rows_policy", "")
    train_filter = ml_plan.get("train_filter") if isinstance(ml_plan.get("train_filter"), dict) else {}
    tf_type = str(train_filter.get("type") or "").strip().lower()
    outcome_analysis = data_profile.get("outcome_analysis", {})

    if tf_type == "label_not_null" or policy == "only_rows_with_label":
        # Check if missingness actually exists
        has_nulls = False
        for col, analysis in outcome_analysis.items():
            if isinstance(analysis, dict) and analysis.get("null_frac", 0) > 0:
                has_nulls = True
                break
        if not has_nulls:
            inconsistencies.append("Plan requires 'only_rows_with_label' but data has 0% outcome missingness.")

    # Inverse check: if there ARE nulls and policy is use_all_rows
    if tf_type == "none" or policy == "use_all_rows":
        for col, analysis in outcome_analysis.items():
            if isinstance(analysis, dict) and analysis.get("present") and analysis.get("null_frac", 0) > 0:
                inconsistencies.append(
                    f"Plan uses 'use_all_rows' but outcome '{col}' has {analysis.get('null_frac', 0)*100:.1f}% missing labels. "
                    f"Cannot train on rows without labels."
                )

    # 2. Split Column vs Candidates
    if tf_type == "split_equals" or policy == "use_split_column":
        split_col = ml_plan.get("split_column")
        if split_col:
            candidates = [c.get("column") for c in data_profile.get("split_candidates", []) if isinstance(c, dict)]
            if split_col not in candidates:
                inconsistencies.append(f"Plan uses split_column '{split_col}' but it is not in split_candidates {candidates}.")
        else:
            inconsistencies.append("Plan uses 'use_split_column' policy but split_column is not specified.")

    # 3. Contract metric vs plan metric
    contract_metric = _extract_contract_primary_metric(contract)
    metric_policy = ml_plan.get("metric_policy", {})
    plan_metric = str(metric_policy.get("primary_metric", "")).lower()

    if contract_metric and plan_metric and plan_metric != "unspecified":
        if contract_metric != plan_metric:
            inconsistencies.append(
                f"Contract requires metric='{contract_metric}' but plan uses '{plan_metric}'."
            )

    # 4. evidence_used coherence
    evidence_used = ml_plan.get("evidence_used", {})
    if evidence_used and isinstance(evidence_used, dict):
        # Check split_candidates coherence
        eu_splits = evidence_used.get("split_candidates", [])
        actual_splits = data_profile.get("split_candidates", [])
        actual_split_cols = {sc.get("column") for sc in actual_splits if isinstance(sc, dict)}

        for eu_split in eu_splits if isinstance(eu_splits, list) else []:
            if isinstance(eu_split, dict):
                eu_col = eu_split.get("column")
                if eu_col and eu_col not in actual_split_cols:
                    inconsistencies.append(
                        f"evidence_used references split_candidate '{eu_col}' but it's not in data_profile.split_candidates."
                    )

    # 5. Leakage policy coherence
    leakage_flags = data_profile.get("leakage_flags", [])
    leakage_policy = ml_plan.get("leakage_policy", {})
    if leakage_flags and leakage_policy.get("action") == "none":
        flagged_cols = [lf.get("column") for lf in leakage_flags if isinstance(lf, dict)]
        warnings.append(
            f"Data has leakage_flags for columns {flagged_cols} but plan leakage_policy.action='none'. "
            f"Consider 'exclude_flagged_columns' unless these are intentionally kept."
        )

    passed = len(inconsistencies) == 0
    return {
        "passed": passed,
        "inconsistencies": inconsistencies,
        "warnings": warnings,
    }

def run_full_coherence_validation(
    ml_plan: Dict[str, Any],
    code: str,
    data_profile: Dict[str, Any],
    contract: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Aggregate all coherence checks.

    Args:
        ml_plan: The ML plan to validate
        code: Generated code string
        data_profile: Data profile facts
        contract: Optional execution contract for metric validation

    Returns:
        Dict with passed status, violations (hard failures), and warnings (soft issues)
    """
    code_res = validate_plan_code_coherence(ml_plan, code, data_profile)
    data_res = validate_plan_data_coherence(ml_plan, data_profile, contract)

    violations = code_res.get("violations", [])
    inconsistencies = data_res.get("inconsistencies", [])
    warnings = data_res.get("warnings", [])

    # Inconsistencies are treated as violations (hard failures)
    all_violations = violations + inconsistencies

    return {
        "passed": len(all_violations) == 0,
        "violations": all_violations,
        "warnings": warnings,
    }
