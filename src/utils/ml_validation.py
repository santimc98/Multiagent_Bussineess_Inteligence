import ast
import re
from typing import Dict, Any, List

def validate_decision_variable_isolation(
    code: str,
    execution_contract: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validates appropriate use of decision variables based on problem type.

    CONTEXT-AWARE VALIDATION (reads from execution_contract):
    - For PRICE OPTIMIZATION (maximize revenue = price × P(success|price)):
      Decision variable MUST be in features to model price sensitivity
    - For RESOURCE ALLOCATION (assign X units to maximize outcome):
      Decision variable should NOT be in features (not causal)

    Args:
        code: Generated Python code to validate
        execution_contract: Contract with objective_analysis and column_roles

    Returns:
        {
            "passed": bool,
            "error_message": str,  # Empty if passed
            "violated_variables": List[str]  # Decision vars found in features (if violation)
        }
    """
    # 1. Identify problem type from contract
    obj_analysis = execution_contract.get("objective_analysis", {})
    problem_type = str(obj_analysis.get("problem_type", "")).lower()
    decision_var = obj_analysis.get("decision_variable")
    success_criteria = str(obj_analysis.get("success_criteria", "")).lower()

    # If not an optimization problem, skip this check
    if problem_type != "optimization":
        return {"passed": True, "error_message": "", "violated_variables": []}

    # 2. Determine if decision variable SHOULD be in features (context-aware)
    # For PRICE/DISCOUNT optimization where we model elasticity (price → probability),
    # decision variable MUST be in features
    is_price_optimization = any(kw in success_criteria for kw in [
        "price *", "revenue", "expected value", "elasticity", "conversion probability"
    ])

    if is_price_optimization and decision_var:
        # For price optimization, decision variable SHOULD be in model_features
        # We're modeling P(success | price, ...) so price must be a feature
        # This is NOT leakage - it's the causal mechanism we're modeling
        return {"passed": True, "error_message": "", "violated_variables": []}

    # 3. For other optimization types, extract decision variables from contract
    # Use column_roles (v4.1) or fallback to feature_availability (legacy)
    decision_vars = []

    column_roles = execution_contract.get("column_roles", {})
    if isinstance(column_roles, dict):
        for col_name, col_info in column_roles.items():
            if isinstance(col_info, dict):
                role = str(col_info.get("role", "")).lower()
                if role == "decision":
                    decision_vars.append(col_name)

    # Fallback to legacy feature_availability if column_roles not found
    if not decision_vars:
        feature_availability = execution_contract.get("feature_availability", [])
        if isinstance(feature_availability, list):
            for item in feature_availability:
                if isinstance(item, dict):
                    avail = str(item.get("availability", "")).lower()
                    if avail in ["decision", "post-decision"]:
                        col = item.get("column")
                        if col:
                            decision_vars.append(col)

    if not decision_vars:
        return {"passed": True, "error_message": "", "violated_variables": []}

    # 3. Scan code for these variables used as features
    violated_variables = []
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            # A) Look for assignments to variable names naming features
            # e.g., features = ['F1', 'price']
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    target_id = ""
                    if isinstance(target, ast.Name):
                        target_id = target.id.lower()
                    elif isinstance(target, ast.Attribute):
                        target_id = target.attr.lower()
                    
                    # Pattern-based matching for feature variable names
                    if any(kw in target_id for kw in ["feature", "x_col", "predictor", "model_cols"]):
                        # Inspect the value being assigned
                        for child in ast.walk(node.value):
                            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                                if child.value in decision_vars:
                                    violated_variables.append(child.value)
                                    
            # B) Look for direct indexing/selection like df[['a', 'price']] or df.loc[:, ['price']]
            if isinstance(node, ast.Subscript):
                # Check slice
                sl = node.slice
                # df[['a', 'b']] -> slice is List
                if isinstance(sl, ast.List):
                    for elt in sl.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            if elt.value in decision_vars:
                                violated_variables.append(elt.value)
                # df.loc[:, ['a', 'b']] -> slice is Tuple, look at index 1
                elif isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
                    col_part = sl.elts[1]
                    if isinstance(col_part, ast.List):
                        for elt in col_part.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                if elt.value in decision_vars:
                                    violated_variables.append(elt.value)
                            
    except Exception:
        # Fallback to robust regex if AST fails (e.g. partial code or syntax error not caught yet)
        for var in decision_vars:
            # Look for the variable name quoted inside a list-like structure in assignment
            # This is broad but avoids missing leakage in less standard code styles.
            # e.g., features = ["price"] or features = ["F1", "price"]
            pattern = rf"['\"]{re.escape(var)}['\"]"
            if re.search(pattern, code):
                # We double check if it's likely a feature list
                if re.search(rf"(feature|x_col|predictor|model_cols).*=.*{re.escape(var)}", code, re.IGNORECASE):
                    violated_variables.append(var)

    violated_variables = sorted(list(set(violated_variables)))
    
    if violated_variables:
        return {
            "passed": False,
            "error_message": f"CAUSAL_VIOLATION: Decision variables {violated_variables} found in feature lists. "
                             f"In optimization problems, these variables cannot be used to train the predictive model "
                             f"as they are unknown at the time of prediction.",
            "violated_variables": violated_variables
        }
        
    return {"passed": True, "error_message": "", "violated_variables": []}


def validate_model_metrics_consistency(
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validates consistency between best_model_name and best_model_auc.
    
    UNIVERSAL CHECK for any ML task (classification/regression).
    Ensures the metrics reported match the model actually selected.
    
    Args:
        metrics: The metrics.json dict (usually from data/metrics.json)
        
    Returns:
        {
            "passed": bool,
            "error_message": str,
            "details": {...}
        }
    """
    # 1. Access model_performance section
    perf = metrics.get("model_performance", {})
    if not perf:
        return {"passed": True, "error_message": "", "details": {}}

    # 2. Extract key fields
    best_name = perf.get("best_model_name")
    # Supports both AUC (classification) and R2/MSE (regression) if present
    best_auc = perf.get("best_model_auc")
    baseline_auc = perf.get("baseline_auc")
    
    # Generic mapping if AUC is missing but other metrics exist
    if best_auc is None:
        best_auc = perf.get("best_model_r2")
    if baseline_auc is None:
        baseline_auc = perf.get("baseline_r2")

    if best_name is None:
        return {"passed": True, "error_message": "", "details": {}}

    # 3. Determine if best_name is a baseline model
    baseline_keywords = ['logisticregression', 'linearregression', 'dummyclassifier', 'dummyregressor', 'baseline']
    is_baseline = any(kw in str(best_name).lower() for kw in baseline_keywords)

    details = {
        "best_model_name": best_name,
        "best_model_auc": best_auc,
        "baseline_auc": baseline_auc,
        "is_baseline_selected": is_baseline
    }

    # 4. Consistency logic
    if best_auc is not None and baseline_auc is not None:
        # Rounding tolerance
        tolerance = 0.01
        
        if is_baseline:
            # If baseline is selected, it must be the best (or equal)
            if abs(best_auc - baseline_auc) > tolerance:
                return {
                    "passed": False,
                    "error_message": f"Inconsistency: Selected best model is a baseline ({best_name}) but its performance ({best_auc}) differs from baseline_auc ({baseline_auc}).",
                    "details": details
                }
        else:
            # If advanced model is selected, it should be at least as good as baseline
            if best_auc < (baseline_auc - tolerance):
                return {
                    "passed": False,
                    "error_message": f"Inconsistency: Selected best model ({best_name}) has performance ({best_auc}) significantly worse than baseline_auc ({baseline_auc}).",
                    "details": details
                }

    return {"passed": True, "error_message": "", "details": details}


def validate_metrics_ci_consistency(metrics: Dict[str, Any]) -> List[str]:
    """
    Validate that metric mean lies within ci_lower/ci_upper when present.
    Returns issue strings like: metrics_schema_inconsistent:<metric_name>
    """
    issues: List[str] = []
    if not isinstance(metrics, dict):
        return issues

    def _is_number(value: Any) -> bool:
        try:
            float(value)
            return True
        except Exception:
            return False

    def _scan(obj: Dict[str, Any], prefix: str) -> None:
        if not isinstance(obj, dict):
            return
        if all(key in obj for key in ("mean", "ci_lower", "ci_upper")):
            mean = obj.get("mean")
            lower = obj.get("ci_lower")
            upper = obj.get("ci_upper")
            if not (_is_number(mean) and _is_number(lower) and _is_number(upper)):
                issues.append(f"metrics_schema_inconsistent:{prefix}")
            else:
                mean_f = float(mean)
                lower_f = float(lower)
                upper_f = float(upper)
                if not (lower_f <= mean_f <= upper_f):
                    issues.append(f"metrics_schema_inconsistent:{prefix}")
        for key, value in obj.items():
            if isinstance(value, dict):
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _scan(value, next_prefix)

    model_perf = metrics.get("model_performance") if isinstance(metrics.get("model_performance"), dict) else None
    if isinstance(model_perf, dict):
        _scan(model_perf, "model_performance")
    else:
        _scan(metrics, "")

    deduped = []
    seen = set()
    for issue in issues:
        if issue in seen:
            continue
        seen.add(issue)
        deduped.append(issue)
    return deduped
