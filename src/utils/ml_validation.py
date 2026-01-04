import ast
import re
from typing import Dict, Any, List

def validate_decision_variable_isolation(
    code: str, 
    execution_contract: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validates that decision variables are not used as model features.
    
    This is a UNIVERSAL check for ANY optimization problem.
    It uses the execution_contract to dynamically identify decision variables.
    
    Args:
        code: Generated Python code to validate
        execution_contract: Contract with feature_availability metadata
        
    Returns:
        {
            "passed": bool,
            "error_message": str,  # Empty if passed
            "violated_variables": List[str]  # Decision vars found in features
        }
    """
    # 1. Identify if it is an optimization/prescriptive problem
    obj_type = str(execution_contract.get("objective_type", "")).lower()
    biz_obj = str(execution_contract.get("business_objective", "")).lower()
    
    # Optimization/Prescriptive keywords
    is_optimization = (obj_type in ["prescriptive", "optimization"]) or \
                      any(kw in biz_obj for kw in ["optimize", "maximize", "minimize"])
    
    if not is_optimization:
        return {"passed": True, "error_message": "", "violated_variables": []}

    # 2. Extract decision variables from feature_availability
    decision_vars = []
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
