import json
from typing import Any, Dict, List, Tuple


PLAN_TYPE = "ml_experiment_plan_v1"


def _as_dict(val: Any) -> Dict[str, Any]:
    return val if isinstance(val, dict) else {}


def parse_ml_plan(text: str) -> Tuple[Dict[str, Any] | None, str | None]:
    """
    Parse an ML experiment plan JSON string.
    Returns (plan, error_message). If the content is not a plan, returns (None, None).
    """
    if not isinstance(text, str):
        return None, "plan_text_not_str"
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return None, "plan_text_empty"
    if not cleaned.startswith("{"):
        return None, None
    try:
        payload = json.loads(cleaned)
    except Exception:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    if payload.get("plan_type") != PLAN_TYPE:
        return None, None
    return payload, None


def validate_ml_plan(plan: Dict[str, Any]) -> List[str]:
    """
    Structure-only validation for an ML plan.
    Returns list of issues; empty list means ok.
    """
    issues: List[str] = []
    if not isinstance(plan, dict):
        return ["plan_not_dict"]
    if plan.get("plan_type") != PLAN_TYPE:
        issues.append("plan_type_missing_or_invalid")
    input_cfg = _as_dict(plan.get("input"))
    if not input_cfg.get("cleaned_path"):
        issues.append("input.cleaned_path_missing")
    dialect = _as_dict(plan.get("dialect"))
    if dialect is None:
        issues.append("dialect_missing")
    outputs = _as_dict(plan.get("outputs"))
    if not outputs.get("weights_path"):
        issues.append("outputs.weights_path_missing")
    experiments = plan.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        issues.append("experiments_missing_or_empty")
    features = plan.get("features")
    if features is not None and not isinstance(features, dict):
        issues.append("features_not_dict")
    target = plan.get("target")
    if target is not None and not isinstance(target, dict):
        issues.append("target_not_dict")
    return issues
