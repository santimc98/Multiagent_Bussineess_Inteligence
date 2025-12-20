import json
from typing import Any, Dict, List, Tuple


PLAN_TYPE = "cleaning_plan_v1"


def _as_dict(val: Any) -> Dict[str, Any]:
    return val if isinstance(val, dict) else {}


def parse_cleaning_plan(text: str) -> Tuple[Dict[str, Any] | None, str | None]:
    """
    Parse a cleaning plan JSON string.
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


def validate_cleaning_plan(plan: Dict[str, Any]) -> List[str]:
    """
    Structure-only validation for a cleaning plan.
    Returns list of issues; empty list means ok.
    """
    issues: List[str] = []
    if not isinstance(plan, dict):
        return ["plan_not_dict"]
    if plan.get("plan_type") != PLAN_TYPE:
        issues.append("plan_type_missing_or_invalid")
    input_cfg = _as_dict(plan.get("input"))
    if not input_cfg.get("path"):
        issues.append("input.path_missing")
    dialect = _as_dict(plan.get("dialect"))
    if not dialect:
        issues.append("dialect_missing")
    output_cfg = _as_dict(plan.get("output"))
    if not output_cfg.get("cleaned_path"):
        issues.append("output.cleaned_path_missing")
    if not output_cfg.get("manifest_path"):
        issues.append("output.manifest_path_missing")
    conversions = plan.get("type_conversions", [])
    if conversions is not None and not isinstance(conversions, list):
        issues.append("type_conversions_not_list")
    derived = plan.get("derived_columns", [])
    if derived is not None and not isinstance(derived, list):
        issues.append("derived_columns_not_list")
    case_rules = _as_dict(plan.get("case_assignment")).get("rules")
    if case_rules is not None and not isinstance(case_rules, list):
        issues.append("case_assignment.rules_not_list")
    return issues
