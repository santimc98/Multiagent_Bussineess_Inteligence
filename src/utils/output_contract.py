import glob
import os
from typing import List, Dict, Any, Tuple


def _split_deliverables(deliverables: List[Any]) -> Tuple[List[str], List[str]]:
    required: List[str] = []
    optional: List[str] = []
    for item in deliverables or []:
        if isinstance(item, dict):
            path = item.get("path") or item.get("output") or item.get("artifact")
            if not path:
                continue
            is_required = item.get("required")
            if is_required is None:
                is_required = True
            if is_required:
                required.append(path)
            else:
                optional.append(path)
        else:
            path = str(item)
            if path:
                required.append(path)
    return required, optional


def _check_paths(paths: List[str]) -> Tuple[List[str], List[str]]:
    present: List[str] = []
    missing: List[str] = []
    for pattern in paths or []:
        try:
            if any(char in pattern for char in ["*", "?", "["]):
                matches = glob.glob(pattern)
                if matches:
                    present.extend(matches)
                else:
                    missing.append(pattern)
            else:
                if os.path.exists(pattern):
                    present.append(pattern)
                else:
                    missing.append(pattern)
        except Exception:
            missing.append(pattern)
    return present, missing


def check_required_outputs(required_outputs: List[Any]) -> Dict[str, object]:
    """
    Best-effort validation of required outputs.
    Supports glob patterns (e.g., static/plots/*.png).
    Returns dict with present, missing, summary.
    Never raises.
    """
    required, optional = _split_deliverables(required_outputs or [])
    present_required, missing_required = _check_paths(required)
    present_optional, missing_optional = _check_paths(optional)
    present = present_required + present_optional
    summary = (
        f"Present: {len(present)}; Missing required: {len(missing_required)}; "
        f"Missing optional: {len(missing_optional)}"
    )
    return {
        "present": present,
        "missing": missing_required,
        "missing_optional": missing_optional,
        "summary": summary,
    }
