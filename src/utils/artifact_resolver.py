"""
Scoped artifact resolver utilities.

Ensures artifact reads are scoped to the current run workspace,
preventing contamination from global/root paths.
"""
import json
import os
from typing import Any, Dict, Optional


def resolve_run_path(state: Dict[str, Any], rel_path: str) -> str:
    """
    Resolve a relative path to the run workspace.

    If workspace is active, prepends work_dir. Otherwise returns rel_path as-is.

    Args:
        state: Agent state dict (may contain work_dir)
        rel_path: Relative path like "data/metrics.json"

    Returns:
        Resolved path scoped to workspace if active
    """
    work_dir = state.get("work_dir")
    if work_dir and state.get("workspace_active"):
        return os.path.join(work_dir, rel_path)
    return rel_path


def exists_scoped(state: Dict[str, Any], rel_path: str) -> bool:
    """
    Check if a file exists within the run workspace scope.

    Args:
        state: Agent state dict
        rel_path: Relative path to check

    Returns:
        True if file exists in scoped path
    """
    resolved = resolve_run_path(state, rel_path)
    return os.path.exists(resolved)


def load_json_scoped(state: Dict[str, Any], rel_path: str, default: Any = None) -> Any:
    """
    Load JSON from a path scoped to the run workspace.

    Args:
        state: Agent state dict
        rel_path: Relative path like "data/metrics.json"
        default: Value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON content or default
    """
    resolved = resolve_run_path(state, rel_path)
    if not os.path.exists(resolved):
        return default
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def read_text_scoped(state: Dict[str, Any], rel_path: str, default: str = "") -> str:
    """
    Read text file from a path scoped to the run workspace.

    Args:
        state: Agent state dict
        rel_path: Relative path
        default: Value to return if file doesn't exist

    Returns:
        File content or default
    """
    resolved = resolve_run_path(state, rel_path)
    if not os.path.exists(resolved):
        return default
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return default


def write_json_scoped(state: Dict[str, Any], rel_path: str, data: Any) -> bool:
    """
    Write JSON to a path scoped to the run workspace.

    Args:
        state: Agent state dict
        rel_path: Relative path like "data/artifact_index.json"
        data: Data to serialize

    Returns:
        True if write succeeded
    """
    resolved = resolve_run_path(state, rel_path)
    try:
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except OSError:
        return False


def get_artifact_from_state_or_scoped(
    state: Dict[str, Any],
    state_key: str,
    rel_path: str,
    default: Any = None
) -> Any:
    """
    Get artifact preferring state, then scoped file, then default.

    Resolution order:
    1. state[state_key] (memory)
    2. load_json_scoped(state, rel_path) (run workspace)
    3. default

    This is the canonical way to read artifacts without cross-run contamination.

    Args:
        state: Agent state dict
        state_key: Key to check in state
        rel_path: Relative path for fallback
        default: Final fallback value

    Returns:
        Resolved artifact value
    """
    # Priority 1: State (memory)
    if state_key in state and state[state_key] is not None:
        return state[state_key]

    # Priority 2: Scoped file (run workspace)
    scoped = load_json_scoped(state, rel_path)
    if scoped is not None:
        return scoped

    # Priority 3: Default
    return default
