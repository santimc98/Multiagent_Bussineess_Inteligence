"""
Run workspace isolation utilities.

Ensures each run operates in an isolated workspace directory to prevent
cross-run contamination from leftover artifacts.
"""
import os
from typing import Dict, Any, Optional


def init_run_workspace(run_dir: str) -> str:
    """
    Initialize a workspace directory for a run.

    Creates work_dir with required subdirectories for artifacts isolation.

    Args:
        run_dir: The run bundle directory (e.g., runs/run_xxx)

    Returns:
        work_dir: Path to the workspace directory (run_dir/work)
    """
    work_dir = os.path.join(run_dir, "work")

    # Create workspace and all required subdirectories
    subdirs = [
        "data",
        "reports",
        "static/plots",
        "analysis",
        "models",
        "plots",
        "artifacts",
    ]

    for subdir in subdirs:
        os.makedirs(os.path.join(work_dir, subdir), exist_ok=True)

    return work_dir


def enter_run_workspace(state: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    """
    Enter the run workspace - changes cwd to isolated workspace.

    This ensures all relative paths (data/, reports/, etc.) resolve
    to the run-specific workspace, not the global root.

    Args:
        state: Agent state dict
        run_dir: The run bundle directory

    Returns:
        Updated state with workspace info
    """
    # Save original cwd for restoration
    state["_orig_cwd"] = os.getcwd()

    # Initialize and enter workspace
    work_dir = init_run_workspace(run_dir)
    state["work_dir"] = work_dir
    state["workspace_active"] = True

    # Change to workspace directory
    os.chdir(work_dir)

    # Double-check required dirs exist (defense in depth)
    for subdir in ["data", "reports", "static/plots"]:
        os.makedirs(subdir, exist_ok=True)

    print(f"WORKSPACE_ENTER: Entered run workspace at {work_dir}")

    return state


def exit_run_workspace(state: Dict[str, Any]) -> None:
    """
    Exit the run workspace - restores original cwd.

    Args:
        state: Agent state dict with _orig_cwd
    """
    orig_cwd = state.get("_orig_cwd")
    if orig_cwd and os.path.isdir(orig_cwd):
        os.chdir(orig_cwd)
        print(f"WORKSPACE_EXIT: Restored cwd to {orig_cwd}")

    state["workspace_active"] = False

    # Note: We don't delete work_dir by default (useful for debug).
    # Set env CLEANUP_RUN_WORKSPACE=1 to enable cleanup in future.


def get_work_dir(state: Dict[str, Any]) -> Optional[str]:
    """
    Get the current work directory from state.

    Returns:
        work_dir if workspace is active, None otherwise
    """
    if state.get("workspace_active"):
        return state.get("work_dir")
    return None


def is_workspace_active(state: Dict[str, Any]) -> bool:
    """Check if we're currently in a run workspace."""
    return bool(state.get("workspace_active"))
