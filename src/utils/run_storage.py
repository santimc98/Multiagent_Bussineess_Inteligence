import json
import os
import shutil
import zipfile
from datetime import datetime
from typing import Any, Dict, Optional

from src.utils.run_bundle import write_run_manifest

RUNS_DIR = "runs"
LATEST_DIR = os.path.join(RUNS_DIR, "latest")
ARCHIVE_DIR = os.path.join(RUNS_DIR, "archive")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def init_run_dir(run_id: str, started_at: Optional[str] = None, runs_dir: str = RUNS_DIR) -> str:
    latest_dir = os.path.join(runs_dir, "latest")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir, ignore_errors=True)
    _ensure_dir(latest_dir)
    try:
        with open(os.path.join(latest_dir, "run_id.txt"), "w", encoding="utf-8") as f:
            f.write(run_id)
    except Exception:
        pass
    run_dir = os.path.join(runs_dir, run_id)
    for sub in ["contracts", "agents", "sandbox", "artifacts", "report"]:
        _ensure_dir(os.path.join(run_dir, sub))
    write_manifest_partial(
        run_id=run_id,
        manifest_path=os.path.join(run_dir, "run_manifest.json"),
        started_at=started_at,
    )
    return run_dir


def write_manifest_partial(
    run_id: str,
    manifest_path: str | None = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    input_info: Optional[Dict[str, Any]] = None,
    agent_models: Optional[Dict[str, Any]] = None,
    status_final: Optional[str] = None,
) -> None:
    path = manifest_path or os.path.join(LATEST_DIR, "run_manifest.json")
    payload = _safe_load_json(path)
    payload.setdefault("run_id", run_id)
    if started_at:
        payload["started_at"] = started_at
    if ended_at:
        payload["ended_at"] = ended_at
    if input_info:
        payload["input"] = input_info
    if agent_models:
        payload["models_by_agent"] = agent_models
    if status_final:
        payload["status_final"] = status_final
    _ensure_dir(os.path.dirname(path))
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def normalize_status(status: str | None) -> str:
    if not status:
        return "CRASH"
    upper = str(status).upper()
    if upper in {"APPROVED", "APPROVE_WITH_WARNINGS", "PASS"}:
        return "PASS"
    if upper in {"NEEDS_IMPROVEMENT"}:
        return "NEEDS_IMPROVEMENT"
    if upper in {"REJECTED"}:
        return "REJECTED"
    if upper in {"FAIL", "FAILED"}:
        return "FAIL"
    if "CRASH" in upper:
        return "CRASH"
    return upper


def finalize_run(
    run_id: str,
    status_final: str,
    state: Optional[Dict[str, Any]] = None,
    keep_last: int = 5,
    runs_dir: str = RUNS_DIR,
) -> None:
    _ensure_dir(os.path.join(runs_dir, "archive"))
    status_final = normalize_status(status_final)
    ended_at = datetime.utcnow().isoformat()
    write_run_manifest(
        run_id,
        state or {},
        status_final=status_final,
        started_at=(state or {}).get("run_start_ts"),
        ended_at=ended_at,
    )
    if status_final != "PASS":
        run_dir = os.path.join(runs_dir, run_id)
        _archive_run(run_id, run_dir, os.path.join(runs_dir, "archive"))
        apply_retention(keep_last=keep_last, archive_dir=os.path.join(runs_dir, "archive"))

def _archive_run(run_id: str, run_dir: str, archive_dir: str) -> Optional[str]:
    if not os.path.isdir(run_dir):
        return None
    _ensure_dir(archive_dir)
    zip_name = f"run_{run_id}.zip"
    zip_path = os.path.join(archive_dir, zip_name)
    base_dir = os.path.abspath(os.path.dirname(run_dir))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(run_dir):
            for file in files:
                path = os.path.join(root, file)
                arcname = os.path.relpath(path, base_dir)
                zf.write(path, arcname)
    return zip_path


def apply_retention(keep_last: int = 5, archive_dir: str = ARCHIVE_DIR) -> None:
    if keep_last is None or keep_last <= 0:
        return
    if not os.path.isdir(archive_dir):
        return
    entries = []
    for name in os.listdir(archive_dir):
        if not name.endswith(".zip"):
            continue
        path = os.path.join(archive_dir, name)
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            continue
        entries.append((mtime, path))
    entries.sort(reverse=True)
    for _, path in entries[keep_last:]:
        try:
            os.remove(path)
        except Exception:
            pass


def clean_workspace_outputs() -> None:
    """Clean workspace outputs - expanded airbag to prevent cross-run contamination."""
    for folder in ["analysis", "models", "reports", os.path.join("static", "plots"), "plots", "artifacts"]:
        if os.path.isdir(folder):
            try:
                shutil.rmtree(folder, ignore_errors=True)
            except Exception:
                pass
    # Expanded list: all artifacts that could cause cross-run contamination
    for path in [
        os.path.join("data", "metrics.json"),
        os.path.join("data", "scored_rows.csv"),
        os.path.join("data", "alignment_check.json"),
        os.path.join("data", "output_contract_report.json"),
        os.path.join("data", "cleaned_data.csv"),
        os.path.join("data", "cleaned_full.csv"),
        os.path.join("data", "cleaning_manifest.json"),
        os.path.join("data", "dataset_profile.json"),
        # P0 FIX: Additional files to prevent cross-run contamination
        os.path.join("data", "produced_artifact_index.json"),
        os.path.join("data", "plan.json"),
        os.path.join("data", "strategy_spec.json"),
        os.path.join("data", "evaluation_spec.json"),
        os.path.join("data", "run_summary.json"),
        os.path.join("data", "steward_summary.json"),
        os.path.join("data", "steward_summary.txt"),
        os.path.join("data", "plot_insights.json"),
        os.path.join("data", "insights.json"),
        os.path.join("data", "integrity_audit_report.json"),
        os.path.join("data", "data_adequacy_report.json"),
        os.path.join("data", "governance_report.json"),
        # Note: dataset_memory.json is intentionally NOT cleaned - it persists across runs
        os.path.join("data", "contract_min.json"),
        os.path.join("data", "execution_contract.json"),
        os.path.join("data", "executive_summary.md"),
    ]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
