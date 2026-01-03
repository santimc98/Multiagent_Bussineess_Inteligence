import os

from src.utils.governance import build_run_summary


def test_run_summary_no_go_on_data_engineer_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    state = {
        "pipeline_aborted_reason": "data_engineer_static_scan_failed",
        "data_engineer_failed": True,
    }
    summary = build_run_summary(state)
    assert summary["run_outcome"] == "NO_GO"
    assert any("pipeline_aborted" in gate for gate in summary["failed_gates"])
