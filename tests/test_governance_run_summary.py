import json
import os

from src.utils.governance import build_run_summary


def test_run_summary_outcome_with_limitations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"auc": 0.51, "baseline_auc": 0.5}, f)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)
    with open("data/execution_contract.json", "w", encoding="utf-8") as f:
        json.dump({"counterfactual_policy": "observational_only"}, f)
    summary = build_run_summary({"review_verdict": "APPROVED"})
    assert summary.get("run_outcome") == "GO_WITH_LIMITATIONS"
    assert summary.get("metric_ceiling_detected") is True


def test_run_summary_integrity_critical_forces_no_go(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"auc": 0.6}, f)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)
    with open("data/integrity_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"issues": [{"type": "MISSING_COLUMN", "severity": "critical"}]},
            f,
        )

    summary = build_run_summary({"review_verdict": "APPROVED"})
    assert summary.get("run_outcome") == "NO_GO"
    assert "integrity_critical" in summary.get("failed_gates", [])
    assert summary.get("integrity_critical_count") == 1


def test_run_summary_integrity_warning_does_not_force_no_go(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"auc": 0.6}, f)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)
    with open("data/integrity_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"issues": [{"type": "OPTIONAL_COLUMN_MISSING", "severity": "warning"}]},
            f,
        )

    summary = build_run_summary({"review_verdict": "APPROVED"})
    assert summary.get("integrity_critical_count") == 0
    assert "integrity_critical" not in summary.get("failed_gates", [])
    assert summary.get("run_outcome") in {"GO", "GO_WITH_LIMITATIONS"}
