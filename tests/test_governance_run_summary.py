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
