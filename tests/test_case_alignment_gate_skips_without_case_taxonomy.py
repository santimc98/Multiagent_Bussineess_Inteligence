import json
import os

from src.graph import graph as graph_mod


class _StubReviewer:
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": ""}


def test_case_alignment_gate_skips_without_case_taxonomy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewer())
    result = graph_mod.run_result_evaluator(state)

    with open(os.path.join("data", "case_alignment_report.json"), "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report.get("status") == "SKIPPED"
    assert "case_taxonomy" in (report.get("skip_reason") or "")
    assert result["case_alignment_report"]["status"] == "SKIPPED"
