import json
import os

from src.graph import graph as graph_mod


class _StubReviewer:
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": ""}


def test_result_evaluator_blocks_inconsistent_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"model_performance": {"lift": {"mean": 0.2, "ci_lower": 0.3, "ci_upper": 0.4}}},
            handle,
        )

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewer())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert any("METRICS_SCHEMA_INCONSISTENT" in item for item in result.get("feedback_history", []))
