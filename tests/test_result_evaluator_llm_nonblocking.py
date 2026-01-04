import json
import os

from src.graph import graph as graph_mod


class _StubReviewer:
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "NEEDS_IMPROVEMENT", "feedback": "LLM concern"}


def test_result_evaluator_llm_nonblocking(tmp_path, monkeypatch):
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
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewer())
    result = graph_mod.run_result_evaluator(state)
    assert result["review_verdict"] == "APPROVE_WITH_WARNINGS"
    assert any("REVIEWER_LLM_NONBLOCKING_WARNING" in item for item in result["feedback_history"])
