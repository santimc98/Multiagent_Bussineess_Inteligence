import types

from src.graph.graph import run_qa_reviewer, qa_reviewer


def test_qa_override_variance_false_positive(monkeypatch):
    code = """
import pandas as pd
df = pd.DataFrame({"y":[1,2,3]})
y = df["y"]
if y.nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
print("Mapping Summary:", {"target": "y", "features": []})
"""

    def fake_review(c, s, b):
        return {
            "status": "REJECTED",
            "feedback": "Missing target variance guard",
            "failed_gates": ["TARGET_VARIANCE"],
            "required_fixes": ["Add variance guard"],
        }

    monkeypatch.setattr(qa_reviewer, "review_code", fake_review)
    state = {
        "generated_code": code,
        "selected_strategy": {},
        "business_objective": "",
        "feedback_history": [],
        "qa_reject_streak": 0,
    }
    result = run_qa_reviewer(state)
    assert result["review_verdict"] in ("APPROVED", "APPROVE_WITH_WARNINGS")
    assert "QA_LLM_FALSE_POSITIVE_OVERRIDDEN" in result["feedback_history"][-1] or "QA_LLM_FALSE_POSITIVE_OVERRIDDEN" in result.get("review_feedback", "")


def test_qa_fail_safe_preserves_gate_context(monkeypatch):
    def fake_review(c, s, b):
        return {
            "status": "REJECTED",
            "feedback": "Fail",
            "failed_gates": ["TARGET_VARIANCE"],
            "required_fixes": [],
        }

    monkeypatch.setattr(qa_reviewer, "review_code", fake_review)
    state = {
        "generated_code": "print('ok')",
        "selected_strategy": {},
        "business_objective": "",
        "feedback_history": [],
        "qa_reject_streak": 5,
    }
    result = run_qa_reviewer(state)
    assert result["review_verdict"] == "REJECTED"
    assert result.get("last_gate_context", {}).get("source") == "qa_reviewer"
