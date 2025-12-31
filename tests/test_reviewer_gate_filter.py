from src.agents.reviewer import apply_reviewer_gate_filter


def test_required_fixes_not_filtered():
    result = {
        "status": "REJECTED",
        "failed_gates": ["gate_a"],
        "required_fixes": ["fix_1", "fix_2"],
        "feedback": "Needs fixes.",
    }
    out = apply_reviewer_gate_filter(result, ["gate_b"])
    assert out["failed_gates"] == []
    assert out["required_fixes"] == ["fix_1", "fix_2"]
    assert out["status"] == "APPROVE_WITH_WARNINGS"
