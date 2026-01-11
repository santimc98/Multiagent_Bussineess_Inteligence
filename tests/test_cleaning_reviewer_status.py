from src.agents.cleaning_reviewer import normalize_cleaning_reviewer_result


def test_cleaning_reviewer_status_normalizes_warning_variant():
    result = {
        "status": "APPROVED_WITH_WARNINGS",
        "feedback": "All good with minor warnings.",
        "failed_checks": ["CHECK_A"],
        "required_fixes": [],
    }
    normalized = normalize_cleaning_reviewer_result(result)
    assert normalized["status"] == "APPROVE_WITH_WARNINGS"
    assert normalized["feedback"] == "All good with minor warnings."
    assert "STATUS_ENUM_NORMALIZED" in normalized["failed_checks"]
