from src.agents.results_advisor import ResultsAdvisorAgent


def test_results_advisor_insights_minimal():
    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights({"artifact_index": []})
    assert isinstance(insights, dict)
    assert insights.get("schema_version") == "1"
    summary_lines = insights.get("summary_lines")
    assert isinstance(summary_lines, list)
    assert summary_lines
