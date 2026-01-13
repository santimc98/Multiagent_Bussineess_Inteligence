import os

from src.agents.results_advisor import ResultsAdvisorAgent


def test_results_advisor_insights_minimal():
    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights({"artifact_index": []})
    assert isinstance(insights, dict)
    assert insights.get("schema_version") == "1"
    summary_lines = insights.get("summary_lines")
    assert isinstance(summary_lines, list)
    assert summary_lines


def test_results_advisor_deployment_recommendation_with_ci(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as handle:
        handle.write('{"model_performance": {"Revenue Lift": {"mean": 1.05, "ci_lower": 0.95, "ci_upper": 1.1}}}')

    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights(
        {"artifact_index": [{"path": "data/metrics.json", "artifact_type": "metrics"}]}
    )
    assert insights.get("deployment_recommendation") == "PILOT"
    assert insights.get("confidence") in {"LOW", "MEDIUM"}
