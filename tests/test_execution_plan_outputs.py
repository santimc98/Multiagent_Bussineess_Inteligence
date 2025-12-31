from src.agents.execution_planner import build_execution_plan


def _artifact_types(plan):
    return {item.get("artifact_type") for item in plan.get("outputs", []) if isinstance(item, dict)}


def test_execution_plan_outputs_by_objective():
    profile = {"column_count": 10}
    classification = _artifact_types(build_execution_plan("classification", profile))
    regression = _artifact_types(build_execution_plan("regression", profile))
    forecasting = _artifact_types(build_execution_plan("forecasting", profile))
    ranking = _artifact_types(build_execution_plan("ranking", profile))

    assert "confusion_matrix" in classification
    assert "residuals" in regression
    assert "forecast" in forecasting
    assert "ranking_scores" in ranking

    assert classification != regression
    assert forecasting != ranking
