from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_adds_reporting_policy_and_preview_deliverable():
    planner = ExecutionPlannerAgent(api_key=None)
    strategy = {"title": "Test Plan", "analysis_type": "predictive", "required_columns": ["A"]}
    contract = planner.generate_contract(strategy, data_summary="", business_objective="Test objective", column_inventory=["A"])
    spec = contract.get("spec_extraction", {}) if isinstance(contract, dict) else {}
    deliverables = spec.get("deliverables", []) if isinstance(spec, dict) else []
    preview_paths = [
        item.get("path")
        for item in deliverables
        if isinstance(item, dict)
    ]
    preview_required_flags = [
        item.get("required")
        for item in deliverables
        if isinstance(item, dict) and item.get("path") == "reports/recommendations_preview.json"
    ]
    assert "reports/recommendations_preview.json" in preview_paths
    assert preview_required_flags and preview_required_flags[0] is False

    reporting_policy = contract.get("reporting_policy", {})
    assert reporting_policy.get("demonstrative_examples_enabled") is True
    assert "NO_GO" in reporting_policy.get("demonstrative_examples_when_outcome_in", [])
    assert reporting_policy.get("max_examples") == 5
    assert reporting_policy.get("require_strong_disclaimer") is True
