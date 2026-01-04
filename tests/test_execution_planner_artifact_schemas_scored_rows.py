from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_artifact_schemas_scored_rows_includes_required_columns():
    planner = ExecutionPlannerAgent(api_key=None)
    strategy = {
        "required_columns": ["CurrentPhase", "1stYearAmount"],
        "analysis_type": "ranking",
        "title": "Ranked scoring for success",
    }
    business_objective = "CurrentPhase contiene 'Contract'; optimiza el precio per segment."
    contract = planner.generate_contract(
        strategy=strategy,
        business_objective=business_objective,
        data_summary="Column Types:\n- Categorical/Boolean: CurrentPhase\n",
        column_inventory=["CurrentPhase", "1stYearAmount", "Size", "Debtors", "Sector"],
    )
    artifact_schemas = contract.get("artifact_schemas")
    assert isinstance(artifact_schemas, dict)
    scored_schema = artifact_schemas.get("data/scored_rows.csv")
    assert isinstance(scored_schema, dict)
    required_columns = scored_schema.get("required_columns") or []
    norm_required = {col.lower() for col in required_columns}
    assert "is_success" in norm_required
    assert "cluster_id" in {col.lower() for col in required_columns}
    assert any(col.lower().startswith("recommended_") for col in required_columns)
    assert "expected_value_at_recommendation" in norm_required
    assert "pred_prob_success" in {col.lower() for col in required_columns}
    patterns = scored_schema.get("allowed_name_patterns") or []
    assert any(pattern.startswith("^recommended_") for pattern in patterns)
    assert any(pattern.startswith("^expected_") for pattern in patterns)
