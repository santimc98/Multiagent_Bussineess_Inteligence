from src.agents.execution_planner import ExecutionPlannerAgent, parse_derive_from_expression


def test_parse_derive_from_expression_simple():
    parsed = parse_derive_from_expression("CurrentPhase == 'Contract'")
    assert parsed.get("column") == "CurrentPhase"
    assert parsed.get("positive_values") == ["Contract"]


def test_canonical_columns_exclude_derived_targets_and_segments():
    planner = ExecutionPlannerAgent(api_key=None)
    strategy = {
        "required_columns": ["CurrentPhase", "Amount"],
        "analysis_type": "predictive",
        "title": "Segmented Conversion",
    }
    business_objective = "Segment accounts and predict conversion success."
    data_summary = "Column Types:\n- Categorical/Boolean: CurrentPhase\n- Numerical: Amount\n"
    contract = planner.generate_contract(
        strategy=strategy,
        data_summary=data_summary,
        business_objective=business_objective,
        column_inventory=["CurrentPhase", "Amount"],
    )
    canonical = contract.get("canonical_columns") or []
    assert "is_success" not in canonical
    assert "cluster_id" not in canonical

    evaluation_spec = planner.generate_evaluation_spec(
        strategy=strategy,
        contract=contract,
        data_summary=data_summary,
        business_objective=business_objective,
        column_inventory=["CurrentPhase", "Amount"],
    )
    spec_canonical = evaluation_spec.get("canonical_columns") or []
    assert "is_success" not in spec_canonical
    assert "cluster_id" not in spec_canonical
