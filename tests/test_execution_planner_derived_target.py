from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_derives_target_from_status_column():
    planner = ExecutionPlannerAgent(api_key=None)
    strategy = {"required_columns": ["CurrentPhase"], "analysis_type": "predictive", "title": "Conversion"}
    business_objective = "Predict conversion probability for contracts and improve win rate."
    data_summary = "Column Types:\n- Categorical/Boolean: CurrentPhase, Owner\n"
    contract = planner.generate_contract(
        strategy=strategy,
        data_summary=data_summary,
        business_objective=business_objective,
        column_inventory=["CurrentPhase", "Owner", "Amount"],
    )
    derived = (contract.get("spec_extraction") or {}).get("derived_columns") or []
    target = next((d for d in derived if isinstance(d, dict) and d.get("role") == "target"), None)
    assert target is not None
    assert target.get("name") == "is_success"
    assert target.get("derived_from") == "CurrentPhase"


def test_execution_planner_derives_positive_labels_from_objective_contains():
    planner = ExecutionPlannerAgent(api_key=None)
    strategy = {"required_columns": ["CurrentPhase"], "analysis_type": "predictive", "title": "Conversion"}
    business_objective = (
        "El campo \"CurrentPhase\" indica el estado; cuando contiene \"Contract\" significa exito."
    )
    data_summary = "Column Types:\n- Categorical/Boolean: CurrentPhase, Sector\n"
    contract = planner.generate_contract(
        strategy=strategy,
        data_summary=data_summary,
        business_objective=business_objective,
        column_inventory=["CurrentPhase", "1stYearAmount", "Size", "Debtors", "Sector"],
    )
    derived = (contract.get("spec_extraction") or {}).get("derived_columns") or []
    target = next((d for d in derived if isinstance(d, dict) and d.get("role") == "target"), None)
    assert target is not None
    assert target.get("derived_from") == "CurrentPhase"
    positive_values = target.get("positive_values") or []
    assert "Contract" in positive_values
    assert "success" not in {str(val).lower() for val in positive_values}
