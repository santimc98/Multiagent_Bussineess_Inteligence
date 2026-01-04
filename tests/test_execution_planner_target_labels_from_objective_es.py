from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_target_labels_from_objective_es():
    planner = ExecutionPlannerAgent(api_key=None)
    strategy = {"analysis_type": "predictive", "required_columns": ["CurrentPhase", "Size"]}
    business_objective = (
        "El campo “CurrentPhase” señala la fase final; cuando contiene “Contract” significa que se cerró un contrato."
    )
    contract = planner.generate_contract(
        strategy=strategy,
        business_objective=business_objective,
        data_summary="Column Types:\n- Categorical/Boolean: CurrentPhase, Size\n",
        column_inventory=["CurrentPhase", "Size", "Debtors"],
    )
    derived = (contract.get("spec_extraction") or {}).get("derived_columns") or []
    target = next((entry for entry in derived if isinstance(entry, dict) and entry.get("role") == "target"), None)
    assert target is not None
    assert target.get("derived_from") == "CurrentPhase"
    assert target.get("column") == "CurrentPhase"
    positive_values = target.get("positive_values") or []
    assert "Contract" in positive_values
    assert "success" not in {str(val).lower() for val in positive_values}

    evaluation_spec = planner.generate_evaluation_spec(
        strategy=strategy,
        contract=contract,
        business_objective=business_objective,
        data_summary="Column Types:\n- Categorical/Boolean: CurrentPhase, Size\n",
        column_inventory=["CurrentPhase", "Size", "Debtors"],
    )
    target_spec = evaluation_spec.get("target") or {}
    derive_from = target_spec.get("derive_from") or {}
    assert derive_from.get("column") == "CurrentPhase"
    assert derive_from.get("positive_values") == ["Contract"]
