from src.agents.execution_planner import build_contract_min


def test_contract_min_canonical_columns_match_contract():
    contract = {
        "canonical_columns": ["A", "B"],
        "allowed_feature_sets": {"audit_only_features": ["C"]},
    }
    strategy = {"required_columns": ["A", "B", "C"]}
    column_inventory = ["A", "B", "C"]
    relevant_columns = ["A", "B", "C"]

    contract_min = build_contract_min(contract, strategy, column_inventory, relevant_columns)

    assert contract_min.get("canonical_columns") == ["A", "B"]
