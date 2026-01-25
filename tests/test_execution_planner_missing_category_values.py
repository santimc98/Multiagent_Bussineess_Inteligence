from src.agents.execution_planner import _ensure_missing_category_values, build_contract_min


def test_missing_category_value_injected_and_propagated_to_gate():
    contract = {
        "preprocessing_requirements": {
            "nan_strategies": [
                {"column": "Alley", "strategy": "missing_category"}
            ]
        },
        "cleaning_gates": [
            {"name": "null_handling_gate", "severity": "HARD", "params": {"columns": ["Alley"], "allow_nans": False}}
        ],
    }
    updated = _ensure_missing_category_values(contract)
    strat = updated["preprocessing_requirements"]["nan_strategies"][0]
    assert strat.get("missing_category_value")
    gate_params = updated["cleaning_gates"][0]["params"]
    assert gate_params.get("missing_category_values", {}).get("Alley") == strat.get("missing_category_value")


def test_missing_category_value_replaces_unsafe_token():
    contract = {
        "preprocessing_requirements": {
            "nan_strategies": [
                {"column": "Alley", "strategy": "missing_category", "missing_category_value": "None"}
            ]
        },
        "cleaning_gates": [
            {"name": "null_handling_gate", "severity": "HARD", "params": {"columns": ["Alley"], "allow_nans": False}}
        ],
    }
    updated = _ensure_missing_category_values(contract)
    val = updated["preprocessing_requirements"]["nan_strategies"][0].get("missing_category_value")
    assert val and val.lower() != "none"


def test_contract_min_includes_preprocessing_requirements_minimal():
    contract = {
        "canonical_columns": ["Id", "Alley"],
        "preprocessing_requirements": {
            "nan_strategies": [
                {"column": "Alley", "strategy": "missing_category", "missing_category_value": "__MISSING__"}
            ]
        },
    }
    contract_min = build_contract_min(contract, {}, ["Id", "Alley"], ["Id", "Alley"])
    prep = contract_min.get("preprocessing_requirements", {})
    assert prep.get("nan_strategies")
    assert prep["nan_strategies"][0]["missing_category_value"] == "__MISSING__"
