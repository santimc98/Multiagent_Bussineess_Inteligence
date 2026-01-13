from src.agents.execution_planner import _build_default_qa_gates


def _find_gate(gates, name):
    for gate in gates:
        if isinstance(gate, dict) and gate.get("name") == name:
            return gate
    return None


def test_default_qa_gates_allow_resampling_for_requires_target():
    contract = {"target_column": "y"}
    gates = _build_default_qa_gates({}, "", contract)
    gate = _find_gate(gates, "no_synthetic_data")
    assert gate is not None
    assert gate.get("params", {}).get("allow_resampling_random") is True


def test_default_qa_gates_allow_resampling_for_validation_method():
    contract = {"validation_requirements": {"method": "cross_validation"}}
    gates = _build_default_qa_gates({}, "", contract)
    gate = _find_gate(gates, "no_synthetic_data")
    assert gate is not None
    assert gate.get("params", {}).get("allow_resampling_random") is True
