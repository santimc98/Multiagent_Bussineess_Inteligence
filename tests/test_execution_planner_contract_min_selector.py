import json

from src.agents.execution_planner import (
    ExecutionPlannerAgent,
    build_contract_min,
    select_relevant_columns,
)


class DummyResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.candidates = []
        self.usage_metadata = None


class DummyClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def generate_content(self, _prompt: str) -> DummyResponse:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return DummyResponse(self._responses[idx])


def test_select_relevant_columns_compact() -> None:
    inventory = [f"col_{i}" for i in range(200)]
    strategy = {
        "required_columns": ["col_1", "col_5", "col_10", "col_20", "col_30", "col_40"],
        "decision_variables": ["col_50"],
        "target_column": "col_60",
    }
    payload = select_relevant_columns(
        strategy=strategy,
        business_objective='Use "col_70" for analysis.',
        domain_expert_critique="",
        column_inventory=inventory,
        data_profile_summary="",
    )
    relevant = payload["relevant_columns"]
    assert set(strategy["required_columns"]).issubset(set(relevant))
    assert len(relevant) <= 30

    contract_min = build_contract_min({}, strategy, inventory, relevant)
    assert contract_min["canonical_columns"]
    assert contract_min["artifact_requirements"]
    assert contract_min["qa_gates"]
    assert contract_min["column_roles"].get("unknown") == []


def test_execution_planner_invalid_json_fallback_contract_min() -> None:
    agent = ExecutionPlannerAgent(api_key=None)
    agent.client = DummyClient(
        [
            '{"contract_version": 2, "rationale":',
            '{"contract_version": 2, "rationale":',
        ]
    )
    inventory = [f"col_{i}" for i in range(10)]
    strategy = {"required_columns": ["col_1", "col_2"]}

    contract = agent.generate_contract(
        strategy=strategy,
        data_summary="",
        business_objective="Test objective",
        column_inventory=inventory,
        output_dialect={"sep": ",", "decimal": ".", "encoding": "utf-8"},
        env_constraints={"forbid_inplace_column_creation": True},
        domain_expert_critique="",
    )

    assert isinstance(contract, dict)
    assert isinstance(agent.last_contract_min, dict)
    assert agent.last_contract_min["canonical_columns"]
    assert agent.last_contract_min["artifact_requirements"]
    assert json.dumps(agent.last_contract_min)
