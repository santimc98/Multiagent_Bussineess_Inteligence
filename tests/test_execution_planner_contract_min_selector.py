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


def test_contract_min_inherits_roles_from_full() -> None:
    inventory = ["A", "B", "C", "D", "E"]
    strategy = {"required_columns": ["A", "B", "C", "D"]}
    full_contract = {
        "business_objective": "Test objective",
        "column_roles": {
            "pre_decision": ["A"],
            "decision": ["B"],
            "outcome": ["C"],
            "post_decision_audit_only": ["D"],
        },
    }
    relevant = ["A", "B", "C", "D"]
    contract_min = build_contract_min(full_contract, strategy, inventory, relevant)
    roles = contract_min.get("column_roles", {})
    assert roles.get("pre_decision") == ["A"]
    assert roles.get("decision") == ["B"]
    assert roles.get("outcome") == ["C"]
    assert roles.get("post_decision_audit_only") == ["D"]
    allowed = contract_min.get("allowed_feature_sets", {})
    assert allowed.get("segmentation_features") == ["A"]
    assert allowed.get("model_features") == ["A", "B"]
    assert set(allowed.get("forbidden_features") or []) == {"C", "D"}


def test_contract_min_normalizes_column_role_maps() -> None:
    inventory = ["ColA", "ColB", "ColC", "ColD"]
    strategy = {"required_columns": ["ColA", "ColB", "ColC", "ColD"]}
    full_contract = {
        "column_roles": {
            "ColA": {"role": "pre_decision"},
            "ColB": {"role": "decision"},
            "ColC": {"role": "outcome"},
            "ColD": {"role": "post_decision_audit_only"},
        }
    }
    relevant = ["ColA", "ColB", "ColC", "ColD"]
    contract_min = build_contract_min(full_contract, strategy, inventory, relevant)
    roles = contract_min.get("column_roles", {})
    assert roles.get("pre_decision") == ["ColA"]
    assert roles.get("decision") == ["ColB"]
    assert roles.get("outcome") == ["ColC"]
    assert roles.get("post_decision_audit_only") == ["ColD"]


def test_contract_min_inherits_scored_rows_schema_required_columns() -> None:
    inventory = ["PassengerId", "Age"]
    strategy = {"required_columns": ["PassengerId", "Age"]}
    full_contract = {
        "artifact_requirements": {
            "required_files": [{"path": "data/scored_rows.csv"}],
            "scored_rows_schema": {"required_columns": ["Individual_Triage_List_CSV"]},
        }
    }
    relevant = ["PassengerId", "Age"]
    contract_min = build_contract_min(full_contract, strategy, inventory, relevant)
    schema = contract_min.get("artifact_requirements", {}).get("scored_rows_schema", {})
    required_cols = schema.get("required_columns", [])
    assert "Individual_Triage_List_CSV" in required_cols
    assert any(path == "data/scored_rows.csv" for path in contract_min.get("required_outputs", []))


def test_contract_min_includes_decisioning_required_columns() -> None:
    inventory = ["id", "feature_a"]
    strategy = {"required_columns": ["id", "feature_a"]}
    full_contract = {
        "decisioning_requirements": {
            "enabled": True,
            "required": True,
            "output": {
                "file": "data/scored_rows.csv",
                "required_columns": [
                    {"name": "priority_score", "role": "score"},
                    {"name": "priority_rank", "role": "priority"},
                    {"name": "explanation", "role": "explanation"},
                ],
            },
        },
        "artifact_requirements": {
            "required_files": [{"path": "data/scored_rows.csv"}],
        },
    }
    contract_min = build_contract_min(full_contract, strategy, inventory, inventory)
    schema = contract_min.get("artifact_requirements", {}).get("scored_rows_schema", {})
    required_cols = schema.get("required_columns", [])
    assert "priority_score" in required_cols
    assert "priority_rank" in required_cols
    assert "explanation" in required_cols


def test_contract_min_adds_prediction_like_required_to_anyof_group() -> None:
    inventory = ["id", "feature_a"]
    strategy = {"required_columns": ["id", "feature_a"]}
    full_contract = {
        "artifact_requirements": {
            "required_files": [{"path": "data/scored_rows.csv"}],
            "scored_rows_schema": {"required_columns": ["survival_probability"]},
        }
    }
    contract_min = build_contract_min(full_contract, strategy, inventory, inventory)
    schema = contract_min.get("artifact_requirements", {}).get("scored_rows_schema", {})
    groups = schema.get("required_any_of_groups", [])
    assert any(
        isinstance(group, list) and "survival_probability" in group
        for group in groups
    )


def test_contract_min_aligns_decisioning_explanation_name() -> None:
    inventory = ["id", "feature_a"]
    strategy = {"required_columns": ["id", "feature_a"]}
    full_contract = {
        "decisioning_requirements": {
            "enabled": True,
            "required": True,
            "output": {
                "file": "data/scored_rows.csv",
                "required_columns": [
                    {"name": "top_drivers", "role": "explanation"},
                ],
            },
        },
        "artifact_requirements": {
            "required_files": [{"path": "data/scored_rows.csv"}],
            "scored_rows_schema": {"required_columns": ["explanation"]},
        },
    }
    contract_min = build_contract_min(full_contract, strategy, inventory, inventory)
    decisioning = contract_min.get("decisioning_requirements", {})
    required_cols = decisioning.get("output", {}).get("required_columns", [])
    assert any(
        isinstance(entry, dict) and entry.get("name") == "explanation"
        for entry in required_cols
    )
