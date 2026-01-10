from src.agents.execution_planner import build_reporting_policy


def test_reporting_policy_builder_has_sections_and_slots():
    execution_plan = {
        "outputs": [
            {"artifact_type": "metrics", "required": True},
            {"artifact_type": "predictions", "required": True},
        ]
    }
    policy = build_reporting_policy(execution_plan, strategy=None)
    assert isinstance(policy, dict)
    assert policy.get("sections")
    assert policy.get("slots")
