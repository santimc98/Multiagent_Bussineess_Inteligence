from src.agents.ml_engineer import MLEngineerAgent


def test_ml_engineer_prompt_renders_decisioning_and_visual_context():
    agent = MLEngineerAgent.__new__(MLEngineerAgent)
    ml_view = {
        "decisioning_requirements": {
            "enabled": True,
            "policy_notes": "Use rank tiers.",
            "output": {"required_columns": [{"name": "priority_rank"}]},
        },
        "visual_requirements": {
            "enabled": True,
            "items": [{"id": "dist"}],
        },
    }
    template = (
        "DECISIONING REQUIREMENTS CONTEXT:\n$decisioning_requirements_context\n"
        "DECISIONING COLUMNS:\n$decisioning_columns_text\n"
        "VISUAL REQUIREMENTS:\n$visual_requirements_context\n"
    )
    prompt = agent._build_system_prompt(template, {}, ml_view=ml_view, execution_contract={})
    assert "$decisioning_requirements_context" not in prompt
    assert "$visual_requirements_context" not in prompt
    assert "DECISIONING REQUIREMENTS CONTEXT" in prompt
    assert "VISUAL REQUIREMENTS" in prompt
    assert "priority_rank" in prompt
    assert "dist" in prompt
