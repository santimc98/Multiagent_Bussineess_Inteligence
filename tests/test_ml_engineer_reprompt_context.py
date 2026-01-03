from src.agents.ml_engineer import MLEngineerAgent


class FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers


def test_incomplete_reprompt_context_has_contract_and_outputs(monkeypatch):
    monkeypatch.setenv("ML_ENGINEER_PROVIDER", "zai")
    monkeypatch.setenv("ZAI_API_KEY", "dummy-zai")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()
    contract = {
        "required_columns": [f"col_{i}" for i in range(300)],
        "required_outputs": ["data/metrics.json", "data/alignment_check.json"],
        "artifact_schemas": {
            "data/scored_rows.csv": {"allowed_name_patterns": ["^segment_.*"]}
        },
    }
    context = agent._build_incomplete_reprompt_context(
        execution_contract=contract,
        required_outputs=contract["required_outputs"],
        iteration_memory_block="ITERATION_MEMORY_CONTEXT: last attempt failed",
        iteration_memory=[],
        feedback_history=["REVIEWER FEEDBACK: add baseline metrics"],
        gate_context={"feedback": "QA TEAM FEEDBACK: fix outputs"},
    )
    assert "CONTRACT_MIN_CONTEXT" in context
    assert "REQUIRED OUTPUTS" in context
    assert len(context) > 3000
    assert "..." not in context
