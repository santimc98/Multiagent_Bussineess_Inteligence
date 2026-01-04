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


def test_reprompt_context_includes_critical_errors(monkeypatch):
    monkeypatch.setenv("ML_ENGINEER_PROVIDER", "zai")
    monkeypatch.setenv("ZAI_API_KEY", "dummy-zai")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()
    iteration_memory = [
        {
            "iteration_id": 1,
            "reviewer_reasons": ["Security violation"],
            "next_actions": ["Remove OS imports"],
        }
    ]
    gate_context = {
        "failed_gates": ["QA_CODE_AUDIT"],
        "feedback": "Price used as feature in optimization",
        "required_fixes": ["Remove price from MODEL_FEATURES"],
    }
    
    context = agent._build_incomplete_reprompt_context(
        execution_contract={},
        required_outputs=[],
        iteration_memory_block="",
        iteration_memory=iteration_memory,
        feedback_history=[],
        gate_context=gate_context,
    )
    
    assert "!!! CRITICAL ERRORS FROM PREVIOUS ATTEMPTS" in context
    assert "ATTEMPT 2 - REJECTED" in context
    assert "Error Type: QA_CODE_AUDIT" in context
    assert "Root Cause: Price used as feature in optimization" in context
    assert "Required Fix: Remove price from MODEL_FEATURES" in context
    
    assert "ATTEMPT 1 - REJECTED" in context
    assert "Error Type: Security violation" in context
    assert "Required Fix: Remove OS imports" in context
