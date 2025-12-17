import json

from src.agents.postmortem import PostmortemAgent


def test_postmortem_runtime_error_retry_ml():
    agent = PostmortemAgent(api_key=None)  # force fallback
    ctx = {
        "execution_output": "Traceback: EXECUTION ERROR",
        "integrity_issues": [],
        "iteration_count": 0,
        "restrategize_count": 0,
    }
    decision = agent.decide(ctx)
    assert decision["action"] == "retry_ml_engineer"


def test_postmortem_integrity_missing_retry_de():
    agent = PostmortemAgent(api_key=None)
    ctx = {
        "execution_output": "",
        "integrity_issues": [{"severity": "critical", "type": "MISSING_COLUMN"}],
        "iteration_count": 0,
        "restrategize_count": 0,
    }
    decision = agent.decide(ctx)
    assert decision["action"] == "retry_data_engineer"


def test_postmortem_restrategize_on_many_iterations():
    agent = PostmortemAgent(api_key=None)
    ctx = {
        "execution_output": "",
        "integrity_issues": [],
        "iteration_count": 3,
        "restrategize_count": 0,
    }
    decision = agent.decide(ctx)
    assert decision["action"] in {"re_strategize", "stop"}


def test_postmortem_dialect_guard_retry_de():
    agent = PostmortemAgent(api_key=None)
    ctx = {
        "execution_output": "",
        "integrity_issues": [],
        "error_message": "CRITICAL: pd.read_csv must use provided dialect parameters.",
        "iteration_count": 0,
        "restrategize_count": 0,
        "missing_repeat_count": 0,
    }
    decision = agent.decide(ctx)
    assert decision["action"] == "retry_data_engineer"


def test_postmortem_cleaning_failed_retry_de():
    agent = PostmortemAgent(api_key=None)
    ctx = {
        "execution_output": "",
        "integrity_issues": [],
        "error_message": "Sandbox Cleaning Failed: json not serializable",
        "iteration_count": 0,
        "restrategize_count": 0,
        "missing_repeat_count": 0,
    }
    decision = agent.decide(ctx)
    assert decision["action"] == "retry_data_engineer"
