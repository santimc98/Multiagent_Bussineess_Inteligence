
import pytest
from unittest.mock import MagicMock, patch
from src.graph.graph import check_execution_status, AgentState

def test_check_execution_status_retry():
    state = {
        "execution_output": "Traceback (most recent call last):\nValueError: bad",
        "execution_attempt": 1
    }
    assert check_execution_status(state) == "retry_fix"

def test_check_execution_status_evaluate():
    state = {
        "execution_output": "Success",
        "execution_attempt": 1
    }
    assert check_execution_status(state) == "evaluate"

def test_check_execution_status_max_retries():
    state = {
        "execution_output": "Traceback (most recent call last):\nValueError: bad",
        "execution_attempt": 4,
        "runtime_fix_count": 3,
        "max_runtime_fix_attempts": 3,
    }
    assert check_execution_status(state) == "failed_runtime"
