
import pytest
from unittest.mock import MagicMock, patch
from src.agents.data_engineer import DataEngineerAgent
from src.agents.ml_engineer import MLEngineerAgent
from src.utils.retries import call_with_retries

# Mock Response Object
class MockResponse:
    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]

# 1. Test Retry Utility
def test_retry_success():
    mock_func = MagicMock(return_value="Success")
    result = call_with_retries(mock_func, max_retries=3)
    assert result == "Success"
    assert mock_func.call_count == 1

# 2. Test Data Engineer - JSON Error Detection
def test_data_engineer_json_error_retry():
    agent = DataEngineerAgent(api_key="fake")
    
    # 1st call: Returns JSON error (simulation)
    # 2nd call: Returns valid code
    json_error = '{"error": {"message": "The engine is currently overloaded", "type": "engine_overloaded_error"}}'
    valid_code = "print('Success')"
    
    mock_create = MagicMock(side_effect=[
        MockResponse(json_error), # FAIL -> Retry 1
        MockResponse(valid_code)  # SUCCESS
    ])
    
    with patch.object(agent.client.chat.completions, 'create', side_effect=mock_create):
        # Should succeed eventually
        code = agent.generate_cleaning_script("audit", {"required_columns": []}, "path")
        assert code.strip().endswith("print('Success')")
        assert mock_create.call_count == 2 # Initial + 1 Retry

def test_data_engineer_overload_sentinel():
    agent = DataEngineerAgent(api_key="fake")
    agent.fallback_client = None
    agent.fallback_model_name = None
    
    # Always fails with JSON error
    json_error = '{"error": {"message": "Overloaded"}}'
    
    # Patch Retries to be fast
    with patch("time.sleep"): 
        with patch.object(agent.client.chat.completions, 'create', return_value=MockResponse(json_error)):
            result = agent.generate_cleaning_script("audit", {}, "path")
            
            # Must return sentinel, NOT raise to crash graph
            assert result.startswith("# Error:")
            assert "Overloaded" in result
