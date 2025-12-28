
import pytest
from unittest.mock import MagicMock, patch
import os
import shutil
import pandas as pd
from src.graph.graph import execute_code

@pytest.fixture
def mock_sandbox_env(tmp_path):
    # Setup directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    plots_dir = tmp_path / "static/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create valid cleaned data
    df = pd.DataFrame({'price': [10, 20, 30], 'category': ['A', 'B', 'A']})
    df.to_csv(data_dir / "cleaned_data.csv", index=False)
    
    # Change CWD for the test
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)

def test_execute_code_triggers_fallback(mock_sandbox_env):
    """
    Test that execute_code generates fallback plots when sandbox returns none.
    """
    # Mock State
    state = {
        'generated_code': 'print("Hello")',
        'execution_output': '',
        'plots_local': [],
        'has_partial_visuals': False
    }

    # Mock Sandbox and Safety Scan
    with patch('src.graph.graph.Sandbox') as MockSandbox, \
         patch('src.graph.graph.scan_code_safety', return_value=(True, [])) as mock_scan, \
         patch.dict(os.environ, {"E2B_API_KEY": "dummy"}):
         
         # Configure Sandbox Mock to return success but NO PLOTS
         mock_instance = MockSandbox.create.return_value.__enter__.return_value
         mock_instance.run_code.return_value.error = None
         mock_instance.run_code.return_value.logs.stdout = ["Done"]
         mock_instance.run_code.return_value.logs.stderr = []
         
         # Mock ls command to return nothing (simulating no plots generated remotely)
         mock_instance.commands.run.return_value.exit_code = 0
         mock_instance.commands.run.return_value.stdout = "" 
         
         # Run
         result = execute_code(state)
         
         # Assert
         assert result['has_partial_visuals'] is False
         assert len(result['plots_local']) == 0
         assert any('fallback_' in p for p in os.listdir('static/plots'))
