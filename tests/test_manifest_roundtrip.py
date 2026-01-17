
import pytest
from unittest.mock import MagicMock, patch, mock_open
import os
from src.graph.graph import execute_code

def test_manifest_roundtrip_upload():
    # Setup State
    state = {
        "generated_code": "import pandas as pd\n# Load Data\ndf = pd.read_csv('data/cleaned_data.csv')",
        "execution_output": "",
        "execution_attempt": 1,
        "run_id": "testrun"
    }
    
    # Mock Sandbox and File Ops
    with patch("src.graph.graph.Sandbox") as MockSandbox, \
         patch("src.graph.graph.os.path.exists") as mock_exists, \
         patch("builtins.open", mock_open(read_data=b"manifest_json_content")) as mock_file, \
         patch("src.graph.graph.scan_code_safety", return_value=(True, [])), \
         patch.dict(os.environ, {"E2B_API_KEY": "mock_key"}):
         
        # Configure Mock Sandbox interactions explicitly
        mock_sb_instance = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_sb_instance
        MockSandbox.create.return_value = mock_ctx

        mock_instance = mock_sb_instance
        
        # Setup run_code return values
        mock_run = MagicMock()
        mock_run.exit_code = 0
        mock_run.stdout = ""
        mock_instance.commands.run.return_value = mock_run

        mock_exec = MagicMock()
        mock_exec.logs.stdout = ["Success"]
        mock_exec.logs.stderr = []
        mock_exec.error = None
        mock_instance.run_code.return_value = mock_exec

        # Simulate local manifest exists
        # Side effect: True for manifest, True for plots dir? Simplified: True always.
        mock_exists.return_value = True 
        
        # Execute Node
        execute_code(state)
        
        # ASSERTIONS
        
        # 1. Verify Upload Call
        # Should upload manifest to /home/user/run/<run_id>/attempt_<k>/cleaning_manifest.json
        expected_suffix = "cleaning_manifest.json"
        
        # Check that we wrote the manifest twice (canonical + root)
        # also we wrote the csv once
        assert mock_instance.files.write.call_count >= 3 # csv + manifest*2
        
        # 2. Verify Code Patching
        # The code passed to run_code should have the remote path
        args, _ = mock_instance.run_code.call_args
        executed_code = args[0]
        
        assert "cleaning_manifest.json" in executed_code
        # Ensure older path is NOT present if we replaced it (assuming input had it)
        # But input didn't have it in this simple string. Let's make input have it
        
def test_manifest_patching_logic():
    state = {
        "generated_code": "import json\nimport pandas as pd\nmd = json.load(open('data/cleaning_manifest.json'))\ndf = pd.read_csv('./data/cleaned_data.csv')",
        "execution_output": "",
        "execution_attempt": 1,
        "csv_path": "data/dummy.csv", # Required by checks? No, but maybe.
        "run_id": "testrun"
    }
    
    with patch("src.graph.graph.Sandbox") as MockSandbox, \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=b"{}")), \
         patch("src.graph.graph.scan_code_safety", return_value=(True, [])), \
         patch.dict(os.environ, {"E2B_API_KEY": "mock_key"}):
         
        mock_instance = MockSandbox.create.return_value.__enter__.return_value
        mock_instance.commands.run.return_value.exit_code = 0
        mock_instance.commands.run.return_value.stdout = ""
        mock_instance.run_code.return_value.logs.stdout = ["ok"]
        mock_instance.run_code.return_value.logs.stderr = []
        mock_instance.run_code.return_value.error = None
        
        # Capture executed code
        result = execute_code(state)
        
        # Verify run_code called
        if not mock_instance.run_code.called:
            pytest.fail("sandbox.run_code was NOT called. Execution logic skipped?")
            
        args, _ = mock_instance.run_code.call_args
        executed_code = args[0]
        
        assert "/cleaning_manifest.json" in executed_code
        assert "/home/user/run/testrun/ml_engineer/" in executed_code
        assert "data/cleaning_manifest.json" not in executed_code # Replaced
