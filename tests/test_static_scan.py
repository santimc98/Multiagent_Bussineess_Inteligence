
import pytest
from src.utils.static_safety_scan import scan_code_safety

def test_safe_code():
    code = """
import pandas as pd
import numpy as np
import os

df = pd.read_csv("data.csv")
os.makedirs("static/plots", exist_ok=True)
print("Hello")
    """
    is_safe, violations = scan_code_safety(code)
    assert is_safe, f"Safe code was blocked: {violations}"

def test_block_subprocess():
    code = """
import subprocess
subprocess.run(["ls", "-l"])
    """
    is_safe, violations = scan_code_safety(code)
    assert not is_safe
    assert any("subprocess" in v for v in violations)

def test_block_os_system():
    code = "import os; os.system('rm -rf /')"
    is_safe, violations = scan_code_safety(code)
    assert not is_safe
    assert any("os.system" in v for v in violations)

def test_block_requests():
    code = "import requests; requests.get('http://google.com')"
    is_safe, violations = scan_code_safety(code)
    assert not is_safe
    assert any("requests" in v for v in violations)

def test_block_pd_io():
    code = "import pandas as pd; pd.io.something()"
    is_safe, violations = scan_code_safety(code)
    assert not is_safe
    assert any("pandas.io" in v or "pd.io" in v for v in violations)

def test_block_exec():
    code = "exec('print(1)')"
    is_safe, violations = scan_code_safety(code)
    assert not is_safe
    assert any("exec" in v for v in violations)

def test_block_filesystem_exploration():
    code = "import os; print(os.listdir('.'))"
    is_safe, violations = scan_code_safety(code)
    assert not is_safe
    assert any("os.listdir" in v for v in violations)

def test_block_np_bool():
    code = "import numpy as np\nflag = np.bool(1)\n"
    is_safe, violations = scan_code_safety(code)
    assert not is_safe
    assert any("np.bool" in v for v in violations)

def test_block_int_sum_chain():
    code = """
import pandas as pd
mask = pd.Series([True, False])
bad = int(mask).sum()
    """
    is_safe, violations = scan_code_safety(code)
    assert not is_safe
    assert any("int(...)" in v for v in violations)

def test_allow_int_sum_scalar():
    code = """
import pandas as pd
mask = pd.Series([True, False])
ok = int(mask.sum())
    """
    is_safe, violations = scan_code_safety(code)
    assert is_safe, f"Valid sum pattern was blocked: {violations}"
