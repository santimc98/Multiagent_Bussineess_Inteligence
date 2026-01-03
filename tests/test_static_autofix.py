from src.graph.graph import _apply_static_autofixes
from src.utils.static_safety_scan import scan_code_safety


def test_auto_fix_np_bool_passes_static_scan():
    code = "import numpy as np\nflag = np.bool(1)\n"
    fixed, fixes = _apply_static_autofixes(code)
    assert "np.bool_" in fixed
    assert fixes
    is_safe, violations = scan_code_safety(fixed)
    assert is_safe
    assert not violations
