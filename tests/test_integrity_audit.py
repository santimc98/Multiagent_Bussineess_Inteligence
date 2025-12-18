import pandas as pd

from src.utils.integrity_audit import run_integrity_audit


def test_percent_scale_suspected():
    df = pd.DataFrame({"col_pct": [50, 100, None]})
    contract = {
        "data_requirements": [
            {"name": "col_pct", "role": "percentage", "expected_range": [0, 1], "allowed_null_frac": 0.5}
        ]
    }
    issues, stats = run_integrity_audit(df, contract)
    types = {i["type"] for i in issues}
    assert "PERCENT_SCALE_SUSPECTED" in types
    assert "MISSING_COLUMN" not in types


def test_categorical_destroyed_and_aliasing():
    df = pd.DataFrame({"col_a": [None] * 10 + ["x"]})
    contract = {
        "data_requirements": [
            {"name": "Col_A", "role": "categorical"},
            {"name": "col_a", "role": "categorical"},
        ]
    }
    issues, stats = run_integrity_audit(df, contract)
    types = {i["type"] for i in issues}
    assert "CATEGORICAL_DESTROYED_BY_PARSING" in types
    assert "ALIASING_RISK" in types


def test_percent_scale_suspected_high_values():
    df = pd.DataFrame({"pct": [10, 20, 30, 40, 50]})
    contract = {
        "data_requirements": [
            {"name": "pct", "role": "percentage", "expected_range": [0, 1]}
        ],
        "validations": [],
    }
    issues, _ = run_integrity_audit(df, contract)
    assert any(i.get("type") == "PERCENT_SCALE_SUSPECTED" for i in issues)
