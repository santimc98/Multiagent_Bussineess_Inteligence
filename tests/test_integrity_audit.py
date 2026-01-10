"""
Integrity Audit Tests - V4.1 Compatible

Tests for integrity_audit using V4.1 contract schema:
- canonical_columns for input columns
- column_roles for role mappings  
- validation_requirements.column_validations for validation rules
- preprocessing_requirements.expected_kinds for type expectations
"""

import pandas as pd

from src.utils.integrity_audit import run_integrity_audit


def test_percent_scale_suspected():
    """V4.1: Test percent scale detection using column_roles and validation_requirements."""
    df = pd.DataFrame({"col_pct": [50, 100, None]})
    contract = {
        "canonical_columns": ["col_pct"],
        "column_roles": {
            "percentage": ["col_pct"]
        },
        "validation_requirements": {
            "column_validations": [
                {"column": "col_pct", "expected_range": [0, 1], "allowed_null_frac": 0.5}
            ]
        }
    }
    issues, stats = run_integrity_audit(df, contract)
    types = {i["type"] for i in issues}
    assert "PERCENT_SCALE_SUSPECTED" in types
    assert "MISSING_COLUMN" not in types


def test_categorical_destroyed_and_aliasing():
    """V4.1: Test categorical detection using column_roles."""
    df = pd.DataFrame({"col_a": [None] * 10 + ["x"]})
    contract = {
        "canonical_columns": ["Col_A", "col_a"],
        "column_roles": {
            "categorical": ["Col_A", "col_a"]
        }
    }
    issues, stats = run_integrity_audit(df, contract)
    types = {i["type"] for i in issues}
    assert "CATEGORICAL_DESTROYED_BY_PARSING" in types
    assert "ALIASING_RISK" in types


def test_percent_scale_suspected_high_values():
    """V4.1: Test percent scale detection with high values."""
    df = pd.DataFrame({"pct": [10, 20, 30, 40, 50]})
    contract = {
        "canonical_columns": ["pct"],
        "column_roles": {
            "percentage": ["pct"]
        },
        "validation_requirements": {
            "column_validations": [
                {"column": "pct", "expected_range": [0, 1]}
            ]
        },
        "validations": [],
    }
    issues, _ = run_integrity_audit(df, contract)
    assert any(i.get("type") == "PERCENT_SCALE_SUSPECTED" for i in issues)


def test_unknown_column_roles_not_enforced():
    """Unknown column_roles should not trigger missing-column issues."""
    df = pd.DataFrame({"req": [1, 2, 3], "extra": [5, 6, 7]})
    unknown_columns = [f"col_unknown_{i}" for i in range(50)]
    contract = {
        "canonical_columns": ["req"],
        "column_roles": {
            "pre_decision": ["req"],
            "unknown": unknown_columns
        }
    }
    issues, _ = run_integrity_audit(df, contract)
    missing = [issue for issue in issues if issue.get("type") == "MISSING_COLUMN"]
    assert not missing, "Unknown column_roles should not enforce requirements"


def test_optional_passthrough_missing_is_warning_only():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    contract = {
        "artifact_requirements": {
            "clean_dataset": {"required_columns": ["A", "B"]},
            "schema_binding": {"optional_passthrough_columns": ["C"]},
        }
    }
    issues, _ = run_integrity_audit(df, contract)
    warnings = [i for i in issues if i.get("severity") == "warning"]
    critical = [i for i in issues if i.get("severity") == "critical"]
    assert any(i.get("type") == "OPTIONAL_COLUMN_MISSING" for i in warnings)
    assert len(critical) == 0


def test_missing_required_is_critical():
    df = pd.DataFrame({"A": [1, 2]})
    contract = {
        "artifact_requirements": {
            "clean_dataset": {"required_columns": ["A", "B"]},
            "schema_binding": {"optional_passthrough_columns": ["C"]},
        }
    }
    issues, _ = run_integrity_audit(df, contract)
    critical = [i for i in issues if i.get("severity") == "critical"]
    assert any(i.get("type") == "MISSING_COLUMN" for i in critical)
