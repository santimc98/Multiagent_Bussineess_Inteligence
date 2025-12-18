from src.graph.graph import ml_quality_preflight


def test_ml_preflight_fails_missing_variance_guard():
    code = """
import pandas as pd
print("Mapping Summary: ...")
feature_cols = ["a", "b"]
X = df[feature_cols]
y = df["target"]
"""
    issues = ml_quality_preflight(code)
    assert "TARGET_VARIANCE_GUARD" in issues


def test_ml_preflight_fails_missing_mapping_summary():
    code = """
import pandas as pd
feature_cols = ["a", "b"]
X = df[feature_cols]
y = df["target"]
if y.nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
"""
    issues = ml_quality_preflight(code)
    assert "MAPPING_SUMMARY" in issues


def test_ml_preflight_passes_minimal_checks():
    code = """
import pandas as pd
feature_cols = ["a", "b"]
print("Mapping Summary: target -> y, features -> feature_cols")
X = df[feature_cols]
y = df["target"]
if y.nunique() < 2:
    raise ValueError("Target has no variance; cannot train meaningful model.")
"""
    issues = ml_quality_preflight(code)
    assert issues == []
