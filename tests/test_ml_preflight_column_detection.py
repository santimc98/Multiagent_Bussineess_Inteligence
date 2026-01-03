from src.graph.graph import _detect_forbidden_df_assignments
from src.graph.graph import _detect_unknown_columns
from src.graph.graph import ml_quality_preflight


def test_detect_unknown_columns_ignores_dict_keys():
    script = """
checks = {"status": "PASS", "col": "Size"}
print(checks["status"])
"""
    allowed_cols = ["Size"]
    assert _detect_unknown_columns(script, allowed_cols, []) == []
    assert _detect_forbidden_df_assignments(script, allowed_cols, []) == []


def test_detect_unknown_columns_from_df_access():
    script = """
x = df["Size"]
y = df.loc[:, "Debtors"]
z = df[["Sector", "Size"]]
"""
    allowed_cols = ["Size", "Debtors"]
    unknown = _detect_unknown_columns(script, allowed_cols, [])
    assert unknown == ["Sector"]


def test_detect_forbidden_df_assignments_and_preflight_issue():
    script = """
df["Size_bin"] = 1
df.loc[:, "segment_key"] = "A"
"""
    allowed_cols = ["Size", "Debtors", "Sector"]
    patterns = ["^segment_.*"]
    forbidden = _detect_forbidden_df_assignments(script, allowed_cols, patterns)
    assert forbidden == ["Size_bin", "segment_key"]
    issues = ml_quality_preflight(script, allowed_columns=allowed_cols, allowed_patterns=patterns)
    assert "DF_COLUMN_ASSIGNMENT_FORBIDDEN" in issues


def test_detect_df_assign_forbidden_columns():
    script = """
df = df.assign(Size_bin=1)
"""
    allowed_cols = ["Size", "Debtors", "Sector"]
    forbidden = _detect_forbidden_df_assignments(script, allowed_cols, [])
    assert forbidden == ["Size_bin"]
