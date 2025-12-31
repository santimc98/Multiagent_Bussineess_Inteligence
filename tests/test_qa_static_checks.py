import pytest

from src.agents.qa_reviewer import run_static_qa_checks


def test_static_qa_blocks_target_noise():
    code = """
import numpy as np
y = df['target']
y = y + np.random.normal(0, 1, len(y))
"""
    result = run_static_qa_checks(code)
    assert result is not None
    assert result["status"] == "REJECTED"
    assert "target_variance_guard" in result.get("failed_gates", [])


def test_static_qa_blocks_missing_variance_guard():
    code = """
import pandas as pd
df = pd.DataFrame({'target': [1, 1, 1]})
print(df.shape)
"""
    result = run_static_qa_checks(code)
    assert result is not None
    assert result["status"] == "REJECTED"
    assert "target_variance_guard" in result.get("failed_gates", [])


def test_static_qa_allows_variance_guard():
    code = """
if df['target'].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
"""
    result = run_static_qa_checks(code)
    assert result is not None
    assert result["status"] in {"PASS", "WARN"}


def test_static_qa_blocks_split_fabrication():
    code = """
df[['a','b']] = df['raw'].str.split(';', expand=True)
"""
    result = run_static_qa_checks(code)
    assert result is not None
    assert "dialect_mismatch_handling" in result.get("failed_gates", [])


def test_static_qa_blocks_missing_group_split_when_inferred():
    code = """
from src.utils.group_split import infer_group_key
if df['target'].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
groups = infer_group_key(df, exclude_cols=['target'])
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
for tr, te in kf.split(df):
    pass
"""
    result = run_static_qa_checks(code)
    assert result is not None
    assert "group_split_required" in result.get("failed_gates", [])


def test_static_qa_allows_group_split_when_used():
    code = """
from src.utils.group_split import infer_group_key
from sklearn.model_selection import GroupKFold
if df['target'].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
groups = infer_group_key(df, exclude_cols=['target'])
gkf = GroupKFold(n_splits=3)
for tr, te in gkf.split(df, df['target'], groups):
    pass
"""
    result = run_static_qa_checks(code)
    assert result is not None
    assert result["status"] in {"PASS", "WARN"}


def test_static_qa_respects_explicit_gates_only():
    code = """
from sklearn.linear_model import LinearRegression
if df['target'].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
model = LinearRegression()
X = df[['a', 'b']]
y = df['target']
model.fit(X, y)
"""
    result = run_static_qa_checks(code, evaluation_spec={"qa_gates": ["target_variance_guard"]})
    assert result is not None
    assert result["status"] in {"PASS", "WARN"}
