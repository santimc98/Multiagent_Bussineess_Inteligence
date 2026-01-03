from src.graph.graph import ml_quality_preflight
from src.graph.graph import run_ml_preflight
from src.graph.graph import _detect_synthetic_data


def test_ml_preflight_fails_missing_variance_guard():
    code = """
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
print("Mapping Summary: ...")
feature_cols = ["a", "b"]
X = df[feature_cols]
y = df["target"]
imputer = SimpleImputer()
baseline = DummyClassifier()
"""
    issues = ml_quality_preflight(code)
    assert "TARGET_VARIANCE_GUARD" in issues


def test_ml_preflight_fails_missing_mapping_summary():
    code = """
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
feature_cols = ["a", "b"]
X = df[feature_cols]
y = df["target"]
if y.nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
imputer = SimpleImputer()
baseline = DummyClassifier()
"""
    issues = ml_quality_preflight(code)
    assert "MAPPING_SUMMARY" in issues


def test_ml_preflight_passes_minimal_checks():
    code = """
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
feature_cols = ["a", "b"]
print("Mapping Summary: target -> y, features -> feature_cols")
X = df[feature_cols]
y = df["target"]
if y.nunique() < 2:
    raise ValueError("Target has no variance; cannot train meaningful model.")
imputer = SimpleImputer()
baseline = DummyClassifier()
"""
    issues = ml_quality_preflight(code)
    assert issues == []


def test_ml_preflight_blocks_dependency_with_suggestion():
    state = {
        "generated_code": "import pulp\n",
        "execution_contract": {"required_dependencies": []},
        "feedback_history": [],
    }
    result = run_ml_preflight(state)
    assert result.get("ml_preflight_failed") is True
    history = result.get("feedback_history", [])
    assert any("DEPENDENCY_BLOCKED" in h for h in history)
    assert any("linprog" in h.lower() for h in history)


def test_ml_preflight_flags_dataframe_literal_overwrite():
    code = """
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
print("Mapping Summary: ...")
df = pd.DataFrame({"age": [1, 2], "income": [3, 4]})
feature_cols = ["age"]
X = df[feature_cols]
y = df["income"]
if y.nunique() < 2:
    raise ValueError("Target has no variance; cannot train meaningful model.")
imputer = SimpleImputer()
baseline = DummyClassifier()
"""
    issues = ml_quality_preflight(code, allowed_columns=["age", "income"])
    assert "DATAFRAME_LITERAL_OVERWRITE" in issues


def test_ml_preflight_flags_unknown_columns_from_literals():
    code = """
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
print("Mapping Summary: ...")
feature_cols = ["a"]
X = df[feature_cols]
y = df["target"]
if y.nunique() < 2:
    raise ValueError("Target has no variance; cannot train meaningful model.")
age = df["age"]
imputer = SimpleImputer()
baseline = DummyClassifier()
"""
    issues = ml_quality_preflight(code, allowed_columns=["a", "target"])
    assert "UNKNOWN_COLUMNS_REFERENCED" in issues


def test_ml_preflight_requires_baseline():
    code = """
import pandas as pd
from sklearn.impute import SimpleImputer
print("Mapping Summary: ...")
feature_cols = ["a", "b"]
X = df[feature_cols]
y = df["target"]
if y.nunique() < 2:
    raise ValueError("Target has no variance; cannot train meaningful model.")
imputer = SimpleImputer()
"""
    issues = ml_quality_preflight(code)
    assert "BASELINE_REQUIRED" in issues


def test_ml_preflight_requires_imputer():
    code = """
import pandas as pd
from sklearn.dummy import DummyClassifier
print("Mapping Summary: ...")
feature_cols = ["a", "b"]
X = df[feature_cols]
y = df["target"]
if y.nunique() < 2:
    raise ValueError("Target has no variance; cannot train meaningful model.")
baseline = DummyClassifier()
"""
    issues = ml_quality_preflight(code)
    assert "IMPUTER_REQUIRED" in issues


def test_ml_preflight_blocks_random_inside_function():
    code = """
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
print("Mapping Summary: ...")
feature_cols = ["a", "b"]
X = df[feature_cols]
y = df["target"]
if y.nunique() < 2:
    raise ValueError("Target has no variance; cannot train meaningful model.")
def sample_noise():
    return np.random.randn(10)
imputer = SimpleImputer()
baseline = DummyClassifier()
"""
    issues = ml_quality_preflight(code, allowed_columns=["a", "b", "target"])
    assert "SYNTHETIC_DATA_DETECTED" in issues


def test_ml_preflight_flags_scored_rows_delta():
    code = """
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
print("Mapping Summary: ...")
feature_cols = ["a", "b"]
X = df[feature_cols]
y = df["target"]
if y.nunique() < 2:
    raise ValueError("Target has no variance; cannot train meaningful model.")
scored_rows = df.copy()
scored_rows["price_delta"] = scored_rows["a"] - 1
imputer = SimpleImputer()
baseline = DummyClassifier()
"""
    issues = ml_quality_preflight(code, allowed_columns=["a", "b", "target"])
    assert "SCORED_ROWS_SCHEMA_VIOLATION" in issues


def test_synthetic_detector_allows_counterfactual_grid():
    code = """
import pandas as pd
feature_cols = ["Size", "Debtors", "Sector", "1stYearAmount"]
df_seg = df[df["Sector"] == "A"]
sim_df = df_seg.copy()
for price in [100, 200, 300]:
    sim_df["1stYearAmount"] = price
    _ = sim_df[feature_cols]
sampled = df.sample(n=10, random_state=42)
"""
    assert _detect_synthetic_data(code) is False


def test_synthetic_detector_blocks_generators():
    code = """
import numpy as np
from faker import Faker
from sklearn.datasets import make_classification
faker = Faker()
X, y = make_classification(n_samples=10, n_features=3)
noise = np.random.randn(10)
"""
    assert _detect_synthetic_data(code) is True
