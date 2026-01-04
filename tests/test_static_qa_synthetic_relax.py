from src.agents.qa_reviewer import run_static_qa_checks


def test_static_qa_allows_aux_dataframe_from_row():
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
if df["age"].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
row = df.iloc[0].to_dict()
single = pd.DataFrame([row])
print("Mapping Summary:", {"target": "age", "features": ["age"]})
"""
    evaluation_spec = {"qa_gates": ["target_variance_guard"], "canonical_columns": ["age"]}
    result = run_static_qa_checks(code, evaluation_spec)
    assert result is not None
    assert result.get("status") != "REJECTED"


def test_static_qa_rejects_literal_dataframe():
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
if df["age"].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
fake = pd.DataFrame({"a": [1, 2]})
print("Mapping Summary:", {"target": "age", "features": ["age"]})
"""
    evaluation_spec = {"qa_gates": ["target_variance_guard"], "canonical_columns": ["age"]}
    result = run_static_qa_checks(code, evaluation_spec)
    assert result is not None
    assert result.get("status") == "REJECTED"
    assert "no_synthetic_data" in (result.get("failed_gates") or [])
