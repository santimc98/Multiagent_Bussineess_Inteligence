import pandas as pd

from src.agents.cleaning_reviewer import _evaluate_gates_deterministic, normalize_gate_name


def test_null_handling_verification_only_target_columns():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [None, None, 1, None, None]})
    gates = [
        {
            "name": "Null_Handling_Verification",
            "severity": "HARD",
            "params": {"columns": ["A"], "allow_nulls": False},
        }
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest={},
        raw_sample=None,
        column_roles={},
        allowed_feature_sets={"model_features": ["A", "B"]},
    )

    assert result["status"] == "APPROVED"
    assert not result.get("hard_failures")
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "null_handling_verification"
    )
    assert gate_entry.get("passed") is True
