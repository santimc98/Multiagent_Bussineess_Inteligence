from src.graph.graph import _artifact_alignment_gate


def test_artifact_alignment_gate_accepts_list_schemas(tmp_path):
    cleaned_path = tmp_path / "cleaned.csv"
    scored_path = tmp_path / "scored.csv"
    cleaned_path.write_text(
        "CurrentPhase,1stYearAmount,Size,Debtors,Sector,Probability\n"
        "A,1,2,3,X,0.1\n"
        "B,2,3,4,Y,0.2\n",
        encoding="utf-8",
    )
    scored_path.write_text(
        "CurrentPhase,1stYearAmount,Size,Debtors,Sector,Probability,is_success,recommended_1stYearAmount,expected_value_at_recommendation\n"
        "A,1,2,3,X,0.1,1,1.5,200\n"
        "B,2,3,4,Y,0.2,0,2.5,250\n",
        encoding="utf-8",
    )
    contract = {
        "required_outputs": ["data/scored_rows.csv"],
        "spec_extraction": {"derived_columns": [{"name": "is_success"}]},
        "artifact_schemas": [
            {
                "path": "data/scored_rows.csv",
                "allowed_name_patterns": ["^recommended_.*", "^expected_.*"],
            }
        ],
    }
    issues = _artifact_alignment_gate(
        str(cleaned_path),
        str(scored_path),
        contract,
        None,
        ",",
        ".",
        "utf-8",
    )
    assert not any(issue.startswith("scored_rows_unknown_columns") for issue in issues)
