from src.utils.contract_validator import normalize_artifact_requirements


def test_normalize_artifact_requirements_preserves_schema_binding() -> None:
    contract = {
        "artifact_requirements": {
            "required_files": [{"path": "data/cleaned_data.csv"}],
            "schema_binding": {
                "required_columns": ["A", "B"],
                "optional_passthrough_columns": ["PoolQC"],
            },
        }
    }
    artifact_req, _warnings = normalize_artifact_requirements(contract)
    schema_binding = artifact_req.get("schema_binding", {})

    assert schema_binding.get("required_columns") == ["A", "B"]
    assert schema_binding.get("optional_passthrough_columns") == ["PoolQC"]
