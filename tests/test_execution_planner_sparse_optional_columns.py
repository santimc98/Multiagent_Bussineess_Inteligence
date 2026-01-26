from src.agents.execution_planner import _apply_sparse_optional_columns


def test_sparse_columns_marked_optional_passthrough() -> None:
    contract = {
        "canonical_columns": ["Id", "PoolQC", "SalePrice", "Alley"],
        "column_roles": {
            "identifiers": ["Id"],
            "outcome": ["SalePrice"],
        },
        "artifact_requirements": {},
    }
    data_profile = {
        "missingness_top30": {
            "PoolQC": 0.9966,
            "Alley": 0.9322,
            "SalePrice": 0.4998,
        }
    }

    updated = _apply_sparse_optional_columns(contract, data_profile, threshold=0.98)
    schema = updated.get("artifact_requirements", {}).get("schema_binding", {})
    optional = schema.get("optional_passthrough_columns", [])

    assert "PoolQC" in optional
    assert "SalePrice" not in optional
    assert "Id" not in optional
    assert "Alley" not in optional


def test_sparse_optional_preserves_existing_order() -> None:
    contract = {
        "canonical_columns": ["Existing", "PoolQC"],
        "artifact_requirements": {
            "schema_binding": {
                "optional_passthrough_columns": ["Existing"],
            }
        },
    }
    data_profile = {"missingness_top30": {"Existing": 0.99, "PoolQC": 0.996}}

    updated = _apply_sparse_optional_columns(contract, data_profile, threshold=0.98)
    schema = updated.get("artifact_requirements", {}).get("schema_binding", {})
    optional = schema.get("optional_passthrough_columns", [])

    assert optional == ["Existing", "PoolQC"]


def test_sparse_optional_uses_available_columns_when_canonical_truncated() -> None:
    contract = {
        "canonical_columns": ["SalePrice"],
        "available_columns": ["SalePrice", "PoolQC"],
        "column_roles": {"SalePrice": "outcome"},
        "artifact_requirements": {},
    }
    data_profile = {"missingness_top30": {"PoolQC": 0.996}}

    updated = _apply_sparse_optional_columns(contract, data_profile, threshold=0.98)
    schema = updated.get("artifact_requirements", {}).get("schema_binding", {})
    optional = schema.get("optional_passthrough_columns", [])

    assert "PoolQC" in optional
