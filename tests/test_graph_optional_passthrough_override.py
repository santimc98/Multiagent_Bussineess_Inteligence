from src.graph.graph import _resolve_requirement_meta, _is_optional_requirement


def test_optional_passthrough_overrides_canonical() -> None:
    contract = {
        "canonical_columns": ["PoolQC", "Id"],
        "artifact_requirements": {
            "schema_binding": {
                "optional_passthrough_columns": ["PoolQC"],
            }
        },
    }

    meta = _resolve_requirement_meta(contract, "PoolQC")
    assert meta.get("required") is False
    assert _is_optional_requirement(meta) is True
