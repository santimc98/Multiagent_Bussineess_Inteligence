from src.graph.graph import _looks_like_filesystem_path, _resolve_required_outputs


def test_looks_like_filesystem_path():
    assert _looks_like_filesystem_path("Priority Rank") is False
    assert _looks_like_filesystem_path("data/metrics.json") is True


def test_resolve_required_outputs_filters_conceptual():
    contract = {
        "evaluation_spec": {
            "required_outputs": [
                "Priority Rank",
                "data/metrics.json",
            ]
        }
    }
    state = {}
    outputs = _resolve_required_outputs(contract, state)

    assert outputs == ["data/metrics.json"]
    reporting = state.get("reporting_requirements", {})
    conceptual = reporting.get("conceptual_outputs", [])
    assert "Priority Rank" in conceptual
