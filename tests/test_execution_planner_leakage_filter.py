from src.agents.execution_planner import _filter_leakage_audit_features


def test_filter_leakage_audit_features_removes_unknown_columns():
    spec = {
        "leakage_policy": {"audit_features": ["CurrentPhase", "Prob", "unknown_feature"]},
    }
    filtered = _filter_leakage_audit_features(spec, ["CurrentPhase"], ["Prob"])
    assert "unknown_feature" in filtered
    assert spec["leakage_policy"]["audit_features"] == ["CurrentPhase", "Prob"]
    detail = spec.get("leakage_policy_detail")
    assert isinstance(detail, dict)
    assert "filtered_audit_features" in detail
    assert detail["filtered_audit_features"] == filtered


def test_filter_leakage_audit_features_drops_all_when_no_known_columns():
    spec = {"leakage_policy": {"audit_features": ["foo", "bar"]}}
    filtered = _filter_leakage_audit_features(spec, None, None)
    assert filtered == ["foo", "bar"]
    assert spec["leakage_policy"]["audit_features"] == []
