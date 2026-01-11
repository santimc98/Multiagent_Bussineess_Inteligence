import json
from pathlib import Path

from src.utils.contract_views import (
    build_de_view,
    build_ml_view,
    build_reviewer_view,
    build_translator_view,
    persist_views,
    trim_to_budget,
)


def _load_fixture(name: str):
    base = Path(__file__).parent / "fixtures" / name
    with base.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_de_view_excludes_prohibited_fields():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    de_view = build_de_view(contract_full, contract_min, artifact_index)
    payload = json.dumps(de_view, ensure_ascii=True)
    assert "strategy_rationale" not in de_view
    assert "case_rules" not in de_view
    assert "weights" not in payload
    assert "optimization" not in payload
    assert de_view.get("required_columns")


def test_ml_view_includes_required_fields():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    ml_view = build_ml_view(contract_full, contract_min, artifact_index)
    assert ml_view.get("required_outputs")
    assert "forbidden_features" in ml_view
    assert isinstance(ml_view.get("forbidden_features"), list)
    assert ml_view.get("objective_type")


def test_ml_view_inherits_roles_when_min_lax():
    contract_min = {
        "canonical_columns": ["feature_a", "target", "audit_col", "entity_id"],
        "column_roles": {
            "pre_decision": ["feature_a", "target", "audit_col", "entity_id"]
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a", "target", "audit_col", "entity_id"],
            "segmentation_features": ["feature_a", "entity_id"],
            "forbidden_features": [],
        },
    }
    contract_full = {
        "canonical_columns": ["feature_a", "target", "audit_col", "entity_id"],
        "column_roles": {
            "pre_decision": ["feature_a", "entity_id"],
            "outcome": ["target"],
            "post_decision_audit_only": ["audit_col"],
        },
    }
    ml_view = build_ml_view(contract_full, contract_min, [])
    forbidden = set(ml_view.get("forbidden_features") or [])
    assert "audit_col" in forbidden
    assert "target" in forbidden
    assert "audit_col" not in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])
    assert "target" not in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])


def test_ml_view_excludes_identifier_columns():
    contract_min = {
        "canonical_columns": ["EntityId", "feature_a"],
        "column_roles": {
            "pre_decision": ["EntityId", "feature_a"]
        },
        "allowed_feature_sets": {
            "model_features": ["EntityId", "feature_a"],
            "segmentation_features": ["EntityId", "feature_a"],
            "forbidden_features": [],
        },
    }
    ml_view = build_ml_view({}, contract_min, [])
    assert "EntityId" in (ml_view.get("identifier_columns") or [])
    assert "EntityId" not in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])


def test_ml_view_preserves_forbidden_features_from_min():
    contract_min = {
        "canonical_columns": ["feature_a", "audit_col", "EntityId"],
        "column_roles": {
            "pre_decision": ["feature_a", "audit_col", "EntityId"],
            "post_decision_audit_only": ["audit_col"],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a", "audit_col", "EntityId"],
            "segmentation_features": ["feature_a", "EntityId"],
            "forbidden_features": ["audit_col"],
        },
    }
    ml_view = build_ml_view({}, contract_min, [])
    forbidden = set(ml_view.get("forbidden_features") or [])
    assert forbidden == {"audit_col", "EntityId"}
    assert "feature_a" in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])
    assert "audit_col" not in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])


def test_ml_view_prefers_full_allowed_feature_sets():
    contract_full = {
        "canonical_columns": ["feature_a", "feature_b", "audit_col", "EntityId"],
        "allowed_feature_sets": {
            "model_features": ["feature_a", "audit_col", "EntityId"],
            "segmentation_features": ["feature_b", "EntityId"],
            "forbidden_for_modeling": ["audit_col"],
            "audit_only_features": ["audit_col"],
        },
    }
    contract_min = {
        "canonical_columns": ["feature_a", "feature_b", "audit_col", "EntityId"],
        "allowed_feature_sets": {
            "model_features": ["feature_b"],
            "segmentation_features": ["feature_a"],
            "forbidden_features": [],
        },
    }
    ml_view = build_ml_view(contract_full, contract_min, [])
    allowed = ml_view.get("allowed_feature_sets") or {}
    assert allowed.get("model_features") == ["feature_a"]
    assert allowed.get("segmentation_features") == ["feature_b"]
    assert ml_view.get("audit_only_columns") == ["audit_col"]
    assert "EntityId" in (ml_view.get("identifier_columns") or [])


def test_reviewer_view_contains_gates_and_outputs():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    reviewer_view = build_reviewer_view(contract_full, contract_min, artifact_index)
    assert reviewer_view.get("reviewer_gates")
    assert reviewer_view.get("required_outputs")


def test_translator_view_contains_policy_and_inventory():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    translator_view = build_translator_view(contract_full, contract_min, artifact_index)
    assert translator_view.get("reporting_policy")
    assert translator_view.get("evidence_inventory")


def test_trim_to_budget_preserves_required_fields():
    payload = {
        "required_columns": ["a", "b"],
        "required_outputs": ["data/metrics.json"],
        "forbidden_features": ["x"],
        "gates": ["gate_a"],
        "long_text": "x" * 5000,
        "items": list(range(200)),
    }
    trimmed = trim_to_budget(payload, max_chars=900)
    assert len(json.dumps(trimmed, ensure_ascii=True)) <= 900
    assert trimmed.get("required_columns") == ["a", "b"]
    assert trimmed.get("required_outputs") == ["data/metrics.json"]
    assert trimmed.get("forbidden_features") == ["x"]
    assert trimmed.get("gates") == ["gate_a"]


def test_persist_views_writes_files(tmp_path):
    views = {
        "de_view": {"role": "data_engineer"},
        "ml_view": {"role": "ml_engineer"},
        "reviewer_view": {"role": "reviewer"},
        "translator_view": {"role": "translator"},
    }
    paths = persist_views(views, base_dir=str(tmp_path))
    assert (tmp_path / "contracts" / "views" / "de_view.json").exists()
    assert (tmp_path / "contracts" / "views" / "ml_view.json").exists()
    assert paths.get("de_view")
