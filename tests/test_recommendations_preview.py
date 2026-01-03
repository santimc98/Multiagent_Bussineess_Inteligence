import json
import os

from src.utils.recommendations_preview import build_recommendations_preview


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_recommendations_preview_from_optimization_results(tmp_path):
    artifacts_dir = tmp_path
    opt_payload = {
        "recommendations": [
            {
                "segment": {"segment_key": "A"},
                "current_action": {"current_value": 1},
                "suggested_action": {"suggested_value": 2},
                "expected_effect": {"metric": "score", "delta": 0.1},
                "support": {"n": 5, "observed_support": True},
            }
        ]
    }
    _write_json(artifacts_dir / "reports" / "optimization_results.json", opt_payload)
    contract = {
        "counterfactual_policy": "counterfactual_supported",
        "recommendation_scope": "within_observed_support_only",
    }
    summary = {"run_outcome": "NO_GO"}
    preview = build_recommendations_preview(contract, summary, str(artifacts_dir), None)
    assert preview["items"]
    assert preview["items"][0]["segment"]["segment_key"] == "A"


def test_recommendations_preview_missing_artifacts(tmp_path):
    contract = {}
    summary = {"run_outcome": "NO_GO"}
    preview = build_recommendations_preview(contract, summary, str(tmp_path), None)
    assert preview["items"] == []
    assert preview["reason"]


def test_recommendations_preview_respects_max_examples(tmp_path):
    artifacts_dir = tmp_path
    items = []
    for idx in range(7):
        items.append(
            {
                "segment": {"segment_key": f"S{idx}"},
                "expected_effect": {"metric": "score", "delta": 0.1},
                "support": {"n": 5, "observed_support": True},
            }
        )
    _write_json(artifacts_dir / "reports" / "optimization_results.json", {"items": items})
    contract = {"reporting_policy": {"max_examples": 3}}
    preview = build_recommendations_preview(contract, {"run_outcome": "GO_WITH_LIMITATIONS"}, str(artifacts_dir), None)
    assert len(preview["items"]) == 3


def test_recommendations_preview_observational_policy_blocks_actions(tmp_path):
    artifacts_dir = tmp_path
    opt_payload = {
        "recommendations": [
            {
                "segment": {"segment_key": "B"},
                "current_action": {"current_value": 1},
                "suggested_action": {"suggested_value": 3},
                "expected_effect": {"metric": "score", "delta": 0.2},
                "support": {"n": 0, "observed_support": False},
            }
        ]
    }
    _write_json(artifacts_dir / "reports" / "optimization_results.json", opt_payload)
    contract = {"counterfactual_policy": "observational_only"}
    preview = build_recommendations_preview(contract, {"run_outcome": "NO_GO"}, str(artifacts_dir), None)
    assert preview["items"]
    assert preview["items"][0]["suggested_action"] == {}
