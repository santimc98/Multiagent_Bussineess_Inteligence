import json
import os

from src.utils.recommendations_preview import build_recommendations_preview


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_obj:
        json.dump(payload, f_obj)


def test_universal_extractor_items_key(tmp_path):
    reports_dir = tmp_path / "reports"
    _write_json(reports_dir / "generic.json", {"items": [{"title": "A", "reason": "R"}]})
    preview = build_recommendations_preview({}, {"run_outcome": "NO_GO"}, str(tmp_path), None)
    assert preview["items"]
    item = preview["items"][0]
    assert "title" in item
    assert "reason" in item
    assert "caveats" in item
    assert "source_path" in item


def test_universal_extractor_recommendations_key(tmp_path):
    reports_dir = tmp_path / "reports"
    _write_json(reports_dir / "steps.json", {"recommendations": [{"name": "Do X"}]})
    preview = build_recommendations_preview({}, {"run_outcome": "NO_GO"}, str(tmp_path), None)
    assert preview["items"]
    item = preview["items"][0]
    assert item.get("title") != ""
    assert "source_path" in item
