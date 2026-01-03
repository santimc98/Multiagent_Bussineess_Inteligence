import csv
import json
import os

from src.utils.segment_enrichment import enrich_segmented_summary_json


def _write_summary(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_obj:
        json.dump(payload, f_obj)


def _write_scored_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f_obj:
        writer = csv.DictWriter(f_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_summary(path):
    with open(path, "r", encoding="utf-8") as f_obj:
        return json.load(f_obj)


def test_segment_metadata_unavailable_without_label(tmp_path):
    summary_path = tmp_path / "reports" / "summary.json"
    scored_path = tmp_path / "data" / "scored_rows.csv"
    _write_summary(summary_path, {"segments": [{"segment_id": "A"}]})
    _write_scored_rows(
        scored_path,
        rows=[{"Sector": "A", "value": "1"}],
        fieldnames=["Sector", "value"],
    )

    assert enrich_segmented_summary_json(
        str(summary_path),
        str(scored_path),
        segment_label_column=None,
    )

    updated = _load_summary(summary_path)
    assert updated.get("segment_metadata_unavailable") is True
    assert "segment_label_column_unavailable" in updated.get("segment_metadata_reason", "")

    with open(scored_path, "r", encoding="utf-8") as f_obj:
        header = f_obj.readline().strip().split(",")
    assert "segment_id" not in header


def test_segment_id_mismatch_adds_caveat_without_sizes(tmp_path):
    summary_path = tmp_path / "reports" / "summary.json"
    scored_path = tmp_path / "data" / "scored_rows.csv"
    _write_summary(summary_path, {"segments": [{"segment_id": "3"}]})
    _write_scored_rows(
        scored_path,
        rows=[{"segment_id": "1", "value": "9"}, {"segment_id": "2", "value": "8"}],
        fieldnames=["segment_id", "value"],
    )

    assert enrich_segmented_summary_json(
        str(summary_path),
        str(scored_path),
        segment_label_column=None,
    )

    updated = _load_summary(summary_path)
    assert updated.get("segment_metadata_unavailable") is True
    caveats = updated.get("caveats", [])
    assert any("do not match" in str(c) for c in caveats)
    assert "segment_size" not in updated["segments"][0]


def test_never_uses_sector_as_fallback(tmp_path):
    summary_path = tmp_path / "reports" / "summary.json"
    scored_path = tmp_path / "data" / "scored_rows.csv"
    _write_summary(summary_path, {"segments": [{"segment_id": "Construccion"}]})
    _write_scored_rows(
        scored_path,
        rows=[{"Sector": "Construccion", "value": "1"}],
        fieldnames=["Sector", "value"],
    )

    assert enrich_segmented_summary_json(
        str(summary_path),
        str(scored_path),
        segment_label_column=None,
    )

    updated = _load_summary(summary_path)
    assert updated.get("segment_metadata_unavailable") is True
    assert "segment_size" not in updated["segments"][0]
