import csv
import json
import os

import pytest

from src.utils.label_enrichment import enrich_outputs


def _write_scored_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f_obj:
        writer = csv.DictWriter(f_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_header(path):
    with open(path, "r", encoding="utf-8") as f_obj:
        return f_obj.readline().strip().split(",")


def _read_rows(path):
    with open(path, "r", encoding="utf-8") as f_obj:
        reader = csv.DictReader(f_obj)
        return list(reader)


def test_label_enrichment_no_label_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scored_path = os.path.join("report", "data", "scored_rows.csv")
    _write_scored_rows(
        scored_path,
        rows=[{"Sector": "A", "value": "1"}],
        fieldnames=["Sector", "value"],
    )

    meta = enrich_outputs(scored_path, summary_json_path=None, label_col_hint=None)
    assert meta.get("status") == "label_column_unavailable"

    header = _read_header(scored_path)
    assert "label_id" not in header

    meta_path = os.path.join("report", "governance", "enrichment_meta.json")
    assert os.path.exists(meta_path)


def test_label_enrichment_creates_stable_mapping(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scored_path = os.path.join("report", "data", "scored_rows.csv")
    _write_scored_rows(
        scored_path,
        rows=[
            {"cluster": "B", "value": "1"},
            {"cluster": "A", "value": "2"},
            {"cluster": "B", "value": "3"},
        ],
        fieldnames=["cluster", "value"],
    )

    meta = enrich_outputs(scored_path, summary_json_path=None, label_col_hint=None)
    assert meta.get("status") == "ok"
    assert meta.get("mapping_meta", {}).get("created") is True
    assert meta.get("id_col_name") == "cluster_id"

    rows = _read_rows(scored_path)
    ids = [row.get("cluster_id") for row in rows]
    assert ids == ["1", "0", "1"]

    meta_path = os.path.join("report", "governance", "enrichment_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f_obj:
        payload = json.load(f_obj)
    mapping = payload.get("mapping_meta", {}).get("mapping", {})
    assert mapping == {"A": 0, "B": 1}
