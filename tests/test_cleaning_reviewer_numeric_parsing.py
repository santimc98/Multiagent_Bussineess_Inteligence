import json
import os

import pytest

from src.agents.cleaning_reviewer import CleaningReviewerAgent


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    old_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


def test_numeric_parsing_validation_passes_with_decimal_comma(tmp_workdir):
    cleaned_path = tmp_workdir / "cleaned.csv"
    cleaned_path.write_text("amount;other\n12727,0;1\n42,5;2\n", encoding="utf-8")
    manifest_path = tmp_workdir / "cleaning_manifest.json"
    manifest_path.write_text(
        json.dumps({"output_dialect": {"sep": ";", "decimal": ",", "encoding": "utf-8"}}),
        encoding="utf-8",
    )

    cleaning_view = {
        "cleaning_gates": [
            {
                "name": "Numeric Parsing Validation",
                "severity": "HARD",
                "params": {"columns": ["amount"], "check": "no_string_remainders"},
            }
        ],
        "required_columns": ["amount", "other"],
        "dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
    }

    agent = CleaningReviewerAgent()
    result = agent.review_cleaning(
        cleaning_view,
        cleaned_csv_path=str(cleaned_path),
        cleaning_manifest_path=str(manifest_path),
    )

    assert result["status"] != "REJECTED"
    assert "numeric_parsing_validation" not in result.get("failed_checks", [])


def test_id_integrity_skips_when_no_id_columns(tmp_workdir):
    cleaned_path = tmp_workdir / "cleaned.csv"
    cleaned_path.write_text("alpha;beta\n1;2\n3;4\n", encoding="utf-8")
    manifest_path = tmp_workdir / "cleaning_manifest.json"
    manifest_path.write_text(
        json.dumps({"output_dialect": {"sep": ";", "decimal": ",", "encoding": "utf-8"}}),
        encoding="utf-8",
    )

    cleaning_view = {
        "cleaning_gates": [{"name": "id_integrity", "severity": "HARD", "params": {}}],
        "required_columns": [],
        "dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
    }

    agent = CleaningReviewerAgent()
    result = agent.review_cleaning(
        cleaning_view,
        cleaned_csv_path=str(cleaned_path),
        cleaning_manifest_path=str(manifest_path),
    )

    assert result["status"] != "REJECTED"
    assert "id_integrity" not in result.get("failed_checks", [])
