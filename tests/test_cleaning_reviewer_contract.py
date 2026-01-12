import json

from src.agents.cleaning_reviewer import CleaningReviewerAgent


def test_cleaning_reviewer_has_review_cleaning():
    agent = CleaningReviewerAgent()
    assert callable(getattr(agent, "review_cleaning", None))


def test_cleaning_reviewer_id_integrity_scientific_notation(tmp_path):
    cleaned_path = tmp_path / "cleaned.csv"
    cleaned_path.write_text("id,value\n1.234e+15,10\n2.345e+14,20\n", encoding="utf-8")
    manifest_path = tmp_path / "cleaning_manifest.json"
    manifest_path.write_text(json.dumps({"row_counts": {"initial": 2, "final": 2}}), encoding="utf-8")

    cleaning_view = {
        "cleaning_gates": [
            {
                "name": "id_integrity",
                "severity": "HARD",
                "params": {
                    "identifier_name_regex": "(?i)^id$",
                    "detect_scientific_notation": True,
                    "min_samples": 1,
                },
            }
        ],
        "required_columns": [],
        "dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
    }

    agent = CleaningReviewerAgent()
    result = agent.review_cleaning(
        cleaning_view,
        cleaned_csv_path=str(cleaned_path),
        cleaning_manifest_path=str(manifest_path),
    )

    assert result["status"] == "REJECTED"
    assert "id_integrity" in result.get("failed_checks", [])
