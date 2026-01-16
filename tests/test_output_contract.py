import os
from pathlib import Path

from src.utils.output_contract import check_required_outputs, check_scored_rows_schema, check_artifact_requirements


def test_output_contract_present(tmp_path: Path):
    file_path = tmp_path / "data" / "cleaned_data.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("x\n1", encoding="utf-8")
    report = check_required_outputs([str(file_path)])
    assert report["missing"] == []
    assert str(file_path) in report["present"]


def test_output_contract_glob_missing(tmp_path: Path):
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        report = check_required_outputs(["static/plots/*.png"])
        assert "static/plots/*.png" in report["missing"]
    finally:
        os.chdir(cwd)


def test_output_contract_optional_missing(tmp_path: Path):
    file_path = tmp_path / "data" / "cleaned_data.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("x\n1", encoding="utf-8")
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        report = check_required_outputs(
            [
                {"path": str(file_path), "required": True},
                {"path": "static/plots/*.png", "required": False},
            ]
        )
        assert report["missing"] == []
        assert "static/plots/*.png" in report.get("missing_optional", [])
    finally:
        os.chdir(cwd)


# P1.6 Tests for any-of groups in scored_rows_schema


def test_scored_rows_any_of_groups_pass_with_synonyms(tmp_path: Path):
    """
    P1.6: any-of groups should match when any synonym is present.
    """
    # Create scored_rows.csv with synonym columns
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scored_rows_path = data_dir / "scored_rows.csv"
    scored_rows_path.write_text("row_id,predicted_prob\n1,0.85\n2,0.45\n", encoding="utf-8")

    # Define any-of groups with synonyms
    required_any_of_groups = [
        ["id", "row_id", "index", "record_id", "case_id"],
        ["prediction", "probability", "predicted_prob"],
    ]

    report = check_scored_rows_schema(
        str(scored_rows_path),
        required_columns=[],
        required_any_of_groups=required_any_of_groups,
    )

    assert report["exists"] is True
    assert report["missing_any_of_groups"] == []
    assert len(report["matched_any_of_groups"]) == 2

    # Check that matched groups have correct columns
    group0_match = report["matched_any_of_groups"][0]
    assert "row_id" in group0_match["matched"]

    group1_match = report["matched_any_of_groups"][1]
    assert "predicted_prob" in group1_match["matched"]


def test_scored_rows_any_of_groups_missing_score_group(tmp_path: Path):
    """
    P1.6: any-of groups should fail when no synonym from a group is present.
    """
    # Create scored_rows.csv with only row_id (missing prediction/score)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scored_rows_path = data_dir / "scored_rows.csv"
    scored_rows_path.write_text("row_id\n1\n2\n", encoding="utf-8")

    # Define any-of groups
    required_any_of_groups = [
        ["id", "row_id", "index"],
        ["prediction", "probability", "score", "risk_score"],
    ]

    report = check_scored_rows_schema(
        str(scored_rows_path),
        required_columns=[],
        required_any_of_groups=required_any_of_groups,
    )

    assert report["exists"] is True
    assert len(report["missing_any_of_groups"]) == 1
    # The second group should be missing
    assert report["missing_any_of_groups"][0] == ["prediction", "probability", "score", "risk_score"]

    # First group should match
    assert len(report["matched_any_of_groups"]) == 1
    assert "row_id" in report["matched_any_of_groups"][0]["matched"]


def test_scored_rows_any_of_groups_case_insensitive(tmp_path: Path):
    """
    P1.6: any-of groups should match case-insensitively.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scored_rows_path = data_dir / "scored_rows.csv"
    # Use mixed case in file
    scored_rows_path.write_text("Row_ID,Predicted_Prob\n1,0.85\n", encoding="utf-8")

    required_any_of_groups = [
        ["id", "row_id", "index"],
        ["prediction", "probability", "predicted_prob"],
    ]

    report = check_scored_rows_schema(
        str(scored_rows_path),
        required_columns=[],
        required_any_of_groups=required_any_of_groups,
    )

    assert report["missing_any_of_groups"] == []
    assert len(report["matched_any_of_groups"]) == 2


def test_scored_rows_any_of_groups_backward_compatible(tmp_path: Path):
    """
    P1.6: Should work when required_any_of_groups is not provided (backward compatibility).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scored_rows_path = data_dir / "scored_rows.csv"
    scored_rows_path.write_text("id,prediction\n1,0\n2,1\n", encoding="utf-8")

    # Call without required_any_of_groups
    report = check_scored_rows_schema(
        str(scored_rows_path),
        required_columns=["id", "prediction"],
        required_any_of_groups=None,
    )

    assert report["exists"] is True
    assert report["present_columns"] == ["id", "prediction"]
    assert report["missing_columns"] == []
    # New fields should be empty/backward compatible
    assert report["missing_any_of_groups"] == []
    assert report["matched_any_of_groups"] == []


def test_artifact_requirements_with_any_of_groups(tmp_path: Path):
    """
    P1.6: check_artifact_requirements should use any-of groups and return warning if missing.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create scored_rows.csv with only row_id (missing prediction)
    scored_rows_path = data_dir / "scored_rows.csv"
    scored_rows_path.write_text("row_id\n1\n2\n", encoding="utf-8")

    # Create a cleaned_data.csv to satisfy required_files
    cleaned_path = data_dir / "cleaned_data.csv"
    cleaned_path.write_text("x\n1\n", encoding="utf-8")

    artifact_requirements = {
        "required_files": [
            {"path": "data/scored_rows.csv", "description": ""},
            {"path": "data/cleaned_data.csv", "description": ""},
        ],
        "scored_rows_schema": {
            "required_columns": [],
            "required_any_of_groups": [
                ["id", "row_id"],
                ["prediction", "probability", "score"],
            ],
            "required_any_of_group_severity": ["warning", "fail"],
        },
    }

    report = check_artifact_requirements(artifact_requirements, work_dir=str(tmp_path))

    # Files should be present
    assert report["files_report"]["missing"] == []

    # Should be ERROR because score group is missing with severity="fail"
    assert report["status"] == "error"
    assert len(report["scored_rows_report"]["missing_any_of_groups"]) == 1

    # Check missing groups have severity
    missing_with_severity = report["scored_rows_report"]["missing_any_of_groups_with_severity"]
    assert len(missing_with_severity) == 1
    assert missing_with_severity[0]["severity"] == "fail"


def test_artifact_requirements_missing_identifier_warning(tmp_path: Path):
    """
    P1.6.1: Missing identifier group with severity="warning" should return warning.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create scored_rows.csv with only prediction (missing identifier)
    scored_rows_path = data_dir / "scored_rows.csv"
    scored_rows_path.write_text("prediction\n0.85\n0.45\n", encoding="utf-8")

    # Create a cleaned_data.csv
    cleaned_path = data_dir / "cleaned_data.csv"
    cleaned_path.write_text("x\n1\n", encoding="utf-8")

    artifact_requirements = {
        "required_files": [
            {"path": "data/scored_rows.csv", "description": ""},
            {"path": "data/cleaned_data.csv", "description": ""},
        ],
        "scored_rows_schema": {
            "required_columns": [],
            "required_any_of_groups": [
                ["id", "row_id"],
                ["prediction", "probability", "score"],
            ],
            "required_any_of_group_severity": ["warning", "fail"],
        },
    }

    report = check_artifact_requirements(artifact_requirements, work_dir=str(tmp_path))

    # Files should be present
    assert report["files_report"]["missing"] == []

    # Should be WARNING because only identifier group is missing (severity="warning")
    assert report["status"] == "warning"
    assert len(report["scored_rows_report"]["missing_any_of_groups"]) == 1

    # Check missing group has severity="warning"
    missing_with_severity = report["scored_rows_report"]["missing_any_of_groups_with_severity"]
    assert len(missing_with_severity) == 1
    assert missing_with_severity[0]["severity"] == "warning"


def test_artifact_requirements_ok_with_any_of_groups(tmp_path: Path):
    """
    P1.6: check_artifact_requirements should return ok when all any-of groups match.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create scored_rows.csv with required synonyms
    scored_rows_path = data_dir / "scored_rows.csv"
    scored_rows_path.write_text("row_id,probability\n1,0.85\n2,0.45\n", encoding="utf-8")

    # Create cleaned_data.csv
    cleaned_path = data_dir / "cleaned_data.csv"
    cleaned_path.write_text("x\n1\n", encoding="utf-8")

    artifact_requirements = {
        "required_files": [
            {"path": "data/scored_rows.csv", "description": ""},
            {"path": "data/cleaned_data.csv", "description": ""},
        ],
        "scored_rows_schema": {
            "required_columns": [],
            "required_any_of_groups": [
                ["id", "row_id"],
                ["prediction", "probability", "predicted_prob", "score"],
            ],
            "required_any_of_group_severity": ["warning", "fail"],
        },
    }

    report = check_artifact_requirements(artifact_requirements, work_dir=str(tmp_path))

    # Should be ok
    assert report["status"] == "ok"
    assert report["scored_rows_report"]["missing_any_of_groups"] == []
    assert report["scored_rows_report"]["missing_any_of_groups_with_severity"] == []
