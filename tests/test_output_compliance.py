"""
Tests for output_contract.py compliance checks (P1.4).
"""
import os
import pytest
import pandas as pd

from src.utils.output_contract import (
    check_scored_rows_schema,
    check_artifact_requirements,
)


class TestScoredRowsSchemaCheck:
    """Test scored_rows column validation."""

    def test_scored_rows_missing_columns(self, tmp_path):
        """Test detection of missing columns in scored_rows.csv."""
        # Create scored_rows.csv with some columns
        scored_path = tmp_path / "scored_rows.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "prediction": [0.8, 0.6, 0.9],
            # Missing: score, priority
        })
        df.to_csv(scored_path, index=False)

        required_columns = ["id", "prediction", "score", "priority"]
        result = check_scored_rows_schema(str(scored_path), required_columns)

        assert result["exists"] is True
        assert "id" in result["present_columns"]
        assert "prediction" in result["present_columns"]
        assert "score" in result["missing_columns"]
        assert "priority" in result["missing_columns"]

    def test_scored_rows_all_columns_present(self, tmp_path):
        """Test when all required columns are present."""
        scored_path = tmp_path / "scored_rows.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [0.8, 0.6, 0.9],
            "priority": ["high", "low", "high"],
        })
        df.to_csv(scored_path, index=False)

        required_columns = ["id", "score", "priority"]
        result = check_scored_rows_schema(str(scored_path), required_columns)

        assert result["exists"] is True
        assert len(result["missing_columns"]) == 0
        assert set(result["present_columns"]) == set(required_columns)

    def test_scored_rows_file_not_found(self):
        """Test handling of missing file."""
        result = check_scored_rows_schema(
            "/nonexistent/path/scored_rows.csv",
            ["id", "score"]
        )

        assert result["exists"] is False
        assert result["missing_columns"] == ["id", "score"]


class TestArtifactRequirementsCheck:
    """Test full artifact requirements validation."""

    def test_check_artifact_requirements_all_present(self, tmp_path):
        """Test when all required artifacts are present."""
        # Create required files
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        (data_dir / "cleaned_data.csv").write_text("id,col\n1,a\n")
        (data_dir / "metrics.json").write_text('{"accuracy": 0.9}')
        (data_dir / "scored_rows.csv").write_text("id,score\n1,0.9\n")

        artifact_requirements = {
            "required_files": [
                {"path": "data/cleaned_data.csv"},
                {"path": "data/metrics.json"},
                {"path": "data/scored_rows.csv"},
            ],
            "scored_rows_schema": {
                "required_columns": ["id", "score"],
            },
        }

        result = check_artifact_requirements(artifact_requirements, str(tmp_path))

        assert result["status"] == "ok"
        assert len(result["files_report"]["missing"]) == 0
        assert len(result["scored_rows_report"]["missing_columns"]) == 0

    def test_check_artifact_requirements_missing_file(self, tmp_path):
        """Test when required file is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        (data_dir / "cleaned_data.csv").write_text("id,col\n1,a\n")
        # metrics.json is missing

        artifact_requirements = {
            "required_files": [
                {"path": "data/cleaned_data.csv"},
                {"path": "data/metrics.json"},
            ],
        }

        result = check_artifact_requirements(artifact_requirements, str(tmp_path))

        assert result["status"] == "error"
        missing_files = result["files_report"]["missing"]
        assert any("metrics.json" in f for f in missing_files)

    def test_check_artifact_requirements_missing_columns(self, tmp_path):
        """Test when scored_rows is missing required columns."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        (data_dir / "cleaned_data.csv").write_text("id,col\n1,a\n")
        (data_dir / "metrics.json").write_text('{"accuracy": 0.9}')
        (data_dir / "scored_rows.csv").write_text("id,prediction\n1,0.9\n")
        # scored_rows.csv has id, prediction but missing score, priority

        artifact_requirements = {
            "required_files": [
                {"path": "data/cleaned_data.csv"},
                {"path": "data/metrics.json"},
                {"path": "data/scored_rows.csv"},
            ],
            "scored_rows_schema": {
                "required_columns": ["id", "score", "priority"],
            },
        }

        result = check_artifact_requirements(artifact_requirements, str(tmp_path))

        # P1.6.1: Status is ERROR for missing required_columns
        assert result["status"] == "error"
        missing_cols = result["scored_rows_report"]["missing_columns"]
        assert "score" in missing_cols
        assert "priority" in missing_cols
