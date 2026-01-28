"""
Tests for cloudrun_launcher required artifacts validation.

Ensures that:
1. When required artifacts are missing, status is "error"
2. Missing artifacts are listed in the response
3. GCS listing is included for diagnostics when artifacts are missing
"""
import pytest
from unittest.mock import patch, MagicMock


def test_missing_required_artifacts_marks_error():
    """When required artifacts are not downloaded, status should be 'error'."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    # Mock all external dependencies
    with patch("src.utils.cloudrun_launcher._ensure_cli") as mock_cli, \
         patch("src.utils.cloudrun_launcher._gsutil_cp") as mock_cp, \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls") as mock_ls, \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        # Setup mocks
        mock_cli.return_value = None
        mock_cp.return_value = None
        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True

        # No files exist in GCS output
        mock_exists.return_value = False
        mock_ls.return_value = []

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
                "scored_rows.csv": "data/scored_rows.csv",
            },
            required_artifacts=["metrics.json", "scored_rows.csv"],
        )

        assert result["status"] == "error"
        assert "missing_artifacts" in result
        assert set(result["missing_artifacts"]) == {"metrics.json", "scored_rows.csv"}
        assert result["error"]["error"] == "missing_required_artifacts"


def test_partial_artifacts_marks_error():
    """When only some required artifacts are downloaded, status should be 'error'."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    with patch("src.utils.cloudrun_launcher._ensure_cli"), \
         patch("src.utils.cloudrun_launcher._gsutil_cp"), \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls") as mock_ls, \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True

        # Only metrics.json exists
        def exists_side_effect(uri, _):
            return "metrics.json" in uri
        mock_exists.side_effect = exists_side_effect
        mock_ls.return_value = ["gs://bucket/outputs/test_run/metrics.json"]

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
                "scored_rows.csv": "data/scored_rows.csv",
                "alignment_check.json": "data/alignment_check.json",
            },
            required_artifacts=["metrics.json", "scored_rows.csv", "alignment_check.json"],
        )

        assert result["status"] == "error"
        assert "scored_rows.csv" in result["missing_artifacts"]
        assert "alignment_check.json" in result["missing_artifacts"]
        assert "metrics.json" not in result["missing_artifacts"]


def test_all_artifacts_present_marks_success():
    """When all required artifacts are downloaded, status should be 'success'."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    with patch("src.utils.cloudrun_launcher._ensure_cli"), \
         patch("src.utils.cloudrun_launcher._gsutil_cp"), \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls"), \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True

        # All required files exist (except error.json)
        def exists_side_effect(uri, _):
            return "error.json" not in uri
        mock_exists.side_effect = exists_side_effect

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
                "scored_rows.csv": "data/scored_rows.csv",
            },
            required_artifacts=["metrics.json", "scored_rows.csv"],
        )

        assert result["status"] == "success"
        assert result["missing_artifacts"] == []


def test_no_required_artifacts_legacy_behavior():
    """When required_artifacts is None, legacy behavior (no artifact check) applies."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    with patch("src.utils.cloudrun_launcher._ensure_cli"), \
         patch("src.utils.cloudrun_launcher._gsutil_cp"), \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls"), \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True

        # No files exist but no required_artifacts specified
        mock_exists.return_value = False

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
            },
            # required_artifacts not specified (None)
        )

        # Without required_artifacts, no artifact check happens
        assert result["status"] == "success"
        assert result["missing_artifacts"] == []
