from src.utils import run_bundle


def test_log_sandbox_attempt_records_failure(tmp_path) -> None:
    run_id = "test-run"
    run_bundle.init_run_bundle(run_id, base_dir=str(tmp_path), enable_tee=False)

    run_bundle.log_sandbox_attempt(
        run_id,
        "ml_engineer",
        1,
        code="print('hi')",
        stdout="",
        stderr="boom",
        outputs_listing=[],
        success=False,
        stage="exception",
        exception_type="ValueError",
        exception_msg="boom",
    )

    record = run_bundle._RUN_ATTEMPTS[run_id][-1]
    assert record.get("success") is False
    assert record.get("stage") == "exception"
    assert record.get("exception_type") == "ValueError"
