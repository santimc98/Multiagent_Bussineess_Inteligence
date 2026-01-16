"""
Tests for sandbox_resilience module.

Tests retry logic, timeout support, and download fallbacks.
"""

import pytest
import inspect
from unittest.mock import Mock, MagicMock, patch

from src.utils.sandbox_resilience import (
    is_transient_error_like,
    is_transient_sandbox_error,
    run_code_with_optional_timeout,
    run_cmd_with_retry,
    safe_download_bytes,
    create_sandbox_with_retry,
)


def test_is_transient_error_like_timeout():
    """Test that timeout errors are detected as transient."""
    err_msg = "Script execution timed out after 10.0 seconds"

    assert is_transient_error_like(err_msg) is True


def test_is_transient_error_like_503():
    """Test that 503 errors are detected as transient."""
    err_msg = "503 Service Unavailable"

    assert is_transient_error_like(err_msg) is True


def test_is_transient_error_like_connection():
    """Test that connection errors are detected as transient."""
    err_msg = "Connection reset by peer"

    assert is_transient_error_like(err_msg) is True


def test_is_transient_error_like_sandbox_expired():
    """Test that sandbox expired errors are detected as transient."""
    # Note: is_transient_sandbox_error takes an Exception, not a string
    err = Exception("sandbox expired: container was terminated")

    assert is_transient_sandbox_error(err) is True


def test_is_transient_sandbox_error_non_transient():
    """Test that script syntax errors are NOT transient."""
    err = Exception("SyntaxError: invalid syntax")

    assert is_transient_sandbox_error(err) is False
    assert is_transient_error_like("SyntaxError: invalid syntax") is False


def test_run_code_with_optional_timeout():
    """Test run_code with timeout when sandbox supports it."""
    mock_sandbox = Mock()

    # Create a callable with proper signature that inspect can read
    call_tracker = []
    def run_code_impl(code, timeout=None):
        call_tracker.append({"code": code, "timeout": timeout})
        return MagicMock()

    # Replace run_code with our implementation (inspect.signature will work)
    mock_sandbox.run_code = run_code_impl

    result = run_code_with_optional_timeout(mock_sandbox, "print('hello')", timeout_s=60)
    assert len(call_tracker) == 1
    assert call_tracker[0]["code"] == "print('hello')"
    assert call_tracker[0]["timeout"] == 60


def test_run_code_with_optional_timeout_no_timeout_param():
    """Test run_code without timeout when sandbox doesn't support it."""
    mock_sandbox = Mock()

    # Create a callable WITHOUT timeout parameter
    call_tracker = []
    def run_code_impl(code):
        call_tracker.append({"code": code})
        return MagicMock()

    mock_sandbox.run_code = run_code_impl

    result = run_code_with_optional_timeout(mock_sandbox, "print('hello')", timeout_s=60)
    assert len(call_tracker) == 1
    assert call_tracker[0]["code"] == "print('hello')"
    # Should NOT have timeout key since function doesn't support it


@patch('src.utils.sandbox_resilience.time.sleep', return_value=None)
def test_run_cmd_with_retry_transient_success(mock_sleep):
    """Test that run_cmd_with_retry succeeds on second attempt after transient error."""
    mock_sandbox = Mock()
    cmd = "ls -la"

    # Create result object
    success_result = Mock()
    success_result.stdout = "file1.txt\nfile2.txt\n"
    success_result.exit_code = 0

    # First call raises transient error, second succeeds
    mock_sandbox.commands.run = Mock(
        side_effect=[Exception("503 Service Unavailable"), success_result]
    )

    result = run_cmd_with_retry(mock_sandbox, cmd, retries=2)

    # Should have called twice and succeeded
    assert mock_sandbox.commands.run.call_count == 2
    assert result.exit_code == 0


@patch('src.utils.sandbox_resilience.time.sleep', return_value=None)
def test_run_cmd_with_retry_non_transient_failure(mock_sleep):
    """Test that run_cmd_with_retry raises immediately on non-transient error."""
    mock_sandbox = Mock()
    cmd = "ls -la"

    # Always fails with syntax error (non-transient)
    mock_sandbox.commands.run = Mock(side_effect=Exception("SyntaxError: invalid syntax"))

    with pytest.raises(Exception) as exc_info:
        run_cmd_with_retry(mock_sandbox, cmd, retries=2)

    # Should have called only once (no retry for non-transient)
    assert mock_sandbox.commands.run.call_count == 1
    assert "SyntaxError" in str(exc_info.value)


def test_run_cmd_with_retry_with_timeout():
    """Test that run_cmd_with_retry passes timeout when supported."""
    mock_sandbox = Mock()
    cmd = "sleep 10"

    # Create a callable with proper signature that inspect can read
    call_tracker = []
    def commands_run_impl(cmd, timeout=None):
        call_tracker.append({"cmd": cmd, "timeout": timeout})
        result = Mock()
        result.stdout = "done"
        result.exit_code = 0
        return result

    mock_sandbox.commands.run = commands_run_impl

    result = run_cmd_with_retry(mock_sandbox, cmd, retries=1, timeout_s=30)

    assert len(call_tracker) == 1
    assert call_tracker[0]["cmd"] == cmd
    assert call_tracker[0]["timeout"] == 30


def test_safe_download_bytes_success():
    """Test successful file download with bytes return."""
    mock_sandbox = Mock()
    remote_path = "/path/to/file.txt"

    mock_sandbox.files.read = Mock(return_value=b"file content")
    result = safe_download_bytes(mock_sandbox, remote_path, max_attempts=2)

    assert result == b"file content"
    assert isinstance(result, bytes)


def test_safe_download_bytes_string_content():
    """Test download when sandbox returns string (auto-converted to bytes)."""
    mock_sandbox = Mock()
    remote_path = "/path/to/file.txt"

    mock_sandbox.files.read = Mock(return_value="string content")
    result = safe_download_bytes(mock_sandbox, remote_path, max_attempts=2)

    assert result == b"string content"
    assert isinstance(result, bytes)


def test_safe_download_bytes_with_base64_fallback():
    """Test download with base64 fallback when files.read fails."""
    import base64
    mock_sandbox = Mock()
    remote_path = "/path/to/file.txt"

    # files.read always fails
    mock_sandbox.files.read = Mock(side_effect=Exception("files.read failed"))

    # base64 fallback succeeds
    cmd_result = Mock()
    cmd_result.exit_code = 0
    cmd_result.stdout = base64.b64encode(b"fallback content").decode("ascii")

    mock_sandbox.commands.run = Mock(return_value=cmd_result)

    result = safe_download_bytes(mock_sandbox, remote_path, max_attempts=2)

    assert result == b"fallback content"
    assert isinstance(result, bytes)
    # Verify shlex.quote was used in the command
    call_args = mock_sandbox.commands.run.call_args[0][0]
    assert "--" in call_args  # Should have -- for safety


def test_safe_download_bytes_all_fail():
    """Test download returns None when all attempts fail."""
    mock_sandbox = Mock()
    remote_path = "/path/to/file.txt"

    # files.read fails
    mock_sandbox.files.read = Mock(side_effect=Exception("files.read failed"))

    # base64 fallback also fails
    cmd_result = Mock()
    cmd_result.exit_code = 1
    cmd_result.stdout = ""
    mock_sandbox.commands.run = Mock(return_value=cmd_result)

    result = safe_download_bytes(mock_sandbox, remote_path, max_attempts=2)

    assert result is None


@patch('src.utils.sandbox_resilience.time.sleep', return_value=None)
def test_create_sandbox_with_retry_success(mock_sleep):
    """Test create_sandbox_with_retry yields exactly once on success."""
    mock_sandbox = Mock()
    mock_sandbox.close = Mock()

    # Use spec=[] to prevent Mock from having .create attribute
    mock_sandbox_cls = Mock(spec=[], return_value=mock_sandbox)

    yield_count = 0
    with create_sandbox_with_retry(mock_sandbox_cls, max_attempts=2) as sandbox:
        yield_count += 1
        assert sandbox is mock_sandbox

    assert yield_count == 1  # Exactly one yield


@patch('src.utils.sandbox_resilience.time.sleep', return_value=None)
def test_create_sandbox_with_retry_transient_then_success(mock_sleep):
    """Test create_sandbox_with_retry retries on transient error then succeeds."""
    mock_sandbox = Mock()
    mock_sandbox.close = Mock()

    # Use a callable class mock that doesn't have .create
    call_count = [0]
    def mock_cls():
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("503 Service Unavailable")
        return mock_sandbox

    yield_count = 0
    with create_sandbox_with_retry(mock_cls, max_attempts=2, step="test") as sandbox:
        yield_count += 1
        assert sandbox is mock_sandbox

    assert yield_count == 1  # Still exactly one yield
    assert call_count[0] == 2


@patch('src.utils.sandbox_resilience.time.sleep', return_value=None)
def test_create_sandbox_with_retry_non_transient_failure(mock_sleep):
    """Test create_sandbox_with_retry raises immediately on non-transient error."""
    call_count = [0]
    def mock_cls():
        call_count[0] += 1
        raise Exception("SyntaxError: invalid")

    with pytest.raises(Exception) as exc_info:
        with create_sandbox_with_retry(mock_cls, max_attempts=2) as sandbox:
            pass

    assert "SyntaxError" in str(exc_info.value)
    assert call_count[0] == 1  # No retry


@patch('src.utils.sandbox_resilience.time.sleep', return_value=None)
def test_create_sandbox_with_retry_all_attempts_fail(mock_sleep):
    """Test create_sandbox_with_retry raises after exhausting all attempts."""
    call_count = [0]
    def mock_cls():
        call_count[0] += 1
        raise Exception("503 Service Unavailable")

    with pytest.raises(Exception) as exc_info:
        with create_sandbox_with_retry(mock_cls, max_attempts=2) as sandbox:
            pass

    assert "All 2 sandbox creation attempts failed" in str(exc_info.value)
    assert call_count[0] == 2


def test_create_sandbox_with_retry_cleanup_on_success():
    """Test that sandbox.close() is called after successful use."""
    mock_sandbox = Mock()
    mock_sandbox.close = Mock()

    # Use spec=[] to prevent Mock from having .create attribute
    mock_sandbox_cls = Mock(spec=[], return_value=mock_sandbox)

    with create_sandbox_with_retry(mock_sandbox_cls, max_attempts=1) as sandbox:
        pass

    mock_sandbox.close.assert_called_once()
