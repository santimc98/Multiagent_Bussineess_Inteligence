"""
Sandbox resilience utilities.

Provides retry logic and robustness for transient failures in sandbox operations.
"""

import os
import re
import time
import inspect
import shlex
from typing import Optional, Any, Callable


# Sandbox timeout constants (P2.2)
DE_TIMEOUT_S = 600  # 10 minutes for Data Engineer
ML_TIMEOUT_SMALL_S = 600  # 10 minutes
ML_TIMEOUT_MEDIUM_S = 900  # 15 minutes
ML_TIMEOUT_LARGE_S = 1200  # 20 minutes


# Patterns that suggest transient/temporary errors
TRANSIENT_ERROR_PATTERNS = [
    "timeout", "timed out", "temporarily unavailable",
    "connection", "expired", "gone",
    "502", "503", "504",
    "connection reset", "broken pipe",
]

_BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".parquet",
    ".pkl",
    ".pickle",
    ".joblib",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
}

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def is_transient_error_like(msg: str) -> bool:
    """
    Check if an error message suggests a transient/temporary failure.

    Args:
        msg: The error message to check

    Returns:
        True if error appears transient
    """
    error_lower = msg.lower()
    return any(pattern in error_lower for pattern in TRANSIENT_ERROR_PATTERNS)


def _is_transient_error(msg: str) -> bool:
    """
    Check if an error message suggests a transient/temporary failure.

    Args:
        msg: The error message to check

    Returns:
        True if error appears transient
    """
    error_lower = msg.lower()
    return any(pattern in error_lower for pattern in TRANSIENT_ERROR_PATTERNS)


def _is_transient_sandbox_error(err: Exception) -> bool:
    """
    Check if an exception is a transient sandbox error.

    Args:
        err: The exception to check

    Returns:
        True if exception appears to be a transient sandbox issue
    """
    err_msg = str(err).lower()

    # Timeout patterns
    if any(p in err_msg for p in ["timeout", "timed out", "deadline", "expired"]):
        return True

    # Network/transport patterns
    if any(p in err_msg for p in ["connection", "transport", "unavailable", "refused", "reset", "broken pipe", "socket", "network"]):
        return True

    # 50x HTTP errors
    if any(p in err_msg for p in ["502", "503", "504", "502 bad gateway", "503 service unavailable", "504 gateway timeout"]):
        return True

    # Sandbox-specific patterns (expired, gone, terminated)
    if any(p in err_msg for p in ["sandbox expired", "sandbox terminated", "sandbox gone", "container", "pod"]):
        return True

    # Rate limits / temporarily unavailable
    if any(p in err_msg for p in ["rate limit", "temporarily unavailable", "too many requests", "quota exceeded"]):
        return True

    return False


def is_transient_sandbox_error(err: Exception) -> bool:
    """
    Public wrapper for checking if an exception is a transient sandbox error.

    Args:
        err: The exception to check

    Returns:
        True if exception appears to be a transient sandbox issue
    """
    return _is_transient_sandbox_error(err)


class _ExecutionLogs:
    def __init__(self, stdout: Optional[str], stderr: Optional[str]) -> None:
        self.stdout = stdout.splitlines() if stdout else []
        self.stderr = stderr.splitlines() if stderr else []


class _ExecutionError:
    def __init__(self, name: str, value: str, traceback: str) -> None:
        self.name = name
        self.value = value
        self.traceback = traceback


class _ExecutionResult:
    def __init__(self, stdout: Optional[str], stderr: Optional[str], exit_code: Optional[int]) -> None:
        self.logs = _ExecutionLogs(stdout, stderr)
        self.exit_code = exit_code
        if exit_code is not None and exit_code != 0:
            error_text = stderr or stdout or f"Process exited with code {exit_code}"
            self.error = _ExecutionError("RuntimeError", error_text, error_text)
        else:
            self.error = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_code_via_commands_run(
    sandbox: Any,
    code: str,
    timeout_s: Optional[int] = None,
    workdir: Optional[str] = None,
) -> Any:
    import posixpath

    run_dir = workdir or "/tmp"
    run_dir = run_dir.rstrip("/") or "/tmp"
    script_path = posixpath.join(run_dir, "run_code.py")

    try:
        sandbox.commands.run(f"mkdir -p {shlex.quote(run_dir)}")
    except Exception:
        pass

    sandbox.files.write(script_path, code)

    if workdir:
        cmd = f"sh -c 'cd {shlex.quote(run_dir)} && python {shlex.quote(script_path)}'"
    else:
        cmd = f"python {shlex.quote(script_path)}"

    supports_timeout = False
    try:
        sig = inspect.signature(sandbox.commands.run)
        supports_timeout = "timeout" in sig.parameters
    except (ValueError, TypeError):
        supports_timeout = False

    if supports_timeout and timeout_s is not None:
        proc = sandbox.commands.run(cmd, timeout=timeout_s)
    else:
        proc = sandbox.commands.run(cmd)

    stdout = _normalize_text(getattr(proc, "stdout", ""))
    stderr = _normalize_text(getattr(proc, "stderr", ""))
    exit_code = getattr(proc, "exit_code", None)
    return _ExecutionResult(stdout, stderr, exit_code)


def run_code_with_optional_timeout(sandbox: Any, code: str, timeout_s: Optional[int] = None) -> Any:
    """
    Run code in sandbox with optional timeout.

    Uses inspect to check if sandbox.run_code accepts timeout parameter.

    Args:
        sandbox: The sandbox instance
        code: The code to run
        timeout_s: Optional timeout in seconds

    Returns:
        Result from sandbox.run_code
    """
    run_code = getattr(sandbox, "run_code", None)
    if callable(run_code):
        sig = inspect.signature(run_code)
        if "timeout" in sig.parameters and timeout_s is not None:
            return run_code(code, timeout=timeout_s)
        return run_code(code)

    return _run_code_via_commands_run(sandbox, code, timeout_s=timeout_s)


def run_cmd_with_retry(sandbox: Any, cmd: str, retries: int = 2, timeout_s: Optional[int] = None) -> Any:
    """
    Run a sandbox command with retry logic for transient errors.

    Retries if error appears to be transient. Supports optional timeout
    if sandbox.commands.run accepts a timeout parameter.

    Args:
        sandbox: The sandbox instance
        cmd: The command to run
        retries: Number of retries (default: 2)
        timeout_s: Optional timeout in seconds (passed if sandbox supports it)

    Returns:
        Result from sandbox.commands.run

    Raises:
        Last exception if all retries fail
    """
    last_error = None

    # Check if sandbox.commands.run supports timeout parameter
    supports_timeout = False
    try:
        sig = inspect.signature(sandbox.commands.run)
        supports_timeout = "timeout" in sig.parameters
    except (ValueError, TypeError):
        supports_timeout = False

    for attempt in range(retries + 1):
        try:
            if supports_timeout and timeout_s is not None:
                return sandbox.commands.run(cmd, timeout=timeout_s)
            else:
                return sandbox.commands.run(cmd)
        except Exception as e:
            last_error = e

            # Check if error appears transient
            if attempt < retries and is_transient_sandbox_error(e):
                # Exponential backoff
                backoff = 2 ** attempt
                print(f"RETRYING_SANDBOX_CMD (attempt {attempt + 1}/{retries}): {str(e)}")
                time.sleep(backoff)
            else:
                # Last attempt or non-transient error
                raise

    raise last_error


def safe_download_bytes(sandbox: Any, remote_path: str, max_attempts: int = 2) -> Optional[bytes]:
    """
    Download a file from sandbox with base64 fallback.

    Args:
        sandbox: The sandbox instance
        remote_path: Path in sandbox to download
        max_attempts: Maximum number of retry attempts

    Returns:
        Content as bytes, or None if download fails
    """
    import base64 as b64_module

    remote_lower = str(remote_path or "").lower()
    ext = os.path.splitext(remote_lower)[1]
    binary_hint = ext in _BINARY_EXTENSIONS
    is_png = ext == ".png"

    for attempt in range(max_attempts):
        content = None
        try:
            content = sandbox.files.read(remote_path)
        except Exception as e:
            error_msg = str(e)
            print(f"DOWNLOAD_ATTEMPT_{attempt + 1}/{max_attempts} FAILED: {error_msg}")

        data = None
        if content is not None:
            if isinstance(content, (bytes, bytearray)):
                data = bytes(content) if isinstance(content, bytearray) else content
            elif hasattr(content, "tobytes"):
                data = content.tobytes()
            elif isinstance(content, str):
                if not binary_hint:
                    return content.encode("utf-8", errors="surrogateescape")

        if data is not None:
            if is_png and not data.startswith(_PNG_SIGNATURE):
                data = None
            else:
                return data

        # Try base64 fallback (always, not just on exception)
        try:
            # Use shlex.quote and "--" to protect against path injection
            cmd = f"base64 -w 0 -- {shlex.quote(remote_path)}"
            proc = sandbox.commands.run(cmd)
            if proc.exit_code == 0 and proc.stdout:
                decoded = b64_module.b64decode(proc.stdout.strip())
                if is_png and not decoded.startswith(_PNG_SIGNATURE):
                    continue
                return decoded  # Already bytes from b64decode
        except Exception as e2:
            print(f"BASE64_FALLBACK_ATTEMPT_{attempt + 1}/{max_attempts} FAILED: {e2}")

    return None


def safe_download_file(sandbox: Any, remote_path: str, max_attempts: int = 2) -> Optional[str]:
    """
    Download a text file from sandbox with base64 fallback.

    For text files (JSON, CSV, etc.) that need to be returned as string.
    For binary files (PNG, PDF), use safe_download_bytes instead.

    Args:
        sandbox: The sandbox instance
        remote_path: Path in sandbox to download
        max_attempts: Maximum number of retry attempts

    Returns:
        Content as string, or None if download fails
    """
    content = safe_download_bytes(sandbox, remote_path, max_attempts)
    if content is None:
        return None
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return None


def create_sandbox_with_retry(SandboxCls, *, max_attempts: int = 2, run_id: Optional[str] = None, step: Optional[str] = None):
    """
    Wrapper for creating sandbox with retry logic.

    Only retries CREATION of the sandbox. Yields exactly once.
    Supports both context-manager style (SandboxCls.create()) and direct instantiation.

    Args:
        SandboxCls: The Sandbox class (e.g., Sandbox from e2b)
        max_attempts: Maximum number of attempts (default: 2)
        run_id: Optional run ID for logging
        step: Optional step name for logging

    Returns:
        Context manager for sandbox that retries creation on transient errors

    Example:
        with create_sandbox_with_retry(Sandbox, max_attempts=2) as sandbox:
            # Use sandbox normally
            sandbox.files.write(path, content)
    """
    from contextlib import contextmanager

    @contextmanager
    def _sandbox_context():
        sandbox = None
        last_error = None
        uses_context_manager = hasattr(SandboxCls, "create") and callable(getattr(SandboxCls, "create"))
        context = None

        # Retry loop for CREATION only
        for attempt in range(1, max_attempts + 1):
            try:
                if step is not None:
                    print(f"SANDBOX_ATTEMPT step={step} attempt={attempt}/{max_attempts}")

                if uses_context_manager:
                    # Use SandboxCls.create() as context manager
                    context = SandboxCls.create()
                    sandbox = context.__enter__()
                else:
                    # Direct instantiation
                    sandbox = SandboxCls()

                # Creation succeeded, break out of retry loop
                break

            except Exception as e:
                last_error = e
                if sandbox is not None and hasattr(sandbox, "close"):
                    try:
                        sandbox.close()
                    except Exception:
                        pass
                    sandbox = None

                if not is_transient_sandbox_error(e):
                    raise  # Non-transient error, propagate immediately

                if attempt < max_attempts:
                    backoff = 2 ** (attempt - 1)
                    print(f"SANDBOX_TRANSIENT_FAILURE_RETRY step={step} attempt={attempt}/{max_attempts}: {e}")
                    time.sleep(backoff)
                else:
                    # All attempts exhausted - preserve original traceback
                    raise RuntimeError(f"All {max_attempts} sandbox creation attempts failed") from last_error

        # Yield exactly once
        try:
            yield sandbox
        finally:
            # Cleanup
            if uses_context_manager and context is not None:
                try:
                    context.__exit__(None, None, None)
                except Exception:
                    pass
            elif sandbox is not None and hasattr(sandbox, "close"):
                try:
                    sandbox.close()
                except Exception:
                    pass

    return _sandbox_context()
