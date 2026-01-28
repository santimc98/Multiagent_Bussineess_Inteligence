import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple


class CloudRunLaunchError(RuntimeError):
    pass


def _resolve_cli_override(binary: str, env_var: str) -> str:
    override = os.getenv(env_var)
    if not override:
        return binary
    override = override.strip().strip('"')
    if os.path.isdir(override):
        candidates = [os.path.join(override, binary)]
        if os.name == "nt":
            candidates.insert(0, os.path.join(override, f"{binary}.cmd"))
            candidates.insert(1, os.path.join(override, f"{binary}.exe"))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        os.environ["PATH"] = override + os.pathsep + os.environ.get("PATH", "")
        return binary
    if os.path.exists(override):
        return override
    return override


def _ensure_cli(binary: str) -> None:
    has_sep = os.path.sep in binary or (os.path.altsep and os.path.altsep in binary)
    if os.path.isabs(binary) or has_sep:
        if not os.path.exists(binary):
            raise CloudRunLaunchError(f"Required CLI not found: {binary}")
        return
    if not shutil.which(binary):
        raise CloudRunLaunchError(f"Required CLI not found: {binary}")


def _run_cmd(args: list[str], timeout_s: Optional[int] = None) -> Tuple[str, str]:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise CloudRunLaunchError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    return proc.stdout, proc.stderr


def _run_gcloud_job_execute(
    *,
    gcloud_bin: str,
    job: str,
    region: str,
    env_vars: str,
    project: Optional[str],
    wait: bool,
) -> Tuple[str, str, str]:
    cmd = [gcloud_bin, "run", "jobs", "execute", job, "--region", region, "--update-env-vars", env_vars]
    if project:
        cmd.extend(["--project", project])
    if wait:
        cmd.append("--wait")
    try:
        stdout, stderr = _run_cmd(cmd)
        return stdout, stderr, "update-env-vars"
    except CloudRunLaunchError as exc:
        err_text = str(exc)
        if "unrecognized arguments" not in err_text or "--update-env-vars" not in err_text:
            raise
    cmd = [gcloud_bin, "run", "jobs", "execute", job, "--region", region, "--set-env-vars", env_vars]
    if project:
        cmd.extend(["--project", project])
    if wait:
        cmd.append("--wait")
    stdout, stderr = _run_cmd(cmd)
    return stdout, stderr, "set-env-vars"


def _gsutil_exists(uri: str, gsutil_bin: str) -> bool:
    _ensure_cli(gsutil_bin)
    proc = subprocess.run(
        [gsutil_bin, "-q", "stat", uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _gsutil_cp(src: str, dest: str, gsutil_bin: str) -> None:
    _ensure_cli(gsutil_bin)
    _run_cmd([gsutil_bin, "-q", "cp", src, dest])


def _gsutil_cat(uri: str, gsutil_bin: str) -> str:
    _ensure_cli(gsutil_bin)
    stdout, _ = _run_cmd([gsutil_bin, "cat", uri])
    return stdout


def _gsutil_ls(uri: str, gsutil_bin: str) -> list[str]:
    """List files at a GCS URI. Returns empty list on error or if nothing found."""
    _ensure_cli(gsutil_bin)
    try:
        proc = subprocess.run(
            [gsutil_bin, "ls", uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            return []
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _upload_json_to_gcs(payload: Dict[str, Any], uri: str, gsutil_bin: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=True)
        tmp_path = tmp.name
    try:
        _gsutil_cp(tmp_path, uri, gsutil_bin)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _upload_text_to_gcs(text: str, uri: str, gsutil_bin: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as tmp:
        tmp.write(text or "")
        tmp_path = tmp.name
    try:
        _gsutil_cp(tmp_path, uri, gsutil_bin)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _normalize_prefix(prefix: str) -> str:
    prefix = str(prefix or "").strip().strip("/")
    return prefix


def launch_heavy_runner_job(
    *,
    run_id: str,
    request: Dict[str, Any],
    dataset_path: str,
    bucket: str,
    job: str,
    region: str,
    input_prefix: str = "inputs",
    output_prefix: str = "outputs",
    dataset_prefix: str = "datasets",
    project: Optional[str] = None,
    download_map: Optional[Dict[str, str]] = None,
    wait: bool = True,
    code_text: Optional[str] = None,
    support_files: Optional[list[Dict[str, str]]] = None,
    data_path: str = "data/cleaned_data.csv",
    required_artifacts: Optional[list[str]] = None,
) -> Dict[str, Any]:
    gcloud_bin = _resolve_cli_override("gcloud", "HEAVY_RUNNER_GCLOUD_BIN")
    gsutil_bin = _resolve_cli_override("gsutil", "HEAVY_RUNNER_GSUTIL_BIN")
    _ensure_cli(gcloud_bin)
    _ensure_cli(gsutil_bin)

    input_prefix = _normalize_prefix(input_prefix)
    output_prefix = _normalize_prefix(output_prefix)
    dataset_prefix = _normalize_prefix(dataset_prefix)

    dataset_uri = request.get("dataset_uri")
    if not dataset_uri or not str(dataset_uri).startswith("gs://"):
        if not dataset_path or not os.path.exists(dataset_path):
            raise CloudRunLaunchError("dataset_path missing or does not exist for heavy runner upload")
        dataset_name = os.path.basename(dataset_path)
        dataset_uri = f"gs://{bucket}/{dataset_prefix}/{run_id}/{dataset_name}"
        _gsutil_cp(dataset_path, dataset_uri, gsutil_bin)
    request["dataset_uri"] = dataset_uri

    if code_text is not None:
        code_uri = f"gs://{bucket}/{input_prefix}/{run_id}/ml_script.py"
        _upload_text_to_gcs(code_text, code_uri, gsutil_bin)
        request["code_uri"] = code_uri
        request["data_path"] = data_path

        uploaded_support = []
        if support_files:
            for item in support_files:
                if not isinstance(item, dict):
                    continue
                local_path = item.get("local_path")
                rel_path = item.get("path")
                if not local_path or not rel_path:
                    continue
                if not os.path.exists(local_path):
                    continue
                rel_path = str(rel_path).lstrip("/").replace("\\", "/")
                support_uri = f"gs://{bucket}/{input_prefix}/{run_id}/support/{rel_path}"
                _gsutil_cp(local_path, support_uri, gsutil_bin)
                uploaded_support.append({"uri": support_uri, "path": rel_path})
        if uploaded_support:
            request["support_files"] = uploaded_support

    input_uri = f"gs://{bucket}/{input_prefix}/{run_id}.json"
    output_uri = f"gs://{bucket}/{output_prefix}/{run_id}/"
    request["output_uri"] = output_uri
    _upload_json_to_gcs(request, input_uri, gsutil_bin)

    env_vars = f"INPUT_URI={input_uri},OUTPUT_URI={output_uri}"
    stdout, stderr, flag_used = _run_gcloud_job_execute(
        gcloud_bin=gcloud_bin,
        job=job,
        region=region,
        env_vars=env_vars,
        project=project,
        wait=wait,
    )

    downloaded: Dict[str, str] = {}
    if download_map:
        for filename, local_path in download_map.items():
            if not filename:
                continue
            remote_path = output_uri + filename
            if not _gsutil_exists(remote_path, gsutil_bin):
                continue
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            _gsutil_cp(remote_path, local_path, gsutil_bin)
            downloaded[filename] = local_path

    error_payload: Optional[Dict[str, Any]] = None
    error_uri = output_uri + "error.json"
    if _gsutil_exists(error_uri, gsutil_bin):
        try:
            error_payload = json.loads(_gsutil_cat(error_uri, gsutil_bin))
        except Exception:
            error_payload = {"error": "Failed to parse error.json", "raw": _gsutil_cat(error_uri, gsutil_bin)}

    # Check for missing required artifacts
    missing_artifacts: list[str] = []
    if required_artifacts:
        for artifact in required_artifacts:
            if artifact not in downloaded:
                missing_artifacts.append(artifact)

    # Build diagnostic info if artifacts are missing
    gcs_listing: list[str] = []
    if missing_artifacts and not error_payload:
        gcs_listing = _gsutil_ls(output_uri, gsutil_bin)
        error_payload = {
            "error": "missing_required_artifacts",
            "missing": missing_artifacts,
            "downloaded": list(downloaded.keys()),
            "gcs_listing": gcs_listing,
            "message": f"Heavy runner job completed but required artifacts missing: {missing_artifacts}",
        }

    return {
        "status": "error" if error_payload else "success",
        "input_uri": input_uri,
        "output_uri": output_uri,
        "dataset_uri": dataset_uri,
        "downloaded": downloaded,
        "missing_artifacts": missing_artifacts,
        "gcs_listing": gcs_listing,
        "job_stdout": stdout,
        "job_stderr": stderr,
        "gcloud_flag": flag_used,
        "gcloud_bin": gcloud_bin,
        "gsutil_bin": gsutil_bin,
        "error": error_payload,
    }
