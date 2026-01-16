import glob
import json
import os
from typing import List, Dict, Any, Tuple, Optional


def get_csv_dialect(work_dir: str = ".") -> Dict[str, Any]:
    """
    Read CSV dialect settings from cleaning_manifest.json.

    Args:
        work_dir: Working directory where data/cleaning_manifest.json may exist

    Returns:
        {"sep": str, "encoding": str} with defaults if manifest not found
    """
    defaults = {"sep": ",", "encoding": "utf-8"}

    manifest_path = os.path.join(work_dir, "data", "cleaning_manifest.json")
    if not os.path.exists(manifest_path):
        return defaults

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        output_dialect = manifest.get("output_dialect", {})
        if not isinstance(output_dialect, dict):
            return defaults

        return {
            "sep": output_dialect.get("sep", defaults["sep"]),
            "encoding": output_dialect.get("encoding", defaults["encoding"]),
        }
    except Exception:
        return defaults


def _split_deliverables(deliverables: List[Any]) -> Tuple[List[str], List[str]]:
    required: List[str] = []
    optional: List[str] = []
    for item in deliverables or []:
        if isinstance(item, dict):
            path = item.get("path") or item.get("output") or item.get("artifact")
            if not path:
                continue
            is_required = item.get("required")
            if is_required is None:
                is_required = True
            if is_required:
                required.append(path)
            else:
                optional.append(path)
        else:
            path = str(item)
            if not path:
                continue

            # CRITICAL FIX: Handle stringified dicts like "{'path': 'data/scored_rows.csv', ...}"
            # These come from execution_contract when artifact schemas are included in required_outputs
            # Extract just the path value and ignore schema metadata
            if path.startswith("{") and "'path':" in path:
                import ast
                try:
                    # Parse as Python literal to extract path
                    parsed = ast.literal_eval(path)
                    if isinstance(parsed, dict) and "path" in parsed:
                        path = parsed["path"]
                except Exception:
                    # If parsing fails, skip this entry (it's malformed)
                    continue

            if path:
                required.append(path)
    return required, optional


def _check_paths(paths: List[str]) -> Tuple[List[str], List[str]]:
    present: List[str] = []
    missing: List[str] = []
    for pattern in paths or []:
        try:
            if any(char in pattern for char in ["*", "?", "["]):
                matches = glob.glob(pattern)
                if matches:
                    present.extend(matches)
                else:
                    missing.append(pattern)
            else:
                if os.path.exists(pattern):
                    present.append(pattern)
                else:
                    missing.append(pattern)
        except Exception:
            missing.append(pattern)
    return present, missing


def check_required_outputs(required_outputs: List[Any]) -> Dict[str, object]:
    """
    Best-effort validation of required outputs.
    Supports glob patterns (e.g., static/plots/*.png).
    Returns dict with present, missing, summary.
    Never raises.
    """
    required, optional = _split_deliverables(required_outputs or [])
    present_required, missing_required = _check_paths(required)
    present_optional, missing_optional = _check_paths(optional)
    present = present_required + present_optional
    summary = (
        f"Present: {len(present)}; Missing required: {len(missing_required)}; "
        f"Missing optional: {len(missing_optional)}"
    )
    return {
        "present": present,
        "missing": missing_required,
        "missing_optional": missing_optional,
        "summary": summary,
    }


def check_scored_rows_schema(
    scored_rows_path: str,
    required_columns: List[str],
    required_any_of_groups: Optional[List[List[str]]] = None,
    required_any_of_group_severity: Optional[List[str]] = None,
    dialect: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Check that scored_rows.csv exists and contains required columns.
    Supports optional any-of groups for semantic column matching with severity.

    Args:
        scored_rows_path: Path to scored_rows.csv
        required_columns: List of required column names (legacy)
        required_any_of_groups: List of groups, each group is a list of column names.
                                At least one column from each group must be present.
        required_any_of_group_severity: List of severity strings ("fail" or "warning") for each group.
                                        Defaults to all "fail" except first group => "warning".
        dialect: Optional {"sep": str, "encoding": str} from get_csv_dialect()

    Returns:
        {
            "exists": bool,
            "present_columns": [...],
            "missing_columns": [...],
            "missing_any_of_groups": [...],
            "matched_any_of_groups": [...],
            "missing_any_of_groups_with_severity": [{"group": [...], "severity": "..."}],
            "summary": str
        }
    """
    if not os.path.exists(scored_rows_path):
        return {
            "exists": False,
            "present_columns": [],
            "missing_columns": required_columns,
            "missing_any_of_groups": required_any_of_groups or [],
            "matched_any_of_groups": [],
            "missing_any_of_groups_with_severity": [],
            "summary": f"File not found: {scored_rows_path}",
        }

    # Use dialect settings or defaults
    sep = dialect.get("sep", ",") if dialect else ","
    encoding = dialect.get("encoding", "utf-8") if dialect else "utf-8"

    try:
        import pandas as pd
        # Only read header to check columns, respecting dialect
        df_header = pd.read_csv(scored_rows_path, nrows=0, sep=sep, encoding=encoding)
        actual_cols = list(df_header.columns)
        actual_norm = {col.strip().lower(): col for col in actual_cols}

        # Check required_columns (legacy, case-insensitive)
        present = []
        missing = []
        for col in required_columns:
            if col.lower() in actual_norm:
                present.append(col)
            else:
                missing.append(col)

        # Default severity: all "fail" except first group => "warning"
        if required_any_of_groups and not required_any_of_group_severity:
            required_any_of_group_severity = ["warning"] + ["fail"] * (len(required_any_of_groups) - 1)

        # Check any-of groups with severity
        missing_groups = []
        matched_groups = []
        missing_with_severity = []
        if required_any_of_groups:
            for idx, group in enumerate(required_any_of_groups):
                if not group or not isinstance(group, list):
                    continue
                group_norm = [g.strip().lower() for g in group if g]
                matches = [actual_norm.get(g) for g in group_norm if g in actual_norm]
                if not matches:
                    missing_groups.append(group)
                    severity = required_any_of_group_severity[idx] if idx < len(required_any_of_group_severity) else "fail"
                    missing_with_severity.append({"group": group, "severity": severity})
                else:
                    matched_groups.append({"group": group, "matched": matches})

        summary = f"Columns - Present: {len(present)}/{len(required_columns)}"
        if missing:
            summary += f"; Missing: {', '.join(missing)}"
        if missing_groups:
            summary += f"; Missing groups: {len(missing_groups)}"

        return {
            "exists": True,
            "present_columns": present,
            "missing_columns": missing,
            "missing_any_of_groups": missing_groups,
            "matched_any_of_groups": matched_groups,
            "missing_any_of_groups_with_severity": missing_with_severity,
            "summary": summary,
        }
    except Exception as e:
        return {
            "exists": True,
            "present_columns": [],
            "missing_columns": required_columns,
            "missing_any_of_groups": required_any_of_groups or [],
            "matched_any_of_groups": [],
            "missing_any_of_groups_with_severity": [],
            "summary": f"Error reading file: {e}",
        }


def check_artifact_requirements(
    artifact_requirements: Dict[str, Any],
    work_dir: str = ".",
) -> Dict[str, Any]:
    """
    P1.4: Check all artifact requirements including files and scored_rows schema.

    Args:
        artifact_requirements: Normalized artifact requirements dict
        work_dir: Working directory for file paths

    Returns:
        {
            "status": "ok" | "warning" | "error",
            "files_report": {...},
            "scored_rows_report": {...},
            "summary": str
        }
    """
    if not isinstance(artifact_requirements, dict):
        return {
            "status": "error",
            "files_report": {},
            "scored_rows_report": {},
            "summary": "Invalid artifact_requirements",
        }

    # Check required files
    required_files = artifact_requirements.get("required_files", [])
    file_paths = []
    for f in required_files:
        if isinstance(f, dict):
            path = f.get("path", "")
        else:
            path = str(f) if f else ""
        if path:
            file_paths.append(os.path.join(work_dir, path))

    files_report = check_required_outputs(file_paths)

    # Check scored_rows schema
    scored_schema = artifact_requirements.get("scored_rows_schema", {})
    required_columns = scored_schema.get("required_columns", [])
    required_any_of_groups = scored_schema.get("required_any_of_groups")
    required_any_of_group_severity = scored_schema.get("required_any_of_group_severity")

    # Find scored_rows file
    scored_rows_path = None
    for f in required_files:
        path = f.get("path", "") if isinstance(f, dict) else str(f)
        if "scored" in path.lower() and path.endswith(".csv"):
            scored_rows_path = os.path.join(work_dir, path)
            break

    if scored_rows_path:
        # Use dialect from cleaning_manifest if available
        dialect = get_csv_dialect(work_dir)
        scored_rows_report = check_scored_rows_schema(
            scored_rows_path,
            required_columns,
            required_any_of_groups,
            required_any_of_group_severity,
            dialect
        )
    else:
        scored_rows_report = {
            "exists": False,
            "present_columns": [],
            "missing_columns": required_columns,
            "missing_any_of_groups": required_any_of_groups or [],
            "matched_any_of_groups": [],
            "missing_any_of_groups_with_severity": [],
            "summary": "No scored_rows file defined in required_files",
        }

    # Determine overall status
    if files_report.get("missing"):
        status = "error"
    elif scored_rows_report.get("missing_columns"):
        status = "error"
    else:
        # Check any-of groups with severity
        missing_with_severity = scored_rows_report.get("missing_any_of_groups_with_severity", [])
        has_fail_severity = any(m["severity"] == "fail" for m in missing_with_severity)
        if has_fail_severity:
            status = "error"
        elif missing_with_severity:
            status = "warning"
        else:
            status = "ok"

    summary = f"Files: {files_report.get('summary', 'N/A')}; Scored rows: {scored_rows_report.get('summary', 'N/A')}"

    return {
        "status": status,
        "files_report": files_report,
        "scored_rows_report": scored_rows_report,
        "summary": summary,
    }
