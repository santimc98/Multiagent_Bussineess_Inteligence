"""
Contract validation utilities.

Ensures contract self-consistency and detects ambiguity between
"outputs as files" vs "outputs as columns".
"""
import os
import re
from typing import Any, Dict, List, Tuple, Optional


# File extensions that indicate a path is an artifact file
FILE_EXTENSIONS = {
    ".csv", ".json", ".md", ".png", ".jpg", ".jpeg", ".pdf",
    ".parquet", ".pkl", ".pickle", ".txt", ".html", ".xlsx", ".xls"
}


def is_file_path(value: str) -> bool:
    """
    Determine if a string looks like a file path.

    A file path contains:
    - A "/" or "\\" (path separator), OR
    - A recognized file extension
    """
    if not isinstance(value, str) or not value.strip():
        return False

    value = value.strip()

    # Contains path separator
    if "/" in value or "\\" in value:
        return True

    # Has a recognized file extension
    _, ext = os.path.splitext(value.lower())
    if ext in FILE_EXTENSIONS:
        return True

    return False


def is_column_name(value: str) -> bool:
    """
    Determine if a string looks like a column name.

    A column name:
    - Does NOT contain "/" or "\\"
    - Does NOT have a file extension
    - Is a valid identifier-like string
    """
    if not isinstance(value, str) or not value.strip():
        return False

    value = value.strip()

    # Contains path separator -> not a column
    if "/" in value or "\\" in value:
        return False

    # Has a file extension -> not a column
    _, ext = os.path.splitext(value.lower())
    if ext in FILE_EXTENSIONS:
        return False

    return True


def detect_output_ambiguity(
    required_outputs: List[Any]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Detect and separate ambiguous entries in required_outputs.

    Returns:
        (files, columns, warnings)
        - files: entries that are clearly file paths
        - columns: entries that are clearly column names
        - warnings: list of {"item": ..., "message": ...} for ambiguous cases
    """
    files = []
    columns = []
    warnings = []

    for item in required_outputs:
        # Handle both string and dict formats
        if isinstance(item, dict):
            path = item.get("path") or item.get("name") or ""
            desc = item.get("description", "")
        else:
            path = str(item) if item else ""
            desc = ""

        if not path:
            continue

        path_clean = path.strip()

        if is_file_path(path_clean):
            files.append({"path": path_clean, "description": desc})
        elif is_column_name(path_clean):
            # This looks like a column, not a file
            columns.append({"name": path_clean, "description": desc})
            warnings.append({
                "item": path_clean,
                "message": f"'{path_clean}' in required_outputs looks like a column name (no extension/path). Moved to required_columns.",
                "action": "moved_to_columns"
            })
        else:
            # Unknown - treat as file with warning
            files.append({"path": path_clean, "description": desc})
            warnings.append({
                "item": path_clean,
                "message": f"'{path_clean}' format is ambiguous. Treating as file path.",
                "action": "kept_as_file"
            })

    return files, columns, warnings


def normalize_artifact_requirements(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Normalize contract to use artifact_requirements structure.

    Converts legacy required_outputs to:
    - artifact_requirements.required_files (paths)
    - scored_rows_schema.required_columns (columns)

    Returns:
        (artifact_requirements, warnings)
    """
    warnings = []

    # Start with existing artifact_requirements if present
    artifact_req = contract.get("artifact_requirements", {})
    if not isinstance(artifact_req, dict):
        artifact_req = {}

    required_files = artifact_req.get("required_files", [])
    if not isinstance(required_files, list):
        required_files = []

    # Normalize mixed types (str vs dict) in required_files
    normalized_files = []
    for item in required_files:
        if isinstance(item, str) and item.strip():
            normalized_files.append({"path": item.strip(), "description": ""})
        elif isinstance(item, dict):
            normalized_files.append(item)
    required_files = normalized_files

    # Merge file_schemas keys if present
    file_schemas = artifact_req.get("file_schemas", {})
    if isinstance(file_schemas, dict):
        existing_paths = {f.get("path") for f in required_files if f.get("path")}
        for path in file_schemas.keys():
            if path and path not in existing_paths:
                required_files.append({"path": path, "description": "From file_schemas"})

    scored_schema = artifact_req.get("scored_rows_schema", {})
    if not isinstance(scored_schema, dict):
        scored_schema = {}

    required_columns = scored_schema.get("required_columns", [])
    if not isinstance(required_columns, list):
        required_columns = []

    # Process legacy required_outputs
    legacy_outputs = contract.get("required_outputs", [])
    if isinstance(legacy_outputs, list) and legacy_outputs:
        files, columns, ambig_warnings = detect_output_ambiguity(legacy_outputs)
        warnings.extend(ambig_warnings)

        # Merge with existing
        existing_paths = {f.get("path", "").lower() for f in required_files}
        for f in files:
            if f["path"].lower() not in existing_paths:
                required_files.append(f)
                existing_paths.add(f["path"].lower())

        existing_cols = {c.get("name", "").lower() if isinstance(c, dict) else str(c).lower() for c in required_columns}
        for c in columns:
            col_name = c.get("name", "")
            if col_name.lower() not in existing_cols:
                required_columns.append(col_name)
                existing_cols.add(col_name.lower())

    # Ensure minimum required files exist
    default_files = [
        {"path": "data/cleaned_data.csv", "description": "Cleaned dataset"},
        {"path": "data/metrics.json", "description": "Model metrics"},
    ]
    existing_paths = {f.get("path", "").lower() for f in required_files}
    for df in default_files:
        if df["path"].lower() not in existing_paths:
            required_files.append(df)

    # Build normalized structure
    artifact_requirements = {
        "required_files": required_files,
        "scored_rows_schema": {
            "required_columns": required_columns,
            "recommended_columns": scored_schema.get("recommended_columns", [])
        },
        "file_schemas": file_schemas
    }

    return artifact_requirements, warnings


def validate_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate contract for self-consistency.

    Rules:
    1) allowed_feature_sets must be subset of (canonical_columns ∪ derived_columns ∪ expandable)
    2) outcome_columns must be in canonical_columns (or explicitly unknown)
    3) decision_columns only if action_space/levers declared
    4) artifact_requirements.required_files must exist as list
    5) scored_rows_schema.required_columns must exist

    Returns:
        {
            "status": "ok" | "warning" | "error",
            "issues": [...],
            "normalized_artifact_requirements": {...}
        }
    """
    issues = []
    status = "ok"

    if not isinstance(contract, dict):
        return {
            "status": "error",
            "issues": [{"rule": "structure", "severity": "error", "message": "Contract must be a dictionary"}],
            "normalized_artifact_requirements": None
        }

    # Normalize artifact_requirements
    artifact_req, normalization_warnings = normalize_artifact_requirements(contract)
    for w in normalization_warnings:
        issues.append({
            "rule": "output_ambiguity",
            "severity": "warning",
            "message": w["message"],
            "item": w["item"]
        })
        if status == "ok":
            status = "warning"

    # Get column sets
    canonical_columns = set(contract.get("canonical_columns", []) or [])
    derived_columns = set(contract.get("derived_columns", []) or [])

    # Get feature selectors expandable columns (placeholder for now)
    selectors = contract.get("feature_selectors", [])
    expandable = set()
    # Feature selectors will be expanded later with actual data

    all_known_columns = canonical_columns | derived_columns | expandable

    # Rule 1: allowed_feature_sets validation
    allowed_sets = contract.get("allowed_feature_sets", {})
    if isinstance(allowed_sets, dict):
        for set_name, features in allowed_sets.items():
            if set_name == "forbidden":
                continue  # Skip forbidden set
            if isinstance(features, list):
                for feat in features:
                    if feat not in all_known_columns and not _matches_any_selector(feat, selectors):
                        # Only warn if we have canonical_columns defined
                        if canonical_columns:
                            issues.append({
                                "rule": "feature_set_consistency",
                                "severity": "warning",
                                "message": f"Feature '{feat}' in allowed_feature_sets.{set_name} not in canonical_columns",
                                "item": feat
                            })
                            if status == "ok":
                                status = "warning"

    # Rule 2: outcome_columns validation
    outcome_columns = contract.get("outcome_columns", [])
    if isinstance(outcome_columns, list):
        for oc in outcome_columns:
            if oc and oc not in all_known_columns:
                # Only warn if canonical_columns is defined
                if canonical_columns:
                    issues.append({
                        "rule": "outcome_column_consistency",
                        "severity": "warning",
                        "message": f"Outcome column '{oc}' not in canonical_columns",
                        "item": oc
                    })
                    if status == "ok":
                        status = "warning"

    # Rule 3: decision_columns validation
    decision_columns = contract.get("decision_columns", [])
    has_action_space = bool(contract.get("action_space") or contract.get("levers"))

    if decision_columns and not has_action_space:
        # Check if any evidence of editable levers
        business_objective = contract.get("business_objective", "")
        has_lever_evidence = any(
            kw in str(business_objective).lower()
            for kw in ["price", "precio", "discount", "descuento", "limit", "límite", "offer", "oferta"]
        )
        if not has_lever_evidence:
            issues.append({
                "rule": "decision_columns_without_levers",
                "severity": "warning",
                "message": f"decision_columns={decision_columns} declared but no action_space/levers defined",
                "item": decision_columns
            })
            if status == "ok":
                status = "warning"

    # Rule 4: artifact_requirements.required_files must exist
    req_files = artifact_req.get("required_files", [])
    if not isinstance(req_files, list) or not req_files:
        issues.append({
            "rule": "missing_required_files",
            "severity": "warning",
            "message": "artifact_requirements.required_files is empty or missing",
            "item": None
        })
        if status == "ok":
            status = "warning"

    # Rule 5: scored_rows_schema should exist
    scored_schema = artifact_req.get("scored_rows_schema", {})
    if not isinstance(scored_schema, dict):
        issues.append({
            "rule": "missing_scored_schema",
            "severity": "warning",
            "message": "artifact_requirements.scored_rows_schema is missing",
            "item": None
        })
        if status == "ok":
            status = "warning"

    return {
        "status": status,
        "issues": issues,
        "normalized_artifact_requirements": artifact_req
    }


def _matches_any_selector(feature: str, selectors: List[Dict]) -> bool:
    """Check if a feature matches any feature selector."""
    if not selectors:
        return False

    for sel in selectors:
        sel_type = sel.get("type", "")
        if sel_type == "regex":
            pattern = sel.get("pattern", "")
            if pattern and re.match(pattern, feature):
                return True
        elif sel_type == "prefix":
            prefix = sel.get("value", "")
            if prefix and feature.startswith(prefix):
                return True

    return False


def validate_output_compliance(
    artifact_requirements: Dict[str, Any],
    work_dir: str = ".",
    scored_rows_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate that required outputs exist and comply with schema.

    Args:
        artifact_requirements: Normalized artifact requirements
        work_dir: Working directory to check files in
        scored_rows_path: Optional explicit path to scored_rows.csv

    Returns:
        {
            "status": "ok" | "warning" | "error",
            "present_files": [...],
            "missing_files": [...],
            "missing_columns_in_scored_rows": [...],
            "warnings": [...]
        }
    """
    present_files = []
    missing_files = []
    missing_columns = []
    warnings = []

    # Check required files
    required_files = artifact_requirements.get("required_files", [])
    for f in required_files:
        path = f.get("path", "") if isinstance(f, dict) else str(f)
        if not path:
            continue

        full_path = os.path.join(work_dir, path)
        if os.path.exists(full_path):
            present_files.append(path)
        else:
            missing_files.append(path)

    # Check scored_rows columns
    scored_schema = artifact_requirements.get("scored_rows_schema", {})
    required_columns = scored_schema.get("required_columns", [])

    # Find scored_rows file
    if not scored_rows_path:
        for f in required_files:
            path = f.get("path", "") if isinstance(f, dict) else str(f)
            if "scored" in path.lower() and path.endswith(".csv"):
                scored_rows_path = path
                break

    if scored_rows_path and required_columns:
        full_path = os.path.join(work_dir, scored_rows_path)
        if os.path.exists(full_path):
            try:
                import pandas as pd
                # Only read header
                df_header = pd.read_csv(full_path, nrows=0)
                actual_columns = set(df_header.columns)

                for col in required_columns:
                    col_name = col.get("name", col) if isinstance(col, dict) else str(col)
                    if col_name and col_name not in actual_columns:
                        missing_columns.append(col_name)
            except Exception as e:
                warnings.append(f"Could not read scored_rows for column check: {e}")

    # Determine status
    if missing_files:
        status = "error"
    elif missing_columns:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "present_files": present_files,
        "missing_files": missing_files,
        "missing_columns_in_scored_rows": missing_columns,
        "warnings": warnings
    }
