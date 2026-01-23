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

# Universal set of metric-like tokens that should NOT appear as required columns in scored_rows_schema
# (metrics belong in metrics.json, not per-row)
METRIC_LIKE_TOKENS = {
    "accuracy", "roc_auc", "auc", "f1", "precision", "recall", "log_loss",
    "rmse", "mae", "mse", "r2", "rmsle", "gini", "normalized_gini",
    "f1_score", "f1-score", "roc-auc", "roc auc", "logloss", "cross_entropy",
    "balanced_accuracy", "cohen_kappa", "matthews_corrcoef", "mcc",
    "mean_squared_error", "mean_absolute_error", "root_mean_squared_error",
}

# Canonical column roles for column_roles normalization
CANONICAL_ROLES = {
    "outcome", "decision", "id", "feature", "timestamp", "group", "weight",
    "ignore", "forbidden", "target", "label", "identifier", "index",
    "segmentation", "audit_only", "protected", "sensitive",
}

# Role synonym mapping for normalization
ROLE_SYNONYM_MAP = {
    "target": "outcome",
    "label": "outcome",
    "identifier": "id",
    "index": "id",
    "ignored": "ignore",
    "excluded": "forbidden",
    "protected_attribute": "protected",
}


def _normalize_metric_name(name: str) -> str:
    """
    Normalize a metric name to canonical form.
    
    Rules:
      - lower()
      - replace spaces/hyphens with underscores
      - remove parentheses and common annotations
      - normalize common variants (roc-auc -> roc_auc, rmsle/rmse_log1p -> same token)
    
    Args:
        name: Raw metric name
        
    Returns:
        Normalized metric name string
    """
    if not isinstance(name, str):
        return str(name) if name is not None else ""
    
    # Lowercase
    s = name.lower().strip()
    
    # Remove common parenthetical annotations
    s = re.sub(r"\s*\([^)]*\)\s*", "", s)
    
    # Replace spaces and hyphens with underscores
    s = re.sub(r"[\s\-]+", "_", s)
    
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    
    # Canonical normalizations
    roc_auc_variants = {"rocauc", "roc_auc", "auc_roc", "auroc", "roc"}
    if s in roc_auc_variants or s == "roc_auc":
        return "roc_auc"
    
    # RMSE log variants
    rmse_log_variants = {"rmsle", "rmse_log", "rmse_log1p", "rmsle_log1p", "log_rmse"}
    if s in rmse_log_variants:
        return "rmsle"
    
    return s


def _is_metric_like_token(name: str) -> bool:
    """
    Check if a column name looks like a metric (should not be in required_columns).
    
    Args:
        name: Column/metric name to check
        
    Returns:
        True if the name looks like a metric
    """
    if not name:
        return False
    normalized = _normalize_metric_name(name)
    return normalized in METRIC_LIKE_TOKENS


def is_probably_path(value: str) -> bool:
    """
    Determine if a string looks like a filesystem path.

    Heuristics:
    - Contains "/" or "\\" or glob "*"
    - Starts with data/, static/, or reports/
    - Ends with a known file extension
    """
    if not isinstance(value, str) or not value.strip():
        return False

    value = value.strip()
    lower = value.lower()

    if "*" in value:
        return True
    if lower.startswith(("data/", "static/", "reports/")):
        return True
    if "/" in value or "\\" in value:
        return True
    _, ext = os.path.splitext(lower)
    if ext in FILE_EXTENSIONS:
        return True
    return False


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

    # Has a recognized file extension
    _, ext = os.path.splitext(value.lower())
    if ext in FILE_EXTENSIONS:
        return True

    # Contains path separator but looks like a conceptual phrase
    if ("/" in value or "\\" in value) and re.search(r"[\s\(\)\[\]\{\}<>]", value):
        return False

    # Contains path separator
    if "/" in value or "\\" in value:
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

    # Disallow spaces or bracketed annotations in column names for this heuristic
    if re.search(r"[\s\(\)\[\]\{\}<>]", value):
        return False

    # Basic identifier-like check (letters, numbers, underscore, dash, dot)
    if not re.match(r"^[A-Za-z0-9_.-]+$", value):
        return False

    return True


def detect_output_ambiguity(
    required_outputs: List[Any]
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Detect and separate ambiguous entries in required_outputs.

    Returns:
        (files, columns, warnings, conceptual_outputs)
        - files: entries that are clearly file paths
        - columns: entries that are clearly column names
        - warnings: list of {"item": ..., "message": ...} for ambiguous cases
        - conceptual_outputs: entries that are non-file, non-column output requests
    """
    files = []
    columns = []
    warnings = []
    conceptual_outputs = []

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
            conceptual_outputs.append({"name": path_clean, "description": desc})
            warnings.append({
                "item": path_clean,
                "message": f"'{path_clean}' looks like a conceptual output. Moved to reporting_requirements.",
                "action": "moved_to_conceptual_outputs"
            })

    return files, columns, warnings, conceptual_outputs


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
        files, columns, ambig_warnings, conceptual_outputs = detect_output_ambiguity(legacy_outputs)
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

        if conceptual_outputs:
            reporting_requirements = contract.get("reporting_requirements")
            if not isinstance(reporting_requirements, dict):
                reporting_requirements = {}
            existing_conceptual = reporting_requirements.get("conceptual_outputs")
            if not isinstance(existing_conceptual, list):
                existing_conceptual = []
            existing_lower = {str(item).lower() for item in existing_conceptual}
            for item in conceptual_outputs:
                name = item.get("name") if isinstance(item, dict) else str(item)
                if not name:
                    continue
                if name.lower() in existing_lower:
                    continue
                existing_conceptual.append(name)
                existing_lower.add(name.lower())
            reporting_requirements["conceptual_outputs"] = existing_conceptual
            existing_narrative = reporting_requirements.get("narrative_outputs")
            if not isinstance(existing_narrative, list):
                existing_narrative = []
            existing_narrative_lower = {str(item).lower() for item in existing_narrative}
            for item in existing_conceptual:
                if str(item).lower() in existing_narrative_lower:
                    continue
                existing_narrative.append(item)
                existing_narrative_lower.add(str(item).lower())
            reporting_requirements["narrative_outputs"] = existing_narrative
            contract["reporting_requirements"] = reporting_requirements
            notes = contract.get("notes_for_engineers")
            if not isinstance(notes, list):
                notes = []
            note = "Conceptual outputs requested (non-file): " + ", ".join(existing_conceptual)
            if note not in notes:
                notes.append(note)
            contract["notes_for_engineers"] = notes

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

    def _extract_path(item: Any) -> str:
        if not item:
            return ""
        if isinstance(item, dict):
            path = item.get("path") or item.get("output") or item.get("artifact")
            return str(path) if path else ""
        return str(item)

    required_plots = artifact_req.get("required_plots", [])
    if not isinstance(required_plots, list):
        required_plots = []
    combined_outputs: List[str] = []
    for source in (required_files, required_plots):
        for entry in source:
            path = _extract_path(entry)
            if path and is_probably_path(path):
                combined_outputs.append(path)
    contract["required_outputs"] = list(dict.fromkeys(combined_outputs))

    return artifact_requirements, warnings


def normalize_allowed_feature_sets(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalize allowed_feature_sets to canonical dict format.
    
    Handles:
      - dict (preferred): normalizes values, unifies synonyms (forbidden, forbidden_features -> forbidden_for_modeling)
      - list[str]: converts to {"model_features": list, ...}
      - list[list[str]]: flattens and converts to dict
      - Other types (None, str, int): returns empty dict with error note
    
    Args:
        contract: The contract dictionary
        
    Returns:
        (normalized_dict, notes) where notes contains repair messages
    """
    notes: List[str] = []
    raw = contract.get("allowed_feature_sets")
    
    # Already a dict - normalize it
    if isinstance(raw, dict):
        normalized: Dict[str, Any] = {}
        
        # Synonym mapping for key unification
        synonym_map = {
            "forbidden": "forbidden_for_modeling",
            "forbidden_features": "forbidden_for_modeling",
        }
        
        for key, value in raw.items():
            # Unify synonyms
            canonical_key = synonym_map.get(key, key)
            if canonical_key != key:
                notes.append(f"Unified '{key}' to '{canonical_key}' in allowed_feature_sets")
            
            # Normalize values to list of unique strings
            if canonical_key == "rationale":
                # Rationale is always kept as a string
                if isinstance(value, str):
                    normalized[canonical_key] = value.strip() if value else ""
                else:
                    normalized[canonical_key] = str(value) if value else ""
            elif isinstance(value, list):
                cleaned = []
                seen = set()
                for item in value:
                    if isinstance(item, str) and item.strip():
                        s = item.strip()
                        if s not in seen:
                            cleaned.append(s)
                            seen.add(s)
                    elif isinstance(item, list):
                        # Flatten nested lists
                        for sub in item:
                            if isinstance(sub, str) and sub.strip():
                                s = sub.strip()
                                if s not in seen:
                                    cleaned.append(s)
                                    seen.add(s)
                normalized[canonical_key] = cleaned
            elif isinstance(value, str):
                # Single string value -> list
                normalized[canonical_key] = [value.strip()] if value.strip() else []
        
        # Ensure required keys exist
        for req_key in ("model_features", "segmentation_features", "audit_only_features", "forbidden_for_modeling"):
            if req_key not in normalized:
                normalized[req_key] = []
        if "rationale" not in normalized:
            normalized["rationale"] = ""
        
        return normalized, notes
    
    # list[str] -> dict
    if isinstance(raw, list):
        flat_features: List[str] = []
        seen = set()
        
        for item in raw:
            if isinstance(item, str) and item.strip():
                s = item.strip()
                if s not in seen:
                    flat_features.append(s)
                    seen.add(s)
            elif isinstance(item, list):
                # list[list[str]] - flatten
                for sub in item:
                    if isinstance(sub, str) and sub.strip():
                        s = sub.strip()
                        if s not in seen:
                            flat_features.append(s)
                            seen.add(s)
        
        if flat_features:
            notes.append(
                f"Normalized allowed_feature_sets from list to dict (model_features={len(flat_features)} features)"
            )
            return {
                "model_features": flat_features,
                "segmentation_features": [],
                "audit_only_features": [],
                "forbidden_for_modeling": [],
                "rationale": "normalized_from_list",
            }, notes
        else:
            notes.append("allowed_feature_sets was an empty list; normalized to empty dict")
            return {
                "model_features": [],
                "segmentation_features": [],
                "audit_only_features": [],
                "forbidden_for_modeling": [],
                "rationale": "normalized_from_empty_list",
            }, notes
    
    # None or unrecognized type
    if raw is None:
        return {
            "model_features": [],
            "segmentation_features": [],
            "audit_only_features": [],
            "forbidden_for_modeling": [],
            "rationale": "",
        }, []
    
    # Invalid type (str, int, etc.)
    notes.append(
        f"allowed_feature_sets had invalid type '{type(raw).__name__}'; cannot normalize"
    )
    return {
        "model_features": [],
        "segmentation_features": [],
        "audit_only_features": [],
        "forbidden_for_modeling": [],
        "rationale": "ERROR: invalid_type_in_source",
    }, notes


def normalize_validation_requirements(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalize validation_requirements to canonical format.
    
    Rules:
      - Ensure dict type
      - Canonicalize metric names with _normalize_metric_name()
      - Move validation_requirements.metrics to metrics_to_report if latter missing
      - Ensure primary_metric is in metrics_to_report
      - If primary_metric missing, try to extract from qa_gates benchmark_kpi_report
    
    Args:
        contract: The contract dictionary
        
    Returns:
        (normalized_validation_requirements, notes)
    """
    notes: List[str] = []
    raw = contract.get("validation_requirements")
    
    if not isinstance(raw, dict):
        raw = {}
    
    # Deep copy to avoid mutation
    normalized = dict(raw)
    
    # Get or initialize metrics_to_report
    metrics_to_report = normalized.get("metrics_to_report")
    if not isinstance(metrics_to_report, list):
        metrics_to_report = []
    
    # Migration: validation_requirements.metrics -> metrics_to_report
    legacy_metrics = normalized.pop("metrics", None)
    if isinstance(legacy_metrics, list) and legacy_metrics:
        for m in legacy_metrics:
            if isinstance(m, str) and m.strip():
                canonical = _normalize_metric_name(m)
                if canonical not in [_normalize_metric_name(x) for x in metrics_to_report]:
                    metrics_to_report.append(canonical)
        notes.append(
            f"Migrated validation_requirements.metrics ({len(legacy_metrics)} items) to metrics_to_report"
        )
    
    # Canonicalize all metrics in metrics_to_report
    canonical_metrics = []
    seen = set()
    for m in metrics_to_report:
        if isinstance(m, str):
            c = _normalize_metric_name(m)
            if c and c not in seen:
                canonical_metrics.append(c)
                seen.add(c)
    normalized["metrics_to_report"] = canonical_metrics
    
    # Handle primary_metric
    primary_metric = normalized.get("primary_metric")
    if isinstance(primary_metric, str) and primary_metric.strip():
        primary_canonical = _normalize_metric_name(primary_metric)
        normalized["primary_metric"] = primary_canonical
        # Ensure it's in metrics_to_report
        if primary_canonical not in seen:
            canonical_metrics.insert(0, primary_canonical)
            notes.append(f"Added primary_metric '{primary_canonical}' to metrics_to_report")
    else:
        # Try to infer from qa_gates
        qa_gates = contract.get("qa_gates")
        if isinstance(qa_gates, list):
            for gate in qa_gates:
                if isinstance(gate, dict):
                    gate_name = gate.get("name", "").lower()
                    if "benchmark" in gate_name or "kpi" in gate_name or "metric" in gate_name:
                        params = gate.get("params", {})
                        if isinstance(params, dict):
                            metric = params.get("metric") or params.get("primary_metric")
                            if isinstance(metric, str) and metric.strip():
                                inferred = _normalize_metric_name(metric)
                                normalized["primary_metric"] = inferred
                                if inferred not in seen:
                                    canonical_metrics.insert(0, inferred)
                                notes.append(
                                    f"Inferred primary_metric '{inferred}' from qa_gates benchmark/kpi gate"
                                )
                                break
    
    normalized["metrics_to_report"] = canonical_metrics
    return normalized, notes


# =============================================================================
# CONTRACT SCHEMA LINTER (Fix #6)
# =============================================================================


def _normalize_role(role: str) -> str:
    """Normalize a role string to canonical form."""
    if not isinstance(role, str):
        return "feature"
    r = role.strip().lower()
    return ROLE_SYNONYM_MAP.get(r, r)


def lint_column_roles(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, str], List[Dict[str, Any]], List[str]]:
    """
    Lint and normalize column_roles to canonical dict format.

    Accepts:
      - dict (standard): {column: role, ...}
      - list[dict]: [{"column": "X", "role": "outcome"}, ...]
      - list[list/tuple]: [["X", "outcome"], ["Y", "feature"], ...]
      - str/int: error (irreparable)

    Returns:
        (normalized_dict, issues, notes)
        - normalized_dict: {column: canonical_role}
        - issues: list of {rule, severity, message, item}
        - notes: repair notes for unknowns
    """
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []
    raw = contract.get("column_roles")

    # None -> empty dict (ok)
    if raw is None:
        return {}, [], []

    # Already a dict - normalize roles
    if isinstance(raw, dict):
        normalized: Dict[str, str] = {}
        for col, role in raw.items():
            if not isinstance(col, str) or not col.strip():
                continue
            col_clean = col.strip()
            if isinstance(role, str):
                role_norm = _normalize_role(role)
                if role_norm not in CANONICAL_ROLES:
                    issues.append({
                        "rule": "contract_schema_lint.column_roles",
                        "severity": "warning",
                        "message": f"Unknown role '{role}' for column '{col_clean}'; keeping as-is",
                        "item": col_clean
                    })
                    notes.append(f"column_roles: unknown role '{role}' for '{col_clean}', kept literal")
                    role_norm = role.strip().lower()  # Keep literal
                normalized[col_clean] = role_norm
            else:
                # Non-string role
                issues.append({
                    "rule": "contract_schema_lint.column_roles",
                    "severity": "warning",
                    "message": f"Role for column '{col_clean}' is not a string (type={type(role).__name__}); defaulting to 'feature'",
                    "item": col_clean
                })
                notes.append(f"column_roles: non-string role for '{col_clean}', defaulted to 'feature'")
                normalized[col_clean] = "feature"
        return normalized, issues, notes

    # list[dict] format: [{"column": "X", "role": "outcome"}, ...]
    if isinstance(raw, list):
        normalized = {}
        for item in raw:
            if isinstance(item, dict):
                col = item.get("column") or item.get("name") or item.get("col")
                role = item.get("role") or item.get("type") or "feature"
                if isinstance(col, str) and col.strip():
                    col_clean = col.strip()
                    role_norm = _normalize_role(role) if isinstance(role, str) else "feature"
                    if role_norm not in CANONICAL_ROLES:
                        issues.append({
                            "rule": "contract_schema_lint.column_roles",
                            "severity": "warning",
                            "message": f"Unknown role '{role}' for column '{col_clean}'; defaulting to 'feature'",
                            "item": col_clean
                        })
                        notes.append(f"column_roles: unknown role '{role}' for '{col_clean}', defaulted to 'feature'")
                        role_norm = "feature"
                    normalized[col_clean] = role_norm
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # list[list/tuple] format: [["X", "outcome"], ...]
                col, role = item[0], item[1]
                if isinstance(col, str) and col.strip():
                    col_clean = col.strip()
                    role_norm = _normalize_role(role) if isinstance(role, str) else "feature"
                    if role_norm not in CANONICAL_ROLES:
                        issues.append({
                            "rule": "contract_schema_lint.column_roles",
                            "severity": "warning",
                            "message": f"Unknown role '{role}' for column '{col_clean}'; defaulting to 'feature'",
                            "item": col_clean
                        })
                        notes.append(f"column_roles: unknown role '{role}' for '{col_clean}', defaulted to 'feature'")
                        role_norm = "feature"
                    normalized[col_clean] = role_norm

        if normalized:
            notes.insert(0, f"column_roles: normalized from list format to dict ({len(normalized)} columns)")
            return normalized, issues, notes
        else:
            # Empty list or all items invalid
            notes.append("column_roles: list was empty or all items invalid, returning empty dict")
            return {}, issues, notes

    # Invalid type (str, int, etc.) - irreparable
    issues.append({
        "rule": "contract_schema_lint.column_roles",
        "severity": "fail",
        "message": f"column_roles has invalid type '{type(raw).__name__}'; must be dict or list",
        "item": "column_roles"
    })
    notes.append(f"column_roles: irreparable type '{type(raw).__name__}', returning empty dict")
    return {}, issues, notes


def lint_required_columns(
    required_columns: Any
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Lint scored_rows_schema.required_columns.

    Rules:
      - Must be list[str]
      - Strip, dedupe
      - Remove metric-like tokens
      - Remove path-like values (contains .csv/.json or '/')

    Returns:
        (clean_columns, issues, notes)
    """
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []

    # Handle None
    if required_columns is None:
        return [], [], []

    # str -> [str]
    if isinstance(required_columns, str):
        required_columns = [required_columns]
        notes.append("required_columns: converted single string to list")

    # Not a list at this point -> try to extract strings
    if not isinstance(required_columns, list):
        issues.append({
            "rule": "contract_schema_lint.required_columns",
            "severity": "warning",
            "message": f"required_columns has invalid type '{type(required_columns).__name__}'; returning empty",
            "item": "required_columns"
        })
        notes.append(f"required_columns: invalid type '{type(required_columns).__name__}', returned empty")
        return [], issues, notes

    clean: List[str] = []
    seen: set = set()

    for item in required_columns:
        # Extract string value
        if isinstance(item, dict):
            col_name = item.get("name") or item.get("column") or ""
        elif isinstance(item, str):
            col_name = item
        else:
            # Non-string, non-dict -> skip
            continue

        if not isinstance(col_name, str) or not col_name.strip():
            continue

        col_clean = col_name.strip()
        col_lower = col_clean.lower()

        # Skip duplicates
        if col_lower in seen:
            continue

        # Check for metric-like token
        if _is_metric_like_token(col_clean):
            issues.append({
                "rule": "contract_schema_lint.required_columns",
                "severity": "warning",
                "message": f"Removed metric-like token '{col_clean}' from required_columns; metrics belong in metrics.json",
                "item": col_clean
            })
            notes.append(f"required_columns: removed metric '{col_clean}'")
            continue

        # Check for path-like value
        if is_file_path(col_clean):
            issues.append({
                "rule": "contract_schema_lint.required_columns",
                "severity": "warning",
                "message": f"Removed path-like value '{col_clean}' from required_columns; looks like a file path",
                "item": col_clean
            })
            notes.append(f"required_columns: removed path-like '{col_clean}'")
            continue

        clean.append(col_clean)
        seen.add(col_lower)

    return clean, issues, notes


def lint_allowed_feature_sets_coherence(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """
    Lint allowed_feature_sets for internal coherence.

    Rules:
      - forbidden_for_modeling ∩ model_features != ∅ → auto-repair (remove from model_features)
      - outcome/decision columns in model_features → warning + auto-repair

    Returns:
        (repaired_allowed_feature_sets, issues, notes)
    """
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []

    allowed_sets = contract.get("allowed_feature_sets")
    if not isinstance(allowed_sets, dict):
        return allowed_sets if allowed_sets else {}, [], []

    # Deep copy to avoid mutation issues
    repaired = {k: list(v) if isinstance(v, list) else v for k, v in allowed_sets.items()}

    model_features = repaired.get("model_features", [])
    if not isinstance(model_features, list):
        model_features = []

    forbidden = repaired.get("forbidden_for_modeling", [])
    if not isinstance(forbidden, list):
        forbidden = []

    # Get outcome and decision columns
    column_roles = contract.get("column_roles", {})
    outcome_cols = set()
    decision_cols = set()

    if isinstance(column_roles, dict):
        for col, role in column_roles.items():
            role_norm = _normalize_role(role) if isinstance(role, str) else ""
            if role_norm == "outcome":
                outcome_cols.add(col)
            elif role_norm == "decision":
                decision_cols.add(col)

    # Also check explicit outcome_columns and decision_columns
    for col in contract.get("outcome_columns", []) or []:
        if isinstance(col, str):
            outcome_cols.add(col)
    for col in contract.get("decision_columns", []) or []:
        if isinstance(col, str):
            decision_cols.add(col)

    # Check 1: forbidden_for_modeling ∩ model_features
    forbidden_set = set(forbidden)
    model_set = set(model_features)
    overlap = forbidden_set & model_set

    if overlap:
        for col in overlap:
            model_features.remove(col)
            issues.append({
                "rule": "contract_schema_lint.feature_set_coherence",
                "severity": "warning",
                "message": f"Column '{col}' in both model_features and forbidden_for_modeling; removed from model_features",
                "item": col
            })
            notes.append(f"feature_set_coherence: removed '{col}' from model_features (was in forbidden)")
        repaired["model_features"] = model_features

    # Check 2: outcome/decision columns in model_features (leakage-by-contract)
    leakage_cols = (outcome_cols | decision_cols) & set(model_features)
    if leakage_cols:
        for col in leakage_cols:
            model_features.remove(col)
            col_type = "outcome" if col in outcome_cols else "decision"
            issues.append({
                "rule": "contract_schema_lint.feature_set_coherence",
                "severity": "warning",
                "message": f"Leakage-by-contract: {col_type} column '{col}' found in model_features; removed",
                "item": col
            })
            notes.append(f"feature_set_coherence: removed {col_type} column '{col}' from model_features (leakage)")
        repaired["model_features"] = model_features

    return repaired, issues, notes


def lint_artifact_requirements_coherence(
    contract: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Lint artifact_requirements for coherence with required_outputs.

    Rules:
      - If scored_rows_schema exists and expects scored_rows file but missing in required_outputs → warning

    Returns:
        (issues, notes)
    """
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []

    artifact_req = contract.get("artifact_requirements")
    if not isinstance(artifact_req, dict):
        return [], []

    scored_schema = artifact_req.get("scored_rows_schema")
    if not isinstance(scored_schema, dict):
        return [], []

    # Check if schema has required_columns (meaning scored_rows is expected)
    required_columns = scored_schema.get("required_columns", [])
    if not required_columns:
        return [], []

    # Check if scored_rows file is in required_files or required_outputs
    required_files = artifact_req.get("required_files", [])
    required_outputs = contract.get("required_outputs", [])

    has_scored_file = False

    # Check in required_files
    for f in required_files:
        path = f.get("path", "") if isinstance(f, dict) else str(f)
        if "scored" in path.lower() and path.endswith(".csv"):
            has_scored_file = True
            break

    # Check in required_outputs
    if not has_scored_file:
        for item in required_outputs:
            path = item.get("path", "") if isinstance(item, dict) else str(item)
            if "scored" in path.lower() and path.endswith(".csv"):
                has_scored_file = True
                break

    if not has_scored_file:
        issues.append({
            "rule": "contract_schema_lint.artifact_coherence",
            "severity": "warning",
            "message": "scored_rows_schema has required_columns but no scored_rows file in required_files/outputs",
            "item": "scored_rows_schema"
        })
        notes.append("artifact_coherence: scored_rows_schema defined but no scored_rows file in outputs")

    return issues, notes


def run_contract_schema_linter(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """
    Run the full contract schema linter.

    This is the main entry point for Fix #6 linting.
    Applies:
      A) column_roles normalization
      B) required_columns linting
      C) allowed_feature_sets coherence
      D) artifact_requirements coherence

    Args:
        contract: The contract dictionary (will be mutated)

    Returns:
        (contract, all_issues, all_notes)
    """
    all_issues: List[Dict[str, Any]] = []
    all_notes: List[str] = []
    has_critical_error = False

    # A) Lint column_roles
    normalized_roles, roles_issues, roles_notes = lint_column_roles(contract)
    contract["column_roles"] = normalized_roles
    all_issues.extend(roles_issues)
    all_notes.extend(roles_notes)

    # Check for critical error in column_roles
    for issue in roles_issues:
        if issue.get("severity") == "fail":
            has_critical_error = True

    # B) Lint required_columns in artifact_requirements
    artifact_req = contract.get("artifact_requirements")
    if isinstance(artifact_req, dict):
        scored_schema = artifact_req.get("scored_rows_schema")
        if isinstance(scored_schema, dict):
            raw_cols = scored_schema.get("required_columns")
            clean_cols, cols_issues, cols_notes = lint_required_columns(raw_cols)
            scored_schema["required_columns"] = clean_cols
            all_issues.extend(cols_issues)
            all_notes.extend(cols_notes)

    # C) Lint allowed_feature_sets coherence
    repaired_afs, afs_issues, afs_notes = lint_allowed_feature_sets_coherence(contract)
    if repaired_afs:
        contract["allowed_feature_sets"] = repaired_afs
    all_issues.extend(afs_issues)
    all_notes.extend(afs_notes)

    # D) Lint artifact_requirements coherence
    artifact_issues, artifact_notes = lint_artifact_requirements_coherence(contract)
    all_issues.extend(artifact_issues)
    all_notes.extend(artifact_notes)

    return contract, all_issues, all_notes


def lint_scored_rows_schema(
    contract: Dict[str, Any]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Lint scored_rows_schema.required_columns to remove metric-like tokens.
    
    Metrics (accuracy, roc_auc, f1, etc.) belong in metrics.json, not per-row columns.
    
    Args:
        contract: The contract dictionary
        
    Returns:
        (clean_required_columns, removed_metric_like, notes)
    """
    notes: List[str] = []
    removed: List[str] = []
    
    artifact_req = contract.get("artifact_requirements")
    if not isinstance(artifact_req, dict):
        return [], [], []
    
    scored_schema = artifact_req.get("scored_rows_schema")
    if not isinstance(scored_schema, dict):
        return [], [], []
    
    required_columns = scored_schema.get("required_columns")
    if not isinstance(required_columns, list):
        return [], [], []
    
    clean_columns: List[str] = []
    for col in required_columns:
        col_name = col.get("name") if isinstance(col, dict) else str(col) if col else ""
        if not col_name:
            continue
        
        if _is_metric_like_token(col_name):
            removed.append(col_name)
            notes.append(
                f"Removed metric-like required column '{col_name}' from scored_rows_schema; "
                "metrics belong in metrics.json"
            )
        else:
            # Preserve original format (str or dict)
            clean_columns.append(col if isinstance(col, dict) else col_name)
    
    return clean_columns, removed, notes


def validate_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate contract for self-consistency and apply strict normalization.

    Rules:
    1) allowed_feature_sets must be subset of (canonical_columns ∪ derived_columns ∪ expandable)
    2) outcome_columns must be in canonical_columns (or explicitly unknown)
    3) decision_columns only if action_space/levers declared
    4) artifact_requirements.required_files must exist as list
    5) scored_rows_schema.required_columns must exist
    6) Contract schema linting (Fix #6): column_roles, required_columns, coherence checks

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

    # Ensure unknowns is a list for traceability
    if not isinstance(contract.get("unknowns"), list):
        contract["unknowns"] = []

    # =========================================================================
    # STRICT NORMALIZATION (auto-repair)
    # =========================================================================
    
    # 1) Normalize allowed_feature_sets
    normalized_afs, afs_notes = normalize_allowed_feature_sets(contract)
    contract["allowed_feature_sets"] = normalized_afs
    
    # Check for critical failure (invalid type that couldn't be normalized)
    afs_has_error = normalized_afs.get("rationale", "").startswith("ERROR:")
    
    for note in afs_notes:
        is_error = "invalid type" in note.lower()
        issues.append({
            "rule": "contract_normalization",
            "severity": "error" if is_error else "warning",
            "message": f"[allowed_feature_sets] {note}",
            "item": "allowed_feature_sets"
        })
        if is_error:
            status = "error"
        elif status == "ok":
            status = "warning"
        
        # Add to unknowns for traceability
        unknowns = contract.get("unknowns")
        if isinstance(unknowns, list):
            unknowns.append(f"Normalized allowed_feature_sets: {note}")
    
    # 2) Normalize validation_requirements
    normalized_val_req, val_req_notes = normalize_validation_requirements(contract)
    contract["validation_requirements"] = normalized_val_req
    
    for note in val_req_notes:
        issues.append({
            "rule": "contract_normalization",
            "severity": "warning",
            "message": f"[validation_requirements] {note}",
            "item": "validation_requirements"
        })
        if status == "ok":
            status = "warning"
        
        # Add to unknowns for traceability
        unknowns = contract.get("unknowns")
        if isinstance(unknowns, list):
            unknowns.append(f"Normalized validation_requirements: {note}")
    
    # 3) Lint scored_rows_schema (remove metric-like columns)
    clean_cols, removed_metrics, lint_notes = lint_scored_rows_schema(contract)
    if removed_metrics:
        # Update the contract
        artifact_req_for_lint = contract.get("artifact_requirements")
        if isinstance(artifact_req_for_lint, dict):
            scored_schema = artifact_req_for_lint.get("scored_rows_schema")
            if isinstance(scored_schema, dict):
                scored_schema["required_columns"] = clean_cols
    
    for note in lint_notes:
        issues.append({
            "rule": "contract_normalization",
            "severity": "warning",
            "message": f"[scored_rows_schema] {note}",
            "item": "scored_rows_schema"
        })
        if status == "ok":
            status = "warning"
        
        # Add to unknowns for traceability
        unknowns = contract.get("unknowns")
        if isinstance(unknowns, list):
            unknowns.append(note)
    
    # Validate allowed_feature_sets is now a proper dict
    if afs_has_error:
        issues.append({
            "rule": "allowed_feature_sets_type",
            "severity": "error",
            "message": "allowed_feature_sets could not be normalized to a valid dict; contract is unusable",
            "item": "allowed_feature_sets"
        })
        status = "error"

    # =========================================================================
    # NORMALIZE ARTIFACT REQUIREMENTS (existing logic)
    # =========================================================================
    
    # Normalize artifact_requirements
    artifact_req, normalization_warnings = normalize_artifact_requirements(contract)
    contract["artifact_requirements"] = artifact_req  # Ensure it's set before linter runs
    for w in normalization_warnings:
        issues.append({
            "rule": "output_ambiguity",
            "severity": "warning",
            "message": w["message"],
            "item": w["item"]
        })
        if status == "ok":
            status = "warning"

    # =========================================================================
    # CONTRACT SCHEMA LINTER (Fix #6)
    # =========================================================================
    _, linter_issues, linter_notes = run_contract_schema_linter(contract)

    # Process linter issues
    for linter_issue in linter_issues:
        issues.append(linter_issue)
        severity = linter_issue.get("severity", "warning")
        if severity == "fail" or severity == "error":
            status = "error"
        elif status == "ok":
            status = "warning"

    # Add linter notes to unknowns for traceability
    unknowns = contract.get("unknowns")
    if isinstance(unknowns, list):
        for note in linter_notes:
            if note not in unknowns:
                unknowns.append(note)

    # Update artifact_req after linter may have modified it
    artifact_req = contract.get("artifact_requirements", artifact_req)

    # Get column sets
    canonical_columns = set(contract.get("canonical_columns", []) or [])
    derived_columns = set(contract.get("derived_columns", []) or [])

    # Get feature selectors expandable columns (placeholder for now)
    selectors = contract.get("feature_selectors", [])
    expandable = set()
    # Feature selectors will be expanded later with actual data

    all_known_columns = canonical_columns | derived_columns | expandable

    # Rule 1: allowed_feature_sets validation (now guaranteed to be a dict after normalization)
    allowed_sets = contract.get("allowed_feature_sets", {})
    if isinstance(allowed_sets, dict):
        for set_name, features in allowed_sets.items():
            # Skip forbidden set and non-feature keys
            if set_name in ("forbidden", "forbidden_for_modeling", "rationale"):
                continue
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
