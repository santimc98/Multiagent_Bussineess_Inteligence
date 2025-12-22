import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def normalize_manifest(manifest: Dict) -> Dict[str, Dict]:
    conversions = {}
    dropped = {}
    column_mapping = {}

    raw_conv = manifest.get("conversions", {}) or {}
    if isinstance(raw_conv, dict):
        for k, v in raw_conv.items():
            conversions[str(k).lower()] = v

    raw_dropped = manifest.get("dropped_columns", {}) or {}
    if isinstance(raw_dropped, list):
        for item in raw_dropped:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).lower()
            if name:
                dropped[name] = item
    elif isinstance(raw_dropped, dict):
        for k, v in raw_dropped.items():
            dropped[str(k).lower()] = v if isinstance(v, dict) else {"reason": str(v)}

    raw_mapping = manifest.get("column_mapping", {}) or {}
    if isinstance(raw_mapping, dict):
        for k, v in raw_mapping.items():
            column_mapping[str(k)] = str(v).lower()

    return {
        "conversions": conversions,
        "dropped_columns": dropped,
        "column_mapping": column_mapping,
    }


def find_raw_column_for_required(required_name: str, raw_columns: Iterable[str], column_mapping: Dict[str, str]) -> Optional[str]:
    required_norm = _normalize(required_name)
    # direct mapping raw->cleaned stored lower
    for raw, cleaned in column_mapping.items():
        if _normalize(cleaned) == required_norm:
            return raw
    for col in raw_columns:
        if _normalize(col) == required_norm:
            return col
    return None


def sample_raw_nonnull_rate(
    csv_path: str,
    dialect: Dict[str, str],
    colname: str,
    n: int = 500,
) -> Tuple[float, float]:
    try:
        sep = dialect.get("sep", ",")
        decimal = dialect.get("decimal", ".")
        encoding = dialect.get("encoding", "utf-8")
        df = pd.read_csv(
            csv_path,
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            usecols=[colname],
            nrows=n,
        )
    except Exception:
        return 0.0, 0.0
    if colname not in df.columns:
        return 0.0, 0.0
    series = df[colname]
    total = len(series)
    non_null = series.dropna()
    non_null_frac = len(non_null) / total if total else 0.0
    digits_ratio = _digit_ratio(series)
    return non_null_frac, digits_ratio


def detect_destructive_drop(
    manifest: Dict,
    required_cols: Iterable[str],
    raw_csv_path: str,
    raw_dialect: Dict[str, str],
    non_null_threshold: float = 0.2,
) -> List[Dict[str, object]]:
    issues = []
    norm = normalize_manifest(manifest)
    conversions = norm["conversions"]
    dropped = norm["dropped_columns"]
    column_mapping = norm["column_mapping"]
    raw_columns = manifest.get("original_columns", []) or []
    suspicious_reasons = {"empty", "100% null", "all_null", "all missing", "all_missing", "100% missing"}

    for req in required_cols:
        req_lower = str(req).lower()
        drop_info = dropped.get(req_lower, {})
        drop_reason = drop_info.get("reason", "") if isinstance(drop_info, dict) else str(drop_info)
        drop_reason_norm = drop_reason.strip().lower()
        conv_type = conversions.get(req_lower, "")
        if drop_reason_norm not in suspicious_reasons:
            continue
        raw_name = find_raw_column_for_required(req_lower, raw_columns, column_mapping) or req
        non_null_frac, digits_ratio = sample_raw_nonnull_rate(raw_csv_path, raw_dialect, raw_name)
        if non_null_frac < non_null_threshold:
            continue
        issues.append(
            {
                "column": req_lower,
                "raw_name": raw_name,
                "drop_reason": drop_reason,
                "conversion_type": conv_type,
                "raw_non_null_frac": non_null_frac,
                "digit_ratio": digits_ratio,
            }
        )
    return issues

# Backward-compatible alias for existing imports
def detect_destructive_conversions(*args, **kwargs):
    return detect_destructive_drop(*args, **kwargs)

def format_issue_report(*args, **kwargs):
    return format_patch_instructions(*args, **kwargs)


def format_patch_instructions(issues: List[Dict[str, object]]) -> str:
    if not issues:
        return ""
    lines = [
        "DESTRUCTIVE_CONVERSION_GUARD: One or more required columns were destroyed by numeric/currency conversion.",
        "Rules:",
        "- Do NOT convert columns with low digits_ratio to numeric_currency.",
        "- If conversion reduces non-nulls drastically, revert to text instead of dropping.",
        "- Never drop required columns because a numeric conversion produced 100% nulls.",
        "- Do NOT drop required columns from the strategy.",
        "- Before declaring a column 'empty', only treat NaN/None/'' (strip=='') as empty; 0/1/'0' are valid values.",
        "- If a column became empty after coercion/parsing, revert and keep as text; log in manifest instead of dropping.",
        "- If you suspect dialect/parse issues, re-read with correct dialect or avoid conversion.",
        "Evidence:"
    ]
    for issue in issues:
        lines.append(
            f"* col='{issue.get('column')}' raw='{issue.get('raw_name')}' conv='{issue.get('conversion_type')}' drop='{issue.get('drop_reason')}' raw_non_null={issue.get('raw_non_null_frac'):.3f} digit_ratio={issue.get('digit_ratio'):.3f}"
        )
    return "\n".join(lines)


def _normalize(name: str) -> str:
    return re.sub(r"[ _]", "", str(name).lower())


def _digit_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    return sum(any(ch.isdigit() for ch in str(x)) for x in non_null) / len(non_null)

# Implementation snippet requested: public helper for sampling raw columns
import pandas as pd

def sample_raw_columns(
    csv_path: str,
    dialect: dict | None,
    usecols: list[str],
    nrows: int = 500,
    dtype: str | None = None,
) -> pd.DataFrame:
    """
    Best-effort: lee una muestra (nrows) del CSV raw usando dialecto, limitando a usecols.
    Nunca lanza excepción: si falla devuelve DataFrame vacío.
    """
    if not usecols:
        return pd.DataFrame()

    d = dialect or {}
    encoding = d.get("encoding", "utf-8")
    sep = d.get("sep", ",")
    decimal = d.get("decimal", ".")

    try:
            return pd.read_csv(
                csv_path,
                encoding=encoding,
                sep=sep,
                decimal=decimal,
                usecols=usecols,
                nrows=nrows,
                dtype=dtype,
                low_memory=False,
            )
    except ValueError:
        # usecols no encaja -> intersectar con header real y reintentar
        try:
            header_cols = pd.read_csv(
                csv_path,
                encoding=encoding,
                sep=sep,
                decimal=decimal,
                nrows=0,
                low_memory=False,
            ).columns
            present = [c for c in usecols if c in set(header_cols)]
            if not present:
                return pd.DataFrame()
            return pd.read_csv(
                csv_path,
                encoding=encoding,
                sep=sep,
                decimal=decimal,
                usecols=present,
                nrows=nrows,
                dtype=dtype,
                low_memory=False,
            )
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
