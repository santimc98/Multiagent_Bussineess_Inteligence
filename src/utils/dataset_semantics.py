import re
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.csv_dialect import coerce_number

_NULL_STRINGS = {"", "na", "n/a", "nan", "null", "none", "nat"}
_TARGET_TOKENS = {"target", "label", "outcome", "response", "result", "groundtruth", "gt", "y"}
_PARTITION_NAME_TOKENS = {"split", "fold", "partition", "set", "bucket", "group"}
_PARTITION_VALUE_TOKENS = {"train", "test", "valid", "validation", "holdout", "dev"}
_DATETIME_TOKENS = {"date", "time", "timestamp", "datetime"}
_TEXT_TOKENS = {"text", "description", "comment", "note", "message", "summary"}
_ID_TOKENS = {"id", "uuid", "guid", "key", "identifier"}
_SUSPECT_TOKENS = {"target", "label", "outcome", "response", "gt", "groundtruth", "leak"}
_GENERIC_BOOLEAN_TOKENS = {"flag", "is", "has", "active", "enabled", "ok", "valid", "approved", "closed", "binary", "confirmed"}
_CONFIDENCE_THRESHOLDS = {"high": 4.0, "medium": 2.0}
_TEXT_OBJECTIVE_TOKENS = {"text", "nlp", "language", "sentiment", "topic", "document", "comment", "message", "description", "summary"}
_GENERIC_BINARY_VALUE_SETS = [
    {"0", "1"},
    {"1", "0"},
    {"true", "false"},
    {"false", "true"},
    {"yes", "no"},
    {"no", "yes"},
    {"y", "n"},
    {"n", "y"},
    {"m", "f"},
    {"f", "m"},
    {"male", "female"},
    {"female", "male"},
    {"on", "off"},
    {"off", "on"},
    {"t", "f"},
    {"f", "t"},
]


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _null_mask(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([], dtype=bool)
    mask = series.isna()
    if series.dtype == object:
        lowered = series.astype(str).str.strip().str.lower()
        mask = mask | lowered.isin(_NULL_STRINGS)
    return mask


def _numeric_series(series: pd.Series, decimal: str) -> pd.Series:
    if series is None:
        return pd.Series([], dtype="float64")
    return series.apply(lambda x: coerce_number(x, decimal))


def _parse_datetime_ratio(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    sample = series.dropna().astype(str)
    if sample.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean())


def _analyze_column(series: pd.Series, decimal: str) -> Dict[str, Any]:
    mask = _null_mask(series)
    total = int(series.shape[0]) if series is not None else 0
    non_null = int((~mask).sum()) if total else 0
    null_frac = float(mask.mean()) if total else 1.0
    values = series[~mask].astype(str) if non_null else pd.Series([], dtype=str)
    n_unique = int(values.nunique(dropna=True)) if non_null else 0
    unique_ratio = float(n_unique / max(non_null, 1))
    numeric_values = _numeric_series(values, decimal)
    numeric_ratio = float(numeric_values.notna().mean()) if non_null else 0.0
    is_numeric_like = numeric_ratio >= 0.8 and non_null >= 5
    is_binary_like = n_unique <= 2 and non_null >= 5
    avg_len = float(values.str.len().mean()) if non_null else 0.0
    return {
        "null_frac": null_frac,
        "n_unique": n_unique,
        "unique_ratio": unique_ratio,
        "numeric_ratio": numeric_ratio,
        "is_numeric_like": is_numeric_like,
        "is_binary_like": is_binary_like,
        "avg_len": avg_len,
    }


def _has_token(name: str, tokens: set[str]) -> bool:
    norm = _normalize_name(name)
    if not norm:
        return False
    for tok in tokens:
        if tok == "y":
            if norm == "y":
                return True
            continue
        if tok in norm:
            return True
    return False


def _detect_partition_values(series: pd.Series) -> Dict[str, List[str]]:
    values = series.dropna().astype(str).str.strip().str.lower()
    unique_vals = sorted({val for val in values if val})
    train_like = [val for val in unique_vals if any(tok in val for tok in _PARTITION_VALUE_TOKENS)]
    test_like = [val for val in unique_vals if any(tok in val for tok in {"test", "holdout", "validation", "valid", "dev"})]
    return {"train_like": train_like, "test_like": test_like}


def _score_target_candidate(name: str, stats: Dict[str, Any]) -> float:
    score = 0.0
    if _has_token(name, _TARGET_TOKENS):
        score += 2.0
    if stats.get("is_binary_like"):
        score += 1.0
    if stats.get("is_numeric_like"):
        score += 0.5
    return score


def infer_dataset_semantics(
    csv_path: str,
    dialect: Dict[str, Any],
    contract_min: Optional[Dict[str, Any]],
    text_context: str | None = None,
    max_sample_rows: int = 2000,
) -> Dict[str, Any]:
    result = {
        "column_roles": {},
        "target_analysis": {
            "target_candidates": [],
            "alternative_candidates": [],
            "partial_label_detected": False,
            "labeled_row_heuristic": "no_target_detected",
            "primary_target": None,
            "target_unclear": False,
            "target_unclear_reason": "",
        },
        "partition_analysis": {"partition_columns": [], "per_partition_counts": {}},
        "leakage_risks": {"correlated_with_target": [], "suspicious_name_columns": []},
        "notes": [],
    }
    if not csv_path:
        result["notes"].append("No csv_path provided; dataset semantics unavailable.")
        return result
    try:
        sep = dialect.get("sep") or ","
        decimal = dialect.get("decimal") or "."
        encoding = dialect.get("encoding") or "utf-8"
        header_df = pd.read_csv(csv_path, nrows=0, sep=sep, decimal=decimal, encoding=encoding)
        columns = [str(col) for col in header_df.columns]
        df_sample = pd.read_csv(
            csv_path,
            nrows=max_sample_rows,
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            low_memory=False,
            dtype=str,
        )
    except Exception as err:
        result["notes"].append(f"Failed to read CSV sample: {err}")
        return result

    column_stats: Dict[str, Dict[str, Any]] = {}
    for col in columns:
        if col not in df_sample.columns:
            continue
        column_stats[col] = _analyze_column(df_sample[col], decimal)

    contract_min = contract_min if isinstance(contract_min, dict) else {}
    contract_targets = [str(col) for col in (contract_min.get("outcome_columns") or []) if col]
    contract_targets_present = [col for col in contract_targets if col in columns]
    target_candidates: List[Dict[str, Any]] = []
    alternative_candidates: List[Dict[str, Any]] = []

    def _build_contract_candidate(column: str) -> Dict[str, Any]:
        stats = column_stats.get(column, {})
        present = column in columns and column in column_stats
        why = ["Declared as outcome in contract", "Contract-defined outcomes take precedence over heuristics"]
        if not present:
            why.append("Outcome column missing from sample/header")
        return {
            "column": column,
            "null_frac": float(stats.get("null_frac", 1.0)),
            "n_unique": int(stats.get("n_unique", 0)),
            "is_binary_like": bool(stats.get("is_binary_like", False)),
            "is_numeric_like": bool(stats.get("is_numeric_like", False)),
            "source": "contract",
            "confidence": "high",
            "why_candidate": why[:5],
        }

    def _is_text_objective(text: str | None) -> bool:
        if not text:
            return False
        normalized = _normalize_name(text)
        if not normalized:
            return False
        return any(tok in normalized for tok in _TEXT_OBJECTIVE_TOKENS)

    text_objective = _is_text_objective(text_context)
    text_context_value = text_context or ""
    text_context_lower = text_context_value.lower() if isinstance(text_context_value, str) else ""
    text_context_norm = _normalize_name(text_context_value)

    def _is_generic_binary_column(name: str) -> bool:
        if not name:
            return False
        norm = _normalize_name(name)
        for tok in _GENERIC_BOOLEAN_TOKENS:
            if norm == tok or norm.startswith(f"{tok}"):
                return True
        return False

    def _is_generic_binary_values(series: pd.Series) -> bool:
        if series is None or series.empty:
            return False
        mask = _null_mask(series)
        values = series[~mask].astype(str).str.strip().str.lower()
        if values.empty:
            return False
        unique_vals = {val for val in values if val}
        if len(unique_vals) < 2 or len(unique_vals) > 4:
            return False
        return any(unique_vals <= expected for expected in _GENERIC_BINARY_VALUE_SETS)

    def _score_heuristic_candidate(col: str, source_tag: str = "heuristic") -> Optional[Dict[str, Any]]:
        stats = column_stats.get(col, {})
        if not stats:
            return None
        score = 0.0
        reasons: List[str] = []
        signals = {"target_token": False, "null_frac": False, "correlation": False, "text_context": False}
        if _has_token(col, _TARGET_TOKENS):
            score += 2.0
            reasons.append("name contains target token")
            signals["target_token"] = True
        null_frac = float(stats.get("null_frac", 1.0))
        if 0.0 < null_frac < 0.9:
            score += 1.0
            reasons.append("moderate null fraction suggests label partiality")
            signals["null_frac"] = True
        if stats.get("is_binary_like"):
            reasons.append("binary signature present")
        if stats.get("is_numeric_like") and stats.get("n_unique", 0) > 1:
            col_series = _numeric_series(df_sample[col], decimal)
            corr_bonus = False
            for other_col in columns:
                if other_col == col or other_col not in df_sample.columns:
                    continue
                other_stats = column_stats.get(other_col, {})
                if not other_stats or not other_stats.get("is_numeric_like"):
                    continue
                other_series = _numeric_series(df_sample[other_col], decimal)
                valid = col_series.notna() & other_series.notna()
                if int(valid.sum()) < 30:
                    continue
                corr = float(col_series[valid].corr(other_series[valid]))
                if pd.isna(corr):
                    continue
                if abs(corr) >= 0.4:
                    score += 1.0
                    corr_bonus = True
                    reasons.append(f"correlation found with {other_col}")
                    signals["correlation"] = True
                    break
            if not corr_bonus:
                reasons.append("numeric column without strong correlation")
        if text_context_lower and (col.lower() in text_context_lower or _normalize_name(col) in text_context_norm):
            score += 1.0
            reasons.append("column referenced in contract/business text")
            signals["text_context"] = True
        if _has_token(col, _ID_TOKENS) or (
            stats.get("unique_ratio", 0.0) >= 0.95 and stats.get("n_unique", 0) >= 20
        ):
            score -= 4.0
            reasons.append("identifier-like signature detected (penalized)")
        if stats.get("is_binary_like"):
            generic_binary = _is_generic_binary_column(col) or _is_generic_binary_values(df_sample[col])
            if generic_binary and not any(signals.values()):
                score -= 1.0
                reasons.append("generic binary indicator without target signals (penalized)")
        value_hits = _detect_partition_values(df_sample[col])
        if value_hits.get("train_like") or value_hits.get("test_like") or _has_token(col, _PARTITION_NAME_TOKENS):
            score -= 4.0
            reasons.append("partition signature detected (penalized)")
        if stats.get("avg_len", 0.0) >= 20 and stats.get("unique_ratio", 0.0) >= 0.3:
            if text_objective:
                reasons.append("text-like column allowed by objective")
            else:
                score -= 2.0
                reasons.append("long text-like column (penalized)")
        if stats.get("n_unique", 0) <= 1:
            score -= 5.0
            reasons.append("constant-like column (penalized)")
        datetime_ratio = _parse_datetime_ratio(df_sample[col])
        if _has_token(col, _DATETIME_TOKENS) or datetime_ratio >= 0.7:
            score -= 2.0
            reasons.append("datetime feature detected (deprioritized)")

        if score <= 0:
            return None
        confidence = "high" if score >= _CONFIDENCE_THRESHOLDS["high"] else "medium" if score >= _CONFIDENCE_THRESHOLDS["medium"] else "low"
        if len(reasons) < 2:
            reasons.append("limited signals; treat as tentative")
        return {
            "column": col,
            "null_frac": float(stats.get("null_frac", 1.0)),
            "n_unique": int(stats.get("n_unique", 0)),
            "is_binary_like": bool(stats.get("is_binary_like", False)),
            "is_numeric_like": bool(stats.get("is_numeric_like", False)),
            "source": source_tag,
            "confidence": confidence,
            "why_candidate": reasons[:5],
            "score": score,
        }

    for col in contract_targets:
        target_candidates.append(_build_contract_candidate(col))

    if contract_targets:
        for col in columns:
            if col in contract_targets:
                continue
            candidate = _score_heuristic_candidate(col, source_tag="heuristic_hint")
            if candidate:
                alternative_candidates.append(candidate)
        alternative_candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        alternative_candidates = alternative_candidates[:3]
        if alternative_candidates:
            result["target_analysis"]["alternative_candidates"] = alternative_candidates
    else:
        for col in columns:
            candidate = _score_heuristic_candidate(col)
            if candidate:
                target_candidates.append(candidate)

    seen_targets = set()
    deduped_targets = []
    for item in target_candidates:
        name = item.get("column")
        if name and name not in seen_targets:
            seen_targets.add(name)
            deduped_targets.append(item)
    if contract_targets:
        target_candidates = deduped_targets
    else:
        deduped_targets.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        target_candidates = deduped_targets[:3]

    result["target_analysis"]["target_candidates"] = target_candidates

    primary_target = None
    if contract_targets:
        primary_target = contract_targets[0]
    elif target_candidates:
        highest_score = max((item.get("score", 0.0) for item in target_candidates), default=0.0)
        if highest_score >= _CONFIDENCE_THRESHOLDS["medium"]:
            primary_target = target_candidates[0].get("column")

    if not contract_targets:
        if not target_candidates:
            result["target_analysis"]["target_unclear"] = True
            result["target_analysis"]["target_unclear_reason"] = "no candidate exceeded heuristic threshold"
        else:
            highest_score = max((item.get("score", 0.0) for item in target_candidates), default=0.0)
            if highest_score < _CONFIDENCE_THRESHOLDS["medium"]:
                result["target_analysis"]["target_unclear"] = True
                result["target_analysis"]["target_unclear_reason"] = "all heuristics scored below medium confidence"

    if primary_target and primary_target in column_stats:
        stats = column_stats[primary_target]
        null_frac = float(stats.get("null_frac", 0.0))
        partial_labels = null_frac > 0.0 and null_frac < 1.0
        result["target_analysis"]["partial_label_detected"] = partial_labels
        result["target_analysis"]["labeled_row_heuristic"] = f"target_notnull:{primary_target}"
        result["target_analysis"]["primary_target"] = primary_target
    elif target_candidates:
        result["target_analysis"]["labeled_row_heuristic"] = "target_detected_but_not_in_sample"

    partition_columns = []
    per_partition_counts = {}
    for col in columns:
        stats = column_stats.get(col, {})
        if not stats:
            continue
        n_unique = int(stats.get("n_unique", 0))
        if n_unique <= 1 or n_unique > 20:
            continue
        if _has_token(col, _PARTITION_NAME_TOKENS):
            partition_columns.append(col)
        else:
            value_hits = _detect_partition_values(df_sample[col])
            if value_hits["train_like"] or value_hits["test_like"]:
                partition_columns.append(col)
        if col in partition_columns:
            counts = df_sample[col].astype(str).value_counts(dropna=True).head(6)
            per_partition_counts[col] = {str(idx): int(val) for idx, val in counts.items()}

    result["partition_analysis"]["partition_columns"] = partition_columns
    result["partition_analysis"]["per_partition_counts"] = per_partition_counts

    column_roles: Dict[str, str] = {}
    dropped_high_card = []
    target_role_cols = set()
    for item in target_candidates:
        if item.get("source") == "contract" or item.get("confidence") in {"medium", "high"}:
            if item.get("column"):
                target_role_cols.add(item.get("column"))

    for col in columns:
        stats = column_stats.get(col, {})
        if not stats:
            continue
        if stats.get("n_unique", 0) <= 1:
            column_roles[col] = "constant_like"
            continue
        if col in target_role_cols:
            column_roles[col] = "target_candidate"
            continue
        if col in partition_columns:
            column_roles[col] = "split_candidate"
            continue
        if _has_token(col, _ID_TOKENS) or (stats.get("unique_ratio", 0.0) >= 0.95 and stats.get("n_unique", 0) >= 20):
            column_roles[col] = "id_like"
            continue
        datetime_ratio = _parse_datetime_ratio(df_sample[col])
        if _has_token(col, _DATETIME_TOKENS) or datetime_ratio >= 0.7:
            column_roles[col] = "datetime_candidate"
            continue
        if stats.get("avg_len", 0.0) >= 20 and stats.get("unique_ratio", 0.0) >= 0.3:
            column_roles[col] = "text_candidate"
            continue
        if stats.get("unique_ratio", 0.0) >= 0.8 and stats.get("n_unique", 0) >= 20:
            column_roles[col] = "high_cardinality"

    if len(column_roles) > 200:
        high_card_cols = [col for col, role in column_roles.items() if role == "high_cardinality"]
        high_card_cols.sort(
            key=lambda item: column_stats.get(item, {}).get("unique_ratio", 0.0),
            reverse=True,
        )
        for col in high_card_cols[50:]:
            dropped_high_card.append(col)
            column_roles.pop(col, None)

    result["column_roles"] = column_roles

    suspicious_names = [col for col in columns if _has_token(col, _SUSPECT_TOKENS)]
    result["leakage_risks"]["suspicious_name_columns"] = suspicious_names[:8]

    correlated = []
    if primary_target and primary_target in df_sample.columns:
        target_series = _numeric_series(df_sample[primary_target], decimal)
        for col in columns:
            if col == primary_target or col not in df_sample.columns:
                continue
            stats = column_stats.get(col, {})
            if not stats or not stats.get("is_numeric_like"):
                continue
            numeric_series = _numeric_series(df_sample[col], decimal)
            valid = target_series.notna() & numeric_series.notna()
            if int(valid.sum()) < 10:
                continue
            corr = float(target_series[valid].corr(numeric_series[valid]))
            if pd.isna(corr):
                continue
            correlated.append(
                {
                    "column": col,
                    "corr": corr,
                    "corr_abs": float(abs(corr)),
                    "n_pairs": int(valid.sum()),
                }
            )
    correlated.sort(key=lambda item: item.get("corr_abs", 0.0), reverse=True)
    result["leakage_risks"]["correlated_with_target"] = correlated[:5]

    notes = []
    if contract_targets:
        notes.append(f"Contract outcome columns present: {contract_targets}")
        missing_contract = [col for col in contract_targets if col not in contract_targets_present]
        if missing_contract:
            notes.append(f"Contract outcomes missing from sample/header: {missing_contract[:5]}")
    if target_candidates:
        notes.append("Target candidates inferred from contract/heuristics.")
    else:
        notes.append("No clear target candidates inferred from sample.")
    if primary_target:
        notes.append(f"Primary target candidate: {primary_target}")
    notes.append(f"Partial labels detected: {result['target_analysis']['partial_label_detected']}")
    if partition_columns:
        notes.append(f"Partition candidates: {partition_columns[:3]}")
    else:
        notes.append("No partition columns detected in sample.")
    id_like = [col for col, role in column_roles.items() if role == "id_like"]
    if id_like:
        notes.append(f"ID-like columns: {id_like[:5]}")
    datetime_cols = [col for col, role in column_roles.items() if role == "datetime_candidate"]
    if datetime_cols:
        notes.append(f"Datetime candidates: {datetime_cols[:5]}")
    text_cols = [col for col, role in column_roles.items() if role == "text_candidate"]
    if text_cols:
        notes.append(f"Text candidates: {text_cols[:5]}")
    high_card = [col for col, role in column_roles.items() if role == "high_cardinality"]
    if high_card:
        notes.append(f"High-cardinality columns: {high_card[:5]}")
    constant_like = [col for col, role in column_roles.items() if role == "constant_like"]
    if constant_like:
        notes.append(f"Constant-like columns: {constant_like[:5]}")
    if dropped_high_card:
        notes.append(f"Column roles truncated for compactness; dropped {len(dropped_high_card)} high-cardinality columns.")
    if suspicious_names:
        notes.append(f"Suspicious name tokens: {suspicious_names[:5]}")
    if correlated:
        notes.append("High correlation candidates identified in sample.")

    result["notes"] = notes[:15]
    return result


def choose_training_mask(
    df_sample: Optional[pd.DataFrame],
    semantics: Dict[str, Any],
    contract_min: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    target_candidates = semantics.get("target_analysis", {}).get("target_candidates") or []
    contract_targets = [str(col) for col in (contract_min.get("outcome_columns") or []) if col]
    primary_target = contract_targets[0] if contract_targets else (target_candidates[0].get("column") if target_candidates else None)
    target_unclear = bool(semantics.get("target_analysis", {}).get("target_unclear"))
    if not contract_targets and target_unclear:
        primary_target = None

    partial_labels = bool(semantics.get("target_analysis", {}).get("partial_label_detected"))
    partition_cols = semantics.get("partition_analysis", {}).get("partition_columns") or []

    training_rule = "use all rows"
    scoring_primary = "use all rows"
    scoring_secondary = None
    rationale: List[str] = []

    if primary_target:
        rationale.append(f"primary target inferred: {primary_target}")
        if partial_labels:
            training_rule = f"use rows where {primary_target} is not null"
            scoring_secondary = f"use rows where {primary_target} is null"
            rationale.append("partial labels detected -> training limited to labeled rows")

    if df_sample is not None and partition_cols:
        for col in partition_cols:
            if col not in df_sample.columns:
                continue
            value_hits = _detect_partition_values(df_sample[col])
            train_like = value_hits.get("train_like") or []
            test_like = value_hits.get("test_like") or []
            if train_like or test_like:
                if train_like:
                    base_rule = f"use partition column {col} with values in {train_like}"
                    training_rule = (
                        f"{base_rule} and {primary_target} not null"
                        if partial_labels and primary_target
                        else base_rule
                    )
                    rationale.append(f"partition {col} has train-like values {train_like}")
                if test_like:
                    if partial_labels:
                        rationale.append(f"partition {col} has scoring subset {test_like} (kept as secondary hint)")
                    else:
                        scoring_primary = f"use partition column {col} with values in {test_like}"
                        rationale.append(f"partition {col} has scoring subset {test_like}")
                break
        if partition_cols and not any("partition column" in note for note in rationale):
            rationale.append("partition columns detected but no clear split labels")

    return {
        "training_rows_rule": training_rule,
        "scoring_rows_rule_primary": scoring_primary,
        "scoring_rows_rule_secondary": scoring_secondary,
        "rationale": rationale[:6],
    }


def summarize_dataset_semantics(
    semantics: Dict[str, Any],
    training_mask: Optional[Dict[str, Any]] = None,
    max_lines: int = 18,
) -> str:
    training_mask = training_mask if isinstance(training_mask, dict) else {}
    target_candidates = semantics.get("target_analysis", {}).get("target_candidates") or []
    target_details = [
        f"{item.get('column')}({item.get('confidence', 'low')})"
        for item in target_candidates
        if item.get("column")
    ]
    alt_candidates = semantics.get("target_analysis", {}).get("alternative_candidates") or []
    alt_details = [
        f"{item.get('column')}({item.get('confidence', 'low')})"
        for item in alt_candidates
        if item.get("column")
    ]
    partition_cols = semantics.get("partition_analysis", {}).get("partition_columns") or []
    column_roles = semantics.get("column_roles") or {}
    id_like = [col for col, role in column_roles.items() if role == "id_like"]
    datetime_cols = [col for col, role in column_roles.items() if role == "datetime_candidate"]
    text_cols = [col for col, role in column_roles.items() if role == "text_candidate"]
    high_card = [col for col, role in column_roles.items() if role == "high_cardinality"]
    constant_like = [col for col, role in column_roles.items() if role == "constant_like"]
    leakage_names = semantics.get("leakage_risks", {}).get("suspicious_name_columns") or []
    correlated = semantics.get("leakage_risks", {}).get("correlated_with_target") or []
    corr_cols = [item.get("column") for item in correlated if item.get("column")]
    partial_labels = semantics.get("target_analysis", {}).get("partial_label_detected")
    labeled_heuristic = semantics.get("target_analysis", {}).get("labeled_row_heuristic") or "unknown"

    lines = []
    lines.append("DATASET_SEMANTICS_SUMMARY:")
    lines.append(f"- target_candidates: {target_details[:5] if target_details else 'none'}")
    lines.append(f"- partial_label_detected: {bool(partial_labels)}")
    lines.append(f"- labeled_row_heuristic: {labeled_heuristic}")
    lines.append(f"- partition_columns: {partition_cols[:5] if partition_cols else 'none'}")
    lines.append(f"- training_rows_rule: {training_mask.get('training_rows_rule', 'use all rows')}")
    lines.append(
        f"- scoring_rows_rule_primary: {training_mask.get('scoring_rows_rule_primary', 'use all rows')}"
    )
    secondary = training_mask.get("scoring_rows_rule_secondary")
    if secondary:
        lines.append(f"- scoring_rows_rule_secondary: {secondary}")
    rationale = training_mask.get("rationale") or []
    if rationale:
        lines.append(f"- rationale: {rationale[:3]}")
    lines.append(f"- id_like: {id_like[:5] if id_like else 'none'}")
    lines.append(f"- datetime_candidates: {datetime_cols[:5] if datetime_cols else 'none'}")
    lines.append(f"- text_candidates: {text_cols[:5] if text_cols else 'none'}")
    lines.append(f"- high_cardinality: {high_card[:5] if high_card else 'none'}")
    lines.append(f"- constant_like: {constant_like[:5] if constant_like else 'none'}")
    lines.append(f"- leakage_name_suspects: {leakage_names[:5] if leakage_names else 'none'}")
    if alt_details:
        lines.append(f"- target_candidate_alternatives: {alt_details[:3]}")
    lines.append(f"- leakage_top_correlations: {corr_cols[:5] if corr_cols else 'none'}")

    return "\n".join(lines[:max_lines])
