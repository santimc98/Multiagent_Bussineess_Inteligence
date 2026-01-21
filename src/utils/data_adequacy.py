import csv
import difflib
import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.json_sanitize import dump_json

from src.utils.contract_v41 import get_outcome_columns, get_column_roles


def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_output_dialect(manifest_path: str = "data/cleaning_manifest.json") -> Dict[str, Any] | None:
    if not os.path.exists(manifest_path):
        return None
    manifest = _safe_load_json(manifest_path)
    dialect = manifest.get("output_dialect") if isinstance(manifest, dict) else None
    if isinstance(dialect, dict) and dialect.get("sep"):
        return dialect
    return None


def _read_sample(path: str, encoding: str, size: int = 50_000) -> str:
    try:
        with open(path, "r", encoding=encoding, errors="replace") as handle:
            return handle.read(size)
    except Exception:
        return ""


def _sniff_delimiter(sample: str) -> str:
    candidates = [",", ";", "\t", "|"]
    if sample:
        try:
            sniffed = csv.Sniffer().sniff(sample, delimiters=candidates)
            if sniffed.delimiter in candidates:
                return sniffed.delimiter
        except Exception:
            pass
    counts = {delim: sample.count(delim) for delim in candidates}
    best = max(counts, key=counts.get) if counts else ","
    return best if counts.get(best, 0) > 0 else ","


def _infer_decimal_from_sample(sample: str, sep: str) -> str:
    comma_hits = len(re.findall(r"\d+,\d+", sample))
    dot_hits = len(re.findall(r"\d+\.\d+", sample))
    if comma_hits > dot_hits:
        return ","
    return "."


def _sniff_csv_dialect(path: str) -> Dict[str, Any]:
    sample = _read_sample(path, "utf-8")
    sep = _sniff_delimiter(sample)
    decimal = _infer_decimal_from_sample(sample, sep)
    return {"sep": sep, "decimal": decimal, "encoding": "utf-8"}


def _read_header_line(path: str, encoding: str) -> str:
    try:
        with open(path, "r", encoding=encoding, errors="replace") as handle:
            return (handle.readline() or "").strip()
    except Exception:
        return ""


def _pick_alternate_sep(header: str, current_sep: str) -> str | None:
    candidates = [",", ";", "\t", "|"]
    best = None
    best_count = 0
    for delim in candidates:
        if delim == current_sep:
            continue
        count = header.count(delim)
        if count > best_count:
            best_count = count
            best = delim
    return best


def _safe_load_csv(path: str) -> tuple[pd.DataFrame | None, str | None]:
    if not os.path.exists(path):
        return None, "file_missing"
    dialect = _load_output_dialect() or _sniff_csv_dialect(path)
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    try:
        df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding)
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"
    if df.shape[1] == 1:
        header = _read_header_line(path, encoding)
        alt_sep = _pick_alternate_sep(header, sep)
        if header and alt_sep:
            sample = _read_sample(path, encoding)
            alt_decimal = _infer_decimal_from_sample(sample, alt_sep)
            try:
                alt_df = pd.read_csv(path, sep=alt_sep, decimal=alt_decimal, encoding=encoding)
                if alt_df.shape[1] > 1:
                    return alt_df, None
            except Exception:
                pass
    return df, None


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _normalize_key(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def _metric_category(metric_name: str) -> str:
    key = _normalize_key(metric_name)
    if any(token in key for token in ["spearman", "kendall", "rank", "corr", "correlation"]):
        return "ranking"
    if any(
        token in key
        for token in [
            "f1",
            "roc",
            "auc",
            "pr_auc",
            "average_precision",
            "precision",
            "recall",
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "logloss",
            "brier",
        ]
    ):
        return "classification"
    if any(token in key for token in ["mae", "rmse", "mse", "mape", "smape", "r2", "r_squared"]):
        return "regression"
    return "other"


def _infer_objective_family_from_metrics(metric_keys: List[str]) -> str | None:
    counts: Dict[str, int] = {}
    for key in metric_keys:
        category = _metric_category(key)
        if category == "other":
            continue
        counts[category] = counts.get(category, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _infer_objective_family(
    contract: Dict[str, Any],
    metrics_report: Dict[str, Any],
    weights: Dict[str, Any],
) -> str | None:
    validation = contract.get("validation_requirements", {}) if isinstance(contract, dict) else {}
    metric_keys: List[str] = []
    metrics_spec = validation.get("metrics") if isinstance(validation, dict) else None
    if isinstance(metrics_spec, dict):
        metric_keys.extend([str(key) for key in metrics_spec.keys()])
    elif isinstance(metrics_spec, list):
        metric_keys.extend([str(key) for key in metrics_spec])
    elif isinstance(metrics_spec, str):
        metric_keys.append(metrics_spec)
    if not metric_keys:
        metric_pool = _extract_metric_pool(weights, metrics_report)
        metric_keys = list(metric_pool.keys())
    return _infer_objective_family_from_metrics(metric_keys)


def _metric_higher_is_better(metric_name: str) -> bool:
    key = _normalize_key(metric_name)
    if any(token in key for token in ["loss", "error", "mae", "rmse", "mse", "mape", "smape", "brier"]):
        return False
    if "r2" in key or "r_squared" in key:
        return True
    return True


def _is_baseline_metric(metric_name: str) -> bool:
    key = _normalize_key(metric_name)
    return any(token in key for token in ["baseline", "dummy", "naive", "null", "default"])


def _collect_metric_entries(obj: Any, prefix: str, pool: Dict[str, float]) -> None:
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        metric_key = f"{prefix}{key}" if prefix else str(key)
        if _is_number(value):
            pool[metric_key] = float(value)
        elif isinstance(value, dict):
            _collect_metric_entries(value, f"{metric_key}.", pool)


def _extract_metric_pool(weights: Dict[str, Any], metrics_report: Dict[str, Any]) -> Dict[str, float]:
    pool: Dict[str, float] = {}
    if isinstance(weights, dict):
        for key in ["metrics", "classification_metrics", "regression_metrics", "model_metrics", "global_metrics"]:
            metrics = weights.get(key)
            if isinstance(metrics, dict):
                for metric_key, metric_val in metrics.items():
                    if _is_number(metric_val):
                        pool[str(metric_key)] = float(metric_val)
        for metric_key, metric_val in weights.items():
            if isinstance(metric_val, (int, float)):
                pool[str(metric_key)] = float(metric_val)
    if isinstance(metrics_report, dict):
        _collect_metric_entries(metrics_report, "", pool)
    return pool


def _select_metric(
    metric_pool: Dict[str, float],
    category: str,
    preference: List[str],
    baseline_only: bool | None = None,
) -> Tuple[str | None, float | None]:
    if not metric_pool:
        return None, None
    candidates = {
        key: value for key, value in metric_pool.items() if _metric_category(key) == category
    }
    if baseline_only is True:
        candidates = {key: value for key, value in candidates.items() if _is_baseline_metric(key)}
    elif baseline_only is False:
        candidates = {key: value for key, value in candidates.items() if not _is_baseline_metric(key)}
    if not candidates:
        return None, None
    normalized = {key: _normalize_key(key) for key in candidates}
    for token in preference:
        for key, norm in normalized.items():
            if token in norm:
                return key, candidates[key]
    chosen = sorted(candidates.keys())[0]
    return chosen, candidates[chosen]


def _align_quality_gates(
    gates: Dict[str, Any], metric_pool: Dict[str, float]
) -> Dict[str, Any]:
    if not isinstance(gates, dict) or not gates:
        return {
            "status": "no_gates",
            "mapped_gates": {},
            "unmapped_gates": {},
            "available_metrics": sorted(metric_pool.keys()),
        }
    mapped: Dict[str, Any] = {}
    unmapped: Dict[str, Any] = {}
    available_metrics = list(metric_pool.keys())
    for gate_key, threshold in gates.items():
        if not gate_key:
            continue
        if gate_key in metric_pool:
            mapped[str(gate_key)] = {
                "metric": gate_key,
                "threshold": threshold,
                "match": "exact",
                "similarity": 1.0,
            }
            continue
        gate_category = _metric_category(gate_key)
        if gate_category == "other":
            unmapped[str(gate_key)] = threshold
            continue
        category_candidates = [key for key in available_metrics if _metric_category(key) == gate_category]
        if not category_candidates:
            unmapped[str(gate_key)] = threshold
            continue
        normalized_gate = _normalize_key(gate_key)
        best_key = None
        best_score = 0.0
        for candidate in category_candidates:
            score = difflib.SequenceMatcher(None, normalized_gate, _normalize_key(candidate)).ratio()
            if score > best_score:
                best_score = score
                best_key = candidate
        if best_key is None:
            unmapped[str(gate_key)] = threshold
            continue
        match_type = "similarity" if best_score >= 0.6 else "category_fallback"
        if best_score < 0.6 and len(category_candidates) > 1:
            unmapped[str(gate_key)] = threshold
            continue
        mapped[str(gate_key)] = {
            "metric": best_key,
            "threshold": threshold,
            "match": match_type,
            "similarity": float(best_score),
        }
    status = "aligned" if not unmapped else "partial"
    return {
        "status": status,
        "mapped_gates": mapped,
        "unmapped_gates": unmapped,
        "available_metrics": sorted(metric_pool.keys()) if metric_pool else None,
    }


def _find_target_column(contract: Dict[str, Any], df: pd.DataFrame | None) -> str | None:
    # V4.1: Use get_outcome_columns first
    target_columns = get_outcome_columns(contract)
    
    # Fallback: check column_roles for target-like roles
    if not target_columns:
        roles = get_column_roles(contract)
        target_roles = {"target", "outcome", "target_label", "derived_label"}
        for role in target_roles:
            if role in roles:
                target_columns.extend(roles[role])
    
    for candidate in target_columns:
        if df is not None and candidate in df.columns:
            return candidate
    return target_columns[0] if target_columns else None


def _infer_target_kind(series: pd.Series | None) -> str:
    if series is None:
        return "unknown"
    try:
        non_null = series.dropna()
        nunique = int(non_null.nunique())
    except Exception:
        return "unknown"
    if nunique <= 2:
        return "binary"
    try:
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().mean() >= 0.9:
            return "numeric"
    except Exception:
        pass
    if nunique <= 12:
        return "categorical"
    return "numeric"


def _binary_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        y = np.asarray(y)
        scores = np.asarray(scores)
        mask = np.isfinite(scores)
        y = y[mask]
        scores = scores[mask]
        if y.size == 0:
            return None
        pos = y == 1
        n_pos = int(pos.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return None
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        sum_ranks_pos = float(ranks[pos].sum())
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)
    except Exception:
        return None


def _best_auc_proxy(df: pd.DataFrame, target: pd.Series, feature_cols: List[str]) -> Dict[str, Any]:
    best_auc = None
    best_feature = None
    if df is None or target is None:
        return {"best_auc": None, "best_feature": None}
    try:
        target_values = pd.to_numeric(target, errors="coerce")
        uniq = sorted(target_values.dropna().unique())
        if len(uniq) != 2:
            return {"best_auc": None, "best_feature": None}
        mapping = {uniq[0]: 0, uniq[1]: 1}
        y = target_values.map(mapping).to_numpy()
    except Exception:
        return {"best_auc": None, "best_feature": None}
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() < 20:
            continue
        auc = _binary_auc(y, numeric.to_numpy())
        if auc is None:
            continue
        auc = max(auc, 1.0 - auc)
        if best_auc is None or auc > best_auc:
            best_auc = auc
            best_feature = col
    if best_auc is None:
        for col in feature_cols:
            if col not in df.columns:
                continue
            series = df[col]
            if series.dropna().nunique() > max(50, int(len(series) * 0.2)):
                continue
            try:
                means = target.groupby(series).mean()
                scores = series.map(means)
            except Exception:
                continue
            auc = _binary_auc(y, pd.to_numeric(scores, errors="coerce").to_numpy())
            if auc is None:
                continue
            auc = max(auc, 1.0 - auc)
            if best_auc is None or auc > best_auc:
                best_auc = auc
                best_feature = col
    return {"best_auc": best_auc, "best_feature": best_feature}


def _best_regression_proxy(df: pd.DataFrame, target: pd.Series, feature_cols: List[str]) -> Dict[str, Any]:
    best_spearman = None
    best_feature = None
    best_r2 = None
    if df is None or target is None:
        return {"best_abs_spearman": None, "best_feature": None, "best_r2_proxy": None}
    target_numeric = pd.to_numeric(target, errors="coerce")
    if target_numeric.notna().sum() < 20:
        return {"best_abs_spearman": None, "best_feature": None, "best_r2_proxy": None}
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= 20:
            try:
                corr = numeric.corr(target_numeric, method="spearman")
            except Exception:
                corr = None
            if corr is not None and np.isfinite(corr):
                abs_corr = float(abs(corr))
                if best_spearman is None or abs_corr > best_spearman:
                    best_spearman = abs_corr
                    best_feature = col
        if series.dropna().nunique() <= max(50, int(len(series) * 0.2)):
            try:
                means = target_numeric.groupby(series).mean()
                overall = float(target_numeric.mean())
                counts = series.value_counts(dropna=True)
                between = sum(counts.get(idx, 0) * (mean - overall) ** 2 for idx, mean in means.items())
                total = float(np.nanvar(target_numeric.to_numpy())) * len(target_numeric.dropna())
                if total > 0:
                    r2_proxy = float(between / total)
                    if best_r2 is None or r2_proxy > best_r2:
                        best_r2 = r2_proxy
            except Exception:
                pass
    best_r2 = best_r2 if best_r2 is not None else (best_spearman ** 2 if best_spearman is not None else None)
    return {"best_abs_spearman": best_spearman, "best_feature": best_feature, "best_r2_proxy": best_r2}


def _calc_lift(baseline: float | None, model: float | None, higher_is_better: bool) -> float | None:
    if baseline is None or model is None:
        return None
    if higher_is_better:
        return model - baseline
    if baseline == 0:
        return None
    return (baseline - model) / baseline


def _segment_coverage(case_summary: pd.DataFrame | None, min_size: int | None) -> Tuple[float | None, int | None]:
    if case_summary is None or min_size is None or "Segment_Size" not in case_summary.columns:
        return None, None
    try:
        small = case_summary["Segment_Size"].astype(float) < float(min_size)
        if small.empty:
            return None, None
        return float(small.mean()), int(small.sum())
    except Exception:
        return None, None


def build_data_adequacy_report(state: Dict[str, Any]) -> Dict[str, Any]:
    contract = _safe_load_json("data/execution_contract.json") or state.get("execution_contract", {})
    weights = _safe_load_json("data/weights.json")
    metrics_report = _safe_load_json("data/metrics.json")
    cleaned, cleaned_err = _safe_load_csv("data/cleaned_data.csv")
    case_summary, _case_summary_err = _safe_load_csv("data/case_summary.csv")
    cleaned_read_failed = cleaned is None and cleaned_err not in (None, "file_missing")
    base_missing = (cleaned is None and not cleaned_read_failed) or (not metrics_report and not weights)
    objective_type = (
        state.get("objective_type")
        or (contract.get("evaluation_spec") or {}).get("objective_type")
        or contract.get("objective_type")
        or ""
    )
    objective_key = str(objective_type or "unknown").lower()
    objective_map = {
        "classification": "classification",
        "classifier": "classification",
        "regression": "regression",
        "forecasting": "forecasting",
        "forecast": "forecasting",
        "ranking": "ranking",
        "prioritization": "ranking",
        "calibration": "ranking",
    }
    objective_family = objective_map.get(objective_key, "unknown")
    if objective_family == "unknown":
        inferred = _infer_objective_family(contract, metrics_report, weights)
        if inferred:
            objective_family = inferred

    if objective_family == "unknown":
        output_report = _safe_load_json("data/output_contract_report.json")
        missing_outputs = output_report.get("missing", []) if isinstance(output_report, dict) else []
        canonical_cols = contract.get("canonical_columns", []) if isinstance(contract, dict) else []
        row_count = int(cleaned.shape[0]) if cleaned is not None else None
        valid_row_fraction = None
        missingness_summary = {}
        if cleaned is not None and canonical_cols:
            cols = [c for c in canonical_cols if c in cleaned.columns]
            if cols:
                subset = cleaned[cols]
                valid_row_fraction = float(subset.notna().all(axis=1).mean()) if len(subset) else None
                missingness_summary = {
                    col: float(subset[col].isna().mean()) for col in cols if col in subset.columns
                }
        reasons: List[str] = []
        if cleaned is None:
            if cleaned_read_failed:
                reasons.append(f"cleaned_data_read_failed:{cleaned_err}")
            else:
                reasons.append("cleaned_data_missing")
        if missing_outputs:
            reasons.append("required_outputs_missing")
        if valid_row_fraction is not None and valid_row_fraction < 0.8:
            reasons.append("low_valid_row_fraction")
        if any(val > 0.5 for val in missingness_summary.values() if isinstance(val, float)):
            reasons.append("high_missingness")
        if any(reason.startswith("cleaned_data_read_failed") for reason in reasons):
            status = "unknown"
        else:
            status = "insufficient_signal" if reasons else "sufficient_signal"
        threshold = int(state.get("data_adequacy_threshold", 3) or 3)
        consecutive = int(state.get("data_adequacy_consecutive", 0) or 0)
        return {
            "status": status,
            "objective_type": objective_family,
            "reasons": reasons,
            "recommendations": [],
            "signals": {
                "row_count": row_count,
                "valid_row_fraction": valid_row_fraction,
                "missing_outputs_count": len(missing_outputs) if isinstance(missing_outputs, list) else None,
                "missingness_summary": missingness_summary,
            },
            "quality_gates_alignment": {},
            "consecutive_data_limited": consecutive,
            "data_limited_threshold": threshold,
            "threshold_reached": consecutive >= threshold,
        }

    metric_pool = _extract_metric_pool(weights, metrics_report)
    use_classification = objective_family == "classification"
    use_regression = objective_family in {"regression", "forecasting"}
    use_ranking = objective_family == "ranking"

    cls_preference = ["f1", "roc_auc", "auc", "pr_auc", "average_precision", "accuracy", "precision", "recall"]
    reg_preference = ["mae", "rmse", "mse", "mape", "smape", "r2"]
    rank_preference = ["spearman", "kendall", "ndcg", "map", "mrr", "gini"]

    cls_metric_name, cls_metric = (None, None)
    cls_baseline_name, cls_baseline = (None, None)
    reg_metric_name, reg_metric = (None, None)
    reg_baseline_name, reg_baseline = (None, None)
    rank_metric_name, rank_metric = (None, None)

    if use_classification:
        cls_metric_name, cls_metric = _select_metric(metric_pool, "classification", cls_preference, baseline_only=False)
        cls_baseline_name, cls_baseline = _select_metric(metric_pool, "classification", cls_preference, baseline_only=True)
    if use_regression:
        reg_metric_name, reg_metric = _select_metric(metric_pool, "regression", reg_preference, baseline_only=False)
        reg_baseline_name, reg_baseline = _select_metric(metric_pool, "regression", reg_preference, baseline_only=True)
    if use_ranking:
        rank_metric_name, rank_metric = _select_metric(metric_pool, "ranking", rank_preference, baseline_only=False)

    cls_higher = _metric_higher_is_better(cls_metric_name) if cls_metric_name else True
    reg_higher = _metric_higher_is_better(reg_metric_name) if reg_metric_name else False
    cls_lift = _calc_lift(cls_baseline, cls_metric, higher_is_better=cls_higher) if use_classification else None
    reg_lift = _calc_lift(reg_baseline, reg_metric, higher_is_better=reg_higher) if use_regression else None

    f1 = cls_metric if cls_metric_name and "f1" in _normalize_key(cls_metric_name) else None
    f1_baseline = cls_baseline if cls_baseline_name and "f1" in _normalize_key(cls_baseline_name) else None
    mae = reg_metric if reg_metric_name and "mae" in _normalize_key(reg_metric_name) else None
    mae_baseline = reg_baseline if reg_baseline_name and "mae" in _normalize_key(reg_baseline_name) else None

    f1_lift = _calc_lift(f1_baseline, f1, higher_is_better=True) if use_classification else None
    mae_lift = _calc_lift(mae_baseline, mae, higher_is_better=False) if use_regression else None

    row_count = int(cleaned.shape[0]) if cleaned is not None else None
    feature_count = None
    if isinstance(weights, dict):
        feat = weights.get("feature_importance") or weights.get("feature_importances")
        if isinstance(feat, dict):
            feature_count = len(feat)

    rows_per_feature = None
    if row_count and feature_count:
        rows_per_feature = row_count / max(1, feature_count)

    target_col = _find_target_column(contract, cleaned)
    class_balance = None
    if use_classification and cleaned is not None and target_col and target_col in cleaned.columns:
        try:
            class_balance = float(cleaned[target_col].mean())
        except Exception:
            class_balance = None

    feature_cols: List[str] = []
    if isinstance(contract, dict):
        # V4.1: Extract features from column_roles
        roles = get_column_roles(contract)
        feature_role_names = {"feature", "pre_decision", "predictor", "driver"}
        for role, cols in roles.items():
            if role.lower() in feature_role_names or "feature" in role.lower():
                feature_cols.extend(cols)
    if cleaned is not None and (not feature_cols):
        if target_col:
            feature_cols = [col for col in cleaned.columns if col != target_col]
        else:
            feature_cols = list(cleaned.columns)

    target_series = cleaned[target_col] if cleaned is not None and target_col in cleaned.columns else None
    target_kind = _infer_target_kind(target_series) if (target_series is not None and (use_classification or use_regression)) else "unknown"
    auc_proxy = {"best_auc": None, "best_feature": None}
    reg_proxy = {"best_abs_spearman": None, "best_feature": None, "best_r2_proxy": None}
    if cleaned is not None and target_series is not None and (use_classification or use_regression):
        if target_kind == "binary":
            auc_proxy = _best_auc_proxy(cleaned, target_series, feature_cols)
        else:
            reg_proxy = _best_regression_proxy(cleaned, target_series, feature_cols)

    quality_gates = contract.get("quality_gates", {}) if isinstance(contract, dict) else {}
    if not isinstance(quality_gates, dict):
        quality_gates = {}
    if not quality_gates and isinstance(contract, dict):
        raw_gates = contract.get("quality_gates_raw")
        if isinstance(raw_gates, list):
            for item in raw_gates:
                if isinstance(item, dict) and item.get("metric") is not None and item.get("threshold") is not None:
                    quality_gates[str(item["metric"])] = item["threshold"]
    min_segment_size = quality_gates.get("min_segment_size")
    small_segment_frac, small_segment_count = _segment_coverage(case_summary, min_segment_size)
    gate_alignment = _align_quality_gates(quality_gates, metric_pool)

    if base_missing:
        metric_pool = {}
        cls_metric_name = cls_metric = cls_baseline_name = cls_baseline = None
        reg_metric_name = reg_metric = reg_baseline_name = reg_baseline = None
        rank_metric_name = rank_metric = None
        cls_lift = reg_lift = None
        f1 = f1_baseline = f1_lift = None
        mae = mae_baseline = mae_lift = None
        row_count = None
        feature_count = None
        rows_per_feature = None
        class_balance = None
        target_kind = None
        small_segment_frac = None
        small_segment_count = None
        auc_proxy = {"best_auc": None, "best_feature": None}
        reg_proxy = {"best_abs_spearman": None, "best_feature": None, "best_r2_proxy": None}

    reasons: List[str] = []
    if cleaned is None:
        if cleaned_read_failed:
            reasons.append(f"cleaned_data_read_failed:{cleaned_err}")
        else:
            reasons.append("cleaned_data_missing")
    if base_missing:
        reasons.append("pipeline_aborted_before_metrics")
    signals: Dict[str, Any] = {
        "objective_type": objective_family,
        "row_count": row_count,
        "feature_count": feature_count,
        "rows_per_feature": rows_per_feature,
        "class_balance": class_balance,
        "small_segment_fraction": small_segment_frac,
        "small_segment_count": small_segment_count,
        "target_kind": target_kind,
        "signal_ceiling_auc_proxy": auc_proxy.get("best_auc"),
        "signal_ceiling_auc_feature": auc_proxy.get("best_feature"),
        "signal_ceiling_abs_spearman": reg_proxy.get("best_abs_spearman"),
        "signal_ceiling_r2_proxy": reg_proxy.get("best_r2_proxy"),
        "signal_ceiling_feature": reg_proxy.get("best_feature"),
        "f1_score_cv_mean": f1,
        "baseline_f1": f1_baseline,
        "f1_lift": f1_lift,
        "mae_cv_mean": mae,
        "baseline_mae": mae_baseline,
        "mae_lift": mae_lift,
        "classification_metric_name": cls_metric_name,
        "classification_metric": cls_metric,
        "classification_baseline_name": cls_baseline_name,
        "classification_baseline": cls_baseline,
        "classification_lift": cls_lift,
        "regression_metric_name": reg_metric_name,
        "regression_metric": reg_metric,
        "regression_baseline_name": reg_baseline_name,
        "regression_baseline": reg_baseline,
        "regression_lift": reg_lift,
        "ranking_metric_name": rank_metric_name,
        "ranking_metric": rank_metric,
        "available_metrics": sorted(metric_pool.keys()),
    }

    if use_classification and not base_missing and cls_metric is None:
        reasons.append("classification_metric_missing")
    if use_regression and not base_missing and reg_metric is None:
        reasons.append("regression_metric_missing")
    if use_ranking and not base_missing and rank_metric is None:
        reasons.append("ranking_metric_missing")
    if use_classification and not base_missing and cls_metric is not None and cls_baseline is None:
        reasons.append("classification_baseline_missing")
    if use_regression and not base_missing and reg_metric is not None and reg_baseline is None:
        reasons.append("regression_baseline_missing")

    if use_classification and not base_missing and cls_lift is not None and cls_lift < 0.05:
        reasons.append("classification_lift_low")
    if use_regression and not base_missing and reg_lift is not None and reg_lift < 0.1:
        reasons.append("regression_lift_low")
    if not base_missing and rows_per_feature is not None and rows_per_feature < 10:
        reasons.append("high_dimensionality_low_sample")
    if use_classification and not base_missing and class_balance is not None and (class_balance < 0.1 or class_balance > 0.9):
        reasons.append("class_imbalance")
    if not base_missing and small_segment_frac is not None and small_segment_frac > 0.3:
        reasons.append("segments_too_small")

    if use_classification and auc_proxy.get("best_auc") is not None:
        if auc_proxy["best_auc"] < 0.6:
            reasons.append("signal_ceiling_low")
        if cls_metric is not None and cls_metric_name and "auc" in _normalize_key(cls_metric_name):
            if auc_proxy["best_auc"] is not None and cls_metric >= 0.95 * auc_proxy["best_auc"]:
                reasons.append("signal_ceiling_reached")
    if use_regression and reg_proxy.get("best_r2_proxy") is not None:
        if reg_proxy["best_r2_proxy"] < 0.05:
            reasons.append("signal_ceiling_low")
        if reg_metric_name and "r2" in _normalize_key(reg_metric_name):
            if reg_metric is not None and reg_metric >= 0.95 * reg_proxy["best_r2_proxy"]:
                reasons.append("signal_ceiling_reached")

    data_limited = (
        len([r for r in reasons if not r.endswith("_missing")]) >= 2
        or "signal_ceiling_reached" in reasons
        or "signal_ceiling_low" in reasons
        or (
            (cls_lift is not None and cls_lift < 0.02)
            and (reg_lift is not None and reg_lift < 0.05)
        )
    )

    recommendations: List[str] = []
    if "classification_lift_low" in reasons:
        recommendations.append("Collect more labeled outcomes or refine the success label definition.")
    if "regression_lift_low" in reasons:
        recommendations.append("Increase the number of successful contracts with reliable 1stYearAmount values.")
    if "high_dimensionality_low_sample" in reasons:
        recommendations.append("Increase sample size or reduce feature dimensionality through aggregation.")
    if "class_imbalance" in reasons:
        recommendations.append("Improve class balance by collecting more rare outcomes or sampling evenly.")
    if "segments_too_small" in reasons:
        recommendations.append("Aggregate segments or collect more cases per segment before recommending prices.")
    if "signal_ceiling_low" in reasons:
        recommendations.append("Increase feature richness or improve data capture to raise the achievable signal ceiling.")
    if "signal_ceiling_reached" in reasons:
        recommendations.append("Current performance is near the data signal ceiling; improvements likely require better data, not tuning.")
    if "classification_metric_missing" in reasons or "regression_metric_missing" in reasons or "ranking_metric_missing" in reasons:
        recommendations.append("Persist model performance metrics alongside weights.json for data adequacy checks.")
    if "classification_baseline_missing" in reasons or "regression_baseline_missing" in reasons:
        recommendations.append("Include baseline metrics (dummy/naive) to quantify lift over trivial models.")

    if cleaned_read_failed:
        status = "unknown"
    elif base_missing:
        status = "insufficient_signal"
    else:
        status = "data_limited" if data_limited else "sufficient_signal"
        if use_classification and cls_metric is None:
            status = "insufficient_signal"
        if use_regression and reg_metric is None:
            status = "insufficient_signal"
        if use_ranking and rank_metric is None:
            status = "insufficient_signal"
    threshold = int(state.get("data_adequacy_threshold", 3) or 3)
    consecutive = int(state.get("data_adequacy_consecutive", 0) or 0)

    return {
        "status": status,
        "reasons": reasons,
        "recommendations": recommendations,
        "signals": signals,
        "quality_gates_alignment": gate_alignment,
        "consecutive_data_limited": consecutive,
        "data_limited_threshold": threshold,
        "threshold_reached": consecutive >= threshold,
    }


def write_data_adequacy_report(state: Dict[str, Any], path: str = "data/data_adequacy_report.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = build_data_adequacy_report(state)
    try:
        dump_json(path, report)
    except Exception:
        pass
