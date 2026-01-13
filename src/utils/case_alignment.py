import csv
import json
import os
import re
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.contract_v41 import get_outcome_columns


def _safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_output_dialect(manifest_path: str = "data/cleaning_manifest.json") -> Dict[str, str]:
    defaults = {"sep": ",", "decimal": ".", "encoding": "utf-8"}
    manifest = _safe_load_json(manifest_path) or {}
    if isinstance(manifest, dict):
        output_dialect = manifest.get("output_dialect") or {}
        if isinstance(output_dialect, dict):
            return {
                "sep": output_dialect.get("sep", defaults["sep"]),
                "decimal": output_dialect.get("decimal", defaults["decimal"]),
                "encoding": output_dialect.get("encoding", defaults["encoding"]),
            }
    return defaults


def _read_text_sample(path: str, encoding: str, max_bytes: int = 50000) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding=encoding, errors="replace") as handle:
            return handle.read(max_bytes)
    except Exception:
        return ""


def _infer_delimiter_from_file(path: str, encoding: str) -> Optional[str]:
    sample = _read_text_sample(path, encoding)
    if not sample:
        return None
    delimiters = [",", ";", "\t", "|"]
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=delimiters)
        if getattr(sniffed, "delimiter", None):
            return sniffed.delimiter
    except Exception:
        pass
    counts = {delim: sample.count(delim) for delim in delimiters}
    best = max(counts, key=counts.get)
    return best if counts.get(best, 0) > 0 else None


def _infer_decimal_from_sample(sample: str) -> str:
    if not sample:
        return "."
    comma_hits = len(re.findall(r"\d+,\d+", sample))
    dot_hits = len(re.findall(r"\d+\.\d+", sample))
    if comma_hits > dot_hits:
        return ","
    if dot_hits > comma_hits:
        return "."
    return "."


def _looks_like_delimiter_mismatch(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty:
        return False
    if df.shape[1] != 1:
        return False
    colname = str(df.columns[0])
    if len(colname) <= 5:
        return False
    return any(token in colname for token in [",", ";", "\t", "|"])


def _read_csv_with_dialect(path: str, dialect: Dict[str, str]) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
    if not path or not os.path.exists(path):
        return None, dialect
    try:
        df = pd.read_csv(
            path,
            sep=dialect.get("sep", ","),
            decimal=dialect.get("decimal", "."),
            encoding=dialect.get("encoding", "utf-8"),
        )
    except Exception:
        df = None
    if _looks_like_delimiter_mismatch(df):
        encoding = dialect.get("encoding", "utf-8")
        inferred_sep = _infer_delimiter_from_file(path, encoding)
        if inferred_sep and inferred_sep != dialect.get("sep"):
            sample = _read_text_sample(path, encoding)
            inferred_decimal = _infer_decimal_from_sample(sample)
            dialect = {"sep": inferred_sep, "decimal": inferred_decimal, "encoding": encoding}
            try:
                df = pd.read_csv(
                    path,
                    sep=dialect.get("sep", ","),
                    decimal=dialect.get("decimal", "."),
                    encoding=dialect.get("encoding", "utf-8"),
                )
            except Exception:
                pass
    return df, dialect


def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _extract_weights(obj: Any) -> Dict[str, float]:
    if isinstance(obj, dict):
        if isinstance(obj.get("weights"), dict):
            return {str(k): float(v) for k, v in obj["weights"].items() if _is_number(v)}
        if isinstance(obj.get("feature_weights"), dict):
            return {str(k): float(v) for k, v in obj["feature_weights"].items() if _is_number(v)}
        if isinstance(obj.get("weights"), list) and isinstance(obj.get("features"), list):
            weights = obj.get("weights") or []
            features = obj.get("features") or []
            if len(weights) == len(features):
                out = {}
                for idx, feat in enumerate(features):
                    val = weights[idx]
                    if _is_number(val):
                        out[str(feat)] = float(val)
                if out:
                    return out
        if all(_is_number(v) for v in obj.values()):
            return {str(k): float(v) for k, v in obj.items()}
    return {}


def _is_number(val: Any) -> bool:
    try:
        float(val)
        return True
    except Exception:
        return False


def _pick_column(df: pd.DataFrame, target_name: Optional[str], kind: str) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    if target_name:
        target_norm = _normalize(target_name)
        for col in cols:
            if _normalize(col) == target_norm:
                return col
    # fallback heuristics
    if kind == "target":
        for col in cols:
            n = _normalize(col)
            if "refscore" in n or "ref_score" in n:
                return col
        for col in cols:
            if "target" in _normalize(col):
                return col
    if kind == "score_mean":
        for col in cols:
            n = _normalize(col)
            if "score" in n and ("mean" in n or "avg" in n):
                return col
        for col in cols:
            if "score" in _normalize(col):
                return col
    return None


def _pick_case_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in df.columns:
        norm = _normalize(col)
        if norm in {"case", "caseid", "case_id", "caso", "casoid"}:
            return col
        if norm.startswith("case") or "case" in norm or "caso" in norm:
            return col
    return None


def _pick_group_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in df.columns:
        norm = _normalize(col)
        if any(tok in norm for tok in ["case", "group", "segment", "bucket", "cluster"]):
            return col
    return None


def _pick_score_column(df: pd.DataFrame, exclude: Optional[str] = None) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in df.columns:
        norm = _normalize(col)
        if exclude and _normalize(exclude) == norm:
            continue
        if "ref" in norm or "target" in norm:
            continue
        if any(tok in norm for tok in ["score", "pred", "prob", "rank"]):
            return col
    return None


def _compute_adjacent_violations(ref: pd.Series, score: pd.Series) -> int:
    df = pd.DataFrame({"ref": ref, "score": score}).dropna()
    if df.empty:
        return 0
    df = df.sort_values("ref")
    refs = df["ref"].values
    scores = df["score"].values
    violations = 0
    for idx in range(len(scores) - 1):
        if refs[idx + 1] > refs[idx] and scores[idx + 1] < scores[idx]:
            violations += 1
    return int(violations)


def _compute_inactive_top_decile_share(ref: pd.Series, score: pd.Series) -> Dict[str, Optional[float]]:
    df = pd.DataFrame({"ref": ref, "score": score}).dropna()
    if df.empty:
        return {"in_top": None, "within_inactive": None}
    ref_q = df["ref"].quantile(0.05)
    score_q = df["score"].quantile(0.9)
    inactive = df["ref"] <= ref_q
    if inactive.sum() == 0:
        return {"in_top": None, "within_inactive": None}
    top_decile = df["score"] >= score_q
    in_top_share = None
    if top_decile.sum() > 0:
        in_top_share = float((inactive & top_decile).sum() / top_decile.sum())
    within_inactive_share = float((inactive & top_decile).sum() / inactive.sum())
    return {"in_top": in_top_share, "within_inactive": within_inactive_share}


def _compute_weight_metrics(weights: Dict[str, float]) -> Dict[str, Any]:
    if not weights:
        return {"max_weight": None, "hhi": None, "near_zero_weights_count": None}
    vals = np.array(list(weights.values()), dtype=float)
    max_weight = float(np.max(vals))
    hhi = float(np.sum(vals ** 2))
    near_zero = int(np.sum(vals < 0.01))
    return {
        "max_weight": max_weight,
        "hhi": hhi,
        "near_zero_weights_count": near_zero,
    }


def _weight_key_variants(key: str) -> Tuple[str, ...]:
    raw = str(key)
    norm = _normalize(raw)
    variants = {norm}
    if norm:
        variants.add(re.sub(r"^w\\d+", "", norm))
        variants.add(re.sub(r"^weight\\d*", "", norm))
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", raw.lower()) if p]
    if parts:
        variants.add(_normalize(parts[-1]))
    cleaned = tuple(v for v in variants if v)
    return cleaned


def _match_weight_for_column(
    col_norm: str,
    weights: Dict[str, float],
    weight_variants: Dict[str, Tuple[str, ...]],
) -> Optional[float]:
    best_key = None
    best_score = -1
    for key, variants in weight_variants.items():
        for variant in variants:
            if not variant:
                continue
            if col_norm == variant:
                score = 2 * len(variant)
            elif variant in col_norm or col_norm in variant:
                score = len(variant)
            else:
                continue
            if score > best_score:
                best_score = score
                best_key = key
    if best_key is None:
        return None
    try:
        return float(weights.get(best_key, 0.0))
    except Exception:
        return None


def build_case_alignment_report(
    contract: Dict[str, Any],
    case_summary_path: str = "data/case_summary.csv",
    weights_path: str = "data/weights.json",
    data_paths: Optional[list[str]] = None,
    scored_rows_path: str = "data/scored_rows.csv",
) -> Dict[str, Any]:
    data_paths = data_paths or []
    contract = contract or {}
    gates = contract.get("quality_gates") or {}
    if not isinstance(gates, dict):
        gates = {}

    defaults = {
        "spearman_min": None,
        "kendall_min": None,
        "violations_max": None,
        "inactive_share_max": None,
        "max_weight_max": None,
        "hhi_max": None,
        "near_zero_max": None,
    }
    thresholds = {**defaults, **{k: v for k, v in gates.items() if v is not None}}

    # V4.1: Identify target name from outcome columns
    target_name = None
    outcome_cols = get_outcome_columns(contract)
    if outcome_cols:
        target_name = outcome_cols[0]

    weights_obj = _safe_load_json(weights_path) or {}
    weights = _extract_weights(weights_obj)

    mode = "case_level"
    ref_series = None
    score_series = None

    dialect = _load_output_dialect()

    # Try group alignment from scored_rows if available (prefer explicit case_summary paths)
    default_case_summary = case_summary_path in {"data/case_summary.csv", os.path.join("data", "case_summary.csv")}
    data_paths_set = set(data_paths or [])
    scored_in_paths = scored_rows_path in data_paths_set
    use_scored_rows = (
        os.path.exists(scored_rows_path)
        and (scored_in_paths or not data_paths)
        and (not case_summary_path or not os.path.exists(case_summary_path) or default_case_summary)
    )
    if use_scored_rows:
        sr, _ = _read_csv_with_dialect(scored_rows_path, dialect)
        if sr is not None and not sr.empty:
            group_col = _pick_group_column(sr)
            ref_col = _pick_column(sr, target_name, "target")
            score_col = _pick_score_column(sr, exclude=ref_col)
            if group_col and score_col:
                group_df = sr[[group_col, score_col] + ([ref_col] if ref_col else [])].copy()
                group_df[score_col] = pd.to_numeric(group_df[score_col], errors="coerce")
                if ref_col:
                    group_df[ref_col] = pd.to_numeric(group_df[ref_col], errors="coerce")
                grouped = group_df.groupby(group_col)
                summary = grouped.agg(
                    n=(score_col, "size"),
                    score_mean=(score_col, "mean"),
                    score_median=(score_col, "median"),
                    ref_mean=(ref_col, "mean") if ref_col else (score_col, "mean"),
                ).reset_index()
                ref_series = pd.to_numeric(summary["ref_mean"], errors="coerce")
                score_series = pd.to_numeric(summary["score_mean"], errors="coerce")
                spearman = None
                kendall = None
                try:
                    if ref_col:
                        spearman = float(ref_series.corr(score_series, method="spearman"))
                        kendall = float(ref_series.corr(score_series, method="kendall"))
                except Exception:
                    pass
                adjacent_violations = _compute_adjacent_violations(ref_series, score_series) if ref_col else 0
                metrics = {
                    "spearman_case_means": spearman,
                    "kendall_case_means": kendall,
                    "adjacent_refscore_violations": int(adjacent_violations),
                    "case_count": int(summary.shape[0]),
                }
                failures = []
                if metrics["case_count"] < 2:
                    failures.append("insufficient_case_variation")
                if spearman is not None and thresholds["spearman_min"] is not None and spearman < thresholds["spearman_min"]:
                    failures.append("spearman_case_means")
                if kendall is not None and thresholds["kendall_min"] is not None and kendall < thresholds["kendall_min"]:
                    failures.append("kendall_case_means")
                if thresholds["violations_max"] is not None and adjacent_violations > thresholds["violations_max"]:
                    failures.append("adjacent_refscore_violations")
                if not gates:
                    status = "PASS" if not failures else "FAIL"
                    explanation = "Group alignment computed from scored_rows; no explicit gates provided."
                else:
                    status = "PASS" if not failures else "FAIL"
                    explanation = "Group alignment gate passed." if status == "PASS" else f"Failed gates: {', '.join(failures)}"
                return {
                    "status": status,
                    "mode": "group_from_scored_rows",
                    "metrics": metrics,
                    "thresholds": thresholds,
                    "failures": failures,
                    "explanation": explanation,
                    "group_summary": summary.to_dict(orient="records"),
                    "source": "computed_from_scored_rows",
                }

    # Try row_level if data available and weights exist
    if weights and data_paths:
        weight_variants = {key: _weight_key_variants(key) for key in weights.keys()}
        for path in data_paths:
            if os.path.exists(path):
                df, _ = _read_csv_with_dialect(path, dialect)
                if df is None:
                    continue
                target_col = _pick_column(df, target_name, "target")
                if not target_col:
                    continue
                feature_cols = []
                w_vec = []
                for col in df.columns:
                    if col == target_col:
                        continue
                    col_norm = _normalize(col)
                    w_val = _match_weight_for_column(col_norm, weights, weight_variants)
                    if w_val is None:
                        continue
                    feature_cols.append(col)
                    w_vec.append(w_val)
                if not feature_cols:
                    continue
                X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
                score = X.mul(w_vec, axis=1).sum(axis=1)
                ref_series = pd.to_numeric(df[target_col], errors="coerce")
                score_series = score
                mode = "row_level"
                break

    case_count = None
    if ref_series is None or score_series is None:
        if os.path.exists(case_summary_path):
            try:
                cs, _ = _read_csv_with_dialect(case_summary_path, dialect)
                if cs is None:
                    cs = pd.DataFrame()
                if not cs.empty:
                    case_col = _pick_case_column(cs)
                    if case_col:
                        case_count = int(cs[case_col].nunique(dropna=True))
                    else:
                        case_count = int(len(cs))
                ref_col = _pick_column(cs, target_name, "target")
                score_col = _pick_column(cs, None, "score_mean")
                if ref_col and score_col:
                    ref_series = pd.to_numeric(cs[ref_col], errors="coerce")
                    score_series = pd.to_numeric(cs[score_col], errors="coerce")
            except Exception:
                pass

    if ref_series is None or score_series is None:
        ref_series = pd.Series(dtype=float)
        score_series = pd.Series(dtype=float)

    # Metrics
    spearman = None
    kendall = None
    try:
        if len(ref_series.dropna()) >= 2 and len(score_series.dropna()) >= 2:
            spearman = float(ref_series.corr(score_series, method="spearman"))
            kendall = float(ref_series.corr(score_series, method="kendall"))
    except Exception:
        pass

    adjacent_violations = _compute_adjacent_violations(ref_series, score_series)
    inactive_shares = _compute_inactive_top_decile_share(ref_series, score_series)
    weight_metrics = _compute_weight_metrics(weights)

    metrics = {
        "spearman_case_means": spearman,
        "kendall_case_means": kendall,
        "adjacent_refscore_violations": adjacent_violations,
        "inactive_top_decile_share": inactive_shares.get("in_top"),
        "inactive_share_within_inactive": inactive_shares.get("within_inactive"),
        "weight_concentration_max": weight_metrics.get("max_weight"),
        "weight_concentration_hhi": weight_metrics.get("hhi"),
        "near_zero_weights_count": weight_metrics.get("near_zero_weights_count"),
        "case_count": case_count,
    }

    failures = []
    if case_count is not None and case_count < 2:
        failures.append("insufficient_case_variation")
    if case_count is None and (spearman is None or kendall is None):
        failures.append("insufficient_case_variation")
    if spearman is not None and thresholds["spearman_min"] is not None and spearman < thresholds["spearman_min"]:
        failures.append("spearman_case_means")
    if kendall is not None and thresholds["kendall_min"] is not None and kendall < thresholds["kendall_min"]:
        failures.append("kendall_case_means")
    if thresholds["violations_max"] is not None and adjacent_violations > thresholds["violations_max"]:
        failures.append("adjacent_refscore_violations")
    inactive_share = metrics.get("inactive_top_decile_share")
    if inactive_share is not None and thresholds["inactive_share_max"] is not None and inactive_share > thresholds["inactive_share_max"]:
        failures.append("inactive_top_decile_share")
    if weight_metrics.get("max_weight") is not None and thresholds["max_weight_max"] is not None:
        if weight_metrics["max_weight"] > thresholds["max_weight_max"]:
            failures.append("max_weight")
    if weight_metrics.get("hhi") is not None and thresholds["hhi_max"] is not None:
        if weight_metrics["hhi"] > thresholds["hhi_max"]:
            failures.append("hhi")
    if weight_metrics.get("near_zero_weights_count") is not None and thresholds["near_zero_max"] is not None:
        if weight_metrics["near_zero_weights_count"] > thresholds["near_zero_max"]:
            failures.append("near_zero_weights_count")

    if not gates:
        status = "SKIPPED"
        explanation = "Case alignment gates not defined in contract; skipping gate evaluation."
    else:
        status = "PASS" if not failures else "FAIL"
        explanation = "Case alignment gate passed." if status == "PASS" else f"Failed gates: {', '.join(failures)}"

    return {
        "status": status,
        "mode": mode,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
        "explanation": explanation,
    }
