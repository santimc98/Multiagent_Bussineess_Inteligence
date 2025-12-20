import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats


def _ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _json_default(o):
    import numpy as _np
    import pandas as _pd
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, (_pd.Timestamp,)):
        return o.isoformat()
    if hasattr(_pd, "Timedelta") and isinstance(o, _pd.Timedelta):
        return o.total_seconds()
    if isinstance(o, (set, tuple)):
        return list(o)
    if o is None:
        return None
    try:
        if isinstance(o, float) and (o != o):
            return None
    except Exception:
        pass
    try:
        if getattr(_pd, "isna", None) and _pd.isna(o):
            return None
    except Exception:
        pass
    return str(o)


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _load_manifest_dialect(manifest_path: str, fallback: Dict[str, str]) -> Dict[str, str]:
    if not manifest_path or not os.path.exists(manifest_path):
        return dict(fallback)
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        out = manifest.get("output_dialect") or {}
        dialect = dict(fallback)
        dialect.update({k: v for k, v in out.items() if v})
        return dialect
    except Exception:
        return dict(fallback)


def _resolve_feature_columns(plan: Dict[str, Any], contract: Dict[str, Any]) -> List[str]:
    features_cfg = plan.get("features") or {}
    plan_cols = features_cfg.get("columns")
    use_contract = features_cfg.get("use_contract_features", True)
    if isinstance(plan_cols, list) and plan_cols:
        return [str(c) for c in plan_cols]
    if not use_contract:
        return []
    reqs = contract.get("data_requirements", []) or []
    cols = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        if req.get("role") != "feature":
            continue
        name = req.get("canonical_name") or req.get("name")
        if name:
            cols.append(name)
    return cols


def _resolve_target(plan: Dict[str, Any], contract: Dict[str, Any]) -> Tuple[str | None, str | None]:
    target_cfg = plan.get("target") or {}
    if target_cfg.get("name"):
        return str(target_cfg.get("name")), str(target_cfg.get("type") or "")
    reqs = contract.get("data_requirements", []) or []
    target_name = None
    target_type = ""
    for role in ("target", "target_benchmark", "baseline_metric"):
        for req in reqs:
            if not isinstance(req, dict):
                continue
            if req.get("role") != role:
                continue
            target_name = req.get("canonical_name") or req.get("name")
            target_type = str(contract.get("spec_extraction", {}).get("target_type", "")) or ""
            if target_name:
                return target_name, target_type
    return None, target_type


def _resolve_baseline_column(contract: Dict[str, Any]) -> str | None:
    reqs = contract.get("data_requirements", []) or []
    for role in ("baseline_metric", "target_benchmark"):
        for req in reqs:
            if not isinstance(req, dict):
                continue
            if req.get("role") != role:
                continue
            return req.get("canonical_name") or req.get("name")
    return None


def _resolve_id_columns(contract: Dict[str, Any]) -> List[str]:
    reqs = contract.get("data_requirements", []) or []
    ids = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        if req.get("role") != "id":
            continue
        name = req.get("canonical_name") or req.get("name")
        if name:
            ids.append(name)
    return ids


def _map_columns(df: pd.DataFrame, columns: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    norm_actual = {_norm_name(col): col for col in df.columns}
    for col in columns:
        if col in df.columns:
            mapping[col] = col
            continue
        norm = _norm_name(col)
        if norm in norm_actual:
            mapping[col] = norm_actual[norm]
    return mapping


def _drop_low_non_null(df: pd.DataFrame, cols: List[str], min_non_null: int) -> Tuple[List[str], List[str]]:
    kept = []
    dropped = []
    for col in cols:
        if col not in df.columns:
            continue
        non_null = df[col].notna().sum()
        if non_null < min_non_null:
            dropped.append(col)
        else:
            kept.append(col)
    return kept, dropped


def _score_from_weights(X: np.ndarray, weights: np.ndarray, scale: float) -> np.ndarray:
    return scale * np.dot(X, weights)


def _objective_factory(X: np.ndarray, y: np.ndarray, objective: str, l2: float, scale: float):
    def _mse(weights):
        pred = _score_from_weights(X, weights, scale)
        mse = np.nanmean((pred - y) ** 2)
        return mse + l2 * np.sum(weights ** 2)

    def _neg_spearman(weights):
        pred = _score_from_weights(X, weights, scale)
        if np.all(np.isnan(pred)) or np.all(pred == pred[0]):
            return 1e6
        corr = stats.spearmanr(pred, y, nan_policy="omit").correlation
        if corr is None or np.isnan(corr):
            return 1e6
        return -corr + l2 * np.sum(weights ** 2)

    if objective == "maximize_spearman":
        return _neg_spearman
    return _mse


def _init_weights(X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    n = X.shape[1]
    if n == 0:
        return []
    uniform = np.ones(n) / n
    corrs = []
    for i in range(n):
        col = X[:, i]
        if np.all(np.isnan(col)) or np.all(col == col[0]):
            corrs.append(0.0)
            continue
        corr = stats.spearmanr(col, y, nan_policy="omit").correlation
        corrs.append(0.0 if corr is None or np.isnan(corr) else max(0.0, corr))
    corr_arr = np.array(corrs, dtype=float)
    if corr_arr.sum() <= 0:
        corr_arr = uniform.copy()
    else:
        corr_arr = corr_arr / corr_arr.sum()
    return [uniform, corr_arr]


def _metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["spearman"] = float(stats.spearmanr(y_pred, y_true, nan_policy="omit").correlation)
    except Exception:
        out["spearman"] = None
    try:
        out["kendall"] = float(stats.kendalltau(y_pred, y_true, nan_policy="omit").correlation)
    except Exception:
        out["kendall"] = None
    out["mse"] = float(np.nanmean((y_pred - y_true) ** 2))
    out["mae"] = float(np.nanmean(np.abs(y_pred - y_true)))
    return out


def _case_summary(df: pd.DataFrame, case_col: str, score_col: str, baseline_col: str | None) -> pd.DataFrame:
    if case_col not in df.columns:
        return pd.DataFrame()
    group = df.groupby(case_col, dropna=False)
    summary = group[score_col].agg(["count", "mean", "median"])
    percentiles = group[score_col].quantile([0.1, 0.5, 0.9]).unstack(level=1)
    percentiles.columns = [f"p{int(p*100)}" for p in percentiles.columns]
    summary = summary.join(percentiles, how="left")
    if baseline_col and baseline_col in df.columns:
        baseline_stats = group[baseline_col].mean().rename("baseline_mean")
        summary = summary.join(baseline_stats, how="left")
    return summary.reset_index()


def execute_ml_plan(plan: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, Any]:
    input_cfg = plan.get("input") or {}
    outputs_cfg = plan.get("outputs") or {}
    manifest_path = input_cfg.get("manifest_path", "data/cleaning_manifest.json")
    cleaned_path = input_cfg.get("cleaned_path", "data/cleaned_data.csv")
    dialect = plan.get("dialect") or {}
    dialect = _load_manifest_dialect(
        manifest_path,
        {
            "sep": dialect.get("sep", ","),
            "decimal": dialect.get("decimal", "."),
            "encoding": dialect.get("encoding", "utf-8"),
        },
    )

    df = pd.read_csv(
        cleaned_path,
        sep=dialect.get("sep", ","),
        decimal=dialect.get("decimal", "."),
        encoding=dialect.get("encoding", "utf-8"),
    )
    if df.empty:
        raise ValueError(f"cleaned dataset empty (sep={dialect.get('sep')}, decimal={dialect.get('decimal')}, encoding={dialect.get('encoding')})")

    require_derived = bool(plan.get("require_derived", True))
    if require_derived:
        derived_reqs = [
            (req.get("canonical_name") or req.get("name"))
            for req in (contract.get("data_requirements", []) or [])
            if isinstance(req, dict) and req.get("source") == "derived"
        ]
        derived_missing = [col for col in derived_reqs if col and col not in df.columns]
        if derived_missing:
            raise ValueError(f"missing derived columns in cleaned data: {derived_missing}")

    feature_cols = _resolve_feature_columns(plan, contract)
    target_name, target_type = _resolve_target(plan, contract)
    if not target_name:
        raise ValueError("target column not resolved from plan or contract")
    feature_cols = [c for c in feature_cols if c != target_name]

    # Map columns by normalized name
    mapping = _map_columns(df, feature_cols + [target_name])
    missing = [col for col in feature_cols + [target_name] if col not in mapping]
    if missing:
        raise ValueError(f"missing required columns after mapping: {missing}")
    if len(set(mapping.values())) != len(mapping.values()):
        raise ValueError("column aliasing detected in feature/target mapping")

    mapped_features = [mapping[col] for col in feature_cols]
    mapped_target = mapping[target_name]
    df = df.rename(columns={v: k for k, v in mapping.items()})

    optional_cols = []
    baseline_col = _resolve_baseline_column(contract)
    if baseline_col:
        optional_cols.append(baseline_col)
    id_cols = _resolve_id_columns(contract)
    optional_cols.extend(id_cols)
    case_col = None
    for req in contract.get("data_requirements", []) or []:
        if isinstance(req, dict) and req.get("canonical_name") and req.get("role") == "metadata":
            if req.get("canonical_name") in {"caso", "case_id", "case"}:
                case_col = req.get("canonical_name")
                break
    case_col = case_col or ("caso" if "caso" in df.columns else None)
    if case_col:
        optional_cols.append(case_col)

    optional_map = _map_columns(df, [c for c in optional_cols if c])
    optional_rename = {}
    for col, actual in optional_map.items():
        if col not in df.columns and actual in df.columns:
            optional_rename[actual] = col
    if optional_rename:
        df = df.rename(columns=optional_rename)

    min_non_null = int((plan.get("features") or {}).get("min_non_null", 20))
    kept_features, dropped_features = _drop_low_non_null(df, feature_cols, min_non_null)
    if not kept_features:
        raise ValueError("no usable features after missingness filter")

    # Convert to numeric
    for col in kept_features + [target_name]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    y = df[target_name]
    if y.nunique(dropna=True) <= 1:
        raise ValueError("Target has no variance; cannot optimize weights.")

    X = df[kept_features].to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y_values)
    if not mask.any():
        raise ValueError("no finite rows after numeric conversion")
    X = X[mask]
    y_values = y_values[mask]

    score_scale = 1.0
    scoring_formula = str((contract.get("spec_extraction") or {}).get("scoring_formula", ""))
    if "100" in scoring_formula and "*" in scoring_formula:
        score_scale = 100.0
    if isinstance(plan.get("score_scale"), (int, float)):
        score_scale = float(plan.get("score_scale"))

    experiments = plan.get("experiments") or []
    selection = plan.get("selection") or {}
    selection_metric = selection.get("metric")
    if not selection_metric:
        selection_metric = "spearman" if target_type == "ranking" else "mse"
    selection_dir = selection.get("direction", "max" if selection_metric in {"spearman", "kendall"} else "min")

    results: List[Dict[str, Any]] = []
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        method = exp.get("method", "optimize_weights")
        if method != "optimize_weights":
            continue
        objective = exp.get("objective", "maximize_spearman" if target_type == "ranking" else "minimize_mse")
        reg_cfg = exp.get("regularization") or {}
        l2 = float(reg_cfg.get("l2", 0.0) or 0.0)
        bounds = [(0.0, 1.0) for _ in kept_features]
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        best = None
        for init in _init_weights(X, y_values):
            if init is None or len(init) != len(kept_features):
                continue
            objective_fn = _objective_factory(X, y_values, objective, l2, score_scale)
            res = optimize.minimize(objective_fn, init, method="SLSQP", bounds=bounds, constraints=[cons])
            if not res.success:
                continue
            weights = np.clip(res.x, 0.0, None)
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            score = _score_from_weights(X, weights, score_scale)
            metrics = _metric_summary(y_values, score)
            metrics["objective"] = objective
            metrics["score_scale"] = score_scale
            hhi = float(np.sum(weights ** 2))
            metrics["hhi"] = hhi
            metrics["max_weight"] = float(weights.max())
            metrics["near_zero_share"] = float(np.mean(weights < 1e-3))
            candidate = {
                "id": exp.get("id") or exp.get("name") or method,
                "method": method,
                "weights": weights,
                "metrics": metrics,
            }
            if best is None:
                best = candidate
            else:
                best_metric = best["metrics"].get(selection_metric)
                cand_metric = metrics.get(selection_metric)
                if cand_metric is None:
                    continue
                if best_metric is None:
                    best = candidate
                elif selection_dir == "max" and cand_metric > best_metric:
                    best = candidate
                elif selection_dir == "min" and cand_metric < best_metric:
                    best = candidate
        if best:
            results.append(best)

    if not results:
        raise ValueError("no successful experiments executed")

    # Select best experiment
    best_exp = results[0]
    for cand in results[1:]:
        cand_metric = cand["metrics"].get(selection_metric)
        best_metric = best_exp["metrics"].get(selection_metric)
        if cand_metric is None:
            continue
        if best_metric is None:
            best_exp = cand
            continue
        if selection_dir == "max" and cand_metric > best_metric:
            best_exp = cand
        elif selection_dir == "min" and cand_metric < best_metric:
            best_exp = cand

    weights = best_exp["weights"]
    X_full = df[kept_features].to_numpy(dtype=float)
    full_mask = np.isfinite(X_full).all(axis=1)
    score_full = np.full(len(df), np.nan, dtype=float)
    if full_mask.any():
        score_full[full_mask] = _score_from_weights(X_full[full_mask], weights, score_scale)
    df["score_nuevo"] = score_full
    if baseline_col and baseline_col in df.columns:
        df[baseline_col] = pd.to_numeric(df[baseline_col], errors="coerce")

    outputs = {
        "weights_path": outputs_cfg.get("weights_path", "data/weights.json"),
        "case_summary_path": outputs_cfg.get("case_summary_path", "data/case_summary.csv"),
        "scored_rows_path": outputs_cfg.get("scored_rows_path", "data/scored_rows.csv"),
    }

    for path in outputs.values():
        _ensure_dir(path)

    weights_payload = {
        "selected_experiment": best_exp["id"],
        "features": kept_features,
        "weights": {feat: float(w) for feat, w in zip(kept_features, weights)},
        "metrics": best_exp["metrics"],
        "dropped_due_to_missingness": dropped_features,
        "selection_metric": selection_metric,
        "selection_direction": selection_dir,
    }
    if case_col and case_col in df.columns and target_name in df.columns:
        try:
            case_means = df.groupby(case_col)[["score_nuevo", target_name]].mean(numeric_only=True)
            case_corr = stats.spearmanr(case_means["score_nuevo"], case_means[target_name], nan_policy="omit").correlation
            weights_payload["metrics"]["case_order_spearman"] = float(case_corr) if case_corr is not None else None
        except Exception:
            weights_payload["metrics"]["case_order_spearman"] = None
    if baseline_col and baseline_col in df.columns:
        try:
            base_corr = stats.spearmanr(df["score_nuevo"], df[baseline_col], nan_policy="omit").correlation
            base_mae = float(np.nanmean(np.abs(df["score_nuevo"] - df[baseline_col])))
            weights_payload["metrics"]["baseline_spearman"] = float(base_corr) if base_corr is not None else None
            weights_payload["metrics"]["baseline_mae"] = base_mae
        except Exception:
            weights_payload["metrics"]["baseline_spearman"] = None
    with open(outputs["weights_path"], "w", encoding="utf-8") as f:
        json.dump(weights_payload, f, indent=2, default=_json_default)

    if case_col and case_col in df.columns:
        summary = _case_summary(df, case_col, "score_nuevo", baseline_col)
        summary.to_csv(outputs["case_summary_path"], index=False)
    else:
        pd.DataFrame().to_csv(outputs["case_summary_path"], index=False)

    id_cols = _resolve_id_columns(contract)
    cols_for_output = []
    for col in id_cols + kept_features + [target_name]:
        if col in df.columns and col not in cols_for_output:
            cols_for_output.append(col)
    for col in [baseline_col, case_col, "score_nuevo"]:
        if col and col in df.columns and col not in cols_for_output:
            cols_for_output.append(col)
    df[cols_for_output].to_csv(outputs["scored_rows_path"], index=False)

    # Plots
    plots = outputs_cfg.get("plots") or []
    plot_paths = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _ensure_dir("static/plots/weights.png")

        # Weights
        plt.figure(figsize=(8, 4))
        plt.bar(kept_features, weights)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        weight_plot = "static/plots/weight_distribution.png"
        plt.savefig(weight_plot, dpi=150)
        plt.close()
        plot_paths.append(weight_plot)

        if case_col and case_col in df.columns and baseline_col and baseline_col in df.columns:
            case_means = df.groupby(case_col)["score_nuevo"].mean().reset_index()
            plt.figure(figsize=(8, 4))
            plt.plot(case_means[case_col].astype(str), case_means["score_nuevo"], marker="o")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            score_plot = "static/plots/score_vs_refscore_by_case.png"
            plt.savefig(score_plot, dpi=150)
            plt.close()
            plot_paths.append(score_plot)
    except Exception:
        plot_paths = plot_paths

    return {
        "execution_output": f"ML_PLAN_EXECUTION: selected={best_exp['id']} metrics={best_exp['metrics']}",
        "plots": plot_paths,
        "weights_path": outputs["weights_path"],
        "case_summary_path": outputs["case_summary_path"],
        "scored_rows_path": outputs["scored_rows_path"],
    }
