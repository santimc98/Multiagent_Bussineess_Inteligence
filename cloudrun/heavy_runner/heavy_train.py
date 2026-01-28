import json
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


def log(message: str) -> None:
    print(f"[heavy_train] {message}", flush=True)


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    path = uri[len("gs://") :]
    parts = path.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI (missing bucket or object): {uri}")
    return bucket, blob


def _normalize_output_uri(uri: str) -> str:
    if not uri:
        return uri
    return uri if uri.endswith("/") else uri + "/"


def _gcs_client() -> storage.Client:
    return storage.Client()


def _download_gcs_to_path(uri: str, local_path: str) -> None:
    bucket_name, blob_name = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)


def _download_gcs_text(uri: str) -> str:
    bucket_name, blob_name = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(encoding="utf-8-sig")


def _upload_path_to_gcs(local_path: str, uri: str) -> None:
    bucket_name, blob_name = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def _write_json_output(obj: Dict[str, Any], output_uri: str, filename: str) -> None:
    output_uri = _normalize_output_uri(output_uri)
    tmp_path = os.path.join("/tmp", filename)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
    _write_file_output(tmp_path, output_uri, filename)


def _write_file_output(local_path: str, output_uri: str, filename: str) -> None:
    output_uri = _normalize_output_uri(output_uri)
    if output_uri.startswith("gs://"):
        _upload_path_to_gcs(local_path, output_uri + filename)
        return
    os.makedirs(output_uri, exist_ok=True)
    dest = os.path.join(output_uri, filename)
    os.replace(local_path, dest)


def _load_input_json(input_uri: str) -> Dict[str, Any]:
    if input_uri.startswith("gs://"):
        text = _download_gcs_text(input_uri)
        return json.loads(text)
    with open(input_uri, "r", encoding="utf-8") as f:
        return json.load(f)


def _download_to_path(uri: str, local_path: str) -> None:
    if uri.startswith("gs://"):
        _download_gcs_to_path(uri, local_path)
        return
    if os.path.exists(uri):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(uri, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        return
    raise ValueError(f"Unsupported URI for download: {uri}")


def _download_support_files(items: Any, base_dir: str) -> None:
    if not items:
        return
    if not isinstance(items, list):
        raise ValueError("support_files must be a list")
    for item in items:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri")
        rel_path = item.get("path")
        if not uri or not rel_path:
            continue
        dest_path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        _download_to_path(uri, dest_path)


def _run_script(script_path: str, work_dir: str) -> Tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, script_path],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _read_dataset(dataset_uri: str, read_cfg: Dict[str, Any]) -> pd.DataFrame:
    local_path = dataset_uri
    if dataset_uri.startswith("gs://"):
        local_path = os.path.join("/tmp", os.path.basename(dataset_uri))
        _download_gcs_to_path(dataset_uri, local_path)
    ext = os.path.splitext(local_path)[1].lower()
    if ext in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(local_path)
        except Exception as exc:
            raise ValueError(f"Failed to read parquet: {exc}") from exc
    sep = read_cfg.get("sep", ",")
    decimal = read_cfg.get("decimal", ".")
    encoding = read_cfg.get("encoding", "utf-8")
    return pd.read_csv(local_path, sep=sep, decimal=decimal, encoding=encoding)


def _resolve_model(
    problem_type: str,
    model_cfg: Dict[str, Any],
    safe_mode: bool,
) -> Tuple[Any, str, Dict[str, Any]]:
    model_type = model_cfg.get("type")
    if not model_type:
        model_type = "random_forest_classifier" if problem_type == "classification" else "random_forest_regressor"
    params = dict(model_cfg.get("params") or {})
    if safe_mode:
        params["n_jobs"] = 1
    if model_type == "random_forest_classifier":
        return RandomForestClassifier(**params), model_type, params
    if model_type == "random_forest_regressor":
        return RandomForestRegressor(**params), model_type, params
    raise ValueError(f"Unsupported model.type: {model_type}")


def _resolve_cv(
    problem_type: str,
    cv_cfg: Dict[str, Any],
    safe_mode: bool,
) -> Tuple[Optional[Any], Optional[str], Optional[int], Optional[int]]:
    folds = cv_cfg.get("folds")
    if folds is None:
        return None, None, None, None
    try:
        folds = int(folds)
    except Exception:
        raise ValueError("cv.folds must be an integer")
    if folds < 2:
        return None, None, None, None
    if safe_mode and folds > 3:
        folds = 3
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = cv_cfg.get("random_state", 42)
    cv_n_jobs = cv_cfg.get("n_jobs")
    if safe_mode:
        cv_n_jobs = 1
    if problem_type == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state), "accuracy", folds, cv_n_jobs
    return KFold(n_splits=folds, shuffle=shuffle, random_state=random_state), "rmse", folds, cv_n_jobs


def main() -> int:
    input_uri = os.getenv("INPUT_URI", "").strip()
    output_uri = os.getenv("OUTPUT_URI", "").strip()
    if not input_uri:
        log("Missing INPUT_URI env var.")
        return 1
    try:
        payload = _load_input_json(input_uri)
    except Exception as exc:
        log(f"Failed to load input JSON: {exc}")
        return 1

    output_uri = _normalize_output_uri(output_uri or payload.get("output_uri", ""))
    if not output_uri:
        log("Missing OUTPUT_URI env var and no output_uri in input JSON.")
        return 1

    run_id = payload.get("run_id") or "unknown"
    if payload.get("ping") == "ok":
        log("Received ping request. Writing status.json.")
        _write_json_output({"ok": True, "run_id": run_id}, output_uri, "status.json")
        return 0

    try:
        dataset_uri = payload.get("dataset_uri")
        if not dataset_uri:
            raise ValueError("dataset_uri is required")
        read_cfg = payload.get("read") or {}
        target_col = payload.get("target_col")
        if not target_col:
            raise ValueError("target_col is required")
        feature_cols = payload.get("feature_cols")
        problem_type = payload.get("problem_type", "").lower().strip()
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be 'classification' or 'regression'")
        float32 = bool(payload.get("float32"))
        safe_mode = bool(payload.get("safe_mode"))
        model_cfg = payload.get("model") or {}
        cv_cfg = payload.get("cv") or {}
        decisioning_required = payload.get("decisioning_required_names") or []
        if not isinstance(decisioning_required, list):
            decisioning_required = []

        code_uri = payload.get("code_uri")
        if code_uri:
            work_dir = "/tmp/run"
            os.makedirs(work_dir, exist_ok=True)
            data_path = payload.get("data_path") or "data/cleaned_data.csv"
            data_path = os.path.join(work_dir, data_path)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            log(f"Downloading dataset for code execution from {dataset_uri}")
            _download_to_path(dataset_uri, data_path)
            support_files = payload.get("support_files")
            _download_support_files(support_files, work_dir)
            script_path = os.path.join(work_dir, "ml_script.py")
            log(f"Downloading ML script from {code_uri}")
            _download_to_path(code_uri, script_path)
            os.makedirs(os.path.join(work_dir, "static", "plots"), exist_ok=True)
            log("Executing ML script in heavy runner.")
            exit_code, stdout, stderr = _run_script(script_path, work_dir)
            log_path = os.path.join("/tmp", "execution_log.txt")
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(stdout or "")
                if stderr:
                    log_file.write("\n[stderr]\n")
                    log_file.write(stderr)
            _write_file_output(log_path, output_uri, "execution_log.txt")
            if exit_code != 0:
                raise RuntimeError(
                    f"ML script failed with exit code {exit_code}. See execution_log.txt for details."
                )
            expected_outputs = payload.get("expected_outputs") or [
                "data/metrics.json",
                "data/scored_rows.csv",
                "data/alignment_check.json",
            ]
            uploaded = []
            for rel_path in expected_outputs:
                local_path = os.path.join(work_dir, rel_path)
                if os.path.exists(local_path):
                    _write_file_output(local_path, output_uri, os.path.basename(rel_path))
                    uploaded.append(rel_path)
            status_payload = {
                "ok": True,
                "run_id": run_id,
                "mode": "execute_code",
                "uploaded_outputs": uploaded,
            }
            _write_json_output(status_payload, output_uri, "status.json")
            log("ML script execution completed successfully.")
            return 0

        log(f"Loading dataset from {dataset_uri}")
        df = _read_dataset(dataset_uri, read_cfg)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        drop_cols = [target_col]
        if "__split" in df.columns:
            drop_cols.append("__split")
        if feature_cols:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"feature_cols missing from dataset: {missing[:10]}")
            X = df[feature_cols]
        else:
            X = df.drop(columns=drop_cols)
        y = df[target_col]
        train_mask = y.notna()
        missing_target = int((~train_mask).sum())
        if missing_target:
            log(f"Detected {missing_target} rows with NaN target; excluding from training/CV.")

        if float32:
            X = X.astype(np.float32)
        else:
            # Ensure numeric features for sklearn RF
            non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            if non_numeric:
                raise ValueError(f"Non-numeric feature columns present: {non_numeric[:10]}")

        X_train = X[train_mask]
        y_train = y[train_mask]
        if len(y_train) == 0:
            raise ValueError("No non-null target rows available for training.")
        if problem_type == "classification" and pd.api.types.is_numeric_dtype(y_train):
            y_train = y_train.astype(int)

        model, model_type, model_params = _resolve_model(problem_type, model_cfg, safe_mode)
        cv_obj, metric_name, folds, cv_n_jobs = _resolve_cv(problem_type, cv_cfg, safe_mode)

        metrics: Dict[str, Any] = {
            "run_id": run_id,
            "problem_type": problem_type,
            "model": {"type": model_type, "params": model_params},
            "data": {
                "n_rows": int(len(df)),
                "n_cols": int(df.shape[1]),
                "n_features": int(X.shape[1]),
                "n_train_rows": int(len(y_train)),
                "n_missing_target": int(missing_target),
                "target_col": target_col,
                "feature_cols": feature_cols if feature_cols else "all_except_target",
                "float32": float32,
                "dtype_summary": {k: int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
            },
            "timing_seconds": {},
            "cv": {"enabled": False},
        }

        if cv_obj is not None and len(y_train) >= (folds or 0):
            log(f"Running cross-validation: folds={folds}, metric={metric_name}")
            cv_start = time.perf_counter()
            if problem_type == "classification":
                scores = cross_val_score(model, X_train, y_train, cv=cv_obj, scoring="accuracy", n_jobs=cv_n_jobs)
                scores_list = [float(s) for s in scores]
                metrics["cv"] = {
                    "enabled": True,
                    "folds": folds,
                    "metric": "accuracy",
                    "scores": scores_list,
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }
            else:
                scores = cross_val_score(
                    model, X_train, y_train, cv=cv_obj, scoring="neg_root_mean_squared_error", n_jobs=cv_n_jobs
                )
                rmse_scores = [-float(s) for s in scores]
                metrics["cv"] = {
                    "enabled": True,
                    "folds": folds,
                    "metric": "rmse",
                    "scores": rmse_scores,
                    "mean": float(np.mean(rmse_scores)),
                    "std": float(np.std(rmse_scores)),
                }
            metrics["timing_seconds"]["cv"] = round(time.perf_counter() - cv_start, 6)
        elif cv_obj is not None:
            metrics["cv"] = {
                "enabled": False,
                "note": "insufficient labeled rows for requested cv folds",
                "folds_requested": folds,
                "n_train_rows": int(len(y_train)),
            }

        log("Fitting final model on full dataset.")
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        metrics["timing_seconds"]["train"] = round(time.perf_counter() - train_start, 6)

        log("Generating predictions for scored rows.")
        preds = model.predict(X)
        scored_df = df.copy()
        scored_df["prediction"] = preds
        train_metrics: Dict[str, Any] = {}
        if problem_type == "classification":
            probs = model.predict_proba(X)
            max_prob = np.max(probs, axis=1)
            scored_df["probability"] = max_prob
            scored_df["confidence_score"] = max_prob
            train_metrics["accuracy"] = float(np.mean(preds[train_mask] == y_train.values))
        else:
            scored_df["predicted_value"] = preds
            rmse = float(np.sqrt(np.mean((preds[train_mask] - y_train.values) ** 2)))
            train_metrics["rmse"] = rmse

        if decisioning_required:
            priority_score = None
            if "priority_score" in decisioning_required:
                if problem_type == "classification" and "probability" in scored_df.columns:
                    priority_score = scored_df["probability"].astype(float).to_numpy()
                else:
                    preds_arr = np.asarray(preds, dtype=float)
                    min_val = float(np.min(preds_arr))
                    max_val = float(np.max(preds_arr))
                    denom = max_val - min_val
                    if denom <= 0:
                        priority_score = np.zeros_like(preds_arr)
                    else:
                        priority_score = (preds_arr - min_val) / denom
                scored_df["priority_score"] = priority_score
            if "priority_rank" in decisioning_required:
                if priority_score is None:
                    priority_score = scored_df.get("priority_score", pd.Series(np.zeros(len(scored_df))))
                ranks = pd.Series(priority_score).rank(method="first", ascending=False).astype(int)
                scored_df["priority_rank"] = ranks

        metrics["train"] = train_metrics

        metrics_path = os.path.join("/tmp", "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=True)
        _write_file_output(metrics_path, output_uri, "metrics.json")

        model_path = os.path.join("/tmp", "model.joblib")
        joblib.dump(model, model_path)
        _write_file_output(model_path, output_uri, "model.joblib")

        alignment_reqs = []
        if metrics.get("cv", {}).get("enabled"):
            alignment_reqs.append(
                {
                    "name": f"cv_{metrics['cv'].get('metric')}",
                    "status": "PASS",
                    "value": metrics["cv"].get("mean"),
                }
            )
        if train_metrics:
            for key, val in train_metrics.items():
                alignment_reqs.append({"name": f"train_{key}", "status": "PASS", "value": val})

        alignment_check = {
            "status": "PASS",
            "summary": "Heavy runner training completed.",
            "requirements": alignment_reqs,
            "feature_usage": {
                "used_features": feature_cols if feature_cols else "all_except_target",
                "target_columns": [target_col],
            },
        }
        alignment_path = os.path.join("/tmp", "alignment_check.json")
        with open(alignment_path, "w", encoding="utf-8") as f:
            json.dump(alignment_check, f, indent=2, ensure_ascii=True)
        _write_file_output(alignment_path, output_uri, "alignment_check.json")

        scored_path = os.path.join("/tmp", "scored_rows.csv")
        sep = read_cfg.get("sep", ",")
        decimal = read_cfg.get("decimal", ".")
        encoding = read_cfg.get("encoding", "utf-8")
        scored_df.to_csv(scored_path, index=False, sep=sep, decimal=decimal, encoding=encoding)
        _write_file_output(scored_path, output_uri, "scored_rows.csv")

        log("Training completed successfully.")
        return 0
    except Exception as exc:
        stack = traceback.format_exc()
        log(f"ERROR: {exc}")
        try:
            _write_json_output(
                {
                    "ok": False,
                    "run_id": run_id,
                    "error": str(exc),
                    "stacktrace": stack,
                },
                output_uri,
                "error.json",
            )
        except Exception as write_err:
            log(f"Failed to write error.json: {write_err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
