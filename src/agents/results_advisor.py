import os
import json
import csv
import re
from statistics import median
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.retries import call_with_retries

load_dotenv()


class ResultsAdvisorAgent:
    """
    Generate insights and improvement advice from evaluation artifacts.
    """

    def __init__(self, api_key: Any = None):
        self.api_key = api_key or os.getenv("MIMO_API_KEY")
        if not self.api_key:
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.xiaomimimo.com/v1",
                timeout=None,
            )
        self.model_name = "mimo-v2-flash"
        self.last_prompt = None
        self.last_response = None

    def generate_ml_advice(self, context: Dict[str, Any]) -> str:
        if not context:
            return ""
        insights = self.generate_insights(context)
        summary_lines = insights.get("summary_lines", []) if isinstance(insights, dict) else []
        if summary_lines:
            return "\n".join(summary_lines)
        if not self.client:
            return self._fallback(context)

        context_snippet = self._truncate(json.dumps(context, ensure_ascii=True), 4000)
        system_prompt = (
            "You are a senior ML reviewer focused on business alignment. "
            "Given evaluation artifacts and business alignment, produce improvement guidance. "
            "Prioritize structural changes when alignment metrics fail (objective/constraints/penalties) "
            "before tuning hyperparameters. Compare current vs previous iteration if provided. "
            "Return 3-6 short lines. Format each line as: "
            "ISSUE: <what failed>; WHY: <root cause>; FIX: <specific change>. "
            "Do NOT include code. Do NOT restate the full metrics dump."
        )
        user_prompt = "CONTEXT:\n" + context_snippet + "\n"
        self.last_prompt = system_prompt + "\n\n" + user_prompt

        def _call_model():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content

        try:
            content = call_with_retries(_call_model, max_retries=2)
        except Exception:
            return self._fallback(context)

        self.last_response = content
        return (content or "").strip()

    def generate_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        produced_index = (
            context.get("produced_artifact_index")
            or context.get("artifact_index")
            or self._safe_load_json("data/produced_artifact_index.json")
            or []
        )
        artifact_index = self._normalize_artifact_index(produced_index)
        if not artifact_index:
            output_report = context.get("output_contract_report") or self._safe_load_json("data/output_contract_report.json")
            present = output_report.get("present", []) if isinstance(output_report, dict) else []
            artifact_index = [
                {"path": path, "artifact_type": self._infer_artifact_type(path)}
                for path in present
                if path
            ]
        objective_type = context.get("objective_type") or context.get("strategy_spec", {}).get("objective_type") or "unknown"

        reporting_policy = context.get("reporting_policy") if isinstance(context, dict) else {}
        slot_defs = reporting_policy.get("slots", []) if isinstance(reporting_policy, dict) else []
        allowed_slots = {slot.get("id") for slot in slot_defs if isinstance(slot, dict) and slot.get("id")}

        metrics_artifacts = self._find_artifacts_by_type(artifact_index, "metrics")
        predictions_artifacts = self._find_artifacts_by_type(artifact_index, "predictions")
        error_artifacts = self._find_artifacts_by_type(artifact_index, "error_analysis")
        importances_artifacts = self._find_artifacts_by_type(artifact_index, "feature_importances")

        metrics_payload = self._safe_load_json(metrics_artifacts[0]) if metrics_artifacts else {}
        predictions_summary = self._summarize_csv(predictions_artifacts[0]) if predictions_artifacts else {}
        error_payload = self._safe_load_json(error_artifacts[0]) if error_artifacts else {}
        importances_payload = self._safe_load_json(importances_artifacts[0]) if importances_artifacts else {}
        alignment_payload = self._safe_load_json("data/alignment_check.json") or {}
        case_alignment_payload = self._safe_load_json("data/case_alignment_report.json") or {}

        metrics_summary = self._extract_metrics_summary(metrics_payload, objective_type)
        if not metrics_summary and isinstance(metrics_payload, dict):
            nested = metrics_payload.get("metrics")
            metrics_summary = self._extract_metrics_summary(nested, objective_type)
        deployment_info = self._compute_deployment_recommendation(metrics_payload, predictions_summary)
        risks = []
        recommendations = []
        summary_lines: List[str] = []

        metrics_present = bool(metrics_artifacts) and isinstance(metrics_payload, dict)
        if not metrics_summary and not metrics_present:
            risks.append("Metrics artifact missing or empty; evaluation confidence is limited.")
            recommendations.append("Generate a metrics artifact aligned to the objective type.")
        elif not metrics_summary and metrics_present:
            recommendations.append("Metrics artifact present but no numeric metrics detected; populate key metrics.")
        if predictions_summary:
            summary_lines.append(
                f"Predictions preview includes {predictions_summary.get('row_count', 0)} rows."
            )
        if error_payload:
            summary_lines.append("Error analysis artifact available; review failure patterns.")
        if importances_payload:
            summary_lines.append("Feature importance artifact available; use for explainability.")

        output_report = context.get("output_contract_report", {}) or {}
        missing_required = output_report.get("missing", []) if isinstance(output_report, dict) else []
        if missing_required:
            risks.append(f"Missing required outputs: {missing_required}")
            recommendations.append("Ensure required artifacts are written before pipeline completion.")

        review_feedback = str(context.get("review_feedback") or "")
        if "leakage" in review_feedback.lower():
            risks.append("Leakage risk flagged in reviewer feedback.")
            recommendations.append("Audit feature availability timing and exclude post-outcome fields.")

        if objective_type == "classification":
            recommendations.append("Check class balance and calibrate thresholds if needed.")
        elif objective_type == "regression":
            recommendations.append("Inspect residuals and consider robust loss if heavy tails exist.")
        elif objective_type == "forecasting":
            recommendations.append("Validate forecast horizon and compare against naive baselines.")
        elif objective_type == "ranking":
            recommendations.append("Validate ordering metrics and consider pairwise loss if rankings are unstable.")

        artifacts_used = []
        for path in (metrics_artifacts + predictions_artifacts + error_artifacts + importances_artifacts):
            artifacts_used.append(path)

        segment_pricing_summary = self._build_segment_pricing_summary(predictions_artifacts[0]) if predictions_artifacts else []
        leakage_audit = self._extract_leakage_audit(alignment_payload)
        validation_summary = self._extract_validation_summary(alignment_payload)
        case_or_bucket_summary = self._extract_case_alignment_summary(case_alignment_payload)

        slot_payloads = {}
        if metrics_summary:
            slot_payloads["model_metrics"] = metrics_summary
        if predictions_summary:
            slot_payloads["predictions_overview"] = predictions_summary
        if segment_pricing_summary:
            slot_payloads["segment_pricing"] = segment_pricing_summary
        if leakage_audit:
            slot_payloads["alignment_risks"] = leakage_audit
        if validation_summary:
            slot_payloads["validation_summary"] = validation_summary
        if case_or_bucket_summary:
            slot_payloads["case_or_bucket_summary"] = case_or_bucket_summary
        if allowed_slots:
            slot_payloads = {k: v for k, v in slot_payloads.items() if k in allowed_slots}

        insights = {
            "schema_version": "1",
            "objective_type": objective_type,
            "artifacts_used": artifacts_used,
            "metrics_summary": metrics_summary,
            "predictions_summary": predictions_summary,
            "overall_scored_rows_row_count": predictions_summary.get("row_count") if isinstance(predictions_summary, dict) else None,
            "segment_pricing_summary": segment_pricing_summary,
            "leakage_audit": leakage_audit,
            "slot_payloads": slot_payloads,
            "risks": risks,
            "recommendations": recommendations,
            "summary_lines": summary_lines,
            "deployment_recommendation": deployment_info.get("deployment_recommendation"),
            "confidence": deployment_info.get("confidence"),
            "primary_metric": deployment_info.get("primary_metric"),
        }
        self.last_response = insights
        if not summary_lines and metrics_summary:
            summary_lines.append("Metrics artifact available; review key performance indicators.")
        if not summary_lines:
            summary_lines.append("Limited artifacts available; generate metrics and predictions for insights.")
        return insights

    def _truncate(self, text: str, max_len: int) -> str:
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len]

    def _safe_load_json(self, path: str) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return {}

    def _load_output_dialect(self, manifest_path: str = "data/cleaning_manifest.json") -> Dict[str, Any]:
        manifest = self._safe_load_json(manifest_path)
        if not isinstance(manifest, dict):
            return {}
        dialect = manifest.get("output_dialect") or manifest.get("dialect") or {}
        if not isinstance(dialect, dict):
            return {}
        sep = dialect.get("sep") or dialect.get("delimiter")
        decimal = dialect.get("decimal")
        encoding = dialect.get("encoding")
        cleaned = {}
        if sep:
            cleaned["sep"] = str(sep)
        if decimal:
            cleaned["decimal"] = str(decimal)
        if encoding:
            cleaned["encoding"] = str(encoding)
        return cleaned

    def _sniff_csv_dialect(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
        except Exception:
            return {"sep": ",", "decimal": ".", "encoding": "utf-8"}

        sample_text = "".join(lines)
        header = lines[0] if lines else ""
        sep = ";" if header.count(";") > header.count(",") else ","
        comma_decimals = len(re.findall(r"\d+,\d+", sample_text))
        dot_decimals = len(re.findall(r"\d+\.\d+", sample_text))
        decimal = "," if comma_decimals > dot_decimals else "."
        return {"sep": sep, "decimal": decimal, "encoding": "utf-8"}

    def _coerce_number(self, raw: Any, decimal: str) -> Optional[float]:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        text = text.replace(" ", "")
        if decimal == ",":
            if text.count(",") == 1 and text.count(".") >= 1:
                text = text.replace(".", "")
            text = text.replace(",", ".")
        try:
            return float(text)
        except Exception:
            return None

    def _read_csv_summary(self, path: str, dialect: Dict[str, Any], max_rows: int) -> Dict[str, Any]:
        sep = dialect.get("sep") or ","
        decimal = dialect.get("decimal") or "."
        encoding = dialect.get("encoding") or "utf-8"
        rows: List[Dict[str, Any]] = []
        row_count_total = 0
        try:
            with open(path, "r", encoding=str(encoding), errors="replace") as f:
                reader = csv.DictReader(f, delimiter=str(sep))
                columns = reader.fieldnames or []
                for row in reader:
                    row_count_total += 1
                    if len(rows) < max_rows:
                        rows.append(row)
            numeric_cols = self._numeric_columns(rows, columns, decimal)
            return {
                "row_count": row_count_total,
                "row_count_total": row_count_total,
                "row_count_sampled": len(rows),
                "columns": columns,
                "numeric_columns": numeric_cols,
                "examples": rows[: min(5, len(rows))],
                "dialect_used": {"sep": sep, "decimal": decimal, "encoding": encoding},
            }
        except Exception:
            return {}

    def _summarize_csv(self, path: str, max_rows: int = 200) -> Dict[str, Any]:
        if not path:
            return {}
        manifest_dialect = self._load_output_dialect()
        dialect = manifest_dialect or self._sniff_csv_dialect(path)
        summary = self._read_csv_summary(path, dialect, max_rows)
        columns = summary.get("columns", [])
        if (
            isinstance(columns, list)
            and len(columns) == 1
            and isinstance(columns[0], str)
            and ";" in columns[0]
        ):
            sniffed = self._sniff_csv_dialect(path)
            if sniffed.get("sep") != dialect.get("sep"):
                summary = self._read_csv_summary(path, sniffed, max_rows)
        return summary

    def _numeric_columns(self, rows: List[Dict[str, Any]], columns: List[str], decimal: str) -> List[str]:
        numeric_cols: List[str] = []
        for col in columns:
            values = []
            for row in rows:
                raw = row.get(col)
                if raw in (None, ""):
                    continue
                values.append(self._coerce_number(raw, decimal))
            non_null = [v for v in values if v is not None]
            if non_null and len(non_null) >= max(1, len(values) // 3):
                numeric_cols.append(col)
        return numeric_cols

    def _extract_metrics_summary(
        self,
        metrics: Dict[str, Any],
        objective_type: str = "unknown",
        max_items: int = 8,
    ) -> List[Dict[str, Any]]:
        if not isinstance(metrics, dict):
            return []
        items: List[Dict[str, Any]] = []
        objective = str(objective_type or "unknown").lower()
        model_perf = metrics.get("model_performance") if isinstance(metrics.get("model_performance"), dict) else {}
        seg_stats = metrics.get("segmentation_stats") if isinstance(metrics.get("segmentation_stats"), dict) else {}
        priority_keys = [
            "accuracy",
            "roc_auc",
            "precision",
            "f1",
            "rmse",
            "mae",
            "r2",
            "silhouette_score",
            "training_samples",
        ]
        for key in priority_keys:
            if key in model_perf:
                num = self._coerce_number(model_perf.get(key), ".")
                if num is not None:
                    items.append({"metric": f"model_performance.{key}", "value": num})
        seg_keys = ["n_segments", "min_segment_size", "median_segment_size"]
        for key in seg_keys:
            if key in seg_stats:
                num = self._coerce_number(seg_stats.get(key), ".")
                if num is not None:
                    items.append({"metric": f"segmentation_stats.{key}", "value": num})
        if len(items) >= max_items:
            return items[:max_items]
        flat_metrics = self._flatten_numeric_metrics(metrics)
        priority_tokens = self._objective_metric_priority(objective)
        used_keys = {item["metric"] for item in items if isinstance(item, dict) and item.get("metric")}
        for token in priority_tokens:
            for key, value in flat_metrics.items():
                norm = key.lower()
                if token in norm and key not in used_keys:
                    items.append({"metric": key, "value": value})
                    used_keys.add(key)
                    if len(items) >= max_items:
                        return items
        if len(items) >= max_items:
            return items[:max_items]
        for key, value in flat_metrics.items():
            if key in used_keys:
                continue
            items.append({"metric": key, "value": value})
            if len(items) >= max_items:
                break
        return items

    def _normalize_metric_key(self, name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

    def _pick_primary_metric_with_ci(self, model_perf: Dict[str, Any]) -> tuple[str | None, Dict[str, Any] | None]:
        if not isinstance(model_perf, dict):
            return None, None
        for key, value in model_perf.items():
            if self._normalize_metric_key(key) == "revenuelift" and isinstance(value, dict):
                return str(key), value
        for key, value in model_perf.items():
            if isinstance(value, dict) and all(k in value for k in ("mean", "ci_lower", "ci_upper")):
                return str(key), value
        return None, None

    def _extract_row_count(self, metrics: Dict[str, Any], predictions_summary: Dict[str, Any]) -> Optional[int]:
        row_count = None
        if isinstance(predictions_summary, dict):
            rc = predictions_summary.get("row_count")
            num = self._coerce_number(rc, ".")
            if num is not None:
                row_count = int(num)
        model_perf = metrics.get("model_performance") if isinstance(metrics.get("model_performance"), dict) else {}
        for key in ["training_samples", "n_samples", "n_rows", "rows", "row_count"]:
            if row_count is not None:
                break
            num = self._coerce_number(model_perf.get(key), ".")
            if num is not None:
                row_count = int(num)
        return row_count

    def _compute_deployment_recommendation(
        self,
        metrics: Dict[str, Any],
        predictions_summary: Dict[str, Any],
        min_rows: int = 200,
    ) -> Dict[str, Any]:
        recommendation = "PILOT"
        confidence = "LOW"
        model_perf = metrics.get("model_performance") if isinstance(metrics.get("model_performance"), dict) else {}
        metric_name, metric_payload = self._pick_primary_metric_with_ci(model_perf)
        if not metric_name or not isinstance(metric_payload, dict):
            return {
                "deployment_recommendation": recommendation,
                "confidence": confidence,
            }
        mean = self._coerce_number(metric_payload.get("mean"), ".")
        lower = self._coerce_number(metric_payload.get("ci_lower"), ".")
        upper = self._coerce_number(metric_payload.get("ci_upper"), ".")
        if mean is None or lower is None or upper is None:
            return {
                "deployment_recommendation": recommendation,
                "confidence": confidence,
            }
        width = max(0.0, upper - lower)
        normalized_width = width / max(abs(mean), 1.0)
        non_negative = all(val >= 0 for val in [mean, lower, upper])
        ratio_like = non_negative and all(abs(val - 1.0) <= 0.2 for val in [mean, lower, upper])
        baseline = 1.0 if ratio_like else 0.0
        includes_baseline = lower <= baseline <= upper
        row_count = self._extract_row_count(metrics, predictions_summary)

        if includes_baseline:
            recommendation = "PILOT"
            confidence = "MEDIUM" if normalized_width <= 0.1 else "LOW"
        else:
            if row_count is not None and row_count >= min_rows:
                recommendation = "GO"
                confidence = "HIGH" if normalized_width <= 0.1 else "MEDIUM"
            else:
                recommendation = "PILOT"
                confidence = "MEDIUM" if normalized_width <= 0.1 else "LOW"

        return {
            "deployment_recommendation": recommendation,
            "confidence": confidence,
            "primary_metric": metric_name,
        }

    def _flatten_numeric_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        if not isinstance(metrics, dict):
            return {}
        flat: Dict[str, float] = {}
        for key, value in metrics.items():
            metric_key = f"{prefix}{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(self._flatten_numeric_metrics(value, f"{metric_key}."))
                continue
            num = self._coerce_number(value, ".")
            if num is not None:
                flat[str(metric_key)] = float(num)
        return flat

    def _objective_metric_priority(self, objective_type: str) -> List[str]:
        objective = str(objective_type or "unknown").lower()
        if objective == "classification":
            return ["roc_auc", "auc", "f1", "precision", "recall", "accuracy", "balanced_accuracy", "pr_auc"]
        if objective == "regression":
            return ["rmse", "mae", "mse", "r2", "mape", "smape"]
        if objective == "forecasting":
            return ["mape", "smape", "rmse", "mae", "coverage", "pinball"]
        if objective == "ranking":
            return ["spearman", "kendall", "ndcg", "map", "mrr", "gini"]
        return ["roc_auc", "f1", "rmse", "mae", "r2", "spearman"]

    def _pick_column(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        lower_map = {col.lower(): col for col in columns}
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        for col in columns:
            col_lower = col.lower()
            for cand in candidates:
                if cand in col_lower:
                    return col
        return None

    def _build_segment_pricing_summary(self, path: str) -> List[Dict[str, Any]]:
        if not path:
            return []
        base_dialect = self._load_output_dialect()
        dialect = base_dialect or self._sniff_csv_dialect(path)
        for _ in range(2):
            sep = dialect.get("sep") or ","
            decimal = dialect.get("decimal") or "."
            encoding = dialect.get("encoding") or "utf-8"
            try:
                with open(path, "r", encoding=str(encoding), errors="replace") as f:
                    reader = csv.DictReader(f, delimiter=str(sep))
                    columns = reader.fieldnames or []
                    if not columns:
                        return []
                    if len(columns) == 1 and ";" in columns[0] and dialect.get("sep") != ";":
                        dialect = self._sniff_csv_dialect(path)
                        continue

                    segment_col = self._pick_column(columns, ["client_segment", "segment", "cluster", "group"])
                    price_col = self._pick_column(
                        columns,
                        [
                            "recommended_price",
                            "optimal_price",
                            "optimal_price_recommendation",
                            "price_recommendation",
                            "recommended_price_value",
                        ],
                    )
                    prob_col = self._pick_column(
                        columns,
                        [
                            "predicted_success_prob",
                            "predicted_probability",
                            "success_prob",
                            "probability",
                        ],
                    )
                    expected_rev_col = self._pick_column(
                        columns,
                        [
                            "expected_revenue",
                            "expected_value",
                            "expected_rev",
                            "expected_profit",
                        ],
                    )

                    if not segment_col:
                        return []

                    buckets: Dict[str, Dict[str, List[float] | int]] = {}
                    for row in reader:
                        seg_raw = row.get(segment_col)
                        if seg_raw is None or seg_raw == "":
                            continue
                        seg_key = str(seg_raw).strip()
                        if seg_key not in buckets:
                            buckets[seg_key] = {
                                "count": 0,
                                "prices": [],
                                "probs": [],
                                "revenues": [],
                            }
                        bucket = buckets[seg_key]
                        bucket["count"] = int(bucket["count"]) + 1
                        if price_col:
                            price_val = self._coerce_number(row.get(price_col), decimal)
                            if price_val is not None:
                                bucket["prices"].append(price_val)
                        if prob_col:
                            prob_val = self._coerce_number(row.get(prob_col), decimal)
                            if prob_val is not None:
                                bucket["probs"].append(prob_val)
                        if expected_rev_col:
                            rev_val = self._coerce_number(row.get(expected_rev_col), decimal)
                            if rev_val is not None:
                                bucket["revenues"].append(rev_val)

                    summary = []
                    for segment, bucket in buckets.items():
                        prices = bucket["prices"]
                        probs = bucket["probs"]
                        revenues = bucket["revenues"]
                        optimal_price = float(median(prices)) if prices else None
                        mean_prob = float(sum(probs) / len(probs)) if probs else None
                        mean_rev = float(sum(revenues) / len(revenues)) if revenues else None
                        summary.append(
                            {
                                "segment": segment,
                                "n": int(bucket["count"]),
                                "optimal_price": optimal_price,
                                "mean_prob": mean_prob,
                                "mean_expected_revenue": mean_rev,
                            }
                        )
                    return summary
            except Exception:
                return []
        return []

    def _extract_leakage_audit(self, alignment_payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(alignment_payload, dict):
            return None

        def _normalize(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not isinstance(candidate, dict):
                return None
            feature = candidate.get("feature") or candidate.get("column") or candidate.get("field")
            target = candidate.get("target") or candidate.get("label") or candidate.get("outcome")
            corr = (
                candidate.get("correlation_coefficient")
                or candidate.get("correlation")
                or candidate.get("corr")
                or candidate.get("spearman")
                or candidate.get("pearson")
            )
            threshold = candidate.get("threshold") or candidate.get("corr_threshold")
            action = candidate.get("action_taken") or candidate.get("action") or candidate.get("action_if_exceeds")
            if corr is None and feature is None and target is None:
                return None
            return {
                "feature": feature,
                "target": target,
                "correlation": corr,
                "threshold": threshold,
                "action_taken": action,
            }

        for key in ("leakage_audit", "leakage", "leakage_check"):
            candidate = alignment_payload.get(key)
            if isinstance(candidate, dict):
                normalized = _normalize(candidate)
                if normalized:
                    return normalized

        requirements = alignment_payload.get("requirements")
        if isinstance(requirements, list):
            for req in requirements:
                if not isinstance(req, dict):
                    continue
                name = str(req.get("name") or req.get("id") or "")
                if "leak" not in name.lower():
                    continue
                evidence = req.get("evidence")
                if isinstance(evidence, dict):
                    normalized = _normalize(evidence)
                    if normalized:
                        return normalized
                normalized = _normalize(req)
                if normalized:
                    return normalized

        return None

    def _extract_validation_summary(self, alignment_payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(alignment_payload, dict) or not alignment_payload:
            return None
        status = alignment_payload.get("status") or alignment_payload.get("overall_status")
        failure_mode = alignment_payload.get("failure_mode")
        summary = alignment_payload.get("summary") or alignment_payload.get("notes")
        if status is None and failure_mode is None and summary is None:
            return None
        return {"status": status, "failure_mode": failure_mode, "summary": summary}

    def _extract_case_alignment_summary(self, payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict) or not payload:
            return None
        status = payload.get("status")
        mode = payload.get("mode")
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        summary = {"status": status, "mode": mode}
        if metrics:
            summary["metrics"] = metrics
        return summary if status or metrics else None

    def _normalize_artifact_index(self, entries: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in entries or []:
            if isinstance(item, dict) and item.get("path"):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"path": item, "artifact_type": "artifact"})
        return normalized

    def _infer_artifact_type(self, path: str) -> str:
        lower = str(path).lower()
        if "metrics" in lower:
            return "metrics"
        if "alignment" in lower:
            return "report"
        if "error" in lower:
            return "error_analysis"
        if "importance" in lower:
            return "feature_importances"
        if "scored_rows" in lower or "predictions" in lower:
            return "predictions"
        return "artifact"

    def _find_artifacts_by_type(self, entries: List[Dict[str, Any]], artifact_type: str) -> List[str]:
        matches = []
        for item in entries or []:
            if not isinstance(item, dict):
                continue
            if item.get("artifact_type") == artifact_type and item.get("path"):
                matches.append(item["path"])
        return matches

    def _fallback(self, context: Dict[str, Any]) -> str:
        lines: List[str] = []
        output_report = context.get("output_contract_report", {}) or {}
        review_feedback = str(context.get("review_feedback") or "")
        if output_report.get("missing"):
            lines.append(
                "ISSUE: required outputs missing; WHY: outputs not saved; FIX: ensure all required artifacts are written."
            )
        if "leakage" in review_feedback.lower():
            lines.append(
                "ISSUE: leakage risk flagged; WHY: post-outcome fields may be in features; FIX: audit feature timing and remove leaks."
            )
        metrics = context.get("metrics") or {}
        if not metrics:
            lines.append(
                "ISSUE: metrics missing; WHY: evaluation artifacts absent; FIX: generate metrics aligned to the objective."
            )
        if not lines:
            lines.append(
                "ISSUE: limited insights; WHY: insufficient artifacts; FIX: produce metrics, predictions, and error analysis."
            )
        return "\n".join(lines)
