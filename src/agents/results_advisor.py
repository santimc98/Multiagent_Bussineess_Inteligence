import os
import json
from typing import Any, Dict, List

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

        return (content or "").strip()

    def generate_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        artifact_index = self._normalize_artifact_index(
            context.get("artifact_index") or self._safe_load_json("data/artifact_index.json") or []
        )
        objective_type = context.get("objective_type") or context.get("strategy_spec", {}).get("objective_type") or "unknown"

        metrics_artifacts = self._find_artifacts_by_type(artifact_index, "metrics")
        predictions_artifacts = self._find_artifacts_by_type(artifact_index, "predictions")
        error_artifacts = self._find_artifacts_by_type(artifact_index, "error_analysis")
        importances_artifacts = self._find_artifacts_by_type(artifact_index, "feature_importances")

        metrics_payload = self._safe_load_json(metrics_artifacts[0]) if metrics_artifacts else {}
        predictions_summary = self._summarize_csv(predictions_artifacts[0]) if predictions_artifacts else {}
        error_payload = self._safe_load_json(error_artifacts[0]) if error_artifacts else {}
        importances_payload = self._safe_load_json(importances_artifacts[0]) if importances_artifacts else {}

        metrics_summary = self._extract_numeric_metrics(metrics_payload)
        if not metrics_summary and isinstance(metrics_payload, dict):
            nested = metrics_payload.get("metrics")
            metrics_summary = self._extract_numeric_metrics(nested)
        risks = []
        recommendations = []
        summary_lines: List[str] = []

        if not metrics_summary:
            risks.append("Metrics artifact missing or empty; evaluation confidence is limited.")
            recommendations.append("Generate a metrics artifact aligned to the objective type.")
        if predictions_summary:
            summary_lines.append(f"Predictions preview includes {predictions_summary.get('row_count', 0)} rows.")
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

        insights = {
            "schema_version": "1",
            "objective_type": objective_type,
            "artifacts_used": artifacts_used,
            "metrics_summary": metrics_summary,
            "predictions_summary": predictions_summary,
            "risks": risks,
            "recommendations": recommendations,
            "summary_lines": summary_lines,
        }
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

    def _summarize_csv(self, path: str, max_rows: int = 200) -> Dict[str, Any]:
        if not path:
            return {}
        try:
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = []
                for _, row in zip(range(max_rows), reader):
                    rows.append(row)
            columns = reader.fieldnames or []
            numeric_cols = self._numeric_columns(rows, columns)
            return {
                "row_count": len(rows),
                "columns": columns,
                "numeric_columns": numeric_cols,
            }
        except Exception:
            return {}

    def _numeric_columns(self, rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
        numeric_cols: List[str] = []
        for col in columns:
            values = []
            for row in rows:
                raw = row.get(col)
                if raw in (None, ""):
                    continue
                try:
                    values.append(float(str(raw).replace(",", ".")))
                except Exception:
                    values.append(None)
            non_null = [v for v in values if v is not None]
            if non_null and len(non_null) >= max(1, len(values) // 3):
                numeric_cols.append(col)
        return numeric_cols

    def _extract_numeric_metrics(self, metrics: Dict[str, Any], max_items: int = 10) -> List[Dict[str, Any]]:
        if not isinstance(metrics, dict):
            return []
        items: List[Dict[str, Any]] = []
        for key, value in metrics.items():
            try:
                num = float(value)
            except Exception:
                continue
            items.append({"metric": str(key), "value": num})
            if len(items) >= max_items:
                break
        return items

    def _normalize_artifact_index(self, entries: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in entries or []:
            if isinstance(item, dict) and item.get("path"):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"path": item, "artifact_type": "artifact"})
        return normalized

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
