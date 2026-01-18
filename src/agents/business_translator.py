import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from string import Template
import json
from typing import Dict, Any, Optional, List
from src.utils.prompting import render_prompt
from src.utils.csv_dialect import (
    load_output_dialect,
    sniff_csv_dialect,
    read_csv_sample,
    coerce_number,
)


def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _normalize_artifact_index(entries):
    normalized = []
    for item in entries or []:
        if isinstance(item, dict) and item.get("path"):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append({"path": item, "artifact_type": "artifact"})
    return normalized

def _first_artifact_path(entries, artifact_type: str):
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        if item.get("artifact_type") == artifact_type and item.get("path"):
            return item.get("path")
    return None

def _facts_from_insights(insights: Dict[str, Any], max_items: int = 8):
    if not isinstance(insights, dict):
        return []
    metrics = insights.get("metrics_summary")
    facts = []
    if isinstance(metrics, list):
        for item in metrics:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric")
            value = item.get("value")
            if metric is None or value is None:
                continue
            facts.append({"source": "insights.json", "metric": metric, "value": value, "labels": {}})
            if len(facts) >= max_items:
                break
    deployment = insights.get("deployment_recommendation")
    confidence = insights.get("confidence")
    if deployment:
        facts.append(
            {
                "source": "insights.json",
                "metric": "deployment_recommendation",
                "value": deployment,
                "labels": {"confidence": confidence or ""},
            }
        )
    return facts

def _safe_load_csv(path: str, max_rows: int = 200):
    try:
        dialect = load_output_dialect() or sniff_csv_dialect(path)
        sample = read_csv_sample(path, dialect, max_rows)
        if not sample:
            return None
        columns = sample.get("columns", [])
        if isinstance(columns, list) and len(columns) == 1 and ";" in columns[0]:
            sniffed = sniff_csv_dialect(path)
            if sniffed.get("sep") != dialect.get("sep"):
                sample = read_csv_sample(path, sniffed, max_rows)
        return sample
    except Exception:
        return None

def _summarize_numeric_columns(rows: List[Dict[str, Any]], columns: List[str], decimal: str, max_cols: int = 12):
    numeric_summary = {}
    for col in columns:
        values = []
        for row in rows:
            raw = row.get(col)
            if raw is None or raw == "":
                continue
            num = coerce_number(raw, decimal)
            if num is not None:
                values.append(num)
        if values:
            numeric_summary[col] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "n": len(values),
            }
        if len(numeric_summary) >= max_cols:
            break
    return numeric_summary

def _truncate_cell(text: str, max_len: int) -> str:
    cleaned = str(text).replace("\n", " ").replace("\r", " ").strip()
    cleaned = cleaned.replace("|", "/")
    if len(cleaned) <= max_len:
        return cleaned
    suffix = " [cut]"
    if max_len <= len(suffix):
        return cleaned[:max_len]
    return cleaned[: max(0, max_len - len(suffix))] + suffix

def render_table_text(headers: List[str], rows: List[List[str]], max_rows: int = 8, max_cell_len: int = 28) -> str:
    if not headers or not rows:
        return "No data available."
    safe_rows = []
    for row in rows[:max_rows]:
        if not isinstance(row, list):
            continue
        safe_rows.append([_truncate_cell(cell, max_cell_len) for cell in row])
    if not safe_rows:
        return "No data available."
    widths = []
    for idx, header in enumerate(headers):
        header_text = _truncate_cell(header, max_cell_len)
        col_width = len(header_text)
        for row in safe_rows:
            if idx < len(row):
                col_width = max(col_width, len(row[idx]))
        widths.append(min(col_width, max_cell_len))
    header_line = "  ".join(
        _truncate_cell(header, max_cell_len).ljust(widths[idx]) for idx, header in enumerate(headers)
    )
    sep_line = "  ".join("-" * widths[idx] for idx in range(len(widths)))
    data_lines = []
    for row in safe_rows:
        padded = []
        for idx, width in enumerate(widths):
            cell = row[idx] if idx < len(row) else ""
            padded.append(_truncate_cell(cell, max_cell_len).ljust(width))
        data_lines.append("  ".join(padded))
    return "\n".join([header_line, sep_line] + data_lines)

def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> List[tuple[str, Any]]:
    items: List[tuple[str, Any]] = []
    if not isinstance(metrics, dict):
        return items
    for key, value in metrics.items():
        metric_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.extend(_flatten_metrics(value, f"{metric_key}."))
        else:
            items.append((metric_key, value))
    return items

def _select_informative_columns(sample: Dict[str, Any], max_cols: int = 8, min_cols: int = 5) -> List[str]:
    if not isinstance(sample, dict):
        return []
    columns = sample.get("columns", []) or []
    rows = sample.get("rows", []) or []
    decimal = (sample.get("dialect_used") or {}).get("decimal") or "."
    if not columns:
        return []
    if not rows:
        return columns[:max_cols]
    stats = []
    for col in columns:
        values = []
        for row in rows[:50]:
            raw = row.get(col)
            if raw not in (None, ""):
                values.append(raw)
        if not values:
            continue
        numeric_hits = sum(1 for val in values if coerce_number(val, decimal) is not None)
        unique_count = len({str(val) for val in values})
        ratio = numeric_hits / max(len(values), 1)
        stats.append(
            {
                "col": col,
                "numeric_ratio": ratio,
                "unique_count": unique_count,
                "non_null": len(values),
            }
        )
    numeric_cols = [item for item in stats if item["numeric_ratio"] >= 0.6]
    cat_cols = [item for item in stats if item["numeric_ratio"] < 0.6]
    numeric_cols.sort(key=lambda item: item["non_null"], reverse=True)
    cat_cols.sort(key=lambda item: item["unique_count"])
    selected = [item["col"] for item in numeric_cols[:4]]
    selected.extend([item["col"] for item in cat_cols[:4] if item["col"] not in selected])
    if len(selected) < min_cols:
        for col in columns:
            if col not in selected:
                selected.append(col)
            if len(selected) >= min_cols:
                break
    return selected[:max_cols]

def _select_scored_columns(sample: Dict[str, Any], max_cols: int = 6) -> List[str]:
    if not isinstance(sample, dict):
        return []
    columns = sample.get("columns", []) or []
    if not columns:
        return []
    tokens = ["pred", "prob", "score", "segment", "cluster", "group", "rank", "risk", "expected", "optimal", "recommend"]
    preferred = []
    for col in columns:
        norm = col.lower()
        if any(tok in norm for tok in tokens):
            preferred.append(col)
    for col in columns:
        if col.lower() in {"row_id", "case_id", "caseid", "id"} and col not in preferred:
            preferred.insert(0, col)
    if not preferred:
        return _select_informative_columns(sample, max_cols=max_cols, min_cols=min(5, max_cols))
    deduped = []
    for col in preferred:
        if col not in deduped:
            deduped.append(col)
    return deduped[:max_cols]

def _rows_from_sample(sample: Dict[str, Any], columns: List[str], max_rows: int = 5) -> List[List[str]]:
    rows = sample.get("rows", []) if isinstance(sample, dict) else []
    out = []
    for row in rows[:max_rows]:
        out.append([str(row.get(col, "")) for col in columns])
    return out

def _metrics_table(metrics_payload: Dict[str, Any], max_items: int = 10) -> str:
    if not isinstance(metrics_payload, dict):
        return "No data available."
    model_perf = metrics_payload.get("model_performance") if isinstance(metrics_payload.get("model_performance"), dict) else {}
    items: List[tuple[str, str]] = []
    if isinstance(model_perf, dict):
        for key, value in model_perf.items():
            if isinstance(value, dict) and {"mean", "ci_lower", "ci_upper"}.issubset(value.keys()):
                mean = value.get("mean")
                lower = value.get("ci_lower")
                upper = value.get("ci_upper")
                items.append((f"model_performance.{key}", f"mean={mean} ci=[{lower}, {upper}]"))
            elif _is_number(value):
                items.append((f"model_performance.{key}", str(value)))
            if len(items) >= max_items:
                break
    if len(items) < max_items:
        flat = _flatten_metrics(metrics_payload)
        for key, value in flat:
            if len(items) >= max_items:
                break
            if key.startswith("model_performance."):
                continue
            if _is_number(value):
                items.append((key, str(value)))
    if not items:
        return "No data available."
    rows = [[metric, val] for metric, val in items[:max_items]]
    return render_table_text(["metric", "value"], rows, max_rows=max_items)

def _recommendations_table(preview: Dict[str, Any], max_rows: int = 3) -> str:
    if not isinstance(preview, dict):
        return "No data available."
    items = preview.get("items")
    if not isinstance(items, list) or not items:
        return "No data available."
    keys = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in item.keys():
            keys[key] = keys.get(key, 0) + 1
    sorted_keys = [k for k, _ in sorted(keys.items(), key=lambda kv: kv[1], reverse=True)]
    headers = sorted_keys[:4]
    if not headers:
        return "No data available."
    rows = []
    for item in items[:max_rows]:
        if not isinstance(item, dict):
            continue
        rows.append([str(item.get(key, "")) for key in headers])
    return render_table_text(headers, rows, max_rows=max_rows)

def _pick_top_examples(rows: List[Dict[str, Any]], columns: List[str], value_keys: List[str], label_keys: List[str], decimal: str, max_rows: int = 3):
    if not rows or not columns:
        return None
    value_key = None
    for key in value_keys:
        if key in columns:
            value_key = key
            break
    if not value_key:
        return None
    def _coerce_num(row):
        raw = row.get(value_key)
        if raw is None or raw == "":
            return None
        return coerce_number(raw, decimal)
    scored = []
    for row in rows:
        val = _coerce_num(row)
        if val is None:
            continue
        scored.append((val, row))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    examples = []
    for val, row in scored[:max_rows]:
        example = {"value_key": value_key, "value": val}
        for label in label_keys:
            if label in row and row.get(label) not in (None, ""):
                example[label] = row.get(label)
        examples.append(example)
    return examples

def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _extract_lift_value(data_adequacy_report: Dict[str, Any], metrics_payload: Dict[str, Any]) -> Optional[float]:
    if isinstance(data_adequacy_report, dict):
        signals = data_adequacy_report.get("signals", {})
        for key in ("classification_lift", "regression_lift", "f1_lift", "mae_lift"):
            lift = _coerce_float(signals.get(key)) if isinstance(signals, dict) else None
            if lift is not None:
                return lift
    if isinstance(metrics_payload, dict):
        model_perf = metrics_payload.get("model_performance") if isinstance(metrics_payload.get("model_performance"), dict) else {}
        for key, value in model_perf.items():
            if "lift" in str(key).lower():
                lift = _coerce_float(value if not isinstance(value, dict) else value.get("mean"))
                if lift is not None:
                    return lift
    return None


def _derive_exec_decision(
    review_verdict: Optional[str],
    data_adequacy_report: Dict[str, Any],
    metrics_payload: Dict[str, Any],
) -> str:
    verdict = str(review_verdict or "").upper()
    status = str(data_adequacy_report.get("status") if isinstance(data_adequacy_report, dict) else "").lower()
    lift = _extract_lift_value(data_adequacy_report, metrics_payload)

    if verdict in {"NEEDS_IMPROVEMENT", "REJECTED"}:
        return "NO_GO"
    if status in {"unknown", "insufficient_signal"}:
        return "GO_WITH_LIMITATIONS"
    if status == "data_limited":
        if lift is not None and lift <= 0:
            return "NO_GO"
        return "GO_WITH_LIMITATIONS"
    if status == "sufficient_signal":
        if lift is None:
            return "GO_WITH_LIMITATIONS"
        return "GO" if lift > 0 else "NO_GO"
    return "GO_WITH_LIMITATIONS"


def _sanitize_report_text(text: str) -> str:
    if not text:
        return text
    while "..." in text:
        text = text.replace("...", ".")
    return text

def _extract_numeric_metrics(metrics: Dict[str, Any], max_items: int = 8):
    if not isinstance(metrics, dict):
        return []
    items = []
    for key, value in metrics.items():
        if _is_number(value):
            items.append((str(key), float(value)))
        if len(items) >= max_items:
            break
    return items

def _build_fact_cards(case_summary_ctx, scored_rows_ctx, weights_ctx, data_adequacy_ctx, max_items: int = 8):
    facts = []
    if isinstance(case_summary_ctx, dict):
        examples = case_summary_ctx.get("examples") or []
        for example in examples:
            value_key = example.get("value_key")
            value = example.get("value")
            if value_key and _is_number(value):
                labels = {k: v for k, v in example.items() if k not in {"value_key", "value"}}
                facts.append({
                    "source": "case_summary",
                    "metric": value_key,
                    "value": float(value),
                    "labels": labels,
                })
    if isinstance(scored_rows_ctx, dict):
        examples = scored_rows_ctx.get("examples") or []
        for example in examples:
            value_key = example.get("value_key")
            value = example.get("value")
            if value_key and _is_number(value):
                labels = {k: v for k, v in example.items() if k not in {"value_key", "value"}}
                facts.append({
                    "source": "predictions",
                    "metric": value_key,
                    "value": float(value),
                    "labels": labels,
                })
    if isinstance(weights_ctx, dict):
        for key in ("metrics", "classification", "regression", "propensity_model", "price_model"):
            metrics = weights_ctx.get(key)
            for metric_key, metric_val in _extract_numeric_metrics(metrics):
                facts.append({
                    "source": "weights",
                    "metric": metric_key,
                    "value": metric_val,
                    "labels": {"model_block": key},
                })
    if isinstance(data_adequacy_ctx, dict):
        signals = data_adequacy_ctx.get("signals", {})
        for metric_key, metric_val in _extract_numeric_metrics(signals):
            facts.append({
                "source": "data_adequacy_report.json",
                "metric": metric_key,
                "value": metric_val,
                "labels": {},
            })
    return facts[:max_items]

class BusinessTranslatorAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Business Translator Agent with Gemini 3 Flash.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
            generation_config={"temperature": 0.2},  # Low temp for evidence-based executive reports
        )
        self.last_prompt = None
        self.last_response = None

    def generate_report(
        self,
        state: Dict[str, Any],
        error_message: Optional[str] = None,
        has_partial_visuals: bool = False,
        plots: Optional[List[str]] = None,
        translator_view: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not isinstance(state, dict):
            state = {"execution_output": str(state), "business_objective": str(error_message or "")}
            error_message = None

        # Sanitize Visuals Context
        has_partial_visuals = bool(has_partial_visuals)
        plots = [str(p).replace("\\", "/") for p in (plots or []) if p]
        plot_reference_mode = "inline"
        if any(not p.startswith("static/plots/") for p in plots):
            plot_reference_mode = "figure_only"
        artifact_index = _normalize_artifact_index(
            state.get("artifact_index") or _safe_load_json("data/produced_artifact_index.json") or []
        )
        translator_view = translator_view or state.get("translator_view") or {}
        view_policy = translator_view.get("reporting_policy") if isinstance(translator_view, dict) else None
        view_inventory = translator_view.get("evidence_inventory") if isinstance(translator_view, dict) else None
        view_constraints = translator_view.get("constraints") if isinstance(translator_view, dict) else None
        if not isinstance(view_inventory, list) or not view_inventory:
            view_inventory = artifact_index

        def _artifact_available(path: str) -> bool:
            if artifact_index:
                return any(item.get("path") == path for item in artifact_index if isinstance(item, dict))
            return os.path.exists(path)
        
        # Safe extraction of strategy info
        strategy = state.get('selected_strategy', {})
        strategy_title = strategy.get('title', 'General Analysis')
        hypothesis = strategy.get('hypothesis', 'N/A')
        analysis_type = str(strategy.get('analysis_type', 'N/A'))
        
        # Review content
        review_verdict = state.get("review_verdict")
        if review_verdict:
            compliance = review_verdict
        else:
            review = state.get('review_feedback', {})
            if isinstance(review, dict):
                compliance = review.get('status', 'PENDING')
            else:
                # If it's a string (e.g. just the feedback text from older legacy flows or simple strings)
                compliance = "REVIEWED" if review else "PENDING"
        
        # Construct JSON Context for Visuals safely using json library
        visuals_context_data = {
            "has_partial_visuals": has_partial_visuals,
            "plots_count": len(plots),
            "plots_list": plots,
            "plot_reference_mode": plot_reference_mode,
        }
        visuals_context_json = json.dumps(visuals_context_data, ensure_ascii=False)
        contract = _safe_load_json("data/execution_contract.json") or {}
        decisioning_context = translator_view.get("decisioning_requirements") or contract.get("decisioning_requirements") or {}
        if not isinstance(decisioning_context, dict):
            decisioning_context = {}
        decisioning_context_json = json.dumps(decisioning_context, ensure_ascii=False)
        decisioning_columns = [
            str(col.get("name"))
            for col in (decisioning_context.get("output", {}).get("required_columns") or [])
            if isinstance(col, dict) and col.get("name")
        ]
        decisioning_columns_text = ", ".join(decisioning_columns) if decisioning_columns else "None requested."
        
        # Load optional artifacts for context
        integrity_audit = _safe_load_json("data/integrity_audit_report.json") or {}
        output_contract_report = _safe_load_json("data/output_contract_report.json") or {}
        case_alignment_report = _safe_load_json("data/case_alignment_report.json") or {}
        data_adequacy_report = _safe_load_json("data/data_adequacy_report.json") or {}
        alignment_check_report = _safe_load_json("data/alignment_check.json") or {}
        plot_insights = _safe_load_json("data/plot_insights.json") or {}
        insights = _safe_load_json("data/insights.json") or {}
        steward_summary = _safe_load_json("data/steward_summary.json") or {}
        cleaning_manifest = _safe_load_json("data/cleaning_manifest.json") or {}
        run_summary = _safe_load_json("data/run_summary.json") or {}
        recommendations_preview = _safe_load_json("reports/recommendations_preview.json") or {}
        metrics_payload = _safe_load_json("data/metrics.json") or {}
        weights_path = _first_artifact_path(artifact_index, "weights")
        predictions_path = _first_artifact_path(artifact_index, "predictions")
        weights_payload = _safe_load_json(weights_path) if weights_path else None
        weights_payload = weights_payload or {}
        scored_rows = _safe_load_csv(predictions_path) if predictions_path else None
        case_summary = None
        cleaned_path = _first_artifact_path(artifact_index, "dataset") or "data/cleaned_data.csv"
        cleaned_rows = _safe_load_csv(cleaned_path, max_rows=100) if _artifact_available(cleaned_path) else None
        business_objective = state.get("business_objective") or contract.get("business_objective") or ""
        executive_decision_label = _derive_exec_decision(
            review_verdict or compliance,
            data_adequacy_report,
            metrics_payload,
        )

        def _summarize_integrity():
            issues = integrity_audit.get("issues", []) if isinstance(integrity_audit, dict) else []
            severity_counts = {}
            for i in issues:
                sev = str(i.get("severity", "unknown"))
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            top = issues[:3]
            return f"Issues by severity: {severity_counts}; Top3: {top}"

        def _summarize_contract():
            if not contract:
                return "No execution contract."
            return {
                "strategy_title": contract.get("strategy_title"),
                "business_objective": contract.get("business_objective"),
                "required_outputs": contract.get("required_outputs", []),
                "validations": contract.get("validations", []),
                "quality_gates": contract.get("quality_gates", {}),
                "business_alignment": contract.get("business_alignment", {}),
                "spec_extraction": contract.get("spec_extraction", {}),
                "iteration_policy": contract.get("iteration_policy", {}),
                "execution_plan": contract.get("execution_plan", {}),
            }

        def _summarize_output_contract():
            if not output_contract_report:
                return "No output contract report."
            miss = output_contract_report.get("missing", [])
            present = output_contract_report.get("present", [])
            return f"Outputs present={len(present)} missing={len(miss)}"

        def _summarize_steward():
            if not steward_summary:
                return "No steward summary."
            summary = steward_summary.get("summary", "")
            encoding = steward_summary.get("encoding")
            sep = steward_summary.get("sep")
            decimal = steward_summary.get("decimal")
            return {
                "summary": summary[:1200],
                "encoding": encoding,
                "sep": sep,
                "decimal": decimal,
                "file_size_bytes": steward_summary.get("file_size_bytes"),
            }

        def _summarize_cleaning():
            if not cleaning_manifest:
                return "No cleaning manifest."
            row_counts = cleaning_manifest.get("row_counts", {})
            conversions = cleaning_manifest.get("conversions", {})
            dropped = cleaning_manifest.get("dropped_rows", {})
            conversion_keys = []
            if isinstance(conversions, dict):
                conversion_keys = list(conversions.keys())[:12]
            elif isinstance(conversions, list):
                conversion_keys = [c.get("column") for c in conversions if isinstance(c, dict) and c.get("column")]
                conversion_keys = conversion_keys[:12]
            return {
                "row_counts": row_counts,
                "dropped_rows": dropped,
                "conversion_keys": conversion_keys,
            }

        def _summarize_weights():
            if not weights_payload:
                return "No weights/metrics payload."
            if not isinstance(weights_payload, dict):
                return weights_payload
            summary = {}
            for key in ("metrics", "weights", "propensity_model", "price_model", "optimization", "regression", "classification"):
                if key in weights_payload:
                    summary[key] = weights_payload.get(key)
            if not summary:
                summary["keys"] = list(weights_payload.keys())[:12]
            return summary

        def _summarize_model_metrics():
            metrics_summary = {}
            if isinstance(weights_payload, dict):
                for key in ("metrics", "propensity_model", "price_model", "optimization", "regression", "classification"):
                    if key in weights_payload:
                        metrics_summary[key] = weights_payload.get(key)
            if isinstance(run_summary, dict):
                run_metrics = run_summary.get("metrics")
                if run_metrics:
                    metrics_summary["run_summary_metrics"] = run_metrics
            if isinstance(insights, dict):
                metrics_summary["insights_metrics"] = insights.get("metrics_summary", [])
            if not metrics_summary:
                return "No explicit model metrics found."
            return metrics_summary

        def _summarize_case_summary():
            if not case_summary:
                return "No case summary artifact."
            columns = case_summary.get("columns", [])
            rows = case_summary.get("rows", [])
            decimal = (case_summary.get("dialect_used") or {}).get("decimal") or "."
            numeric_summary = _summarize_numeric_columns(rows, columns, decimal)
            examples = _pick_top_examples(rows, columns, value_keys=columns, label_keys=columns, decimal=decimal)
            return {
                "row_count_sampled": case_summary.get("row_count_sampled", 0),
                "columns": columns,
                "numeric_summary": numeric_summary,
                "examples": examples,
            }

        def _summarize_scored_rows():
            if not scored_rows:
                return "No predictions artifact."
            columns = scored_rows.get("columns", [])
            rows = scored_rows.get("rows", [])
            decimal = (scored_rows.get("dialect_used") or {}).get("decimal") or "."
            numeric_summary = _summarize_numeric_columns(rows, columns, decimal)
            examples = _pick_top_examples(rows, columns, value_keys=columns, label_keys=columns, decimal=decimal)
            return {
                "row_count_sampled": scored_rows.get("row_count_sampled", 0),
                "columns": columns,
                "numeric_summary": numeric_summary,
                "examples": examples,
            }

        def _summarize_run():
            if not run_summary:
                return "No run_summary.json."
            return {
                "status": run_summary.get("status"),
                "run_outcome": run_summary.get("run_outcome"),
                "failed_gates": run_summary.get("failed_gates", []),
                "warnings": run_summary.get("warnings", []),
                "metrics": run_summary.get("metrics", {}),
                "metric_ceiling_detected": run_summary.get("metric_ceiling_detected"),
                "ceiling_reason": run_summary.get("ceiling_reason"),
                "baseline_vs_model": run_summary.get("baseline_vs_model", []),
            }

        def _summarize_gate_context():
            gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context") or {}
            if not gate_context:
                return "No gate context."
            if isinstance(gate_context, dict):
                return {
                    "source": gate_context.get("source"),
                    "status": gate_context.get("status"),
                    "failed_gates": gate_context.get("failed_gates", []),
                    "required_fixes": gate_context.get("required_fixes", []),
                }
            return str(gate_context)[:1200]

        def _summarize_review_feedback():
            feedback = state.get("review_feedback") or state.get("execution_feedback") or ""
            if isinstance(feedback, dict):
                return feedback
            if isinstance(feedback, str):
                return feedback[:2000]
            return str(feedback)[:2000]

        def _summarize_data_adequacy():
            if not data_adequacy_report:
                return "No data adequacy report."
            return {
                "status": data_adequacy_report.get("status"),
                "reasons": data_adequacy_report.get("reasons", []),
                "recommendations": data_adequacy_report.get("recommendations", []),
                "signals": data_adequacy_report.get("signals", {}),
                "quality_gates_alignment": data_adequacy_report.get("quality_gates_alignment", {}),
                "consecutive_data_limited": data_adequacy_report.get("consecutive_data_limited"),
                "data_limited_threshold": data_adequacy_report.get("data_limited_threshold"),
                "threshold_reached": data_adequacy_report.get("threshold_reached"),
            }

        def _summarize_alignment_check():
            if not alignment_check_report:
                return "No alignment check."
            return {
                "status": alignment_check_report.get("status"),
                "failure_mode": alignment_check_report.get("failure_mode"),
                "summary": alignment_check_report.get("summary"),
                "requirements": alignment_check_report.get("requirements", []),
            }

        def _summarize_case_alignment():
            if not case_alignment_report:
                return "No case alignment report."
            status = case_alignment_report.get("status")
            failures = case_alignment_report.get("failures", [])
            metrics = case_alignment_report.get("metrics", {})
            thresholds = case_alignment_report.get("thresholds", {})
            return f"Status={status}; Failures={failures}; KeyMetrics={metrics}"

        def _case_alignment_business_status():
            if not case_alignment_report:
                return {
                    "label": "NO_DATA",
                    "status": "UNKNOWN",
                    "message": "No se encontró reporte de alineación de casos.",
                    "recommendation": "Revisar si el proceso generó data/case_alignment_report.json.",
                }
            status = case_alignment_report.get("status")
            failures = case_alignment_report.get("failures", [])
            metrics = case_alignment_report.get("metrics", {})
            thresholds = case_alignment_report.get("thresholds", {})
            if status == "SKIPPED":
                return {
                    "label": "PENDIENTE_DEFINICION_GATES",
                    "status": "SKIPPED",
                    "message": "No se definieron gates de alineación de casos en el contrato.",
                    "recommendation": "Definir métricas y umbrales en el contrato para evaluar preparación de negocio.",
                }
            if status == "PASS":
                return {
                    "label": "APTO_CONDICIONAL",
                    "status": "PASS",
                    "message": "La alineación con la lógica de casos cumple los umbrales definidos.",
                    "key_metrics": metrics,
                }
            # FAIL
            details = []
            for failure in failures:
                metric_val = metrics.get(failure)
                thresh_val = thresholds.get(failure.replace("case_means", "min"), thresholds.get(failure))
                if metric_val is not None:
                    details.append(f"{failure}={metric_val} (umbral={thresh_val})")
                else:
                    details.append(f"{failure} (umbral={thresh_val})")
            return {
                "label": "NO_APTO_PARA_PRODUCCION",
                "status": "FAIL",
                "message": "La solución no cumple los criterios de alineación por casos.",
                "details": details,
                "recommendation": "Priorizar reducción de violaciones entre casos antes de considerar producción.",
            }

        contract_context = _summarize_contract()
        integrity_context = _summarize_integrity()
        output_contract_context = _summarize_output_contract()
        case_alignment_context = _summarize_case_alignment()
        case_alignment_business_status = _case_alignment_business_status()
        alignment_check_context = _summarize_alignment_check()
        gate_context = _summarize_gate_context()
        review_feedback_context = _summarize_review_feedback()
        steward_context = _summarize_steward()
        cleaning_context = _summarize_cleaning()
        weights_context = _summarize_weights()
        case_summary_context = _summarize_case_summary()
        scored_rows_context = _summarize_scored_rows()
        run_summary_context = _summarize_run()
        data_adequacy_context = _summarize_data_adequacy()
        model_metrics_context = _summarize_model_metrics()
        facts_context = _facts_from_insights(insights) or _build_fact_cards(case_summary_context, scored_rows_context, weights_context, data_adequacy_context)
        artifacts_context = view_inventory if view_inventory else []
        evidence_paths = []
        for item in artifacts_context or []:
            if isinstance(item, dict) and item.get("path"):
                evidence_paths.append(str(item.get("path")))
            elif isinstance(item, str):
                evidence_paths.append(str(item))
        evidence_paths = [p for idx, p in enumerate(evidence_paths) if p and p not in evidence_paths[:idx]]
        evidence_paths = evidence_paths[:8]
        reporting_policy_context = view_policy if isinstance(view_policy, dict) else {}
        if not reporting_policy_context:
            reporting_policy_context = contract.get("reporting_policy", {}) if isinstance(contract, dict) else {}
        translator_view_context = translator_view if isinstance(translator_view, dict) else {}
        slot_payloads = {}
        if isinstance(insights, dict):
            slot_payloads = insights.get("slot_payloads") or {}
            if not slot_payloads:
                if insights.get("metrics_summary"):
                    slot_payloads["model_metrics"] = insights.get("metrics_summary")
                if insights.get("predictions_summary"):
                    slot_payloads["predictions_overview"] = insights.get("predictions_summary")
                if insights.get("segment_pricing_summary"):
                    slot_payloads["segment_pricing"] = insights.get("segment_pricing_summary")
                if insights.get("leakage_audit"):
                    slot_payloads["alignment_risks"] = insights.get("leakage_audit")

        slot_defs = reporting_policy_context.get("slots", []) if isinstance(reporting_policy_context, dict) else []
        missing_required_slots = []
        if isinstance(slot_defs, list):
            for slot in slot_defs:
                if not isinstance(slot, dict):
                    continue
                if slot.get("mode") != "required":
                    continue
                slot_id = slot.get("id")
                insights_key = slot.get("insights_key")
                payload = slot_payloads.get(slot_id) if slot_id else None
                if payload is None and insights_key and isinstance(insights, dict):
                    payload = insights.get(insights_key)
                if payload:
                    continue
                missing_required_slots.append(
                    {"id": slot_id, "sources": slot.get("sources", []), "insights_key": insights_key}
                )
        slot_coverage_context = {
            "slot_payloads": slot_payloads,
            "missing_required_slots": missing_required_slots,
        }

        cleaned_sample_table_text = "No data available."
        if isinstance(cleaned_rows, dict) and cleaned_rows.get("rows"):
            cleaned_cols = _select_informative_columns(cleaned_rows, max_cols=8, min_cols=5)
            cleaned_rows_list = _rows_from_sample(cleaned_rows, cleaned_cols, max_rows=5)
            cleaned_sample_table_text = render_table_text(cleaned_cols, cleaned_rows_list, max_rows=5)

        scored_sample_table_text = "No data available."
        if isinstance(scored_rows, dict) and scored_rows.get("rows"):
            scored_cols = _select_scored_columns(scored_rows, max_cols=6)
            scored_rows_list = _rows_from_sample(scored_rows, scored_cols, max_rows=5)
            scored_sample_table_text = render_table_text(scored_cols, scored_rows_list, max_rows=5)

        metrics_table_text = _metrics_table(metrics_payload, max_items=10)
        recommendations_table_text = _recommendations_table(recommendations_preview, max_rows=3)
        evidence_paths_text = "\n".join(f"- {path}" for path in evidence_paths) if evidence_paths else "No data available."

        # Define Template
        SYSTEM_PROMPT_TEMPLATE = Template("""
        You are a Senior Executive Translator and Data Storyteller.
        Your goal is to translate technical outputs into a decision-ready business narrative.
        
        TONE: Professional, evidence-driven, decisive. Avoid unnecessary jargon.
        STYLE: Prioritize decision, evidence, risks, and next actions. No fluff.
        
        *** FORMATTING CONSTRAINTS (CRITICAL) ***
        1. **LANGUAGE:** DETECT the language of the 'Business Objective' in the state. GENERATE THE REPORT IN THAT SAME LANGUAGE. (If objective is Spanish, output Spanish).
        2. **NO MARKDOWN TABLES:** The PDF generator breaks on tables. DO NOT use table syntax. Use bulleted lists, key-value pairs, or the provided fixed-width text tables (no pipes).
           - Bad: | Metric | Value |
           - Good: 
             * Metric: Value
        3. **NO ELLIPSIS:** Never output the literal sequence "..." anywhere in the report.
        4. **NO TRUNCATED SENTENCES:** Do not leave sentences hanging; rewrite as complete, concise sentences.
        
        CONTEXT:
        - Business Objective: $business_objective
        - Strategy: $strategy_title
        - Hypothesis: $hypothesis
        - Compliance Check: $compliance
        - Executive Decision Label (deterministic): $executive_decision_label
        - Contract: $contract_context
        - Integrity Audit: $integrity_context
        - Output Contract: $output_contract_context
        - Case Alignment QA: $case_alignment_context
        - Business Readiness (Case Alignment): $case_alignment_business_status
        - Alignment Check: $alignment_check_context
        - Gate Context: $gate_context
        - Review Feedback: $review_feedback
        - Steward Summary: $steward_context
        - Cleaning Summary: $cleaning_context
        - Run Summary: $run_summary_context
        - Data Adequacy: $data_adequacy_context
        - Data Adequacy Report (verbatim): $data_adequacy_report_json
        - Insights (primary): $insights_context
        - Fact Cards (use as evidence): $facts_context
        - Model Metrics & Weights: $weights_context
        - Model Metrics (Expanded): $model_metrics_context
        - Case Summary Snapshot: $case_summary_context
        - Scored Rows Snapshot: $scored_rows_context
        - Plot Insights (data-driven): $plot_insights_json
        - Artifacts Available: $artifacts_context
        - Recommendations Preview: $recommendations_preview_context
        - Cleaned Data Sample Table (text): $cleaned_sample_table_text
        - Scored Rows Sample Table (text): $scored_sample_table_text
        - Metrics Table (text): $metrics_table_text
        - Recommendations Table (text): $recommendations_table_text
        - Evidence Paths (max 8): $evidence_paths_text
        - Reporting Policy: $reporting_policy_context
        - Translator View: $translator_view_context
        - Slot Payloads: $slot_payloads_context
        - Slot Coverage: $slot_coverage_context
        - Decisioning Requirements (json): $decisioning_context_json
        - Decisioning Columns (text): $decisioning_columns_text

        GUIDANCE:
        - Use Insights as the primary evidence source; only reference other artifacts if they add clear value.
        - Include 2-4 short "tablas de muestra" using the provided text tables. Label them explicitly as samples and keep 3-5 rows maximum. Do NOT use markdown tables or pipes; use the text tables as-is.
        - If reporting_policy.demonstrative_examples_enabled is true AND run_outcome is in reporting_policy.demonstrative_examples_when_outcome_in,
          you MUST include a section titled "Ejemplos ilustrativos (no aptos para producción)".
          Use recommendations_preview.items (max 3-5) and include strong disclaimers plus support (n, observed_support if available).
          If items are empty, explain why using recommendations_preview.reason and mention which artifact was missing.
        - If run_outcome is GO, you may include a short "Recommendations Snapshot" section if recommendations_preview.items exists,
          without the "illustrative" labeling.
        - For every plot you mention, include 1-3 concrete findings (n/%/top categories) drawn from plot_insights_json. If no quantitative insights are available for a plot, explicitly say "No se dispone de insights cuantitativos para este gráfico" (or equivalent in the target language).
        - Do NOT invent segment/category names; only mention names that appear in facts, insights, or plot_insights_json.
        - Data Adequacy must be reported verbatim: status, up to 3 reason tags, and up to 3 recommendations from data_adequacy_report_json. Do not rephrase status (e.g., keep "sufficient_signal" as-is).
        - If data_adequacy_report_json reasons include classification_baseline_missing or regression_baseline_missing, call it out as a limitation and advise adding Dummy baselines.
        - Plot paths must use forward slashes. If VISUALS CONTEXT plot_reference_mode is "figure_only", reference plots as "Figure: <filename>" instead of inline markdown images.
        - Slot-driven reporting:
          For each slot in reporting_policy.slots:
            * if mode == "required": include it using evidence (payload or artifact); if missing => write "No disponible" and cite missing sources.
            * if mode == "conditional" or "optional": include only if payload exists; otherwise omit without inventing.
          Structure the report using reporting_policy.sections order; do not add top-level sections outside that list, EXCEPT the final mandatory section titled "Evidencia usada".
        - Decision Policy / Actions:
          If DECISIONING REQUIREMENTS (json) indicates enabled=true, add a dedicated section titled "Decision Policy / Actions" after "Evidence & Metrics".
          Describe each required decision column (name, type, role, derivation logic) using the JSON context and mention how the column supports prioritized actions or flags.
          Reference the Scored Rows Sample Table to show concrete values for those columns. Do not invent values nor attributes beyond what the sample shows.

        ERROR CONDITION:
        $error_condition
        
        VISUALS CONTEXT (JSON):
        $visuals_context_json

        ERROR LOGIC (HIGHEST PRIORITY):
        Determine execution status:
        - ERROR if "$error_condition" != "No critical errors." OR Run Summary indicates FAIL.
        - SUCCESS otherwise.

        IF ERROR DETECTED:
        1. Title: "EXECUTION FAILURE REPORT" (in the target language).
        2. Status Line: START with "⛔ BLOCKED / FALLO CRÍTICO".
        3. Explain the failure based on error_condition and run_summary (non-technical, executive tone).
        4. State which critical artifacts are missing and why that blocks a decision (use Evidence Paths + artifact_index).
        5. If VISUALS CONTEXT indicates plots_list is non-empty:
           - Include a short subsection "Visual evidence available" and list each plot.
           - If plot_reference_mode is "figure_only", reference as "Figure: <filename>" instead of inline images.
           - Otherwise embed each plot using forward slashes:
             ![<filename>](<exact_path_from_plots_list>)
           - Under each plot add 1 bullet: what it suggests (or "No disponible" if unknown).
           - Do NOT claim "no visualizations" when plots_list is non-empty.
        6. Close with 2-4 "Next actions" that would unblock the pipeline (data / pipeline / validation).

        IF SUCCESS (Only if NO Error):
        OUTPUT FORMAT (MANDATORY):
        - You MUST generate ONLY the top-level sections listed in reporting_policy.sections, in that exact order.
        - Each section MUST be a level-2 heading: "## <Section Title>".
        - You MAY use short subheadings or bullet lists inside a section, but do NOT add new top-level sections.
        - Exception: you WILL add the final mandatory top-level section "## Evidencia usada" at the end (see later instruction).

        SECTION TITLE GUIDELINES:
        - decision -> Executive Decision
        - objective_approach -> Objective & Approach
        - evidence_metrics -> Evidence & Metrics
        - business_impact -> Business Impact
        - risks_limitations -> Risks & Limitations
        - next_actions -> Recommended Next Actions
        - visual_insights -> Visual Insights
        - If a section key is not listed above: convert it to Title Case (replace "_" with " ").

        SLOT-TO-SECTION GUIDELINES (use best judgment; do NOT invent):
        - model_metrics, predictions_overview, forecast_summary, ranking_top -> Evidence & Metrics
        - explainability, error_analysis -> Evidence & Metrics (or Risks & Limitations if negative)
        - alignment_risks -> Risks & Limitations
        - segment_pricing -> Business Impact (and/or Recommended Next Actions if it contains recommendations)

        REQUIRED CONTENT BY SECTION (adapt to the objective and available evidence):
        - Executive Decision:
          * First line: readiness (GO / GO_WITH_LIMITATIONS / NO_GO) + 1-sentence reason.
          * Use the Executive Decision Label (deterministic) as the readiness label.
          * Ground the reason in gates/reviewer verdict/alignment check; if a required artifact is missing, downgrade readiness.
        - Objective & Approach:
          * Restate the business objective in plain language.
          * Summarize the chosen strategy and the minimum viable method used (segmentation + model + optimization).
          * Briefly describe the data preparation at a high level (no code, no hallucinations).
        - Evidence & Metrics:
          * Cite at least 3 concrete numbers (metrics, counts, segment sizes, ranges) from facts_context, metrics.json summary, scored_rows snapshot, or alignment_check.
          * For each number: include the source artifact name/path.
          * If a required number is unavailable: write "No disponible" + which artifact is missing.
          * Include required/available slots from reporting_policy.slots (respect mode).
        - Business Impact:
          * Translate the evidence into business implications (pricing decision logic, expected value, risk/return).
          * If there is a segment-based recommendation, explain how segments differ WITHOUT inventing segment names.
        - Risks & Limitations:
          * List the top 3-6 risks/limitations (data quality, leakage risk, small sample size, misalignment warnings).
          * Include an "Execution Trace" bullet list: summarize iterations, warnings, and reviewer verdict using run_summary + review_feedback + gate_context (no speculation).
          * Report Data Adequacy verbatim: status, up to 3 reason tags, and up to 3 recommendations from data_adequacy_report_json.
          * If Data Adequacy indicates data_limited/insufficient_signal, state it clearly and tie it to confidence.
          * If Data Adequacy reports missing baseline metrics, call it out and advise adding Dummy baselines.
          * If Alignment Check is WARN/FAIL, state it explicitly and the practical implication.
        - Recommended Next Actions:
          * 2-5 specific, actionable steps (quick wins + structural data improvements) aligned to the objective.
          * If Data Adequacy provides recommendations, reuse them (do NOT invent).
        - Visual Insights (ONLY if this section exists in reporting_policy.sections):
          * For EACH plot in VISUALS CONTEXT plots_list:
            - Use forward slashes in paths. If plot_reference_mode is "figure_only", reference as "Figure: <filename>" instead of inline images.
            - Otherwise embed the plot using the exact path string:
              ![<filename>](<exact_path_from_plots_list>)
            - Add 1-3 bullets with concrete takeaways, grounded in plot_insights_json/facts/metrics. If not available, write "No disponible".
          * Do NOT describe only the chart type; state what it implies for the business decision.
          * Do NOT claim "no visualizations" when plots_list is non-empty.

        Ensure logical consistency: do not claim elasticity, uplift, or improvements unless supported by metrics or plots.
        If quality gates are missing or misaligned, explicitly state that evaluation confidence is reduced.
        IF DATA ADEQUACY:
        If Data Adequacy indicates status "data_limited" AND threshold_reached is true,
        explicitly say the performance ceiling is likely due to data quality/coverage,
        and list 2-4 concrete data improvement steps from Data Adequacy recommendations.
        If Data Adequacy indicates "insufficient_signal", state that metrics are incomplete
        and the report should be treated as directional, not decision-grade.
        If run_outcome is GO_WITH_LIMITATIONS, explicitly state limitations and scope of validity.

        If Business Readiness indicates NO_APTO_PARA_PRODUCCION, explicitly state it and summarize
        the main reasons using gate context and review feedback in executive language.
        If Alignment Check is WARN or FAIL, explicitly state the limitation and whether it is data-limited
        or method-choice, and reflect it in Risks & Limitations.

        End the report with a final section titled "Evidencia usada" listing up to 8 artifact paths from Evidence Paths.
                OUTPUT: Markdown format (NO TABLES).
        """)
        
        error_condition_str = f"CRITICAL ERROR ENCOUNTERED: {error_message}" if error_message else "No critical errors."

        system_prompt = SYSTEM_PROMPT_TEMPLATE.substitute(
            business_objective=business_objective,
            strategy_title=strategy_title,
            hypothesis=hypothesis,
            compliance=compliance,
            executive_decision_label=executive_decision_label,
            error_condition=error_condition_str,
            visuals_context_json=visuals_context_json,
            analysis_type=analysis_type,
            contract_context=json.dumps(contract_context, ensure_ascii=False),
            integrity_context=integrity_context,
            output_contract_context=output_contract_context,
            case_alignment_context=case_alignment_context,
            case_alignment_business_status=json.dumps(case_alignment_business_status, ensure_ascii=False),
            gate_context=json.dumps(gate_context, ensure_ascii=False),
            review_feedback=json.dumps(review_feedback_context, ensure_ascii=False),
            steward_context=json.dumps(steward_context, ensure_ascii=False),
            cleaning_context=json.dumps(cleaning_context, ensure_ascii=False),
            run_summary_context=json.dumps(run_summary_context, ensure_ascii=False),
            data_adequacy_context=json.dumps(data_adequacy_context, ensure_ascii=False),
            data_adequacy_report_json=json.dumps(data_adequacy_report, ensure_ascii=False),
            insights_context=json.dumps(insights, ensure_ascii=False),
            alignment_check_context=json.dumps(alignment_check_context, ensure_ascii=False),
            facts_context=json.dumps(facts_context, ensure_ascii=False),
            weights_context=json.dumps(weights_context, ensure_ascii=False),
            model_metrics_context=json.dumps(model_metrics_context, ensure_ascii=False),
            case_summary_context=json.dumps(case_summary_context, ensure_ascii=False),
            scored_rows_context=json.dumps(scored_rows_context, ensure_ascii=False),
            plot_insights_json=json.dumps(plot_insights, ensure_ascii=False),
            artifacts_context=json.dumps(artifacts_context, ensure_ascii=False),
            recommendations_preview_context=json.dumps(recommendations_preview, ensure_ascii=False),
            cleaned_sample_table_text=cleaned_sample_table_text,
            scored_sample_table_text=scored_sample_table_text,
            metrics_table_text=metrics_table_text,
            recommendations_table_text=recommendations_table_text,
            evidence_paths_text=evidence_paths_text,
            reporting_policy_context=json.dumps(reporting_policy_context, ensure_ascii=False),
            translator_view_context=json.dumps(translator_view_context, ensure_ascii=False),
            slot_payloads_context=json.dumps(slot_payloads, ensure_ascii=False),
            slot_coverage_context=json.dumps(slot_coverage_context, ensure_ascii=False),
            decisioning_context_json=decisioning_context_json,
            decisioning_columns_text=decisioning_columns_text,
        )
        
        # Execution Results
        execution_results = state.get('execution_output', 'No execution results available.')

        USER_MESSAGE_TEMPLATE = """
        Generate the Executive Report.

        *** EXECUTION FINDINGS (RESULTS & METRICS) ***
        $execution_results

        *** INSTRUCTIONS ***
        Use ONLY the structured context provided in the system prompt above.
        Do NOT invent numbers or claims not supported by the artifacts.
        Follow reporting_policy.slots. If a required slot is missing, write "No disponible" and cite missing sources.
        Cite source artifact names for all metrics (e.g., "Fuente: metrics.json").
        """

        user_message = render_prompt(
            USER_MESSAGE_TEMPLATE,
            execution_results=execution_results
        )

        full_prompt = system_prompt + "\n\n" + user_message
        self.last_prompt = full_prompt

        try:
            response = self.model.generate_content(full_prompt)
            content = (getattr(response, "text", "") or "").strip()
            content = _sanitize_report_text(content)
            self.last_response = content
            return content
        except Exception as e:
            return f"Error generating report: {e}"
