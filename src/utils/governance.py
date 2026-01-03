import json
import os
from datetime import datetime
from typing import Any, Dict


def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _flatten_metrics(obj: Any, prefix: str = "", out: Dict[str, float] | None = None) -> Dict[str, float]:
    if out is None:
        out = {}
    if not isinstance(obj, dict):
        return out
    for key, value in obj.items():
        metric_key = f"{prefix}{key}" if prefix else str(key)
        if _is_number(value):
            out[metric_key] = float(value)
        elif isinstance(value, dict):
            _flatten_metrics(value, f"{metric_key}.", out)
    return out


def _metric_higher_is_better(name: str) -> bool:
    key = str(name or "").lower()
    if any(token in key for token in ["loss", "error", "mae", "rmse", "mse", "mape", "smape", "logloss", "brier"]):
        return False
    return True


def _extract_baseline_vs_model(metric_pool: Dict[str, float]) -> Dict[str, Any]:
    baseline_vs_model = []
    baseline_keys = [k for k in metric_pool.keys() if any(tok in k.lower() for tok in ["baseline", "dummy", "naive", "null"])]
    for base_key in baseline_keys:
        base_val = metric_pool.get(base_key)
        norm = base_key.lower()
        for prefix in ["baseline_", "dummy_", "naive_", "null_"]:
            if norm.startswith(prefix):
                model_key = base_key[len(prefix):]
                break
        else:
            model_key = base_key.replace("baseline.", "", 1)
        model_val = metric_pool.get(model_key)
        if model_val is None:
            continue
        higher_is_better = _metric_higher_is_better(model_key)
        delta = (model_val - base_val) if higher_is_better else (base_val - model_val)
        baseline_vs_model.append(
            {
                "metric": model_key,
                "baseline": base_val,
                "model": model_val,
                "delta": delta,
                "higher_is_better": higher_is_better,
            }
        )
    return {"pairs": baseline_vs_model, "metric_pool": metric_pool}


def _detect_metric_ceiling(
    baseline_vs_model: Dict[str, Any],
    data_adequacy: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    ceiling_detected = False
    reason = None
    pairs = baseline_vs_model.get("pairs", []) if isinstance(baseline_vs_model, dict) else []
    for pair in pairs:
        metric = str(pair.get("metric", "")).lower()
        delta = pair.get("delta")
        if delta is None:
            continue
        if "auc" in metric:
            if delta < thresholds["auc"]:
                ceiling_detected = True
                reason = "low_signal"
        if "f1" in metric:
            if delta < thresholds["f1"]:
                ceiling_detected = True
                reason = "low_signal"
        if "r2" in metric:
            if delta < thresholds["r2"]:
                ceiling_detected = True
                reason = "low_signal"
        if any(tok in metric for tok in ["mae", "rmse", "mape"]):
            if delta < thresholds["error"]:
                ceiling_detected = True
                reason = "low_signal"
    if isinstance(data_adequacy, dict):
        if data_adequacy.get("status") in {"data_limited", "insufficient_signal"}:
            ceiling_detected = True
            reason = reason or "low_signal"
        reasons = data_adequacy.get("reasons", []) or []
        if any("high_dimensionality_low_sample" in r for r in reasons):
            ceiling_detected = True
            reason = "small_n"
    metric_pool = baseline_vs_model.get("metric_pool", {}) if isinstance(baseline_vs_model, dict) else {}
    for key, value in metric_pool.items():
        if "cv_std" in str(key).lower() and _is_number(value) and value >= thresholds["cv_std"]:
            ceiling_detected = True
            reason = "high_variance_cv"
            break
    return {"metric_ceiling_detected": ceiling_detected, "ceiling_reason": reason}


def build_governance_report(state: Dict[str, Any]) -> Dict[str, Any]:
    contract = _safe_load_json("data/execution_contract.json") or state.get("execution_contract", {})
    output_contract = _safe_load_json("data/output_contract_report.json")
    case_alignment = _safe_load_json("data/case_alignment_report.json")
    alignment_check = _safe_load_json("data/alignment_check.json")
    integrity = _safe_load_json("data/integrity_audit_report.json")

    issues = integrity.get("issues", []) if isinstance(integrity, dict) else []
    severity_counts = {}
    for issue in issues:
        sev = str(issue.get("severity", "unknown"))
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    review_verdict = state.get("last_successful_review_verdict") or state.get("review_verdict")
    gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context")

    return {
        "run_id": state.get("run_id"),
        "timestamp": datetime.utcnow().isoformat(),
        "strategy_title": contract.get("strategy_title", ""),
        "business_objective": contract.get("business_objective", ""),
        "review_verdict": review_verdict,
        "last_gate_context": gate_context,
        "output_contract": output_contract,
        "case_alignment": case_alignment,
        "alignment_check": alignment_check,
        "integrity_issues_summary": severity_counts,
        "budget_counters": state.get("budget_counters", {}),
        "run_budget": state.get("run_budget", {}),
        "data_risks": contract.get("data_risks", []),
    }


def write_governance_report(state: Dict[str, Any], path: str = "data/governance_report.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = build_governance_report(state)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def build_run_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    case_alignment = _safe_load_json("data/case_alignment_report.json")
    output_contract = _safe_load_json("data/output_contract_report.json")
    data_adequacy = _safe_load_json("data/data_adequacy_report.json")
    alignment_check = _safe_load_json("data/alignment_check.json")
    status = state.get("last_successful_review_verdict") or state.get("review_verdict") or "UNKNOWN"
    failed_gates = []
    gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context")
    if isinstance(gate_context, dict):
        failed_gates = gate_context.get("failed_gates", []) or []
    if isinstance(case_alignment, dict) and case_alignment.get("status") == "FAIL":
        failed_gates.extend(case_alignment.get("failures", []))
    if isinstance(output_contract, dict) and output_contract.get("missing"):
        failed_gates.append("output_contract_missing")
    adequacy_summary = {}
    if isinstance(data_adequacy, dict):
        alignment = data_adequacy.get("quality_gates_alignment", {}) if isinstance(data_adequacy, dict) else {}
        alignment_summary = {}
        if isinstance(alignment, dict) and alignment:
            mapped = alignment.get("mapped_gates", {}) if isinstance(alignment, dict) else {}
            unmapped = alignment.get("unmapped_gates", {}) if isinstance(alignment, dict) else {}
            alignment_summary = {
                "status": alignment.get("status"),
                "mapped_gate_count": len(mapped) if isinstance(mapped, dict) else 0,
                "unmapped_gate_count": len(unmapped) if isinstance(unmapped, dict) else 0,
            }
        adequacy_summary = {
            "status": data_adequacy.get("status"),
            "reasons": data_adequacy.get("reasons", []),
            "recommendations": data_adequacy.get("recommendations", []),
            "consecutive_data_limited": data_adequacy.get("consecutive_data_limited"),
            "data_limited_threshold": data_adequacy.get("data_limited_threshold"),
            "threshold_reached": data_adequacy.get("threshold_reached"),
            "quality_gates_alignment": alignment_summary,
        }
    warnings = []
    if state.get("qa_budget_exceeded"):
        warnings.append("QA_INCOMPLETE: QA budget exceeded; QA audit skipped.")
    contract = _safe_load_json("data/execution_contract.json") or state.get("execution_contract", {})
    metrics_report = _safe_load_json("data/metrics.json")
    weights_report = _safe_load_json("data/weights.json")
    metric_pool = {}
    metric_pool.update(_flatten_metrics(metrics_report))
    metric_pool.update(_flatten_metrics(weights_report))
    baseline_vs_model = _extract_baseline_vs_model(metric_pool)
    thresholds = {
        "auc": float(os.getenv("CEILING_DELTA_AUC", "0.02")),
        "f1": float(os.getenv("CEILING_DELTA_F1", "0.03")),
        "r2": float(os.getenv("CEILING_DELTA_R2", "0.02")),
        "error": float(os.getenv("CEILING_DELTA_ERROR", "0.03")),
        "cv_std": float(os.getenv("CEILING_CV_STD", "0.05")),
    }
    ceiling_info = _detect_metric_ceiling(baseline_vs_model, data_adequacy, thresholds)
    failed_gates_lower = [str(item).lower() for item in failed_gates if item]
    critical_tokens = [
        "synthetic",
        "leakage",
        "security",
        "output_contract_missing",
        "dataframe_literal_overwrite",
        "unknown_columns_referenced",
        "df_column_assignment_forbidden",
    ]
    critical_hit = any(any(tok in gate for tok in critical_tokens) for gate in failed_gates_lower)
    output_missing = bool(isinstance(output_contract, dict) and output_contract.get("missing"))
    if output_missing or critical_hit or status in {"REJECTED", "FAIL", "CRASH"}:
        run_outcome = "NO_GO"
    else:
        counterfactual_policy = ""
        if isinstance(contract, dict):
            counterfactual_policy = str(contract.get("counterfactual_policy") or "")
        if ceiling_info.get("metric_ceiling_detected") or counterfactual_policy == "observational_only":
            run_outcome = "GO_WITH_LIMITATIONS"
        else:
            run_outcome = "GO"
    return {
        "run_id": state.get("run_id"),
        "status": status,
        "run_outcome": run_outcome,
        "failed_gates": list(dict.fromkeys(failed_gates)),
        "warnings": warnings,
        "budget_counters": state.get("budget_counters", {}),
        "data_adequacy": adequacy_summary,
        "metric_ceiling_detected": ceiling_info.get("metric_ceiling_detected"),
        "ceiling_reason": ceiling_info.get("ceiling_reason"),
        "baseline_vs_model": baseline_vs_model.get("pairs", []),
        "metrics": {
            "baseline_vs_model": baseline_vs_model.get("pairs", []),
            "metric_pool_size": len(metric_pool),
        },
        "alignment_check": {
            "status": alignment_check.get("status"),
            "failure_mode": alignment_check.get("failure_mode"),
            "summary": alignment_check.get("summary"),
        } if isinstance(alignment_check, dict) and alignment_check else {},
    }
