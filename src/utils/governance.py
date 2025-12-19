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


def build_governance_report(state: Dict[str, Any]) -> Dict[str, Any]:
    contract = _safe_load_json("data/execution_contract.json") or state.get("execution_contract", {})
    output_contract = _safe_load_json("data/output_contract_report.json")
    case_alignment = _safe_load_json("data/case_alignment_report.json")
    integrity = _safe_load_json("data/integrity_audit_report.json")
    postmortem = _safe_load_json("data/postmortem_decision.json")

    issues = integrity.get("issues", []) if isinstance(integrity, dict) else []
    severity_counts = {}
    for issue in issues:
        sev = str(issue.get("severity", "unknown"))
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    return {
        "run_id": state.get("run_id"),
        "timestamp": datetime.utcnow().isoformat(),
        "strategy_title": contract.get("strategy_title", ""),
        "business_objective": contract.get("business_objective", ""),
        "review_verdict": state.get("review_verdict"),
        "last_gate_context": state.get("last_gate_context"),
        "output_contract": output_contract,
        "case_alignment": case_alignment,
        "integrity_issues_summary": severity_counts,
        "postmortem_decision": postmortem,
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
    status = state.get("review_verdict") or "UNKNOWN"
    failed_gates = []
    if isinstance(state.get("last_gate_context"), dict):
        failed_gates = state.get("last_gate_context", {}).get("failed_gates", []) or []
    if isinstance(case_alignment, dict) and case_alignment.get("status") == "FAIL":
        failed_gates.extend(case_alignment.get("failures", []))
    if isinstance(output_contract, dict) and output_contract.get("missing"):
        failed_gates.append("output_contract_missing")
    return {
        "run_id": state.get("run_id"),
        "status": status,
        "failed_gates": list(dict.fromkeys(failed_gates)),
        "budget_counters": state.get("budget_counters", {}),
    }
