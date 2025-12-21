import json
import os
from typing import Dict, Any, List
from string import Template

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

UNSET = object()


class PostmortemAgent:
    """
    Tech-lead style agent to decide the next action after a failure or poor outcome.
    Returns structured decisions; falls back to heuristics if LLM is unavailable.
    """

    def __init__(self, api_key: Any = UNSET):
        if api_key is UNSET:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        else:
            self.api_key = api_key
        if self.api_key is None or self.api_key == "":
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
                timeout=None,
            )
        self.model_name = "deepseek-reasoner"

    def _infer_failure_cause(self, text: str) -> str:
        if not text:
            return ""
        lower = text.lower()
        if "missing required columns" in lower or "mapping failed" in lower:
            return "Required columns not found after canonicalization or alias mapping."
        if "cleaning_plan_not_allowed" in lower or "plan output" in lower:
            return "Model returned a JSON plan instead of executable Python code."
        if "dialect" in lower or "read_csv" in lower:
            return "CSV dialect mismatch (sep/decimal/encoding) or missing dialect params."
        if "target has no variance" in lower:
            return "Target has no variance; optimization/training is not feasible."
        if "no successful experiments executed" in lower:
            return "Optimization failed; constraints or missingness left no viable solution."
        if "parsererror" in lower or "tokenizing data" in lower:
            return "CSV dialect/quoting mismatch; enforce detected sep/decimal/encoding."
        if "nameerror" in lower:
            return "Undefined name referenced in the generated code."
        if "keyerror" in lower and "not in index" in lower:
            return "Column name mismatch after normalization; mapping resolution failed."
        if "typeerror" in lower or "ufunc" in lower:
            return "Type conversion missing; numeric ops executed on string/object data."
        if "json" in lower and "default" in lower:
            return "Manifest json.dump missing default for numpy/pandas types."
        if "syntaxerror" in lower:
            return "Generated code is not valid Python syntax."
        return ""

    def _build_de_context_payload(self, context: Dict[str, Any], reason: str) -> str:
        last_gate = context.get("last_gate_context") or {}
        required_fixes = last_gate.get("required_fixes", [])
        gate_feedback = last_gate.get("feedback", "")
        err_msg = context.get("error_message", "") or ""
        exec_out = context.get("execution_output", "") or ""
        why = ""
        if required_fixes:
            why = f"Gate required fixes: {required_fixes}"
        else:
            why = self._infer_failure_cause(gate_feedback or err_msg or exec_out)
        if not reason:
            reason = err_msg or gate_feedback or "Data Engineer failure."
        if not why:
            why = "Unknown root cause. Inspect header mapping, canonicalization, and required input checks."
        lines = ["POSTMORTEM_CONTEXT_FOR_DE:"]
        if reason:
            lines.append(f"FAILURE_SUMMARY: {reason}")
        if why:
            lines.append(f"WHY_IT_HAPPENED: {why}")
        if gate_feedback:
            lines.append(f"LAST_GATE_FEEDBACK: {gate_feedback}")
        if err_msg:
            lines.append(f"ERROR_MESSAGE: {err_msg}")
        if exec_out:
            lines.append(f"EXECUTION_OUTPUT_TAIL: {exec_out[-1200:]}")
        lines.append("FIX_GUIDANCE: Fix the root cause and regenerate the full cleaning script.")
        return "\n".join(lines)

    def _build_ml_context_payload(self, context: Dict[str, Any], reason: str) -> str:
        last_gate = context.get("last_gate_context") or {}
        required_fixes = last_gate.get("required_fixes", [])
        gate_feedback = last_gate.get("feedback", "")
        err_msg = context.get("error_message", "") or ""
        exec_out = context.get("execution_output", "") or ""
        why = ""
        if required_fixes:
            why = f"Gate required fixes: {required_fixes}"
        else:
            why = self._infer_failure_cause(gate_feedback or err_msg or exec_out)
        if not reason:
            reason = err_msg or gate_feedback or "ML Engineer failure."
        if not why:
            why = "Unknown root cause. Inspect feature mapping, target variance, and dialect usage."
        lines = ["POSTMORTEM_CONTEXT_FOR_ML:"]
        if reason:
            lines.append(f"FAILURE_SUMMARY: {reason}")
        if why:
            lines.append(f"WHY_IT_HAPPENED: {why}")
        if gate_feedback:
            lines.append(f"LAST_GATE_FEEDBACK: {gate_feedback}")
        if err_msg:
            lines.append(f"ERROR_MESSAGE: {err_msg}")
        if exec_out:
            lines.append(f"EXECUTION_OUTPUT_TAIL: {exec_out[-1200:]}")
        lines.append("FIX_GUIDANCE: Fix the root cause and regenerate the full ML script.")
        return "\n".join(lines)

    def _fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        exec_out = context.get("execution_output", "") or ""
        issues = context.get("integrity_issues", []) or []
        iteration = context.get("iteration_count", 0)
        restrat = context.get("restrategize_count", 0)
        err_msg = (context.get("error_message") or "").lower()
        feedback_history = context.get("feedback_history", []) or []
        variance_count = sum(1 for item in feedback_history if "target has no variance" in str(item).lower())
        missing_repeat = context.get("missing_repeat_count", 0) or 0
        last_gate = context.get("last_gate_context") or {}
        last_source = str(last_gate.get("source", "")).lower()

        def _decision(action: str, reason: str, context_patch: Dict[str, Any] | None = None) -> Dict[str, Any]:
            if context_patch is None:
                if action == "retry_data_engineer":
                    context_patch = {
                        "target": "data_engineer_audit_override",
                        "payload": self._build_de_context_payload(context, reason),
                    }
                elif action == "retry_ml_engineer":
                    context_patch = {
                        "target": "ml_engineer_audit_override",
                        "payload": self._build_ml_context_payload(context, reason),
                    }
                else:
                    context_patch = {
                        "target": "feedback_history",
                        "payload": f"POSTMORTEM: {reason}",
                    }
            return {
                "action": action,
                "reason": reason,
                "instructions": ["Address highlighted issues before next attempt."],
                "context_patch": context_patch,
                "should_reset": {
                    "reset_ml_patch_context": action != "retry_ml_engineer",
                    "reset_review_streaks": True,
                }
            }

        if last_source in {"ml_preflight", "qa_reviewer", "reviewer"}:
            return _decision("retry_ml_engineer", f"Last gate failed in {last_source}; retry ML engineer.")
        if last_source in {"data_engineer", "integrity_audit", "column_mapping", "dialect_guard", "cleaning"} or "dialect" in last_source:
            reason = f"Last gate failed in {last_source}; retry Data Engineer."
            de_patch = {
                "target": "data_engineer_audit_override",
                "payload": self._build_de_context_payload(context, reason),
            }
            return _decision("retry_data_engineer", reason, de_patch)
        if "case_alignment_gate_failed" in err_msg or "case_alignment_gate_failed" in " ".join(str(x).lower() for x in feedback_history):
            if restrat >= 1:
                return _decision("re_strategize", "Case alignment gate failed repeatedly; re-strategize.")
            return _decision("retry_ml_engineer", "Case alignment gate failed; retry ML with ranking loss + regularization.")

        if "target has no variance" in err_msg:
            if variance_count >= 2 and restrat < 2:
                return _decision("re_strategize", "Repeated target variance failure; re-strategize.")
            return _decision("retry_ml_engineer", "Target has no variance; retry ML engineer.")

        if "missing required columns" in err_msg or "mapping failed" in err_msg or "dialect" in err_msg or "pd.read_csv" in err_msg or "cleaning failed" in err_msg or "json" in err_msg:
            if missing_repeat >= 1:
                return _decision("re_strategize", "Repeated missing input columns; re-strategize.")
            reason = "Cleaning/manifest/dialect failure; retry Data Engineer."
            de_patch = {
                "target": "data_engineer_audit_override",
                "payload": self._build_de_context_payload(context, reason),
            }
            return _decision("retry_data_engineer", reason, de_patch)

        if "traceback" in exec_out.lower() or "execution error" in exec_out.lower():
            action = "retry_ml_engineer"
            reason = "Runtime error detected."
        elif any(i.get("severity") == "critical" and i.get("type") in {"MISSING_COLUMN", "ALIASING_RISK"} for i in issues):
            action = "retry_data_engineer"
            reason = "Critical integrity issue (missing/alias)."
        elif iteration >= 2:
            if restrat >= 2:
                action = "stop"
                reason = "Multiple attempts and re-strategies exhausted."
            else:
                action = "re_strategize"
                reason = "Repeated attempts with low quality; suggest new strategy."
        else:
            action = "retry_ml_engineer"
            reason = "Default retry for improvement."

        if action == "retry_data_engineer":
            de_patch = {
                "target": "data_engineer_audit_override",
                "payload": self._build_de_context_payload(context, reason),
            }
            return _decision(action, reason, de_patch)
        if action == "retry_ml_engineer":
            ml_patch = {
                "target": "ml_engineer_audit_override",
                "payload": self._build_ml_context_payload(context, reason),
            }
            return _decision(action, reason, ml_patch)
        return _decision(action, reason)

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            return self._fallback(context)

        system_prompt = """
You are a Principal Data Scientist acting as a postmortem lead.
Decide the next action after a failed/weak run. Return STRICT JSON only.
Allowed actions: retry_data_engineer | retry_ml_engineer | re_strategize | stop.
Provide clear instructions (no code) and a context_patch to inject into next agent.
If action is retry_data_engineer, context_patch MUST target "data_engineer_audit_override" and include:
- FAILURE_SUMMARY
- WHY_IT_HAPPENED (root cause explanation)
- FIX_GUIDANCE
If action is retry_ml_engineer, context_patch MUST target "ml_engineer_audit_override" and include:
- FAILURE_SUMMARY
- WHY_IT_HAPPENED (root cause explanation)
- FIX_GUIDANCE
"""
        user_template = Template(
            """
BUSINESS_OBJECTIVE: $business_objective
SELECTED_STRATEGY: $selected_strategy
EXECUTION_CONTRACT: $execution_contract
INTEGRITY_ISSUES: $integrity_issues
EXECUTION_OUTPUT_TAIL: $execution_output
REVIEW_FEEDBACK: $review_feedback
FEEDBACK_HISTORY: $feedback_history
ITERATION_COUNT: $iteration_count
RESTRATEGIZE_COUNT: $restrategize_count
ERROR_MESSAGE: $error_message
MISSING_REPEAT_COUNT: $missing_repeat_count
LAST_GATE_CONTEXT: $last_gate_context
"""
        )
        user_prompt = user_template.substitute(
            business_objective=context.get("business_objective", ""),
            selected_strategy=json.dumps(context.get("selected_strategy", {}), indent=2),
            execution_contract=json.dumps(context.get("execution_contract", {}), indent=2),
            integrity_issues=json.dumps(context.get("integrity_issues", []), indent=2),
            execution_output=str(context.get("execution_output", ""))[:2000],
            review_feedback=context.get("review_feedback", ""),
            feedback_history=json.dumps(context.get("feedback_history", []), indent=2),
            iteration_count=context.get("iteration_count", 0),
            restrategize_count=context.get("restrategize_count", 0),
            error_message=context.get("error_message", ""),
            missing_repeat_count=context.get("missing_repeat_count", 0),
            last_gate_context=json.dumps(context.get("last_gate_context", {}), indent=2),
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content.strip()
            # Strip fences if any
            content = content.replace("```json", "").replace("```", "").strip()
            decision = json.loads(content)
            if not isinstance(decision, dict):
                return self._fallback(context)
            # Minimal sanity
            if decision.get("action") not in {"retry_data_engineer", "retry_ml_engineer", "re_strategize", "stop"}:
                return self._fallback(context)
            # Normalize context_patch
            cp = decision.get("context_patch")
            if isinstance(cp, str):
                try:
                    cp = json.loads(cp)
                except Exception:
                    cp = {}
            if cp is not None and not isinstance(cp, dict):
                decision["context_patch"] = {"target": "feedback_history", "payload": f"POSTMORTEM_BAD_SHAPE: {cp}"}
            else:
                decision["context_patch"] = cp or {}
            if decision.get("action") == "retry_data_engineer":
                existing_payload = ""
                if isinstance(decision.get("context_patch"), dict):
                    existing_payload = str(decision["context_patch"].get("payload") or "")
                auto_payload = self._build_de_context_payload(context, decision.get("reason", ""))
                payload = existing_payload
                if auto_payload and auto_payload not in payload:
                    payload = f"{payload}\n\n{auto_payload}" if payload else auto_payload
                decision["context_patch"] = {
                    "target": "data_engineer_audit_override",
                    "payload": payload,
                }
            if decision.get("action") == "retry_ml_engineer":
                existing_payload = ""
                if isinstance(decision.get("context_patch"), dict):
                    existing_payload = str(decision["context_patch"].get("payload") or "")
                auto_payload = self._build_ml_context_payload(context, decision.get("reason", ""))
                payload = existing_payload
                if auto_payload and auto_payload not in payload:
                    payload = f"{payload}\n\n{auto_payload}" if payload else auto_payload
                decision["context_patch"] = {
                    "target": "ml_engineer_audit_override",
                    "payload": payload,
                }
            # Normalize should_reset
            if decision.get("should_reset") is not None and not isinstance(decision.get("should_reset"), dict):
                decision["should_reset"] = {}
            return decision
        except Exception:
            return self._fallback(context)
