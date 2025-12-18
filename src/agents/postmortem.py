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

        def _decision(action: str, reason: str) -> Dict[str, Any]:
            return {
                "action": action,
                "reason": reason,
                "instructions": ["Address highlighted issues before next attempt."],
                "context_patch": {
                    "target": "feedback_history",
                    "payload": f"POSTMORTEM: {reason}"
                },
                "should_reset": {
                    "reset_ml_patch_context": action != "retry_ml_engineer",
                    "reset_review_streaks": True,
                }
            }

        if last_source in {"ml_preflight", "qa_reviewer", "reviewer"}:
            return _decision("retry_ml_engineer", f"Last gate failed in {last_source}; retry ML engineer.")
        if last_source in {"data_engineer", "integrity_audit", "column_mapping", "dialect_guard", "cleaning"} or "dialect" in last_source:
            return _decision("retry_data_engineer", f"Last gate failed in {last_source}; retry Data Engineer.")

        if "target has no variance" in err_msg:
            if variance_count >= 2 and restrat < 2:
                return _decision("re_strategize", "Repeated target variance failure; re-strategize.")
            return _decision("retry_ml_engineer", "Target has no variance; retry ML engineer.")

        if "missing required columns" in err_msg or "mapping failed" in err_msg or "dialect" in err_msg or "pd.read_csv" in err_msg or "cleaning failed" in err_msg or "json" in err_msg:
            if missing_repeat >= 1:
                return _decision("re_strategize", "Repeated missing input columns; re-strategize.")
            return _decision("retry_data_engineer", "Cleaning/manifest/dialect failure; retry Data Engineer.")

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

        return _decision(action, reason)

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            return self._fallback(context)

        system_prompt = """
You are a Principal Data Scientist acting as a postmortem lead.
Decide the next action after a failed/weak run. Return STRICT JSON only.
Allowed actions: retry_data_engineer | retry_ml_engineer | re_strategize | stop.
Provide clear instructions (no code) and a context_patch to inject into next agent.
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
            # Normalize should_reset
            if decision.get("should_reset") is not None and not isinstance(decision.get("should_reset"), dict):
                decision["should_reset"] = {}
            return decision
        except Exception:
            return self._fallback(context)
