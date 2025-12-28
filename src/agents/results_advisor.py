import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.retries import call_with_retries

load_dotenv()


class ResultsAdvisorAgent:
    """
    Generate ML improvement advice from evaluation artifacts.
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

    def _truncate(self, text: str, max_len: int) -> str:
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len]

    def _fallback(self, context: Dict[str, Any]) -> str:
        lines: List[str] = []
        case_report = context.get("case_alignment_report", {}) or {}
        prev_report = context.get("previous_case_alignment_report", {}) or {}
        failures = case_report.get("failures", []) or []
        metrics = case_report.get("metrics", {}) or {}
        weights = context.get("weights", {}) or {}
        weights_uniformity = context.get("weights_uniformity", {}) or {}
        output_report = context.get("output_contract_report", {}) or {}
        review_feedback = str(context.get("review_feedback") or "")
        if prev_report:
            prev_metrics = prev_report.get("metrics", {}) or {}
            try:
                curr_spear = float(metrics.get("spearman_case_means", 0))
                prev_spear = float(prev_metrics.get("spearman_case_means", 0))
                if curr_spear < prev_spear:
                    lines.append(
                        "ISSUE: case-level alignment regressed vs prior run; WHY: objective still global-first; FIX: make case-order penalty primary and increase its weight."
                    )
            except Exception:
                pass

        if failures:
            if "spearman_case_means" in failures:
                lines.append(
                    "ISSUE: case-level correlation too low; WHY: objective optimizes global ranking only; FIX: optimize case-mean ordering first and penalize case-order violations."
                )
            if "adjacent_refscore_violations" in failures:
                lines.append(
                    "ISSUE: adjacent RefScore order violations; WHY: no monotonic constraints; FIX: add pairwise case-order penalty or constraints between adjacent RefScore groups."
                )
        if metrics.get("weight_concentration_hhi"):
            hhi = float(metrics.get("weight_concentration_hhi", 0))
            if hhi > 0.5:
                lines.append(
                    "ISSUE: weight concentration high; WHY: optimization collapses to one feature; FIX: add L2/entropy regularization or cap max weight."
                )
        if weights_uniformity.get("uniform") is True:
            lines.append(
                "ISSUE: weights are effectively uniform; WHY: optimizer likely fell back to equal weights or objective is too flat; FIX: change initialization/penalties to break symmetry and force weight differentiation."
            )
        if "data leakage" in review_feedback.lower() or "leakage" in review_feedback.lower():
            lines.append(
                "ISSUE: data leakage risk flagged; WHY: target or proxy is inside feature set; FIX: explicitly exclude target-derived columns from features and validate post-mapping."
            )
        if "successflag" in review_feedback.lower() and "feature" in review_feedback.lower():
            lines.append(
                "ISSUE: SuccessFlag used as feature; WHY: target leakage in conversion model; FIX: remove SuccessFlag from features and use only pre-contract attributes."
            )
        if "contract" in review_feedback.lower() and "only" in review_feedback.lower() and "success" in review_feedback.lower():
            lines.append(
                "ISSUE: pricing model trained only on wins; WHY: biases price ceiling estimate; FIX: train pricing on full dataset and optimize revenue via price*probability."
            )
        if isinstance(weights, dict):
            weight_list = weights.get("optimized_weights") or []
            if weight_list and max(weight_list) > 0.6:
                lines.append(
                    "ISSUE: single weight dominates; WHY: unconstrained optimization; FIX: add max-weight constraint or penalty for concentration."
                )
        if output_report.get("missing"):
            lines.append(
                "ISSUE: required outputs missing; WHY: outputs not saved; FIX: ensure all contract outputs are written before exit."
            )
        if not lines:
            lines.append(
                "ISSUE: alignment gates failed; WHY: objective not matching business ordering; FIX: optimize case-level ranking before global correlation."
            )
        return "\n".join(lines)
