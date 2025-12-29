import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.retries import call_with_retries

load_dotenv()


class CleaningReviewerAgent:
    """
    Senior Cleaning Reviewer. Audits data cleaning outputs for destructive conversions,
    leakage risks, and invalid transformations using manifest + samples + stats.
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

    def review_cleaning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "APPROVE_WITH_WARNINGS",
                "feedback": "Cleaning reviewer skipped: MIMO_API_KEY not configured.",
                "failed_checks": [],
                "required_fixes": [],
            }

        system_prompt = (
            "You are a Senior Data Cleaning Reviewer. "
            "Audit the cleaning output using the manifest, raw samples, cleaned previews, and summary stats. "
            "Your job is to detect destructive parsing (null explosions), misinterpreted number formats "
            "(thousands/decimal separators), incorrect imputation, and missing/aliased columns. "
            "Use the execution_contract (data_requirements, expected_kind, expected_range, feature_availability, "
            "spec_extraction, decision_variables, missing_sentinels) as ground truth for variable meaning. "
            "Use steward_summary as authoritative context for domain semantics. "
            "Reason like a senior: base decisions on evidence, explain interpretation and impact, then decide. "
            "If results are unsafe for downstream modeling, REJECT with clear, actionable fixes. "
            "If evidence is mixed, prefer APPROVE_WITH_WARNINGS rather than rejecting. "
            "Incorporate steward_summary and raw_pattern_stats to infer expected formats and units. "
            "If cleaned_value_counts suggest binarization or collapse, call it out explicitly, but do not "
            "reject if the cleaned range matches expected_range and the contract allows sentinel imputation. "
            "Differentiate between warnings and hard failures: "
            "- REJECT only if required columns are missing, derived columns in spec_extraction are absent, "
            "or numeric parsing causes large-scale corruption (null explosions or order-of-magnitude shifts). "
            "- If missing_sentinels are provided, treat those sentinel values as expected and recommend handling "
            "in modeling rather than rejecting cleaning. "
            "- Leakage audits belong to modeling; if a leakage check is missing in cleaning, issue a WARNING "
            "unless the cleaning step used a post-decision field to derive a pre-decision column. "
            "- If a field is flagged as post-decision in feature_availability, treat leakage as a WARNING "
            "unless the column was used for deriving pre-decision features. "
            "Do NOT provide code. Provide reasoning with Evidence -> Interpretation -> Impact -> Decision."
        )

        output_format = (
            "Return a JSON object with this schema: "
            "{"
            "\"status\": \"APPROVED\" | \"APPROVE_WITH_WARNINGS\" | \"REJECTED\", "
            "\"feedback\": \"Senior-level explanation\", "
            "\"failed_checks\": [\"list\"], "
            "\"required_fixes\": [\"list\"]"
            "}"
        )

        user_prompt = (
            "CONTEXT (JSON):\n"
            + json.dumps(context, ensure_ascii=False) + "\n\n"
            + output_format
        )

        def _call_model():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return response.choices[0].message.content

        try:
            content = call_with_retries(_call_model, max_retries=2)
            result = json.loads(content)
        except Exception:
            return {
                "status": "REJECTED",
                "feedback": "Cleaning reviewer failed to parse JSON response.",
                "failed_checks": ["JSON_PARSE_ERROR"],
                "required_fixes": ["Ensure the reviewer returns valid JSON."],
            }

        status = result.get("status")
        if status not in {"APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"}:
            result["status"] = "REJECTED"
            result["feedback"] = "Cleaning reviewer returned invalid status."

        for field in ["failed_checks", "required_fixes"]:
            val = result.get(field, [])
            if isinstance(val, str):
                result[field] = [val]
            elif not isinstance(val, list):
                result[field] = []
        if "feedback" not in result:
            result["feedback"] = ""
        return result
