import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.retries import call_with_retries

load_dotenv()


class FailureExplainerAgent:
    """
    Explains runtime failures using code + traceback + context.
    Returns a short, plain-text diagnosis to feed back into the next attempt.
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

    def explain_data_engineer_failure(
        self,
        code: str,
        error_details: str,
        context: Dict[str, Any] | None = None,
    ) -> str:
        if not code or not error_details:
            return ""
        if not self.client:
            return self._fallback(error_details)

        ctx = context or {}
        code_snippet = self._truncate(code, 6000)
        error_snippet = self._truncate(error_details, 4000)
        context_snippet = self._truncate(str(ctx), 2000)

        system_prompt = (
            "You are a senior debugging assistant. "
            "Given the generated Python cleaning code, the traceback/error, and context, "
            "explain why the failure happened. "
            "Return concise plain text (2-6 short lines). "
            "Do NOT include code. Do NOT restate the full traceback. "
            "Focus on root cause and the specific location/pattern that broke."
        )
        user_prompt = (
            "CODE:\n"
            + code_snippet + "\n\n"
            "ERROR:\n"
            + error_snippet + "\n\n"
            "CONTEXT:\n"
            + context_snippet + "\n"
        )

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
            return self._fallback(error_details)

        return (content or "").strip()

    def explain_ml_failure(
        self,
        code: str,
        error_details: str,
        context: Dict[str, Any] | None = None,
    ) -> str:
        if not code or not error_details:
            return ""
        if not self.client:
            return self._fallback(error_details)

        ctx = context or {}
        code_snippet = self._truncate(code, 6000)
        error_snippet = self._truncate(error_details, 4000)
        context_snippet = self._truncate(str(ctx), 2000)

        system_prompt = (
            "You are a senior ML debugging assistant. "
            "Given the generated ML Python code, the runtime error output, and context, "
            "explain why the failure happened. "
            "Return concise plain text (3-6 short lines). "
            "Use this format with short lines: "
            "WHERE: <location or step>, WHY: <root cause>, FIX: <what to change>. "
            "Do NOT include code. Do NOT restate the full traceback. "
            "Focus on root cause and the specific logic mistake."
        )
        user_prompt = (
            "CODE:\n"
            + code_snippet + "\n\n"
            "ERROR:\n"
            + error_snippet + "\n\n"
            "CONTEXT:\n"
            + context_snippet + "\n"
        )

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
            return self._fallback(error_details)

        return (content or "").strip()

    def _fallback(self, error_details: str) -> str:
        lower = error_details.lower()
        if "list of cases must be same length as list of conditions" in lower:
            return "np.select called with mismatched conditions vs choices list lengths."
        if "numpy.bool_" in lower and "not serializable" in lower:
            return "json.dumps failed because numpy.bool_ values are not handled by _json_default."
        if "keyerror" in lower and "not in index" in lower:
            return "Column selection list includes names that are not present after renaming."
        if "missing required columns" in lower:
            return "Required columns are missing after header normalization/mapping."
        return ""

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        head = text[: limit // 2]
        tail = text[-(limit // 2) :]
        return f"{head}\n...[truncated]...\n{tail}"
