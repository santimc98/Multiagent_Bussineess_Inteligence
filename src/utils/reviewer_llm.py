from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

_PLACEHOLDER_KEYS = {"dummy", "test", "placeholder", "changeme"}


def _is_placeholder_key(value: Optional[str]) -> bool:
    if not value:
        return True
    return value.strip().lower() in _PLACEHOLDER_KEYS


def init_reviewer_llm(
    api_key: Optional[str] = None,
) -> Tuple[str, Any, Optional[str], Optional[str]]:
    """
    Returns (provider, client, model_name, warning).
    provider: "gemini" | "mimo" | "none".
    """
    if api_key and not _is_placeholder_key(api_key):
        from openai import OpenAI

        model_name = os.getenv("REVIEWER_MIMO_MODEL", "mimo-v2-flash")
        client = OpenAI(api_key=api_key, base_url="https://api.xiaomimimo.com/v1")
        return "mimo", client, model_name, None

    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key and not _is_placeholder_key(google_key):
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        genai.configure(api_key=google_key)
        generation_config = {
            "temperature": float(os.getenv("REVIEWER_GEMINI_TEMPERATURE", "0.2")),
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": int(os.getenv("REVIEWER_GEMINI_MAX_TOKENS", "8192")),
            "response_mime_type": "application/json",
        }
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        model_name = os.getenv("REVIEWER_GEMINI_MODEL", "gemini-3-flash-preview")
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        return "gemini", model, model_name, None

    mimo_key = os.getenv("MIMO_API_KEY")
    if mimo_key and not _is_placeholder_key(mimo_key):
        from openai import OpenAI

        model_name = os.getenv("REVIEWER_MIMO_MODEL", "mimo-v2-flash")
        client = OpenAI(api_key=mimo_key, base_url="https://api.xiaomimimo.com/v1")
        return "mimo", client, model_name, "GEMINI_API_KEY missing; falling back to MIMO reviewers."

    return "none", None, None, "No reviewer LLM API key configured (GOOGLE_API_KEY/MIMO_API_KEY)."
