import logging
from typing import Iterable, Tuple, Any, Dict, List


def _response_has_content(response: Any) -> bool:
    if response is None:
        return False
    choices = getattr(response, "choices", None)
    if not choices:
        return False
    first = choices[0]
    msg = getattr(first, "message", None)
    if msg is None:
        return False
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return True
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        return True
    return False


def call_chat_with_fallback(
    llm_client: Any,
    messages: List[Dict[str, str]],
    model_chain: Iterable[str],
    *,
    call_kwargs: Dict[str, Any],
    logger: logging.Logger | None,
    context_tag: str,
) -> Tuple[Any, str]:
    last_exc: Exception | None = None
    for model in model_chain:
        if not model:
            continue
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=messages,
                **(call_kwargs or {}),
            )
            if not _response_has_content(response):
                raise ValueError("EMPTY_COMPLETION")
            return response, model
        except Exception as exc:  # pragma: no cover - safety net
            last_exc = exc
            if logger:
                logger.warning(
                    "LLM_FALLBACK_WARNING context=%s model=%s error=%s message=%s",
                    context_tag,
                    model,
                    type(exc).__name__,
                    str(exc)[:200],
                )
            else:
                print(
                    f"LLM_FALLBACK_WARNING context={context_tag} model={model} "
                    f"error={type(exc).__name__} message={str(exc)[:200]}"
                )
            continue
    if last_exc is not None:
        raise last_exc
    raise ValueError("No models provided for fallback.")
