import logging
from types import SimpleNamespace

from src.utils.llm_fallback import call_chat_with_fallback


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))]


class _DummyCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, model, messages, **kwargs):
        self.calls.append(model)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _DummyClient:
    def __init__(self, responses):
        completions = _DummyCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self._completions = completions


def test_call_chat_with_fallback_uses_second_model(caplog):
    client = _DummyClient([Exception("boom"), _DummyResponse("ok")])
    logger = logging.getLogger("test_fallback")
    with caplog.at_level(logging.WARNING):
        response, model_used = call_chat_with_fallback(
            client,
            messages=[{"role": "user", "content": "hi"}],
            model_chain=["model-a", "model-b"],
            call_kwargs={},
            logger=logger,
            context_tag="unit_test",
        )
    assert model_used == "model-b"
    assert response.choices[0].message.content == "ok"
    assert any("model-a" in record.message for record in caplog.records)
