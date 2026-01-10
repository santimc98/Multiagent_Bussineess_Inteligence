import json
import os

from src.agents.business_translator import BusinessTranslatorAgent


class _EchoModel:
    def generate_content(self, prompt):
        class _Resp:
            def __init__(self, text):
                self.text = text
        return _Resp(prompt)


def test_translator_required_slot_missing_mentions_no_disponible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reporting_policy": {
                    "sections": ["decision"],
                    "slots": [
                        {
                            "id": "model_metrics",
                            "mode": "required",
                            "insights_key": "metrics_summary",
                            "sources": ["data/metrics.json"],
                        }
                    ],
                    "constraints": {"no_markdown_tables": True},
                }
            },
            f,
        )
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Objetivo de prueba"}
    )
    assert "No disponible" in report


def test_translator_without_segment_pricing_slot_does_not_require_segments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reporting_policy": {
                    "sections": ["decision"],
                    "slots": [
                        {
                            "id": "model_metrics",
                            "mode": "required",
                            "insights_key": "metrics_summary",
                            "sources": ["data/metrics.json"],
                        }
                    ],
                    "constraints": {"no_markdown_tables": True},
                }
            },
            f,
        )
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics_summary": [{"metric": "accuracy", "value": 0.8}]}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Objetivo de prueba"}
    )
    assert "segment_pricing" not in report
