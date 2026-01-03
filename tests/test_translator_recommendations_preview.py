import json
import os

from src.agents.business_translator import BusinessTranslatorAgent


class _EchoModel:
    def generate_content(self, prompt):
        class _Resp:
            def __init__(self, text):
                self.text = text
        return _Resp(prompt)


def test_translator_includes_illustrative_examples_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "NO_GO"}, f)
    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reporting_policy": {
                    "demonstrative_examples_enabled": True,
                    "demonstrative_examples_when_outcome_in": ["NO_GO", "GO_WITH_LIMITATIONS"],
                    "max_examples": 5,
                    "require_strong_disclaimer": True,
                },
                "spec_extraction": {
                    "deliverables": [
                        {"path": "reports/recommendations_preview.json", "required": False}
                    ]
                },
            },
            f,
        )
    with open(os.path.join("reports", "recommendations_preview.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "illustrative_only",
                "items": [{"segment": {"segment_key": "A"}, "expected_effect": {"metric": "score"}}],
                "reason": "",
                "caveats": ["Illustrative examples only."],
            },
            f,
        )

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Mejorar resultados"}
    )
    assert "Ejemplos ilustrativos" in report
