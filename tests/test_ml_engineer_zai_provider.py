from src.agents.ml_engineer import MLEngineerAgent


class FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout


def test_ml_engineer_zai_provider_uses_zai_base_url(monkeypatch):
    monkeypatch.setenv("ML_ENGINEER_PROVIDER", "zai")
    monkeypatch.setenv("ZAI_API_KEY", "dummy-zai")
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()

    assert agent.client.base_url == "https://api.z.ai/api/paas/v4/"
    assert agent.model_name == "glm-4.7"
