from src.graph import graph as graph_module


class StubDataEngineer:
    def __init__(self) -> None:
        self.calls = []
        self.model_name = "stub"
        self.last_prompt = None
        self.last_response = None

    def generate_cleaning_script(
        self,
        data_audit,
        strategy,
        input_path,
        business_objective="",
        csv_encoding="utf-8",
        csv_sep=",",
        csv_decimal=".",
        execution_contract=None,
    ) -> str:
        self.calls.append(data_audit)
        if len(self.calls) == 1:
            self.last_response = "```python\nprint('clean')\n```"
            return "print('clean')"
        self.last_response = "print('clean')"
        return "# Error: stop"


def test_data_engineer_does_not_retry_on_code_fence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    state = {
        "csv_path": str(csv_path),
        "selected_strategy": {"required_columns": []},
        "business_objective": "test",
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "data_summary": "summary",
        "execution_contract": {"required_outputs": ["data/cleaned_data.csv"]},
    }

    stub = StubDataEngineer()
    monkeypatch.setattr(graph_module, "data_engineer", stub)

    graph_module.run_data_engineer(state)

    assert len(stub.calls) == 1
    assert all("CODE_FENCE_GUARD" not in call for call in stub.calls)
