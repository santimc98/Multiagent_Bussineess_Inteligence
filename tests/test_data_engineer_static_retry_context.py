from src.graph import graph as graph_module
from src.utils.run_bundle import init_run_bundle


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
            return "import requests\nprint('hi')\n"
        return "# Error: stop"


def test_static_scan_retry_includes_violation_context(tmp_path, monkeypatch):
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
        "run_id": "testrun",
        "run_start_epoch": 0,
    }
    run_dir = tmp_path / "runs" / state["run_id"]
    init_run_bundle(state["run_id"], state, run_dir=str(run_dir))

    stub = StubDataEngineer()
    monkeypatch.setattr(graph_module, "data_engineer", stub)

    graph_module.run_data_engineer(state)

    assert len(stub.calls) == 2
    assert "STATIC_SCAN_VIOLATIONS" in stub.calls[1]
    assert "requests" in stub.calls[1]
