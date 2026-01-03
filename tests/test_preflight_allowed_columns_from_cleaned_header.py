import os

from src.graph.graph import _resolve_allowed_columns_for_gate


def test_preflight_allowed_columns_from_cleaned_header(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "cleaned_data.csv"), "w", encoding="utf-8") as f:
        f.write("Size,Debtors\n1,10\n")
    state = {"csv_sep": ",", "csv_decimal": ".", "csv_encoding": "utf-8"}
    contract = {
        "canonical_columns": ["Size", "Debtors", "is_success"],
        "data_requirements": [
            {"name": "is_success", "source": "derived"},
        ],
    }
    allowed = _resolve_allowed_columns_for_gate(state, contract, {})
    assert "Size" in allowed
    assert "Debtors" in allowed
    assert "is_success" not in allowed
