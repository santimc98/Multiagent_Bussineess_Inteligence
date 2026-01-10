import json
import os

from src.agents.business_translator import _safe_load_csv


def test_business_translator_parses_semicolon_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "cleaning_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"output_dialect": {"sep": ";", "decimal": ",", "encoding": "utf-8"}}, f)

    scored_rows_path = os.path.join("data", "scored_rows.csv")
    with open(scored_rows_path, "w", encoding="utf-8") as f:
        f.write("Size;Debtors;Probability\n")
        f.write("10,5;2,0;0,7\n")
        f.write("11,0;3,5;0,8\n")

    parsed = _safe_load_csv(scored_rows_path, max_rows=10)
    assert parsed is not None
    assert isinstance(parsed.get("columns"), list)
    assert len(parsed.get("columns", [])) == 3
    assert parsed.get("row_count_total") == 2
