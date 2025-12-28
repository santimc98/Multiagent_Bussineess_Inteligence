import json
import os

import pandas as pd
import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("E2B_API_KEY", "dummy")

from src.utils.dialect_guardrails import (
    assert_not_single_column_delimiter_mismatch,
    get_output_dialect_from_manifest,
)
from src.graph.graph import run_engineer


def test_output_dialect_override_used_before_ml(monkeypatch, tmp_path):
    manifest_path = tmp_path / "cleaning_manifest.json"
    manifest_path.write_text(
        json.dumps({"output_dialect": {"sep": ";", "decimal": ",", "encoding": "latin-1"}}),
        encoding="utf-8",
    )

    sep, decimal, encoding, updated = get_output_dialect_from_manifest(
        str(manifest_path), ",", ".", "utf-8"
    )
    assert updated is True
    assert (sep, decimal, encoding) == (";", ",", "latin-1")

    calls = {}

    def fake_generate_code(
        self,
        strategy,
        data_path,
        feedback_history,
        **kwargs,
    ):
        calls["csv_encoding"] = kwargs.get("csv_encoding")
        calls["csv_sep"] = kwargs.get("csv_sep")
        calls["csv_decimal"] = kwargs.get("csv_decimal")
        return "# ok"

    monkeypatch.setattr(
        "src.agents.ml_engineer.MLEngineerAgent.generate_code", fake_generate_code, raising=True
    )

    state = {
        "selected_strategy": {"title": "t", "analysis_type": "predictive"},
        "feedback_history": [],
        "data_summary": "",
        "business_objective": "",
        "csv_encoding": encoding,
        "csv_sep": sep,
        "csv_decimal": decimal,
        "last_generated_code": None,
        "last_gate_context": None,
    }

    run_engineer(state)

    assert calls["csv_encoding"] == "latin-1"
    assert calls["csv_sep"] == ";"
    assert calls["csv_decimal"] == ","


def test_single_column_guardrail_triggers_value_error():
    colname = "customer_id,price,region,campaign,extra_fields"
    df = pd.DataFrame({colname: [1, 2, 3]})
    with pytest.raises(ValueError) as excinfo:
        assert_not_single_column_delimiter_mismatch(df, ";", ",", "latin-1")
    message = str(excinfo.value)
    assert "Delimiter/Dialect mismatch" in message
    assert "sep=';'" in message
