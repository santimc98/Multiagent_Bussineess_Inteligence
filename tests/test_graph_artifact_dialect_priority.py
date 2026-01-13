import json
import os

import pytest

from src.graph import graph as graph_mod


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    old_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


def test_artifact_gate_dialect_prefers_state(tmp_workdir):
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "cleaning_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"output_dialect": {"sep": ";", "decimal": ",", "encoding": "latin-1"}}, handle)

    state = {"csv_sep": "\t", "csv_decimal": ".", "csv_encoding": "utf-8"}
    contract = {"output_dialect": {"sep": ";", "decimal": ",", "encoding": "latin-1"}}

    dialect = graph_mod._resolve_artifact_gate_dialect(state, contract)
    assert dialect["sep"] == "\t"
    assert dialect["decimal"] == "."
    assert dialect["encoding"] == "utf-8"
