import os
from pathlib import Path

from src.utils.output_contract import check_required_outputs


def test_output_contract_present(tmp_path: Path):
    file_path = tmp_path / "data" / "cleaned_data.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("x\n1", encoding="utf-8")
    report = check_required_outputs([str(file_path)])
    assert report["missing"] == []
    assert str(file_path) in report["present"]


def test_output_contract_glob_missing(tmp_path: Path):
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        report = check_required_outputs(["static/plots/*.png"])
        assert "static/plots/*.png" in report["missing"]
    finally:
        os.chdir(cwd)


def test_output_contract_optional_missing(tmp_path: Path):
    file_path = tmp_path / "data" / "cleaned_data.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("x\n1", encoding="utf-8")
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        report = check_required_outputs(
            [
                {"path": str(file_path), "required": True},
                {"path": "static/plots/*.png", "required": False},
            ]
        )
        assert report["missing"] == []
        assert "static/plots/*.png" in report.get("missing_optional", [])
    finally:
        os.chdir(cwd)
