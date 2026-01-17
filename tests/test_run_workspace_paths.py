import os
from src.utils.run_workspace import enter_run_workspace


def test_enter_run_workspace_patches_relative_csv_path(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "x.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    state = {"csv_path": os.path.join("data", "x.csv")}
    run_dir = tmp_path / "runs" / "abc"

    state = enter_run_workspace(state, str(run_dir))

    assert os.path.isabs(state["csv_path"])
    assert os.path.exists(state["csv_path"])
    expected_work_dir = os.path.normpath(os.path.join(str(run_dir), "work"))
    assert os.path.normcase(os.getcwd()) == os.path.normcase(expected_work_dir)
