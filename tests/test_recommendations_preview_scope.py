from pathlib import Path

from src.utils.recommendations_preview import build_recommendations_preview


def test_recommendations_preview_rejects_out_of_scope_root(tmp_path):
    run_root = tmp_path / "run_root"
    outside_root = tmp_path / "outside_root"
    run_root.mkdir()
    outside_root.mkdir()

    preview = build_recommendations_preview(
        contract={},
        governance_summary={},
        artifacts_dir=str(outside_root),
        run_scoped_root=str(run_root),
    )

    assert preview["sources_checked"]
    assert "root_out_of_scope" in preview["sources_checked"][0]["reasons"]
    assert "no_valid_sources" in preview["reason"]
