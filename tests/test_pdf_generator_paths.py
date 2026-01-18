from pathlib import Path

from src.utils.pdf_generator import resolve_image_path


def test_resolve_image_path_falls_back_to_artifacts(tmp_path: Path) -> None:
    plots_dir = tmp_path / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "plot.png"
    plot_path.write_bytes(b"test")

    resolved = resolve_image_path("static/plots/plot.png", str(tmp_path))

    assert resolved == str(plot_path)
