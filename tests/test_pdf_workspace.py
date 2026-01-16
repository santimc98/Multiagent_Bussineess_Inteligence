import os
import time
from pathlib import Path
import pytest


def test_pdf_generated_in_workdir_not_cwd(tmp_path: Path, monkeypatch):
    """
    P1.6: PDF should be generated in work_dir even if cwd is elsewhere.
    Verify no PDF is created outside work_dir.
    """
    from src.graph.graph import generate_pdf_artifact
    from src.utils.pdf_generator import convert_report_to_pdf

    # Create work_dir structure
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir()
    data_dir = work_dir / "data"
    data_dir.mkdir()
    summary_path = data_dir / "executive_summary.md"
    summary_path.write_text("# Test Report\n\nContent", encoding="utf-8")

    # Create separate cwd_root
    cwd_root = tmp_path / "cwd_root"
    cwd_root.mkdir()

    # Create bundle_dir for copy
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    # Monkeypatch chdir to simulate cwd != work_dir
    original_cwd = os.getcwd()
    monkeypatch.chdir(cwd_root)

    # Monkeypatch convert_report_to_pdf to avoid pandoc/latex
    def fake_convert(md: str, pdf_path: str) -> bool:
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n%%EOF")
        return True

    monkeypatch.setattr("src.graph.graph.convert_report_to_pdf", fake_convert)

    try:
        # Call generate_pdf_artifact with work_dir in state
        state = {
            "work_dir": str(work_dir),
            "final_report": "# Test Report\n\nContent",
            "run_id": None,
            "run_bundle_dir": str(bundle_dir),
            "run_start_epoch": time.time(),
        }

        result = generate_pdf_artifact(state)

        # Assertions
        assert result["pdf_path"] is not None, "PDF path should be returned"
        pdf_path = result["pdf_path"]
        assert os.path.exists(pdf_path), f"PDF should exist at {pdf_path}"

        # PDF should be inside work_dir
        pdf_posix = Path(pdf_path).as_posix()
        work_dir_posix = work_dir.as_posix()
        assert pdf_posix.startswith(work_dir_posix), f"PDF {pdf_path} should be in work_dir {work_dir}"

        # NO PDF should exist in cwd_root
        cwd_pdfs = list(cwd_root.glob("final_report*.pdf"))
        assert len(cwd_pdfs) == 0, f"No PDF should be in cwd_root, found {cwd_pdfs}"

        # PDF should be copied to bundle_dir/report/
        bundle_pdf = bundle_dir / "report" / "final_report.pdf"
        assert bundle_pdf.exists(), f"PDF should be copied to bundle at {bundle_pdf}"

    finally:
        os.chdir(original_cwd)


def test_pdf_includes_plots_from_workdir(tmp_path: Path, monkeypatch):
    """
    P1.6: PDF should find plots from work_dir even if cwd is elsewhere.
    """
    from src.graph.graph import generate_pdf_artifact
    from src.utils.pdf_generator import convert_report_to_pdf

    # Create work_dir structure with plots
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir()
    plots_dir = work_dir / "report" / "static" / "plots"
    plots_dir.mkdir(parents=True)

    # Create a dummy plot
    plot_path = plots_dir / "test_plot.png"
    plot_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    # Create separate cwd_root
    cwd_root = tmp_path / "cwd_root"
    cwd_root.mkdir()

    # Create bundle_dir
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    # Monkeypatch chdir
    original_cwd = os.getcwd()
    monkeypatch.chdir(cwd_root)

    # Monkeypatch convert_report_to_pdf to capture the markdown
    captured_md = []

    def fake_convert(md: str, pdf_path: str) -> bool:
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        captured_md.append(md)
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n%%EOF")
        return True

    monkeypatch.setattr("src.graph.graph.convert_report_to_pdf", fake_convert)

    try:
        state = {
            "work_dir": str(work_dir),
            "final_report": "# Report\n\n",
            "run_id": None,
            "run_bundle_dir": str(bundle_dir),
            "run_start_epoch": time.time(),
            "plots_local": [],  # No plots in state, should find in filesystem
        }

        result = generate_pdf_artifact(state)

        # Verify PDF was created
        assert result["pdf_path"] is not None
        assert os.path.exists(result["pdf_path"])

        # Verify plot was included in markdown
        assert len(captured_md) == 1
        md = captured_md[0]
        assert "test_plot.png" in md, f"Plot reference should be in markdown: {md}"
    finally:
        os.chdir(original_cwd)


def test_pdf_generation_uses_workdir_as_cwd(tmp_path: Path, monkeypatch):
    """
    P1.6.1: PDF generation should temporarily change CWD to work_dir.
    Verify that os.getcwd() during convert_report_to_pdf is work_dir.
    """
    from src.graph.graph import generate_pdf_artifact

    # Create work_dir structure
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir()
    data_dir = work_dir / "data"
    data_dir.mkdir()
    summary_path = data_dir / "executive_summary.md"
    summary_path.write_text("# Test Report\n\nContent", encoding="utf-8")

    # Create separate cwd_root
    cwd_root = tmp_path / "cwd_root"
    cwd_root.mkdir()

    # Create bundle_dir
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    # Monkeypatch chdir to simulate cwd != work_dir
    original_cwd = os.getcwd()
    monkeypatch.chdir(cwd_root)

    # Capture CWD during convert_report_to_pdf
    captured_cwd = []

    def fake_convert(md: str, pdf_path: str) -> bool:
        captured_cwd.append(os.getcwd())
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n%%EOF")
        return True

    monkeypatch.setattr("src.graph.graph.convert_report_to_pdf", fake_convert)

    try:
        # Call generate_pdf_artifact with work_dir in state
        state = {
            "work_dir": str(work_dir),
            "final_report": "# Test Report\n\nContent",
            "run_id": None,
            "run_bundle_dir": str(bundle_dir),
            "run_start_epoch": time.time(),
        }

        result = generate_pdf_artifact(state)

        # Verify CWD during convert was work_dir
        assert len(captured_cwd) == 1, "convert_report_to_pdf should be called once"
        convert_cwd = Path(captured_cwd[0]).as_posix()
        work_dir_posix = work_dir.as_posix()
        assert convert_cwd == work_dir_posix, f"CWD should be work_dir: {convert_cwd} vs {work_dir_posix}"

        # Verify PDF was created
        assert result["pdf_path"] is not None
        assert os.path.exists(result["pdf_path"])
    finally:
        os.chdir(original_cwd)
