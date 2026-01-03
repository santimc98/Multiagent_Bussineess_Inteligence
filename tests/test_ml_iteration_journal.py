from src.graph.graph import _append_ml_iteration_journal
from src.graph.graph import _build_ml_iteration_memory_block
from src.graph.graph import _load_ml_iteration_journal


def test_ml_iteration_journal_append_and_load(tmp_path):
    entry = {
        "iteration_id": 1,
        "code_hash": "abc123",
        "preflight_issues": ["UNKNOWN_COLUMNS_REFERENCED"],
        "runtime_error": None,
        "outputs_present": [],
        "outputs_missing": ["data/metrics.json"],
        "reviewer_verdict": "REJECTED",
        "reviewer_reasons": ["unknown_columns"],
        "qa_verdict": "UNKNOWN",
        "qa_reasons": [],
        "next_actions": ["Use only contract columns"],
    }
    written = _append_ml_iteration_journal("run1", entry, [], base_dir=str(tmp_path))
    path = tmp_path / "run1" / "ml_iteration_journal.jsonl"
    assert path.exists()
    entries = _load_ml_iteration_journal("run1", base_dir=str(tmp_path))
    assert entries and entries[0]["iteration_id"] == 1
    assert written == [1]


def test_ml_iteration_memory_block_compact():
    entries = [
        {
            "iteration_id": 1,
            "preflight_issues": ["UNKNOWN_COLUMNS_REFERENCED"],
            "runtime_error": None,
            "outputs_missing": ["data/metrics.json"],
            "reviewer_verdict": "REJECTED",
            "reviewer_reasons": ["unknown_columns"],
            "qa_verdict": "UNKNOWN",
            "qa_reasons": ["synthetic"],
            "next_actions": ["Use only contract columns", "Add baseline"],
        },
        {
            "iteration_id": 2,
            "preflight_issues": [],
            "runtime_error": {"type": "runtime_error", "message": "ValueError"},
            "outputs_missing": ["data/alignment_check.json"],
            "reviewer_verdict": "NEEDS_IMPROVEMENT",
            "reviewer_reasons": ["baseline_missing"],
            "qa_verdict": "UNKNOWN",
            "qa_reasons": [],
            "next_actions": ["Add baseline model", "Write required outputs"],
        },
    ]
    block = _build_ml_iteration_memory_block(entries, max_chars=220)
    assert "Last attempt summary" in block
    assert "Top recurring failures" in block
    assert "baseline_missing" in block or "unknown_columns" in block
    lines = [line for line in block.splitlines() if line.startswith("DO:") or line.startswith("Don't")]
    assert len(lines) <= 6
