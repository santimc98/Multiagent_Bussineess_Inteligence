from pathlib import Path


def _extract_function_block(source: str, func_name: str) -> str:
    marker = f"def {func_name}("
    start = source.find(marker)
    if start == -1:
        return ""
    next_def = source.find("\ndef ", start + len(marker))
    if next_def == -1:
        return source[start:]
    return source[start:next_def]


def test_run_data_engineer_defines_attempt_id_before_use():
    source = Path("src/graph/graph.py").read_text(encoding="utf-8")
    block = _extract_function_block(source, "run_data_engineer")
    assert block, "run_data_engineer not found"
    assign_pos = block.find("attempt_id =")
    use_pos = block.find("attempt=attempt_id")
    assert assign_pos != -1, "attempt_id assignment missing in run_data_engineer"
    assert use_pos != -1, "attempt=attempt_id usage missing in run_data_engineer"
    assert assign_pos < use_pos, "attempt_id assigned after use in run_data_engineer"
