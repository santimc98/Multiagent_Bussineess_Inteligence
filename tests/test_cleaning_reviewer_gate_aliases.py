from src.agents.cleaning_reviewer import _check_row_count_sanity, _normalize_gate_name


def test_normalize_gate_name_numeric_parsing_verification():
    assert _normalize_gate_name("Numeric Parsing Verification") == "numeric_parsing_validation"


def test_row_count_sanity_supports_alternate_manifest_keys():
    manifest = {"row_counts": {"original": 352, "after_cleaning": 282}}
    params = {"max_drop_pct": 5.0}
    issues = _check_row_count_sanity(manifest, params)
    assert issues
