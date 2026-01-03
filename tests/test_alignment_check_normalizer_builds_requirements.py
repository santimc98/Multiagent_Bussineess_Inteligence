from src.graph.graph import _normalize_alignment_check


def test_alignment_check_normalizer_builds_requirements():
    alignment_check = {
        "status": "PASS",
        "checks": [
            {"id": "req_1", "status": "PASS", "evidence": ["ok"]},
            {"name": "req_2", "status": "FAIL", "evidence": "bad"},
        ],
    }

    normalized, issues = _normalize_alignment_check(alignment_check, [])
    reqs = normalized.get("requirements")
    assert isinstance(reqs, list)

    status_map = {req.get("id"): req.get("status") for req in reqs}
    assert status_map.get("req_1") == "PASS"
    assert status_map.get("req_2") == "FAIL"
    assert "alignment_missing_requirement_status" not in issues
