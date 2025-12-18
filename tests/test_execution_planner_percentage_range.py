from src.agents.execution_planner import enforce_percentage_ranges


def test_percentage_role_sets_range():
    contract = {
        "data_requirements": [
            {"name": "col_pct", "role": "percentage", "expected_range": None}
        ],
        "notes_for_engineers": [],
    }
    updated = enforce_percentage_ranges(contract)
    req = updated["data_requirements"][0]
    assert req["expected_range"] == [0, 1]
    assert any("Percentages must be normalized" in n for n in updated["notes_for_engineers"])
