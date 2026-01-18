import csv
from pathlib import Path

from src.graph.graph import _check_decisioning_columns


def test_decisioning_rank_no_upper_bound(tmp_path: Path) -> None:
    scored_rows = tmp_path / "data" / "scored_rows.csv"
    scored_rows.parent.mkdir(parents=True, exist_ok=True)

    with scored_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["priority_rank"])
        writer.writeheader()
        writer.writerow({"priority_rank": "999"})

    decisioning = {
        "output": {
            "file": str(scored_rows),
            "required_columns": [
                {
                    "name": "priority_rank",
                    "constraints": {"non_null_rate_min": 0.5, "range": {"min": 1, "max": None}},
                }
            ],
        }
    }

    result = _check_decisioning_columns(decisioning, max_rows=10)

    assert result.get("constraint_violations") == []
