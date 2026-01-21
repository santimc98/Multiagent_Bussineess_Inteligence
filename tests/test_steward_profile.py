import json
from pathlib import Path

import pandas as pd

from src.agents.steward import build_dataset_profile, write_dataset_profile


def test_dataset_profile_written(tmp_path: Path):
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "amount": [10.5, 20.0, None],
            "event_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    profile = build_dataset_profile(
        df=df,
        objective="Predict churn",
        dialect_info={"sep": ",", "decimal": "."},
        encoding="utf-8",
        file_size_bytes=123,
        was_sampled=False,
        sample_size=3,
        pii_findings={"detected": False, "findings": []},
    )
    out_path = tmp_path / "dataset_profile.json"
    write_dataset_profile(profile, path=str(out_path))
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    for key in [
        "rows",
        "cols",
        "columns",
        "type_hints",
        "missing_frac",
        "cardinality",
        "pii_findings",
        "sampling",
        "dialect",
    ]:
        assert key in saved
