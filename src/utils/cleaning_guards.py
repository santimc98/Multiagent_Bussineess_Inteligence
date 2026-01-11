import re
from typing import Any, Dict, List

import pandas as pd

_IDENTIFIER_TOKENS = {
    "id",
    "uuid",
    "guid",
    "key",
    "codigo",
    "code",
    "cod",
    "identifier",
    "reference",
    "ref",
    "account",
    "entity",
}


def is_identifier_like(name: str) -> bool:
    if not name:
        return False
    raw = str(name)
    lowered = raw.lower()
    tokens = [t for t in re.split(r"[^0-9a-zA-Z]+", lowered) if t]
    if any(token in _IDENTIFIER_TOKENS for token in tokens):
        return True
    if re.search(r"[A-Za-z0-9]+(?:id|ID)$", raw):
        return True
    return False


def detect_identifier_scientific_notation(
    series: pd.Series,
    sample_size: int = 200,
    e_ratio_threshold: float = 0.1,
    dot0_ratio_threshold: float = 0.6,
) -> Dict[str, Any]:
    result = {"flag": False, "e_ratio": 0.0, "dot0_ratio": 0.0, "examples": []}
    if series is None or series.empty:
        return result
    if series.dtype != object:
        return result
    values = series.dropna().astype(str).head(sample_size)
    if values.empty:
        return result
    matches_e = [v for v in values if "e+" in v.lower() or "e-" in v.lower()]
    matches_dot0 = [v for v in values if re.search(r"\.0+$", v.strip())]
    total = len(values)
    e_ratio = len(matches_e) / total if total else 0.0
    dot0_ratio = len(matches_dot0) / total if total else 0.0
    flag = e_ratio >= e_ratio_threshold or dot0_ratio >= dot0_ratio_threshold
    result.update(
        {
            "flag": flag,
            "e_ratio": e_ratio,
            "dot0_ratio": dot0_ratio,
            "examples": (matches_e + matches_dot0)[:5],
        }
    )
    return result
