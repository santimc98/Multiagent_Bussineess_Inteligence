import csv
import random
from collections import deque
from typing import Any, Dict, List

import pandas as pd

_NULL_STRINGS = {"", "na", "n/a", "nan", "null", "none", "nat"}


def read_header(csv_path: str, dialect: Dict[str, Any]) -> List[str]:
    if not csv_path:
        return []
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    try:
        header_df = pd.read_csv(csv_path, nrows=0, sep=sep, decimal=decimal, encoding=encoding)
        return [str(col) for col in header_df.columns]
    except Exception:
        pass
    try:
        with open(csv_path, "r", encoding=encoding, errors="replace") as handle:
            reader = csv.reader(handle, delimiter=sep)
            header = next(reader, [])
        return [str(col) for col in header]
    except Exception:
        return []


def scan_missingness(
    csv_path: str,
    dialect: Dict[str, Any],
    col: str,
    chunksize: int = 200000,
) -> Dict[str, Any]:
    if not csv_path or not col:
        return {"column": col, "total": 0, "missing": 0, "null_frac_exact": None}
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    total = 0
    missing = 0
    try:
        reader = pd.read_csv(
            csv_path,
            usecols=[col],
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            dtype="string",
            keep_default_na=False,
            chunksize=max(1, int(chunksize)),
            low_memory=False,
        )
    except Exception:
        return {"column": col, "total": 0, "missing": 0, "null_frac_exact": None}

    for chunk in reader:
        if col not in chunk.columns:
            continue
        series = chunk[col]
        total += int(series.shape[0])
        cleaned = series.astype("string").str.strip()
        lowered = cleaned.str.lower()
        missing_mask = series.isna() | (lowered == "") | lowered.isin(_NULL_STRINGS)
        missing += int(missing_mask.sum())
    null_frac_exact = float(missing / total) if total else None
    return {"column": col, "total": int(total), "missing": int(missing), "null_frac_exact": null_frac_exact}


def scan_uniques(
    csv_path: str,
    dialect: Dict[str, Any],
    col: str,
    chunksize: int = 200000,
    max_unique: int = 20,
) -> Dict[str, Any]:
    if not csv_path or not col:
        return {"column": col, "unique_values": [], "counts_hint": [], "total": 0}
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    total = 0
    counts: Dict[str, int] = {}
    try:
        reader = pd.read_csv(
            csv_path,
            usecols=[col],
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            dtype="string",
            keep_default_na=False,
            chunksize=max(1, int(chunksize)),
            low_memory=False,
        )
    except Exception:
        return {"column": col, "unique_values": [], "counts_hint": [], "total": 0}

    for chunk in reader:
        if col not in chunk.columns:
            continue
        series = chunk[col]
        cleaned = series.astype("string").str.strip()
        lowered = cleaned.str.lower()
        missing_mask = series.isna() | (lowered == "") | lowered.isin(_NULL_STRINGS)
        values = cleaned[~missing_mask]
        total += int(values.shape[0])
        for raw_val in values.tolist():
            val = str(raw_val)
            if val in counts:
                counts[val] += 1
            elif len(counts) < max_unique:
                counts[val] = 1

    counts_hint = [
        {"value": key, "count": int(count)} for key, count in sorted(counts.items(), key=lambda item: -item[1])
    ]
    unique_values = [entry["value"] for entry in counts_hint]
    return {"column": col, "unique_values": unique_values, "counts_hint": counts_hint, "total": int(total)}


def sample_rows(
    csv_path: str,
    dialect: Dict[str, Any],
    head_n: int = 50,
    tail_n: int = 50,
    random_n: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    if not csv_path:
        return {"head": [], "tail": [], "random": [], "total_rows": 0}
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    head: List[Dict[str, Any]] = []
    tail: deque[Dict[str, Any]] = deque(maxlen=max(0, int(tail_n)))
    reservoir: List[Dict[str, Any]] = []
    total_rows = 0
    rng = random.Random(seed)
    try:
        reader = pd.read_csv(
            csv_path,
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            dtype="string",
            keep_default_na=False,
            chunksize=5000,
            low_memory=False,
        )
    except Exception:
        return {"head": [], "tail": [], "random": [], "total_rows": 0}

    for chunk in reader:
        records = chunk.to_dict(orient="records")
        for row in records:
            total_rows += 1
            if len(head) < head_n:
                head.append(row)
            if tail_n > 0:
                tail.append(row)
            if random_n > 0:
                if len(reservoir) < random_n:
                    reservoir.append(row)
                else:
                    idx = rng.randrange(total_rows)
                    if idx < random_n:
                        reservoir[idx] = row

    return {
        "head": head,
        "tail": list(tail),
        "random": reservoir,
        "total_rows": total_rows,
    }
