import csv
import json
import re
from typing import Any, Dict, List, Optional


def load_output_dialect(manifest_path: str = "data/cleaning_manifest.json") -> Dict[str, Any]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return {}
    if not isinstance(manifest, dict):
        return {}
    dialect = manifest.get("output_dialect") or manifest.get("dialect") or {}
    if not isinstance(dialect, dict):
        return {}
    sep = dialect.get("sep") or dialect.get("delimiter")
    decimal = dialect.get("decimal")
    encoding = dialect.get("encoding")
    cleaned: Dict[str, Any] = {}
    if sep:
        cleaned["sep"] = str(sep)
    if decimal:
        cleaned["decimal"] = str(decimal)
    if encoding:
        cleaned["encoding"] = str(encoding)
    return cleaned


def sniff_csv_dialect(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
    except Exception:
        return {"sep": ",", "decimal": ".", "encoding": "utf-8"}

    sample_text = "".join(lines)
    header = lines[0] if lines else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    comma_decimals = len(re.findall(r"\d+,\d+", sample_text))
    dot_decimals = len(re.findall(r"\d+\.\d+", sample_text))
    decimal = "," if comma_decimals > dot_decimals else "."
    return {"sep": sep, "decimal": decimal, "encoding": "utf-8"}


def coerce_number(raw: Any, decimal: str) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace(" ", "")
    if decimal == ",":
        if text.count(",") == 1 and text.count(".") >= 1:
            text = text.replace(".", "")
        text = text.replace(",", ".")
    try:
        return float(text)
    except Exception:
        return None


def read_csv_sample(path: str, dialect: Dict[str, Any], max_rows: int) -> Dict[str, Any]:
    if not path:
        return {}
    sep = dialect.get("sep") or ","
    encoding = dialect.get("encoding") or "utf-8"
    rows: List[Dict[str, Any]] = []
    row_count_total = 0
    try:
        with open(path, "r", encoding=str(encoding), errors="replace") as f:
            reader = csv.DictReader(f, delimiter=str(sep))
            columns = reader.fieldnames or []
            for row in reader:
                row_count_total += 1
                if len(rows) < max_rows:
                    rows.append(row)
        return {
            "columns": columns,
            "rows": rows,
            "row_count_total": row_count_total,
            "row_count_sampled": len(rows),
            "dialect_used": {
                "sep": sep,
                "decimal": dialect.get("decimal") or ".",
                "encoding": encoding,
            },
        }
    except Exception:
        return {}
