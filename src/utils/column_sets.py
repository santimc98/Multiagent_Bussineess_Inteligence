import re
from typing import Any, Dict, List

_NUMERIC_SUFFIX_RE = re.compile(r"^(.*?)(\d+)$")
_EXPLICIT_ROLE_ALLOWLIST = {"target_candidate", "id_like", "split_candidate", "constant_like"}


def _normalize_columns(columns: List[Any]) -> List[str]:
    return [str(col) for col in columns if col is not None and str(col).strip()]


def _collect_explicit_columns(columns: List[str], roles: Dict[str, str] | None) -> List[str]:
    if not isinstance(roles, dict) or not roles:
        return []
    explicit: List[str] = []
    seen = set()
    for col in columns:
        if roles.get(col) in _EXPLICIT_ROLE_ALLOWLIST and col not in seen:
            seen.add(col)
            explicit.append(col)
    return explicit


def build_column_sets(
    columns: List[str],
    roles: Dict[str, str] | None = None,
    max_listed: int = 50,
) -> Dict[str, Any]:
    cols = _normalize_columns(columns or [])
    explicit_columns = _collect_explicit_columns(cols, roles)
    explicit_set = set(explicit_columns)
    remaining = [col for col in cols if col not in explicit_set]

    groups: Dict[str, List[tuple[int, str]]] = {}
    for col in remaining:
        match = _NUMERIC_SUFFIX_RE.match(col)
        if not match:
            continue
        prefix = match.group(1)
        if not prefix:
            continue
        idx = int(match.group(2))
        groups.setdefault(prefix, []).append((idx, col))

    sets: List[Dict[str, Any]] = []
    covered: set[str] = set()
    set_index = 1

    for prefix in sorted(groups.keys()):
        items = groups[prefix]
        if len(items) < 50:
            continue
        indices = sorted({idx for idx, _ in items})
        start = min(indices)
        end = max(indices)
        span = max(end - start + 1, 1)
        coverage = len(indices) / span
        matched = [col for _, col in items]

        if coverage >= 0.95:
            selector = {"type": "prefix_numeric_range", "prefix": prefix, "start": int(start), "end": int(end)}
        else:
            selector = {"type": "regex", "pattern": r"^" + re.escape(prefix) + r"\d+$"}

        if len(matched) >= 50:
            sets.append({"name": f"SET_{set_index}", "selector": selector, "count": len(matched)})
            covered.update(matched)
            set_index += 1

    if not sets and len(cols) >= 700:
        selector = {"type": "all_columns_except", "except_columns": explicit_columns}
        sets.append({"name": "SET_1", "selector": selector, "count": len(remaining)})
        covered.update(remaining)

    leftovers = [col for col in remaining if col not in covered]
    leftovers_sample = {
        "columns": leftovers[:max_listed],
        "total_leftovers": len(leftovers),
    }

    return {
        "explicit_columns": explicit_columns,
        "sets": sets,
        "leftovers_sample": leftovers_sample,
        "leftovers_count": len(leftovers),
    }


def expand_column_sets(columns: List[str], sets_spec: Dict[str, Any]) -> Dict[str, Any]:
    cols = _normalize_columns(columns or [])
    explicit_columns = _normalize_columns(sets_spec.get("explicit_columns") or [])
    explicit_set = set(explicit_columns)
    sets = sets_spec.get("sets") if isinstance(sets_spec, dict) else []
    expanded: List[str] = []
    expanded_seen = set()
    debug: Dict[str, int] = {}

    for entry in sets if isinstance(sets, list) else []:
        name = entry.get("name") or "SET"
        selector = entry.get("selector") if isinstance(entry, dict) else {}
        if not isinstance(selector, dict):
            selector = {}
        selector_type = selector.get("type")
        matched: List[str] = []

        if selector_type == "prefix_numeric_range":
            prefix = str(selector.get("prefix") or "")
            start = selector.get("start")
            end = selector.get("end")
            if prefix and isinstance(start, int) and isinstance(end, int):
                for col in cols:
                    if col in explicit_set:
                        continue
                    match = _NUMERIC_SUFFIX_RE.match(col)
                    if not match:
                        continue
                    if match.group(1) != prefix:
                        continue
                    idx = int(match.group(2))
                    if start <= idx <= end:
                        matched.append(col)
        elif selector_type == "regex":
            pattern = selector.get("pattern")
            if pattern:
                regex = re.compile(str(pattern))
                matched = [col for col in cols if col not in explicit_set and regex.match(col)]
        elif selector_type == "all_numeric_except":
            excluded = set(_normalize_columns(selector.get("except_columns") or [])) | explicit_set
            matched = [col for col in cols if col not in excluded]
        elif selector_type == "all_columns_except":
            excluded = set(_normalize_columns(selector.get("except_columns") or [])) | explicit_set
            matched = [col for col in cols if col not in excluded]

        for col in matched:
            if col in expanded_seen or col in explicit_set:
                continue
            expanded_seen.add(col)
            expanded.append(col)
        debug[name] = len(matched)

    return {
        "expanded_feature_columns": expanded,
        "debug": debug,
    }


def summarize_column_sets(column_sets: Dict[str, Any], max_sets: int = 5) -> str:
    if not isinstance(column_sets, dict) or not column_sets:
        return ""
    explicit = column_sets.get("explicit_columns") or []
    sets = column_sets.get("sets") or []
    leftovers = column_sets.get("leftovers_sample") or {}
    lines = ["COLUMN_SETS_SUMMARY:"]
    lines.append(f"- explicit_columns_count: {len(explicit)}")
    if sets:
        set_summaries = []
        for entry in sets[:max_sets]:
            selector = entry.get("selector") if isinstance(entry, dict) else {}
            selector_type = selector.get("type") if isinstance(selector, dict) else "unknown"
            count = entry.get("count")
            name = entry.get("name") or "SET"
            if selector_type == "prefix_numeric_range":
                prefix = selector.get("prefix")
                start = selector.get("start")
                end = selector.get("end")
                detail = f"{prefix}{start}-{end}"
            elif selector_type == "regex":
                detail = selector.get("pattern")
            else:
                detail = selector_type
            count_text = count if isinstance(count, int) else "unknown"
            set_summaries.append(f"{name}({detail}, count={count_text})")
        lines.append(f"- sets: {set_summaries}")
    else:
        lines.append("- sets: none")
    total_leftovers = leftovers.get("total_leftovers")
    if isinstance(total_leftovers, int) and total_leftovers > 0:
        lines.append(f"- leftovers_total: {total_leftovers}")
    return "\n".join(lines)
