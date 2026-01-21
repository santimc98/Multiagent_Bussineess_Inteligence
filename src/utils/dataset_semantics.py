from typing import Any, Dict, List, Optional


def _as_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _extract_primary_target(semantics: Dict[str, Any]) -> str | None:
    primary = semantics.get("primary_target")
    if isinstance(primary, str) and primary.strip():
        return primary
    target_info = semantics.get("target_analysis")
    if isinstance(target_info, dict):
        primary = target_info.get("primary_target")
        if isinstance(primary, str) and primary.strip():
            return primary
    return None


def summarize_dataset_semantics(
    semantics: Dict[str, Any],
    training_mask: Optional[Dict[str, Any]] = None,
    max_lines: int = 18,
) -> str:
    semantics = semantics if isinstance(semantics, dict) else {}
    training_mask = training_mask if isinstance(training_mask, dict) else {}

    primary_target = _extract_primary_target(semantics)
    target_info = semantics.get("target_analysis") if isinstance(semantics.get("target_analysis"), dict) else {}
    partial_labels = target_info.get("partial_label_detected") if isinstance(target_info, dict) else None
    null_frac_exact = target_info.get("target_null_frac_exact") if isinstance(target_info, dict) else None
    missing_exact = target_info.get("target_missing_count_exact") if isinstance(target_info, dict) else None
    total_exact = target_info.get("target_total_count_exact") if isinstance(target_info, dict) else None

    split_candidates = _as_list(semantics.get("split_candidates"))
    partition_info = semantics.get("partition_analysis") if isinstance(semantics.get("partition_analysis"), dict) else {}
    if not split_candidates:
        split_candidates = _as_list(partition_info.get("partition_columns"))
    id_candidates = _as_list(semantics.get("id_candidates"))

    lines: List[str] = []
    lines.append("DATASET_SEMANTICS_SUMMARY:")
    lines.append(f"- primary_target: {primary_target or 'missing'}")
    if partial_labels is not None:
        lines.append(f"- partial_label_detected: {bool(partial_labels)}")
    if null_frac_exact is not None:
        ratio = round(float(null_frac_exact), 6)
        if total_exact is not None and missing_exact is not None:
            lines.append(f"- target_null_frac_exact: {ratio} ({missing_exact}/{total_exact})")
        else:
            lines.append(f"- target_null_frac_exact: {ratio}")
    if split_candidates:
        lines.append(f"- split_candidates: {split_candidates[:5]}")
    if id_candidates:
        lines.append(f"- id_candidates: {id_candidates[:5]}")
    if training_mask:
        lines.append(f"- training_rows_rule: {training_mask.get('training_rows_rule', '')}")
        lines.append(f"- scoring_rows_rule_primary: {training_mask.get('scoring_rows_rule_primary', '')}")
        secondary = training_mask.get("scoring_rows_rule_secondary")
        if secondary:
            lines.append(f"- scoring_rows_rule_secondary: {secondary}")
        rationale = training_mask.get("rationale")
        if isinstance(rationale, list) and rationale:
            lines.append(f"- rationale: {rationale[:3]}")

    return "\n".join(lines[:max_lines])
