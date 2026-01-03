import csv
import fnmatch
import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple


def _safe_load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_load_rows(path: str, max_rows: int = 500) -> Tuple[List[str], List[Dict[str, Any]]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, Any]] = []
            for _, row in zip(range(max_rows), reader):
                rows.append(row)
            return reader.fieldnames or [], rows
    except Exception:
        return [], []


def _is_number(value: Any) -> bool:
    try:
        float(str(value).replace(",", "."))
        return True
    except Exception:
        return False


def _normalize_reporting_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
    policy = contract.get("reporting_policy")
    if not isinstance(policy, dict):
        policy = {}
    defaults = {
        "demonstrative_examples_enabled": True,
        "demonstrative_examples_when_outcome_in": ["NO_GO", "GO_WITH_LIMITATIONS"],
        "max_examples": 5,
        "require_strong_disclaimer": True,
    }
    merged = dict(defaults)
    merged.update({k: v for k, v in policy.items() if v is not None})
    return merged


def _normalize_deliverables(contract: Dict[str, Any]) -> Dict[str, Any]:
    required_paths: List[str] = []
    optional_paths: List[str] = []
    kinds_by_path: Dict[str, str] = {}
    spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
    deliverables = spec.get("deliverables") if isinstance(spec, dict) else None
    if isinstance(deliverables, list) and deliverables:
        for item in deliverables:
            if isinstance(item, dict):
                path = item.get("path") or item.get("output") or item.get("artifact")
                if not path:
                    continue
                kind = item.get("kind")
                if not kind:
                    lower = str(path).lower()
                    if lower.startswith("static/plots/") or lower.endswith((".png", ".jpg", ".jpeg", ".svg")):
                        kind = "plot"
                if kind:
                    kinds_by_path[str(path)] = str(kind)
                is_required = item.get("required")
                if is_required is None:
                    is_required = True
                if is_required:
                    required_paths.append(str(path))
                else:
                    optional_paths.append(str(path))
            elif isinstance(item, str):
                required_paths.append(item)
    else:
        required_paths.extend(contract.get("required_outputs", []) or [])
    requires_plots = any(str(kind).lower() == "plot" for kind in kinds_by_path.values())
    return {
        "required_paths": required_paths,
        "optional_paths": optional_paths,
        "kinds_by_path": kinds_by_path,
        "requires_plots": requires_plots,
    }


def _path_matches(pattern: str, rel_path: str) -> bool:
    if not pattern or not rel_path:
        return False
    if any(ch in pattern for ch in ["*", "?", "["]):
        return fnmatch.fnmatch(rel_path, pattern)
    return pattern == rel_path


def _count_present(paths: List[str], file_set: set[str]) -> Tuple[int, List[str]]:
    present = []
    for path in paths:
        if any(_path_matches(path, rel) for rel in file_set):
            present.append(path)
    return len(present), present


def _normalize_produced_index(entries: Any) -> set[str]:
    file_set: set[str] = set()
    if isinstance(entries, dict):
        entries = entries.get("present") or entries.get("paths") or []
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, dict):
                path = item.get("path")
            else:
                path = item
            if path:
                file_set.add(str(path).replace("\\", "/"))
    return file_set


def _load_output_contract_report(root_dir: str) -> Dict[str, Any]:
    if not root_dir:
        return {}
    candidates = [
        os.path.join(root_dir, "data", "output_contract_report.json"),
        os.path.join(root_dir, "report", "output_contract_report.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            payload = _safe_load_json(path)
            if isinstance(payload, dict):
                return payload
    return {}


def _ensure_item_fields(item: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    normalized = dict(item)
    if "title" not in normalized:
        title = None
        for key in ("title", "name", "action", "recommendation", "label", "id", "segment_id"):
            val = normalized.get(key)
            if val not in (None, ""):
                title = str(val)
                break
        normalized["title"] = title or ""
    if "reason" not in normalized:
        reason = None
        for key in ("reason", "rationale", "notes", "because"):
            val = normalized.get(key)
            if val not in (None, ""):
                reason = val
                break
        normalized["reason"] = reason if reason is not None else ""
    caveats = normalized.get("caveats")
    if not isinstance(caveats, list):
        if caveats in (None, ""):
            caveats = []
        else:
            caveats = [str(caveats)]
    normalized["caveats"] = caveats
    normalized["source_path"] = source_path
    return normalized


def _extract_items_from_payload(payload: Any, source_path: str) -> List[Dict[str, Any]]:
    if not payload:
        return []
    candidates: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        candidates = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        for key in ("items", "recommendations", "actions", "next_steps"):
            if isinstance(payload.get(key), list):
                candidates = [item for item in payload.get(key) if isinstance(item, dict)]
                break
    return [_ensure_item_fields(item, source_path) for item in candidates]


def _extract_recommended_pairs(payload: Any, source_path: str) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    items: List[Dict[str, Any]] = []
    for key, value in payload.items():
        if not str(key).lower().startswith("recommended_"):
            continue
        items.append(
            _ensure_item_fields(
                {
                    "title": str(key),
                    "value": value,
                },
                source_path,
            )
        )
    return items


def _extract_items_from_legacy_summary(payload: Any, source_path: str) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return []
    items: List[Dict[str, Any]] = []
    for entry in segments:
        if not isinstance(entry, dict):
            continue
        items.append(_ensure_item_fields(dict(entry), source_path))
    return items


def _load_cleaning_dialect(artifacts_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(artifacts_dir, "data", "cleaning_manifest.json")
    manifest = _safe_load_json(manifest_path)
    if not isinstance(manifest, dict):
        return {}
    return manifest.get("output_dialect") or {}


def _count_segment_support(
    cleaned_data_path: str,
    segment: Dict[str, Any],
    dialect: Dict[str, Any],
) -> Dict[str, Any]:
    if not cleaned_data_path or not os.path.exists(cleaned_data_path):
        return {}
    if not segment:
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}
    usecols = [col for col in segment.keys() if col]
    if not usecols:
        return {}
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    count = 0
    try:
        for chunk in pd.read_csv(
            cleaned_data_path,
            usecols=usecols,
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            chunksize=10000,
            dtype=str,
            low_memory=False,
        ):
            mask = None
            for col, val in segment.items():
                if col not in chunk.columns:
                    return {}
                col_series = chunk[col].astype(str)
                target = str(val)
                match = col_series.eq(target)
                mask = match if mask is None else (mask & match)
            if mask is not None:
                count += int(mask.sum())
    except Exception:
        return {}
    return {"n": count, "observed_support": count > 0}


def _apply_observational_policy(
    item: Dict[str, Any],
    counterfactual_policy: str,
) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    if counterfactual_policy != "observational_only":
        return item, notes
    support = item.get("support") or {}
    if isinstance(support, dict) and support.get("observed_support") is True:
        return item, notes
    if item.get("suggested_action"):
        item["suggested_action"] = {}
        notes.append("Removed suggested_action due to observational_only policy without verified support.")
    return item, notes


def _extract_items_from_optimization(payload: Any, source_path: str) -> List[Dict[str, Any]]:
    if not payload:
        return []
    candidates: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        candidates = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        for key in ["recommendations", "items", "results", "recommendation_items"]:
            if isinstance(payload.get(key), list):
                candidates = [item for item in payload.get(key) if isinstance(item, dict)]
                break
    return [_ensure_item_fields(item, source_path) for item in candidates]


def _build_items_from_scored_rows(path: str, max_examples: int) -> List[Dict[str, Any]]:
    columns, rows = _safe_load_rows(path, max_rows=max(200, max_examples * 40))
    if not rows or not columns:
        return []
    candidate_cols = [
        col for col in columns
        if any(tok in str(col).lower() for tok in ["score", "prob", "pred", "risk"])
    ]
    candidate_cols = candidate_cols or columns
    score_col = None
    scored_rows = []
    for col in candidate_cols:
        values = []
        for row in rows:
            val = row.get(col)
            if _is_number(val):
                values.append(float(str(val).replace(",", ".")))
        if values:
            score_col = col
            scored_rows = values
            break
    if not score_col:
        return []
    scored_pairs = []
    for row in rows:
        val = row.get(score_col)
        if not _is_number(val):
            continue
        score = float(str(val).replace(",", "."))
        scored_pairs.append((score, row))
    if not scored_pairs:
        return []
    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    items: List[Dict[str, Any]] = []
    for score, row in scored_pairs[:max_examples]:
        segment = {}
        segment_keys = [
            key for key in columns
            if any(tok in str(key).lower() for tok in ["segment", "cluster", "group", "tier", "band", "bucket", "category"])
        ]
        for key in segment_keys:
            val = row.get(key)
            if val not in (None, ""):
                segment[key] = val
        if not segment:
            for key in columns:
                if key == score_col:
                    continue
                val = row.get(key)
                if val in (None, "") or _is_number(val):
                    continue
                segment[key] = val
                if len(segment) >= 2:
                    break
        items.append(
            _ensure_item_fields(
                {
                    "segment": segment,
                    "score": score,
                    "metric": score_col,
                },
                "data/scored_rows.csv",
            )
        )
    return items


def _load_iteration_journal(run_dir: str) -> List[Dict[str, Any]]:
    journal_path = os.path.join(run_dir, "report", "governance", "ml_iteration_journal.jsonl")
    if not os.path.exists(journal_path):
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with open(journal_path, "r", encoding="utf-8") as f_journal:
            for line in f_journal:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    entries.append(payload)
    except Exception:
        return []
    return entries


def _entry_has_blocking_issue(entry: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    fields = []
    for key in ("preflight_issues", "reviewer_reasons", "qa_reasons", "failed_gates"):
        vals = entry.get(key)
        if isinstance(vals, list):
            fields.extend([str(v).lower() for v in vals if v])
    joined = " ".join(fields)
    if "synthetic" in joined:
        return True
    contract_tokens = [
        "unknown_columns",
        "unknown_columns_referenced",
        "df_column_assignment_forbidden",
        "output_contract_missing",
        "contract_missing_outputs",
        "contract_literal_guard",
        "must_reference_contract_columns",
    ]
    return any(tok in joined for tok in contract_tokens)


def _blocking_reasons_from_entry(entry: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if not isinstance(entry, dict):
        return reasons
    for key in ("preflight_issues", "reviewer_reasons", "qa_reasons", "failed_gates"):
        vals = entry.get(key)
        if isinstance(vals, list):
            reasons.extend([str(v) for v in vals if v])
    return list(dict.fromkeys(reasons))


def _hash_code_text(code: str) -> str:
    if not code:
        return ""
    return hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()[:12]


def _attempt_is_blocked(attempt_dir: str, journal_entries: List[Dict[str, Any]]) -> bool:
    if not journal_entries:
        return False
    code_path = os.path.join(attempt_dir, "code_sent.py")
    if not os.path.exists(code_path):
        return False
    try:
        with open(code_path, "r", encoding="utf-8") as f_code:
            code = f_code.read()
    except Exception:
        return False
    code_hash = _hash_code_text(code)
    if not code_hash:
        return False
    for entry in journal_entries:
        if entry.get("code_hash") == code_hash and _entry_has_blocking_issue(entry):
            return True
    return False


def _attempt_blocking_reasons(attempt_dir: str, journal_entries: List[Dict[str, Any]]) -> List[str]:
    if not journal_entries:
        return []
    code_path = os.path.join(attempt_dir, "code_sent.py")
    if not os.path.exists(code_path):
        return []
    try:
        with open(code_path, "r", encoding="utf-8") as f_code:
            code = f_code.read()
    except Exception:
        return []
    code_hash = _hash_code_text(code)
    if not code_hash:
        return []
    for entry in journal_entries:
        if entry.get("code_hash") == code_hash and _entry_has_blocking_issue(entry):
            return _blocking_reasons_from_entry(entry)
    return []


def _list_downloaded_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for base, _, names in os.walk(root_dir):
        for name in names:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root_dir)
            files.append(rel.replace("\\", "/"))
    return files


def _score_source_root(
    root_dir: str,
    deliverable_spec: Dict[str, Any],
    blocking_reasons: List[str] | None = None,
    file_set_override: set[str] | None = None,
) -> Dict[str, Any]:
    files = []
    if file_set_override is not None:
        file_set = set(file_set_override)
    else:
        files = _list_downloaded_files(root_dir) if root_dir and os.path.isdir(root_dir) else []
        file_set = set(files)
    required_paths = deliverable_spec.get("required_paths") or []
    optional_paths = deliverable_spec.get("optional_paths") or []
    requires_plots = bool(deliverable_spec.get("requires_plots"))

    required_total = len(required_paths)
    required_present, present_required = _count_present(required_paths, file_set)
    optional_present, present_optional = _count_present(optional_paths, file_set)
    plots_present = any(path.startswith("static/plots/") for path in file_set)

    output_report = _load_output_contract_report(root_dir)
    reasons = list(blocking_reasons or [])
    if required_total and not output_report:
        if file_set_override is None or "data/output_contract_report.json" not in file_set_override:
            reasons.append("output_contract_missing")
    missing_required = output_report.get("missing", []) if isinstance(output_report, dict) else []
    if missing_required:
        reasons.append("output_contract_missing")

    required_ratio = 1.0 if required_total == 0 else (required_present / max(required_total, 1))
    score = (10.0 * required_ratio) + (2.0 * optional_present)
    if requires_plots and plots_present:
        score += 1.0
    if reasons:
        score = -1e9

    return {
        "root": root_dir,
        "files": files,
        "integrity_pass": not reasons,
        "reasons": reasons,
        "score": score,
        "score_breakdown": {
            "required_present": required_present,
            "required_total": required_total,
            "optional_present": optional_present,
            "plots_present": plots_present,
            "required_ratio": required_ratio,
        },
        "present_required": present_required,
        "present_optional": present_optional,
    }


def _select_best_attempt(run_dir: str, deliverable_spec: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    sandbox_dir = os.path.join(run_dir, "sandbox", "ml_engineer")
    if not os.path.isdir(sandbox_dir):
        return None, []
    journal_entries = _load_iteration_journal(run_dir)
    attempt_dirs = []
    for name in os.listdir(sandbox_dir):
        if not name.startswith("attempt_"):
            continue
        attempt_path = os.path.join(sandbox_dir, name)
        if os.path.isdir(attempt_path):
            attempt_dirs.append((name, attempt_path))
    if not attempt_dirs:
        return None, []
    candidates: List[Dict[str, Any]] = []
    for name, attempt_path in attempt_dirs:
        downloaded_root = os.path.join(attempt_path, "downloaded_artifacts")
        if not os.path.isdir(downloaded_root):
            continue
        blocking = _attempt_blocking_reasons(attempt_path, journal_entries)
        scored = _score_source_root(downloaded_root, deliverable_spec, blocking)
        attempt_num = int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else 0
        scored.update({"attempt": attempt_num, "name": name})
        candidates.append(scored)
    if not candidates:
        return None, []
    candidates.sort(key=lambda item: (item["score"], item.get("attempt", 0)), reverse=True)
    best = candidates[0] if candidates[0].get("integrity_pass") else None
    return best, candidates


def build_recommendations_preview(
    contract: Dict[str, Any],
    governance_summary: Dict[str, Any] | None,
    artifacts_dir: str,
    cleaned_data_path: str | None = None,
    produced_artifact_index: Any | None = None,
    run_scoped_root: str | None = None,
) -> Dict[str, Any]:
    contract = contract or {}
    policy = _normalize_reporting_policy(contract)
    deliverable_spec = _normalize_deliverables(contract)
    artifacts_dir = artifacts_dir or ""
    artifacts_abs = os.path.abspath(artifacts_dir) if artifacts_dir else ""
    run_root_abs = os.path.abspath(run_scoped_root) if run_scoped_root else ""
    root_out_of_scope = False
    if run_root_abs and artifacts_abs:
        try:
            root_out_of_scope = os.path.commonpath([artifacts_abs, run_root_abs]) != run_root_abs
        except ValueError:
            root_out_of_scope = True
    governance_summary = governance_summary or {}
    run_outcome = governance_summary.get("run_outcome") or governance_summary.get("status") or "UNKNOWN"
    counterfactual_policy = contract.get("counterfactual_policy") or "unknown"
    recommendation_scope = contract.get("recommendation_scope") or ""
    max_examples = int(policy.get("max_examples") or 5)
    sources_checked: List[Dict[str, Any]] = []
    sources_used: List[str] = []
    caveats: List[str] = []
    items: List[Dict[str, Any]] = []
    reason = ""
    chosen_source: Dict[str, Any] | None = None

    def _label_source(prefix: str, rel_path: str) -> str:
        rel = rel_path.replace("\\", "/")
        return f"{prefix}{rel}" if prefix else rel

    def _scan_reports_dir(root_dir: str, prefix: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        checked: List[str] = []
        used: List[str] = []
        reports_dir = os.path.join(root_dir, "reports")
        if not os.path.isdir(reports_dir):
            return [], checked, used
        report_files = [name for name in os.listdir(reports_dir) if name.lower().endswith(".json")]
        report_files.sort()

        for name in report_files:
            rel_path = _label_source(prefix, f"reports/{name}")
            checked.append(rel_path)
            payload = _safe_load_json(os.path.join(reports_dir, name))
            items = _extract_items_from_payload(payload, rel_path)
            if not items:
                items = _extract_recommended_pairs(payload, rel_path)
            if not items:
                items = _extract_items_from_legacy_summary(payload, rel_path)
            if not items:
                items = _extract_items_from_optimization(payload, rel_path)
            if items:
                used.append(rel_path)
                return items, checked, used
        return [], checked, used

    def _collect_items_from_root(root_dir: str, prefix: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        checked: List[str] = []
        used: List[str] = []
        items, checked, used = _scan_reports_dir(root_dir, prefix)
        return items, checked, used

    candidates: List[Dict[str, Any]] = []
    artifact_blocking: List[str] = []
    if isinstance(governance_summary, dict):
        for key in ("failed_gates", "preflight_issues", "qa_reasons", "reviewer_reasons"):
            vals = governance_summary.get(key)
            if isinstance(vals, list):
                artifact_blocking.extend([str(v) for v in vals if v])
    produced_file_set = _normalize_produced_index(produced_artifact_index)
    if root_out_of_scope:
        artifacts_eval = {
            "root": artifacts_dir,
            "files": [],
            "integrity_pass": False,
            "reasons": ["root_out_of_scope"],
            "score": -1e9,
            "score_breakdown": {
                "required_present": 0,
                "required_total": len(deliverable_spec.get("required_paths") or []),
                "optional_present": 0,
                "plots_present": False,
                "required_ratio": 0.0,
            },
            "present_required": [],
            "present_optional": [],
        }
    else:
        artifacts_eval = _score_source_root(
            artifacts_dir,
            deliverable_spec,
            artifact_blocking,
            produced_file_set if produced_file_set else None,
        )
    sources_checked.append(
        {
            "attempt_root": artifacts_abs or artifacts_dir,
            "integrity_pass": artifacts_eval["integrity_pass"],
            "reasons": artifacts_eval["reasons"],
            "score_breakdown": artifacts_eval["score_breakdown"],
        }
    )
    if artifacts_eval["integrity_pass"]:
        candidates.append({"kind": "artifacts", **artifacts_eval})

    attempt_candidates: List[Dict[str, Any]] = []
    if governance_summary and governance_summary.get("run_id"):
        run_id = str(governance_summary.get("run_id"))
        run_dir = os.path.join("runs", run_id)
        best_attempt, attempt_candidates = _select_best_attempt(run_dir, deliverable_spec)
        for candidate in attempt_candidates:
            sources_checked.append(
                {
                    "attempt_root": os.path.abspath(candidate["root"]),
                    "integrity_pass": candidate["integrity_pass"],
                    "reasons": candidate["reasons"],
                    "score_breakdown": candidate["score_breakdown"],
                }
            )
        if best_attempt and best_attempt.get("integrity_pass"):
            best_attempt["run_id"] = run_id
            candidates.append({"kind": "sandbox_attempt", **best_attempt})

    if candidates:
        candidates.sort(key=lambda item: item.get("score", -1e9), reverse=True)
        best = candidates[0]
        if best.get("kind") == "sandbox_attempt":
            prefix = f"sandbox/ml_engineer/attempt_{best['attempt']}/downloaded_artifacts/"
            attempt_items, checked, used = _collect_items_from_root(best["root"], prefix)
            sources_used.extend(used)
            items = attempt_items
            chosen_source = {
                "kind": "sandbox_attempt",
                "run_id": best.get("run_id"),
                "attempt": best.get("attempt"),
                "root": os.path.abspath(best.get("root")),
                "score_breakdown": best.get("score_breakdown"),
            }
        else:
            items, checked, used = _collect_items_from_root(artifacts_dir, "")
            sources_used.extend(used)
            chosen_source = {
                "kind": "artifacts",
                "root": os.path.abspath(artifacts_dir),
                "score_breakdown": best.get("score_breakdown"),
            }

    items = [item for item in items if isinstance(item, dict)]
    if items:
        processed = []
        for item in items:
            item, notes = _apply_observational_policy(item, counterfactual_policy)
            if notes:
                caveat_list = item.get("caveats")
                if not isinstance(caveat_list, list):
                    caveat_list = []
                caveat_list.extend(notes)
                item["caveats"] = caveat_list
            processed.append(item)
        items = processed[:max_examples]

    dialect = _load_cleaning_dialect(artifacts_dir)
    if cleaned_data_path and items:
        for item in items:
            support = item.get("support") if isinstance(item.get("support"), dict) else {}
            if support and support.get("n") is not None:
                continue
            segment = item.get("segment") if isinstance(item.get("segment"), dict) else {}
            support_update = _count_segment_support(cleaned_data_path, segment, dialect)
            if support_update:
                merged = dict(support)
                merged.update(support_update)
                item["support"] = merged

    if not items:
        reason = "insufficient_artifacts"
        if not chosen_source and sources_checked:
            reasons = []
            for entry in sources_checked:
                for item in entry.get("reasons", []) if isinstance(entry, dict) else []:
                    reasons.append(str(item))
            if reasons:
                deduped = list(dict.fromkeys(reasons))[:6]
                reason = f"no_valid_sources: {', '.join(deduped)}"
    if run_outcome in {"NO_GO", "GO_WITH_LIMITATIONS"}:
        caveats.append("Illustrative examples only; not production-ready.")
        caveats.append("Do not claim causal impact from these examples.")
    if isinstance(chosen_source, dict) and chosen_source.get("kind") == "sandbox_attempt":
        caveats.append("Examples derived from sandbox attempt outputs; not promoted artifacts.")
    if counterfactual_policy == "observational_only":
        caveats.append("Observational-only policy: stay within observed support.")
    if not caveats:
        caveats.append("Use recommendations only within contract scope.")

    status = "standard"
    risk_level = "low"
    if run_outcome == "GO_WITH_LIMITATIONS":
        status = "illustrative_only"
        risk_level = "medium"
    if run_outcome == "NO_GO":
        status = "illustrative_only"
        risk_level = "high"

    return {
        "schema_version": "RecommendationPreview.v1",
        "generated_at": datetime.utcnow().isoformat(),
        "run_outcome": run_outcome,
        "counterfactual_policy": counterfactual_policy,
        "recommendation_scope": recommendation_scope,
        "status": status,
        "risk_level": risk_level,
        "illustrative_only": True,
        "reason": reason,
        "caveats": caveats,
        "policy_used": {
            "demonstrative_examples_enabled": policy.get("demonstrative_examples_enabled"),
            "demonstrative_examples_when_outcome_in": policy.get("demonstrative_examples_when_outcome_in"),
            "max_examples": max_examples,
            "require_strong_disclaimer": policy.get("require_strong_disclaimer"),
            "counterfactual_policy": counterfactual_policy,
            "recommendation_scope": recommendation_scope,
        },
        "sources_checked": sources_checked,
        "sources_used": sources_used,
        "chosen_source": chosen_source,
        "items": items,
    }
