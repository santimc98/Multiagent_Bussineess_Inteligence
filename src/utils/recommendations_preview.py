import csv
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


def _extract_segment_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    segment = {}
    if isinstance(item.get("segment"), dict):
        segment.update(item.get("segment") or {})
    segment_keys = [
        k for k in item.keys()
        if any(tok in str(k).lower() for tok in ["segment", "cluster", "group", "tier", "band", "bucket", "category"])
    ]
    for key in segment_keys:
        val = item.get(key)
        if val not in (None, ""):
            segment[key] = val
    return segment


def _extract_actions_from_item(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    current_action: Dict[str, Any] = {}
    suggested_action: Dict[str, Any] = {}
    for key, value in item.items():
        if value in (None, ""):
            continue
        key_lower = str(key).lower()
        if key_lower.startswith(("current_", "existing_", "baseline_", "prev_")):
            current_action[key] = value
        elif key_lower.startswith(("suggested_", "recommended_", "proposed_", "new_")):
            suggested_action[key] = value
    if isinstance(item.get("current_action"), dict):
        current_action.update(item.get("current_action") or {})
    if isinstance(item.get("suggested_action"), dict):
        suggested_action.update(item.get("suggested_action") or {})
    return current_action, suggested_action


def _extract_expected_effect(item: Dict[str, Any]) -> Dict[str, Any]:
    expected_effect: Dict[str, Any] = {}
    if isinstance(item.get("expected_effect"), dict):
        expected_effect.update(item.get("expected_effect") or {})
    for key, value in item.items():
        if value in (None, ""):
            continue
        key_lower = str(key).lower()
        if any(tok in key_lower for tok in ["delta", "uplift", "impact", "lift", "effect"]):
            expected_effect[key] = value
    if "metric" not in expected_effect:
        metric = item.get("metric") or item.get("metric_name")
        if metric:
            expected_effect["metric"] = metric
    if "notes" not in expected_effect:
        notes = item.get("notes") or item.get("reason")
        if notes:
            expected_effect["notes"] = notes
    return expected_effect


def _extract_support(item: Dict[str, Any]) -> Dict[str, Any]:
    support: Dict[str, Any] = {}
    if isinstance(item.get("support"), dict):
        support.update(item.get("support") or {})
    for key in ["n", "count", "support_n", "sample_size"]:
        if key in item and item.get(key) not in (None, ""):
            support.setdefault("n", item.get(key))
    for key in ["observed_support", "in_support", "within_support"]:
        if key in item:
            support["observed_support"] = bool(item.get(key))
    return support


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


def _extract_items_from_optimization(payload: Any) -> List[Dict[str, Any]]:
    if not payload:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ["recommendations", "items", "results", "recommendation_items"]:
            if isinstance(payload.get(key), list):
                return [item for item in payload.get(key) if isinstance(item, dict)]
    return []


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
        current_action, suggested_action = _extract_actions_from_item(row)
        expected_effect = {"metric": score_col, "value": score, "notes": "Ranked by observed score."}
        items.append(
            {
                "segment": segment,
                "current_action": current_action,
                "suggested_action": suggested_action,
                "expected_effect": expected_effect,
                "support": {},
            }
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


def _list_downloaded_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for base, _, names in os.walk(root_dir):
        for name in names:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root_dir)
            files.append(rel.replace("\\", "/"))
    return files


def _select_best_attempt(run_dir: str) -> Dict[str, Any] | None:
    sandbox_dir = os.path.join(run_dir, "sandbox", "ml_engineer")
    if not os.path.isdir(sandbox_dir):
        return None
    journal_entries = _load_iteration_journal(run_dir)
    attempt_dirs = []
    for name in os.listdir(sandbox_dir):
        if not name.startswith("attempt_"):
            continue
        attempt_path = os.path.join(sandbox_dir, name)
        if os.path.isdir(attempt_path):
            attempt_dirs.append((name, attempt_path))
    if not attempt_dirs:
        return None
    candidates: List[Dict[str, Any]] = []
    for name, attempt_path in attempt_dirs:
        if _attempt_is_blocked(attempt_path, journal_entries):
            continue
        downloaded_root = os.path.join(attempt_path, "downloaded_artifacts")
        if not os.path.isdir(downloaded_root):
            continue
        files = _list_downloaded_files(downloaded_root)
        file_set = set(files)
        has_metrics = "data/metrics.json" in file_set
        has_preview_source = any(
            rel in file_set
            for rel in ("reports/optimization_results.json", "data/predictions.csv", "data/scored_rows.csv")
        )
        if not (has_metrics and has_preview_source):
            continue
        score = 0
        if "data/metrics.json" in file_set:
            score += 3
        if "reports/price_optimization_summary.json" in file_set:
            score += 3
        if "reports/optimization_results.json" in file_set:
            score += 2
        if "data/scored_rows.csv" in file_set or "data/predictions.csv" in file_set:
            score += 2
        if "data/alignment_check.json" in file_set:
            score += 1
        if any(path.startswith("static/plots/") for path in file_set):
            score += 1
        attempt_num = int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else 0
        candidates.append(
            {
                "attempt": attempt_num,
                "name": name,
                "root": downloaded_root,
                "score": score,
                "files": files,
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["score"], item["attempt"]), reverse=True)
    return candidates[0]


def build_recommendations_preview(
    contract: Dict[str, Any],
    governance_summary: Dict[str, Any] | None,
    artifacts_dir: str,
    cleaned_data_path: str | None = None,
) -> Dict[str, Any]:
    contract = contract or {}
    policy = _normalize_reporting_policy(contract)
    governance_summary = governance_summary or {}
    run_outcome = governance_summary.get("run_outcome") or governance_summary.get("status") or "UNKNOWN"
    counterfactual_policy = contract.get("counterfactual_policy") or "unknown"
    recommendation_scope = contract.get("recommendation_scope") or ""
    max_examples = int(policy.get("max_examples") or 5)
    sources_checked: List[str] = []
    sources_used: List[str] = []
    caveats: List[str] = []
    items: List[Dict[str, Any]] = []
    reason = ""
    chosen_source: Dict[str, Any] | None = None

    def _label_source(prefix: str, rel_path: str) -> str:
        rel = rel_path.replace("\\", "/")
        return f"{prefix}{rel}" if prefix else rel

    def _collect_items_from_root(root_dir: str, prefix: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        collected: List[Dict[str, Any]] = []
        checked: List[str] = []
        used: List[str] = []
        opt_path = os.path.join(root_dir, "reports", "optimization_results.json")
        if os.path.exists(opt_path):
            checked.append(_label_source(prefix, "reports/optimization_results.json"))
            payload = _safe_load_json(opt_path)
            for raw_item in _extract_items_from_optimization(payload):
                segment = _extract_segment_from_item(raw_item)
                current_action, suggested_action = _extract_actions_from_item(raw_item)
                expected_effect = _extract_expected_effect(raw_item)
                support = _extract_support(raw_item)
                item = {
                    "segment": segment,
                    "current_action": current_action,
                    "suggested_action": suggested_action,
                    "expected_effect": expected_effect,
                    "support": support,
                    "source": _label_source(prefix, "reports/optimization_results.json"),
                }
                item, notes = _apply_observational_policy(item, counterfactual_policy)
                if notes:
                    expected_effect.setdefault("notes", "; ".join(notes))
                collected.append(item)
            if collected:
                used.append(_label_source(prefix, "reports/optimization_results.json"))

        if not collected:
            for rel_path in ["data/predictions.csv", "data/scored_rows.csv"]:
                path = os.path.join(root_dir, rel_path)
                if not os.path.exists(path):
                    continue
                checked.append(_label_source(prefix, rel_path))
                extracted = _build_items_from_scored_rows(path, max_examples=max_examples)
                for item in extracted:
                    item["source"] = _label_source(prefix, rel_path)
                    item, notes = _apply_observational_policy(item, counterfactual_policy)
                    if notes:
                        item["expected_effect"].setdefault("notes", "; ".join(notes))
                    collected.append(item)
                if collected:
                    used.append(_label_source(prefix, rel_path))
                    break
        return collected, checked, used
    items, checked, used = _collect_items_from_root(artifacts_dir, "")
    sources_checked.extend(checked)
    sources_used.extend(used)
    if items:
        chosen_source = {
            "kind": "artifacts",
            "root": os.path.abspath(artifacts_dir),
        }

    if not items and governance_summary and governance_summary.get("run_id"):
        run_id = str(governance_summary.get("run_id"))
        run_dir = os.path.join("runs", run_id)
        attempt = _select_best_attempt(run_dir)
        if attempt:
            prefix = f"sandbox/ml_engineer/attempt_{attempt['attempt']}/downloaded_artifacts/"
            attempt_items, checked, used = _collect_items_from_root(attempt["root"], prefix)
            sources_checked.extend(checked)
            sources_used.extend(used)
            if attempt_items:
                items = attempt_items
            chosen_source = {
                "kind": "sandbox_attempt",
                "run_id": run_id,
                "attempt": attempt["attempt"],
                "root": os.path.abspath(attempt["root"]),
            }

    items = [item for item in items if isinstance(item, dict)]
    if items:
        items = items[:max_examples]

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
