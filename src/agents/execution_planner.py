import json
import os
import ast
from typing import Dict, Any, List
from string import Template
import re
import difflib

from dotenv import load_dotenv
from src.agents.prompts import SENIOR_PLANNER_PROMPT
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from src.utils.contract_validation import (
    ensure_role_runbooks,
    DEFAULT_DATA_ENGINEER_RUNBOOK,
    DEFAULT_ML_ENGINEER_RUNBOOK,
    validate_spec_extraction_structure,
)

load_dotenv()


def _create_v41_skeleton(
    strategy: Dict[str, Any],
    business_objective: str,
    column_inventory: List[str] | None = None,
    output_dialect: Dict[str, str] | None = None,
    reason: str = "LLM failure"
) -> Dict[str, Any]:
    """
    Returns a complete, safe V4.1 schema skeleton with all required fields.
    Used as fallback when LLM fails or for validation.
    """
    strategy_title = strategy.get("title", "Unknown") if isinstance(strategy, dict) else "Unknown"
    required_cols = strategy.get("required_columns", []) if isinstance(strategy, dict) else []
    available_cols = column_inventory or []
    
    # Minimal safe canonical_columns from strategy.required_columns filtered by inventory
    canonical_cols = []
    missing_cols = []
    fuzzy_matches = {}
    
    if required_cols and available_cols:
        inv_norm = {re.sub(r"[^0-9a-zA-Z]+", "", str(c).lower()): c for c in available_cols}
        for req in required_cols:
            req_norm = re.sub(r"[^0-9a-zA-Z]+", "", str(req).lower())
            if req_norm in inv_norm:
                canonical_cols.append(inv_norm[req_norm])
            else:
                # Column missing from inventory, try fuzzy matching
                missing_cols.append(str(req))
                # Use difflib to find close matches
                close_matches = difflib.get_close_matches(
                    str(req).lower(),
                    [str(c).lower() for c in available_cols],
                    n=3,
                    cutoff=0.6
                )
                if close_matches:
                    # Map back to original case
                    matched_originals = [
                        c for c in available_cols 
                        if str(c).lower() in close_matches
                    ]
                    fuzzy_matches[str(req)] = matched_originals[:3]
    
    n_rows = 0  # Unknown without data profile parsing
    
    return {
        "contract_version": 2,
        "strategy_title": strategy_title,
        "business_objective": business_objective or "",
        
        "missing_columns_handling": {
            "missing_from_inventory": missing_cols,
            "attempted_fuzzy_matches": fuzzy_matches,
            "resolution": "unknown" if not missing_cols else "require_verification",
            "impact": "none" if not missing_cols else f"{len(missing_cols)} required columns not found",
            "contract_updates": {
                "canonical_columns_update": "Fuzzy matches suggested" if fuzzy_matches else "",
                "artifact_schema_update": "",
                "derived_plan_update": "",
                "gates_update": ""
            }
        },
        
        "execution_constraints": {
            "inplace_column_creation_policy": "unknown_or_forbidden",
            "preferred_patterns": ["df = df.assign(...)", "derived_arrays_then_concat", "build_new_df_from_columns"],
            "rationale": "Fallback: prefer safe patterns when uncertain."
        },
        
        "objective_analysis": {
            "problem_type": "unknown",
            "decision_variable": None,
            "business_decision": "unknown",
            "success_criteria": "unknown",
            "complexity": "unknown"
        },
        
        "data_analysis": {
            "dataset_size": n_rows,
            "features_with_nulls": [],
            "type_distribution": {},
            "risk_features": [],
            "data_sufficiency": "unknown"
        },
        
        "column_roles": {
            "pre_decision": [],
            "decision": [],
            "outcome": [],
            "post_decision_audit_only": [],
            "unknown": available_cols
        },
        
        "preprocessing_requirements": {},
        
        "feature_engineering_plan": {
            "derived_columns": []
        },
        
        "validation_requirements": {
            "method": "cross_validation",
            "stratification": False,
            "min_samples_required": 100
        },
        
        "leakage_execution_plan": {
            "audit_features": [],
            "method": "correlation_with_target",
            "threshold": 0.9,
            "action_if_exceeds": "exclude_from_features"
        },
        
        "optimization_specification": None,
        "segmentation_constraints": None,
        
        "data_limited_mode": {
            "is_active": False,
            "activation_reasons": [],
            "fallback_methodology": "unknown",
            "minimum_outputs": ["data/metrics.json", "data/scored_rows.csv", "data/alignment_check.json"],
            "artifact_reductions_allowed": True
        },
        
        "allowed_feature_sets": {
            "segmentation_features": [],
            "model_features": [],
            "audit_only_features": [],
            "forbidden_for_modeling": [],
            "rationale": "Fallback: empty feature sets pending analysis."
        },
        
        "artifact_requirements": {
            "required_files": ["data/cleaned_data.csv", "data/metrics.json"],
            "file_schemas": {}
        },
        
        "qa_gates": [
            {"id": "mapping_summary", "required": True, "description": "Column mapping validated"},
            {"id": "consistency_checks", "required": True, "description": "Basic consistency verified"},
            {"id": "outputs_required", "required": True, "description": "Required outputs present"}
        ],
        
        "reviewer_gates": [
            {"id": "methodology_alignment", "required": True, "description": "Methodology aligns with objective"},
            {"id": "business_value", "required": True, "description": "Business value demonstrated"}
        ],
        
        "data_engineer_runbook": DEFAULT_DATA_ENGINEER_RUNBOOK,
        "ml_engineer_runbook": DEFAULT_ML_ENGINEER_RUNBOOK,
        
        "available_columns": available_cols,
        "canonical_columns": canonical_cols,
        "derived_columns": [],
        "required_outputs": ["data/cleaned_data.csv", "data/metrics.json", "data/alignment_check.json"],
        
        "iteration_policy": {
            "max_iterations": 3,
            "early_stop_on_success": True
        },
        
        "unknowns": [
            {
                "item": reason,
                "impact": "Using skeletal V4.1 fallback",
                "mitigation": "Manual review required",
                "requires_verification": True
            }
        ],
        
        "assumptions": [
            "Minimal safe defaults used due to planner unavailability"
        ],
        
        "notes_for_engineers": [
            "This is a fallback contract. Proceed conservatively.",
            "Verify column inventory and semantics manually if possible."
        ]
    }


def ensure_v41_schema(contract: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """
    Validates and fills missing V4.1 schema keys.
    Adds to 'unknowns' array when filling defaults.
    
    Args:
        contract: Contract dict from LLM
        strict: If True, raise error on missing keys (for tests)
    
    Returns:
        Contract with all V4.1 keys present
    """
    if not isinstance(contract, dict):
        return contract
    
    required_keys = [
        "contract_version", "strategy_title", "business_objective",
        "missing_columns_handling", "execution_constraints",
        "objective_analysis", "data_analysis", "column_roles",
        "preprocessing_requirements", "feature_engineering_plan",
        "validation_requirements", "leakage_execution_plan",
        "optimization_specification", "segmentation_constraints",
        "data_limited_mode", "allowed_feature_sets",
        "artifact_requirements", "qa_gates", "reviewer_gates",
        "data_engineer_runbook", "ml_engineer_runbook",
        "available_columns", "canonical_columns", "derived_columns",
        "required_outputs", "iteration_policy", "unknowns",
        "assumptions", "notes_for_engineers"
    ]
    
    repairs = []
    
    for key in required_keys:
        if key not in contract:
            if strict:
                raise ValueError(f"Missing required V4.1 key: {key}")
            
            # Fill with safe default
            if key == "contract_version":
                contract[key] = 2
            elif key in ("optimization_specification", "segmentation_constraints"):
                contract[key] = None
            elif key in ("unknowns", "assumptions", "notes_for_engineers", "available_columns", 
                         "canonical_columns", "derived_columns", "required_outputs"):
                contract[key] = []
            elif key in ("qa_gates", "reviewer_gates"):
                contract[key] = []
            elif key in ("strategy_title", "business_objective"):
                contract[key] = ""
            else:
                contract[key] = {}
            
            repairs.append(f"Added missing key: {key}")
    
    # Ensure unknowns is a list
    unknowns = contract.get("unknowns")
    if not isinstance(unknowns, list):
        unknowns = []
        contract["unknowns"] = unknowns
    
    # Add repair notes to unknowns
    for repair in repairs:
        unknowns.append({
            "item": repair,
            "impact": "Schema validation filled missing field",
            "mitigation": "Review LLM output quality",
            "requires_verification": False
        })
    
    return contract


def _normalize_column_identifier(value: Any) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "", str(value).lower())
    return cleaned


def _build_allowed_column_norms(column_sets: List[str] | None, *more_sets: List[str] | None) -> set[str]:
    norms: set[str] = set()
    for collection in (column_sets, *more_sets):
        if not isinstance(collection, list):
            continue
        for col in collection:
            normed = _normalize_column_identifier(col)
            if normed:
                norms.add(normed)
    return norms


def _filter_leakage_audit_features(
    spec: Dict[str, Any],
    canonical_columns: List[str] | None,
    column_inventory: List[str] | None,
) -> List[str]:
    policy = spec.get("leakage_policy")
    if not isinstance(policy, dict):
        return []
    features = policy.get("audit_features")
    if not isinstance(features, list):
        return []
    allowed_norms = _build_allowed_column_norms(canonical_columns, column_inventory)
    filtered_out: List[str] = []

    if not allowed_norms:
        filtered_out = [str(item) for item in features if item]
        policy["audit_features"] = []
    else:
        kept: List[str] = []
        for item in features:
            if not item:
                continue
            normed = _normalize_column_identifier(item)
            if normed not in allowed_norms:
                filtered_out.append(str(item))
                continue
            kept.append(item)
        policy["audit_features"] = kept

    if filtered_out:
        detail = spec.get("leakage_policy_detail")
        if not isinstance(detail, dict):
            detail = {}
            spec["leakage_policy_detail"] = detail
        detail.setdefault("filtered_audit_features", [])
        existing = detail["filtered_audit_features"]
        existing.extend(filtered_out)
    return filtered_out

def parse_derive_from_expression(expr: str) -> Dict[str, Any]:
    if not expr or not isinstance(expr, str):
        return {}
    text = expr.strip()
    if not text:
        return {}

    def _coerce_values(raw: str) -> List[str]:
        if not raw:
            return []
        cleaned = raw.strip()
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
            if isinstance(parsed, str):
                return [parsed]
            return [str(parsed)]
        except Exception:
            pass
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", "\""}:
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()
        if "," in cleaned:
            parts = [part.strip(" \"'") for part in cleaned.split(",") if part.strip()]
            return parts
        return [cleaned] if cleaned else []

    match = re.match(r"^\s*([A-Za-z0-9_][A-Za-z0-9_ %\.\-]*)\s*==\s*(.+?)\s*$", text)
    if match:
        column = match.group(1).strip()
        values = _coerce_values(match.group(2))
        return {"column": column, "positive_values": values}
    match = re.match(r"^\s*([A-Za-z0-9_][A-Za-z0-9_ %\.\-]*)\s+in\s+(.+?)\s*$", text, flags=re.IGNORECASE)
    if match:
        column = match.group(1).strip()
        values = _coerce_values(match.group(2))
        return {"column": column, "positive_values": values}
    token_match = re.search(r"[A-Za-z0-9_][A-Za-z0-9_ %\.\-]*", text)
    if token_match:
        column = token_match.group(0).strip()
        return {"column": column, "positive_values": []}
    return {}

def enforce_percentage_ranges(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    reqs = contract.get("data_requirements", []) or []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        role = (req.get("role") or "").lower()
        expected = req.get("expected_range")
        if role == "percentage" and not expected:
            req["expected_range"] = [0, 1]
    notes = contract.get("notes_for_engineers")
    if not isinstance(notes, list):
        notes = []
    note = "Percentages must be normalized to 0-1; if values look like 0-100 scale, divide by 100."
    if note not in notes:
        notes.append(note)
    contract["notes_for_engineers"] = notes
    contract["data_requirements"] = reqs
    return contract


def build_dataset_profile(data_summary: str, column_inventory: List[str] | None = None) -> Dict[str, Any]:
    profile: Dict[str, Any] = {"column_count": len(column_inventory or [])}
    summary = (data_summary or "").strip()
    if summary:
        profile["summary_excerpt"] = summary[:400]
    return profile


def build_execution_plan(objective_type: str, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    objective = (objective_type or "unknown").lower()
    gates = [
        {"id": "data_ok", "description": "Data availability and basic quality checks pass.", "required": True},
        {"id": "target_ok", "description": "Target is valid with sufficient variation.", "required": True},
        {"id": "leakage_ok", "description": "No post-outcome leakage in features.", "required": True},
        {"id": "runtime_ok", "description": "Pipeline executes without runtime failures.", "required": True},
        {"id": "eval_ok", "description": "Evaluation meets objective-specific thresholds.", "required": True},
    ]

    base_outputs = [
        {"artifact_type": "clean_dataset", "required": True, "description": "Cleaned dataset for downstream use."},
        {"artifact_type": "artifact_index", "required": True, "description": "Typed inventory of produced artifacts."},
        {"artifact_type": "insights", "required": True, "description": "Unified insights for downstream reporting."},
        {"artifact_type": "executive_summary", "required": True, "description": "Business-facing summary."},
    ]

    objective_outputs: Dict[str, List[Dict[str, Any]]] = {
        "classification": [
            {"artifact_type": "metrics", "required": True, "description": "Classification metrics."},
            {"artifact_type": "predictions", "required": True, "description": "Predicted labels/probabilities."},
            {"artifact_type": "confusion_matrix", "required": False, "description": "Error breakdown by class."},
        ],
        "regression": [
            {"artifact_type": "metrics", "required": True, "description": "Regression metrics."},
            {"artifact_type": "predictions", "required": True, "description": "Predicted numeric outputs."},
            {"artifact_type": "residuals", "required": False, "description": "Residual diagnostics."},
        ],
        "forecasting": [
            {"artifact_type": "metrics", "required": True, "description": "Forecasting metrics."},
            {"artifact_type": "forecast", "required": True, "description": "Forecast outputs."},
            {"artifact_type": "backtest", "required": False, "description": "Historical forecast evaluation."},
        ],
        "ranking": [
            {"artifact_type": "metrics", "required": True, "description": "Ranking metrics."},
            {"artifact_type": "ranking_scores", "required": True, "description": "Ranked scores output."},
            {"artifact_type": "ranking_report", "required": False, "description": "Ranking diagnostics."},
        ],
    }
    optional_common = [
        {"artifact_type": "feature_importances", "required": False, "description": "Explainability artifact."},
        {"artifact_type": "error_analysis", "required": False, "description": "Failure mode analysis."},
        {"artifact_type": "plots", "required": False, "description": "Diagnostic plots."},
    ]

    outputs = list(base_outputs)
    outputs.extend(objective_outputs.get(objective, [{"artifact_type": "metrics", "required": True, "description": "Evaluation metrics."}]))
    outputs.extend(optional_common)

    return {
        "schema_version": "1",
        "objective_type": objective,
        "dataset_profile": dataset_profile or {},
        "gates": gates,
        "outputs": outputs,
    }


class ExecutionPlannerAgent:
    """
    LLM-driven planner that emits an execution contract (JSON) to guide downstream agents.
    Falls back to heuristic contract if the model call fails.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            self.client = None
        else:
            genai.configure(api_key=self.api_key)
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            self.client = genai.GenerativeModel(
                model_name="gemini-3-flash-preview",
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
        self.model_name = "gemini-3-flash-preview"
        self.last_prompt = None
        self.last_response = None

    def generate_contract(
        self,
        strategy: Dict[str, Any],
        data_summary: str = "",
        business_objective: str = "",
        column_inventory: list[str] | None = None,
        output_dialect: Dict[str, str] | None = None,
        env_constraints: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        def _norm(name: str) -> str:
            return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

        def _canonicalize_name(name: str) -> str:
            return str(name)

        def _resolve_exact_header(name: str) -> str | None:
            if not name or not column_inventory:
                return None
            norm_name = _norm(name)
            if not norm_name:
                return None
            best_match = None
            best_score = 0.0
            for raw in column_inventory:
                if raw is None:
                    continue
                raw_str = str(raw)
                raw_norm = _norm(raw_str)
                if raw_norm == norm_name:
                    return raw_str
                score = difflib.SequenceMatcher(None, norm_name, raw_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_match = raw_str
            if best_score >= 0.9:
                return best_match
            return None

        def _contract_column_norms(contract_obj: Dict[str, Any] | None = None) -> set[str]:
            norms: set[str] = set()
            canonical = contract_obj.get("canonical_columns") if isinstance(contract_obj, dict) else []
            if isinstance(canonical, list):
                for col in canonical:
                    if col:
                        norm_col = _norm(col)
                        if norm_col:
                            norms.add(norm_col)
            for col in column_inventory or []:
                if not col:
                    continue
                norm_col = _norm(col)
                if norm_col:
                    norms.add(norm_col)
            return norms

        def _filter_columns_against_contract(columns: List[str] | None, contract_obj: Dict[str, Any] | None = None) -> List[str]:
            if not columns:
                return []
            allowed = _contract_column_norms(contract_obj)
            if not allowed:
                return [col for col in columns if col]
            filtered: List[str] = []
            for col in columns:
                if not col:
                    continue
                if _norm(col) in allowed:
                    filtered.append(col)
            return filtered

        def _normalize_artifact_schema_payload(raw: Any) -> Dict[str, Dict[str, Any]]:
            normalized: Dict[str, Dict[str, Any]] = {}
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if not key:
                        continue
                    normalized[str(key)] = dict(value) if isinstance(value, dict) else {}
                return normalized
            if isinstance(raw, list):
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    path = item.get("path") or item.get("artifact") or item.get("output")
                    if not path:
                        continue
                    normalized[str(path)] = item
            return normalized

        def _parse_summary_kinds(summary_text: str) -> Dict[str, str]:
            kind_map: Dict[str, str] = {}
            if not summary_text:
                return kind_map
            for raw_line in summary_text.splitlines():
                line = raw_line.strip()
                if not line.startswith("-"):
                    continue
                content = line.lstrip("-").strip()
                if ":" not in content:
                    continue
                label, cols = content.split(":", 1)
                label_lower = label.strip().lower()
                cols_list = [c.strip() for c in re.split(r"[;,]", cols) if c.strip()]
                kind = None
                if "date" in label_lower:
                    kind = "datetime"
                elif "numerical" in label_lower or "numeric" in label_lower:
                    kind = "numeric"
                elif "categor" in label_lower or "boolean" in label_lower or "identifier" in label_lower:
                    kind = "categorical"
                if kind:
                    for col in cols_list:
                        kind_map[_norm(col)] = kind
            return kind_map

        def _guess_kind_from_name(name: str) -> str | None:
            if not name:
                return None
            raw = str(name)
            norm_name = _norm(raw)
            if not norm_name:
                return None
            if any(tok in norm_name for tok in ["date", "time", "fecha", "day", "month", "year"]):
                return "datetime"
            if any(tok in norm_name for tok in ["salesrep", "owner", "channel", "sector", "category", "type", "status", "phase", "segment", "email", "phone", "country", "city", "region", "industry", "name"]):
                return "categorical"
            if any(tok in norm_name for tok in ["id", "uuid", "code", "ref"]):
                return "categorical"
            if "%" in raw or any(tok in norm_name for tok in ["pct", "percent", "ratio", "rate", "prob", "score", "amount", "price", "size", "debt", "count", "number", "num", "qty"]):
                return "numeric"
            return None

        def _extract_formula_tokens(formula: str) -> List[str]:
            if not formula:
                return []
            tokens = re.findall(r"[A-Za-z%_][A-Za-z0-9_%]*", formula)
            ignore = {
                "score_nuevo",
                "scorenuevo",
                "score",
                "w",
            }
            cleaned: List[str] = []
            for tok in tokens:
                tok_norm = _norm(tok)
                if not tok_norm:
                    continue
                if tok_norm in ignore:
                    continue
                if tok_norm.startswith("w") and tok_norm[1:].isdigit():
                    continue
                cleaned.append(tok)
            # preserve order, unique
            seen = set()
            unique = []
            for tok in cleaned:
                tok_norm = _norm(tok)
                if tok_norm in seen:
                    continue
                seen.add(tok_norm)
                unique.append(tok)
            return unique

        def _guess_expected_range(name: str) -> List[float] | None:
            if not name:
                return None
            lower = name.lower()
            if "%" in name or "norm" in lower or "score" in lower or "impact" in lower:
                return [0, 1]
            return None

        def _ensure_formula_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            spec = contract.get("spec_extraction")
            if not isinstance(spec, dict):
                return contract
            formula = spec.get("scoring_formula") or ""
            if not isinstance(formula, str) or not formula.strip():
                return contract
            reqs = contract.get("data_requirements", []) or []
            existing = {_norm(r.get("canonical_name") or r.get("name")) for r in reqs if isinstance(r, dict)}
            derived_cols = spec.get("derived_columns") or []
            derived_names = {
                _norm(dc.get("name"))
                for dc in derived_cols
                if isinstance(dc, dict) and dc.get("name")
            }
            tokens = _extract_formula_tokens(formula)
            for tok in tokens:
                tok_norm = _norm(tok)
                if not tok_norm or tok_norm in existing:
                    continue
                source = "derived" if tok_norm in derived_names else "input"
                raw_match = _resolve_exact_header(tok)
                canonical = raw_match or tok
                reqs.append(
                    {
                        "name": tok,
                        "role": "feature",
                        "expected_range": _guess_expected_range(tok),
                        "allowed_null_frac": None,
                        "source": source,
                        "expected_kind": "numeric",
                        "canonical_name": canonical,
                    }
                )
                existing.add(tok_norm)
            contract["data_requirements"] = reqs
            return contract

        def _apply_expected_kind(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            kind_map = _parse_summary_kinds(data_summary)
            reqs = contract.get("data_requirements", []) or []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                if req.get("expected_kind"):
                    continue
                name = req.get("name")
                norm_name = _norm(name) if name else ""
                if norm_name in kind_map:
                    req["expected_kind"] = kind_map[norm_name]
                    continue
                inferred = _guess_kind_from_name(name or "")
                if inferred:
                    req["expected_kind"] = inferred
                    continue
                role = (req.get("role") or "").lower()
                if role in {"percentage", "risk_score", "probability", "ratio"}:
                    req["expected_kind"] = "numeric"
                elif role == "categorical":
                    req["expected_kind"] = "categorical"
                elif role == "date":
                    req["expected_kind"] = "datetime"
                else:
                    req["expected_kind"] = "unknown"
            contract["data_requirements"] = reqs
            return contract

        def _apply_inventory_source(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            inv_norm = {_norm(c) for c in (column_inventory or []) if c is not None}
            reqs = contract.get("data_requirements", []) or []
            updated_reqs = []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("name")
                if not name:
                    continue
                norm_name = _norm(name)
                best_match = None
                best_score = 0.0
                if inv_norm and norm_name:
                    for candidate in inv_norm:
                        score = difflib.SequenceMatcher(None, norm_name, candidate).ratio()
                        if score > best_score:
                            best_score = score
                            best_match = candidate
                if not inv_norm:
                    req["source"] = req.get("source", "input") or "input"
                elif norm_name and (norm_name in inv_norm or best_score >= 0.9):
                    req["source"] = req.get("source", "input") or "input"
                else:
                    req["source"] = "derived"
                updated_reqs.append(req)
            contract["data_requirements"] = updated_reqs
            return contract

        def _ensure_strategy_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            required = strategy.get("required_columns", []) if isinstance(strategy, dict) else []
            if not required:
                return contract
            reqs = contract.get("data_requirements", []) or []
            existing = {_norm(r.get("canonical_name") or r.get("name")) for r in reqs if isinstance(r, dict)}
            for col in required:
                if not col:
                    continue
                norm = _norm(col)
                if not norm or norm in existing:
                    continue
                raw_match = _resolve_exact_header(col)
                canonical = raw_match or _canonicalize_name(col)
                reqs.append(
                    {
                        "name": col,
                        "role": "feature",
                        "expected_range": _guess_expected_range(col),
                        "allowed_null_frac": None,
                        "source": "input",
                        "expected_kind": None,
                        "canonical_name": canonical,
                    }
                )
                existing.add(norm)
            contract["data_requirements"] = reqs
            return contract

        def _attach_canonical_names(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            reqs = contract.get("data_requirements", []) or []
            canonical_cols: List[str] = []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("name")
                if not name:
                    continue
                source = (req.get("source") or "input").lower()
                canonical = req.get("canonical_name")
                if source == "derived":
                    if not canonical:
                        canonical = _canonicalize_name(name)
                        req["canonical_name"] = canonical
                    continue
                raw_match = _resolve_exact_header(canonical or name)
                if raw_match:
                    canonical = raw_match
                    req["canonical_name"] = canonical
                elif not canonical:
                    canonical = _canonicalize_name(name)
                    req["canonical_name"] = canonical
                if canonical:
                    canonical_cols.append(canonical)
            contract["data_requirements"] = reqs
            if canonical_cols:
                contract["canonical_columns"] = canonical_cols
                notes = contract.get("notes_for_engineers")
                if not isinstance(notes, list):
                    notes = []
                note = "Use data_requirements.canonical_name for consistent column references across agents."
                if note not in notes:
                    notes.append(note)
                contract["notes_for_engineers"] = notes
            return contract

        def _propagate_business_alignment(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            ba = contract.get("business_alignment")
            if not isinstance(ba, dict):
                return contract
            runbooks = contract.get("role_runbooks")
            if isinstance(runbooks, dict):
                ml_runbook = runbooks.get("ml_engineer")
                if isinstance(ml_runbook, dict):
                    ml_runbook["business_alignment"] = ba
                    runbooks["ml_engineer"] = ml_runbook
                    contract["role_runbooks"] = runbooks
            return contract

        def _ensure_case_id_requirement(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") or {}
            case_taxonomy = spec.get("case_taxonomy")
            if not isinstance(case_taxonomy, list) or not case_taxonomy:
                return contract
            reqs = contract.get("data_requirements", []) or []
            case_names = {"case_id", "case", "caso", "caseid", "case_id"}
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = (req.get("name") or "").strip()
                if _norm(name) in case_names:
                    return contract
                canonical = (req.get("canonical_name") or "").strip()
                if _norm(canonical) in case_names:
                    return contract
            reqs.append(
                {
                    "name": "Case_ID",
                    "role": "case_id",
                    "expected_range": None,
                    "allowed_null_frac": 0.0,
                    "source": "derived",
                    "expected_kind": "categorical",
                    "canonical_name": "Case_ID",
                }
            )
            contract["data_requirements"] = reqs
            notes = contract.get("notes_for_engineers")
            if not isinstance(notes, list):
                notes = []
            note = "Case taxonomy present; include a case identifier column (e.g., Case_ID) in cleaned outputs."
            if note not in notes:
                notes.append(note)
            contract["notes_for_engineers"] = notes
            return contract

        def _attach_strategy_context(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            strategy_ctx = {
                "title": strategy.get("title") if isinstance(strategy, dict) else None,
                "analysis_type": strategy.get("analysis_type") if isinstance(strategy, dict) else None,
                "hypothesis": strategy.get("hypothesis") if isinstance(strategy, dict) else None,
                "techniques": strategy.get("techniques", []) if isinstance(strategy, dict) else [],
                "required_columns": strategy.get("required_columns", []) if isinstance(strategy, dict) else [],
                "estimated_difficulty": strategy.get("estimated_difficulty") if isinstance(strategy, dict) else None,
            }
            contract["strategy_context"] = strategy_ctx
            runbooks = contract.get("role_runbooks")
            if isinstance(runbooks, dict):
                ml_runbook = runbooks.get("ml_engineer")
                if isinstance(ml_runbook, dict):
                    ml_runbook["strategy_context"] = strategy_ctx
                    runbooks["ml_engineer"] = ml_runbook
                    contract["role_runbooks"] = runbooks
            return contract

        def _detect_canonical_collisions() -> List[tuple[str, List[str]]]:
            names = [c for c in (column_inventory or []) if c is not None]
            buckets: Dict[str, List[str]] = {}
            for name in names:
                canon = _canonicalize_name(name)
                if not canon:
                    continue
                buckets.setdefault(canon, []).append(str(name))
            return [(canon, vals) for canon, vals in buckets.items() if len(vals) > 1]

        def _has_numeric_conversion_risk(risk_items: List[str]) -> bool:
            marker = "Ensure numeric conversion before comparisons/normalization"
            return any(marker in risk for risk in risk_items)

        def _has_canonical_collision_risk(risk_items: List[str]) -> bool:
            marker = "Potential normalization collisions in column names"
            return any(marker in risk for risk in risk_items)

        def _extract_data_risks(contract: Dict[str, Any]) -> List[str]:
            risks: List[str] = []
            summary_text = data_summary or ""
            summary_lower = summary_text.lower()
            numeric_name_tokens = {
                "pct",
                "percent",
                "ratio",
                "rate",
                "prob",
                "probability",
                "score",
                "norm",
                "amount",
                "value",
                "price",
                "cost",
                "revenue",
                "income",
                "importe",
                "monto",
                "saldo",
                "age",
                "days",
                "term",
            }

            def _looks_numeric_name(col_name: str) -> bool:
                if "%" in col_name:
                    return True
                norm_name = _norm(col_name)
                return any(tok in norm_name for tok in numeric_name_tokens)

            def _parse_column_dtypes(text: str) -> List[tuple[str, str]]:
                cols: List[tuple[str, str]] = []
                in_section = False
                for raw_line in text.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    lower = line.lower()
                    if lower.startswith("key columns"):
                        in_section = True
                        continue
                    if in_section and lower.startswith("potential "):
                        break
                    if in_section and lower.startswith("example rows"):
                        break
                    if not in_section:
                        continue
                    if not line.startswith("-"):
                        continue
                    match = re.match(r"-\s*(.+?):\s*([^,]+),", line)
                    if match:
                        col = match.group(1).strip()
                        dtype = match.group(2).strip()
                        cols.append((col, dtype))
                return cols

            # Surface explicit alert/critical lines from steward summary
            for raw_line in summary_text.splitlines():
                line = raw_line.strip()
                lower = line.lower()
                if not line:
                    continue
                if "alert" in lower or "critical" in lower or "warning" in lower:
                    risks.append(line)
            # Sampling warning
            if "sample" in summary_lower and ("5000" in summary_lower or "sampled" in summary_lower):
                risks.append("Summary indicates sampling; verify results on full dataset.")
            # Dialect/encoding hints
            if "delimiter" in summary_lower or "dialect" in summary_lower or "encoding" in summary_lower:
                risks.append("Potential dialect/encoding sensitivity; enforce manifest output_dialect on load.")
            # Variance/constant hints
            if "no variation" in summary_lower or "no variance" in summary_lower or "constant" in summary_lower:
                risks.append("Potential low-variance/constant columns; guard for target/feature variance.")

            # Object dtypes on numeric-looking columns (type conversion risks)
            for col, dtype in _parse_column_dtypes(summary_text):
                dtype_lower = dtype.lower()
                if "object" in dtype_lower or "string" in dtype_lower:
                    if _looks_numeric_name(col):
                        risks.append(
                            f"Column '{col}' appears numeric/percentage but dtype is '{dtype}'. "
                            "Ensure numeric conversion before comparisons/normalization to avoid type errors."
                        )

            collisions = _detect_canonical_collisions()
            if collisions:
                examples = []
                for canon, originals in collisions[:3]:
                    sample = ", ".join(originals[:3])
                    examples.append(f"{canon}: {sample}")
                suffix = "; ".join(examples)
                risks.append(
                    "Potential normalization collisions in column names; ensure column selection is unambiguous "
                    f"after canonicalization (examples: {suffix})."
                )

            inv_norm = {_norm(c) for c in (column_inventory or []) if c is not None}
            missing_inputs: List[str] = []
            derived_needed: List[str] = []
            for req in contract.get("data_requirements", []) or []:
                if not isinstance(req, dict):
                    continue
                name = req.get("name")
                if not name:
                    continue
                source = req.get("source", "input") or "input"
                if source == "input" and inv_norm:
                    if _norm(name) not in inv_norm:
                        missing_inputs.append(name)
                if source == "derived":
                    derived_needed.append(name)
            if missing_inputs:
                risks.append(
                    f"Input requirements not found in header inventory: {missing_inputs}. "
                    "Use normalized mapping after canonicalization; do not fail before mapping."
                )
            if derived_needed:
                risks.append(
                    f"Derived columns required: {derived_needed}. "
                    "Derive after mapping; do not expect in raw input."
                )

            # Deduplicate and cap
            seen = set()
            unique = []
            for r in risks:
                if r not in seen:
                    seen.add(r)
                    unique.append(r)
            return unique[:8]

        def _attach_data_risks(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            risks = _extract_data_risks(contract)
            if risks:
                existing = contract.get("data_risks")
                if not isinstance(existing, list):
                    existing = []
                combined = existing + [r for r in risks if r not in existing]
                contract["data_risks"] = combined
                notes = contract.get("notes_for_engineers")
                if not isinstance(notes, list):
                    notes = []
                for r in risks:
                    note = f"DATA_RISK: {r}"
                    if note not in notes:
                        notes.append(note)
                contract["notes_for_engineers"] = notes
                if _has_numeric_conversion_risk(risks):
                    planner_self_check = contract.get("planner_self_check")
                    if not isinstance(planner_self_check, list):
                        planner_self_check = []
                    msg = (
                        "Flagged numeric-looking object columns; warn DE to convert to numeric before "
                        "comparisons/normalization."
                    )
                    if msg not in planner_self_check:
                        planner_self_check.append(msg)
                    contract["planner_self_check"] = planner_self_check
                if _has_canonical_collision_risk(risks):
                    planner_self_check = contract.get("planner_self_check")
                    if not isinstance(planner_self_check, list):
                        planner_self_check = []
                    msg = (
                        "Detected potential column-name collisions after canonicalization; ensure unambiguous "
                        "selection before validation."
                    )
                    if msg not in planner_self_check:
                        planner_self_check.append(msg)
                    contract["planner_self_check"] = planner_self_check
            return contract

        def _infer_objective_type() -> str:
            objective_text = (business_objective or "").lower()
            strategy_obj = strategy if isinstance(strategy, dict) else {}
            analysis_type = str(strategy_obj.get("analysis_type") or "").lower()
            techniques_text = " ".join([str(t) for t in (strategy_obj.get("techniques") or [])]).lower()
            signal_text = " ".join([objective_text, analysis_type, techniques_text])
            prescriptive_tokens = [
                "optimiz",
                "maximize",
                "minimize",
                "pricing",
                "precio",
                "optimal",
                "optimo",
                "revenue",
                "expected value",
                "recommend",
                "allocation",
                "decision",
                "prescriptive",
                "ranking",
                "scoring",
            ]
            predictive_tokens = [
                "predict",
                "classification",
                "regression",
                "forecast",
                "probability",
                "probabilidad",
                "clasific",
                "conversion",
                "convert",
                "churn",
                "contract",
                "propensity",
                "predictive",
            ]
            causal_tokens = [
                "causal",
                "uplift",
                "impact",
                "intervention",
                "treatment",
            ]
            if any(tok in signal_text for tok in prescriptive_tokens):
                return "prescriptive"
            if any(tok in signal_text for tok in predictive_tokens):
                return "predictive"
            if any(tok in signal_text for tok in causal_tokens):
                return "causal"
            return "descriptive"

        def _safe_column_name(name: str) -> str:
            if not name:
                return ""
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()

        def _recommended_column_name(name: str) -> str:
            if not name:
                return ""
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")

        def _collect_text_blob() -> str:
            parts = []
            if business_objective:
                parts.append(business_objective)
            if data_summary:
                parts.append(data_summary)
            if isinstance(strategy, dict):
                parts.append(json.dumps(strategy, ensure_ascii=True))
            return "\n".join(parts)

        def _infer_segmentation_required() -> bool:
            text = _collect_text_blob().lower()
            tokens = ["segment", "segmentation", "cluster", "clustering", "typology", "tipolog"]
            return any(tok in text for tok in tokens)

        def _estimate_n_rows(summary_text: str) -> int | None:
            profile_path = os.path.join("data", "dataset_profile.json")
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, "r", encoding="utf-8") as f_profile:
                        profile = json.load(f_profile)
                    if isinstance(profile, dict):
                        rows = profile.get("rows") or profile.get("row_count") or profile.get("n_rows")
                        if isinstance(rows, int) and rows >= 0:
                            return rows
                except Exception:
                    pass
            if summary_text:
                match = re.search(r"\brows\s*[:=]\s*(\d+)", summary_text, flags=re.IGNORECASE)
                if match:
                    try:
                        return int(match.group(1))
                    except Exception:
                        return None
            return None

        def _compute_segmentation_constraints(n_rows: int | None) -> Dict[str, Any]:
            n_val = int(n_rows) if isinstance(n_rows, int) and n_rows >= 0 else 0
            max_segments = min(15, max(3, n_val // 20))
            min_segment_size = max(10, n_val // 100)
            if n_val > 0:
                max_segments = min(max_segments, max(2, n_val // max(min_segment_size, 1)))
            preferred_max = min(10, max_segments)
            preferred_min = 2 if preferred_max >= 2 else 1
            return {
                "n_rows_estimate": n_val if n_rows is not None else None,
                "max_segments": int(max_segments),
                "min_segment_size": int(min_segment_size),
                "preferred_k_range": [int(preferred_min), int(preferred_max)],
            }

        def _attach_segmentation_constraints(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            constraints = contract.get("segmentation_constraints")
            if not isinstance(constraints, dict):
                constraints = {}
            n_rows = _estimate_n_rows(data_summary)
            computed = _compute_segmentation_constraints(n_rows)
            for key, value in computed.items():
                if value is None:
                    continue
                constraints[key] = value
            contract["segmentation_constraints"] = constraints
            return contract

        def _extract_summary_candidates(summary_text: str) -> List[str]:
            if not summary_text:
                return []
            candidates: List[str] = []
            label_tokens = ["status", "phase", "stage", "outcome", "result"]
            for raw_line in summary_text.splitlines():
                line = raw_line.strip()
                line_lower = line.lower()
                if not any(tok in line_lower for tok in label_tokens):
                    continue
                if ":" not in line:
                    continue
                _, cols = line.split(":", 1)
                cols_list = [c.strip() for c in re.split(r"[;,]", cols) if c.strip()]
                candidates.extend(cols_list)
            return candidates

        def _status_candidate_score(name: str) -> int:
            if not name:
                return -1
            norm = _norm(name)
            if not norm:
                return -1
            if any(tok in norm for tok in ["prob", "score", "pred"]):
                return -1
            tokens = [
                "status",
                "phase",
                "stage",
                "outcome",
                "result",
                "success",
                "won",
                "win",
                "closed",
                "churn",
                "conversion",
                "convert",
                "approval",
                "approved",
            ]
            score = 0
            for idx, tok in enumerate(tokens):
                if tok in norm:
                    score += (len(tokens) - idx)
            return score

        def _find_status_candidate() -> str | None:
            candidates: List[str] = []
            for raw in column_inventory or []:
                if raw is None:
                    continue
                raw_str = str(raw)
                if _status_candidate_score(raw_str) > 0:
                    candidates.append(raw_str)
            for candidate in _extract_summary_candidates(data_summary):
                resolved = _resolve_exact_header(candidate) or candidate
                if _status_candidate_score(resolved) > 0:
                    candidates.append(resolved)
            best = None
            best_score = -1
            for cand in candidates:
                score = _status_candidate_score(cand)
                if score > best_score:
                    best_score = score
                    best = cand
            return best

        def _parse_label_list(text: str) -> List[str]:
            if not text:
                return []
            parts = [p.strip(" \"'") for p in re.split(r"[;,/|]", text) if p.strip()]
            return [p for p in parts if p]

        def _extract_positive_labels(summary_text: str) -> List[str]:
            if not summary_text:
                return []
            for pattern in [
                r"positive labels?\s*[:=]\s*([^\n]+)",
                r"positive_values?\s*[:=]\s*([^\n]+)",
                r"success labels?\s*[:=]\s*([^\n]+)",
            ]:
                match = re.search(pattern, summary_text, flags=re.IGNORECASE)
                if match:
                    labels = _parse_label_list(match.group(1))
                    if labels:
                        return labels
            return []

        def _normalize_quotes(text: str) -> str:
            if not text:
                return ""
            return (
                text.replace("\u201c", "\"")
                .replace("\u201d", "\"")
                .replace("\u2018", "'")
                .replace("\u2019", "'")
            )

        def _column_mentioned(text: str, column: str) -> bool:
            if not text or not column:
                return False
            return _norm(column) in _norm(text)

        def _column_name_pattern(column: str) -> str:
            if not column:
                return ""
            tokens = re.findall(r"[A-Za-z0-9]+", column)
            if not tokens:
                return re.escape(column)
            return r"[\\s_\\-\\.]*".join([re.escape(tok) for tok in tokens])

        def _extract_positive_labels_from_objective(
            objective_text: str,
            status_col: str | None,
        ) -> tuple[List[str], str | None]:
            if not objective_text or not status_col:
                return [], None
            normalized_text = _normalize_quotes(objective_text)
            resolved_col = _resolve_exact_header(status_col) or status_col
            column_referenced = _column_mentioned(normalized_text, resolved_col)
            col_pattern = _column_name_pattern(resolved_col)
            if not col_pattern:
                return [], None

            label_pattern = r"(?P<label>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_%\\.-]+)"
            contains_patterns = []
            equals_patterns = []
            if column_referenced:
                contains_patterns.extend(
                    [
                        rf"{col_pattern}[^\n\r]{{0,200}}?(contiene|contains)[^\n\r]{{0,80}}?{label_pattern}",
                        rf"(contiene|contains)[^\n\r]{{0,80}}?{label_pattern}[^\n\r]{{0,200}}?{col_pattern}",
                    ]
                )
                equals_patterns.extend(
                    [
                        rf"{col_pattern}[^\n\r]{{0,200}}?(==|=|equals|equal to|es|igual)[^\n\r]{{0,80}}?{label_pattern}",
                        rf"(==|=|equals|equal to|es|igual)[^\n\r]{{0,80}}?{label_pattern}[^\n\r]{{0,200}}?{col_pattern}",
                    ]
                )

            phase_tokens = r"(status|phase|stage|estado|fase)"
            contains_patterns.append(
                rf"{phase_tokens}[^\n\r]{{0,60}}?(contiene|contains)[^\n\r]{{0,80}}?{label_pattern}"
            )
            equals_patterns.append(
                rf"{phase_tokens}[^\n\r]{{0,60}}?(==|=|equals|equal to|es|igual)[^\n\r]{{0,80}}?{label_pattern}"
            )
            equals_patterns.append(
                rf"{label_pattern}(?:\s+|\s*[-_/:,]+\s*){phase_tokens}"
            )

            for pattern in contains_patterns:
                match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
                if match:
                    labels = _parse_label_list(match.group("label"))
                    if labels:
                        return labels, "contains"

            for pattern in equals_patterns:
                match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
                if match:
                    labels = _parse_label_list(match.group("label"))
                    if labels:
                        return labels, "equals"

            labels = _extract_positive_labels(normalized_text)
            if labels:
                return labels, None
            return [], None

        def _deliverable_id_from_path(path: str) -> str:
            base = os.path.basename(str(path)) or str(path)
            cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_").lower()
            return cleaned or "deliverable"

        def _infer_deliverable_kind(path: str) -> str:
            lower = str(path).lower()
            if "plots" in lower or lower.endswith((".png", ".jpg", ".jpeg", ".svg")):
                return "plot"
            if lower.endswith(".csv"):
                return "dataset"
            if lower.endswith(".json"):
                if "metrics" in lower:
                    return "metrics"
                if "weights" in lower:
                    return "weights"
                if "alignment" in lower or "report" in lower:
                    return "report"
                return "json"
            return "artifact"

        def _default_deliverable_description(path: str, kind: str) -> str:
            known = {
                "data/cleaned_data.csv": "Cleaned dataset used for downstream modeling.",
                "data/metrics.json": "Model metrics and validation summary.",
                "data/weights.json": "Feature weights or scoring coefficients.",
                "data/case_summary.csv": "Per-case scoring summary.",
                "data/case_alignment_report.json": "Case alignment QA metrics.",
                "data/scored_rows.csv": "Row-level scores and key features.",
                "data/alignment_check.json": "Alignment check results for contract requirements.",
                "static/plots/*.png": "Required diagnostic plots.",
                "reports/recommendations_preview.json": "Illustrative recommendation examples for the executive report.",
            }
            if path in known:
                return known[path]
            if kind == "plot":
                return "Diagnostic plots required by the contract."
            if kind == "metrics":
                return "Metrics artifact required by the contract."
            if kind == "weights":
                return "Weights or scoring artifact required by the contract."
            return "Requested deliverable."

        def _build_deliverable(
            path: str,
            required: bool = True,
            kind: str | None = None,
            description: str | None = None,
            deliverable_id: str | None = None,
        ) -> Dict[str, Any]:
            if not path:
                return {}
            kind_val = kind or _infer_deliverable_kind(path)
            desc_val = description or _default_deliverable_description(path, kind_val)
            deliverable_id = deliverable_id or _deliverable_id_from_path(path)
            return {
                "id": deliverable_id,
                "path": path,
                "required": bool(required),
                "kind": kind_val,
                "description": desc_val,
            }

        def _normalize_deliverables(
            raw: Any,
            default_required: bool = True,
            required_paths: set[str] | None = None,
        ) -> List[Dict[str, Any]]:
            if not raw or not isinstance(raw, list):
                return []
            required_paths = {str(p) for p in (required_paths or set()) if p}
            normalized: List[Dict[str, Any]] = []
            for item in raw:
                if isinstance(item, str):
                    path = item
                    required = path in required_paths if required_paths else default_required
                    deliverable = _build_deliverable(path, required=required)
                    if deliverable:
                        normalized.append(deliverable)
                    continue
                if not isinstance(item, dict):
                    continue
                path = item.get("path") or item.get("output") or item.get("artifact")
                if not path:
                    continue
                required = item.get("required")
                if required is None:
                    required = path in required_paths if required_paths else default_required
                deliverable = _build_deliverable(
                    path=path,
                    required=bool(required),
                    kind=item.get("kind"),
                    description=item.get("description"),
                    deliverable_id=item.get("id"),
                )
                if deliverable:
                    normalized.append(deliverable)
            return normalized

        def _merge_deliverables(
            base: List[Dict[str, Any]],
            overrides: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            merged = list(base)
            by_path = {item.get("path"): idx for idx, item in enumerate(merged) if item.get("path")}
            for item in overrides:
                path = item.get("path")
                if not path:
                    continue
                if path in by_path:
                    existing = merged[by_path[path]]
                    for key in ("id", "kind", "description"):
                        if item.get(key):
                            existing[key] = item.get(key)
                    if "required" in item and item.get("required") is not None:
                        existing["required"] = bool(item.get("required"))
                    merged[by_path[path]] = existing
                else:
                    merged.append(item)
                    by_path[path] = len(merged) - 1
            return merged

        def _ensure_unique_deliverable_ids(deliverables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen: set[str] = set()
            for item in deliverables:
                base_id = item.get("id") or _deliverable_id_from_path(item.get("path") or "")
                candidate = base_id
                suffix = 2
                while candidate in seen:
                    candidate = f"{base_id}_{suffix}"
                    suffix += 1
                item["id"] = candidate
                seen.add(candidate)
            return deliverables

        def _derive_deliverables(
            objective_type: str,
            strategy_obj: Dict[str, Any],
            spec_obj: Dict[str, Any],
        ) -> List[Dict[str, Any]]:
            deliverables: List[Dict[str, Any]] = []

            def _add(path: str, required: bool = True, kind: str | None = None, description: str | None = None) -> None:
                item = _build_deliverable(path, required=required, kind=kind, description=description)
                if item:
                    deliverables.append(item)

            _add("data/cleaned_data.csv", True, "dataset", "Cleaned dataset used for downstream modeling.")
            _add("data/metrics.json", True, "metrics", "Model metrics and validation summary.")
            _add("static/plots/*.png", False, "plot", "Optional diagnostic plots.")
            _add("data/predictions.csv", False, "predictions", "Optional predictions output.")
            _add("data/feature_importances.json", False, "feature_importances", "Optional feature importance output.")
            _add("data/error_analysis.json", False, "error_analysis", "Optional error analysis output.")
            _add(
                "reports/recommendations_preview.json",
                False,
                "report",
                "Optional illustrative recommendation preview for executive reporting.",
            )

            target_type = str(spec_obj.get("target_type") or "").lower()
            scoring_formula = spec_obj.get("scoring_formula")
            analysis_type = str(strategy_obj.get("analysis_type") or "").lower()
            techniques_text = " ".join([str(t) for t in (strategy_obj.get("techniques") or [])]).lower()
            signal_text = " ".join([analysis_type, techniques_text, target_type, str(scoring_formula or "").lower()])
            if any(tok in signal_text for tok in ["ranking", "scoring", "weight", "weights", "optimization", "optimiz", "priorit"]):
                _add("data/weights.json", False, "weights", "Optional weights artifact for legacy consumers.")
                _add("data/case_summary.csv", False, "dataset", "Optional legacy case summary output.")
                _add("data/scored_rows.csv", False, "predictions", "Optional legacy scored rows output.")
                _add("data/case_alignment_report.json", False, "report", "Optional legacy alignment report.")
            return deliverables

        def _apply_deliverables(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            spec = contract.get("spec_extraction")
            if not isinstance(spec, dict):
                spec = {}
            legacy_required = contract.get("required_outputs", []) or []
            derived = _derive_deliverables(_infer_objective_type(), strategy or {}, spec)
            legacy = _normalize_deliverables(legacy_required, default_required=True)
            existing = _normalize_deliverables(spec.get("deliverables"), default_required=True, required_paths=set(legacy_required))
            deliverables = _merge_deliverables(derived, legacy)
            deliverables = _merge_deliverables(deliverables, existing)
            deliverables = _ensure_unique_deliverable_ids(deliverables)
            spec["deliverables"] = deliverables
            contract["spec_extraction"] = spec
            contract["required_outputs"] = [item["path"] for item in deliverables if item.get("required")]
            return contract

        def _merge_unique(values: List[str], extras: List[str]) -> List[str]:
            seen: set[str] = set()
            out: List[str] = []
            for item in values + extras:
                if not item:
                    continue
                text = str(item)
                if text in seen:
                    continue
                seen.add(text)
                out.append(text)
            return out

        def _has_deliverable(contract: Dict[str, Any], path: str) -> bool:
            if not isinstance(contract, dict):
                return False
            if path in (contract.get("required_outputs") or []):
                return True
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            deliverables = spec.get("deliverables")
            if isinstance(deliverables, list):
                for item in deliverables:
                    if isinstance(item, dict) and item.get("path") == path:
                        return True
                    if isinstance(item, str) and item == path:
                        return True
            return False

        def _build_scored_rows_schema(contract: Dict[str, Any]) -> Dict[str, Any] | None:
            if not _has_deliverable(contract, "data/scored_rows.csv"):
                return None
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            derived_cols: List[str] = []
            required_cols: List[str] = []
            derived = spec.get("derived_columns")
            if isinstance(derived, list):
                for entry in derived:
                    if isinstance(entry, dict):
                        name = entry.get("name") or entry.get("canonical_name")
                    elif isinstance(entry, str):
                        name = entry
                    else:
                        name = None
                    if name:
                        derived_cols.append(str(name))

            canonical_cols = contract.get("canonical_columns") if isinstance(contract.get("canonical_columns"), list) else []
            required_cols.extend([str(c) for c in canonical_cols if c])

            target_name = _target_name_from_contract(contract, derived if isinstance(derived, list) else [])
            if target_name:
                required_cols.append(str(target_name))
                safe_target = _safe_column_name(target_name)
                if safe_target:
                    required_cols.append(f"pred_{safe_target}")
                else:
                    required_cols.append("prediction")
                if _norm(target_name) in {"issuccess", "success"}:
                    required_cols.append("pred_prob_success")
            else:
                required_cols.append("prediction")

            if _infer_segmentation_required():
                required_cols.append("cluster_id")

            decision_vars = contract.get("decision_variables") or []
            if isinstance(decision_vars, list):
                for var in decision_vars:
                    if not var:
                        continue
                    safe_var = _recommended_column_name(var)
                    if safe_var:
                        required_cols.append(f"recommended_{safe_var}")
                if decision_vars:
                    required_cols.append("expected_value_at_recommendation")

            objective_type = _infer_objective_type()
            allowed_patterns = [
                r".*_probability$",
                r".*_score$",
                r".*_rank$",
                r".*_segment$",
                r"^segment_.*",
                r"^typology_.*",
                r".*_cluster$",
                r".*_group$",
            ]
            if objective_type in {"predictive", "prescriptive", "forecasting", "ranking"}:
                allowed_patterns.append(r"^pred(icted)?_.*")

            decision_vars = contract.get("decision_variables") or []
            decision_context = bool(decision_vars)
            if not decision_context:
                combined_text = "\n".join(
                    [
                        contract.get("business_objective") or business_objective or "",
                        data_summary or "",
                        json.dumps(strategy, ensure_ascii=True) if isinstance(strategy, dict) else "",
                    ]
                )
                decision_context = _detect_decision_context(combined_text)

            if decision_context:
                allowed_patterns.extend(
                    [
                        r"^(expected|optimal|recommended)_.*(revenue|value|price|profit|margin|cost).*",
                        r"^recommended_.*",
                        r"^expected_value_at_recommendation$",
                    ]
                )
            extra_patterns = ["^recommended_.*", "^expected_.*"]
            for pattern in extra_patterns:
                if pattern not in allowed_patterns:
                    allowed_patterns.append(pattern)

            return {
                "rowcount": "match_cleaned",
                "min_overlap": 1,
                "required_columns": _merge_unique(required_cols, []),
                "allowed_extra_columns": derived_cols,
                "allowed_name_patterns": allowed_patterns,
            }

        def _pattern_name(name: str) -> str:
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).lower()).strip("_")

        def _sanitize_allowed_extras(
            extras: List[str],
            patterns: List[str],
            allowlist: set[str],
            max_count: int = 30,
        ) -> List[str]:
            if not extras:
                return []
            allowed = []
            seen = set()
            norm_allowlist = {_norm(item) for item in allowlist if item}
            pattern_list = [str(pat) for pat in patterns if isinstance(pat, str) and pat.strip()]
            for item in extras:
                if not item:
                    continue
                raw = str(item)
                norm = _norm(raw)
                if not norm or norm in seen:
                    continue
                if norm in norm_allowlist:
                    allowed.append(raw)
                    seen.add(norm)
                else:
                    target = _pattern_name(raw)
                    matched = False
                    for pattern in pattern_list:
                        try:
                            if re.search(pattern, target):
                                matched = True
                                break
                        except re.error:
                            continue
                    if matched:
                        allowed.append(raw)
                        seen.add(norm)
                if len(allowed) >= max_count:
                    break
            return allowed

        def _apply_artifact_schemas(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            schemas = _normalize_artifact_schema_payload(contract.get("artifact_schemas"))
            spec_schemas = _normalize_artifact_schema_payload(spec.get("artifact_schemas"))
            for path, payload in spec_schemas.items():
                if path not in schemas:
                    schemas[path] = payload
            scored_schema = _build_scored_rows_schema(contract)
            if scored_schema:
                existing = schemas.get("data/scored_rows.csv")
                if not isinstance(existing, dict):
                    existing = {}
                merged = dict(existing)
                for key in ("rowcount", "min_overlap"):
                    if key not in merged and key in scored_schema:
                        merged[key] = scored_schema[key]
                if scored_schema.get("required_columns") and not merged.get("required_columns"):
                    merged["required_columns"] = scored_schema.get("required_columns")
                merged["allowed_extra_columns"] = _merge_unique(
                    scored_schema.get("allowed_extra_columns", []),
                    merged.get("allowed_extra_columns", []) or [],
                )
                merged["allowed_name_patterns"] = _merge_unique(
                    scored_schema.get("allowed_name_patterns", []),
                    merged.get("allowed_name_patterns", []) or [],
                )
                extra_patterns = ["^recommended_.*", "^expected_.*"]
                for pattern in extra_patterns:
                    if pattern not in merged.get("allowed_name_patterns", []):
                        merged.setdefault("allowed_name_patterns", []).append(pattern)
                base_allowlist = {
                    "is_success",
                    "success_probability",
                    "client_segment",
                    "cluster_id",
                    "expected_value_at_recommendation",
                }
                base_allowlist.update({str(c) for c in (merged.get("allowed_extra_columns") or []) if c})
                merged["allowed_extra_columns"] = _sanitize_allowed_extras(
                    merged.get("allowed_extra_columns", []) or [],
                    merged.get("allowed_name_patterns", []) or [],
                    base_allowlist,
                    max_count=30,
                )
                if merged.get("allowed_name_patterns"):
                    merged["allowed_name_patterns"] = _merge_unique(
                        merged["allowed_name_patterns"],
                        [],
                    )
                schemas["data/scored_rows.csv"] = merged
            contract["artifact_schemas"] = schemas
            return contract

        def _ensure_spec_extraction(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            spec = contract.get("spec_extraction")
            if not isinstance(spec, dict):
                spec = {}
            spec["derived_columns"] = spec.get("derived_columns") if isinstance(spec.get("derived_columns"), list) else []
            spec["case_taxonomy"] = spec.get("case_taxonomy") if isinstance(spec.get("case_taxonomy"), list) else []
            spec["constraints"] = spec.get("constraints") if isinstance(spec.get("constraints"), list) else []
            spec["deliverables"] = spec.get("deliverables") if isinstance(spec.get("deliverables"), list) else []
            spec["scoring_formula"] = spec.get("scoring_formula") if isinstance(spec.get("scoring_formula"), str) else None
            spec["target_type"] = spec.get("target_type") if isinstance(spec.get("target_type"), str) else None
            spec["leakage_policy"] = spec.get("leakage_policy") if isinstance(spec.get("leakage_policy"), str) else None
            if not isinstance(spec.get("leakage_policy_detail"), dict):
                spec["leakage_policy_detail"] = {}
            contract["spec_extraction"] = spec
            planner_self_check = contract.get("planner_self_check")
            if not isinstance(planner_self_check, list):
                contract["planner_self_check"] = []
            return contract

        def _normalize_derived_columns(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
            derived = spec.get("derived_columns")
            normalized: List[Dict[str, Any]] = []
            if isinstance(derived, list):
                for entry in derived:
                    if isinstance(entry, dict):
                        normalized.append(entry)
                    elif isinstance(entry, str):
                        normalized.append({"name": entry})
            spec["derived_columns"] = normalized
            return normalized

        def _ensure_requirement(
            reqs: List[Dict[str, Any]],
            name: str,
            role: str,
            expected_kind: str | None = None,
            source: str = "derived",
            derived_owner: str = "ml_engineer",
        ) -> None:
            if not name:
                return
            norm_name = _norm(name)
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                existing = req.get("canonical_name") or req.get("name")
                if existing and _norm(existing) == norm_name:
                    if not req.get("source"):
                        req["source"] = source
                    if role and not req.get("role"):
                        req["role"] = role
                    if expected_kind and not req.get("expected_kind"):
                        req["expected_kind"] = expected_kind
                    if derived_owner and not req.get("derived_owner"):
                        req["derived_owner"] = derived_owner
                    return
            payload = {
                "name": name,
                "role": role,
                "expected_range": None,
                "allowed_null_frac": None,
                "source": source,
                "expected_kind": expected_kind or "unknown",
                "canonical_name": name,
                "derived_owner": derived_owner,
            }
            reqs.append(payload)

        def _column_candidate_from_value(value: Any) -> str | None:
            if isinstance(value, str):
                parsed = parse_derive_from_expression(value)
                candidate = parsed.get("column") if parsed else None
                return candidate or value
            if isinstance(value, dict):
                for key in ("column", "source_column", "base_column", "from_column"):
                    val = value.get(key)
                    if isinstance(val, str):
                        return val
                return None
            return None

        def _validate_column(column: str | None, allowed_norms: set[str]) -> str | None:
            if not column:
                return None
            resolved = _resolve_exact_header(column) or column
            normed = _norm(resolved)
            if not normed:
                return None
            if allowed_norms and normed not in allowed_norms:
                return None
            return resolved

        def _extract_explicit_is_success(derived_entries: List[Dict[str, Any]], allowed_norms: set[str]) -> tuple[str | None, Dict[str, Any] | None]:
            for entry in derived_entries:
                if not isinstance(entry, dict):
                    continue
                role = str(entry.get("role") or "").lower()
                if "target" not in role and "label" not in role:
                    continue
                name = entry.get("canonical_name") or entry.get("name")
                if not name or _norm(name) not in {"issuccess", "success"}:
                    continue
                candidates: List[str | None] = []
                for key in ("column", "derived_from", "source_column", "base_column", "from_column"):
                    candidates.append(_column_candidate_from_value(entry.get(key)))
                depends = entry.get("depends_on")
                if isinstance(depends, list):
                    for item in depends:
                        if isinstance(item, str):
                            candidates.append(item)
                for candidate in candidates:
                    resolved = _validate_column(candidate, allowed_norms)
                    if resolved:
                        return resolved, entry
            return None, None

        def _target_name_from_contract(contract: Dict[str, Any], derived: List[Dict[str, Any]]) -> str | None:
            reqs = contract.get("data_requirements", []) or []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                role = str(req.get("role") or "").lower()
                if "target_source" in role:
                    continue
                if "target" in role or "label" in role:
                    return req.get("canonical_name") or req.get("name")
            for entry in derived:
                role = str(entry.get("role") or "").lower()
                if "target" in role or "label" in role:
                    return entry.get("name") or entry.get("canonical_name")
            return None

        def _ensure_derived_columns(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            derived = _normalize_derived_columns(spec)
            derived_keys = {_norm(entry.get("name") or entry.get("canonical_name") or "") for entry in derived}
            reqs = contract.get("data_requirements", []) or []

            objective_type = _infer_objective_type()
            target_req = None
            target_req_name = None
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                role = str(req.get("role") or "").lower()
                if "target" in role or "label" in role:
                    target_req = req
                    target_req_name = req.get("canonical_name") or req.get("name")
                    break
            target_in_inventory = bool(_resolve_exact_header(target_req_name)) if target_req_name else False
            target_role = str(target_req.get("role") or "").lower() if target_req else ""
            target_source_like = any(tok in target_role for tok in ["target_source", "outcome", "status", "phase"])
            target_is_input = (
                target_req
                and str(target_req.get("source") or "input").lower() == "input"
                and target_in_inventory
                and not target_source_like
            )
            allowed_norms = _contract_column_norms(contract)
            explicit_status_column, _ = _extract_explicit_is_success(derived, allowed_norms)
            status_source_col = explicit_status_column
            target_explicit = bool(explicit_status_column) or target_is_input
            if objective_type in {"predictive", "prescriptive", "forecasting"} and not target_explicit:
                status_col = _find_status_candidate()
                status_source_col = status_col
                if status_col:
                    resolved_status = _resolve_exact_header(status_col) or status_col
                    allowed_norms = _contract_column_norms(contract)
                    status_norm = _norm(resolved_status) if resolved_status else ""
                    if not resolved_status or (allowed_norms and status_norm not in allowed_norms):
                        resolved_status = None
                    if resolved_status:
                        status_source_col = resolved_status
                        positive_labels, rule_hint = _extract_positive_labels_from_objective(business_objective, resolved_status)
                        if target_req and str(target_req.get("source") or "").lower() == "derived":
                            target_req["name"] = "is_success"
                            target_req["canonical_name"] = "is_success"
                            target_req["role"] = "target"
                            if not target_req.get("expected_kind"):
                                target_req["expected_kind"] = "categorical"
                            if not target_req.get("derived_owner"):
                                target_req["derived_owner"] = "ml_engineer"
                            target_req_name = "is_success"
                        rule = "1 if status in positive_labels else 0"
                        if rule_hint == "contains":
                            rule = "1 if status contains positive_labels else 0"
                        entry = {
                            "name": "is_success",
                            "canonical_name": "is_success",
                            "role": "target",
                            "dtype": "boolean",
                            "derived_from": resolved_status,
                            "column": resolved_status,
                            "rule": rule,
                            "positive_values": positive_labels,
                        }
                        derived = [
                            item
                            for item in derived
                            if _norm(item.get("name") or item.get("canonical_name") or "") != _norm("is_success")
                        ]
                        derived.append(entry)
                        derived_keys.add(_norm("is_success"))
                        _ensure_requirement(reqs, "is_success", "target", expected_kind="categorical")
                        if not positive_labels:
                            checklist = contract.get("compliance_checklist")
                            if not isinstance(checklist, list):
                                checklist = []
                            note = (
                                "Infer positive_labels for is_success from unique values of the status column "
                                f"('{resolved_status}') before training."
                            )
                            if note not in checklist:
                                checklist.append(note)
                            contract["compliance_checklist"] = checklist

            if _infer_segmentation_required() and _norm("cluster_id") not in derived_keys:
                derived.append(
                    {
                        "name": "cluster_id",
                        "canonical_name": "cluster_id",
                        "role": "segment",
                        "dtype": "integer",
                        "derived_from": "pre_decision_features",
                        "rule": "Cluster rows using only pre-decision features.",
                    }
                )
                derived_keys.add(_norm("cluster_id"))
                _ensure_requirement(reqs, "cluster_id", "segment", expected_kind="categorical")

            status_col = status_source_col or _find_status_candidate()
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                role = str(req.get("role") or "").lower()
                if "target" not in role and "label" not in role:
                    continue
                if str(req.get("source") or "input").lower() != "derived":
                    continue
                name = req.get("canonical_name") or req.get("name")
                if not name:
                    continue
                norm_name = _norm(name)
                if norm_name in derived_keys:
                    continue
                dtype = "boolean" if name.lower().startswith(("is_", "has_")) or "success" in name.lower() else "numeric"
                entry = {
                    "name": name,
                    "canonical_name": name,
                    "role": "target",
                    "dtype": dtype,
                    "derived_from": status_col or "source_status",
                    "rule": "Derive target per contract instructions.",
                }
                if status_col and name.lower() == "is_success":
                    entry["rule"] = "1 if status in positive_labels else 0"
                derived.append(entry)
                derived_keys.add(norm_name)

            decision_vars = contract.get("decision_variables") or []
            if isinstance(decision_vars, list):
                for var in decision_vars:
                    if not var:
                        continue
                    safe_var = _safe_column_name(var)
                    if not safe_var:
                        continue
                    rec_name = f"recommended_{safe_var}"
                    if _norm(rec_name) not in derived_keys:
                        derived.append(
                            {
                                "name": rec_name,
                                "canonical_name": rec_name,
                                "role": "recommendation",
                                "dtype": "float",
                                "derived_from": str(var),
                                "rule": "Optimize expected value over decision variable.",
                            }
                        )
                        derived_keys.add(_norm(rec_name))
                        _ensure_requirement(reqs, rec_name, "recommendation", expected_kind="numeric")
                if decision_vars:
                    ev_name = "expected_value_at_recommendation"
                    if _norm(ev_name) not in derived_keys:
                        derived.append(
                            {
                                "name": ev_name,
                                "canonical_name": ev_name,
                                "role": "expected_value",
                                "dtype": "float",
                                "derived_from": "recommended_decision",
                                "rule": "Expected value at the recommended decision.",
                            }
                        )
                        derived_keys.add(_norm(ev_name))
                        _ensure_requirement(reqs, ev_name, "expected_value", expected_kind="numeric")

            spec["derived_columns"] = derived
            contract["spec_extraction"] = spec
            contract["data_requirements"] = reqs
            return contract

        def _refresh_canonical_columns(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            reqs = contract.get("data_requirements", []) or []
            derived = []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                source = (req.get("source") or "input").lower()
                if source == "derived":
                    continue
                name = req.get("canonical_name") or req.get("name")
                if name:
                    derived.append(str(name))
            existing = contract.get("canonical_columns") if isinstance(contract.get("canonical_columns"), list) else []
            input_norms = {
                _norm(req.get("canonical_name") or req.get("name") or "")
                for req in reqs
                if isinstance(req, dict) and str(req.get("source") or "input").lower() == "input"
            }
            existing = [col for col in existing if _norm(col) in input_norms]
            combined = []
            seen = set()
            for col in list(existing) + derived:
                norm = _norm(col)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                combined.append(col)
            if combined:
                contract["canonical_columns"] = combined
            return contract

        def _infer_availability(name: str, role: str | None, decision_vars: List[str]) -> str:
            norm = _norm(name)
            role_lower = (role or "").lower()
            decision_norms = {_norm(v) for v in decision_vars if v}
            if norm in decision_norms or "decision" in role_lower:
                return "decision"
            if any(tok in norm for tok in ["pred", "prob", "score", "recommend", "optimal", "expectedvalue"]):
                return "post_decision-audit_only"
            if any(tok in norm for tok in ["post", "after"]):
                return "post_decision-audit_only"
            if "target" in role_lower or "label" in role_lower:
                return "outcome"
            if any(tok in norm for tok in ["status", "phase", "stage", "outcome", "result", "success", "churn", "conversion"]):
                return "outcome"
            if any(tok in norm for tok in ["segment", "cluster", "group"]):
                return "pre_decision"
            return "pre_decision"

        def _ensure_feature_availability(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            canonical = contract.get("canonical_columns") if isinstance(contract.get("canonical_columns"), list) else []
            reqs = contract.get("data_requirements", []) or []
            availability = contract.get("feature_availability")
            if not isinstance(availability, list):
                availability = []
            by_norm = { _norm(item.get("column")): item for item in availability if isinstance(item, dict) and item.get("column") }
            decision_vars = contract.get("decision_variables") or []
            role_map = {}
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("canonical_name") or req.get("name")
                if name:
                    role_map[_norm(name)] = req.get("role")
            for col in canonical:
                norm = _norm(col)
                if not norm:
                    continue
                entry = by_norm.get(norm)
                role = role_map.get(norm)
                availability_label = _infer_availability(col, role, decision_vars if isinstance(decision_vars, list) else [])
                if entry:
                    if not entry.get("availability"):
                        entry["availability"] = availability_label
                    if not entry.get("rationale"):
                        entry["rationale"] = "Availability inferred from contract context."
                else:
                    availability.append(
                        {
                            "column": col,
                            "availability": availability_label,
                            "rationale": "Availability inferred from contract context.",
                        }
                    )
            contract["feature_availability"] = availability
            if not contract.get("availability_summary"):
                counts = {"pre_decision": 0, "decision": 0, "outcome": 0, "post_decision-audit_only": 0}
                for item in availability:
                    if not isinstance(item, dict):
                        continue
                    label = item.get("availability")
                    if label in counts:
                        counts[label] += 1
                contract["availability_summary"] = (
                    "Auto-generated availability summary: "
                    f"pre_decision={counts['pre_decision']}, decision={counts['decision']}, "
                    f"outcome={counts['outcome']}, post_decision-audit_only={counts['post_decision-audit_only']}."
                )
            return contract

        def _attach_leakage_policy_detail(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else {}
            feature_availability = contract.get("feature_availability") or []
            audit_only: List[str] = []
            forbidden: List[str] = []
            for item in feature_availability:
                if not isinstance(item, dict):
                    continue
                col = item.get("column")
                availability = str(item.get("availability") or "").lower()
                if not col:
                    continue
                if "post" in availability:
                    audit_only.append(col)
                    forbidden.append(col)
                if availability == "outcome":
                    forbidden.append(col)
            if isinstance(strategy, dict):
                for key in ("leakage_risk", "leakage_features", "leakage_columns", "leakage_risk_columns"):
                    vals = strategy.get(key)
                    if isinstance(vals, list):
                        for col in vals:
                            if col:
                                audit_only.append(str(col))
                                forbidden.append(str(col))
            reqs = contract.get("data_requirements", []) or []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("canonical_name") or req.get("name")
                role = str(req.get("role") or "").lower()
                if name and ("target" in role or "label" in role):
                    forbidden.append(name)
            canonical = contract.get("canonical_columns") if isinstance(contract.get("canonical_columns"), list) else []
            for col in canonical:
                norm = _norm(col)
                if any(tok in norm for tok in ["prob", "score", "pred"]):
                    audit_only.append(col)
                    forbidden.append(col)
            detail = {
                "audit_only": list(dict.fromkeys([c for c in audit_only if c])),
                "forbidden_as_feature": list(dict.fromkeys([c for c in forbidden if c])),
            }
            if detail["audit_only"] or detail["forbidden_as_feature"]:
                spec["leakage_policy_detail"] = detail
                contract["spec_extraction"] = spec
            return contract

        def _complete_contract_inference(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            contract = _ensure_derived_columns(contract)
            contract = _refresh_canonical_columns(contract)
            contract = _ensure_feature_availability(contract)
            contract = _attach_leakage_policy_detail(contract)
            return contract

        def _attach_spec_extraction_issues(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            issues = validate_spec_extraction_structure(contract)
            if not issues:
                return contract
            contract["spec_extraction_issues"] = issues
            notes = contract.get("notes_for_engineers")
            if not isinstance(notes, list):
                notes = []
            for issue in issues:
                note = f"SPEC_EXTRACTION_ISSUE: {issue}"
                if note not in notes:
                    notes.append(note)
            contract["notes_for_engineers"] = notes
            return contract

        def _build_feature_semantics(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
            if not isinstance(contract, dict):
                return []
            reqs = contract.get("data_requirements", []) or []
            semantics: List[Dict[str, Any]] = []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                name = req.get("canonical_name") or req.get("name")
                if not name:
                    continue
                role = req.get("role")
                kind = req.get("expected_kind")
                norm_name = _norm(name)
                meaning = None
                expectation = None
                risk = None

                if any(tok in norm_name for tok in ["amount", "price", "value", "revenue", "importe", "monto"]):
                    meaning = "monetary value of the contract or deal size"
                    expectation = "larger values typically indicate higher revenue; treat as outcome or pricing target"
                elif any(tok in norm_name for tok in ["size", "turnover", "facturacion", "ingresos"]):
                    meaning = "client scale or capacity (turnover / size)"
                    expectation = "larger clients may support higher contract values; conversion impact may be non-linear"
                elif any(tok in norm_name for tok in ["debt", "debtor", "risk", "riim"]):
                    meaning = "risk or exposure proxy tied to client behavior"
                    expectation = "higher risk may reduce conversion or price tolerance"
                elif any(tok in norm_name for tok in ["sector", "industry"]):
                    meaning = "industry segment describing client context"
                    expectation = "segment effects are categorical; compare within sector"
                elif any(tok in norm_name for tok in ["phase", "status", "contract"]):
                    meaning = "deal outcome or stage indicator"
                    expectation = "use to derive success labels, not as a predictive feature for conversion"
                    risk = "post-outcome fields can leak target information"
                elif any(tok in norm_name for tok in ["probability", "score"]):
                    meaning = "prior probability or scoring output"
                    expectation = "validate whether it is input signal or model output before use"
                elif any(tok in norm_name for tok in ["date", "time", "day", "month", "year"]):
                    meaning = "temporal marker for event timing"
                    expectation = "use to derive cycle duration or ordering; avoid leaking post-outcome dates"
                    risk = "post-event dates can leak conversion outcome"
                elif any(tok in norm_name for tok in ["channel", "salesrep", "owner", "agent"]):
                    meaning = "operational or assignment attribute"
                    expectation = "high-cardinality categorical; may require grouping to avoid overfitting"
                elif any(tok in norm_name for tok in ["reason", "comment", "note", "desc"]):
                    meaning = "free-text context or explanation"
                    expectation = "high-cardinality text; use with caution or exclude in small samples"

                if meaning is None:
                    meaning = "feature or identifier relevant to the business context"

                semantics.append(
                    {
                        "column": name,
                        "role": role,
                        "expected_kind": kind,
                        "business_meaning": meaning,
                        "directional_expectation": expectation,
                        "risk_notes": risk,
                    }
                )
            return semantics

        def _build_business_sanity_checks(
            contract: Dict[str, Any],
            feature_semantics: List[Dict[str, Any]],
        ) -> List[str]:
            if not isinstance(contract, dict):
                return []
            checks: List[str] = []
            roles = [str(req.get("role") or "").lower() for req in contract.get("data_requirements", []) if isinstance(req, dict)]
            has_reg_target = any("target_regression" in role or role == "target" for role in roles)
            has_cls_target = any("target_classification" in role for role in roles)
            if has_reg_target and has_cls_target:
                checks.append(
                    "If a conversion model is trained, ensure contract-value fields are not used as predictors when they only exist after success."
                )
            for item in feature_semantics:
                if not isinstance(item, dict):
                    continue
                if item.get("risk_notes"):
                    checks.append(f"Review potential leakage for column '{item.get('column')}'.")
            checks.append("If predicted conversion rises with higher price in most segments, re-check leakage or label leakage.")
            checks.append("If recommendations exceed historical max by a large margin, treat as a hypothesis and justify with evidence.")
            checks.append("If a segment has very few samples, aggregate or downweight before recommending prices.")
            return checks

        def _attach_semantic_guidance(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            semantics = contract.get("feature_semantics")
            if not isinstance(semantics, list) or not semantics:
                semantics = _build_feature_semantics(contract)
                contract["feature_semantics"] = semantics
            sanity = contract.get("business_sanity_checks")
            if not isinstance(sanity, list) or not sanity:
                contract["business_sanity_checks"] = _build_business_sanity_checks(contract, semantics)
            return contract

        def _attach_probability_audit_note(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            notes = contract.get("notes_for_engineers")
            if not isinstance(notes, list):
                notes = []
            note = (
                "Probability columns (e.g., *prob*, *probability*, *score*) are post-decision audit only; "
                "do not use for segmentation or modeling."
            )
            if note not in notes:
                notes.append(note)
            contract["notes_for_engineers"] = notes
            return contract

        def _detect_decision_context(text: str) -> bool:
            if not text:
                return False
            norm_text = _norm(text)
            if not norm_text:
                return False
            tokens = [
                "price",
                "pricing",
                "precio",
                "tarifa",
                "quote",
                "offer",
                "cotizacion",
                "importe",
                "amount",
                "valor",
                "premium",
                "rate",
                "fee",
                "cost",
            ]
            token_norms = [_norm(tok) for tok in tokens if tok]
            return any(tok and tok in norm_text for tok in token_norms)

        def _is_price_like(name: str) -> bool:
            if not name:
                return False
            norm_name = _norm(name)
            tokens = [
                "price",
                "precio",
                "amount",
                "importe",
                "valor",
                "fee",
                "tarifa",
                "premium",
                "rate",
                "cost",
            ]
            token_norms = [_norm(tok) for tok in tokens if tok]
            return any(tok and tok in norm_name for tok in token_norms)

        def _extract_missing_sentinel(text: str) -> float | int | None:
            if not text:
                return None
            phrases = [
                "no offer",
                "not offered",
                "sin oferta",
                "sin propuesta",
                "no quote",
                "no proposal",
                "no bid",
                "sin precio",
            ]
            lower = text.lower()
            for sentence in re.split(r"[\n\.]", lower):
                if not any(p in sentence for p in phrases):
                    continue
                match = re.search(r"(?:=|->|:)\s*([0-9]+(?:[.,][0-9]+)?)", sentence)
                if not match:
                    match = re.search(r"\b([0-9]+(?:[.,][0-9]+)?)\b", sentence)
                if not match:
                    continue
                raw_val = match.group(1).replace(",", ".")
                try:
                    return float(raw_val) if "." in raw_val else int(raw_val)
                except ValueError:
                    continue
            return None

        def _assign_derived_owners(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction") or {}
            formulas = spec.get("formulas") if isinstance(spec, dict) else {}
            formula_keys = set()
            if isinstance(formulas, dict):
                formula_keys = {_norm(k) for k in formulas.keys()}
            derived_spec = spec.get("derived_columns") if isinstance(spec, dict) else []
            derived_keys = set()
            if isinstance(derived_spec, list):
                for entry in derived_spec:
                    if isinstance(entry, dict):
                        name = entry.get("name")
                        if name:
                            derived_keys.add(_norm(name))
            reqs = contract.get("data_requirements", []) or []
            for req in reqs:
                if not isinstance(req, dict):
                    continue
                if req.get("source", "input") != "derived":
                    continue
                if req.get("derived_owner"):
                    continue
                name = req.get("canonical_name") or req.get("name")
                norm_name = _norm(name) if name else ""
                role = (req.get("role") or "").lower()
                if norm_name in derived_keys or norm_name in formula_keys:
                    req["derived_owner"] = "data_engineer"
                elif "segment" in role or "group" in role or "cluster" in role:
                    req["derived_owner"] = "ml_engineer"
                else:
                    req["derived_owner"] = "ml_engineer"
            contract["data_requirements"] = reqs
            return contract

        def _attach_variable_semantics(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            combined_text = "\n".join(
                [
                    business_objective or "",
                    data_summary or "",
                    json.dumps(strategy, ensure_ascii=True) if isinstance(strategy, dict) else "",
                ]
            )
            decision_context = _detect_decision_context(combined_text)
            decision_vars: List[str] = []
            if decision_context:
                for req in contract.get("data_requirements", []) or []:
                    if not isinstance(req, dict):
                        continue
                    name = req.get("canonical_name") or req.get("name")
                    if not name:
                        continue
                    if _is_price_like(name) or (req.get("role") or "").lower() == "target_regression":
                        req["decision_variable"] = True
                        decision_vars.append(name)

            if decision_vars:
                unique_vars: List[str] = []
                seen = set()
                for item in decision_vars:
                    key = _norm(item)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    unique_vars.append(item)
                contract["decision_variables"] = unique_vars

                feature_availability = contract.get("feature_availability")
                if not isinstance(feature_availability, list):
                    feature_availability = []
                avail_map = {
                    _norm(item.get("column")): item
                    for item in feature_availability
                    if isinstance(item, dict) and item.get("column")
                }
                for col in unique_vars:
                    key = _norm(col)
                    entry = avail_map.get(key)
                    if not entry:
                        entry = {"column": col}
                        feature_availability.append(entry)
                    entry["availability"] = "decision"
                    entry.setdefault(
                        "rationale",
                        "Decision variable controlled by the business; usable for optimization or elasticity modeling.",
                    )
                contract["feature_availability"] = feature_availability

                summary = contract.get("availability_summary") or ""
                if "decision" not in summary.lower():
                    summary = (summary + " " if summary else "") + (
                        "Decision variables may be used in optimization/elasticity modeling when available."
                    )
                contract["availability_summary"] = summary

                sentinel_value = _extract_missing_sentinel(combined_text)
                if sentinel_value is not None:
                    missing_sentinels = []
                    for col in unique_vars:
                        missing_sentinels.append(
                            {
                                "column": col,
                                "sentinel": sentinel_value,
                                "meaning": "not_observed",
                                "action": "treat_as_missing",
                            }
                        )
                    contract["missing_sentinels"] = missing_sentinels
                    notes = contract.get("notes_for_engineers")
                    if not isinstance(notes, list):
                        notes = []
                    note = (
                        "Missing sentinel detected for decision variables; treat sentinel values as missing "
                        "when modeling elasticity and document any derived observed flags."
                    )
                    if note not in notes:
                        notes.append(note)
                    contract["notes_for_engineers"] = notes

            return contract

        def _ensure_availability_reasoning(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            if "feature_availability" not in contract:
                contract["feature_availability"] = []
            if "availability_summary" not in contract:
                contract["availability_summary"] = ""
            return contract

        def _attach_counterfactual_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            if contract.get("counterfactual_policy"):
                return contract
            decision_vars = contract.get("decision_variables") or []
            if not isinstance(decision_vars, list) or not decision_vars:
                return contract
            spec = contract.get("spec_extraction") or {}
            case_taxonomy = spec.get("case_taxonomy") if isinstance(spec.get("case_taxonomy"), list) else []
            availability_summary = contract.get("availability_summary") or ""
            feature_availability = contract.get("feature_availability") or []
            reqs = contract.get("data_requirements") or []
            combined_text = " ".join(
                [
                    availability_summary,
                    json.dumps(feature_availability, ensure_ascii=True) if isinstance(feature_availability, list) else "",
                    json.dumps(reqs, ensure_ascii=True) if isinstance(reqs, list) else "",
                    data_summary or "",
                ]
            ).lower()
            evidence_tokens = [
                "random", "randomized", "experiment", "a/b", "ab test", "treatment",
                "control", "holdout", "policy change", "uplift", "causal", "instrument",
            ]
            has_counterfactual = any(tok in combined_text for tok in evidence_tokens)
            if not case_taxonomy:
                has_counterfactual = False
            if not has_counterfactual:
                contract["counterfactual_policy"] = "observational_only"
                contract["recommendation_scope"] = "within_observed_support_only"
                contract["required_limitations_section"] = True
                contract["required_next_steps"] = True
            else:
                contract["counterfactual_policy"] = "counterfactual_supported"
                contract["recommendation_scope"] = "supported_by_experiment"
            return contract

        def _ensure_iteration_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            if not isinstance(contract.get("compliance_checklist"), list):
                contract["compliance_checklist"] = []
            if not isinstance(contract.get("iteration_policy"), dict):
                contract["iteration_policy"] = {}
            return contract

        def _attach_reporting_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            policy = contract.get("reporting_policy")
            if not isinstance(policy, dict):
                policy = {}
            if "demonstrative_examples_enabled" not in policy:
                policy["demonstrative_examples_enabled"] = True
            if "demonstrative_examples_when_outcome_in" not in policy:
                policy["demonstrative_examples_when_outcome_in"] = ["NO_GO", "GO_WITH_LIMITATIONS"]
            if "max_examples" not in policy:
                policy["max_examples"] = 5
            if "require_strong_disclaimer" not in policy:
                policy["require_strong_disclaimer"] = True
            contract["reporting_policy"] = policy
            return contract

        def _normalize_alignment_requirements(items: Any) -> List[Dict[str, Any]]:
            if not isinstance(items, list):
                return []
            normalized: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for idx, item in enumerate(items):
                if isinstance(item, str) and item.strip():
                    req_id = item.strip()
                    item = {"id": req_id, "requirement": req_id}
                if not isinstance(item, dict):
                    continue
                req_id = str(item.get("id") or item.get("name") or item.get("key") or f"custom_{idx}")
                if not req_id or req_id in seen:
                    continue
                seen.add(req_id)
                req = {
                    "id": req_id,
                    "requirement": str(item.get("requirement") or item.get("description") or "").strip(),
                    "rationale": str(item.get("rationale") or "").strip(),
                    "success_criteria": item.get("success_criteria") if isinstance(item.get("success_criteria"), list) else [],
                    "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
                    "applies_when": str(item.get("applies_when") or "always"),
                    "failure_mode_on_miss": str(item.get("failure_mode_on_miss") or "method_choice"),
                }
                normalized.append(req)
            return normalized

        def _detect_segment_context(text: str) -> bool:
            if not text:
                return False
            norm_text = _norm(text)
            tokens = [
                "segment",
                "segmentation",
                "segmented",
                "cluster",
                "cohort",
                "case",
                "bucket",
                "grupo",
                "segmento",
                "clase",
                "caso",
            ]
            token_norms = [_norm(tok) for tok in tokens if tok]
            return any(tok and tok in norm_text for tok in token_norms)

        def _build_alignment_requirements(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
            requirements: List[Dict[str, Any]] = []

            def _add(req_id: str, requirement: str, rationale: str, success: List[str], applies_when: str, failure_mode: str):
                requirements.append(
                    {
                        "id": req_id,
                        "requirement": requirement,
                        "rationale": rationale,
                        "success_criteria": success,
                        "evidence": [],
                        "applies_when": applies_when,
                        "failure_mode_on_miss": failure_mode,
                    }
                )

            _add(
                "objective_alignment",
                "Methodology directly answers the business objective and the strategy analysis_type.",
                "Prevents optimizing an easier proxy that misses the business goal.",
                [
                    "Approach matches analysis_type/techniques or a justified alternative is documented.",
                    "Outputs support the business decision stated in the objective.",
                ],
                "always",
                "method_choice",
            )

            decision_vars = contract.get("decision_variables") or []
            if isinstance(decision_vars, list) and decision_vars:
                preview = [str(v) for v in decision_vars[:5]]
                if len(decision_vars) > 5:
                    preview.append("...")
                _add(
                    "decision_variable_handling",
                    f"Decision variables are modeled as controllable inputs ({', '.join(preview)}).",
                    "Pricing/decision inputs must drive elasticity or optimization rather than be ignored.",
                    [
                        "Decision variables are used in modeling or optimization.",
                        "If excluded, explain why and quantify the impact on recommendations.",
                    ],
                    "decision_variables_present",
                    "method_choice",
                )

            segment_required = False

        def _fallback() -> Dict[str, Any]:
            return _create_v41_skeleton(
                strategy=strategy,
                business_objective=business_objective,
                column_inventory=column_inventory,
                output_dialect=output_dialect,
                reason="Planner LLM Failed"
            )

        if not self.client:
            return _fallback()

        strategy_json = json.dumps(strategy, indent=2)
        column_inventory_json = json.dumps(column_inventory or [])
        
        # Ensure data_summary is a string
        data_summary_str = ""
        if isinstance(data_summary, dict):
            data_summary_str = json.dumps(data_summary, indent=2)
        else:
            data_summary_str = str(data_summary)

        # Construct Input
        user_input = f"""
strategy:
{strategy_json}

business_objective:
{business_objective}

column_inventory:
{column_inventory_json}

data_profile_summary:
{data_summary_str}

output_dialect:
{json.dumps(output_dialect or "unknown")}

env_constraints:
{json.dumps(env_constraints or {"forbid_inplace_column_creation": True})}
"""
        full_prompt = SENIOR_PLANNER_PROMPT + "\n\nINPUTS:\n" + user_input
        self.last_prompt = full_prompt
        
        try:
            response = self.client.generate_content(full_prompt)
            content = response.text
            self.last_response = content
            # Clean markdown
            content = content.replace("```json", "").replace("```", "").strip()
            # Handle potential preamble text if model is chatty (though V4.1 says JSON only)
            if "{" in content:
                content = content[content.find("{"):content.rfind("}")+1]
                
            contract = json.loads(content)
            
            # Validate structure
            if not isinstance(contract, dict):
                # Fallback if output is not a dict
                return _fallback()
            
            # Ensure V4.1 schema completeness
            contract = ensure_v41_schema(contract)
            
            return contract

        except Exception:
            return _fallback()

    def generate_evaluation_spec(
        self,
        strategy: Dict[str, Any],
        contract: Dict[str, Any],
        data_summary: str = "",
        business_objective: str = "",
        column_inventory: list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate evaluation spec by extracting ONLY from V4.1 contract fields.
        NO legacy fields (data_requirements, spec_extraction) allowed.
        """
        if not isinstance(contract, dict):
            return {
                "confidence": 0.1,
                "qa_gates": [],
                "reviewer_gates": [],
                "artifact_requirements": {},
                "notes": ["Invalid contract structure"],
                "source": "error_fallback",
                "unknowns": ["Contract is not a valid dictionary"]
            }
        
        # Extract ONLY from V4.1 fields
        qa_gates = contract.get("qa_gates", [])
        reviewer_gates = contract.get("reviewer_gates", [])
        artifact_requirements = contract.get("artifact_requirements", {})
        validation_requirements = contract.get("validation_requirements", {})
        leakage_execution_plan = contract.get("leakage_execution_plan", {})
        allowed_feature_sets = contract.get("allowed_feature_sets", {})
        canonical_columns = contract.get("canonical_columns", [])
        derived_columns = contract.get("derived_columns", [])
        required_outputs = contract.get("required_outputs", [])
        data_limited_mode = contract.get("data_limited_mode", {})
        
        # Build evaluation spec from V4.1 contract
        spec = {
            "qa_gates": qa_gates if isinstance(qa_gates, list) else [],
            "reviewer_gates": reviewer_gates if isinstance(reviewer_gates, list) else [],
            "artifact_requirements": artifact_requirements if isinstance(artifact_requirements, dict) else {},
            "validation_requirements": validation_requirements if isinstance(validation_requirements, dict) else {},
            "leakage_execution_plan": leakage_execution_plan if isinstance(leakage_execution_plan, dict) else {},
            "allowed_feature_sets": allowed_feature_sets if isinstance(allowed_feature_sets, dict) else {},
            "canonical_columns": canonical_columns if isinstance(canonical_columns, list) else [],
            "derived_columns": derived_columns if isinstance(derived_columns, list) else [],
            "required_outputs": required_outputs if isinstance(required_outputs, list) else [],
            "data_limited_mode": data_limited_mode if isinstance(data_limited_mode, dict) else {},
            "confidence": 0.9,
            "source": "contract_driven_v41",
            "notes": ["Extracted directly from V4.1 contract fields"]
        }
        
        # Add unknowns if critical fields are missing
        unknowns = []
        if not qa_gates:
            unknowns.append("qa_gates missing from contract")
        if not reviewer_gates:
            unknowns.append("reviewer_gates missing from contract")
        if not required_outputs:
            unknowns.append("required_outputs missing from contract")
        
        if unknowns:
            spec["unknowns"] = unknowns
            spec["confidence"] = 0.6  # Lower confidence if fields missing
        
        return spec

