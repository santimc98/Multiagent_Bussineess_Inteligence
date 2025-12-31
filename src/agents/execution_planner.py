import json
import os
from typing import Dict, Any, List
from string import Template
import re
import difflib

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from src.utils.contract_validation import (
    ensure_role_runbooks,
    DEFAULT_DATA_ENGINEER_RUNBOOK,
    DEFAULT_ML_ENGINEER_RUNBOOK,
    validate_spec_extraction_structure,
)

load_dotenv()

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

    def generate_contract(self, strategy: Dict[str, Any], data_summary: str = "", business_objective: str = "", column_inventory: list[str] | None = None) -> Dict[str, Any]:
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
                    if canonical:
                        canonical_cols.append(canonical)
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
            contract["spec_extraction"] = spec
            planner_self_check = contract.get("planner_self_check")
            if not isinstance(planner_self_check, list):
                contract["planner_self_check"] = []
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

        def _ensure_iteration_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            if not isinstance(contract.get("compliance_checklist"), list):
                contract["compliance_checklist"] = []
            if not isinstance(contract.get("iteration_policy"), dict):
                contract["iteration_policy"] = {}
            return contract

        def _normalize_alignment_requirements(items: Any) -> List[Dict[str, Any]]:
            if not isinstance(items, list):
                return []
            normalized: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                req_id = str(item.get("id") or f"custom_{idx}")
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
            spec = contract.get("spec_extraction") or {}
            case_taxonomy = spec.get("case_taxonomy") if isinstance(spec, dict) else None
            if isinstance(case_taxonomy, list) and case_taxonomy:
                segment_required = True
            roles = [
                str(req.get("role") or "").lower()
                for req in contract.get("data_requirements", [])
                if isinstance(req, dict)
            ]
            if any(tok in role for role in roles for tok in ["segment", "group", "cluster"]):
                segment_required = True
            combined_text = "\n".join(
                [
                    business_objective or "",
                    data_summary or "",
                    json.dumps(strategy, ensure_ascii=True) if isinstance(strategy, dict) else "",
                ]
            )
            if _detect_segment_context(combined_text):
                segment_required = True
            if segment_required:
                _add(
                    "segment_alignment",
                    "Segment-level or case-level analysis is produced when segmentation is part of the strategy.",
                    "Global-only results can mask segment variability required for the decision.",
                    [
                        "Segment-level metrics or elasticity are reported.",
                        "If only one segment is valid, warn and aggregate with rationale.",
                    ],
                    "segmentation_relevant",
                    "data_limited",
                )

            feature_availability = contract.get("feature_availability")
            if isinstance(feature_availability, list) and feature_availability:
                _add(
                    "availability_alignment",
                    "Training uses only pre-decision features; post-outcome fields are excluded or audited.",
                    "Ensures recommendations are based on information available at decision time.",
                    [
                        "Features used for modeling are available at decision time.",
                        "Post-outcome fields are excluded or documented as leakage risks.",
                    ],
                    "feature_availability_present",
                    "method_choice",
                )

            quality_gates = contract.get("quality_gates")
            if isinstance(quality_gates, dict) and quality_gates:
                _add(
                    "validation_minimum",
                    "Validation metrics are computed and compared to quality gates.",
                    "Keeps the solution aligned with the minimum acceptance criteria.",
                    [
                        "Metrics are reported and checked against quality_gates.",
                        "Shortfalls are explained with data limitations or remediation steps.",
                    ],
                    "quality_gates_present",
                    "method_choice",
                )

            return requirements

        def _attach_alignment_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            existing = contract.get("alignment_requirements")
            base_reqs = _build_alignment_requirements(contract)
            merged: List[Dict[str, Any]] = []
            base_by_id = {req.get("id"): dict(req) for req in base_reqs if isinstance(req, dict)}
            merged.extend(base_reqs)

            normalized_existing = _normalize_alignment_requirements(existing)
            for item in normalized_existing:
                req_id = item.get("id")
                if not req_id:
                    continue
                if req_id in base_by_id:
                    base_item = base_by_id[req_id]
                    for key in ("requirement", "rationale", "applies_when", "failure_mode_on_miss"):
                        if item.get(key):
                            base_item[key] = item[key]
                    if item.get("success_criteria"):
                        base_item["success_criteria"] = item["success_criteria"]
                    if item.get("evidence"):
                        base_item["evidence"] = item["evidence"]
                else:
                    merged.append(item)
                    base_by_id[req_id] = item

            contract["alignment_requirements"] = merged

            checklist = contract.get("compliance_checklist")
            if not isinstance(checklist, list):
                checklist = []
            for req in merged:
                req_id = req.get("id")
                req_text = req.get("requirement")
                if not req_id or not req_text:
                    continue
                line = f"Alignment requirement ({req_id}): {req_text}"
                if line not in checklist:
                    checklist.append(line)
            contract["compliance_checklist"] = checklist
            return contract

        def _build_evaluation_spec_fallback(contract: Dict[str, Any]) -> Dict[str, Any]:
            objective_text = (contract.get("business_objective") or business_objective or "").lower()
            strategy_text = json.dumps(strategy or {}).lower()
            objective_type = "descriptive"
            if any(tok in objective_text for tok in ["optimiz", "maximize", "optimal price", "precio", "revenue", "expected value"]):
                objective_type = "prescriptive"
            elif any(tok in objective_text for tok in ["predict", "classif", "regress", "forecast", "probability"]):
                objective_type = "predictive"

            decision_vars = contract.get("decision_variables") or []
            decision_var = decision_vars[0] if decision_vars else None
            segmentation_required = any(tok in strategy_text for tok in ["segment", "cluster", "typology", "tipolog"])
            feature_availability = contract.get("feature_availability") or []
            pre_decision = [
                item.get("column")
                for item in feature_availability
                if isinstance(item, dict) and str(item.get("availability", "")).lower() == "pre-decision"
            ]
            leakage_features = [
                item.get("column")
                for item in feature_availability
                if isinstance(item, dict) and "post" in str(item.get("availability", "")).lower()
            ]
            derived_target_required = False
            for req in contract.get("data_requirements", []) or []:
                if not isinstance(req, dict):
                    continue
                if str(req.get("role", "")).lower() == "target" and str(req.get("source", "input")).lower() == "derived":
                    derived_target_required = True
                    break

            qa_gates = [
                "mapping_summary",
                "canonical_mapping",
                "consistency_checks",
                "target_variance_guard",
                "outputs_required",
            ]
            if derived_target_required:
                qa_gates.append("target_derivation_required")
            if leakage_features:
                qa_gates.append("leakage_prevention")
            if segmentation_required:
                qa_gates.append("segmentation_predecision")
            reviewer_gates = ["methodology_alignment", "business_value"]
            if objective_type in {"predictive", "prescriptive"}:
                reviewer_gates.append("validation_required")
            if objective_type == "prescriptive":
                reviewer_gates.append("price_optimization")

            alignment_requirements = contract.get("alignment_requirements") or []
            if not alignment_requirements:
                alignment_requirements = [
                    {"id": "objective_alignment", "description": "Output aligns with business objective.", "required": True},
                    {"id": "decision_variable_handling", "description": "Decision variables used correctly.", "required": bool(decision_var)},
                    {"id": "segment_alignment", "description": "Segmentation uses only pre-decision features.", "required": segmentation_required},
                    {"id": "validation_minimum", "description": "Validation performed per policy.", "required": True},
                ]

            deliverables = (contract.get("spec_extraction") or {}).get("deliverables")
            evidence_requirements: List[Dict[str, Any]] = []
            if isinstance(deliverables, list) and deliverables:
                for item in deliverables:
                    if isinstance(item, dict) and item.get("path"):
                        evidence_requirements.append(
                            {"artifact": item.get("path"), "required": bool(item.get("required", True))}
                        )
                    elif isinstance(item, str):
                        evidence_requirements.append({"artifact": item, "required": True})
            else:
                evidence_requirements = [{"artifact": out, "required": True} for out in contract.get("required_outputs", [])]

            return {
                "objective_type": objective_type,
                "segmentation": {
                    "required": segmentation_required,
                    "features": pre_decision if segmentation_required else [],
                    "confidence": 0.4,
                    "rationale": "Derived from strategy text and feature availability."
                },
                "decision_variable": {
                    "name": decision_var,
                    "confidence": 0.4,
                    "rationale": "Derived from contract decision_variables."
                },
                "leakage_policy": {
                    "audit_features": leakage_features,
                    "correlation_threshold": 0.9,
                    "exclude_on_leakage": True
                },
                "validation_policy": {
                    "require_cv": objective_type in {"predictive", "prescriptive"},
                    "baseline_required": objective_type in {"predictive", "prescriptive"}
                },
                "qa_gates": qa_gates,
                "reviewer_gates": reviewer_gates,
                "alignment_requirements": alignment_requirements,
                "evidence_requirements": evidence_requirements,
                "confidence": 0.4,
                "justification": "Fallback evaluation spec derived heuristically from contract and strategy."
            }

        def _normalize_quality_gates(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            gates = contract.get("quality_gates")
            if isinstance(gates, dict) and gates:
                return contract
            raw = contract.get("quality_gates_raw")
            if not isinstance(raw, list):
                return contract
            normalized: Dict[str, Any] = {}
            for item in raw:
                if not isinstance(item, dict):
                    continue
                metric = item.get("metric")
                threshold = item.get("threshold")
                if metric and threshold is not None:
                    normalized[str(metric)] = threshold
            if normalized:
                contract["quality_gates"] = normalized
                planner_self_check = contract.get("planner_self_check")
                if not isinstance(planner_self_check, list):
                    planner_self_check = []
                msg = "Derived quality_gates from quality_gates_raw for downstream gating."
                if msg not in planner_self_check:
                    planner_self_check.append(msg)
                contract["planner_self_check"] = planner_self_check
            return contract

        def _attach_spec_extraction_to_runbook(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction")
            if not isinstance(spec, dict):
                return contract
            runbooks = contract.get("role_runbooks")
            if not isinstance(runbooks, dict):
                return contract
            ml_runbook = runbooks.get("ml_engineer")
            if not isinstance(ml_runbook, dict):
                return contract
            ml_runbook["spec_extraction"] = spec
            runbooks["ml_engineer"] = ml_runbook
            contract["role_runbooks"] = runbooks
            return contract

        def _attach_business_alignment(contract: Dict[str, Any]) -> Dict[str, Any]:
            return _propagate_business_alignment(contract)

        def _fallback() -> Dict[str, Any]:
            required_cols = strategy.get("required_columns", []) or []
            data_requirements: List[Dict[str, Any]] = []
            for col in required_cols:
                name_lower = str(col).lower()
                role = "feature"
                expected_range = None
                if any(tok in name_lower for tok in ["pct", "percent", "ratio", "prob", "rate"]):
                    role = "percentage"
                    expected_range = [0, 1]
                elif "score" in name_lower:
                    role = "risk_score"
                    expected_range = [0, 1]
                data_requirements.append(
                    {
                        "name": col,
                        "role": role,
                        "expected_range": expected_range,
                        "allowed_null_frac": None,
                        "source": "input",
                    }
                )
            outputs = ["data/cleaned_data.csv"]
            title_lower = str(strategy.get("title", "")).lower()
            required_deps: List[str] = []
            if "xgboost" in title_lower:
                required_deps.append("xgboost")
            if "statsmodel" in title_lower or "statsmodels" in title_lower:
                required_deps.append("statsmodels")
            if "parquet" in title_lower:
                required_deps.append("pyarrow")
            if "excel" in title_lower or "xlsx" in title_lower:
                required_deps.append("openpyxl")
            contract = {
                "contract_version": 1,
                "strategy_title": strategy.get("title", ""),
                "business_objective": business_objective,
                "required_outputs": outputs,
                "data_requirements": data_requirements,
                "required_dependencies": required_deps,
                "feature_availability": [],
                "availability_summary": "Planner fallback used; no explicit availability reasoning provided.",
                "compliance_checklist": [],
                "iteration_policy": {
                    "compliance_bootstrap_max": 2,
                    "metric_improvement_max": 6,
                    "runtime_fix_max": 3,
                },
                "quality_gates": {
                    "spearman_min": 0.85,
                    "violations_max": 0,
                    "inactive_share_max": 0.01,
                    "max_weight_max": 0.70,
                    "hhi_max": 0.60,
                    "near_zero_max": 1,
                },
                "optimization_preferences": {
                    "regularization": {"l2": 0.05, "concentration_penalty": 0.1},
                    "ranking_loss": "hinge_pairwise",
                },
                "business_alignment": {},
                "validations": [],
                "notes_for_engineers": [
                    "Refine roles/ranges using data_summary evidence; adjust in Patch Mode if needed.",
                    "Align cleaning/modeling with this contract; avoid hardcoded business rules.",
                    "Planner fallback used; spec_extraction may be empty.",
                ],
                "role_runbooks": {
                    "data_engineer": DEFAULT_DATA_ENGINEER_RUNBOOK,
                    "ml_engineer": DEFAULT_ML_ENGINEER_RUNBOOK,
                },
                "spec_extraction": {
                    "derived_columns": [],
                    "case_taxonomy": [],
                    "constraints": [],
                    "deliverables": [],
                    "scoring_formula": None,
                    "target_type": None,
                    "leakage_policy": None,
                },
                "planner_self_check": [],
            }
            contract = _ensure_formula_requirements(contract)
            contract = _ensure_strategy_requirements(contract)
            contract = _apply_inventory_source(contract)
            contract = _apply_expected_kind(contract)
            contract = _ensure_case_id_requirement(contract)
            contract = _attach_canonical_names(contract)
            contract = enforce_percentage_ranges(contract)
            contract = ensure_role_runbooks(contract)
            contract = _attach_data_risks(contract)
            contract = _attach_spec_extraction_issues(contract)
            contract = _normalize_quality_gates(contract)
            contract = _ensure_spec_extraction(contract)
            contract = _apply_deliverables(contract)
            contract = _attach_business_alignment(contract)
            contract = _attach_strategy_context(contract)
            contract = _attach_semantic_guidance(contract)
            contract = _assign_derived_owners(contract)
            contract = _attach_variable_semantics(contract)
            contract = _ensure_availability_reasoning(contract)
            contract = _attach_alignment_requirements(contract)
            contract = _ensure_iteration_policy(contract)
            return _attach_spec_extraction_to_runbook(contract)

        if not self.client:
            return _fallback()

        column_inventory_json = json.dumps(column_inventory or [])
        system_prompt = Template(
            """
You are a senior execution planner inside a multi-agent system. Produce a JSON contract that helps
downstream AI engineers reason better (guidance, not rigid imperative rules).

Requirements:
- Output JSON ONLY (no markdown/code fences).
- Include: contract_version, strategy_title, business_objective, required_outputs, data_requirements, validations,
  notes_for_engineers, required_dependencies, data_risks, spec_extraction, planner_self_check,
  compliance_checklist, iteration_policy.
- Include feature_availability (list of {column, availability, rationale}) and availability_summary (string).
  Use this to reason about pre-decision vs post-outcome fields and leakage risk. This is a reasoning aid, not a rule.
- data_requirements: list of {name, role, expected_range, allowed_null_frac, source, expected_kind}.
  expected_kind in {numeric, datetime, categorical, unknown}.
- Each data_requirement may include source: "input" | "derived" (default input).
- Use expected_range when the business context implies it (e.g., [0,1] for normalized scores/ratios).
- validations: generic checks (ranking coherence, out-of-range, weight constraints).
- Prefer using columns from the strategy and data_summary; if something must be derived, mark source="derived"
  and explain in data_risks.
- SPEC EXTRACTION deliverables must be a list of objects with keys {id, path, required, kind, description}.
- required_outputs must equal [d.path for d in spec_extraction.deliverables if d.required] and include data/cleaned_data.csv.
- Include role_runbooks to guide reasoning (goals/must/must_not/safe_idioms/reasoning_checklist/validation_checklist).
- COLUMN INVENTORY (detected from CSV header) to help decide source input/derived: $column_inventory
- required_dependencies is optional; include only if strongly implied by the strategy title or data_summary.
- ALWAYS include quality_gates with explicit metrics + thresholds for this specific business objective.
  Do NOT leave quality_gates empty. Do NOT rely on defaults.
- If you use quality_gates_raw (list form), also populate quality_gates (metric -> threshold) so evaluators can act.
- ALWAYS include business_alignment with ordered optimization priorities and acceptance criteria.
  This must be derived from the objective (not generic).
- SPEC EXTRACTION: map explicit formulas, derived columns, cases, constraints, deliverables, target_type, leakage_policy.
  If not stated, leave empty/null; do not invent.
- Infer expected_kind using evidence from data_summary and column inventory. If a column looks like a person/role/channel/sector
  or identifier, treat as categorical; if it looks like a date/time, treat as datetime; otherwise use numeric only when justified.
- Provide feature_semantics (short business meaning per column) and business_sanity_checks (checks to interpret results).
  These are reasoning aids, not hard rules.
- Include compliance_checklist (list of concrete compliance items that must pass before metric tuning begins).
  Derive it from the contract itself (quality_gates, deliverables, leakage policy) rather than generic boilerplate.
- Include alignment_requirements: concise alignment checks derived from objective/strategy (decision variables,
  segment-level validity, validation minima). Keep them universal and evidence-focused.
- Include iteration_policy with:
  * compliance_bootstrap_max (iterations to fix compliance issues),
  * metric_improvement_max (iterations to improve metrics),
  * runtime_fix_max (runtime error retries).
  Choose values appropriate to the task difficulty and budget.
- If the objective involves pricing, segmentation, or threshold-based recommendations, consider whether a derived tier/segment
  column (e.g., size deciles or sector groups) would materially improve downstream reasoning; include it only if justified
  by the strategy and data. Do not treat tiering as universally required.
- SELF CHECK: short statements confirming you defined quality_gates and business_alignment consistent with the objective.
  """
        ).substitute(column_inventory=column_inventory_json)
        user_prompt_template = Template(
            """
BUSINESS_OBJECTIVE:
$business_objective

STRATEGY:
$strategy_json

DATA_SUMMARY:
$data_summary

CSV COLUMN INVENTORY:
$column_inventory

Return the contract JSON.
"""
        )
        user_prompt = user_prompt_template.substitute(
            business_objective=business_objective,
            strategy_json=json.dumps(strategy, indent=2),
            data_summary=data_summary,
            column_inventory=json.dumps(column_inventory or []),
        )
        try:
            full_prompt = system_prompt + "\n\nUSER_INPUT:\n" + user_prompt
            response = self.client.generate_content(full_prompt)
            content = response.text
            content = content.replace("```json", "").replace("```", "").strip()
            contract = json.loads(content)
            if not isinstance(contract, dict) or "data_requirements" not in contract:
                return _fallback()
            if "required_dependencies" not in contract:
                contract["required_dependencies"] = []
            if "quality_gates" not in contract:
                contract["quality_gates"] = {}
            if "optimization_preferences" not in contract:
                contract["optimization_preferences"] = {}
            contract = _ensure_formula_requirements(contract)
            contract = _ensure_strategy_requirements(contract)
            contract = _apply_inventory_source(contract)
            contract = _apply_expected_kind(contract)
            contract = _ensure_case_id_requirement(contract)
            contract = _attach_canonical_names(contract)
            contract = enforce_percentage_ranges(contract)
            contract = ensure_role_runbooks(contract)
            contract = _attach_data_risks(contract)
            contract = _attach_spec_extraction_issues(contract)
            contract = _normalize_quality_gates(contract)
            contract = _ensure_spec_extraction(contract)
            contract = _apply_deliverables(contract)
            contract = _attach_business_alignment(contract)
            contract = _attach_strategy_context(contract)
            contract = _attach_semantic_guidance(contract)
            contract = _assign_derived_owners(contract)
            contract = _attach_variable_semantics(contract)
            contract = _ensure_availability_reasoning(contract)
            contract = _attach_alignment_requirements(contract)
            contract = _ensure_iteration_policy(contract)
            return _attach_spec_extraction_to_runbook(contract)
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
        def _derive_flag_defaults(spec: Dict[str, Any]) -> Dict[str, Any]:
            objective_type = str(spec.get("objective_type") or "").lower()
            requires_target = spec.get("requires_target")
            if requires_target is None:
                requires_target = objective_type in {"predictive", "prescriptive", "forecasting"}
            requires_time_series_split = spec.get("requires_time_series_split")
            if requires_time_series_split is None:
                requires_time_series_split = objective_type == "forecasting"
            requires_supervised_split = spec.get("requires_supervised_split")
            if requires_supervised_split is None:
                requires_supervised_split = bool(requires_target)
            requires_row_scoring = spec.get("requires_row_scoring")
            if requires_row_scoring is None:
                deliverables = (contract.get("spec_extraction") or {}).get("deliverables") if isinstance(contract, dict) else []
                requires_row_scoring = False
                if isinstance(deliverables, list):
                    for item in deliverables:
                        if isinstance(item, dict) and item.get("path") == "data/scored_rows.csv":
                            requires_row_scoring = bool(item.get("required", True))
                            break
                        if isinstance(item, str) and item == "data/scored_rows.csv":
                            requires_row_scoring = True
                            break
            spec["requires_target"] = bool(requires_target)
            spec["requires_time_series_split"] = bool(requires_time_series_split)
            spec["requires_supervised_split"] = bool(requires_supervised_split)
            spec["requires_row_scoring"] = bool(requires_row_scoring)
            return spec

        def _fallback() -> Dict[str, Any]:
            # Reuse the heuristic builder inside generate_contract scope via a lightweight copy
            objective_text = (contract.get("business_objective") or business_objective or "").lower()
            strategy_text = json.dumps(strategy or {}).lower()
            objective_type = "descriptive"
            if any(tok in objective_text for tok in ["optimiz", "maximize", "optimal price", "precio", "revenue", "expected value"]):
                objective_type = "prescriptive"
            elif any(tok in objective_text for tok in ["predict", "classif", "regress", "forecast", "probability"]):
                objective_type = "predictive"

            decision_vars = contract.get("decision_variables") or []
            decision_var = decision_vars[0] if decision_vars else None
            segmentation_required = any(tok in strategy_text for tok in ["segment", "cluster", "typology", "tipolog"])
            feature_availability = contract.get("feature_availability") or []
            pre_decision = [
                item.get("column")
                for item in feature_availability
                if isinstance(item, dict) and str(item.get("availability", "")).lower() == "pre-decision"
            ]
            leakage_features = [
                item.get("column")
                for item in feature_availability
                if isinstance(item, dict) and "post" in str(item.get("availability", "")).lower()
            ]
            qa_gates = ["mapping_summary", "consistency_checks", "target_variance_guard", "outputs_required"]
            if leakage_features:
                qa_gates.append("leakage_prevention")
            if segmentation_required:
                qa_gates.append("segmentation_predecision")
            reviewer_gates = ["methodology_alignment", "business_value"]
            if objective_type in {"predictive", "prescriptive"}:
                reviewer_gates.append("validation_required")
            if objective_type == "prescriptive":
                reviewer_gates.append("price_optimization")

            alignment_requirements = contract.get("alignment_requirements") or []
            if not alignment_requirements:
                alignment_requirements = [
                    {"id": "objective_alignment", "description": "Output aligns with business objective.", "required": True},
                    {"id": "decision_variable_handling", "description": "Decision variables used correctly.", "required": bool(decision_var)},
                    {"id": "segment_alignment", "description": "Segmentation uses only pre-decision features.", "required": segmentation_required},
                    {"id": "validation_minimum", "description": "Validation performed per policy.", "required": True},
                ]

            deliverables = (contract.get("spec_extraction") or {}).get("deliverables")
            evidence_requirements: List[Dict[str, Any]] = []
            if isinstance(deliverables, list) and deliverables:
                for item in deliverables:
                    if isinstance(item, dict) and item.get("path"):
                        evidence_requirements.append(
                            {"artifact": item.get("path"), "required": bool(item.get("required", True))}
                        )
                    elif isinstance(item, str):
                        evidence_requirements.append({"artifact": item, "required": True})
            else:
                evidence_requirements = [{"artifact": out, "required": True} for out in contract.get("required_outputs", [])]

            spec = {
                "objective_type": objective_type,
                "segmentation": {
                    "required": segmentation_required,
                    "features": pre_decision if segmentation_required else [],
                    "confidence": 0.4,
                    "rationale": "Derived from strategy text and feature availability."
                },
                "decision_variable": {
                    "name": decision_var,
                    "confidence": 0.4,
                    "rationale": "Derived from contract decision_variables."
                },
                "leakage_policy": {
                    "audit_features": leakage_features,
                    "correlation_threshold": 0.9,
                    "exclude_on_leakage": True
                },
                "validation_policy": {
                    "require_cv": objective_type in {"predictive", "prescriptive"},
                    "baseline_required": objective_type in {"predictive", "prescriptive"}
                },
                "qa_gates": qa_gates,
                "reviewer_gates": reviewer_gates,
                "alignment_requirements": alignment_requirements,
                "evidence_requirements": evidence_requirements,
                "confidence": 0.4,
                "justification": "Fallback evaluation spec derived heuristically from contract and strategy."
            }
            return _derive_flag_defaults(spec)

        if not self.client:
            return _fallback()

        spec_template = {
            "objective_type": "prescriptive|predictive|descriptive|causal|unknown",
            "segmentation": {"required": False, "features": [], "confidence": 0.0, "rationale": ""},
            "decision_variable": {"name": None, "confidence": 0.0, "rationale": ""},
            "leakage_policy": {"audit_features": [], "correlation_threshold": 0.9, "exclude_on_leakage": True},
            "validation_policy": {"require_cv": False, "baseline_required": False},
            "qa_gates": [],
            "reviewer_gates": [],
            "alignment_requirements": [],
            "evidence_requirements": [],
            "requires_target": False,
            "requires_supervised_split": False,
            "requires_time_series_split": False,
            "requires_row_scoring": False,
            "confidence": 0.0,
            "justification": ""
        }

        system_prompt = (
            "You are a senior evaluation architect. Derive an evaluation_spec for QA/Reviewer based on the "
            "business objective, strategy, and execution contract. Do NOT use static templates. "
            "If unsure, set low confidence and mark requirements as non-required instead of inventing gates. "
            "Return JSON only, matching the template. "
            "Gate vocabulary (use only these ids): "
            "mapping_summary, consistency_checks, target_variance_guard, leakage_prevention, outputs_required, "
            "segmentation_predecision, decision_variable_handling, validation_required, price_optimization, "
            "methodology_alignment, business_value. "
            "Include requires_target/requires_supervised_split/requires_time_series_split/requires_row_scoring flags."
        )
        user_payload = {
            "business_objective": business_objective,
            "strategy": strategy,
            "contract_summary": {
                "required_outputs": contract.get("required_outputs"),
                "decision_variables": contract.get("decision_variables"),
                "feature_availability": contract.get("feature_availability"),
                "alignment_requirements": contract.get("alignment_requirements"),
                "quality_gates": contract.get("quality_gates"),
            },
            "data_summary": data_summary,
            "column_inventory": column_inventory,
            "template": spec_template,
        }

        try:
            response = self.client.generate_content(
                json.dumps({"system": system_prompt, "input": user_payload})
            )
            content = getattr(response, "text", "")
            spec = json.loads(content) if content else {}
            if not isinstance(spec, dict):
                return _fallback()
            return _derive_flag_defaults(spec)
        except Exception:
            return _fallback()
