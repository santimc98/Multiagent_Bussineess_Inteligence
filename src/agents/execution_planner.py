import json
import os
from typing import Dict, Any, List
from string import Template
import re
import difflib

from dotenv import load_dotenv
from openai import OpenAI
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


class ExecutionPlannerAgent:
    """
    LLM-driven planner that emits an execution contract (JSON) to guide downstream agents.
    Falls back to heuristic contract if the model call fails.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
                timeout=None,
            )
        self.model_name = "deepseek-reasoner"

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
                role = (req.get("role") or "").lower()
                if role in {"percentage", "feature", "risk_score", "probability", "ratio"}:
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
                canonical = req.get("canonical_name")
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
            if not isinstance(contract, dict):
                return contract
            spec = contract.get("spec_extraction", {}) if isinstance(contract.get("spec_extraction"), dict) else {}
            target_type = str(spec.get("target_type") or "").lower()
            has_cases = bool(spec.get("case_taxonomy"))
            quality_gates = contract.get("quality_gates", {}) if isinstance(contract.get("quality_gates"), dict) else {}
            priorities: List[Dict[str, Any]] = []
            acceptance: Dict[str, Any] = {}
            notes: List[str] = []

            if has_cases and target_type in {"ordinal", "ranking"}:
                priorities.append(
                    {
                        "priority": 1,
                        "goal": "minimize_case_order_violations",
                        "metric": "case_order_violations",
                        "direction": "minimize",
                    }
                )
                priorities.append(
                    {
                        "priority": 2,
                        "goal": "maximize_case_level_rank_correlation",
                        "metric": "case_level_spearman",
                        "direction": "maximize",
                    }
                )
                priorities.append(
                    {
                        "priority": 3,
                        "goal": "maximize_global_rank_correlation",
                        "metric": "spearman",
                        "direction": "maximize",
                    }
                )
                acceptance["case_order_violations_max"] = quality_gates.get("violations_max", 0)
                acceptance["spearman_min"] = quality_gates.get("spearman_min")
                notes.append(
                    "Primary objective is case ordering consistency; global ranking is secondary when case taxonomy is defined."
                )
            else:
                priorities.append(
                    {
                        "priority": 1,
                        "goal": "maximize_global_rank_correlation",
                        "metric": "spearman",
                        "direction": "maximize",
                    }
                )
                acceptance["spearman_min"] = quality_gates.get("spearman_min")

            contract["business_alignment"] = {
                "optimization_priorities": priorities,
                "acceptance_criteria": acceptance,
                "notes": notes,
            }

            runbooks = contract.get("role_runbooks")
            if isinstance(runbooks, dict):
                ml_runbook = runbooks.get("ml_engineer")
                if isinstance(ml_runbook, dict):
                    ml_runbook["business_alignment"] = contract["business_alignment"]
                    runbooks["ml_engineer"] = ml_runbook
                    contract["role_runbooks"] = runbooks
            return contract

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
            if any(tok in title_lower for tok in ["score", "weight", "ranking", "priorit"]):
                outputs.extend(["data/weights.json", "data/case_summary.csv", "static/plots/*.png"])
                outputs.append("data/case_alignment_report.json")
            else:
                outputs.extend(["data/metrics.json", "static/plots/*.png"])
            contract = {
                "contract_version": 1,
                "strategy_title": strategy.get("title", ""),
                "business_objective": business_objective,
                "required_outputs": outputs,
                "data_requirements": data_requirements,
                "required_dependencies": required_deps,
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
            contract = _apply_inventory_source(contract)
            contract = _apply_expected_kind(contract)
            contract = _attach_canonical_names(contract)
            contract = enforce_percentage_ranges(contract)
            contract = ensure_role_runbooks(contract)
            contract = _attach_data_risks(contract)
            contract = _attach_spec_extraction_issues(contract)
            contract = _ensure_spec_extraction(contract)
            contract = _attach_business_alignment(contract)
            return _attach_spec_extraction_to_runbook(contract)

        if not self.client:
            return _fallback()

        column_inventory_json = json.dumps(column_inventory or [])
        system_prompt = Template(
            """
You are a senior execution planner. Produce a JSON contract to guide data cleaning and modeling.
Requirements:
- Output JSON ONLY (no markdown/code fences).
- Include: contract_version, strategy_title, business_objective, required_outputs, data_requirements, validations, notes_for_engineers, required_dependencies, data_risks, spec_extraction, planner_self_check.
        - data_requirements: list of {name, role, expected_range, allowed_null_frac, source, expected_kind}. expected_kind in {numeric, datetime, categorical, unknown}.
- Each data_requirement may include source: "input" | "derived" (default input). If role==target and the name is not present in the column inventory from data_summary, mark source="derived" if reasonable.
- expected_range e.g. [0,1] for probabilities/scores/percentages if implied by the column description.
- validations: generic checks (e.g., ranking_coherence spearman, out_of_range).
  - Do NOT invent columns; base everything on the provided strategy and data_summary.
  - If a required column is NOT present in the column inventory, mark source="derived" and add a data_risks note.
  - data_risks should list potential failure modes (missing columns, low variance, dialect/encoding sensitivity, sampling).
  - required_outputs should always include data/cleaned_data.csv. For scoring/weights/ranking strategies, also include data/weights.json, data/case_summary.csv, and at least one plot like static/plots/*.png. For standard classification/regression, include data/metrics.json and one plot.
  - Include role_runbooks for data_engineer and ml_engineer with: goals, must, must_not, safe_idioms, reasoning_checklist, validation_checklist, and (for DE) manifest_requirements; (for ML) methodology and outputs.
  - If role == "date", use expected_range=null. Do not mix numeric ranges with null (avoid [0, null]); either null or a full numeric range when appropriate.
          - COLUMN INVENTORY (detected from CSV header) to help decide source input/derived: $column_inventory
          - required_dependencies is optional; include only if strongly implied by the strategy title or data_summary. Use module names (e.g., "xgboost", "statsmodels", "pyarrow", "openpyxl"). Otherwise use [].
        - Include quality_gates (spearman_min, violations_max, inactive_share_max, max_weight_max, hhi_max, near_zero_max) and optimization_preferences (regularization + ranking_loss).
        - Include business_alignment with optimization_priorities (ordered list with priority/goal/metric/direction) and acceptance_criteria tied to quality_gates and spec_extraction.
        - Include role_runbooks as above to guide engineers with run-time safe idioms and reasoning checks.
  - SPEC EXTRACTION (from business_objective + strategy): include a spec_extraction object with:
    - scoring_formula: a string formula if explicitly stated, else null.
    - derived_columns: list of {name, formula, depends_on, constraints} for any explicitly described derived fields.
    - case_taxonomy: list of {case_id_or_name, conditions, ref_score, precedence} if the objective defines cases or tables.
    - constraints: list of explicit constraints (e.g., weight non-negativity, sum-to-one) if stated.
    - deliverables: list of explicitly requested outputs (tables, checks, comparisons) if stated.
    - target_type: "ranking" or "ordinal" if the objective says ordinal/ranking, else null.
    - leakage_policy: "allow_deterministic_for_design" if the target/case is defined using input variables; else null.
    - If a spec element is not stated, leave it empty/null. Do NOT invent.
  - SELF CHECK: include planner_self_check list with short statements verifying:
    - If scoring_formula is not null, it appears in spec_extraction.
    - If case_taxonomy is non-empty, deliverables include a case-level table.
    - If constraints are stated, they appear in constraints.
    - If target_type is ordinal/ranking, include a ranking validation in validations.
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            contract = json.loads(content)
            if not isinstance(contract, dict) or "data_requirements" not in contract:
                return _fallback()
            if "required_dependencies" not in contract:
                contract["required_dependencies"] = []
            if "quality_gates" not in contract:
                contract["quality_gates"] = {
                    "spearman_min": 0.85,
                    "violations_max": 0,
                    "inactive_share_max": 0.01,
                    "max_weight_max": 0.70,
                    "hhi_max": 0.60,
                    "near_zero_max": 1,
                }
            if "optimization_preferences" not in contract:
                contract["optimization_preferences"] = {
                    "regularization": {"l2": 0.05, "concentration_penalty": 0.1},
                    "ranking_loss": "hinge_pairwise",
                }
            contract = _apply_inventory_source(contract)
            contract = _apply_expected_kind(contract)
            contract = _attach_canonical_names(contract)
            contract = enforce_percentage_ranges(contract)
            contract = ensure_role_runbooks(contract)
            contract = _attach_data_risks(contract)
            contract = _attach_spec_extraction_issues(contract)
            contract = _ensure_spec_extraction(contract)
            contract = _attach_business_alignment(contract)
            return _attach_spec_extraction_to_runbook(contract)
        except Exception:
            return _fallback()
