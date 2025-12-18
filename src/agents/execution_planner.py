import json
import os
from typing import Dict, Any, List
from string import Template
import re
import difflib

from dotenv import load_dotenv
from openai import OpenAI

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
            else:
                outputs.extend(["data/metrics.json", "static/plots/*.png"])
            contract = {
                "contract_version": 1,
                "strategy_title": strategy.get("title", ""),
                "business_objective": business_objective,
                "required_outputs": outputs,
                "data_requirements": data_requirements,
                "required_dependencies": required_deps,
                "validations": [],
                "notes_for_engineers": [
                    "Refine roles/ranges using data_summary evidence; adjust in Patch Mode if needed.",
                    "Align cleaning/modeling with this contract; avoid hardcoded business rules.",
                ],
            }
            return contract

        if not self.client:
            return _fallback()

        column_inventory_json = json.dumps(column_inventory or [])
        system_prompt = Template(
            """
You are a senior execution planner. Produce a JSON contract to guide data cleaning and modeling.
Requirements:
- Output JSON ONLY (no markdown/code fences).
- Include: contract_version, strategy_title, business_objective, required_outputs, data_requirements, validations, notes_for_engineers, required_dependencies.
        - data_requirements: list of {name, role, expected_range, allowed_null_frac, source}. Roles: target|feature|percentage|probability|categorical|date|risk_score. source: "input" | "derived".
- Each data_requirement may include source: "input" | "derived" (default input). If role==target and the name is not present in the column inventory from data_summary, mark source="derived" if reasonable.
- expected_range e.g. [0,1] for probabilities/scores/percentages if implied by the column description.
- validations: generic checks (e.g., ranking_coherence spearman, out_of_range).
- Do NOT invent columns; base everything on the provided strategy and data_summary.
- required_outputs should always include data/cleaned_data.csv. For scoring/weights/ranking strategies, also include data/weights.json, data/case_summary.csv, and at least one plot like static/plots/*.png. For standard classification/regression, include data/metrics.json and one plot.
- If role == "date", use expected_range=null. Do not mix numeric ranges with null (avoid [0, null]); either null or a full numeric range when appropriate.
        - COLUMN INVENTORY (detected from CSV header) to help decide source input/derived: $column_inventory
        - required_dependencies is optional; include only if strongly implied by the strategy title or data_summary. Use module names (e.g., "xgboost", "statsmodels", "pyarrow", "openpyxl"). Otherwise use [].
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
            return enforce_percentage_ranges(_apply_inventory_source(contract))
        except Exception:
            return _fallback()
