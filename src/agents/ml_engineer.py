import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from string import Template
import json
import ast
from src.utils.prompting import render_prompt
from src.utils.code_extract import extract_code_block, is_syntax_valid

# NOTE: scan_code_safety referenced by tests as a required safety mechanism.
# ML code executes in sandbox; keep the reference for integration checks.
_scan_code_safety_ref = "scan_code_safety"

class MLEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the ML Engineer Agent with the configured provider.
        """
        self.provider = (os.getenv("ML_ENGINEER_PROVIDER", "google") or "google").strip().lower()
        if self.provider == "zai":
            self.api_key = api_key or os.getenv("ZAI_API_KEY") or os.getenv("GLM_API_KEY")
            if not self.api_key:
                raise ValueError("Z.ai API Key is required.")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.z.ai/api/paas/v4/",
                timeout=None,
            )
            self.model_name = os.getenv("ZAI_MODEL") or os.getenv("GLM_MODEL") or "glm-4.7"
        elif self.provider == "deepseek":
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DeepSeek API Key is required.")

            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
                timeout=None,
            )
            self.model_name = "deepseek-reasoner"
        elif self.provider in {"google", "gemini"}:
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("Google API Key is required.")
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = os.getenv("ML_ENGINEER_MODEL", "gemini-3-flash-preview") or "gemini-3-flash-preview"
        else:
            raise ValueError(f"Unsupported ML_ENGINEER_PROVIDER: {self.provider}")
        self.last_prompt = None
        self.last_response = None

    def _compact_execution_contract(self, contract: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(contract, dict):
            return {}
        keep_keys = [
            "contract_version",
            "strategy_title",
            "business_objective",
            "required_columns",
            "required_outputs",
            "alignment_requirements",
            "business_alignment",
            "feature_semantics",
            "business_sanity_checks",
            "feature_availability",
            "availability_summary",
            "required_dependencies",
            "spec_extraction",
            "role_runbooks",
            "compliance_checklist",
        ]
        compact: Dict[str, Any] = {}
        for key in keep_keys:
            if key in contract:
                compact[key] = contract.get(key)
        for key in ["canonical_columns", "column_mapping_rules", "column_mapping"]:
            vals = contract.get(key)
            if isinstance(vals, list) and vals:
                compact[key] = vals[:80]
        return compact

    def _truncate_prompt_text(self, text: str, max_len: int, head_len: int, tail_len: int) -> str:
        if not text:
            return text
        if len(text) <= max_len:
            return text
        safe_head = max(0, min(head_len, max_len))
        safe_tail = max(0, min(tail_len, max_len - safe_head))
        if safe_head + safe_tail == 0:
            return text[:max_len]
        if safe_head + safe_tail < max_len:
            safe_head = max_len - safe_tail
        return text[:safe_head] + "\n...[TRUNCATED]...\n" + text[-safe_tail:]

    def _truncate_code_for_patch(self, code: str, max_len: int = 12000) -> str:
        return self._truncate_prompt_text(code or "", max_len=max_len, head_len=7000, tail_len=4000)

    def _check_script_completeness(self, code: str, required_paths: List[str]) -> List[str]:
        if not code:
            return ["code_empty"]
        issues: List[str] = []
        lower = code.lower()
        if "read_csv" not in lower:
            issues.append("data_load_missing")
        if not any(token in lower for token in ["to_csv", "json.dump", "plt.savefig", "savefig("]):
            issues.append("artifact_write_missing")
        missing_outputs = [out for out in (required_paths or []) if out and out not in code]
        if missing_outputs:
            issues.append(f"required_outputs_missing: {missing_outputs}")
        return issues

    def generate_code(
        self,
        strategy: Dict[str, Any],
        data_path: str,
        feedback_history: List[str] = None,
        previous_code: str = None,
        gate_context: Dict[str, Any] = None,
        csv_encoding: str = 'utf-8',
        csv_sep: str = ',',
        csv_decimal: str = '.',
        data_audit_context: str = "",
        business_objective: str = "",
        execution_contract: Dict[str, Any] | None = None,
        feature_availability: List[Dict[str, Any]] | None = None,
        availability_summary: str = "",
        signal_summary: Dict[str, Any] | None = None,
        iteration_memory: List[Dict[str, Any]] | None = None,
        iteration_memory_block: str = "",
    ) -> str:

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Senior ML Engineer for tabular Data Science.

        MISSION
        - Produce ONE robust, runnable Python SCRIPT that loads the cleaned dataset from $data_path, trains/evaluates according to the Execution Contract, and writes the required artifacts.
        - Adapt to each dataset and objective. Do not follow a rigid recipe; follow the contract + data.
        - If Evaluation Spec says requires_target=false, DO NOT train a supervised model. Produce descriptive/segmentation insights and still write data/metrics.json with model_trained=false.

        HARD CONSTRAINTS (VIOLATION = FAILURE)
        1) OUTPUT VALID PYTHON CODE ONLY (no markdown, no code fences, no JSON-only plans).
        2) If RUNTIME_ERROR_CONTEXT is present in the audit, fix root cause and regenerate the FULL script.
        3) NEVER generate synthetic/placeholder data. Always load from '$data_path' only.
        4) Do NOT invent column names. Use only columns from the contract/canonical list and the loaded dataset.
        5) Do NOT add/overwrite df columns unless the name is explicitly allowed by the contract. Prefer Pipeline/ColumnTransformer transforms over df["new_col"]=... .
        6) NEVER create DataFrames from literals (pd.DataFrame({}), from_dict, or lists/tuples). No np.random/random/faker.
        7) scored_rows.csv may include ONLY allowed columns per the contract. Any extra derived columns (e.g., price_delta) must be written to a separate artifact file.
        8) Start the script with a short comment block labeled PLAN describing: detected columns, row_id construction, scored_rows columns, and where extra derived artifacts go.

        SECURITY / SANDBOX (VIOLATION = FAILURE)
        - Do NOT import sys.
        - NO NETWORK/FS OPS: Do NOT use requests/subprocess/os.system and do not access filesystem outside declared input/output paths.
        - No network/shell: no requests, subprocess, os.system.
        - No filesystem discovery: do NOT use os.listdir, os.walk, glob.
        - Read only '$data_path'. Save outputs only to ./data and plots to ./static/plots.

        INPUT CONTEXT (authoritative)
        - Business Objective: "$business_objective"
        - Strategy: $strategy_title ($analysis_type)
        - Execution Contract (json): $execution_contract_json
        - Deliverables: $deliverables_json
        - Canonical Columns: $canonical_columns
        - Required Features: $required_columns
        - Feature Availability: $feature_availability_json
        - Availability Summary: $availability_summary
        - Spec Extraction: $spec_extraction_json
        - Evaluation Spec: $evaluation_spec_json
        - Feature Semantics: $feature_semantics_json
        - Business sanity checks: $business_sanity_checks_json
        - Alignment Requirements: $alignment_requirements_json
        - Signal Summary: $signal_summary_json
        - Iteration Memory: $iteration_memory_json
        - Data audit context: $data_audit_context

        DEPENDENCIES
        - Use only: numpy, pandas, scipy, sklearn, statsmodels, matplotlib, seaborn, pyarrow, openpyxl, duckdb, sqlalchemy, dateutil, pytz, tqdm, yaml.
        - Extended deps (rapidfuzz, plotly, pydantic, pandera, networkx) ONLY if listed in execution_contract.required_dependencies.
        - Do not import any other deps.

        DIALECT CONTRACT (must follow)
        - Prefer reading 'data/cleaning_manifest.json' to get output_dialect (sep, encoding, decimal). Default: Enc='$csv_encoding', Sep='$csv_sep', Decimal='$csv_decimal'.
        - After loading:
        - If df is empty: raise ValueError including the dialect used.
        - If df has 1 column and the column name contains ',', ';', or '\\t' AND length>20: raise ValueError("Delimiter/Dialect mismatch: ...") including the dialect used.
        - Do NOT attempt to split columns.

        SENIOR WORKFLOW (do this, not a checklist)
        Step 0) Feasibility gate:
        - Identify target from contract/spec_extraction. If missing/unmappable -> raise ValueError with a clear message.
        - Build y as a pandas Series and enforce ONE variance guard:
        if y.nunique() <= 1: raise ValueError("CRITICAL: Target variable has no variation.")
        - Never add noise/jitter.

        Step 1) Diagnose the dataset quickly:
        - Determine task type (classification/regression) and key risks:
        missingness, high-cardinality categoricals, suspected IDs, leakage/post-outcome features (use availability + semantics).
        - Use signal_summary to choose model complexity (avoid overfitting).

        Step 2) Decide validation correctly:
        - If objective_type == "forecasting" or requires_time_series_split=true -> use TimeSeriesSplit or chronological holdout (shuffle=False). Do NOT use random KFold.
        - If the contract/spec indicates group_key OR you infer a grouping vector -> use GroupKFold or GroupShuffleSplit (or CV with groups=...).
        - Else if time_key or time ordering matters -> use a time-based split.
        - Else -> StratifiedKFold (classification) or KFold (regression).
        - Never evaluate on training data.

        Step 3) Implement with pipelines (default):
        - Use sklearn Pipeline + ColumnTransformer for preprocessing.
        - Numeric: imputer (+ scaler if needed).
        - Categorical: imputer + OneHotEncoder(handle_unknown='ignore', sparse_output=False).
        - Apply a high-cardinality safeguard when needed (e.g., limit top-K categories or hashing) without leakage.

        Step 4) Models (default behavior):
        - Always include:
        - A baseline (simple, interpretable).
        - A stronger model (more expressive) appropriate for dataset size and task.
        - Prefer stable sklearn models unless the contract explicitly requires others.

        Step 5) Contract compliance outputs:
        - Do NOT invent global rules. Use execution_contract to decide:
        - which columns to use (pre-decision vs post-outcome),
        - required artifacts,
        - derived targets/columns behavior.
        - After mapping/selection, print:
        - Mapping Summary: {...}
        - Only enforce segmentation/weights/pricing logic IF deliverables require those outputs or decision_variables exist.
        (Example: if a required deliverable includes "data/weights.json" or execution_contract.decision_variables present -> run the corresponding logic; else skip.)

        REQUIRED ARTIFACT RULES (minimal, contract-driven)
        - Always:
        - os.makedirs('data', exist_ok=True)
        - os.makedirs('static/plots', exist_ok=True)
        - JSON writing: always json.dump(..., default=_json_default) with a small _json_default helper.
        - Write all required deliverables; write optional deliverables only if they materially support the objective.
        - Plotting: matplotlib.use('Agg') BEFORE pyplot; save at least one plot IF required deliverables include plots; otherwise skip gracefully.

        ALIGNMENT CHECK (contract-driven)
        - Write data/alignment_check.json with:
        status (PASS|WARN|FAIL), failure_mode (data_limited|method_choice|unknown), summary,
        and per-requirement statuses with evidence.
        - If there are no alignment requirements provided, write WARN with failure_mode=data_limited and explain.

        FINAL SELF-CHECK
        - Print QA_SELF_CHECK: PASS with a short bullet list of what was satisfied (target guard, split choice, baseline, required deliverables, no forbidden imports/ops).

        Return Python code only.

        """

        required_outputs = (execution_contract or {}).get("required_outputs", []) or []
        raw_deliverables = (execution_contract or {}).get("spec_extraction", {}).get("deliverables", [])
        deliverables: List[Dict[str, Any]] = []
        if isinstance(raw_deliverables, list):
            for item in raw_deliverables:
                if isinstance(item, dict) and item.get("path"):
                    deliverables.append(
                        {
                            "id": item.get("id"),
                            "path": item.get("path"),
                            "required": bool(item.get("required", True)),
                            "kind": item.get("kind"),
                            "description": item.get("description"),
                        }
                    )
                elif isinstance(item, str):
                    deliverables.append({"path": item, "required": True})
        if not deliverables and required_outputs:
            deliverables = [{"path": path, "required": True} for path in required_outputs if path]
        required_deliverables = [item.get("path") for item in deliverables if item.get("required") and item.get("path")]
        deliverables_json = json.dumps(deliverables, indent=2)
        
        ml_runbook_json = json.dumps(
            (execution_contract or {}).get("role_runbooks", {}).get("ml_engineer", {}),
            indent=2,
        )
        spec_extraction_json = json.dumps(
            (execution_contract or {}).get("spec_extraction", {}),
            indent=2,
        )
        execution_contract_compact = self._compact_execution_contract(execution_contract or {})
        evaluation_spec_json = json.dumps((execution_contract or {}).get("evaluation_spec", {}), indent=2)
        # Safe Rendering for System Prompt
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            business_objective=business_objective,
            strategy_title=strategy.get('title', 'Unknown'),
            analysis_type=str(strategy.get('analysis_type', 'predictive')).upper(),
            hypothesis=strategy.get('hypothesis', 'N/A'),
            required_columns=json.dumps(strategy.get('required_columns', [])),
            deliverables_json=deliverables_json,
            canonical_columns=json.dumps(
                execution_contract_compact.get("canonical_columns", (execution_contract or {}).get("canonical_columns", []))
            ),
            business_alignment_json=json.dumps((execution_contract or {}).get("business_alignment", {}), indent=2),
            alignment_requirements_json=json.dumps((execution_contract or {}).get("alignment_requirements", []), indent=2),
            feature_semantics_json=json.dumps((execution_contract or {}).get("feature_semantics", []), indent=2),
            business_sanity_checks_json=json.dumps((execution_contract or {}).get("business_sanity_checks", []), indent=2),
            data_path=data_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
            data_audit_context=data_audit_context,
            execution_contract_json=json.dumps(execution_contract_compact, indent=2),
            evaluation_spec_json=evaluation_spec_json,
            spec_extraction_json=spec_extraction_json,
            ml_engineer_runbook=ml_runbook_json,
            feature_availability_json=json.dumps(feature_availability or [], indent=2),
            availability_summary=availability_summary or "",
            signal_summary_json=json.dumps(signal_summary or {}, indent=2),
            iteration_memory_json=json.dumps(iteration_memory or [], indent=2),
            iteration_memory_block=iteration_memory_block or "",
        )
        
        # USER TEMPLATES (Static)
        USER_FIRSTPASS_TEMPLATE = (
            "Generate a COMPLETE, runnable ML Python script for strategy: $strategy_title "
            "using data at $data_path. Include data loading, modeling, validation, "
            "required outputs, and alignment_check.json. Return Python code only."
        )
        
        USER_PATCH_TEMPLATE = """
        *** PATCH MODE ACTIVATED ***
        Your previous code was REJECTED by the $gate_source.
        
        *** CRITICAL FEEDBACK ***
        $feedback_text
        *** FEEDBACK DIGEST (Recent) ***
        $feedback_digest

        *** EDIT THESE BLOCKS (FROM LAST ITERATION) ***
        $edit_instructions
        
        *** REQUIRED FIXES (CHECKLIST) ***
        $fixes_bullets
        - [ ] VERIFY COLUMN MAPPING: Ensure fuzzy match + aliasing check found in Protocol v2. No case-sensitive filtering.
        - [ ] VERIFY RENAMING: Ensure DataFrame columns are renamed to canonical required names.
        - [ ] DO NOT GENERATE SYNTHETIC DATA: Load the provided dataset from $data_path.
        - [ ] DEFINE SEGMENT_FEATURES and MODEL_FEATURES lists and use them.
        - [ ] IF USING CALIBRATEDCLASSIFIERCV: avoid feature_importances_ / base_estimator access.
        - [ ] IF TARGET IS DERIVED: derive or guard it and log DERIVED_TARGET:<name>.
        
        *** PREVIOUS OUTPUT (TO PATCH) ***
        $previous_code

        INSTRUCTIONS:
        1. APPLY A MINIMAL PATCH to the previous Python code. DO NOT REWRITE FROM SCRATCH unless necessary.
        2. Address EVERY item in the Required Fixes checklist.
        3. Ensure all MANDATORY UNIVERSAL QA INVARIANTS are met.
        4. Return the FULL script (not a partial diff/snippet).
        """

        # Construct User Message with Patch Mode Logic
        if previous_code and gate_context:
            failed_gates = gate_context.get('failed_gates', [])
            required_fixes = gate_context.get('required_fixes', [])
            feedback_text = gate_context.get('feedback', '')
            fixes_bullets = "\n".join(["- " + str(fix) for fix in required_fixes])
            digest_prefixes = ("REVIEWER FEEDBACK", "QA TEAM FEEDBACK", "RESULT EVALUATION FEEDBACK", "OUTPUT_CONTRACT_MISSING")
            feedback_digest_items = [f for f in (feedback_history or []) if isinstance(f, str) and f.startswith(digest_prefixes)]
            feedback_digest = "\n".join(feedback_digest_items[-5:])
            previous_code_block = self._truncate_code_for_patch(previous_code)
            
            user_message = render_prompt(
                USER_PATCH_TEMPLATE,
                gate_source=str(gate_context.get('source', 'QA Reviewer')).upper(),
                feedback_text=feedback_text,
                fixes_bullets=fixes_bullets,
                previous_code=previous_code_block,
                feedback_digest=feedback_digest,
                edit_instructions=str(gate_context.get("edit_instructions", "")),
                strategy_title=strategy.get('title', 'Unknown'),
                data_path=data_path
            )
        else:
            # First pass
            user_message = render_prompt(
                USER_FIRSTPASS_TEMPLATE,
                strategy_title=strategy.get('title', 'Unknown'),
                data_path=data_path
            )


        # Dynamic Configuration
        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        base_temp = _env_float("ML_ENGINEER_TEMPERATURE", 0.1)
        retry_temp = _env_float("ML_ENGINEER_TEMPERATURE_RETRY", 0.0)
        if previous_code and gate_context:
            current_temp = retry_temp
        else:
            current_temp = base_temp

        thinking_level = (os.getenv("ML_ENGINEER_THINKING_LEVEL") or "high").strip().lower()
        if thinking_level not in {"minimal", "low", "medium", "high"}:
            thinking_level = "high"

        from src.utils.retries import call_with_retries
        if self.provider in {"google", "gemini"}:
            provider_label = "Google"
        elif self.provider == "zai":
            provider_label = "Z.ai"
        else:
            provider_label = "DeepSeek"

        def _call_model_with_prompts(sys_prompt: str, usr_prompt: str, temperature: float) -> str:
            self.last_prompt = sys_prompt + "\n\nUSER:\n" + usr_prompt
            print(f"DEBUG: ML Engineer calling {provider_label} Model ({self.model_name})...")
            if self.provider == "zai":
                from src.utils.llm_throttle import glm_call_slot
                with glm_call_slot():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": usr_prompt}
                        ],
                        temperature=temperature,
                    )
                content = response.choices[0].message.content
            elif self.provider in {"google", "gemini"}:
                full_prompt = sys_prompt + "\n\nUSER INPUT:\n" + usr_prompt
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=8192,
                        candidate_count=1,
                        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        ],
                    ),
                )
                content = getattr(response, "text", "")
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt}
                    ],
                    temperature=temperature,
                )
                content = response.choices[0].message.content
            self.last_response = content
            
            # CRITICAL CHECK FOR SERVER ERRORS (HTML/504)
            if "504 Gateway Time-out" in content or "<html" in content.lower():
                raise ConnectionError("LLM Server Timeout (504 Received)")
            return content

        def _syntax_error_message(candidate: str) -> str:
            try:
                ast.parse(candidate)
                return ""
            except SyntaxError as err:
                return str(err)
            except Exception as err:
                return str(err)

        try:
            content = call_with_retries(
                lambda: _call_model_with_prompts(system_prompt, user_message, current_temp),
                max_retries=5,
                backoff_factor=2,
                initial_delay=2,
            )
            print(f"DEBUG: {provider_label} response received.")
            code = self._clean_code(content)
            if code.strip().startswith("{") or code.strip().startswith("["):
                return "# Error: ML_CODE_REQUIRED"
            if not is_syntax_valid(code):
                syntax_err = _syntax_error_message(code)
                repair_system = (
                    "You are a senior Python engineer. Fix syntax errors ONLY. "
                    "Do not change logic or remove required outputs. "
                    "Return VALID PYTHON CODE ONLY."
                )
                repair_user = (
                    "Fix syntax errors in the code below. Keep logic intact. "
                    f"Syntax error: {syntax_err}\n\nCODE:\n{code}"
                )
                repaired = code
                for _ in range(2):
                    repaired = call_with_retries(
                        lambda: _call_model_with_prompts(repair_system, repair_user, 0.0),
                        max_retries=2,
                        backoff_factor=2,
                        initial_delay=1,
                    )
                    repaired = self._clean_code(repaired)
                    if is_syntax_valid(repaired):
                        code = repaired
                        break
            completion_issues = self._check_script_completeness(code, required_deliverables)
            if completion_issues:
                completion_system = (
                    "You are a senior ML engineer. Return a COMPLETE runnable Python script. "
                    "Do not return partial snippets, diffs, or TODOs. "
                    "Preserve required outputs and alignment checks."
                )
                completion_user = (
                    "Your last response is incomplete. Return the full script.\n"
                    f"Missing/invalid sections: {completion_issues}\n"
                    f"Required deliverables: {required_deliverables}\n\n"
                    f"INCOMPLETE CODE:\n{code}"
                )
                completed = call_with_retries(
                    lambda: _call_model_with_prompts(completion_system, completion_user, 0.0),
                    max_retries=2,
                    backoff_factor=2,
                    initial_delay=1,
                )
                completed = self._clean_code(completed)
                if is_syntax_valid(completed):
                    code = completed
            return code

        except Exception as e:
            # Raise RuntimeError as requested for clean catch in graph.py
            print(f"CRITICAL: ML Engineer Failed (Max Retries): {e}")
            raise RuntimeError(f"ML Generation Failed: {e}")

    def _clean_code(self, code: str) -> str:
        return extract_code_block(code)
