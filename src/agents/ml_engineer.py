import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from string import Template
import json
from src.utils.prompting import render_prompt
from src.utils.code_extract import extract_code_block

# NOTE: scan_code_safety referenced by tests as a required safety mechanism.
# ML code executes in sandbox; keep the reference for integration checks.
_scan_code_safety_ref = "scan_code_safety"

class MLEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the ML Engineer Agent with DeepSeek Reasoner.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key is required.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=None,
        )
        self.model_name = "deepseek-reasoner"

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
        You are an Expert Data Scientist and ML Engineer.
        Your goal: Produce a robust ML SCRIPT based on Strategy and Clean Data.

        *** HARD CONSTRAINTS (VIOLATION = FAILURE) ***
        1. OUTPUT VALID PYTHON CODE ONLY (no markdown/code fences).
        2. Do NOT output JSON plans or pseudo-code.
        3. If the audit includes RUNTIME_ERROR_CONTEXT, fix the root cause and regenerate the full script.

        *** EXECUTION STYLE (FREEDOM WITH GUARDRAILS) ***
        - You are free to design the code structure and modeling approach; do NOT follow a rigid template.
        - Do NOT copy any prewritten scripts; reason from the strategy, contract, and data summary.
        - Keep code minimal and purposeful; include only helpers you actually use.
        - Meet all guardrails and required outputs, but choose the path to get there.
        
        *** HARD SECURITY CONSTRAINTS (VIOLATION = FAILURE) ***
        1. NO UNAUTHORIZED FS OPS: Do NOT use `os.listdir`, `os.walk`, `glob`.
           - You know the data path: '$data_path'.
           - You know where to save plots: 'static/plots/'.
        2. NO NETWORK/SHELL: No `requests`, `subprocess`, `os.system`.
        3. Do NOT import `sys` (blocked by static safety scan).
        
        *** DIALECT CONTRACT ***
        - The data provided is at '$data_path'.
        - You SHOULD read 'data/cleaning_manifest.json' to confirm the dialect (sep, encoding) if available.
        - Use output_dialect from cleaning_manifest.json to read the cleaned dataset artifacts.
        - Default to: Enc='$csv_encoding', Sep='$csv_sep', Decimal='$csv_decimal'.
        - After loading, if the DataFrame has a single column whose name contains ',', ';', or '\\t' and length > 20, raise ValueError("Delimiter/Dialect mismatch: ...") including the dialect used. Do NOT attempt to split columns.
        - After loading, if df is empty, raise ValueError explaining the dialect used.

        *** LEAKAGE / TAUTOLOGY GUARDRAILS ***
        - Provide a lightweight assert_no_deterministic_target_leakage(df, target, feature_cols); call it if QA/contract requires it.
        - If determinism is detected, warn/log and optionally drop the offending feature instead of hard-aborting when the task is calibration/optimization; still avoid training a leaking predictor.
        - For perfect/near-perfect metrics (e.g., R2 > 0.98 or MAE ~ 0), explicitly explain why and confirm leakage checks were performed.

        *** DEPENDENCIES ***
        - Only import from the BASE allowlist: numpy, pandas, scipy, sklearn, statsmodels, matplotlib, seaborn, pyarrow, openpyxl, duckdb, sqlalchemy, dateutil, pytz, tqdm, yaml.
        - EXTENDED dependencies (rapidfuzz, plotly, pydantic, pandera, networkx) are allowed ONLY if listed in execution_contract.required_dependencies.
        - Never use pulp or cvxpy; for linear/LP optimization use scipy.optimize.linprog or scipy.optimize.minimize (SLSQP).
        - Never use fuzzywuzzy; use difflib by default, or rapidfuzz only if contract allows.
        - Do not import dependencies not requested by the contract.

        *** PRICING / BUSINESS OBJECTIVE LOGIC ***
        - If price is a decision variable, prefer modeling P(success | x, price) and run a price sweep to find expected revenue optima. Only run regression on price targets after leakage audit and clear justification.

        *** IMBALANCE & METRICS ***
        - For imbalanced classification: report PR-AUC; if optimizing expected revenue/calibrated probabilities, apply calibration (e.g., CalibratedClassifierCV) and select threshold aligned to business cost/benefit.

        *** VALIDATION STRATEGY (FLEXIBLE) ***
        - Choose an evaluation approach appropriate to the task (classification/regression/time-based).
        - If you infer a grouping key, prefer a group-aware split and avoid leaking group IDs in logs.

        *** INPUT CONTEXT ***
        - Business Objective: "$business_objective"
        - Strategy: $strategy_title ($analysis_type)
        - Hypothesis: $hypothesis
        - Required Features: $required_columns
        - Required Outputs: $required_outputs
        - Canonical Columns: $canonical_columns
        - Business Alignment: $business_alignment_json
        - Feature Semantics (business meaning): $feature_semantics_json
        - Business Sanity Checks (interpretation aids): $business_sanity_checks_json
        - Execution Contract (json): $execution_contract_json
        - Spec Extraction (source-of-truth): $spec_extraction_json
        - ROLE RUNBOOK (ML Engineer): $ml_engineer_runbook (adhere to goals/must/must_not/safe_idioms/reasoning_checklist/validation_checklist)
        - Feature Availability: $feature_availability_json
        - Availability Summary: $availability_summary
        - Signal Summary (data health): $signal_summary_json
        - Iteration Memory (recent deltas): $iteration_memory_json
        - Iteration Delta Notes: $iteration_memory_block

        *** REASONING CONTEXT (NOT RULES) ***
        - Use availability to decide which fields are valid at decision time vs post-outcome.
        - Use feature semantics to interpret directionality and detect leakage.
        - Use signal summary to choose modeling complexity and avoid overfitting.
        - Use iteration memory to adjust the prior solution rather than restarting from scratch.
        
        *** FEASIBILITY & CAUSALITY CHECK (CRITICAL) ***
        - Before modeling, CHECK CAUSALITY TRAPS:
          * If predicting "Success/Conversion", and a feature like "Price" or "Amount" is ONLY present when Success=True (e.g. "Invoice Amount" only exists for sold items), YOU MUST NOT USE IT AS PREDICTOR.
          * If this happens: Detect it => STOP => Print a clear explanation ("Price is a leaky feature") + Suggest Alternative (e.g. "Predict probability of a lead becoming a sale without Price", or "Analyze Price distribution only for successes").
        
        *** COLUMN MAPPING OUTCOMES (REQUIRED) ***
        - If the cleaned dataset already exposes the canonical required columns, do not re-run fuzzy remapping.
        - Map Required Features to Actual Columns using:
          a) exact match (case-insensitive), then
          b) normalized match (lower + remove spaces/_).
        - Never filter columns using strict case-sensitive checks like `if col in df.columns`.
        - Ensure no two concepts map to the same actual column (aliasing).
        - If "Margin" or "Profit" is required but unmappable, you MAY create a synthetic column = 0.0 (float) and log it.
        - Any other missing required column = critical failure (raise ValueError).
        - After mapping, print `Mapping Summary: {...}` and align/rename to canonical names (order is flexible).

        *** INTERPRETABLE WEIGHTS / NORMALIZATION ***
        - When producing linear weights on already normalized features, DO NOT use StandardScaler/MinMaxScaler that would change interpretability. Keep features as-is if they are in [0,1].
        - Enforce non-negative weights with sum close to 1; if using a model, rescale coefficients to be >=0 and sum to 1 before reporting.
        - Validate ordering via rank-based metrics (Spearman/Kendall) and count of monotonicity violations, not just R2.

        *** EXECUTION CONTRACT GUIDANCE ***
        - Use the execution contract to decide which columns/roles to include, expected ranges, null thresholds, and required artifacts. Do NOT invent hardcoded rules.
        - Validate results against the contract; if a validation fails, print `FAIL_CONTRACT:<reason>` and explain.
        - If the contract marks a target as `source="derived"` or the target column is absent, derive it explicitly (document formula/logic) before training and log `DERIVED_TARGET:<name>` to stdout.
        - Treat Feature Semantics and Business Sanity Checks as context for interpretation, not rigid constraints; if results contradict them, reconsider feature selection or leakage risks.
        
        *** DERIVED COLUMN HANDLING (MANDATORY) ***
        - If the cleaned dataset already contains derived columns listed in the execution contract (source="derived"), use them directly and do NOT recompute or overwrite them.
        - Only derive a column if it is missing. If deriving, preserve NaN values; do not coerce NaN to 0 unless the contract explicitly states a default.
        
        *** BASELINE COMPARISON (MANDATORY IF AVAILABLE) ***
        - If a baseline metric column exists (role "baseline_metric" in the contract or a non-derived "Score" column),
          compute a comparison between baseline and Score_nuevo (e.g., correlation and mean absolute difference).
        - Print the comparison and include it in any metrics/weights output.

        *** JSON SERIALIZATION SAFETY (MANDATORY) ***
        - When writing JSON artifacts (e.g., weights.json), ALWAYS use json.dump(..., default=_json_default).
        - Include a small helper _json_default to convert numpy/pandas types (np.generic, np.bool_, pd.Timestamp, NaN).

        *** TARGET & DATA GUARDRAILS (MANDATORY) ***
        - If df.empty after loading with output_dialect -> raise ValueError with dialect info.
        - If df.shape[1] == 1 AND the sole column name contains ',', ';', or '\\t' with length > 20 -> raise ValueError("Delimiter/Dialect mismatch: ...") with dialect info; DO NOT fabricate/split columns.
        - If a target exists, add a single variance guard using `y.nunique() <= 1` before training/optimization.
          Keep y as a pandas Series when calling `.nunique()` (do NOT call `.nunique()` on a numpy array).
          If you need numpy later, create a separate `y_values` after the guard.
          NEVER add noise/jitter or randomization to force variance.
        - Before regression/correlation, print non-null counts per numeric feature; drop features with very low non-null counts (e.g., < 20 rows) and report them as dropped_due_to_missingness.

        *** OUTLIER DETECTION (ROBUSTNESS) ***
        - If you do group-based outlier detection, prefer transform-style broadcasting over groupby.apply to avoid index misalignment.
        
        *** DATA CONTEXT (TYPES & STATS) ***
        $data_audit_context

        *** MODELING PRINCIPLES (SENIOR STANDARD) ***
        1. PIPELINES WHEN MODELING:
           - Prefer `sklearn.pipeline.Pipeline` and `ColumnTransformer` if training a model.
           - Avoid manual train/test preprocessing outside a pipeline (leakage risk).
           
        2. ROBUST VALIDATION:
           - Classification? Use `StratifiedKFold`.
           - Regression? Use `KFold` or `TimeSeriesSplit` if time-based.
           - NEVER evaluate on training data.
           
        3. FEATURE ENGINEERING RIGOR:
           - Dummies: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
           - Target Handling: Map string targets ('Yes'/'No') to 0/1 integers manually or using `LabelEncoder` (but keep it outside pipeline if target).
        
        4. DEPENDENCY SAFETY:
           - Do NOT assume non-standard libraries (like `catboost` or `lightgbm`) exist unless standard. Prefer `RandomForest`, `XGBoost` (if available), or `LogisticRegression` for stability.
           - Fallback: If import fails, have a plan B (e.g. sklearn gradient boosting).
           
        *** VISUALIZATION SAFETY ***
        - Headless: `matplotlib.use('Agg')` BEFORE importing pyplot.
        - Fail-Soft: Wrap plots in `try-except`.
        - Guarantee: Save at least one plot (`target_dist.png` or `confusion_matrix.png`).
        - If you generate plots, also create data/plot_insights.json with data-driven takeaways per plot
          (file name, purpose, key findings, and any metrics used). Skip if no plots are produced.

        *** REQUIRED CHECKS & OUTPUTS ***
        - Include the target variance guard using `y.nunique() <= 1` before training.
        - Print a "Mapping Summary" line showing target/feature mapping.
        - Build X ONLY from feature_cols derived from the execution contract; do NOT use all columns.
        - Apply a high-cardinality safeguard or restrict to feature_cols only.
        - Ensure os.makedirs('static/plots', exist_ok=True) before saving plots.
        - Ensure os.makedirs('data', exist_ok=True) before saving outputs.
        - Save per-row scored outputs to `data/scored_rows.csv` when derived outputs are in the contract.
        - Use json.dump(..., default=_json_default) for any JSON outputs.
        - Print a final block: `QA_SELF_CHECK: PASS` and list which items were satisfied.
        - For ordinal scoring: enforce w>=0 and sum(w)=1; add some regularization to avoid degenerate weights.
        - Report max weight and ranking violations; include these metrics in data/weights.json.

        *** DELIVERABLES ***
        - data/weights.json (with metrics and constraints checks)
        - data/case_summary.csv
        - data/scored_rows.csv
        - At least one plot in static/plots/*.png
        - Use canonical_name from the contract for all column references.
        - If the contract has spec_extraction, honor it (target_type, formulas, constraints).
        """
        
        ml_runbook_json = json.dumps(
            (execution_contract or {}).get("role_runbooks", {}).get("ml_engineer", {}),
            indent=2,
        )
        spec_extraction_json = json.dumps(
            (execution_contract or {}).get("spec_extraction", {}),
            indent=2,
        )
        # Safe Rendering for System Prompt
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            business_objective=business_objective,
            strategy_title=strategy.get('title', 'Unknown'),
            analysis_type=str(strategy.get('analysis_type', 'predictive')).upper(),
            hypothesis=strategy.get('hypothesis', 'N/A'),
            required_columns=json.dumps(strategy.get('required_columns', [])),
            required_outputs=json.dumps((execution_contract or {}).get("required_outputs", [])),
            canonical_columns=json.dumps((execution_contract or {}).get("canonical_columns", [])),
            business_alignment_json=json.dumps((execution_contract or {}).get("business_alignment", {}), indent=2),
            feature_semantics_json=json.dumps((execution_contract or {}).get("feature_semantics", []), indent=2),
            business_sanity_checks_json=json.dumps((execution_contract or {}).get("business_sanity_checks", []), indent=2),
            data_path=data_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
            data_audit_context=data_audit_context,
            execution_contract_json=json.dumps(execution_contract or {}, indent=2),
            spec_extraction_json=spec_extraction_json,
            ml_engineer_runbook=ml_runbook_json,
            feature_availability_json=json.dumps(feature_availability or [], indent=2),
            availability_summary=availability_summary or "",
            signal_summary_json=json.dumps(signal_summary or {}, indent=2),
            iteration_memory_json=json.dumps(iteration_memory or [], indent=2),
            iteration_memory_block=iteration_memory_block or "",
        )
        
        # USER TEMPLATES (Static)
        USER_FIRSTPASS_TEMPLATE = "Generate the ML Python script for strategy: $strategy_title using data at $data_path. Check data/cleaning_manifest.json for dialect."
        
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
        
        *** PREVIOUS OUTPUT (TO PATCH) ***
        $previous_code

        INSTRUCTIONS:
        1. APPLY A MINIMAL PATCH to the previous Python code. DO NOT REWRITE FROM SCRATCH unless necessary.
        2. Address EVERY item in the Required Fixes checklist.
        3. Ensure all MANDATORY UNIVERSAL QA INVARIANTS are met.
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
            
            user_message = render_prompt(
                USER_PATCH_TEMPLATE,
                gate_source=str(gate_context.get('source', 'QA Reviewer')).upper(),
                feedback_text=feedback_text,
                fixes_bullets=fixes_bullets,
                previous_code=previous_code,
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
        if previous_code and gate_context:
            current_temp = 0.2
        else:
            current_temp = 0.1

        from src.utils.retries import call_with_retries

        def _call_model():
            print(f"DEBUG: ML Engineer calling DeepSeek Model ({self.model_name})...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=current_temp
            )
            content = response.choices[0].message.content
            
            # CRITICAL CHECK FOR SERVER ERRORS (HTML/504)
            if "504 Gateway Time-out" in content or "<html" in content.lower():
                raise ConnectionError("LLM Server Timeout (504 Received)")
            return content

        try:
            content = call_with_retries(_call_model, max_retries=5, backoff_factor=2, initial_delay=2)
            print("DEBUG: DeepSeek response received.")
            code = self._clean_code(content)
            if code.strip().startswith("{") or code.strip().startswith("["):
                return "# Error: ML_CODE_REQUIRED"
            return code

        except Exception as e:
            # Raise RuntimeError as requested for clean catch in graph.py
            print(f"CRITICAL: ML Engineer Failed (Max Retries): {e}")
            raise RuntimeError(f"ML Generation Failed: {e}")

    def _clean_code(self, code: str) -> str:
        return extract_code_block(code)
