import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from string import Template
import json
from src.utils.prompting import render_prompt
from src.utils.code_extract import extract_code_block
from src.utils.static_safety_scan import scan_code_safety

class MLEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the ML Engineer Agent with DeepSeek (Moonshot AI).
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key is required.")
        
        # Initialize OpenAI Client for DeepSeek (Moonshot)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=None
        )
        self.model_name = "deepseek-reasoner" # Fallback/Standard
        # USER REQUESTED: Switch to 'deepseek-reasoner' due to credit limits.
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
    ) -> str:

        SYSTEM_PROMPT_TEMPLATE = """
        You are an Expert Data Scientist and ML Engineer.
        Your goal: Execute a robust, "bulletproof" analysis based on Strategy and Clean Data.
        
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

        *** VALIDATION STRATEGY ***
        - If a group key can be inferred (e.g., via infer_group_key helper), you MUST use GroupKFold/GroupShuffleSplit with that group vector. Log to stdout: "Using GroupSplit on <group_key> (hashed)" without printing raw IDs. If no group key, derive groups via hashing duplicates of X to avoid leakage.

        *** INPUT CONTEXT ***
        - Business Objective: "$business_objective"
        - Strategy: $strategy_title ($analysis_type)
        - Hypothesis: $hypothesis
        - Required Features: $required_columns
        - Execution Contract (json): $execution_contract_json
        
        *** FEASIBILITY & CAUSALITY CHECK (CRITICAL) ***
        - Before modeling, CHECK CAUSALITY TRAPS:
          * If predicting "Success/Conversion", and a feature like "Price" or "Amount" is ONLY present when Success=True (e.g. "Invoice Amount" only exists for sold items), YOU MUST NOT USE IT AS PREDICTOR.
          * If this happens: Detect it => STOP => Print a clear explanation ("Price is a leaky feature") + Suggest Alternative (e.g. "Predict probability of a lead becoming a sale without Price", or "Analyze Price distribution only for successes").
        
        *** COLUMN MAPPING PROTOCOL v2 (ROBUSTNESS) ***
        1. MAPPING STRATEGY:
           - If the cleaned dataset already exposes the canonical required columns, DO NOT re-run fuzzy remapping; use them directly.
           - You MUST map Required Features to Actual Columns using this prioritization:
             a) Exact Match (Case-Insensitive).
             b) Fuzzy Match (Normalized: lower + remove spaces/_).
           - NEVER filter columns using strict case-sensitive checks like `if col in df.columns`.
        
        2. VALIDATION & ALIASING:
           - Ensure no two concepts map to the same actual column (Aliasing).
           - If "Margin" or "Profit" is required but significantly missing/unmappable, YOU MAY create a Synthetic Column = 0.0 (float), but must log this in the summary.
           - Other missing required columns = CRITICAL FAILURE (Raise ValueError).

        3. EXECUTION ORDER:
           - Step A: Build Mapping Dict.
           - Step B: Print `Mapping Summary: {...}`.
           - Step C: Select columns `df = df[[actual_col1, actual_col2...]]`.
           - Step D: Rename to Canonical Names `df.columns = [required_name1, required_name2...]`.

        *** INTERPRETABLE WEIGHTS / NORMALIZATION ***
        - When producing linear weights on already normalized features, DO NOT use StandardScaler/MinMaxScaler that would change interpretability. Keep features as-is if they are in [0,1].
        - Enforce non-negative weights with sum close to 1; if using a model, rescale coefficients to be >=0 and sum to 1 before reporting.
        - Validate ordering via rank-based metrics (Spearman/Kendall) and count of monotonicity violations, not just R2.

        *** EXECUTION CONTRACT GUIDANCE ***
        - Use the execution contract to decide which columns/roles to include, expected ranges, null thresholds, and required artifacts. Do NOT invent hardcoded rules.
        - Validate results against the contract; if a validation fails, print `FAIL_CONTRACT:<reason>` and explain.
        - If the contract marks a target as `source="derived"` or the target column is absent, derive it explicitly (document formula/logic) before training and log `DERIVED_TARGET:<name>` to stdout.

        *** TARGET & DATA GUARDRAILS (MANDATORY) ***
        - If df.empty after loading with output_dialect -> raise ValueError with dialect info.
        - If df.shape[1] == 1 AND the sole column name contains ',', ';', or '\\t' with length > 20 -> raise ValueError("Delimiter/Dialect mismatch: ...") with dialect info; DO NOT fabricate/split columns.
        - If a target exists, add EXACTLY one guard: `if y.nunique() <= 1: raise ValueError("Target has no variance; cannot train meaningful model.")` (or `< 2`) before training/optimization. NEVER add noise/jitter or randomization to force variance.
        - Before regression/correlation, print non-null counts per numeric feature; drop features with very low non-null counts (e.g., < 20 rows) and report them as dropped_due_to_missingness.

        *** OUTLIER DETECTION (ROBUSTNESS) ***
        - When detecting outliers by group:
          * DO NOT use `groupby().apply(...)` to return a mask, as this often returns a nested object/incompatible index causing ValueError.
          * USE `groupby().transform(...)` to broadcast metrics (Q1, Q3, Mean, Std) to the original shape.
          * Example (Safe):
             `g = df.groupby('col')[target]`
             `q1 = g.transform(lambda x: x.quantile(0.25))`
             `mask = (df[target] < q1 - 1.5*iqr)`
          * This ensures 1:1 alignment with the DataFrame.
        
        *** DATA CONTEXT (TYPES & STATS) ***
        $data_audit_context

        *** MODELING PRINCIPLES (SENIOR STANDARD) ***
        1. PIPELINES ARE MANDATORY:
           - Use `sklearn.pipeline.Pipeline` and `ColumnTransformer`.
           - NEVER manually process training and test sets separately without a pipeline (avoids leakage).
           - `Imputer` -> `Scaler`/`Encoder` -> `Model`.
           
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

        *** MANDATORY QA CHECKLIST (IMPLEMENT IN CODE) ***
        - Add a target variance guard: `if y.nunique() <= 1: raise ValueError("Target has no variance; cannot train meaningful model.")` (or `<2`).
        - Print a "Mapping Summary" line showing target/feature mapping.
        - Build X ONLY from feature_cols derived from the execution contract; do NOT use all columns.
        - Implement a high-cardinality safeguard: either (a) compute/drop high-card columns via nunique/len threshold, or (b) use only feature_cols and ignore extra columns.
        - Ensure os.makedirs('static/plots', exist_ok=True) before saving plots.
        - Ensure os.makedirs('data', exist_ok=True) before saving outputs.
        - Print a final block: `QA_SELF_CHECK: PASS` and list which checklist items were satisfied.
        - For ordinal scoring: optimize a ranking-aware loss at case level, and add regularization to avoid degenerate weights (e.g., L2 on weights and/or max-weight penalty). Enforce w>=0 and sum(w)=1.
        - Report HHI/entropy, max weight, near-zero weights, and ranking violations in stdout; save these metrics into data/weights.json.

        *** OUTPUT REQUIREMENTS ***
        - Valid Python Code ONLY.
        - Save model (optional, pickle).
        - Save metrics (print to stdout).
        - Save plots to `static/plots/`.
        """
        
        # Safe Rendering for System Prompt
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            business_objective=business_objective,
            strategy_title=strategy.get('title', 'Unknown'),
            analysis_type=str(strategy.get('analysis_type', 'predictive')).upper(),
            hypothesis=strategy.get('hypothesis', 'N/A'),
            required_columns=json.dumps(strategy.get('required_columns', [])),
            data_path=data_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
            data_audit_context=data_audit_context,
            execution_contract_json=json.dumps(execution_contract or {}, indent=2)
        )
        
        # USER TEMPLATES (Static)
        USER_FIRSTPASS_TEMPLATE = "Generate code for strategy: $strategy_title using data at $data_path. Check data/cleaning_manifest.json for dialect."
        
        USER_PATCH_TEMPLATE = """
        *** PATCH MODE ACTIVATED ***
        Your previous code was REJECTED by the $gate_source.
        
        *** CRITICAL FEEDBACK ***
        $feedback_text
        *** FEEDBACK DIGEST (Recent) ***
        $feedback_digest
        
        *** REQUIRED FIXES (CHECKLIST) ***
        $fixes_bullets
        - [ ] VERIFY COLUMN MAPPING: Ensure fuzzy match + aliasing check found in Protocol v2. No case-sensitive filtering.
        - [ ] VERIFY RENAMING: Ensure DataFrame columns are renamed to canonical required names.
        
        *** PREVIOUS CODE (TO PATCH) ***
        ```python
        $previous_code
        ```
        
        INSTRUCTIONS:
        1. APPLY A MINIMAL PATCH to the previous code. DO NOT REWRITE FROM SCRATCH unless necessary.
        2. Address EVERY item in the Required Fixes checklist.
        3. Ensure all MANDATORY UNIVERSAL QA INVARIANTS are met.
        4. At the end of the script, add a comment block:
           # QA FIX CHECKLIST:
           # [x] Fix 1...
           # [x] Fix 2...
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
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=current_temp
                )
            except Exception as e:
                # Fallback Logic
                error_str = str(e).lower()
                if "not found" in error_str or "model" in error_str:
                     print(f"WARNING: Model {self.model_name} failed ({e}). Attempting FALLBACK to 'deepseek-chat'.")
                     self.model_name = "deepseek-chat" # Updates for subsequent calls too
                     response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=current_temp
                    )
                     print("FALLBACK MODEL USED: deepseek-chat")
                else:
                    raise e
            
            content = response.choices[0].message.content
            
            # CRITICAL CHECK FOR SERVER ERRORS (HTML/504)
            if "504 Gateway Time-out" in content or "<html" in content.lower():
                raise ConnectionError("DeepSeek Server Timeout (504 Recieved)")
            return content

        try:
            content = call_with_retries(_call_model, max_retries=3)
            print("DEBUG: DeepSeek response received.")
            
            code = self._clean_code(content)
            
            # STATIC SAFETY SCAN (Security)
            is_safe, violations = scan_code_safety(code)
            if not is_safe:
                error_msg = f"Security Check Failed. Violations: {violations}"
                print(f"CRITICAL: {error_msg}")
                import json as _json
                msg = "GENERATED CODE BLOCKED BY STATIC SCAN: " + _json.dumps(violations)
                return f"raise ValueError({msg!r})"
            
            return code

        except Exception as e:
            # Raise RuntimeError as requested for clean catch in graph.py
            print(f"CRITICAL: ML Engineer Failed (Max Retries): {e}")
            raise RuntimeError(f"ML Generation Failed: {e}")

    def _clean_code(self, code: str) -> str:
        return extract_code_block(code)
