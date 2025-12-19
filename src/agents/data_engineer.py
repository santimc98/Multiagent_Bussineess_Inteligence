import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class DataEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Data Engineer Agent with DeepSeek (Moonshot AI).
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
        self.model_name = "deepseek-reasoner" # Standard model ID. 'kimi-k2-thinking' might be non-standard, falling back to v1-8k if needed, or respecting user Input if strictly required. 
        # USER REQUESTED: Switch to 'deepseek-reasoner' due to credit limits.
        self.model_name = "deepseek-reasoner"

    def generate_cleaning_script(
        self,
        data_audit: str,
        strategy: Dict[str, Any],
        input_path: str,
        business_objective: str = "",
        csv_encoding: str = "utf-8",
        csv_sep: str = ",",
        csv_decimal: str = ".",
        execution_contract: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generates a Python script to clean and standardize the dataset.
        """
        from src.utils.prompting import render_prompt
        from src.utils.static_safety_scan import scan_code_safety
        import json

        contract_json = json.dumps(execution_contract or {}, indent=2)
        de_runbook_json = json.dumps(
            (execution_contract or {}).get("role_runbooks", {}).get("data_engineer", {}),
            indent=2,
        )
        
        # SYSTEM TEMPLATE (Static, Safe, No F-Strings)
        SYSTEM_TEMPLATE = """
        You are a Senior Data Engineer. Produce a robust cleaned dataset for downstream ML.
        
        *** HARD CONSTRAINTS (VIOLATION = FAILURE) ***
        1. NO PRIVATE APIS: Prohibited `pd.io.*`, `pandas.io.*`, `ParserBase`. Use public API `pd.read_csv` only.
        2. NO NETWORK/FS OPS: Prohibited `os.system`, `subprocess`, `requests`.
        3. OUTPUT: Valid Python Code ONLY. No markdown.
        4. Do NOT import `sys`.
        
        *** INPUT PARAMETERS ***
        - Input: '$input_path'
        - Encoding: '$csv_encoding' | Sep: '$csv_sep' | Decimal: '$csv_decimal'
        - Business Objective: "$business_objective"
        - Required Columns (Strategy): $required_columns
        - Execution Contract (json): $execution_contract_json
        - ROLE RUNBOOK (Data Engineer): $data_engineer_runbook (you MUST adhere to goals/must/must_not/safe_idioms/reasoning_checklist/validation_checklist)

        *** DATA AUDIT ***
        $data_audit
        
        *** ENGINEERING PRINCIPLES (SENIOR STANDARD) ***

        *** EXECUTION STYLE (FREEDOM WITH GUARDRAILS) ***
        - You are free to design the code structure and logic; do NOT follow a rigid template.
        - Do NOT copy any prewritten scripts; reason from the data audit + execution contract.
        - Prioritize correctness, clarity, and meeting contract requirements over boilerplate.
        - Keep the script minimal: only include helpers you actually use.
        
        1. UNIVERSAL SUFFICIENCY:
           - PRESERVE EVERYTHING by default. Filter logic:
             * KEEP: Required columns (Strategy).
             * KEEP: Any column that looks like signal (even if not required).
             * DROP: Only if 100% Null or Constant (nunique<=1).
           - "Universal" means the output CSV should support MULTIPLE possible strategies, not just the current one.
           - Do NOT drop required/derived columns solely because they are constant; keep and record a "constant_column" warning in the manifest instead.
        
        2. ROBUST LOADING & NAMING:
           - Try/Except load (Smart Metadata -> Fallback Python Engine -> Latin1).
           - The FIRST pd.read_csv MUST include sep, decimal, encoding from the provided dialect variables (do NOT hardcode literal values).
           - COLUMN NAMING:
             * Normalize to `snake_case`.
             * DEDUPLICATION (CRITICAL):
               * Implement deterministic deduplication using a counter and record the mapping.
             * Empty names -> `unknown_col_<n>` (Use a counter, do NOT use braces in string).
        
        3. SEMANTIC TYPE INTELLIGENCE:
           - Money/Percentage: Clean regex -> Float.
           - Dates: `pd.to_datetime(..., errors='coerce')`.
             * If a column has few unique values and patterns like "0-7", "-7-0", "-15--7", treat it as categorical string, NOT as a date.
             * Do NOT use short substrings like "fe"/"fv" for date detection; if used, require exact normalized match or clear patterns ("fecha", "date", "venc", "emision", "obs").
           - Categories: `str.strip().str.title()`.
           - Booleans: Map {yes, y, true, 1} -> True.
           - Missing semantics: Use `is_effectively_missing` (None/NaN/''/whitespace) when deciding emptiness; NEVER treat 0 or "0" as missing.
           - Numeric/Currency parsing: prefer shared helper `safe_convert_numeric_currency` from `src.utils.type_inference` to ensure consistent audit signals.
           - Parsing hygiene (MANDATORY):
             * Do NOT blindly strip '.' characters; infer decimal/thousands from patterns (only '.' present -> decimal='.'; only ',' -> decimal=','; if both, decide by rightmost separator).
             * For percentages: always strip '%' and parse using the detected decimal separator.
             * If a conversion is reverted, restore exactly the original series (no downstream overwrites or partial coercion).
           - Percentage role handling (MANDATORY):
             * If role==percentage in the contract, parse with the detected decimal separator and normalize to 0-1 when most values are in [1,100] or p50 > 1.
             * Keep both raw and normalized versions only if explicitly required by the contract; otherwise overwrite with normalized values.
           - Contract type enforcement (MANDATORY):
             * For each required column, respect expected_kind from the contract: numeric -> pd.to_numeric, datetime -> pd.to_datetime, categorical -> keep as string/categorical.
             * If coercion produces too many NaN (e.g., >80%), try the next candidate mapping; if none, raise a clear ValueError.
             * Ensure required columns with expected_kind==numeric do NOT end up as datetime dtype.
        
        4. MANIFEST ARTIFACT (AUDITABILITY):
           - In addition to 'data/cleaned_data.csv', you MUST save 'data/cleaning_manifest.json'.
           - Do NOT create downstream ML outputs (weights.json, case_summary.csv); only the cleaned CSV + manifest.
           - Structure:
             {
               "input_dialect": {"sep": "$csv_sep", "decimal": "$csv_decimal", "encoding": "$csv_encoding"},
               "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
               "original_columns": ["Raw Name 1", ...],
               "column_mapping": {"Raw Name 1": "raw_name_1"},
               "dropped_columns": [{"name": "col_x", "reason": "empty"}],
               "conversions": {"price": "clean_currency"},
               "conversions_meta": {"price": {"parse_success_rate": 0.97, "reverted": false, "reason": ""}},
               "type_checks": [{"column": "col", "expected_kind": "numeric", "observed_dtype": "float64", "coercion_applied": "to_numeric", "na_frac_after": 0.01}],
               "rows_before": 1000,
            "rows_after": 950
           }
           - MANIFEST JSON SAFETY (MANDATORY):
             * Define helper:
                 def _json_default(o):
                     import numpy as _np, pandas as _pd
                     if isinstance(o, _np.generic):
                         return o.item()
                     if isinstance(o, (_pd.Timestamp,)):
                         return o.isoformat()
                     if hasattr(_pd, "Timedelta") and isinstance(o, _pd.Timedelta):
                         return o.total_seconds()
                     if isinstance(o, (set, tuple)):
                         return list(o)
                     if o is None:
                         return None
                     try:
                         from math import isnan
                         if isinstance(o, float) and (o != o or isnan(o)):
                             return None
                     except Exception:
                         pass
                     try:
                         if getattr(_pd, "isna", None) and _pd.isna(o):
                             return None
                     except Exception:
                         pass
                     return str(o)
             * Save manifest with: json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)

        *** NUMERIC CURRENCY INFERENCE (STRICT) ***
        - Only classify a column as numeric_currency if a high fraction of non-null values contain digits AND a sampled parse_success_rate >= 0.9 (sample up to 200 non-null values).
        - parse_success_rate = non_null_after_parse / non_null_before on the sample; include this in manifest under conversions_meta.
        - Use a robust localized number parser (handle EU/US thousands/decimal) and document decimal_hint/thousands_hint in conversions_meta.
        - If conversion drops non_null_after to 0 or below a post_drop_threshold (e.g., 0.9 of before), REVERT to text/categorical, keep original values, and log conversion_reverted_due_to_low_parse_rate in manifest.
        - conversions_meta MUST include for every numeric conversion: parse_success_rate, digits_ratio, decimal_hint, thousands_hint, sample_size, reverted (bool), and reason.
        - NEVER drop a column solely because a numeric conversion produced 100% nulls; revert instead. Always record the reversion in conversions_meta.

        *** EXECUTION CONTRACT ENFORCEMENT ***
        - Read the Execution Contract JSON provided. Use it to decide expected ranges, null thresholds, and outputs. Do NOT invent requirements beyond the contract/strategy.
        - Apply transformations to meet expected ranges/types; if you cannot, document in manifest why (conversions_meta or warnings).
        - Print a block "CLEANING_VALIDATION" summarizing checks vs contract (null_frac, ranges) using actual computed stats.
        - Validation must be TYPE-SAFE: if expected_range is null or contains null -> do not compare numerically; just report null_frac, dtype, and safe min/max display. If dtype is datetime, do not run numeric range checks; optionally report min_date/max_date. For bool/categorical/object, do not use numeric min/max; you may attempt pd.to_numeric(errors='coerce') only for diagnostics and if conversion_rate is low, log COERCION_LOW warning. Never raise; if validation cannot run, print SKIPPED_VALIDATION:<reason> and continue.
        - For datetime columns: only validate non-nullness and display min_date/max_date. For open-ended ranges (None on min or max), treat as unbounded. If types are incompatible, skip and warn without raising.
        - Validation must never raise exceptions; always catch and log SKIPPED_VALIDATION with the reason.
        - Validation results rows MUST have a stable schema for ALL rows (even missing columns). Never index dict keys that may be missing; use .get() with defaults.
        - When checking if a required column exists, match via normalization: lowercase + remove all non-alphanumeric (so ImporteNorm, Importe_Norm, importe norm match).
        - Normalize column names for matching using a case/spacing-insensitive method.
        - When a required column is missing, emit a validation row with "MISSING" status and null-safe fields.
        - When printing validation rows, use .get() with defaults to avoid KeyError and format nulls safely.
             
        *** DERIVED COLUMNS (MANDATORY) ***
        - When deriving contract columns (e.g., Case, RefScore, Score_nuevo), NEVER hardcode raw column names.
        - Build a map of normalized names -> actual column names after canonicalization and use it to access source columns.
        - If a required source column is missing, raise a clear ValueError instead of defaulting all rows.
        - Do NOT validate required column presence before canonicalization; check after normalization mapping.
        - Only enforce existence for source="input" columns. For source="derived", derive after mapping.
        - When printing validation summaries, guard None values:
          actual = str(result.get('actual_column') or 'MISSING') before slicing/formatting.
        - When reading dtype for validation, guard duplicate column names:
          series = df[actual_col]; if isinstance(series, pd.DataFrame), use series = series.iloc[:, 0] and log a warning.

        *** OUTPUT REQUIREMENTS ***
        - Save `data/cleaned_data.csv` and `data/cleaning_manifest.json`.
        - Print "CLEANING_VALIDATION" and "CLEANING_SUCCESS".
        """
        
        # USER TEMPLATE (Static)
        USER_TEMPLATE = "Generate the cleaning script following Principles."
        
        # Rendering
        system_prompt = render_prompt(
            SYSTEM_TEMPLATE,
            input_path=input_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
            business_objective=business_objective,
            required_columns=str(strategy.get('required_columns', [])),
            data_audit=data_audit,
            execution_contract_json=contract_json,
            data_engineer_runbook=de_runbook_json,
        )

        from src.utils.retries import call_with_retries

        def _call_model():
            print(f"DEBUG: Data Engineer calling DeepSeek Model ({self.model_name})...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": USER_TEMPLATE}
                    ],
                    temperature=0.1
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
                            {"role": "user", "content": USER_TEMPLATE}
                        ],
                        temperature=0.1
                    )
                     print("FALLBACK MODEL USED: deepseek-chat")
                else:
                    raise e
                    
            content = response.choices[0].message.content
            
             # CRITICAL CHECK FOR SERVER ERRORS (HTML/504)
            if "504 Gateway Time-out" in content or "<html" in content.lower():
                raise ConnectionError("DeepSeek Server Timeout (504 Recieved)")

            # Check for JSON error messages that are NOT valid code
            content_stripped = content.strip()
            if content_stripped.startswith("{") or content_stripped.startswith("["):
                try:
                    import json
                    json_content = json.loads(content_stripped)
                    if isinstance(json_content, dict):
                        # Standard error formats
                        if "error" in json_content or "errorMessage" in json_content:
                            raise ConnectionError(f"API Error Detected (JSON): {content_stripped}")
                except Exception:
                    pass # Not valid JSON, proceed
            
            # Text based fallback for Error/Overloaded keywords
            content_lower = content.lower()
            if "error" in content_lower and ("overloaded" in content_lower or "rate limit" in content_lower or "429" in content_lower):
                 raise ConnectionError(f"API Error Detected (Text): {content_stripped}")
            
            return content

        try:
            content = call_with_retries(_call_model, max_retries=3)
            print("DEBUG: DeepSeek response received.")
            
            code = self._clean_code(content)
            
            # STATIC SAFETY SCAN
            is_safe, violations = scan_code_safety(code)
            if not is_safe:
                error_msg = f"Security Check Failed. Violations: {violations}"
                print(f"CRITICAL: {error_msg}")
                # Return a script that raises the error so the Reviewer catches it
                return f"raise ValueError('GENERATED CODE BLOCKED BY STATIC SCAN: {violations}')"
                
            return code
            
        except Exception as e:
            error_msg = f"Data Engineer Failed (Max Retries): {str(e)}"
            print(f"CRITICAL: {error_msg}")
            # Return sentinel for graph.py to handle cleanly
            return f"# Error: {error_msg}"

    def _clean_code(self, code: str) -> str:
        """
        Removes markdown formatting and artifacts from the generated code.
        """
        code = re.sub(r'```python', '', code)
        code = re.sub(r'```', '', code)
        return code.strip()
