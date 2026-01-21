import os
import re
import ast
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from src.utils.static_safety_scan import scan_code_safety
from src.utils.code_extract import extract_code_block
from src.utils.senior_protocol import SENIOR_ENGINEERING_PROTOCOL
from src.utils.contract_v41 import get_cleaning_gates
from openai import OpenAI

load_dotenv()


class DataEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Data Engineer Agent with DeepSeek Reasoner (primary) and OpenRouter fallback.
        """
        # --- PRIMARY: DeepSeek ---
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key is required.")

        # Configurable base_url and timeout
        # DeepSeek Reasoner is slow by design (chain-of-thought reasoning)
        # Data cleaning scripts are complex: 300s (5 min) provides adequate margin
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        deepseek_timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "300"))

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=deepseek_base_url,
            timeout=deepseek_timeout,
        )
        # Configurable model name
        self.model_name = os.getenv("DEEPSEEK_DE_PRIMARY_MODEL", "deepseek-reasoner")

        # --- FALLBACK: OpenRouter ---
        self.fallback_client = None
        self.fallback_model_name = None
        or_key = os.getenv("OPENROUTER_API_KEY")
        if or_key:
            fallback_timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120"))
            self.fallback_client = OpenAI(
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=fallback_timeout,
                default_headers={"HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "")},
            )
            self.fallback_model_name = os.getenv("OPENROUTER_DE_FALLBACK_MODEL") or os.getenv("OPENROUTER_ML_PRIMARY_MODEL") or "z-ai/glm-4.7"

        self.last_prompt = None
        self.last_response = None

    def _extract_nonempty(self, response) -> str:
        """
        Extracts non-empty content from LLM response.
        Raises ValueError("EMPTY_COMPLETION") if content is empty (CAUSA RAÍZ 2).
        This triggers retry logic in call_with_retries.
        """
        msg = response.choices[0].message
        content = (msg.content or "").strip()

        # Fallback: some models expose reasoning separately
        if not content:
            reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning.strip()

        if not content:
            print("ERROR: LLM returned EMPTY_COMPLETION. Will retry.")
            raise ValueError("EMPTY_COMPLETION")

        return content

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
        contract_min: Optional[Dict[str, Any]] = None,
        de_view: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generates a Python script to clean and standardize the dataset.
        """
        from src.utils.prompting import render_prompt

        contract = contract_min or execution_contract or {}
        from src.utils.context_pack import compress_long_lists, summarize_long_list, COLUMN_LIST_POINTER

        contract_json = json.dumps(compress_long_lists(contract)[0], indent=2)
        de_view = de_view or {}
        de_view_json = json.dumps(compress_long_lists(de_view)[0], indent=2)
        cleaning_gates = get_cleaning_gates(contract) or get_cleaning_gates(execution_contract or {}) or []
        cleaning_gates_json = json.dumps(compress_long_lists(cleaning_gates)[0], indent=2)

        # --- FIX CAUSA RAÍZ 1: Leer runbook correcto V4.1 ---
        # Primero intentar clave canónica V4.1, luego fallback a legacy
        de_runbook = contract.get("data_engineer_runbook")
        if not de_runbook:
            de_runbook = (contract.get("role_runbooks") or {}).get("data_engineer", {})
        de_runbook_json = json.dumps(compress_long_lists(de_runbook)[0], indent=2)

        # SYSTEM TEMPLATE with PYTHON SYNTAX GOTCHAS (Fix CAUSA RAÍZ 3)
        SYSTEM_TEMPLATE = """
        You are a Senior Data Engineer. Produce a robust cleaning SCRIPT for downstream ML.

        === SENIOR ENGINEERING PROTOCOL ===
        $senior_engineering_protocol
        
        *** HARD CONSTRAINTS (VIOLATION = FAILURE) ***
        1. OUTPUT VALID PYTHON CODE ONLY (no markdown/code fences).
        2. Do NOT output JSON plans or pseudo-code.
        3. NO NETWORK/FS OPS: Do NOT use requests/subprocess/os.system and do not access filesystem outside declared input/output paths.
        4. NO SYS MODULE: Do NOT use 'import sys' or 'sys.exit()'. Your code runs in a controlled sandbox, not as a standalone script.
           - For error handling, use: raise ValueError("descriptive error message") or print("ERROR: ...") and return early.
           - Example (WRONG): sys.exit(1)
           - Example (CORRECT): raise FileNotFoundError(f"Input file '{input_path}' not found")
        5. BAN pandas private APIs: do not use pandas.io.* or pd.io.parsers.*.
        6. If the audit includes RUNTIME_ERROR_CONTEXT, fix the root cause and regenerate the full script.
        7. Do NOT use np.bool (deprecated). Use bool or np.bool_ if needed.
        8. Never call int(series) or float(series). For boolean masks use int(mask.sum()).
           If you fill NaN with a sentinel (e.g., 'Unknown'), log nulls via original_nulls = int(col.isna().sum());
           nulls_before = original_nulls; nulls_after_na = int(cleaned.isna().sum()); filled_nulls = original_nulls.
        9. SAFE READ: You MUST read input with pd.read_csv(..., dtype=str, low_memory=False) to preserve ID fidelity.
           If you choose not to use dtype=str, you MUST define dtype/converters for identifier-like columns (id/key/cod/entity).

        COMMENT BLOCK REQUIREMENT:
        - At the top of the script, include comment sections:
          # Decision Log:
          # Assumptions:
          # Risks & Checks:

        *** SCOPE OF WORK (NON-NEGOTIABLE) ***
        - Output ONLY: data/cleaned_data.csv and data/cleaning_manifest.json.
        - MUST NOT: compute scores, case assignment, weight fitting, regression/optimization, correlations, rank checks.
        - MUST: parse types, normalize numeric formats, preserve canonical column names.
        - Manifest MUST include: output_dialect, row_counts, conversions.
        - Do NOT impute outcome/target columns. Use data/dataset_semantics.json + data/dataset_training_mask.json (Steward-decided); if partial labels exist, preserve missingness. Do not invent targets.
        - Preserve partition/split columns if they exist or are detected in the Dataset Semantics Summary.
        - If you create a partition column (split/fold/bucket), document it in the manifest and do not drop it.
        - For wide datasets, avoid enumerating all columns in code comments or logic. If data/column_sets.json exists, use src.utils.column_sets.expand_column_sets to manage column lists; fall back gracefully if the file is missing.
        - Do NOT drop columns just because they are missing from a truncated list; use selectors + explicit columns from column_sets.json when available.
        - If column_sets.json is present, preserve all columns matched by its selectors plus explicit_columns unless the contract explicitly forbids them.
        - Never assume canonical_columns is the full inventory on wide datasets. Use data/column_inventory.json + data/column_sets.json as source of truth when present.
        - cleaned_data.csv should retain nearly all feature columns; only drop columns if they are explicitly forbidden, constant, or 100% missing and the contract allows it.

        *** PYTHON SYNTAX GOTCHAS (CRITICAL) ***
        - Column names starting with a digit (e.g., '1stYearAmount') are NOT valid Python identifiers.
        - NEVER use: df.assign(1stYearAmount=...) - This causes SyntaxError!
        - ALWAYS use: df.assign(**{'1stYearAmount': ...}) or df['1stYearAmount'] = ...
        - DO NOT rescale numeric columns (e.g., divide/multiply by 100) to "normalize ranges" in cleaning.
          Only parse. Rescaling is allowed ONLY if the column is percent-like with evidence:
          (name contains '%' OR raw samples contain '%'). "score" does NOT imply percent.
        - For numeric parsing: ALWAYS sanitize symbols first (strip currency/letters; keep digits, sign, separators, parentheses, and %) and handle repeated thousands separators like '23.351.746'.
        
        *** INPUT PARAMETERS ***
        - Input: '$input_path'
        - Encoding: '$csv_encoding' | Sep: '$csv_sep' | Decimal: '$csv_decimal'
        - DE Cleaning Objective: "$business_objective"
        - Required Columns (DE View): $required_columns
        - Optional Passthrough Columns (keep if present): $optional_passthrough_columns
        - DE_VIEW_CONTEXT (json): $de_view_context
        - CONTRACT_MIN_CONTEXT (json): $contract_min_context
        - CLEANING_GATES_CONTEXT (json): $cleaning_gates_context
        - ROLE RUNBOOK (Data Engineer): $data_engineer_runbook (adhere to goals/must/must_not/safe_idioms/reasoning_checklist/validation_checklist)

        *** DATA AUDIT ***
        $data_audit
        
        *** CLEANING OUTPUT REQUIREMENTS ***
        - CRITICAL: Read input with pd.read_csv(..., sep='$csv_sep', decimal='$csv_decimal', encoding='$csv_encoding', dtype=str, low_memory=False). DO NOT rely on defaults.
        - Save cleaned CSV to data/cleaned_data.csv.
        - Save manifest to data/cleaning_manifest.json (use _safe_dump_json if present; otherwise json.dump(..., default=_json_default)).
        - CRITICAL: Manifest MUST include "output_dialect": {"sep": "...", "decimal": "...", "encoding": "..."} matching the saved file.
        - Use standard CSV (sep=',', decimal='.', encoding='utf-8') for output unless forbidden.
        - Use canonical_name from the contract for all column references.
        - Derive required columns using clear, deterministic logic.
        - Build a header map for lookup (normalize only for matching), but preserve canonical_name exactly (including spaces/symbols) in the output.
        - Canonical columns must contain cleaned values (do not leave raw strings in canonical columns while writing cleaned_* shadows).
        - Optional passthrough columns: if present in the input, keep them in the cleaned output without modification; if missing, do NOT fabricate them.
        - Print a CLEANING_VALIDATION section that reports dtype and null_frac for each required column (no advanced metrics).
        - Use DATA AUDIT + steward summary to avoid destructive parsing (null explosions) and misinterpreted number formats.
        - If a derived column has derived_owner='ml_engineer', do NOT create placeholders; leave it absent and document in the manifest.
        - Manifest audit counts: include n_cols_in, n_cols_out, kept_by_selectors_count, dropped_forbidden_count, dropped_constant_count.

        *** GATE CHECKLIST (CONTRACT-DRIVEN) ***
        - Enumerate cleaning_gates by column and requirement (max_null_fraction, allow_nulls, required_columns, etc.).
        - Before writing cleaned_data.csv, compute null_fraction for each gated column.
        - If any HARD gate is violated, raise ValueError with a clear message: "CLEANING_GATE_FAILED: <gate_name> <details>".
        - If a gate references a column that is missing, raise ValueError (do not fabricate columns).
        """

        USER_TEMPLATE = "Generate the cleaning script following Principles."

        # Rendering
        required_columns_payload = de_view.get("required_columns") or strategy.get("required_columns", [])
        if isinstance(required_columns_payload, list) and len(required_columns_payload) > 80:
            required_columns_payload = summarize_long_list(required_columns_payload)
            required_columns_payload["note"] = COLUMN_LIST_POINTER
        optional_passthrough_payload = de_view.get("optional_passthrough_columns") or []
        if isinstance(optional_passthrough_payload, list) and len(optional_passthrough_payload) > 80:
            optional_passthrough_payload = summarize_long_list(optional_passthrough_payload)
            optional_passthrough_payload["note"] = COLUMN_LIST_POINTER

        system_prompt = render_prompt(
            SYSTEM_TEMPLATE,
            input_path=input_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
            business_objective=business_objective,
            required_columns=json.dumps(required_columns_payload),
            optional_passthrough_columns=json.dumps(optional_passthrough_payload),
            data_audit=data_audit,
            contract_min_context=contract_json,
            de_view_context=de_view_json,
            data_engineer_runbook=de_runbook_json,
            cleaning_gates_context=cleaning_gates_json,
            senior_engineering_protocol=SENIOR_ENGINEERING_PROTOCOL,
        )
        self.last_prompt = system_prompt + "\n\nUSER:\n" + USER_TEMPLATE
        print(f"DEBUG: DE System Prompt Len: {len(system_prompt)}")
        print(f"DEBUG: DE System Prompt Preview: {system_prompt[:300]}...")
        if len(system_prompt) < 100:
            print("CRITICAL: System Prompt is suspiciously short!")

        from src.utils.retries import call_with_retries

        def _call_model():
            print(f"DEBUG: Data Engineer calling DeepSeek Model ({self.model_name})...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_TEMPLATE}
                ],
                temperature=0.1
            )
            # Use _extract_nonempty to handle EMPTY_COMPLETION (CAUSA RAÍZ 2)
            content = self._extract_nonempty(response)
            print(f"DEBUG: Primary DE Response Preview: {content[:200]}...")
            self.last_response = content

            # CRITICAL CHECK FOR SERVER ERRORS (HTML/504)
            if "504 Gateway Time-out" in content or "<html" in content.lower():
                raise ConnectionError("LLM Server Timeout (504 Received)")

            # Check for JSON error messages that are NOT valid code
            content_stripped = content.strip()
            if content_stripped.startswith("{") or content_stripped.startswith("["):
                try:
                    import json
                    json_content = json.loads(content_stripped)
                    if isinstance(json_content, dict):
                        if "error" in json_content or "errorMessage" in json_content:
                            raise ConnectionError(f"API Error Detected (JSON): {content_stripped}")
                except Exception:
                    pass

            # Text based fallback for Error/Overloaded keywords
            content_lower = content.lower()
            if "error" in content_lower and ("overloaded" in content_lower or "rate limit" in content_lower or "429" in content_lower):
                raise ConnectionError(f"API Error Detected (Text): {content_stripped}")

            return content

        def _call_fallback():
            if not self.fallback_client:
                raise ValueError("No fallback client configured.")
            print(f"DEBUG: Data Engineer calling Fallback Model ({self.fallback_model_name})...")
            response = self.fallback_client.chat.completions.create(
                model=self.fallback_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_TEMPLATE}
                ],
                temperature=0.1
            )
            # Use _extract_nonempty to handle EMPTY_COMPLETION (CAUSA RAÍZ 2)
            content = self._extract_nonempty(response)
            print(f"DEBUG: Fallback DE Response Preview: {content[:200]}...")
            self.last_response = content
            return content

        injection = "\n".join(
            [
                "import os",
                "import json",
                "from datetime import date, datetime",
                "try:",
                "    import numpy as np",
                "except Exception:",
                "    np = None",
                "try:",
                "    import pandas as pd",
                "except Exception:",
                "    pd = None",
                "",
                "os.makedirs('data', exist_ok=True)",
                "",
                "def _to_jsonable(value):",
                "    if value is None:",
                "        return None",
                "    if isinstance(value, (str, int, bool)):",
                "        return value",
                "    if isinstance(value, float):",
                "        return None if value != value else value",
                "    if isinstance(value, (datetime, date)):",
                "        return value.isoformat()",
                "    if isinstance(value, (list, tuple, set)):",
                "        return [_to_jsonable(item) for item in value]",
                "    if isinstance(value, dict):",
                "        return {str(k): _to_jsonable(v) for k, v in value.items()}",
                "    if isinstance(value, (bytes, bytearray)):",
                "        return value.decode('utf-8', errors='replace')",
                "    if np is not None:",
                "        if isinstance(value, np.bool_):",
                "            return bool(value)",
                "        if isinstance(value, np.integer):",
                "            return int(value)",
                "        if isinstance(value, np.floating):",
                "            return float(value)",
                "        if isinstance(value, np.ndarray):",
                "            return [_to_jsonable(item) for item in value.tolist()]",
                "    if pd is not None:",
                "        if value is pd.NA:",
                "            return None",
                "        if isinstance(value, pd.Timestamp):",
                "            return value.isoformat()",
                "        try:",
                "            if pd.isna(value) is True:",
                "                return None",
                "        except Exception:",
                "            pass",
                "    return str(value)",
                "",
                "_ORIG_JSON_DUMP = json.dump",
                "_ORIG_JSON_DUMPS = json.dumps",
                "",
                "def _safe_dump_json(obj, fp, **kwargs):",
                "    payload = _to_jsonable(obj)",
                "    kwargs.pop('default', None)",
                "    return _ORIG_JSON_DUMP(payload, fp, **kwargs)",
                "",
                "def _safe_dumps_json(obj, **kwargs):",
                "    payload = _to_jsonable(obj)",
                "    kwargs.pop('default', None)",
                "    return _ORIG_JSON_DUMPS(payload, **kwargs)",
                "",
                "json.dump = _safe_dump_json",
                "json.dumps = _safe_dumps_json",
                "",
            ]
        ) + "\n"

        try:
            content = call_with_retries(_call_model, max_retries=5, backoff_factor=2, initial_delay=2)
            print("DEBUG: DeepSeek response received.")

            code = self._clean_code(content)

            return injection + code

        except Exception as e:
            print(f"WARNING: Primary Data Engineer Model (DeepSeek) failed: {str(e)}")
            if self.fallback_client:
                print(f"DEBUG: Attempting Fallback to {self.fallback_model_name}...")
                try:
                    content = call_with_retries(_call_fallback, max_retries=3, backoff_factor=2, initial_delay=2)
                    print("DEBUG: Fallback response received.")
                    code = self._clean_code(content)
                    return injection + code
                except Exception as e_bk:
                    error_msg = f"Data Engineer Failed (Primary & Fallback): {str(e)} | Backup: {str(e_bk)}"
            else:
                error_msg = f"Data Engineer Failed (Max Retries, No Fallback): {str(e)}"

            print(f"CRITICAL: {error_msg}")
            return f"# Error: {error_msg}"

    def _clean_code(self, code: str) -> str:
        """
        Extracts code from markdown blocks, validates syntax, and applies auto-fixes.
        Raises ValueError if code is empty or has unfixable syntax errors (CAUSA RAÍZ 2 & 3).
        """
        # Step 1: Extract code from markdown
        cleaned = (extract_code_block(code) or "").strip()

        if not cleaned:
            print("ERROR: EMPTY_CODE_AFTER_EXTRACTION")
            raise ValueError("EMPTY_CODE_AFTER_EXTRACTION")

        # Step 2: Validate syntax
        try:
            ast.parse(cleaned)
            return cleaned
        except SyntaxError as e:
            print(f"DEBUG: SyntaxError detected: {e}. Attempting auto-fix...")

        # Step 3: Auto-fix common pattern: .assign(1stYearAmount=...) -> .assign(**{'1stYearAmount': ...})
        # Pattern: .assign(DIGIT_START_IDENT=VALUE)
        pattern = r'\.assign\(\s*([0-9][a-zA-Z0-9_]*)\s*=\s*([^)]+)\)'
        
        def fix_assign(match):
            col_name = match.group(1)
            value = match.group(2)
            return f".assign(**{{'{col_name}': {value}}})"

        fixed = re.sub(pattern, fix_assign, cleaned)

        # Step 4: Validate fixed code
        try:
            ast.parse(fixed)
            print("DEBUG: Auto-fix successful.")
            return fixed
        except SyntaxError as e2:
            print(f"ERROR: Auto-fix failed. Syntax still invalid: {e2}")
            raise ValueError(f"UNFIXABLE_SYNTAX_ERROR: {e2}")
