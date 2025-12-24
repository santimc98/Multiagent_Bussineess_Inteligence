import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class DataEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Data Engineer Agent with MIMO v2 Flash.
        """
        self.api_key = api_key or os.getenv("MIMO_API_KEY")
        if not self.api_key:
            raise ValueError("MIMO API Key is required.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.xiaomimimo.com/v1",
            timeout=None,
        )
        self.model_name = "mimo-v2-flash"

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
        You are a Senior Data Engineer. Produce a robust cleaning SCRIPT for downstream ML.
        
        *** HARD CONSTRAINTS (VIOLATION = FAILURE) ***
        1. OUTPUT VALID PYTHON CODE ONLY (no markdown/code fences).
        2. Do NOT output JSON plans or pseudo-code.
        3. NO NETWORK/FS OPS: Do NOT use requests/subprocess/os.system and do not access filesystem outside declared input/output paths.
        4. BAN pandas private APIs: do not use pandas.io.* or pd.io.parsers.*.
        5. If the audit includes RUNTIME_ERROR_CONTEXT, fix the root cause and regenerate the full script.
        
        *** INPUT PARAMETERS ***
        - Input: '$input_path'
        - Encoding: '$csv_encoding' | Sep: '$csv_sep' | Decimal: '$csv_decimal'
        - Business Objective: "$business_objective"
        - Required Columns (Strategy): $required_columns
        - Execution Contract (json): $execution_contract_json (for reasoning only)
        - ROLE RUNBOOK (Data Engineer): $data_engineer_runbook (adhere to goals/must/must_not/safe_idioms/reasoning_checklist/validation_checklist)

        *** DATA AUDIT ***
        $data_audit
        
        *** CLEANING OUTPUT REQUIREMENTS ***
        - Read input using the provided dialect (sep/decimal/encoding).
        - Save cleaned CSV to data/cleaned_data.csv.
        - Save manifest to data/cleaning_manifest.json (json.dump(..., default=_json_default)).
        - Use canonical_name from the contract for all column references.
        - Derive required columns using clear, deterministic logic.
        - Build a header map for lookup (normalize only for matching), but preserve canonical_name exactly (including spaces/symbols) in the output.
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
            print(f"DEBUG: Data Engineer calling MIMO Model ({self.model_name})...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_TEMPLATE}
                ],
                temperature=0.1
            )

            content = response.choices[0].message.content
            
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
            content = call_with_retries(_call_model, max_retries=5, backoff_factor=2, initial_delay=2)
            print("DEBUG: MIMO response received.")
            
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
