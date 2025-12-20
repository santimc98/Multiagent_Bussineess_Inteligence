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
        Generates a JSON cleaning plan to clean and standardize the dataset.
        """
        from src.utils.prompting import render_prompt
        from src.utils.cleaning_plan import parse_cleaning_plan, validate_cleaning_plan
        from src.utils.static_safety_scan import scan_code_safety
        import json

        contract_json = json.dumps(execution_contract or {}, indent=2)
        de_runbook_json = json.dumps(
            (execution_contract or {}).get("role_runbooks", {}).get("data_engineer", {}),
            indent=2,
        )
        
        # SYSTEM TEMPLATE (Static, Safe, No F-Strings)
        SYSTEM_TEMPLATE = """
        You are a Senior Data Engineer. Produce a robust cleaning PLAN for downstream ML.
        
        *** HARD CONSTRAINTS (VIOLATION = FAILURE) ***
        1. OUTPUT JSON ONLY (no markdown/code fences).
        2. Do NOT output Python code.
        3. NO NETWORK/FS OPS: Do NOT use requests/subprocess/os.system and do not access filesystem outside declared input/output paths.
        4. BAN pandas private APIs: do not use pandas.io.* or pd.io.parsers.*.
        
        *** INPUT PARAMETERS ***
        - Input: '$input_path'
        - Encoding: '$csv_encoding' | Sep: '$csv_sep' | Decimal: '$csv_decimal'
        - Business Objective: "$business_objective"
        - Required Columns (Strategy): $required_columns
        - Execution Contract (json): $execution_contract_json (for reasoning only)
        - ROLE RUNBOOK (Data Engineer): $data_engineer_runbook (adhere to goals/must/must_not/safe_idioms/reasoning_checklist/validation_checklist)

        *** DATA AUDIT ***
        $data_audit
        
        *** CLEANING PLAN OUTPUT (JSON) ***
        Output a JSON object with plan_type="cleaning_plan_v1".
        Required keys:
          {
            "plan_type": "cleaning_plan_v1",
            "input": {"path": "$input_path"},
            "dialect": {"sep": "$csv_sep", "decimal": "$csv_decimal", "encoding": "$csv_encoding"},
            "output": {"cleaned_path": "data/cleaned_data.csv", "manifest_path": "data/cleaning_manifest.json"},
            "normalize": {"method": "snake_case_dedup"},
            "column_mapping": {"use_canonical_names": true, "required": [<canonical names>]},
            "type_conversions": [
              {"column": "...", "kind": "numeric|categorical|datetime", "role": "...", "normalize_0_1": true|false}
            ],
            "derived_columns": [
              {"name": "...", "op": "copy|clip|case_when", ...}
            ],
            "case_assignment": {
              "case_id_col": "...",
              "refscore_col": "...",
              "rules": [
                {"case_id": 1, "ref_score": 1.0, "when": [ {"col":"...", "op":"<", "value": 3}, ... ] }
              ],
              "default": {"case_id": 9, "ref_score": -1.0}
            },
            "warnings": []
          }
        - Use canonical_name from the contract for all column references.
        - Use structured conditions only (no free-text rules).
        - If a column is numeric/percentage, include a conversion entry with normalize_0_1 when needed.
        """
        
        # USER TEMPLATE (Static)
        USER_TEMPLATE = "Generate the cleaning plan following Principles."
        
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
            
            plan, _ = parse_cleaning_plan(content)
            if plan:
                issues = validate_cleaning_plan(plan)
                if issues:
                    return f"# Error: CLEANING_PLAN_INVALID: {', '.join(issues)}"
                return json.dumps(plan, indent=2, ensure_ascii=False)

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
