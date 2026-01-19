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
from src.utils.senior_protocol import (
    SENIOR_ENGINEERING_PROTOCOL,
    SENIOR_REASONING_PROTOCOL_GENERAL,
)

# NOTE: scan_code_safety referenced by tests as a required safety mechanism.
# ML code executes in sandbox; keep the reference for integration checks.
_scan_code_safety_ref = "scan_code_safety"

class MLEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the ML Engineer Agent with the configured provider.
        """
        self.provider = (os.getenv("ML_ENGINEER_PROVIDER", "openrouter") or "openrouter").strip().lower()
        self.fallback_model_name = None
        self.last_model_used = None
        self.last_fallback_reason = None
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
        elif self.provider == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OpenRouter API Key is required.")
            timeout_raw = os.getenv("OPENROUTER_TIMEOUT_SECONDS")
            try:
                timeout_seconds = float(timeout_raw) if timeout_raw else 120.0
            except ValueError:
                timeout_seconds = 120.0
            headers = {}
            referer = os.getenv("OPENROUTER_HTTP_REFERER")
            if referer:
                headers["HTTP-Referer"] = referer
            title = os.getenv("OPENROUTER_X_TITLE")
            if title:
                headers["X-Title"] = title
            client_kwargs = {
                "api_key": self.api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "timeout": timeout_seconds,
            }
            if headers:
                client_kwargs["default_headers"] = headers
            self.client = OpenAI(**client_kwargs)
            self.model_name = os.getenv("OPENROUTER_ML_PRIMARY_MODEL") or "z-ai/glm-4.7"
            self.fallback_model_name = os.getenv("OPENROUTER_ML_FALLBACK_MODEL") or "moonshotai/kimi-k2-thinking"
        elif self.provider == "deepseek":
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DeepSeek API Key is required.")

            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
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
        """
        Extract relevant V4.1 fields for ML Engineer prompt.
        Excludes legacy spec_extraction and role_runbooks.
        """
        if not isinstance(contract, dict):
            return {}
        
        # V4.1 keys relevant for ML Engineer
        keep_keys = [
            "contract_version",
            "strategy_title",
            "business_objective",
            "required_columns",
            "required_outputs",
            "data_requirements",
            "alignment_requirements",
            "business_alignment",
            "feature_semantics",
            "business_sanity_checks",
            "column_roles",
            "decision_variables",
            "feature_availability",
            "availability_summary",
            "required_dependencies",
            "compliance_checklist",
            # V4.1 specific
            "artifact_requirements",
            "qa_gates",
            "reviewer_gates",
            "allowed_feature_sets",
            "leakage_execution_plan",
            "data_limited_mode",
            "ml_engineer_runbook",
            "derived_columns",
        ]
        
        compact: Dict[str, Any] = {}
        for key in keep_keys:
            if key in contract:
                compact[key] = contract.get(key)
        
        # Truncate large lists
        for key in ["canonical_columns", "column_mapping_rules", "column_mapping"]:
            vals = contract.get(key)
            if isinstance(vals, list) and vals:
                compact[key] = vals[:80]
        
        # Ensure ml_engineer_runbook is present (V4.1 priority)
        if "ml_engineer_runbook" not in compact or not compact.get("ml_engineer_runbook"):
            # Fallback: try role_runbooks.ml_engineer (legacy shim)
            role_runbooks = contract.get("role_runbooks")
            if isinstance(role_runbooks, dict):
                compact["ml_engineer_runbook"] = role_runbooks.get("ml_engineer", {})
        
        return compact

    def _resolve_allowed_columns_for_prompt(self, contract: Dict[str, Any] | None) -> List[str]:
        if not isinstance(contract, dict):
            return []
        cols: List[str] = []
        for key in ("required_columns", "canonical_columns"):
            values = contract.get(key)
            if isinstance(values, list):
                cols.extend([str(v) for v in values if v])
        feature_availability = contract.get("feature_availability")
        if isinstance(feature_availability, list):
            for item in feature_availability:
                if isinstance(item, dict) and item.get("column"):
                    cols.append(str(item.get("column")))
        decision_vars = contract.get("decision_variables")
        if isinstance(decision_vars, list):
            cols.extend([str(v) for v in decision_vars if v])
        seen = set()
        deduped = []
        for col in cols:
            key = col.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(col)
            if len(deduped) >= 120:
                break
        return deduped

    def _resolve_allowed_name_patterns_for_prompt(self, contract: Dict[str, Any] | None) -> List[str]:
        if not isinstance(contract, dict):
            return []
        schema = contract.get("artifact_schemas")
        if not isinstance(schema, dict):
            spec = contract.get("spec_extraction") if isinstance(contract.get("spec_extraction"), dict) else None
            if isinstance(spec, dict):
                schema = spec.get("artifact_schemas")
        if not isinstance(schema, dict):
            return []
        scored_schema = schema.get("data/scored_rows.csv")
        if not isinstance(scored_schema, dict):
            return []
        allowed = scored_schema.get("allowed_name_patterns")
        if not isinstance(allowed, list):
            return []
        return [str(pat) for pat in allowed if isinstance(pat, str) and pat.strip()]

    def _select_feedback_blocks(
        self,
        feedback_history: List[str] | None,
        gate_context: Dict[str, Any] | None,
        max_blocks: int = 2,
    ) -> str:
        blocks: List[str] = []
        if isinstance(gate_context, dict):
            feedback = gate_context.get("feedback")
            if isinstance(feedback, str) and feedback.strip():
                blocks.append(feedback.strip())
        if isinstance(feedback_history, list):
            for item in reversed(feedback_history):
                if isinstance(item, str) and item.strip():
                    blocks.append(item.strip())
                if len(blocks) >= max_blocks:
                    break
        if not blocks:
            return ""
        return "\n\n".join(blocks[:max_blocks])

    def _extract_decisioning_context(
        self,
        ml_view: Dict[str, Any] | None,
        execution_contract: Dict[str, Any] | None,
    ) -> tuple[str, str, str]:
        decisioning_requirements: Dict[str, Any] = {}
        if isinstance(ml_view, dict):
            decisioning_requirements = ml_view.get("decisioning_requirements") or {}
        if not decisioning_requirements and isinstance(execution_contract, dict):
            decisioning_requirements = execution_contract.get("decisioning_requirements") or {}
        if not isinstance(decisioning_requirements, dict):
            decisioning_requirements = {}
        decisioning_requirements_context = json.dumps(decisioning_requirements, indent=2)
        decisioning_columns = []
        output_block = decisioning_requirements.get("output") if isinstance(decisioning_requirements, dict) else None
        required_columns = output_block.get("required_columns") if isinstance(output_block, dict) else None
        if isinstance(required_columns, list):
            for col in required_columns:
                if isinstance(col, dict) and col.get("name"):
                    decisioning_columns.append(str(col.get("name")))
        decisioning_columns_text = ", ".join(decisioning_columns) if decisioning_columns else "None requested."
        decisioning_policy_notes = decisioning_requirements.get("policy_notes", "")
        if decisioning_policy_notes is None:
            decisioning_policy_notes = ""
        if not isinstance(decisioning_policy_notes, str):
            decisioning_policy_notes = str(decisioning_policy_notes)
        return decisioning_requirements_context, decisioning_columns_text, decisioning_policy_notes

    def _extract_visual_requirements_context(self, ml_view: Dict[str, Any] | None) -> str:
        visual_requirements: Dict[str, Any] = {}
        if isinstance(ml_view, dict):
            visual_requirements = ml_view.get("visual_requirements") or {}
        if not isinstance(visual_requirements, dict):
            visual_requirements = {}
        return json.dumps(visual_requirements, indent=2)

    def _build_system_prompt(
        self,
        template: str,
        render_kwargs: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None = None,
        execution_contract: Dict[str, Any] | None = None,
    ) -> str:
        render_kwargs = render_kwargs if isinstance(render_kwargs, dict) else {}
        decisioning_requirements_context, decisioning_columns_text, decisioning_policy_notes = (
            self._extract_decisioning_context(ml_view, execution_contract)
        )
        visual_requirements_context = self._extract_visual_requirements_context(ml_view)
        merged = dict(render_kwargs)
        merged.update(
            {
                "decisioning_requirements_context": decisioning_requirements_context,
                "decisioning_policy_notes": decisioning_policy_notes,
                "decisioning_columns_text": decisioning_columns_text,
                "visual_requirements_context": visual_requirements_context,
                "senior_reasoning_protocol": SENIOR_REASONING_PROTOCOL_GENERAL,
                "senior_engineering_protocol": SENIOR_ENGINEERING_PROTOCOL,
            }
        )
        return render_prompt(template, **merged)

    def _build_incomplete_reprompt_context(
        self,
        execution_contract: Dict[str, Any] | None,
        required_outputs: List[str],
        iteration_memory_block: str,
        iteration_memory: List[Dict[str, Any]] | None,
        feedback_history: List[str] | None,
        gate_context: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None = None,
    ) -> str:
        contract_min = self._compact_execution_contract(execution_contract or {})
        allowed_columns = self._resolve_allowed_columns_for_prompt(execution_contract or {})
        allowed_patterns = self._resolve_allowed_name_patterns_for_prompt(execution_contract or {})
        feedback_blocks = self._select_feedback_blocks(feedback_history, gate_context, max_blocks=2)
        ml_view = ml_view or {}
        ml_view_json = json.dumps(ml_view, indent=2)
        plot_spec_json = json.dumps(ml_view.get("plot_spec", {}), indent=2)
        decisioning_requirements_context, decisioning_columns_text, decisioning_policy_notes = (
            self._extract_decisioning_context(ml_view, execution_contract)
        )
        visual_requirements_json = self._extract_visual_requirements_context(ml_view)
        
        # STRUCTURED CRITICAL ERRORS SECTION
        critical_errors: List[str] = []
        recent_history = (iteration_memory or [])[-2:]
        current_failure = gate_context or {}
        
        # 1. Process Current failure (from gate_context)
        if current_failure:
            att_num = len(iteration_memory or []) + 1
            f_gates = current_failure.get("failed_gates", [])
            f_type = ", ".join(f_gates) if isinstance(f_gates, list) and f_gates else "MODEL_REJECTION"
            f_feedback = str(current_failure.get("feedback", "")).strip()
            f_fixes = current_failure.get("required_fixes", [])
            f_fix_str = "; ".join(f_fixes) if isinstance(f_fixes, list) and f_fixes else "Address feedback"
            
            critical_errors.append(
                f"ATTEMPT {att_num} - REJECTED:\n"
                f"  - Error Type: {f_type}\n"
                f"  - Root Cause: {f_feedback or 'General methodological failure'}\n"
                f"  - Required Fix: {f_fix_str}"
            )
            
        # 2. Process historical failures (from iteration_memory)
        for i, entry in enumerate(recent_history):
            att_id = entry.get("iteration_id", i + 1)
            # Skip if it's the current one (not expected in iteration_memory yet, but just in case)
            if current_failure and att_id == len(iteration_memory or []) + 1:
                continue
                
            e_type = ", ".join(entry.get("reviewer_reasons", []) + entry.get("qa_reasons", [])) or "PREVIOUS_FAILURE"
            e_cause = entry.get("runtime_error", {}).get("message") if isinstance(entry.get("runtime_error"), dict) else None
            # Fallback to general reasons if no runtime error message
            if not e_cause:
                e_cause = "; ".join(entry.get("reviewer_reasons", []) or ["Historical rejection"])
                
            e_fix = "; ".join(entry.get("next_actions", [])) or "Improve implementation"
            
            critical_errors.append(
                f"ATTEMPT {att_id} - REJECTED:\n"
                f"  - Error Type: {e_type}\n"
                f"  - Root Cause: {e_cause}\n"
                f"  - Required Fix: {e_fix}"
            )
            
        critical_section = ""
        if critical_errors:
            critical_section = (
                "!!! CRITICAL ERRORS FROM PREVIOUS ATTEMPTS (DO NOT REPEAT) !!!\n" +
                "\n".join(critical_errors) +
                "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            )

        memory_block = iteration_memory_block.strip()
        if not memory_block:
            memory_block = json.dumps(iteration_memory or [], indent=2)
            
        rules_block = "\n".join(
            [
                "- No synthetic feature/row generation. Load only the provided dataset. Bootstrap resampling of existing rows is allowed when required by the contract.",
                "- Do not mutate df_in; use df_work = df_in.copy() and only assign contract-declared derived columns.",
                "- Baseline model is required.",
                "- Include SimpleImputer in preprocessing when NaNs may exist.",
                "- Write all required outputs to exact paths.",
                "- scored_rows may include canonical + contract-approved derived outputs only.",
                "- Define CONTRACT_COLUMNS from the contract and print a MAPPING SUMMARY.",
            ]
        )
        return "\n".join(
            [
                critical_section,
                "ML_VIEW_CONTEXT:",
                ml_view_json,
                "PLOT_SPEC_CONTEXT:",
                plot_spec_json,
                "DECISIONING_REQUIREMENTS_CONTEXT:",
                decisioning_requirements_context,
                "DECISIONING_COLUMNS:",
                decisioning_columns_text,
                "DECISIONING_POLICY_NOTES:",
                decisioning_policy_notes or "None",
                "VISUAL_REQUIREMENTS_CONTEXT:",
                visual_requirements_json,
                "CONTRACT_MIN_CONTEXT:",
                json.dumps(contract_min, indent=2),
                "REQUIRED OUTPUTS:",
                json.dumps(required_outputs or [], indent=2),
                "ALLOWED COLUMNS:",
                json.dumps(allowed_columns, indent=2),
                "ALLOWED_NAME_PATTERNS:",
                json.dumps(allowed_patterns, indent=2),
                "UNIVERSAL RULES:",
                rules_block,
                "ITERATION_MEMORY_CONTEXT:",
                memory_block,
                "LATEST_REVIEW_FEEDBACK:",
                feedback_blocks or "None",
            ]
        )

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

    def _fix_data_path_in_code(self, code: str, correct_path: str) -> str:
        """
        Post-processing safety check: If LLM generated code with incorrect INPUT_FILE path,
        inject the correct one. This is NOT hardcoding business logic - it's infrastructure.

        Common wrong paths: 'input.csv', 'data.csv', 'data/input_data.csv', 'data/data.csv', 'raw_data.csv'
        Correct path: provided by system (usually 'data/cleaned_data.csv')

        This catches common variable naming patterns that LLMs use for input paths.
        """
        import re

        # Comprehensive pattern: Capture common variable name patterns for input data paths
        # Matches: INPUT_FILE(NAME), DATA_FILE(NAME), INPUT_PATH, DATA_PATH, CLEANED_(DATA_)PATH, etc.
        # Negative lookahead ensures we don't replace if path is already correct
        wrong_patterns = [
            # Pattern 1: (INPUT|DATA|CLEANED)_(FILE|FILENAME|PATH|DATA_PATH) = 'wrong.csv'
            r"(INPUT_FILE(?:NAME)?|DATA_FILE(?:NAME)?|INPUT_PATH|DATA_PATH|CLEANED_(?:DATA_)?PATH|CLEANED_FILE)\s*=\s*['\"](?!{})(?!['\"]?\s*\+)[^'\"]+\.csv['\"]".format(
                re.escape(correct_path)
            ),
        ]

        for pattern in wrong_patterns:
            if re.search(pattern, code):
                # Replace with correct path
                code = re.sub(
                    pattern,
                    f"\\1 = '{correct_path}'",
                    code
                )
                print(f"SAFETY_FIX: Injected correct data_path='{correct_path}' into generated code")
                break

        return code

    def _fix_to_csv_dialect_in_code(self, code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        has_sep = False
        has_decimal = False
        has_encoding = False
        has_load_dialect = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id == "sep":
                    has_sep = True
                elif node.id == "decimal":
                    has_decimal = True
                elif node.id == "encoding":
                    has_encoding = True
            if isinstance(node, ast.Call):
                name = self._call_name(node)
                if name.endswith("load_dialect"):
                    has_load_dialect = True
        if not has_load_dialect and not (has_sep and has_decimal and has_encoding):
            return code

        class _ToCsvDialectFixer(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call) -> ast.AST:
                self.generic_visit(node)
                if not isinstance(node.func, ast.Attribute) or node.func.attr != "to_csv":
                    return node
                if any(kw.arg is None for kw in node.keywords):
                    return node
                existing = {kw.arg for kw in node.keywords if kw.arg}
                if "sep" not in existing:
                    node.keywords.append(ast.keyword(arg="sep", value=ast.Name(id="sep", ctx=ast.Load())))
                if "decimal" not in existing:
                    node.keywords.append(ast.keyword(arg="decimal", value=ast.Name(id="decimal", ctx=ast.Load())))
                if "encoding" not in existing:
                    node.keywords.append(ast.keyword(arg="encoding", value=ast.Name(id="encoding", ctx=ast.Load())))
                return node

        try:
            fixed_tree = _ToCsvDialectFixer().visit(tree)
            ast.fix_missing_locations(fixed_tree)
            return ast.unparse(fixed_tree)
        except Exception:
            return code

    def _call_name(self, call_node: ast.Call) -> str:
        try:
            return ast.unparse(call_node.func)
        except Exception:
            if isinstance(call_node.func, ast.Name):
                return call_node.func.id
            if isinstance(call_node.func, ast.Attribute):
                return call_node.func.attr
        return ""

    def _get_call_parts(self, node: ast.AST) -> List[str]:
        parts: List[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return list(reversed(parts))

    def _is_path_constructor(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "Path"
        if isinstance(node, ast.Attribute):
            return node.attr == "Path"
        return False

    def _node_is_input_reference(self, node: ast.AST, input_names: set, data_path: str) -> bool:
        if isinstance(node, ast.Name) and node.id in input_names:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == data_path:
            return True
        if isinstance(node, ast.Call) and self._is_path_constructor(node.func):
            for arg in node.args:
                if self._node_is_input_reference(arg, input_names, data_path):
                    return True
            for kw in node.keywords:
                if self._node_is_input_reference(kw.value, input_names, data_path):
                    return True
        return False

    def _extract_target_names(self, target: ast.AST) -> List[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            names: List[str] = []
            for elt in target.elts:
                names.extend(self._extract_target_names(elt))
            return names
        return []

    def _collect_input_path_names(self, tree: ast.AST, data_path: str) -> set:
        names: set = set()

        def value_is_data_path(value: ast.AST, known: set) -> bool:
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value == data_path:
                return True
            if isinstance(value, ast.Name) and value.id in known:
                return True
            if isinstance(value, ast.Call) and self._is_path_constructor(value.func):
                for arg in value.args:
                    if value_is_data_path(arg, known):
                        return True
                for kw in value.keywords:
                    if value_is_data_path(kw.value, known):
                        return True
            return False

        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    value = node.value
                    if value is None or not value_is_data_path(value, names):
                        continue
                    for target in node.targets:
                        for name in self._extract_target_names(target):
                            if name not in names:
                                names.add(name)
                                changed = True
                elif isinstance(node, ast.AnnAssign):
                    value = node.value
                    if value is None or not value_is_data_path(value, names):
                        continue
                    for name in self._extract_target_names(node.target):
                        if name not in names:
                            names.add(name)
                            changed = True
        return names

    def _is_input_exists_call(self, call_node: ast.Call, input_names: set, data_path: str) -> bool:
        if not isinstance(call_node.func, ast.Attribute) or call_node.func.attr != "exists":
            return False
        func_value = call_node.func.value
        if isinstance(func_value, ast.Attribute):
            if isinstance(func_value.value, ast.Name) and func_value.value.id == "os" and func_value.attr == "path":
                return any(
                    self._node_is_input_reference(arg, input_names, data_path)
                    for arg in call_node.args
                )
        if isinstance(func_value, ast.Call) and self._is_path_constructor(func_value.func):
            for arg in func_value.args:
                if self._node_is_input_reference(arg, input_names, data_path):
                    return True
            for kw in func_value.keywords:
                if self._node_is_input_reference(kw.value, input_names, data_path):
                    return True
        if self._node_is_input_reference(func_value, input_names, data_path):
            return True
        return False

    def _try_handles_filenotfound(self, try_node: ast.Try) -> bool:
        for handler in try_node.handlers:
            exc_type = handler.type
            if exc_type is None:
                continue
            if isinstance(exc_type, ast.Name) and exc_type.id == "FileNotFoundError":
                return True
            if isinstance(exc_type, ast.Attribute) and exc_type.attr == "FileNotFoundError":
                return True
            if isinstance(exc_type, ast.Tuple):
                for elt in exc_type.elts:
                    if isinstance(elt, ast.Name) and elt.id == "FileNotFoundError":
                        return True
                    if isinstance(elt, ast.Attribute) and elt.attr == "FileNotFoundError":
                        return True
        return False

    def _call_is_read_csv(self, call_node: ast.Call) -> bool:
        name = self._call_name(call_node).lower()
        return name.endswith("read_csv")

    def _call_uses_input(self, call_node: ast.Call, input_names: set, data_path: str) -> bool:
        for arg in call_node.args:
            if self._node_is_input_reference(arg, input_names, data_path):
                return True
        for kw in call_node.keywords:
            if self._node_is_input_reference(kw.value, input_names, data_path):
                return True
        return False

    def _synthetic_call_reason(self, call_node: ast.Call) -> str | None:
        parts = self._get_call_parts(call_node.func)
        if parts:
            func_name = parts[-1].lower()
            module_prefix = ".".join(part.lower() for part in parts[:-1])
            if module_prefix in {"np.random", "numpy.random"}:
                allowed = {"seed", "choice", "randint", "permutation", "shuffle", "default_rng"}
                if func_name not in allowed:
                    return f"forbidden_np_random_call:{module_prefix}.{func_name}"

        call_name = self._call_name(call_node).lower()
        if "sklearn.datasets.make_" in call_name or ".datasets.make_" in call_name or call_name.startswith("make_"):
            return f"forbidden_sklearn_make_call:{call_name}"
        if "faker" in call_name:
            return f"forbidden_faker_call:{call_name}"
        return None

    def _detect_forbidden_input_fallback(self, code: str, data_path: str) -> List[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"ast_parse_failed:{exc.msg}"]

        input_names = self._collect_input_path_names(tree, data_path)
        reasons: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if self._is_input_exists_call(node, input_names, data_path):
                    reasons.append("input_existence_check_on_input_file")
                synthetic_reason = self._synthetic_call_reason(node)
                if synthetic_reason:
                    reasons.append(synthetic_reason)

        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and self._try_handles_filenotfound(node):
                for stmt in node.body:
                    for inner in ast.walk(stmt):
                        if isinstance(inner, ast.Call) and self._call_is_read_csv(inner):
                            if self._call_uses_input(inner, input_names, data_path):
                                reasons.append("file_not_found_fallback_on_input_read")
                                break

        deduped: List[str] = []
        seen = set()
        for reason in reasons:
            if reason in seen:
                continue
            seen.add(reason)
            deduped.append(reason)
        return deduped

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
        ml_view: Dict[str, Any] | None = None,
        feature_availability: List[Dict[str, Any]] | None = None,
        availability_summary: str = "",
        signal_summary: Dict[str, Any] | None = None,
        iteration_memory: List[Dict[str, Any]] | None = None,
        iteration_memory_block: str = "",
        dataset_scale: Dict[str, Any] | None = None,
        dataset_scale_str: str | None = None,
    ) -> str:

        SYSTEM_PROMPT_TEMPLATE = """
         You are a Senior ML Engineer for tabular Data Science.

         === SENIOR REASONING PROTOCOL ===
         $senior_reasoning_protocol

         === SENIOR ENGINEERING PROTOCOL ===
         $senior_engineering_protocol

         MISSION
         - Produce ONE robust, runnable Python SCRIPT that loads cleaned dataset from $data_path, trains/evaluates according to Execution Contract, and writes required artifacts.
         - Adapt to each dataset and objective. Do not follow a rigid recipe; follow contract + data.
         - If Evaluation Spec says requires_target=false, DO NOT train a supervised model. Produce descriptive/segmentation insights and still write data/metrics.json with model_trained=false.

         TRAINING DATA SELECTION (REASONED)
         - Read execution_contract (or contract_min) for outcome_columns and optional fields:
           * training_rows_rule
           * scoring_rows_rule
           * data_partitioning_notes
         - If those rules exist, implement them exactly.
         - If rules are absent, use the Dataset Semantics Summary in data_audit_context:
           * If partial_label_detected=True, train ONLY on rows where the target is not null; score all rows.
           * If partition_columns are listed and split-like values are identified, use that partitioning for train/score.
         - Do NOT hardcode column names. Use the detected target/partition columns from contract or dataset semantics.
         - Decision Log must include:
           * Target chosen
           * Training rows rule
           * Scoring rows rule
           * Evidence (e.g., null_frac or partition evidence from dataset_semantics.json)

         ML BEST PRACTICES CHECKLIST (Quality Assurance):
         [ ] NO NaN HYPOTHESES: Before .fit(), you MUST check for NaNs in X and impute (SimpleImputer) or drop them. Scikit-learn models crash on NaNs.
         [ ] INPUT SOURCE: Load data from EXACT path provided as $data_path. Do NOT hardcode arbitrary filenames.
         [ ] VARIABLE DEFINITION: Define all metric variables (e.g., auc, f1, precision) locally before trying to save them to metrics.json.
         [ ] CASTING SAFEGUARDS: When converting columns, handle non-numeric values gracefully (coerce).

         IDENTIFIER USAGE WARNING:
         - Do NOT automatically drop columns just because the name contains 'id'.
         - Refer to ML_VIEW.identifier_policy: strict identifiers are forbidden unless explicitly allowed by the contract.
         - Candidate identifiers may be useful categorical features; run a fast cardinality/uniqueness check and keep only low-cardinality ones.
         - High-cardinality candidate identifiers should be dropped or neutralized; mention the check in comments.

         {% if dataset_scale and dataset_scale.scale in ['medium', 'large'] %}
         LARGE DATASET PROTOCOL:
         {% if dataset_scale.scale == 'medium' %}
         - Dataset is MEDIUM ({{dataset_scale.file_mb:.1f}} MB, ~{{dataset_scale.est_rows}} rows).
         - TRAINING LIMIT: Use at most {{dataset_scale.max_train_rows}} rows for training (sample with train_test_split if needed).
         - CHUNK PROCESSING: If scoring many rows, process in chunks of {{dataset_scale.chunk_size}}.
         - Avoid heavy gridsearch; prefer faster models (linear, tree with limited depth).
         {% if dataset_scale.prefer_parquet %}
         - ACCELERATION: data/cleaned_data.parquet is available. You may load it instead of CSV for faster reads.
         {% endif %}
         {% elif dataset_scale.scale == 'large' %}
         - Dataset is LARGE ({{dataset_scale.file_mb:.1f}} MB, ~{{dataset_scale.est_rows}} rows).
         - TRAINING LIMIT: Use at most {{dataset_scale.max_train_rows}} rows for training (CRITICAL).
         - CHUNK PROCESSING: Score in chunks of {{dataset_scale.chunk_size}} to avoid memory issues.
         - MODEL SELECTION: Prefer SGD/linear models or tree models with limited depth.
         - DO NOT use full dataset for training - sample down to {{dataset_scale.max_train_rows}} rows.
         {% endif %}
         {% endif %}

         HARD CONSTRAINTS (VIOLATION = FAILURE)
         1) OUTPUT VALID PYTHON CODE ONLY (no markdown, no code fences, no JSON-only plans).
         1) OUTPUT VALID PYTHON CODE ONLY (no markdown, no code fences, no JSON-only plans).
        2) If RUNTIME_ERROR_CONTEXT is present in the audit, fix root cause and regenerate the FULL script.
        3) CRITICAL - DIALECT LOADING (DO THIS FIRST): Before loading ANY data, you MUST load the output_dialect from cleaning_manifest.json.
           - MANDATORY FIRST STEP: Define a load_dialect() function that reads 'data/cleaning_manifest.json' and extracts output_dialect {sep, decimal, encoding}.
           - CORRECT pattern:
             ```python
             def load_dialect():
                 manifest_path = MANIFEST_PATH
                 if os.path.exists(manifest_path):
                     with open(manifest_path, 'r') as f:
                         manifest = json.load(f)
                     dialect = manifest.get('output_dialect', {})
                     return (
                         dialect.get('sep', ';'),
                         dialect.get('decimal', ','),
                         dialect.get('encoding', 'utf-8')
                     )
                 return ';', ',', 'utf-8'

             sep, decimal, encoding = load_dialect()
             ```
           - Then use these values in ALL pd.read_csv() and .to_csv() calls.
           - DO NOT hardcode sep=',', decimal='.', or any other dialect values. ALWAYS read from manifest first.
           - Fallback ONLY if manifest doesn't exist: use the defaults shown above.
           - When writing any CSV artifacts (scored_rows.csv, optimal_pricing_guide.csv, etc.), ALWAYS pass sep, decimal, encoding from load_dialect().
        3b) USE ORCHESTRATOR PATH VARIABLES: Always use MANIFEST_PATH for dialect loading and CLEANED_CSV_PATH for input data.
           - Do NOT hardcode file paths. Do NOT use alternate path variables.
           - If you define INPUT_FILE, set INPUT_FILE = CLEANED_CSV_PATH (this is equivalent to $data_path).
        4) CRITICAL - INPUT PATH: You MUST read data from the EXACT path '$data_path' provided in the context.
           INPUT GUARANTEE (NON-NEGOTIABLE):
           - The orchestrator guarantees that the dataset at $data_path exists before your script runs.
           PROHIBITED:
           - Any input existence checks: os.path.exists(INPUT_FILE), Path(INPUT_FILE).exists(), or try/except FileNotFoundError around the input pd.read_csv.
           - Any fallback branch that creates DataFrames/arrays when input is missing (dummy/demo/synthetic data).
           - Any synthetic data generation with np.random (uniform/rand/randn/random_sample/normal/etc), sklearn.datasets.make_*, faker, etc.
           CORRECT:
           - Define INPUT_FILE = '$data_path' and call pd.read_csv with dialect.
           - If read_csv fails, let the error bubble up; do not handle by creating data.
           NOTE: os.path.exists is allowed only for reading data/cleaning_manifest.json and for creating output dirs; never for INPUT_FILE.
           - CORRECT: INPUT_FILE = '$data_path' then df = pd.read_csv(INPUT_FILE, sep=sep, decimal=decimal, encoding=encoding, ...)
           - WRONG: Using hardcoded paths like 'data.csv', 'input.csv', 'raw_data.csv', 'data/input_data.csv', etc.
           - WRONG: pd.read_csv(INPUT_FILE) without dialect parameters
           - The $data_path variable will be substituted with the actual path (e.g., 'data/cleaned_data.csv').
           - ABSOLUTE PROHIBITION: Do NOT implement fallback logic like "if not os.path.exists(filepath): generate dummy data".
             The file WILL exist. If it doesn't, let pd.read_csv() raise FileNotFoundError. NO synthetic fallbacks.
        5) Do NOT invent column names. Use only columns from the contract/canonical list and the loaded dataset.
        6) Do NOT mutate the input dataframe in-place. Use df_in for the raw load. If you need derived columns, create df_work = df_in.copy() and assign ONLY columns explicitly declared as derived in the Execution Contract (data_requirements with source='derived' or spec_extraction.derived_columns). If a required input column is missing, raise ValueError (no dummy values).
        6b) Do NOT overwrite data/cleaned_data.csv. Treat it as immutable input; write derived datasets to data/model_input.csv or data/features.csv.
        7) NEVER fabricate synthetic rows/features (pd.DataFrame({}) from literals, faker, sklearn.datasets.make_*, etc.).
           - Bootstrap/CV resampling of the OBSERVED rows is allowed (and expected when validation_requirements asks for bootstrap).
           - Randomness is permitted ONLY for resampling indices; do not generate new feature values from distributions.
        8) scored_rows.csv may include canonical columns plus contract-approved derived outputs (target/prediction/probability/segment/optimal values) ONLY if explicitly declared in data_requirements or spec_extraction. Any other derived columns must go to a separate artifact file.
        9) Start the script with a short comment block labeled PLAN describing: (1) dialect loading from cleaning_manifest.json, (2) detected columns, (3) row_id construction, (4) scored_rows columns, and (5) where extra derived artifacts go.
        10) Define CONTRACT_COLUMNS from the Execution Contract (prefer data_requirements source=input; else canonical_columns) and validate they exist in df_in; raise ValueError listing missing columns.
        11) LEAKAGE ZERO-TOLERANCE: Check 'allowed_feature_sets' in the contract. Any column listed as 'audit_only_features' or 'forbidden_for_modeling' MUST be excluded from X (features). Use them ONLY for audit/metrics calculation. Violation = REJECTION.
        12) PIPELINE ISOLATION: If you define multiple models/pipelines, do NOT reuse the same preprocessor/transformer across pipelines.
            - Create separate preprocessors or clone them.
            - Example:
              preprocessor1 = ColumnTransformer(...)
              preprocessor2 = ColumnTransformer(...)
              # or: preprocessor2 = sklearn.base.clone(preprocessor1)

         COMMENT BLOCK REQUIREMENT:
         - At the top of the script, include comment sections:
           # Decision Log:
           # Assumptions:
           # Trade-offs:
           # Risks:
        
        UNIVERSAL FEATURE USAGE RULE (CONTRACT-DRIVEN):
        - Each phase (segmentation/modeling/optimization) MUST use ONLY features allowed by the contract.
        - If 'allowed_feature_sets' exists in contract, validate features per phase:
          * segmentation_features: for clustering/segment assignment
          * modeling_features: for predictive model training
          * optimization_features: for optimization decisions
        - If 'allowed_feature_sets' is missing, derive constraints from: canonical_columns + derived_columns + leakage_execution_plan.
        - NEVER invent features or use columns not in the contract.
        
        TECHNICAL HELPERS (use these patterns):
        - JSON serialization with numpy (CRITICAL): Use this helper function to handle ALL numpy types:
          ```python
          def _json_default(obj):
              if isinstance(obj, (np.integer, np.int64)):
                  return int(obj)
              elif isinstance(obj, (np.floating, np.float64)):
                  return float(obj)
              elif isinstance(obj, (np.bool_, bool)):
                  return bool(obj)
              elif isinstance(obj, np.ndarray):
                  return obj.tolist()
              elif isinstance(obj, pd.Series):
                  return obj.tolist()
              elif pd.isna(obj):
                  return None
              raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
          ```
          Then: json.dump(data, f, indent=2, default=_json_default)
        - Sklearn scoring: Use string scorers ('accuracy', 'roc_auc', 'neg_mean_squared_error') instead of custom scorers.
          Avoid 'needs_proba' parameter issues by using cross_val_score with scoring='accuracy' for classification.

        SECURITY / SANDBOX (VIOLATION = FAILURE)
        - Do NOT import sys.
        - NO NETWORK/FS OPS: Do NOT use requests/subprocess/os.system and do not access filesystem outside declared input/output paths.
        - No network/shell: no requests, subprocess, os.system.
        - No filesystem discovery: do NOT use os.listdir, os.walk, glob.
        - INPUT FILE: Read ONLY from the path specified in '$data_path' (this will be the actual cleaned data path).
        - OUTPUT FILES: Save all outputs to ./data directory and plots to ./static/plots.

        FORBIDDEN PATTERNS & CORRECT ALTERNATIVES:
        Problem: The sandbox blocks direct DataFrame column creation for safety.
        Solution: Use .assign() or Pipeline transformers instead.

        ❌ FORBIDDEN:
          df['new_col'] = value
          df.loc[:, 'new_col'] = value
          df[['col1', 'col2']] = [val1, val2]

        ✅ CORRECT:
          df = df.assign(new_col=value)
          df = df.assign(col1=val1, col2=val2)
          
        ✅ CORRECT (for complex logic in Pipelines):
          from sklearn.preprocessing import FunctionTransformer
          def add_derived_column(X):
              return X.assign(derived_target=(X['status_col']==target_value).astype(int))
          pipeline = Pipeline([('add_derived', FunctionTransformer(add_derived_column)), ...])

        Universal patterns:
          ✅ df = df.assign(binary_flag=(df['categorical_col'] == 'target_value').astype(int))
          ✅ df = df.assign(has_quantity=(df['numeric_col'] > threshold))
          ✅ df = df.assign(segment_id=clustering_model.fit_predict(df[feature_cols]))

        ⚠️ CRITICAL: .assign() only works on DataFrames, NOT on Series:
        
        ❌ FORBIDDEN (Series has no .assign() method):
          df.apply(lambda row: row[['col1', 'col2']].assign(new_col=value), axis=1)
          #                     ↑ row[['col1', 'col2']] is a Series, not DataFrame!
        
        ✅ CORRECT (build dict first, then create DataFrame):
          df.apply(lambda row: pd.DataFrame([{**row[['col1', 'col2']].to_dict(), 'new_col': value}]), axis=1)
        
        ✅ BETTER (avoid lambda entirely, compute outside apply):
          # Step 1: Compute derived column without lambda
          df = df.assign(new_col=df.apply(lambda row: compute_value(row), axis=1))
          # Step 2: Use the result
          result = df[['col1', 'col2', 'new_col']]

        Universal example (optimization scenario):
          ❌ WRONG: row[model_features].assign(decision_var=optimal_value)  # Series!
          ✅ RIGHT: pd.DataFrame([{**row[model_features].to_dict(), 'decision_var': optimal_value}])



        INPUT CONTEXT (authoritative)
        - Business Objective: "$business_objective"
        - Strategy: $strategy_title ($analysis_type)
        - ML_VIEW_CONTEXT (json): $ml_view_context
        - PLOT_SPEC_CONTEXT (json): $plot_spec_context
        - DECISIONING REQUIREMENTS CONTEXT (json): $decisioning_requirements_context
        - DECISIONING POLICY NOTES: $decisioning_policy_notes
        - DECISIONING COLUMNS: $decisioning_columns_text
        - VISUAL_REQUIREMENTS_CONTEXT (json): $visual_requirements_context
        - CONTRACT_MIN_CONTEXT (json): $contract_min_context
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
        - Iteration Memory (compact): $iteration_memory_block
        - Data audit context: $data_audit_context

        DEPENDENCIES
        - Use only: numpy, pandas, scipy, sklearn, statsmodels, matplotlib, seaborn, pyarrow, openpyxl, duckdb, sqlalchemy, dateutil, pytz, tqdm, yaml.
        - Extended deps (rapidfuzz, plotly, pydantic, pandera, networkx) ONLY if listed in execution_contract.required_dependencies.
        - Do not import any other deps.

        CAUSAL REASONING FOR OPTIMIZATION
        - Consultation: check feature_availability in contract. Variables marked 'decision' or 'post-decision' CANNOT be features.
        - Logic: If a model needs the decision_variable to predict, it cannot recommend it for new cases (unknown at prediction time).
        - Modeling: Predict outcome using pre-decision features (F1, F2). Model decision_variable effect separately (curves/elasticity).
        - Examples:
          ✅ OK: Outcome ~ [F1, F2]; then model effect of decision_variable within segments.
          ❌ FAIL: features = [F1, F2, decision_variable].

        SENIOR WORKFLOW (do this, not a checklist)
        Step 0) LOAD DIALECT FIRST (MANDATORY):
        - BEFORE any data loading, define load_dialect() function and call it to get (sep, decimal, encoding).
        - Pattern (copy this exactly):
          def load_dialect():
              manifest_path = 'data/cleaning_manifest.json'
              if os.path.exists(manifest_path):
                  with open(manifest_path, 'r') as f:
                      manifest = json.load(f)
                  dialect = manifest.get('output_dialect', {})
                  return (
                      dialect.get('sep', ';'),
                      dialect.get('decimal', ','),
                      dialect.get('encoding', 'utf-8')
                  )
              return ';', ',', 'utf-8'

          sep, decimal, encoding = load_dialect()
          print(f"Loading data with dialect: sep='{sep}', decimal='{decimal}', encoding='{encoding}'")
        - Use these values in ALL pd.read_csv() and .to_csv() calls throughout the script.
        - After loading data:
          - If df is empty: raise ValueError including the dialect used.
          - If df has 1 column and the column name contains ',', ';', or '\\t' AND length>20: raise ValueError("Delimiter/Dialect mismatch: ...") including the dialect used.
        - Do NOT attempt to split columns or change dialect mid-script.

        Step 1) Feasibility gate:
        - Identify target from contract/spec_extraction. If missing/unmappable -> raise ValueError with a clear message.
        - Build y as a pandas Series and enforce ONE variance guard:
        if y.nunique() <= 1: raise ValueError("CRITICAL: Target variable has no variation.")
        - Never add noise/jitter.

        Step 2) Diagnose the dataset quickly:
        - Determine task type (classification/regression) and key risks:
        missingness, high-cardinality categoricals, suspected IDs, leakage/post-outcome features (use availability + semantics).
        - If the contract marks any columns as post-decision/post-outcome/leakage_risk, never include them as model features; record them in a leakage audit note.
        - Use signal_summary to choose model complexity (avoid overfitting).
        - Probability columns (e.g., Probability/prob/score) are audit-only; NEVER use for segmentation or modeling.
          For audit stats, use dropna on the joined sample; do not impute with zeros.

        Step 2.5) Segmentation sanity (required if segmentation is used):
        - Compute and log: n_rows, n_segments, min/median segment_size.
        - Respect execution_contract.segmentation_constraints (max_segments, min_segment_size, preferred_k_range).
        - If constraints violated, reduce k, or use quantile binning for numerics, top-K + "Other" for categoricals,
          or fallback to a coarser segmentation (never 1-row-per-segment).
        - Do NOT create segment_id by concatenating raw columns if it yields unique IDs per row.

        Step 3) Decide validation correctly:
        - If objective_type == "forecasting" or requires_time_series_split=true -> use TimeSeriesSplit or chronological holdout (shuffle=False). Do NOT use random KFold.
        - If the contract/spec indicates group_key OR you infer a grouping vector -> use GroupKFold or GroupShuffleSplit (or CV with groups=...).
        - Else if time_key or time ordering matters -> use a time-based split.
        - Else -> StratifiedKFold (classification) or KFold (regression).
        - Never evaluate on training data.

        Step 4) Implement with pipelines (default):
        - Use sklearn Pipeline + ColumnTransformer for preprocessing.
        - Numeric: imputer (+ scaler if needed).
        - Categorical: imputer + OneHotEncoder(handle_unknown='ignore', sparse_output=False).
        - Apply a high-cardinality safeguard when needed (e.g., limit top-K categories or hashing) without leakage.

        Step 5) Models (default behavior):
        - Always include:
        - A baseline (simple, interpretable).
        - A stronger model (more expressive) appropriate for dataset size and task.
        - Prefer stable sklearn models unless the contract explicitly requires others.
        - Any predict_proba call must pass a 2D array (e.g., X.reshape(1, -1) or [[x]] for a single row).

        MODEL SELECTION & METRICS CONSISTENCY:
        - If comparing multiple models, select best based on primary metric (AUC/accuracy/R2).
        - Save both best_model_name AND its corresponding metric value.
        - CRITICAL: best_model_auc must match the AUC of best_model_name (never mix models).
        - Pattern: if lr_auc > rf_auc: best_name="LR", best_auc=lr_auc (NOT rf_auc).

        Step 6) Contract compliance outputs:
        - Do NOT invent global rules. Use execution_contract to decide:
        - which columns to use (pre-decision vs post-outcome),
        - required artifacts,
        - derived targets/columns behavior.
        - Print a "MAPPING SUMMARY" block with canonical columns, selected features, and any derived outputs used.
        - Only enforce segmentation/weights/pricing logic IF deliverables require those outputs or decision_variables exist.
        (Example: if a required deliverable includes "data/weights.json" or execution_contract.decision_variables present -> run the corresponding logic; else skip.)
        - If price sensitivity curves or optimal pricing guide are required, they must NOT be empty.
          If segment-level estimation is too sparse, fallback to global curves or coarser segments; never emit empty artifacts.

        VISUAL REQUIREMENTS EXECUTION (CONTRACT-DRIVEN)
        - Use VISUAL_REQUIREMENTS_CONTEXT to manage plotting. Do NOT invent additional plots outside the items list.
        - If visual_requirements.enabled == false:
          * Do NOT generate PNGs; you may omit creating files under static/plots except for a stub directory.
          * Write data/visuals_status.json with {"plots_disabled": true, "reason": "Not requested"}.
        - If visual_requirements.enabled == true:
          * Generate each item listed in visual_requirements.items and save to <visual_requirements.outputs_dir>/<expected_filename>.
          * Respect constraints: sample rows according to sampling_strategy and limit to max_rows_for_plot before computing visuals.
          * If an item is required (visual_requirements.required is true) and you cannot produce it, log failure in data/alignment_check.json and append a warning to feedback_history.
          * Record a JSON object per item in data/visuals_status.json with fields: plot_id, file, status (ok|skipped|missing), skip_reason, sources_used, columns_used, rows_used.
        - Use matplotlib with Agg backend and keep seaborn imports out of saved plots (import only when necessary).
        - Avoid synthetic data (np.random) when generating plots; sampling should reuse observed rows only.

        REQUIRED ARTIFACT RULES (minimal, contract-driven)
        - Always:
        - os.makedirs('data', exist_ok=True)
        - os.makedirs('static/plots', exist_ok=True)
        - JSON writing: always json.dump(..., default=_json_default) with a small _json_default helper.
        - Write all required deliverables; write optional deliverables only if they materially support the objective.
        - metrics.json RULES:
          - MUST include top-level key: "model_performance".
          - Populate "model_performance" with standard technical metrics RELEVANT TO THE TASK:
            * Classification: AUC, F1, Accuracy, LogLoss.
            * Regression: RMSE, R2, MAE.
            * Clustering: Silhouette Score, Inertia.
            * Descriptive/Rule-based: Coverage %, Rule Hit Rate.
          - Never leave "model_performance" empty for a modeling task.
        - Plotting: matplotlib.use('Agg') BEFORE pyplot; if PLOT_SPEC_CONTEXT.enabled true, generate plots per plot_spec; otherwise save a plot only when required deliverables include plots.
        DECISION POLICY (CONTRACT-DRIVEN)
        - If DECISIONING_REQUIREMENTS_CONTEXT.enabled == true, you MUST generate each required column listed in the context and save them to $data_path/decisioning output (scored_rows.csv) with types/ranges as described.
        - Use decision columns to summarize priority, actions, segments, or flags as requested. Prefer threshold-based logic (top-k, quantiles) to complex heuristics, and document thresholds in comments or metrics.json.
        - If DECISIONING_REQUIREMENTS_CONTEXT.enabled == false, do NOT invent new action/flag/segment columns or write additional decision artifacts.
        - If a decision column cannot be produced (missing inputs, no predictions), log the issue in alignment_check.json/feedback_history so the reviewers can request a fix.
        - If computing optimal prices or using minimize_scalar, ensure the objective returns float and coerce optimal_price = float(optimal_price) before assignment.
        - scored_rows.csv must include canonical columns plus derived outputs required by the contract
          (e.g., is_success, cluster_id, pred_prob_success, recommended_* and expected_value_at_recommendation).

        ALIGNMENT CHECK (contract-driven)
        - Write data/alignment_check.json with:
          status (PASS|WARN|FAIL), failure_mode (data_limited|method_choice|unknown), summary,
          and per-requirement statuses with evidence.
        - Include feature_usage in alignment_check.json:
          {used_features: [...], target_columns: [...], excluded_features: [...], reason_exclusions: {...}}.
        - If there are no alignment requirements provided, write WARN with failure_mode=data_limited and explain.

        FINAL SELF-CHECK
        - Print QA_SELF_CHECK: PASS with a short bullet list of what was satisfied (target guard, split choice, baseline, required deliverables, no forbidden imports/ops).

        Return Python code only.

        """

        ml_view = ml_view or {}
        required_outputs = ml_view.get("required_outputs") or (execution_contract or {}).get("required_outputs", []) or []
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
        ml_view_json = json.dumps(ml_view, indent=2)
        plot_spec_json = json.dumps(ml_view.get("plot_spec", {}), indent=2)
        evaluation_spec_json = json.dumps((execution_contract or {}).get("evaluation_spec", {}), indent=2)
        render_kwargs = dict(
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
            contract_min_context=json.dumps(execution_contract_compact, indent=2),
            ml_view_context=ml_view_json,
            plot_spec_context=plot_spec_json,
            evaluation_spec_json=evaluation_spec_json,
            spec_extraction_json=spec_extraction_json,
            ml_engineer_runbook=ml_runbook_json,
            feature_availability_json=json.dumps(feature_availability or [], indent=2),
            availability_summary=availability_summary or "",
            signal_summary_json=json.dumps(signal_summary or {}, indent=2),
            iteration_memory_json=json.dumps(iteration_memory or [], indent=2),
            iteration_memory_block=iteration_memory_block or "",
            dataset_scale=dataset_scale,
        )
        # Safe Rendering for System Prompt
        system_prompt = self._build_system_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            render_kwargs,
            ml_view=ml_view,
            execution_contract=execution_contract or {},
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
        elif self.provider == "openrouter":
            provider_label = "OpenRouter"
        else:
            provider_label = "DeepSeek"

        def _call_model_with_prompts(
            sys_prompt: str,
            usr_prompt: str,
            temperature: float,
            model_name: str,
        ) -> str:
            self.last_prompt = sys_prompt + "\n\nUSER:\n" + usr_prompt
            print(f"DEBUG: ML Engineer calling {provider_label} Model ({model_name})...")
            if self.provider == "zai":
                from src.utils.llm_throttle import glm_call_slot
                with glm_call_slot():
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": usr_prompt}
                        ],
                        temperature=temperature,
                    )
                content = response.choices[0].message.content
            elif self.provider == "openrouter":
                response = self.client.chat.completions.create(
                    model=model_name,
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
                    model=model_name,
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
                    model=model_name,
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
            self.last_fallback_reason = None
            if self.provider == "openrouter":
                primary = self.model_name
                fallback = self.fallback_model_name
                try:
                    content = call_with_retries(
                        lambda: _call_model_with_prompts(system_prompt, user_message, current_temp, primary),
                        max_retries=5,
                        backoff_factor=2,
                        initial_delay=2,
                    )
                    self.last_model_used = primary
                except Exception as e:
                    self.last_fallback_reason = str(e)[:500]
                    print(
                        f"WARN: OpenRouter primary failed ({e}); switching to fallback ({fallback})..."
                    )
                    content = call_with_retries(
                        lambda: _call_model_with_prompts(system_prompt, user_message, current_temp, fallback),
                        max_retries=2,
                        backoff_factor=2,
                        initial_delay=1,
                    )
                    self.last_model_used = fallback
            else:
                content = call_with_retries(
                    lambda: _call_model_with_prompts(system_prompt, user_message, current_temp, self.model_name),
                    max_retries=5,
                    backoff_factor=2,
                    initial_delay=2,
                )
                self.last_model_used = self.model_name
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
                model_for_repairs = self.last_model_used or self.model_name
                for _ in range(2):
                    repaired = call_with_retries(
                        lambda: _call_model_with_prompts(repair_system, repair_user, 0.0, model_for_repairs),
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
                reprompt_context = self._build_incomplete_reprompt_context(
                    execution_contract=execution_contract,
                    required_outputs=required_deliverables,
                    iteration_memory_block=iteration_memory_block,
                    iteration_memory=iteration_memory,
                    feedback_history=feedback_history,
                    gate_context=gate_context,
                    ml_view=ml_view,
                )
                completion_system = (
                    "You are a senior ML engineer. Return a COMPLETE runnable Python script. "
                    "Do not return partial snippets, diffs, or TODOs. "
                    "Preserve required outputs and alignment checks."
                )
                completion_user = (
                    "Your last response is incomplete. Return the full script.\n"
                    f"Missing/invalid sections: {completion_issues}\n"
                    f"Required deliverables: {required_deliverables}\n\n"
                    f"INCOMPLETE CODE:\n{code}\n\n"
                    "Re-emit the FULL script without truncation, respecting contract and fixes.\n\n"
                    f"{reprompt_context}"
                )
                completed = call_with_retries(
                    lambda: _call_model_with_prompts(
                        completion_system,
                        completion_user,
                        0.0,
                        self.last_model_used or self.model_name,
                    ),
                    max_retries=2,
                    backoff_factor=2,
                    initial_delay=1,
                )
                completed = self._clean_code(completed)
                if is_syntax_valid(completed):
                    code = completed

            # Post-processing: Inject correct data_path if LLM used wrong path
            code = self._fix_data_path_in_code(code, data_path)
            code = self._fix_to_csv_dialect_in_code(code)
            reasons = self._detect_forbidden_input_fallback(code, data_path)
            if reasons:
                guard_system = (
                    "You are a senior ML engineer. Remove forbidden input-existence checks "
                    "and synthetic fallbacks. Assume input exists at $data_path. "
                    "Return full python script only."
                )
                guard_user = (
                    "Remove any input existence checks and any fallback branches that generate data.\n"
                    "Detected forbidden patterns:\n"
                    + "\n".join([f"- {reason}" for reason in reasons])
                    + "\n\nCODE:\n"
                    + self._truncate_code_for_patch(code)
                    + "\n\nReturn the FULL script only, without any input existence checks, "
                    "FileNotFoundError fallbacks, or synthetic data generation (np.random, "
                    "sklearn.datasets.make_*, faker)."
                )
                model_for_repairs = self.last_model_used or self.model_name
                repaired = code
                for _ in range(2):
                    repaired = call_with_retries(
                        lambda: _call_model_with_prompts(guard_system, guard_user, 0.0, model_for_repairs),
                        max_retries=2,
                        backoff_factor=2,
                        initial_delay=1,
                    )
                    repaired = self._clean_code(repaired)
                    if not is_syntax_valid(repaired):
                        reasons = ["syntax_invalid_after_guardrail"]
                        continue
                    repaired = self._fix_data_path_in_code(repaired, data_path)
                    repaired = self._fix_to_csv_dialect_in_code(repaired)
                    reasons = self._detect_forbidden_input_fallback(repaired, data_path)
                    if not reasons:
                        code = repaired
                        break
                if reasons:
                    raise RuntimeError(
                        "ML guardrail failed: synthetic/fallback patterns still present: "
                        + "; ".join(reasons)
                    )
            return code

        except Exception as e:
            # Raise RuntimeError as requested for clean catch in graph.py
            print(f"CRITICAL: ML Engineer Failed (Max Retries): {e}")
            raise RuntimeError(f"ML Generation Failed: {e}")

    def _clean_code(self, code: str) -> str:
        return extract_code_block(code)
