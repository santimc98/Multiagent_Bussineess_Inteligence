import os
import logging
from datetime import datetime
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
from src.utils.llm_fallback import call_chat_with_fallback

# NOTE: scan_code_safety referenced by tests as a required safety mechanism.
# ML code executes in sandbox; keep the reference for integration checks.
_scan_code_safety_ref = "scan_code_safety"

# ============================================================================
# ML PLAN SCHEMA CONSTANTS (TAREA 1)
# ============================================================================

REQUIRED_PLAN_KEYS = [
    "training_rows_policy",
    "train_filter",
    "metric_policy",
    "cv_policy",
    "scoring_policy",
    "leakage_policy",
    "notes",
    "plan_source",
]

DEFAULT_PLAN: Dict[str, Any] = {
    "training_rows_policy": "unspecified",
    "training_rows_rule": None,
    "split_column": None,
    "train_filter": {
        "type": "unspecified",
        "column": None,
        "value": None,
        "rule": None,
    },
    "metric_policy": {
        "primary_metric": "unspecified",
        "secondary_metrics": [],
        "report_with_cv": True,
        "notes": "",
    },
    "cv_policy": {
        "strategy": "unspecified",
        "n_splits": 5,
        "shuffle": True,
        "stratified": None,
        "notes": "",
    },
    "scoring_policy": {
        "generate_scores": True,
        "score_rows": "unspecified",
    },
    "leakage_policy": {
        "action": "unspecified",
        "flagged_columns": [],
        "notes": "",
    },
    "evidence": [],
    "assumptions": [],
    "open_questions": [],
    "notes": [],
    "evidence_used": {},  # Structured evidence digest for QA coherence checks
    "plan_source": "fallback",
}


class MLEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the ML Engineer Agent with the configured provider.
        """
        self.logger = logging.getLogger(__name__)
        self.provider = (os.getenv("ML_ENGINEER_PROVIDER", "openrouter") or "openrouter").strip().lower()
        self.fallback_model_name = None
        self.last_model_used = None
        self.last_fallback_reason = None
        self.last_training_policy_warnings = None
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
            self.model_name = (
                os.getenv("ML_ENGINEER_PRIMARY_MODEL")
                or os.getenv("OPENROUTER_ML_PRIMARY_MODEL")
                or "moonshotai/kimi-k2.5"
            )
            self.fallback_model_name = (
                os.getenv("ML_ENGINEER_FALLBACK_MODEL")
                or os.getenv("OPENROUTER_ML_FALLBACK_MODEL")
                or "z-ai/glm-4.7"
            )
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
        V4.1 CUTOVER: No legacy keys (data_requirements, spec_extraction, role_runbooks).
        """
        if not isinstance(contract, dict):
            return {}

        # V4.1 keys relevant for ML Engineer (no legacy keys)
        keep_keys = [
            "contract_version",
            "strategy_title",
            "business_objective",
            "canonical_columns",  # V4.1: replaces required_columns/data_requirements
            "required_outputs",
            "alignment_requirements",
            "business_alignment",
            "feature_semantics",
            "business_sanity_checks",
            "column_roles",
            "column_roles",
            # V4.1: availability_summary removed
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

        # V4.1: Use ml_engineer_runbook only, no legacy role_runbooks fallback

        return compact

    def _resolve_allowed_columns_for_prompt(self, contract: Dict[str, Any] | None) -> List[str]:
        """V4.1: Use canonical_columns only, no legacy required_columns."""
        if not isinstance(contract, dict):
            return []
        cols: List[str] = []
        # V4.1: Only use canonical_columns (no legacy required_columns)
        canonical = contract.get("canonical_columns")
        if isinstance(canonical, list):
            cols.extend([str(v) for v in canonical if v])
        # Also include derived_columns
        derived = contract.get("derived_columns")
        if isinstance(derived, list):
            for item in derived:
                if isinstance(item, str) and item:
                    cols.append(item)
                elif isinstance(item, dict) and item.get("name"):
                    cols.append(str(item.get("name")))
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
        """V4.1: Use artifact_requirements.file_schemas only, no legacy fallback."""
        if not isinstance(contract, dict):
            return []
        # V4.1: Use artifact_requirements.file_schemas
        artifact_reqs = contract.get("artifact_requirements", {})
        schema = artifact_reqs.get("file_schemas") if isinstance(artifact_reqs, dict) else {}
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
        from src.utils.context_pack import compress_long_lists

        contract_min = compress_long_lists(self._compact_execution_contract(execution_contract or {}))[0]
        allowed_columns = compress_long_lists(self._resolve_allowed_columns_for_prompt(execution_contract or {}))[0]
        allowed_patterns = self._resolve_allowed_name_patterns_for_prompt(execution_contract or {})
        feedback_blocks = self._select_feedback_blocks(feedback_history, gate_context, max_blocks=2)
        ml_view = ml_view or {}
        ml_view_json = json.dumps(compress_long_lists(ml_view)[0], indent=2)
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
        """
        DEPRECATED: Static syntax checks are no longer enforced.

        Validation Philosophy Change (v5.0):
        - Instead of checking if code SAYS .read_csv, we let code RUN and check RESULTS.
        - The Reviewer/QA Agent validates outputs exist and are correct.
        - This method returns empty list to maintain API compatibility.
        """
        # Static checks removed - trust execution-based validation
        return []

    def _extract_training_context(
        self,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
        ml_plan: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        contract = execution_contract or {}
        view = ml_view or {}
        plan = ml_plan or {}

        outcome_columns = contract.get("outcome_columns")
        if not isinstance(outcome_columns, list) or not outcome_columns:
            outcome_columns = view.get("outcome_columns")
        if not isinstance(outcome_columns, list):
            outcome_columns = []

        target_column = None
        if outcome_columns:
            target_column = str(outcome_columns[0])

        training_rows_rule = contract.get("training_rows_rule") or view.get("training_rows_rule")
        scoring_rows_rule = contract.get("scoring_rows_rule") or view.get("scoring_rows_rule")
        secondary_scoring_subset = contract.get("secondary_scoring_subset") or view.get("secondary_scoring_subset")

        training_rows_policy = plan.get("training_rows_policy")
        split_column = plan.get("split_column")
        train_filter = plan.get("train_filter") if isinstance(plan.get("train_filter"), dict) else None

        canonical_cols = contract.get("canonical_columns")
        if not isinstance(canonical_cols, list):
            canonical_cols = view.get("canonical_columns")
        if not isinstance(canonical_cols, list):
            canonical_cols = []

        dataset_semantics = contract.get("dataset_semantics") if isinstance(contract.get("dataset_semantics"), dict) else {}
        partition_notes = dataset_semantics.get("data_partitioning_notes") if isinstance(dataset_semantics.get("data_partitioning_notes"), list) else []
        if not isinstance(split_column, str) or not split_column.strip():
            split_column = None
        if split_column is None:
            evidence = plan.get("evidence_used") if isinstance(plan.get("evidence_used"), dict) else {}
            split_candidates = evidence.get("split_candidates")
            if isinstance(split_candidates, list):
                for cand in split_candidates:
                    if not isinstance(cand, dict):
                        continue
                    col = cand.get("column")
                    if isinstance(col, str) and col.strip():
                        split_column = col.strip()
                        break
        if split_column is None and "__split" in canonical_cols:
            split_column = "__split"

        if not isinstance(training_rows_policy, str) or not training_rows_policy.strip():
            notes_text = " ".join(str(note) for note in partition_notes).lower()
            if split_column and ("split" in notes_text or "train/test" in notes_text or "train test" in notes_text):
                training_rows_policy = "use_split_column"
            elif isinstance(training_rows_rule, str):
                rule_lower = training_rows_rule.lower()
                if any(token in rule_lower for token in ("not missing", "not null", "notna", "non-null")):
                    training_rows_policy = "only_rows_with_label"
            if not training_rows_policy:
                training_rows_policy = "use_all_rows"

        if not isinstance(train_filter, dict):
            train_filter = None
        if train_filter is None:
            if training_rows_policy == "use_split_column" and split_column:
                train_filter = {
                    "type": "split_equals",
                    "column": split_column,
                    "value": "train",
                    "rule": None,
                }
            elif training_rows_policy == "only_rows_with_label" and target_column:
                train_filter = {
                    "type": "label_not_null",
                    "column": target_column,
                    "value": None,
                    "rule": None,
                }

        return {
            "target_column": target_column,
            "training_rows_rule": training_rows_rule,
            "scoring_rows_rule": scoring_rows_rule,
            "secondary_scoring_subset": secondary_scoring_subset,
            "training_rows_policy": training_rows_policy,
            "split_column": split_column,
            "train_filter": train_filter,
        }

    def _code_mentions_label_filter(self, code: str, target: str | None) -> bool:
        """
        DEPRECATED: Static regex-based code analysis is no longer enforced.

        Validation Philosophy Change (v5.0):
        - We no longer parse code looking for .dropna/.notna patterns.
        - Execution-based validation: run the code, check if output has nulls.
        - Always returns True to avoid blocking valid implementations.
        """
        return True  # Trust the LLM; validate results, not syntax

    def _code_mentions_split_usage(self, code: str, split_column: str | None) -> bool:
        """
        DEPRECATED: Static regex-based code analysis is no longer enforced.

        Validation Philosophy Change (v5.0):
        - We no longer parse code looking for split column references.
        - Execution-based validation: run the code, check train/test split in results.
        - Always returns True to avoid blocking valid implementations.
        """
        return True  # Trust the LLM; validate results, not syntax

    def _check_training_policy_compliance(
        self,
        code: str,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
        ml_plan: Dict[str, Any] | None,
    ) -> List[str]:
        """
        DEPRECATED: Static AST/regex-based policy compliance checks are no longer enforced.

        Validation Philosophy Change (v5.0):
        =====================================
        OLD: Parse code text looking for .dropna(), split column references, etc.
             Flag code as invalid if specific patterns weren't found.
             Problem: Brittle - flags valid implementations that use different syntax.

        NEW: Execution-based validation.
             - Let the code RUN in the sandbox.
             - Reviewer/QA Agent checks if RESULTS are correct:
               * Does the output have nulls where it shouldn't?
               * Was train/test split done correctly?
               * Are metrics reasonable?
             - Self-correction: ML Engineer receives stderr/traceback and fixes errors.

        This method returns empty list to maintain API compatibility while
        removing the static linting that blocked valid code.
        """
        # Static checks removed - execution-based validation is now the standard
        # The Reviewer and QA agents validate results, not code syntax
        return []

    def _describe_train_filter(
        self,
        train_filter: Dict[str, Any] | None,
        target_column: str | None,
        split_column: str | None,
    ) -> str:
        if not isinstance(train_filter, dict):
            return "No explicit train_filter provided."
        tf_type = str(train_filter.get("type") or "").strip().lower()
        column = train_filter.get("column") or (target_column if tf_type == "label_not_null" else split_column)
        value = train_filter.get("value")
        rule = train_filter.get("rule")
        if tf_type == "label_not_null":
            return f"Filter training rows where '{column}' is not null."
        if tf_type == "split_equals":
            return f"Filter training rows where '{column}' == '{value or 'train'}'."
        if tf_type == "custom_rule":
            return f"Apply custom training rule: {rule}"
        if tf_type == "none":
            return "Use all rows for training."
        return "Train filter is unspecified; infer from plan/contract."

    def _build_training_policy_checklist(
        self,
        code: str,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
        ml_plan: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        context = self._extract_training_context(execution_contract, ml_view, ml_plan)
        plan = ml_plan or {}
        metric_policy = plan.get("metric_policy", {}) if isinstance(plan.get("metric_policy"), dict) else {}
        cv_policy = plan.get("cv_policy", {}) if isinstance(plan.get("cv_policy"), dict) else {}
        target = context.get("target_column")
        split_column = context.get("split_column")
        return {
            "target_column": target,
            "training_rows_policy": context.get("training_rows_policy"),
            "train_filter": context.get("train_filter"),
            "code_has_label_filter": self._code_mentions_label_filter(code, target),
            "code_has_split_filter": self._code_mentions_split_usage(code, split_column),
            "primary_metric": metric_policy.get("primary_metric"),
            "cv_policy": {
                "strategy": cv_policy.get("strategy"),
                "n_splits": cv_policy.get("n_splits"),
                "shuffle": cv_policy.get("shuffle"),
                "stratified": cv_policy.get("stratified"),
            },
        }

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

    def _execute_llm_call(self, sys_prompt: str, usr_prompt: str, temperature: float = 0.1) -> str:
        """Helper to execute LLM call with current provider."""
        model_name = self.model_name
        
        # Determine label
        if self.provider in {"google", "gemini"}:
            provider_label = "Google"
        elif self.provider == "zai":
            provider_label = "Z.ai"
        elif self.provider == "openrouter":
            provider_label = "OpenRouter"
        else:
            provider_label = "DeepSeek"
            
        self.last_prompt = sys_prompt + "\n\nUSER:\n" + usr_prompt
        print(f"DEBUG: ML Engineer (Plan) calling {provider_label} Model ({model_name})...")
        
        try:
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
                return response.choices[0].message.content
            elif self.provider == "openrouter":
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt},
                ]
                response, model_used = call_chat_with_fallback(
                    self.client,
                    messages,
                    [self.model_name, self.fallback_model_name],
                    call_kwargs={"temperature": temperature},
                    logger=self.logger,
                    context_tag="ml_engineer_plan",
                )
                self.last_model_used = model_used
                self.logger.info("ML_ENGINEER_MODEL_USED: %s", model_used)
                return response.choices[0].message.content
            elif self.provider in {"google", "gemini"}:
                full_prompt = sys_prompt + "\n\nUSER INPUT:\n" + usr_prompt
                from google.genai import types
                
                # Thinking config logic
                thinking_level = (os.getenv("ML_ENGINEER_THINKING_LEVEL") or "high").strip().lower()
                if thinking_level not in {"minimal", "low", "medium", "high"}:
                    thinking_level = "high"
                    
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
                return getattr(response, "text", "")
            else:
                # Deepseek or fallback
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt}
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content
        except Exception as e:
            # Check for 504
            if "504" in str(e):
                raise ConnectionError("LLM Server Timeout (504 Received)")
            raise e

    def generate_ml_plan(
        self,
        data_profile: Dict[str, Any],
        execution_contract: Dict[str, Any] | None = None,
        strategy: Dict[str, Any] | None = None,
        business_objective: str = "",
        llm_call: Any = None,
    ) -> Dict[str, Any]:
        """
        Generate ml_plan.json using LLM reasoning (Facts -> Plan).

        NEVER returns {} - always returns a complete plan with all REQUIRED_PLAN_KEYS.

        Args:
            data_profile: Data profile with facts (outcome_analysis, split_candidates, etc.)
            execution_contract: V4.1 execution contract
            strategy: Selected strategy dict
            business_objective: Business objective string
            llm_call: Optional callable for LLM call injection (for testing).
                      Signature: llm_call(system_prompt, user_prompt) -> str

        Returns:
            Complete ML plan dict (never empty)
        """
        import copy
        contract = execution_contract or {}
        profile = data_profile or {}
        strategy_dict = strategy or {}

        PLAN_PROMPT = """
You are a Senior ML Engineer. Your task is to reason about the data facts and contract requirements to produce a Robust ML Plan.

*** DATA FACTS (Facts Only - CANONICAL EVIDENCE) ***
$data_profile_json

*** EXECUTION CONTRACT ***
$execution_contract_json

*** STRATEGY ***
$strategy_json

*** BUSINESS OBJECTIVE ***
"$business_objective"

*** UNIVERSAL CONSTRAINTS (apply always) ***
1. If outcome_analysis shows any outcome column with null_frac > 0, you CANNOT use "use_all_rows" for training_rows_policy.
   Reason: You cannot compute loss/metric on rows without labels.
2. METRIC SOURCE PRIORITY: Use contract.validation_requirements.primary_metric if set. Otherwise use evaluation_spec.primary_metric.
   If qa_gates specify a metric, use that exact metric name. Do NOT invent or normalize metric names.
3. If leakage_flags exist in data_profile, leakage_policy.action should be "exclude_flagged_columns" unless contract explicitly allows them.
4. If no primary metric is specified anywhere, infer a minimal metric from analysis_type and state this explicitly in notes.

*** INSTRUCTIONS ***
1. Analyze 'outcome_analysis' to check for partial labels. If null_frac > 0 for any outcome, set training_rows_policy to "only_rows_with_label".
2. Check 'split_candidates'. If a split column exists AND has values like 'train'/'test', consider "use_split_column". If you choose NOT to use it, explain why in evidence_used.split_evaluation.
3. Check contract for primary_metric (validation_requirements.primary_metric or evaluation_spec primary metric / qa_gates metric). Use that exact metric name.
4. DO NOT invent rules. Base every decision on 'evidence' found in the data_profile.
5. CRITICAL: Populate 'evidence_used' with STRUCTURED facts you used for decisions. This enables QA to verify coherence.
6. If you choose "only_rows_with_label", set train_filter.type="label_not_null" and train_filter.column=target.
7. If you choose "use_split_column", set train_filter.type="split_equals", train_filter.column=split_column, and train_filter.value to the training value (e.g., "train").
8. Avoid ambiguity: select ONE training policy and make it explicit via train_filter.

*** REQUIRED OUTPUT (JSON ONLY, NO MARKDOWN) ***
{
  "training_rows_policy": "use_all_rows | only_rows_with_label | use_split_column | custom",
  "training_rows_rule": "string rule if custom, or null",
  "split_column": "col_name or null",
  "train_filter": {
      "type": "none | label_not_null | split_equals | custom_rule",
      "column": "column name or null",
      "value": "value for split_equals or null",
      "rule": "rule string for custom_rule or null"
  },
  "metric_policy": {
      "primary_metric": "metric_name",
      "secondary_metrics": [],
      "report_with_cv": true,
      "notes": "brief justification"
  },
  "cv_policy": {
      "strategy": "StratifiedKFold | KFold | TimeSeriesSplit | GroupKFold | auto",
      "n_splits": 5,
      "shuffle": true,
      "stratified": true,
      "notes": "brief justification"
  },
  "scoring_policy": {
      "generate_scores": true,
      "score_rows": "all | labeled_only | unlabeled_only"
  },
  "leakage_policy": {
      "action": "none | exclude_flagged_columns | manual_review",
      "flagged_columns": [],
      "notes": "brief justification"
  },
  "evidence_used": {
      "outcome_null_frac": {"column": "target_col", "null_frac": 0.3},
      "split_candidates": [{"column": "__split", "values": ["train", "test"]}],
      "split_evaluation": "used split column because..." or "ignored split because...",
      "contract_primary_metric": "roc_auc or null if not specified",
      "analysis_type": "classification"
  },
  "evidence": ["fact1 from profile", "fact2 from profile"],
  "assumptions": [],
  "open_questions": [],
  "notes": ["brief reasoning notes"]
}
"""

        # Check if we can make LLM calls
        has_llm_init = getattr(self, "model_name", None) is not None
        can_execute_llm = has_llm_init or llm_call is not None

        if not can_execute_llm:
            # Agent not initialized, cannot call LLM
            result = self._normalize_ml_plan(None, source="missing_llm_init")
            result = self._derive_train_filter(result, profile, contract)
            result["notes"] = ["Agent not initialized; cannot call LLM. Using fallback defaults."]
            return result

        from src.utils.context_pack import compress_long_lists

        # Prepare context
        try:
            render_kwargs = {
                "data_profile_json": json.dumps(compress_long_lists(profile)[0], indent=2),
                "execution_contract_json": json.dumps(self._compact_execution_contract(contract), indent=2),
                "strategy_json": json.dumps(strategy_dict, indent=2),
                "business_objective": business_objective,
            }
        except Exception as render_err:
            result = self._normalize_ml_plan(None, source="render_error")
            result = self._derive_train_filter(result, profile, contract)
            result["notes"] = [f"Failed to render prompt context: {render_err}"]
            return result

        system_prompt = render_prompt(PLAN_PROMPT, **render_kwargs)
        user_prompt = "Generate the ML Plan JSON now."

        # Determine which LLM call function to use
        if llm_call is not None:
            execute_call = llm_call
        else:
            execute_call = lambda sys, usr: self._execute_llm_call(sys, usr, temperature=0.1)

        try:
            # First attempt
            response = execute_call(system_prompt, user_prompt)
            if hasattr(self, 'last_response'):
                self.last_response = response
            parsed = self._parse_json_response(response)
            ml_plan = self._normalize_ml_plan(parsed, source="llm")
            ml_plan = self._apply_contract_metric_override(ml_plan, contract)

            # Check if we got meaningful data
            if ml_plan.get("training_rows_policy") != "unspecified":
                ml_plan = self._derive_train_filter(ml_plan, profile, contract)
                return ml_plan

            # Retry attempt
            print("Warning: ML Plan generation returned incomplete JSON, retrying...")
            response = execute_call(system_prompt, user_prompt + "\nCRITICAL: Return VALID JSON ONLY. No markdown.")
            if hasattr(self, 'last_response'):
                self.last_response = response
            parsed = self._parse_json_response(response)
            ml_plan = self._normalize_ml_plan(parsed, source="llm_retry")
            ml_plan = self._apply_contract_metric_override(ml_plan, contract)
            ml_plan = self._derive_train_filter(ml_plan, profile, contract)

            return ml_plan

        except Exception as e:
            print(f"ML Engineer Plan Gen Error: {e}")
            result = self._normalize_ml_plan(None, source="llm_error")
            result = self._derive_train_filter(result, profile, contract)
            result["notes"] = [f"LLM call failed: {e}"]
            return result

    def _apply_contract_metric_override(
        self,
        ml_plan: Dict[str, Any],
        contract: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(ml_plan, dict):
            return ml_plan
        contract_metric = self._resolve_contract_primary_metric(contract)
        if not contract_metric:
            return ml_plan
        metric_policy = ml_plan.get("metric_policy")
        if not isinstance(metric_policy, dict):
            metric_policy = {}
            ml_plan["metric_policy"] = metric_policy
        current = str(metric_policy.get("primary_metric") or "").strip()
        if not current or current.lower() == "unspecified" or current != contract_metric:
            metric_policy["primary_metric"] = contract_metric
            notes = ml_plan.get("notes")
            if not isinstance(notes, list):
                notes = []
            notes.append(f"Primary metric aligned to contract: {contract_metric}")
            ml_plan["notes"] = notes
        evidence = ml_plan.get("evidence_used")
        if isinstance(evidence, dict):
            evidence["contract_primary_metric"] = contract_metric
        return ml_plan

    def _resolve_contract_primary_metric(self, contract: Dict[str, Any] | None) -> str | None:
        if not isinstance(contract, dict):
            return None
        validation = contract.get("validation_requirements")
        if isinstance(validation, dict):
            primary = validation.get("primary_metric")
            if isinstance(primary, str) and primary.strip():
                return primary.strip()
        evaluation_spec = contract.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            validation = evaluation_spec.get("validation_requirements")
            if isinstance(validation, dict):
                primary = validation.get("primary_metric")
                if isinstance(primary, str) and primary.strip():
                    return primary.strip()
            primary = evaluation_spec.get("primary_metric")
            if isinstance(primary, str) and primary.strip():
                return primary.strip()
            qa_gates = evaluation_spec.get("qa_gates")
            if isinstance(qa_gates, list):
                for gate in qa_gates:
                    if not isinstance(gate, dict):
                        continue
                    params = gate.get("params")
                    if isinstance(params, dict):
                        metric = params.get("metric")
                        if isinstance(metric, str) and metric.strip():
                            return metric.strip()
        return None

    def _parse_json_response(self, text: str) -> Any:
        try:
            cleaned = text.strip()
            # Remove markdown fences
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                # Fallback for generic block
                parts = cleaned.split("```")
                if len(parts) >= 2:
                     cleaned = parts[1].strip()
            
            # More robust regex
            import re
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            return json.loads(cleaned)
        except Exception:
            return None

    def _normalize_ml_plan(self, parsed: Any, source: str = "llm") -> Dict[str, Any]:
        """
        Normalize parsed LLM response to a complete ML plan.

        NEVER returns {} or None - always returns a complete plan with all required keys.

        Args:
            parsed: Raw parsed response (can be dict, list, None, etc.)
            source: Plan source label (e.g., "llm", "fallback", "missing_llm_init")

        Returns:
            Complete ML plan dict with all REQUIRED_PLAN_KEYS
        """
        import copy

        # Start with a copy of DEFAULT_PLAN
        result = copy.deepcopy(DEFAULT_PLAN)
        result["plan_source"] = source

        # Handle list input
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                result["notes"] = ["Parsed response was list but could not extract dict"]
                return result

        # Handle non-dict input
        if not isinstance(parsed, dict):
            result["notes"] = [f"Parsed response was {type(parsed).__name__}, using defaults"]
            return result

        # Merge parsed values into result
        # training_rows_policy
        if "training_rows_policy" in parsed and isinstance(parsed["training_rows_policy"], str):
            result["training_rows_policy"] = parsed["training_rows_policy"]

        # training_rows_rule
        if "training_rows_rule" in parsed:
            result["training_rows_rule"] = parsed["training_rows_rule"]

        # split_column
        if "split_column" in parsed:
            result["split_column"] = parsed["split_column"]

        # train_filter
        if "train_filter" in parsed:
            tf = parsed["train_filter"]
            if isinstance(tf, dict):
                result["train_filter"] = tf
            elif isinstance(tf, str):
                result["train_filter"] = {
                    "type": "custom_rule",
                    "column": None,
                    "value": None,
                    "rule": tf,
                }

        # metric_policy
        if "metric_policy" in parsed:
            mp = parsed["metric_policy"]
            if isinstance(mp, dict):
                if "primary_metric" in mp:
                    result["metric_policy"]["primary_metric"] = str(mp["primary_metric"])
                if "secondary_metrics" in mp and isinstance(mp["secondary_metrics"], list):
                    result["metric_policy"]["secondary_metrics"] = mp["secondary_metrics"]
                if "report_with_cv" in mp:
                    result["metric_policy"]["report_with_cv"] = bool(mp["report_with_cv"])
                if "notes" in mp:
                    result["metric_policy"]["notes"] = str(mp.get("notes", ""))
            elif isinstance(mp, str):
                # Handle case where metric_policy is just a string
                result["metric_policy"]["primary_metric"] = mp

        # cv_policy
        if "cv_policy" in parsed:
            cp = parsed["cv_policy"]
            if isinstance(cp, dict):
                if "strategy" in cp:
                    result["cv_policy"]["strategy"] = str(cp["strategy"])
                if "n_splits" in cp:
                    try:
                        result["cv_policy"]["n_splits"] = int(cp["n_splits"])
                    except (ValueError, TypeError):
                        pass
                if "shuffle" in cp:
                    result["cv_policy"]["shuffle"] = bool(cp["shuffle"])
                if "stratified" in cp:
                    result["cv_policy"]["stratified"] = cp["stratified"]
                if "notes" in cp:
                    result["cv_policy"]["notes"] = str(cp.get("notes", ""))

        # scoring_policy
        if "scoring_policy" in parsed:
            sp = parsed["scoring_policy"]
            if isinstance(sp, dict):
                if "generate_scores" in sp:
                    result["scoring_policy"]["generate_scores"] = bool(sp["generate_scores"])
                if "score_rows" in sp:
                    result["scoring_policy"]["score_rows"] = str(sp["score_rows"])

        # leakage_policy
        if "leakage_policy" in parsed:
            lp = parsed["leakage_policy"]
            if isinstance(lp, dict):
                if "action" in lp:
                    result["leakage_policy"]["action"] = str(lp["action"])
                if "flagged_columns" in lp and isinstance(lp["flagged_columns"], list):
                    result["leakage_policy"]["flagged_columns"] = lp["flagged_columns"]
                if "notes" in lp:
                    result["leakage_policy"]["notes"] = str(lp.get("notes", ""))

        # evidence
        if "evidence" in parsed:
            ev = parsed["evidence"]
            if isinstance(ev, list):
                result["evidence"] = ev
            elif isinstance(ev, str):
                result["evidence"] = [ev]

        # assumptions
        if "assumptions" in parsed:
            assum = parsed["assumptions"]
            if isinstance(assum, list):
                result["assumptions"] = assum
            elif isinstance(assum, str):
                result["assumptions"] = [assum]

        # open_questions
        if "open_questions" in parsed:
            oq = parsed["open_questions"]
            if isinstance(oq, list):
                result["open_questions"] = oq
            elif isinstance(oq, str):
                result["open_questions"] = [oq]

        # notes
        if "notes" in parsed:
            notes = parsed["notes"]
            if isinstance(notes, list):
                result["notes"] = notes
            elif isinstance(notes, str):
                result["notes"] = [notes]

        # evidence_used (structured evidence digest for QA coherence checks)
        if "evidence_used" in parsed:
            eu = parsed["evidence_used"]
            if isinstance(eu, dict):
                result["evidence_used"] = eu
            else:
                result["evidence_used"] = {}
        else:
            # Ensure evidence_used always exists (even if empty) for QA
            result["evidence_used"] = {}

        # Set plan_source to "llm" if we got valid data
        if result["training_rows_policy"] != "unspecified":
            result["plan_source"] = source

        return result

    def _infer_outcome_column(
        self,
        execution_contract: Dict[str, Any] | None,
        data_profile: Dict[str, Any] | None,
    ) -> str | None:
        contract = execution_contract or {}
        profile = data_profile or {}
        outcome_columns = contract.get("outcome_columns")
        if isinstance(outcome_columns, list) and outcome_columns:
            return str(outcome_columns[0])
        outcome_analysis = profile.get("outcome_analysis", {})
        if isinstance(outcome_analysis, dict) and outcome_analysis:
            for key in outcome_analysis.keys():
                return str(key)
        return None

    def _split_candidate_info(
        self,
        data_profile: Dict[str, Any] | None,
        preferred_column: str | None = None,
    ) -> Dict[str, Any]:
        profile = data_profile or {}
        split_candidates = profile.get("split_candidates", [])
        if not isinstance(split_candidates, list):
            return {"column": None, "values": [], "has_train_test": False}

        def _candidate_values(candidate: Dict[str, Any]) -> List[str]:
            values = candidate.get("values")
            if isinstance(values, list):
                return [str(v) for v in values]
            sample = candidate.get("unique_values_sample")
            if isinstance(sample, list):
                return [str(v) for v in sample]
            uniques = candidate.get("unique_values")
            if isinstance(uniques, list):
                return [str(v) for v in uniques]
            return []

        def _has_train_test(values: List[str]) -> bool:
            lowered = {v.strip().lower() for v in values if isinstance(v, str)}
            return "train" in lowered and "test" in lowered

        for cand in split_candidates:
            if not isinstance(cand, dict):
                continue
            col = cand.get("column")
            if preferred_column and str(col) != str(preferred_column):
                continue
            values = _candidate_values(cand)
            return {
                "column": str(col) if col is not None else None,
                "values": values,
                "has_train_test": _has_train_test(values),
            }

        # Fallback: first candidate
        for cand in split_candidates:
            if not isinstance(cand, dict):
                continue
            col = cand.get("column")
            values = _candidate_values(cand)
            return {
                "column": str(col) if col is not None else None,
                "values": values,
                "has_train_test": _has_train_test(values),
            }

        return {"column": None, "values": [], "has_train_test": False}

    def _derive_train_filter(
        self,
        plan: Dict[str, Any],
        data_profile: Dict[str, Any],
        execution_contract: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        import copy

        result = copy.deepcopy(plan)
        result.setdefault("train_filter", {})
        train_filter = result.get("train_filter")
        if not isinstance(train_filter, dict):
            train_filter = {}

        tf_type = str(train_filter.get("type") or "").strip().lower()
        if tf_type in {"", "unspecified"}:
            tf_type = ""

        outcome_col = self._infer_outcome_column(execution_contract, data_profile)
        outcome_analysis = (data_profile or {}).get("outcome_analysis", {})
        has_null_labels = False
        if isinstance(outcome_analysis, dict):
            for _, analysis in outcome_analysis.items():
                if isinstance(analysis, dict) and analysis.get("present") and analysis.get("null_frac", 0) > 0:
                    has_null_labels = True
                    break

        training_rows_policy = str(result.get("training_rows_policy") or "").strip().lower()
        training_rows_rule = result.get("training_rows_rule")
        split_column = result.get("split_column")
        split_info = self._split_candidate_info(data_profile, split_column)
        if split_column is None and split_info.get("column"):
            split_column = split_info["column"]

        evidence_used = result.get("evidence_used") if isinstance(result.get("evidence_used"), dict) else {}
        split_eval = str(evidence_used.get("split_evaluation", "")).lower()
        uses_split = "split" in split_eval and "use" in split_eval

        if tf_type == "custom_rule" and not train_filter.get("rule") and isinstance(training_rows_rule, str):
            train_filter["rule"] = training_rows_rule

        known_policies = {"use_all_rows", "only_rows_with_label", "use_split_column", "custom", "unspecified"}
        if not tf_type:
            if training_rows_policy and training_rows_policy not in known_policies:
                # Preserve explicit custom policy from LLM; do not infer a filter
                result["train_filter"] = train_filter
                return result
            if training_rows_policy == "custom" and not isinstance(training_rows_rule, str):
                # Preserve explicit custom policy without forcing defaults
                result["train_filter"] = train_filter
                return result

        if tf_type:
            # normalize column/value if missing
            if tf_type == "label_not_null":
                train_filter.setdefault("column", outcome_col)
                train_filter.setdefault("value", None)
                train_filter.setdefault("rule", None)
            elif tf_type == "split_equals":
                train_filter.setdefault("column", split_column)
                if train_filter.get("value") is None and split_info.get("has_train_test"):
                    train_filter["value"] = "train"
                train_filter.setdefault("rule", None)
            elif tf_type == "custom_rule":
                train_filter.setdefault("column", outcome_col)
            elif tf_type == "none":
                train_filter.setdefault("column", None)
                train_filter.setdefault("value", None)
                train_filter.setdefault("rule", None)
        else:
            # Derive a deterministic filter to remove ambiguity
            if isinstance(training_rows_rule, str) and training_rows_rule.strip() and training_rows_policy not in {"use_all_rows", "only_rows_with_label", "use_split_column"}:
                train_filter = {
                    "type": "custom_rule",
                    "column": outcome_col,
                    "value": None,
                    "rule": training_rows_rule,
                }
            elif training_rows_policy == "custom" and isinstance(training_rows_rule, str):
                train_filter = {
                    "type": "custom_rule",
                    "column": outcome_col,
                    "value": None,
                    "rule": training_rows_rule,
                }
            elif split_column and split_info.get("has_train_test") and (training_rows_policy == "use_split_column" or uses_split):
                train_filter = {
                    "type": "split_equals",
                    "column": split_column,
                    "value": "train",
                    "rule": None,
                }
            elif split_column and split_info.get("has_train_test") and has_null_labels:
                # Avoid ambiguity: split is the explicit training mask when train/test is present
                train_filter = {
                    "type": "split_equals",
                    "column": split_column,
                    "value": "train",
                    "rule": None,
                }
            elif has_null_labels:
                train_filter = {
                    "type": "label_not_null",
                    "column": outcome_col,
                    "value": None,
                    "rule": None,
                }
            else:
                train_filter = {
                    "type": "none",
                    "column": None,
                    "value": None,
                    "rule": None,
                }

        result["train_filter"] = train_filter

        # Align training_rows_policy with train_filter to remove ambiguity
        tf_type = str(train_filter.get("type") or "").strip().lower()
        policy_is_known = training_rows_policy in known_policies or not training_rows_policy
        if policy_is_known:
            if tf_type == "split_equals":
                result["training_rows_policy"] = "use_split_column"
                if split_column:
                    result["split_column"] = split_column
                if isinstance(training_rows_rule, str):
                    rule_lower = training_rows_rule.lower()
                    if any(token in rule_lower for token in ("not missing", "not null", "notna", "non-null")):
                        result["training_rows_rule"] = f"rows where {split_column} == 'train'"
                elif split_column:
                    result["training_rows_rule"] = f"rows where {split_column} == 'train'"
            elif tf_type == "label_not_null":
                result["training_rows_policy"] = "only_rows_with_label"
                if isinstance(training_rows_rule, str) is False and outcome_col:
                    result["training_rows_rule"] = f"rows where {outcome_col} is not missing"
            elif tf_type == "none":
                result["training_rows_policy"] = "use_all_rows"
            elif tf_type == "custom_rule":
                result["training_rows_policy"] = "custom"

        return result

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
        ml_plan: Dict[str, Any] | None = None,
        # V4.1: availability_summary parameter removed
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
         - CRITICAL: If 'ML_PLAN_CONTEXT' is present (in Data Audit), you MUST implement that plan exactly (training_rows_policy, train_filter, metric_policy, cv_policy). Do not deviate.

         TRAINING DATA SELECTION (STEWARD-DRIVEN)
         - Read execution_contract (or contract_min) for outcome_columns and optional fields:
           * training_rows_rule
           * scoring_rows_rule (primary)
           * secondary_scoring_subset (optional)
           * data_partitioning_notes
         - If those rules exist, implement them exactly.
         - If ML_PLAN_CONTEXT provides train_filter, treat it as authoritative and implement it exactly.
         - If training_rows_policy is "only_rows_with_label" or "use_split_column", your code must explicitly filter train_df before fit().
         - Scoring guidance:
           * Always produce scored_rows.csv for the primary scoring_rows_rule.
           * If scoring_rows_rule says "use all rows", score all rows.
           * If secondary_scoring_subset exists, keep scored_rows.csv for the primary rule and either:
             - add a labeled/unlabeled flag column (only if allowed by the contract schema), or
             - emit a separate artifact and document it.
         - If rules are absent, use Steward outputs (dataset_training_mask.json / dataset_semantics.json) when provided. Do NOT invent target or split logic.
         - If no target is provided by contract or Steward artifacts, stop and report ERROR (contract/inputs missing); do not guess.
         - Do NOT hardcode column names. Use the target/partition columns from contract or Steward outputs.
         - Safety airbag: before training, if y contains missing values, filter train_mask = y.notna() and log it (this does NOT choose the target).
         - Decision Log must include:
           * Target chosen
           * Training rows rule
           * Scoring rows rule
           * Secondary scoring subset (if any)
           * Evidence (e.g., missingness from dataset_semantics.json or dataset_training_mask.json)

         WIDE DATASET HANDLING (NO COLUMN ENUMERATION)
         - If data/column_sets.json exists, use src.utils.column_sets.expand_column_sets at runtime to build feature lists.
         - Avoid enumerating hundreds of columns in code or prompts; rely on selectors + explicit_columns in column_sets.json.
         - If column_sets.json is missing, fall back to contract-driven column selection logic.
         - Do NOT hardcode prefixes (e.g., startswith("pixel") or similar). Always use selectors or data-driven inference.
         - If no selectors are available, use all numeric columns except explicit outcome/id/partition columns and forbidden/audit-only features.
         - Decision Log must include n_features_used and feature_source ("column_sets" or "fallback_numeric").

         ROW-LEVEL EXPLANATIONS (MANDATORY IF CONTRACT REQUIRES DRIVERS/EXPLANATIONS)
         - Each row must have top_drivers with 2-3 drivers and direction (e.g., "Age low (down), Income high (up)").
         - Allowed methods (choose ONE, justify in comments):
           A) Linear/logistic model: rank by coefficient * feature_value (post-processing space).
           B) Tree/boosting: use global feature_importances and adjust by row deviation vs median.
           C) Simple rule-based: use z-scores of key features when no interpretable model is available.
         - Record in metrics.json (or a small artifact):
           * explanation_method
           * features_used_for_explanations (max 10)
         - If the method yields identical drivers for all rows, add context-by-values so row-level drivers differ.

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
        6) Do NOT mutate the input dataframe in-place. Use df_in for the raw load. If you need derived columns, create df_work = df_in.copy() and assign ONLY columns explicitly declared as derived in the Execution Contract (contract.derived_columns). If a required input column is missing, raise ValueError (no dummy values).
        6b) Do NOT overwrite data/cleaned_data.csv. Treat it as immutable input; write derived datasets to data/model_input.csv or data/features.csv.
        7) NEVER fabricate synthetic rows/features (pd.DataFrame({}) from literals, faker, sklearn.datasets.make_*, etc.).
           - Bootstrap/CV resampling of the OBSERVED rows is allowed (and expected when validation_requirements asks for bootstrap).
           - Randomness is permitted ONLY for resampling indices; do not generate new feature values from distributions.
        8) scored_rows.csv may include canonical columns plus contract-approved derived outputs ONLY if explicitly declared in the contract.
           Acceptable sources: artifact_requirements.scored_rows_schema (derived_columns/required_columns/allowed_extra_columns/allowed_name_patterns) or execution_contract.derived_columns.
           Any other derived columns must go to a separate artifact file.
        9) Start the script with a short comment block labeled PLAN describing: (1) dialect loading from cleaning_manifest.json, (2) detected columns, (3) row_id construction, (4) scored_rows columns, and (5) where extra derived artifacts go.
        10) Define CONTRACT_COLUMNS from the Execution Contract (use canonical_columns) and validate they exist in df_in; raise ValueError listing missing columns.
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
        - Sklearn scoring: Prefer built-in string scorers ONLY when they match the contract/plan metric.
          If the required metric is not available as a string (e.g., RMSLE, RMSE_log1p), compute it explicitly
          after predictions. Do NOT substitute a different metric to fit a scoring string.

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
        - Evaluation Spec: $evaluation_spec_json
        - Feature Semantics: $feature_semantics_json
        - Business sanity checks: $business_sanity_checks_json
        - Alignment Requirements: $alignment_requirements_json
        - Signal Summary: $signal_summary_json
        - Iteration Memory: $iteration_memory_json
        - Iteration Memory (compact): $iteration_memory_block
        - Data audit context: $data_audit_context

        DEPENDENCIES
        - Core ML: numpy, pandas, scipy, sklearn, statsmodels, joblib
        - Gradient Boosting: xgboost, lightgbm, catboost (use for large datasets or when sklearn underperforms)
        - Preprocessing: category_encoders, imbalanced-learn (for class imbalance: SMOTE, ADASYN, etc.)
        - Hyperparameter Tuning: optuna (for efficient Bayesian optimization, better than GridSearchCV for large search spaces)
        - Explainability: shap (for feature importance and model explanations)
        - Visualization: matplotlib, seaborn, plotly
        - Data I/O: pyarrow, openpyxl, duckdb, sqlalchemy
        - Utilities: dateutil, pytz, tqdm, yaml
        - Extended deps (rapidfuzz, pydantic, pandera, networkx) ONLY if listed in execution_contract.required_dependencies.
        - Do not import deep learning frameworks (tensorflow, keras, torch) unless explicitly required by contract.

        COMPUTE CONTEXT (HARDWARE-AWARE OPTIMIZATION)
        =============================================
        You have access to TWO execution environments. Choose hyperparameters accordingly:

        1. DEFAULT ENVIRONMENT (E2B Sandbox):
           - Resources: 2 vCPU, 8GB RAM
           - Best for: Standard workflows, small-medium datasets (<100k rows)
           - Recommendations:
             * n_jobs=2 (match vCPU count)
             * Prefer sklearn over heavy frameworks
             * Use chunked processing for >50k rows
             * Avoid large GridSearchCV (use RandomizedSearchCV or optuna with n_trials<50)
             * batch_size: 1000-5000 for iterative operations

        2. HEAVY ENVIRONMENT (Cloud Run):
           - Resources: 4-8 vCPU, 32GB RAM
           - Best for: Large datasets, deep ensembles, memory-intensive operations
           - Recommendations:
             * n_jobs=4 to n_jobs=8 (scale with vCPU)
             * Can use full GridSearchCV with moderate param grids
             * LightGBM/XGBoost with larger num_leaves, more trees
             * batch_size: 10000-50000 for iterative operations
             * Can load entire dataset into memory for most tabular tasks

        ENVIRONMENT SELECTION HEURISTICS:
        - Dataset > 500MB or > 500k rows → Heavy environment recommended
        - Ensemble with > 100 estimators → Heavy environment recommended
        - Deep hyperparameter search (> 100 combinations) → Heavy environment recommended
        - Simple baseline or EDA → Default environment is sufficient

        ADAPTIVE CODING PATTERNS:
        ```python
        # Pattern: Detect environment and adapt
        import os
        N_CPUS = int(os.environ.get('N_CPUS', 2))  # Injected by orchestrator
        IS_HEAVY = N_CPUS >= 4

        # Adapt hyperparameters
        n_jobs = N_CPUS
        n_estimators = 200 if IS_HEAVY else 50
        cv_folds = 10 if IS_HEAVY else 5
        ```

        SELF-CORRECTION PROTOCOL (EXECUTION-BASED VALIDATION)
        =====================================================
        You will receive execution feedback (stdout/stderr/traceback) after each run.
        When you see runtime errors:

        1. READ THE TRACEBACK CAREFULLY - identify the exact line and error type
        2. COMMON FIXES:
           - MemoryError → Reduce batch_size, use chunked processing, or request Heavy env
           - ValueError (NaN) → Add SimpleImputer before model.fit()
           - KeyError → Check column names against canonical_columns
           - FileNotFoundError → Use exact paths from context ($data_path)
        3. REGENERATE THE FULL SCRIPT with the fix applied
        4. DO NOT add defensive try/except that swallows errors - let them surface

        You are judged on RESULTS, not syntax. The Reviewer validates:
        - Output files exist and contain valid data
        - Metrics are reasonable for the task
        - No data leakage in train/test split
        - Predictions align with business requirements

        CAUSAL REASONING FOR OPTIMIZATION
        - Consultation: check column_roles in contract. Variables marked 'decision' or 'post-decision' CANNOT be features.
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
        - Identify target from contract.column_roles or contract.outcome_columns. If missing/unmappable -> raise ValueError with a clear message.
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
        - REQUIRED: include a high-cardinality guard with an explicit nunique check, e.g.
          if df[col].nunique() > threshold: apply top-K grouping or hashing. This must be in code.

        Step 2.5) Segmentation sanity (required if segmentation is used):
        - Compute and log: n_rows, n_segments, min/median segment_size.
        - Respect execution_contract.segmentation_constraints (max_segments, min_segment_size, preferred_k_range).
        - If constraints violated, reduce k, or use quantile binning for numerics, top-K + "Other" for categoricals,
          or fallback to a coarser segmentation (never 1-row-per-segment).
        - Do NOT create segment_id by concatenating raw columns if it yields unique IDs per row.

        Step 3) Decide validation correctly:
        - If contract/plan provides an explicit validation method (validation_requirements or ml_plan.cv_policy), follow it.
        - If objective_type == "forecasting" or requires_time_series_split=true -> use TimeSeriesSplit or chronological holdout (shuffle=False). Do NOT use random KFold.
        - If the contract/spec indicates group_key OR you infer a grouping vector -> use GroupKFold or GroupShuffleSplit (or CV with groups=...).
        - Else if time_key or time ordering matters -> use a time-based split.
        - Else -> StratifiedKFold (classification) or KFold (regression).
        - Never evaluate on training data.

        Step 4) Implement with pipelines (contract-aware default):
        - Use sklearn Pipeline + ColumnTransformer for preprocessing UNLESS the contract/plan specifies an explicit
          preprocessing or encoding requirement. Contract/plan overrides defaults.
        - Numeric: imputer (+ scaler if needed) when not contradicted by contract parsing requirements.
        - Categorical: imputer + encoder (OneHotEncoder or contract-specified ordinal mapping).
        - Apply a high-cardinality safeguard when needed (e.g., limit top-K categories or hashing) without leakage.

        Step 5) Models (contract-first):
        - If the contract/plan specifies model family or a single model, follow that exactly.
        - If not specified, choose based on dataset characteristics:
          * Small datasets (<10k rows): sklearn models (LogisticRegression, RandomForest, GradientBoosting)
          * Medium datasets (10k-100k rows): Consider xgboost or lightgbm for better performance
          * Large datasets (>100k rows): Prefer lightgbm (faster) or catboost (handles categoricals natively)
          * High-cardinality categoricals: catboost handles them well without encoding
          * Class imbalance: Use imbalanced-learn (SMOTE, ADASYN) or model's built-in class_weight
        - For calibrated probabilities: sklearn's CalibratedClassifierCV (note: does NOT accept random_state)
        - Any predict_proba call must pass a 2D array (e.g., X.reshape(1, -1) or [[x]] for a single row).

        MODEL SELECTION & METRICS CONSISTENCY:
        - If comparing multiple models, select best based on the PRIMARY METRIC from the contract/plan
          (ml_plan.metric_policy.primary_metric, validation_requirements.primary_metric, or evaluation_spec.qa_gates).
        - Save both best_model_name AND its corresponding metric value.
        - CRITICAL: best_model_metric must match the metric value of best_model_name (never mix models).
        - Pattern: if modelA_metric > modelB_metric: best_name="A", best_metric=modelA_metric (NOT modelB_metric).

        Step 6) Contract compliance outputs:
        - Do NOT invent global rules. Use execution_contract to decide:
        - which columns to use (pre-decision vs post-outcome),
        - required artifacts,
        - derived targets/columns behavior.
        - Print a "MAPPING SUMMARY" block with canonical columns, selected features, and any derived outputs used.
          MUST include the literal text "MAPPING SUMMARY" in stdout.
        - Only enforce segmentation/weights/pricing logic IF deliverables require those outputs or decision_columns exist.
        (Example: if a required deliverable includes "data/weights.json" or decision_columns are present -> run the corresponding logic; else skip.)
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
          - Populate "model_performance" ONLY with metrics REQUIRED by the contract/plan/evaluation_spec/qa_gates.
          - If no metrics are specified by contract/plan, choose a minimal, task-appropriate set and document the rationale.
          - Never omit a contract-required metric (including transformed metrics like RMSE_log1p or RMSLE).
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
        - Print QA_SELF_CHECK: PASS with a short bullet list of what was satisfied (target guard, split choice, model choice, required deliverables, no forbidden imports/ops).

        Return Python code only.

        """

        from src.utils.context_pack import compress_long_lists, summarize_long_list, COLUMN_LIST_POINTER

        ml_view = ml_view or {}
        required_outputs = ml_view.get("required_outputs") or (execution_contract or {}).get("required_outputs", []) or []
        # V4.1: Build deliverables from required_outputs, no spec_extraction
        deliverables: List[Dict[str, Any]] = []
        if required_outputs:
            deliverables = [{"path": path, "required": True} for path in required_outputs if path]
        required_deliverables = [item.get("path") for item in deliverables if item.get("required") and item.get("path")]
        deliverables_json = json.dumps(compress_long_lists(deliverables)[0], indent=2)
        
        # V4.1: Use ml_engineer_runbook directly, no legacy role_runbooks
        ml_runbook_json = json.dumps(
            compress_long_lists((execution_contract or {}).get("ml_engineer_runbook", {}))[0],
            indent=2,
        )
        # V4.1: No spec_extraction - removed
        execution_contract_compact = self._compact_execution_contract(execution_contract or {})
        execution_contract_compact = compress_long_lists(execution_contract_compact)[0]
        ml_view_payload = compress_long_lists(ml_view)[0]
        ml_view_json = json.dumps(ml_view_payload, indent=2)
        plot_spec_json = json.dumps(compress_long_lists(ml_view.get("plot_spec", {}))[0], indent=2)
        evaluation_spec_json = json.dumps(
            compress_long_lists((execution_contract or {}).get("evaluation_spec", {}))[0],
            indent=2,
        )
        required_columns_payload = strategy.get('required_columns', [])
        if isinstance(required_columns_payload, list) and len(required_columns_payload) > 80:
            required_columns_payload = summarize_long_list(required_columns_payload)
            required_columns_payload["note"] = COLUMN_LIST_POINTER

        render_kwargs = dict(
            business_objective=business_objective,
            strategy_title=strategy.get('title', 'Unknown'),
            analysis_type=str(strategy.get('analysis_type', 'predictive')).upper(),
            hypothesis=strategy.get('hypothesis', 'N/A'),
            required_columns=json.dumps(required_columns_payload),
            deliverables_json=deliverables_json,
            canonical_columns=json.dumps(
                execution_contract_compact.get("canonical_columns", (execution_contract or {}).get("canonical_columns", []))
            ),
            business_alignment_json=json.dumps(
                compress_long_lists((execution_contract or {}).get("business_alignment", {}))[0],
                indent=2,
            ),
            alignment_requirements_json=json.dumps(
                compress_long_lists((execution_contract or {}).get("alignment_requirements", []))[0],
                indent=2,
            ),
            feature_semantics_json=json.dumps(
                compress_long_lists((execution_contract or {}).get("feature_semantics", []))[0],
                indent=2,
            ),
            business_sanity_checks_json=json.dumps(
                compress_long_lists((execution_contract or {}).get("business_sanity_checks", []))[0],
                indent=2,
            ),
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
            ml_engineer_runbook=ml_runbook_json,
            # V4.1: availability_summary removed
            signal_summary_json=json.dumps(compress_long_lists(signal_summary or {})[0], indent=2),
            iteration_memory_json=json.dumps(compress_long_lists(iteration_memory or [])[0], indent=2),
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
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]

                def _call_openrouter():
                    self.last_prompt = system_prompt + "\n\nUSER:\n" + user_message
                    response, model_used = call_chat_with_fallback(
                        self.client,
                        messages,
                        [self.model_name, self.fallback_model_name],
                        call_kwargs={"temperature": current_temp},
                        logger=self.logger,
                        context_tag="ml_engineer",
                    )
                    if model_used != self.model_name:
                        self.last_fallback_reason = "fallback_used"
                    self.last_model_used = model_used
                    self.logger.info("ML_ENGINEER_MODEL_USED: %s", model_used)
                    content = response.choices[0].message.content
                    self.last_response = content
                    if "504 Gateway Time-out" in content or "<html" in content.lower():
                        raise ConnectionError("LLM Server Timeout (504 Received)")
                    return content

                content = call_with_retries(
                    _call_openrouter,
                    max_retries=5,
                    backoff_factor=2,
                    initial_delay=2,
                )
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

            checklist = self._build_training_policy_checklist(
                code=code,
                execution_contract=execution_contract,
                ml_view=ml_view,
                ml_plan=ml_plan,
            )
            print("ML_TRAINING_CHECKLIST: " + json.dumps(checklist))

            training_issues = self._check_training_policy_compliance(
                code=code,
                execution_contract=execution_contract,
                ml_view=ml_view,
                ml_plan=ml_plan,
            )
            if training_issues:
                context = self._extract_training_context(execution_contract, ml_view, ml_plan)
                required_filter = self._describe_train_filter(
                    context.get("train_filter"),
                    context.get("target_column"),
                    context.get("split_column"),
                )
                compliance_system = (
                    "You are a senior ML engineer. Ensure the script implements the training "
                    "row selection policy from the plan/contract. Return FULL python script only."
                )
                compliance_user = (
                    "Plan/contract compliance issues detected:\n"
                    + "\n".join([f"- {issue}" for issue in training_issues])
                    + "\n\nTraining context:\n"
                    + json.dumps(context, indent=2)
                    + "\n\nRequired training filter:\n"
                    + required_filter
                    + "\n\nChecklist:\n"
                    + json.dumps(checklist, indent=2)
                    + "\n\nCODE:\n"
                    + self._truncate_code_for_patch(code)
                    + "\n\nFix the training/scoring row selection so it matches the plan. "
                    "Only edit the row-selection block (train_df / X_train / y construction). "
                    "Keep outputs and other logic intact."
                )
                repaired = call_with_retries(
                    lambda: _call_model_with_prompts(
                        compliance_system,
                        compliance_user,
                        0.0,
                        self.last_model_used or self.model_name,
                    ),
                    max_retries=2,
                    backoff_factor=2,
                    initial_delay=1,
                )
                repaired = self._clean_code(repaired)
                repaired_valid = is_syntax_valid(repaired)
                if repaired_valid:
                    repaired = self._fix_data_path_in_code(repaired, data_path)
                    repaired = self._fix_to_csv_dialect_in_code(repaired)
                remaining = self._check_training_policy_compliance(
                    code=repaired if repaired_valid else code,
                    execution_contract=execution_contract,
                    ml_view=ml_view,
                    ml_plan=ml_plan,
                )
                if not remaining and repaired_valid:
                    code = repaired
                elif remaining:
                    fallback_code = repaired if repaired_valid else (code if is_syntax_valid(code) else None)
                    if fallback_code is None:
                        raise RuntimeError(
                            "ML training policy compliance failed after repair: "
                            + "; ".join(remaining)
                        )
                    self.last_training_policy_warnings = {
                        "issues": remaining,
                        "context": context,
                        "required_filter": required_filter,
                        "checklist": checklist,
                    }
                    print("ML_TRAINING_POLICY_WARNING: " + json.dumps(self.last_training_policy_warnings))
                    code = fallback_code
            return code

        except Exception as e:
            # Raise RuntimeError as requested for clean catch in graph.py
            print(f"CRITICAL: ML Engineer Failed (Max Retries): {e}")
            raise RuntimeError(f"ML Generation Failed: {e}")

    def _clean_code(self, code: str) -> str:
        return extract_code_block(code)
