import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from string import Template
import json
from typing import Dict, Any, Optional, List
from src.utils.prompting import render_prompt


def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

class BusinessTranslatorAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Business Translator Agent with Gemini 3 Flash.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
            generation_config={"temperature": 0.7},
        )

    def generate_report(self, state: Dict[str, Any], error_message: Optional[str] = None, has_partial_visuals: bool = False, plots: Optional[List[str]] = None) -> str:
        
        # Sanitize Visuals Context
        has_partial_visuals = bool(has_partial_visuals)
        plots = plots or []
        
        # Safe extraction of strategy info
        strategy = state.get('selected_strategy', {})
        strategy_title = strategy.get('title', 'General Analysis')
        hypothesis = strategy.get('hypothesis', 'N/A')
        analysis_type = str(strategy.get('analysis_type', 'N/A'))
        
        # Review content
        review_verdict = state.get("review_verdict")
        if review_verdict:
            compliance = review_verdict
        else:
            review = state.get('review_feedback', {})
            if isinstance(review, dict):
                compliance = review.get('status', 'PENDING')
            else:
                # If it's a string (e.g. just the feedback text from older legacy flows or simple strings)
                compliance = "REVIEWED" if review else "PENDING"
        
        # Construct JSON Context for Visuals safely using json library
        visuals_context_data = {
            "has_partial_visuals": has_partial_visuals,
            "plots_count": len(plots),
            "plots_list": plots
        }
        visuals_context_json = json.dumps(visuals_context_data, ensure_ascii=False)
        
        # Load optional artifacts for context
        contract = _safe_load_json("data/execution_contract.json") or {}
        integrity_audit = _safe_load_json("data/integrity_audit_report.json") or {}
        output_contract_report = _safe_load_json("data/output_contract_report.json") or {}
        case_alignment_report = _safe_load_json("data/case_alignment_report.json") or {}
        plot_insights = _safe_load_json("data/plot_insights.json") or {}

        def _summarize_integrity():
            issues = integrity_audit.get("issues", []) if isinstance(integrity_audit, dict) else []
            severity_counts = {}
            for i in issues:
                sev = str(i.get("severity", "unknown"))
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            top = issues[:3]
            return f"Issues by severity: {severity_counts}; Top3: {top}"

        def _summarize_contract():
            req_out = contract.get("required_outputs", [])
            validations = contract.get("validations", [])
            return f"Required outputs: {req_out}; Validations: {validations}"

        def _summarize_output_contract():
            if not output_contract_report:
                return "No output contract report."
            miss = output_contract_report.get("missing", [])
            present = output_contract_report.get("present", [])
            return f"Outputs present={len(present)} missing={len(miss)}"


        def _summarize_case_alignment():
            if not case_alignment_report:
                return "No case alignment report."
            status = case_alignment_report.get("status")
            failures = case_alignment_report.get("failures", [])
            metrics = case_alignment_report.get("metrics", {})
            thresholds = case_alignment_report.get("thresholds", {})
            return f"Status={status}; Failures={failures}; KeyMetrics={metrics}"

        def _case_alignment_business_status():
            if not case_alignment_report:
                return {
                    "label": "NO_DATA",
                    "status": "UNKNOWN",
                    "message": "No se encontró reporte de alineación de casos.",
                    "recommendation": "Revisar si el proceso generó data/case_alignment_report.json.",
                }
            status = case_alignment_report.get("status")
            failures = case_alignment_report.get("failures", [])
            metrics = case_alignment_report.get("metrics", {})
            thresholds = case_alignment_report.get("thresholds", {})
            if status == "SKIPPED":
                return {
                    "label": "PENDIENTE_DEFINICION_GATES",
                    "status": "SKIPPED",
                    "message": "No se definieron gates de alineación de casos en el contrato.",
                    "recommendation": "Definir métricas y umbrales en el contrato para evaluar preparación de negocio.",
                }
            if status == "PASS":
                return {
                    "label": "APTO_CONDICIONAL",
                    "status": "PASS",
                    "message": "La alineación con la lógica de casos cumple los umbrales definidos.",
                    "key_metrics": metrics,
                }
            # FAIL
            details = []
            for failure in failures:
                metric_val = metrics.get(failure)
                thresh_val = thresholds.get(failure.replace("case_means", "min"), thresholds.get(failure))
                if metric_val is not None:
                    details.append(f"{failure}={metric_val} (umbral={thresh_val})")
                else:
                    details.append(f"{failure} (umbral={thresh_val})")
            return {
                "label": "NO_APTO_PARA_PRODUCCION",
                "status": "FAIL",
                "message": "La solución no cumple los criterios de alineación por casos.",
                "details": details,
                "recommendation": "Priorizar reducción de violaciones entre casos antes de considerar producción.",
            }

        contract_context = _summarize_contract()
        integrity_context = _summarize_integrity()
        output_contract_context = _summarize_output_contract()
        case_alignment_context = _summarize_case_alignment()
        case_alignment_business_status = _case_alignment_business_status()

        # Define Template
        SYSTEM_PROMPT_TEMPLATE = Template("""
        You are a Data Storyteller and Executive Consultant.
        Your goal is to translate technical data outputs into a compelling business narrative for the CEO.
        
        TONE: Professional, insightful, concise. Avoid technical jargon where possible.
        
        *** FORMATTING CONSTRAINTS (CRITICAL) ***
        1. **LANGUAGE:** DETECT the language of the 'Business Objective' in the state. GENERATE THE REPORT IN THAT SAME LANGUAGE. (If objective is Spanish, output Spanish).
        2. **NO MARKDOWN TABLES:** The PDF generator breaks on tables. DO NOT use table syntax. Use bulleted lists or key-value pairs instead.
           - Bad: | Metric | Value |
           - Good: 
             * Metric: Value
        
        CONTEXT:
        - Strategy: $strategy_title
        - Hypothesis: $hypothesis
        - Compliance Check: $compliance
        - Contract: $contract_context
        - Integrity Audit: $integrity_context
        - Output Contract: $output_contract_context
        - Case Alignment QA: $case_alignment_context
        - Business Readiness (Case Alignment): $case_alignment_business_status
        - Plot Insights (data-driven): $plot_insights_json
        
        ERROR CONDITION:
        $error_condition
        
        VISUALS CONTEXT (JSON):
        $visuals_context_json

        IF ERROR: 
        Explain clearly what went wrong in non-technical terms and suggest next steps.
        CRITICAL: If "has_partial_visuals" is true, you MUST state: "Despite individual errors, partial visualizations were generated." and refer to the plots listed in "plots_list". Do NOT say "No visualizations created" if they exist.
        
        IF SUCCESS:
        Synthesize the journey:
        1. Context: What did we set out to discover?
        2. Execution: What did the agents do? (Cleaned data, ran $analysis_type model).
        3. Explain the provided charts or results (found in the context below).
        4. Conclusion: Did we validate the hypothesis?
        Use Plot Insights to describe what each chart *shows* (key numbers, shifts, thresholds), not just
        what the chart type is.
        If Business Readiness indicates NO_APTO_PARA_PRODUCCION, explicitly state it and summarize the main failure reasons and recommendation in executive language.
        
        OUTPUT: Markdown format (NO TABLES).
        """)
        
        error_condition_str = f"CRITICAL ERROR ENCOUNTERED: {error_message}" if error_message else "No critical errors."

        system_prompt = SYSTEM_PROMPT_TEMPLATE.substitute(
            strategy_title=strategy_title,
            hypothesis=hypothesis,
            compliance=compliance,
            error_condition=error_condition_str,
            visuals_context_json=visuals_context_json,
            analysis_type=analysis_type,
            contract_context=contract_context,
            integrity_context=integrity_context,
            output_contract_context=output_contract_context,
            case_alignment_context=case_alignment_context,
            case_alignment_business_status=json.dumps(case_alignment_business_status, ensure_ascii=False),
            plot_insights_json=json.dumps(plot_insights, ensure_ascii=False)
        )
        
        # Execution Results
        execution_results = state.get('execution_output', 'No execution results available.')
        
        USER_MESSAGE_TEMPLATE = """
        Generate the Executive Report.
        
        *** EXECUTION FINDINGS (RESULTS & METRICS) ***
        $execution_results
        
        *** FULL CONTEXT (STATE) ***
        $final_state_str
        """
        
        user_message = render_prompt(
            USER_MESSAGE_TEMPLATE,
            execution_results=execution_results,
            final_state_str=str(state)
        )

        full_prompt = f"{system_prompt}\n\n{user_message}"

        try:
            response = self.model.generate_content(full_prompt)
            return (getattr(response, "text", "") or "").strip()
        except Exception as e:
            return f"Error generating report: {e}"
