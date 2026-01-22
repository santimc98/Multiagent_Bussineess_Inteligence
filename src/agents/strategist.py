import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from src.utils.senior_protocol import SENIOR_STRATEGY_PROTOCOL

load_dotenv()

class StrategistAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Strategist Agent with Gemini 3 Flash Preview.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required.")

        genai.configure(api_key=self.api_key)
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
            generation_config=generation_config,
            safety_settings=self.safety_settings,
        )
        self.last_prompt = None
        self.last_response = None

    def generate_strategies(self, data_summary: str, user_request: str) -> Dict[str, Any]:
        """
        Generates a single strategy based on the data summary and user request.
        """
        
        from src.utils.prompting import render_prompt

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Chief Data Strategist inside a multi-agent system. Your goal is to craft ONE optimal strategy
        that downstream AI engineers can execute successfully.

        === SENIOR STRATEGY PROTOCOL ===
        $senior_strategy_protocol

        *** DATASET SUMMARY ***
        $data_summary

        *** USER REQUEST ***
        "$user_request"

        *** YOUR TASK ***
        Design a strategy using FIRST PRINCIPLES REASONING. Do not classify the problem into pre-defined categories.
        Instead, reason through WHAT the business is trying to achieve and HOW data science can help.

        CRITICAL: Your reasoning must be universal and adaptable to ANY objective, not hardcoded to specific problem types.

        *** STEP 1: UNDERSTAND THE BUSINESS QUESTION (First Principles) ***
        Before proposing techniques, answer these fundamental questions:

        1. **WHAT is the business trying to LEARN or ACHIEVE?**
           - Are they trying to UNDERSTAND patterns? (descriptive/exploratory)
           - Are they trying to PREDICT future outcomes? (predictive)
           - Are they trying to OPTIMIZE a decision? (prescriptive)
           - Are they trying to EXPLAIN why something happens? (causal/diagnostic)
           - Are they trying to RANK/PRIORITIZE items? (comparative)

        2. **WHAT is the DECISION or ACTION that follows from this analysis?**
           - Will the business change prices? (optimization)
           - Will they target specific customers? (classification/segmentation)
           - Will they forecast demand? (time series prediction)
           - Will they adjust a process? (causal inference)

        3. **WHAT is the SUCCESS METRIC from the business perspective?**
           - Maximizing revenue/profit? (optimization metric)
           - Minimizing error in prediction? (accuracy metric)
           - Understanding customer segments? (descriptive metric)
           - Identifying causal relationships? (statistical significance)

        *** STEP 2: TRANSLATE TO DATA SCIENCE OBJECTIVE ***
        Based on your answers above, explicitly state:
        - **objective_type**: One of [descriptive, predictive, prescriptive, causal, comparative]
        - **objective_reasoning**: WHY you chose this type (2-3 sentences connecting business goal to objective type)
        - **success_metric**: What metric best captures business success (not generic ML metrics)

        Examples of objective_reasoning (DO NOT COPY, USE AS REFERENCE):
        - "The business wants to find the optimal price point to maximize expected revenue (Price × Success Probability).
           This is a PRESCRIPTIVE objective because the goal is not just to predict success, but to recommend the best
           price per customer segment. Success metric: Expected Revenue."

        - "The business wants to predict customer churn in the next 30 days to enable proactive retention.
           This is a PREDICTIVE objective because the goal is forecasting a future binary outcome.
           Success metric: Precision at top 20% (cost of false positives is high)."

        - "The business wants to understand what customer segments exist based on behavior patterns.
           This is a DESCRIPTIVE objective because the goal is pattern discovery, not prediction.
           Success metric: Segment interpretability and separation quality (silhouette score)."

        *** STEP 3: DESIGN STRATEGY WITH TEAM CAPABILITIES IN MIND ***
        You are designing a plan for AI engineers (Data Engineer, ML Engineer) with specific strengths and limitations.
        Consider dataset size and complexity when proposing strategies.
        
        ML Engineer Capabilities:
        ✅ Strong: Supervised learning (Linear/Logistic Regression, Random Forest, Gradient Boosting), 
            clustering (KMeans, DBSCAN), segmentation analysis, baseline comparisons, 
            simple pipelines, visualization, cross-validation
        ⚠️ Limited: Complex causal inference (small n), advanced time series (requires n>=1000), 
            deep learning, automatic feature engineering beyond standard transforms
        ❌ Cannot: Generate synthetic data, access external APIs, train models requiring >5000 samples
        
        Data Engineer Capabilities:
        ✅ Strong: Data cleaning, type inference, missing value handling, outlier detection,
            basic transformations, duplicate removal, format standardization
        ⚠️ Limited: Complex feature engineering, advanced imputation strategies
        ❌ Cannot: External data joins, API calls, database writes
        
        Dataset Scale Guidance (UNIVERSAL):
        - Use dataset_scale_hints or the data summary to infer small/medium/large.
        - Small or limited scale: favor simple, interpretable approaches; penalize complexity.
        - Medium or large scale: allow more complex methods if justified, but prefer robust baselines.
        
        Strategy Complexity Rules:
        1. Small or limited datasets: Prefer simple, interpretable models over complex methods.
        2. If proposing COMPLEX techniques (elasticity modeling, causal inference, multi-stage optimization):
           → Include FALLBACK to a simpler approach if data limits apply.
        3. Always consider: Can this be implemented with available data and engineer capabilities?
        
        Implementation Feasibility Check:
        Before finalizing strategy, ask yourself:
        - Does this require data we don't have (time series with n=300)?
        - Does this need capabilities engineers lack (deep learning, causal DAGs)?
        - Is there a simpler approach that delivers 80% of the value with 20% of the risk?

        
        *** DATA SCIENCE FIRST PRINCIPLES (UNIVERSAL REASONING) ***
        1. **REPRESENTATIVENESS (The "Bias" Check):**
           - Does your selected data subset represent the *Full Reality* of the problem?
           - *Rule:* NEVER filter the target variable to a single class if the goal is comparison or prediction.
        
        2. **SIGNAL MAXIMIZATION (The "Feature" Check):**
           - *Action:* Select ALL columns that might carry information. Be broad.
           
        3. **TARGET CLARITY:**
           - What exactly are we solving for? (e.g. Price Optimization -> Target = "Success Probability" given Price).
           
        *** STEP 4: EVALUATE APPROPRIATE METRICS ***
        Based on your objective_type, reason through what metrics best measure success.
        DO NOT use pre-defined metric lists. Instead, think:
        - What does the business care about? (revenue, accuracy, interpretability, coverage)
        - What are the risks? (false positives costly? false negatives worse?)
        - What validates the approach? (cross-validation, time split, holdout)

        Examples (DO NOT COPY LITERALLY):
        - Prescriptive (optimization): Expected Value, Revenue Lift, Opportunity Cost
        - Predictive (classification): Precision@K, ROC-AUC, F1 (depends on cost asymmetry)
        - Predictive (regression): MAE, RMSE, MAPE (depends on scale sensitivity)
        - Descriptive (segmentation): Silhouette Score, Segment Size Distribution, Interpretability
        - Causal: ATE (Average Treatment Effect), Statistical Significance, Confidence Intervals

        *** CRITICAL OUTPUT RULES ***
        - RETURN ONLY RAW JSON. NO MARKDOWN. NO COMMENTS.
        - The output must be a dictionary with a single key "strategies" containing a LIST of 3 objects.
        - The object must include these keys:
          {
            "title": "Strategy name",
            "objective_type": "One of: descriptive, predictive, prescriptive, causal, comparative",
            "objective_reasoning": "2-3 sentences explaining WHY this objective_type fits the business goal",
            "success_metric": "Primary business metric (not generic ML metric)",
            "recommended_evaluation_metrics": ["list", "of", "metrics", "to", "track"],
            "validation_strategy": "cross_validation | time_split | holdout | bootstrap (with reasoning)",
            "analysis_type": "Brief label (e.g. 'Price Optimization', 'Churn Prediction')",
            "hypothesis": "What you expect to find or achieve",
            "required_columns": ["exact", "column", "names", "from", "summary"],
            "techniques": ["list", "of", "data science techniques"],
            "estimated_difficulty": "Low | Medium | High",
            "reasoning": "Why this strategy is optimal for the data and objective"
          }
        - "required_columns": Use EXACT column names from the dataset summary.
        - "objective_reasoning" is MANDATORY and must connect business goal → objective_type.
        - "reasoning" must include: why this fits the objective, what could fail, and a fallback if data limits apply.
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            data_summary=data_summary,
            user_request=user_request,
            senior_strategy_protocol=SENIOR_STRATEGY_PROTOCOL,
        )
        self.last_prompt = system_prompt
        
        try:
            response = self.model.generate_content(system_prompt)
            content = response.text
            self.last_response = content
            cleaned_content = self._clean_json(content)
            parsed = json.loads(cleaned_content)

            # Normalization (Fix for crash 'list' object has no attribute 'get')
            payload = self._normalize_strategist_output(parsed)

            # Build strategy_spec from LLM reasoning (not hardcoded inference)
            strategy_spec = self._build_strategy_spec_from_llm(payload, data_summary, user_request)
            payload["strategy_spec"] = strategy_spec
            return payload

        except Exception as e:
            print(f"Strategist Error: {e}")
            # Fallback simple strategy
            fallback = {"strategies": [{
                "title": "Error Fallback Strategy",
                "objective_type": "descriptive",
                "objective_reasoning": "API error occurred. Defaulting to descriptive analysis as safest fallback.",
                "success_metric": "Data quality summary",
                "recommended_evaluation_metrics": ["completeness", "consistency"],
                "validation_strategy": "manual_review",
                "analysis_type": "statistical",
                "hypothesis": "Could not generate complex strategy. Analyzing basic correlations.",
                "required_columns": [],
                "techniques": ["correlation_analysis"],
                "estimated_difficulty": "Low",
                "reasoning": f"Gemini API Failed: {e}"
            }]}
            strategy_spec = self._build_strategy_spec_from_llm(fallback, data_summary, user_request)
            fallback["strategy_spec"] = strategy_spec
            return fallback

    def _normalize_strategist_output(self, parsed: Any) -> Dict[str, Any]:
        """
        Normalize the LLM output into a stable dictionary structure.
        """
        strategies = []
        if isinstance(parsed, dict):
            raw_strategies = parsed.get("strategies")
            if isinstance(raw_strategies, list):
                # Filter non-dict elements
                strategies = [s for s in raw_strategies if isinstance(s, dict)]
                parsed["strategies"] = strategies
                return parsed
            elif raw_strategies is None:
                # Interpret the dict itself as a single strategy if explicit "strategies" key is missing
                strategies = [parsed]
            elif isinstance(raw_strategies, dict):
                 strategies = [raw_strategies]
            else:
                 strategies = []
            
            # If parsed had 'strategies' but it wasn't a list, we just normalized it into 'strategies'.
            # However, if parsed is the strategy itself, we wrap it.
            if "strategies" not in parsed:
                 return {"strategies": strategies}
            # Update normalized strategies
            parsed["strategies"] = strategies
            return parsed
        
        elif isinstance(parsed, list):
            strategies = [elem for elem in parsed if isinstance(elem, dict)]
            return {"strategies": strategies}
        
        else:
            return {"strategies": []}

    def _clean_json(self, text: str) -> str:
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()

    def _build_strategy_spec_from_llm(self, strategy_payload: Dict[str, Any], data_summary: str, user_request: str) -> Dict[str, Any]:
        """
        Build strategy_spec using LLM-generated reasoning instead of hardcoded inference.

        The LLM now provides:
        - objective_type (reasoned, not inferred)
        - objective_reasoning (explicit connection to business goal)
        - success_metric (business metric, not generic ML metric)
        - recommended_evaluation_metrics (reasoned, not mapped from dict)
        - validation_strategy (reasoned, not mapped from dict)
        """
        strategies = []
        if isinstance(strategy_payload, dict):
            strategies = strategy_payload.get("strategies", []) or []
        primary = strategies[0] if strategies else {}

        # Use LLM-generated objective_type (with fallback to heuristic if missing)
        objective_type = primary.get("objective_type", "descriptive")

        # If LLM didn't provide objective_type (backward compatibility), use simple heuristic
        if not objective_type or objective_type == "descriptive":
            combined = " ".join([str(user_request or "").lower()])
            if any(tok in combined for tok in ["optimiz", "maximize", "minimize", "optimal", "best price"]):
                objective_type = "prescriptive"
            elif any(tok in combined for tok in ["predict", "forecast", "estimate future"]):
                objective_type = "predictive"
            elif any(tok in combined for tok in ["explain", "why", "cause", "impact of"]):
                objective_type = "causal"
            else:
                objective_type = "descriptive"

        # Use LLM-generated metrics (with sensible defaults if missing)
        metrics = primary.get("recommended_evaluation_metrics", [])
        if not metrics:
            # Sensible defaults based on objective_type
            if objective_type == "prescriptive":
                metrics = ["expected_value", "revenue_lift"]
            elif objective_type == "predictive":
                metrics = ["accuracy", "roc_auc"]
            else:
                metrics = ["summary_statistics"]

        validation_strategy = primary.get("validation_strategy", "cross_validation")

        evaluation_plan = {
            "objective_type": objective_type,
            "metrics": metrics,
            "validation": {
                "strategy": validation_strategy,
                "notes": "Adjust validation to data volume and temporal ordering.",
            },
        }

        # Keep simple leakage heuristics (universal)
        leakage_risks: List[str] = []
        combined = " ".join([str(data_summary or "").lower(), str(user_request or "").lower()])
        if any(tok in combined for tok in ["post", "after", "outcome", "result"]):
            leakage_risks.append("Potential post-outcome fields may leak target information.")
        if "target" in combined:
            leakage_risks.append("Exclude target or target-derived fields from features.")

        # Universal recommended artifacts (not hardcoded to objective_type)
        recommended_artifacts = [
            {"artifact_type": "clean_dataset", "required": True, "rationale": "Base dataset for modeling."},
            {"artifact_type": "metrics", "required": True, "rationale": "Objective evaluation results."},
            {"artifact_type": "predictions_or_scores", "required": True, "rationale": "Primary output (predictions, scores, or recommendations)."},
            {"artifact_type": "explainability", "required": False, "rationale": "Feature importances or model explanations."},
            {"artifact_type": "diagnostics", "required": False, "rationale": "Error analysis or quality diagnostics."},
            {"artifact_type": "visualizations", "required": False, "rationale": "Plots for interpretability."},
        ]

        return {
            "objective_type": objective_type,
            "evaluation_plan": evaluation_plan,
            "leakage_risks": leakage_risks,
            "recommended_artifacts": recommended_artifacts,
        }
