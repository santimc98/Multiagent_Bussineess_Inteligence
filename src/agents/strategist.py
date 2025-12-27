import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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

    def generate_strategies(self, data_summary: str, user_request: str) -> Dict[str, Any]:
        """
        Generates a single strategy based on the data summary and user request.
        """
        
        from src.utils.prompting import render_prompt

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Chief Data Strategist inside a multi-agent system. Your goal is to craft ONE optimal strategy
        that downstream AI engineers can execute successfully.
        
        *** DATASET SUMMARY ***
        $data_summary
        
        *** USER REQUEST ***
        "$user_request"
        
        *** YOUR TASK ***
        - Provide ONE strategy that maximizes business alignment AND is feasible for the AI engineers to execute.
        - Think about the capabilities/constraints of the system (cleaning, mapping, optimization, reporting).
        - State which variables are required and which data science techniques should be used.
        - In your reasoning, explicitly assess feature availability timing (pre-decision vs post-outcome),
          leakage risk, and signal sufficiency. If you see risk, propose a fallback approach (e.g., descriptive
          segmentation instead of predictive modeling). This is reasoning guidance, not a rigid rule.
        
        *** DATA SCIENCE FIRST PRINCIPLES (UNIVERSAL REASONING) ***
        1. **REPRESENTATIVENESS (The "Bias" Check):**
           - Does your selected data subset represent the *Full Reality* of the problem?
           - *Rule:* NEVER filter the target variable to a single class if the goal is comparison or prediction.
        
        2. **SIGNAL MAXIMIZATION (The "Feature" Check):**
           - *Action:* Select ALL columns that might carry information. Be broad.
           
        3. **TARGET CLARITY:**
           - What exactly are we solving for? (e.g. Price Optimization -> Target = "Success Probability" given Price).
           
        *** CRITICAL OUTPUT RULES ***
        - RETURN ONLY RAW JSON. NO MARKDOWN. NO COMMENTS.
        - The output must be a dictionary with a single key "strategies" containing a LIST of 1 object.
        - The object keys: "title", "analysis_type", "hypothesis", "required_columns" (list of strings),
          "techniques" (list of strings), "estimated_difficulty", "reasoning".
        - "required_columns": Use EXACT column names from the summary.
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            data_summary=data_summary,
            user_request=user_request
        )
        
        try:
            response = self.model.generate_content(system_prompt)
            content = response.text
            cleaned_content = self._clean_json(content)
            single_strategy = json.loads(cleaned_content)
            
            return single_strategy
            
        except Exception as e:
            print(f"Strategist Error: {e}")
            # Fallback simple strategy
            return {"strategies": [{
                "title": "Error Fallback Strategy",
                "analysis_type": "statistical",
                "hypothesis": "Could not generate complex strategy. Analyzing basic correlations.",
                "required_columns": [],
                "techniques": ["correlation_analysis"],
                "estimated_difficulty": "Low",
                "reasoning": f"Gemini API Failed: {e}"
            }]}

    def _clean_json(self, text: str) -> str:
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()
