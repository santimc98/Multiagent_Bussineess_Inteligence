import os\r
import json\r
import re\r
from typing import Dict, Any, List\r
from dotenv import load_dotenv\r
from openai import OpenAI\r
\r
load_dotenv()\r
\r
class DomainExpertAgent:\r
    def __init__(self, api_key: str = None):\r
        \"\"\"\r
        Initializes the Domain Expert Agent with MIMO v2 Flash.\r
        Role: Senior Business Analyst / Product Owner.\r
        \"\"\"\r
        self.api_key = api_key or os.getenv("MIMO_API_KEY")\r
        if not self.api_key:\r
            raise ValueError("MIMO API Key is required for Domain Expert.")\r
        \r
        # Initialize OpenAI-compatible client for MIMO\r
        self.client = OpenAI(\r
            api_key=self.api_key,\r
            base_url="https://api.xiaomimimo.com/v1"\r
        )\r
        self.model_name = "mimo-v2-flash"\r
        self.last_prompt = None\r
        self.last_response = None\r
\r
    def evaluate_strategies(self, data_summary: str, business_objective: str, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:\r
        \"\"\"\r
        Critiques and scores multiple strategies based on business alignment and feasibility.\r
        \"\"\"\r
        \r
        strategies_text = json.dumps(strategies, indent=2)\r
        \r
        from src.utils.prompting import render_prompt\r
\r
        SYSTEM_PROMPT_TEMPLATE = \"\"\"\r
        You are a Senior Industry Expert and Business Analyst.\r
        Your goal is to critique technical data science proposals and select the one that delivers maximum BUSINESS VALUE.\r
        \r
        *** BUSINESS OBJECTIVE ***\r
        \"$business_objective\"\r
        \r
        *** DATA CONTEXT ***\r
        $data_summary\r
        \r
        *** CANDIDATE STRATEGIES ***\r
        $strategies_text\r
        \r
        *** YOUR TASK ***\r
        Evaluate each strategy (0-10 Score) based on:\r
        1. **Business Alignment:** Does it directly answer the business question? (High score) or just explore data? (Low score).\r
        2. **Technical Feasibility:** Do we have the data? Is the approach realistic given dataset size and complexity?\r
        3. **Implementability:** Can our AI engineers (ML + Data) execute this with their capabilities?\r
           - Consider dataset size from data summary (small n = harder for complex models)\r
           - Penalize strategies that require capabilities engineers lack (e.g. causal inference with n<500)\r
           - Reward strategies with fallback plans for complex approaches\r
        4. **Risk Assessment:** Overfitting risk? \"Black box\" non-explainability where transparency is needed?\r
        \r
        Implementability Guidelines:\r
        - Small datasets (n<500) + Complex techniques (causal, elasticity) = Lower score (-1 to -2 pts)\r
        - Large datasets (n>=500) + Standard techniques (LR, RF, segmentation) = Higher score\r
        - Strategies with explicit fallbacks = Bonus (+0.5 pts) for risk mitigation\r
        \r
        *** OUTPUT FORMAT ***\r
        Return a JSON object with:\r
        {{\r
            \"reviews\": [\r
                {{\r
                    \"title\": \"Strategy Title\",\r
                    \"score\": 8.5,\r
                    \"reasoning\": \"Strong alignment with pricing goal...\",\r
                    \"risks\": [\"Potential overfitting due to small sample\"],\r
                    \"recommendation\": \"Proceed with caution on feature selection.\"\r
                }},\r
                ...\r
            ]\r
        }}\r
        \"\"\"\r
        \r
        system_prompt = render_prompt(\r
            SYSTEM_PROMPT_TEMPLATE,\r
            business_objective=business_objective,\r
            data_summary=data_summary,\r
            strategies_text=strategies_text\r
        )\r
        self.last_prompt = system_prompt\r
        \r
        messages = [\r
            {\"role\": \"system\", \"content\": system_prompt},\r
            {\"role\": \"user\", \"content\": \"Evaluate these strategies.\"}\r
        ]\r
\r
        try:\r
            response = self.client.chat.completions.create(\r
                model=self.model_name,\r
                messages=messages,\r
                response_format={'type': 'json_object'},\r
                temperature=0.1\r
            )\r
            \r
            content = response.choices[0].message.content\r
            self.last_response = content\r
            cleaned_content = self._clean_json(content)\r
            return json.loads(cleaned_content)\r
            \r
        except Exception as e:\r
            print(f\"Domain Expert Error: {e}\")\r
            # Fallback: Return empty reviews, graph will handle selection fallback\r
            return {\"reviews\": []}\r
\r
    def _clean_json(self, text: str) -> str:\r
        text = re.sub(r'```json', '', text)\r
        text = re.sub(r'```', '', text)\r
        return text.strip()\r
