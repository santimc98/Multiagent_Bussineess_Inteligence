# 🏭 The Insight Foundry
### Autonomous Multi-Agent Data Science Department powered by Gemini 3 Pro

![Gemini](https://img.shields.io/badge/Powered%20by-Gemini%203%20Pro-4285F4?style=for-the-badge&logo=google)
![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-FF4B4B?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Code-Python-3776AB?style=for-the-badge&logo=python)

---

## 🚀 The Elevator Pitch
**The Insight Foundry** is not just a tool; it's an entire Data Science department in a box. It transforms raw, messy business data (CSV) into strategic, actionable business insights without a single line of human code.

Orchestrated by **LangGraph** and powered by the cognitive reasoning of **Gemini 3 Pro**, a team of 5 specialized AI agents collaborates to audit data, formulate strategies, write and execute machine learning code, and translate technical metrics into executive reports.

---

## 🏗️ Architecture

```mermaid
graph LR
    User[User Input<br/>(CSV + Goal)] --> Steward
    subgraph "The Insight Foundry"
        Steward[👮 Data Steward<br/>(Audit & Clean)] --> Strategist[🧠 Strategist<br/>(Planning)]
        Strategist --> Engineer[🛠️ ML Engineer<br/>(Coding)]
        Engineer --> Executor[⚙️ Code Execution<br/>(Sandboxed Subprocess)]
        Executor --> Translator[💼 Business Translator<br/>(Reporting)]
    end
    Translator --> Insight[Final Strategic Insight]
    style Strategist fill:#f9f,stroke:#333,stroke-width:2px
```

---

## 🤖 The Team (Agents)

Our system mimics a high-performing human data team:

1.  **👮 The Data Steward**: The gatekeeper. Uses Pandas and Gemini to audit data quality, detect anomalies, and ensure the pipeline is fed with clean, understood data.
2.  **🧠 The Strategist (CDO)**: Powered by **Gemini 3 Pro**. It doesn't just look at data; it understands business context. It formulates hypotheses and strategic plans (e.g., "Analyze churn by tenure to identify at-risk loyalists").
3.  **🛠️ The ML Engineer**: A senior Python developer. Takes the strategy and writes robust, production-ready Scikit-Learn code to execute it. It handles imputation, encoding, and model training autonomously.
4.  **💼 The Business Translator**: The storyteller. Takes raw accuracy scores and F1 metrics and converts them into ROI, business risks, and actionable next steps for stakeholders.

---

## ✨ Key Features

-   **🛡️ Self-Healing Pipelines**: Missing values? Typos? The ML Engineer agent detects these issues and writes code to impute or clean them automatically, ensuring the pipeline never breaks on "messy" real-world data.
-   **💼 Business-First Approach**: We don't stop at `Accuracy: 0.85`. We tell you *what that means* for your bottom line. The system is aligned with business objectives, not just loss functions.
-   **🔄 End-to-End Automation**: From `upload.csv` to a Board-ready report in minutes. No notebooks, no manual tuning, just results.

---

## 🛠️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/the-insight-foundry.git
    cd the-insight-foundry
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```
    Optional: configure the ML Engineer LLM provider (default is deepseek). For Z.ai GLM-4.7:
    ```env
    ML_ENGINEER_PROVIDER=zai
    ZAI_API_KEY=your_api_key_here
    # Alternatively: GLM_API_KEY=your_api_key_here
    ML_ENGINEER_MODEL=glm-4.7
    GLM_MAX_CONCURRENCY=2
    GLM_USE_FILE_LOCK=1
    ```
    Note: set `GLM_MAX_CONCURRENCY=2` for GLM-4.7.

4.  **Run the Foundry**
    ```bash
    streamlit run app.py
    ```

---

## 📸 Screenshots

![Demo Screenshot](path/to/screenshot.png)

---

*Built with ❤️ for the Gemini Hackathon.*
