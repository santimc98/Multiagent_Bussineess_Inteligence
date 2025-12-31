import sys
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agents.steward import StewardAgent
from agents.business_translator import BusinessTranslatorAgent

def create_dummy_csv():
    data = {
        'customer_id': [101, 102, 103, 104, 105],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'signup_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'spend': [100.50, 200.00, None, 50.25, 300.10],
        'active': [True, False, True, True, False]
    }
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/dummy_test.csv', index=False)
    return 'data/dummy_test.csv'

def test_remaining_agents():
    print("--- Starting Test for Steward and Translator ---")
    
    # 1. Test Steward
    print("\n[1] Testing StewardAgent...")
    csv_path = create_dummy_csv()
    try:
        steward = StewardAgent()
        summary = steward.analyze_data(csv_path)
        print("Steward Output:")
        print(summary)
        summary_text = summary.get("summary") if isinstance(summary, dict) else summary
        if summary_text and "DATA SUMMARY:" in summary_text:
            print(">> Steward Test PASSED")
        else:
            print(">> Steward Test WARNING: Output format might be incorrect")
    except Exception as e:
        print(f">> Steward Test FAILED: {e}")

    # 2. Test Translator
    print("\n[2] Testing BusinessTranslatorAgent...")
    mock_code_output = """
    Accuracy: 0.85
    Classification Report:
                  precision    recall  f1-score   support
           0       0.88      0.90      0.89       100
           1       0.75      0.70      0.72        40
    """
    mock_objective = "Reduce customer churn by identifying at-risk users."
    
    try:
        translator = BusinessTranslatorAgent()
        report = translator.generate_report(mock_code_output, mock_objective)
        print("Translator Output:")
        print(report[:200] + "...") # Print first 200 chars
        if len(report) > 50:
            print(">> Translator Test PASSED")
        else:
            print(">> Translator Test FAILED: Output too short")
    except Exception as e:
        print(f">> Translator Test FAILED: {e}")

if __name__ == "__main__":
    test_remaining_agents()
