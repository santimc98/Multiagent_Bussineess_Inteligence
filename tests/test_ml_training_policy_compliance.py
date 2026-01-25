from src.agents.ml_engineer import MLEngineerAgent


def _agent():
    return MLEngineerAgent.__new__(MLEngineerAgent)


def test_training_policy_accepts_split_usage_when_plan_prefers_split():
    agent = _agent()
    code = """
import pandas as pd
df = pd.read_csv('data/cleaned_data.csv')
train_df = df[df['__split'] == 'train'].copy()
"""
    execution_contract = {"outcome_columns": ["SalePrice"]}
    ml_view = {}
    ml_plan = {
        "training_rows_policy": "only_rows_with_label",
        "split_column": "__split",
        "evidence_used": {
            "split_evaluation": "Used split column '__split' because it delineates train/test.",
            "split_candidates": [{"column": "__split", "values": ["train", "test"]}],
        },
    }
    issues = agent._check_training_policy_compliance(code, execution_contract, ml_view, ml_plan)
    assert issues == []


def test_training_policy_flags_missing_filter_when_no_split_or_label():
    agent = _agent()
    code = "df = pd.read_csv('data/cleaned_data.csv')"
    execution_contract = {"outcome_columns": ["SalePrice"]}
    ml_view = {}
    ml_plan = {"training_rows_policy": "only_rows_with_label"}
    issues = agent._check_training_policy_compliance(code, execution_contract, ml_view, ml_plan)
    assert "training_rows_filter_missing" in issues
