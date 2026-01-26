"""
Tests for the Senior Reasoning Flow: Evidence → Decision Plan → Execution → Reviewer Consistency.

These tests use synthetic data (no Kaggle dependency) to verify:
1. data_profile.json captures outcome missingness correctly
2. ml_plan.json chooses coherent policy based on evidence
3. Reviewer rejects if plan says A but code does B
"""

import pandas as pd
import pytest

from src.agents.steward import build_data_profile
from src.agents.ml_engineer import MLEngineerAgent
from src.utils.ml_plan_validation import (
    validate_plan_code_coherence,
    validate_plan_data_coherence,
    run_full_coherence_validation,
)


class TestDataProfile:
    """Test that build_data_profile captures evidence correctly."""

    def test_captures_basic_stats(self):
        """Data profile captures row/column counts."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
        })
        profile = build_data_profile(df)

        assert profile["basic_stats"]["n_rows"] == 5
        assert profile["basic_stats"]["n_cols"] == 3
        assert "feature1" in profile["basic_stats"]["columns"]

    def test_captures_outcome_missingness(self):
        """Data profile captures partial outcome labels."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [0, 1, None, None, None, 0, 1, None, None, None],
        })
        contract = {"outcome_columns": ["target"]}
        profile = build_data_profile(df, contract, analysis_type="classification")

        assert "target" in profile["outcome_analysis"]
        outcome_info = profile["outcome_analysis"]["target"]
        assert outcome_info["present"] is True
        assert outcome_info["null_frac"] == 0.6  # 6 out of 10 missing
        assert outcome_info["non_null_count"] == 4

    def test_captures_split_candidates(self):
        """Data profile detects columns with split/train/test tokens."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "is_train": ["train", "train", "test", "test"],
            "fold_id": [0, 0, 1, 1],
            "target": [0, 1, 0, 1],
        })
        profile = build_data_profile(df)

        split_cols = [c["column"] for c in profile["split_candidates"]]
        assert "is_train" in split_cols
        assert "fold_id" in split_cols

    def test_captures_constant_columns(self):
        """Data profile detects constant columns."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "constant_col": ["same", "same", "same", "same"],
            "target": [0, 1, 0, 1],
        })
        profile = build_data_profile(df)

        assert "constant_col" in profile["constant_columns"]

    def test_captures_high_cardinality(self):
        """Data profile detects high cardinality columns."""
        df = pd.DataFrame({
            "id": list(range(100)),  # 100 unique out of 100 = 100% unique
            "feature1": [1, 2] * 50,
            "target": [0, 1] * 50,
        })
        profile = build_data_profile(df)

        high_card_cols = [c["column"] for c in profile["high_cardinality_columns"]]
        assert "id" in high_card_cols

    def test_captures_leakage_flags(self):
        """Data profile flags columns whose names contain outcome name."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "target_encoded": [0.1, 0.2, 0.1, 0.2],  # Contains "target"
        })
        contract = {"outcome_columns": ["target"]}
        profile = build_data_profile(df, contract)

        flagged_cols = [f["column"] for f in profile["leakage_flags"]]
        assert "target_encoded" in flagged_cols


class TestMLPlanGeneration:
    """Test that ML plan generation uses evidence correctly."""

    def test_plan_uses_only_rows_with_label_for_missing_outcome(self):
        """Plan chooses only_rows_with_label when outcome has missingness."""
        engineer = MLEngineerAgent.__new__(MLEngineerAgent)

        data_profile = {
            "basic_stats": {"n_rows": 100, "n_cols": 5},
            "outcome_analysis": {
                "target": {
                    "present": True,
                    "null_frac": 0.3,  # 30% missing
                    "non_null_count": 70,
                    "total_count": 100,
                }
            },
            "split_candidates": [],
        }
        contract = {"outcome_columns": ["target"]}
        strategy = {"analysis_type": "classification"}

        # Fake LLM call that returns expected plan for outcome missingness scenario
        def fake_llm(sys_prompt, usr_prompt):
            return '''{
                "training_rows_policy": "only_rows_with_label",
                "training_rows_rule": null,
                "split_column": null,
                "metric_policy": {
                    "primary_metric": "roc_auc",
                    "secondary_metrics": ["f1", "accuracy"],
                    "report_with_cv": true,
                    "notes": "Classification with partial labels"
                },
                "cv_policy": {
                    "strategy": "StratifiedKFold",
                    "n_splits": 5,
                    "shuffle": true,
                    "stratified": true,
                    "notes": "Stratified for classification"
                },
                "scoring_policy": {
                    "generate_scores": true,
                    "score_rows": "all"
                },
                "leakage_policy": {
                    "action": "none",
                    "flagged_columns": [],
                    "notes": ""
                },
                "evidence": ["outcome 'target' has 30% missingness (null_frac=0.3)"],
                "assumptions": [],
                "open_questions": [],
                "notes": ["Using only_rows_with_label due to outcome missingness"]
            }'''

        plan = engineer.generate_ml_plan(data_profile, contract, strategy, llm_call=fake_llm)

        assert plan["training_rows_policy"] == "only_rows_with_label"
        assert any("missingness" in str(e).lower() for e in plan["evidence"])

    def test_plan_uses_split_column_when_detected(self):
        """Plan chooses use_split_column when split candidates exist with train values."""
        engineer = MLEngineerAgent.__new__(MLEngineerAgent)

        data_profile = {
            "basic_stats": {"n_rows": 100, "n_cols": 5},
            "outcome_analysis": {
                "target": {
                    "present": True,
                    "null_frac": 0.0,
                    "non_null_count": 100,
                    "total_count": 100,
                }
            },
            "split_candidates": [
                {"column": "is_train", "unique_values_sample": ["train", "test"]}
            ],
        }
        contract = {"outcome_columns": ["target"]}
        strategy = {"analysis_type": "classification"}

        # Fake LLM call that returns expected plan for split column scenario
        def fake_llm(sys_prompt, usr_prompt):
            return '''{
                "training_rows_policy": "use_split_column",
                "training_rows_rule": null,
                "split_column": "is_train",
                "metric_policy": {
                    "primary_metric": "roc_auc",
                    "secondary_metrics": [],
                    "report_with_cv": true,
                    "notes": ""
                },
                "cv_policy": {
                    "strategy": "StratifiedKFold",
                    "n_splits": 5,
                    "shuffle": true,
                    "stratified": true,
                    "notes": ""
                },
                "scoring_policy": {
                    "generate_scores": true,
                    "score_rows": "all"
                },
                "leakage_policy": {
                    "action": "none",
                    "flagged_columns": [],
                    "notes": ""
                },
                "evidence": ["Split column 'is_train' detected with train/test values"],
                "assumptions": [],
                "open_questions": [],
                "notes": ["Using split column for train/test separation"]
            }'''

        plan = engineer.generate_ml_plan(data_profile, contract, strategy, llm_call=fake_llm)

        assert plan["training_rows_policy"] == "use_split_column"
        assert plan["split_column"] == "is_train"

    def test_plan_derives_split_filter_when_evidence_prefers_split(self):
        """Plan derives train_filter=split_equals to remove ambiguity when evidence says to use split."""
        engineer = MLEngineerAgent.__new__(MLEngineerAgent)

        data_profile = {
            "basic_stats": {"n_rows": 200, "n_cols": 5},
            "outcome_analysis": {
                "target": {
                    "present": True,
                    "null_frac": 0.5,
                    "non_null_count": 100,
                    "total_count": 200,
                }
            },
            "split_candidates": [
                {"column": "__split", "values": ["train", "test"]}
            ],
        }
        contract = {"outcome_columns": ["target"]}
        strategy = {"analysis_type": "classification"}

        def fake_llm(sys_prompt, usr_prompt):
            return '''{
                "training_rows_policy": "only_rows_with_label",
                "training_rows_rule": null,
                "split_column": "__split",
                "metric_policy": {
                    "primary_metric": "roc_auc",
                    "secondary_metrics": [],
                    "report_with_cv": true,
                    "notes": ""
                },
                "cv_policy": {
                    "strategy": "StratifiedKFold",
                    "n_splits": 5,
                    "shuffle": true,
                    "stratified": true,
                    "notes": ""
                },
                "scoring_policy": {
                    "generate_scores": true,
                    "score_rows": "all"
                },
                "leakage_policy": {
                    "action": "none",
                    "flagged_columns": [],
                    "notes": ""
                },
                "evidence_used": {
                    "outcome_null_frac": {"column": "target", "null_frac": 0.5},
                    "split_candidates": [{"column": "__split", "values": ["train", "test"]}],
                    "split_evaluation": "Used split column __split to separate train/test."
                },
                "evidence": [],
                "assumptions": [],
                "open_questions": [],
                "notes": []
            }'''

        plan = engineer.generate_ml_plan(data_profile, contract, strategy, llm_call=fake_llm)

        assert plan["train_filter"]["type"] == "split_equals"
        assert plan["training_rows_policy"] == "use_split_column"

    def test_plan_uses_contract_training_rule_when_specified(self):
        """Plan respects contract training_rows_rule."""
        engineer = MLEngineerAgent.__new__(MLEngineerAgent)

        data_profile = {
            "basic_stats": {"n_rows": 100, "n_cols": 5},
            "outcome_analysis": {},
            "split_candidates": [],
        }
        contract = {
            "training_rows_rule": "filter_by_date_column",
        }
        strategy = {"analysis_type": "regression"}

        import json
        fake_resp = {
            "training_rows_policy": "filter_by_date_column", 
            "metric_policy": {"primary_metric": "r2"},
            "cv_policy": {}, 
            "notes": []
        }
        
        plan = engineer.generate_ml_plan(data_profile, contract, strategy, llm_call=lambda s, u: json.dumps(fake_resp))

        assert plan["training_rows_policy"] == "filter_by_date_column"

    def test_plan_metric_matches_analysis_type(self):
        """Plan chooses appropriate metric for analysis type."""
        engineer = MLEngineerAgent.__new__(MLEngineerAgent)

        data_profile = {"basic_stats": {"n_rows": 100, "n_cols": 3}, "outcome_analysis": {}, "split_candidates": []}
        import json

        # Classification should use roc_auc
        strategy = {"analysis_type": "classification"}
        fake_cls = {"training_rows_policy": "use_all", "metric_policy": {"primary_metric": "roc_auc"}, "cv_policy": {}, "notes": []}
        plan = engineer.generate_ml_plan(data_profile, {}, strategy, llm_call=lambda s, u: json.dumps(fake_cls))
        assert plan["metric_policy"]["primary_metric"] == "roc_auc"

        # Regression should use r2
        strategy = {"analysis_type": "regression"}
        fake_reg = {"training_rows_policy": "use_all", "metric_policy": {"primary_metric": "r2"}, "cv_policy": {}, "notes": []}
        plan = engineer.generate_ml_plan(data_profile, {}, strategy, llm_call=lambda s, u: json.dumps(fake_reg))
        assert plan["metric_policy"]["primary_metric"] == "r2"


class TestPlanCodeCoherence:
    """Test that coherence validation catches mismatches between plan and code."""

    def test_rejects_missing_label_filter(self):
        """Rejects code that doesn't filter when plan says only_rows_with_label."""
        ml_plan = {
            "training_rows_policy": "only_rows_with_label",
            "evidence": ["outcome has 30% missing"],
            "metric_policy": {"primary_metric": "roc_auc"},
        }
        code = """
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data.csv')
X = df[['feature1', 'feature2']]
y = df['target']  # No filtering for missing values!
model = LogisticRegression()
model.fit(X, y)
"""
        result = validate_plan_code_coherence(ml_plan, code)

        assert result["passed"] is False
        assert result["status"] == "REJECTED"
        assert any("training_rows_policy" in v for v in result["violations"])

    def test_approves_code_with_label_filter(self):
        """Approves code that filters when plan says only_rows_with_label."""
        ml_plan = {
            "training_rows_policy": "only_rows_with_label",
            "evidence": ["outcome has 30% missing"],
            "metric_policy": {"primary_metric": "roc_auc"},
        }
        code = """
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data.csv')
# Filter to rows with valid labels
train_mask = df['target'].notna()
df_train = df[train_mask]
X = df_train[['feature1', 'feature2']]
y = df_train['target']
model = LogisticRegression()
model.fit(X, y)
"""
        result = validate_plan_code_coherence(ml_plan, code)

        assert result["passed"] is True
        assert result["status"] == "APPROVED"

    def test_rejects_missing_split_column_reference(self):
        """Rejects code that doesn't use split column when plan specifies it."""
        ml_plan = {
            "training_rows_policy": "use_split_column",
            "split_column": "xyzzy_partition_col",  # Very unique column name
            "split_values": ["modeling", "holdout"],
            "metric_policy": {"primary_metric": "accuracy"},
        }
        # Code does NOT mention xyzzy_partition_col at all
        code = """
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X = df[['feature1', 'feature2']]
y = df['target']
# Using random split instead of the designated partition column!
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
"""
        result = validate_plan_code_coherence(ml_plan, code)

        assert result["passed"] is False
        assert any("split_column" in v for v in result["violations"])

    def test_approves_code_using_split_column(self):
        """Approves code that uses the split column from plan."""
        ml_plan = {
            "training_rows_policy": "use_split_column",
            "split_column": "is_train",
            "split_values": ["train", "test"],
            "metric_policy": {"primary_metric": "accuracy"},
        }
        code = """
import pandas as pd

df = pd.read_csv('data.csv')
train_df = df[df['is_train'] == 'train']
test_df = df[df['is_train'] == 'test']
X_train = train_df[['feature1', 'feature2']]
y_train = train_df['target']
"""
        result = validate_plan_code_coherence(ml_plan, code)

        assert result["passed"] is True


class TestPlanDataCoherence:
    """Test coherence between ml_plan and data_profile."""

    def test_flags_missing_outcome_evidence_for_label_filter(self):
        """Flags inconsistency when plan says filter labels but no missingness in data."""
        ml_plan = {
            "training_rows_policy": "only_rows_with_label",
            "evidence": [],
        }
        data_profile = {
            "outcome_analysis": {
                "target": {
                    "present": True,
                    "null_frac": 0.0,  # No missing values
                    "non_null_count": 100,
                    "total_count": 100,
                }
            },
            "split_candidates": [],
        }

        result = validate_plan_data_coherence(ml_plan, data_profile)

        assert result["passed"] is False
        assert any("missingness" in i.lower() for i in result["inconsistencies"])

    def test_flags_missing_split_candidate(self):
        """Flags inconsistency when plan says use split column but not in candidates."""
        ml_plan = {
            "training_rows_policy": "use_split_column",
            "split_column": "is_train",
        }
        data_profile = {
            "outcome_analysis": {},
            "split_candidates": [],  # No split candidates
        }

        result = validate_plan_data_coherence(ml_plan, data_profile)

        assert result["passed"] is False
        assert any("split_candidates" in i.lower() for i in result["inconsistencies"])


class TestFullCoherenceValidation:
    """Test the combined validation function."""

    def test_full_validation_catches_all_issues(self):
        """Full validation combines plan-code and plan-data checks."""
        ml_plan = {
            "training_rows_policy": "only_rows_with_label",
            "evidence": [],
            "metric_policy": {"primary_metric": "roc_auc"},
        }
        code = """
df = pd.read_csv('data.csv')
X = df[['f1']]
y = df['target']  # No filter!
model.fit(X, y)
"""
        data_profile = {
            "outcome_analysis": {
                "target": {"present": True, "null_frac": 0.0}  # No missingness
            },
            "split_candidates": [],
        }

        result = run_full_coherence_validation(ml_plan, code, data_profile)

        assert result["passed"] is False
        # Should have violations from plan-code check
        assert len(result["violations"]) > 0 or len(result["warnings"]) > 0


class TestEndToEndSyntheticScenario:
    """End-to-end test with a realistic synthetic scenario."""

    def test_titanic_like_scenario(self):
        """
        Simulates a Titanic-like scenario where some rows don't have survival labels.

        This tests the full flow:
        1. data_profile detects partial labels
        2. ml_plan chooses only_rows_with_label
        3. Code that filters is approved
        4. Code that doesn't filter is rejected
        """
        # Create synthetic Titanic-like data
        df = pd.DataFrame({
            "PassengerId": list(range(1, 101)),
            "Pclass": [1, 2, 3] * 33 + [1],
            "Age": [22, 38, 26, 35] * 25,
            "Fare": [7.25, 71.28, 7.92, 53.10] * 25,
            # 70% have labels (train), 30% don't (test/scoring)
            "Survived": [0, 1, 0, 1] * 17 + [None] * 32,
        })

        contract = {
            "outcome_columns": ["Survived"],
            "column_roles": {
                "outcome": ["Survived"],
                "id": ["PassengerId"],
            },
        }

        # Step 1: Build data profile
        profile = build_data_profile(df, contract, "classification")

        # Verify profile captures the evidence
        assert profile["outcome_analysis"]["Survived"]["null_frac"] == 0.32

        # Step 2: Generate ML plan
        engineer = MLEngineerAgent.__new__(MLEngineerAgent)
        strategy = {"analysis_type": "classification"}
        import json
        fake_plan = {
            "training_rows_policy": "only_rows_with_label",
            "metric_policy": {"primary_metric": "accuracy"},
            "cv_policy": {}, "notes": []
        }
        plan = engineer.generate_ml_plan(profile, contract, strategy, llm_call=lambda s,u: json.dumps(fake_plan))

        # Verify plan makes correct decision
        assert plan["training_rows_policy"] == "only_rows_with_label"

        # Step 3: Validate good code
        good_code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/cleaned_data.csv')
# Filter to labeled rows only (as per ml_plan)
df_train = df[df['Survived'].notna()]
X = df_train[['Pclass', 'Age', 'Fare']]
y = df_train['Survived']
model = RandomForestClassifier()
model.fit(X, y)
"""
        good_result = validate_plan_code_coherence(plan, good_code, profile)
        assert good_result["passed"] is True

        # Step 4: Validate bad code
        bad_code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/cleaned_data.csv')
# BUG: Using all rows including unlabeled ones!
X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']
model = RandomForestClassifier()
model.fit(X, y)
"""
        bad_result = validate_plan_code_coherence(plan, bad_code, profile)
        assert bad_result["passed"] is False
