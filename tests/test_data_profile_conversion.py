"""
Tests for data profile conversion and ML plan validation coherence.

These tests validate the unified evidence layer:
- convert_dataset_profile_to_data_profile correctly maps schema
- ML plan validation catches Facts ↔ Plan contradictions
- No network/LLM dependencies
"""

import pytest
from src.utils.data_profile_compact import (
    convert_dataset_profile_to_data_profile,
    compact_data_profile_for_llm,
    _is_dataset_profile_schema,
    _extract_outcome_columns,
)
from src.utils.ml_plan_validation import (
    validate_ml_plan_constraints,
    validate_plan_data_coherence,
    run_full_coherence_validation,
)


# =============================================================================
# Fixtures: Synthetic data for tests
# =============================================================================

@pytest.fixture
def synthetic_dataset_profile():
    """A synthetic dataset_profile.json with all key fields."""
    return {
        "rows": 100,
        "cols": 5,
        "columns": ["id", "feature1", "target", "__split", "derived_target_score"],
        "type_hints": {
            "id": "numeric",
            "feature1": "numeric",
            "target": "numeric",
            "__split": "categorical",
            "derived_target_score": "numeric",
        },
        "missing_frac": {
            "id": 0.0,
            "feature1": 0.05,
            "target": 0.3,  # 30% missing labels
            "__split": 0.0,
            "derived_target_score": 0.3,
        },
        "cardinality": {
            "id": {"unique": 100, "top_values": [{"value": "1", "count": 1}]},
            "feature1": {"unique": 50, "top_values": [{"value": "1.5", "count": 5}]},
            "target": {
                "unique": 2,
                "top_values": [
                    {"value": "0.0", "count": 35},
                    {"value": "1.0", "count": 35},
                    {"value": "nan", "count": 30},
                ],
            },
            "__split": {
                "unique": 2,
                "top_values": [
                    {"value": "train", "count": 70},
                    {"value": "test", "count": 30},
                ],
            },
            "derived_target_score": {"unique": 10, "top_values": []},
        },
        "pii_findings": {"detected": False, "findings": []},
        "sampling": {"was_sampled": False, "sample_size": 100, "file_size_bytes": 5000},
        "dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
    }


@pytest.fixture
def contract_with_outcome():
    """Contract that specifies outcome_columns and primary_metric."""
    return {
        "outcome_columns": ["target"],
        "column_roles": {"outcome": ["target"], "identifier": ["id"]},
        "validation_requirements": {"primary_metric": "roc_auc"},
        "evaluation_spec": {"primary_metric": "roc_auc", "analysis_type": "classification"},
    }


@pytest.fixture
def strategy_classification():
    """Strategy for classification task."""
    return {"analysis_type": "classification", "title": "Binary Classification"}


# =============================================================================
# Tests: convert_dataset_profile_to_data_profile
# =============================================================================

class TestConvertDatasetProfile:
    """Tests for dataset_profile -> data_profile conversion."""

    def test_basic_stats_mapping(self, synthetic_dataset_profile, contract_with_outcome):
        """Test that basic_stats are correctly mapped."""
        result = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        assert result["basic_stats"]["n_rows"] == 100
        assert result["basic_stats"]["n_cols"] == 5
        assert len(result["basic_stats"]["columns"]) == 5

    def test_outcome_analysis_null_frac(self, synthetic_dataset_profile, contract_with_outcome):
        """Test that outcome_analysis correctly extracts null_frac from missing_frac."""
        result = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        assert "target" in result["outcome_analysis"]
        outcome = result["outcome_analysis"]["target"]
        assert outcome["present"] is True
        assert outcome["null_frac"] == 0.3
        assert outcome["non_null_count"] == 70  # 100 * (1 - 0.3)
        assert outcome["total_count"] == 100

    def test_split_candidates_detected(self, synthetic_dataset_profile, contract_with_outcome):
        """Test that __split column is detected as split candidate."""
        result = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        assert len(result["split_candidates"]) > 0
        split_cols = [sc["column"] for sc in result["split_candidates"]]
        assert "__split" in split_cols

        # Check unique_values_sample contains train/test
        split_candidate = next(sc for sc in result["split_candidates"] if sc["column"] == "__split")
        assert "train" in split_candidate["unique_values_sample"]
        assert "test" in split_candidate["unique_values_sample"]

    def test_high_cardinality_detected(self, synthetic_dataset_profile, contract_with_outcome):
        """Test that high cardinality columns (like id) are detected."""
        result = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        # id has unique=100 out of 100 rows = 100% unique
        high_card_cols = [hc["column"] for hc in result["high_cardinality_columns"]]
        assert "id" in high_card_cols

    def test_leakage_flags_detected(self, synthetic_dataset_profile, contract_with_outcome):
        """Test that columns containing outcome name are flagged for leakage."""
        result = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        # derived_target_score contains "target" -> should be flagged
        flagged_cols = [lf["column"] for lf in result["leakage_flags"]]
        assert "derived_target_score" in flagged_cols

    def test_schema_version_and_source(self, synthetic_dataset_profile, contract_with_outcome):
        """Test that schema_version and source are set."""
        result = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        assert result["schema_version"] == "1.0"
        assert result["source"] == "converted_from_dataset_profile"
        assert "generated_at" in result

    def test_classification_inferred_type(self, synthetic_dataset_profile, contract_with_outcome):
        """Test that inferred_type is set correctly for classification."""
        result = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome, analysis_type="classification"
        )

        assert result["outcome_analysis"]["target"]["inferred_type"] == "classification"


# =============================================================================
# Tests: _is_dataset_profile_schema
# =============================================================================

class TestSchemaDetection:
    """Tests for schema detection."""

    def test_detects_dataset_profile_schema(self, synthetic_dataset_profile):
        """Test that dataset_profile schema is correctly detected."""
        assert _is_dataset_profile_schema(synthetic_dataset_profile) is True

    def test_detects_data_profile_schema(self, synthetic_dataset_profile, contract_with_outcome):
        """Test that data_profile schema is correctly detected as NOT dataset_profile."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )
        assert _is_dataset_profile_schema(data_profile) is False


# =============================================================================
# Tests: compact_data_profile_for_llm
# =============================================================================

class TestCompactDataProfile:
    """Tests for compact_data_profile_for_llm."""

    def test_auto_converts_dataset_profile(self, synthetic_dataset_profile, contract_with_outcome):
        """Test that dataset_profile is auto-converted when passed to compact function."""
        result = compact_data_profile_for_llm(
            synthetic_dataset_profile,
            contract=contract_with_outcome,
            analysis_type="classification",
        )

        # Should have data_profile keys after conversion
        assert "basic_stats" in result
        assert "outcome_analysis" in result
        assert "split_candidates" in result
        assert result["outcome_analysis"]["target"]["null_frac"] == 0.3

    def test_without_contract_returns_warning(self, synthetic_dataset_profile):
        """Test that dataset_profile without contract returns warning."""
        result = compact_data_profile_for_llm(synthetic_dataset_profile, contract=None)

        assert "_warning" in result
        assert "no contract provided" in result["_warning"]


# =============================================================================
# Tests: validate_ml_plan_constraints
# =============================================================================

class TestMLPlanConstraints:
    """Tests for ML plan constraint validation."""

    def test_rejects_use_all_rows_with_missing_labels(
        self, synthetic_dataset_profile, contract_with_outcome, strategy_classification
    ):
        """Test that use_all_rows is rejected when outcome has missing labels."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        bad_plan = {
            "training_rows_policy": "use_all_rows",  # WRONG! Data has 30% missing labels
            "metric_policy": {"primary_metric": "roc_auc"},
            "plan_source": "llm",
            "evidence_used": {},
        }

        result = validate_ml_plan_constraints(
            bad_plan, data_profile, contract_with_outcome, strategy_classification
        )

        assert result["ok"] is False
        assert len(result["violations"]) > 0
        assert any("use_all_rows" in v for v in result["violations"])

    def test_accepts_only_rows_with_label(
        self, synthetic_dataset_profile, contract_with_outcome, strategy_classification
    ):
        """Test that only_rows_with_label is accepted when outcome has missing labels."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        good_plan = {
            "training_rows_policy": "only_rows_with_label",
            "metric_policy": {"primary_metric": "roc_auc"},
            "plan_source": "llm",
            "evidence_used": {
                "outcome_null_frac": {"column": "target", "null_frac": 0.3},
                "split_evaluation": "Using label filter because 30% labels are missing",
            },
        }

        result = validate_ml_plan_constraints(
            good_plan, data_profile, contract_with_outcome, strategy_classification
        )

        assert result["ok"] is True
        assert len(result["violations"]) == 0

    def test_rejects_metric_mismatch_with_contract(
        self, synthetic_dataset_profile, contract_with_outcome, strategy_classification
    ):
        """Test that plan metric must match contract metric."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        bad_plan = {
            "training_rows_policy": "only_rows_with_label",
            "metric_policy": {"primary_metric": "accuracy"},  # Contract says roc_auc!
            "plan_source": "llm",
            "evidence_used": {},
        }

        result = validate_ml_plan_constraints(
            bad_plan, data_profile, contract_with_outcome, strategy_classification
        )

        assert result["ok"] is False
        assert any("METRIC_CONTRACT_MISMATCH" in v for v in result["violations"])

    def test_warns_split_not_evaluated(
        self, synthetic_dataset_profile, contract_with_outcome, strategy_classification
    ):
        """Test that warning is raised if split candidates exist but not evaluated."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        plan_without_split_eval = {
            "training_rows_policy": "only_rows_with_label",
            "metric_policy": {"primary_metric": "roc_auc"},
            "plan_source": "llm",
            "evidence_used": {},  # No split_evaluation!
        }

        result = validate_ml_plan_constraints(
            plan_without_split_eval, data_profile, contract_with_outcome, strategy_classification
        )

        # Should have warning about split not evaluated
        assert len(result["warnings"]) > 0
        assert any("SPLIT_NOT_EVALUATED" in w for w in result["warnings"])


# =============================================================================
# Tests: validate_plan_data_coherence
# =============================================================================

class TestPlanDataCoherence:
    """Tests for ML plan vs data profile coherence."""

    def test_coherence_fails_use_all_rows_with_nulls(
        self, synthetic_dataset_profile, contract_with_outcome
    ):
        """Test coherence check fails when use_all_rows but data has null labels."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        bad_plan = {
            "training_rows_policy": "use_all_rows",
            "metric_policy": {"primary_metric": "roc_auc"},
        }

        result = validate_plan_data_coherence(bad_plan, data_profile, contract_with_outcome)

        assert result["passed"] is False
        assert len(result["inconsistencies"]) > 0

    def test_coherence_fails_invalid_split_column(
        self, synthetic_dataset_profile, contract_with_outcome
    ):
        """Test coherence check fails when split_column not in candidates."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        bad_plan = {
            "training_rows_policy": "use_split_column",
            "split_column": "nonexistent_column",
            "metric_policy": {"primary_metric": "roc_auc"},
        }

        result = validate_plan_data_coherence(bad_plan, data_profile, contract_with_outcome)

        assert result["passed"] is False
        assert any("split_column" in i for i in result["inconsistencies"])

    def test_coherence_passes_valid_split_column(
        self, synthetic_dataset_profile, contract_with_outcome
    ):
        """Test coherence check passes when split_column is valid."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        good_plan = {
            "training_rows_policy": "use_split_column",
            "split_column": "__split",
            "metric_policy": {"primary_metric": "roc_auc"},
        }

        result = validate_plan_data_coherence(good_plan, data_profile, contract_with_outcome)

        assert result["passed"] is True
        assert len(result["inconsistencies"]) == 0


# =============================================================================
# Tests: run_full_coherence_validation
# =============================================================================

class TestFullCoherenceValidation:
    """Tests for full coherence validation (plan + code + data)."""

    def test_full_validation_catches_code_mismatch(
        self, synthetic_dataset_profile, contract_with_outcome
    ):
        """Test that code without filter is caught when plan requires filter."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        plan = {
            "training_rows_policy": "only_rows_with_label",
            "metric_policy": {"primary_metric": "roc_auc"},
        }

        # Code that doesn't filter nulls
        bad_code = """
import pandas as pd
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']
# No filtering for nulls!
"""

        result = run_full_coherence_validation(plan, bad_code, data_profile, contract_with_outcome)

        assert result["passed"] is False
        assert any("notna" in v or "dropna" in v for v in result["violations"])

    def test_full_validation_passes_good_code(
        self, synthetic_dataset_profile, contract_with_outcome
    ):
        """Test that code with proper filter passes validation."""
        data_profile = convert_dataset_profile_to_data_profile(
            synthetic_dataset_profile, contract_with_outcome
        )

        plan = {
            "training_rows_policy": "only_rows_with_label",
            "metric_policy": {"primary_metric": "roc_auc"},
        }

        # Code that properly filters nulls
        good_code = """
import pandas as pd
df = pd.read_csv('data.csv')
df = df[df['target'].notna()]
X = df.drop('target', axis=1)
y = df['target']
"""

        result = run_full_coherence_validation(plan, good_code, data_profile, contract_with_outcome)

        # Code check should pass (plan-data coherence may still have issues)
        code_violations = [v for v in result["violations"] if "notna" in v or "dropna" in v]
        assert len(code_violations) == 0


# =============================================================================
# Tests: _extract_outcome_columns
# =============================================================================

class TestExtractOutcomeColumns:
    """Tests for outcome column extraction from contract."""

    def test_extracts_from_outcome_columns(self):
        """Test extraction from outcome_columns field."""
        contract = {"outcome_columns": ["target", "label"]}
        result = _extract_outcome_columns(contract)
        assert result == ["target", "label"]

    def test_extracts_from_column_roles(self):
        """Test extraction from column_roles.outcome field."""
        contract = {"column_roles": {"outcome": ["target"]}}
        result = _extract_outcome_columns(contract)
        assert result == ["target"]

    def test_ignores_unknown(self):
        """Test that 'unknown' outcome is ignored."""
        contract = {"outcome_columns": ["unknown", "target"]}
        result = _extract_outcome_columns(contract)
        assert "unknown" not in result
        assert "target" in result

    def test_handles_string_outcome(self):
        """Test that string outcome_columns is handled."""
        contract = {"outcome_columns": "target"}
        result = _extract_outcome_columns(contract)
        assert result == ["target"]

    def test_extracts_from_v41_column_to_role_format(self):
        """Test extraction from V4.1 column -> role format (inverted)."""
        # V4.1 format C: column_roles = {column_name: role_string}
        contract = {
            "column_roles": {
                "PassengerId": "identifier",
                "Survived": "outcome",
                "Age": "feature",
            }
        }
        result = _extract_outcome_columns(contract)
        assert "Survived" in result

    def test_extracts_from_v41_column_to_role_dict_format(self):
        """Test extraction from V4.1 column -> {role: ...} format."""
        # V4.1 format C with metadata: column_roles = {column_name: {role: ..., ...}}
        contract = {
            "column_roles": {
                "PassengerId": {"role": "identifier"},
                "Survived": {"role": "outcome", "nullable": False},
                "Age": {"role": "feature"},
            }
        }
        result = _extract_outcome_columns(contract)
        assert "Survived" in result


# =============================================================================
# Tests: V4.1 Contract Schema Support
# =============================================================================

class TestV41ContractSupport:
    """Tests for V4.1 contract schema support in conversion."""

    def test_convert_with_v41_inverted_column_roles(self):
        """Test conversion with V4.1 column -> role format populates outcome_analysis."""
        dataset_profile = {
            "rows": 1309,
            "cols": 13,
            "columns": ["PassengerId", "Survived", "Pclass", "Age"],
            "type_hints": {
                "PassengerId": "numeric",
                "Survived": "numeric",
                "Pclass": "numeric",
                "Age": "numeric",
            },
            "missing_frac": {
                "PassengerId": 0.0,
                "Survived": 0.319,  # ~32% missing (test set)
                "Pclass": 0.0,
                "Age": 0.2,
            },
            "cardinality": {
                "PassengerId": {"unique": 1309, "top_values": []},
                "Survived": {"unique": 2, "top_values": [
                    {"value": "0.0", "count": 549},
                    {"value": "1.0", "count": 342},
                ]},
                "Pclass": {"unique": 3, "top_values": []},
                "Age": {"unique": 98, "top_values": []},
            },
        }

        # V4.1 inverted format: column -> role
        contract_v41 = {
            "column_roles": {
                "PassengerId": "identifier",
                "Survived": "outcome",
                "Pclass": "feature",
                "Age": "feature",
            }
        }

        result = convert_dataset_profile_to_data_profile(
            dataset_profile, contract_v41, analysis_type="classification"
        )

        # Must detect Survived as outcome
        assert "Survived" in result["outcome_analysis"]
        outcome = result["outcome_analysis"]["Survived"]
        assert outcome["present"] is True
        # null_frac should be ~0.319 from missing_frac
        assert abs(outcome["null_frac"] - 0.319) < 0.01
        assert outcome["total_count"] == 1309
        # non_null_count = 1309 * (1 - 0.319) ≈ 891
        assert outcome["non_null_count"] == int(1309 * (1 - 0.319))

    def test_convert_with_v41_role_dict_format(self):
        """Test conversion with V4.1 column -> {role: ...} format."""
        dataset_profile = {
            "rows": 100,
            "cols": 3,
            "columns": ["id", "target", "feature1"],
            "type_hints": {"id": "numeric", "target": "numeric", "feature1": "numeric"},
            "missing_frac": {"id": 0.0, "target": 0.25, "feature1": 0.0},
            "cardinality": {
                "id": {"unique": 100, "top_values": []},
                "target": {"unique": 2, "top_values": []},
                "feature1": {"unique": 50, "top_values": []},
            },
        }

        contract_v41 = {
            "column_roles": {
                "id": {"role": "identifier"},
                "target": {"role": "outcome", "nullable": True},
                "feature1": {"role": "feature"},
            }
        }

        result = convert_dataset_profile_to_data_profile(dataset_profile, contract_v41)

        assert "target" in result["outcome_analysis"]
        assert result["outcome_analysis"]["target"]["null_frac"] == 0.25

    def test_validation_detects_missing_labels_from_v41_contract(self):
        """Test that validation catches missing labels with V4.1 contract format."""
        dataset_profile = {
            "rows": 1309,
            "cols": 2,
            "columns": ["Survived", "Age"],
            "type_hints": {"Survived": "numeric", "Age": "numeric"},
            "missing_frac": {"Survived": 0.32, "Age": 0.2},
            "cardinality": {
                "Survived": {"unique": 2, "top_values": []},
                "Age": {"unique": 98, "top_values": []},
            },
        }

        contract_v41 = {"column_roles": {"Survived": "outcome", "Age": "feature"}}
        strategy = {"analysis_type": "classification"}

        data_profile = convert_dataset_profile_to_data_profile(
            dataset_profile, contract_v41, analysis_type="classification"
        )

        # Plan that incorrectly uses all rows
        bad_plan = {
            "training_rows_policy": "use_all_rows",
            "metric_policy": {"primary_metric": "roc_auc"},
            "plan_source": "llm",
            "evidence_used": {},
        }

        result = validate_ml_plan_constraints(bad_plan, data_profile, contract_v41, strategy)

        # Must detect the violation
        assert result["ok"] is False
        assert any("use_all_rows" in v for v in result["violations"])
