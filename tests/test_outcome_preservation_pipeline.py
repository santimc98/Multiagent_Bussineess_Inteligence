"""
Tests for outcome preservation pipeline (Fix #7).

Tests that supervised learning contracts properly preserve outcome columns
through the pipeline from Execution Planner → Data Engineer → ML Engineer.
"""

import pytest

from src.utils.contract_validator import (
    lint_column_roles,
    lint_outcome_presence_and_coherence,
    run_contract_schema_linter,
    validate_contract,
    _is_supervised_contract,
    _extract_role_from_value,
)
from src.utils.contract_v41 import get_outcome_columns


class TestExtractRoleFromValue:
    """Tests for _extract_role_from_value helper."""

    def test_string_role_passthrough(self):
        """String role should pass through with normalization."""
        role, from_dict = _extract_role_from_value("target")
        assert role == "outcome"  # target is normalized to outcome
        assert from_dict is False

    def test_dict_with_role_key(self):
        """Dict with 'role' key should extract role."""
        role, from_dict = _extract_role_from_value({"role": "outcome"})
        assert role == "outcome"
        assert from_dict is True

    def test_dict_with_column_role_key(self):
        """Dict with 'column_role' key should extract role."""
        role, from_dict = _extract_role_from_value({"column_role": "target"})
        assert role == "outcome"  # target normalized to outcome
        assert from_dict is True

    def test_dict_with_type_key(self):
        """Dict with 'type' key should extract role."""
        role, from_dict = _extract_role_from_value({"type": "id"})
        assert role == "id"
        assert from_dict is True

    def test_dict_without_recognizable_key(self):
        """Dict without recognizable role key should default to feature."""
        role, from_dict = _extract_role_from_value({"description": "some column"})
        assert role == "feature"
        assert from_dict is True

    def test_int_defaults_to_feature(self):
        """Non-string, non-dict value should default to feature."""
        role, from_dict = _extract_role_from_value(42)
        assert role == "feature"
        assert from_dict is False


class TestLintColumnRolesDictValues:
    """Tests for lint_column_roles handling dict values."""

    def test_dict_value_with_role_extracts_outcome(self):
        """
        Case 3: column_roles dict-values parse.
        {"Survived": {"role": "outcome"}} should end up as "outcome" (not default feature).
        """
        contract = {
            "column_roles": {
                "Survived": {"role": "outcome"},
                "PassengerId": {"role": "id"},
                "Age": {"role": "feature"},
            }
        }
        normalized, issues, notes = lint_column_roles(contract)

        assert normalized["Survived"] == "outcome"
        assert normalized["PassengerId"] == "id"
        assert normalized["Age"] == "feature"

        # Should have info issues about parsing from dict
        info_issues = [i for i in issues if i.get("severity") == "info"]
        assert len(info_issues) == 3

    def test_dict_value_with_target_normalizes_to_outcome(self):
        """Dict with target role should normalize to outcome."""
        contract = {
            "column_roles": {
                "SalePrice": {"role": "target"},
            }
        }
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized["SalePrice"] == "outcome"

    def test_dict_value_with_label_normalizes_to_outcome(self):
        """Dict with label role should normalize to outcome."""
        contract = {
            "column_roles": {
                "y": {"column_role": "label"},
            }
        }
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized["y"] == "outcome"

    def test_mixed_string_and_dict_values(self):
        """Should handle mix of string and dict role values."""
        contract = {
            "column_roles": {
                "target": "outcome",
                "id_col": {"role": "id"},
                "feature1": "feature",
            }
        }
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized["target"] == "outcome"
        assert normalized["id_col"] == "id"
        assert normalized["feature1"] == "feature"


class TestIsSupervisedContract:
    """Tests for _is_supervised_contract detection."""

    def test_rmse_primary_metric_is_supervised(self):
        """Contract with RMSE primary metric should be supervised."""
        contract = {
            "validation_requirements": {
                "primary_metric": "rmse"
            }
        }
        assert _is_supervised_contract(contract) is True

    def test_roc_auc_primary_metric_is_supervised(self):
        """Contract with ROC_AUC primary metric should be supervised."""
        contract = {
            "validation_requirements": {
                "primary_metric": "roc_auc"
            }
        }
        assert _is_supervised_contract(contract) is True

    def test_accuracy_in_metrics_to_report_is_supervised(self):
        """Contract with accuracy in metrics_to_report should be supervised."""
        contract = {
            "validation_requirements": {
                "metrics_to_report": ["accuracy", "f1"]
            }
        }
        assert _is_supervised_contract(contract) is True

    def test_no_metric_is_not_supervised(self):
        """Contract without supervised metrics should not be supervised."""
        contract = {
            "validation_requirements": {}
        }
        assert _is_supervised_contract(contract) is False

    def test_benchmark_gate_is_supervised(self):
        """Contract with benchmark QA gate should be supervised."""
        contract = {
            "qa_gates": [
                {"name": "benchmark_kpi_report", "severity": "HARD"}
            ]
        }
        assert _is_supervised_contract(contract) is True


class TestLintOutcomePresenceAndCoherence:
    """Tests for lint_outcome_presence_and_coherence function."""

    def test_supervised_without_outcome_is_critical_error(self):
        """
        Case 2: Contrato supervisado sin ninguna forma de outcome.
        validate_contract() => status="error"
        """
        contract = {
            "validation_requirements": {
                "primary_metric": "rmse"
            },
            "canonical_columns": ["feature1", "feature2"],
            "column_roles": {},
        }
        issues, notes, critical = lint_outcome_presence_and_coherence(contract)

        assert critical is True
        assert any("outcome_required" in i.get("rule", "") for i in issues)
        assert any(i.get("severity") == "error" for i in issues)

    def test_supervised_with_column_role_outcome_is_ok(self):
        """Supervised contract with outcome in column_roles should be ok."""
        contract = {
            "validation_requirements": {
                "primary_metric": "rmse"
            },
            "canonical_columns": ["SalePrice", "feature1"],
            "column_roles": {"SalePrice": "outcome"},
            "allowed_feature_sets": {"model_features": [], "forbidden_for_modeling": []},
        }
        issues, notes, critical = lint_outcome_presence_and_coherence(contract)

        assert critical is False
        # Should have set outcome_columns
        assert contract["outcome_columns"] == ["SalePrice"]

    def test_auto_repair_adds_outcome_to_canonical(self):
        """
        Case 1: Contrato supervisado con column_roles={"SalePrice":{"role":"target"}}
        y canonical_columns sin SalePrice:
        - validate_contract() auto-repara: SalePrice ∈ canonical_columns
        """
        contract = {
            "validation_requirements": {
                "primary_metric": "rmse"
            },
            "canonical_columns": ["feature1", "feature2"],
            "column_roles": {"SalePrice": "outcome"},
            "allowed_feature_sets": {"model_features": [], "forbidden_for_modeling": []},
        }
        issues, notes, critical = lint_outcome_presence_and_coherence(contract)

        assert critical is False
        # Should have auto-added SalePrice to canonical_columns
        assert "SalePrice" in contract["canonical_columns"]
        # Should have warning about auto-add
        assert any("Auto-added outcome" in i.get("message", "") for i in issues)

    def test_auto_repair_adds_outcome_to_forbidden(self):
        """Outcome should be auto-added to forbidden_for_modeling."""
        contract = {
            "validation_requirements": {
                "primary_metric": "accuracy"
            },
            "canonical_columns": ["target"],
            "column_roles": {"target": "outcome"},
            "allowed_feature_sets": {"model_features": ["feature1"], "forbidden_for_modeling": []},
        }
        issues, notes, critical = lint_outcome_presence_and_coherence(contract)

        assert critical is False
        assert "target" in contract["allowed_feature_sets"]["forbidden_for_modeling"]

    def test_infer_outcome_from_validation_requirements_params(self):
        """Should infer outcome from validation_requirements.params.target."""
        contract = {
            "validation_requirements": {
                "primary_metric": "rmse",
                "params": {"target": "SalePrice"}
            },
            "canonical_columns": [],
            "column_roles": {},
            "allowed_feature_sets": {"model_features": [], "forbidden_for_modeling": []},
        }
        issues, notes, critical = lint_outcome_presence_and_coherence(contract)

        assert critical is False
        assert contract["outcome_columns"] == ["SalePrice"]
        assert "SalePrice" in contract["canonical_columns"]

    def test_infer_outcome_from_objective_analysis(self):
        """Should infer outcome from objective_analysis.target_column."""
        contract = {
            "validation_requirements": {
                "primary_metric": "f1"
            },
            "objective_analysis": {
                "target_column": "Survived"
            },
            "canonical_columns": [],
            "column_roles": {},
            "allowed_feature_sets": {"model_features": [], "forbidden_for_modeling": []},
        }
        issues, notes, critical = lint_outcome_presence_and_coherence(contract)

        assert critical is False
        assert contract["outcome_columns"] == ["Survived"]


class TestValidateContractOutcomeIntegration:
    """Integration tests for validate_contract with outcome validation."""

    def test_supervised_no_outcome_returns_error_status(self):
        """validate_contract returns error status when supervised but no outcome."""
        contract = {
            "validation_requirements": {
                "primary_metric": "rmse"
            },
            "canonical_columns": ["feature1"],
            "artifact_requirements": {
                "required_files": [{"path": "data/cleaned_data.csv"}],
                "scored_rows_schema": {"required_columns": []}
            }
        }
        result = validate_contract(contract)

        assert result["status"] == "error"
        # Should have issue about missing outcome
        outcome_issues = [
            i for i in result["issues"]
            if "outcome" in i.get("rule", "").lower() or "outcome" in i.get("message", "").lower()
        ]
        assert len(outcome_issues) > 0

    def test_dict_role_values_properly_parsed(self):
        """validate_contract properly parses dict role values."""
        contract = {
            "validation_requirements": {
                "primary_metric": "rmse"
            },
            "column_roles": {
                "SalePrice": {"role": "target"},
                "Id": {"role": "id"},
            },
            "canonical_columns": ["Id"],
            "artifact_requirements": {
                "required_files": [{"path": "data/cleaned_data.csv"}],
                "scored_rows_schema": {"required_columns": []}
            }
        }
        result = validate_contract(contract)

        # Should NOT be error because outcome is inferable
        assert result["status"] != "error" or all(
            "outcome_required" not in i.get("rule", "") for i in result["issues"]
        )
        # SalePrice should be in canonical_columns now
        assert "SalePrice" in contract["canonical_columns"]

    def test_house_prices_scenario_auto_repair(self):
        """
        Reproduce House Prices bug:
        - Contract with SalePrice as target but not in canonical_columns
        - Should auto-repair to include SalePrice
        """
        contract = {
            "business_objective": "Predict house sale prices",
            "validation_requirements": {
                "primary_metric": "rmse",
                "params": {"target": "SalePrice"}
            },
            "canonical_columns": ["OverallQual", "GrLivArea", "GarageCars"],  # Missing SalePrice!
            "column_roles": {},
            "artifact_requirements": {
                "required_files": [{"path": "data/cleaned_data.csv"}],
                "scored_rows_schema": {"required_columns": []}
            },
            "allowed_feature_sets": {
                "model_features": ["OverallQual", "GrLivArea", "GarageCars"],
                "forbidden_for_modeling": []
            }
        }
        result = validate_contract(contract)

        # Should auto-repair: SalePrice in canonical_columns
        assert "SalePrice" in contract["canonical_columns"]
        # Should auto-repair: SalePrice in outcome_columns
        assert contract.get("outcome_columns") == ["SalePrice"]
        # Should auto-repair: SalePrice in forbidden_for_modeling
        assert "SalePrice" in contract["allowed_feature_sets"]["forbidden_for_modeling"]
        # Should have unknowns notes about repairs
        assert len(contract.get("unknowns", [])) > 0


class TestGetOutcomeColumnsV41:
    """Tests for get_outcome_columns in contract_v41.py."""

    def test_explicit_outcome_columns(self):
        """Should return explicit outcome_columns if provided."""
        contract = {"outcome_columns": ["target"]}
        assert get_outcome_columns(contract) == ["target"]

    def test_column_roles_outcome(self):
        """Should return from column_roles if outcome role exists."""
        contract = {
            "column_roles": {
                "outcome": ["Survived"]
            }
        }
        assert get_outcome_columns(contract) == ["Survived"]

    def test_validation_requirements_params_target(self):
        """Should return from validation_requirements.params.target."""
        contract = {
            "validation_requirements": {
                "params": {"target": "SalePrice"}
            }
        }
        assert get_outcome_columns(contract) == ["SalePrice"]

    def test_objective_analysis_fallback(self):
        """Should return from objective_analysis as last fallback."""
        contract = {
            "objective_analysis": {
                "target_column": "Survived"
            }
        }
        assert get_outcome_columns(contract) == ["Survived"]

    def test_unknown_filtered_out(self):
        """Should filter out 'unknown' values."""
        contract = {"outcome_columns": ["target", "unknown", "Unknown"]}
        assert get_outcome_columns(contract) == ["target"]


class TestRunContractSchemaLinterIntegration:
    """Integration tests for run_contract_schema_linter."""

    def test_returns_critical_error_for_missing_outcome(self):
        """Linter should return critical error for supervised without outcome."""
        contract = {
            "validation_requirements": {"primary_metric": "accuracy"},
            "canonical_columns": ["feature1"],
            "artifact_requirements": {
                "required_files": [],
                "scored_rows_schema": {"required_columns": []}
            }
        }
        _, issues, notes, critical = run_contract_schema_linter(contract)

        assert critical is True

    def test_no_critical_error_for_non_supervised(self):
        """Non-supervised contract should not have critical error for missing outcome."""
        contract = {
            "validation_requirements": {},
            "canonical_columns": ["feature1"],
            "artifact_requirements": {
                "required_files": [],
                "scored_rows_schema": {"required_columns": []}
            }
        }
        _, issues, notes, critical = run_contract_schema_linter(contract)

        assert critical is False
