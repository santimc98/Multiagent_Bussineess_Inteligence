"""
Tests for contract_validator strict normalization (V4.1 auto-repair).

Tests the following normalization behaviors:
  - normalize_allowed_feature_sets: list -> dict, nested lists -> flattened, synonym unification
  - normalize_validation_requirements: metric canonicalization, metrics -> metrics_to_report migration
  - lint_scored_rows_schema: removal of metric-like columns from required_columns
  - validate_contract integration: issues/unknowns traceability, status escalation to error on invalid types
"""
import pytest
from src.utils.contract_validator import (
    _normalize_metric_name,
    _is_metric_like_token,
    normalize_allowed_feature_sets,
    normalize_validation_requirements,
    lint_scored_rows_schema,
    validate_contract,
)


class TestNormalizeMetricName:
    """Test _normalize_metric_name helper."""

    def test_basic_lowercase(self):
        assert _normalize_metric_name("RMSE") == "rmse"
        assert _normalize_metric_name("Accuracy") == "accuracy"

    def test_replace_spaces_and_hyphens(self):
        assert _normalize_metric_name("ROC-AUC") == "roc_auc"
        assert _normalize_metric_name("ROC AUC") == "roc_auc"
        assert _normalize_metric_name("roc auc") == "roc_auc"

    def test_roc_auc_variants(self):
        assert _normalize_metric_name("roc-auc") == "roc_auc"
        assert _normalize_metric_name("roc_auc") == "roc_auc"
        assert _normalize_metric_name("auc_roc") == "roc_auc"
        assert _normalize_metric_name("auroc") == "roc_auc"

    def test_rmse_log_variants(self):
        assert _normalize_metric_name("RMSLE") == "rmsle"
        assert _normalize_metric_name("rmse_log1p") == "rmsle"
        assert _normalize_metric_name("RMSE_log") == "rmsle"
        assert _normalize_metric_name("log_rmse") == "rmsle"

    def test_parentheses_removal(self):
        # Parenthetical annotations are stripped entirely
        assert _normalize_metric_name("F1 (macro)") == "f1"
        assert _normalize_metric_name("accuracy (weighted)") == "accuracy"

    def test_non_string_input(self):
        assert _normalize_metric_name(123) == "123"
        assert _normalize_metric_name(None) == ""


class TestIsMetricLikeToken:
    """Test _is_metric_like_token helper."""

    def test_known_metrics(self):
        assert _is_metric_like_token("accuracy") is True
        assert _is_metric_like_token("Accuracy") is True
        assert _is_metric_like_token("ROC-AUC") is True
        assert _is_metric_like_token("f1") is True
        assert _is_metric_like_token("F1-Score") is True
        assert _is_metric_like_token("rmsle") is True

    def test_non_metrics(self):
        assert _is_metric_like_token("Id") is False
        assert _is_metric_like_token("Survived") is False
        assert _is_metric_like_token("prediction") is False
        assert _is_metric_like_token("probability") is False
        assert _is_metric_like_token("Pclass") is False

    def test_empty_and_none(self):
        assert _is_metric_like_token("") is False
        assert _is_metric_like_token(None) is False


class TestNormalizeAllowedFeatureSets:
    """Test normalize_allowed_feature_sets function."""

    def test_list_to_dict(self):
        """allowed_feature_sets = ["A", "B"] -> dict with model_features."""
        contract = {"allowed_feature_sets": ["A", "B", "C"]}
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert isinstance(result, dict)
        assert result["model_features"] == ["A", "B", "C"]
        assert result["segmentation_features"] == []
        assert result["audit_only_features"] == []
        assert result["forbidden_for_modeling"] == []
        assert result["rationale"] == "normalized_from_list"
        assert len(notes) == 1
        assert "list to dict" in notes[0]

    def test_nested_list_flatten(self):
        """allowed_feature_sets = [["A", "B"]] -> flattened dict."""
        contract = {"allowed_feature_sets": [["Pclass", "Sex", "Age"]]}
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["model_features"] == ["Pclass", "Sex", "Age"]
        assert result["rationale"] == "normalized_from_list"

    def test_mixed_nested_list(self):
        """allowed_feature_sets = ["A", ["B", "C"]] -> flattened."""
        contract = {"allowed_feature_sets": ["A", ["B", "C"], "D"]}
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["model_features"] == ["A", "B", "C", "D"]

    def test_dict_with_forbidden_synonym(self):
        """forbidden_features -> forbidden_for_modeling."""
        contract = {
            "allowed_feature_sets": {
                "model_features": ["A", "B"],
                "forbidden_features": ["target", "id"],
            }
        }
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["forbidden_for_modeling"] == ["target", "id"]
        assert "forbidden_features" not in result
        assert any("Unified" in n for n in notes)

    def test_dict_with_forbidden_key(self):
        """forbidden -> forbidden_for_modeling."""
        contract = {
            "allowed_feature_sets": {
                "model_features": ["X"],
                "forbidden": ["Y"],
            }
        }
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["forbidden_for_modeling"] == ["Y"]

    def test_dict_already_correct(self):
        """Properly structured dict passes through with minimal changes."""
        contract = {
            "allowed_feature_sets": {
                "model_features": ["A", "B"],
                "segmentation_features": ["C"],
                "audit_only_features": [],
                "forbidden_for_modeling": ["target"],
                "rationale": "Expert selection",
            }
        }
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["model_features"] == ["A", "B"]
        assert result["segmentation_features"] == ["C"]
        assert result["forbidden_for_modeling"] == ["target"]
        assert result["rationale"] == "Expert selection"
        assert len(notes) == 0

    def test_dict_deduplicate(self):
        """Duplicate values in lists are removed."""
        contract = {
            "allowed_feature_sets": {
                "model_features": ["A", "B", "A", "B", "C"],
            }
        }
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["model_features"] == ["A", "B", "C"]

    def test_invalid_type_int(self):
        """allowed_feature_sets = 123 -> error status."""
        contract = {"allowed_feature_sets": 123}
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["rationale"].startswith("ERROR:")
        assert any("invalid type" in n.lower() for n in notes)

    def test_invalid_type_string(self):
        """allowed_feature_sets = 'some string' -> error."""
        contract = {"allowed_feature_sets": "Pclass,Sex,Age"}
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["rationale"].startswith("ERROR:")

    def test_none_value(self):
        """allowed_feature_sets = None -> empty dict with no error."""
        contract = {"allowed_feature_sets": None}
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["model_features"] == []
        assert result["rationale"] == ""
        assert len(notes) == 0

    def test_missing_key(self):
        """No allowed_feature_sets key -> empty dict."""
        contract = {}
        result, notes = normalize_allowed_feature_sets(contract)
        
        assert result["model_features"] == []
        assert len(notes) == 0


class TestNormalizeValidationRequirements:
    """Test normalize_validation_requirements function."""

    def test_metrics_migration(self):
        """validation_requirements.metrics -> metrics_to_report."""
        contract = {
            "validation_requirements": {
                "metrics": ["RMSE_log1p", "accuracy"],
            }
        }
        result, notes = normalize_validation_requirements(contract)
        
        assert "rmsle" in result["metrics_to_report"]
        assert "accuracy" in result["metrics_to_report"]
        assert "metrics" not in result
        assert any("Migrated" in n for n in notes)

    def test_primary_metric_canonicalization(self):
        """primary_metric is canonicalized."""
        contract = {
            "validation_requirements": {
                "primary_metric": "ROC-AUC",
                "metrics_to_report": ["accuracy"],
            }
        }
        result, notes = normalize_validation_requirements(contract)
        
        assert result["primary_metric"] == "roc_auc"
        assert "roc_auc" in result["metrics_to_report"]

    def test_primary_metric_added_to_metrics(self):
        """primary_metric is added to metrics_to_report if missing."""
        contract = {
            "validation_requirements": {
                "primary_metric": "rmse",
                "metrics_to_report": ["mae"],
            }
        }
        result, notes = normalize_validation_requirements(contract)
        
        assert "rmse" in result["metrics_to_report"]
        assert "mae" in result["metrics_to_report"]

    def test_infer_primary_from_qa_gates(self):
        """Primary metric inferred from qa_gates benchmark gate."""
        contract = {
            "validation_requirements": {},
            "qa_gates": [
                {"name": "benchmark_kpi_report", "params": {"metric": "accuracy"}}
            ]
        }
        result, notes = normalize_validation_requirements(contract)
        
        assert result["primary_metric"] == "accuracy"
        assert any("Inferred" in n for n in notes)

    def test_deduplication(self):
        """Duplicate metrics are removed."""
        contract = {
            "validation_requirements": {
                "metrics_to_report": ["accuracy", "Accuracy", "ACCURACY"],
            }
        }
        result, notes = normalize_validation_requirements(contract)
        
        assert result["metrics_to_report"].count("accuracy") == 1


class TestLintScoredRowsSchema:
    """Test lint_scored_rows_schema function."""

    def test_remove_metric_columns(self):
        """Metric-like columns are removed."""
        contract = {
            "artifact_requirements": {
                "scored_rows_schema": {
                    "required_columns": ["Id", "prediction", "Accuracy", "ROC-AUC", "F1-Score"]
                }
            }
        }
        clean, removed, notes = lint_scored_rows_schema(contract)
        
        assert "Id" in clean
        assert "prediction" in clean
        assert "Accuracy" not in clean
        assert "ROC-AUC" not in clean
        assert "F1-Score" not in clean
        assert "Accuracy" in removed
        assert "ROC-AUC" in removed
        assert "F1-Score" in removed
        assert len(notes) == 3

    def test_no_metrics_present(self):
        """Non-metric columns are preserved."""
        contract = {
            "artifact_requirements": {
                "scored_rows_schema": {
                    "required_columns": ["Id", "Survived", "prediction", "probability"]
                }
            }
        }
        clean, removed, notes = lint_scored_rows_schema(contract)
        
        assert clean == ["Id", "Survived", "prediction", "probability"]
        assert removed == []
        assert notes == []

    def test_missing_schema(self):
        """Missing schema returns empty results."""
        contract = {}
        clean, removed, notes = lint_scored_rows_schema(contract)
        
        assert clean == []
        assert removed == []
        assert notes == []


class TestValidateContractIntegration:
    """Test full validate_contract integration with normalization."""

    def test_list_feature_sets_normalized(self):
        """allowed_feature_sets list is normalized to dict."""
        contract = {
            "allowed_feature_sets": ["A", "B", "C"],
            "canonical_columns": ["A", "B", "C", "target"],
        }
        result = validate_contract(contract)
        
        # Contract should be mutated
        assert isinstance(contract["allowed_feature_sets"], dict)
        assert contract["allowed_feature_sets"]["model_features"] == ["A", "B", "C"]
        
        # Should have normalization issue
        issues = result["issues"]
        assert any("contract_normalization" in i["rule"] for i in issues)

    def test_nested_list_normalized(self):
        """Nested list [[...]] is flattened."""
        contract = {
            "allowed_feature_sets": [["Pclass", "Sex", "Age"]],
        }
        result = validate_contract(contract)
        
        assert contract["allowed_feature_sets"]["model_features"] == ["Pclass", "Sex", "Age"]

    def test_invalid_type_returns_error(self):
        """Invalid allowed_feature_sets type -> status=error."""
        contract = {
            "allowed_feature_sets": 123,
        }
        result = validate_contract(contract)
        
        assert result["status"] == "error"
        assert any(
            i["severity"] == "error" and "allowed_feature_sets" in str(i["item"])
            for i in result["issues"]
        )

    def test_metric_columns_removed(self):
        """Metric-like columns are removed from scored_rows_schema."""
        contract = {
            "artifact_requirements": {
                "scored_rows_schema": {
                    "required_columns": ["Id", "prediction", "Accuracy", "ROC-AUC"]
                }
            }
        }
        result = validate_contract(contract)
        
        scored_cols = contract["artifact_requirements"]["scored_rows_schema"]["required_columns"]
        assert "Accuracy" not in scored_cols
        assert "ROC-AUC" not in scored_cols
        assert "Id" in scored_cols
        assert "prediction" in scored_cols

    def test_validation_requirements_normalized(self):
        """validation_requirements.metrics is migrated."""
        contract = {
            "validation_requirements": {
                "metrics": ["RMSE_log1p"],
            }
        }
        result = validate_contract(contract)
        
        val_req = contract["validation_requirements"]
        assert "metrics" not in val_req
        assert "rmsle" in val_req["metrics_to_report"]

    def test_traceability_in_unknowns(self):
        """Repairs are added to unknowns list."""
        contract = {
            "unknowns": [],
            "allowed_feature_sets": ["A", "B"],
        }
        result = validate_contract(contract)
        
        # Unknowns should have the normalization note
        assert any("Normalized" in u for u in contract.get("unknowns", []))

    def test_forbidden_features_synonym(self):
        """forbidden_features is unified to forbidden_for_modeling."""
        contract = {
            "allowed_feature_sets": {
                "model_features": ["A"],
                "forbidden_features": ["target"],
            }
        }
        result = validate_contract(contract)
        
        afs = contract["allowed_feature_sets"]
        assert "forbidden_features" not in afs
        assert afs["forbidden_for_modeling"] == ["target"]

    def test_existing_valid_contract_passes(self):
        """Properly structured contract passes with ok/warning status."""
        contract = {
            "canonical_columns": ["A", "B", "target"],
            "column_roles": {"target": "outcome", "A": "feature", "B": "feature"},
            "allowed_feature_sets": {
                "model_features": ["A", "B"],
                "segmentation_features": [],
                "audit_only_features": [],
                "forbidden_for_modeling": ["target"],
                "rationale": "Preselected",
            },
            "validation_requirements": {
                "primary_metric": "accuracy",
                "metrics_to_report": ["accuracy", "roc_auc"],
            },
            "artifact_requirements": {
                "required_files": [{"path": "data/metrics.json"}],
                "scored_rows_schema": {
                    "required_columns": ["Id", "prediction"],
                },
            },
        }
        result = validate_contract(contract)

        # Should not be error (target is properly defined as outcome)
        assert result["status"] in ("ok", "warning")
