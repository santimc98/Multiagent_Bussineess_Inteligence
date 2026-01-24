"""
Tests for Contract Schema Linter (Fix #6).

Tests the deterministic linting and auto-repair of contract schemas.
"""

import pytest

from src.utils.contract_validator import (
    lint_column_roles,
    lint_required_columns,
    lint_allowed_feature_sets_coherence,
    lint_artifact_requirements_coherence,
    run_contract_schema_linter,
    validate_contract,
)


class TestLintColumnRoles:
    """Tests for lint_column_roles function."""

    def test_dict_format_passthrough(self):
        """Standard dict format should pass through with normalized roles."""
        contract = {
            "column_roles": {
                "target_col": "outcome",
                "id_col": "id",
                "feature1": "feature",
            }
        }
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized == {
            "target_col": "outcome",
            "id_col": "id",
            "feature1": "feature",
        }
        assert len(issues) == 0

    def test_list_dict_to_dict_normalized(self):
        """
        Case 1: column_roles list[dict] -> dict normalized.
        """
        contract = {
            "column_roles": [
                {"column": "X", "role": "outcome"},
                {"column": "Y", "role": "feature"},
                {"name": "Z", "type": "id"},
            ]
        }
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized == {"X": "outcome", "Y": "feature", "Z": "id"}
        assert any("normalized from list" in n for n in notes)

    def test_list_list_to_dict_normalized(self):
        """
        Case 2: column_roles list[list] -> dict normalized.
        """
        contract = {
            "column_roles": [
                ["X", "outcome"],
                ["Y", "feature"],
                ("Z", "id"),  # tuple also works
            ]
        }
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized == {"X": "outcome", "Y": "feature", "Z": "id"}
        assert any("normalized from list" in n for n in notes)

    def test_invalid_type_str_returns_error(self):
        """
        Case 3: column_roles tipo str -> status error.
        """
        contract = {"column_roles": "not a valid format"}
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized == {}
        assert len(issues) == 1
        assert issues[0]["severity"] == "fail"
        assert "invalid type" in issues[0]["message"]

    def test_invalid_type_int_returns_error(self):
        """
        Case 3: column_roles tipo int -> status error.
        """
        contract = {"column_roles": 42}
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized == {}
        assert len(issues) == 1
        assert issues[0]["severity"] == "fail"
        assert "invalid type" in issues[0]["message"]

    def test_unknown_role_triggers_warning(self):
        """Unknown roles should trigger warning but keep literal."""
        contract = {
            "column_roles": {
                "col1": "completely_custom_role",
            }
        }
        normalized, issues, notes = lint_column_roles(contract)
        assert "col1" in normalized
        # Either keeps literal or defaults to feature
        assert normalized["col1"] in ("completely_custom_role", "feature")
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "Unknown role" in issues[0]["message"]

    def test_role_synonym_normalization(self):
        """Role synonyms should be normalized to canonical form."""
        contract = {
            "column_roles": {
                "target": "target",  # should become "outcome"
                "idx": "identifier",  # should become "id"
            }
        }
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized["target"] == "outcome"
        assert normalized["idx"] == "id"

    def test_none_returns_empty_dict(self):
        """None column_roles should return empty dict without issues."""
        contract = {"column_roles": None}
        normalized, issues, notes = lint_column_roles(contract)
        assert normalized == {}
        assert len(issues) == 0


class TestLintRequiredColumns:
    """Tests for lint_required_columns function."""

    def test_metric_like_tokens_removed(self):
        """
        Case 4: scored_rows_schema.required_columns con métricas -> se eliminan + warning.
        """
        required = ["id", "score", "accuracy", "roc_auc", "prediction"]
        clean, issues, notes = lint_required_columns(required)
        assert "accuracy" not in clean
        assert "roc_auc" not in clean
        assert "id" in clean
        assert "score" in clean
        assert "prediction" in clean
        # Should have warnings for removed metrics
        metric_issues = [i for i in issues if "metric-like" in i["message"]]
        assert len(metric_issues) == 2  # accuracy and roc_auc

    def test_path_like_values_removed(self):
        """
        Case 5: required_columns con "metrics.json" o "scored_rows.csv" -> se elimina + warning.
        """
        required = ["id", "metrics.json", "data/scored_rows.csv", "prediction"]
        clean, issues, notes = lint_required_columns(required)
        assert "metrics.json" not in clean
        assert "data/scored_rows.csv" not in clean
        assert "id" in clean
        assert "prediction" in clean
        # Should have warnings for removed paths
        path_issues = [i for i in issues if "path-like" in i["message"]]
        assert len(path_issues) == 2

    def test_string_converted_to_list(self):
        """Single string should be converted to list."""
        clean, issues, notes = lint_required_columns("single_column")
        assert clean == ["single_column"]
        assert any("converted single string" in n for n in notes)

    def test_deduplication(self):
        """Duplicate columns should be removed."""
        required = ["id", "score", "ID", "Score", "id"]  # case-insensitive dedup
        clean, issues, notes = lint_required_columns(required)
        # Should have only one of each (case-preserved for first occurrence)
        assert len(clean) == 2
        assert "id" in clean
        assert "score" in clean

    def test_invalid_type_returns_empty(self):
        """Invalid types should return empty list with warning."""
        clean, issues, notes = lint_required_columns({"not": "a list"})
        assert clean == []
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"

    def test_mixed_formats_handled(self):
        """Mixed str and dict formats should be handled."""
        required = [
            "col1",
            {"name": "col2"},
            {"column": "col3"},
            None,  # should be skipped
            123,   # should be skipped
        ]
        clean, issues, notes = lint_required_columns(required)
        assert "col1" in clean
        assert "col2" in clean
        assert "col3" in clean
        assert len(clean) == 3


class TestLintAllowedFeatureSetsCoherence:
    """Tests for lint_allowed_feature_sets_coherence function."""

    def test_overlap_forbidden_model_features_auto_repair(self):
        """
        Case 6: overlap forbidden_for_modeling vs model_features -> auto-repair quitando del model_features.
        """
        contract = {
            "allowed_feature_sets": {
                "model_features": ["feature1", "feature2", "forbidden_col", "feature3"],
                "forbidden_for_modeling": ["forbidden_col", "other_forbidden"],
            }
        }
        repaired, issues, notes = lint_allowed_feature_sets_coherence(contract)
        assert "forbidden_col" not in repaired["model_features"]
        assert "feature1" in repaired["model_features"]
        assert "feature2" in repaired["model_features"]
        assert "feature3" in repaired["model_features"]
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "removed from model_features" in issues[0]["message"]

    def test_outcome_in_model_features_auto_repair(self):
        """
        Case 7: outcome/decision dentro de model_features -> warning + auto-repair quitándolas.
        """
        contract = {
            "column_roles": {
                "target": "outcome",
                "decision_col": "decision",
            },
            "allowed_feature_sets": {
                "model_features": ["feature1", "target", "decision_col", "feature2"],
                "forbidden_for_modeling": [],
            }
        }
        repaired, issues, notes = lint_allowed_feature_sets_coherence(contract)
        assert "target" not in repaired["model_features"]
        assert "decision_col" not in repaired["model_features"]
        assert "feature1" in repaired["model_features"]
        assert "feature2" in repaired["model_features"]
        # Should have 2 warnings (one for outcome, one for decision)
        leakage_issues = [i for i in issues if "Leakage-by-contract" in i["message"]]
        assert len(leakage_issues) == 2

    def test_outcome_columns_explicit(self):
        """Outcome columns from explicit outcome_columns list should be detected."""
        contract = {
            "outcome_columns": ["target_explicit"],
            "allowed_feature_sets": {
                "model_features": ["feature1", "target_explicit"],
                "forbidden_for_modeling": [],
            }
        }
        repaired, issues, notes = lint_allowed_feature_sets_coherence(contract)
        assert "target_explicit" not in repaired["model_features"]
        assert any("Leakage-by-contract" in i["message"] for i in issues)

    def test_no_issues_when_coherent(self):
        """No issues when feature sets are coherent."""
        contract = {
            "column_roles": {"target": "outcome"},
            "allowed_feature_sets": {
                "model_features": ["feature1", "feature2"],
                "forbidden_for_modeling": ["forbidden1"],
            }
        }
        repaired, issues, notes = lint_allowed_feature_sets_coherence(contract)
        assert len(issues) == 0
        assert repaired["model_features"] == ["feature1", "feature2"]


class TestLintArtifactRequirementsCoherence:
    """Tests for lint_artifact_requirements_coherence function."""

    def test_missing_scored_file_warning(self):
        """Warning when scored_rows_schema has columns but no scored file in outputs."""
        contract = {
            "artifact_requirements": {
                "required_files": [{"path": "data/metrics.json"}],
                "scored_rows_schema": {
                    "required_columns": ["id", "score"]
                }
            },
            "required_outputs": ["data/metrics.json"]
        }
        issues, notes = lint_artifact_requirements_coherence(contract)
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "scored_rows file" in issues[0]["message"]

    def test_no_warning_when_scored_file_present(self):
        """No warning when scored file is present."""
        contract = {
            "artifact_requirements": {
                "required_files": [
                    {"path": "data/metrics.json"},
                    {"path": "data/scored_rows.csv"}
                ],
                "scored_rows_schema": {
                    "required_columns": ["id", "score"]
                }
            }
        }
        issues, notes = lint_artifact_requirements_coherence(contract)
        assert len(issues) == 0


class TestRunContractSchemaLinter:
    """Tests for run_contract_schema_linter function."""

    def test_full_linter_integration(self):
        """Full linter should process all checks."""
        contract = {
            "column_roles": [["X", "outcome"], ["Y", "feature"]],
            "artifact_requirements": {
                "required_files": [{"path": "data/scored.csv"}],
                "scored_rows_schema": {
                    "required_columns": ["id", "accuracy", "score"]
                }
            },
            "allowed_feature_sets": {
                "model_features": ["X", "Y", "Z"],
                "forbidden_for_modeling": [],
            }
        }
        result_contract, issues, notes, _ = run_contract_schema_linter(contract)

        # column_roles should be normalized
        assert isinstance(result_contract["column_roles"], dict)
        assert result_contract["column_roles"]["X"] == "outcome"

        # accuracy should be removed from required_columns
        scored_cols = result_contract["artifact_requirements"]["scored_rows_schema"]["required_columns"]
        assert "accuracy" not in scored_cols
        assert "id" in scored_cols
        assert "score" in scored_cols

        # X (outcome) should be removed from model_features
        model_features = result_contract["allowed_feature_sets"]["model_features"]
        assert "X" not in model_features
        assert "Y" in model_features


class TestValidateContractIntegration:
    """Integration tests for validate_contract with linter."""

    def test_linter_issues_in_validation_result(self):
        """Linter issues should appear in validate_contract result."""
        contract = {
            "column_roles": 42,  # Invalid type
            "artifact_requirements": {
                "required_files": [{"path": "data/cleaned_data.csv"}],
                "scored_rows_schema": {"required_columns": []}
            }
        }
        result = validate_contract(contract)
        assert result["status"] == "error"
        # Should have issue about invalid column_roles type
        lint_issues = [i for i in result["issues"] if "contract_schema_lint" in i.get("rule", "")]
        assert len(lint_issues) > 0
        assert any("invalid type" in i["message"] for i in lint_issues)

    def test_traceability_unknowns_contains_repair_notes(self):
        """
        Case 8: traceability: unknowns contiene nota cuando hay repair.
        """
        contract = {
            "column_roles": [["X", "outcome"], ["Y", "feature"]],
            "artifact_requirements": {
                "required_files": [{"path": "data/scored.csv"}],
                "scored_rows_schema": {
                    "required_columns": ["id", "accuracy"]
                }
            },
            "allowed_feature_sets": {
                "model_features": ["X", "Y"],
                "forbidden_for_modeling": [],
            }
        }
        result = validate_contract(contract)

        # unknowns should contain repair notes
        unknowns = contract.get("unknowns", [])
        assert isinstance(unknowns, list)

        # Should have notes about:
        # - column_roles normalization from list
        # - accuracy removal from required_columns
        # - X removal from model_features (leakage)
        assert any("column_roles" in u for u in unknowns)

    def test_no_dataset_specific_hardcodes(self):
        """Validation should not use dataset-specific hardcodes."""
        # Contract with generic column names - no special treatment expected
        contract = {
            "column_roles": {"col_a": "feature", "col_b": "outcome"},
            "canonical_columns": ["col_a", "col_b", "col_c"],
            "artifact_requirements": {
                "required_files": [{"path": "data/cleaned_data.csv"}],
                "scored_rows_schema": {"required_columns": ["col_a"]}
            },
            "allowed_feature_sets": {
                "model_features": ["col_a", "col_c"],
                "forbidden_for_modeling": [],
            }
        }
        result = validate_contract(contract)
        # Should work without any special casing for column names
        assert result["status"] in ("ok", "warning")

    def test_backward_compatibility_preserved(self):
        """Existing fields and structure should be preserved."""
        contract = {
            "business_objective": "Test objective",
            "canonical_columns": ["col1", "col2"],
            "artifact_requirements": {
                "required_files": [{"path": "data/cleaned_data.csv"}],
                "scored_rows_schema": {"required_columns": ["col1"]}
            },
            "validation_requirements": {"primary_metric": "roc_auc"},
            "custom_field": "should be preserved",
        }
        result = validate_contract(contract)

        # Original fields should still be present
        assert contract["business_objective"] == "Test objective"
        assert contract["custom_field"] == "should be preserved"
        assert "normalized_artifact_requirements" in result

    def test_warning_status_for_non_critical_issues(self):
        """Non-critical linter issues should result in warning, not error."""
        contract = {
            "column_roles": {"col": "completely_unknown_role"},
            "artifact_requirements": {
                "required_files": [{"path": "data/cleaned_data.csv"}],
                "scored_rows_schema": {"required_columns": []}
            }
        }
        result = validate_contract(contract)
        # Unknown role is warning, not error
        assert result["status"] in ("ok", "warning")
        assert result["status"] != "error"
