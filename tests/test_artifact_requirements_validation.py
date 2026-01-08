"""
Test artifact_requirements validation to ensure required_columns ⊆ canonical_columns.

This test validates the fix for the bug where artifact_requirements.required_columns
included columns not in canonical_columns (e.g., 'Account', 'FiscalId' which were
marked as role="unknown" metadata columns).
"""

import pytest
from src.agents.execution_planner import validate_artifact_requirements


def test_artifact_requirements_moves_non_canonical_to_optional():
    """Test that non-canonical columns in required_columns are moved to optional_passthrough_columns."""
    contract = {
        'canonical_columns': ['CurrentPhase', '1stYearAmount', 'Size', 'Debtors', 'Sector', 'Probability'],
        'derived_columns': ['is_success', 'client_segment'],
        'available_columns': [
            'CurrentPhase', '1stYearAmount', 'Size', 'Debtors', 'Sector', 'Probability',
            'Account', 'FiscalId'  # Metadata columns not in canonical
        ],
        'artifact_requirements': {
            'schema_binding': {
                'required_columns': ['Account', 'FiscalId', 'CurrentPhase', '1stYearAmount']
            }
        },
        'unknowns': []
    }

    result = validate_artifact_requirements(contract)

    # Verify required_columns only contains canonical columns
    required_columns = result['artifact_requirements']['schema_binding']['required_columns']
    assert set(required_columns) == {'CurrentPhase', '1stYearAmount'}

    # Verify non-canonical columns moved to optional
    optional = result['artifact_requirements']['schema_binding']['optional_passthrough_columns']
    assert set(optional) == {'Account', 'FiscalId'}

    # Verify unknowns documents the change
    assert len(result['unknowns']) == 1
    assert 'Account' in result['unknowns'][0]['item']
    assert 'FiscalId' in result['unknowns'][0]['item']
    assert result['unknowns'][0]['auto_corrected'] is True


def test_artifact_requirements_preserves_valid_required_columns():
    """Test that valid canonical columns in required_columns are preserved."""
    contract = {
        'canonical_columns': ['col1', 'col2', 'col3'],
        'derived_columns': ['derived1'],
        'available_columns': ['col1', 'col2', 'col3', 'metadata'],
        'artifact_requirements': {
            'schema_binding': {
                'required_columns': ['col1', 'col2']  # All valid
            }
        },
        'unknowns': []
    }

    result = validate_artifact_requirements(contract)

    # Verify all required_columns are preserved
    required_columns = result['artifact_requirements']['schema_binding']['required_columns']
    assert set(required_columns) == {'col1', 'col2'}

    # Verify no optional_passthrough_columns added
    optional = result['artifact_requirements']['schema_binding'].get('optional_passthrough_columns', [])
    assert optional == []

    # Verify no unknowns added
    assert len(result['unknowns']) == 0


def test_artifact_requirements_allows_derived_columns():
    """Test that derived columns can be in required_columns."""
    contract = {
        'canonical_columns': ['col1', 'col2'],
        'derived_columns': ['is_success', 'segment'],
        'available_columns': ['col1', 'col2'],
        'artifact_requirements': {
            'schema_binding': {
                'required_columns': ['col1', 'is_success']  # Mix of canonical and derived
            }
        },
        'unknowns': []
    }

    result = validate_artifact_requirements(contract)

    # Verify both canonical and derived columns are allowed
    required_columns = result['artifact_requirements']['schema_binding']['required_columns']
    assert set(required_columns) == {'col1', 'is_success'}

    # Verify no changes
    assert len(result['unknowns']) == 0


def test_artifact_requirements_case_insensitive_matching():
    """Test that column matching is case-insensitive."""
    contract = {
        'canonical_columns': ['CurrentPhase', '1stYearAmount'],
        'derived_columns': [],
        'available_columns': ['CurrentPhase', '1stYearAmount', 'Account'],
        'artifact_requirements': {
            'schema_binding': {
                'required_columns': ['currentphase', '1styearamount']  # Lowercase
            }
        },
        'unknowns': []
    }

    result = validate_artifact_requirements(contract)

    # Verify case-insensitive matching works
    required_columns = result['artifact_requirements']['schema_binding']['required_columns']
    assert len(required_columns) == 2
    assert 'currentphase' in required_columns
    assert '1styearamount' in required_columns


def test_artifact_requirements_handles_missing_schema_binding():
    """Test that validation handles missing schema_binding gracefully."""
    contract = {
        'canonical_columns': ['col1'],
        'derived_columns': [],
        'available_columns': ['col1'],
        'artifact_requirements': {},  # No schema_binding
        'unknowns': []
    }

    result = validate_artifact_requirements(contract)

    # Verify contract is returned unchanged
    assert result == contract


def test_artifact_requirements_handles_invalid_types():
    """Test that validation handles invalid types gracefully."""
    contract = {
        'canonical_columns': ['col1'],
        'derived_columns': [],
        'available_columns': ['col1'],
        'artifact_requirements': {
            'schema_binding': {
                'required_columns': 'not_a_list'  # Invalid type
            }
        },
        'unknowns': []
    }

    result = validate_artifact_requirements(contract)

    # Verify contract is returned unchanged
    assert result['artifact_requirements']['schema_binding']['required_columns'] == 'not_a_list'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
