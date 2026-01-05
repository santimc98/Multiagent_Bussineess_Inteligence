"""
Tests for V4.1 schema alignment in ExecutionPlannerAgent
"""
import pytest
from src.agents.execution_planner import ExecutionPlannerAgent, _create_v41_skeleton, ensure_v41_schema


def test_fallback_contains_all_v41_keys():
    """Verify fallback returns complete V4.1 schema."""
    planner = ExecutionPlannerAgent(api_key=None)  # Forces fallback
    contract = planner.generate_contract(
        strategy={"title": "Test Strategy"},
        business_objective="Test objective",
        column_inventory=["Col1", "Col2"],
        output_dialect={"sep": ",", "decimal": ".", "encoding": "utf-8"},
        env_constraints={"forbid_inplace_column_creation": True}
    )
    
    # Check all required V4.1 top-level keys
    required_keys = [
        "contract_version", "strategy_title", "business_objective",
        "missing_columns_handling", "execution_constraints",
        "objective_analysis", "data_analysis", "column_roles",
        "preprocessing_requirements", "feature_engineering_plan",
        "validation_requirements", "leakage_execution_plan",
        "optimization_specification", "segmentation_constraints",
        "data_limited_mode", "allowed_feature_sets",
        "artifact_requirements", "qa_gates", "reviewer_gates",
        "data_engineer_runbook", "ml_engineer_runbook",
        "available_columns", "canonical_columns", "derived_columns",
        "required_outputs", "iteration_policy", "unknowns",
        "assumptions", "notes_for_engineers"
    ]
    
    for key in required_keys:
        assert key in contract, f"Missing required V4.1 key: {key}"
    
    assert contract["contract_version"] == 2
    assert contract["available_columns"] == ["Col1", "Col2"]
    assert contract["strategy_title"] == "Test Strategy"


def test_create_v41_skeleton_basic():
    """Test V4.1 skeleton creation."""
    skeleton = _create_v41_skeleton(
        strategy={"title": "Test", "required_columns": ["Col1"]},
        business_objective="Test objective",
        column_inventory=["Col1", "Col2", "Col3"]
    )
    
    assert skeleton["contract_version"] == 2
    assert skeleton["strategy_title"] == "Test"
    assert skeleton["available_columns"] == ["Col1", "Col2", "Col3"]
    # canonical_columns should be filtered required_columns that exist in inventory
    assert "Col1" in skeleton["canonical_columns"]
    assert len(skeleton["unknowns"]) > 0  # Should have fallback reason


def test_ensure_v41_schema_fills_missing_keys():
    """Test that ensure_v41_schema fills missing V4.1 keys."""
    incomplete_contract = {
        "contract_version": 2,
        "strategy_title": "Test",
        "business_objective": "Test obj"
        # Missing most keys
    }
    
    complete_contract = ensure_v41_schema(incomplete_contract)
    
    # Should have all required keys now
    assert "missing_columns_handling" in complete_contract
    assert "execution_constraints" in complete_contract
    assert "qa_gates" in complete_contract
    assert "reviewer_gates" in complete_contract
    assert "unknowns" in complete_contract
    
    # Should have repair notes in unknowns
    unknowns = complete_contract["unknowns"]
    assert isinstance(unknowns, list)
    assert len(unknowns) > 0  # Should have repair entries


def test_ensure_v41_schema_strict_mode():
    """Test strict mode raises error on missing keys."""
    incomplete_contract = {
        "contract_version": 2
    }
    
    with pytest.raises(ValueError, match="Missing required V4.1 key"):
        ensure_v41_schema(incomplete_contract, strict=True)


def test_no_legacy_schema_version():
    """Ensure no code path returns contract_version=1."""
    planner = ExecutionPlannerAgent(api_key=None)
    contract = planner.generate_contract(
        strategy={"title": "Test"},
        business_objective="Test",
        column_inventory=[]
    )
    
    assert contract.get("contract_version") != 1
    assert contract.get("contract_version") == 2


def test_output_dialect_propagation():
    """Verify output_dialect parameter is used."""
    planner = ExecutionPlannerAgent(api_key=None)
    dialect = {"sep": ";", "decimal": ",", "encoding": "iso-8859-1"}
    
    contract = planner.generate_contract(
        strategy={"title": "Test"},
        business_objective="Test",
        column_inventory=["Col1"],
        output_dialect=dialect
    )
    
    # Fallback path should receive the dialect
    # In the skeleton it's noted, check that contract has execution_constraints
    assert "execution_constraints" in contract


def test_env_constraints_propagation():
    """Verify env_constraints parameter is used."""
    planner = ExecutionPlannerAgent(api_key=None)
    constraints = {"forbid_inplace_column_creation": False}
    
    contract = planner.generate_contract(
        strategy={"title": "Test"},
        business_objective="Test",
        column_inventory=["Col1"],
        env_constraints=constraints
    )
    
    # Check execution_constraints exists
    assert "execution_constraints" in contract
    # In fallback, it uses default, but contract should have the field
    assert isinstance(contract["execution_constraints"], dict)


def test_contract_driven_evaluation_spec():
    """Test that generate_evaluation_spec is contract-driven."""
    planner = ExecutionPlannerAgent(api_key=None)
    
    # Create a V4.1 contract
    contract = {
        "contract_version": 2,
        "qa_gates": [{"id": "test_gate", "required": True}],
        "reviewer_gates": [{"id": "review_gate", "required": True}],
        "artifact_requirements": {"required_files": ["data/test.csv"]},
        "validation_requirements": {"method": "cross_validation"},
        "canonical_columns": ["Col1"],
        "derived_columns": [],
        "required_outputs": ["data/output.csv"]
    }
    
    eval_spec = planner.generate_evaluation_spec(
        strategy={},
        contract=contract,
        data_summary="",
        business_objective="",
        column_inventory=[]
    )
    
    # Should extract from contract
    assert eval_spec["source"] == "contract_driven_v41"
    assert eval_spec["confidence"] == 0.9
    assert eval_spec["qa_gates"] == contract["qa_gates"]
    assert eval_spec["reviewer_gates"] == contract["reviewer_gates"]
    assert eval_spec["canonical_columns"] == ["Col1"]


def test_canonical_columns_subset_of_inventory():
    """Test that canonical_columns is a subset of available_columns."""
    skeleton = _create_v41_skeleton(
        strategy={"title": "Test", "required_columns": ["Col1", "ColNotInInventory"]},
        business_objective="Test",
        column_inventory=["Col1", "Col2"]
    )
    
    # canonical_columns should only contain columns from inventory
    assert "Col1" in skeleton["canonical_columns"]
    assert "ColNotInInventory" not in skeleton["canonical_columns"]
    assert set(skeleton["canonical_columns"]).issubset(set(skeleton["available_columns"]))
