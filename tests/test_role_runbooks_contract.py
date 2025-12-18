from src.utils.contract_validation import ensure_role_runbooks, DEFAULT_DATA_ENGINEER_RUNBOOK, DEFAULT_ML_ENGINEER_RUNBOOK


def test_ensure_role_runbooks_injects_defaults():
    contract = {"contract_version": 1, "data_requirements": []}
    updated = ensure_role_runbooks(contract)
    rb = updated.get("role_runbooks", {})
    assert rb
    de = rb.get("data_engineer")
    ml = rb.get("ml_engineer")
    assert de["safe_idioms"]
    assert ml["must"]
    # Defaults preserved
    assert "Avoid double sum" in " ".join(DEFAULT_DATA_ENGINEER_RUNBOOK["safe_idioms"])
    assert "variance guard" in " ".join(DEFAULT_ML_ENGINEER_RUNBOOK["must"]).lower()


def test_ensure_role_runbooks_sanitizes_shapes():
    contract = {"role_runbooks": {"data_engineer": {"must": "bad"}, "ml_engineer": {"must_not": "bad"}}}
    updated = ensure_role_runbooks(contract)
    de = updated["role_runbooks"]["data_engineer"]
    ml = updated["role_runbooks"]["ml_engineer"]
    assert isinstance(de["must"], list)
    assert isinstance(ml["must_not"], list)


def test_ensure_role_runbooks_preserves_defaults_when_empty():
    contract = {"role_runbooks": {"data_engineer": {"goals": []}, "ml_engineer": {"safe_idioms": []}}}
    updated = ensure_role_runbooks(contract)
    de = updated["role_runbooks"]["data_engineer"]
    ml = updated["role_runbooks"]["ml_engineer"]
    assert de["goals"] == DEFAULT_DATA_ENGINEER_RUNBOOK["goals"]
    assert ml["safe_idioms"] == DEFAULT_ML_ENGINEER_RUNBOOK["safe_idioms"]
