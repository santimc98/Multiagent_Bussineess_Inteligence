from src.utils.sandbox_deps import check_dependency_precheck


def test_dependency_precheck_blocks_banned_import():
    code = "import fuzzywuzzy\n"
    result = check_dependency_precheck(code, required_dependencies=[])
    assert "fuzzywuzzy" in result["banned"]


def test_dependency_precheck_allows_base_imports():
    code = "import pandas\nfrom sklearn.linear_model import LogisticRegression\nimport json\n"
    result = check_dependency_precheck(code, required_dependencies=[])
    assert result["blocked"] == []
    assert result["banned"] == []


def test_dependency_precheck_allows_extended_when_contract_requests():
    code = "import xgboost\n"
    result = check_dependency_precheck(code, required_dependencies=["xgboost"])
    assert result["blocked"] == []
    assert result["banned"] == []


def test_dependency_precheck_blocks_extended_when_not_requested():
    code = "import xgboost\n"
    result = check_dependency_precheck(code, required_dependencies=[])
    assert "xgboost" in result["blocked"]
