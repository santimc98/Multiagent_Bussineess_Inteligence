import ast
import sys
from typing import Iterable, Dict, List, Set

BASE_ALLOWLIST = ["pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn"]
EXTENDED_ALLOWLIST = ["statsmodels", "xgboost", "pyarrow", "openpyxl"]
BANNED_ALLOWLIST = ["fuzzywuzzy", "rapidfuzz", "shap", "torch", "tensorflow", "spacy", "prophet"]

PIP_BASE = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn"]
PIP_EXTENDED = {
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
    "pyarrow": "pyarrow",
    "openpyxl": "openpyxl",
}


def _stdlib_modules() -> Set[str]:
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)
    return {
        "abc", "argparse", "asyncio", "base64", "collections", "contextlib", "csv",
        "dataclasses", "datetime", "enum", "functools", "glob", "hashlib", "itertools",
        "json", "logging", "math", "os", "pathlib", "random", "re", "statistics",
        "string", "sys", "time", "typing", "uuid", "warnings",
    }


def extract_import_roots(code: str) -> Set[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return set()
    roots: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def check_dependency_precheck(code: str, required_dependencies: Iterable[str] | None = None) -> Dict[str, List[str]]:
    imports = extract_import_roots(code)
    required = {str(dep).split(".")[0] for dep in (required_dependencies or []) if dep}
    stdlib = _stdlib_modules()

    banned = sorted(imports & set(BANNED_ALLOWLIST))
    allowed = set(BASE_ALLOWLIST) | (set(EXTENDED_ALLOWLIST) & required) | stdlib
    blocked = sorted([imp for imp in imports if imp not in allowed])
    return {
        "imports": sorted(imports),
        "blocked": blocked,
        "banned": banned,
    }


def get_sandbox_install_packages(required_dependencies: Iterable[str] | None = None) -> Dict[str, List[str]]:
    required = {str(dep).split(".")[0] for dep in (required_dependencies or []) if dep}
    extra = [PIP_EXTENDED[d] for d in EXTENDED_ALLOWLIST if d in required]
    return {"base": list(PIP_BASE), "extra": extra}
