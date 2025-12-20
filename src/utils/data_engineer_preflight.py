import ast
from typing import List


def data_engineer_preflight(code: str) -> List[str]:
    """
    Deterministic static checks for common Data Engineer pitfalls.
    Returns a list of issues; empty list means allow.
    """
    issues: List[str] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return issues

    # Import guard
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root = (alias.name or "").split(".")[0]
                if root == "sys":
                    issues.append("Do not import sys; remove sys import.")
                    break

    # sum(x.sum()) pattern guard
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum":
            if node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Call):
                    func = arg0.func
                    if (isinstance(func, ast.Attribute) and func.attr == "sum") or (
                        isinstance(func, ast.Name) and func.id == "sum"
                    ):
                        issues.append("Avoid sum(x.sum()); use mask.mean() or mask.sum()/len(mask) for ratios.")
                        break
    # Guard against any()/all() applied to a single membership test
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"any", "all"}:
            if len(node.args) != 1:
                continue
            arg0 = node.args[0]
            if not isinstance(arg0, ast.Compare):
                continue
            if not any(isinstance(op, (ast.In, ast.NotIn)) for op in arg0.ops):
                continue
            if not arg0.comparators:
                continue
            comp = arg0.comparators[0]
            if isinstance(comp, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                issues.append(
                    "any()/all() expects an iterable; avoid passing a single membership test (it returns a bool)."
                )
                break
    # Guard against slicing None for actual_column in validation summaries
    if "actual_column" in code:
        for line in code.splitlines():
            if "actual_column" in line and "[:" in line:
                issues.append(
                    "Guard actual_column when printing: use actual = str(res.get('actual_column') or 'MISSING') before slicing."
                )
                break
    if "df[actual_col].dtype" in code or "df[actual_col].dtypes" in code:
        issues.append(
            "Guard duplicate column labels: assign series = df[actual_col]; if isinstance(series, pd.DataFrame) use series = series.iloc[:, 0] before accessing dtype."
        )
    return issues
