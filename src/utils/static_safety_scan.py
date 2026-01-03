import ast
import re
from typing import Tuple, List, Set

def scan_code_safety(code: str) -> Tuple[bool, List[str]]:
    """
    Scans generated Python code for forbidden patterns using AST analysis and Regex fallbacks.
    Returns (is_safe: bool, violations: List[str]).
    
    POLICY:
    - ALLOW: os.path.*, os.makedirs (data/plots).
    - BLOCK: os.system, subprocess, network libs, internals.
    - BLOCK: eval, exec, compile.
    """
    violations = []
    
    # 1. AST Analysis
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax Error in code: {e}"]

    # Config: Blocked Modules & Calls
    BLOCKED_MODULES = {
        "subprocess", "socket", "requests", "httpx", "urllib", "ftplib",
        "paramiko", "selenium", "playwright", "openai", "google.generativeai",
        "sys", "builtins" # prevent import builtins; exec
    }
    
    # Allow 'os' but block dangerous attributes
    BLOCKED_CALLS = {
        "eval", "exec", "compile", "__import__",
        "os.system", "os.popen", "os.spawn", "os.execl", "os.execv",
        "shutil.rmtree", "os.remove", "os.unlink", "os.rmdir",
        "pathlib.Path.unlink", "glob.glob", "glob.iglob"
    }
    
    # Specific Attribute Blocks (e.g. pd.io)
    BLOCKED_ATTRS = {
        "pandas.io": "Private Pandas API",
        "pd.io": "Private Pandas API",
        "subprocess.run": "Subprocess",
        "subprocess.Popen": "Subprocess"
    }

    class SecurityVisitor(ast.NodeVisitor):
        def __init__(self):
            self.errors = []
            
        def visit_Import(self, node):
            for alias in node.names:
                base_module = alias.name.split('.')[0]
                if base_module in BLOCKED_MODULES:
                    self.errors.append(f"Importing '{alias.name}' is PROHIBITED.")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module:
                base_module = node.module.split('.')[0]
                if base_module in BLOCKED_MODULES:
                    self.errors.append(f"Importing from '{node.module}' is PROHIBITED.")
                # Check for `from pandas import io`
                if base_module == "pandas" and "io" in [n.name for n in node.names]:
                     self.errors.append("Importing 'pandas.io' is PROHIBITED.")
            self.generic_visit(node)
            
        def visit_Call(self, node):
            func_name = self._get_func_name(node.func)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "sum":
                base_call = node.func.value
                if isinstance(base_call, ast.Call) and isinstance(base_call.func, ast.Name):
                    if base_call.func.id == "int":
                        self.errors.append(
                            "Likely bug: int(...) returns scalar; .sum() here indicates you meant int((...).sum())."
                        )
            if func_name in BLOCKED_CALLS:
                self.errors.append(f"Calling '{func_name}' is PROHIBITED.")
            
            # Check for os.listdir/walk explicitly (filesystem recon)
            if func_name in ["os.listdir", "os.walk", "os.scandir"]:
                self.errors.append(f"Filesystem exploration ('{func_name}') is PROHIBITED.")
                
            self.generic_visit(node)

        def visit_Attribute(self, node):
            # Catch direct attribute access without call (e.g. passing os.system around)
            attr_name = self._get_func_name(node)
            if attr_name in BLOCKED_CALLS or attr_name in BLOCKED_ATTRS:
                 # We allow if it's just checking existence, but blocking execution is done in visit_Call.
                 # However, `pd.io` is an attribute usage we want to block entirely.
                 if attr_name in BLOCKED_ATTRS:
                     self.errors.append(f"Usage of '{attr_name}' is PROHIBITED ({BLOCKED_ATTRS[attr_name]}).")
            self.generic_visit(node)

        def _get_func_name(self, node):
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._get_func_name(node.value)}.{node.attr}"
            return ""

    visitor = SecurityVisitor()
    visitor.visit(tree)
    violations.extend(visitor.errors)
    
    # 2. Regex Fallbacks (for dynamic things AST might miss or string injection)
    # ParserBase
    if re.search(r'\bParserBase\b', code):
        violations.append("Usage of 'ParserBase' is PROHIBITED.")

    if re.search(r"\bnp\.bool(?!_)\b", code) or re.search(r"\bnumpy\.bool(?!_)\b", code):
        violations.append("Usage of 'np.bool' is PROHIBITED. Use 'np.bool_' instead.")
        
    return (len(violations) == 0, violations)
