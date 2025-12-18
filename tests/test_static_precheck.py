from src.graph.graph import detect_undefined_names


def test_precheck_allows_dunder_main_block():
    code = """
if __name__ == "__main__":
    print("ok")
"""
    assert detect_undefined_names(code) == []


def test_precheck_flags_missing_function():
    code = """
def main():
    df = load_data_with_dialect()
    return df
"""
    undefined = detect_undefined_names(code)
    assert "load_data_with_dialect" in undefined


def test_precheck_allows_lambda_args():
    code = """
import pandas as pd
f = lambda x: x.quantile(0.25)
df = pd.DataFrame({"a":[1,2,3]})
q = df["a"].agg(lambda x: x.quantile(0.25))
"""
    undefined = detect_undefined_names(code)
    assert undefined == []
