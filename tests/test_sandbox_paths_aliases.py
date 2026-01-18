from src.utils import sandbox_paths


def test_cleaned_aliases_include_train_test() -> None:
    aliases = sandbox_paths.COMMON_CLEANED_ALIASES
    assert "train.csv" in aliases
    assert "test.csv" in aliases
    assert "data/train.csv" in aliases
    assert "data/test.csv" in aliases
