from src.utils.feature_selectors import infer_feature_selectors, expand_feature_selectors


def test_feature_selectors_reconstruct_pixels() -> None:
    columns = [f"pixel{i}" for i in range(784)]
    selectors, remaining = infer_feature_selectors(columns, max_list_size=200, min_group_size=50)

    expanded = set(expand_feature_selectors(columns, selectors))
    reconstructed = expanded.union(set(remaining))

    assert reconstructed == set(columns)
