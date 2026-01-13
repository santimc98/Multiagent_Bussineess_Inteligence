from src.utils.ml_validation import validate_metrics_ci_consistency


def test_metrics_ci_consistency_detects_out_of_range_mean():
    metrics = {
        "model_performance": {
            "accuracy": {"mean": 0.9, "ci_lower": 0.95, "ci_upper": 0.98}
        }
    }
    issues = validate_metrics_ci_consistency(metrics)
    assert "metrics_schema_inconsistent:model_performance.accuracy" in issues
