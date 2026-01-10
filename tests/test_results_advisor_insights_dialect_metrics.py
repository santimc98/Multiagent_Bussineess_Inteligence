import json
import os

from src.agents.results_advisor import ResultsAdvisorAgent


def test_insights_scored_rows_dialect_and_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "cleaning_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"output_dialect": {"sep": ";", "decimal": ",", "encoding": "utf-8"}}, f)

    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_performance": {"accuracy": 0.82, "roc_auc": 0.91, "training_samples": 120},
                "segmentation_stats": {"n_segments": 2, "min_segment_size": 5, "median_segment_size": 8},
            },
            f,
        )

    scored_rows_path = os.path.join("data", "scored_rows.csv")
    with open(scored_rows_path, "w", encoding="utf-8") as f:
        f.write("client_segment;recommended_price;predicted_success_prob;expected_revenue\n")
        f.write("0;10,5;0,7;100,0\n")
        f.write("1;20,0;0,6;200,0\n")
        f.write("1;30,0;0,8;300,0\n")

    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights(
        {
            "artifact_index": [
                {"path": "data/metrics.json", "artifact_type": "metrics"},
                {"path": "data/scored_rows.csv", "artifact_type": "predictions"},
            ]
        }
    )

    predictions_summary = insights.get("predictions_summary", {})
    assert isinstance(predictions_summary.get("columns"), list)
    assert len(predictions_summary.get("columns", [])) > 1
    assert predictions_summary.get("row_count") == 3

    metrics_summary = insights.get("metrics_summary", [])
    assert metrics_summary
    assert "Metrics artifact missing or empty" not in " ".join(insights.get("risks", []))

    segment_summary = insights.get("segment_pricing_summary", [])
    assert len(segment_summary) >= 2
