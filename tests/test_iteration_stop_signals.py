from src.graph import graph as graph_mod


def test_check_evaluation_stops_on_data_limited():
    state = {
        "last_iteration_type": "metric",
        "data_adequacy_report": {
            "threshold_reached": True,
            "reasons": ["signal_ceiling_reached"],
        },
        "feedback_history": [],
        "iteration_count": 0,
    }

    decision = graph_mod.check_evaluation(state)

    assert decision == "approved"
    assert any("DATA_LIMITED_STOP" in msg for msg in state.get("feedback_history", []))


def test_check_evaluation_stops_on_plateau():
    signature = "objective_type=classification;method=cross_validation"
    state = {
        "last_iteration_type": "metric",
        "metric_history": [
            {"primary_metric_name": "roc_auc", "lift": 0.005, "eval_signature": signature},
            {"primary_metric_name": "roc_auc", "lift": 0.004, "eval_signature": signature},
        ],
        "execution_contract": {"iteration_policy": {"plateau_window": 2, "plateau_epsilon": 0.01}},
        "feedback_history": [],
        "iteration_count": 0,
    }

    decision = graph_mod.check_evaluation(state)

    assert decision == "approved"
    assert any("PLATEAU_STOP" in msg for msg in state.get("feedback_history", []))
