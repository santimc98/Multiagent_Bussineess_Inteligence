import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.case_alignment import build_case_alignment_report


def test_case_alignment_gate_detects_violations(tmp_path: Path):
    case_summary = tmp_path / "case_summary.csv"
    df = pd.DataFrame(
        {
            "refscore": [0.1, 0.2, 0.3, 0.4],
            "score_nuevo_mean": [0.4, 0.3, 0.2, 0.1],
        }
    )
    df.to_csv(case_summary, index=False)
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"weights": {"a": 0.9, "b": 0.1}}))

    contract = {
        "quality_gates": {"spearman_min": 0.85, "violations_max": 0},
        "data_requirements": [{"name": "refscore", "role": "target"}],
    }
    report = build_case_alignment_report(
        contract=contract,
        case_summary_path=str(case_summary),
        weights_path=str(weights),
        data_paths=[],
    )
    assert report["status"] == "FAIL"
    assert "adjacent_refscore_violations" in report["failures"]


def test_case_alignment_gate_passes_with_monotonic_scores(tmp_path: Path):
    case_summary = tmp_path / "case_summary.csv"
    df = pd.DataFrame(
        {
            "refscore": [0.1, 0.2, 0.3, 0.4],
            "score_nuevo_mean": [0.1, 0.2, 0.3, 0.4],
        }
    )
    df.to_csv(case_summary, index=False)
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"weights": {"a": 0.5, "b": 0.5}}))

    contract = {
        "quality_gates": {"spearman_min": 0.8, "violations_max": 0},
        "data_requirements": [{"name": "refscore", "role": "target"}],
    }
    report = build_case_alignment_report(
        contract=contract,
        case_summary_path=str(case_summary),
        weights_path=str(weights),
        data_paths=[],
    )
    assert report["status"] == "PASS"


def test_case_alignment_adjacent_violations_ignore_ties(tmp_path: Path):
    case_summary = tmp_path / "case_summary.csv"
    df = pd.DataFrame(
        {
            "refscore": [0.1, 0.1, 0.2, 0.3],
            "score_nuevo_mean": [0.4, 0.1, 0.2, 0.3],
        }
    )
    df.to_csv(case_summary, index=False)
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"weights": {"a": 0.5, "b": 0.5}}))

    contract = {"data_requirements": [{"name": "refscore", "role": "target"}]}
    report = build_case_alignment_report(
        contract=contract,
        case_summary_path=str(case_summary),
        weights_path=str(weights),
        data_paths=[],
    )
    assert report["metrics"]["adjacent_refscore_violations"] == 0


def test_case_alignment_inactive_share_in_top(tmp_path: Path):
    case_summary = tmp_path / "case_summary.csv"
    refscore = list(np.linspace(0.0, 1.0, 20))
    score = list(reversed(refscore))
    df = pd.DataFrame({"refscore": refscore, "score_nuevo_mean": score})
    df.to_csv(case_summary, index=False)
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"weights": {"a": 0.5, "b": 0.5}}))

    contract = {"data_requirements": [{"name": "refscore", "role": "target"}]}
    report = build_case_alignment_report(
        contract=contract,
        case_summary_path=str(case_summary),
        weights_path=str(weights),
        data_paths=[],
    )
    assert report["metrics"]["inactive_top_decile_share"] == pytest.approx(0.5)
    assert report["metrics"]["inactive_share_within_inactive"] == pytest.approx(1.0)


def test_case_alignment_row_level_weight_suffix_mapping(tmp_path: Path):
    data_path = tmp_path / "cleaned_full.csv"
    df = pd.DataFrame({"RefScore": [0.1, 0.2, 0.3], "FeatureA": [0.1, 0.2, 0.3]})
    df.to_csv(data_path, index=False)
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"weights": {"w1_featurea": 1.0}}))

    contract = {"data_requirements": [{"name": "RefScore", "role": "target"}]}
    report = build_case_alignment_report(
        contract=contract,
        case_summary_path=str(tmp_path / "missing.csv"),
        weights_path=str(weights),
        data_paths=[str(data_path)],
    )
    assert report["mode"] == "row_level"
    assert report["metrics"]["spearman_case_means"] == pytest.approx(1.0)
