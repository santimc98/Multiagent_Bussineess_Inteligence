import argparse
import json
import os
import sys
from typing import Any, Dict, List

from src.utils.contract_views import (
    build_de_view,
    build_ml_view,
    build_cleaning_view,
    build_qa_view,
    build_reviewer_view,
    build_translator_view,
    build_results_advisor_view,
    persist_views,
)


def _load_json(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _artifact_index_from_required_outputs(required_outputs: List[str]) -> List[Dict[str, Any]]:
    entries = []
    for path in required_outputs or []:
        if not path:
            continue
        entries.append({"path": path, "artifact_type": "artifact"})
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph utilities")
    parser.add_argument("--dry_views", action="store_true", help="Generate contract views without invoking LLMs.")
    parser.add_argument("--contract_full", type=str, default="", help="Path to execution contract JSON.")
    parser.add_argument("--contract_min", type=str, default="", help="Path to contract_min JSON.")
    parser.add_argument("--artifact_index", type=str, default="", help="Path to artifact index JSON.")
    parser.add_argument("--output_dir", type=str, default="data", help="Base output directory.")
    parser.add_argument("--run_bundle_dir", type=str, default="", help="Optional run bundle dir for persistence.")
    args = parser.parse_args()

    if not args.dry_views:
        parser.print_help()
        return 1

    contract_full = _load_json(args.contract_full) if args.contract_full else _load_json("data/execution_contract.json") or {}
    contract_min = _load_json(args.contract_min) if args.contract_min else _load_json("data/contract_min.json") or {}
    artifact_index = _load_json(args.artifact_index) if args.artifact_index else _load_json("data/produced_artifact_index.json")
    if not isinstance(artifact_index, list):
        required_outputs = contract_min.get("required_outputs") if isinstance(contract_min, dict) else []
        if not isinstance(required_outputs, list):
            required_outputs = []
        artifact_index = _artifact_index_from_required_outputs(required_outputs)

    if not contract_full and not contract_min:
        print("dry_views error: missing contract_full/contract_min. Provide paths or data/*.json files.")
        return 2

    de_view = build_de_view(contract_full, contract_min, artifact_index)
    ml_view = build_ml_view(contract_full, contract_min, artifact_index)
    cleaning_view = build_cleaning_view(contract_full, contract_min, artifact_index)
    qa_view = build_qa_view(contract_full, contract_min, artifact_index)
    reviewer_view = build_reviewer_view(contract_full, contract_min, artifact_index)
    translator_view = build_translator_view(contract_full, contract_min, artifact_index)
    results_advisor_view = build_results_advisor_view(contract_full, contract_min, artifact_index)
    views = {
        "de_view": de_view,
        "ml_view": ml_view,
        "cleaning_view": cleaning_view,
        "qa_view": qa_view,
        "reviewer_view": reviewer_view,
        "translator_view": translator_view,
        "results_advisor_view": results_advisor_view,
    }
    persisted = persist_views(
        views,
        base_dir=args.output_dir,
        run_bundle_dir=args.run_bundle_dir or None,
    )
    print("dry_views completed:")
    for key, path in persisted.items():
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
