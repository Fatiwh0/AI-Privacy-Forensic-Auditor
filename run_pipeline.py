from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, List

from src.utils import ensure_output_dirs, load_yaml, save_json

PIPELINE_STEPS: List[str] = [
    "ingestion",
    "preprocessing",
    "eda",
    "reliability",
    "association_rules",
    "clustering",
    "classification",
    "mia_shadow",
    "defense_dp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Privacy Forensic Auditor pipeline")
    parser.add_argument(
        "--config",
        default="config/dataset_adult.yaml",
        help="Path to dataset config YAML.",
    )
    parser.add_argument(
        "--experiments",
        default="config/experiments.yaml",
        help="Path to experiments config YAML.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=PIPELINE_STEPS,
        help="Subset of steps to run in order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned steps and exit.",
    )
    return parser.parse_args()


def run_step(step_name: str, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    module = importlib.import_module(f"src.{step_name}")
    if not hasattr(module, "run"):
        return {"step": step_name, "status": "error", "message": "Missing run() in module."}
    return module.run(config=config, context=context)


def main() -> None:
    args = parse_args()
    dataset_cfg = load_yaml(args.config)
    experiments_cfg = load_yaml(args.experiments)
    ensure_output_dirs(experiments_cfg)

    if args.dry_run:
        print("Planned steps:")
        for step in args.steps:
            print(f"- {step}")
        return

    context: Dict[str, Any] = {"experiments_cfg": experiments_cfg}
    step_results: List[Dict[str, Any]] = []

    for step in args.steps:
        result = run_step(step, dataset_cfg, context)
        step_results.append(result)
        print(f"[{result.get('status', 'unknown')}] {step}: {result.get('message', '')}")

    summary = {
        "config": str(Path(args.config)),
        "experiments": str(Path(args.experiments)),
        "steps": step_results,
    }
    save_json(summary, "outputs/reports/pipeline_summary.json")
    print("Saved summary: outputs/reports/pipeline_summary.json")


if __name__ == "__main__":
    main()

