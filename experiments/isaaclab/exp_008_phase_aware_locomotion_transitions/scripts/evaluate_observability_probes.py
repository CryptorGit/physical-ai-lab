"""Validate and summarize the Stage 0 diagnostic-probe outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/exp_008_phase_aware_locomotion_transitions/stage0_observability_and_controllability"),
    )
    args = parser.parse_args()
    required = [
        "probe_results.json",
        "per_feature_condition_results.json",
        "age_matched_results.json",
        "timing_leakage_audit.json",
        "observability_classification.json",
    ]
    missing = [name for name in required if not (args.results / name).is_file()]
    if missing:
        raise SystemExit(f"missing probe outputs: {missing}")
    result = json.loads((args.results / "observability_classification.json").read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
