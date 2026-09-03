"""GPU-MJX v60 diagnostic launcher using the parity-qualified evaluator.

The underlying v59 entrypoint is checkpoint-generic; this wrapper freezes the
v60 command contract and keeps every run in a separate artifact directory.
It does not modify or overwrite the legacy evaluator or v59 results.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
V59_EVALUATOR = (
    EXP_ROOT / "scripts" / "evaluate_v59_corrected_15s_diagnostic.py"
)
V59_COMMANDS = (
    EXP_ROOT
    / "artifacts"
    / "v59_corrected_15s_diagnostic"
    / "command_manifest.json"
)
OUTPUT_ROOT = (
    EXP_ROOT / "artifacts" / "v60_bounded_yaw_pilot" / "evaluations"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--label",
        required=True,
        choices=("parent_v52", "control_1m", "treatment_1m", "treatment_5m"),
    )
    parser.add_argument("--condition", required=True, choices=("D", "S"))
    parser.add_argument("--master-seed", type=int, default=20260730)
    args = parser.parse_args()

    output = OUTPUT_ROOT / args.label / f"condition_{args.condition.lower()}"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(V59_COMMANDS, output / "command_manifest.json")
    invocation = {
        "diagnostic_only": True,
        "formal_acceptance_eligible": False,
        "enough_episodes": False,
        "gpu_mjx_required": True,
        "checkpoint": args.checkpoint,
        "label": args.label,
        "condition": args.condition,
        "master_seed": args.master_seed,
        "seconds": 15,
        "commands": 19,
        "seeds": 5,
        "underlying_evaluator": str(V59_EVALUATOR),
    }
    (output / "v60_evaluation_invocation.json").write_text(
        json.dumps(invocation, indent=2) + "\n", encoding="utf-8"
    )
    command = [
        sys.executable,
        str(V59_EVALUATOR),
        "--condition",
        args.condition,
        "--checkpoint",
        args.checkpoint,
        "--output",
        str(output),
        "--seconds",
        "15",
        "--seeds",
        "5",
        "--master-seed",
        str(args.master_seed),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
