"""Read-only A8 validation coverage evaluation for saved A7-R2 checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
R2 = BASE / "phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2"
OUT = BASE / "phase_w2_p1_a8_offline_start_teacher_oracle"
RAW = OUT / "raw/validation_coverage"
EVALUATOR = HERE.parent / "evaluate_w2_p1_a7_r3.py"
ISAAC = Path.home() / "workspace/IsaacLab/isaaclab.bat"
UPDATES = (0, 1, 10, 20, 45, 75, 100, 120, 130, 140, 150)


def checkpoint(update: int) -> Path:
    return R2 / f"checkpoints/model_{update:03d}.pt"


def run(update: int, condition: int) -> dict:
    direction = (condition // 3) * 45.0
    yaw = (-0.3, 0.0, 0.3)[condition % 3]
    output = RAW / f"update_{update:03d}_condition_{condition:02d}.csv"
    result = output.with_suffix(".json")
    if result.exists():
        row = json.loads(result.read_text(encoding="utf-8"))["row"]
        if int(row["episodes"]) == 300:
            return row
    command = [
        str(ISAAC), "-p", str(EVALUATOR), "--policy", str(checkpoint(update)),
        "--batch", "4", "--split", "validation", "--direction", str(direction),
        "--speed", "0.3", "--yaw", str(yaw), "--episodes", "300",
        "--group", "start_matrix", "--output", str(output), "--headless", "--device", "cuda:0",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(result.read_text(encoding="utf-8"))["row"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", required=True)
    parser.add_argument("--condition-start", type=int, default=0)
    parser.add_argument("--condition-end", type=int, default=24)
    args = parser.parse_args()
    for update in [int(value) for value in args.updates.split(",")]:
        for condition in range(args.condition_start, args.condition_end):
            run(update, condition)
            print(json.dumps({"update": update, "condition": condition, "status": "COMPLETE"}), flush=True)
