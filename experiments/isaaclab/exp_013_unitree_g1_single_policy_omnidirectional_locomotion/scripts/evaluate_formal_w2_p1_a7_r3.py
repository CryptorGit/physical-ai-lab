"""Frozen held-out A7-R3 teacher authorization without fallback."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = BASE / "phase_w2_p1_a7_r3_start_retention_recovery"; RAW = OUT / "raw/formal_heldout"
EVALUATOR = HERE.parent / "evaluate_w2_p1_a7_r3.py"; GUARD = HERE.parent / "evaluate_w2_p1_a7_r2.py"
ISAAC = Path.home() / "workspace/IsaacLab/isaaclab.bat"


def evaluate(policy: Path, mode: str, conditions: list[tuple[float, float, float]], episodes: int, destination: str) -> list[dict]:
    rows = []
    for index, (direction, speed, yaw) in enumerate(conditions):
        output = RAW / f"{mode}_{index:02d}.csv"
        if not output.with_suffix(".json").exists():
            command = [str(ISAAC), "-p", str(EVALUATOR), "--policy", str(policy), "--batch", "5", "--split", "heldout", "--direction", str(direction), "--speed", str(speed), "--yaw", str(yaw), "--episodes", str(episodes), "--group", mode, "--output", str(output), "--headless", "--device", "cuda:0"]
            with output.with_suffix(".log").open("w", encoding="utf-8") as log: subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows.append(json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["row"])
        print(json.dumps({"mode": mode, "condition": index, "status": "COMPLETE"}), flush=True)
    with (OUT / destination).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / destination).with_suffix(".json").write_text(json.dumps({"split": "heldout", "fallback": False, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    selected = json.loads((OUT / "selected_checkpoint.json").read_text(encoding="utf-8")); policy = OUT / selected["path"]
    matrix_conditions = [(direction, 0.3, yaw) for direction in range(0, 360, 45) for yaw in (-0.3, 0.0, 0.3)]
    pure_conditions = [(0.0, 0.0, yaw) for yaw in (-0.3, 0.3)]
    boundary_conditions = [(180.0, speed, yaw) for speed in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35) for yaw in (-0.3, 0.3)]
    evaluate(policy, "start_matrix", matrix_conditions, 300, "formal_start_matrix.csv")
    evaluate(policy, "pure_yaw", pure_conditions, 300, "formal_pure_yaw_start.csv")
    evaluate(policy, "rear_boundary", boundary_conditions, 200, "formal_rear_speed_boundary.csv")
    static_output = RAW / "static_retention.csv"
    if not static_output.with_suffix(".json").exists():
        command = [str(ISAAC), "-p", str(GUARD), "--policy", str(policy), "--batch", "5", "--split", "heldout", "--mode", "guard", "--output", str(static_output), "--headless", "--device", "cuda:0"]
        with static_output.with_suffix(".log").open("w", encoding="utf-8") as log: subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__": main()
