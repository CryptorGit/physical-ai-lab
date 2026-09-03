"""Evaluate every saved A7-R2 checkpoint for R3-A rescue eligibility."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
R2 = BASE / "phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2"
OUT = BASE / "phase_w2_p1_a7_r3_start_retention_recovery"
RAW = OUT / "raw/existing_checkpoint_rescue"
EVALUATOR = HERE.parent / "evaluate_w2_p1_a7_r3.py"
ISAAC = Path.home() / "workspace/IsaacLab/isaaclab.bat"
UPDATES = (0, 1, 10, 20, 45, 75, 100, 120, 130, 140, 150)


def run(update: int, condition: int) -> dict:
    direction = (condition // 3) * 45.0
    yaw = (-0.3, 0.0, 0.3)[condition % 3]
    output = RAW / f"update_{update:03d}_condition_{condition:02d}.csv"
    if output.with_suffix(".json").exists():
        return json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["row"]
    command = [
        str(ISAAC), "-p", str(EVALUATOR), "--policy", str(R2 / f"checkpoints/model_{update:03d}.pt"),
        "--batch", "4", "--split", "validation", "--direction", str(direction), "--speed", "0.3",
        "--yaw", str(yaw), "--episodes", "200", "--group", "start_matrix", "--output", str(output),
        "--headless", "--device", "cuda:0",
    ]
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["row"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    timeline = json.loads((R2 / "a7_capability_timeline.json").read_text(encoding="utf-8"))["rows"]
    all_rows = []
    eligibility = []
    for update in UPDATES:
        matrix = []
        for condition in range(24):
            row = run(update, condition)
            row["update"] = update
            matrix.append(row)
            all_rows.append(row)
            print(json.dumps({"update": update, "condition": condition, "status": "COMPLETE"}), flush=True)
        quick = [row for row in timeline if int(row["update"]) == update]
        groups = {name: [row for row in quick if row["group"] == name] for name in {row["group"] for row in quick}}
        rear_negative = next(row for row in matrix if row["direction"] == 180.0 and row["yaw"] < 0)
        rear_positive = next(row for row in matrix if row["direction"] == 180.0 and row["yaw"] > 0)
        target = next(row for row in matrix if row["direction"] == 315.0 and row["yaw"] > 0)
        matrix_pass = all(row["endpoint_success"] >= 0.90 and row["acquisition_0p20"] >= 0.85 and row["fall_rate"] <= 0.05 for row in matrix)
        rear_pass = all(row["endpoint_success"] >= 0.95 and row["acquisition_0p20"] >= 0.90 and row["fall_rate"] <= 0.02 for row in (rear_negative, rear_positive))
        static_pass = (
            sum(row["endpoint_success"] >= 0.90 for row in groups["zero_yaw"]) == 16
            and min(row["endpoint_success"] for row in groups["forward_anchor"]) >= 0.95
            and min(row["endpoint_success"] for row in groups["pure_yaw"]) >= 0.90
            and sum(row["endpoint_success"] >= 0.90 for row in groups["moving_turn"]) == 24
        )
        safety_pass = (
            max(row["fall_rate"] for row in matrix) <= 0.02
            and max(row["dangerous_slip_rate"] for row in matrix) <= 0.10
            and max(row["impact_rate"] for row in matrix) <= 0.05
            and max(row["saturation_rate"] for row in matrix) <= 0.05
        )
        eligibility.append({
            "update": update,
            "minimum_acquisition": min(row["acquisition_0p20"] for row in matrix),
            "aggregate_acquisition": sum(row["acquisition_0p20"] for row in matrix) / 24,
            "minimum_endpoint": min(row["endpoint_success"] for row in matrix),
            "rear_negative_acquisition": rear_negative["acquisition_0p20"],
            "rear_positive_acquisition": rear_positive["acquisition_0p20"],
            "target_acquisition": target["acquisition_0p20"],
            "matrix_pass": matrix_pass, "rear_pass": rear_pass, "static_pass": static_pass,
            "safety_pass": safety_pass, "eligible": matrix_pass and rear_pass and static_pass and safety_pass,
        })
        (OUT / "existing_checkpoint_eligibility.json").write_text(json.dumps({"status": "IN_PROGRESS", "rows": eligibility}, indent=2) + "\n", encoding="utf-8")
    columns = list(all_rows[0])
    with (OUT / "existing_checkpoint_rescue_timeline.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "existing_checkpoint_rescue_timeline.json").write_text(json.dumps({"split": "validation", "episodes_per_condition": 200, "rows": all_rows}, indent=2) + "\n", encoding="utf-8")
    eligible = [row for row in eligibility if row["eligible"]]
    selected = sorted(eligible, key=lambda row: (-row["minimum_acquisition"], -min(row["rear_negative_acquisition"], row["rear_positive_acquisition"]), -row["target_acquisition"], -row["aggregate_acquisition"], row["update"]))[0] if eligible else None
    (OUT / "existing_checkpoint_eligibility.json").write_text(json.dumps({"status": "PASS" if selected else "NO_RESCUE", "rows": eligibility, "selected": selected}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
