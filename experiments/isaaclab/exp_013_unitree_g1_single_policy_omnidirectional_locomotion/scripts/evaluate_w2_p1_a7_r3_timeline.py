"""Validation capability timeline and checkpoint selection for A7-R3."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = BASE / "phase_w2_p1_a7_r3_start_retention_recovery"
RAW = OUT / "raw/capability_timeline"
EVALUATOR = HERE.parent / "evaluate_w2_p1_a7_r3.py"
GUARD = HERE.parent / "evaluate_w2_p1_a7_r2.py"
ISAAC = Path.home() / "workspace/IsaacLab/isaaclab.bat"
UPDATES = (0, 1, 5, 10, 15, 20, 25, 30)


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def condition(update: int, index: int) -> dict:
    direction = (index // 3) * 45.0; yaw = (-0.3, 0.0, 0.3)[index % 3]
    output = RAW / f"update_{update:03d}_condition_{index:02d}.csv"
    if not output.with_suffix(".json").exists():
        command = [str(ISAAC), "-p", str(EVALUATOR), "--policy", str(OUT / f"checkpoints/model_{update:03d}.pt"), "--batch", "4", "--split", "validation", "--direction", str(direction), "--speed", "0.3", "--yaw", str(yaw), "--episodes", "200", "--output", str(output), "--headless", "--device", "cuda:0"]
        with output.with_suffix(".log").open("w", encoding="utf-8") as log: subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["row"]


def guard(update: int) -> list[dict]:
    output = RAW / f"update_{update:03d}_static.csv"
    if not output.with_suffix(".json").exists():
        command = [str(ISAAC), "-p", str(GUARD), "--policy", str(OUT / f"checkpoints/model_{update:03d}.pt"), "--batch", "4", "--split", "validation", "--mode", "guard", "--output", str(output), "--headless", "--device", "cuda:0"]
        with output.with_suffix(".log").open("w", encoding="utf-8") as log: subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["rows"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True); rows, candidates = [], []
    for update in UPDATES:
        matrix = []
        for index in range(24):
            row = condition(update, index); row["update"] = update; matrix.append(row); rows.append(row)
            print(json.dumps({"update": update, "condition": index, "status": "COMPLETE"}), flush=True)
        static = guard(update); groups = {name: [row for row in static if row["group"] == name] for name in {row["group"] for row in static}}
        target = next(row for row in matrix if row["direction"] == 315.0 and row["yaw"] == 0.3)
        mirror = next(row for row in matrix if row["direction"] == 45.0 and row["yaw"] == -0.3)
        rear_negative = next(row for row in matrix if row["direction"] == 180.0 and row["yaw"] == -0.3)
        rear_positive = next(row for row in matrix if row["direction"] == 180.0 and row["yaw"] == 0.3)
        matrix_pass = all(row["endpoint_success"] >= 0.90 and row["acquisition_0p20"] >= 0.85 and row["fall_rate"] <= 0.05 for row in matrix)
        target_pass = target["endpoint_success"] >= 0.95 and target["acquisition_0p20"] >= 0.85 and target["fall_rate"] <= 0.02
        mirror_pass = mirror["endpoint_success"] >= 0.95 and mirror["acquisition_0p20"] >= 0.85 and mirror["fall_rate"] <= 0.02
        rear_pass = all(row["endpoint_success"] >= 0.95 and row["acquisition_0p20"] >= 0.90 and row["fall_rate"] <= 0.02 for row in (rear_negative, rear_positive))
        static_pass = sum(row["endpoint_success"] >= 0.90 for row in groups["zero_yaw"]) == 16 and min(row["endpoint_success"] for row in groups["forward_anchor"]) >= 0.95 and min(row["endpoint_success"] for row in groups["pure_yaw"]) >= 0.90 and sum(row["endpoint_success"] >= 0.90 for row in groups["moving_turn"]) == 24
        safety_pass = max(row["fall_rate"] for row in matrix) <= 0.02 and max(row["dangerous_slip_rate"] for row in matrix) <= 0.10 and max(row["impact_rate"] for row in matrix) <= 0.05 and max(row["saturation_rate"] for row in matrix) <= 0.05
        candidates.append({"update": update, "minimum_acquisition": min(row["acquisition_0p20"] for row in matrix), "target_acquisition": target["acquisition_0p20"], "mirror_acquisition": mirror["acquisition_0p20"], "rear_negative_acquisition": rear_negative["acquisition_0p20"], "rear_positive_acquisition": rear_positive["acquisition_0p20"], "target_mirror_difference": abs(target["acquisition_0p20"] - mirror["acquisition_0p20"]), "yaw_resets": target["yaw_timer_resets"] + mirror["yaw_timer_resets"], "longest_yaw_pass": min(target["longest_yaw_pass_s"], mirror["longest_yaw_pass_s"]), "matrix_pass": matrix_pass, "target_pass": target_pass, "mirror_pass": mirror_pass, "rear_pass": rear_pass, "static_pass": static_pass, "safety_pass": safety_pass, "eligible": matrix_pass and target_pass and mirror_pass and rear_pass and static_pass and safety_pass})
        (OUT / "a7_r3_capability_timeline.json").write_text(json.dumps({"status": "IN_PROGRESS", "rows": rows, "candidates": candidates}, indent=2) + "\n", encoding="utf-8")
    with (OUT / "a7_r3_capability_timeline.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    eligible = [row for row in candidates if row["eligible"]]
    selected = sorted(eligible, key=lambda row: (-row["minimum_acquisition"], -row["target_acquisition"], -min(row["rear_negative_acquisition"], row["rear_positive_acquisition"]), row["target_mirror_difference"], row["yaw_resets"], -row["longest_yaw_pass"], row["update"]))[0] if eligible else None
    (OUT / "a7_r3_capability_timeline.json").write_text(json.dumps({"status": "PASS" if selected else "NO_ELIGIBLE", "rows": rows, "candidates": candidates, "selected": selected}, indent=2) + "\n", encoding="utf-8")
    if selected:
        path = OUT / f"checkpoints/model_{selected['update']:03d}.pt"
        (OUT / "selected_checkpoint.json").write_text(json.dumps({"source": "A7-R3 localized retention recovery", "selected_update": selected["update"], "path": str(path.relative_to(OUT)).replace("\\", "/"), "sha256": sha(path), "selection_split": "validation only", "heldout_fallback": False, "rationale": selected}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
