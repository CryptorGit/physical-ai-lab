"""Run the frozen held-out A7-R2 teacher authorization matrix."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2"
RAW = OUT / "raw/formal_heldout"
EVALUATOR = HERE.parent / "evaluate_w2_p1_a7_r2.py"
ISAAC = Path.home() / "workspace/IsaacLab/isaaclab.bat"
POLICY = OUT / "checkpoints/model_075.pt"


def evaluate(mode: str, count: int, episodes: int, destination: str) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        output = RAW / f"{mode}_{index:02d}.csv"
        if output.with_suffix(".json").exists():
            existing = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            existing_rows = existing.get("rows", [])
            if existing_rows and "acquisition_0p10" in existing_rows[0] and "yaw_mae" in existing_rows[0]:
                rows.extend(existing_rows)
                print(json.dumps({"mode": mode, "condition_index": index, "status": "REUSED_EXACT"}), flush=True)
                continue
        command = [
            str(ISAAC), "-p", str(EVALUATOR), "--policy", str(POLICY),
            "--batch", "5", "--split", "heldout", "--mode", mode,
            "--condition-index", str(index), "--episodes", str(episodes),
            "--output", str(output), "--headless", "--device", "cuda:0",
        ]
        with output.with_suffix(".log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        print(json.dumps({"mode": mode, "condition_index": index, "status": "COMPLETE"}), flush=True)
    columns = list(rows[0])
    target = OUT / destination
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    target.with_suffix(".json").write_text(
        json.dumps({"split": "heldout", "selected_update": 75, "fallback": False, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


RAW.mkdir(parents=True, exist_ok=True)
evaluate("formal_matrix", 24, 300, "formal_start_matrix.csv")
evaluate("pure_yaw", 2, 300, "formal_pure_yaw_start.csv")
evaluate("rear_boundary", 12, 200, "formal_rear_speed_boundary.csv")
