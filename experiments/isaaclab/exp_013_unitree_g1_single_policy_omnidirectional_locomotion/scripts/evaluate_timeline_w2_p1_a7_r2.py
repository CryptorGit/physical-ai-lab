"""Run the preregistered A7-R2 validation checkpoint timeline."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2"
RAW = OUT / "raw/timeline"
EVALUATOR = HERE.parent / "evaluate_w2_p1_a7_r2.py"
ISAAC = Path.home() / "workspace/IsaacLab/isaaclab.bat"
UPDATES = (0, 1, 10, 20, 45, 75, 100, 120, 130, 140, 150)


def run(update: int, mode: str, episodes: int | None = None) -> list[dict]:
    destination = RAW / f"update_{update:03d}_{mode}.csv"
    command = [
        str(ISAAC), "-p", str(EVALUATOR),
        "--policy", str(OUT / f"checkpoints/model_{update:03d}.pt"),
        "--batch", "4", "--split", "validation", "--mode", mode,
        "--output", str(destination), "--headless", "--device", "cuda:0",
    ]
    if episodes is not None:
        command[command.index("--output"):command.index("--output")] = ["--episodes", str(episodes)]
    with destination.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    payload = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
    return [{"update": update, "mode": mode, **row} for row in payload["rows"]]


RAW.mkdir(parents=True, exist_ok=True)
rows: list[dict] = []
for update in UPDATES:
    rows.extend(run(update, "timeline", 800))
    rows.extend(run(update, "guard"))
    print(json.dumps({"update": update, "status": "VALIDATION_TIMELINE_COMPLETE"}), flush=True)

columns = sorted({key for row in rows for key in row})
with (OUT / "a7_capability_timeline.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
(OUT / "a7_capability_timeline.json").write_text(
    json.dumps({"split": "validation", "updates": list(UPDATES), "rows": rows}, indent=2) + "\n",
    encoding="utf-8",
)
