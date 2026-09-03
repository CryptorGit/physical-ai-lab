"""Select the W1B-R2 checkpoint and execute the complete read-only suite."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
CHECKPOINTS = (0, 1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
EVALUATOR = HERE.parent / "evaluate_w1b_r2.py"
TIMELINE_EVALUATOR = HERE.parent / "evaluate_w1b_r2_timeline.py"
RUN_EVALUATOR = HERE.parent / "evaluate_w1b_r2_run.py"


def checkpoint(iteration: int) -> Path:
    label = "initial" if iteration == 0 else str(iteration)
    return OUT / f"checkpoints/model_{label}.pt"


def evaluate(mode: str, path: Path, tag: str) -> dict:
    target = OUT / f"_raw_{mode}_{tag}.json"
    if not target.exists():
        subprocess.run(
            [
                sys.executable, str(EVALUATOR), "--mode", mode,
                "--checkpoint", str(path), "--tag", tag, "--headless",
            ],
            cwd=REPO,
            check=True,
        )
    return json.loads(target.read_text(encoding="utf-8"))


ranks = []
for iteration in CHECKPOINTS:
    timeline_target = OUT / f"_raw_capability_timeline_{iteration}.json"
    if not timeline_target.exists() or len(json.loads(
        timeline_target.read_text(encoding="utf-8")
    ).get("rows", [])) < 50:
        subprocess.run(
            [
                sys.executable, str(TIMELINE_EVALUATOR),
                "--mode", "capability",
                "--checkpoint", str(checkpoint(iteration)),
                "--tag", f"timeline_{iteration}", "--headless",
            ],
            cwd=REPO,
            check=True,
        )
    payload = json.loads(timeline_target.read_text(encoding="utf-8"))
    rows = payload["rows"]
    moving_rows = [row for row in rows if row["kind"] == "moving"]
    zero = [row for row in rows if row["condition"].startswith("ZERO_D")]
    pure = [row for row in rows if row["kind"] == "pure"]
    by_name = {row["condition"]: row for row in rows}
    moving_by_name = {row["condition"]: row for row in moving_rows}
    mirror_differences = []
    for row in moving_rows:
        direction = float(row["direction_deg"])
        yaw = float(row["yaw_cmd"])
        mate = moving_by_name.get(
            f"MOVE_D{((-direction) % 360):05.1f}_Y{-yaw:+.1f}"
        )
        if mate:
            mirror_differences.append(
                abs(row["success_rate"] - mate["success_rate"])
            )
    tracked = (
        "MOVE_D000.0_Y+0.3", "MOVE_D045.0_Y+0.3",
        "MOVE_D135.0_Y+0.3", "MOVE_D225.0_Y-0.3",
        "MOVE_D180.0_Y-0.3", "MOVE_D180.0_Y+0.3",
    )
    ranks.append({
        "iteration": iteration,
        "zero_yaw_pass": sum(bool(row["gate_pass"]) for row in zero),
        "moving_pass": sum(bool(row["gate_pass"]) for row in moving_rows),
        "pure_pass": sum(bool(row["gate_pass"]) for row in pure),
        "simultaneous_success": sum(
            row["both_correct_rate"] for row in moving_rows
        ) / len(moving_rows),
        "parent_failure_improvements": sum(
            moving_by_name[name]["success_rate"] >= .9
            for name in tracked
        ),
        "mirror_success_difference": sum(mirror_differences) / len(mirror_differences),
        "yaw_mae": sum(
            row["yaw_rate_mae"] for row in pure + moving_rows
        ) / len(pure + moving_rows),
        "translation_mae": sum(
            row["vector_velocity_mae"] for row in zero + moving_rows
        ) / len(zero + moving_rows),
        "forward_0p6": by_name["FWD_0P6"]["success_rate"],
        "forward_1p2": by_name["FWD_1P2"]["success_rate"],
        "fall": sum(row["fall_rate"] for row in rows) / len(rows),
        "dangerous_slip": sum(row["dangerous_slip_rate"] for row in rows) / len(rows),
    })

eligible = [row for row in ranks if row["zero_yaw_pass"] == 16]
if not eligible:
    raise SystemExit("EXP013_W1B_R2_TRANSLATION_YAW_INTERFERENCE")
eligible.sort(key=lambda row: (
    -row["moving_pass"],
    -row["pure_pass"],
    -row["simultaneous_success"],
    -row["parent_failure_improvements"],
    row["mirror_success_difference"],
    row["yaw_mae"],
    row["translation_mae"],
    -min(row["forward_0p6"], row["forward_1p2"]),
    row["fall"],
    row["dangerous_slip"],
    -row["iteration"],
))
selected_row = eligible[0]
selected = checkpoint(selected_row["iteration"])
(OUT / "selected_checkpoint.json").write_text(
    json.dumps({
        **selected_row,
        "path": str(selected.relative_to(REPO)),
        "ranked_candidates": ranks,
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

for mode in ("zero", "pure", "moving", "independence", "path", "random"):
    evaluate(mode, selected, "selected")

run_target = OUT / "_raw_run_selected.json"
if not run_target.exists():
    subprocess.run(
        [
            sys.executable, str(RUN_EVALUATOR), "--suite", "run",
            "--checkpoint", str(selected), "--tag", "selected", "--headless",
        ],
        cwd=REPO,
        check=True,
    )
print(selected_row["iteration"], selected)
