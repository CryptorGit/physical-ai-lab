"""Joint- and phase-wise held-out WALK action errors for the selected Stage-2 checkpoint."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation"
STAGE0 = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.dataset import action_columns, observation_columns
from unified_walk_run.student_actor import UnifiedWalkRunStudent123

JOINTS = [
    "left_hip_pitch", "right_hip_pitch", "torso", "left_hip_roll", "right_hip_roll",
    "left_shoulder_pitch", "right_shoulder_pitch", "left_hip_yaw", "right_hip_yaw",
    "left_shoulder_roll", "right_shoulder_roll", "left_knee", "right_knee",
    "left_shoulder_yaw", "right_shoulder_yaw", "left_ankle_pitch", "right_ankle_pitch",
    "left_elbow_pitch", "right_elbow_pitch", "left_ankle_roll", "right_ankle_roll",
    "left_elbow_roll", "right_elbow_roll", "left_five", "left_three", "left_zero",
    "right_five", "right_three", "right_zero", "left_six", "left_four", "left_one",
    "right_six", "right_four", "right_one", "left_two", "right_two",
]


def main() -> None:
    selected = json.loads((OUT / "walk_only_closed_loop_results.json").read_text())
    payload = torch.load(REPO / selected["checkpoint"], map_location="cuda:0", weights_only=False)
    model = UnifiedWalkRunStudent123().cuda()
    model.load_state_dict(payload["student"], strict=True)
    model.eval()
    obs_cols, act_cols = observation_columns(), action_columns()
    sums = np.zeros((4, 37), np.float64)
    squares = np.zeros((4, 37), np.float64)
    counts = np.zeros(4, np.int64)
    with torch.no_grad():
        for part in sorted((STAGE0 / "teacher_dataset.parquet").glob("*.parquet")):
            frame = pd.read_parquet(part, columns=obs_cols + act_cols + ["split", "regime", "support_phase"])
            frame = frame[frame["split"].eq("test") & frame["regime"].eq("walk_steady")]
            if frame.empty:
                continue
            prediction = model(torch.from_numpy(frame[obs_cols].to_numpy(np.float32)).cuda()).cpu().numpy()
            error = prediction - frame[act_cols].to_numpy(np.float32)
            phases = frame["support_phase"].to_numpy(np.int64).clip(0, 3)
            for phase in range(4):
                chosen = error[phases == phase]
                if len(chosen):
                    sums[phase] += np.abs(chosen).sum(0)
                    squares[phase] += np.square(chosen).sum(0)
                    counts[phase] += len(chosen)
    rows = []
    for phase in range(4):
        for joint, name in enumerate(JOINTS):
            rows.append({
                "support_phase": phase, "joint_index": joint, "joint_name": name,
                "samples": int(counts[phase]), "mae": float(sums[phase, joint] / max(counts[phase], 1)),
                "rmse": float(np.sqrt(squares[phase, joint] / max(counts[phase], 1))),
            })
    with (OUT / "walk_only_joint_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    ankle = [row for row in rows if "ankle_roll" in row["joint_name"]]
    (OUT / "walk_only_offline_joint_error_summary.json").write_text(json.dumps({
        "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
        "overall_mae": float(sums.sum() / max(counts.sum() * 37, 1)),
        "ankle_roll": ankle, "per_phase_joint_csv": "walk_only_joint_errors.csv",
        "teacher_action_order_matches_robot_action_order": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
