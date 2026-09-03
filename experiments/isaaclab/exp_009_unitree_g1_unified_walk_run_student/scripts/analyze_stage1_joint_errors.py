"""Detailed joint/support/action-delta errors for the frozen Stage-0 student."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
STAGE0 = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.dataset import action_columns, observation_columns
from unified_walk_run.student_actor import UnifiedWalkRunStudent123

OBS, ACT = observation_columns(), action_columns()
columns = OBS + ACT + ["regime", "split", "episode_id", "sequence_step", "support_phase"]
frames = []
for part in sorted((STAGE0 / "teacher_dataset.parquet").glob("*.parquet")):
    frame = pd.read_parquet(part, columns=columns)
    frame = frame[frame["split"].eq("test")]
    if not frame.empty:
        frames.append(frame)
data = pd.concat(frames, ignore_index=True).sort_values(["episode_id", "sequence_step"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UnifiedWalkRunStudent123().to(device)
model.load_state_dict(torch.load(STAGE0 / "checkpoints/epoch_10.pt", map_location=device, weights_only=False)["student"])
prediction = []
with torch.no_grad():
    values = data[OBS].to_numpy(np.float32)
    for begin in range(0, len(values), 16384):
        prediction.append(model(torch.from_numpy(values[begin:begin + 16384]).to(device)).cpu().numpy())
prediction = np.concatenate(prediction)
target = data[ACT].to_numpy(np.float32)
error = prediction - target
rows = []
for (regime, phase), indices in data.groupby(["regime", "support_phase"]).groups.items():
    index = np.asarray(list(indices))
    for joint in range(37):
        value = error[index, joint]
        rows.append({
            "regime": regime, "support_phase": int(phase), "joint_index": joint,
            "mae": float(np.abs(value).mean()), "rmse": float(np.sqrt(np.square(value).mean())),
            "p95_abs_error": float(np.quantile(np.abs(value), .95)),
            "error_lag1_autocorrelation": float(np.corrcoef(value[:-1], value[1:])[0, 1]) if len(value) > 2 else 0.0,
        })
with (OUT / "jointwise_action_errors.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

valid_previous = data.groupby("episode_id").cumcount().to_numpy() > 0
teacher_delta = target[1:] - target[:-1]
student_delta = prediction[1:] - prediction[:-1]
same_episode = data["episode_id"].to_numpy()[1:] == data["episode_id"].to_numpy()[:-1]
delta_error = student_delta[same_episode] - teacher_delta[same_episode]
summary = {
    "rows": len(data),
    "test_split_only": True,
    "action_delta_mae": float(np.abs(delta_error).mean()),
    "action_delta_p95": float(np.quantile(np.abs(delta_error), .95)),
    "support_phase_rows": len(rows),
}
(OUT / "jointwise_action_error_summary.json").write_text(__import__("json").dumps(summary, indent=2) + "\n")
