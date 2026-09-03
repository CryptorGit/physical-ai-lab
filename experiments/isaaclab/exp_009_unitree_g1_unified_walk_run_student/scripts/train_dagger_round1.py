"""Single allowed DAgger round; offline supervised updates only."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"
CFG_PATH = EXP / "configs/stage0_multiteacher_distillation.yaml"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.dataset import action_columns, observation_columns
from unified_walk_run.distillation_loss import distillation_loss
from unified_walk_run.student_actor import UnifiedWalkRunStudent123


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_part(path):
    obs_cols, act_cols = observation_columns(), action_columns()
    frame = pd.read_parquet(path, columns=obs_cols + act_cols + ["episode_id", "sequence_step", "split"])
    frame = frame[frame["split"].eq("train")].sort_values(["episode_id", "sequence_step"])
    previous = frame.groupby("episode_id", sort=False).cumcount() > 0
    previous_obs = frame.groupby("episode_id", sort=False)[obs_cols].shift(1)
    previous_action = frame.groupby("episode_id", sort=False)[act_cols].shift(1)
    frame = frame[previous]
    return (
        frame[obs_cols].to_numpy(np.float32),
        frame[act_cols].to_numpy(np.float32),
        previous_obs.loc[previous].to_numpy(np.float32),
        previous_action.loc[previous].to_numpy(np.float32),
    )


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    seed = cfg["experiment"]["training_seed"] + 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(cfg["experiment"]["device"])
    parent = OUT / "checkpoints/epoch_1.pt"
    payload = torch.load(parent, map_location=device, weights_only=False)
    student = UnifiedWalkRunStudent123().to(device)
    student.load_state_dict(payload["student"], strict=True)
    training, dagger = cfg["training"], cfg["dagger"]
    optimizer = torch.optim.AdamW(student.parameters(), lr=dagger["learning_rate"], weight_decay=training["weight_decay"])
    parts = sorted((OUT / "teacher_dataset.parquet").glob("*.parquet")) + sorted((OUT / "dagger_dataset.parquet").glob("*.parquet"))
    curves = []
    for epoch in range(1, dagger["epochs"] + 1):
        order = list(parts)
        random.Random(seed + epoch).shuffle(order)
        total, loss_sum = 0, 0.0
        student.train()
        for part in order:
            obs, action, previous_obs, previous_action = load_part(part)
            permutation = np.random.default_rng(seed + epoch + total).permutation(len(obs))
            for begin in range(0, len(obs), training["batch_size"]):
                ids = permutation[begin : begin + training["batch_size"]]
                x, y = torch.as_tensor(obs[ids], device=device), torch.as_tensor(action[ids], device=device)
                px, py = torch.as_tensor(previous_obs[ids], device=device), torch.as_tensor(previous_action[ids], device=device)
                loss, _ = distillation_loss(
                    student(x), y, student(px), py, delta=training["action_huber_delta"],
                    action_weight=training["action_loss_weight"], action_delta_weight=training["action_delta_loss_weight"],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), training["gradient_clip_norm"])
                optimizer.step()
                total += len(ids)
                loss_sum += float(loss.detach()) * len(ids)
        row = {"epoch": epoch, "train_loss": loss_sum / max(total, 1), "samples": total}
        curves.append(row)
        print(f"[exp009 dagger] epoch={epoch} loss={row['train_loss']:.7f}", flush=True)
    path = OUT / "checkpoints/dagger_round1.pt"
    torch.save({
        "student": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": dagger["epochs"],
        "dagger_round": 1, "parent": str(parent), "parent_sha256": sha(parent), "ppo": 0, "reward": False,
    }, path)
    with (OUT / "dagger_training_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    (OUT / "dagger_training_summary.json").write_text(json.dumps({
        "rounds": 1, "parent": str(parent), "parent_sha256": sha(parent),
        "checkpoint": str(path), "sha256": sha(path), "epochs": dagger["epochs"],
        "teacher_parts": len(list((OUT / "teacher_dataset.parquet").glob("*.parquet"))),
        "dagger_parts": len(list((OUT / "dagger_dataset.parquet").glob("*.parquet"))),
        "ppo": 0, "reward": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
