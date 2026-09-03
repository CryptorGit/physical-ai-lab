"""Guarded offline residual distillation; refuses an inadequate envelope."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage4_frozen_walk_speed_residual"
CFG_PATH = EXP / "configs/stage4_frozen_walk_speed_residual.yaml"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.frozen_walk_residual import FrozenWalkSpeedResidualController123  # noqa: E402
from unified_walk_run.student_actor import UnifiedWalkRunStudent123  # noqa: E402

OBS = [f"obs_{index:03d}" for index in range(123)]
ACT = [f"action_{index:03d}" for index in range(37)]


def load_base(device):
    payload = torch.load(
        REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        map_location=device, weights_only=False,
    )
    state = {key.removeprefix("mlp."): value for key, value in payload["actor_state_dict"].items() if key.startswith("mlp.")}
    model = UnifiedWalkRunStudent123().to(device)
    model.load_state_dict(state, strict=True)
    return model


def frame_tensors(frame, device):
    frame = frame.sort_values(["episode_id", "sequence_step"], kind="stable")
    observation = torch.from_numpy(frame[OBS].to_numpy(np.float32).copy()).to(device)
    action = torch.from_numpy(frame[ACT].to_numpy(np.float32).copy()).to(device)
    walk = torch.from_numpy((frame["regime"] == "walk_steady").to_numpy()).to(device)
    episode = frame["episode_id"].to_numpy()
    step = frame["sequence_step"].to_numpy()
    consecutive = torch.from_numpy(
        np.concatenate(([False], (episode[1:] == episode[:-1]) & (step[1:] == step[:-1] + 1)))
    ).to(device)
    return observation, action, walk, consecutive


def losses(controller, observation, teacher_action, walk, consecutive, cfg):
    base, bounded, gate = controller.forward_components(observation)
    predicted = base.clone()
    active = torch.nonzero(gate > 0, as_tuple=False).flatten()
    if len(active):
        predicted[active] = base[active] + gate[active, None] * bounded[active]
    nonwalk = ~walk
    action = F.huber_loss(predicted[nonwalk], teacher_action[nonwalk], delta=cfg["action_huber_delta"]) if nonwalk.any() else predicted.sum() * 0
    delta = F.huber_loss(
        predicted[1:][consecutive[1:]] - predicted[:-1][consecutive[1:]],
        teacher_action[1:][consecutive[1:]] - teacher_action[:-1][consecutive[1:]],
        delta=cfg["action_huber_delta"],
    ) if consecutive[1:].any() else predicted.sum() * 0
    zero_anchor = F.huber_loss(bounded[walk], torch.zeros_like(bounded[walk]), delta=cfg["action_huber_delta"]) if walk.any() else bounded.sum() * 0
    total = (
        cfg["action_loss_weight"] * action
        + cfg["action_delta_loss_weight"] * delta
        + cfg["walk_zero_anchor_weight"] * zero_anchor
    )
    return total, {"action": action, "action_delta": delta, "zero_anchor": zero_anchor}


def main() -> None:
    audit = json.loads((OUT / "residual_parameterization_audit.json").read_text(encoding="utf-8"))
    if not audit["pass"]:
        raise SystemExit("BLOCKED: RESIDUAL_PARAMETERIZATION_INADEQUATE; optimizer was not constructed")
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    device = torch.device(cfg["experiment"]["device"])
    torch.manual_seed(cfg["experiment"]["training_seed"])
    controller = FrozenWalkSpeedResidualController123(
        load_base(device), torch.tensor(cfg["controller"]["residual"]["per_joint_bounds"], device=device)
    ).to(device)
    optimizer = torch.optim.AdamW(
        controller.residual.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    parts = sorted((REPO / cfg["dataset"]["path"]).glob("*.parquet"))
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best, patience = float("inf"), 0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        random.Random(cfg["experiment"]["training_seed"] + epoch).shuffle(parts)
        for part in parts:
            frame = pd.read_parquet(part, columns=OBS + ACT + ["regime", "split", "episode_id", "sequence_step"])
            frame = frame[frame["split"] == "train"]
            if not len(frame):
                continue
            observation, teacher, walk, consecutive = frame_tensors(frame, device)
            for start in range(0, len(frame), cfg["training"]["batch_size"]):
                sl = slice(start, start + cfg["training"]["batch_size"])
                loss, _ = losses(controller, observation[sl], teacher[sl], walk[sl], consecutive[sl], cfg["training"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(controller.residual.parameters(), cfg["training"]["gradient_clip_norm"])
                optimizer.step()
        # Full validation is intentionally deterministic and uses the same grouped parts.
        validation_total, validation_rows = 0.0, 0
        with torch.no_grad():
            for part in sorted(parts):
                frame = pd.read_parquet(part, columns=OBS + ACT + ["regime", "split", "episode_id", "sequence_step"])
                frame = frame[frame["split"] == "validation"]
                if not len(frame):
                    continue
                observation, teacher, walk, consecutive = frame_tensors(frame, device)
                value, _ = losses(controller, observation, teacher, walk, consecutive, cfg["training"])
                validation_total += float(value) * len(frame)
                validation_rows += len(frame)
        validation = validation_total / validation_rows
        payload = {
            "residual": controller.residual.state_dict(), "residual_bounds": controller.residual_bounds,
            "optimizer": optimizer.state_dict(), "epoch": epoch, "validation_loss": validation,
        }
        if epoch in cfg["training"]["checkpoint_epochs"]:
            torch.save(payload, checkpoint_dir / f"epoch_{epoch}.pt")
        if validation < best - cfg["training"]["early_stopping_min_delta"]:
            best, patience = validation, 0
            torch.save(payload, checkpoint_dir / "best_validation.pt")
        else:
            patience += 1
        if epoch >= cfg["training"]["early_stopping_minimum_epoch"] and patience >= cfg["training"]["early_stopping_patience_epochs"]:
            break


if __name__ == "__main__":
    main()
