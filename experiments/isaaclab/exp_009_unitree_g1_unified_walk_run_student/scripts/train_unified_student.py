"""Offline-only multi-teacher distillation."""

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
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src"),
]
from g1_walk_centered.experts import load_walk_expert
from unified_walk_run.dataset import action_columns, observation_columns
from unified_walk_run.distillation_loss import distillation_loss
from unified_walk_run.student_actor import UnifiedWalkRunStudent123


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_part(path, split):
    obs_cols, act_cols = observation_columns(), action_columns()
    columns = obs_cols + act_cols + ["episode_id", "sequence_step", "split"]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["split"].eq(split)].sort_values(["episode_id", "sequence_step"])
    if frame.empty:
        return None
    previous_indices = frame.groupby("episode_id", sort=False).cumcount() > 0
    previous_obs = frame.groupby("episode_id", sort=False)[obs_cols].shift(1)
    previous_action = frame.groupby("episode_id", sort=False)[act_cols].shift(1)
    frame = frame[previous_indices]
    return (
        frame[obs_cols].to_numpy(np.float32),
        frame[act_cols].to_numpy(np.float32),
        previous_obs.loc[previous_indices].to_numpy(np.float32),
        previous_action.loc[previous_indices].to_numpy(np.float32),
    )


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    seed = cfg["experiment"]["training_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg["experiment"]["device"])
    data_parts = sorted((OUT / "teacher_dataset.parquet").glob("*.parquet"))
    if not data_parts:
        raise RuntimeError("teacher dataset missing")
    dataset_hashes = {part.name: sha_file(part) for part in data_parts}
    dataset_hash = canonical_hash(dataset_hashes)
    walk = load_walk_expert(REPO / cfg["teachers"]["walk"]["path"], device=device)
    student = UnifiedWalkRunStudent123().to(device)
    student.initialize_from_walk(walk.actor)
    probe = torch.randn(32, 123, generator=torch.Generator(device=device).manual_seed(seed), device=device)
    with torch.no_grad():
        bitwise = torch.equal(student(probe), walk.actor(probe))
    if not bitwise:
        raise RuntimeError("student/WALK initialization is not bitwise identical")
    initial_state_bytes = b"".join(value.detach().cpu().numpy().tobytes() for value in student.state_dict().values())
    initialization_hash = sha_bytes(initial_state_bytes)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    init_path = checkpoint_dir / "initial.pt"
    torch.save({"student": student.state_dict(), "epoch": 0, "dataset_hash": dataset_hash}, init_path)
    training = cfg["training"]
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    curves = []
    checkpoint_manifest = []
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    best_path = checkpoint_dir / "best_validation.pt"

    def evaluate(split):
        student.eval()
        total, action_sum, delta_sum = 0, 0.0, 0.0
        with torch.no_grad():
            for part in data_parts:
                loaded = load_part(part, split)
                if loaded is None:
                    continue
                obs, action, prev_obs, prev_action = loaded
                for begin in range(0, len(obs), training["batch_size"]):
                    end = begin + training["batch_size"]
                    x = torch.as_tensor(obs[begin:end], device=device)
                    y = torch.as_tensor(action[begin:end], device=device)
                    px = torch.as_tensor(prev_obs[begin:end], device=device)
                    py = torch.as_tensor(prev_action[begin:end], device=device)
                    current, previous = student(x), student(px)
                    _, terms = distillation_loss(
                        current, y, previous, py,
                        delta=training["action_huber_delta"],
                        action_weight=training["action_loss_weight"],
                        action_delta_weight=training["action_delta_loss_weight"],
                    )
                    count = len(x)
                    total += count
                    action_sum += float(terms["action_huber"]) * count
                    delta_sum += float(terms["action_delta_huber"]) * count
        return {
            "count": total,
            "action_huber": action_sum / max(total, 1),
            "action_delta_huber": delta_sum / max(total, 1),
            "weighted_total": (
                training["action_loss_weight"] * action_sum
                + training["action_delta_loss_weight"] * delta_sum
            ) / max(total, 1),
        }

    for epoch in range(1, training["epochs"] + 1):
        student.train()
        order = list(data_parts)
        random.Random(seed + epoch).shuffle(order)
        total, loss_sum = 0, 0.0
        for part in order:
            loaded = load_part(part, "train")
            if loaded is None:
                continue
            obs, action, prev_obs, prev_action = loaded
            permutation = np.random.default_rng(seed + epoch + total).permutation(len(obs))
            for begin in range(0, len(obs), training["batch_size"]):
                indices = permutation[begin : begin + training["batch_size"]]
                x = torch.as_tensor(obs[indices], device=device)
                y = torch.as_tensor(action[indices], device=device)
                px = torch.as_tensor(prev_obs[indices], device=device)
                py = torch.as_tensor(prev_action[indices], device=device)
                current, previous = student(x), student(px)
                loss, _ = distillation_loss(
                    current, y, previous, py,
                    delta=training["action_huber_delta"],
                    action_weight=training["action_loss_weight"],
                    action_delta_weight=training["action_delta_loss_weight"],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), training["gradient_clip_norm"])
                optimizer.step()
                total += len(indices)
                loss_sum += float(loss.detach()) * len(indices)
        validation = evaluate("validation")
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(total, 1),
            "validation_loss": validation["weighted_total"],
            "validation_action_huber": validation["action_huber"],
            "validation_action_delta_huber": validation["action_delta_huber"],
        }
        curves.append(row)
        improved = validation["weighted_total"] < best_loss - training["early_stopping_min_delta"]
        if improved:
            best_loss, best_epoch, patience = validation["weighted_total"], epoch, 0
            torch.save(
                {"student": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "dataset_hash": dataset_hash},
                best_path,
            )
        else:
            patience += 1
        if epoch in training["checkpoint_epochs"]:
            path = checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(
                {"student": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "dataset_hash": dataset_hash},
                path,
            )
            checkpoint_manifest.append({"epoch": epoch, "path": str(path.relative_to(REPO)), "sha256": sha_file(path)})
        print(f"[exp009 distill] epoch={epoch} train={row['train_loss']:.7f} val={row['validation_loss']:.7f}", flush=True)
        if patience >= training["early_stopping_patience_epochs"] and epoch >= training["early_stopping_minimum_epoch"]:
            break
    final_path = checkpoint_dir / "final.pt"
    torch.save(
        {"student": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": curves[-1]["epoch"], "dataset_hash": dataset_hash},
        final_path,
    )
    checkpoint_manifest.extend([
        {"epoch": 0, "path": str(init_path.relative_to(REPO)), "sha256": sha_file(init_path)},
        {"epoch": best_epoch, "path": str(best_path.relative_to(REPO)), "sha256": sha_file(best_path), "role": "best_validation"},
        {"epoch": curves[-1]["epoch"], "path": str(final_path.relative_to(REPO)), "sha256": sha_file(final_path), "role": "final"},
    ])
    test = evaluate("test")
    with (OUT / "training_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    (OUT / "dataset_hashes.json").write_text(json.dumps({"parts": dataset_hashes, "dataset_sha256": dataset_hash}, indent=2) + "\n")
    (OUT / "student_initialization.json").write_text(json.dumps({
        "source": cfg["teachers"]["walk"]["path"],
        "strict_copy": True,
        "bitwise_action_match": bitwise,
        "initialization_sha256": initialization_hash,
        "all_student_parameters_trainable": all(parameter.requires_grad for parameter in student.parameters()),
        "teacher_gradients": 0,
    }, indent=2) + "\n")
    (OUT / "student_architecture.json").write_text(json.dumps({
        "name": "UnifiedWalkRunStudent123",
        "layers": [123, 256, 128, 128, 37],
        "activation": "ELU",
        "parameter_count": sum(parameter.numel() for parameter in student.parameters()),
        "single_head": True,
        "teacher_identity_input": False,
        "action_scale": 0.5,
    }, indent=2) + "\n")
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2) + "\n")
    (OUT / "offline_evaluation.json").write_text(json.dumps({
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test": test,
        "early_stopped": curves[-1]["epoch"] < training["epochs"],
        "completed_epochs": curves[-1]["epoch"],
    }, indent=2) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "test": test}, indent=2))


if __name__ == "__main__":
    main()
