"""Offline Stage 2 dynamics-sensitive distillation; never performs RL or teacher updates."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation"
STAGE0 = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"
STAGE1 = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis"
CFG_PATH = EXP / "configs/stage2_dynamics_sensitive_distillation.yaml"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.dataset import action_columns, observation_columns
from unified_walk_run.dynamics_sensitive_loss import DynamicsSensitivityTable, dynamics_sensitive_distillation_loss
from unified_walk_run.student_actor import UnifiedWalkRunStudent123

OBS, ACT = observation_columns(), action_columns()
REGIME_ID = {"walk_steady": 0, "run_steady": 1, "walk_to_run": 2}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parts() -> list[Path]:
    value = sorted((STAGE0 / "teacher_dataset.parquet").glob("*.parquet"))
    if not value:
        raise RuntimeError("Stage 0 fixed teacher dataset is missing")
    return value


def load_part(path: Path, split: str, scope: str):
    columns = OBS + ACT + ["episode_id", "sequence_step", "split", "regime", "support_phase"]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["split"].eq(split)]
    if scope == "walk_only":
        frame = frame[frame["regime"].eq("walk_steady")]
    frame = frame.sort_values(["episode_id", "sequence_step"])
    if frame.empty:
        return None
    has_previous = frame.groupby("episode_id", sort=False).cumcount() > 0
    previous_obs = frame.groupby("episode_id", sort=False)[OBS].shift(1)
    previous_action = frame.groupby("episode_id", sort=False)[ACT].shift(1)
    frame = frame[has_previous]
    return (
        frame[OBS].to_numpy(np.float32),
        frame[ACT].to_numpy(np.float32),
        previous_obs.loc[has_previous].to_numpy(np.float32),
        previous_action.loc[has_previous].to_numpy(np.float32),
        frame["regime"].map(REGIME_ID).to_numpy(np.int64),
        frame["support_phase"].to_numpy(np.int64).clip(0, 3),
    )


def make_table(device: torch.device) -> DynamicsSensitivityTable:
    payload = torch.load(OUT / "sensitivity_table.pt", map_location="cpu", weights_only=False)
    return DynamicsSensitivityTable(
        payload["jacobian"], payload["contact"], payload["critical_indices"],
        payload["centroids"], payload["observation_scale"],
    ).to(device)


def initialize_student(device: torch.device) -> UnifiedWalkRunStudent123:
    student = UnifiedWalkRunStudent123().to(device)
    initial = torch.load(STAGE0 / "checkpoints/initial.pt", map_location=device, weights_only=False)["student"]
    student.load_state_dict(initial, strict=True)
    if sum(parameter.numel() for parameter in student.parameters()) != 85925:
        raise RuntimeError("student parameter contract changed")
    return student


def initialize_calibration_student(device: torch.device) -> UnifiedWalkRunStudent123:
    """Use the frozen Stage-1 standard-Huber error distribution for non-degenerate calibration."""
    payload = torch.load(STAGE1 / "checkpoints/single_walk_steady.pt", map_location=device, weights_only=False)
    source = payload["model"]
    translated = {
        (name[len("network."):] if name.startswith("network.") else name): value for name, value in source.items()
    }
    student = UnifiedWalkRunStudent123().to(device)
    student.load_state_dict(translated, strict=True)
    return student


def calibration(student, table, data_parts, cfg, device, scope):
    sample_terms = {"action": [], "action_delta": [], "dynamic": [], "contact": []}
    limit = 131072
    student.eval()
    with torch.no_grad():
        for part in data_parts:
            loaded = load_part(part, "train", scope)
            if loaded is None:
                continue
            obs, action, prev_obs, prev_action, regime, phase = loaded
            remaining = limit - sum(len(value) for value in sample_terms["action"])
            if remaining <= 0:
                break
            if len(obs) > remaining:
                index = np.random.default_rng(cfg["experiment"]["seed"]).choice(len(obs), remaining, replace=False)
                obs, action, prev_obs, prev_action, regime, phase = (
                    value[index] for value in (obs, action, prev_obs, prev_action, regime, phase)
                )
            for begin in range(0, len(obs), cfg["training"]["batch_size"]):
                end = begin + cfg["training"]["batch_size"]
                x, y = torch.from_numpy(obs[begin:end]).to(device), torch.from_numpy(action[begin:end]).to(device)
                px, py = torch.from_numpy(prev_obs[begin:end]).to(device), torch.from_numpy(prev_action[begin:end]).to(device)
                r, p = torch.from_numpy(regime[begin:end]).to(device), torch.from_numpy(phase[begin:end]).to(device)
                current, previous = student(x), student(px)
                error = current - y
                action_per = F.huber_loss(current, y, delta=cfg["training"]["action_huber_delta"], reduction="none").mean(1)
                delta_per = F.huber_loss(
                    current - previous, y - py, delta=cfg["training"]["action_huber_delta"], reduction="none"
                ).mean(1)
                dynamic_per, contact_per = table.terms(error, x, r, p, cfg["training"]["action_huber_delta"])
                for key, values in (
                    ("action", action_per), ("action_delta", delta_per),
                    ("dynamic", dynamic_per), ("contact", contact_per),
                ):
                    sample_terms[key].append(values.cpu().numpy())
    medians = {key: float(np.median(np.concatenate(value))) for key, value in sample_terms.items()}
    epsilon = float(cfg["calibration"]["epsilon"])
    reference = max(medians["action"], epsilon)
    lambda_dynamic = min(
        float(cfg["calibration"]["lambda_dynamic_max"]),
        float(cfg["calibration"]["dynamic_target_median"]) * reference / max(medians["dynamic"], epsilon),
    )
    lambda_contact = min(
        float(cfg["calibration"]["lambda_contact_max"]),
        float(cfg["calibration"]["contact_target_median"]) * reference / max(medians["contact"], epsilon),
    )
    return {
        "sample_count": int(sum(len(value) for value in sample_terms["action"])),
        "unweighted_medians": medians,
        "action_reference_normalized": 1.0,
        "weighted_median_targets_relative_to_action": {
            "action": 1.0,
            "action_delta_weight": cfg["training"]["action_delta_loss_weight"],
            "dynamic": cfg["calibration"]["dynamic_target_median"],
            "contact": cfg["calibration"]["contact_target_median"],
        },
        "lambda_dynamic": lambda_dynamic,
        "lambda_contact": lambda_contact,
        "lambda_dynamic_max": cfg["calibration"]["lambda_dynamic_max"],
        "lambda_contact_max": cfg["calibration"]["lambda_contact_max"],
        "calibrated_once_before_training": True,
        "validation_dependent": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("walk_only", "mixed"), required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    seed = int(cfg["experiment"]["seed"]) + (0 if args.scope == "walk_only" else 100)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(cfg["experiment"]["device"])
    OUT.mkdir(parents=True, exist_ok=True)
    data_parts = parts()
    dataset_hashes = {str(path.relative_to(REPO)): sha(path) for path in data_parts}
    dataset_hash = canonical_hash(dataset_hashes)
    table = make_table(device)
    student = initialize_student(device)
    calibration_student = initialize_calibration_student(device)
    calibrated = calibration(calibration_student, table, data_parts, cfg, device, args.scope)
    calibrated["calibration_student"] = str(
        (STAGE1 / "checkpoints/single_walk_steady.pt").relative_to(REPO)
    )
    calibrated["calibration_student_sha256"] = sha(STAGE1 / "checkpoints/single_walk_steady.pt")
    del calibration_student
    if args.scope == "walk_only":
        write_json("loss_calibration.json", calibrated)
    training = cfg["training"]
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    checkpoint_dir = OUT / "checkpoints" / args.scope
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_path = checkpoint_dir / "initial.pt"
    torch.save({"student": student.state_dict(), "epoch": 0, "scope": args.scope, "dataset_hash": dataset_hash}, initial_path)
    curves, manifest = [], []
    best_loss, best_epoch, patience = float("inf"), 0, 0

    def evaluate(split: str):
        sums = {key: 0.0 for key in ("action", "action_delta", "dynamic", "contact", "total")}
        count = 0
        student.eval()
        with torch.no_grad():
            for part in data_parts:
                loaded = load_part(part, split, args.scope)
                if loaded is None:
                    continue
                obs, action, prev_obs, prev_action, regime, phase = loaded
                for begin in range(0, len(obs), training["batch_size"]):
                    end = begin + training["batch_size"]
                    x, y = torch.from_numpy(obs[begin:end]).to(device), torch.from_numpy(action[begin:end]).to(device)
                    px, py = torch.from_numpy(prev_obs[begin:end]).to(device), torch.from_numpy(prev_action[begin:end]).to(device)
                    r, p = torch.from_numpy(regime[begin:end]).to(device), torch.from_numpy(phase[begin:end]).to(device)
                    loss, terms = dynamics_sensitive_distillation_loss(
                        student(x), y, x, student(px), py, r, p, table,
                        huber_delta=training["action_huber_delta"],
                        action_delta_weight=training["action_delta_loss_weight"],
                        lambda_dynamic=calibrated["lambda_dynamic"],
                        lambda_contact=calibrated["lambda_contact"],
                    )
                    size = len(x); count += size
                    sums["total"] += float(loss) * size
                    for key in ("action", "action_delta", "dynamic", "contact"):
                        sums[key] += float(terms[key]) * size
        return {"count": count, **{key: value / max(count, 1) for key, value in sums.items()}}

    for epoch in range(1, int(training["epochs"]) + 1):
        student.train()
        shuffled = list(data_parts)
        random.Random(seed + epoch).shuffle(shuffled)
        train_sum = count = 0
        for part in shuffled:
            loaded = load_part(part, "train", args.scope)
            if loaded is None:
                continue
            obs, action, prev_obs, prev_action, regime, phase = loaded
            order = np.random.default_rng(seed + epoch + count).permutation(len(obs))
            for begin in range(0, len(order), training["batch_size"]):
                index = order[begin:begin + training["batch_size"]]
                x, y = torch.from_numpy(obs[index]).to(device), torch.from_numpy(action[index]).to(device)
                px, py = torch.from_numpy(prev_obs[index]).to(device), torch.from_numpy(prev_action[index]).to(device)
                r, p = torch.from_numpy(regime[index]).to(device), torch.from_numpy(phase[index]).to(device)
                loss, _ = dynamics_sensitive_distillation_loss(
                    student(x), y, x, student(px), py, r, p, table,
                    huber_delta=training["action_huber_delta"],
                    action_delta_weight=training["action_delta_loss_weight"],
                    lambda_dynamic=calibrated["lambda_dynamic"],
                    lambda_contact=calibrated["lambda_contact"],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), training["gradient_clip_norm"])
                optimizer.step()
                train_sum += float(loss.detach()) * len(index); count += len(index)
        validation = evaluate("validation")
        row = {
            "epoch": epoch, "train_loss": train_sum / max(count, 1), "validation_loss": validation["total"],
            "validation_action": validation["action"], "validation_action_delta": validation["action_delta"],
            "validation_dynamic": validation["dynamic"], "validation_contact": validation["contact"],
        }
        curves.append(row)
        improved = validation["total"] < best_loss - training["early_stopping_min_delta"]
        if improved:
            best_loss, best_epoch, patience = validation["total"], epoch, 0
            torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "scope": args.scope, "dataset_hash": dataset_hash, "calibration": calibrated},
                       checkpoint_dir / "best_validation.pt")
        else:
            patience += 1
        if epoch in training["checkpoint_epochs"]:
            path = checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "scope": args.scope, "dataset_hash": dataset_hash, "calibration": calibrated}, path)
            manifest.append({"epoch": epoch, "path": str(path.relative_to(REPO)), "sha256": sha(path)})
        print(f"[stage2 {args.scope}] epoch={epoch} train={row['train_loss']:.7f} val={row['validation_loss']:.7f}", flush=True)
        if patience >= training["early_stopping_patience_epochs"] and epoch >= training["early_stopping_minimum_epoch"]:
            break
    final_path = checkpoint_dir / "final.pt"
    torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": curves[-1]["epoch"],
                "scope": args.scope, "dataset_hash": dataset_hash, "calibration": calibrated}, final_path)
    for role, path, epoch in (
        ("initial", initial_path, 0),
        ("best_validation", checkpoint_dir / "best_validation.pt", best_epoch),
        ("final", final_path, curves[-1]["epoch"]),
    ):
        manifest.append({"role": role, "epoch": epoch, "path": str(path.relative_to(REPO)), "sha256": sha(path)})
    with (OUT / f"{args.scope}_training_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader(); writer.writerows(curves)
    write_json(f"{args.scope}_checkpoint_manifest.json", {
        "scope": args.scope, "dataset_sha256": dataset_hash, "checkpoints": manifest,
        "best_validation_epoch": best_epoch, "test": evaluate("test"),
        "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "teacher_gradients": 0, "ppo_training": 0, "reward_optimization": 0,
    })
    print(json.dumps({"scope": args.scope, "best_epoch": best_epoch, "best_loss": best_loss}))


if __name__ == "__main__":
    main()
