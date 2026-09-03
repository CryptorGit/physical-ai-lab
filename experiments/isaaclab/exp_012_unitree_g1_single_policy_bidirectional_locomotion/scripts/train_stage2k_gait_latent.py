"""Diagnostic supervised endpoint representability training for Stage 2K."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"
RAW = OUT / "raw"
DATASET = RAW / "gait_latent_endpoint_dataset.pt"
RUN = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
RUN_SHA = "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9"
SAVE_STEPS = {0, 500, 1000, 2000, 5000, 10000, 20000}
JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_elbow_roll_joint",
    "right_elbow_roll_joint", "left_five_joint", "left_three_joint", "left_zero_joint",
    "right_five_joint", "right_three_joint", "right_zero_joint", "left_six_joint",
    "left_four_joint", "left_one_joint", "right_six_joint", "right_four_joint",
    "right_one_joint", "left_two_joint", "right_two_joint",
]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TeacherActor(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.mlp.load_state_dict(OrderedDict(
            (key.removeprefix("mlp."), value) for key, value in state.items() if key.startswith("mlp.")
        ), strict=True)

    def forward(self, observation):
        return self.mlp(observation)


class GaitLatentActor(nn.Module):
    """124D actor with a separately evaluated gait column for exact zero-column identity."""

    def __init__(self, run_state):
        super().__init__()
        first_weight = run_state["mlp.0.weight"]
        self.first_base_weight = nn.Parameter(first_weight.clone())
        self.first_gait_column = nn.Parameter(torch.zeros(first_weight.shape[0], 1, dtype=first_weight.dtype))
        self.first_bias = nn.Parameter(run_state["mlp.0.bias"].clone())
        self.hidden = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 37),
        )
        mapped = OrderedDict({
            "1.weight": run_state["mlp.2.weight"], "1.bias": run_state["mlp.2.bias"],
            "3.weight": run_state["mlp.4.weight"], "3.bias": run_state["mlp.4.bias"],
            "5.weight": run_state["mlp.6.weight"], "5.bias": run_state["mlp.6.bias"],
        })
        self.hidden.load_state_dict(mapped, strict=True)
        self.std = nn.Parameter(run_state["distribution.std_param"].clone())

    def forward(self, observation, gait_cmd):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        first = first + gait_cmd.reshape(-1, 1) * self.first_gait_column.T
        return self.hidden(first)

    def combined_first_weight(self):
        return torch.cat((self.first_base_weight, self.first_gait_column), dim=1)


def exact_kl(teacher_mean, teacher_std, student_mean, student_std):
    return (
        torch.log(student_std / teacher_std)
        + (teacher_std.square() + (teacher_mean - student_mean).square()) / (2 * student_std.square())
        - .5
    ).sum(-1)


def sample_batch(data, split_by_group, split_name, batch_size, device, generator):
    # Exactly 50% WALK and 50% RUN; RUN portion is as equal as possible over 1.2/2.4/2.6.
    walk_count = batch_size // 2
    run_count = batch_size - walk_count
    counts = [run_count // 3, run_count // 3, run_count - 2 * (run_count // 3)]
    selections = []
    for name, count in zip(("WALK_1P2", "RUN_1P2", "RUN_2P4", "RUN_2P6"), [walk_count, *counts]):
        episodes = torch.tensor(split_by_group[name][split_name], dtype=torch.long)
        episode_ids = episodes[torch.randint(len(episodes), (count,), generator=generator)]
        timesteps = torch.randint(0, 500, (count,), generator=generator)
        selections.append((timesteps, episode_ids))
    timesteps = torch.cat([item[0] for item in selections])
    episodes = torch.cat([item[1] for item in selections])
    permutation = torch.randperm(batch_size, generator=generator)
    timesteps, episodes = timesteps[permutation], episodes[permutation]
    observation = data["observation"][timesteps, episodes].to(device)
    action = data["teacher_action"][timesteps, episodes].to(device)
    gait = data["gait_cmd"][episodes].to(device)
    teacher = data["teacher_id"][episodes]
    teacher_std = data["teacher_std"][teacher].to(device)
    group = data["group_id"][episodes]
    return observation, gait, action, teacher_std, group


def evaluate(model, data, split_by_group, split_name, device, sample_count=40000):
    generator = torch.Generator().manual_seed(20267022 if split_name == "validation" else 20267023)
    model.eval()
    totals = {index: {"mse": [], "cosine": [], "kl": []} for index in range(4)}
    batches = math.ceil(sample_count / 4096)
    with torch.no_grad():
        for _ in range(batches):
            size = min(4096, sample_count)
            observation, gait, teacher_action, teacher_std, group = sample_batch(
                data, split_by_group, split_name, size, device, generator
            )
            student_action = model(observation, gait)
            student_std = model.std.clamp_min(1e-6).expand_as(teacher_std)
            mse = (student_action - teacher_action).square().mean(-1)
            cosine = nn.functional.cosine_similarity(student_action, teacher_action)
            kl = exact_kl(teacher_action, teacher_std, student_action, student_std)
            for group_id in range(4):
                mask = group.to(device) == group_id
                totals[group_id]["mse"].append(mse[mask].cpu())
                totals[group_id]["cosine"].append(cosine[mask].cpu())
                totals[group_id]["kl"].append(kl[mask].cpu())
    result = {}
    group_names = ("WALK_1P2", "RUN_1P2", "RUN_2P4", "RUN_2P6")
    for group_id, name in enumerate(group_names):
        values = {key: torch.cat(items) for key, items in totals[group_id].items()}
        result[name] = {
            "samples": len(values["mse"]), "action_mse": float(values["mse"].mean()),
            "action_cosine": float(values["cosine"].mean()), "gaussian_kl": float(values["kl"].mean()),
            "mse_p95": float(torch.quantile(values["mse"], .95)),
        }
    result["aggregate_loss"] = sum(
        value["action_mse"] + .1 * value["gaussian_kl"] for value in result.values()
    ) / 4
    return result


def save_checkpoint(model, optimizer, step, validation):
    path = RAW / "checkpoints" / f"student_step_{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "validation": validation, "architecture": [124, 256, 128, 128, 37],
        "gait_command": {"WALK": 0.0, "RUN": 1.0},
    }, path)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sha(RUN) != RUN_SHA:
        raise RuntimeError("GAIT_LATENT_RUN_TEACHER_PROVENANCE_FAIL")
    torch.manual_seed(20267021)
    random.seed(20267021)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    split = json.loads((OUT / "gait_latent_dataset_split.json").read_text(encoding="utf-8"))
    split_by_group = split["stratified_groups"]
    run_payload = torch.load(RUN, map_location="cpu", weights_only=False)
    run_state = run_payload["actor_state_dict"]
    teacher = TeacherActor(run_state).to(device).eval()
    model = GaitLatentActor(run_state).to(device)
    # Identity is audited on real endpoint observations and five required command values.
    identity_observation = data["observation"][:8, :64].reshape(-1, 123).to(device)
    identity_rows = []
    with torch.no_grad():
        expected = teacher(identity_observation)
        for gait_value in (0.0, .25, .5, .75, 1.0):
            actual = model(identity_observation, torch.full((len(identity_observation),), gait_value, device=device))
            identity_rows.append({
                "gait_cmd": gait_value, "bitwise_equal": bool(torch.equal(expected, actual)),
                "max_absolute_difference": float((expected - actual).abs().max()),
            })
    identity_pass = all(row["bitwise_equal"] for row in identity_rows)
    dump("student_initialization_contract.json", {
        "architecture": [124, 256, 128, 128, 37], "parent": "RUN teacher model_5244.pt",
        "copied": ["first 123 input columns", "biases", "hidden layers", "output layer", "std"],
        "new_gait_column_initialization": "all zeros", "gait_column_nonzero_count": 0,
        "forward_contract": "F.linear(original_123D, copied_weight, bias) + gait_cmd * gait_column",
    })
    dump("student_initial_identity_audit.json", {
        "samples": len(identity_observation), "commands": identity_rows,
        "actor_mean_bitwise_all": identity_pass, "std_bitwise": bool(torch.equal(model.std.cpu(), run_state["distribution.std_param"])),
        "classification": "PASS" if identity_pass else "GAIT_LATENT_INITIALIZATION_IDENTITY_FAIL",
    })
    if not identity_pass:
        raise RuntimeError("GAIT_LATENT_INITIALIZATION_IDENTITY_FAIL")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = """stage: 2K
training_type: diagnostic_supervised_endpoint_preflight
seed: 20267021
architecture: [124, 256, 128, 128, 37]
batch_size: 2048
maximum_optimizer_steps: 20000
learning_rate: 0.0003
optimizer: Adam
action_loss: mean_squared_error
gaussian_kl_coefficient: 0.1
validation_interval: 500
early_stopping_patience_evaluations: 5
pre_registered_action_mse_threshold: 0.001
action_cosine_threshold: 0.98
gaussian_kl_threshold: 0.05
student_std: trainable_from_RUN_teacher_initialization
critic_training: false
ppo_training: false
"""
    (OUT / "resolved_supervised_training_config.yaml").write_text(config, encoding="utf-8")
    generator = torch.Generator().manual_seed(20267021)
    curves, checkpoints = [], []
    best_loss, best_path, no_improvement = float("inf"), None, 0
    initial_validation = evaluate(model, data, split_by_group, "validation", device, sample_count=20000)
    initial_path = save_checkpoint(model, optimizer, 0, initial_validation)
    checkpoints.append((0, initial_path, initial_validation))
    model.train()
    for step in range(1, 20001):
        observation, gait, teacher_action, teacher_std, _ = sample_batch(
            data, split_by_group, "train", 2048, device, generator
        )
        optimizer.zero_grad(set_to_none=True)
        student_action = model(observation, gait)
        student_std = model.std.clamp_min(1e-6).expand_as(teacher_std)
        action_loss = nn.functional.mse_loss(student_action, teacher_action)
        kl_loss = exact_kl(teacher_action, teacher_std, student_action, student_std).mean()
        loss = action_loss + .1 * kl_loss
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % 500 == 0:
            validation = evaluate(model, data, split_by_group, "validation", device, sample_count=20000)
            curves.append({
                "step": step, "train_action_mse": float(action_loss), "train_gaussian_kl": float(kl_loss),
                "train_total_loss": float(loss), "gradient_norm": float(gradient_norm),
                "validation_loss": validation["aggregate_loss"],
                "gait_column_norm": float(model.first_gait_column.norm()),
                "std_mean": float(model.std.mean()),
            })
            if step in SAVE_STEPS:
                path = save_checkpoint(model, optimizer, step, validation)
                checkpoints.append((step, path, validation))
            if validation["aggregate_loss"] < best_loss - 1e-8:
                best_loss = validation["aggregate_loss"]
                best_path = RAW / "checkpoints" / f"student_best_step_{step}.pt"
                torch.save({
                    "step": step, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "validation": validation,
                    "architecture": [124, 256, 128, 128, 37],
                }, best_path)
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= 5:
                break
    with (OUT / "training_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    if best_path is None:
        best_path = initial_path
    selected_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected_payload["model_state_dict"], strict=True)
    held_out = evaluate(model, data, split_by_group, "held_out", device, sample_count=40000)
    endpoint_rows = {}
    static_pass = True
    for name in ("WALK_1P2", "RUN_1P2", "RUN_2P4", "RUN_2P6"):
        value = held_out[name]
        passed = value["action_mse"] <= .001 and value["action_cosine"] >= .98 and value["gaussian_kl"] <= .05
        endpoint_rows[name] = {**value, "pass": passed}
        static_pass &= passed
    endpoint_rows["aggregate_classification"] = (
        "GAIT_LATENT_ENDPOINT_STATIC_PASS" if static_pass else "GAIT_LATENT_ENDPOINT_STATIC_FAIL"
    )
    dump("static_endpoint_evaluation.json", endpoint_rows)
    selected_dir = OUT / "student"
    selected_dir.mkdir(exist_ok=True)
    selected_path = selected_dir / "selected_gait_latent_student.pt"
    torch.save({
        "step": selected_payload["step"], "model_state_dict": model.state_dict(),
        "architecture": [124, 256, 128, 128, 37], "held_out": endpoint_rows,
        "teacher_runtime_required": False,
    }, selected_path)
    checkpoint_rows = []
    for checkpoint_step, path, validation in checkpoints:
        checkpoint_rows.append({
            "step": checkpoint_step, "path": str(path.relative_to(REPO)), "sha256": sha(path),
            "validation_loss": validation["aggregate_loss"], "git_tracked": False,
        })
    dump("checkpoint_manifest.json", {
        "training_checkpoints": checkpoint_rows, "selected_path": str(selected_path.relative_to(REPO)),
        "selected_sha256": sha(selected_path), "selection_rule": "minimum validation endpoint loss",
        "latest_automatically_selected": False,
    })
    dump("selected_student_checkpoint.json", {
        "path": str(selected_path.relative_to(REPO)), "sha256": sha(selected_path),
        "step": selected_payload["step"], "validation_loss": selected_payload["validation"]["aggregate_loss"],
        "held_out_static_classification": endpoint_rows["aggregate_classification"],
    })


if __name__ == "__main__":
    main()
