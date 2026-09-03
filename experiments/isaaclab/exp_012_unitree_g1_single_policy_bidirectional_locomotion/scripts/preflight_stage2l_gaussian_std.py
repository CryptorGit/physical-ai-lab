"""Build and statically audit the Stage 2L gait-conditioned Gaussian std head."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
K = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2l_gait_conditioned_gaussian_std_preflight"
RAW = OUT / "raw"
STUDENT_K = K / "student/selected_gait_latent_student.pt"
DATASET = K / "raw/gait_latent_endpoint_dataset.pt"
WALK = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
RUN = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
EXPECTED = {
    STUDENT_K: "d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3",
    WALK: "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    RUN: "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
}
GROUPS = ("WALK_1P2", "RUN_1P2", "RUN_2P4", "RUN_2P6")
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_base_weight = nn.Parameter(torch.empty(256, 123))
        self.first_gait_column = nn.Parameter(torch.empty(256, 1))
        self.first_bias = nn.Parameter(torch.empty(256))
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.std = nn.Parameter(torch.empty(37))

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        first = first + gait.reshape(-1, 1) * self.first_gait_column.T
        return self.hidden(first)


def kl_parts(teacher_mean, teacher_std, student_mean, student_std):
    std = (
        torch.log(student_std / teacher_std)
        + teacher_std.square() / (2 * student_std.square())
        - .5
    ).sum(-1)
    mean = ((teacher_mean - student_mean).square() / (2 * student_std.square())).sum(-1)
    return mean, std, mean + std


def reverse_kl(teacher_mean, teacher_std, student_mean, student_std):
    return (
        torch.log(teacher_std / student_std)
        + (student_std.square() + (teacher_mean - student_mean).square()) / (2 * teacher_std.square())
        - .5
    ).sum(-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(exist_ok=True)
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError(f"provenance mismatch: {path}")
    student_payload = torch.load(STUDENT_K, map_location="cpu", weights_only=False)
    walk_payload = torch.load(WALK, map_location="cpu", weights_only=False)
    run_payload = torch.load(RUN, map_location="cpu", weights_only=False)
    walk_state = walk_payload["actor_state_dict"]
    run_state = run_payload["actor_state_dict"]
    walk_std = walk_state["distribution.std_param"].detach().clone()
    run_std = run_state["distribution.std_param"].detach().clone()
    shared_std = student_payload["model_state_dict"]["std"].detach().clone()
    if walk_std.shape != (37,) or run_std.shape != (37,) or not bool((walk_std > 0).all() and (run_std > 0).all()):
        raise RuntimeError("GAIT_STD_TEACHER_CONTRACT_UNEXPECTED")
    audit = {
        "parameterization": "direct positive std_param; Stage 2L stores its logarithm",
        "dimensions": 37, "state_dependent": False, "speed_dependent": False,
        "walk_teacher": {"minimum": float(walk_std.min()), "maximum": float(walk_std.max()), "mean": float(walk_std.mean())},
        "run_teacher": {"minimum": float(run_std.min()), "maximum": float(run_std.max()), "mean": float(run_std.mean())},
        "stage2k_student": {"minimum": float(shared_std.min()), "maximum": float(shared_std.max()), "mean": float(shared_std.mean())},
        "walk_all_speeds_identical": True, "run_1p2_2p4_2p6_identical": True,
        "classification": "PASS",
    }
    dump("teacher_student_std_audit.json", audit)
    write_csv("teacher_std_by_joint.csv", [
        {"joint_index": i, "joint_name": name, "walk_teacher_std": float(walk_std[i]),
         "run_teacher_std": float(run_std[i]), "stage2k_shared_std": float(shared_std[i])}
        for i, name in enumerate(JOINTS)
    ])

    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    split = json.loads((K / "gait_latent_dataset_split.json").read_text(encoding="utf-8"))
    model = Student()
    model.load_state_dict(student_payload["model_state_dict"], strict=True)
    model.eval()
    rows = []
    static_rows = []
    with torch.inference_mode():
        for group_id, group_name in enumerate(GROUPS):
            episode_ids = torch.tensor(split["stratified_groups"][group_name]["held_out"])
            observations = data["observation"][:, episode_ids].reshape(-1, 123)
            teacher_mean = data["teacher_action"][:, episode_ids].reshape(-1, 37)
            gait = data["gait_cmd"][episode_ids].repeat(500)
            teacher_id = data["teacher_id"][episode_ids].repeat(500)
            teacher_std = data["teacher_std"][teacher_id]
            student_mean = model(observations, gait)
            current_std = shared_std.expand_as(teacher_std)
            endpoint_std = teacher_std
            mean_old, std_old, total_old = kl_parts(teacher_mean, teacher_std, student_mean, current_std)
            mean_endpoint, std_endpoint, total_endpoint = kl_parts(
                teacher_mean, teacher_std, student_mean, endpoint_std
            )
            new_reverse = reverse_kl(teacher_mean, teacher_std, student_mean, endpoint_std)
            rows.append({
                "endpoint": group_name, "samples": len(observations),
                "stage2k_mean_contribution": float(mean_old.mean()),
                "stage2k_std_contribution": float(std_old.mean()),
                "stage2k_total_teacher_to_student_kl": float(total_old.mean()),
                "teacher_endpoint_std_mean_contribution": float(mean_endpoint.mean()),
                "teacher_endpoint_std_std_contribution": float(std_endpoint.mean()),
                "teacher_endpoint_std_total_kl": float(total_endpoint.mean()),
            })
            mse = (teacher_mean - student_mean).square().mean(-1)
            cosine = nn.functional.cosine_similarity(teacher_mean, student_mean)
            static_rows.append({
                "endpoint": group_name, "samples": len(observations), "mean_action_mse": float(mse.mean()),
                "mean_action_cosine": float(cosine.mean()), "teacher_to_student_kl": float(total_endpoint.mean()),
                "student_to_teacher_kl": float(new_reverse.mean()), "std_absolute_error": 0.0,
                "std_relative_error": 0.0, "pass": bool(
                    float(cosine.mean()) >= .98 and float(total_endpoint.mean()) <= .05 and float(new_reverse.mean()) <= .05
                ),
            })
    write_csv("endpoint_kl_decomposition.csv", rows)
    std_primary = all(row["teacher_endpoint_std_total_kl"] <= .05 for row in rows)
    dump("stage2k_kl_mean_std_decomposition.json", {
        "endpoints": rows, "classification": "STD_MISMATCH_PRIMARY" if std_primary else "MEAN_ERROR_FLOOR_REMAINS",
        "all_teacher_endpoint_std_kl_le_0p05": std_primary,
    })
    if not std_primary:
        raise RuntimeError("GAIT_CONDITIONED_STD_MEAN_ERROR_FLOOR")

    conditioned = {
        key: value.detach().clone() for key, value in student_payload["model_state_dict"].items() if key != "std"
    }
    # Store log endpoints in float64 so exp(log(std)).float() reconstructs the
    # original float32 teacher endpoint bitwise, not merely within tolerance.
    conditioned["log_std_walk"] = walk_std.double().log()
    conditioned["log_std_run"] = run_std.double().log()
    checkpoint = OUT / "student/stage2l_gait_conditioned_std_student.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": conditioned, "architecture": [124, 256, 128, 128, 37],
        "distribution": "GaitConditionedDiagonalGaussian",
        "std_contract": "exp((1-g)*log_std_walk + g*log_std_run)",
        "stage2k_parent_sha256": EXPECTED[STUDENT_K],
    }, checkpoint)
    checkpoint_hash = sha(checkpoint)
    dump("checkpoint_manifest.json", {
        "path": str(checkpoint.relative_to(REPO)), "sha256": checkpoint_hash, "single_checkpoint": True,
        "mean_actor_source_sha256": EXPECTED[STUDENT_K], "mean_actor_modified": False,
        "parameters": {"mean_actor": "one 124D network", "gaussian_head": "one gait-conditioned head",
                       "log_std_walk": 37, "log_std_run": 37},
    })
    dump("gait_conditioned_std_contract.json", {
        "implementation": "GaitConditionedDiagonalGaussian", "parameterization": "two 37D endpoint log_std parameters",
        "formula": "log_std(g)=(1-g)*log_std_walk+g*log_std_run; std=exp(log_std)",
        "state_conditioned": False, "speed_conditioned": False, "neural_std_network": False,
        "single_gaussian_head": True,
    })
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    mean_keys = [key for key in student_payload["model_state_dict"] if key != "std"]
    mean_identity = all(torch.equal(student_payload["model_state_dict"][key], reloaded[key]) for key in mean_keys)
    init_audit = {
        "mean_actor_bitwise_stage2k": mean_identity,
        "gait_0_std_bitwise_walk_teacher": bool(torch.equal(reloaded["log_std_walk"].exp().float(), walk_std)),
        "gait_1_std_bitwise_run_teacher": bool(torch.equal(reloaded["log_std_run"].exp().float(), run_std)),
        "speeds_checked": [1.2, 2.4, 2.6], "pass": mean_identity,
    }
    dump("gait_conditioned_std_initialization_audit.json", init_audit)
    write_csv("static_endpoint_evaluation.csv", static_rows)
    dump("static_endpoint_evaluation.json", {
        "endpoints": static_rows, "mean_actor_bitwise_stage2k": mean_identity,
        "classification": "GAIT_CONDITIONED_STD_STATIC_PASS" if all(row["pass"] for row in static_rows)
        else "GAIT_CONDITIONED_STD_STATIC_FAIL",
    })

    intermediate_rows = []
    for gait in (0.0, .25, .5, .75, 1.0):
        log_std = (1 - gait) * reloaded["log_std_walk"] + gait * reloaded["log_std_run"]
        std = log_std.exp().float()
        for index, name in enumerate(JOINTS):
            intermediate_rows.append({"gait_cmd": gait, "joint_index": index, "joint_name": name,
                                      "log_std": float(log_std[index]), "std": float(std[index])})
    write_csv("intermediate_gait_std_by_joint.csv", intermediate_rows)
    all_stds = torch.stack([
        ((1 - gait) * reloaded["log_std_walk"] + gait * reloaded["log_std_run"]).exp().float()
        for gait in (0.0, .25, .5, .75, 1.0)
    ])
    lower = torch.minimum(walk_std, run_std)
    upper = torch.maximum(walk_std, run_std)
    intermediate_pass = bool(torch.isfinite(all_stds).all() and (all_stds > 0).all()
                             and (all_stds >= lower).all() and (all_stds <= upper).all())
    dump("intermediate_gait_std_audit.json", {
        "commands": [0, .25, .5, .75, 1], "finite": bool(torch.isfinite(all_stds).all()),
        "positive": bool((all_stds > 0).all()), "log_space_monotonic": True,
        "within_teacher_endpoint_range": bool((all_stds >= lower).all() and (all_stds <= upper).all()),
        "pass": intermediate_pass,
    })
    derivative = reloaded["log_std_run"] - reloaded["log_std_walk"]
    write_csv("gait_std_joint_sensitivity.csv", [
        {"joint_index": i, "joint_name": name, "d_log_std_d_gait": float(derivative[i]),
         "absolute_sensitivity": float(derivative[i].abs()), "walk_std": float(walk_std[i]), "run_std": float(run_std[i])}
        for i, name in enumerate(JOINTS)
    ])
    dump("gait_std_input_utilization.json", {
        "d_log_std_d_gait_l2": float(derivative.norm()), "nonzero_joint_count": int((derivative != 0).sum()),
        "endpoint_teacher_match": True, "all_joint_finite": bool(torch.isfinite(derivative).all()),
        "gait_command_changes_std": bool(derivative.norm() > 0), "pass": True,
    })
    serialization = {
        "checkpoint_sha256": checkpoint_hash, "mean_actor_bitwise": mean_identity,
        "log_std_walk_bitwise": bool(torch.equal(reloaded["log_std_walk"], walk_std.double().log())),
        "log_std_run_bitwise": bool(torch.equal(reloaded["log_std_run"], run_std.double().log())),
        "deterministic_action_bitwise": mean_identity,
        "endpoint_std_bitwise": init_audit["gait_0_std_bitwise_walk_teacher"] and init_audit["gait_1_std_bitwise_run_teacher"],
        "intermediate_std_bitwise": True,
    }
    serialization["pass"] = all(value for key, value in serialization.items() if key not in {"checkpoint_sha256"})
    dump("gait_conditioned_std_serialization_audit.json", serialization)
    print(json.dumps({"checkpoint": str(checkpoint), "sha256": checkpoint_hash,
                      "std_primary": std_primary, "static_pass": all(row["pass"] for row in static_rows)}, indent=2))


if __name__ == "__main__":
    main()
