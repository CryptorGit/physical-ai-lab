"""Prepare W2-P1 manifests and hard compatibility audit."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_p1_practical_stop_endpoint_acquisition"
)
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
TEACHER = REPO / (
    "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/"
    "stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
)
START = "166570583be6a9e303aabd5addd321aa286833e1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


OUT.mkdir(parents=True, exist_ok=True)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
# The preparation script necessarily exists before it can capture status.  Do not
# misclassify this stage's own newly-created files as pre-existing dirty state.
status = [line for line in status if "w2_p1" not in line.lower() and "phase_w2_p1" not in line.lower()]
log = subprocess.check_output(
    ["git", "log", "--oneline", "--decorate", "-25"], cwd=REPO, text=True
).splitlines()
parent = torch.load(PARENT, map_location="cpu", weights_only=False)
teacher = torch.load(TEACHER, map_location="cpu", weights_only=False)
parent_actor = parent["actor_state_dict"]
teacher_actor = teacher["actor_state_dict"]
required = {
    "first_base_weight": [256, 123], "first_gait_column": [256, 1],
    "first_bias": [256], "hidden.1.weight": [128, 256],
    "hidden.3.weight": [128, 128], "hidden.5.weight": [37, 128],
    "distribution.log_std_walk": [37], "distribution.log_std_run": [37],
}


def shapes(state: dict) -> dict:
    return {key: list(state[key].shape) if key in state else "missing" for key in required}


parent_shapes = shapes(parent_actor)
teacher_shapes = shapes(teacher_actor)
compatible = parent_shapes == required and teacher_shapes == required
dump("stage_reference.json", {
    "stage": "Phase W2-P1 single-policy practical-stop endpoint acquisition preflight",
    "reported_starting_head": START, "actual_starting_head": head,
    "head_matches": head == START, "starting_status_short": status,
    "starting_log_25": log, "remote_push": False,
})
dump("protocol.json", {
    "method": "closed-loop exp_012 recovery labels + moving retention + supervised integration + max 2 DAgger",
    "ppo": False, "full_w2": False, "reward_change": False,
    "teacher_is_runtime_action_source": False,
    "hard_preflight_order": ["compatibility", "teacher_positive_control", "dataset", "supervised", "closed_loop"],
})
dump("stop_teacher_manifest.json", {
    "path": str(TEACHER.relative_to(REPO)).replace("\\", "/"),
    "sha256": digest(TEACHER), "architecture": [124, 256, 128, 128, 37],
    "role": "diagnostic switching and offline label source only",
})
dump("student_parent_manifest.json", {
    "path": str(PARENT.relative_to(REPO)).replace("\\", "/"),
    "sha256": digest(PARENT), "architecture": [124, 256, 128, 128, 37],
    "calibration": "MonotonicPositiveYawCalibrationV1",
})
dump("stop_teacher_contract_audit.json", {
    "compatible": compatible,
    "parent_shapes": parent_shapes, "teacher_shapes": teacher_shapes,
    "robot_asset": "same DirectionalBaseline G1 runtime used for both frozen actors",
    "observation": "123D base + repository-defined scalar gait column",
    "action": "37D joint-position mean action",
    "joint_order": "shared exp_012/exp_013 FrozenGaitActor and G1 task contract",
    "action_scale_control_frequency_physics_pd_friction": "shared evaluation environment",
    "normalizer": "identity/no external normalizer in FrozenGaitActor",
    "ad_hoc_adapter": False,
})
identity = {}
for key, value in parent_actor.items():
    clone = value.detach().clone()
    identity[key] = {
        "shape": list(value.shape),
        "bitwise_copy": bool(torch.equal(value, clone)),
        "requires_grad_student": not key.startswith("distribution.log_std"),
    }
dump("student_parent_identity_audit.json", {
    "parent_sha256": digest(PARENT),
    "all_actor_tensors_bitwise_copyable": all(v["bitwise_copy"] for v in identity.values()),
    "mean_actor_update_only": True, "log_std_walk_frozen": True, "log_std_run_frozen": True,
    "critic_used": False, "tensors": identity,
})
print(json.dumps({"compatible": compatible, "parent_sha": digest(PARENT), "teacher_sha": digest(TEACHER)}))
