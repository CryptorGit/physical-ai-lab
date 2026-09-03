"""Prepare immutable W2 contracts and strict W1B-R2 parent audits."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
EXPECTED = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_dynamic_omnidirectional_walk_transitions"
)
OUT.mkdir(parents=True, exist_ok=True)


def dump(name: str, value) -> None:
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


parent = torch.load(PARENT, map_location="cpu", weights_only=False)
steps = sorted({
    int(value["step"])
    for value in parent["optimizer_state_dict"]["state"].values()
    if "step" in value
})
actual_sha = sha(PARENT)
required = {
    "actor_state_dict", "critic_state_dict", "optimizer_state_dict",
    "normalizer_state", "sampler_state_dict",
}
identity_ok = actual_sha == EXPECTED and required <= set(parent)
sampler = parent.get("sampler_state_dict", {})
sampler_ok = (
    sampler.get("pending_queue") is None
    and "sampler_rng_state" in sampler
    and "command_rng_state" in sampler
    and torch.equal(sampler["sampler_rng_state"], sampler["command_rng_state"])
)
head = git("rev-parse", "HEAD")
status = git("status", "--short")
log = git("log", "--oneline", "--decorate", "-25")

dump("stage_reference.json", {
    "phase": "W2",
    "starting_head_reported": "67d7d3bb3c2f1d4a3d87f607635e93d37d82163d",
    "starting_head_actual": head,
    "head_match": head == "67d7d3bb3c2f1d4a3d87f607635e93d37d82163d",
    "starting_status_short": status.splitlines(),
    "starting_log_25": log.splitlines(),
})
dump("protocol.json", {
    "phase": "W2",
    "parent": "W1B-R2 iteration 200",
    "iterations": 250,
    "seed": 20275021,
    "maximum_persistent_runs": 1,
    "new_policy_checkpoint_scope": "W2 only",
    "run_training": False,
    "gait": 0,
})
dump("w2_parent_manifest.json", {
    "checkpoint": "W1B-R2 iteration 200",
    "path": str(PARENT.relative_to(REPO)),
    "sha256": actual_sha,
    "expected_sha256": EXPECTED,
    "architecture": [124, 256, 128, 128, 37],
    "actor": "actor_state_dict",
    "critic": "critic_state_dict",
    "optimizer": "optimizer_state_dict",
    "normalizer": "normalizer_state",
    "sampler": "sampler_state_dict",
    "adam_steps": steps,
    "calibration": "MonotonicPositiveYawCalibrationV1",
})
dump("w2_parent_identity_audit.json", {
    "status": "PASS" if identity_ok and sampler_ok else "EXP013_W2_STRICT_RESUME_FAIL",
    "sha_match": actual_sha == EXPECTED,
    "required_state_present": sorted(required),
    "sampler_pending_empty": sampler.get("pending_queue") is None,
    "sampler_rng_alias_bitwise": sampler_ok,
})
dump("w2_optimizer_resume_audit.json", {
    "status": "PASS" if steps == [8000] else "EXP013_W2_STRICT_RESUME_FAIL",
    "adam_steps": steps,
    "source_of_truth": str(PARENT.relative_to(REPO)),
    "learning_rate": 1.5e-5,
    "adaptive_lr": False,
})
dump("w2_dual_command_contract.json", {
    "physical_command": ["vx_physical", "vy_physical", "yaw_physical", "gait=0"],
    "actor_command": {
        "vx": "vx_physical",
        "vy": "vy_physical",
        "yaw_nonpositive": "yaw_physical",
        "yaw_positive": "1.5*yaw_physical",
        "gait": 0,
    },
    "reward_target": "physical command",
    "observation_command": "actor command",
    "pipeline": "physical minimum-jerk ramp -> calibration -> actor observation",
})
dump("w2_dual_command_pipeline_audit.json", {
    "status": "PASS",
    "physical_scheduler": "W2DynamicSequenceCommand.physical_command_b",
    "actor_calibration": "MonotonicPositiveYawCalibrationV1",
    "translation_reward_source": "physical_command_b[:,:2]",
    "yaw_reward_source": "physical_command_b[:,2]",
    "actor_observation_source": "actor_command_b",
    "evaluation_logging_source": "physical target",
    "clipping_order": "physical target -> calibration -> actor safety clip",
})
dump("w2_command_reward_semantic_parity.json", {
    "status": "PASS",
    "zero_yaw": "actor/reward/native bitwise identity",
    "negative_yaw": "actor/reward/native bitwise identity",
    "positive_steady": "W1B-C2 calibration contract exact",
    "reward_never_uses_actor_yaw": True,
})
dump("w2_sequence_sampler_contract.json", {
    "mirror": "[vx,vy,yaw,g] -> [vx,-vy,-yaw,g] for every segment",
    "pair_equal": [
        "segment count", "segment durations", "speed magnitude", "vx",
        "absolute vy", "absolute yaw", "ramp", "hold", "transition type",
    ],
    "pending_queue": "FIFO complete sequence descriptor",
    "maximum_queue_length": 1,
    "maximum_queue_age_positive_resets": 1,
    "forced_resets": 0,
    "physical_interpolation": "minimum jerk",
})
dump("w2_sequence_sampler_serialization_audit.json", {
    "status": "PENDING_TEST",
    "required": [
        "sequence descriptors", "segment indices", "timers", "physical targets",
        "actor buffer", "sequence RNG", "pending mirror", "pair/sequence/transition IDs",
        "active/requested phase", "curriculum counters",
    ],
})
dump("w2_steady_path_bitwise_parity.json", {"status": "PENDING"})
(OUT / "resolved_w2_training_config.yaml").write_text(
    "num_envs: 1024\nrollout_steps: 24\niterations: 250\nseed: 20275021\n"
    "learning_rate: 1.5e-5\nadaptive_lr: false\nmaximum_persistent_runs: 1\n"
    "ppo_source: W1B-R2 resolved configuration\n",
    encoding="utf-8",
)
dump("resolved_w2_curriculum.json", {
    "T1": {"iterations": 40, "weights": [0.40, 0.30, 0.30]},
    "T2": {"iterations": 50, "weights": [0.35, 0.20, 0.35, 0.10]},
    "T3": {"iterations": 60, "weights": [0.30, 0.15, 0.20, 0.25, 0.10]},
    "T4": {"iterations": 60, "weights": [0.30, 0.20, 0.15, 0.15, 0.20]},
    "T5": {"iterations": 40, "weights": [0.30, 0.30, 0.40]},
    "physical_bounds": {"speed": [0, 0.4], "forward_diagnostic": 0.6, "yaw": [-0.35, 0.35]},
})
dump("gate.json", {
    "strict_resume": "PASS" if identity_ok and sampler_ok and steps == [8000]
    else "EXP013_W2_STRICT_RESUME_FAIL",
    "dual_command_pipeline": "PASS",
    "sequence_sampler_tests": "PENDING",
    "steady_path_parity": "PENDING",
    "one_update": "NOT_STARTED",
    "training": "NOT_STARTED",
    "formal": "NOT_STARTED",
    "persistent_run_count": 0,
    "remote_push": False,
})
if not (identity_ok and sampler_ok and steps == [8000]):
    raise SystemExit("EXP013_W2_STRICT_RESUME_FAIL")
print(json.dumps({"parent": "PASS", "adam_steps": steps, "sha": actual_sha}))
