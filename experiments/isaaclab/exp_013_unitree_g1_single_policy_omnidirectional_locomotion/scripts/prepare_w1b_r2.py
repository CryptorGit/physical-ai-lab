"""Prepare W1B-R2 immutable contracts and strict parent audits."""
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
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
R1 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r1_evaluation_parity_corrected_rerun"
)
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
)
EXPECTED = "bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244"
START = "65f6dcf49968c1567abeac0b1b1d6326efec7e4b"
OUT.mkdir(parents=True, exist_ok=True)


def dump(name: str, value) -> None:
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True, encoding="utf-8"
    ).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


head = git("rev-parse", "HEAD")
status = git("status", "--short")
log = git("log", "--oneline", "--decorate", "-25")
dump("stage_reference.json", {
    "phase": "W1B-R2",
    "starting_head_reported": START,
    "starting_head_actual": head,
    "head_match": head == START,
    "starting_status_short": status.splitlines(),
    "starting_log_25": log.splitlines(),
})
dump("protocol.json", {
    "repair": "DETERMINISTIC_PENDING_MIRROR_QUEUE",
    "maximum_persistent_runs": 1,
    "iterations": 200,
    "seed": 20274021,
    "parent": "W1A2 iteration 80",
    "gate_before_training": True,
    "reward_curriculum_ppo_formal_gate_unchanged": True,
    "resume_from_w1b_r1_iteration10": False,
})
parent = torch.load(PARENT, map_location="cpu", weights_only=False)
steps = sorted({
    int(value["step"])
    for value in parent["optimizer_state_dict"]["state"].values()
    if "step" in value
})
manifest = {
    "checkpoint": "W1A2 iteration 80",
    "path": str(PARENT.relative_to(REPO)),
    "sha256": sha(PARENT),
    "expected_sha256": EXPECTED,
    "architecture": [124, 256, 128, 128, 37],
    "command_indices": {"vx": 9, "vy": 10, "yaw_rate": 11, "gait": 123},
    "command_frame": "robot body",
    "command_scale": 1.0,
    "actor": "actor_state_dict",
    "critic": "critic_state_dict",
    "optimizer": "optimizer_state_dict",
    "normalizer": "Identity",
    "adam_steps": steps,
}
dump("w1b_r2_parent_manifest.json", manifest)
identity_ok = (
    manifest["sha256"] == EXPECTED
    and all(key in parent for key in (
        "actor_state_dict", "critic_state_dict", "optimizer_state_dict"
    ))
)
dump("w1b_r2_parent_identity_audit.json", {
    "status": "PASS" if identity_ok else "EXP013_W1B_R2_STRICT_RESUME_FAIL",
    "sha_match": manifest["sha256"] == EXPECTED,
    "actor_present": "actor_state_dict" in parent,
    "critic_present": "critic_state_dict" in parent,
    "optimizer_present": "optimizer_state_dict" in parent,
    "normalizer_identity": True,
})
dump("w1b_r2_optimizer_resume_audit.json", {
    "status": "PASS" if steps == [4000] else "EXP013_W1B_R2_STRICT_RESUME_FAIL",
    "adam_steps": steps,
    "expected": [4000],
    "bitwise_source": str(PARENT.relative_to(REPO)),
    "learning_rate": 1.5e-5,
    "adaptive_lr": False,
})
dump("pending_mirror_queue_contract.json", {
    "classification": "DETERMINISTIC_PENDING_MIRROR_QUEUE",
    "queue": "FIFO",
    "maximum_length": 1,
    "maximum_age_positive_reset_events": 1,
    "input_order_preserved": True,
    "forced_reset": 0,
    "missing_assignment": 0,
    "duplicate_assignment": 0,
    "mirror": "[vx,vy,yaw,g] -> [vx,-vy,-yaw,g]",
    "same_call_pairing_required": False,
    "guarantee_window": "current and next positive partial-reset event",
    "odd_algorithm": "K+1 bases, K mirrors, queue final exact mirror",
    "pending_consumption": "first slot of next positive reset event",
    "fail_closed_queue_length_above_one": True,
})
dump("sampler_phase_transition_contract.json", {
    "empty_queue": "activate requested phase immediately",
    "pending_queue": (
        "consume old-phase pending mirror in first slot, then activate requested "
        "phase for remaining slots"
    ),
    "discard_pending": False,
    "reclassify_pending": False,
    "self_mirror_filler": False,
    "forced_reset": False,
    "serialized": [
        "active phase", "requested phase", "transition pending flag",
    ],
})
state_keys = [
    "pending queue", "sampler RNG", "command RNG", "next pair ID",
    "reset event counter", "active phase", "requested phase",
    "phase-transition state", "current command buffer", "curriculum counters",
]
dump("pending_queue_serialization_contract.json", {
    "checkpoint_required": [
        "actor", "critic", "optimizer", "normalizer", *state_keys,
    ],
    "legacy_parent_migration": {
        "allowed_only_for_new_w1b_r2_run": True,
        "pending_queue": "empty",
        "pair_id": 0,
        "reset_event_counter": 0,
        "active_phase": "Y1_FORWARD_MOVING_TURNS",
        "requested_phase": "Y1_FORWARD_MOVING_TURNS",
        "sampler_rng": "initialized from W1B seed contract",
    },
    "w1b_r2_resume_missing_state": "fail-closed",
    "fresh_process_next_assignment_bitwise": True,
})
(OUT / "resolved_w1b_r2_training_config.yaml").write_text(
    (R1 / "resolved_w1b_r1_training_config.yaml").read_text(encoding="utf-8"),
    encoding="utf-8",
)
(OUT / "resolved_w1b_r2_curriculum.json").write_text(
    (R1 / "resolved_w1b_r1_curriculum.json").read_text(encoding="utf-8"),
    encoding="utf-8",
)
dump("gate.json", {
    "parent_strict_restore": (
        "PASS" if identity_ok and steps == [4000]
        else "EXP013_W1B_R2_STRICT_RESUME_FAIL"
    ),
    "sampler_boundary": "PENDING",
    "even_path_parity": "PENDING",
    "odd_path_determinism": "PENDING",
    "distribution": "PENDING",
    "serialization": "PENDING",
    "evaluation_parity": "PENDING",
    "training_prefix_parity": "PENDING",
    "training": "NOT_STARTED",
    "formal_evaluation": "NOT_STARTED",
    "persistent_run_count": 0,
    "remote_push": False,
})
print(head, identity_ok, steps)
