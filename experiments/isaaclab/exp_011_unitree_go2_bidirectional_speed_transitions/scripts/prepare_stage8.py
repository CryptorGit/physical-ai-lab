"""Freeze Stage 8 heading-diagnosis provenance and observation contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage8_low_speed_heading_diagnosis"
START = "b99e980abb9b74f93c85779748719867c88a7e2a"
SELECTED = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"
STAGE4 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training/checkpoints/model_50.pt"
OFFICIAL = REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
EXPECTED = {
    "stage7_selected": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
    "stage4_selected": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
    "official_parent": "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
}
PROTOCOL_SHA = "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual = {key: sha(path) for key, path in {
        "stage7_selected": SELECTED, "stage4_selected": STAGE4, "official_parent": OFFICIAL
    }.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"CHECKPOINT_PROVENANCE_FAIL: {actual}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != START:
        raise RuntimeError(f"STARTING_HEAD_MISMATCH: {head}")
    status_now = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
    stage8_paths = (
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/prepare_stage8.py",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/evaluate_stage8_heading.py",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/finalize_stage8.py",
        "research/exp_011_go2_low_speed_heading_diagnosis_report.md",
        "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage8_low_speed_heading_diagnosis/",
    )
    status = [
        row for row in status_now
        if not any(row[3:].replace("\\", "/").startswith(path) for path in stage8_paths)
        and row[3:].replace("\\", "/") not in (
            "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
            "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
        )
    ]
    dump("starting_repository_state.json", {
        "starting_head": head, "starting_status": status,
        "unrelated_dirty_paths": [row[3:] for row in status],
    })
    dump("stage7_reference.json", {
        "classification": "GO2_LOW_SPEED_GAIT_STABILIZED_PARTIAL",
        "selected_iteration": 50, "selected_checkpoint": str(SELECTED.resolve()),
        "selected_sha256": actual["stage7_selected"],
        "stage4_checkpoint": str(STAGE4.resolve()), "stage4_sha256": actual["stage4_selected"],
        "official_checkpoint": str(OFFICIAL.resolve()), "official_sha256": actual["official_parent"],
        "protected_low_speed_fall_rates": {"0.2": 0.02, "0.3": 0.0, "0.4": 0.02, "0.5": 0.0, "0.6": 0.0},
        "anchor_retention": "PASS", "anchor_sequence_completion": 1.0,
    })
    dump("protocol.json", {
        "stage": 8, "diagnostic_target": "LOW_SPEED_HEADING_FAILURE",
        "seed_root": 20267901, "deterministic_policy": True, "ppo_updates": 0,
        "reward_optimization": 0, "evaluation_protocol": "GO2_ENDPOINT_EVALUATION_V1",
        "evaluation_protocol_sha256": PROTOCOL_SHA,
        "checkpoints": actual,
        "conditions": {
            "steady_mps": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0],
            "steady_episodes": 50,
            "low_transitions": ["0->0.2", "0->0.4", "0->0.6", "0.6->0.4", "0.6->0.2", "0.6->0"],
            "low_transition_episodes": 50,
            "anchor_transitions": ["0->1.2", "1.2->2.0", "2.0->1.2", "1.2->0"],
            "anchor_transition_episodes": 20,
            "yaw_probe": {"speeds": [0.2, 0.4, 0.6, 1.2], "yaw_rates": [-0.1, -0.05, 0.0, 0.05, 0.1], "episodes": 20, "duration_s": 5.0},
            "feedback": {"kp": 1.0, "omega_max": 0.1, "no_search": True},
        },
    })
    seeds = {}
    for family, count in (("steady", 50), ("low_transition", 50), ("anchor_transition", 20), ("yaw_probe", 20), ("feedback", 20)):
        seeds[family] = list(range(20267901, 20267901 + count))
    dump("diagnostic_seed_manifest.json", {
        "seed_root": 20267901, "selection": "pre-generated consecutive seeds",
        "same_sets_across_checkpoints": True, "sets": seeds,
    })
    fields = [
        {"slice": [0, 3], "name": "base_linear_velocity", "frame": "body", "present": True},
        {"slice": [3, 6], "name": "base_angular_velocity", "frame": "body", "present": True},
        {"slice": [6, 9], "name": "projected_gravity", "frame": "body", "present": True},
        {"slice": [9, 12], "name": "velocity_command_vx_vy_yaw_rate", "frame": "body-command", "present": True},
        {"slice": [12, 24], "name": "joint_position_relative_default", "present": True},
        {"slice": [24, 36], "name": "joint_velocity", "present": True},
        {"slice": [36, 48], "name": "previous_action", "present": True},
    ]
    absent = ["absolute_world_yaw", "initial_heading", "target_heading", "heading_error", "world_position", "path_lateral_error"]
    dump("heading_observation_contract.json", {
        "dimension": 48, "source": "registered ObservationManager and Stage 1 frozen contract",
        "field_order": fields, "absent_fields": absent, "observation_history": False,
        "absolute_heading_observable": False,
        "policy_can_observe": ["instantaneous body yaw rate", "body attitude through projected gravity", "commanded yaw rate", "proprioception", "previous action"],
        "policy_cannot_observe": ["sign or magnitude of accumulated world-heading error relative to the episode reference"],
    })
    dump("heading_observability_classification.json", {
        "classification": "ABSOLUTE_HEADING_UNOBSERVABLE",
        "statement": (
            "The policy can suppress instantaneous yaw-rate bias, but cannot directly observe "
            "the sign or magnitude of accumulated world-heading error and drive it back to zero."
        ),
        "not_yet_a_causal_primary_classification": True,
    })
    dump("policy_mirror_equivariance.json", {
        "status": "NOT_EXECUTED_DUE_TO_UNVERIFIED_MIRROR_MAPPING",
        "reason": (
            "The asset joint-axis transforms and the exact projected-gravity/angular-velocity "
            "reflection map were not yet jointly verified; no guessed mirrored observation was evaluated."
        ),
        "mirror_action_applied_to_simulation": False,
    })


if __name__ == "__main__":
    main()
