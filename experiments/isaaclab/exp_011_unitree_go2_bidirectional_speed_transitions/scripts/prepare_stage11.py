"""Freeze Stage 11 contracts and run offline reward tests before simulation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction"
PARENT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"
EXPECTED_PARENT = "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd"
START = "b975ebbd43f703265693ccda2cd429a2cae943ca"
sys.path.insert(0, str(EXP / "src"))

from go2_bidirectional.stage11_tasks.reward import run_unit_tests  # noqa: E402


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
stage11_markers = (
    "stage11_", "stage11_tasks", "prepare_stage11.py", "preflight_stage11.py",
    "train_stage11.py", "evaluate_stage11.py", "finalize_stage11.py",
    "run_stage11_tangential_slip.ps1", "exp_011_go2_tangential_slip_reduction_report.md",
)
unrelated = [line for line in status if not any(marker in line for marker in stage11_markers)]
if head != START:
    raise SystemExit(f"unexpected starting HEAD {head}")
if sha(PARENT) != EXPECTED_PARENT:
    raise SystemExit("Stage 7 selected checkpoint hash mismatch")
dump("starting_repository_state.json", {
    "starting_head": head, "starting_status": unrelated, "unrelated_dirty_paths": unrelated,
})
dump("stage10_reference.json", {
    "classification": "PHASE_GATED_FIXED_HEADING_PASS",
    "checkpoint": str(PARENT.resolve()), "checkpoint_sha256": EXPECTED_PARENT,
    "controller": {"name": "PhaseGatedFixedHeadingController", "kp": 1.0, "yaw_limit": 0.10},
    "protected": True,
})
protocol = {
    "stage": 11,
    "single_learning_change": "contact-conditioned tangential-relative-slip penalty",
    "task_id": "Isaac-Exp011-Go2-Tangential-Slip-v0",
    "parent_sha256": EXPECTED_PARENT,
    "controller": "frozen Stage 10 PhaseGatedFixedHeadingController",
    "curriculum": "frozen Stage 7 15/35/30/20",
    "reward_addition_count": 1,
    "pilot": {"num_envs": 2048, "iterations": 200, "seed": 20261001},
    "preflight": {"num_envs": 2048, "rollout_batches": 10},
    "validation_seed_root": 20270901, "formal_seed_root": 20271901,
    "production_status": "DIAGNOSTIC_CANDIDATE",
}
dump("protocol.json", protocol)
dump("phase_gated_heading_contract.json", {
    "source": "Stage 10 frozen contract", "kp": 1.0, "yaw_limit_rad_s": 0.10,
    "heading_error": "wrap(reference-current)", "steady_activation_s": [1.0, 1.5],
    "transition_activation": "after 0.5 s target acquisition, 0.5 s minimum jerk",
    "mutated": False,
})
controller_source = EXP / "src/go2_bidirectional/phase_gated_heading.py"
dump("phase_gated_heading_hash.json", {"path": str(controller_source), "sha256": sha(controller_source)})
dump("contact_telemetry_contract.json", {
    "source": "PhysX RigidContactView",
    "contact_fields": ["world point", "world normal", "normal force", "foot body", "static ground"],
    "friction_fields": ["world friction force"],
    "surface_velocity": "v_b + omega_b x (p_c - x_b)",
    "ground_surface_velocity": 0.0,
    "relative_tangent": "v_rel - dot(v_rel,n_hat)*n_hat",
    "units": {"velocity": "m/s", "force": "N", "position": "m"},
})
dump("tangential_slip_reward_contract.json", {
    "name": "go2_contact_tangential_slip",
    "stable_contact": {"normal_force_n_gt": 5.0, "contact_age_steps_gte": 3},
    "causal": True, "future_release_used": False,
    "force_weight": {"reference_n": 100.0, "clip": [0.0, 1.0]},
    "robust_score": {"v_free": 0.20, "v_scale": 0.30, "huber_cap": 5.0},
    "legacy_anchor_displacement": False, "foot_link_origin_velocity": False,
    "weight": "calibrated once from preflight",
})
tests = run_unit_tests()
dump("tangential_slip_reward_unit_tests.json", tests)
if not tests["all_pass"]:
    raise SystemExit("tangential slip reward unit tests failed")
dump("stage7_vs_stage11_curriculum_diff.json", {
    "semantic_curriculum_difference": 0,
    "ratios": {"ZERO_HOLD": 0.15, "LOW_SPEED_STEADY": 0.35, "LOW_SPEED_TRANSITION": 0.30, "CAPABILITY_ANCHOR": 0.20},
    "speeds_pairs_holds_ramp": "bitwise source inheritance from Stage 7 command",
})
print("Stage 11 offline contracts and unit tests: PASS")
