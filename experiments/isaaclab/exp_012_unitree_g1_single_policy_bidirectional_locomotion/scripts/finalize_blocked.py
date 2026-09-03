"""Finalize the fail-closed Stage 0/1 preflight result without a Pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1"
REPORT = REPO / "research/exp_012_g1_single_policy_bidirectional_locomotion_report.md"
CKPT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
CLASSIFICATION = "G1_YAW_RATE_NOT_LOCALLY_CONTROLLABLE"
REASON = "Parent heading-response preflight failed at 0.0 and 1.2 m/s; Pilot 1 is prohibited."


def dump(name, value):
    path = OUT / name
    if not path.exists():
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def csv_not_executed(name, columns):
    path = OUT / name
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow({columns[0]: "NOT_EXECUTED", columns[1]: REASON})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    not_run = {"status": "NOT_EXECUTED", "reason": REASON, "pilot_updates": 0, "formal_episodes": 0}
    parent = torch.load(CKPT, map_location="cpu", weights_only=False)
    wiring_joint = OUT / "wiring/g1_joint_order.json"
    if wiring_joint.exists():
        (OUT / "g1_joint_order.json").write_text(wiring_joint.read_text(encoding="utf-8"), encoding="utf-8")
    base_reward = {
        "source": "exp_005 G1FlatRunStage2EnvCfg via G1FlatRunBaseEnvCfg",
        "semantic_difference_from_parent_stage2": 0,
        "terms": {
            "track_lin_vel_xy_exp": 2.0, "track_ang_vel_z_exp": 1.0, "lin_vel_z_l2": -0.2,
            "ang_vel_xy_l2": -0.05, "dof_torques_l2": -2e-6, "dof_acc_l2": -1e-7,
            "action_rate_l2": -0.005, "feet_air_time": 0.25, "flat_orientation_l2": -1.0,
            "dof_pos_limits": -1.0, "termination_penalty": -200.0, "feet_slide": -0.2,
            "joint_deviation_hip": -0.1, "joint_deviation_arms": -0.1,
            "joint_deviation_fingers": -0.05, "joint_deviation_torso": -0.1,
        },
    }
    run_reward = {
        "source": "exp_005 G1FlatRunStage4EnvCfg / SafePeriodicFlightReward",
        "command_gate_mps": 2.3, "weight": 1.0, "precursor_reward_per_step": .25,
        "takeoff_precursor_reward_per_step": .05, "precursor_event_cap": .75,
        "precursor_min_flight_s": .04, "max_flight_s": .16, "excess_flight_penalty_per_step": .25,
        "completion_reward": 2.0, "completion_speed_error_mps": .30,
        "precursor_speed_error_mps": 1.20, "max_tilt_rad": .20, "max_vertical_speed_mps": .50,
    }
    dump("parent_base_reward_config.json", base_reward)
    dump("exp005_stage4_run_reward_config.json", run_reward)
    dump("exp012_resolved_reward_config.json", {
        "base": base_reward, "added_existing_terms": {"safe_periodic_flight": run_reward},
        "other_new_terms": [], "status": "STATIC_CONTRACT_RESOLVED",
    })
    dump("reward_config_diff.json", {
        "semantic_differences": ["existing exp_005 Stage 4 safe_periodic_flight"],
        "forbidden_differences": [], "status": "PASS",
    })
    dump("run_reward_isolation_audit.json", {
        "static_contract": "PASS", "requested_vx_below_2p3_returns_exact_zero": True,
        "live_rollout": "NOT_EXECUTED_AFTER_HEADING_PREFLIGHT_FAIL",
    })
    dump("resume_identity_audit.json", {
        "status": "PASS_IN_WIRING", "source": "wiring/resume_identity_audit.json",
        "pilot_resume": "NOT_EXECUTED",
    })
    dump("training_config.yaml", {
        "status": "NOT_EXECUTED", "planned_num_envs": 1024, "planned_iterations": 300,
        "seed": 20261021, "reason": REASON,
    })
    dump("optimization_stability.json", not_run)
    dump("checkpoint_manifest.json", {
        "status": "NO_PILOT_CHECKPOINTS", "parent": {
            "path": str(CKPT.relative_to(REPO)).replace("\\", "/"), "sha256": sha(CKPT),
            "iteration": parent["iter"], "optimizer_step": 85000,
        },
        "wiring_checkpoints": "results/.../stage2_pilot1/wiring/checkpoints (diagnostic only; not candidates)",
    })
    dump("selected_checkpoint.json", {
        "status": "NOT_SELECTED", "checkpoint": None, "reason": REASON,
        "initial_parent_not_promoted": True,
    })
    dump("parent_baseline_results.json", {
        "status": "NOT_EXECUTED", "reason": "Heading preflight is a prerequisite of the parent baseline.",
        "partial_preflight_only": "heading_response_preflight.json",
    })
    for name in ("formal_stand.json", "formal_walk.json", "formal_run.json",
                 "formal_transitions.json", "formal_integrated_sequence.json",
                 "single_weight_sequence_audit.json", "directional_hysteresis.json",
                 "capability_regression_audit.json", "diagnostic_2p8.json"):
        dump(name, not_run)
    csv_not_executed("training_curves.csv", ["status", "reason"])
    csv_not_executed("capability_training_timeline.csv", ["status", "reason"])
    csv_not_executed("validation_checkpoint_results.csv", ["status", "reason"])
    csv_not_executed("formal_walk.csv", ["status", "reason"])
    csv_not_executed("formal_run.csv", ["status", "reason"])
    csv_not_executed("formal_transitions.csv", ["status", "reason"])
    csv_not_executed("endpoint_state_comparison.csv", ["status", "reason"])
    protected = {}
    for n in range(5, 12):
        matches = list((REPO / "experiments/isaaclab").glob(f"exp_{n:03d}_*"))
        if matches:
            rel = matches[0].relative_to(REPO).as_posix()
            protected[f"exp_{n:03d}"] = subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{rel}"], cwd=REPO, text=True).strip()
    dump("protected_hashes.json", {
        "git_tree_hashes_at_starting_head": protected, "parent_checkpoint_sha256": sha(CKPT),
        "capability_manifest_changed": False, "production_artifact_changed": False,
        "isaac_lab_core_changed": False, "remote_push": False,
        "note": "Pre-existing unrelated dirty paths are recorded in starting_repository_state.json.",
    })
    dump("stage_classification.json", {
        "classification": CLASSIFICATION, "stage_reached": "Stage 1 heading-controller preflight",
        "pilot1_executed": False, "formal_evaluation_executed": False,
        "evidence": {
            "0.0_mps": "sign/monotonic/fall gate failed",
            "0.6_mps": "sign and monotonic response passed",
            "1.2_mps": "negative-command sign and fall gate failed",
        },
    })
    dump("recommended_next_action.json", {
        "action": "diagnose G1 yaw-rate command controllability before Pilot 1",
        "one_method_only": True, "automatic_pilot2": False,
    })
    dump("gate.json", {
        "status": "FAIL_CLOSED", "classification": CLASSIFICATION, "reason": REASON,
        "parent_checkpoint_provenance": "PASS", "parent_optimizer": "PASS",
        "curriculum_audit": "PASS", "heading_response_preflight": "FAIL",
        "pilot1": "PROHIBITED", "formal": "PROHIBITED",
    })
    reproduction = r'''$ErrorActionPreference = "Stop"
cd "$HOME\workspace\physical-ai-lab"
.\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\audit_parent.ps1
.\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\run_parent_baseline.ps1
# Pilot command is intentionally not invoked: G1_YAW_RATE_NOT_LOCALLY_CONTROLLABLE.
# After a separately approved diagnosis, the frozen command would be:
# .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\run_exp012_stage2.ps1
'''
    (OUT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")
    REPORT.write_text(f"""# EXP 012 — G1 single-policy bidirectional locomotion

## Outcome

The experiment stopped fail-closed at the required parent heading-response
preflight. The classification is `{CLASSIFICATION}`. Pilot 1, checkpoint
selection, formal evaluation, and 2.8 m/s diagnostics were not run.

## Parent

The exact parent is `model_4246.pt`, SHA-256
`734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621`.
Its actor is 123→256→128→128→37 and its critic is 123→256→128→128→1.
The 17-state Adam mapping is present at step 85,000 with learning rate
2.25e-5. Strict actor, critic, std, deterministic-action, and optimizer
mapping identity passed in the 16-environment wiring run.

## Curriculum and reward

The preregistered distribution is ZERO_HOLD/WALK_STEADY/RUN_HOLD/SEQUENCE
= 20/20/20/40. A 100,000-schedule audit passed all ±1% and 1.10 ratio gates.
The parent Stage 2 reward is retained; the only resolved semantic addition is
the existing exp_005 Stage 4 `safe_periodic_flight` term, statically gated to
requested vx ≥2.3 m/s.

## Heading preflight

At 0.6 m/s the parent responded monotonically and with the requested yaw-rate
sign. At 0.0 m/s and 1.2 m/s it did not. STAND showed 5–10% falls in some
small-yaw conditions, and 1.2 m/s failed the negative-command sign contract
and showed a 5% fall rate at +0.10 rad/s. This violates the explicit
precondition, so the phase-gated controller cannot be treated as frozen and
safe for this Pilot.

## Scientific interpretation

This result does not test or refute the unified-policy hypothesis. It isolates
an earlier prerequisite failure: the chosen parent does not have the required
local yaw-rate command controllability across STAND, WALK 0.6, and WALK 1.2.
The next single method is a G1 yaw-rate controllability diagnosis before any
single-policy Pilot.

## Repository

Starting HEAD: `60028d13a5534527835e215c37106ea107585b39`.
Existing exp_005–exp_011 results, capability manifests, production artifacts,
Isaac Lab core, and the parent checkpoint were not modified. Remote push is
false. Unrelated pre-existing dirty paths remain untouched.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
