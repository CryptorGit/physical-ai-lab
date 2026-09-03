"""Freeze Stage 2W-B FULL_PASS results and package the production WALK expert."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage2wb_walk_stabilization"
OLD = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage2w_independent_walk"
ART = REPO / "artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_steady_state_expert_v1"
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_22-32-29_stage2w_independent_walk_pilot1_1024_150/model_150.pt"
PILOT1 = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-12-44_stage2wb_stabilization_pilot1_valid_1024_100"
PILOT2 = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100"
SELECTED = PILOT2 / "model_100.pt"
SOURCE_REVISION = "41567217a5ec4a461bc31d2f3c236f2c35c359b9"


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def group_mean(rows, key):
    result = {}
    for speed in ("0.6", "0.8", "1.0", "1.2"):
        values = [float(row[key]) for row in rows if row["target_speed_mps"] == speed]
        result[speed] = sum(values) / len(values)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    formal = read(OUT / "full_range_formal_candidate_summary.json")
    with (OUT / "full_range_formal_candidate_episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected_hash = sha(SELECTED)
    stage2w_gate = OLD / "gate.json"
    stage2w_episodes = OLD / "episodes.csv"
    write(
        OUT / "stage2w_reference.json",
        {
            "commit": SOURCE_REVISION,
            "classification": "FAIL",
            "eligible_for_stage3": False,
            "selected_checkpoint": rel(PARENT),
            "selected_checkpoint_sha256": sha(PARENT),
            "protected_files": {
                rel(stage2w_gate): sha(stage2w_gate),
                rel(stage2w_episodes): sha(stage2w_episodes),
            },
            "modified_by_stage2wb": False,
        },
    )
    controllers = {
        label: read(OUT / f"controller_{label}_summary.json")
        for label in ("zero_yaw", "current", "lower_bandwidth")
    }
    write(
        OUT / "controller_only_comparison.json",
        {
            "checkpoint": rel(PARENT),
            "checkpoint_sha256": sha(PARENT),
            "seed": 20260804,
            "episodes_per_speed": 8,
            "same_seed_reset_and_speed_assignment": True,
            "candidates": controllers,
            "selected": "current",
            "formal_candidate_from_controller_only": False,
            "root_cause_classification": "POLICY_RESPONSE_DOMINATED",
        },
    )
    controller = {
        "name": "Stage2W_Current_FixedTarget",
        "mode": "FixedTarget",
        "k_heading": 0.8,
        "k_yaw_rate": 0.10,
        "yaw_rate_limit_radps": 0.30,
        "low_pass_alpha": 0.15,
        "slew_limit_radps_per_control_step": 0.01,
        "selection_reason": "Best controller-only success and heading/path performance; lower bandwidth degraded recovery.",
    }
    write(OUT / "selected_heading_controller.json", controller)
    write(
        OUT / "training_config.json",
        {
            "scope": "INDEPENDENT_STEADY_WALK_ONLY",
            "parent": rel(PARENT),
            "parent_sha256": sha(PARENT),
            "actor_architecture": [123, 256, 128, 128, 37],
            "observation_dimension": 123,
            "action_dimension": 37,
            "action_scale": 0.5,
            "optimizer": "reset_each_pilot",
            "speed_distribution": {"0.6": 0.20, "0.8": 0.20, "1.0": 0.30, "1.2": 0.30},
            "heading_response_training": {
                "type": "smooth_low_frequency_sinusoidal_target_heading_perturbation",
                "probability": 0.5,
                "amplitude_max_rad": 0.06,
                "frequency_hz": [0.08, 0.15],
                "high_frequency_reversal": False,
            },
            "pilot1": {"run": rel(PILOT1), "iterations": 100, "num_envs": 1024, "seed": 20260805},
            "pilot2": {"run": rel(PILOT2), "iterations": 100, "num_envs": 1024, "seed": 20260807},
            "distinct_pilot_count": 2,
            "world_xy_policy_observation": False,
            "absolute_heading_policy_observation": False,
            "stand_run_transition_experts_loaded": False,
        },
    )
    write(
        OUT / "reward_delta.json",
        {
            "base": "Stage 2W pilot-1 reward profile",
            "unchanged_terms": "all Stage 2W terms including ankle effort hinge",
            "pilot1_single_delta": {"yaw_rate_oscillation": -0.02},
            "pilot2_only_change_from_pilot1": {"yaw_rate_oscillation": {"old": -0.02, "new": -0.05}},
            "ankle_penalty_increased": False,
            "fixed_foot_phase_or_lead_foot_reward": False,
        },
    )
    pilot1_summaries = {
        "model_0_parent": read(OUT / "pilot1_parent_model_0_summary.json"),
        "model_50": read(OUT / "pilot1_valid_model_50_summary.json"),
        "model_100": read(OUT / "pilot1_valid_model_100_summary.json"),
    }
    pilot2_summaries = {
        "model_0_parent": read(OUT / "pilot2_parent_model_0_summary.json"),
        "model_50": read(OUT / "pilot2_model_50_summary.json"),
        "model_100": read(OUT / "pilot2_model_100_summary.json"),
    }
    write(
        OUT / "pilot_results.json",
        {
            "pilot1": pilot1_summaries,
            "pilot2": pilot2_summaries,
            "duplicate_preflight_note": (
                "A repeated pilot-1 run with identical config and seed produced bitwise-identical "
                "model_50/model_100 hashes; it is a reproduction check, not a third tuning pilot."
            ),
        },
    )
    write(
        OUT / "checkpoint_sweep.json",
        {
            "selection_priority": [
                "fall", "WALK success", "heading error", "speed tracking", "path drift",
                "dangerous slip", "long-dwell saturation", "action-rate", "instantaneous ankle effort",
            ],
            "pilot1": pilot1_summaries,
            "pilot2": pilot2_summaries,
            "selected_checkpoint": rel(SELECTED),
            "selected_checkpoint_sha256": selected_hash,
            "selected_reason": "100% pilot success and lower heading p95 than pilot2 model_50.",
        },
    )
    write(OUT / "full_range_formal.json", formal)
    write(
        OUT / "walk_low_formal.json",
        {
            "status": "NOT_RUN_FULL_RANGE_PASSED",
            "candidate_range_mps": [0.6, 0.8],
            "episode_selection_from_full_range": False,
        },
    )
    write(OUT / "per_speed_results.json", formal["per_speed"])
    shutil.copyfile(OUT / "full_range_formal_candidate_episodes.csv", OUT / "episodes.csv")
    write(OUT / "failure_counts.json", formal["failure_counts"])
    write(
        OUT / "path_drift_diagnostics.json",
        {
            "thresholds": {"max_cross_track_m": 0.30, "max_drift_rate_mps": 0.08, "failure_rate_max": 0.05},
            "overall_failure_rate": formal["overall"]["path_drift_failure_rate"],
            "per_speed_failure_rate": {
                speed: values["path_drift_failure_rate"] for speed, values in formal["per_speed"].items()
            },
            "max_cross_track_mean_by_speed_m": group_mean(rows, "path_drift_max_m"),
            "gate_pass": formal["gate_checks"]["path_drift_failure_le_0_05"],
        },
    )
    write(
        OUT / "yaw_oscillation_diagnostics.json",
        {
            "controller_only": {
                label: {
                    "yaw_reversal_frequency_mean_hz": data["overall"]["yaw_reversal_frequency_mean_hz"],
                    "yaw_command_saturation_fraction_mean": data["overall"]["yaw_command_saturation_fraction_mean"],
                    "action_rate_p95": data["overall"]["action_rate_p95"],
                }
                for label, data in controllers.items()
            },
            "formal": {
                "yaw_reversal_frequency_mean_hz": formal["overall"]["yaw_reversal_frequency_mean_hz"],
                "yaw_command_saturation_fraction_mean": formal["overall"]["yaw_command_saturation_fraction_mean"],
                "action_rate_p95": formal["overall"]["action_rate_p95"],
            },
        },
    )
    gait_keys = (
        "stance_duration_asymmetry_s", "slip_asymmetry_mps", "hip_yaw_action_asymmetry_mean",
        "hip_roll_action_asymmetry_mean", "lateral_velocity_abs_mean_mps",
    )
    write(
        OUT / "gait_asymmetry_diagnostics.json",
        {
            "formal_mean": {
                key: sum(float(row[key]) for row in rows) / len(rows) for key in gait_keys
            },
            "formal_mean_by_speed": {key: group_mean(rows, key) for key in gait_keys},
            "fixed_lead_foot_enforced": False,
            "fixed_gait_phase_enforced": False,
        },
    )
    gate = {
        "stage": "Stage 2W-B",
        "status": "FULL_PASS",
        "eligible_for_stage3": True,
        "supported_walk_speed_range_mps": [0.6, 1.2],
        "capability_class": "FULL",
        "selected_checkpoint": rel(SELECTED),
        "selected_checkpoint_sha256": selected_hash,
        "selected_heading_controller": controller,
        "metrics": formal["overall"],
        "per_speed": formal["per_speed"],
        "gate_checks": formal["gate_checks"],
        "failures": [],
        "warnings": [
            "Two formal episodes failed heading/path criteria, but all aggregate and per-speed frozen gates passed."
        ],
        "artifact": rel(ART),
        "stage2w_results_unchanged": True,
        "transition_experts_started": False,
        "source_git_revision": SOURCE_REVISION,
    }
    write(OUT / "gate.json", gate)
    commands = rf"""# Exact Stage 2W-B reproduction commands
cd "$HOME\workspace\physical-ai-lab"
$parent = ".\{rel(PARENT).replace('/', chr(92))}"
$pilot1 = ".\{rel(PILOT1).replace('/', chr(92))}"
$pilot2 = ".\{rel(PILOT2).replace('/', chr(92))}"
$out = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage2wb_walk_stabilization"

.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint $parent -Mode timeline -Label stage2w_failure_replay -HeadingMode FixedTarget -KHeading 0.8 -KYawRate 0.10 -YawRateLimit 0.30 -LowPassAlpha 0.15 -SlewLimit 0.01 -Seed 20260731 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint $parent -Mode controller -Label controller_zero_yaw -HeadingMode ZeroYaw -Seed 20260804 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint $parent -Mode controller -Label controller_current -HeadingMode FixedTarget -KHeading 0.8 -KYawRate 0.10 -YawRateLimit 0.30 -LowPassAlpha 0.15 -SlewLimit 0.01 -Seed 20260804 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint $parent -Mode controller -Label controller_lower_bandwidth -HeadingMode FixedTarget -KHeading 0.5 -KYawRate 0.10 -YawRateLimit 0.25 -LowPassAlpha 0.08 -SlewLimit 0.005 -Seed 20260804 -Output $out

.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\train_walk_stabilization.ps1 -Checkpoint $parent -Iterations 100 -Seed 20260805 -RunName stage2wb_stabilization_pilot1_valid_1024_100 -YawOscillationWeight -0.02
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\train_walk_stabilization.ps1 -Checkpoint "$pilot1\model_100.pt" -Iterations 100 -Seed 20260807 -RunName stage2wb_stabilization_pilot2_1024_100 -YawOscillationWeight -0.05
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint "$pilot2\model_100.pt" -Mode formal-full -Label full_range_formal_candidate -HeadingMode FixedTarget -KHeading 0.8 -KYawRate 0.10 -YawRateLimit 0.30 -LowPassAlpha 0.15 -SlewLimit 0.01 -Seed 20260809 -Output $out

.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_walk_steady_state.ps1 -Speed 1.0
"""
    (OUT / "reproduction_commands.ps1").write_text(commands)

    ART.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SELECTED, ART / "model_100.pt")
    write(
        ART / "checkpoint.json",
        {
            "file": "model_100.pt",
            "sha256": selected_hash,
            "source": rel(SELECTED),
            "parent": rel(PARENT),
            "parent_sha256": sha(PARENT),
        },
    )
    write(ART / "supported_range.json", {"target_speed_mps": [0.6, 1.2], "unsupported_behavior": "REJECT_NO_CLAMP"})
    shutil.copyfile(EXP / "state_contracts.json", ART / "state_contract.json")
    write(
        ART / "observation_action_contract.json",
        {
            "observation_dimension": 123,
            "layout": {
                "base_linear_velocity": [0, 3], "base_angular_velocity": [3, 6],
                "projected_gravity": [6, 9], "velocity_command": [9, 12],
                "joint_position": [12, 49], "joint_velocity": [49, 86], "previous_action": [86, 123],
            },
            "world_xy_in_observation": False,
            "absolute_heading_in_observation": False,
            "action_dimension": 37,
            "action_scale": 0.5,
            "semantics": "default_joint_position + 0.5 * normalized_position_action",
        },
    )
    write(ART / "heading_controller.json", controller)
    write(ART / "formal_metrics.json", formal)
    write(
        ART / "failure_taxonomy.json",
        {
            "classes": [
                "fall", "walk_not_sustained", "speed_tracking_failure", "heading_failure",
                "path_drift_failure", "ankle_torque_saturation", "dangerous_slip_failure",
                "excessive_flight_failure", "unsupported_walk_speed",
            ]
        },
    )
    (ART / "reproduction_commands.ps1").write_text(commands)
    write(
        ART / "source_revision.json",
        {
            "training_source_git_revision": SOURCE_REVISION,
            "strategy_revision": "c13f74c756e6000821754b538355bc1da86d9d84",
            "stage2w_negative_result_commit": SOURCE_REVISION,
        },
    )
    sums = []
    for path in sorted(ART.iterdir()):
        if path.name != "SHA256SUMS":
            sums.append(f"{sha(path)}  {path.name}")
    (ART / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
