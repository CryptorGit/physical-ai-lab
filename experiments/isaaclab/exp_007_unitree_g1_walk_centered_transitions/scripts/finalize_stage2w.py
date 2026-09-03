"""Freeze the completed Stage 2W audit without rerunning simulation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage2w_independent_walk"
PILOT1 = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_22-32-29_stage2w_independent_walk_pilot1_1024_150"
PILOT2 = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_22-39-22_stage2w_independent_walk_pilot2_1024_100"


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def grouped(rows: list[dict[str, str]], key: str) -> dict[str, dict]:
    result = {}
    for value in ("0.6", "0.8", "1.0", "1.2"):
        subset = [row for row in rows if row["target_speed_mps"] == value]
        numbers = [float(row[key]) for row in subset]
        result[value] = {
            "episodes": len(subset),
            "mean": sum(numbers) / len(numbers),
            "max": max(numbers),
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    strategy_commit = "c13f74c756e6000821754b538355bc1da86d9d84"
    formal = read_json(OUT / "formal_model_150_summary.json")
    with (OUT / "formal_model_150_episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    write_json(
        "strategy_revision.json",
        {
            **read_json(EXP / "strategy_revision.json"),
            "strategy_commit": strategy_commit,
            "stage2w_outcome": "FAIL",
            "next_stage_allowed": False,
        },
    )
    shutil.copyfile(EXP / "transition_graph.json", OUT / "state_graph_v2.json")
    shutil.copyfile(EXP / "state_contracts.json", OUT / "state_contracts.json")
    shutil.copyfile(EXP / "transition_contracts.json", OUT / "transition_contracts.json")

    parent_a = read_json(OUT / "parent_model4246_summary.json")
    parent_b = read_json(OUT / "parent_stage2r_p1_model50_summary.json")
    write_json(
        "parent_comparison.json",
        {
            "scope": "STEADY_WALK_ONLY",
            "seed": 20260728,
            "episodes_per_speed": 5,
            "selection_order_frozen_before_comparison": [
                "walk_success",
                "fall",
                "path_drift",
                "heading",
                "saturation",
                "speed_tracking",
            ],
            "stand_performance_considered": False,
            "candidates": {
                "model_4246": {
                    "checkpoint": parent_a["checkpoint"],
                    "sha256": parent_a["checkpoint_sha256"],
                    "overall": parent_a["overall"],
                },
                "stage2r_pilot1_model_50": {
                    "checkpoint": parent_b["checkpoint"],
                    "sha256": parent_b["checkpoint_sha256"],
                    "overall": parent_b["overall"],
                },
            },
            "selected_parent": "model_4246",
            "reason": "Higher steady-WALK success (15% vs 10%) with zero falls and lower heading error; STAND was excluded from selection.",
        },
    )
    write_json(
        "training_config.json",
        {
            "stage": "Stage 2W",
            "scope": "STEADY_WALK_ONLY",
            "observation_dimension": 123,
            "action_dimension": 37,
            "action_scale": 0.5,
            "steady_commands_mps": [0.6, 0.8, 1.0, 1.2],
            "unsupported_commands": "reject_without_clamp",
            "command_ramp": {"type": "minimum_jerk", "duration_s": 1.5},
            "pilot1": {
                "run": rel(PILOT1),
                "parent": parent_a["checkpoint"],
                "num_envs": 1024,
                "iterations": 150,
                "seed": 20260729,
                "checkpoints": [0, 50, 100, 150],
                "optimizer": "reset",
                "exploration_std": "strict_load_then_reset_trainable_to_0.25",
            },
            "pilot2": {
                "run": rel(PILOT2),
                "parent": rel(PILOT1 / "model_150.pt"),
                "num_envs": 1024,
                "iterations": 100,
                "seed": 20260801,
                "checkpoints": [0, 50, 100],
                "optimizer": "reset",
                "exploration_std": "inherited",
            },
            "formal": {
                "candidate": rel(PILOT1 / "model_150.pt"),
                "seed": 20260731,
                "episodes": 50,
                "speed_counts": {"0.6": 13, "0.8": 13, "1.0": 12, "1.2": 12},
            },
            "run_expert_loaded": False,
            "stand_gate_enabled": False,
            "transition_training_enabled": False,
            "world_xy_policy_input": False,
            "source_git_revision": strategy_commit,
        },
    )
    training_runs = {}
    for label, directory in (("pilot1", PILOT1), ("pilot2", PILOT2)):
        event = next(directory.glob("events.out.*"))
        run = read_json(directory / "stage2w_run.json")
        checkpoints = sorted(directory.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
        training_runs[label] = {
            "resolved_preflight": read_json(directory / "stage2w_preflight.json"),
            "run": run,
            "event_file": rel(event),
            "event_sha256": sha(event),
            "checkpoints": [{"path": rel(path), "sha256": sha(path)} for path in checkpoints],
        }
    write_json("training_runs.json", training_runs)
    write_json(
        "reward_definition.json",
        {
            "base": "G1FlatRunStage2EnvCfg",
            "common_terms_retained": [
                "linear velocity tracking",
                "yaw-rate tracking",
                "upright posture",
                "vertical velocity suppression",
                "roll/pitch angular velocity suppression",
                "torque cost",
                "action-rate cost",
                "joint acceleration cost",
                "joint-limit penalty",
                "foot slip penalty",
                "termination penalty",
            ],
            "stage2w_terms": {
                "pilot1": {
                    "track_lin_vel_xy_exp": 2.0,
                    "heading_error_l2": -1.0,
                    "lateral_velocity_l2": -0.5,
                    "cross_track_error_l2": -0.25,
                    "ankle_pitch_effort_hinge_above_95pct": -0.25,
                },
                "pilot2": {
                    "track_lin_vel_xy_exp": 2.5,
                    "heading_error_l2": -2.0,
                    "lateral_velocity_l2": -0.75,
                    "cross_track_error_l2": -0.5,
                    "ankle_pitch_effort_hinge_above_95pct": -0.25,
                },
            },
            "stand_specific_reward": False,
            "transition_reward": False,
            "fixed_gait_phase_reward": False,
        },
    )

    sweep_labels = [
        "parent_model4246",
        "parent_stage2r_p1_model50",
        "pilot1_model_50",
        "pilot1_model_100",
        "pilot1_model_150",
        "pilot2_model_50",
        "pilot2_model_100",
    ]
    write_json(
        "checkpoint_sweep.json",
        {
            "evaluations": {
                label: read_json(OUT / f"{label}_summary.json") for label in sweep_labels
            },
            "best_diagnostic_checkpoint": rel(PILOT1 / "model_150.pt"),
            "best_diagnostic_checkpoint_sha256": sha(PILOT1 / "model_150.pt"),
            "formal_candidate": True,
            "formal_pass": False,
            "pilot2_result": "DEGRADED_HEADING_AND_SUCCESS; branch rejected",
            "selection_note": "Best pilot checkpoint was selected rather than latest checkpoint.",
        },
    )
    write_json("formal_summary.json", formal)
    write_json("per_speed_results.json", formal["per_speed"])
    shutil.copyfile(OUT / "formal_model_150_episodes.csv", OUT / "episodes.csv")
    write_json("failure_counts.json", formal["failure_counts"])

    saturation = {
        "definition": {
            "joint": "left/right ankle_pitch",
            "instantaneous_threshold": 0.95,
            "long_dwell_threshold_s": 0.20,
        },
        "overall": {
            "failure_rate": formal["overall"]["saturation_failure_rate"],
            "max_fraction": max(float(row["ankle_saturation_fraction"]) for row in rows),
            "max_dwell_s": max(float(row["ankle_saturation_max_dwell_s"]) for row in rows),
            "left_p95_max": max(float(row["ankle_effort_p95_left"]) for row in rows),
            "right_p95_max": max(float(row["ankle_effort_p95_right"]) for row in rows),
        },
        "fraction_by_speed": grouped(rows, "ankle_saturation_fraction"),
        "max_dwell_by_speed": grouped(rows, "ankle_saturation_max_dwell_s"),
    }
    write_json("saturation_diagnostics.json", saturation)
    write_json(
        "path_drift_diagnostics.json",
        {
            "gate_frozen_before_training": formal["path_gate_frozen"],
            "overall_failure_rate": formal["overall"]["path_drift_failure_rate"],
            "max_cross_track_by_speed": grouped(rows, "path_drift_max_m"),
            "drift_rate_by_speed": grouped(rows, "path_drift_rate_mps"),
            "gate_pass": formal["gate_checks"]["path_drift_gate"],
        },
    )
    write_json(
        "gate.json",
        {
            "stage": "Stage 2W",
            "status": "FAIL",
            "eligible_for_stage3": False,
            "supported_walk_speed_range": None,
            "failures": [
                "overall WALK success 86% < 95%",
                "1.0 m/s success 75% < 90%",
                "1.2 m/s success 83.33% < 90%",
                "heading error p95 0.182207 rad > 0.12 rad",
            ],
            "warnings": [
                "0.6 and 0.8 m/s each reached 92.31%, but no formal supported range is published because the full requested continuous range failed.",
                "The best checkpoint remains diagnostic only; no artifact or production capability was created.",
            ],
            "metrics": formal["overall"],
            "per_speed": formal["per_speed"],
            "thresholds": {
                "overall_success_min": 0.95,
                "each_speed_success_min": 0.90,
                "fall_max": 0.02,
                "heading_error_p95_rad_max": 0.12,
                "speed_error_mean_mps_max": 0.20,
                "long_dwell_saturation_failure_max": 0.05,
                "dangerous_slip_failure_max": 0.05,
                "excessive_flight_failure_max": 0.05,
                "path_drift_failure_max": 0.05,
            },
            "selected_parent": parent_a["checkpoint"],
            "selected_parent_sha256": parent_a["checkpoint_sha256"],
            "best_diagnostic_checkpoint": rel(PILOT1 / "model_150.pt"),
            "best_diagnostic_checkpoint_sha256": sha(PILOT1 / "model_150.pt"),
            "artifact_created": False,
            "capability_manifest_updated_to_pass": False,
            "stand_expert_unchanged": True,
            "run_expert_loaded": False,
            "transition_expert_trained": False,
            "stage2a_b_r_unchanged": True,
            "source_git_revision": strategy_commit,
            "finalization_git_revision": revision,
        },
    )
    commands = r"""# Stage 2W exact reproduction commands (PowerShell)
cd "$HOME\workspace\physical-ai-lab"

$parent4246 = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$parent2r = ".\logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_21-43-52_stage2r_r1_pilot1_1024_100\model_50.pt"
$pilot1 = ".\logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_22-32-29_stage2w_independent_walk_pilot1_1024_150"
$pilot2 = ".\logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_22-39-22_stage2w_independent_walk_pilot2_1024_100"
$out = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage2w_independent_walk"

.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_steady_state.ps1 -Checkpoint $parent4246 -Mode preflight -Label parent_model4246 -Output $out -Seed 20260728
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_steady_state.ps1 -Checkpoint $parent2r -Mode preflight -Label parent_stage2r_p1_model50 -Output $out -Seed 20260728

.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\train_walk_steady_state.ps1 -Checkpoint $parent4246 -NumEnvs 1024 -Iterations 150 -Seed 20260729 -RunName stage2w_independent_walk_pilot1_1024_150 -RewardProfile pilot1
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_steady_state.ps1 -Checkpoint "$pilot1\model_50.pt" -Mode pilot -Label pilot1_model_50 -Output $out -Seed 20260730
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_steady_state.ps1 -Checkpoint "$pilot1\model_100.pt" -Mode pilot -Label pilot1_model_100 -Output $out -Seed 20260730
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_steady_state.ps1 -Checkpoint "$pilot1\model_150.pt" -Mode pilot -Label pilot1_model_150 -Output $out -Seed 20260730
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_steady_state.ps1 -Checkpoint "$pilot1\model_150.pt" -Mode formal -Label formal_model_150 -Output $out -Seed 20260731

.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\train_walk_steady_state.ps1 -Checkpoint "$pilot1\model_150.pt" -NumEnvs 1024 -Iterations 100 -Seed 20260801 -RunName stage2w_independent_walk_pilot2_1024_100 -RewardProfile pilot2
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_steady_state.ps1 -Checkpoint "$pilot2\model_50.pt" -Mode pilot -Label pilot2_model_50 -Output $out -Seed 20260802
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_steady_state.ps1 -Checkpoint "$pilot2\model_100.pt" -Mode pilot -Label pilot2_model_100 -Output $out -Seed 20260802

# Diagnostic GUI only: the formal gate failed.
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_walk_steady_state.ps1 -Speed 1.0
"""
    (OUT / "reproduction_commands.ps1").write_text(commands)


if __name__ == "__main__":
    main()
