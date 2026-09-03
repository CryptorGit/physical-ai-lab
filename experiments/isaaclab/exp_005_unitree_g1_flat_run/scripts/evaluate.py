"""Evaluate G1 velocity tracking and gait phases at fixed forward speeds."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from g1_flat_run.robustness import apply_robustness_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)


DEFAULT_TASK_ID = "Isaac-Velocity-Flat-G1-Run-Stage3-Eval-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint path.")
parser.add_argument("--task", default=DEFAULT_TASK_ID, help="Registered evaluation task ID.")
parser.add_argument("--speeds", type=float, nargs="+", default=[2.3, 2.4, 2.5, 2.6])
parser.add_argument("--episodes_per_speed", type=int, default=3)
parser.add_argument(
    "--parallel_envs_per_speed",
    type=int,
    default=1,
    help="Parallel replicas per fixed speed; episodes_per_speed must be divisible by this value.",
)
parser.add_argument("--max_steps", type=int, default=3200)
parser.add_argument(
    "--steady_state_start_s",
    type=float,
    default=2.0,
    help="Exclude this initial acceleration interval from steady-state velocity errors.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--curriculum_stage", type=int, choices=(0, 1, 2), default=0)
parser.add_argument("--output_dir", default="")
parser.add_argument("--condition_name", default="baseline")
parser.add_argument("--friction_scale", type=float, default=1.0)
parser.add_argument("--mass_scale", type=float, default=1.0)
parser.add_argument("--com_shift_x_m", type=float, default=0.0)
parser.add_argument("--stiffness_scale", type=float, default=1.0)
parser.add_argument("--damping_scale", type=float, default=1.0)
parser.add_argument("--action_delay_steps", type=int, choices=(0, 1, 2), default=0)
parser.add_argument("--external_force_axis", choices=("none", "x", "y"), default="none")
parser.add_argument("--external_force_n", type=float, default=60.0)
parser.add_argument("--external_force_start_s", type=float, default=8.0)
parser.add_argument("--external_force_duration_s", type=float, default=0.20)
parser.add_argument("--small_rough_terrain", action="store_true")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def _zero_metrics(num_envs: int, device: str) -> dict[str, torch.Tensor]:
    names = (
        "reward",
        "steps",
        "actual_speed",
        "command_speed",
        "absolute_error",
        "yaw_forward_speed",
        "yaw_forward_absolute_error",
        "lateral_absolute_speed",
        "lateral_squared_speed",
        "yaw_rate_absolute_error",
        "yaw_rate_squared_error",
        "xy_tracking_error",
        "steady_steps",
        "steady_yaw_forward_speed",
        "steady_yaw_forward_absolute_error",
        "steady_lateral_absolute_speed",
        "steady_yaw_rate_absolute_error",
        "steady_xy_tracking_error",
        "flight_steps",
        "single_support_steps",
        "double_support_steps",
        "slip_speed",
        "contact_samples",
        "feet_slide_raw",
        "feet_slide_weighted",
        "feet_slide_weight",
        "high_speed_feet_slide_raw",
        "high_speed_feet_slide_weighted",
        "high_speed_feet_slide_weight",
        "track_lin_vel_raw",
        "track_lin_vel_weighted",
        "track_lin_vel_weight",
        "quality_gated_excess_slip_raw",
        "quality_gated_excess_slip_weighted",
        "quality_gated_excess_slip_weight",
        "quality_gated_excess_slip_eligible_steps",
        "contact_duration",
        "contact_events",
        "flight_events",
        "completed_flight_events",
        "flight_duration_sum",
        "flight_duration_20ms",
        "flight_duration_40ms",
        "flight_duration_60ms",
        "flight_duration_ge_80ms",
        "max_flight_duration",
        "alternation_opportunities",
        "alternating_landings",
        "normal_landings",
        "max_consecutive_running_cycles",
        "torso_tilt",
        "max_torso_tilt",
        "absolute_vertical_speed",
        "max_absolute_vertical_speed",
        "landing_left",
        "landing_right",
        "same_side_landings",
        "double_foot_landings",
        "suspected_chatter_events",
        "contact_filter_disagreement_steps",
        "safe_periodic_flight_fires",
        "safe_periodic_flight_raw_reward",
        "safe_periodic_flight_weighted_reward",
        "precursor_reward_step_count",
        "precursor_raw_reward",
        "completion_reward_fire_count",
        "completion_raw_reward",
        "excess_flight_penalty_step_count",
        "left_stride_length_m",
        "right_stride_length_m",
        "step_frequency_hz",
        "left_contact_time_s",
        "right_contact_time_s",
        "left_mean_horizontal_force_n",
        "right_mean_horizontal_force_n",
        "left_mean_vertical_force_n",
        "right_mean_vertical_force_n",
        "left_max_landing_impact_n",
        "right_max_landing_impact_n",
        "left_contact_slip_mps",
        "right_contact_slip_mps",
        "base_vertical_excursion_m",
        "stride_asymmetry",
        "contact_time_asymmetry",
        "max_joint_speed_limit_ratio",
        "max_joint_torque_limit_ratio",
        "max_velocity_limit_fraction",
        "max_torque_limit_fraction",
        "reset_phase_initialization",
        "reset_not_single_foot",
        "reset_flight_too_short",
        "reset_flight_too_long",
        "reset_command_too_slow",
        "reset_tracking_error",
        "reset_torso_tilt",
        "reset_vertical_speed",
        "reset_same_side_landing",
        "external_force_applied",
        "external_force_signed_n",
        "external_force_recovered",
        "external_force_recovery_time_s",
    )
    metrics = {name: torch.zeros(num_envs, device=device) for name in names}
    for condition in (
        "high_command",
        "tracking",
        "torso_tilt",
        "vertical_speed",
        "short_flight",
        "single_foot",
        "alternating",
    ):
        metrics[f"pass_{condition}"] = torch.zeros(num_envs, device=device)
        metrics[f"fail_{condition}"] = torch.zeros(num_envs, device=device)
    return metrics


def _zero_high_speed_metrics(num_envs: int, num_joints: int, device: str) -> dict[str, torch.Tensor]:
    """Allocate Stage 5 gait, load and actuator diagnostic accumulators."""
    scalar_names = (
        "left_stride_sum", "right_stride_sum", "left_stride_count", "right_stride_count",
        "left_contact_steps", "right_contact_steps", "left_contact_events", "right_contact_events",
        "left_horizontal_force_sum", "right_horizontal_force_sum",
        "left_vertical_force_sum", "right_vertical_force_sum",
        "left_force_samples", "right_force_samples", "left_max_impact", "right_max_impact",
        "left_slip_sum", "right_slip_sum", "left_slip_samples", "right_slip_samples",
        "min_base_height", "max_base_height",
    )
    result = {name: torch.zeros(num_envs, device=device) for name in scalar_names}
    result["min_base_height"].fill_(torch.inf)
    result["max_base_height"].fill_(-torch.inf)
    for name in (
        "max_joint_speed", "max_joint_speed_ratio", "max_joint_torque", "max_joint_torque_ratio",
        "velocity_limit_steps", "torque_limit_steps",
    ):
        result[name] = torch.zeros((num_envs, num_joints), device=device)
    return result


def _reset_high_speed_metrics(metrics: dict[str, torch.Tensor], env_id: int) -> None:
    for name, values in metrics.items():
        if name == "min_base_height":
            values[env_id] = torch.inf
        elif name == "max_base_height":
            values[env_id] = -torch.inf
        else:
            values[env_id] = 0.0


def _record_episode(
    env_id: int,
    speed: float,
    episode_index: int,
    metrics: dict[str, torch.Tensor],
    *,
    completed: bool,
    fell: bool,
    flight_durations_ms: list[int],
    landing_sequence: list[str],
    high_speed_metrics: dict[str, torch.Tensor],
    joint_names: list[str],
    step_dt: float,
    landing_impacts: list[list[float]],
    precontact_downward_speeds: list[list[float]],
) -> dict[str, float | int | bool | str]:
    steps = max(int(metrics["steps"][env_id].item()), 1)
    contact_samples = max(float(metrics["contact_samples"][env_id].item()), 1.0)
    contact_events = max(float(metrics["contact_events"][env_id].item()), 1.0)
    completed_flights = max(float(metrics["completed_flight_events"][env_id].item()), 1.0)
    alternation_opportunities = max(float(metrics["alternation_opportunities"][env_id].item()), 1.0)
    left_stride_count = max(float(high_speed_metrics["left_stride_count"][env_id].item()), 1.0)
    right_stride_count = max(float(high_speed_metrics["right_stride_count"][env_id].item()), 1.0)
    left_events = max(float(high_speed_metrics["left_contact_events"][env_id].item()), 1.0)
    right_events = max(float(high_speed_metrics["right_contact_events"][env_id].item()), 1.0)
    left_force_samples = max(float(high_speed_metrics["left_force_samples"][env_id].item()), 1.0)
    right_force_samples = max(float(high_speed_metrics["right_force_samples"][env_id].item()), 1.0)
    left_slip_samples = max(float(high_speed_metrics["left_slip_samples"][env_id].item()), 1.0)
    right_slip_samples = max(float(high_speed_metrics["right_slip_samples"][env_id].item()), 1.0)
    left_stride = float(high_speed_metrics["left_stride_sum"][env_id].item()) / left_stride_count
    right_stride = float(high_speed_metrics["right_stride_sum"][env_id].item()) / right_stride_count
    left_contact_time = float(high_speed_metrics["left_contact_steps"][env_id].item()) * step_dt / left_events
    right_contact_time = float(high_speed_metrics["right_contact_steps"][env_id].item()) * step_dt / right_events
    max_speed_ratio = float(high_speed_metrics["max_joint_speed_ratio"][env_id].max().item())
    max_torque_ratio = float(high_speed_metrics["max_joint_torque_ratio"][env_id].max().item())
    max_velocity_limit_fraction = float(
        (high_speed_metrics["velocity_limit_steps"][env_id] / steps).max().item()
    )
    max_torque_limit_fraction = float(
        (high_speed_metrics["torque_limit_steps"][env_id] / steps).max().item()
    )
    all_impacts = landing_impacts[0] + landing_impacts[1]
    all_precontact_speeds = precontact_downward_speeds[0] + precontact_downward_speeds[1]

    def percentile(values: list[float], quantile: float) -> float:
        return float(np.percentile(values, quantile)) if values else 0.0

    record: dict[str, float | int | bool | str] = {
        "command_speed_mps": speed,
        "episode": episode_index,
        "completed": completed,
        "fell": fell,
        "episode_length_steps": steps,
        "reward": float(metrics["reward"][env_id].item()),
        "mean_actual_speed_mps": float(metrics["actual_speed"][env_id].item()) / steps,
        "mean_command_speed_mps": float(metrics["command_speed"][env_id].item()) / steps,
        "mean_abs_speed_error_mps": float(metrics["absolute_error"][env_id].item()) / steps,
        "mean_yaw_frame_forward_speed_mps": float(metrics["yaw_forward_speed"][env_id].item()) / steps,
        "mean_abs_yaw_frame_forward_error_mps": float(
            metrics["yaw_forward_absolute_error"][env_id].item()
        ) / steps,
        "mean_abs_lateral_speed_mps": float(metrics["lateral_absolute_speed"][env_id].item()) / steps,
        "rms_lateral_speed_mps": float(
            np.sqrt(float(metrics["lateral_squared_speed"][env_id].item()) / steps)
        ),
        "mean_abs_yaw_rate_error_rad_s": float(
            metrics["yaw_rate_absolute_error"][env_id].item()
        ) / steps,
        "rms_yaw_rate_error_rad_s": float(
            np.sqrt(float(metrics["yaw_rate_squared_error"][env_id].item()) / steps)
        ),
        "mean_xy_tracking_error_mps": float(metrics["xy_tracking_error"][env_id].item()) / steps,
        "flight_phase_fraction": float(metrics["flight_steps"][env_id].item()) / steps,
        "single_support_fraction": float(metrics["single_support_steps"][env_id].item()) / steps,
        "double_support_fraction": float(metrics["double_support_steps"][env_id].item()) / steps,
        "mean_contact_foot_slip_mps": float(metrics["slip_speed"][env_id].item()) / contact_samples,
        # Reward-manager raw terms are step means.  The evaluator slip above is
        # the same physical samples normalized by contacted-foot samples.
        "feet_slide_raw_per_step": float(metrics["feet_slide_raw"][env_id].item()) / steps,
        "feet_slide_weight": float(metrics["feet_slide_weight"][env_id].item()) / steps,
        "feet_slide_weighted_episode": float(metrics["feet_slide_weighted"][env_id].item()),
        "high_speed_feet_slide_raw_per_step": float(
            metrics["high_speed_feet_slide_raw"][env_id].item()
        ) / steps,
        "high_speed_feet_slide_weight": float(
            metrics["high_speed_feet_slide_weight"][env_id].item()
        ) / steps,
        "high_speed_feet_slide_weighted_episode": float(
            metrics["high_speed_feet_slide_weighted"][env_id].item()
        ),
        "track_lin_vel_raw_per_step": float(metrics["track_lin_vel_raw"][env_id].item()) / steps,
        "track_lin_vel_weight": float(metrics["track_lin_vel_weight"][env_id].item()) / steps,
        "track_lin_vel_weighted_episode": float(metrics["track_lin_vel_weighted"][env_id].item()),
        "quality_gated_excess_slip_raw_per_step": float(
            metrics["quality_gated_excess_slip_raw"][env_id].item()
        ) / steps,
        "quality_gated_excess_slip_weight": float(
            metrics["quality_gated_excess_slip_weight"][env_id].item()
        ) / steps,
        "quality_gated_excess_slip_weighted_episode": float(
            metrics["quality_gated_excess_slip_weighted"][env_id].item()
        ),
        "quality_gated_excess_slip_eligible_fraction": float(
            metrics["quality_gated_excess_slip_eligible_steps"][env_id].item()
        ) / steps,
        "mean_contact_duration_s": float(metrics["contact_duration"][env_id].item()) / contact_events,
        "flight_phase_event_count": int(metrics["flight_events"][env_id].item()),
        "flight_duration_distribution_ms": json.dumps(flight_durations_ms, separators=(",", ":")),
        "flight_events_20ms": int(metrics["flight_duration_20ms"][env_id].item()),
        "flight_events_40ms": int(metrics["flight_duration_40ms"][env_id].item()),
        "flight_events_60ms": int(metrics["flight_duration_60ms"][env_id].item()),
        "flight_events_ge_80ms": int(metrics["flight_duration_ge_80ms"][env_id].item()),
        "mean_flight_duration_s": float(metrics["flight_duration_sum"][env_id].item()) / completed_flights,
        "max_flight_duration_s": float(metrics["max_flight_duration"][env_id].item()),
        "alternating_contact_rate": float(metrics["alternating_landings"][env_id].item())
        / alternation_opportunities,
        "normal_landing_rate": float(metrics["normal_landings"][env_id].item()) / completed_flights,
        "max_consecutive_running_cycles": int(metrics["max_consecutive_running_cycles"][env_id].item()),
        "mean_torso_tilt_rad": float(metrics["torso_tilt"][env_id].item()) / steps,
        "max_torso_tilt_rad": float(metrics["max_torso_tilt"][env_id].item()),
        "mean_abs_vertical_speed_mps": float(metrics["absolute_vertical_speed"][env_id].item()) / steps,
        "max_abs_vertical_speed_mps": float(metrics["max_absolute_vertical_speed"][env_id].item()),
        "landing_sequence": ",".join(landing_sequence),
        "left_landing_count": int(metrics["landing_left"][env_id].item()),
        "right_landing_count": int(metrics["landing_right"][env_id].item()),
        "same_side_consecutive_landing_count": int(metrics["same_side_landings"][env_id].item()),
        "double_foot_landing_count": int(metrics["double_foot_landings"][env_id].item()),
        "suspected_chatter_event_count": int(metrics["suspected_chatter_events"][env_id].item()),
        "contact_filter_disagreement_fraction": float(metrics["contact_filter_disagreement_steps"][env_id].item())
        / steps,
        "safe_periodic_flight_fire_count": int(metrics["safe_periodic_flight_fires"][env_id].item()),
        "safe_periodic_flight_raw_reward": float(metrics["safe_periodic_flight_raw_reward"][env_id].item()),
        "safe_periodic_flight_weighted_reward": float(
            metrics["safe_periodic_flight_weighted_reward"][env_id].item()
        ),
        "precursor_reward_step_count": int(metrics["precursor_reward_step_count"][env_id].item()),
        "precursor_raw_reward": float(metrics["precursor_raw_reward"][env_id].item()),
        "completion_reward_fire_count": int(metrics["completion_reward_fire_count"][env_id].item()),
        "completion_raw_reward": float(metrics["completion_raw_reward"][env_id].item()),
        "excess_flight_penalty_step_count": int(
            metrics["excess_flight_penalty_step_count"][env_id].item()
        ),
        "left_stride_length_m": left_stride,
        "right_stride_length_m": right_stride,
        "step_frequency_hz": (
            float(high_speed_metrics["left_contact_events"][env_id].item())
            + float(high_speed_metrics["right_contact_events"][env_id].item())
        ) / max(steps * step_dt, step_dt),
        "left_contact_time_s": left_contact_time,
        "right_contact_time_s": right_contact_time,
        "left_mean_horizontal_force_n": float(high_speed_metrics["left_horizontal_force_sum"][env_id].item()) / left_force_samples,
        "right_mean_horizontal_force_n": float(high_speed_metrics["right_horizontal_force_sum"][env_id].item()) / right_force_samples,
        "left_mean_vertical_force_n": float(high_speed_metrics["left_vertical_force_sum"][env_id].item()) / left_force_samples,
        "right_mean_vertical_force_n": float(high_speed_metrics["right_vertical_force_sum"][env_id].item()) / right_force_samples,
        "left_max_landing_impact_n": float(high_speed_metrics["left_max_impact"][env_id].item()),
        "right_max_landing_impact_n": float(high_speed_metrics["right_max_impact"][env_id].item()),
        "landing_impact_mean_n": float(np.mean(all_impacts)) if all_impacts else 0.0,
        "landing_impact_median_n": percentile(all_impacts, 50),
        "landing_impact_p95_n": percentile(all_impacts, 95),
        "landing_impact_p99_n": percentile(all_impacts, 99),
        "landing_impact_over_3500_rate": (
            sum(value > 3500.0 for value in all_impacts) / len(all_impacts) if all_impacts else 0.0
        ),
        "left_landing_impact_mean_n": float(np.mean(landing_impacts[0])) if landing_impacts[0] else 0.0,
        "left_landing_impact_median_n": percentile(landing_impacts[0], 50),
        "left_landing_impact_p95_n": percentile(landing_impacts[0], 95),
        "left_landing_impact_p99_n": percentile(landing_impacts[0], 99),
        "right_landing_impact_mean_n": float(np.mean(landing_impacts[1])) if landing_impacts[1] else 0.0,
        "right_landing_impact_median_n": percentile(landing_impacts[1], 50),
        "right_landing_impact_p95_n": percentile(landing_impacts[1], 95),
        "right_landing_impact_p99_n": percentile(landing_impacts[1], 99),
        "precontact_downward_speed_mean_mps": (
            float(np.mean(all_precontact_speeds)) if all_precontact_speeds else 0.0
        ),
        "precontact_downward_speed_p95_mps": percentile(all_precontact_speeds, 95),
        "left_precontact_downward_speed_mean_mps": (
            float(np.mean(precontact_downward_speeds[0])) if precontact_downward_speeds[0] else 0.0
        ),
        "right_precontact_downward_speed_mean_mps": (
            float(np.mean(precontact_downward_speeds[1])) if precontact_downward_speeds[1] else 0.0
        ),
        "left_contact_slip_mps": float(high_speed_metrics["left_slip_sum"][env_id].item()) / left_slip_samples,
        "right_contact_slip_mps": float(high_speed_metrics["right_slip_sum"][env_id].item()) / right_slip_samples,
        "base_vertical_excursion_m": float(
            high_speed_metrics["max_base_height"][env_id].item()
            - high_speed_metrics["min_base_height"][env_id].item()
        ),
        "stride_asymmetry": abs(left_stride - right_stride) / max((left_stride + right_stride) * 0.5, 1.0e-6),
        "contact_time_asymmetry": abs(left_contact_time - right_contact_time) / max((left_contact_time + right_contact_time) * 0.5, 1.0e-6),
        "max_joint_speed_limit_ratio": max_speed_ratio,
        "max_joint_torque_limit_ratio": max_torque_ratio,
        "max_velocity_limit_fraction": max_velocity_limit_fraction,
        "max_torque_limit_fraction": max_torque_limit_fraction,
        "external_force_applied": bool(metrics["external_force_applied"][env_id].item()),
        "external_force_signed_n": float(metrics["external_force_signed_n"][env_id].item()),
        "external_force_recovered": bool(metrics["external_force_recovered"][env_id].item()),
        "external_force_recovery_time_s": float(
            metrics["external_force_recovery_time_s"][env_id].item()
        ),
        "joint_max_speed_rad_s": json.dumps(dict(zip(joint_names, high_speed_metrics["max_joint_speed"][env_id].tolist())), separators=(",", ":")),
        "joint_max_speed_limit_ratio": json.dumps(dict(zip(joint_names, high_speed_metrics["max_joint_speed_ratio"][env_id].tolist())), separators=(",", ":")),
        "joint_max_torque_nm": json.dumps(dict(zip(joint_names, high_speed_metrics["max_joint_torque"][env_id].tolist())), separators=(",", ":")),
        "joint_max_torque_limit_ratio": json.dumps(dict(zip(joint_names, high_speed_metrics["max_joint_torque_ratio"][env_id].tolist())), separators=(",", ":")),
        "joint_velocity_limit_fraction": json.dumps(dict(zip(joint_names, (high_speed_metrics["velocity_limit_steps"][env_id] / steps).tolist())), separators=(",", ":")),
        "joint_torque_limit_fraction": json.dumps(dict(zip(joint_names, (high_speed_metrics["torque_limit_steps"][env_id] / steps).tolist())), separators=(",", ":")),
        "landing_impact_distribution_n": json.dumps(landing_impacts, separators=(",", ":")),
        "precontact_downward_speed_distribution_mps": json.dumps(
            precontact_downward_speeds, separators=(",", ":")
        ),
        "curriculum_stage": args_cli.curriculum_stage,
        "curriculum_upper_speed_mps": (
            4.55 if "Stage7" in args_cli.task else (3.8, 3.9, 4.0)[args_cli.curriculum_stage]
        ),
    }
    steady_steps = max(float(metrics["steady_steps"][env_id].item()), 1.0)
    record.update(
        {
            "steady_state_start_s": max((steps - steady_steps) * step_dt, 0.0),
            "steady_mean_yaw_frame_forward_speed_mps": float(
                metrics["steady_yaw_forward_speed"][env_id].item()
            ) / steady_steps,
            "steady_mean_abs_forward_error_mps": float(
                metrics["steady_yaw_forward_absolute_error"][env_id].item()
            ) / steady_steps,
            "steady_mean_abs_lateral_speed_mps": float(
                metrics["steady_lateral_absolute_speed"][env_id].item()
            ) / steady_steps,
            "steady_mean_abs_yaw_rate_error_rad_s": float(
                metrics["steady_yaw_rate_absolute_error"][env_id].item()
            ) / steady_steps,
            "steady_mean_xy_tracking_error_mps": float(
                metrics["steady_xy_tracking_error"][env_id].item()
            ) / steady_steps,
        }
    )
    for condition in (
        "high_command",
        "tracking",
        "torso_tilt",
        "vertical_speed",
        "short_flight",
        "single_foot",
        "alternating",
    ):
        record[f"pass_{condition}_count"] = int(metrics[f"pass_{condition}"][env_id].item())
        record[f"fail_{condition}_count"] = int(metrics[f"fail_{condition}"][env_id].item())
    for reason in (
        "phase_initialization",
        "not_single_foot",
        "flight_too_short",
        "flight_too_long",
        "command_too_slow",
        "tracking_error",
        "torso_tilt",
        "vertical_speed",
        "same_side_landing",
    ):
        record[f"cycle_reset_{reason}_count"] = int(metrics[f"reset_{reason}"][env_id].item())
    record["classification"] = _classify_episode(record)
    return record


def _episode_quality_gates(record: dict[str, float | int | bool]) -> dict[str, bool]:
    """Return the episode-level gates used by the final classification."""
    return {
        "completed": bool(record["completed"]),
        "not_fallen": not bool(record["fell"]),
        "episode_length": int(record["episode_length_steps"]) >= 900,
        "speed_error": float(
            record.get("steady_mean_abs_forward_error_mps", record["mean_abs_speed_error_mps"])
        ) <= 0.25,
        "flight_events": int(record["flight_phase_event_count"]) >= 4,
        "safe_cycles": int(record["max_consecutive_running_cycles"]) >= 3,
        "alternating": float(record["alternating_contact_rate"]) >= 0.80,
        "normal_landing": float(record["normal_landing_rate"]) >= 0.80,
        "flight_duration": 0.04 <= float(record["mean_flight_duration_s"]) <= 0.16,
        "slip": float(record["mean_contact_foot_slip_mps"]) <= 0.55,
        "joint_velocity_saturation": float(record["max_velocity_limit_fraction"]) <= 0.05,
        "joint_torque_saturation": float(record["max_torque_limit_fraction"]) <= 0.20,
        "impact_p95": float(record["landing_impact_p95_n"]) <= 3500.0,
        "impact_exceedance": float(record["landing_impact_over_3500_rate"]) <= 0.05,
        "vertical_excursion": float(record["base_vertical_excursion_m"]) <= 0.30,
        "stride_asymmetry": float(record["stride_asymmetry"]) <= 0.20,
        "contact_time_asymmetry": float(record["contact_time_asymmetry"]) <= 0.20,
    }


def _classify_episode(record: dict[str, float | int | bool]) -> str:
    """Separate sustained alternating running from isolated hops and unstable flight."""
    gates = _episode_quality_gates(record)
    for gate_name, passed in gates.items():
        record[f"gate_{gate_name}_passed"] = passed
    if not bool(record["completed"]):
        record["physical_quality_gate_passed"] = False
        return "incomplete"

    stable = all(gates[name] for name in ("not_fallen", "episode_length", "speed_error"))
    periodic = all(
        gates[name]
        for name in ("flight_events", "safe_cycles", "alternating", "normal_landing", "flight_duration")
    )
    physical_quality = all(
        gates[name]
        for name in (
            "slip", "joint_velocity_saturation", "joint_torque_saturation", "impact_p95",
            "impact_exceedance", "vertical_excursion", "stride_asymmetry", "contact_time_asymmetry",
        )
    )
    record["physical_quality_gate_passed"] = physical_quality
    if stable and periodic and physical_quality:
        return "periodic_running"
    if stable and periodic:
        return "periodic_running_with_physical_risk"
    if stable and int(record["flight_phase_event_count"]) > 0:
        return "stable_with_isolated_flight"
    if stable:
        return "stable_high_speed_walk"
    if int(record["flight_phase_event_count"]) > 0:
        return "unstable_with_flight"
    return "not_stable"


def _summarize(records: list[dict[str, float | int | bool]]) -> list[dict[str, float | int | bool | str]]:
    grouped: dict[float, list[dict[str, float | int | bool]]] = defaultdict(list)
    for record in records:
        grouped[float(record["command_speed_mps"])].append(record)

    summaries = []
    numeric_fields = (
        "episode_length_steps",
        "reward",
        "mean_actual_speed_mps",
        "mean_command_speed_mps",
        "mean_abs_speed_error_mps",
        "mean_yaw_frame_forward_speed_mps",
        "mean_abs_yaw_frame_forward_error_mps",
        "mean_abs_lateral_speed_mps",
        "rms_lateral_speed_mps",
        "mean_abs_yaw_rate_error_rad_s",
        "rms_yaw_rate_error_rad_s",
        "mean_xy_tracking_error_mps",
        "steady_state_start_s",
        "steady_mean_yaw_frame_forward_speed_mps",
        "steady_mean_abs_forward_error_mps",
        "steady_mean_abs_lateral_speed_mps",
        "steady_mean_abs_yaw_rate_error_rad_s",
        "steady_mean_xy_tracking_error_mps",
        "flight_phase_fraction",
        "single_support_fraction",
        "double_support_fraction",
        "mean_contact_foot_slip_mps",
        "feet_slide_raw_per_step",
        "feet_slide_weight",
        "feet_slide_weighted_episode",
        "high_speed_feet_slide_raw_per_step",
        "high_speed_feet_slide_weight",
        "high_speed_feet_slide_weighted_episode",
        "track_lin_vel_raw_per_step",
        "track_lin_vel_weight",
        "track_lin_vel_weighted_episode",
        "quality_gated_excess_slip_raw_per_step",
        "quality_gated_excess_slip_weight",
        "quality_gated_excess_slip_weighted_episode",
        "quality_gated_excess_slip_eligible_fraction",
        "mean_contact_duration_s",
        "flight_phase_event_count",
        "flight_events_20ms",
        "flight_events_40ms",
        "flight_events_60ms",
        "flight_events_ge_80ms",
        "mean_flight_duration_s",
        "max_flight_duration_s",
        "alternating_contact_rate",
        "normal_landing_rate",
        "max_consecutive_running_cycles",
        "mean_torso_tilt_rad",
        "max_torso_tilt_rad",
        "mean_abs_vertical_speed_mps",
        "max_abs_vertical_speed_mps",
        "left_landing_count",
        "right_landing_count",
        "same_side_consecutive_landing_count",
        "double_foot_landing_count",
        "suspected_chatter_event_count",
        "contact_filter_disagreement_fraction",
        "safe_periodic_flight_fire_count",
        "safe_periodic_flight_raw_reward",
        "safe_periodic_flight_weighted_reward",
        "precursor_reward_step_count",
        "precursor_raw_reward",
        "completion_reward_fire_count",
        "completion_raw_reward",
        "excess_flight_penalty_step_count",
        "left_stride_length_m",
        "right_stride_length_m",
        "step_frequency_hz",
        "left_contact_time_s",
        "right_contact_time_s",
        "left_mean_horizontal_force_n",
        "right_mean_horizontal_force_n",
        "left_mean_vertical_force_n",
        "right_mean_vertical_force_n",
        "left_max_landing_impact_n",
        "right_max_landing_impact_n",
        "landing_impact_mean_n",
        "landing_impact_median_n",
        "landing_impact_p95_n",
        "landing_impact_p99_n",
        "landing_impact_over_3500_rate",
        "left_landing_impact_mean_n",
        "left_landing_impact_median_n",
        "left_landing_impact_p95_n",
        "left_landing_impact_p99_n",
        "right_landing_impact_mean_n",
        "right_landing_impact_median_n",
        "right_landing_impact_p95_n",
        "right_landing_impact_p99_n",
        "precontact_downward_speed_mean_mps",
        "precontact_downward_speed_p95_mps",
        "left_precontact_downward_speed_mean_mps",
        "right_precontact_downward_speed_mean_mps",
        "left_contact_slip_mps",
        "right_contact_slip_mps",
        "base_vertical_excursion_m",
        "stride_asymmetry",
        "contact_time_asymmetry",
        "max_joint_speed_limit_ratio",
        "max_joint_torque_limit_ratio",
        "max_velocity_limit_fraction",
        "max_torque_limit_fraction",
        "external_force_applied",
        "external_force_signed_n",
        "external_force_recovered",
        "external_force_recovery_time_s",
        "curriculum_stage",
        "curriculum_upper_speed_mps",
    )
    numeric_fields += tuple(
        f"{result}_{condition}_count"
        for condition in (
            "high_command",
            "tracking",
            "torso_tilt",
            "vertical_speed",
            "short_flight",
            "single_foot",
            "alternating",
        )
        for result in ("pass", "fail")
    )
    numeric_fields += tuple(
        f"cycle_reset_{reason}_count"
        for reason in (
            "phase_initialization",
            "not_single_foot",
            "flight_too_short",
            "flight_too_long",
            "command_too_slow",
            "tracking_error",
            "torso_tilt",
            "vertical_speed",
            "same_side_landing",
        )
    )
    for speed, speed_records in sorted(grouped.items()):
        completed = [record for record in speed_records if bool(record["completed"])]
        denominator = max(len(completed), 1)
        summary: dict[str, float | int | bool | str] = {
            "command_speed_mps": speed,
            "episodes": len(speed_records),
            "completed_episodes": len(completed),
            "fall_rate": sum(bool(record["fell"]) for record in completed) / denominator,
            "periodic_running_rate": sum(
                record["classification"] == "periodic_running" for record in speed_records
            )
            / len(speed_records),
            "physical_quality_gate_pass_rate": sum(
                bool(record["physical_quality_gate_passed"]) for record in speed_records
            ) / len(speed_records),
        }
        for field in numeric_fields:
            summary[field] = sum(float(record[field]) for record in speed_records) / len(speed_records)
        for gate_name in _episode_quality_gates(speed_records[0]):
            pass_count = sum(bool(record[f"gate_{gate_name}_passed"]) for record in speed_records)
            summary[f"gate_{gate_name}_pass_count"] = pass_count
            summary[f"gate_{gate_name}_fail_count"] = len(speed_records) - pass_count

        enough_data = len(completed) == len(speed_records)
        stable = (
            enough_data
            and float(summary["fall_rate"]) <= 0.1
            and float(summary["steady_mean_abs_forward_error_mps"]) <= 0.25
            and float(summary["episode_length_steps"]) >= 900.0
        )
        summary["stable_tracking"] = stable
        summary["running_evidence"] = stable and float(summary["periodic_running_rate"]) >= 0.80
        if summary["running_evidence"]:
            summary["classification"] = "periodic_running"
        elif stable and float(summary["flight_phase_event_count"]) > 0:
            summary["classification"] = "stable_with_isolated_flight"
        elif stable:
            summary["classification"] = "stable_high_speed_walk"
        else:
            summary["classification"] = "not_stable"
        summaries.append(summary)
    return summaries


def _write_results(
    records: list[dict[str, float | int | bool]],
    flight_event_records: list[dict[str, float | int | bool | str]],
    landing_event_records: list[dict[str, float | int | bool | str]],
    temporal_event_records: list[dict[str, float | int | bool | str]],
    checkpoint: Path,
    robustness_metadata: dict[str, object],
) -> None:
    """Write episode and per-speed summaries before the simulation app closes."""
    summaries = _summarize(records)
    episode_outcomes = {
        (float(record["command_speed_mps"]), int(record["episode"])): record for record in records
    }
    for event in landing_event_records:
        outcome = episode_outcomes[(float(event["command_speed_mps"]), int(event["episode"]))]
        event["fell_later"] = bool(outcome["fell"])
        if bool(outcome["fell"]):
            event["time_to_fall_s"] = max(
                (int(outcome["episode_length_steps"]) - int(event["step"])) * 0.02, 0.0
            )

    for summary in summaries:
        speed = float(summary["command_speed_mps"])
        events = [event for event in landing_event_records if float(event["command_speed_mps"]) == speed]
        impacts = [float(event["landing_impact_n"]) for event in events]
        left_impacts = [
            float(event["landing_impact_n"]) for event in events if event["foot"] == "L"
        ]
        right_impacts = [
            float(event["landing_impact_n"]) for event in events if event["foot"] == "R"
        ]
        downward_speeds = [float(event["precontact_downward_speed_mps"]) for event in events]

        def set_distribution(prefix: str, values: list[float]) -> None:
            summary[f"{prefix}_mean_n"] = float(np.mean(values)) if values else 0.0
            summary[f"{prefix}_median_n"] = float(np.percentile(values, 50)) if values else 0.0
            summary[f"{prefix}_p95_n"] = float(np.percentile(values, 95)) if values else 0.0
            summary[f"{prefix}_p99_n"] = float(np.percentile(values, 99)) if values else 0.0

        set_distribution("landing_impact", impacts)
        set_distribution("left_landing_impact", left_impacts)
        set_distribution("right_landing_impact", right_impacts)
        summary["landing_impact_over_3500_rate"] = (
            sum(value > 3500.0 for value in impacts) / len(impacts) if impacts else 0.0
        )
        summary["precontact_downward_speed_mean_mps"] = (
            float(np.mean(downward_speeds)) if downward_speeds else 0.0
        )
        summary["precontact_downward_speed_p95_mps"] = (
            float(np.percentile(downward_speeds, 95)) if downward_speeds else 0.0
        )
        summary["landing_velocity_saturation_overlap_rate"] = (
            sum(int(event["velocity_saturated_joint_count"]) > 0 for event in events) / len(events)
            if events else 0.0
        )
        summary["landing_torque_saturation_overlap_rate"] = (
            sum(int(event["torque_saturated_joint_count"]) > 0 for event in events) / len(events)
            if events else 0.0
        )
        high_impact_events = [event for event in events if bool(event["over_3500_n"])]
        summary["high_impact_velocity_saturation_overlap_rate"] = (
            sum(int(event["velocity_saturated_joint_count"]) > 0 for event in high_impact_events)
            / len(high_impact_events)
            if high_impact_events else 0.0
        )
        summary["high_impact_torque_saturation_overlap_rate"] = (
            sum(int(event["torque_saturated_joint_count"]) > 0 for event in high_impact_events)
            / len(high_impact_events)
            if high_impact_events else 0.0
        )
        summary["high_impact_within_0p5s_of_fall_rate"] = (
            sum(
                bool(event["fell_later"])
                and event["time_to_fall_s"] != ""
                and float(event["time_to_fall_s"]) <= 0.5
                for event in high_impact_events
            )
            / len(high_impact_events)
            if high_impact_events else 0.0
        )
    output_dir = (
        Path(args_cli.output_dir)
        if args_cli.output_dir
        else REPOSITORY_ROOT
        / "results"
        / "exp_005_unitree_g1_flat_run"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if flight_event_records:
        with (output_dir / "flight_events.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flight_event_records[0]))
            writer.writeheader()
            writer.writerows(flight_event_records)
    if landing_event_records:
        with (output_dir / "landing_events.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(landing_event_records[0]))
            writer.writeheader()
            writer.writerows(landing_event_records)
    if temporal_event_records:
        with (output_dir / "temporal_events.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(temporal_event_records[0]))
            writer.writeheader()
            writer.writerows(temporal_event_records)
    gate_value_fields = {
        "completed": ("completed", "true"),
        "not_fallen": ("fell", "false"),
        "episode_length": ("episode_length_steps", ">=900"),
        "speed_error": ("steady_mean_abs_forward_error_mps", "<=0.25 m/s after 2 s"),
        "flight_events": ("flight_phase_event_count", ">=4"),
        "safe_cycles": ("max_consecutive_running_cycles", ">=3"),
        "alternating": ("alternating_contact_rate", ">=0.80"),
        "normal_landing": ("normal_landing_rate", ">=0.80"),
        "flight_duration": ("mean_flight_duration_s", "0.04..0.16 s"),
        "slip": ("mean_contact_foot_slip_mps", "<=0.55 m/s"),
        "joint_velocity_saturation": ("max_velocity_limit_fraction", "<=0.05"),
        "joint_torque_saturation": ("max_torque_limit_fraction", "<=0.20"),
        "impact_p95": ("landing_impact_p95_n", "<=3500 N"),
        "impact_exceedance": ("landing_impact_over_3500_rate", "<=0.05"),
        "vertical_excursion": ("base_vertical_excursion_m", "<=0.30 m"),
        "stride_asymmetry": ("stride_asymmetry", "<=0.20"),
        "contact_time_asymmetry": ("contact_time_asymmetry", "<=0.20"),
    }
    quality_gate_rows = []
    for record in records:
        for gate_name, (value_field, criterion) in gate_value_fields.items():
            quality_gate_rows.append(
                {
                    "command_speed_mps": record["command_speed_mps"],
                    "episode": record["episode"],
                    "classification": record["classification"],
                    "gate": gate_name,
                    "passed": record[f"gate_{gate_name}_passed"],
                    "value": record[value_field],
                    "criterion": criterion,
                }
            )
    with (output_dir / "quality_gates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(quality_gate_rows[0]))
        writer.writeheader()
        writer.writerows(quality_gate_rows)
    joint_rows = []
    for record in records:
        speed_values = json.loads(str(record["joint_max_speed_rad_s"]))
        speed_ratios = json.loads(str(record["joint_max_speed_limit_ratio"]))
        torque_values = json.loads(str(record["joint_max_torque_nm"]))
        torque_ratios = json.loads(str(record["joint_max_torque_limit_ratio"]))
        velocity_limit_fractions = json.loads(str(record["joint_velocity_limit_fraction"]))
        torque_limit_fractions = json.loads(str(record["joint_torque_limit_fraction"]))
        for joint_name in speed_values:
            joint_rows.append(
                {
                    "command_speed_mps": record["command_speed_mps"],
                    "episode": record["episode"],
                    "joint": joint_name,
                    "max_speed_rad_s": speed_values[joint_name],
                    "max_speed_limit_ratio": speed_ratios[joint_name],
                    "max_torque_nm": torque_values[joint_name],
                    "max_torque_limit_ratio": torque_ratios[joint_name],
                    "velocity_limit_fraction": velocity_limit_fractions[joint_name],
                    "velocity_limit_time_s": velocity_limit_fractions[joint_name] * float(record["episode_length_steps"]) * 0.02,
                    "torque_limit_fraction": torque_limit_fractions[joint_name],
                    "torque_limit_time_s": torque_limit_fractions[joint_name] * float(record["episode_length_steps"]) * 0.02,
                }
            )
    with (output_dir / "joint_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joint_rows[0]))
        writer.writeheader()
        writer.writerows(joint_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "task": args_cli.task,
                "condition_name": args_cli.condition_name,
                "robustness": robustness_metadata,
                "steps_limit": args_cli.max_steps,
                "flight_event_records": len(flight_event_records),
                "landing_event_records": len(landing_event_records),
                "temporal_event_records": len(temporal_event_records),
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Evaluation output: {output_dir}")
    for summary in summaries:
        print(
            f"command={summary['command_speed_mps']:.2f} m/s "
            f"actual={summary['mean_actual_speed_mps']:.2f} m/s "
            f"error={summary['mean_abs_speed_error_mps']:.2f} m/s "
            f"fall_rate={summary['fall_rate']:.2f} "
            f"flight={summary['flight_phase_fraction']:.3f} "
            f"events={summary['flight_phase_event_count']:.1f} "
            f"20/40/60/80+={summary['flight_events_20ms']:.1f}/"
            f"{summary['flight_events_40ms']:.1f}/{summary['flight_events_60ms']:.1f}/"
            f"{summary['flight_events_ge_80ms']:.1f} "
            f"safe_fires={summary['safe_periodic_flight_fire_count']:.1f} "
            f"cycles={summary['max_consecutive_running_cycles']:.1f} "
            f"slip={summary['mean_contact_foot_slip_mps']:.2f} m/s "
            f"joint_v/t={summary['max_joint_speed_limit_ratio']:.2f}/"
            f"{summary['max_joint_torque_limit_ratio']:.2f} "
            f"impact={max(summary['left_max_landing_impact_n'], summary['right_max_landing_impact_n']):.0f} N "
            f"classification={summary['classification']}"
        )


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).resolve(strict=True)
    if args_cli.parallel_envs_per_speed < 1:
        raise ValueError("parallel_envs_per_speed must be at least one")
    if args_cli.episodes_per_speed % args_cli.parallel_envs_per_speed != 0:
        raise ValueError("episodes_per_speed must be divisible by parallel_envs_per_speed")
    episodes_per_env = args_cli.episodes_per_speed // args_cli.parallel_envs_per_speed
    evaluation_speeds = [
        speed
        for speed in args_cli.speeds
        for _ in range(args_cli.parallel_envs_per_speed)
    ]
    episode_offsets = [
        replica * episodes_per_env
        for _ in args_cli.speeds
        for replica in range(args_cli.parallel_envs_per_speed)
    ]
    num_envs = len(evaluation_speeds)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    robustness_metadata = apply_robustness_config(
        env_cfg,
        num_envs=num_envs,
        friction_scale=args_cli.friction_scale,
        mass_scale=args_cli.mass_scale,
        com_shift_x_m=args_cli.com_shift_x_m,
        stiffness_scale=args_cli.stiffness_scale,
        damping_scale=args_cli.damping_scale,
        small_rough_terrain=args_cli.small_rough_terrain,
    )
    robustness_metadata.update(
        {
            "action_delay_steps": args_cli.action_delay_steps,
            "external_force_axis": args_cli.external_force_axis,
            "external_force_n": args_cli.external_force_n,
            "external_force_start_s": args_cli.external_force_start_s,
            "external_force_duration_s": args_cli.external_force_duration_s,
        }
    )
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        agent_cfg.device = raw_env.unwrapped.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=raw_env.unwrapped.device)

        unwrapped = raw_env.unwrapped
        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene.sensors["contact_forces"]
        foot_body_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        torso_body_ids, _ = robot.find_bodies("torso_link")
        torso_body_ids_tensor = torch.tensor(torso_body_ids, dtype=torch.int32, device=unwrapped.device)
        sensor_foot_ids = [contact_sensor.body_names.index(name) for name in foot_names]
        command_term = unwrapped.command_manager.get_term("base_velocity")
        fixed_speeds = torch.tensor(evaluation_speeds, device=unwrapped.device)

        def reward_term_cfg(name: str):
            try:
                return unwrapped.reward_manager.get_term_cfg(name)
            except (KeyError, ValueError):
                return None

        feet_slide_cfg = reward_term_cfg("feet_slide")
        high_speed_feet_slide_cfg = reward_term_cfg("high_speed_feet_slide")
        quality_gated_excess_slip_cfg = reward_term_cfg("quality_gated_excess_slip")
        track_lin_vel_cfg = reward_term_cfg("track_lin_vel_xy_exp")
        feet_slide_weight = float(feet_slide_cfg.weight) if feet_slide_cfg is not None else 0.0
        high_speed_feet_slide_weight = (
            float(high_speed_feet_slide_cfg.weight) if high_speed_feet_slide_cfg is not None else 0.0
        )
        track_lin_vel_weight = float(track_lin_vel_cfg.weight) if track_lin_vel_cfg is not None else 0.0
        quality_gated_excess_slip_weight = (
            float(quality_gated_excess_slip_cfg.weight)
            if quality_gated_excess_slip_cfg is not None
            else 0.0
        )
        track_lin_vel_std = (
            float(track_lin_vel_cfg.params.get("std", 0.5)) if track_lin_vel_cfg is not None else 0.5
        )
        acceptable_tracking_error = (
            float(track_lin_vel_cfg.params.get("acceptable_error_mps", 0.0))
            if track_lin_vel_cfg is not None
            else 0.0
        )
        excess_slip_params = (
            quality_gated_excess_slip_cfg.params if quality_gated_excess_slip_cfg is not None else {}
        )
        excess_slip_threshold = float(excess_slip_params.get("slip_threshold_mps", 0.50))
        excess_slip_min_command = float(excess_slip_params.get("min_command_speed_mps", 4.40))
        excess_slip_max_error = float(excess_slip_params.get("max_tracking_error_mps", 0.25))

        metrics = _zero_metrics(num_envs, unwrapped.device)
        high_speed_metrics = _zero_high_speed_metrics(num_envs, len(robot.joint_names), unwrapped.device)
        previous_in_flight = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
        previous_contacts = torch.zeros((num_envs, 2), dtype=torch.bool, device=unwrapped.device)
        flight_duration = torch.zeros(num_envs, device=unwrapped.device)
        event_precursor_reward = torch.zeros(num_envs, device=unwrapped.device)
        takeoff_foot = torch.full((num_envs,), -1, dtype=torch.long, device=unwrapped.device)
        last_landing_foot = torch.full((num_envs,), -1, dtype=torch.long, device=unwrapped.device)
        consecutive_running_cycles = torch.zeros(num_envs, device=unwrapped.device)
        last_landing_xy = torch.zeros((num_envs, 2, 2), device=unwrapped.device)
        has_landing_xy = torch.zeros((num_envs, 2), dtype=torch.bool, device=unwrapped.device)
        flight_durations_ms: list[list[int]] = [[] for _ in range(num_envs)]
        landing_sequences: list[list[str]] = [[] for _ in range(num_envs)]
        landing_impacts: list[list[list[float]]] = [[[], []] for _ in range(num_envs)]
        precontact_downward_speeds: list[list[list[float]]] = [[[], []] for _ in range(num_envs)]
        flight_event_records: list[dict[str, float | int | bool | str]] = []
        landing_event_records: list[dict[str, float | int | bool | str]] = []
        temporal_event_records: list[dict[str, float | int | bool | str]] = []
        episode_counts = [0] * num_envs
        active = torch.ones(num_envs, dtype=torch.bool, device=unwrapped.device)
        records: list[dict[str, float | int | bool]] = []
        wrapped_env.reset()
        previous_foot_vertical_velocity = robot.data.body_lin_vel_w.torch[:, foot_body_ids, 2].clone()
        delayed_action_history: list[torch.Tensor] | None = None
        recovery_streak = torch.zeros(num_envs, device=unwrapped.device)

        stage4_shaping = any(
            stage in args_cli.task
            for stage in ("Stage4", "Stage5", "Stage6", "Stage7", "Stage8", "Stage9")
        )
        continuous_cycle_reward = "Stage9" in args_cli.task
        precursor_reward_per_step = 0.25 if stage4_shaping else 0.0
        takeoff_precursor_reward_per_step = 0.05 if stage4_shaping else 0.0
        precursor_event_cap = 0.75 if stage4_shaping else 0.0
        completion_reward_value = 2.0 if stage4_shaping else 1.0
        excess_flight_penalty_per_step = 0.25 if stage4_shaping else 0.0
        steady_state_start_steps = max(int(round(args_cli.steady_state_start_s / unwrapped.step_dt)), 0)
        force_start_steps = max(int(round(args_cli.external_force_start_s / unwrapped.step_dt)), 0)
        force_duration_steps = max(
            int(round(args_cli.external_force_duration_s / unwrapped.step_dt)), 1
        )
        force_end_steps = force_start_steps + force_duration_steps
        recovery_hold_steps = max(int(round(0.50 / unwrapped.step_dt)), 1)

        for _ in range(args_cli.max_steps):
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0] = fixed_speeds
            observations = wrapped_env.get_observations()

            with torch.inference_mode():
                policy_actions = policy(observations)
            policy_actions = policy_actions.clone()
            if args_cli.action_delay_steps > 0:
                if delayed_action_history is None:
                    delayed_action_history = [
                        torch.zeros_like(policy_actions) for _ in range(args_cli.action_delay_steps)
                    ]
                actions = delayed_action_history.pop(0)
                delayed_action_history.append(policy_actions)
            else:
                actions = policy_actions

            if args_cli.external_force_axis != "none":
                robot.permanent_wrench_composer.reset()
                episode_steps = metrics["steps"].long()
                force_active = (
                    active
                    & (episode_steps >= force_start_steps)
                    & (episode_steps < force_end_steps)
                )
                force_env_ids = torch.nonzero(force_active, as_tuple=False).flatten().to(torch.int32)
                if len(force_env_ids) > 0:
                    signed_forces = torch.zeros(
                        (len(force_env_ids), 1, 3), device=unwrapped.device
                    )
                    global_episode_ids = torch.tensor(
                        [
                            episode_offsets[env_id] + episode_counts[env_id]
                            for env_id in force_env_ids.tolist()
                        ],
                        device=unwrapped.device,
                    )
                    signs = torch.where(
                        global_episode_ids.remainder(2) == 0,
                        torch.ones_like(global_episode_ids, dtype=torch.float32),
                        -torch.ones_like(global_episode_ids, dtype=torch.float32),
                    )
                    axis_index = 0 if args_cli.external_force_axis == "x" else 1
                    signed_forces[:, 0, axis_index] = signs * args_cli.external_force_n
                    robot.permanent_wrench_composer.add_forces_and_torques_index(
                        forces=signed_forces,
                        body_ids=torso_body_ids_tensor,
                        env_ids=force_env_ids,
                        is_global=True,
                    )
                    metrics["external_force_applied"][force_env_ids.long()] = 1.0
                    metrics["external_force_signed_n"][force_env_ids.long()] = (
                        signs * args_cli.external_force_n
                    )
            with torch.inference_mode():
                _, rewards, dones, _ = wrapped_env.step(actions)

            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0] = fixed_speeds

            actual_speed = robot.data.root_lin_vel_b.torch[:, 0]
            yaw_frame_speed = quat_apply_inverse(
                yaw_quat(robot.data.root_quat_w.torch), robot.data.root_lin_vel_w.torch[:, :3]
            )
            yaw_frame_forward_speed = yaw_frame_speed[:, 0]
            lateral_speed = yaw_frame_speed[:, 1]
            yaw_rate_error = robot.data.root_ang_vel_w.torch[:, 2]
            diagnostic_speed = yaw_frame_forward_speed if stage4_shaping else actual_speed
            forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_foot_ids, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 1.0
            # Three physics samples (15 ms with the current configuration) suppress
            # single-sample PhysX spikes while preserving the landing impulse.
            robust_vertical_force = forces[:, :, :, 2].abs().mean(dim=1)
            raw_forces = contact_sensor.data.net_forces_w.torch[:, sensor_foot_ids, :]
            raw_contacts = raw_forces.norm(dim=-1) > 1.0
            contact_count = contacts.sum(dim=1)
            foot_speed = robot.data.body_lin_vel_w.torch[:, foot_body_ids, :2].norm(dim=-1)
            feet_slide_raw = (foot_speed * contacts).sum(dim=1)
            high_speed_ramp = torch.clamp((fixed_speeds - 4.40) / 0.10, min=0.0, max=1.0)
            # This counterfactual raw value is reported for every stage.  Its
            # weight is zero when the configured reward term is absent.
            high_speed_feet_slide_raw = feet_slide_raw * high_speed_ramp
            tracking_error_sq = (fixed_speeds - yaw_frame_forward_speed).square() + yaw_frame_speed[:, 1].square()
            xy_tracking_error = tracking_error_sq.sqrt()
            cycle_tracking_quality = torch.exp(
                -((yaw_frame_forward_speed - fixed_speeds) / 0.30).square()
                -(lateral_speed / 0.20).square()
            )
            if acceptable_tracking_error > 0.0:
                tracking_reward_error_sq = torch.maximum(
                    tracking_error_sq,
                    torch.full_like(tracking_error_sq, acceptable_tracking_error**2),
                )
            else:
                tracking_reward_error_sq = tracking_error_sq
            track_lin_vel_raw = torch.exp(-tracking_reward_error_sq / track_lin_vel_std**2)
            excess_slip_eligible = (fixed_speeds >= excess_slip_min_command) & (
                tracking_error_sq.sqrt() <= excess_slip_max_error
            )
            normalized_slip_excess = torch.relu(foot_speed - excess_slip_threshold) / excess_slip_threshold
            quality_gated_excess_slip_raw = (
                normalized_slip_excess.square() * contacts
            ).sum(dim=1) * excess_slip_eligible
            first_contact = contacts & ~previous_contacts
            first_air = contact_sensor.compute_first_air(unwrapped.step_dt).torch[:, sensor_foot_ids].clone()
            last_contact = contact_sensor.data.last_contact_time.torch[:, sensor_foot_ids]
            current_air = contact_sensor.data.current_air_time.torch[:, sensor_foot_ids]
            simultaneous_flight_duration = current_air.amin(dim=1)
            in_flight = contact_count == 0
            flight_started = in_flight & ~previous_in_flight
            landed = ~in_flight & previous_in_flight
            previous_single_support = previous_contacts.sum(dim=1) == 1
            takeoff_foot[flight_started] = torch.where(
                previous_single_support[flight_started],
                previous_contacts[flight_started].to(torch.int64).argmax(dim=1),
                -1,
            )
            event_precursor_reward[flight_started] = 0.0
            flight_duration[in_flight & active] += unwrapped.step_dt
            duration_steps = torch.round(flight_duration / unwrapped.step_dt).to(torch.long)
            landing_foot = contacts.to(torch.int64).argmax(dim=1)
            single_foot_landing = landed & (contact_count == 1) & active
            has_previous_landing = last_landing_foot >= 0
            alternation_opportunity = single_foot_landing & has_previous_landing
            alternating_landing = alternation_opportunity & (landing_foot != last_landing_foot)

            torso_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b.torch[:, 2], -1.0, 1.0))
            absolute_vertical_speed = robot.data.root_lin_vel_b.torch[:, 2].abs()
            short_flight = (flight_duration >= 0.04 - 1.0e-6) & (flight_duration <= 0.16 + 1.0e-6)
            normal_landing = (
                single_foot_landing
                & short_flight
                & ((diagnostic_speed - fixed_speeds).abs() <= 0.30)
                & (torso_tilt <= 0.20)
                & (absolute_vertical_speed <= 0.50)
            )
            safe_running_cycle = normal_landing & alternating_landing
            consecutive_running_cycles[landed & ~safe_running_cycle] = 0.0
            consecutive_running_cycles[safe_running_cycle] += 1.0

            active_float = active.float()
            joint_speed = robot.data.joint_vel.torch.abs()
            joint_speed_limit = torch.clamp(robot.data.joint_vel_limits.torch.abs(), min=1.0e-6)
            joint_torque = robot.data.applied_torque.torch.abs()
            joint_torque_limit = torch.clamp(robot.data.joint_effort_limits.torch.abs(), min=1.0e-6)
            joint_speed_ratio = joint_speed / joint_speed_limit
            joint_torque_ratio = joint_torque / joint_torque_limit
            high_speed_metrics["max_joint_speed"] = torch.maximum(
                high_speed_metrics["max_joint_speed"], joint_speed * active_float[:, None]
            )
            high_speed_metrics["max_joint_speed_ratio"] = torch.maximum(
                high_speed_metrics["max_joint_speed_ratio"], joint_speed_ratio * active_float[:, None]
            )
            high_speed_metrics["max_joint_torque"] = torch.maximum(
                high_speed_metrics["max_joint_torque"], joint_torque * active_float[:, None]
            )
            high_speed_metrics["max_joint_torque_ratio"] = torch.maximum(
                high_speed_metrics["max_joint_torque_ratio"], joint_torque_ratio * active_float[:, None]
            )
            high_speed_metrics["velocity_limit_steps"] += (joint_speed_ratio >= 0.95).float() * active_float[:, None]
            high_speed_metrics["torque_limit_steps"] += (joint_torque_ratio >= 0.95).float() * active_float[:, None]
            current_horizontal_force = torch.linalg.norm(raw_forces[:, :, :2], dim=-1)
            current_vertical_force = raw_forces[:, :, 2].abs()
            velocity_saturated = joint_speed_ratio >= 0.95
            torque_saturated = joint_torque_ratio >= 0.95
            foot_xy = robot.data.body_pos_w.torch[:, foot_body_ids, :2]
            for foot_id, side in enumerate(("left", "right")):
                contact_active = contacts[:, foot_id].float() * active_float
                landing_active = first_contact[:, foot_id] & active
                high_speed_metrics[f"{side}_contact_steps"] += contact_active
                high_speed_metrics[f"{side}_contact_events"] += landing_active.float()
                high_speed_metrics[f"{side}_horizontal_force_sum"] += current_horizontal_force[:, foot_id] * contact_active
                high_speed_metrics[f"{side}_vertical_force_sum"] += current_vertical_force[:, foot_id] * contact_active
                high_speed_metrics[f"{side}_force_samples"] += contact_active
                high_speed_metrics[f"{side}_max_impact"] = torch.maximum(
                    high_speed_metrics[f"{side}_max_impact"], current_vertical_force[:, foot_id] * landing_active
                )
                high_speed_metrics[f"{side}_slip_sum"] += foot_speed[:, foot_id] * contact_active
                high_speed_metrics[f"{side}_slip_samples"] += contact_active
                has_previous = landing_active & has_landing_xy[:, foot_id]
                stride = torch.linalg.norm(foot_xy[:, foot_id] - last_landing_xy[:, foot_id], dim=-1)
                high_speed_metrics[f"{side}_stride_sum"] += stride * has_previous
                high_speed_metrics[f"{side}_stride_count"] += has_previous.float()
                last_landing_xy[landing_active, foot_id] = foot_xy[landing_active, foot_id]
                has_landing_xy[landing_active, foot_id] = True
                true_landing = landing_active & previous_in_flight
                for env_id in torch.nonzero(true_landing, as_tuple=False).flatten().tolist():
                    impact = float(current_vertical_force[env_id, foot_id].item())
                    short_mean_impact = float(robust_vertical_force[env_id, foot_id].item())
                    precontact_downward = max(
                        -float(previous_foot_vertical_velocity[env_id, foot_id].item()), 0.0
                    )
                    landing_impacts[env_id][foot_id].append(impact)
                    precontact_downward_speeds[env_id][foot_id].append(precontact_downward)
                    velocity_names = [
                        robot.joint_names[index]
                        for index in torch.nonzero(velocity_saturated[env_id], as_tuple=False).flatten().tolist()
                    ]
                    torque_names = [
                        robot.joint_names[index]
                        for index in torch.nonzero(torque_saturated[env_id], as_tuple=False).flatten().tolist()
                    ]
                    landing_event_records.append(
                        {
                            "command_speed_mps": evaluation_speeds[env_id],
                            "episode": episode_offsets[env_id] + episode_counts[env_id],
                            "step": int(metrics["steps"][env_id].item()),
                            "time_s": float(metrics["steps"][env_id].item()) * unwrapped.step_dt,
                            "foot": "L" if foot_id == 0 else "R",
                            "landing_impact_n": impact,
                            "landing_impact_short_mean_n": short_mean_impact,
                            "over_3500_n": impact > 3500.0,
                            "precontact_vertical_speed_mps": float(
                                previous_foot_vertical_velocity[env_id, foot_id].item()
                            ),
                            "precontact_downward_speed_mps": precontact_downward,
                            "velocity_saturated_joint_count": len(velocity_names),
                            "torque_saturated_joint_count": len(torque_names),
                            "velocity_saturated_joints": ",".join(velocity_names),
                            "torque_saturated_joints": ",".join(torque_names),
                            "max_joint_velocity_ratio": float(joint_speed_ratio[env_id].max().item()),
                            "max_joint_torque_ratio": float(joint_torque_ratio[env_id].max().item()),
                            "fell_later": False,
                            "time_to_fall_s": "",
                        }
                    )
            base_height = robot.data.root_pos_w.torch[:, 2]
            high_speed_metrics["min_base_height"] = torch.where(
                active, torch.minimum(high_speed_metrics["min_base_height"], base_height), high_speed_metrics["min_base_height"]
            )
            high_speed_metrics["max_base_height"] = torch.where(
                active, torch.maximum(high_speed_metrics["max_base_height"], base_height), high_speed_metrics["max_base_height"]
            )
            metrics["reward"] += rewards * active_float
            metrics["steps"] += active_float
            metrics["actual_speed"] += actual_speed * active_float
            metrics["command_speed"] += fixed_speeds * active_float
            metrics["absolute_error"] += (actual_speed - fixed_speeds).abs() * active_float
            forward_absolute_error = (yaw_frame_forward_speed - fixed_speeds).abs()
            metrics["yaw_forward_speed"] += yaw_frame_forward_speed * active_float
            metrics["yaw_forward_absolute_error"] += forward_absolute_error * active_float
            metrics["lateral_absolute_speed"] += lateral_speed.abs() * active_float
            metrics["lateral_squared_speed"] += lateral_speed.square() * active_float
            metrics["yaw_rate_absolute_error"] += yaw_rate_error.abs() * active_float
            metrics["yaw_rate_squared_error"] += yaw_rate_error.square() * active_float
            metrics["xy_tracking_error"] += xy_tracking_error * active_float
            steady = (metrics["steps"] >= steady_state_start_steps) & active
            steady_float = steady.float()
            metrics["steady_steps"] += steady_float
            metrics["steady_yaw_forward_speed"] += yaw_frame_forward_speed * steady_float
            metrics["steady_yaw_forward_absolute_error"] += forward_absolute_error * steady_float
            metrics["steady_lateral_absolute_speed"] += lateral_speed.abs() * steady_float
            metrics["steady_yaw_rate_absolute_error"] += yaw_rate_error.abs() * steady_float
            metrics["steady_xy_tracking_error"] += xy_tracking_error * steady_float
            if args_cli.external_force_axis != "none":
                after_force = (
                    active
                    & (metrics["steps"] >= force_end_steps)
                    & (metrics["external_force_applied"] > 0.0)
                    & (metrics["external_force_recovered"] == 0.0)
                )
                recovery_good = (
                    (forward_absolute_error <= 0.25)
                    & (lateral_speed.abs() <= 0.20)
                    & after_force
                )
                recovery_streak = torch.where(
                    recovery_good,
                    recovery_streak + 1.0,
                    torch.zeros_like(recovery_streak),
                )
                newly_recovered = after_force & (recovery_streak >= recovery_hold_steps)
                metrics["external_force_recovered"][newly_recovered] = 1.0
                metrics["external_force_recovery_time_s"][newly_recovered] = (
                    metrics["steps"][newly_recovered] - force_end_steps
                ) * unwrapped.step_dt
            metrics["flight_steps"] += (contact_count == 0).float() * active_float
            metrics["single_support_steps"] += (contact_count == 1).float() * active_float
            metrics["double_support_steps"] += (contact_count == 2).float() * active_float
            metrics["slip_speed"] += (foot_speed * contacts).sum(dim=1) * active_float
            metrics["contact_samples"] += contacts.sum(dim=1) * active_float
            metrics["feet_slide_raw"] += feet_slide_raw * active_float
            metrics["feet_slide_weighted"] += (
                feet_slide_raw * feet_slide_weight * unwrapped.step_dt * active_float
            )
            metrics["feet_slide_weight"] += feet_slide_weight * active_float
            metrics["high_speed_feet_slide_raw"] += high_speed_feet_slide_raw * active_float
            metrics["high_speed_feet_slide_weighted"] += (
                high_speed_feet_slide_raw * high_speed_feet_slide_weight * unwrapped.step_dt * active_float
            )
            metrics["high_speed_feet_slide_weight"] += high_speed_feet_slide_weight * active_float
            metrics["track_lin_vel_raw"] += track_lin_vel_raw * active_float
            metrics["track_lin_vel_weighted"] += (
                track_lin_vel_raw * track_lin_vel_weight * unwrapped.step_dt * active_float
            )
            metrics["track_lin_vel_weight"] += track_lin_vel_weight * active_float
            metrics["quality_gated_excess_slip_raw"] += quality_gated_excess_slip_raw * active_float
            metrics["quality_gated_excess_slip_weighted"] += (
                quality_gated_excess_slip_raw
                * quality_gated_excess_slip_weight
                * unwrapped.step_dt
                * active_float
            )
            metrics["quality_gated_excess_slip_weight"] += (
                quality_gated_excess_slip_weight * active_float
            )
            metrics["quality_gated_excess_slip_eligible_steps"] += (
                excess_slip_eligible.float() * active_float
            )
            metrics["contact_duration"] += (last_contact * first_air).sum(dim=1) * active_float
            metrics["contact_events"] += first_air.sum(dim=1) * active_float
            metrics["flight_events"] += flight_started.float() * active_float
            metrics["completed_flight_events"] += landed.float() * active_float
            metrics["flight_duration_sum"] += flight_duration * landed.float() * active_float
            metrics["flight_duration_20ms"] += (landed & (duration_steps == 1)).float() * active_float
            metrics["flight_duration_40ms"] += (landed & (duration_steps == 2)).float() * active_float
            metrics["flight_duration_60ms"] += (landed & (duration_steps == 3)).float() * active_float
            metrics["flight_duration_ge_80ms"] += (landed & (duration_steps >= 4)).float() * active_float
            metrics["max_flight_duration"] = torch.maximum(
                metrics["max_flight_duration"], simultaneous_flight_duration * active_float
            )
            metrics["alternation_opportunities"] += alternation_opportunity.float() * active_float
            metrics["alternating_landings"] += alternating_landing.float() * active_float
            metrics["normal_landings"] += normal_landing.float() * active_float
            metrics["max_consecutive_running_cycles"] = torch.maximum(
                metrics["max_consecutive_running_cycles"], consecutive_running_cycles * active_float
            )
            metrics["torso_tilt"] += torso_tilt * active_float
            metrics["max_torso_tilt"] = torch.maximum(metrics["max_torso_tilt"], torso_tilt * active_float)
            metrics["absolute_vertical_speed"] += absolute_vertical_speed * active_float
            metrics["max_absolute_vertical_speed"] = torch.maximum(
                metrics["max_absolute_vertical_speed"], absolute_vertical_speed * active_float
            )
            metrics["landing_left"] += (single_foot_landing & (landing_foot == 0)).float() * active_float
            metrics["landing_right"] += (single_foot_landing & (landing_foot == 1)).float() * active_float
            same_side_landing = (
                single_foot_landing & has_previous_landing & (landing_foot == last_landing_foot)
            )
            metrics["same_side_landings"] += same_side_landing.float() * active_float
            metrics["double_foot_landings"] += (landed & (contact_count == 2)).float() * active_float
            suspected_chatter = (
                landed & (duration_steps == 1) & single_foot_landing & (takeoff_foot == landing_foot)
            )
            metrics["suspected_chatter_events"] += suspected_chatter.float() * active_float
            metrics["contact_filter_disagreement_steps"] += (
                (raw_contacts != contacts).any(dim=1).float() * active_float
            )
            diagnostic_conditions = {
                "high_command": fixed_speeds >= 2.3 - 1.0e-6,
                "tracking": (diagnostic_speed - fixed_speeds).abs() <= 0.30,
                "torso_tilt": torso_tilt <= 0.20,
                "vertical_speed": absolute_vertical_speed <= 0.50,
                "short_flight": short_flight,
                "single_foot": single_foot_landing,
                "alternating": alternating_landing,
            }
            common_precursor_safety = (
                in_flight
                & diagnostic_conditions["high_command"]
                & diagnostic_conditions["torso_tilt"]
                & diagnostic_conditions["vertical_speed"]
                & (flight_duration <= 0.16 + 1.0e-6)
            )
            sustained_precursor = (
                common_precursor_safety
                & (
                    torch.ones_like(in_flight)
                    if continuous_cycle_reward
                    else (
                        (diagnostic_speed - fixed_speeds).abs()
                        <= (1.20 if stage4_shaping else 0.30)
                    )
                )
                & (flight_duration >= 0.04 - 1.0e-6)
            )
            safe_takeoff = (
                common_precursor_safety
                & (
                    torch.ones_like(in_flight)
                    if continuous_cycle_reward
                    else diagnostic_conditions["tracking"]
                )
                & (flight_duration < 0.04 - 1.0e-6)
            )
            remaining_precursor_cap = torch.clamp(precursor_event_cap - event_precursor_reward, min=0.0)
            requested_precursor = torch.where(
                sustained_precursor,
                torch.full_like(remaining_precursor_cap, precursor_reward_per_step),
                torch.where(
                    safe_takeoff,
                    torch.full_like(remaining_precursor_cap, takeoff_precursor_reward_per_step),
                    torch.zeros_like(remaining_precursor_cap),
                ),
            )
            if continuous_cycle_reward:
                requested_precursor *= cycle_tracking_quality
            precursor_reward = torch.minimum(
                requested_precursor, remaining_precursor_cap
            )
            event_precursor_reward += precursor_reward
            excess_flight = in_flight & (flight_duration > 0.16 + 1.0e-6)
            excess_penalty = excess_flight.float() * excess_flight_penalty_per_step
            reward_cycle = safe_running_cycle
            if continuous_cycle_reward:
                reward_cycle = (
                    alternating_landing
                    & single_foot_landing
                    & short_flight
                    & diagnostic_conditions["high_command"]
                    & diagnostic_conditions["torso_tilt"]
                    & diagnostic_conditions["vertical_speed"]
                )
            completion_raw_reward = reward_cycle.float() * completion_reward_value
            if continuous_cycle_reward:
                completion_raw_reward *= cycle_tracking_quality
            safe_raw_reward = precursor_reward + completion_raw_reward - excess_penalty

            metrics["safe_periodic_flight_fires"] += (safe_raw_reward != 0.0).float() * active_float
            metrics["safe_periodic_flight_raw_reward"] += safe_raw_reward * active_float
            metrics["safe_periodic_flight_weighted_reward"] += safe_raw_reward * unwrapped.step_dt * active_float
            metrics["precursor_reward_step_count"] += (precursor_reward > 0.0).float() * active_float
            metrics["precursor_raw_reward"] += precursor_reward * active_float
            metrics["completion_reward_fire_count"] += reward_cycle.float() * active_float
            metrics["completion_raw_reward"] += completion_raw_reward * active_float
            metrics["excess_flight_penalty_step_count"] += excess_flight.float() * active_float
            for name, condition in diagnostic_conditions.items():
                metrics[f"pass_{name}"] += (landed & condition).float() * active_float
                metrics[f"fail_{name}"] += (landed & ~condition).float() * active_float

            for env_id in torch.nonzero(landed & active, as_tuple=False).flatten().tolist():
                duration_ms = int(duration_steps[env_id].item()) * int(round(unwrapped.step_dt * 1000))
                flight_durations_ms[env_id].append(duration_ms)
                if int(contact_count[env_id].item()) == 2:
                    landing_token = "B"
                elif int(landing_foot[env_id].item()) == 0:
                    landing_token = "L"
                else:
                    landing_token = "R"
                landing_sequences[env_id].append(landing_token)

                if bool(safe_running_cycle[env_id].item()):
                    reset_reason = "none"
                elif not bool(single_foot_landing[env_id].item()):
                    reset_reason = "not_single_foot"
                elif not bool(short_flight[env_id].item()) and duration_steps[env_id].item() < 2:
                    reset_reason = "flight_too_short"
                elif not bool(short_flight[env_id].item()):
                    reset_reason = "flight_too_long"
                elif not bool(diagnostic_conditions["high_command"][env_id].item()):
                    reset_reason = "command_too_slow"
                elif abs(float(diagnostic_speed[env_id].item() - fixed_speeds[env_id].item())) > 0.30:
                    reset_reason = "tracking_error"
                elif float(torso_tilt[env_id].item()) > 0.20:
                    reset_reason = "torso_tilt"
                elif float(absolute_vertical_speed[env_id].item()) > 0.50:
                    reset_reason = "vertical_speed"
                elif not bool(has_previous_landing[env_id].item()):
                    reset_reason = "phase_initialization"
                else:
                    reset_reason = "same_side_landing"

                if reset_reason != "none":
                    metrics[f"reset_{reset_reason}"][env_id] += 1.0
                flight_event_records.append(
                    {
                        "command_speed_mps": evaluation_speeds[env_id],
                        "episode": episode_offsets[env_id] + episode_counts[env_id],
                        "event": len(flight_durations_ms[env_id]) - 1,
                        "duration_ms": duration_ms,
                        "takeoff_foot": "L" if takeoff_foot[env_id].item() == 0 else "R" if takeoff_foot[env_id].item() == 1 else "B_or_unknown",
                        "landing": landing_token,
                        "actual_speed_mps": float(actual_speed[env_id].item()),
                        "yaw_frame_speed_mps": float(yaw_frame_forward_speed[env_id].item()),
                        "yaw_frame_lateral_speed_mps": float(lateral_speed[env_id].item()),
                        "yaw_rate_error_rad_s": float(yaw_rate_error[env_id].item()),
                        "xy_tracking_error_mps": float(xy_tracking_error[env_id].item()),
                        "speed_error_mps": abs(
                            float(diagnostic_speed[env_id].item() - fixed_speeds[env_id].item())
                        ),
                        "torso_tilt_rad": float(torso_tilt[env_id].item()),
                        "abs_vertical_speed_mps": float(absolute_vertical_speed[env_id].item()),
                        **{f"pass_{name}": bool(condition[env_id].item()) for name, condition in diagnostic_conditions.items()},
                        "suspected_chatter": bool(suspected_chatter[env_id].item()),
                        "safe_periodic_flight_fired": bool(reward_cycle[env_id].item()),
                        "cycle_tracking_quality": float(cycle_tracking_quality[env_id].item()),
                        "event_precursor_raw_reward": float(event_precursor_reward[env_id].item()),
                        "completion_raw_reward": float(completion_raw_reward[env_id].item()),
                        "event_total_weighted_reward": float(
                            (event_precursor_reward[env_id] + completion_raw_reward[env_id]).item()
                        ) * unwrapped.step_dt,
                        "cycle_reset_reason": reset_reason,
                    }
                )

            last_landing_foot[single_foot_landing] = landing_foot[single_foot_landing]
            flight_duration[landed] = 0.0
            previous_in_flight.copy_(in_flight)
            previous_contacts.copy_(contacts)

            base_contact = unwrapped.termination_manager.get_term("base_contact")
            noteworthy = active & (
                first_contact.any(dim=1)
                | velocity_saturated.any(dim=1)
                | torque_saturated.any(dim=1)
                | (dones > 0)
            )
            for env_id in torch.nonzero(noteworthy, as_tuple=False).flatten().tolist():
                temporal_event_records.append(
                    {
                        "command_speed_mps": evaluation_speeds[env_id],
                        "episode": episode_offsets[env_id] + episode_counts[env_id],
                        "step": int(metrics["steps"][env_id].item()),
                        "time_s": float(metrics["steps"][env_id].item()) * unwrapped.step_dt,
                        "left_first_contact": bool(first_contact[env_id, 0].item()),
                        "right_first_contact": bool(first_contact[env_id, 1].item()),
                        "left_impact_short_mean_n": float(robust_vertical_force[env_id, 0].item()),
                        "right_impact_short_mean_n": float(robust_vertical_force[env_id, 1].item()),
                        "velocity_saturated": bool(velocity_saturated[env_id].any().item()),
                        "torque_saturated": bool(torque_saturated[env_id].any().item()),
                        "max_joint_velocity_ratio": float(joint_speed_ratio[env_id].max().item()),
                        "max_joint_torque_ratio": float(joint_torque_ratio[env_id].max().item()),
                        "terminated": bool((dones[env_id] > 0).item()),
                        "fell": bool(base_contact[env_id].item()),
                    }
                )
            for env_id in torch.nonzero((dones > 0) & active, as_tuple=False).flatten().tolist():
                records.append(
                    _record_episode(
                        env_id,
                        evaluation_speeds[env_id],
                        episode_offsets[env_id] + episode_counts[env_id],
                        metrics,
                        completed=True,
                        fell=bool(base_contact[env_id].item()),
                        flight_durations_ms=flight_durations_ms[env_id],
                        landing_sequence=landing_sequences[env_id],
                        high_speed_metrics=high_speed_metrics,
                        joint_names=robot.joint_names,
                        step_dt=unwrapped.step_dt,
                        landing_impacts=landing_impacts[env_id],
                        precontact_downward_speeds=precontact_downward_speeds[env_id],
                    )
                )
                episode_counts[env_id] += 1
                for values in metrics.values():
                    values[env_id] = 0.0
                _reset_high_speed_metrics(high_speed_metrics, env_id)
                previous_in_flight[env_id] = False
                previous_contacts[env_id] = False
                flight_duration[env_id] = 0.0
                event_precursor_reward[env_id] = 0.0
                takeoff_foot[env_id] = -1
                last_landing_foot[env_id] = -1
                consecutive_running_cycles[env_id] = 0.0
                last_landing_xy[env_id] = 0.0
                has_landing_xy[env_id] = False
                flight_durations_ms[env_id] = []
                landing_sequences[env_id] = []
                landing_impacts[env_id] = [[], []]
                precontact_downward_speeds[env_id] = [[], []]
                recovery_streak[env_id] = 0.0
                if delayed_action_history is not None:
                    for buffered_actions in delayed_action_history:
                        buffered_actions[env_id] = 0.0
                if episode_counts[env_id] >= episodes_per_env:
                    active[env_id] = False

            policy.reset(dones)
            previous_foot_vertical_velocity.copy_(
                robot.data.body_lin_vel_w.torch[:, foot_body_ids, 2]
            )
            if not active.any():
                break

        for env_id in torch.nonzero(active & (metrics["steps"] > 0), as_tuple=False).flatten().tolist():
            records.append(
                _record_episode(
                    env_id,
                    evaluation_speeds[env_id],
                    episode_offsets[env_id] + episode_counts[env_id],
                    metrics,
                    completed=False,
                    fell=False,
                    flight_durations_ms=flight_durations_ms[env_id],
                    landing_sequence=landing_sequences[env_id],
                    high_speed_metrics=high_speed_metrics,
                    joint_names=robot.joint_names,
                    step_dt=unwrapped.step_dt,
                    landing_impacts=landing_impacts[env_id],
                    precontact_downward_speeds=precontact_downward_speeds[env_id],
                )
            )

        _write_results(
            records,
            flight_event_records,
            landing_event_records,
            temporal_event_records,
            checkpoint,
            robustness_metadata,
        )
        wrapped_env.close()


if __name__ == "__main__":
    main()
