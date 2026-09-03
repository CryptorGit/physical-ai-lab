"""Evaluate individual command skills or RUN->TURN->RUN->STOP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner


SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from g1_command_skills.evaluation import FAILURE_CLASSES, classify_failure  # noqa: E402
from g1_command_skills.command_observation import apply_command_ablation  # noqa: E402
from g1_command_skills.fixed_feedback import StopFeedbackConfig, StopFixedFeedbackController  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


SKILL_NAMES = {0: "RUN", 1: "STOP", 2: "TURN", 3: "CROUCH", 4: "STEP_OVER", 5: "LAND"}
RUN, STOP, TURN = 0, 1, 2
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Motion-Flat-G1-Command-Sequence-Eval-v0")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="results/exp_006_unitree_g1_command_skills/evaluation")
parser.add_argument(
    "--command-ablation",
    choices=("normal", "new_command_zero", "legacy_command_zero", "all_command_zero", "shuffle", "zero"),
    default="normal",
)
parser.add_argument(
    "--stop-residual-ablation",
    choices=("current", "yaw_mask", "yaw_ankle_roll_mask", "lateral_mask", "symmetric"),
    default="current",
)
parser.add_argument("--stop-feedback-k-heading", type=float, default=0.0)
parser.add_argument("--stop-feedback-k-yaw-rate", type=float, default=0.0)
parser.add_argument("--stop-feedback-k-roll", type=float, default=0.0)
parser.add_argument("--stop-feedback-k-roll-rate", type=float, default=0.0)
parser.add_argument("--stop-feedback-alpha", type=float, default=1.0)
parser.add_argument("--stop-feedback-max-delta-per-step", type=float, default=1.0)
parser.add_argument("--stop-feedback-braking-scale", type=float, default=1.0)
parser.add_argument("--stop-feedback-hold-scale", type=float, default=1.0)
parser.add_argument("--stop-feedback-single-support-scale", type=float, default=1.0)
parser.add_argument("--stop-feedback-flight-scale", type=float, default=1.0)
parser.add_argument("--stop-feedback-yaw-soft-threshold", type=float, default=float("inf"))
parser.add_argument("--stop-feedback-yaw-hard-threshold", type=float, default=float("inf"))
parser.add_argument("--stop-feedback-hard-guard-mode", choices=("zero", "damping_only"), default="zero")
parser.add_argument("--stop-feedback-flight-hard-zero", action="store_true")
parser.add_argument("--stop-feedback-contact-zero-steps", type=int, default=0)
parser.add_argument("--stop-feedback-contact-ramp-steps", type=int, default=0)
parser.add_argument("--stop-feedback-hard-action-limit", type=float, default=0.03)
parser.add_argument("--stop-feedback-hard-disable-torso", action="store_true")
parser.add_argument("--stop-feedback-ankle-soft", type=float, default=1.0)
parser.add_argument("--stop-feedback-ankle-hard", type=float, default=1.01)
parser.add_argument("--stop-feedback-joint-velocity-soft", type=float, default=1.0)
parser.add_argument("--stop-feedback-joint-velocity-hard", type=float, default=1.01)
parser.add_argument("--stop-feedback-tilt-soft", type=float, default=10.0)
parser.add_argument("--stop-feedback-tilt-hard", type=float, default=11.0)
parser.add_argument("--stop-feedback-angular-soft", type=float, default=100.0)
parser.add_argument("--stop-feedback-angular-hard", type=float, default=101.0)
parser.add_argument("--stop-feedback-worsening-yaw-scale", type=float, default=1.0)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(round((len(ordered) - 1) * q / 100.0)), len(ordered) - 1)]


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator > 1.0e-12 else 0.0


def summarize_stop_entry_strata(records: list[dict]) -> dict[str, dict]:
    strata = {
        "in_range_le_1.4": [],
        "moderate_tail_1.4_1.8": [],
        "high_tail_gt_1.8": [],
    }
    for record in records:
        speed = float(record.get("stop_entry_speed_mps", 0.0))
        name = "in_range_le_1.4" if speed <= 1.4 else "moderate_tail_1.4_1.8" if speed <= 1.8 else "high_tail_gt_1.8"
        strata[name].append(record)
    result = {}
    for name, rows in strata.items():
        result[name] = {
            "count": len(rows),
            "success_rate": mean([float(row["success"]) for row in rows]),
            "fall_rate": mean([float(row["fall"]) for row in rows]),
            "saturation_failure_rate": mean([
                float(
                    row["joint_velocity_saturation_fraction"] > 0.05
                    or row["ankle_torque_saturation_fraction"] > 0.20
                )
                for row in rows
            ]),
            "heading_error_rad": mean([row["heading_error_rad"] for row in rows]),
            "stop_position_error_m": mean([row["stop_position_error_m"] for row in rows]),
            "stop_hold_end_speed_mps": mean([row["stop_hold_end_speed_mps"] for row in rows]),
            "stop_speed_mps": mean([row["stop_speed_mps"] for row in rows]),
            "stop_hold_success_rate": mean([float(row["stop_hold_success"]) for row in rows]),
            "parent_action_deviation_norm": mean([row["parent_action_deviation_norm"] for row in rows]),
            "yaw_rate_abs_p99_rps": mean([row["actual_yaw_rate_abs_p99_rps"] for row in rows]),
            "yaw_rate_abs_max_rps": max([row["actual_yaw_rate_abs_max_rps"] for row in rows], default=0.0),
        }
    return result


def new_segment(
    episode: int,
    index: int,
    skill: int,
    time_s: float,
    start_xy: list[float],
    start_heading: float,
    commanded_turn_angle_rad: float = 0.0,
) -> dict:
    return {
        "episode": episode,
        "segment_index": index,
        "skill": SKILL_NAMES[skill],
        "start_time_s": time_s,
        "start_xy": start_xy,
        "start_heading": start_heading,
        "speed_errors": [], "heading_errors": [], "stop_errors": [], "stop_speeds": [],
        "pelvis_errors": [], "tilts": [], "slips": [], "knee_saturation": [],
        "ankle_torque_saturation": [], "landing_impacts": [], "course_deviations": [],
        "path_lateral_errors": [], "path_heading_errors": [], "path_forward_velocities": [],
        "path_lateral_velocities": [], "residual_action_norms": [],
        "raw_residual_action_norms": [],
        "corrective_residual_norms": [], "parent_action_deviation_norms": [],
        "residual_path_correction_rms": [], "residual_propulsion_rms": [],
        "tracking_time_s": None, "stabilization_time_s": None,
        "previous_contacts": None, "strike_sides": [], "airborne_samples": 0,
        "commanded_turn_angle_rad": commanded_turn_angle_rad,
        "actual_accumulated_yaws": [], "turn_angle_errors": [], "turn_completion_time_s": None,
        "legacy_command_vx": [], "legacy_command_vy": [], "legacy_command_yaw_rate": [],
        "new_command_heading_sin": [], "new_command_heading_cos": [], "new_command_turn_angle": [],
        "stop_entry_speeds": [], "stop_initial_distances": [], "stop_required_decelerations": [],
        "stop_braking_targets": [], "stop_progress_values": [], "stop_hold_progress_values": [],
        "stop_curve": [], "saturation_joint_names": set(), "action_magnitudes": [],
        "feedback_norm_double": [], "feedback_norm_single": [], "feedback_norm_flight": [],
        "feedback_spike_guard_count": 0, "feedback_hard_guard_count": 0,
        "feedback_slew_limiter_count": 0,
    }


def finalize_segment(segment: dict, end_time_s: float, fall: bool = False) -> dict:
    duration = max(end_time_s - segment["start_time_s"], 0.0)
    strikes = segment["strike_sides"]
    alternating = sum(a != b for a, b in zip(strikes, strikes[1:]))
    periodic_running = len(strikes) >= 3 and alternating >= len(strikes) - 2
    absolute_lateral_errors = [abs(value) for value in segment["path_lateral_errors"]]
    stop_curve = segment["stop_curve"]
    braking_curve = [point for point in stop_curve if point.get("stop_phase") == "braking"]
    hold_curve = [point for point in stop_curve if point.get("stop_phase") == "hold"]
    braking_heading_errors = [abs(point["stop_heading_error_rad"]) for point in braking_curve]
    hold_heading_errors = [abs(point["stop_heading_error_rad"]) for point in hold_curve]
    raw_exit_speed = segment["stop_speeds"][-1] if segment["stop_speeds"] else 0.0
    braking_end_speed = braking_curve[-1]["speed_mps"] if braking_curve else raw_exit_speed
    hold_start_speed = hold_curve[0]["speed_mps"] if hold_curve else 0.0
    hold_end_speed = hold_curve[-1]["speed_mps"] if hold_curve else 0.0
    final_assessment_speed = hold_end_speed if hold_curve else raw_exit_speed
    hold_max_speed = max([point["speed_mps"] for point in hold_curve], default=0.0)
    heading_failure_point = next(
        (point for point in stop_curve if abs(point["stop_heading_error_rad"]) > 0.12), None
    )
    stop_heading_errors = [abs(point["stop_heading_error_rad"]) for point in stop_curve]
    signed_actual_yaw_rates = [point["actual_yaw_rate_rps"] for point in stop_curve]
    actual_yaw_rates = [abs(value) for value in signed_actual_yaw_rates]
    legacy_yaw_commands = [point["legacy_yaw_rate_command_rps"] for point in stop_curve]
    raw_stop_residual_norms = [point["stop_residual_norm"] for point in stop_curve]
    masked_stop_residual_norms = [point["masked_residual_norm"] for point in stop_curve]
    braking_tracking_errors = [
        abs(point["forward_speed_mps"] - point["braking_target_speed_mps"])
        for point in braking_curve
    ]
    record = {
        "episode": segment["episode"],
        "segment_index": segment["segment_index"],
        "skill": segment["skill"],
        "duration_s": duration,
        "success": False,
        "failure_class": "",
        "command_tracking_time_s": segment["tracking_time_s"] if segment["tracking_time_s"] is not None else duration,
        "speed_error_mps": mean(segment["speed_errors"]),
        "heading_error_rad": mean(segment["heading_errors"]),
        "mean_step_heading_error_rad": mean(segment["heading_errors"]),
        "stop_position_error_m": segment["stop_errors"][-1] if segment["stop_errors"] else 0.0,
        "stop_target_longitudinal_m": segment["stop_longitudinal"][-1] if segment.get("stop_longitudinal") else 0.0,
        "stop_speed_mps": final_assessment_speed,
        "stop_entry_speed_mps": segment["stop_entry_speeds"][0] if segment["stop_entry_speeds"] else 0.0,
        "stop_exit_speed_mps": segment["stop_speeds"][-1] if segment["stop_speeds"] else 0.0,
        "stop_phase_min_speed_mps": min(segment["stop_speeds"], default=0.0),
        "stop_braking_end_speed_mps": braking_end_speed,
        "stop_hold_start_speed_mps": hold_start_speed,
        "stop_hold_end_speed_mps": hold_end_speed,
        "stop_hold_observed": bool(hold_curve),
        "stop_hold_max_speed_mps": hold_max_speed,
        "stop_hold_max_speed_rebound_mps": max(hold_max_speed - hold_start_speed, 0.0) if hold_curve else 0.0,
        "braking_heading_error_mean_rad": mean(braking_heading_errors),
        "braking_heading_error_p95_rad": percentile(braking_heading_errors, 95),
        "braking_heading_error_max_rad": max(braking_heading_errors, default=0.0),
        "hold_heading_error_mean_rad": mean(hold_heading_errors),
        "hold_heading_error_p95_rad": percentile(hold_heading_errors, 95),
        "hold_heading_error_max_rad": max(hold_heading_errors, default=0.0),
        "stop_heading_error_mean_rad": mean(stop_heading_errors),
        "stop_heading_error_p95_rad": percentile(stop_heading_errors, 95),
        "stop_heading_error_max_rad": max(stop_heading_errors, default=0.0),
        "actual_yaw_rate_mean_rps": mean(signed_actual_yaw_rates),
        "actual_yaw_rate_abs_mean_rps": mean(actual_yaw_rates),
        "actual_yaw_rate_abs_p95_rps": percentile(actual_yaw_rates, 95),
        "actual_yaw_rate_abs_p99_rps": percentile(actual_yaw_rates, 99),
        "actual_yaw_rate_abs_max_rps": max(actual_yaw_rates, default=0.0),
        "actual_yaw_rate_over_1_5_fraction": mean([value > 1.5 for value in actual_yaw_rates]),
        "actual_yaw_rate_over_2_5_fraction": mean([value > 2.5 for value in actual_yaw_rates]),
        "actual_yaw_rate_over_4_0_fraction": mean([value > 4.0 for value in actual_yaw_rates]),
        "legacy_yaw_rate_command_mean_rps": mean(legacy_yaw_commands),
        "legacy_yaw_rate_command_abs_max_rps": max([abs(value) for value in legacy_yaw_commands], default=0.0),
        "stop_raw_residual_norm": mean(raw_stop_residual_norms),
        "stop_masked_residual_norm": mean(masked_stop_residual_norms),
        "braking_speed_tracking_error_mps": mean(braking_tracking_errors),
        "heading_failure_time_s": heading_failure_point["time_s"] if heading_failure_point else "",
        "fall_heading_error_rad": abs(stop_curve[-1]["stop_heading_error_rad"]) if fall and stop_curve else "",
        "fall_yaw_rate_rps": stop_curve[-1]["actual_yaw_rate_rps"] if fall and stop_curve else "",
        "stop_initial_distance_m": segment["stop_initial_distances"][0] if segment["stop_initial_distances"] else 0.0,
        "stop_required_deceleration_mps2": mean(segment["stop_required_decelerations"]),
        "stop_braking_target_exit_mps": segment["stop_braking_targets"][-1] if segment["stop_braking_targets"] else 0.0,
        "stop_hold_progress": max(segment["stop_hold_progress_values"], default=0.0),
        "stop_hold_success": max(segment["stop_hold_progress_values"], default=0.0) >= 1.0,
        "post_stop_hold_fall": bool(fall and max(segment["stop_hold_progress_values"], default=0.0) > 0.0),
        "stop_braking_curve_json": json.dumps(segment["stop_curve"], separators=(",", ":")),
        "pelvis_height_error_m": mean(segment["pelvis_errors"]),
        "obstacle_contact_rate": 0.0,
        "clearance_m": 0.0,
        "landing_impact_n": percentile(segment["landing_impacts"], 95),
        "recovery_time_s": segment["stabilization_time_s"] if segment["stabilization_time_s"] is not None else duration,
        "fall": fall,
        "fall_rate": 1.0 if fall else 0.0,
        "foot_slip_mps": mean(segment["slips"]),
        "joint_velocity_saturation_fraction": mean(segment["knee_saturation"]),
        "ankle_torque_saturation_fraction": mean(segment["ankle_torque_saturation"]),
        "saturation_joint_names": ";".join(sorted(segment["saturation_joint_names"])),
        "maximum_action_magnitude": max(segment["action_magnitudes"], default=0.0),
        "stabilization_after_transition_s": segment["stabilization_time_s"] if segment["stabilization_time_s"] is not None else duration,
        "course_deviation_m": max(absolute_lateral_errors, default=0.0),
        "path_lateral_error_mean": mean(absolute_lateral_errors),
        "path_lateral_error_p95": percentile(absolute_lateral_errors, 95),
        "path_lateral_error_max": max(absolute_lateral_errors, default=0.0),
        "path_lateral_error_signed_mean": mean(segment["path_lateral_errors"]),
        "path_heading_error": mean(segment["path_heading_errors"]),
        "path_forward_velocity": mean(segment["path_forward_velocities"]),
        "path_lateral_velocity": mean([abs(value) for value in segment["path_lateral_velocities"]]),
        "residual_action_norm": mean(segment["residual_action_norms"]),
        "raw_residual_action_norm": mean(segment["raw_residual_action_norms"]),
        "corrective_residual_norm": mean(segment["corrective_residual_norms"]),
        "parent_action_deviation_norm": mean(segment["parent_action_deviation_norms"]),
        "parent_action_deviation_norm_p95": percentile(segment["parent_action_deviation_norms"], 95),
        "parent_action_deviation_norm_max": max(segment["parent_action_deviation_norms"], default=0.0),
        "feedback_norm_double_support": mean(segment["feedback_norm_double"]),
        "feedback_norm_single_support": mean(segment["feedback_norm_single"]),
        "feedback_norm_flight": mean(segment["feedback_norm_flight"]),
        "feedback_spike_guard_count": segment["feedback_spike_guard_count"],
        "feedback_hard_guard_count": segment["feedback_hard_guard_count"],
        "feedback_slew_limiter_count": segment["feedback_slew_limiter_count"],
        "residual_action_norm_p95": percentile(segment["residual_action_norms"], 95),
        "residual_vs_lateral_error_correlation": correlation(
            absolute_lateral_errors, segment["residual_action_norms"]
        ),
        "residual_path_correction_rms": mean(segment["residual_path_correction_rms"]),
        "residual_propulsion_rms": mean(segment["residual_propulsion_rms"]),
        "tilt_rad": max(segment["tilts"], default=0.0),
        "timeout": False,
        "fall_segment": segment["skill"] if fall else "",
        "fall_time_s": end_time_s if fall else "",
        "first_failure_segment": "",
        "first_failure_time_s": "",
        "periodic_running": periodic_running,
        "coordinate_frame": "robot-local yaw frame; headings are wrapped target-minus-world-heading",
        "legacy_command_vx_mean": mean(segment["legacy_command_vx"]),
        "legacy_command_vy_mean": mean(segment["legacy_command_vy"]),
        "legacy_command_yaw_rate_mean": mean(segment["legacy_command_yaw_rate"]),
        "legacy_command_yaw_rate_signed_peak": max(
            segment["legacy_command_yaw_rate"], key=abs, default=0.0
        ),
        "new_command_heading_sin_mean": mean(segment["new_command_heading_sin"]),
        "new_command_heading_cos_mean": mean(segment["new_command_heading_cos"]),
        "new_command_turn_angle_mean": mean(segment["new_command_turn_angle"]),
    }
    commanded_turn = float(segment["commanded_turn_angle_rad"])
    actual_turn = segment["actual_accumulated_yaws"][-1] if segment["actual_accumulated_yaws"] else 0.0
    turn_errors = segment["turn_angle_errors"]
    record.update({
        "commanded_turn_angle_rad": commanded_turn,
        "actual_accumulated_yaw_rad": actual_turn,
        "final_turn_angle_error_rad": abs(commanded_turn - actual_turn) if segment["skill"] == "TURN" else 0.0,
        "maximum_turn_angle_error_rad": max(turn_errors, default=0.0),
        "turn_completion_time_s": (
            segment["turn_completion_time_s"] if segment["turn_completion_time_s"] is not None else duration
        ),
        "turn_success": False,
        "straight_recovery_success": False,
        "post_turn_heading_error_rad": 0.0,
        "post_turn_path_lateral_error_m": 0.0,
    })
    if segment["skill"] == "TURN":
        # Keep the generic heading field identical to the gate quantity.
        record["heading_error_rad"] = record["final_turn_angle_error_rad"]
    record["failure_class"] = classify_failure(record)
    record["success"] = record["failure_class"] == ""
    return record


def attach_turn_recovery(records: list[dict]) -> None:
    """Attach post-turn straight metrics, then classify TURN exactly once."""
    for index, record in enumerate(records):
        if record["skill"] != "TURN":
            continue
        next_run = records[index + 1] if index + 1 < len(records) and records[index + 1]["skill"] == "RUN" else None
        if next_run is not None:
            record["post_turn_heading_error_rad"] = next_run["path_heading_error"]
            record["post_turn_path_lateral_error_m"] = next_run["path_lateral_error_p95"]
            record["straight_recovery_success"] = bool(
                not next_run["fall"]
                and record["post_turn_heading_error_rad"] <= 0.12
                and record["post_turn_path_lateral_error_m"] <= 0.75
            )
        record["turn_success"] = bool(
            record["final_turn_angle_error_rad"] <= 0.12 and record["straight_recovery_success"]
        )
        record["failure_class"] = classify_failure(record)
        record["success"] = record["failure_class"] == ""
        record["turn_success"] = record["success"]


def make_episode_record(episode: int, records: list[dict], time_s: float, expected_segments: int) -> dict:
    """Finalize episode attribution from real segments only (never auto-reset state)."""
    attach_turn_recovery(records)
    first_failed = next((record for record in records if not record["success"]), None)
    first_failure_segment = first_failed["skill"] if first_failed else ""
    first_failure_time = (
        float(first_failed["duration_s"]) + sum(float(r["duration_s"]) for r in records[: records.index(first_failed)])
        if first_failed else ""
    )
    fall_record = next((record for record in records if record["fall"]), None)
    fall_segment = fall_record["skill"] if fall_record else ""
    fall_time = fall_record["fall_time_s"] if fall_record else ""
    for record in records:
        record["fall_segment"] = fall_segment
        record["fall_time_s"] = fall_time
        record["first_failure_segment"] = first_failure_segment
        record["first_failure_time_s"] = first_failure_time
    run_records = [record for record in records if record["skill"] == "RUN"]
    turn_record = next((record for record in records if record["skill"] == "TURN"), None)
    stop_record = next((record for record in records if record["skill"] == "STOP"), None)
    return {
        "episode": episode,
        "sequence_complete": first_failure_segment == "" and len(records) >= expected_segments,
        "first_failed_skill": first_failure_segment,
        "first_failure_reason": first_failed["failure_class"] if first_failed else "",
        "first_failure_segment": first_failure_segment,
        "first_failure_time_s": first_failure_time,
        "fall": fall_record is not None,
        "fall_segment": fall_segment,
        "fall_time_s": fall_time,
        "duration_s": time_s,
        "stop_entry_speed_mps": stop_record["stop_entry_speed_mps"] if stop_record else 0.0,
        "stop_exit_speed_mps": stop_record["stop_exit_speed_mps"] if stop_record else 0.0,
        "stop_phase_min_speed_mps": stop_record["stop_phase_min_speed_mps"] if stop_record else 0.0,
        "stop_braking_end_speed_mps": stop_record["stop_braking_end_speed_mps"] if stop_record else 0.0,
        "stop_hold_start_speed_mps": stop_record["stop_hold_start_speed_mps"] if stop_record else 0.0,
        "stop_hold_end_speed_mps": stop_record["stop_hold_end_speed_mps"] if stop_record else 0.0,
        "stop_hold_observed": stop_record["stop_hold_observed"] if stop_record else False,
        "stop_hold_max_speed_mps": stop_record["stop_hold_max_speed_mps"] if stop_record else 0.0,
        "stop_hold_max_speed_rebound_mps": stop_record["stop_hold_max_speed_rebound_mps"] if stop_record else 0.0,
        "braking_heading_error_mean_rad": stop_record["braking_heading_error_mean_rad"] if stop_record else 0.0,
        "braking_heading_error_p95_rad": stop_record["braking_heading_error_p95_rad"] if stop_record else 0.0,
        "braking_heading_error_max_rad": stop_record["braking_heading_error_max_rad"] if stop_record else 0.0,
        "hold_heading_error_mean_rad": stop_record["hold_heading_error_mean_rad"] if stop_record else 0.0,
        "hold_heading_error_p95_rad": stop_record["hold_heading_error_p95_rad"] if stop_record else 0.0,
        "hold_heading_error_max_rad": stop_record["hold_heading_error_max_rad"] if stop_record else 0.0,
        "stop_heading_error_mean_rad": stop_record["stop_heading_error_mean_rad"] if stop_record else 0.0,
        "stop_heading_error_p95_rad": stop_record["stop_heading_error_p95_rad"] if stop_record else 0.0,
        "stop_heading_error_max_rad": stop_record["stop_heading_error_max_rad"] if stop_record else 0.0,
        "actual_yaw_rate_mean_rps": stop_record["actual_yaw_rate_mean_rps"] if stop_record else 0.0,
        "actual_yaw_rate_abs_mean_rps": stop_record["actual_yaw_rate_abs_mean_rps"] if stop_record else 0.0,
        "actual_yaw_rate_abs_p95_rps": stop_record["actual_yaw_rate_abs_p95_rps"] if stop_record else 0.0,
        "actual_yaw_rate_abs_max_rps": stop_record["actual_yaw_rate_abs_max_rps"] if stop_record else 0.0,
        "legacy_yaw_rate_command_mean_rps": stop_record["legacy_yaw_rate_command_mean_rps"] if stop_record else 0.0,
        "legacy_yaw_rate_command_abs_max_rps": stop_record["legacy_yaw_rate_command_abs_max_rps"] if stop_record else 0.0,
        "stop_raw_residual_norm": stop_record["stop_raw_residual_norm"] if stop_record else 0.0,
        "stop_masked_residual_norm": stop_record["stop_masked_residual_norm"] if stop_record else 0.0,
        "braking_speed_tracking_error_mps": stop_record["braking_speed_tracking_error_mps"] if stop_record else 0.0,
        "heading_failure_time_s": stop_record["heading_failure_time_s"] if stop_record else "",
        "fall_heading_error_rad": stop_record["fall_heading_error_rad"] if stop_record else "",
        "fall_yaw_rate_rps": stop_record["fall_yaw_rate_rps"] if stop_record else "",
        "stop_initial_distance_m": stop_record["stop_initial_distance_m"] if stop_record else 0.0,
        "stop_required_deceleration_mps2": stop_record["stop_required_deceleration_mps2"] if stop_record else 0.0,
        "stop_hold_success": stop_record["stop_hold_success"] if stop_record else False,
        "post_stop_hold_fall": stop_record["post_stop_hold_fall"] if stop_record else False,
        "stop_braking_curve_json": stop_record["stop_braking_curve_json"] if stop_record else "[]",
        "saturation_joint_names": stop_record["saturation_joint_names"] if stop_record else "",
        "joint_velocity_saturation_fraction": stop_record["joint_velocity_saturation_fraction"] if stop_record else 0.0,
        "ankle_torque_saturation_fraction": stop_record["ankle_torque_saturation_fraction"] if stop_record else 0.0,
        "maximum_action_magnitude": stop_record["maximum_action_magnitude"] if stop_record else 0.0,
        "path_lateral_error_mean": mean([r["path_lateral_error_mean"] for r in run_records]),
        "path_lateral_error_p95": max([r["path_lateral_error_p95"] for r in run_records], default=0.0),
        "path_lateral_error_max": max([r["path_lateral_error_max"] for r in run_records], default=0.0),
        "path_heading_error": mean([r["path_heading_error"] for r in run_records]),
        "path_forward_velocity": mean([r["path_forward_velocity"] for r in run_records]),
        "path_lateral_velocity": mean([r["path_lateral_velocity"] for r in run_records]),
        "commanded_turn_angle_rad": turn_record["commanded_turn_angle_rad"] if turn_record else 0.0,
        "actual_accumulated_yaw_rad": turn_record["actual_accumulated_yaw_rad"] if turn_record else 0.0,
        "final_turn_angle_error_rad": turn_record["final_turn_angle_error_rad"] if turn_record else 0.0,
        "maximum_turn_angle_error_rad": turn_record["maximum_turn_angle_error_rad"] if turn_record else 0.0,
        "turn_completion_time_s": turn_record["turn_completion_time_s"] if turn_record else 0.0,
        "turn_success": turn_record["turn_success"] if turn_record else False,
        "straight_recovery_success": turn_record["straight_recovery_success"] if turn_record else False,
        "post_turn_heading_error_rad": turn_record["post_turn_heading_error_rad"] if turn_record else 0.0,
        "post_turn_path_lateral_error_m": turn_record["post_turn_path_lateral_error_m"] if turn_record else 0.0,
    }


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).resolve(strict=True)
    output = (REPOSITORY_ROOT / args_cli.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        unwrapped = raw_env.unwrapped
        agent_cfg.device = unwrapped.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        )
        policy = runner.get_inference_policy(device=unwrapped.device)
        actor = runner.alg.actor
        robot = unwrapped.scene["robot"]
        term = unwrapped.command_manager.get_term("base_velocity")
        contact = unwrapped.scene.sensors["contact_forces"]
        foot_body_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_foot_ids = [contact.body_names.index(name) for name in foot_names]
        pelvis_ids, _ = robot.find_bodies("pelvis")
        knee_ids, _ = robot.find_joints(".*_knee_joint")
        ankle_ids, ankle_names = robot.find_joints(".*_ankle_.*_joint")
        all_joint_ids, all_joint_names = robot.find_joints(".*")
        path_correction_ids, _ = robot.find_joints(".*_(hip_roll|hip_yaw|ankle_roll)_joint|torso_joint")
        propulsion_ids, _ = robot.find_joints(".*_(hip_pitch|knee|ankle_pitch)_joint")
        left_hip_yaw_ids, _ = robot.find_joints("left_hip_yaw_joint")
        right_hip_yaw_ids, _ = robot.find_joints("right_hip_yaw_joint")
        torso_yaw_ids, _ = robot.find_joints("torso_joint")
        left_ankle_roll_ids, _ = robot.find_joints("left_ankle_roll_joint")
        right_ankle_roll_ids, _ = robot.find_joints("right_ankle_roll_joint")
        left_hip_roll_ids, _ = robot.find_joints("left_hip_roll_joint")
        right_hip_roll_ids, _ = robot.find_joints("right_hip_roll_joint")
        left_hip_pitch_ids, _ = robot.find_joints("left_hip_pitch_joint")
        right_hip_pitch_ids, _ = robot.find_joints("right_hip_pitch_joint")
        left_knee_ids, _ = robot.find_joints("left_knee_joint")
        right_knee_ids, _ = robot.find_joints("right_knee_joint")
        left_ankle_pitch_ids, _ = robot.find_joints("left_ankle_pitch_joint")
        right_ankle_pitch_ids, _ = robot.find_joints("right_ankle_pitch_joint")
        actor.configure_stop_residual_ablation(args_cli.stop_residual_ablation, {
            "torso": torso_yaw_ids[0],
            "left_hip_yaw": left_hip_yaw_ids[0], "right_hip_yaw": right_hip_yaw_ids[0],
            "left_ankle_roll": left_ankle_roll_ids[0], "right_ankle_roll": right_ankle_roll_ids[0],
            "left_hip_roll": left_hip_roll_ids[0], "right_hip_roll": right_hip_roll_ids[0],
            "left_hip_pitch": left_hip_pitch_ids[0], "right_hip_pitch": right_hip_pitch_ids[0],
            "left_knee": left_knee_ids[0], "right_knee": right_knee_ids[0],
            "left_ankle_pitch": left_ankle_pitch_ids[0], "right_ankle_pitch": right_ankle_pitch_ids[0],
        })
        # Contact gating is deliberately outside the frozen actor because foot
        # contact is not part of the legacy 123-D policy observation.
        actor.configure_stop_fixed_feedback(0.0, 0.0, 0.0, 0.0)
        feedback_config = StopFeedbackConfig(
            k_heading=args_cli.stop_feedback_k_heading,
            k_yaw_rate=args_cli.stop_feedback_k_yaw_rate,
            alpha=args_cli.stop_feedback_alpha,
            max_delta_per_step=args_cli.stop_feedback_max_delta_per_step,
            braking_scale=args_cli.stop_feedback_braking_scale,
            hold_scale=args_cli.stop_feedback_hold_scale,
            single_support_scale=args_cli.stop_feedback_single_support_scale,
            flight_scale=args_cli.stop_feedback_flight_scale,
            yaw_soft_threshold=args_cli.stop_feedback_yaw_soft_threshold,
            yaw_hard_threshold=args_cli.stop_feedback_yaw_hard_threshold,
            hard_guard_mode=args_cli.stop_feedback_hard_guard_mode,
            flight_hard_zero=args_cli.stop_feedback_flight_hard_zero,
            contact_recovery_zero_steps=args_cli.stop_feedback_contact_zero_steps,
            contact_recovery_ramp_steps=args_cli.stop_feedback_contact_ramp_steps,
            hard_guard_action_limit=args_cli.stop_feedback_hard_action_limit,
            hard_guard_disable_torso=args_cli.stop_feedback_hard_disable_torso,
            ankle_utilization_soft=args_cli.stop_feedback_ankle_soft,
            ankle_utilization_hard=args_cli.stop_feedback_ankle_hard,
            joint_velocity_soft=args_cli.stop_feedback_joint_velocity_soft,
            joint_velocity_hard=args_cli.stop_feedback_joint_velocity_hard,
            tilt_soft_rad=args_cli.stop_feedback_tilt_soft,
            tilt_hard_rad=args_cli.stop_feedback_tilt_hard,
            angular_velocity_soft_rps=args_cli.stop_feedback_angular_soft,
            angular_velocity_hard_rps=args_cli.stop_feedback_angular_hard,
            worsening_yaw_scale=args_cli.stop_feedback_worsening_yaw_scale,
        )
        feedback_controller = StopFixedFeedbackController(1, 37, unwrapped.device, feedback_config)

        skill_records: list[dict] = []
        episode_records: list[dict] = []
        stop_curve_records: list[dict] = []
        wrapped.reset()
        episode = 0
        step_in_episode = 0
        segment_index = 0
        current_skill = int(term.skill_id[0].item())
        position = robot.data.root_pos_w.torch[0, :2].tolist()
        heading = float(robot.data.heading_w.torch[0].item())
        commanded_turn = float(term.commanded_turn_angle_rad[0].item()) if current_skill == TURN else 0.0
        turn_start_heading = float(term.turn_start_heading_w[0].item()) if current_skill == TURN else heading
        segment = new_segment(
            episode, segment_index, current_skill, 0.0, position, turn_start_heading, commanded_turn
        )

        while episode < args_cli.episodes:
            observations = wrapped.get_observations()
            source_policy = observations["policy"]
            segment["legacy_command_vx"].append(float(source_policy[0, 9].item()))
            segment["legacy_command_vy"].append(float(source_policy[0, 10].item()))
            segment["legacy_command_yaw_rate"].append(float(source_policy[0, 11].item()))
            segment["new_command_heading_sin"].append(float(source_policy[0, 123 + 12].item()))
            segment["new_command_heading_cos"].append(float(source_policy[0, 123 + 13].item()))
            segment["new_command_turn_angle"].append(float(source_policy[0, 123 + 14].item()))
            observations = apply_command_ablation(observations, args_cli.command_ablation, term)
            pre_step_forces = contact.data.net_forces_w_history.torch[0, :, sensor_foot_ids, :]
            pre_step_support_count = (
                pre_step_forces.norm(dim=-1).amax(dim=0) > 1.0
            ).sum().reshape(1)
            pre_step_contacts = (pre_step_forces.norm(dim=-1).amax(dim=0) > 1.0).reshape(1, 2)
            left_ankle_pair = [left_ankle_pitch_ids[0], left_ankle_roll_ids[0]]
            right_ankle_pair = [right_ankle_pitch_ids[0], right_ankle_roll_ids[0]]
            pre_step_ankle_utilization = torch.stack([
                (
                    robot.data.applied_torque.torch[:, ids].abs()
                    / robot.data.joint_effort_limits.torch[:, ids].abs().clamp_min(1.0e-6)
                ).amax(dim=1)
                for ids in (left_ankle_pair, right_ankle_pair)
            ], dim=1)
            pre_step_joint_velocity_utilization = (
                robot.data.joint_vel.torch[:, all_joint_ids].abs()
                / robot.data.joint_vel_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
            ).amax(dim=1)
            pre_gravity = robot.data.projected_gravity_b.torch
            pre_roll = torch.atan2(-pre_gravity[:, 1], -pre_gravity[:, 2])
            pre_pitch = torch.atan2(
                pre_gravity[:, 0], torch.sqrt(pre_gravity[:, 1].square() + pre_gravity[:, 2].square())
            )
            with torch.inference_mode():
                actor_components = actor.diagnostic_components(observations)
                actions = policy(observations)
            feedback_action, feedback_diagnostics = feedback_controller.step(
                observations["policy"], term.skill_id == STOP, term.stop_hold_progress, pre_step_support_count,
                contacts=pre_step_contacts, ankle_utilization=pre_step_ankle_utilization,
                joint_velocity_utilization=pre_step_joint_velocity_utilization,
                roll_pitch=torch.stack((pre_roll, pre_pitch), dim=1),
                angular_velocity=robot.data.root_ang_vel_b.torch,
            )
            actions = actions + feedback_action
            segment["action_magnitudes"].append(float(actions[0].abs().max().item()))
            _, _, dones, infos = wrapped.step(actions)
            step_in_episode += 1
            time_s = step_in_episode * float(unwrapped.step_dt)
            done = bool(dones[0].item())
            if done:
                timeout_tensor = infos.get("time_outs") if isinstance(infos, dict) else None
                timed_out = bool(timeout_tensor[0].item()) if timeout_tensor is not None else time_s >= env_cfg.episode_length_s - unwrapped.step_dt
                # RSL-RL auto-resets before returning. Attribute termination to
                # the pre-step segment and never create a zero-duration reset segment.
                skill_records.append(finalize_segment(segment, time_s, fall=not timed_out))
                episode_skills = [record for record in skill_records if record["episode"] == episode]
                expected = 4 if "Sequence" in args_cli.task else 3 if "Turn" in args_cli.task else 2 if "Stop" in args_cli.task else 1
                episode_records.append(make_episode_record(episode, episode_skills, time_s, expected))
                episode += 1
                feedback_controller.reset()
                step_in_episode = 0
                segment_index = 0
                if episode < args_cli.episodes:
                    current_skill = int(term.skill_id[0].item())
                    position = robot.data.root_pos_w.torch[0, :2].tolist()
                    heading = float(robot.data.heading_w.torch[0].item())
                    commanded_turn = float(term.commanded_turn_angle_rad[0].item()) if current_skill == TURN else 0.0
                    turn_start_heading = float(term.turn_start_heading_w[0].item()) if current_skill == TURN else heading
                    segment = new_segment(episode, 0, current_skill, 0.0, position, turn_start_heading, commanded_turn)
                continue
            new_skill = int(term.skill_id[0].item())
            if new_skill != current_skill:
                skill_records.append(finalize_segment(segment, time_s))
                segment_index += 1
                current_skill = new_skill
                position = robot.data.root_pos_w.torch[0, :2].tolist()
                heading = float(robot.data.heading_w.torch[0].item())
                commanded_turn = float(term.commanded_turn_angle_rad[0].item()) if current_skill == TURN else 0.0
                turn_start_heading = float(term.turn_start_heading_w[0].item()) if current_skill == TURN else heading
                segment = new_segment(
                    episode, segment_index, current_skill, time_s, position, turn_start_heading, commanded_turn
                )

            speed_error = abs(float(robot.data.root_lin_vel_b.torch[0, 0].item()) - float(term.target_speed[0].item()))
            heading_error = abs(float(term.heading_error[0].item()))
            stop_error = float(torch.linalg.norm(term.target_displacement_b[0]).item()) if current_skill == STOP else 0.0
            stop_speed = float(torch.linalg.norm(robot.data.root_lin_vel_b.torch[0, :2]).item())
            pelvis_height = float(robot.data.body_pos_w.torch[0, pelvis_ids[0], 2].item())
            tilt = float(torch.linalg.norm(robot.data.projected_gravity_b.torch[0, :2]).item())
            forces = contact.data.net_forces_w_history.torch[0, :, sensor_foot_ids, :]
            contacts = forces.norm(dim=-1).amax(dim=0) > 1.0
            contact_list = [bool(value) for value in contacts.tolist()]
            if not any(contact_list):
                segment["airborne_samples"] += 1
            if segment["previous_contacts"] is not None:
                for side, (before, after) in enumerate(zip(segment["previous_contacts"], contact_list)):
                    if not before and after:
                        segment["strike_sides"].append(side)
            segment["previous_contacts"] = contact_list
            foot_speed = robot.data.body_lin_vel_w.torch[0, foot_body_ids, :2].norm(dim=-1)
            slip = float((foot_speed * contacts).sum().item() / max(int(contacts.sum().item()), 1))
            joint_velocity_ratio = robot.data.joint_vel.torch[0, all_joint_ids].abs() / robot.data.joint_vel_limits.torch[0, all_joint_ids].abs().clamp_min(1.0e-6)
            ankle_ratio = robot.data.applied_torque.torch[0, ankle_ids].abs() / robot.data.joint_effort_limits.torch[0, ankle_ids].abs().clamp_min(1.0e-6)
            path_active = current_skill == RUN
            path_lateral_error = float(term.path_lateral_error[0].item()) if path_active else 0.0
            selected_residual = actor_components["selected_residual"][0]
            selected_raw_residual = actor_components["selected_raw_residual"][0]
            residual_norm = float(torch.linalg.vector_norm(selected_residual).item())
            raw_residual_norm = float(torch.linalg.vector_norm(selected_raw_residual).item())
            corrective_residual = actor_components["total_stop_correction"][0] + feedback_action[0]
            parent_action_deviation = actor_components["parent_action_deviation"][0] + feedback_action[0]
            corrective_residual_norm = float(torch.linalg.vector_norm(corrective_residual).item())
            parent_action_deviation_norm = float(torch.linalg.vector_norm(parent_action_deviation).item())
            path_correction_rms = float(selected_residual[path_correction_ids].square().mean().sqrt().item())
            propulsion_rms = float(selected_residual[propulsion_ids].square().mean().sqrt().item())
            segment["speed_errors"].append(speed_error)
            segment["heading_errors"].append(heading_error)
            segment["stop_errors"].append(stop_error)
            segment.setdefault("stop_longitudinal", []).append(
                float(term.target_displacement_b[0, 0].item()) if current_skill == STOP else 0.0
            )
            segment["stop_speeds"].append(stop_speed)
            segment["pelvis_errors"].append(abs(pelvis_height - float(term.target_pelvis_height[0].item())))
            segment["tilts"].append(tilt)
            segment["slips"].append(slip)
            saturated = joint_velocity_ratio >= 0.95
            segment["knee_saturation"].append(float(saturated.float().mean().item()))
            segment["ankle_torque_saturation"].append(float((ankle_ratio >= 0.95).float().mean().item()))
            segment["saturation_joint_names"].update(
                f"velocity:{name}" for name, hit in zip(all_joint_names, saturated.tolist()) if hit
            )
            segment["saturation_joint_names"].update(
                f"ankle_torque:{name}" for name, hit in zip(ankle_names, (ankle_ratio >= 0.95).tolist()) if hit
            )
            segment["landing_impacts"].append(float(forces[:, :, 2].abs().mean(dim=0).max().item()))
            segment["course_deviations"].append(abs(path_lateral_error))
            segment["path_lateral_errors"].append(path_lateral_error)
            segment["path_heading_errors"].append(abs(float(term.heading_error[0].item())) if path_active else 0.0)
            segment["path_forward_velocities"].append(float(term.path_forward_velocity[0].item()) if path_active else 0.0)
            segment["path_lateral_velocities"].append(float(term.path_lateral_velocity[0].item()) if path_active else 0.0)
            segment["residual_action_norms"].append(residual_norm)
            segment["raw_residual_action_norms"].append(raw_residual_norm)
            segment["corrective_residual_norms"].append(corrective_residual_norm)
            segment["parent_action_deviation_norms"].append(parent_action_deviation_norm)
            feedback_norm = float(feedback_diagnostics["feedback_norm"][0].item())
            support_count = int(feedback_diagnostics["support_count"][0].item())
            segment[
                "feedback_norm_double" if support_count >= 2 else
                "feedback_norm_single" if support_count == 1 else "feedback_norm_flight"
            ].append(feedback_norm)
            segment["feedback_spike_guard_count"] += int(feedback_diagnostics["spike_guard_active"][0].item())
            segment["feedback_hard_guard_count"] += int(feedback_diagnostics["hard_guard_active"][0].item())
            segment["feedback_slew_limiter_count"] += int(feedback_diagnostics["slew_limiter_active"][0].item())
            segment["residual_path_correction_rms"].append(path_correction_rms)
            segment["residual_propulsion_rms"].append(propulsion_rms)
            if current_skill == STOP:
                entry_speed = float(term.stop_entry_speed[0].item())
                initial_distance = float(term.stop_initial_distance[0].item())
                required_deceleration = float(term.stop_required_deceleration[0].item())
                braking_target = float(term.stop_braking_target_speed[0].item())
                stop_progress = float(term.stop_progress[0].item())
                hold_progress = float(term.stop_hold_progress[0].item())
                curve_point = {
                    "time_s": round(time_s - segment["start_time_s"], 6),
                    "remaining_distance_m": float(term.target_displacement_b[0, 0].item()),
                    "forward_speed_mps": float(robot.data.root_lin_vel_b.torch[0, 0].item()),
                    "speed_mps": stop_speed,
                    "required_deceleration_mps2": required_deceleration,
                    "braking_target_speed_mps": braking_target,
                    "stop_progress": stop_progress,
                    "hold_progress": hold_progress,
                    "stop_phase": "hold" if hold_progress > 0.0 else "braking",
                    "stop_target_heading_w": float(term.stop_target_heading_w[0].item()),
                    "current_heading_w": float(robot.data.heading_w.torch[0].item()),
                    "stop_heading_error_rad": float(term.heading_error[0].item()),
                    "legacy_yaw_rate_command_rps": float(term.vel_command_b[0, 2].item()),
                    "actual_yaw_rate_rps": float(robot.data.root_ang_vel_b.torch[0, 2].item()),
                    "stop_residual_norm": raw_residual_norm,
                    "masked_residual_norm": residual_norm,
                    "corrective_residual_norm": corrective_residual_norm,
                    "parent_action_deviation_norm": parent_action_deviation_norm,
                    "fixed_feedback_norm": feedback_norm,
                    "feedback_support_count": support_count,
                    "feedback_spike_guard_active": bool(feedback_diagnostics["spike_guard_active"][0].item()),
                    "feedback_hard_guard_active": bool(feedback_diagnostics["hard_guard_active"][0].item()),
                    "feedback_slew_limiter_active": bool(feedback_diagnostics["slew_limiter_active"][0].item()),
                    "feedback_left_contact": bool(pre_step_contacts[0, 0].item()),
                    "feedback_right_contact": bool(pre_step_contacts[0, 1].item()),
                    "left_ankle_utilization": float(pre_step_ankle_utilization[0, 0].item()),
                    "right_ankle_utilization": float(pre_step_ankle_utilization[0, 1].item()),
                    "feedback_raw_signal": float(feedback_diagnostics["raw_signal"][0].item()),
                    "feedback_filtered_signal": float(feedback_diagnostics["filtered_signal"][0].item()),
                    "feedback_recovery_scale": float(feedback_diagnostics["contact_recovery_scale"][0].item()),
                    "feedback_safety_scale": float(feedback_diagnostics["combined_safety_scale"][0].item()),
                    "left_hip_yaw_residual": float(selected_residual[left_hip_yaw_ids[0]].item()),
                    "right_hip_yaw_residual": float(selected_residual[right_hip_yaw_ids[0]].item()),
                    "torso_yaw_residual": float(selected_residual[torso_yaw_ids[0]].item()),
                    "left_ankle_roll_residual": float(selected_residual[left_ankle_roll_ids[0]].item()),
                    "right_ankle_roll_residual": float(selected_residual[right_ankle_roll_ids[0]].item()),
                }
                segment["stop_entry_speeds"].append(entry_speed)
                segment["stop_initial_distances"].append(initial_distance)
                segment["stop_required_decelerations"].append(required_deceleration)
                segment["stop_braking_targets"].append(braking_target)
                segment["stop_progress_values"].append(stop_progress)
                segment["stop_hold_progress_values"].append(hold_progress)
                segment["stop_curve"].append(curve_point)
                stop_curve_records.append({"episode": episode, "segment_index": segment_index, **curve_point})
            if current_skill == TURN:
                accumulated_yaw = float(term.actual_accumulated_yaw_rad[0].item())
                turn_error = abs(float(term.commanded_turn_angle_rad[0].item()) - accumulated_yaw)
                segment["actual_accumulated_yaws"].append(accumulated_yaw)
                segment["turn_angle_errors"].append(turn_error)
                if segment["turn_completion_time_s"] is None and turn_error <= 0.12:
                    segment["turn_completion_time_s"] = time_s - segment["start_time_s"]
            if segment["tracking_time_s"] is None and speed_error <= 0.35 and heading_error <= 0.12:
                segment["tracking_time_s"] = time_s - segment["start_time_s"]
            if segment["stabilization_time_s"] is None and tilt <= 0.15 and speed_error <= 0.35:
                segment["stabilization_time_s"] = time_s - segment["start_time_s"]

            # STOP evaluation ends at the requested hold boundary. A later
            # simulation event must not retroactively turn a completed STOP into a failure.
            if current_skill == STOP and bool(term.stop_hold_complete[0].item()):
                skill_records.append(finalize_segment(segment, time_s))
                episode_skills = [record for record in skill_records if record["episode"] == episode]
                expected = 4 if "Sequence" in args_cli.task else 2
                episode_records.append(make_episode_record(episode, episode_skills, time_s, expected))
                episode += 1
                step_in_episode = 0
                segment_index = 0
                if episode < args_cli.episodes:
                    wrapped.reset()
                    feedback_controller.reset()
                    current_skill = int(term.skill_id[0].item())
                    position = robot.data.root_pos_w.torch[0, :2].tolist()
                    heading = float(robot.data.heading_w.torch[0].item())
                    commanded_turn = float(term.commanded_turn_angle_rad[0].item()) if current_skill == TURN else 0.0
                    turn_start_heading = float(term.turn_start_heading_w[0].item()) if current_skill == TURN else heading
                    segment = new_segment(episode, 0, current_skill, 0.0, position, turn_start_heading, commanded_turn)
                continue

        write_csv(output / "skills.csv", skill_records)
        write_csv(output / "episodes.csv", episode_records)
        write_csv(output / "stop_curve.csv", stop_curve_records)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in skill_records:
            grouped[record["skill"]].append(record)
        skill_summary = {
            skill: {
                "count": len(records),
                "success_rate": mean([float(r["success"]) for r in records]),
                "speed_error_mps": mean([r["speed_error_mps"] for r in records]),
                "heading_error_rad": mean([r["heading_error_rad"] for r in records]),
                "mean_step_heading_error_rad": mean([r["mean_step_heading_error_rad"] for r in records]),
                "stop_position_error_m": mean([r["stop_position_error_m"] for r in records]),
                "stop_speed_mps": mean(
                    [r["stop_hold_end_speed_mps"] for r in records if r["stop_hold_observed"]]
                    or [r["stop_speed_mps"] for r in records]
                ),
                "stop_entry_speed_mps": mean([r["stop_entry_speed_mps"] for r in records]),
                "stop_exit_speed_mps": mean([r["stop_exit_speed_mps"] for r in records]),
                "stop_phase_min_speed_mps": mean([r["stop_phase_min_speed_mps"] for r in records]),
                "stop_braking_end_speed_mps": mean([r["stop_braking_end_speed_mps"] for r in records]),
                "stop_hold_start_speed_mps": mean([r["stop_hold_start_speed_mps"] for r in records if r["stop_hold_observed"]]),
                "stop_hold_end_speed_mps": mean([r["stop_hold_end_speed_mps"] for r in records if r["stop_hold_observed"]]),
                "stop_hold_max_speed_mps": mean([r["stop_hold_max_speed_mps"] for r in records if r["stop_hold_observed"]]),
                "stop_hold_max_speed_rebound_mps": mean([r["stop_hold_max_speed_rebound_mps"] for r in records if r["stop_hold_observed"]]),
                "braking_heading_error_mean_rad": mean([r["braking_heading_error_mean_rad"] for r in records]),
                "braking_heading_error_p95_rad": mean([r["braking_heading_error_p95_rad"] for r in records]),
                "braking_heading_error_max_rad": max([r["braking_heading_error_max_rad"] for r in records], default=0.0),
                "hold_heading_error_mean_rad": mean([r["hold_heading_error_mean_rad"] for r in records]),
                "hold_heading_error_p95_rad": mean([r["hold_heading_error_p95_rad"] for r in records]),
                "hold_heading_error_max_rad": max([r["hold_heading_error_max_rad"] for r in records], default=0.0),
                "stop_heading_error_mean_rad": mean([r["stop_heading_error_mean_rad"] for r in records]),
                "stop_heading_error_p95_rad": mean([r["stop_heading_error_p95_rad"] for r in records]),
                "stop_heading_error_max_rad": max([r["stop_heading_error_max_rad"] for r in records], default=0.0),
                "actual_yaw_rate_mean_rps": mean([r["actual_yaw_rate_mean_rps"] for r in records]),
                "actual_yaw_rate_abs_mean_rps": mean([r["actual_yaw_rate_abs_mean_rps"] for r in records]),
                "actual_yaw_rate_abs_p95_rps": mean([r["actual_yaw_rate_abs_p95_rps"] for r in records]),
                "actual_yaw_rate_abs_p99_rps": mean([r["actual_yaw_rate_abs_p99_rps"] for r in records]),
                "actual_yaw_rate_abs_max_rps": max([r["actual_yaw_rate_abs_max_rps"] for r in records], default=0.0),
                "actual_yaw_rate_over_1_5_fraction": mean([r["actual_yaw_rate_over_1_5_fraction"] for r in records]),
                "actual_yaw_rate_over_2_5_fraction": mean([r["actual_yaw_rate_over_2_5_fraction"] for r in records]),
                "actual_yaw_rate_over_4_0_fraction": mean([r["actual_yaw_rate_over_4_0_fraction"] for r in records]),
                "legacy_yaw_rate_command_mean_rps": mean([r["legacy_yaw_rate_command_mean_rps"] for r in records]),
                "legacy_yaw_rate_command_abs_max_rps": max([r["legacy_yaw_rate_command_abs_max_rps"] for r in records], default=0.0),
                "stop_raw_residual_norm": mean([r["stop_raw_residual_norm"] for r in records]),
                "stop_masked_residual_norm": mean([r["stop_masked_residual_norm"] for r in records]),
                "braking_speed_tracking_error_mps": mean([r["braking_speed_tracking_error_mps"] for r in records]),
                "stop_initial_distance_m": mean([r["stop_initial_distance_m"] for r in records]),
                "stop_required_deceleration_mps2": mean([r["stop_required_deceleration_mps2"] for r in records]),
                "stop_hold_success_rate": mean([float(r["stop_hold_success"]) for r in records]),
                "post_stop_hold_fall_rate": mean([float(r["post_stop_hold_fall"]) for r in records]),
                "periodic_running_rate": mean([float(r["periodic_running"]) for r in records]),
                "fall_rate": mean([float(r["fall"]) for r in records]),
                "stabilization_after_transition_s": mean([r["stabilization_after_transition_s"] for r in records]),
                "path_lateral_error_mean": mean([r["path_lateral_error_mean"] for r in records]),
                "path_lateral_error_p95": mean([r["path_lateral_error_p95"] for r in records]),
                "path_lateral_error_max": max([r["path_lateral_error_max"] for r in records], default=0.0),
                "path_heading_error": mean([r["path_heading_error"] for r in records]),
                "path_forward_velocity": mean([r["path_forward_velocity"] for r in records]),
                "path_lateral_velocity": mean([r["path_lateral_velocity"] for r in records]),
                "residual_action_norm": mean([r["residual_action_norm"] for r in records]),
                "raw_residual_action_norm": mean([r["raw_residual_action_norm"] for r in records]),
                "corrective_residual_norm": mean([r["corrective_residual_norm"] for r in records]),
                "parent_action_deviation_norm": mean([r["parent_action_deviation_norm"] for r in records]),
                "parent_action_deviation_norm_p95": mean(
                    [r["parent_action_deviation_norm_p95"] for r in records]
                ),
                "parent_action_deviation_norm_max": max(
                    [r["parent_action_deviation_norm_max"] for r in records], default=0.0
                ),
                "feedback_norm_double_support": mean([r["feedback_norm_double_support"] for r in records]),
                "feedback_norm_single_support": mean([r["feedback_norm_single_support"] for r in records]),
                "feedback_norm_flight": mean([r["feedback_norm_flight"] for r in records]),
                "feedback_spike_guard_count": sum(r["feedback_spike_guard_count"] for r in records),
                "feedback_hard_guard_count": sum(r["feedback_hard_guard_count"] for r in records),
                "feedback_slew_limiter_count": sum(r["feedback_slew_limiter_count"] for r in records),
                "residual_vs_lateral_error_correlation": mean(
                    [r["residual_vs_lateral_error_correlation"] for r in records]
                ),
                "residual_path_correction_rms": mean([r["residual_path_correction_rms"] for r in records]),
                "residual_propulsion_rms": mean([r["residual_propulsion_rms"] for r in records]),
                "joint_velocity_saturation_fraction": mean([r["joint_velocity_saturation_fraction"] for r in records]),
                "ankle_torque_saturation_fraction": mean([r["ankle_torque_saturation_fraction"] for r in records]),
                "saturation_failure_rate": mean([
                    float(
                        r["joint_velocity_saturation_fraction"] > 0.05
                        or r["ankle_torque_saturation_fraction"] > 0.20
                    )
                    for r in records
                ]),
                "maximum_action_magnitude": max([r["maximum_action_magnitude"] for r in records], default=0.0),
                "commanded_turn_angle_rad": mean([r["commanded_turn_angle_rad"] for r in records]),
                "actual_accumulated_yaw_rad": mean([r["actual_accumulated_yaw_rad"] for r in records]),
                "final_turn_angle_error_rad": mean([r["final_turn_angle_error_rad"] for r in records]),
                "maximum_turn_angle_error_rad": mean([r["maximum_turn_angle_error_rad"] for r in records]),
                "turn_completion_time_s": mean([r["turn_completion_time_s"] for r in records]),
                "turn_success_rate": mean([float(r["turn_success"]) for r in records]),
                "straight_recovery_success_rate": mean(
                    [float(r["straight_recovery_success"]) for r in records]
                ),
                "post_turn_heading_error_rad": mean([r["post_turn_heading_error_rad"] for r in records]),
                "post_turn_path_lateral_error_m": mean(
                    [r["post_turn_path_lateral_error_m"] for r in records]
                ),
            }
            for skill, records in grouped.items()
        }
        primary_skill = next(
            (name for name in ("RUN", "TURN", "STOP") if f"-Command-{name.title()}-" in args_cli.task),
            None,
        )
        primary_records = grouped.get(primary_skill, []) if primary_skill else skill_records
        failure_reason_counts = Counter(
            record["failure_class"] for record in primary_records if record["failure_class"]
        )
        summary = {
            "checkpoint": str(checkpoint), "task": args_cli.task, "episodes": args_cli.episodes,
            "command_ablation": args_cli.command_ablation,
            "stop_residual_ablation": args_cli.stop_residual_ablation,
            "stop_fixed_feedback_gains": {
                "k_heading": args_cli.stop_feedback_k_heading,
                "k_yaw_rate": args_cli.stop_feedback_k_yaw_rate,
                "k_roll": args_cli.stop_feedback_k_roll,
                "k_roll_rate": args_cli.stop_feedback_k_roll_rate,
            },
            "stop_fixed_feedback_safety": {
                "alpha": args_cli.stop_feedback_alpha,
                "max_delta_per_step": args_cli.stop_feedback_max_delta_per_step,
                "braking_scale": args_cli.stop_feedback_braking_scale,
                "hold_scale": args_cli.stop_feedback_hold_scale,
                "double_support_scale": 1.0,
                "single_support_scale": args_cli.stop_feedback_single_support_scale,
                "flight_scale": args_cli.stop_feedback_flight_scale,
                "yaw_soft_threshold": args_cli.stop_feedback_yaw_soft_threshold,
                "yaw_hard_threshold": args_cli.stop_feedback_yaw_hard_threshold,
                "hard_guard_mode": args_cli.stop_feedback_hard_guard_mode,
                "flight_hard_zero": args_cli.stop_feedback_flight_hard_zero,
                "contact_recovery_zero_steps": args_cli.stop_feedback_contact_zero_steps,
                "contact_recovery_ramp_steps": args_cli.stop_feedback_contact_ramp_steps,
                "hard_guard_action_limit": args_cli.stop_feedback_hard_action_limit,
                "hard_guard_disable_torso": args_cli.stop_feedback_hard_disable_torso,
                "ankle_utilization_soft": args_cli.stop_feedback_ankle_soft,
                "ankle_utilization_hard": args_cli.stop_feedback_ankle_hard,
                "joint_velocity_soft": args_cli.stop_feedback_joint_velocity_soft,
                "joint_velocity_hard": args_cli.stop_feedback_joint_velocity_hard,
                "tilt_soft_rad": args_cli.stop_feedback_tilt_soft,
                "tilt_hard_rad": args_cli.stop_feedback_tilt_hard,
                "angular_velocity_soft_rps": args_cli.stop_feedback_angular_soft,
                "angular_velocity_hard_rps": args_cli.stop_feedback_angular_hard,
                "worsening_yaw_scale": args_cli.stop_feedback_worsening_yaw_scale,
            },
            "stop_curriculum": (
                "C" if "StopC" in args_cli.task else "B" if "StopB" in args_cli.task else "A"
                if "Stop" in args_cli.task else None
            ),
            "stop_entry_speed_strata": summarize_stop_entry_strata(grouped.get("STOP", [])),
            "stop_primary_stratum": "in_range_le_1.4" if "Stop-Eval" in args_cli.task else None,
            "command_ablation_schema_version": 2,
            "coordinate_frame": "path-local for RUN; TURN yaw is unwrapped from fixed world-yaw at command entry",
            "turn_heading_metric_definition": "abs(commanded_turn_angle - accumulated_unwrapped_yaw) at TURN end",
            "sequence_completion_rate": mean([float(r["sequence_complete"]) for r in episode_records]),
            "skill_success_rate": mean([float(r["success"]) for r in primary_records]),
            "course_deviation_failure_rate": mean(
                [float(r["failure_class"] == "course_deviation") for r in primary_records]
            ),
            "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
            "success_count": sum(bool(record["success"]) for record in primary_records),
            "fall_rate": mean([float(r["fall"]) for r in episode_records]),
            "fall_rate_definition": "episode-level any segment fall",
            "stop_window_fall_rate": mean([float(r["fall"]) for r in grouped.get("STOP", [])]),
            "stop_window_definition": "STOP entry through completion of the configured post-stop hold",
            "skills": skill_summary, "failure_classes": FAILURE_CLASSES,
        }
        turn_records = grouped.get("TURN", [])
        summary["turn_straight_recovery_rate"] = mean(
            [float(record["straight_recovery_success"]) for record in turn_records]
        )
        turn_buckets = {
            "left_45": [], "right_45": [], "left_90": [], "right_90": [],
        }
        for record in turn_records:
            commanded = float(record["commanded_turn_angle_rad"])
            side = "left" if commanded >= 0.0 else "right"
            degrees = 45 if abs(abs(math.degrees(commanded)) - 45.0) <= 1.0 else 90
            turn_buckets[f"{side}_{degrees}"].append(record)
        summary["turn_angle_results"] = {
            name: {
                "count": len(records),
                "success_rate": mean([float(record["turn_success"]) for record in records]),
                "final_turn_angle_error_rad": mean(
                    [record["final_turn_angle_error_rad"] for record in records]
                ),
                "straight_recovery_success_rate": mean(
                    [float(record["straight_recovery_success"]) for record in records]
                ),
            }
            for name, records in turn_buckets.items()
        }
        for name, values in summary["turn_angle_results"].items():
            summary[f"{name}_success_rate"] = values["success_rate"]
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary_rows = [{"skill": skill, **values} for skill, values in skill_summary.items()]
        write_csv(output / "summary.csv", summary_rows)
        print(json.dumps(summary, indent=2))
        raw_env.close()


if __name__ == "__main__":
    main()
