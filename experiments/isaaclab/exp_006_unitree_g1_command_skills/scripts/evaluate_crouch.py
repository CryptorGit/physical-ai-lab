"""Evaluate isolated CROUCH Stage A and write episode/curve CSV plus JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

CROUCH = 3
EVALUATION_SCHEMA_VERSION = 3
STANDING_BASE_OPTION_ID = "stage2_fastwalk_model4246"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Motion-Flat-G1-Command-Crouch-Eval-v0")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--fixed-depth", type=float, default=None)
parser.add_argument("--fixed-depths", default="")
parser.add_argument("--console-status", action="store_true")
parser.add_argument("--output", default="results/exp_006_unitree_g1_command_skills/crouch_stage_a")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * q / 100.0), len(ordered) - 1)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_trace(episode: int, term, robot, env_id: int = 0) -> dict:
    return {
        "episode": episode,
        "entry_height": float(term.crouch_entry_height[env_id].item()), "entry_height_fixed": False,
        "requested_depth_m": float(term.crouch_requested_depth_m[env_id].item()),
        "applied_depth_m": float(term.crouch_applied_depth_m[env_id].item()),
        "command_supported": bool(term.crouch_command_supported[env_id].item()),
        "command_clamped": bool(term.crouch_command_clamped[env_id].item()),
        "unsupported_reason_code": int(term.crouch_unsupported_reason_code[env_id].item()),
        "commanded_drop": float(term.crouch_applied_depth_m[env_id].item()),
        "down_duration": float(term.crouch_down_duration[env_id].item()),
        "hold_duration": float(term.crouch_hold_duration[env_id].item()),
        "return_duration": float(term.crouch_return_duration[env_id].item()),
        "stand_hold_duration": float(term.crouch_stand_hold_duration[env_id].item()),
        "heights": [], "phase_heights": [], "height_errors": [], "vertical_velocities": [], "tilts": [],
        "hold_errors": [], "return_errors": [], "slips": [], "contact_loss": [],
        "support_states": [], "support_phases": [],
        "hip_differences": [], "knee_differences": [], "ankle_differences": [],
        "velocity_saturation": [], "torque_saturation": [], "actions": [], "residual_norms": [],
        "ankle_pitch_residual_saturation": [],
        "stabilization_time": None, "elapsed": 0.0,
        "settle_heights": [], "settle_speeds": [], "settle_support": [],
        "settle_success": False, "settle_time": 0.0, "down_entry_speed": 0.0,
        "down_entry_height_range": 0.0, "base_transition_failure": False,
        "initial_height_observed": float(robot.data.root_pos_w.torch[env_id, 2].item()),
    }


def finalize(trace: dict, fall: bool = False) -> dict:
    hold_errors = trace["hold_errors"]
    return_errors = trace["return_errors"]
    entry = trace["entry_height"]
    hold_heights = [entry - error for error in []]  # documentation placeholder; actual drop is derived below
    del hold_heights
    # Depth is geometric and independent of target error/reward bookkeeping.
    hold_signed = trace.get("hold_signed_errors", [])
    actual_minimum_height = min(trace["heights"], default=entry)
    actual_drop = entry - actual_minimum_height
    final_depth_error = abs(actual_drop - trace["commanded_drop"])
    hold_success = bool(hold_errors) and percentile(hold_errors, 95) <= 0.04
    down_reached = trace["command_supported"] and actual_drop >= max(0.0, trace["commanded_drop"] - 0.04)
    return_kinematic_success = bool(return_errors) and percentile(return_errors, 95) <= 0.05
    return_success = down_reached and return_kinematic_success
    hold_heights = [height for height, phase in trace["phase_heights"] if phase == 2]
    return_heights = [height for height, phase in trace["phase_heights"] if phase == 3]
    contact_loss_fraction = mean(trace["contact_loss"])
    dt = trace.get("step_dt", 0.02)
    states = trace["support_states"]
    phases = trace["support_phases"]
    state_names = ("flight", "left_single", "right_single", "double")
    fractions = {name: mean([float(state == index) for state in states]) for index, name in enumerate(state_names)}
    max_run_steps = {name: 0 for name in state_names}
    current_state, run_steps = None, 0
    support_switch_count = 0
    for state in states:
        if state == current_state:
            run_steps += 1
        else:
            if current_state is not None:
                max_run_steps[state_names[current_state]] = max(max_run_steps[state_names[current_state]], run_steps)
                support_switch_count += 1
            current_state, run_steps = state, 1
    if current_state is not None:
        max_run_steps[state_names[current_state]] = max(max_run_steps[state_names[current_state]], run_steps)
    max_duration = {name: steps * dt for name, steps in max_run_steps.items()}
    support_loss_flight_steps = 0
    for index, state in enumerate(states):
        if state != 0 or index == 0 or states[index - 1] not in (1, 2):
            continue
        end = index
        while end < len(states) and states[end] == 0:
            end += 1
        support_loss_flight_steps = max(support_loss_flight_steps, end - index)
    support_loss_flight_duration = support_loss_flight_steps * dt
    phase_support = {}
    for phase_id, phase_name in enumerate(("settle", "down", "hold", "return", "stand_hold")):
        selected = [state for state, phase in zip(states, phases) if phase == phase_id]
        phase_support[phase_name] = {
            f"{name}_fraction": mean([float(state == index) for state in selected])
            for index, name in enumerate(state_names)
        }
    flight_failure = max_duration["flight"] > 0.10
    prolonged_single = max(max_duration["left_single"], max_duration["right_single"]) > 0.50
    support_foot_loss = support_loss_flight_duration > 0.06
    switching_rate = support_switch_count / max(trace["elapsed"], 1.0e-6)
    unstable_switching = support_switch_count >= 6 and switching_rate > 4.0
    record = {
        "episode": trace["episode"], "skill": "CROUCH",
        "requested_depth_m": trace["requested_depth_m"],
        "supported_depth_min_m": 0.08,
        "supported_depth_max_m": 0.10,
        "applied_depth_m": trace["applied_depth_m"],
        "command_supported": trace["command_supported"],
        "command_clamped": trace["command_clamped"],
        "unsupported_reason": (
            "" if trace["unsupported_reason_code"] == 0 else
            "below_supported_range" if trace["unsupported_reason_code"] == 1 else
            "DEEP_CROUCH_RETURN_UNRESOLVED"
        ),
        "commanded_height_drop_m": trace["commanded_drop"],
        "actual_height_drop_m": actual_drop,
        "final_depth_error_m": final_depth_error,
        "hold_height_mean_error_m": mean(hold_errors),
        "hold_height_p95_error_m": percentile(hold_errors, 95),
        "hold_height_max_error_m": max(hold_errors, default=0.0),
        "target_minimum_pelvis_height_m": entry - trace["commanded_drop"],
        "minimum_pelvis_height_m": actual_minimum_height,
        "down_reached": down_reached,
        "hold_start_height_drop_m": entry - hold_heights[0] if hold_heights else 0.0,
        "hold_height_drop_mean_m": entry - mean(hold_heights) if hold_heights else 0.0,
        "hold_height_drop_p95_m": percentile([entry - value for value in hold_heights], 95),
        "return_start_pelvis_height_m": return_heights[0] if return_heights else entry,
        "vertical_velocity_p95_mps": percentile([abs(v) for v in trace["vertical_velocities"]], 95),
        "vertical_velocity_max_mps": max([abs(v) for v in trace["vertical_velocities"]], default=0.0),
        "crouch_hold_success": hold_success,
        "return_to_stand_success": return_success,
        "return_kinematic_success": return_kinematic_success,
        "stand_hold_success": return_success,
        "return_height_error_m": mean(return_errors),
        "return_height_p95_error_m": percentile(return_errors, 95),
        "fall": fall,
        "standing_base_candidate": trace.get("standing_base_candidate", STANDING_BASE_OPTION_ID),
        "base_option_id": "crouch_standing_base",
        "settle_success": trace["settle_success"],
        "settle_time_s": trace["settle_time"],
        "settle_speed_mean_mps": mean(trace["settle_speeds"]),
        "settle_pelvis_height_range_m": (
            max(trace["settle_heights"]) - min(trace["settle_heights"]) if trace["settle_heights"] else 0.0
        ),
        "settle_double_support_fraction": mean([float(state == 3) for state in trace["settle_support"]]),
        "base_transition_failure": trace["base_transition_failure"],
        "down_entry_speed_mps": trace["down_entry_speed"],
        "down_entry_pelvis_height_range_m": trace["down_entry_height_range"],
        "foot_contact_loss": flight_failure or prolonged_single or support_foot_loss or unstable_switching,
        # Compatibility field: historically this meant "not double support".
        # Keep it explicit and do not use it as the CROUCH safety failure.
        "foot_contact_loss_fraction": contact_loss_fraction,
        "non_double_support_fraction": contact_loss_fraction,
        "left_foot_contact_loss_fraction": mean([float(state in (0, 2)) for state in states]),
        "right_foot_contact_loss_fraction": mean([float(state in (0, 1)) for state in states]),
        "double_support_fraction": fractions["double"],
        "single_support_fraction": fractions["left_single"] + fractions["right_single"],
        "both_feet_airborne_fraction": fractions["flight"],
        "both_feet_airborne_duration_s": fractions["flight"] * len(states) * dt,
        "single_support_duration_s": (fractions["left_single"] + fractions["right_single"]) * len(states) * dt,
        "double_support_duration_s": fractions["double"] * len(states) * dt,
        "maximum_both_feet_airborne_duration_s": max_duration["flight"],
        "maximum_support_loss_flight_duration_s": support_loss_flight_duration,
        "maximum_left_single_support_duration_s": max_duration["left_single"],
        "maximum_right_single_support_duration_s": max_duration["right_single"],
        "support_switch_count": support_switch_count,
        "support_switch_rate_hz": switching_rate,
        "both_feet_airborne_failure": flight_failure,
        "support_foot_loss_failure": support_foot_loss,
        "prolonged_single_support_failure": prolonged_single,
        "unstable_contact_switching_failure": unstable_switching,
        "foot_slip_mps": mean(trace["slips"]),
        "foot_slip_max_mps": max(trace["slips"], default=0.0),
        "left_right_hip_pitch_difference_rad": mean(trace["hip_differences"]),
        "left_right_knee_difference_rad": mean(trace["knee_differences"]),
        "left_right_ankle_pitch_difference_rad": mean(trace["ankle_differences"]),
        "joint_velocity_saturation_fraction": mean(trace["velocity_saturation"]),
        "torque_saturation_fraction": mean(trace["torque_saturation"]),
        "saturation_failure": (
            mean(trace["velocity_saturation"]) > 0.05 or mean(trace["torque_saturation"]) > 0.05
        ),
        "joint_limit_proximity_max": max(trace.get("joint_limit_proximity", []), default=0.0),
        "ankle_torque_saturation_fraction": 0.0,
        "ankle_pitch_residual_saturation_fraction": mean(trace["ankle_pitch_residual_saturation"]),
        "maximum_action_magnitude": max(trace["actions"], default=0.0),
        "residual_action_norm": mean(trace["residual_norms"]),
        "residual_action_norm_p95": percentile(trace["residual_norms"], 95),
        "stabilization_time_s": trace["stabilization_time"] if trace["stabilization_time"] is not None else 0.0,
        "tilt_rad": max(trace["tilts"], default=0.0),
        "down_duration_s": trace["down_duration"], "hold_duration_s": trace["hold_duration"],
        "return_duration_s": trace["return_duration"], "stand_hold_duration_s": trace["stand_hold_duration"],
        "duration_s": trace["elapsed"], "timeout": False,
    }
    for phase_name, values in phase_support.items():
        for metric, value in values.items():
            record[f"{phase_name}_{metric}"] = value
    record["failure_class"] = (
        "unsupported_crouch_depth" if not record["command_supported"] else classify_failure(record)
    )
    record["success"] = record["failure_class"] == ""
    return record


def save_results(output: Path, checkpoint: Path, records: list[dict], curves: list[dict]) -> dict:
    write_csv(output / "episodes.csv", records)
    write_csv(output / "skills.csv", records)
    write_csv(output / "crouch_curve.csv", curves)
    failure_counts = Counter(record["failure_class"] for record in records if record["failure_class"])
    crouch = {
        "count": len(records), "success_rate": mean([float(r["success"]) for r in records]),
        "supported_command_rate": mean([float(r["command_supported"]) for r in records]),
        "depth_error_m": mean([r["final_depth_error_m"] for r in records]),
        "depth_error_p95_m": percentile([r["final_depth_error_m"] for r in records], 95),
        "hold_success_rate": mean([float(r["crouch_hold_success"]) for r in records]),
        "settle_success_rate": mean([float(r["settle_success"]) for r in records]),
        "settle_time_s": mean([r["settle_time_s"] for r in records if r["settle_success"]]),
        "return_success_rate": mean([float(r["return_to_stand_success"]) for r in records]),
        "return_kinematic_success_rate": mean([float(r["return_kinematic_success"]) for r in records]),
        "down_reached_rate": mean([float(r["down_reached"]) for r in records]),
        "stand_hold_success_rate": mean([float(r["stand_hold_success"]) for r in records]),
        "return_height_error_m": mean([r["return_height_error_m"] for r in records]),
        "fall_rate": mean([float(r["fall"]) for r in records]),
        "saturation_failure_rate": mean([float(r["saturation_failure"]) for r in records]),
        "foot_contact_loss_rate": mean([float(r["foot_contact_loss"]) for r in records]),
        "dangerous_contact_failure_rate": mean([float(r["foot_contact_loss"]) for r in records]),
        "both_feet_airborne_failure_rate": mean([float(r["both_feet_airborne_failure"]) for r in records]),
        "support_foot_loss_failure_rate": mean([float(r["support_foot_loss_failure"]) for r in records]),
        "prolonged_single_support_failure_rate": mean([float(r["prolonged_single_support_failure"]) for r in records]),
        "unstable_contact_switching_failure_rate": mean([float(r["unstable_contact_switching_failure"]) for r in records]),
        "double_support_fraction": mean([r["double_support_fraction"] for r in records]),
        "single_support_fraction": mean([r["single_support_fraction"] for r in records]),
        "both_feet_airborne_fraction": mean([r["both_feet_airborne_fraction"] for r in records]),
        "commanded_height_drop_m": mean([r["commanded_height_drop_m"] for r in records]),
        "actual_height_drop_m": mean([r["actual_height_drop_m"] for r in records]),
        "hold_height_p95_error_m": mean([r["hold_height_p95_error_m"] for r in records]),
        "minimum_pelvis_height_m": min([r["minimum_pelvis_height_m"] for r in records], default=0.0),
        "vertical_velocity_p95_mps": mean([r["vertical_velocity_p95_mps"] for r in records]),
        "vertical_velocity_max_mps": max([r["vertical_velocity_max_mps"] for r in records], default=0.0),
        "maximum_action_magnitude": max([r["maximum_action_magnitude"] for r in records], default=0.0),
        "residual_action_norm": mean([r["residual_action_norm"] for r in records]),
        "ankle_pitch_residual_saturation_fraction": mean([
            r["ankle_pitch_residual_saturation_fraction"] for r in records
        ]),
        "unsupported_command_count": sum(not r["command_supported"] for r in records),
        "stabilization_time_s": mean([r["stabilization_time_s"] for r in records]),
        "transition_duration_s": mean([
            r["down_duration_s"] + r["return_duration_s"] for r in records
        ]),
    }
    summary = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "checkpoint": str(checkpoint), "task": args_cli.task, "episodes": len(records),
        "seed": args_cli.seed, "crouch_curriculum": "SHALLOW_A", "skills": {"CROUCH_SHALLOW": crouch},
        "controller": "scripted_shallow_v1",
        "learned_crouch_residual_enabled": False,
        "supported_depth_range_m": [0.08, 0.10],
        "standing_base_option_id": STANDING_BASE_OPTION_ID,
        "skill_success_rate": crouch["success_rate"], "fall_rate": crouch["fall_rate"],
        "failure_reason_counts": dict(sorted(failure_counts.items())), "failure_classes": FAILURE_CLASSES,
        "coordinate_frame": "pelvis height relative to CROUCH entry; no world XY command",
        "contact_failure_definition": {
            "brief_single_support_is_failure": False,
            "both_feet_airborne": "maximum continuous flight > 0.10 s",
            "support_foot_loss": "single-support to flight transition with continuous flight > 0.06 s",
            "prolonged_single_support": "maximum continuous left or right single support > 0.50 s",
            "unstable_contact_switching": "at least 6 switches and switch rate > 4 Hz",
            "foot_contact_loss_fraction_compatibility": "legacy non-double-support fraction; not used as a failure",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def evaluate_parallel(wrapped, unwrapped, actor, policy, robot, term, contact, sensor_foot_ids,
                      foot_body_ids, hip_ids, knee_ids, ankle_pitch_ids, all_joint_ids,
                      crouch_joint_ids, crouch_joint_labels, crouch_limits) -> tuple[list[dict], list[dict]]:
    """Evaluate one episode per environment; this preserves the requested episode count."""
    wrapped.reset()
    num_envs = min(args_cli.num_envs, args_cli.episodes)
    traces = [new_trace(env_id, term, robot, env_id) for env_id in range(num_envs)]
    for trace in traces:
        trace["step_dt"] = float(unwrapped.step_dt)
    active = set(range(num_envs))
    records, curves = [], []
    while active:
        observations = wrapped.get_observations()
        with torch.inference_mode():
            components = actor.diagnostic_components(observations)
            actions = policy(observations)
        _, _, dones, infos = wrapped.step(actions)
        timeout_tensor = infos.get("time_outs") if isinstance(infos, dict) else None
        for env_id in list(active):
            trace = traces[env_id]
            trace["elapsed"] += float(unwrapped.step_dt)
            done = bool(dones[env_id].item())
            timed_out = bool(timeout_tensor[env_id].item()) if done and timeout_tensor is not None else False
            if done:
                record = finalize(trace, fall=not timed_out)
                record["timeout"] = timed_out
                if timed_out and not record["failure_class"]:
                    record["failure_class"], record["success"] = "timeout", False
                records.append(record)
                active.remove(env_id)
                continue
            height = float(robot.data.root_pos_w.torch[env_id, 2].item())
            vertical_velocity = float(robot.data.root_lin_vel_w.torch[env_id, 2].item())
            target = float(term.target_pelvis_height[env_id].item())
            phase = int(term.crouch_phase[env_id].item())
            if bool(term.crouch_entry_height_fixed[env_id].item()) and not trace["entry_height_fixed"]:
                trace["entry_height"] = float(term.crouch_entry_height[env_id].item())
                trace["entry_height_fixed"] = True
                trace["settle_success"] = True
                trace["settle_time"] = float(term.crouch_settle_time[env_id].item())
                trace["down_entry_speed"] = float(term.crouch_down_entry_speed[env_id].item())
                trace["down_entry_height_range"] = float(
                    (term.crouch_settle_height_max[env_id] - term.crouch_settle_height_min[env_id]).item()
                )
            error = height - target
            tilt = float(torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[env_id, :2]).item())
            forces = contact.data.net_forces_w_history.torch[env_id, :, sensor_foot_ids, :]
            contacts = forces.norm(dim=-1).amax(dim=0) > 5.0
            foot_speed = robot.data.body_lin_vel_w.torch[env_id, foot_body_ids, :2].norm(dim=-1)
            slip = float((foot_speed * contacts).sum().item() / max(int(contacts.sum().item()), 1))
            velocity_ratio = robot.data.joint_vel.torch[env_id, all_joint_ids].abs() / robot.data.joint_vel_limits.torch[env_id, all_joint_ids].abs().clamp_min(1.0e-6)
            torque_ratio = robot.data.applied_torque.torch[env_id, all_joint_ids].abs() / robot.data.joint_effort_limits.torch[env_id, all_joint_ids].abs().clamp_min(1.0e-6)
            joints = robot.data.joint_pos.torch[env_id]
            residual = components["selected_residual"][env_id]
            residual_norm = float(torch.linalg.vector_norm(residual).item())
            trace["heights"].append(height); trace["phase_heights"].append((height, phase)); trace["height_errors"].append(abs(error))
            trace["vertical_velocities"].append(vertical_velocity); trace["tilts"].append(tilt)
            trace.setdefault("hold_signed_errors", [])
            if phase == 0:
                trace["settle_heights"].append(height)
                trace["settle_speeds"].append(float(robot.data.root_lin_vel_b.torch[env_id, :2].norm().item()))
            if phase == 2:
                trace["hold_errors"].append(abs(error)); trace["hold_signed_errors"].append(error)
            if phase == 4:
                return_error = abs(height - trace["entry_height"])
                trace["return_errors"].append(return_error)
                if trace["stabilization_time"] is None and return_error <= 0.05 and abs(vertical_velocity) <= 0.08 and tilt <= 0.15:
                    return_start = trace["down_duration"] + trace["hold_duration"] + trace["return_duration"]
                    trace["stabilization_time"] = max(0.0, trace["elapsed"] - return_start)
            trace["slips"].append(slip); trace["contact_loss"].append(float(not bool(contacts.all().item())))
            support_state = 3 if bool(contacts.all().item()) else 1 if bool(contacts[0].item()) else 2 if bool(contacts[1].item()) else 0
            if phase == 0: trace["settle_support"].append(support_state)
            trace["support_states"].append(support_state); trace["support_phases"].append(phase)
            trace["hip_differences"].append(abs(float(joints[hip_ids[0]].item() - joints[hip_ids[1]].item())))
            trace["knee_differences"].append(abs(float(joints[knee_ids[0]].item() - joints[knee_ids[1]].item())))
            trace["ankle_differences"].append(abs(float(joints[ankle_pitch_ids[0]].item() - joints[ankle_pitch_ids[1]].item())))
            trace["velocity_saturation"].append(float(bool((velocity_ratio >= 0.95).any().item())))
            trace["torque_saturation"].append(float(bool((torque_ratio >= 0.95).any().item())))
            trace["actions"].append(float(actions[env_id].abs().max().item())); trace["residual_norms"].append(residual_norm)
            trace.setdefault("joint_limit_proximity", []).append(float(term.joint_limit_proximity[env_id].item()))
            trace["ankle_pitch_residual_saturation"].append(mean([
                float(abs(float(residual[joint_id].item())) >= 0.99 * limit)
                for joint_id, limit in zip(ankle_pitch_ids, crouch_limits[-2:])
            ]))
            curves.append({
                "episode": env_id, "time_s": trace["elapsed"], "phase": phase,
                "phase_progress": float(term.crouch_phase_progress[env_id].item()),
                "hold_progress": float(term.crouch_hold_progress[env_id].item()),
                "return_progress": float(term.crouch_return_progress[env_id].item()),
                "entry_pelvis_height_m": trace["entry_height"], "commanded_height_drop_m": trace["commanded_drop"],
                "requested_depth_m": trace["requested_depth_m"], "applied_depth_m": trace["applied_depth_m"],
                "command_supported": trace["command_supported"], "command_clamped": trace["command_clamped"],
                "target_relative_height_m": target - trace["entry_height"], "target_absolute_pelvis_height_m": target,
                "pelvis_height_m": height, "height_error_m": error, "vertical_velocity_mps": vertical_velocity,
                "target_vertical_velocity_mps": float(term.target_vertical_velocity[env_id].item()),
                "commanded_vx_mps": float(observations["policy"][env_id, 9].item()),
                "commanded_vy_mps": float(observations["policy"][env_id, 10].item()),
                "commanded_yaw_rate_rps": float(observations["policy"][env_id, 11].item()),
                "actual_forward_speed_mps": float(robot.data.root_lin_vel_b.torch[env_id, 0].item()),
                "crouch_gate": float(components["gate"][env_id, CROUCH].item()),
                "stand_base_gate": float(components["stand_base_gate"][env_id, 0].item()),
                "running_base_action_norm": float(torch.linalg.vector_norm(components["running_base_action"][env_id]).item()),
                "standing_base_action_norm": float(torch.linalg.vector_norm(components["standing_base_action"][env_id]).item()),
                "selected_base_action_norm": float(torch.linalg.vector_norm(components["selected_base_action"][env_id]).item()),
                "base_action_difference": float(torch.linalg.vector_norm(components["base_action_difference"][env_id]).item()),
                "base_crossfade_progress": float(components["base_crossfade_progress"][env_id, 0].item()),
                "both_feet_contact": bool(contacts.all().item()), "foot_slip_mps": slip,
                "left_foot_contact": bool(contacts[0].item()), "right_foot_contact": bool(contacts[1].item()),
                "support_state": ("flight", "left_single", "right_single", "double")[support_state],
                "joint_limit_proximity": float(term.joint_limit_proximity[env_id].item()),
                "residual_action_norm": residual_norm, "maximum_action_magnitude": float(actions[env_id].abs().max().item()),
                "scripted_crouch_offset_norm": float(torch.linalg.vector_norm(components["scripted_crouch_offset"][env_id]).item()),
                **{f"{label}_{field}": value for label, joint_id, limit in zip(crouch_joint_labels, crouch_joint_ids, crouch_limits)
                   for field, value in (
                       ("joint_position_rad", float(joints[joint_id].item())), ("residual", float(residual[joint_id].item())),
                       ("residual_saturated", bool(abs(float(residual[joint_id].item())) >= 0.99 * limit)),
                       ("base_action", float(components["base_action"][env_id, joint_id].item())),
                       ("final_action", float(components["action_mean"][env_id, joint_id].item())),
                       ("velocity_utilization", float(velocity_ratio[joint_id].item())),
                       ("torque_utilization", float(torque_ratio[joint_id].item())),
                   )},
            })
            if args_cli.console_status and env_id == 0 and len(trace["heights"]) % max(1, round(0.25 / float(unwrapped.step_dt))) == 0:
                print(
                    f"requested={trace['requested_depth_m']:.3f}m applied={trace['applied_depth_m']:.3f}m "
                    f"actual={max(0.0, trace['entry_height'] - height):.3f}m phase={phase} "
                    f"supported={trace['command_supported']}"
                )
            complete = bool(term.crouch_stand_hold_complete[env_id].item())
            settle_failed = bool(term.crouch_settle_failure[env_id].item())
            if settle_failed:
                trace["base_transition_failure"] = bool(term.crouch_base_transition_progress[env_id].item() < 1.0)
            if complete or settle_failed:
                records.append(finalize(trace)); active.remove(env_id)
    records.sort(key=lambda row: row["episode"])
    return records, curves


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).resolve(strict=True)
    output = Path(args_cli.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.fixed_depth is not None:
        env_cfg.commands.base_velocity.crouch_height_drop_range_m = (args_cli.fixed_depth, args_cli.fixed_depth)
    if args_cli.fixed_depths:
        env_cfg.commands.base_velocity.crouch_evaluation_depths_m = tuple(
            float(value) for value in args_cli.fixed_depths.split(",")
        )
    env_cfg.scene.num_envs = min(max(args_cli.num_envs, 1), args_cli.episodes)
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
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
        policy, actor = runner.get_inference_policy(device=unwrapped.device), runner.alg.actor
        robot = unwrapped.scene["robot"]
        term = unwrapped.command_manager.get_term("base_velocity")
        contact = unwrapped.scene.sensors["contact_forces"]
        foot_body_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_foot_ids = [contact.body_names.index(name) for name in foot_names]
        hip_ids, _ = robot.find_joints(["left_hip_pitch_joint", "right_hip_pitch_joint"], preserve_order=True)
        knee_ids, _ = robot.find_joints(["left_knee_joint", "right_knee_joint"], preserve_order=True)
        ankle_pitch_ids, _ = robot.find_joints(["left_ankle_pitch_joint", "right_ankle_pitch_joint"], preserve_order=True)
        crouch_joint_ids = hip_ids + knee_ids + ankle_pitch_ids
        crouch_joint_labels = ("left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee", "left_ankle_pitch", "right_ankle_pitch")
        crouch_limits = (0.25, 0.25, 0.25, 0.25, 0.15, 0.15)
        all_joint_ids, _ = robot.find_joints(".*")

        if env_cfg.scene.num_envs > 1:
            records, curves = evaluate_parallel(
                wrapped, unwrapped, actor, policy, robot, term, contact, sensor_foot_ids,
                foot_body_ids, hip_ids, knee_ids, ankle_pitch_ids, all_joint_ids,
                crouch_joint_ids, crouch_joint_labels, crouch_limits,
            )
            save_results(output, checkpoint, records, curves)
            raw_env.close()
            return

        wrapped.reset()  # Environment/reset mutation deliberately remains outside inference mode.
        records, curves = [], []
        episode = 0
        trace = new_trace(episode, term, robot)
        trace["step_dt"] = float(unwrapped.step_dt)
        while episode < args_cli.episodes:
            observations = wrapped.get_observations()
            with torch.inference_mode():
                components = actor.diagnostic_components(observations)
                actions = policy(observations)
            _, _, dones, infos = wrapped.step(actions)
            trace["elapsed"] += float(unwrapped.step_dt)
            done = bool(dones[0].item())
            timeout_tensor = infos.get("time_outs") if isinstance(infos, dict) else None
            timed_out = bool(timeout_tensor[0].item()) if done and timeout_tensor is not None else False
            if done:
                record = finalize(trace, fall=not timed_out)
                record["timeout"] = timed_out
                if timed_out and not record["failure_class"]:
                    record["failure_class"], record["success"] = "timeout", False
                records.append(record)
                episode += 1
                if episode < args_cli.episodes:
                    trace = new_trace(episode, term, robot)
                    trace["step_dt"] = float(unwrapped.step_dt)
                continue

            height = float(robot.data.root_pos_w.torch[0, 2].item())
            vertical_velocity = float(robot.data.root_lin_vel_w.torch[0, 2].item())
            target = float(term.target_pelvis_height[0].item())
            phase = int(term.crouch_phase[0].item())
            if bool(term.crouch_entry_height_fixed[0].item()) and not trace["entry_height_fixed"]:
                trace["entry_height"] = float(term.crouch_entry_height[0].item())
                trace["entry_height_fixed"] = True
                trace["settle_success"] = True
                trace["settle_time"] = float(term.crouch_settle_time[0].item())
                trace["down_entry_speed"] = float(term.crouch_down_entry_speed[0].item())
                trace["down_entry_height_range"] = float(
                    (term.crouch_settle_height_max[0] - term.crouch_settle_height_min[0]).item()
                )
            error = height - target
            tilt = float(torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[0, :2]).item())
            forces = contact.data.net_forces_w_history.torch[0, :, sensor_foot_ids, :]
            contacts = forces.norm(dim=-1).amax(dim=0) > 5.0
            foot_speed = robot.data.body_lin_vel_w.torch[0, foot_body_ids, :2].norm(dim=-1)
            slip = float((foot_speed * contacts).sum().item() / max(int(contacts.sum().item()), 1))
            velocity_ratio = robot.data.joint_vel.torch[0, all_joint_ids].abs() / robot.data.joint_vel_limits.torch[0, all_joint_ids].abs().clamp_min(1.0e-6)
            torque_ratio = robot.data.applied_torque.torch[0, all_joint_ids].abs() / robot.data.joint_effort_limits.torch[0, all_joint_ids].abs().clamp_min(1.0e-6)
            joints = robot.data.joint_pos.torch[0]
            residual_norm = float(torch.linalg.vector_norm(components["selected_residual"][0]).item())
            trace["heights"].append(height)
            trace["phase_heights"].append((height, phase))
            trace["height_errors"].append(abs(error))
            trace["vertical_velocities"].append(vertical_velocity)
            trace["tilts"].append(tilt)
            trace.setdefault("hold_signed_errors", [])
            if phase == 0:
                trace["settle_heights"].append(height)
                trace["settle_speeds"].append(float(robot.data.root_lin_vel_b.torch[0, :2].norm().item()))
            if phase == 2:
                trace["hold_errors"].append(abs(error))
                trace["hold_signed_errors"].append(error)
            if phase == 4:
                return_error = abs(height - trace["entry_height"])
                trace["return_errors"].append(return_error)
                if trace["stabilization_time"] is None and return_error <= 0.05 and abs(vertical_velocity) <= 0.08 and tilt <= 0.15:
                    return_start = trace["down_duration"] + trace["hold_duration"] + trace["return_duration"]
                    trace["stabilization_time"] = max(0.0, trace["elapsed"] - return_start)
            trace["slips"].append(slip)
            trace["contact_loss"].append(float(not bool(contacts.all().item())))
            support_state = 3 if bool(contacts.all().item()) else 1 if bool(contacts[0].item()) else 2 if bool(contacts[1].item()) else 0
            if phase == 0:
                trace["settle_support"].append(support_state)
            trace["support_states"].append(support_state)
            trace["support_phases"].append(phase)
            trace["hip_differences"].append(abs(float(joints[hip_ids[0]].item() - joints[hip_ids[1]].item())))
            trace["knee_differences"].append(abs(float(joints[knee_ids[0]].item() - joints[knee_ids[1]].item())))
            trace["ankle_differences"].append(abs(float(joints[ankle_pitch_ids[0]].item() - joints[ankle_pitch_ids[1]].item())))
            trace["velocity_saturation"].append(float(bool((velocity_ratio >= 0.95).any().item())))
            trace["torque_saturation"].append(float(bool((torque_ratio >= 0.95).any().item())))
            trace["actions"].append(float(actions[0].abs().max().item()))
            trace["residual_norms"].append(residual_norm)
            trace["ankle_pitch_residual_saturation"].append(mean([
                float(abs(float(components["selected_residual"][0, joint_id].item())) >= 0.99 * limit)
                for joint_id, limit in zip(ankle_pitch_ids, crouch_limits[-2:])
            ]))
            curves.append({
                "episode": episode, "time_s": trace["elapsed"], "phase": phase,
                "phase_progress": float(term.crouch_phase_progress[0].item()),
                "hold_progress": float(term.crouch_hold_progress[0].item()),
                "return_progress": float(term.crouch_return_progress[0].item()),
                "entry_pelvis_height_m": trace["entry_height"], "commanded_height_drop_m": trace["commanded_drop"],
                "target_relative_height_m": target - trace["entry_height"], "target_absolute_pelvis_height_m": target,
                "pelvis_height_m": height,
                "height_error_m": error, "vertical_velocity_mps": vertical_velocity,
                "target_vertical_velocity_mps": float(term.target_vertical_velocity[0].item()),
                "commanded_vx_mps": float(observations["policy"][0, 9].item()),
                "commanded_vy_mps": float(observations["policy"][0, 10].item()),
                "commanded_yaw_rate_rps": float(observations["policy"][0, 11].item()),
                "actual_forward_speed_mps": float(robot.data.root_lin_vel_b.torch[0, 0].item()),
                "crouch_gate": float(components["gate"][0, CROUCH].item()),
                "stand_base_gate": float(components["stand_base_gate"][0, 0].item()),
                "running_base_action_norm": float(torch.linalg.vector_norm(components["running_base_action"][0]).item()),
                "standing_base_action_norm": float(torch.linalg.vector_norm(components["standing_base_action"][0]).item()),
                "selected_base_action_norm": float(torch.linalg.vector_norm(components["selected_base_action"][0]).item()),
                "base_action_difference": float(torch.linalg.vector_norm(components["base_action_difference"][0]).item()),
                "base_crossfade_progress": float(components["base_crossfade_progress"][0, 0].item()),
                "both_feet_contact": bool(contacts.all().item()), "foot_slip_mps": slip,
                "left_foot_contact": bool(contacts[0].item()), "right_foot_contact": bool(contacts[1].item()),
                "support_state": ("flight", "left_single", "right_single", "double")[support_state],
                "joint_limit_proximity": float(term.joint_limit_proximity[0].item()),
                "residual_action_norm": residual_norm, "maximum_action_magnitude": float(actions[0].abs().max().item()),
                **{
                    f"{label}_{field}": value
                    for label, joint_id, limit in zip(crouch_joint_labels, crouch_joint_ids, crouch_limits)
                    for field, value in (
                        ("joint_position_rad", float(joints[joint_id].item())),
                        ("residual", float(components["selected_residual"][0, joint_id].item())),
                        ("residual_saturated", bool(abs(float(components["selected_residual"][0, joint_id].item())) >= 0.99 * limit)),
                        ("base_action", float(components["base_action"][0, joint_id].item())),
                        ("final_action", float(components["action_mean"][0, joint_id].item())),
                        ("velocity_utilization", float(velocity_ratio[joint_id].item())),
                        ("torque_utilization", float(torque_ratio[joint_id].item())),
                    )
                },
            })
            if args_cli.console_status and len(trace["heights"]) % max(1, round(0.25 / float(unwrapped.step_dt))) == 0:
                print(
                    f"requested={trace['requested_depth_m']:.3f}m applied={trace['applied_depth_m']:.3f}m "
                    f"actual={max(0.0, trace['entry_height'] - height):.3f}m phase={phase} "
                    f"supported={trace['command_supported']}"
                )
            if bool(term.crouch_settle_failure[0].item()):
                trace["base_transition_failure"] = bool(
                    term.crouch_base_transition_progress[0].item() < 1.0
                )
                records.append(finalize(trace))
                episode += 1
                if episode < args_cli.episodes:
                    wrapped.reset()
                    trace = new_trace(episode, term, robot)
                    trace["step_dt"] = float(unwrapped.step_dt)
                continue
            if bool(term.crouch_stand_hold_complete[0].item()):
                records.append(finalize(trace))
                episode += 1
                if episode < args_cli.episodes:
                    wrapped.reset()
                    trace = new_trace(episode, term, robot)
                    trace["step_dt"] = float(unwrapped.step_dt)

        write_csv(output / "episodes.csv", records)
        write_csv(output / "skills.csv", records)
        write_csv(output / "crouch_curve.csv", curves)
        failure_counts = Counter(record["failure_class"] for record in records if record["failure_class"])
        crouch = {
            "count": len(records), "success_rate": mean([float(r["success"]) for r in records]),
            "depth_error_m": mean([r["final_depth_error_m"] for r in records]),
            "hold_success_rate": mean([float(r["crouch_hold_success"]) for r in records]),
            "settle_success_rate": mean([float(r["settle_success"]) for r in records]),
            "settle_time_s": mean([r["settle_time_s"] for r in records if r["settle_success"]]),
            "return_success_rate": mean([float(r["return_to_stand_success"]) for r in records]),
            "return_kinematic_success_rate": mean([float(r["return_kinematic_success"]) for r in records]),
            "down_reached_rate": mean([float(r["down_reached"]) for r in records]),
            "stand_hold_success_rate": mean([float(r["stand_hold_success"]) for r in records]),
            "return_height_error_m": mean([r["return_height_error_m"] for r in records]),
            "fall_rate": mean([float(r["fall"]) for r in records]),
            "saturation_failure_rate": mean([float(r["failure_class"] == "saturation_failure") for r in records]),
            "foot_contact_loss_rate": mean([float(r["foot_contact_loss"]) for r in records]),
            "both_feet_airborne_failure_rate": mean([float(r["both_feet_airborne_failure"]) for r in records]),
            "support_foot_loss_failure_rate": mean([float(r["support_foot_loss_failure"]) for r in records]),
            "prolonged_single_support_failure_rate": mean([float(r["prolonged_single_support_failure"]) for r in records]),
            "unstable_contact_switching_failure_rate": mean([float(r["unstable_contact_switching_failure"]) for r in records]),
            "double_support_fraction": mean([r["double_support_fraction"] for r in records]),
            "single_support_fraction": mean([r["single_support_fraction"] for r in records]),
            "both_feet_airborne_fraction": mean([r["both_feet_airborne_fraction"] for r in records]),
            "commanded_height_drop_m": mean([r["commanded_height_drop_m"] for r in records]),
            "actual_height_drop_m": mean([r["actual_height_drop_m"] for r in records]),
            "hold_height_p95_error_m": mean([r["hold_height_p95_error_m"] for r in records]),
            "minimum_pelvis_height_m": min([r["minimum_pelvis_height_m"] for r in records], default=0.0),
            "vertical_velocity_p95_mps": mean([r["vertical_velocity_p95_mps"] for r in records]),
            "vertical_velocity_max_mps": max([r["vertical_velocity_max_mps"] for r in records], default=0.0),
            "maximum_action_magnitude": max([r["maximum_action_magnitude"] for r in records], default=0.0),
            "residual_action_norm": mean([r["residual_action_norm"] for r in records]),
            "ankle_pitch_residual_saturation_fraction": mean([
                r["ankle_pitch_residual_saturation_fraction"] for r in records
            ]),
            "stabilization_time_s": mean([r["stabilization_time_s"] for r in records]),
        }
        summary = {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "checkpoint": str(checkpoint), "task": args_cli.task, "episodes": args_cli.episodes,
            "seed": args_cli.seed, "crouch_curriculum": "A", "skills": {"CROUCH": crouch},
            "standing_base_option_id": STANDING_BASE_OPTION_ID,
            "skill_success_rate": crouch["success_rate"], "fall_rate": crouch["fall_rate"],
            "failure_reason_counts": dict(sorted(failure_counts.items())), "failure_classes": FAILURE_CLASSES,
        "coordinate_frame": "pelvis height relative to CROUCH entry; no world XY command",
        "contact_failure_definition": {
            "brief_single_support_is_failure": False,
            "both_feet_airborne": "maximum continuous flight > 0.10 s",
            "support_foot_loss": "single-support to flight transition with continuous flight > 0.06 s",
            "prolonged_single_support": "maximum continuous left or right single support > 0.50 s",
            "unstable_contact_switching": "at least 6 switches and switch rate > 4 Hz",
            "foot_contact_loss_fraction_compatibility": "legacy non-double-support fraction; not used as a failure",
        },
    }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        raw_env.close()


if __name__ == "__main__":
    main()
