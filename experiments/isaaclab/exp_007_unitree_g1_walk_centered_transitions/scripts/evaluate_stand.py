"""Formal Stage 1 evaluation of the frozen Stage 2 expert as STAND home state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
EXP005 = REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run"
sys.path[:0] = [str(EXP / "src"), str(EXP005 / "src")]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import (  # noqa: E402
    canonical_state_from_legacy_observation,
    to_walk_observation,
)
from g1_walk_centered.models.transition_bridge import NOT_IMPLEMENTED_STAGE_0, TransitionBridge  # noqa: E402
from g1_walk_centered.tasks.evaluation import (  # noqa: E402
    GATE_THRESHOLDS,
    classify_failures,
    evaluate_gate,
    mean,
    percentile,
    retention_vs_exp006,
    summarize_failures,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


EXPECTED_CHECKPOINT_SHA256 = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
TASK = "Isaac-Velocity-Flat-G1-Run-Eval-v0"
FAILURE_FLAG_NAMES = (
    "stand_settle_failure", "stand_hold_failure", "fall", "torso_contact",
    "excessive_horizontal_motion", "excessive_vertical_motion", "posture_failure",
    "double_support_failure", "flight_failure", "dangerous_support_loss",
    "foot_slip_failure", "saturation_failure", "joint_limit_failure",
    "action_routing_failure", "timeout",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=20260723)
parser.add_argument("--settle-timeout-s", type=float, default=2.0)
parser.add_argument("--settle-hold-s", type=float, default=0.4)
parser.add_argument("--stand-hold-s", type=float, default=8.0)
parser.add_argument("--output", required=True)
parser.add_argument("--run-label", choices=("smoke", "formal"), default="formal")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def primitive(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): primitive(item) for key, item in value.items()}
    return str(value)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    value = lambda name: [float(record[name]) for record in records]
    return {
        "episodes": len(records),
        "settle_success_rate": mean(value("settle_success")),
        "settle_time_mean_s": mean([float(row["settle_time_s"]) for row in records if row["settle_success"]]),
        "stand_hold_success_rate": mean(value("stand_hold_success")),
        "fall_rate": mean(value("fall")),
        "torso_contact_rate": mean(value("torso_contact")),
        "final_double_support_rate": mean(value("final_double_support")),
        "double_support_fraction": mean(value("double_support_fraction")),
        "single_support_fraction": mean(value("single_support_fraction")),
        "flight_fraction": mean(value("flight_fraction")),
        "support_switch_count_mean": mean(value("support_switch_count")),
        "prolonged_single_support_rate": mean(value("prolonged_single_support")),
        "dangerous_support_failure_rate": mean(value("dangerous_support_loss")),
        "horizontal_speed_mean_mps": mean(value("horizontal_speed_mean_mps")),
        "horizontal_speed_p95_mps": mean(value("horizontal_speed_p95_mps")),
        "horizontal_speed_pooled_p95_mps": percentile(
            [sample for row in records for sample in json.loads(row["_horizontal_speed_samples"])], 95
        ),
        "horizontal_speed_max_mps": max(value("horizontal_speed_max_mps")),
        "vertical_speed_mean_mps": mean(value("vertical_speed_mean_mps")),
        "vertical_speed_p95_mps": mean(value("vertical_speed_p95_mps")),
        "yaw_rate_abs_mean_rps": mean(value("yaw_rate_abs_mean_rps")),
        "yaw_rate_abs_p95_rps": mean(value("yaw_rate_abs_p95_rps")),
        "pelvis_height_mean_m": mean(value("pelvis_height_mean_m")),
        "pelvis_height_range_mean_m": mean(value("pelvis_height_range_m")),
        "pelvis_vertical_range_mean_m": mean(value("pelvis_vertical_range_m")),
        "horizontal_drift_mean_m": mean(value("horizontal_drift_m")),
        "horizontal_drift_max_m": max(value("horizontal_drift_m")),
        "final_horizontal_speed_mean_mps": mean(value("final_horizontal_speed_mps")),
        "final_vertical_speed_abs_mean_mps": mean(abs(float(row["final_vertical_speed_mps"])) for row in records),
        "roll_abs_mean_rad": mean(value("roll_abs_mean_rad")),
        "roll_abs_p95_rad": mean(value("roll_abs_p95_rad")),
        "roll_abs_max_rad": max(value("roll_abs_max_rad")),
        "pitch_abs_mean_rad": mean(value("pitch_abs_mean_rad")),
        "pitch_abs_p95_rad": mean(value("pitch_abs_p95_rad")),
        "pitch_abs_max_rad": max(value("pitch_abs_max_rad")),
        "projected_gravity_error_mean_rad": mean(value("projected_gravity_error_mean_rad")),
        "left_contact_fraction": mean(value("left_contact_fraction")),
        "right_contact_fraction": mean(value("right_contact_fraction")),
        "foot_slip_mean_mps": mean(value("foot_slip_mean_mps")),
        "foot_slip_p95_mps": mean(value("foot_slip_p95_mps")),
        "unstable_contact_switching_rate": mean(value("unstable_contact_switching")),
        "contact_asymmetry_mean": mean(value("contact_asymmetry")),
        "both_feet_airborne_event_count": sum(value("both_feet_airborne_event_count")),
        "joint_velocity_saturation_fraction": mean(value("joint_velocity_saturation_fraction")),
        "torque_saturation_fraction": mean(value("torque_saturation_fraction")),
        "ankle_torque_saturation_fraction": mean(value("ankle_torque_saturation_fraction")),
        "knee_velocity_saturation_fraction": mean(value("knee_velocity_saturation_fraction")),
        "joint_limit_proximity_max": max(value("joint_limit_proximity_max")),
        "saturation_failure_rate": mean(value("saturation_failure")),
        "joint_limit_failure_rate": mean(value("joint_limit_failure")),
        "action_magnitude_mean": mean(value("action_magnitude_mean")),
        "action_magnitude_p95": mean(value("action_magnitude_p95")),
        "action_magnitude_max": max(value("action_magnitude_max")),
        "action_rate_mean_per_s": mean(value("action_rate_mean_per_s")),
        "action_rate_p95_per_s": mean(value("action_rate_p95_per_s")),
        "run_contribution_norm_max": max(value("run_contribution_norm_max")),
        "transition_bridge_norm_max": max(value("transition_bridge_norm_max")),
        "scripted_offset_norm_max": max(value("scripted_offset_norm_max")),
        "action_routing_failure_rate": mean(value("action_routing_failure")),
    }


def main() -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash_before = sha256_file(checkpoint)
    if checkpoint_hash_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {checkpoint_hash_before}")

    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.episodes
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    evaluation_config = {
        "task": TASK,
        "terrain": primitive(env_cfg.scene.terrain.terrain_type),
        "episodes": args.episodes,
        "seed": args.seed,
        "physics_timestep_s": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "control_timestep_s": float(env_cfg.sim.dt * env_cfg.decimation),
        "settle": {
            "horizontal_speed_max_mps": 0.08,
            "vertical_speed_max_mps": 0.05,
            "roll_abs_max_rad": 0.10,
            "pitch_abs_max_rad": 0.10,
            "double_support_required": True,
            "hold_duration_s": args.settle_hold_s,
            "timeout_s": args.settle_timeout_s,
        },
        "stand_hold_s": args.stand_hold_s,
        "command": {"vx_mps": 0.0, "vy_mps": 0.0, "yaw_rate_radps": 0.0},
        "external_disturbance": False,
        "reset_base": primitive(env_cfg.events.reset_base.params),
        "reset_robot_joints": primitive(env_cfg.events.reset_robot_joints.params),
        "reset_comparison_to_exp006": "IDENTICAL_TASK_CONFIG_AND_SEED",
        "parallel_episode_independence": "one reset episode per environment; terminal environments excluded immediately",
    }

    with launch_simulation(env_cfg, args):
        raw = gym.make(TASK, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped
        expert = load_walk_expert(checkpoint, device=env.device)
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        contact = env.scene.sensors["contact_forces"]
        foot_body_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_foot_ids = [contact.body_names.index(name) for name in foot_names]
        torso_sensor_ids = [
            index for index, name in enumerate(contact.body_names)
            if "torso" in name.lower() or "pelvis" in name.lower()
        ]
        ankle_joint_ids, _ = robot.find_joints(".*ankle.*")
        knee_joint_ids, _ = robot.find_joints(".*knee.*")
        all_joint_ids, all_joint_names = robot.find_joints(".*")

        wrapped.reset()
        n = args.episodes
        dt = float(env.step_dt)
        initial_root_pos = robot.data.root_pos_w.torch.detach().clone()
        initial_root_quat = robot.data.root_quat_w.torch.detach().clone()
        initial_root_velocity = torch.cat(
            (robot.data.root_lin_vel_w.torch, robot.data.root_ang_vel_w.torch), dim=1
        ).detach().clone()
        initial_joint_pos_noise = (
            robot.data.joint_pos.torch[:, all_joint_ids] - robot.data.default_joint_pos.torch[:, all_joint_ids]
        ).detach().clone()
        initial_joint_velocity = robot.data.joint_vel.torch[:, all_joint_ids].detach().clone()

        settle_required = max(1, round(args.settle_hold_s / dt))
        settle_timeout = max(settle_required, round(args.settle_timeout_s / dt))
        hold_required = max(1, round(args.stand_hold_s / dt))
        active = torch.ones(n, dtype=torch.bool, device=env.device)
        settled = torch.zeros(n, dtype=torch.bool, device=env.device)
        failed_settle = torch.zeros(n, dtype=torch.bool, device=env.device)
        fallen = torch.zeros(n, dtype=torch.bool, device=env.device)
        torso_contact = torch.zeros(n, dtype=torch.bool, device=env.device)
        settle_streak = torch.zeros(n, dtype=torch.long, device=env.device)
        settle_times = torch.zeros(n, device=env.device)
        hold_steps = torch.zeros(n, dtype=torch.long, device=env.device)
        previous_action = torch.zeros(n, 37, device=env.device)
        adapter_exact = True
        finite_all = True
        contact_samples_finite = True
        traces = [{
            "horizontal": [], "vertical": [], "yaw": [], "height": [], "xy": [],
            "roll": [], "pitch": [], "gravity_error": [], "support": [],
            "left_contact": [], "right_contact": [], "slip": [],
            "velocity_sat": [], "torque_sat": [], "ankle_torque_sat": [],
            "knee_velocity_sat": [], "joint_limit": [], "action_magnitude": [],
            "action_rate": [], "run_norm": [], "bridge_norm": [], "scripted_norm": [],
            "switches": 0, "previous_support": None, "flight_run": 0,
            "single_run": 0, "max_flight_run": 0, "max_single_run": 0,
            "airborne_events": 0,
        } for _ in range(n)]

        max_steps = settle_timeout + hold_required + 2
        zero_command = MotionCommand(0.0, 0.0)
        for step in range(max_steps):
            command_term.vel_command_b.zero_()
            observations = wrapped.get_observations()
            legacy = observations["policy"]
            headings = robot.data.heading_w.torch
            state = canonical_state_from_legacy_observation(legacy, heading_w_rad=headings)
            rebuilt = to_walk_observation(state, zero_command)
            adapter_exact = adapter_exact and bool(torch.equal(legacy, rebuilt))
            with torch.inference_mode():
                actions = expert(state, zero_command)
            finite_all = finite_all and bool(torch.isfinite(legacy).all() and torch.isfinite(actions).all())
            run_contribution = torch.zeros_like(actions)
            bridge_contribution = torch.zeros_like(actions)
            scripted_offset = torch.zeros_like(actions)
            routed_actions = actions + run_contribution + bridge_contribution + scripted_offset
            with torch.inference_mode():
                _, _, dones, _ = wrapped.step(routed_actions)
            command_term.vel_command_b.zero_()

            horizontal = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
            vertical_signed = robot.data.root_lin_vel_w.torch[:, 2]
            vertical = vertical_signed.abs()
            yaw = robot.data.root_ang_vel_b.torch[:, 2]
            gravity = robot.data.projected_gravity_b.torch
            roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
            pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1].square() + gravity[:, 2].square()))
            gravity_error = torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))
            force_history = contact.data.net_forces_w_history.torch
            contact_samples_finite = contact_samples_finite and bool(torch.isfinite(force_history).all())
            foot_forces = force_history[:, :, sensor_foot_ids, :]
            contacts = foot_forces.norm(dim=-1).amax(dim=1) > 5.0
            contact_count = contacts.sum(dim=1)
            if torso_sensor_ids:
                torso_force = force_history[:, :, torso_sensor_ids, :].norm(dim=-1).amax(dim=(1, 2))
                torso_contact |= active & (torso_force > 5.0)
            safe_settle = (
                (horizontal <= 0.08) & (vertical <= 0.05)
                & (roll.abs() <= 0.10) & (pitch.abs() <= 0.10)
                & (contact_count == 2)
            )
            waiting = active & ~settled & ~failed_settle
            settle_streak[waiting] = torch.where(
                safe_settle[waiting],
                settle_streak[waiting] + 1,
                torch.zeros_like(settle_streak[waiting]),
            )
            newly_settled = waiting & (settle_streak >= settle_required)
            settled[newly_settled] = True
            settle_times[newly_settled] = (step + 1) * dt
            timed_out = waiting & ~newly_settled & ((step + 1) >= settle_timeout)
            failed_settle[timed_out] = True
            settle_times[timed_out] = (step + 1) * dt

            foot_speed = robot.data.body_lin_vel_w.torch[:, foot_body_ids, :2].norm(dim=-1)
            velocity = robot.data.joint_vel.torch[:, all_joint_ids].abs()
            velocity_limit = robot.data.joint_vel_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
            torque = robot.data.applied_torque.torch[:, all_joint_ids].abs()
            effort_limit = robot.data.joint_effort_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
            velocity_ratio = velocity / velocity_limit
            torque_ratio = torque / effort_limit
            hard_limits = robot.data.joint_pos_limits.torch[:, all_joint_ids]
            joint_position = robot.data.joint_pos.torch[:, all_joint_ids]
            default_position = robot.data.default_joint_pos.torch[:, all_joint_ids]
            upper_excursion = (
                (joint_position - default_position)
                / (hard_limits[..., 1] - default_position).clamp_min(1.0e-6)
            )
            lower_excursion = (
                (default_position - joint_position)
                / (default_position - hard_limits[..., 0]).clamp_min(1.0e-6)
            )
            # Some hand joints have a default pose at a hard-limit endpoint.
            # Measuring excursion from the configured default (the exp_006
            # authority convention) avoids declaring that valid home pose a
            # limit failure while still reaching 1.0 at either approached limit.
            joint_limit = torch.maximum(upper_excursion, lower_excursion).clamp_min(0.0).amax(dim=1)
            action_magnitude = torch.linalg.vector_norm(actions, dim=1)
            action_rate = torch.linalg.vector_norm(actions - previous_action, dim=1) / dt
            previous_action = actions.detach().clone()

            collecting = active & (settled | failed_settle) & (hold_steps < hold_required)
            for env_id in torch.nonzero(collecting, as_tuple=False).flatten().tolist():
                trace = traces[env_id]
                support = int(contact_count[env_id].item())
                trace["horizontal"].append(float(horizontal[env_id].item()))
                trace["vertical"].append(float(vertical[env_id].item()))
                trace.setdefault("vertical_signed", []).append(float(vertical_signed[env_id].item()))
                trace["yaw"].append(float(yaw[env_id].item()))
                trace["height"].append(float(robot.data.root_pos_w.torch[env_id, 2].item()))
                trace["xy"].append(robot.data.root_pos_w.torch[env_id, :2].detach().cpu().tolist())
                trace["roll"].append(float(roll[env_id].item()))
                trace["pitch"].append(float(pitch[env_id].item()))
                trace["gravity_error"].append(float(gravity_error[env_id].item()))
                trace["support"].append(support)
                trace["left_contact"].append(float(contacts[env_id, 0].item()))
                trace["right_contact"].append(float(contacts[env_id, 1].item()))
                trace["slip"].extend([
                    float(foot_speed[env_id, foot].item())
                    for foot in range(2) if bool(contacts[env_id, foot].item())
                ])
                trace["velocity_sat"].append(float(bool((velocity_ratio[env_id] >= 0.95).any().item())))
                trace["torque_sat"].append(float(bool((torque_ratio[env_id] >= 0.95).any().item())))
                trace["ankle_torque_sat"].append(float(bool((torque_ratio[env_id, ankle_joint_ids] >= 0.95).any().item())))
                trace["knee_velocity_sat"].append(float(bool((velocity_ratio[env_id, knee_joint_ids] >= 0.95).any().item())))
                trace["joint_limit"].append(float(joint_limit[env_id].item()))
                trace["action_magnitude"].append(float(action_magnitude[env_id].item()))
                if len(trace["action_magnitude"]) > 1:
                    trace["action_rate"].append(float(action_rate[env_id].item()))
                trace["run_norm"].append(float(torch.linalg.vector_norm(run_contribution[env_id]).item()))
                trace["bridge_norm"].append(float(torch.linalg.vector_norm(bridge_contribution[env_id]).item()))
                trace["scripted_norm"].append(float(torch.linalg.vector_norm(scripted_offset[env_id]).item()))
                if trace["previous_support"] is not None and trace["previous_support"] != support:
                    trace["switches"] += 1
                if support == 0 and trace["previous_support"] != 0:
                    trace["airborne_events"] += 1
                trace["previous_support"] = support
                trace["flight_run"] = trace["flight_run"] + 1 if support == 0 else 0
                trace["single_run"] = trace["single_run"] + 1 if support == 1 else 0
                trace["max_flight_run"] = max(trace["max_flight_run"], trace["flight_run"])
                trace["max_single_run"] = max(trace["max_single_run"], trace["single_run"])
            hold_steps[collecting] += 1
            fallen |= active & dones.bool()
            active[dones.bool()] = False
            active[(settled | failed_settle) & (hold_steps >= hold_required)] = False
            if not bool(active.any().item()):
                break

        # Empirical auto-reset probe after the measured window.  It is excluded
        # from every formal trace and forces only the task time-out termination.
        env.episode_length_buf[:] = int(env.max_episode_length) - 1
        command_term.vel_command_b.zero_()
        probe_observations = wrapped.get_observations()
        probe_state = canonical_state_from_legacy_observation(
            probe_observations["policy"], heading_w_rad=robot.data.heading_w.torch
        )
        with torch.inference_mode():
            probe_actions = expert(probe_state, zero_command)
            _, _, probe_dones, _ = wrapped.step(probe_actions)
        auto_reset_probe = {
            "timeout_done_all": bool(probe_dones.bool().all().item()),
            "episode_length_reset_all": bool((env.episode_length_buf <= 1).all().item()),
            "post_reset_state_finite": bool(
                torch.isfinite(robot.data.root_pos_w.torch).all()
                and torch.isfinite(robot.data.joint_pos.torch).all()
            ),
        }
        auto_reset_probe["passed"] = all(auto_reset_probe.values())

        records = []
        for env_id, trace in enumerate(traces):
            support = trace["support"]
            horizontal = trace["horizontal"]
            height = trace["height"]
            complete = len(horizontal) >= hold_required
            flight_fraction = mean(float(state == 0) for state in support)
            max_flight_s = trace["max_flight_run"] * dt
            max_single_s = trace["max_single_run"] * dt
            dangerous = max_flight_s > 0.10 or max_single_s > 0.50
            velocity_sat = mean(trace["velocity_sat"])
            torque_sat = mean(trace["torque_sat"])
            ankle_sat = mean(trace["ankle_torque_sat"])
            knee_sat = mean(trace["knee_velocity_sat"])
            saturation_failure = velocity_sat > 0.05 or torque_sat > 0.05
            joint_limit_max = max(trace["joint_limit"], default=0.0)
            routing_failure = any(
                max(trace[name], default=0.0) != 0.0
                for name in ("run_norm", "bridge_norm", "scripted_norm")
            )
            final_double = bool(support and support[-1] == 2)
            drift = (
                math.dist(trace["xy"][0], trace["xy"][-1])
                if len(trace["xy"]) >= 2 else 0.0
            )
            hold_success = bool(
                settled[env_id] and complete and not fallen[env_id] and not torso_contact[env_id]
                and mean(horizontal) <= 0.05 and percentile(horizontal, 95) <= 0.10
                and (max(height) - min(height) if height else math.inf) <= 0.04
                and flight_fraction == 0.0 and not dangerous and not saturation_failure
                and joint_limit_max < 0.95 and final_double and not routing_failure
            )
            record = {
                "episode": env_id,
                "seed": args.seed,
                "checkpoint": str(checkpoint),
                "settle_success": bool(settled[env_id].item()),
                "settle_time_s": float(settle_times[env_id].item()),
                "stand_hold_success": hold_success,
                "fall": bool(fallen[env_id].item()),
                "torso_contact": bool(torso_contact[env_id].item()),
                "hold_duration_s": len(horizontal) * dt,
                "final_double_support": final_double,
                "double_support_fraction": mean(float(state == 2) for state in support),
                "single_support_fraction": mean(float(state == 1) for state in support),
                "flight_fraction": flight_fraction,
                "support_switch_count": trace["switches"],
                "maximum_both_feet_airborne_s": max_flight_s,
                "maximum_single_support_s": max_single_s,
                "prolonged_single_support": max_single_s > 0.50,
                "dangerous_support_loss": dangerous,
                "horizontal_speed_mean_mps": mean(horizontal),
                "horizontal_speed_p95_mps": percentile(horizontal, 95),
                "horizontal_speed_max_mps": max(horizontal, default=0.0),
                "vertical_speed_mean_mps": mean(trace["vertical"]),
                "vertical_speed_p95_mps": percentile(trace["vertical"], 95),
                "yaw_rate_abs_mean_rps": mean(abs(value) for value in trace["yaw"]),
                "yaw_rate_abs_p95_rps": percentile([abs(value) for value in trace["yaw"]], 95),
                "pelvis_height_mean_m": mean(height),
                "pelvis_height_range_m": max(height) - min(height) if height else 0.0,
                "pelvis_vertical_range_m": max(height) - min(height) if height else 0.0,
                "horizontal_drift_m": drift,
                "final_horizontal_speed_mps": horizontal[-1] if horizontal else 0.0,
                "final_vertical_speed_mps": trace.get("vertical_signed", [0.0])[-1],
                "roll_abs_mean_rad": mean(abs(value) for value in trace["roll"]),
                "roll_abs_p95_rad": percentile([abs(value) for value in trace["roll"]], 95),
                "roll_abs_max_rad": max([abs(value) for value in trace["roll"]], default=0.0),
                "pitch_abs_mean_rad": mean(abs(value) for value in trace["pitch"]),
                "pitch_abs_p95_rad": percentile([abs(value) for value in trace["pitch"]], 95),
                "pitch_abs_max_rad": max([abs(value) for value in trace["pitch"]], default=0.0),
                "projected_gravity_error_mean_rad": mean(trace["gravity_error"]),
                "final_roll_rad": trace["roll"][-1] if trace["roll"] else 0.0,
                "final_pitch_rad": trace["pitch"][-1] if trace["pitch"] else 0.0,
                "left_contact_fraction": mean(trace["left_contact"]),
                "right_contact_fraction": mean(trace["right_contact"]),
                "foot_slip_mean_mps": mean(trace["slip"]),
                "foot_slip_p95_mps": percentile(trace["slip"], 95),
                "unstable_contact_switching": trace["switches"] > 4,
                "contact_asymmetry": abs(mean(trace["left_contact"]) - mean(trace["right_contact"])),
                "both_feet_airborne_event_count": trace["airborne_events"],
                "joint_velocity_saturation_fraction": velocity_sat,
                "torque_saturation_fraction": torque_sat,
                "ankle_torque_saturation_fraction": ankle_sat,
                "knee_velocity_saturation_fraction": knee_sat,
                "joint_limit_proximity_max": joint_limit_max,
                "saturation_failure": saturation_failure,
                "joint_limit_failure": joint_limit_max >= 0.95,
                "action_magnitude_mean": mean(trace["action_magnitude"]),
                "action_magnitude_p95": percentile(trace["action_magnitude"], 95),
                "action_magnitude_max": max(trace["action_magnitude"], default=0.0),
                "action_rate_mean_per_s": mean(trace["action_rate"]),
                "action_rate_p95_per_s": percentile(trace["action_rate"], 95),
                "run_contribution_norm_max": max(trace["run_norm"], default=0.0),
                "transition_bridge_norm_max": max(trace["bridge_norm"], default=0.0),
                "scripted_offset_norm_max": max(trace["scripted_norm"], default=0.0),
                "action_routing_failure": routing_failure,
                "timeout": not complete and not bool(fallen[env_id].item()),
                "reset_root_position_w": json.dumps(initial_root_pos[env_id].cpu().tolist()),
                "reset_root_quaternion_wxyz": json.dumps(initial_root_quat[env_id].cpu().tolist()),
                "reset_root_velocity_w": json.dumps(initial_root_velocity[env_id].cpu().tolist()),
                "reset_joint_position_noise_abs_max_rad": float(initial_joint_pos_noise[env_id].abs().max().item()),
                "reset_joint_velocity_abs_max_rps": float(initial_joint_velocity[env_id].abs().max().item()),
                "_horizontal_speed_samples": json.dumps(horizontal),
            }
            primary, flags = classify_failures(record)
            record["primary_failure"] = primary
            record["failure_flags"] = flags
            for name in FAILURE_FLAG_NAMES:
                record[f"failure_{name}"] = flags[name]
            records.append(record)

        metrics = aggregate(records)
        stage0_gate = json.loads(
            (REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/stage0_gate.json").read_text(encoding="utf-8")
        )
        stage0_bitwise = json.loads(
            (REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/bitwise_reference.json").read_text(encoding="utf-8")
        )
        stage0_action_order = json.loads(
            (REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/action_order.json").read_text(encoding="utf-8")
        )
        checkpoint_hash_after = sha256_file(checkpoint)
        routing_preflight = {
            "checkpoint_sha256_match": checkpoint_hash_before == EXPECTED_CHECKPOINT_SHA256,
            "expert_weights_unchanged": checkpoint_hash_before == checkpoint_hash_after,
            "actor_loaded": True,
            "actor_trainable_parameters": sum(parameter.requires_grad for parameter in expert.actor.parameters()),
            "action_order_match": all_joint_names == stage0_action_order["joint_names"],
            "adapter_observation_bitwise_equal_every_step": adapter_exact,
            "stage0_eligible": bool(stage0_gate["eligible_for_stage1"]),
            "stage0_adapter_reference_maintained": bool(stage0_bitwise["all_adapter_actions_bitwise_equal"]),
            "run_contribution_bitwise_zero": metrics["run_contribution_norm_max"] == 0.0,
            "transition_bridge_contribution_bitwise_zero": metrics["transition_bridge_norm_max"] == 0.0,
            "scripted_offset_bitwise_zero": metrics["scripted_offset_norm_max"] == 0.0,
            "transition_bridge_invoked_in_formal_path": False,
            "transition_bridge_interface_probe": TransitionBridge()(),
            "transition_bridge_interface_expected": NOT_IMPLEMENTED_STAGE_0,
            "finite_observations_actions": finite_all,
            "contact_samples_finite": contact_samples_finite,
            "contact_sensor_update_period_s": float(contact.cfg.update_period),
            "contact_sensor_history_length": int(contact.cfg.history_length),
            "stale_contact_sensor_check": contact_samples_finite and float(contact.cfg.update_period) == 0.0,
            "explicit_env_reset_called": True,
            "manager_based_auto_reset_contract": True,
            "empirical_auto_reset_probe": auto_reset_probe,
            "parallel_episode_independence": True,
            "command_zero_every_step": True,
            "router_invoked": False,
        }
        static_checks = {
            "checkpoint_sha256_match": routing_preflight["checkpoint_sha256_match"],
            "expert_weights_unchanged": routing_preflight["expert_weights_unchanged"],
            "action_order_match": routing_preflight["action_order_match"],
            "run_contribution_bitwise_zero": routing_preflight["run_contribution_bitwise_zero"],
            "transition_bridge_output_bitwise_zero": routing_preflight["transition_bridge_contribution_bitwise_zero"],
            "scripted_offset_bitwise_zero": routing_preflight["scripted_offset_bitwise_zero"],
            "stage0_adapter_reference_maintained": routing_preflight["stage0_adapter_reference_maintained"] and adapter_exact,
            "finite_observations_actions": finite_all,
            "contact_sensor_fresh": routing_preflight["stale_contact_sensor_check"],
            "auto_reset_probe": auto_reset_probe["passed"],
        }
        gate_pass, gate_failures = evaluate_gate(metrics, static_checks)
        retention = retention_vs_exp006(metrics)
        warnings = [
            "exp_007 observes 8.0 s while the exp_006 reference observed 6.0 s; the longer window is stricter."
        ]
        if args.run_label == "smoke":
            warnings.append("SMOKE_ONLY: these episodes are not formal evidence.")
        summary = {
            "stage": 1,
            "run_label": args.run_label,
            "status": "PASS" if gate_pass else "FAIL",
            "eligible_for_stage2": bool(gate_pass and args.run_label == "formal" and args.episodes == 50),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash_after,
            "metrics": metrics,
            "evaluation_config": evaluation_config,
            "routing": routing_preflight,
            "retention": retention,
            "failures": gate_failures,
            "warnings": warnings,
            "git_revision": git_revision(),
            "training_performed": False,
            "transition_evaluation_performed": False,
        }
        gate = {
            "stage": 1,
            "status": summary["status"],
            "eligible_for_stage2": summary["eligible_for_stage2"],
            "failures": gate_failures,
            "warnings": warnings,
            "metrics": metrics,
            "thresholds": GATE_THRESHOLDS,
            "retention": retention["status"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash_after,
            "stage0_reference": {
                "commit": "feb2bc10ecefa37dbd407c81d2e2b5d158da69a5",
                "eligible_for_stage1": stage0_gate["eligible_for_stage1"],
                "adapter_bitwise": stage0_bitwise["all_adapter_actions_bitwise_equal"],
            },
            "git_revision": git_revision(),
            "static_checks": static_checks,
        }
        failure_counts = summarize_failures(records)
        csv_records = []
        for record in records:
            value = dict(record)
            value["failure_flags"] = json.dumps(value["failure_flags"], sort_keys=True)
            value.pop("_horizontal_speed_samples")
            csv_records.append(value)
        write_csv(output / "episodes.csv", csv_records)
        long_rows = []
        for record in csv_records:
            for name, value in record.items():
                if name in {"episode", "seed"} or not isinstance(value, (int, float, bool)):
                    continue
                long_rows.append({"episode": record["episode"], "metric": name, "value": value})
        write_csv(output / "metrics_long.csv", long_rows)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output / "gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        (output / "failure_counts.json").write_text(json.dumps(failure_counts, indent=2) + "\n", encoding="utf-8")
        (output / "retention_vs_exp006.json").write_text(json.dumps(retention, indent=2) + "\n", encoding="utf-8")
        (output / "routing_preflight.json").write_text(json.dumps(routing_preflight, indent=2) + "\n", encoding="utf-8")
        provenance = {
            "checkpoint_reference": str(checkpoint.relative_to(REPO)),
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
            "sha256_before": checkpoint_hash_before,
            "sha256_after": checkpoint_hash_after,
            "copied": False,
            "modified": False,
            "actor": "LegacyWalkActor 123->256->128->128->37",
            "source_experiment": "exp_005_unitree_g1_flat_run Stage 2",
        }
        (output / "checkpoint_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        reproduction = (
            "cd \"$HOME\\workspace\\physical-ai-lab\"\n"
            ".\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions"
            "\\scripts\\evaluate_stand.ps1\n"
        )
        (output / "reproduction_command.txt").write_text(reproduction, encoding="utf-8")
        print(json.dumps({
            "status": summary["status"],
            "eligible_for_stage2": summary["eligible_for_stage2"],
            "metrics": metrics,
            "failures": gate_failures,
            "retention": retention["status"],
            "output": str(output),
        }, indent=2))
        wrapped.close()
        if not gate_pass:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
