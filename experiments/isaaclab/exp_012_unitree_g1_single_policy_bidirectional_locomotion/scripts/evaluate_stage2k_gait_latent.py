"""Closed-loop endpoint, authority, toggle, and matrix evaluation for Stage 2K."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"
RAW = OUT / "raw"
STUDENT = OUT / "student/selected_gait_latent_student.pt"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("endpoints", "authority0", "authority1", "toggleA", "toggleB", "matrix"), required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_elbow_roll_joint",
    "right_elbow_roll_joint", "left_five_joint", "left_three_joint", "left_zero_joint",
    "right_five_joint", "right_three_joint", "right_zero_joint", "left_six_joint",
    "left_four_joint", "left_one_joint", "right_six_joint", "right_four_joint",
    "right_one_joint", "left_two_joint", "right_two_joint",
]


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_base_weight = nn.Parameter(torch.empty(256, 123))
        self.first_gait_column = nn.Parameter(torch.empty(256, 1))
        self.first_bias = nn.Parameter(torch.empty(256))
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.std = nn.Parameter(torch.empty(37))

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        first = first + gait.reshape(-1, 1) * self.first_gait_column.T
        return self.hidden(first)


def minimum_jerk(value):
    value = torch.clamp(value, 0.0, 1.0)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions(mode):
    if mode == "endpoints":
        return [
            {"name": "walk_1p2", "speed": 1.2, "gait": 0.0, "episodes": 100, "duration": 10.0},
            {"name": "run_1p2", "speed": 1.2, "gait": 1.0, "episodes": 100, "duration": 10.0},
            {"name": "run_2p4", "speed": 2.4, "gait": 1.0, "episodes": 100, "duration": 10.0},
            {"name": "run_2p6", "speed": 2.6, "gait": 1.0, "episodes": 100, "duration": 10.0},
        ]
    if mode.startswith("authority"):
        gait = 0.0 if mode == "authority0" else 1.0
        return [{"name": mode, "speed": 1.2, "gait": gait, "episodes": 100, "duration": 10.0}]
    if mode.startswith("toggle"):
        return [{"name": mode, "speed": 1.2, "gait": None, "episodes": 100, "duration": 12.0}]
    return [
        {"name": f"speed_{speed:.2f}_gait_{gait:.2f}", "speed": speed, "gait": gait, "episodes": 20, "duration": 10.0}
        for speed in (.6, .8, 1.0, 1.2, 2.0, 2.4, 2.6)
        for gait in (0.0, .25, .5, .75, 1.0)
    ]


def gait_at(mode, t, count, device):
    if mode == "toggleA":
        if t < 5:
            return torch.zeros(count, device=device)
        if t < 7:
            return minimum_jerk(torch.full((count,), (t - 5) / 2, device=device))
        return torch.ones(count, device=device)
    if mode == "toggleB":
        if t < 5:
            return torch.ones(count, device=device)
        if t < 7:
            return 1 - minimum_jerk(torch.full((count,), (t - 5) / 2, device=device))
        return torch.zeros(count, device=device)
    raise RuntimeError("not a toggle")


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    output = {}
    for name, values in grouped.items():
        output[name] = {
            "episodes": len(values),
            "walk_like_rate": sum(row["gait_classification"] == "WALK_LIKE" for row in values) / len(values),
            "periodic_running_rate": sum(row["gait_classification"] == "PERIODIC_RUNNING" for row in values) / len(values),
            "fall_rate": sum(row["fall"] for row in values) / len(values),
            "speed_mae": sum(row["speed_mae"] for row in values) / len(values),
            "flight_fraction": sum(row["flight_fraction"] for row in values) / len(values),
            "stride_frequency_hz": sum(row["stride_frequency_hz"] for row in values) / len(values),
            "double_support_fraction": sum(row["double_support_fraction"] for row in values) / len(values),
            "heading_p95": sum(row["heading_p95"] for row in values) / len(values),
            "dangerous_slip_rate": sum(row["dangerous_slip"] for row in values) / len(values),
            "impact_failure_rate": sum(row["impact_failure"] for row in values) / len(values),
            "long_dwell_saturation_rate": sum(row["long_dwell_saturation"] for row in values) / len(values),
            "completion_reward_fires": sum(row["completion_reward_fires"] for row in values),
            "base_height_mean": sum(row["base_height_mean"] for row in values) / len(values),
            "base_pitch_abs_mean": sum(row["base_pitch_abs_mean"] for row in values) / len(values),
            "vertical_velocity_abs_mean": sum(row["vertical_velocity_abs_mean"] for row in values) / len(values),
            "action_norm_mean": sum(row["action_norm_mean"] for row in values) / len(values),
        }
    return output


def main():
    specs = conditions(args.mode)
    count = sum(spec["episodes"] for spec in specs)
    duration = max(spec["duration"] for spec in specs)
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = count
    cfg.episode_length_s = duration
    cfg.seed = 20267121
    agent_cfg.seed = 20267121
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    payload = torch.load(STUDENT, map_location="cpu", weights_only=False)
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        # Constructing the runner validates the unchanged observation/action contract; it is never updated.
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        student = Student().to(runner.device)
        student.load_state_dict(payload["model_state_dict"], strict=True)
        student.eval()
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        reward_term = env.reward_manager.get_term_cfg("safe_periodic_flight").func
        sensor = env.scene.sensors["contact_forces"]
        sensor_feet = [index for index, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [
            next(index for index, name in enumerate(robot.body_names) if name == sensor.body_names[sensor_id])
            for sensor_id in sensor_feet
        ]
        spec_id = torch.empty(count, dtype=torch.long, device=runner.device)
        speed = torch.empty(count, device=runner.device)
        fixed_gait = torch.empty(count, device=runner.device)
        cursor = 0
        for index, spec in enumerate(specs):
            right = cursor + spec["episodes"]
            spec_id[cursor:right] = index
            speed[cursor:right] = spec["speed"]
            fixed_gait[cursor:right] = spec["gait"] if spec["gait"] is not None else 0
            cursor = right
        command.external_override[:, 0] = speed
        command.external_override[:, 1:] = 0
        obs, _ = wrapped.reset()
        obs = obs.to(runner.device)
        initial_observation_hash = hashlib.sha256(obs["policy"].detach().cpu().numpy().tobytes()).hexdigest()
        reference_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        steps = round(duration / float(env.step_dt))
        fallen = torch.zeros(count, dtype=torch.bool, device=runner.device)
        speed_error = torch.zeros(count, device=runner.device)
        flight_steps = torch.zeros(count, device=runner.device)
        flight_events = torch.zeros(count, dtype=torch.long, device=runner.device)
        flight_streak = torch.zeros_like(flight_events)
        maximum_flight = torch.zeros_like(flight_events)
        safe_flights = torch.zeros_like(flight_events)
        alternating = torch.zeros_like(flight_events)
        last_landing = torch.full_like(flight_events, -1)
        double_steps = torch.zeros(count, device=runner.device)
        heading_values = []
        slip_streak = torch.zeros_like(flight_events)
        dangerous_slip = torch.zeros_like(fallen)
        impact = torch.zeros_like(fallen)
        saturation_streak = torch.zeros_like(flight_events)
        saturation = torch.zeros_like(fallen)
        completions = torch.zeros_like(flight_events)
        base_height = torch.zeros(count, device=runner.device)
        pitch_abs = torch.zeros(count, device=runner.device)
        vertical_abs = torch.zeros(count, device=runner.device)
        action_norm = torch.zeros(count, device=runner.device)
        endpoint_samples = []
        # Toggle-only phase counters.
        source_flight = torch.zeros(count, device=runner.device)
        source_steps = torch.zeros(count, device=runner.device)
        target_flight = torch.zeros(count, device=runner.device)
        target_steps = torch.zeros(count, device=runner.device)
        target_events = torch.zeros(count, dtype=torch.long, device=runner.device)
        target_alternating = torch.zeros_like(target_events)
        transition_time = torch.full((count,), float("nan"), device=runner.device)
        for step in range(steps):
            t = step * float(env.step_dt)
            gait = gait_at(args.mode, t, count, runner.device) if args.mode.startswith("toggle") else fixed_gait
            command.external_override[:, 0] = speed
            command.external_override[:, 1:] = 0
            if step == 0:
                obs = wrapped.get_observations().to(runner.device)
            with torch.inference_mode():
                action = student(obs["policy"], gait)
            if args.mode == "endpoints" and step % 10 == 0:
                endpoint_samples.append({
                    "observation": obs["policy"].detach().cpu(),
                    "spec_id": spec_id.detach().cpu(),
                })
            obs, _, dones, extras = wrapped.step(action)
            obs = obs.to(runner.device)
            timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
            fallen |= dones.bool() & ~timeout
            actual = robot.data.root_lin_vel_b[:, 0]
            speed_error += (actual - speed).abs()
            forces = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1)
            contacts = forces > 5
            in_flight = contacts.sum(-1) == 0
            previous = flight_streak.clone()
            flight_events += (in_flight & (flight_streak == 0)).long()
            flight_steps += in_flight.float()
            double_steps += (contacts.sum(-1) == 2).float()
            flight_streak = torch.where(in_flight, flight_streak + 1, torch.zeros_like(flight_streak))
            maximum_flight = torch.maximum(maximum_flight, flight_streak)
            landing = ~in_flight & (previous > 0)
            single = landing & (contacts.sum(-1) == 1)
            foot = contacts.long().argmax(-1)
            safe = single & (previous >= 2) & (previous <= 8)
            alt = safe & (last_landing >= 0) & (foot != last_landing)
            safe_flights += safe.long()
            alternating += alt.long()
            last_landing[single] = foot[single]
            completions += (reward_term.last_raw_reward >= 1.0).long()
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
            slipping = ((foot_speed > .55) & contacts).any(-1)
            slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
            dangerous_slip |= slip_streak >= 5
            impact |= forces.amax(-1) > 3500
            limits = robot.data.joint_vel_limits
            if limits.ndim == 3:
                limits = limits[..., 1].abs()
            saturated = (robot.data.joint_vel.abs() / limits.clamp_min(1e-6) > .95).any(-1)
            saturation_streak = torch.where(saturated, saturation_streak + 1, torch.zeros_like(saturation_streak))
            saturation |= saturation_streak >= 5
            heading = wrapped_heading_error(reference_yaw, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs()
            heading_values.append(heading.detach().cpu())
            projected = obs["policy"][:, 6:9]
            pitch = torch.atan2(-projected[:, 0], torch.sqrt(projected[:, 1].square() + projected[:, 2].square()).clamp_min(1e-8))
            base_height += robot.data.root_pos_w[:, 2]
            pitch_abs += pitch.abs()
            vertical_abs += robot.data.root_lin_vel_b[:, 2].abs()
            action_norm += action.norm(dim=-1)
            if args.mode.startswith("toggle"):
                source = t < 5
                target = t >= 7
                if source:
                    source_flight += in_flight.float()
                    source_steps += 1
                if target:
                    target_flight += in_flight.float()
                    target_steps += 1
                    target_events += (in_flight & (previous == 0)).long()
                    target_alternating += alt.long()
                    target_is_run = args.mode == "toggleA"
                    acquired = (
                        (target_events >= 4) & (target_alternating >= 3)
                        if target_is_run else
                        ((target_flight / target_steps.clamp_min(1)) < .10) & (target_steps >= 50)
                    )
                    newly = acquired & torch.isnan(transition_time)
                    transition_time[newly] = t - 7
        heading_tensor = torch.stack(heading_values)
        rows = []
        for env_id in range(count):
            spec = specs[int(spec_id[env_id])]
            periodic = int(flight_events[env_id]) >= 4 and int(safe_flights[env_id]) >= 3 and int(alternating[env_id]) >= 3
            flight_fraction = float(flight_steps[env_id] / steps)
            gait_label = (
                "FALL" if bool(fallen[env_id]) else "PERIODIC_RUNNING" if periodic else
                "WALK_LIKE" if flight_fraction < .10 else "ISOLATED_FLIGHT"
            )
            row = {
                "condition": spec["name"], "episode": env_id - sum(item["episodes"] for item in specs[:int(spec_id[env_id])]),
                "target_speed": spec["speed"], "gait_cmd": spec["gait"] if spec["gait"] is not None else args.mode,
                "gait_classification": gait_label, "fall": bool(fallen[env_id]),
                "speed_mae": float(speed_error[env_id] / steps),
                "flight_fraction": flight_fraction, "max_flight_duration_s": int(maximum_flight[env_id]) * float(env.step_dt),
                "flight_event_count": int(flight_events[env_id]), "safe_flight_count": int(safe_flights[env_id]),
                "alternating_landing_count": int(alternating[env_id]),
                "stride_frequency_hz": float(flight_events[env_id]) / duration,
                "double_support_fraction": float(double_steps[env_id] / steps),
                "heading_p95": float(torch.quantile(heading_tensor[:, env_id], .95)),
                "dangerous_slip": bool(dangerous_slip[env_id]), "impact_failure": bool(impact[env_id]),
                "long_dwell_saturation": bool(saturation[env_id]),
                "completion_reward_fires": int(completions[env_id]),
                "base_height_mean": float(base_height[env_id] / steps),
                "base_pitch_abs_mean": float(pitch_abs[env_id] / steps),
                "vertical_velocity_abs_mean": float(vertical_abs[env_id] / steps),
                "action_norm_mean": float(action_norm[env_id] / steps),
            }
            if args.mode.startswith("toggle"):
                row.update({
                    "source_flight_fraction": float(source_flight[env_id] / source_steps[env_id].clamp_min(1)),
                    "target_flight_fraction": float(target_flight[env_id] / target_steps[env_id].clamp_min(1)),
                    "target_flight_events": int(target_events[env_id]),
                    "target_alternating_landings": int(target_alternating[env_id]),
                    "transition_time_s": float(transition_time[env_id]) if torch.isfinite(transition_time[env_id]) else "",
                })
            rows.append(row)
        output_csv = RAW / f"{args.mode}_evaluation.csv"
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = {
            "mode": args.mode, "seed": 20267121, "student_sha256": sha(STUDENT),
            "initial_observation_sha256": initial_observation_hash, "summary": summarize(rows),
            "teacher_calls": 0, "expert_calls": 0, "checkpoint_switches": 0, "action_blends": 0,
        }
        (RAW / f"{args.mode}_evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if endpoint_samples:
            torch.save(endpoint_samples, RAW / "student_endpoint_state_samples.pt")
        wrapped.close()


if __name__ == "__main__":
    main()
