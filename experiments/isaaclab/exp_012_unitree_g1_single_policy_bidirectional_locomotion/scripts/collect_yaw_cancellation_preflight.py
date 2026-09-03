"""Collect one frozen-parent condition for the Stage 1B yaw-cancellation preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage1b_speed_conditioned_yaw_cancellation"
CKPT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
EXPECTED_SHA = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.yaw_bias_canceller import G1SpeedConditionedYawBiasCancellerV1, minimum_jerk  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("steady", "transition"), required=True)
parser.add_argument("--controller", choices=("off", "on"), required=True)
parser.add_argument("--seed", type=int, default=20261201)
parser.add_argument("--checkpoint", type=Path, default=CKPT)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

STEADY_SPEEDS = (0.0, 0.6, 0.8, 1.0, 1.2)
SEQUENCE_SPEEDS = (0.0, 0.6, 0.8, 1.0, 1.2, 1.0, 0.8, 0.6, 0.0)
HOLD_S = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3.0)
RAMP_S = 1.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence_command(time_s: float) -> tuple[float, int, str]:
    cursor = 0.0
    for index, speed in enumerate(SEQUENCE_SPEEDS):
        hold = HOLD_S[index]
        if cursor <= time_s < cursor + hold:
            return speed, index, "hold"
        cursor += hold
        if index == len(SEQUENCE_SPEEDS) - 1:
            break
        if cursor <= time_s < cursor + RAMP_S:
            tau = (time_s - cursor) / RAMP_S
            source, target = speed, SEQUENCE_SPEEDS[index + 1]
            return source + (target - source) * minimum_jerk(tau), index, "ramp"
        cursor += RAMP_S
    return 0.0, len(SEQUENCE_SPEEDS) - 1, "final_hold"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(args.checkpoint) != EXPECTED_SHA:
        raise RuntimeError("parent checkpoint SHA mismatch")
    episodes = 50
    num_envs = episodes * len(STEADY_SPEEDS) if args.mode == "steady" else episodes
    duration_s = 8.0 if args.mode == "steady" else sum(HOLD_S) + RAMP_S * (len(SEQUENCE_SPEEDS) - 1)
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = duration_s + 2.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    agent_cfg.seed = args.seed
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device

    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        env = raw.unwrapped
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        term.external_override.zero_()
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args.checkpoint.resolve()), strict=True, map_location=runner.device)
        policy = runner.get_inference_policy(device=runner.device)
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        sensor_feet = [index for index, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        foot_bodies, _ = robot.find_bodies(".*_ankle_roll_link", preserve_order=True)
        all_joints, _ = robot.find_joints(".*", preserve_order=True)
        dt = float(env.step_dt)
        steps = round(duration_s / dt)
        controllers = [G1SpeedConditionedYawBiasCancellerV1(dt) for _ in range(num_envs)]

        obs, _ = wrapped.reset()
        obs = obs.to(runner.device)
        if args.mode == "steady":
            target_speed = torch.tensor(
                [speed for speed in STEADY_SPEEDS for _ in range(episodes)], device=runner.device
            )
        else:
            target_speed = torch.zeros(num_envs, device=runner.device)
        alive = torch.ones(num_envs, dtype=torch.bool, device=runner.device)
        fell = torch.zeros(num_envs, dtype=torch.bool, device=runner.device)
        counts = torch.zeros(num_envs, device=runner.device)
        sums = {
            key: torch.zeros(num_envs, device=runner.device)
            for key in ("speed", "speed_error", "yaw", "lateral", "tilt", "slip", "impact", "flight",
                        "double_support", "single_support", "velocity_saturation", "torque_saturation", "offset_abs")
        }
        heading = torch.full((steps, num_envs), torch.nan, device=runner.device)
        yaw_abs = torch.full_like(heading, torch.nan)
        action_rate = torch.full_like(heading, torch.nan)
        references = torch.zeros(num_envs, device=runner.device)
        velocity_dwell = torch.zeros((num_envs, len(all_joints)), device=runner.device)
        torque_dwell = torch.zeros_like(velocity_dwell)
        long_saturation = torch.zeros(num_envs, dtype=torch.bool, device=runner.device)
        previous_action = torch.zeros((num_envs, 37), device=runner.device)
        segment_values = [[{"speed_error": [], "yaw": []} for _ in SEQUENCE_SPEEDS] for _ in range(num_envs)]
        activation_jumps = [[] for _ in range(num_envs)]
        deactivation_jumps = [[] for _ in range(num_envs)]
        previous_states = ["DISABLED"] * num_envs
        stand_action_hash = hashlib.sha256()
        stand_command_hash = hashlib.sha256()
        offset_max_stand = 0.0
        final_stand_good = torch.zeros(num_envs, device=runner.device)
        final_stand_count = torch.zeros(num_envs, device=runner.device)

        for step in range(steps):
            time_s = step * dt
            if args.mode == "transition":
                speed_value, segment, segment_phase = sequence_command(time_s)
                target_speed.fill_(speed_value)
            else:
                segment, segment_phase = 0, "steady"
            offsets, policy_yaws, states = [], [], []
            for env_id, controller in enumerate(controllers):
                result = controller.step(float(target_speed[env_id]), 0.0) if args.controller == "on" else {
                    "offset": 0.0, "policy_yaw_rate": 0.0, "state": "DISABLED"
                }
                offsets.append(float(result["offset"]))
                policy_yaws.append(float(result["policy_yaw_rate"]))
                states.append(str(result["state"]))
            offset_tensor = torch.tensor(offsets, device=runner.device)
            term.external_override[:, 0] = target_speed
            term.external_override[:, 1] = 0.0
            term.external_override[:, 2] = torch.tensor(policy_yaws, device=runner.device)
            with torch.inference_mode():
                action = policy(obs)
            obs, _, dones, _ = wrapped.step(action)
            obs = obs.to(runner.device)
            rate = torch.linalg.vector_norm(action - previous_action, dim=1)
            previous_action.copy_(action)
            if args.mode == "steady":
                stand = target_speed == 0.0
                stand_action_hash.update(action[stand].detach().cpu().contiguous().numpy().tobytes())
                stand_command_hash.update(term.external_override[stand].detach().cpu().contiguous().numpy().tobytes())
                offset_max_stand = max(offset_max_stand, float(offset_tensor[stand].abs().max()))

            just_fell = dones.bool() & alive
            fell |= just_fell
            valid = alive & ~just_fell
            alive &= ~just_fell
            reference_time = 2.0 if args.mode == "steady" else HOLD_S[0]
            if step == round(reference_time / dt):
                references.copy_(robot.data.heading_w)
            quality = valid & (time_s >= reference_time)
            if quality.any():
                speed = robot.data.root_lin_vel_b[:, 0]
                yaw = robot.data.root_ang_vel_w[:, 2]
                heading_error = torch.atan2(
                    torch.sin(robot.data.heading_w - references),
                    torch.cos(robot.data.heading_w - references),
                )
                tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
                force_history = sensor.data.net_forces_w_history[:, -1, sensor_feet, :]
                forces = force_history.norm(dim=-1)
                contacts = forces > 5.0
                foot_speed = robot.data.body_lin_vel_w[:, foot_bodies, :2].norm(dim=-1)
                slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(dim=1)
                impact = forces.amax(dim=1)
                velocity_ratio = (
                    robot.data.joint_vel[:, all_joints].abs()
                    / robot.data.joint_vel_limits[:, all_joints].abs().clamp_min(1.0e-6)
                )
                torque_ratio = (
                    robot.data.applied_torque[:, all_joints].abs()
                    / robot.data.joint_effort_limits[:, all_joints].abs().clamp_min(1.0e-6)
                )
                values = {
                    "speed": speed,
                    "speed_error": (speed - target_speed).abs(),
                    "yaw": yaw,
                    "lateral": robot.data.root_lin_vel_b[:, 1].abs(),
                    "tilt": tilt,
                    "slip": slip,
                    "impact": impact,
                    "flight": (~contacts.any(dim=1)).float(),
                    "double_support": contacts.all(dim=1).float(),
                    "single_support": (contacts.sum(dim=1) == 1).float(),
                    "velocity_saturation": (velocity_ratio >= 0.95).any(dim=1).float(),
                    "torque_saturation": (torque_ratio >= 0.95).any(dim=1).float(),
                    "offset_abs": offset_tensor.abs(),
                }
                for key, value in values.items():
                    sums[key][quality] += value[quality]
                counts[quality] += 1
                heading[step, quality] = heading_error.abs()[quality]
                yaw_abs[step, quality] = yaw.abs()[quality]
                action_rate[step, quality] = rate[quality]
                velocity_dwell = torch.where(
                    (velocity_ratio >= 0.95) & quality.unsqueeze(1),
                    velocity_dwell + dt,
                    torch.zeros_like(velocity_dwell),
                )
                torque_dwell = torch.where(
                    (torque_ratio >= 0.95) & quality.unsqueeze(1),
                    torque_dwell + dt,
                    torch.zeros_like(torque_dwell),
                )
                long_saturation |= (velocity_dwell >= 0.05).any(dim=1)
                long_saturation |= (torque_dwell >= 0.20).any(dim=1)
                if args.mode == "transition":
                    for env_id in torch.where(quality)[0].tolist():
                        segment_values[env_id][segment]["speed_error"].append(float(values["speed_error"][env_id]))
                        segment_values[env_id][segment]["yaw"].append(float(yaw[env_id]))
                    if time_s >= duration_s - HOLD_S[-1]:
                        final_window = quality
                        final_stand_good[final_window] += (
                            (speed[final_window].abs() <= 0.08)
                            & contacts[final_window].all(dim=1)
                        ).float()
                        final_stand_count[final_window] += 1
            if args.mode == "transition":
                for env_id, state in enumerate(states):
                    if state != previous_states[env_id]:
                        if state == "ACTIVATING":
                            activation_jumps[env_id].append(float(rate[env_id]))
                        if state == "DEACTIVATING":
                            deactivation_jumps[env_id].append(float(rate[env_id]))
                    previous_states[env_id] = state

        rows = []
        for env_id in range(num_envs):
            denominator = max(float(counts[env_id]), 1.0)
            valid_heading = heading[:, env_id][torch.isfinite(heading[:, env_id])]
            valid_yaw = yaw_abs[:, env_id][torch.isfinite(yaw_abs[:, env_id])]
            valid_rate = action_rate[:, env_id][torch.isfinite(action_rate[:, env_id])]
            row = {
                "mode": args.mode,
                "controller": args.controller,
                "episode": env_id % episodes,
                "paired_slot": env_id,
                "target_speed": float(target_speed[env_id]) if args.mode == "steady" else -1.0,
                "fall": bool(fell[env_id]),
                "actual_speed": float(sums["speed"][env_id] / denominator),
                "speed_mae": float(sums["speed_error"][env_id] / denominator),
                "yaw_rate_mean": float(sums["yaw"][env_id] / denominator),
                "yaw_rate_p95": float(torch.quantile(valid_yaw, 0.95)) if valid_yaw.numel() else 0.0,
                "heading_p50": float(torch.quantile(valid_heading, 0.50)) if valid_heading.numel() else 0.0,
                "heading_p90": float(torch.quantile(valid_heading, 0.90)) if valid_heading.numel() else 0.0,
                "heading_p95": float(torch.quantile(valid_heading, 0.95)) if valid_heading.numel() else 0.0,
                "heading_p99": float(torch.quantile(valid_heading, 0.99)) if valid_heading.numel() else 0.0,
                "heading_drift_slope": float(sums["yaw"][env_id] / denominator),
                "lateral_velocity": float(sums["lateral"][env_id] / denominator),
                "gravity_tilt": float(sums["tilt"][env_id] / denominator),
                "slip": float(sums["slip"][env_id] / denominator),
                "impact": float(sums["impact"][env_id] / denominator),
                "flight_fraction": float(sums["flight"][env_id] / denominator),
                "double_support_fraction": float(sums["double_support"][env_id] / denominator),
                "single_support_fraction": float(sums["single_support"][env_id] / denominator),
                "joint_velocity_saturation": float(sums["velocity_saturation"][env_id] / denominator),
                "joint_torque_saturation": float(sums["torque_saturation"][env_id] / denominator),
                "long_dwell_saturation": bool(long_saturation[env_id]),
                "action_rate_p95": float(torch.quantile(valid_rate, 0.95)) if valid_rate.numel() else 0.0,
                "activation_action_jump": max(activation_jumps[env_id], default=0.0),
                "deactivation_action_jump": max(deactivation_jumps[env_id], default=0.0),
                "offset_abs_mean": float(sums["offset_abs"][env_id] / denominator),
                "sequence_completion": bool(args.mode == "transition" and alive[env_id]),
                "final_speed": float(robot.data.root_lin_vel_b[env_id, 0]),
                "final_stand_hold": bool(
                    args.mode == "transition" and not fell[env_id]
                    and float(final_stand_good[env_id] / final_stand_count[env_id].clamp_min(1.0)) >= 0.95
                ),
                "final_stand_good_fraction": float(
                    final_stand_good[env_id] / final_stand_count[env_id].clamp_min(1.0)
                ) if args.mode == "transition" else 0.0,
                "segment_metrics": json.dumps(segment_values[env_id], separators=(",", ":"))
                if args.mode == "transition" else "",
            }
            rows.append(row)
        raw_name = f"_{args.mode}_{args.controller}_rows.csv"
        with (OUT / raw_name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (OUT / f"_{args.mode}_{args.controller}_trace.json").write_text(json.dumps({
            "mode": args.mode,
            "controller": args.controller,
            "seed": args.seed,
            "episodes": episodes,
            "duration_s": duration_s,
            "checkpoint_sha256": EXPECTED_SHA,
            "stand_action_trace_sha256": stand_action_hash.hexdigest() if args.mode == "steady" else None,
            "stand_policy_command_trace_sha256": stand_command_hash.hexdigest() if args.mode == "steady" else None,
            "stand_offset_abs_max": offset_max_stand if args.mode == "steady" else None,
            "ppo_updates": 0,
            "policy_gradients": 0,
        }, indent=2) + "\n", encoding="utf-8")
        wrapped.close()


if __name__ == "__main__":
    main()
