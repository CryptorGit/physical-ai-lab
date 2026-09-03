"""Collect matched deterministic WALK/RUN-at-1.2 positive-control trajectories."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2j_low_speed_action_manifold_reachability"
RAW = OUT / "raw"
CHECKPOINTS = {
    "W0": REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
    "R0": REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt",
    "R1": REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2i_reverse_continuation_phase_r1/checkpoints/model_1.pt",
}
EXPECTED = {
    "W0": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "R0": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
    "R1": "707bd50a8a168f2b247965ff6977e41da1d560094a1d5328737eaa76963f3ecd",
}
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(exist_ok=True)
    for name, path in CHECKPOINTS.items():
        if sha(path) != EXPECTED[name]:
            raise RuntimeError(f"STAGE2J_CHECKPOINT_PROVENANCE_FAIL:{name}")
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 300
    cfg.seed = 20268021
    cfg.episode_length_s = 20.0
    agent_cfg.seed = 20268021
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        actors, critics = {}, {}
        for name, path in CHECKPOINTS.items():
            payload = torch.load(path, map_location=runner.device, weights_only=False)
            actor = copy.deepcopy(runner.alg.actor)
            critic = copy.deepcopy(runner.alg.critic)
            actor.load_state_dict(payload["actor_state_dict"], strict=True)
            critic.load_state_dict(payload["critic_state_dict"], strict=True)
            actor.eval()
            critic.eval()
            actors[name], critics[name] = actor, critic
        group_names = ("W0", "R0", "R1")
        group_index = torch.arange(300, device=runner.device) // 100
        env = wrapped.unwrapped
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        command.external_override[:, 0] = 1.2
        command.external_override[:, 1:] = 0.0
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        sensor_feet = [index for index, value in enumerate(sensor.body_names) if "ankle_roll" in value]
        robot_feet = [
            next(index for index, value in enumerate(robot.body_names) if value == sensor.body_names[sensor_id])
            for sensor_id in sensor_feet
        ]
        reward_names = list(env.reward_manager.active_terms)
        obs, _ = wrapped.reset()
        obs = obs.to(runner.device)
        reference_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        fields = {key: [] for key in (
            "observation", "action", "value", "reward", "reward_components", "contact",
            "base_height", "base_pitch", "vertical_velocity", "actual_speed", "heading",
            "foot_speed", "joint_position", "joint_velocity",
        )}
        falls = torch.zeros(300, dtype=torch.bool, device=runner.device)
        flight_streak = torch.zeros(300, dtype=torch.long, device=runner.device)
        flight_events = torch.zeros_like(flight_streak)
        max_flight = torch.zeros_like(flight_streak)
        safe_flights = torch.zeros_like(flight_streak)
        alternating = torch.zeros_like(flight_streak)
        last_landing = torch.full_like(flight_streak, -1)
        slip_streak = torch.zeros_like(flight_streak)
        saturation_streak = torch.zeros_like(flight_streak)
        dangerous_slip = torch.zeros(300, dtype=torch.bool, device=runner.device)
        long_dwell_saturation = torch.zeros_like(dangerous_slip)
        impact_failure = torch.zeros_like(dangerous_slip)
        joint_velocity_limit = robot.data.joint_vel_limits.abs().clamp_min(1.0e-6)
        for step in range(500):
            if step == 0:
                obs = wrapped.get_observations().to(runner.device)
            action = torch.zeros((300, 37), device=runner.device)
            value = torch.zeros(300, device=runner.device)
            with torch.inference_mode():
                for index, name in enumerate(group_names):
                    mask = group_index == index
                    action[mask] = actors[name](obs[mask], stochastic_output=False)
                    value[mask] = critics[name](obs[mask]).squeeze(-1)
            obs_before = obs.clone()
            obs, reward, dones, extras = wrapped.step(action)
            obs = obs.to(runner.device)
            timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
            falls |= dones.bool() & ~timeouts
            forces = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1)
            contacts = forces > 5.0
            flight = contacts.sum(-1) == 0
            previous = flight_streak.clone()
            flight_events += (flight & (flight_streak == 0)).long()
            flight_streak = torch.where(flight, flight_streak + 1, torch.zeros_like(flight_streak))
            max_flight = torch.maximum(max_flight, flight_streak)
            landing = ~flight & (previous > 0)
            single = landing & (contacts.sum(-1) == 1)
            foot = contacts.long().argmax(-1)
            safe = single & (previous >= 2) & (previous <= 8)
            alt = safe & (last_landing >= 0) & (foot != last_landing)
            safe_flights += safe.long()
            alternating += alt.long()
            last_landing[single] = foot[single]
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
            slip_now = (contacts & (foot_speed > 0.55)).any(-1)
            slip_streak = torch.where(slip_now, slip_streak + 1, torch.zeros_like(slip_streak))
            dangerous_slip |= slip_streak >= 5
            joint_velocity_fraction = (robot.data.joint_vel.abs() / joint_velocity_limit).amax(-1)
            saturation_streak = torch.where(
                joint_velocity_fraction >= 0.95, saturation_streak + 1, torch.zeros_like(saturation_streak)
            )
            long_dwell_saturation |= saturation_streak >= 5
            impact_failure |= forces.amax(-1) > 3500.0
            policy_observation = obs_before["policy"]
            projected_gravity = policy_observation[:, 6:9]
            pitch = torch.atan2(
                -projected_gravity[:, 0],
                torch.sqrt(projected_gravity[:, 1].square() + projected_gravity[:, 2].square()).clamp_min(1.0e-8),
            )
            values = {
                "observation": policy_observation, "action": action, "value": value, "reward": reward,
                "reward_components": env.reward_manager._step_reward.clone(),
                "contact": contacts, "base_height": robot.data.root_pos_w[:, 2],
                "base_pitch": pitch, "vertical_velocity": robot.data.root_lin_vel_b[:, 2],
                "actual_speed": robot.data.root_lin_vel_b[:, 0],
                "heading": wrapped_heading_error(reference_yaw, yaw_from_quat_wxyz(robot.data.root_quat_w)),
                "foot_speed": foot_speed, "joint_position": robot.data.joint_pos,
                "joint_velocity": robot.data.joint_vel,
            }
            for key, tensor in values.items():
                fields[key].append(tensor.detach().cpu())
        tensors = {key: torch.stack(value) for key, value in fields.items()}
        tensors["group_index"] = group_index.cpu()
        tensors["reward_names"] = reward_names
        tensors["checkpoint_names"] = group_names
        tensors["checkpoint_paths"] = {name: str(path.relative_to(REPO)) for name, path in CHECKPOINTS.items()}
        torch.save(tensors, RAW / "positive_control_trajectories.pt")
        records = []
        dt = float(env.step_dt)
        for env_id in range(300):
            name = group_names[int(group_index[env_id])]
            contacts = tensors["contact"][:, env_id]
            flight_fraction = float((contacts.sum(-1) == 0).float().mean())
            periodic = int(flight_events[env_id]) >= 4 and int(safe_flights[env_id]) >= 3 and int(alternating[env_id]) >= 3
            gait = "FALL" if falls[env_id] else "PERIODIC_RUNNING" if periodic else "ISOLATED_FLIGHT" if flight_events[env_id] else "WALK_LIKE"
            records.append({
                "checkpoint": name, "episode": env_id % 100, "gait": gait,
                "success": gait == "WALK_LIKE", "fall": bool(falls[env_id]),
                "speed_mae": float((tensors["actual_speed"][:, env_id] - 1.2).abs().mean()),
                "flight_fraction": flight_fraction, "max_flight_duration_s": int(max_flight[env_id]) * dt,
                "flight_event_count": int(flight_events[env_id]),
                "safe_flight_count": int(safe_flights[env_id]),
                "alternating_landing_count": int(alternating[env_id]),
                "single_support_fraction": float((contacts.sum(-1) == 1).float().mean()),
                "double_support_fraction": float((contacts.sum(-1) == 2).float().mean()),
                "stride_frequency_hz": float(flight_events[env_id]) / 10.0,
                "step_length_m": (
                    float(tensors["actual_speed"][:, env_id].mean()) / max(float(flight_events[env_id]) / 10.0, 1.0e-8)
                    if int(flight_events[env_id]) > 0 else 0.0
                ),
                "base_height_mean": float(tensors["base_height"][:, env_id].mean()),
                "base_pitch_abs_mean": float(tensors["base_pitch"][:, env_id].abs().mean()),
                "vertical_velocity_abs_mean": float(tensors["vertical_velocity"][:, env_id].abs().mean()),
                "action_norm_mean": float(torch.linalg.vector_norm(tensors["action"][:, env_id], dim=-1).mean()),
                "action_rate_mean": float(torch.linalg.vector_norm(torch.diff(tensors["action"][:, env_id], dim=0), dim=-1).mean()),
                "heading_p95": float(torch.quantile(tensors["heading"][:, env_id].abs(), .95)),
                "dangerous_slip": bool(dangerous_slip[env_id]),
                "impact_failure": bool(impact_failure[env_id]),
                "long_dwell_saturation": bool(long_dwell_saturation[env_id]),
                "episode_return": float(tensors["reward"][:, env_id].sum()),
            })
        with (OUT / "walk_run_positive_control_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        summary = {}
        for index, name in enumerate(group_names):
            rows = records[index * 100:(index + 1) * 100]
            summary[name] = {
                "episodes": 100, "walk_success_rate": sum(row["success"] for row in rows) / 100,
                "fall_rate": sum(row["fall"] for row in rows) / 100,
                "periodic_running_rate": sum(row["gait"] == "PERIODIC_RUNNING" for row in rows) / 100,
                **{
                    key: sum(row[key] for row in rows) / 100 for key in (
                        "speed_mae", "flight_fraction", "max_flight_duration_s",
                        "flight_event_count", "safe_flight_count", "alternating_landing_count",
                        "single_support_fraction", "double_support_fraction", "stride_frequency_hz",
                        "step_length_m",
                        "base_height_mean", "base_pitch_abs_mean", "vertical_velocity_abs_mean",
                        "action_norm_mean", "action_rate_mean", "heading_p95", "episode_return",
                    )
                },
                "dangerous_slip_rate": sum(row["dangerous_slip"] for row in rows) / 100,
                "impact_failure_rate": sum(row["impact_failure"] for row in rows) / 100,
                "long_dwell_saturation_rate": sum(row["long_dwell_saturation"] for row in rows) / 100,
            }
        dump("walk_run_positive_control_comparison.json", summary)
        wrapped.close()


if __name__ == "__main__":
    main()
