"""Collect one fresh-process, frozen-checkpoint Phase-A on-policy corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2g_event_stratified_on_policy_preflight"
RAW = OUT / "raw"
CHECKPOINT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight/checkpoints/model_50.pt"
EXPECTED_SHA = "4edbb595e28e24dc09cf39e8245c7be1b1bebf792798a73af2e562075d0fe952"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--batch-index", type=int, required=True)
parser.add_argument("--seed-root", type=int, default=20268021)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    digest = hashlib.sha256()
    if hasattr(value, "keys"):
        for key in sorted(value.keys()):
            digest.update(key.encode())
            digest.update(value[key].contiguous().numpy().tobytes())
    else:
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def discounted_gae(reward, value, done, next_value, gamma=.99, lam=.95):
    steps, envs = reward.shape
    advantage = torch.zeros_like(reward)
    running = torch.zeros(envs)
    future_value = next_value.cpu()
    for step in range(steps - 1, -1, -1):
        not_done = (~done[step]).float()
        delta = reward[step] + gamma * future_value * not_done - value[step]
        running = delta + gamma * lam * not_done * running
        advantage[step] = running
        future_value = value[step]
    return advantage, advantage + value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    if sha(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError("PRIMARY_CHECKPOINT_HASH_MISMATCH")
    seed = args.seed_root + args.batch_index
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp012-G1-PhaseA-RunAcquisition-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 1024
    cfg.seed = seed
    cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    agent_cfg.seed = seed
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    import importlib.metadata
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-PhaseA-RunAcquisition-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(CHECKPOINT), load_cfg={
            "actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False,
        }, strict=True, map_location=runner.device)
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        command_term = env.command_manager.get_term("base_velocity")
        reward_term = env.reward_manager.get_term_cfg("safe_periodic_flight").func
        reward_names = list(env.reward_manager.active_terms)
        sensor_feet = list(reward_term.foot_ids)
        robot_feet = [
            next(i for i, body in enumerate(robot.body_names) if body == sensor.body_names[int(sensor_id)])
            for sensor_id in sensor_feet
        ]
        obs, _ = wrapped.reset()
        obs = obs.to(runner.device)
        reference_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        episode_id = torch.arange(1024, device=runner.device, dtype=torch.long)
        next_episode_id = 1024
        previous_landing = torch.full((1024,), -1, device=runner.device, dtype=torch.long)
        was_flight = torch.zeros(1024, device=runner.device, dtype=torch.bool)
        flight_duration = torch.zeros(1024, device=runner.device)
        slip_streak = torch.zeros(1024, device=runner.device, dtype=torch.long)
        saturation_streak = torch.zeros(1024, device=runner.device, dtype=torch.long)
        storage = defaultdict(list)
        event_rows = []
        dt = float(env.step_dt)
        steps = round(20.0 / dt)
        with torch.inference_mode():
            for step in range(steps):
                policy_obs = obs["policy"] if hasattr(obs, "keys") else obs
                action = runner.alg.actor(obs, stochastic_output=True)
                mean_action, std_action = (value.clone() for value in runner.alg.actor.output_distribution_params)
                log_probability = runner.alg.actor.get_output_log_prob(action)
                value = runner.alg.critic(obs).squeeze(-1)
                command = command_term.vel_command_b.clone()
                current_episode = episode_id.clone()
                current_time = env.episode_length_buf.float() * dt
                obs, reward, dones, extras = wrapped.step(action)
                obs = obs.to(runner.device)
                reward = reward.to(runner.device)
                dones = dones.to(runner.device).bool()
                timeouts = extras.get("time_outs", torch.zeros_like(dones)).to(runner.device).bool()
                forces = sensor.data.net_forces_w_history[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1)
                contacts = forces > 1.0
                flight = contacts.sum(-1) == 0
                takeoff = flight & ~was_flight
                flight_duration = torch.where(flight, flight_duration + dt, flight_duration)
                landing = was_flight & ~flight
                single_landing = landing & (contacts.sum(-1) == 1)
                landing_side = contacts.long().argmax(-1)
                previous_side = previous_landing.clone()
                previous_landing[single_landing] = landing_side[single_landing]
                raw_run = reward_term.last_raw_reward.detach().clone()
                completion = raw_run >= 1.0
                precursor = (raw_run > 0) & (raw_run < .10)
                safe_flight = (raw_run >= .10) & (raw_run < 1.0)
                tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
                heading = wrapped_heading_error(reference_yaw, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs()
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
                slipping_now = ((foot_speed > .55) & contacts).any(-1)
                slip_streak = torch.where(slipping_now, slip_streak + 1, torch.zeros_like(slip_streak))
                slip = slip_streak >= 5
                impact = forces.amax(-1) > 3500.0
                limits = robot.data.joint_vel_limits
                if limits.ndim == 3:
                    limits = limits[..., 1].abs()
                saturated_now = (robot.data.joint_vel.abs() / torch.clamp(limits, min=1e-6) > .95).any(-1)
                saturation_streak = torch.where(
                    saturated_now, saturation_streak + 1, torch.zeros_like(saturation_streak)
                )
                saturation = saturation_streak >= 5
                fall = dones & ~timeouts
                gait_state = torch.where(
                    fall, torch.full_like(episode_id, 4),
                    torch.where(flight, torch.full_like(episode_id, 2), torch.ones_like(episode_id)),
                )
                components = env.reward_manager._step_reward.detach().clone()
                for key, value in (
                    ("observation", policy_obs), ("action", action), ("mean", mean_action),
                    ("std", std_action), ("old_logp", log_probability), ("value", value),
                    ("reward", reward), ("reward_components", components),
                    ("environment_id", torch.arange(1024, device=runner.device)),
                    ("episode_id", current_episode), ("episode_time", current_time),
                    ("rollout_timestep", torch.full((1024,), step, device=runner.device)),
                    ("command", command), ("actual_speed", robot.data.root_lin_vel_b[:, 0]),
                    ("gait_state", gait_state), ("contact_state", contacts),
                    ("flight_duration", flight_duration), ("takeoff", takeoff),
                    ("precursor", precursor), ("safe_flight", safe_flight),
                    ("completion", completion), ("landing", landing),
                    ("landing_side", torch.where(single_landing, landing_side, torch.full_like(landing_side, -1))),
                    ("previous_landing_side", previous_side), ("fall", fall), ("tilt", tilt),
                    ("heading", heading), ("slip", slip), ("impact", impact),
                    ("saturation", saturation), ("done", dones),
                ):
                    storage[key].append(value.detach().cpu())
                for env_id in torch.where(completion)[0].tolist():
                    event_rows.append({
                        "batch_index": args.batch_index, "seed": seed, "environment_id": env_id,
                        "episode_id": int(current_episode[env_id]), "rollout_timestep": step,
                        "episode_time_s": float(current_time[env_id]),
                        "target_speed": float(command[env_id, 0]),
                        "actual_speed": float(robot.data.root_lin_vel_b[env_id, 0]),
                        "landing_side": int(landing_side[env_id]),
                        "previous_landing_side": int(previous_side[env_id]),
                    })
                flight_duration[landing] = 0
                was_flight.copy_(flight)
                if dones.any():
                    reset_ids = torch.where(dones)[0]
                    count = len(reset_ids)
                    episode_id[reset_ids] = torch.arange(
                        next_episode_id, next_episode_id + count, device=runner.device
                    )
                    next_episode_id += count
                    previous_landing[reset_ids] = -1
                    was_flight[reset_ids] = False
                    flight_duration[reset_ids] = 0
                    slip_streak[reset_ids] = 0
                    saturation_streak[reset_ids] = 0
                    reference_yaw[reset_ids] = yaw_from_quat_wxyz(robot.data.root_quat_w)[reset_ids]
                if step % 100 == 0:
                    print(
                        f"[Stage2G collect] batch={args.batch_index} step={step}/{steps} "
                        f"completion={len(event_rows)}",
                        flush=True,
                    )
            next_value = runner.alg.critic(obs).squeeze(-1)
        corpus = {key: torch.stack(values) for key, values in storage.items()}
        advantage, returns = discounted_gae(
            corpus["reward"], corpus["value"], corpus["done"], next_value
        )
        corpus["raw_advantage"] = advantage
        corpus["normalized_advantage"] = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        corpus["return"] = returns
        corpus["reward_names"] = reward_names
        corpus["joint_names"] = list(robot.joint_names)
        corpus["checkpoint_sha256"] = EXPECTED_SHA
        corpus["batch_index"] = args.batch_index
        corpus["seed"] = seed
        raw_path = RAW / f"on_policy_batch_{args.batch_index}.pt"
        torch.save(corpus, raw_path)
        hashes = {
            key: tensor_hash(value) for key, value in corpus.items()
            if torch.is_tensor(value) or hasattr(value, "keys")
        }
        unique_episodes = len({row["episode_id"] for row in event_rows})
        summary = {
            "batch_index": args.batch_index, "seed": seed, "checkpoint": str(CHECKPOINT.relative_to(REPO)),
            "checkpoint_sha256": EXPECTED_SHA, "raw_path": str(raw_path.relative_to(REPO)),
            "raw_sha256": sha(raw_path), "samples": steps * 1024, "steps": steps,
            "completion_events": len(event_rows), "completion_episodes": unique_episodes,
            "on_policy": True, "action_std_multiplier": 1.0, "yaw_nonzero_samples": int(
                torch.count_nonzero(corpus["command"][..., 2])
            ), "controller": "OFF", "event_rows": event_rows, "tensor_hashes": hashes,
        }
        (OUT / f"batch_{args.batch_index}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raw.close()


if __name__ == "__main__":
    main()
