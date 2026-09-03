"""Inspect and step the official Isaac Lab Unitree G1 flat environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


TASK_ID = "Isaac-Velocity-Flat-G1-v0"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID, help="Registered Isaac Lab task ID.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of vectorized environments.")
parser.add_argument("--max_steps", type=int, default=128, help="Number of environment steps to execute.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    """Create the official task, print its interface, and run zero actions."""
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        spec = gym.spec(args_cli.task)
        env = gym.make(args_cli.task, cfg=env_cfg)

        try:
            unwrapped = env.unwrapped
            robot = unwrapped.scene["robot"]
            observations, reset_info = env.reset(seed=args_cli.seed)

            print("\n=== Unitree G1 flat environment inspection ===")
            print(f"Isaac Lab package version: {importlib.metadata.version('isaaclab')}")
            print(f"Isaac Lab tasks version: {importlib.metadata.version('isaaclab_tasks')}")
            print(f"Task ID: {args_cli.task}")
            print(f"Entry point: {spec.entry_point}")
            print(f"Environment class: {type(unwrapped).__module__}.{type(unwrapped).__name__}")
            print(f"Robot USD: {unwrapped.cfg.scene.robot.spawn.usd_path}")
            print(f"Terrain type: {unwrapped.cfg.scene.terrain.terrain_type}")
            print(f"Joint count: {robot.num_joints}")
            for index, name in enumerate(robot.joint_names):
                print(f"  joint[{index:02d}]: {name}")

            print(f"Observation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")
            for group_name, group_observation in observations.items():
                print(f"Observation[{group_name!r}] shape: {tuple(group_observation.shape)}")
            print(f"Observation terms: {unwrapped.observation_manager.active_terms['policy']}")
            print(f"Action terms: {unwrapped.action_manager.active_terms}")
            print(f"Reward terms: {unwrapped.reward_manager.active_terms}")
            print(f"Termination terms: {unwrapped.termination_manager.active_terms}")
            print(f"Reset info keys: {sorted(reset_info)}")

            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            reward_sum = torch.zeros(args_cli.num_envs, device=unwrapped.device)
            terminated_count = 0
            truncated_count = 0
            episode_log_keys: set[str] = set()

            with torch.inference_mode():
                for _ in range(args_cli.max_steps):
                    _, rewards, terminated, truncated, info = env.step(actions)
                    reward_sum += rewards
                    terminated_count += int(terminated.sum().item())
                    truncated_count += int(truncated.sum().item())
                    if "log" in info:
                        episode_log_keys.update(info["log"].keys())

            print("\n=== Finite zero-action rollout ===")
            print(f"Steps: {args_cli.max_steps}")
            print(f"Mean cumulative reward: {reward_sum.mean().item():.6f}")
            print(f"Terminated episodes: {terminated_count}")
            print(f"Truncated episodes: {truncated_count}")
            print(f"Episode log keys observed: {sorted(episode_log_keys)}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
