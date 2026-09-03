from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env import OpenDuckCalibratedWalkEnv


HERE = Path(__file__).resolve().parent


def rollout(
    policy=None,
    steps=600,
    command_velocity=0.10,
    reference_residual=False,
):
    env = OpenDuckCalibratedWalkEnv(
        seed=29,
        episode_steps=steps,
        command_velocity=command_velocity,
        reference_residual=reference_residual,
    )
    observation, _ = env.reset()
    start_x = float(env.data.qpos[0])
    min_height = float(env.data.qpos[2])
    min_upright = 1.0
    total_reward = 0.0
    completed_steps = 0
    terminated = False
    reward_term_sums = {}

    for _ in range(steps):
        if policy is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        for name, value in info["reward_terms"].items():
            reward_term_sums[name] = reward_term_sums.get(name, 0.0) + value
        total_reward += reward
        completed_steps += 1
        min_height = min(min_height, float(env.data.qpos[2]))
        min_upright = min(min_upright, float(env._base_rotation()[2, 2]))
        if terminated or truncated:
            break

    result = {
        "steps": completed_steps,
        "terminated": terminated,
        "distance_x_m": float(env.data.qpos[0] - start_x),
        "min_base_height_m": min_height,
        "min_upright_cosine": min_upright,
        "total_reward": total_reward,
        "mean_reward_terms": {
            name: value / completed_steps
            for name, value in reward_term_sums.items()
        },
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--command-velocity", type=float, default=0.10)
    parser.add_argument("--reference-residual", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        policy = PPO.load(args.checkpoint, device="cpu")
        result = rollout(
            policy=policy,
            command_velocity=args.command_velocity,
            reference_residual=args.reference_residual,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    checkpoint = HERE / "artifacts" / "ppo_calibrated_walk_smoke.zip"
    smoke_policy = PPO.load(checkpoint)
    trained_checkpoint = HERE / "artifacts" / "ppo_calibrated_walk.zip"
    trained_policy = PPO.load(trained_checkpoint)
    stand_checkpoint = HERE / "artifacts" / "ppo_stage1_stand.zip"
    stand_policy = PPO.load(stand_checkpoint)
    results = {
        "zero_action_safe_init": rollout(policy=None),
        "smoke_policy": rollout(policy=smoke_policy),
        "trained_100k_policy": rollout(policy=trained_policy),
        "stage1_stand_policy": rollout(
            policy=stand_policy, command_velocity=0.0
        ),
        "deployment_allowed": False,
        "note": "Deployment remains disabled until the trained policy passes.",
    }
    trained_result = results["trained_100k_policy"]
    results["deployment_allowed"] = bool(
        not trained_result["terminated"]
        and trained_result["steps"] == 600
        and trained_result["distance_x_m"] >= 0.6
        and trained_result["min_upright_cosine"] >= 0.8
    )
    output_path = HERE / "artifacts" / "validation.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
