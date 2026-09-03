from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from env import OpenDuckCalibratedWalkEnv


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--command-velocity", type=float, default=0.10)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--output-name", default="ppo_calibrated_walk")
    args = parser.parse_args()

    env = OpenDuckCalibratedWalkEnv(
        seed=args.seed, command_velocity=args.command_velocity
    )
    check_env(env, warn=True)
    env.close()

    steps = 2_048 if args.smoke else args.steps
    monitored_env = Monitor(
        OpenDuckCalibratedWalkEnv(
            seed=args.seed, command_velocity=args.command_velocity
        )
    )
    if args.pretrained:
        model = PPO.load(
            args.pretrained,
            env=monitored_env,
            learning_rate=1e-4,
            ent_coef=0.0,
            device="auto",
        )
    else:
        model = PPO(
            "MlpPolicy",
            monitored_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=128,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs={"log_std_init": -1.5},
            verbose=1,
            seed=args.seed,
            device="auto",
        )
    model.learn(total_timesteps=steps, progress_bar=False)

    output_dir = HERE / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        "ppo_calibrated_walk_smoke" if args.smoke else args.output_name
    )
    model.save(output_path)
    print(f"saved={output_path}.zip")


if __name__ == "__main__":
    main()
