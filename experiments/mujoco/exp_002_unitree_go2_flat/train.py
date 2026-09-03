"""PPOを使ってUnitree Go2の地球平面歩行を学習する。"""

from __future__ import annotations

from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env import Go2FlatEnv, load_config


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


def main() -> None:
    """PPO学習を実行し、モデルを保存する。"""
    config = load_config()
    training = config["training"]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(Go2FlatEnv())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="go2_flat",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=float(training["learning_rate"]),
        n_steps=int(training["n_steps"]),
        batch_size=int(training["batch_size"]),
        n_epochs=int(training["n_epochs"]),
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        clip_range=float(training["clip_range"]),
        tensorboard_log=str(LOG_DIR),
        verbose=1,
        device=device,
        seed=int(config["experiment"]["seed"]),
    )

    model.learn(
        total_timesteps=int(training["total_timesteps"]),
        callback=checkpoint_callback,
        progress_bar=True,
    )

    output_path = CHECKPOINT_DIR / "go2_flat_final"
    model.save(output_path)

    env.close()

    print(f"Saved model: {output_path}.zip")


if __name__ == "__main__":
    main()