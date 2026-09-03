from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import OpenDuckCalibratedWalkEnv


HERE = Path(__file__).resolve().parent


def make_parallel_env(
    n_envs: int,
    command_velocity: float,
    seed: int,
    *,
    reference_residual: bool = False,
):
    env = make_vec_env(
        OpenDuckCalibratedWalkEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={
            "command_velocity": command_velocity,
            "reference_residual": reference_residual,
        },
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    return env


def checkpoint_callback(output_dir: Path, n_envs: int, prefix: str):
    return CheckpointCallback(
        save_freq=max(250_000 // n_envs, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix=prefix,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=24)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--stand-steps", type=int, default=1_000_000)
    parser.add_argument("--walk-steps", type=int, default=5_000_000)
    parser.add_argument("--walk-speed", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="parallel24_gpu")
    parser.add_argument("--stand-checkpoint", type=Path)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training requested but this Python environment has no "
            "CUDA-enabled PyTorch."
        )

    output_dir = HERE / "artifacts" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stand_checkpoint:
        stand_env = None
        model = PPO.load(
            args.stand_checkpoint,
            device=args.device,
        )
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.zero_()
            model.policy.log_std.fill_(-3.0)
        print(f"loaded_stand={args.stand_checkpoint}", flush=True)
    else:
        stand_env = make_parallel_env(
            args.n_envs, command_velocity=0.0, seed=args.seed
        )
        model = PPO(
            "MlpPolicy",
            stand_env,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs={
                "activation_fn": torch.nn.ELU,
                "log_std_init": -1.5,
                "net_arch": {
                    "pi": [2048, 2048, 1024],
                    "vf": [2048, 2048, 1024],
                },
            },
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(output_dir / "tensorboard"),
        )
        print(
            f"phase=stand n_envs={args.n_envs} "
            f"rollout={args.n_envs * args.n_steps} "
            f"batch={args.batch_size} device={model.device}",
            flush=True,
        )
        model.learn(
            total_timesteps=args.stand_steps,
            callback=checkpoint_callback(output_dir, args.n_envs, "stand"),
            tb_log_name="stand",
        )
        stand_path = output_dir / "ppo_stage1_stand"
        model.save(stand_path)
        stand_env.close()

    walk_env = make_parallel_env(
        args.n_envs,
        command_velocity=args.walk_speed,
        seed=args.seed + 10_000,
        reference_residual=True,
    )
    model.set_env(walk_env)
    walk_learning_rate = 5e-5 if args.stand_checkpoint else 1e-4
    model.learning_rate = walk_learning_rate
    model.lr_schedule = lambda _: walk_learning_rate
    if args.stand_checkpoint:
        model.n_epochs = 3
        model.target_kl = 0.02
        model.policy.optimizer = torch.optim.Adam(
            model.policy.parameters(),
            lr=walk_learning_rate,
            eps=1e-5,
        )
    print(
        f"phase=walk n_envs={args.n_envs} "
        f"rollout={args.n_envs * args.n_steps} "
        f"batch={args.batch_size} device={model.device}",
        flush=True,
    )
    model.learn(
        total_timesteps=args.walk_steps,
        reset_num_timesteps=False,
        callback=checkpoint_callback(output_dir, args.n_envs, "walk"),
        tb_log_name="walk",
    )
    walk_path = output_dir / "ppo_stage2_walk"
    model.save(walk_path)
    walk_env.close()
    if not args.stand_checkpoint:
        print(f"saved_stand={stand_path}.zip", flush=True)
    print(f"saved_walk={walk_path}.zip", flush=True)


if __name__ == "__main__":
    # Required by Windows multiprocessing spawn.
    main()
