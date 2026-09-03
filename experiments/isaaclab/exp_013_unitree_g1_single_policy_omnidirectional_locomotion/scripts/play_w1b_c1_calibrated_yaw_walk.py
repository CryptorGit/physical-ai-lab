"""W1BC1CalibratedYawWalk playback; calibration changes commands, never actions."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
EXP = HERE.parent.parent
CHECKPOINT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
sys.path.insert(0, str(ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(ROOT / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from g1_omnidirectional.yaw_calibration import calibrate_yaw  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=float, default=0.0)
parser.add_argument("--speed", type=float, default=0.3)
parser.add_argument("--yaw", type=float, default=0.3, help="physical target yaw rate")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

with launch_simulation(args) as simulation_app:
    cfg = resolve_task_config("Isaac-Velocity-Flat-Unitree-G1-DirectionalBaseline-v0", args)
    cfg.scene.num_envs = 1
    cfg.observations.policy.enable_corruption = False
    env = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-Unitree-G1-DirectionalBaseline-v0", cfg=cfg))
    actor = FrozenGaitActor(CHECKPOINT, device=env.device)
    obs, _ = env.get_observations()
    target = torch.tensor(float(args.yaw), device=env.device)
    actor_yaw = calibrate_yaw(target)
    direction = math.radians(args.direction)
    while simulation_app.is_running():
        env.unwrapped.command_manager.get_term("base_velocity").vel_command_b[:, 0] = args.speed * math.cos(direction)
        env.unwrapped.command_manager.get_term("base_velocity").vel_command_b[:, 1] = args.speed * math.sin(direction)
        env.unwrapped.command_manager.get_term("base_velocity").vel_command_b[:, 2] = actor_yaw
        with torch.inference_mode():
            action = actor(obs, gait=0.0)
        obs, _, _, _ = env.step(action)
        actual = env.unwrapped.scene["robot"].data.root_ang_vel_b[0, 2].item()
        linear = env.unwrapped.scene["robot"].data.root_lin_vel_b[0, :2]
        print(
            "\rMODE W1BC1CalibratedYawWalk | "
            f"TARGET VX/VY {args.speed*math.cos(direction):+.2f}/{args.speed*math.sin(direction):+.2f} | "
            f"TARGET YAW {args.yaw:+.2f} | ACTOR YAW INPUT {actor_yaw.item():+.2f} | "
            f"ACTUAL VX/VY {linear[0].item():+.2f}/{linear[1].item():+.2f} | "
            f"ACTUAL YAW {actual:+.2f} | CALIBRATION POSITIVE x1.50 NEGATIVE x1.00 | "
            f"CHECKPOINT SHA {CHECKPOINT.name}",
            end="",
        )
    env.close()
