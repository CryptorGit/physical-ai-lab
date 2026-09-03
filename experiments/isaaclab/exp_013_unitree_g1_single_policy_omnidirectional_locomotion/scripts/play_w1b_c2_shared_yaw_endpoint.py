"""W1BC2SharedYawEndpointEvaluator playback; evaluator never changes actions."""
from __future__ import annotations

import argparse
import math
import sys
from collections import deque
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

parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=float, default=0.0)
parser.add_argument("--speed", type=float, default=0.3)
parser.add_argument("--yaw", type=float, default=0.3, help="physical target yaw rate")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

with launch_simulation(args) as simulation_app:
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.observations.policy.enable_corruption = False
    env = RslRlVecEnvWrapper(
        gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
        clip_actions=acfg.clip_actions,
    )
    actor = FrozenGaitActor(CHECKPOINT).to(env.unwrapped.device).eval()
    command = env.unwrapped.command_manager.get_term("base_velocity")
    command.external_override_enabled = True
    obs, _ = env.reset()
    obs = obs["policy"].to(env.unwrapped.device)
    direction = math.radians(args.direction)
    actor_yaw = float(calibrate_yaw(args.yaw))
    yaw_window = deque(maxlen=300)
    acquired_at = None
    step = 0
    while simulation_app.is_running():
        command.external_override[:, 0] = args.speed * math.cos(direction)
        command.external_override[:, 1] = args.speed * math.sin(direction)
        command.external_override[:, 2] = actor_yaw
        if step == 0:
            command._update_command()
            obs = env.get_observations()["policy"].to(env.unwrapped.device)
        with torch.inference_mode():
            action = actor(obs, torch.zeros(1, device=env.unwrapped.device))
        obs, _, done, _ = env.step(action)
        obs = obs["policy"].to(env.unwrapped.device)
        robot = env.unwrapped.scene["robot"]
        actual_yaw = float(robot.data.root_ang_vel_b[0, 2])
        actual_v = robot.data.root_lin_vel_b[0, :2]
        yaw_window.append(actual_yaw)
        if acquired_at is None and actual_yaw * args.yaw > 0:
            acquired_at = step * env.unwrapped.step_dt
        mean_yaw = sum(yaw_window) / len(yaw_window)
        yaw_mae = sum(abs(value - args.yaw) for value in yaw_window) / len(yaw_window)
        endpoint_active = len(yaw_window) == yaw_window.maxlen
        yaw_limit = .15 if args.speed <= .05 else .20
        endpoint_pass = endpoint_active and mean_yaw * args.yaw > 0 and yaw_mae <= yaw_limit
        vector_mae = float(torch.linalg.vector_norm(
            actual_v - torch.tensor(
                [args.speed * math.cos(direction), args.speed * math.sin(direction)],
                device=actual_v.device,
            )
        ))
        print(
            "\rMODE W1BC2SharedYawEndpointEvaluator | "
            f"TARGET VX/VY {args.speed*math.cos(direction):+.2f}/{args.speed*math.sin(direction):+.2f} | "
            f"ACTUAL VX/VY {actual_v[0]:+.2f}/{actual_v[1]:+.2f} | "
            f"PHYSICAL TARGET YAW {args.yaw:+.2f} | ACTOR YAW INPUT {actor_yaw:+.2f} | "
            f"ACTUAL YAW {actual_yaw:+.2f} | ENDPOINT WINDOW {'ACTIVE' if endpoint_active else 'INACTIVE'} | "
            f"ENDPOINT MEAN/MAE {mean_yaw:+.2f}/{yaw_mae:.2f} | ENDPOINT PASS {endpoint_pass} | "
            f"ACQUISITION {acquired_at if acquired_at is not None else float('nan'):.2f}s | "
            f"VECTOR MAE {vector_mae:.2f} | FALL {bool(done[0])} | CHECKPOINT SHA {CHECKPOINT.name}",
            end="",
        )
        step += 1
    env.close()
