"""Single-expert GUI playback for the immutable exp_005 STAND/WALK actor."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("STAND", "WALK"), required=True)
parser.add_argument("--speed", type=float, default=1.5)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=20260723)
parser.add_argument("--duration-s", type=float, default=30.0)
parser.add_argument("--validate-only", action="store_true", help="Resolve config/checkpoint without launching Isaac Sim.")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    speed = 0.0 if args.mode == "STAND" else float(args.speed)
    if not 0.0 <= speed <= 2.2:
        raise ValueError("Stage 2 WALK expert accepts showcase speed in [0.0, 2.2] m/s")
    task = "Isaac-Velocity-Flat-G1-Run-Stage2-v0"
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 20.0
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (7.5, -8.5, 5.0)
    env_cfg.viewer.lookat = (2.5, 0.0, 0.7)
    if args.device is not None:
        env_cfg.sim.device = args.device
    print(f"actor=exp005_stage2_legacy_mlp_123x37")
    print(f"checkpoint={checkpoint}")
    print(f"command={args.mode} target_speed_mps={speed:.3f}")
    print("environments=1 camera=world_orientation_fixed transition_switching=false")
    if args.validate_only:
        print("preflight=PASS simulation_started=false")
        return

    with launch_simulation(env_cfg, args):
        raw = gym.make(task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped
        agent_cfg.device = env.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
        policy = runner.get_inference_policy(device=env.device)
        term = env.command_manager.get_term("base_velocity")
        robot = env.scene["robot"]
        wrapped.reset()
        dt = float(env.step_dt)
        max_steps = max(1, round(args.duration_s / dt))
        for step in range(max_steps):
            term.vel_command_b[:, :] = 0.0
            term.vel_command_b[:, 0] = speed
            observations = wrapped.get_observations()
            with torch.inference_mode():
                actions = policy(observations)
                wrapped.step(actions)
            if step % max(1, round(1.0 / dt)) == 0:
                heading = float(robot.data.heading_w.torch[0].item())
                actual_speed = float(robot.data.root_lin_vel_b.torch[0, 0].item())
                print(json.dumps({"time_s": round(step * dt, 2), "mode": args.mode, "target_speed_mps": speed, "actual_speed_mps": actual_speed, "actual_heading_rad": heading}))
        wrapped.close()


if __name__ == "__main__":
    main()
