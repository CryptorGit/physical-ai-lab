"""Capture the fixed Stage-2 standing state and authoritative Isaac dynamics metadata."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve(); EXP = SCRIPT.parent.parent; REPO = EXP.parents[2]
sys.path[:0] = [str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--settle-s", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=20260722)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser); sys.argv = [sys.argv[0]] + hydra


def values(tensor):
    return tensor.detach().cpu().tolist()


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    env_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1; env_cfg.seed = args.seed
    if args.device is not None: env_cfg.sim.device = args.device
    with launch_simulation(env_cfg, args):
        raw = gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions); env = raw.unwrapped
        agent_cfg.device = env.device; agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False, "optimizer": False,
                                               "iteration": False, "rnd": False})
        policy = runner.get_inference_policy(device=env.device)
        robot = env.scene["robot"]; command = env.command_manager.get_term("base_velocity")
        wrapped.reset()
        action = None
        for _ in range(round(args.settle_s / float(env.step_dt))):
            command.vel_command_b.zero_(); obs = wrapped.get_observations()
            import torch
            with torch.inference_mode(): action = policy(obs); wrapped.step(action)
            command.vel_command_b.zero_()
        joint_ids, joint_names = robot.find_joints(".*")
        foot_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        terrain = env_cfg.scene.terrain.physics_material
        result = {
            "checkpoint": str(checkpoint), "seed": args.seed, "settle_s": args.settle_s,
            "step_dt_s": float(env.step_dt), "joint_names": joint_names, "foot_body_names": foot_names,
            "root_pos_w": values(robot.data.root_pos_w.torch[0]), "root_quat_w": values(robot.data.root_quat_w.torch[0]),
            "root_lin_vel_w": values(robot.data.root_lin_vel_w.torch[0]), "root_ang_vel_w": values(robot.data.root_ang_vel_w.torch[0]),
            "joint_pos": values(robot.data.joint_pos.torch[0, joint_ids]), "joint_vel": values(robot.data.joint_vel.torch[0, joint_ids]),
            "default_joint_pos": values(robot.data.default_joint_pos.torch[0, joint_ids]),
            "hard_joint_pos_limits": values(robot.data.joint_pos_limits.torch[0, joint_ids]),
            "joint_velocity_limits": values(robot.data.joint_vel_limits.torch[0, joint_ids]),
            "joint_effort_limits": values(robot.data.joint_effort_limits.torch[0, joint_ids]),
            "foot_body_pos_w": values(robot.data.body_pos_w.torch[0, foot_ids]),
            "foot_body_quat_w": values(robot.data.body_quat_w.torch[0, foot_ids]),
            "mass_matrix": values(robot.data.mass_matrix.torch[0]),
            "foot_link_jacobian_w": values(robot.data.body_link_jacobian_w.torch[0, foot_ids]),
            "last_actor_action": values(action[0]),
            "terrain_friction": {"static": float(terrain.static_friction), "dynamic": float(terrain.dynamic_friction),
                                 "restitution": float(terrain.restitution), "combine_mode": str(terrain.friction_combine_mode)},
            "isaac_usd": str(env_cfg.scene.robot.spawn.usd_path),
        }
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "root_height": result["root_pos_w"][2],
                          "root_speed": sum(x*x for x in result["root_lin_vel_w"][:2])**0.5,
                          "mass_matrix_shape": [len(result["mass_matrix"]), len(result["mass_matrix"][0])]}, indent=2))
        raw.close()


if __name__ == "__main__": main()
