"""GUI playback of the certified Stage 2 STAND route only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=20260723)
parser.add_argument("--duration-s", type=float, default=30.0)
parser.add_argument("--validate-only", action="store_true")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    task = "Isaac-Velocity-Flat-G1-Run-Eval-v0"
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (4.5, -5.5, 3.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.7)
    if args.device is not None:
        env_cfg.sim.device = args.device
    print("active_state=STAND active_expert=Stage_2_model_4246")
    print(f"checkpoint={checkpoint}")
    print("command_vx=0.0 command_vy=0.0 command_yaw_rate=0.0")
    print("routing=stage2_action_only RUN=0 bridge=0 scripted_offset=0 camera=world_orientation_fixed")
    if args.validate_only:
        load_walk_expert(checkpoint)
        print("preflight=PASS simulation_started=false")
        return

    with launch_simulation(env_cfg, args):
        raw = gym.make(task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped
        expert = load_walk_expert(checkpoint, device=env.device)
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        contact = env.scene.sensors["contact_forces"]
        foot_body_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_foot_ids = [contact.body_names.index(name) for name in foot_names]
        all_joint_ids, _ = robot.find_joints(".*")
        wrapped.reset()
        zero = MotionCommand(0.0, 0.0)
        dt = float(env.step_dt)
        for step in range(max(1, round(args.duration_s / dt))):
            command_term.vel_command_b.zero_()
            observations = wrapped.get_observations()
            state = canonical_state_from_legacy_observation(
                observations["policy"], heading_w_rad=robot.data.heading_w.torch
            )
            with torch.inference_mode():
                actions = expert(state, zero)
                wrapped.step(actions)
            if step % max(1, round(0.5 / dt)) == 0:
                gravity = robot.data.projected_gravity_b.torch
                roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
                pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1].square() + gravity[:, 2].square()))
                forces = contact.data.net_forces_w_history.torch[:, :, sensor_foot_ids, :]
                contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
                velocity_ratio = (
                    robot.data.joint_vel.torch[:, all_joint_ids].abs()
                    / robot.data.joint_vel_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
                )
                torque_ratio = (
                    robot.data.applied_torque.torch[:, all_joint_ids].abs()
                    / robot.data.joint_effort_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
                )
                payload = {
                    "active_state": "STAND",
                    "active_expert": "Stage 2 model_4246",
                    "command_vx_vy_yaw": [0.0, 0.0, 0.0],
                    "horizontal_speed_mps": float(robot.data.root_lin_vel_b.torch[0, :2].norm().item()),
                    "roll_rad": float(roll[0].item()),
                    "pitch_rad": float(pitch[0].item()),
                    "contact_state": ["left" if contacts[0, 0] else "", "right" if contacts[0, 1] else ""],
                    "double_support": bool(contacts[0].all().item()),
                    "flight": bool((~contacts[0]).all().item()),
                    "saturation": bool((velocity_ratio[0] >= 0.95).any() or (torque_ratio[0] >= 0.95).any()),
                    "hold_elapsed_s": round((step + 1) * dt, 2),
                }
                print("STAND_STATUS " + json.dumps(payload))
        wrapped.close()


if __name__ == "__main__":
    main()
