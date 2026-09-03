"""GUI diagnostic playback for the frozen Stage 2 STAND-WALK controller."""

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
from g1_walk_centered.heading_controller import FixedHeadingConfig, fixed_heading_yaw_rate  # noqa: E402
from g1_walk_centered.stand_walk_controller import Phase, velocity_command  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--speed", type=float, default=1.2)
parser.add_argument("--ramp-duration", type=float, default=2.0)
parser.add_argument("--heading-mode", choices=("ZeroYaw", "FixedTarget"), default="ZeroYaw")
parser.add_argument("--seed", type=int, default=20260724)
parser.add_argument("--validate-only", action="store_true")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def main() -> None:
    if args.speed not in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
        raise ValueError("Stage 2 audit GUI accepts only explicitly audited speeds from 0.2 to 2.0 m/s; unsupported speeds are not clamped")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    print("WARNING: Stage 2 formal result is FAIL; this is diagnostic playback, not a supported production route.")
    print(f"active_expert=Stage_2_model_4246 checkpoint={checkpoint}")
    print(f"target_speed={args.speed} ramp=minimum_jerk/{args.ramp_duration}s vy=0 heading_mode={args.heading_mode}")
    print("routing=Stage2_only RUN=0 bridge=0 scripted_offset=0 camera=world_orientation_fixed")
    if args.validate_only:
        load_walk_expert(checkpoint)
        print("preflight=PASS simulation_started=false")
        return

    task = "Isaac-Velocity-Flat-G1-Run-Eval-v0"
    cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 30.0
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (5.5, -7.0, 3.5)
    cfg.viewer.lookat = (2.0, 0.0, 0.7)
    if args.device is not None:
        cfg.sim.device = args.device

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped
        expert = load_walk_expert(checkpoint, device=env.device)
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        _, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        all_joints, _ = robot.find_joints(".*")
        wrapped.reset()
        dt = float(env.step_dt)
        phase = torch.tensor([int(Phase.INITIAL_STAND_SETTLE)], device=env.device)
        elapsed = torch.zeros(1, device=env.device)
        streak = 0
        initial_heading = robot.data.heading_w.torch.clone()
        target_heading = initial_heading.clone()
        previous = torch.zeros(1, 37, device=env.device)
        result = "IN_PROGRESS"
        for step in range(round(29.0 / dt)):
            cmd = velocity_command(
                phase,
                elapsed,
                torch.tensor([args.speed], device=env.device),
                torch.tensor([args.ramp_duration], device=env.device),
            )
            yaw_cmd = torch.zeros_like(cmd)
            if args.heading_mode == "FixedTarget":
                yaw_cmd, _ = fixed_heading_yaw_rate(
                    target_heading,
                    robot.data.heading_w.torch,
                    robot.data.root_ang_vel_b.torch[:, 2],
                    FixedHeadingConfig(1.25, 0.10, 0.50),
                )
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0] = cmd
            command_term.vel_command_b[:, 2] = yaw_cmd
            obs = wrapped.get_observations()["policy"]
            state = canonical_state_from_legacy_observation(obs, heading_w_rad=robot.data.heading_w.torch)
            with torch.inference_mode():
                action = expert(state, MotionCommand(cmd, torch.zeros_like(cmd), target_yaw_rate_radps=yaw_cmd))
                _, _, done, _ = wrapped.step(action)
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            support = int(contacts[0].sum().item())
            speed = float(robot.data.root_lin_vel_b.torch[0, :2].norm().item())
            vz = float(abs(robot.data.root_lin_vel_w.torch[0, 2].item()))
            gravity = robot.data.projected_gravity_b.torch
            roll = float(torch.atan2(gravity[0, 1], -gravity[0, 2]).item())
            pitch = float(torch.atan2(-gravity[0, 0], torch.sqrt(gravity[0, 1] ** 2 + gravity[0, 2] ** 2)).item())
            heading = float(abs(torch.atan2(torch.sin(robot.data.heading_w.torch[0] - target_heading[0]), torch.cos(robot.data.heading_w.torch[0] - target_heading[0])).item()))
            action_rate = float(torch.linalg.vector_norm(action - previous).item() / dt)
            previous = action.clone()
            velocity_ratio = (robot.data.joint_vel.torch[:, all_joints].abs() / robot.data.joint_vel_limits.torch[:, all_joints].abs().clamp_min(1e-6)).amax()
            safe = speed <= .08 and vz <= .05 and abs(roll) <= .10 and abs(pitch) <= .10 and support == 2
            reached = speed >= max(.20, .75 * args.speed) and abs(speed - args.speed) <= .20 and heading <= .12 and abs(roll) <= .20 and abs(pitch) <= .20
            p = int(phase.item())
            if bool(done[0]):
                result = "FAIL: fall_or_timeout"
                break
            if p == 0:
                streak = streak + 1 if safe else 0
                if streak * dt >= .4: target_heading[0] = robot.data.heading_w.torch[0]; phase[0] = 1; elapsed[0] = 0; streak = 0
                elif elapsed[0] >= 2.0: result = "FAIL: initial_stand_settle_failure"; break
            elif p == 1 and elapsed[0] >= 1.2: phase[0] = 2; elapsed[0] = 0
            elif p == 2 and elapsed[0] >= args.ramp_duration: phase[0] = 3; elapsed[0] = 0
            elif p == 3:
                streak = streak + 1 if reached else 0
                if streak * dt >= .4: phase[0] = 4; elapsed[0] = 0; streak = 0
                elif elapsed[0] >= 3.0: result = "FAIL: target_speed_not_reached"; break
            elif p == 4 and elapsed[0] >= 3.5: phase[0] = 5; elapsed[0] = 0
            elif p == 5 and elapsed[0] >= args.ramp_duration: phase[0] = 6; elapsed[0] = 0
            elif p == 6:
                streak = streak + 1 if safe else 0
                if streak * dt >= .4: phase[0] = 7; elapsed[0] = 0; streak = 0
                elif elapsed[0] >= 3.0: result = "FAIL: double_support_recovery_failure"; break
            elif p == 7 and elapsed[0] >= 5.0:
                result = "PASS: diagnostic_sequence_complete"
                break
            elapsed += dt
            if step % max(1, round(.25 / dt)) == 0:
                payload = {"phase": Phase(int(phase.item())).name, "heading_mode": args.heading_mode, "target_speed_mps": args.speed, "command_vx_mps": float(cmd.item()), "command_yaw_rate_radps": float(yaw_cmd.item()), "ramp_progress": min(1.0, float(elapsed.item()) / args.ramp_duration) if int(phase.item()) in (2, 5) else None, "actual_speed_mps": speed, "active_expert": "Stage 2 model_4246", "contact_state": contacts[0].tolist(), "double_support": support == 2, "heading_error_rad": heading, "action_rate": action_rate, "saturation_instantaneous": bool(velocity_ratio >= .95)}
                print("STAND_WALK_STATUS " + json.dumps(payload))
        print(f"FINAL_RESULT {result}")
        wrapped.close()


if __name__ == "__main__":
    main()
