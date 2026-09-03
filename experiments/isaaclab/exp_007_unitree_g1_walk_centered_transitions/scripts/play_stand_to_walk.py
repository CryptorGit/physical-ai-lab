"""GUI playback of the Stage 3 STAND -> edge -> WALK hard-switch route."""

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
import g1_walk_centered.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stand-checkpoint", required=True)
parser.add_argument("--transition-checkpoint", required=True)
parser.add_argument("--walk-checkpoint", required=True)
parser.add_argument("--speed", type=float, choices=(0.6, 0.8, 1.0, 1.2), default=1.0)
parser.add_argument("--seed", type=int, default=20260818)
parser.add_argument("--validate-only", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def minimum_jerk(u):
    u = max(0.0, min(1.0, u))
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def main() -> None:
    paths = [Path(value).resolve(strict=True) for value in (
        args.stand_checkpoint, args.transition_checkpoint, args.walk_checkpoint
    )]
    print("ROUTE=STAND -> STAND_TO_WALK -> WALK; HARD_SWITCH=true; ACTION_BLEND=false")
    print(f"TARGET_SPEED={args.speed}; SUPPORTED_DISCRETE=[0.6,0.8,1.0,1.2]")
    if args.validate_only:
        for path in paths:
            load_walk_expert(path)
        print("preflight=PASS simulation_started=false")
        return
    cfg, agent = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 20.0
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (6.0, -7.5, 3.8)
    cfg.viewer.lookat = (3.0, 0.0, 0.8)
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        stand, transition, walk = [load_walk_expert(path, device=env.device) for path in paths]
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in names]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        wrapped.reset()
        heading = robot.data.heading_w.torch.clone()
        path_origin = robot.data.root_pos_w.torch[:, :2].clone()
        state_name, elapsed, streak, switches, previous_support = "STAND", 0.0, 0.0, 0, 0
        filtered_yaw = torch.zeros(1, device=env.device)
        previous_action = torch.zeros(1, 37, device=env.device)
        entry_jump = exit_jump = 0.0
        result = "IN_PROGRESS"
        dt = float(env.step_dt)
        for step in range(round(12.0 / dt)):
            if state_name == "STAND" and elapsed >= 2.0:
                state_name, elapsed = "STAND_TO_WALK", 0.0
                heading[:] = robot.data.heading_w.torch
                path_origin[:] = robot.data.root_pos_w.torch[:, :2]
            command_vx = 0.0 if state_name == "STAND" else (
                args.speed * minimum_jerk(elapsed / 1.5) if state_name == "STAND_TO_WALK" else args.speed
            )
            error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
            raw = (0.8 * error - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            low = filtered_yaw + 0.15 * (raw - filtered_yaw)
            filtered_yaw += (low - filtered_yaw).clamp(-0.01, 0.01)
            if state_name == "STAND":
                filtered_yaw.zero_()
            term.vel_command_b.zero_()
            term.vel_command_b[:, 0] = command_vx
            term.vel_command_b[:, 2] = filtered_yaw
            legacy = wrapped.get_observations()["policy"]
            canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
            command = MotionCommand(torch.tensor([command_vx], device=env.device), heading, target_yaw_rate_radps=filtered_yaw)
            active = stand if state_name == "STAND" else transition if state_name == "STAND_TO_WALK" else walk
            with torch.inference_mode():
                action = active(canonical, command)
                _, _, done, _ = wrapped.step(action)
            if state_name == "STAND_TO_WALK" and elapsed <= dt * 1.5:
                entry_jump = float(torch.linalg.vector_norm(action - previous_action))
            if state_name == "WALK" and elapsed <= dt * 1.5:
                exit_jump = float(torch.linalg.vector_norm(action - previous_action))
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            support = int(contacts[0, 0].item()) + 2 * int(contacts[0, 1].item())
            if state_name == "STAND_TO_WALK" and support and support != previous_support:
                switches += 1
            previous_support = support
            speed = float(robot.data.root_lin_vel_b.torch[0, 0])
            heading_error = abs(float(error[0]))
            good = speed >= 0.75 * args.speed and abs(speed - args.speed) <= 0.20 and heading_error <= 0.12 and switches >= 2
            streak = streak + dt if state_name == "STAND_TO_WALK" and good else 0.0
            detector = streak >= 0.4
            if detector:
                state_name, elapsed, streak = "WALK", 0.0, 0.0
            ankle = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            )
            cross = abs(float((robot.data.root_pos_w.torch[0, :2] - path_origin[0])[1]))
            if step % max(1, round(0.25 / dt)) == 0:
                print("STAGE3_STATUS " + json.dumps({
                    "state": state_name, "active_model": (
                        "stage2_model_4246" if state_name == "STAND" else
                        "stand_to_walk_transition_v1" if state_name == "STAND_TO_WALK" else
                        "walk_steady_state_expert_v1"
                    ), "target_speed_mps": args.speed, "actual_speed_mps": speed,
                    "transition_elapsed_s": elapsed, "completion_detector": detector,
                    "support_state": support, "heading_error_rad": heading_error,
                    "entry_action_jump_l2": entry_jump, "exit_action_jump_l2": exit_jump,
                    "path_drift_m": cross, "ankle_effort_max": float(ankle.max()),
                    "walk_takeover_result": result,
                }))
            previous_action[:] = action
            if bool(done[0]):
                result = "FAIL"
                break
            if state_name == "WALK" and elapsed >= 3.0:
                result = "PASS"
                break
            elapsed += dt
        print("FINAL_RESULT=" + result)
        wrapped.close()


if __name__ == "__main__":
    main()
