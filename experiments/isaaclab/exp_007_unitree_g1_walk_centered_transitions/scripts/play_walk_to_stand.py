"""GUI playback of the Stage 4 WALK -> edge -> STAND hard-switch route."""

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
parser.add_argument("--walk-checkpoint", required=True)
parser.add_argument("--stand-to-walk-checkpoint", required=True)
parser.add_argument("--walk-to-stand-checkpoint", required=True)
parser.add_argument("--speed", type=float, choices=(0.6, 0.8, 1.0, 1.2), default=1.0)
parser.add_argument("--seed", type=int, default=20260827)
parser.add_argument("--validate-only", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def minimum_jerk(u):
    u = max(0.0, min(1.0, u))
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def main() -> None:
    paths = [
        Path(value).resolve(strict=True)
        for value in (
            args.stand_checkpoint,
            args.walk_checkpoint,
            args.stand_to_walk_checkpoint,
            args.walk_to_stand_checkpoint,
        )
    ]
    print("ROUTE=WALK -> WALK_TO_STAND -> STAND; HARD_SWITCH=true; ACTION_BLEND=false")
    print("CONTROLLER_TYPE=LEARNED_TRANSITION_EXPERT")
    print(f"SOURCE_SPEED={args.speed}; SUPPORTED_DISCRETE=[0.6,0.8,1.0,1.2]")
    if args.validate_only:
        for path in paths:
            load_walk_expert(path)
        print("preflight=PASS simulation_started=false")
        return
    cfg, agent = resolve_task_config(
        "Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 24.0
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (6.0, -7.5, 3.8)
    cfg.viewer.lookat = (3.0, 0.0, 0.8)
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        env = wrapped.unwrapped
        stand, walk, start, stop = [
            load_walk_expert(path, device=env.device) for path in paths
        ]
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in names]
        wrapped.reset()
        heading = robot.data.heading_w.torch.clone()
        stop_origin = robot.data.root_pos_w.torch[:, :2].clone()
        phase, elapsed, streak, source_switches = "SOURCE_SETUP", 0.0, 0.0, 0
        previous_support = 0
        no_switch_elapsed = 0.0
        filtered_yaw = torch.zeros(1, device=env.device)
        previous_action = torch.zeros(1, 37, device=env.device)
        entry_jump = exit_jump = 0.0
        stop_requested = False
        result = "IN_PROGRESS"
        dt = float(env.step_dt)
        for step in range(round(22.0 / dt)):
            if phase == "SOURCE_SETUP" and elapsed >= 2.0:
                phase, elapsed, streak = "SOURCE_EDGE", 0.0, 0.0
                heading[:] = robot.data.heading_w.torch
            if phase == "SOURCE_SETUP":
                command_vx = 0.0
            elif phase == "SOURCE_EDGE":
                command_vx = args.speed * minimum_jerk(elapsed / 1.5)
            elif phase == "WALK":
                command_vx = args.speed
            elif phase == "WALK_TO_STAND":
                command_vx = args.speed * (1.0 - minimum_jerk(elapsed / 1.6))
            else:
                command_vx = 0.0
            error = torch.atan2(
                torch.sin(heading - robot.data.heading_w.torch),
                torch.cos(heading - robot.data.heading_w.torch),
            )
            raw = (
                0.8 * error - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]
            ).clamp(-0.3, 0.3)
            low = filtered_yaw + 0.15 * (raw - filtered_yaw)
            filtered_yaw += (low - filtered_yaw).clamp(-0.01, 0.01)
            if phase in ("SOURCE_SETUP", "STAND"):
                filtered_yaw.zero_()
            term.vel_command_b.zero_()
            term.vel_command_b[:, 0] = command_vx
            term.vel_command_b[:, 2] = filtered_yaw
            legacy = wrapped.get_observations()["policy"]
            state = canonical_state_from_legacy_observation(
                legacy, heading_w_rad=robot.data.heading_w.torch
            )
            command = MotionCommand(
                torch.tensor([command_vx], device=env.device),
                heading,
                target_yaw_rate_radps=filtered_yaw,
            )
            active = (
                stand
                if phase in ("SOURCE_SETUP", "STAND")
                else start
                if phase == "SOURCE_EDGE"
                else walk
                if phase == "WALK"
                else stop
            )
            with torch.inference_mode():
                action = active(state, command)
                _, _, done, _ = wrapped.step(action)
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            support = int(contacts[0, 0]) + 2 * int(contacts[0, 1])
            speed = float(robot.data.root_lin_vel_b.torch[0, 0])
            horizontal = float(robot.data.root_lin_vel_b.torch[0, :2].norm())
            vertical = abs(float(robot.data.root_lin_vel_w.torch[0, 2]))
            heading_error = abs(float(error[0]))
            g = robot.data.projected_gravity_b.torch[0]
            roll = float(torch.atan2(g[1], -g[2]))
            pitch = float(torch.atan2(-g[0], torch.sqrt(g[1] ** 2 + g[2] ** 2)))
            if phase == "WALK_TO_STAND" and elapsed <= dt * 1.5:
                entry_jump = float(torch.linalg.vector_norm(action - previous_action))
            if phase == "STAND" and elapsed <= dt * 1.5:
                exit_jump = float(torch.linalg.vector_norm(action - previous_action))
            if phase == "SOURCE_EDGE" and support and support != previous_support:
                source_switches += 1
            if phase == "WALK_TO_STAND":
                if support and support != previous_support:
                    no_switch_elapsed = 0.0
                else:
                    no_switch_elapsed += dt
            previous_support = support
            if phase == "SOURCE_EDGE":
                good = (
                    speed >= 0.75 * args.speed
                    and abs(speed - args.speed) <= 0.20
                    and heading_error <= 0.12
                    and source_switches >= 2
                )
                streak = streak + dt if good else 0.0
                if streak >= 0.4:
                    phase, elapsed, streak = "WALK", 0.0, 0.0
            elif phase == "WALK" and elapsed >= 3.0:
                phase, elapsed, streak = "WALK_TO_STAND", 0.0, 0.0
                stop_requested = True
                stop_origin[:] = robot.data.root_pos_w.torch[:, :2]
            elif phase == "WALK_TO_STAND":
                good = (
                    horizontal <= 0.08
                    and vertical <= 0.05
                    and heading_error <= 0.12
                    and abs(roll) <= 0.10
                    and abs(pitch) <= 0.10
                    and support == 3
                    and no_switch_elapsed >= 0.4
                )
                streak = streak + dt if good else 0.0
                if streak >= 0.4:
                    phase, elapsed, streak = "STAND", 0.0, 0.0
            displacement = robot.data.root_pos_w.torch[0, :2] - stop_origin[0]
            forward = torch.stack((torch.cos(heading[0]), torch.sin(heading[0])))
            stopping_distance = float((displacement * forward).sum())
            if step % max(1, round(0.25 / dt)) == 0 and phase != "SOURCE_SETUP":
                print("STAGE4_STATUS " + json.dumps({
                    "state": phase,
                    "active_controller": (
                        "stand_to_walk_transition_v1"
                        if phase == "SOURCE_EDGE"
                        else "walk_steady_state_expert_v1"
                        if phase == "WALK"
                        else "walk_to_stand_transition_v1"
                        if phase == "WALK_TO_STAND"
                        else "stage2_model_4246"
                    ),
                    "controller_type": "LEARNED_TRANSITION_EXPERT",
                    "source_speed_mps": args.speed,
                    "actual_speed_mps": speed,
                    "stop_request": stop_requested,
                    "transition_elapsed_s": elapsed if phase == "WALK_TO_STAND" else 0.0,
                    "completion_detector": phase == "STAND",
                    "support_state": support,
                    "heading_error_rad": heading_error,
                    "residual_speed_mps": horizontal,
                    "stopping_distance_m": stopping_distance,
                    "entry_action_jump_l2": entry_jump,
                    "exit_action_jump_l2": exit_jump,
                    "stand_takeover_result": result,
                }))
            previous_action[:] = action
            if bool(done[0]):
                result = "FAIL"
                break
            if phase == "STAND" and elapsed >= 5.0:
                result = "PASS"
                break
            elapsed += dt
        print("FINAL_RESULT=" + result)
        wrapped.close()


if __name__ == "__main__":
    main()
