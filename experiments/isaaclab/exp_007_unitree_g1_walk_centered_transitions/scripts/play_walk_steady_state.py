"""GUI playback of the Stage 2W-B FULL_PASS steady-WALK expert."""

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
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--speed", type=float, choices=(0.6, 0.8, 1.0, 1.2), default=1.0)
parser.add_argument("--seed", type=int, default=20260803)
parser.add_argument("--validate-only", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    print("STAGE2WB_STATUS=FULL_PASS")
    print("STATE=WALK CAPABILITY=FULL TRANSITION=NONE MODEL=walk_steady_state_expert_v1")
    print("SUPPORTED_RANGE_MPS=[0.6,1.2]")
    print("RUN_expert=NOT_LOADED transition_expert=NOT_LOADED transition_bridge=ZERO")
    print(f"target_speed_mps={args.speed} camera=world_orientation_fixed")
    expert = load_walk_expert(checkpoint)
    if args.validate_only:
        print("preflight=PASS simulation_started=false")
        return

    task = "Isaac-Velocity-Flat-G1-IndependentWalk-Eval-v0"
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 60.0
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (6.0, -7.5, 3.8)
    cfg.viewer.lookat = (3.0, 0.0, 0.8)
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        expert = load_walk_expert(checkpoint, device=env.device)
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        wrapped.reset()
        term.target_speed[:] = args.speed
        term.target_heading_w[:] = robot.data.heading_w.torch
        term.path_origin_xy[:] = robot.data.root_pos_w.torch[:, :2]
        target_heading = term.target_heading_w.clone()
        previous_action = torch.zeros(1, 37, device=env.device)
        dt = float(env.step_dt)
        result = "WALK_HOLD_COMPLETE"
        for step in range(round(30.0 / dt)):
            obs = wrapped.get_observations()["policy"]
            command = term.vel_command_b.clone()
            state = canonical_state_from_legacy_observation(obs, heading_w_rad=robot.data.heading_w.torch)
            with torch.inference_mode():
                action = expert(
                    state,
                    MotionCommand(command[:, 0], target_heading, target_yaw_rate_radps=command[:, 2]),
                )
                _, _, done, _ = wrapped.step(action)
            if bool(done[0]):
                result = "DIAGNOSTIC_FAIL_FALL_OR_TIMEOUT"
                break
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            ankle_ratio = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            )
            heading_error = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            displacement = robot.data.root_pos_w.torch[:, :2] - term.path_origin_xy
            normal = torch.stack((-torch.sin(target_heading), torch.cos(target_heading)), dim=1)
            cross_track = (displacement * normal).sum(dim=1).abs()
            action_rate = torch.linalg.vector_norm(action - previous_action, dim=1) / dt
            previous_action = action.clone()
            if step % max(1, round(0.25 / dt)) == 0:
                print(
                    "WALK_STEADY_STATE_STATUS "
                    + json.dumps(
                        {
                            "stage2wb_status": "FULL_PASS",
                            "state": "WALK",
                            "capability": "FULL",
                            "supported_range_mps": [0.6, 1.2],
                            "model": "walk_steady_state_expert_v1",
                            "transition": "NONE",
                            "target_speed_mps": args.speed,
                            "command_speed_mps": float(command[0, 0]),
                            "actual_speed_mps": float(robot.data.root_lin_vel_b.torch[0, 0]),
                            "heading_error_rad": abs(float(heading_error[0])),
                            "yaw_rate_command_radps": float(command[0, 2]),
                            "path_drift_m": float(cross_track[0]),
                            "ankle_effort_utilization": [float(value) for value in ankle_ratio[0]],
                            "support_foot": (
                                "DOUBLE" if bool(contacts[0].all()) else
                                "LEFT" if bool(contacts[0, 0]) else
                                "RIGHT" if bool(contacts[0, 1]) else "FLIGHT"
                            ),
                            "action_rate": float(action_rate[0]),
                            "saturation_instantaneous": bool((ankle_ratio[0] >= 0.95).any()),
                        }
                    )
                )
        print("FINAL_RESULT=" + result)
        wrapped.close()


if __name__ == "__main__":
    main()
