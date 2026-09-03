"""GUI-only playback of the best Stage 2R diagnostic checkpoint.

This is deliberately labelled NO_GO and does not expose a production route.
"""

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
parser.add_argument("--speed", type=float, choices=(0.6, 0.8, 1.0, 1.2), default=1.0)
parser.add_argument("--seed", type=int, default=20260726)
parser.add_argument("--validate-only", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def minimum_jerk(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    print("STAGE2R_STATUS=NO_GO_RETRAIN")
    print("WARNING=diagnostic checkpoint only; no supported WALK range or formal transition claim")
    print(f"active_expert=unified_stage2r_diagnostic checkpoint={checkpoint}")
    print("RUN_expert=NOT_LOADED transition_bridge=ZERO scripted_offset=ZERO")
    print("camera=world_orientation_fixed heading_controller=smoothed_fixed_target")
    if args.validate_only:
        load_walk_expert(checkpoint)
        print("preflight=PASS simulation_started=false")
        return

    task = "Isaac-Velocity-Flat-G1-Run-Eval-v0"
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 22.0
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (5.0, -7.5, 3.5)
    cfg.viewer.lookat = (2.5, 0.0, 0.8)
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        expert = load_walk_expert(checkpoint, device=env.device)
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        _, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        wrapped.reset()
        target_heading = robot.data.heading_w.torch.clone()
        yaw_command = torch.zeros(1, device=env.device)
        dt = float(env.step_dt)
        result = "DIAGNOSTIC_COMPLETE_NOT_FORMAL"
        for step in range(round(17.0 / dt)):
            elapsed = step * dt
            if elapsed < 2.0:
                phase, vx, progress = "STAND_HOLD", 0.0, 0.0
            elif elapsed < 4.0:
                progress = (elapsed - 2.0) / 2.0
                phase, vx = "ACCELERATION_RAMP", args.speed * minimum_jerk(progress)
            elif elapsed < 8.0:
                phase, vx, progress = "WALK_HOLD", args.speed, 1.0
            elif elapsed < 10.0:
                progress = (elapsed - 8.0) / 2.0
                phase, vx = "DECELERATION_RAMP", args.speed * (1.0 - minimum_jerk(progress))
            else:
                phase, vx, progress = "FINAL_STAND_HOLD", 0.0, 1.0
            error = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            raw = (0.8 * error - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            low_pass = yaw_command + 0.15 * (raw - yaw_command)
            yaw_command += (low_pass - yaw_command).clamp(-0.01, 0.01)
            vx_tensor = torch.tensor([vx], device=env.device)
            term.vel_command_b.zero_()
            term.vel_command_b[:, 0] = vx
            term.vel_command_b[:, 2] = yaw_command
            obs = wrapped.get_observations()["policy"]
            state = canonical_state_from_legacy_observation(obs, heading_w_rad=robot.data.heading_w.torch)
            with torch.inference_mode():
                action = expert(state, MotionCommand(vx_tensor, target_heading, target_yaw_rate_radps=yaw_command))
                _, _, done, _ = wrapped.step(action)
            if bool(done[0]):
                result = "DIAGNOSTIC_FAIL_FALL_OR_TIMEOUT"
                break
            contacts = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1) > 5.0
            ankle_ratio = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            )
            if step % max(1, round(0.25 / dt)) == 0:
                print(
                    "UNIFIED_STAND_WALK_STATUS "
                    + json.dumps(
                        {
                            "classification": "NO_GO_RETRAIN",
                            "phase": phase,
                            "active_expert": "unified_stage2r_diagnostic",
                            "target_speed_mps": args.speed,
                            "command_speed_mps": vx,
                            "actual_speed_mps": float(robot.data.root_lin_vel_b.torch[0, :2].norm()),
                            "ramp_progress": progress,
                            "heading_error_rad": abs(float(error[0])),
                            "yaw_rate_command_radps": float(yaw_command[0]),
                            "ankle_torque_utilization": [float(value) for value in ankle_ratio[0]],
                            "support_state": [bool(value) for value in contacts[0]],
                            "double_support": int(contacts[0].sum()) == 2,
                            "saturation_instantaneous": bool((ankle_ratio[0] >= 0.95).any()),
                        }
                    )
                )
        print("FINAL_RESULT=" + result)
        wrapped.close()


if __name__ == "__main__":
    main()
