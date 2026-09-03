"""Interactive W1AAllDirectionWalk playback; no action correction."""

from __future__ import annotations

import argparse
import hashlib
import math
import msvcrt
import sys
from pathlib import Path

import gymnasium as gym
import torch

EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main():
    cfg, agent_cfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs, cfg.episode_length_s = 1, 3600.
    if args.device:
        cfg.sim.device = agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=agent_cfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        actor = FrozenGaitActor(args.checkpoint).to(device).eval()
        robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[i]) for i in feet]
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        angle, speed, step, slip_streak, sat_streak = 0., .3, 0, 0, 0
        obs, _ = wrapped.reset()
        obs = obs.to(device)
        checksum = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        print("W1AAllDirectionWalk: A/D direction +/-5deg, W/S speed +/-0.05m/s, X stop, ESC quit")
        running = True
        while running:
            while msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "\x1b": running = False
                elif key == "a": angle = (angle + 5) % 360
                elif key == "d": angle = (angle - 5) % 360
                elif key == "w": speed = min(1.2, speed + .05)
                elif key == "s": speed = max(0., speed - .05)
                elif key == "x": speed = 0.
            radians = math.radians(angle)
            command = torch.tensor([speed * math.cos(radians), speed * math.sin(radians), 0.], device=device)
            term.external_override[0] = command
            with torch.inference_mode():
                action = actor(obs["policy"], torch.zeros(1, device=device))
            obs, _, dones, _ = wrapped.step(action)
            obs = obs.to(device)
            actual = robot.data.root_lin_vel_b[0, :2]
            actual_speed = torch.linalg.vector_norm(actual)
            actual_direction = math.degrees(math.atan2(float(actual[1]), float(actual[0]))) % 360
            direction_error = abs((actual_direction - angle + 180) % 360 - 180) if speed > .05 else float("nan")
            vector_mae = torch.linalg.vector_norm(actual - command[:2])
            forces = sensor.data.net_forces_w_history[0, -1, feet, :].norm(dim=-1)
            contacts = forces > 5
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[0, robot_feet, :2], dim=-1)
            slipping = bool(((foot_speed > .55) & contacts).any())
            slip_streak = slip_streak + 1 if slipping else 0
            limits = robot.data.joint_vel_limits[0]
            if limits.ndim == 2: limits = limits[:, 1].abs()
            saturated = bool((robot.data.joint_vel[0].abs() / limits.clamp_min(1e-6) > .95).any())
            sat_streak = sat_streak + 1 if saturated else 0
            gravity = robot.data.projected_gravity_b[0]
            roll = torch.atan2(gravity[1].abs(), gravity[2].abs().clamp_min(1e-6))
            pitch = torch.atan2(gravity[0].abs(), gravity[2].abs().clamp_min(1e-6))
            flight = not bool(contacts.any())
            gait = "FALL" if bool(dones[0]) else ("WALK_LIKE" if not flight else "ISOLATED_FLIGHT")
            if step % 10 == 0:
                print(" | ".join((
                    f"TARGET VX {float(command[0]):+.2f}", f"TARGET VY {float(command[1]):+.2f}",
                    f"ACTUAL VX {float(actual[0]):+.2f}", f"ACTUAL VY {float(actual[1]):+.2f}",
                    f"TARGET DIRECTION {angle:.1f}", f"ACTUAL DIRECTION {actual_direction:.1f}",
                    f"VECTOR MAE {float(vector_mae):.3f}", f"DIRECTION ERROR {direction_error:.1f}",
                    f"DETECTED GAIT {gait}", f"ROLL {float(roll):.3f}", f"PITCH {float(pitch):.3f}",
                    f"SLIP {slip_streak >= 5}", f"IMPACT {float(forces.max()) > 3500}",
                    f"SATURATION {sat_streak >= 5}", f"FALL {bool(dones[0])}",
                    f"CHECKPOINT SHA {checksum[:16]}")))
            step += 1
        wrapped.close()


if __name__ == "__main__":
    main()
