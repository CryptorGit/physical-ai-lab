"""Interactive DirectionalBaseline playback for the frozen selected parent."""

from __future__ import annotations

import argparse
import hashlib
import math
import msvcrt
import sys
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=20261399)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main():
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.5
    cfg.episode_length_s = 3600.
    cfg.seed = args.seed
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped
        actor = FrozenGaitActor(args.checkpoint).to(env.device).eval()
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        foot_bodies = [next(i for i, name in enumerate(robot.body_names) if name == sensor.body_names[j])
                       for j in feet]
        checkpoint_sha = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        command = torch.zeros(3, device=env.device)
        gait = torch.zeros(1, device=env.device)
        obs, _ = wrapped.reset()
        obs = obs.to(env.device)
        print("DirectionalBaseline keys: W/S vx, A/D vy, Q/E yaw, G gait, X stop, ESC quit")
        running = True
        step = 0
        slip_streak = 0
        saturation_streak = 0
        while running:
            while msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "\x1b":
                    running = False
                elif key == "w":
                    command[0] = min(float(command[0]) + .1, 2.4)
                elif key == "s":
                    command[0] = max(float(command[0]) - .1, -2.4)
                elif key == "a":
                    command[1] = min(float(command[1]) + .1, 2.4)
                elif key == "d":
                    command[1] = max(float(command[1]) - .1, -2.4)
                elif key == "q":
                    command[2] = min(float(command[2]) + .1, 1.0)
                elif key == "e":
                    command[2] = max(float(command[2]) - .1, -1.0)
                elif key == "g":
                    gait[0] = 1 - gait[0]
                elif key == "x":
                    command.zero_()
            term.external_override[0] = command
            with torch.inference_mode():
                action = actor(obs["policy"], gait)
            obs, _, dones, extras = wrapped.step(action)
            obs = obs.to(env.device)
            forces = sensor.data.net_forces_w_history[0, -1, feet, :].norm(dim=-1)
            contacts = forces > 5
            actual = robot.data.root_lin_vel_b[0]
            actual_yaw = robot.data.root_ang_vel_b[0, 2]
            cmd_speed = torch.linalg.vector_norm(command[:2])
            actual_speed = torch.linalg.vector_norm(actual[:2])
            if float(cmd_speed) > .05 and float(actual_speed) > .02:
                cosine = torch.dot(command[:2], actual[:2]) / (cmd_speed * actual_speed)
                direction_error = math.degrees(math.acos(max(-1., min(1., float(cosine)))))
            else:
                direction_error = float("nan")
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[0, foot_bodies, :2], dim=-1)
            slipping = bool(((foot_speed > .55) & contacts).any())
            slip_streak = slip_streak + 1 if slipping else 0
            limits = robot.data.joint_vel_limits[0]
            if limits.ndim == 2:
                limits = limits[:, 1].abs()
            saturated = bool((robot.data.joint_vel[0].abs() / limits.clamp_min(1e-6) > .95).any())
            saturation_streak = saturation_streak + 1 if saturated else 0
            flight = not bool(contacts.any())
            detected = "FALL" if bool(dones[0]) else (
                "PERIODIC_RUNNING" if flight and float(gait[0]) > .5 else
                "STAND_OR_NEAR_STAND" if float(actual_speed) < .1 else "WALK_LIKE"
            )
            if step % 10 == 0:
                fields = (
                    f"TARGET VX {float(command[0]):+.2f}", f"TARGET VY {float(command[1]):+.2f}",
                    f"ACTUAL VX {float(actual[0]):+.2f}", f"ACTUAL VY {float(actual[1]):+.2f}",
                    f"TARGET YAW RATE {float(command[2]):+.2f}", f"ACTUAL YAW RATE {float(actual_yaw):+.2f}",
                    f"GAIT COMMAND {'RUN' if float(gait[0]) > .5 else 'WALK'}", f"DETECTED GAIT {detected}",
                    f"DIRECTION ERROR {direction_error:.1f}", f"SPEED ERROR {abs(float(actual_speed-cmd_speed)):.2f}",
                    f"LEFT CONTACT {bool(contacts[0])}", f"RIGHT CONTACT {bool(contacts[1])}",
                    f"FLIGHT {flight}", f"SLIP {slip_streak >= 5}",
                    f"IMPACT {float(forces.max()) > 3500}", f"SATURATION {saturation_streak >= 5}",
                    f"FALL {bool(dones[0])}", f"CHECKPOINT SHA {checkpoint_sha[:16]}",
                )
                print(" | ".join(fields))
            root = robot.data.root_pos_w[0].cpu()
            env.sim.set_camera_view(
                eye=(float(root[0] - 3.5), float(root[1] - 4.0), 2.2),
                target=(float(root[0] + .5), float(root[1]), .9),
            )
            step += 1
        wrapped.close()


if __name__ == "__main__":
    main()
