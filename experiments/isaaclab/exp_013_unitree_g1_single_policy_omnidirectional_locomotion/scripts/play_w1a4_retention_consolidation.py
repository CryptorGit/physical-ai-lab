"""W1A4RetentionConsolidation playback; the selected actor alone supplies actions."""
import argparse
import hashlib
import math
import msvcrt
import sys
from pathlib import Path

import gymnasium as gym
import torch

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parents[2]
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks
import g1_omnidirectional.tasks
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument(
    "--reference",
    default=str(
        REPO
        / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
        "phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
    ),
)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main():
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600
    with launch_simulation(cfg, args):
        wrapper = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env = wrapper.unwrapped
        actor = FrozenGaitActor(args.checkpoint).to(env.device).eval()
        reference = FrozenGaitActor(args.reference).to(env.device).eval()
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        feet = [index for index, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[index]) for index in feet]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        angle, speed, step, slip_steps, saturation_steps = 0.0, 0.3, 0, 0, 0
        obs, _ = wrapper.reset()
        obs = obs.to(env.device)
        checksum = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        print("W1A4RetentionConsolidation: A/D direction, W/S speed, X stop, ESC quit")
        running = True
        while running:
            while msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "\x1b":
                    running = False
                elif key == "a":
                    angle = (angle + 5) % 360
                elif key == "d":
                    angle = (angle - 5) % 360
                elif key == "w":
                    speed = min(1.2, speed + 0.05)
                elif key == "s":
                    speed = max(0.0, speed - 0.05)
                elif key == "x":
                    speed = 0.0
            radians = math.radians(angle)
            target = torch.tensor(
                [speed * math.cos(radians), speed * math.sin(radians), 0.0], device=env.device
            )
            command.external_override[0] = target
            with torch.inference_mode():
                mean = actor(obs["policy"], torch.zeros(1, device=env.device))
                ref_mean = reference(obs["policy"], torch.zeros(1, device=env.device))
                std = reference.log_std_walk.exp()
                reference_kl = float((0.5 * ((ref_mean - mean) / std).square().sum(-1)).mean())
            obs, _, done, _ = wrapper.step(mean)
            obs = obs.to(env.device)
            actual = robot.data.root_lin_vel_b[0, :2]
            actual_speed = float(torch.linalg.vector_norm(actual))
            actual_direction = math.degrees(math.atan2(float(actual[1]), float(actual[0]))) % 360
            direction_error = abs((actual_direction - angle + 180) % 360 - 180) if speed > 0.05 else float("nan")
            mae = float(torch.linalg.vector_norm(actual - target[:2]))
            forces = sensor.data.net_forces_w_history[0, -1, feet, :].norm(dim=-1)
            contacts = forces > 5
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[0, robot_feet, :2], dim=-1)
            slipping = bool(((foot_speed > 0.55) & contacts).any())
            slip_steps = slip_steps + 1 if slipping else 0
            limits = robot.data.joint_vel_limits[0]
            limits = limits[:, 1].abs() if limits.ndim == 2 else limits
            saturated = bool((robot.data.joint_vel[0].abs() / limits.clamp_min(1e-6) > 0.95).any())
            saturation_steps = saturation_steps + 1 if saturated else 0
            gravity = robot.data.projected_gravity_b[0]
            roll = float(torch.atan2(gravity[1].abs(), gravity[2].abs().clamp_min(1e-6)))
            pitch = float(torch.atan2(gravity[0].abs(), gravity[2].abs().clamp_min(1e-6)))
            gait = "FALL" if bool(done[0]) else ("WALK_LIKE" if contacts.any() else "ISOLATED_FLIGHT")
            if step % 10 == 0:
                print(
                    f"TARGET VX/VY {target[0]:+.2f}/{target[1]:+.2f} | "
                    f"ACTUAL VX/VY {actual[0]:+.2f}/{actual[1]:+.2f} | "
                    f"TARGET/ACTUAL DIRECTION {angle:.1f}/{actual_direction:.1f} | "
                    f"TARGET/ACTUAL SPEED {speed:.2f}/{actual_speed:.2f} | "
                    f"VECTOR MAE {mae:.3f} | DIRECTION ERROR {direction_error:.1f} | "
                    f"LOW-SPEED REFERENCE KL {reference_kl:.5f} | WALK GAIT {gait} | "
                    f"ROLL/PITCH {roll:.3f}/{pitch:.3f} | SLIP {slip_steps >= 5} | "
                    f"IMPACT {float(forces.max()) > 3500} | SATURATION {saturation_steps >= 5} | "
                    f"FALL {bool(done[0])} | CHECKPOINT SHA {checksum[:16]}"
                )
            step += 1
        wrapper.close()


if __name__ == "__main__":
    main()
