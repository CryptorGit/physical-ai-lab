"""Interactive unified-student diagnostic player."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]

import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_walk_observation  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from unified_walk_run.command_profile import minimum_jerk  # noqa: E402
from unified_walk_run.student_actor import UnifiedWalkRunStudent123  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--start-speed", type=float, required=True)
parser.add_argument("--target-speed", type=float, required=True)
parser.add_argument("--student-checkpoint", type=Path, required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main():
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = 1
    task_cfg.episode_length_s = 120.0
    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg), clip_actions=agent_cfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        payload = torch.load(args.student_checkpoint, map_location=device, weights_only=False)
        student = UnifiedWalkRunStudent123().to(device)
        student.load_state_dict(payload["student"], strict=True)
        student.eval()
        heading = env.scene["robot"].data.heading_w.torch.clone()
        command_term = env.command_manager.get_term("base_velocity")
        previous_contacts = torch.zeros(1, 2, dtype=torch.bool, device=device)
        valid_streak = 0
        for step in range(round(120 / float(env.step_dt))):
            elapsed = step * float(env.step_dt)
            alpha = minimum_jerk(torch.tensor([max(0.0, elapsed - 4.0) / 1.4], device=device))
            speed = args.start_speed + (args.target_speed - args.start_speed) * alpha
            error = torch.atan2(torch.sin(heading - env.scene["robot"].data.heading_w.torch), torch.cos(heading - env.scene["robot"].data.heading_w.torch))
            yaw = error.clamp(-1.0, 1.0)
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = speed, yaw
            legacy = wrapped.get_observations()["policy"]
            canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=env.scene["robot"].data.heading_w.torch)
            motion = MotionCommand(speed, heading, target_yaw_rate_radps=yaw)
            with torch.no_grad():
                action = student(to_walk_observation(canonical, motion))
                wrapped.step(action)
            actual = float(env.scene["robot"].data.root_lin_vel_b.torch[0, 0])
            speed_ok = abs(actual - args.target_speed) <= 0.2
            valid_streak = valid_streak + 1 if speed_ok and abs(float(error[0])) <= 0.12 else 0
            if step % 25 == 0:
                print(
                    f"controller=UNIFIED_LOCOMOTION target={float(speed[0]):.2f} actual={actual:.2f} "
                    f"walk_valid_streak={valid_streak} heading={float(error[0]):.3f}"
                )
        wrapped.close()


if __name__ == "__main__":
    main()
