"""Interactive diagnostic player for the frozen-WALK residual controller."""

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
    str(REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from unified_walk_run.command_profile import minimum_jerk  # noqa: E402
from unified_walk_run.frozen_walk_residual import FrozenWalkSpeedResidualController123  # noqa: E402
from unified_walk_run.student_actor import UnifiedWalkRunStudent123  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--start-speed", type=float, required=True)
parser.add_argument("--target-speed", type=float, required=True)
parser.add_argument("--residual-checkpoint", type=Path, required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def load_base(device):
    payload = torch.load(
        REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        map_location=device, weights_only=False,
    )
    state = {key.removeprefix("mlp."): value for key, value in payload["actor_state_dict"].items() if key.startswith("mlp.")}
    base = UnifiedWalkRunStudent123().to(device)
    base.load_state_dict(state, strict=True)
    return base


def main():
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = 1
    task_cfg.episode_length_s = 120.0
    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg), clip_actions=agent_cfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        payload = torch.load(args.residual_checkpoint, map_location=device, weights_only=False)
        controller = FrozenWalkSpeedResidualController123(load_base(device), payload["residual_bounds"].to(device)).to(device)
        controller.residual.load_state_dict(payload["residual"], strict=True)
        controller.eval()
        command_term = env.command_manager.get_term("base_velocity")
        heading = env.scene["robot"].data.heading_w.torch.clone()
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = env.scene["robot"].find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        walk_streak = 0
        for step in range(round(120 / float(env.step_dt))):
            elapsed = step * float(env.step_dt)
            alpha = minimum_jerk(torch.tensor([max(0.0, elapsed - 4.0) / 1.4], device=device))
            speed = args.start_speed + (args.target_speed - args.start_speed) * alpha
            error = torch.atan2(torch.sin(heading - env.scene["robot"].data.heading_w.torch), torch.cos(heading - env.scene["robot"].data.heading_w.torch))
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = speed, error.clamp(-1.0, 1.0)
            observation = wrapped.get_observations()["policy"]
            with torch.no_grad():
                base, residual, gate = controller.forward_components(observation)
                action = controller(observation)
                _, _, dones, _ = wrapped.step(action)
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(1) > 5
            support = int(contacts[0, 0]) + 2 * int(contacts[0, 1])
            actual = float(env.scene["robot"].data.root_lin_vel_b.torch[0, 0])
            valid = abs(actual - args.target_speed) <= 0.2 and abs(float(error[0])) <= 0.12 and support != 0 and not bool(dones[0])
            walk_streak = walk_streak + 1 if valid else 0
            if step % 25 == 0:
                print(
                    "CONTROLLER=FROZEN_WALK+SPEED_RESIDUAL "
                    f"requested={float(speed[0]):.2f} base_speed={min(float(speed[0]),1.2):.2f} "
                    f"gate={float(gate[0]):.3f} base_norm={float(base.norm()):.3f} "
                    f"residual_norm={float(residual.norm()):.3f} final_norm={float(action.norm()):.3f} "
                    f"gait={'flight' if support == 0 else 'support'} support={support} "
                    f"walk_valid_streak={walk_streak} periodic_running=diagnostic "
                    f"fall={bool(dones[0])} slip=diagnostic impact=diagnostic saturation=diagnostic"
                )
        wrapped.close()


if __name__ == "__main__":
    main()
