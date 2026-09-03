"""Single exp_006 RUN expert GUI playback; RUN or RUN-with-heading only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--speed", type=float, required=True)
parser.add_argument("--turn-degrees", type=float, default=0.0)
parser.add_argument("--direction", choices=("Left", "Right"), default="Left")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=20260723)
parser.add_argument("--duration-s", type=float, default=22.0)
parser.add_argument("--validate-only", action="store_true", help="Resolve config/checkpoint without launching Isaac Sim.")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if args.speed not in (2.4, 2.6, 2.8, 3.0):
        raise ValueError("RUN_LOW GUI accepts only audited command points 2.4/2.6/2.8/3.0 m/s")
    turning = abs(args.turn_degrees) > 1.0e-9
    if turning and abs(args.turn_degrees) not in (45.0, 90.0):
        raise ValueError("Stage 0 GUI accepts only formal 45 or 90 degree TURN magnitudes")
    task = "Isaac-Motion-Flat-G1-Command-TurnFull-Eval-v0" if turning else "Isaac-Motion-Flat-G1-Command-Run-Eval-v0"
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 30.0
    env_cfg.seed = args.seed
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (11.0, -12.0, 8.0)
    env_cfg.viewer.lookat = (4.0, 2.0 if args.direction == "Left" else -2.0, 0.7)
    reset = env_cfg.events.reset_base.params
    reset["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    reset["velocity_range"] = {name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
    command = env_cfg.commands.base_velocity
    command.run_speed_range = (args.speed, args.speed)
    if turning:
        command.rehearsal_probabilities = (0.0, 1.0, 0.0, 0.0, 0.0)
        command.turn_angles_deg = (abs(args.turn_degrees),)
        command.turn_angle_probabilities = (1.0,)
        command.turn_direction_probabilities = (1.0, 0.0) if args.direction == "Left" else (0.0, 1.0)
        command.turn_speed_range = (min(args.speed, 2.0), min(args.speed, 2.0))
        command.turn_script_durations_s = (3.0, 8.0, 4.0)
        command.phase_duration_jitter_fraction = 0.0
        command.deterministic_turn_evaluation = False
    if args.device is not None:
        env_cfg.sim.device = args.device
    signed_angle = abs(args.turn_degrees) * (1.0 if args.direction == "Left" else -1.0)
    turn_speed = min(args.speed, 2.0) if turning else None
    print("actor=G1CommandResidualActor selected_route=RUN/TURN candidate=A")
    print(f"checkpoint={checkpoint}")
    print(f"command=RUN run_speed_mps={args.speed:.3f} turn_degrees={signed_angle:.1f} turn_speed_mps={turn_speed}")
    print("environments=1 camera=world_orientation_fixed camera_yaw_follow=false walk_run_switching=false")
    if args.validate_only:
        print("preflight=PASS simulation_started=false")
        return

    with launch_simulation(env_cfg, args):
        raw = gym.make(task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped
        agent_cfg.device = env.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
        policy = runner.get_inference_policy(device=env.device)
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        wrapped.reset()
        dt = float(env.step_dt)
        max_steps = max(1, round(args.duration_s / dt))
        last_segment = int(term.segment_index[0].item()) if turning else 0
        last_commanded = 0.0
        last_actual = 0.0
        for step in range(max_steps):
            observations = wrapped.get_observations()
            with torch.inference_mode():
                actions = policy(observations)
                wrapped.step(actions)
            heading = float(robot.data.heading_w.torch[0].item())
            actual_speed = float(robot.data.root_lin_vel_b.torch[0, 0].item())
            target_speed = float(term.vel_command_b[0, 0].item())
            segment = int(term.segment_index[0].item()) if turning else 0
            if turning:
                last_commanded = float(term.commanded_turn_angle_rad[0].item())
                last_actual = float(term.actual_accumulated_yaw_rad[0].item())
            if step % max(1, round(0.5 / dt)) == 0:
                print(json.dumps({"time_s": round(step * dt, 2), "phase": segment, "target_speed_mps": target_speed, "actual_speed_mps": actual_speed, "actual_heading_deg": math.degrees(heading), "accumulated_yaw_deg": math.degrees(last_actual)}))
            if turning and last_segment == 1 and segment == 2:
                print(f"TURN_RESULT commanded_angle_deg={math.degrees(last_commanded):.3f} actual_angle_deg={math.degrees(last_actual):.3f} final_error_deg={abs(math.degrees(last_commanded-last_actual)):.3f}")
            last_segment = segment
        if turning:
            print(f"TURN_FINAL commanded_angle_deg={math.degrees(last_commanded):.3f} actual_angle_deg={math.degrees(last_actual):.3f} final_error_deg={abs(math.degrees(last_commanded-last_actual)):.3f}")
        wrapped.close()


if __name__ == "__main__":
    main()
