"""Simulator preflight for CROUCH phase commands, rewards, resets and joint mapping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="results/exp_006_unitree_g1_command_skills/crouch_preflight_env.json")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    task = "Isaac-Motion-Flat-G1-Command-Crouch-Eval-v0"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    with launch_simulation(cfg, args_cli):
        env = gym.make(task, cfg=cfg)
        unwrapped = env.unwrapped
        observations, _ = env.reset()
        term = unwrapped.command_manager.get_term("base_velocity")
        robot = unwrapped.scene["robot"]
        ids = torch.tensor([0], device=unwrapped.device)
        phase_rows = []
        checkpoints = [
            0.25 * term.crouch_down_duration[0],
            term.crouch_down_duration[0] + 0.50 * term.crouch_hold_duration[0],
            term.crouch_down_duration[0] + term.crouch_hold_duration[0] + 0.50 * term.crouch_return_duration[0],
            term.crouch_down_duration[0] + term.crouch_hold_duration[0] + term.crouch_return_duration[0] + 0.50 * term.crouch_stand_hold_duration[0],
        ]
        for elapsed in checkpoints:
            term.segment_elapsed[ids] = elapsed
            term._update_command()
            command = term.extra_command[0]
            phase_rows.append({
                "phase": int(term.crouch_phase[0].item()),
                "relative_target_height": float(command[16].item()),
                "height_error": float(command[14].item()),
                "vertical_velocity": float(command[17].item()),
                "joint_limit_proximity": float(command[18].item()),
                "return_progress": float(command[19].item()),
                "entry_height_offset": float(command[20].item()),
                "target_vertical_velocity": float(command[21].item()),
                "phase_column": float(command[24].item()),
                "hold_progress": float(command[26].item()),
            })
        reset_success = True
        for _ in range(3):
            try:
                env.reset()
            except RuntimeError:
                reset_success = False
                break
        mutable_names = [name for name in term._MUTABLE_STATE_NAMES if isinstance(getattr(term, name, None), torch.Tensor)]
        mutable_normal = all(not torch.is_inference(getattr(term, name)) for name in mutable_names)
        auto_reset = False
        for _ in range(int(cfg.episode_length_s / unwrapped.step_dt) + 10):
            _, _, terminated, truncated, _ = env.step(torch.zeros(1, unwrapped.action_manager.total_action_dim, device=unwrapped.device))
            if bool((terminated | truncated)[0].item()):
                auto_reset = True
                break
        action_indices = [0, 1, 11, 12, 15, 16]
        action_joint_names = [robot.joint_names[index] for index in action_indices]
        expected = [
            "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        ]
        required_rewards = (
            "crouch_height_tracking", "crouch_vertical_velocity", "crouch_depth_progress",
            "crouch_hold_height", "crouch_return_height", "crouch_upright", "crouch_symmetry",
            "crouch_foot_contact", "crouch_foot_contact_loss", "crouch_foot_slip",
            "crouch_action_rate", "crouch_joint_velocity_saturation", "crouch_torque_saturation",
            "crouch_joint_limit", "crouch_hold_completion", "crouch_return_completion",
        )
        result = {
            "task_registered": task in gym.registry,
            "observation_dim": int(observations["policy"].shape[-1]),
            "skill_id": int(term.skill_id[0].item()),
            "commanded_drop_in_stage_a_range": 0.08 <= float(term.crouch_commanded_drop[0].item()) <= 0.15,
            "phase_sequence": [row["phase"] for row in phase_rows],
            "phase_rows": phase_rows,
            "action_joint_names": action_joint_names,
            "expected_action_joint_names": expected,
            "repeated_reset_success": reset_success,
            "auto_reset_success": auto_reset,
            "mutable_tensors_are_normal": mutable_normal,
            "mutable_tensor_count": len(mutable_names),
            "required_rewards_registered": all(name in unwrapped.reward_manager.active_terms for name in required_rewards),
        }
        result["passed"] = (
            result["task_registered"] and result["observation_dim"] == 152 and result["skill_id"] == 3
            and result["commanded_drop_in_stage_a_range"] and result["phase_sequence"] == [0, 1, 2, 3]
            and result["action_joint_names"] == expected and result["repeated_reset_success"]
            and result["auto_reset_success"] and result["mutable_tensors_are_normal"]
            and result["required_rewards_registered"]
        )
        output = (REPOSITORY_ROOT / args_cli.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        env.close()
        if not result["passed"]:
            raise RuntimeError("CROUCH environment preflight failed")


if __name__ == "__main__":
    main()
