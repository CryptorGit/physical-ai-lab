"""Verify registration, observation layout and deterministic command switching."""

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


EXTRA_COMMAND_DIM = 29
POLICY_OBSERVATION_DIM = 152
RUN, STOP, TURN = 0, 1, 2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="results/exp_006_unitree_g1_command_skills/smoke_env.json")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    task = "Isaac-Motion-Flat-G1-Command-Sequence-Eval-v0"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    with launch_simulation(cfg, args_cli):
        env = gym.make(task, cfg=cfg)
        unwrapped = env.unwrapped
        observations, _ = env.reset()
        term = unwrapped.command_manager.get_term("base_velocity")
        ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
        observed_sequence = [int(term.skill_id[0].item())]
        command_snapshots = [term.extra_command[0].detach().cpu().tolist()]
        for _ in range(3):
            term.segment_elapsed[ids] = term.segment_duration[ids]
            term._advance_scripts()
            term._update_command()
            observed_sequence.append(int(term.skill_id[0].item()))
            command_snapshots.append(term.extra_command[0].detach().cpu().tolist())
        policy_obs = observations["policy"]
        correction_indices = (2, 7, 8, 3, 4, 19, 20, 0, 1)
        correction_joint_names = [unwrapped.scene["robot"].joint_names[index] for index in correction_indices]
        expected_correction_joint_names = [
            "torso_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
            "left_hip_pitch_joint", "right_hip_pitch_joint",
        ]
        # Execute one environment step so every newly registered reward term is called.
        env.step(torch.zeros(1, unwrapped.action_manager.total_action_dim, device=unwrapped.device))
        result = {
            "task": task,
            "registered": task in gym.registry,
            "observation_dim": int(policy_obs.shape[-1]),
            "expected_observation_dim": POLICY_OBSERVATION_DIM,
            "action_dim": unwrapped.action_manager.total_action_dim,
            "extra_command_dim": EXTRA_COMMAND_DIM,
            "sequence_ids": observed_sequence,
            "expected_sequence_ids": [RUN, TURN, RUN, STOP],
            "one_hot_sums": [sum(snapshot[:6]) for snapshot in command_snapshots],
            "local_heading_encoding_finite": all(
                math_value == math_value for snapshot in command_snapshots for math_value in snapshot[12:16]
            ),
            "run_path_lookahead_local_xy": command_snapshots[0][14:16],
            "run_path_state": {
                "signed_lateral_error_m": command_snapshots[0][17],
                "path_forward_velocity_mps": command_snapshots[0][18],
                "path_lateral_velocity_mps": command_snapshots[0][19],
                "curvature_per_m": command_snapshots[0][20],
            },
            "stop_correction_joint_names": correction_joint_names,
            "expected_stop_correction_joint_names": expected_correction_joint_names,
            "corrective_rewards_registered": all(
                name in unwrapped.reward_manager.active_terms
                for name in (
                    "stop_heading_tail", "stop_yaw_rate_tracking", "stop_attitude_stability",
                    "stop_instability_tail", "stop_hold_heading", "stop_parent_action_deviation",
                )
            ),
        }
        result["passed"] = (
            result["registered"]
            and result["observation_dim"] == POLICY_OBSERVATION_DIM
            and result["action_dim"] == 37
            and result["sequence_ids"] == result["expected_sequence_ids"]
            and all(abs(value - 1.0) < 1.0e-6 for value in result["one_hot_sums"])
            and 0.8 <= result["run_path_lookahead_local_xy"][0] <= 1.2
            and abs(result["run_path_state"]["signed_lateral_error_m"]) < 1.0e-4
            and abs(result["run_path_state"]["curvature_per_m"]) < 1.0e-6
            and result["stop_correction_joint_names"] == result["expected_stop_correction_joint_names"]
            and result["corrective_rewards_registered"]
        )
        output = (REPOSITORY_ROOT / args_cli.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        env.close()
        if not result["passed"]:
            raise RuntimeError("command environment smoke failed")


if __name__ == "__main__":
    main()
