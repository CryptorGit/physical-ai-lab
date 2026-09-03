"""Simulator preflight for CROUCH standing option, SETTLE, phases, and resets."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    task = "Isaac-Motion-Flat-G1-Command-Crouch-Eval-v0"
    cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    with launch_simulation(cfg, args_cli):
        raw = gym.make(task, cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = raw.unwrapped
        agent_cfg.device = env.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(str(args_cli.checkpoint.resolve(strict=True)), load_cfg={
            "actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False,
        })
        policy, actor = runner.get_inference_policy(device=env.device), runner.alg.actor
        repeated_reset = True
        for _ in range(3):
            try:
                wrapped.reset()
            except RuntimeError:
                repeated_reset = False
        observations = wrapped.get_observations()
        term = env.command_manager.get_term("base_velocity")
        robot = env.scene["robot"]
        phases = []
        rows = []
        entry_fixed_edges = 0
        was_fixed = bool(term.crouch_entry_height_fixed[0].item())
        base_exclusive = True
        stand_gate_values = []
        auto_reset = False
        for _ in range(int(cfg.episode_length_s / env.step_dt) + 20):
            with torch.inference_mode():
                components = actor.diagnostic_components(observations)
                actions = policy(observations)
            base_exclusive &= torch.equal(
                components["selected_base_action"], components["standing_base_action"]
            )
            stand_gate_values.append(float(components["stand_base_gate"][0, 0].item()))
            observations, _, dones, _ = wrapped.step(actions)
            fixed = bool(term.crouch_entry_height_fixed[0].item())
            if fixed and not was_fixed:
                entry_fixed_edges += 1
            was_fixed = fixed
            phase = int(term.crouch_phase[0].item())
            if not phases or phase != phases[-1]:
                phases.append(phase)
                rows.append({
                    "phase": phase, "time_s": float(term.segment_elapsed[0].item()),
                    "entry_height_fixed": fixed,
                    "entry_height_m": float(term.crouch_entry_height[0].item()),
                    "target_height_m": float(term.target_pelvis_height[0].item()),
                    "return_progress": float(term.crouch_return_progress[0].item()),
                    "base_crossfade_progress": float(term.crouch_base_transition_progress[0].item()),
                })
            if bool(dones[0].item()):
                auto_reset = True
                break
        mutable = [getattr(term, name) for name in term._MUTABLE_STATE_NAMES if isinstance(getattr(term, name, None), torch.Tensor)]
        action_indices = [0, 1, 11, 12, 15, 16]
        report = {
            "task_registered": task in gym.registry, "observation_dim": int(observations["policy"].shape[-1]),
            "repeated_reset_success": repeated_reset, "auto_reset_success": auto_reset,
            "mutable_tensors_are_normal": all(not torch.is_inference(tensor) for tensor in mutable),
            "phase_sequence": phases, "phase_rows": rows,
            "settle_success": any(row["phase"] >= 1 for row in rows),
            "entry_height_fixed_exactly_once": entry_fixed_edges == 1,
            "stand_base_gate_min": min(stand_gate_values), "stand_base_gate_max": max(stand_gate_values),
            "standing_base_exclusive": base_exclusive,
            "action_joint_names": [robot.joint_names[index] for index in action_indices],
        }
        report["return_target_is_entry_height"] = any(
            row["phase"] == 4 and abs(row["target_height_m"] - row["entry_height_m"]) <= 1.0e-7
            for row in rows
        )
        report["passed"] = all((
            report["task_registered"], report["observation_dim"] == 152,
            report["repeated_reset_success"], report["auto_reset_success"],
            report["mutable_tensors_are_normal"], report["phase_sequence"][:5] == [0, 1, 2, 3, 4],
            report["settle_success"], report["entry_height_fixed_exactly_once"],
            report["stand_base_gate_min"] == 1.0, report["stand_base_gate_max"] == 1.0,
            report["standing_base_exclusive"], report["return_target_is_entry_height"],
        ))
        args_cli.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        wrapped.close()
        if not report["passed"]:
            raise RuntimeError("CROUCH standing-option environment preflight failed")


if __name__ == "__main__":
    main()
