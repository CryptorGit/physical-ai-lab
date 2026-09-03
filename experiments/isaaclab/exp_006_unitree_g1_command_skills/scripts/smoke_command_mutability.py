"""Verify repeated reset, auto-reset, and mutable MotionCommand tensor types."""

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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="results/exp_006_unitree_g1_command_skills/smoke_command_mutability.json")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def inference_state(term) -> dict[str, bool]:
    return {
        name: bool(torch.is_inference(value))
        for name in term._MUTABLE_STATE_NAMES
        if isinstance((value := getattr(term, name, None)), torch.Tensor)
    }


def main() -> None:
    task = "Isaac-Motion-Flat-G1-Command-Stop-Eval-v0"
    cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    with launch_simulation(cfg, args_cli):
        raw_env = gym.make(task, cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        term = raw_env.unwrapped.command_manager.get_term("base_velocity")
        snapshots = []
        for reset_index in range(3):
            wrapped.reset()
            snapshots.append({"event": f"explicit_reset_{reset_index + 1}", "inference": inference_state(term)})

        auto_reset_seen = False
        for _ in range(round(float(cfg.episode_length_s) / float(raw_env.unwrapped.step_dt)) + 10):
            action = torch.zeros((1, raw_env.unwrapped.action_manager.total_action_dim), device=raw_env.unwrapped.device)
            _, _, dones, _ = wrapped.step(action)
            if bool(dones[0].item()):
                auto_reset_seen = True
                snapshots.append({"event": "auto_reset", "inference": inference_state(term)})
                break

        inference_tensors = sorted({
            name
            for snapshot in snapshots
            for name, is_inference in snapshot["inference"].items()
            if is_inference
        })
        result = {
            "task": task,
            "explicit_reset_count": 3,
            "auto_reset_seen": auto_reset_seen,
            "mutable_tensor_count": len(snapshots[0]["inference"]),
            "inference_tensor_names": inference_tensors,
            "all_mutable_tensors_normal": not inference_tensors,
            "snapshots": snapshots,
        }
        result["passed"] = auto_reset_seen and result["all_mutable_tensors_normal"]
        output = (REPOSITORY_ROOT / args_cli.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        raw_env.close()
        if not result["passed"]:
            raise RuntimeError("mutable command-state smoke failed")


if __name__ == "__main__":
    main()
