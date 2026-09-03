"""Compare parent and TURN-updated actions on one identical RUN trajectory."""

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
parser.add_argument("--parent", required=True)
parser.add_argument("--candidate", required=True)
parser.add_argument("--output", required=True)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    task = "Isaac-Motion-Flat-G1-Command-Run-Eval-v0"
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    with launch_simulation(env_cfg, args_cli):
        raw_env = gym.make(task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        agent_cfg.device = raw_env.unwrapped.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        parent_runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        candidate_runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        actor_only = {"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False}
        parent_runner.load(str(Path(args_cli.parent).resolve(strict=True)), load_cfg=actor_only)
        candidate_runner.load(str(Path(args_cli.candidate).resolve(strict=True)), load_cfg=actor_only)
        parent_actor, candidate_actor = parent_runner.alg.actor, candidate_runner.alg.actor
        wrapped.reset()
        steps = 0
        max_abs_difference = 0.0
        all_bitwise_equal = True
        while True:
            observations = wrapped.get_observations()
            with torch.inference_mode():
                parent_action = parent_actor(observations)
                candidate_action = candidate_actor(observations)
                all_bitwise_equal &= bool(torch.equal(parent_action, candidate_action))
                max_abs_difference = max(
                    max_abs_difference, float((parent_action - candidate_action).abs().max().item())
                )
                _, _, dones, _ = wrapped.step(parent_action)
            steps += 1
            if bool(dones[0].item()):
                break
        report = {
            "task": task,
            "parent": str(Path(args_cli.parent).resolve()),
            "candidate": str(Path(args_cli.candidate).resolve()),
            "steps": steps,
            "run_actions_bitwise_equal_every_step": all_bitwise_equal,
            "maximum_absolute_action_difference": max_abs_difference,
            "passed": all_bitwise_equal and max_abs_difference == 0.0,
        }
        output = (REPOSITORY_ROOT / args_cli.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raw_env.close()
        if not report["passed"]:
            raise RuntimeError("RUN action equivalence smoke failed")


if __name__ == "__main__":
    main()
