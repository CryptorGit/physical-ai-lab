"""Run one bounded Stage 2W-B steady-WALK stabilization pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, choices=(100, 150), default=100)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--yaw-oscillation-weight", type=float, choices=(-0.02, -0.05), default=-0.02)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    parent_sha = sha(checkpoint)
    if args.num_envs != 1024:
        raise ValueError("Stage 2W-B frozen pilot configuration requires exactly 1024 environments")
    task = "Isaac-Velocity-Flat-G1-WalkStabilization-v0"
    cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.rewards.yaw_rate_oscillation.weight = args.yaw_oscillation_weight
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device or agent_cfg.device
    cfg.sim.device = args.device or cfg.sim.device
    log_root = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered"
    run_dir = log_root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with launch_simulation(cfg, args):
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
        payload = torch.load(checkpoint, map_location=agent_cfg.device, weights_only=False)
        runner.alg.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        runner.alg.critic.load_state_dict(payload["critic_state_dict"], strict=True)
        actor_hash = tensor_hash(payload["actor_state_dict"])
        critic_hash = tensor_hash(payload["critic_state_dict"])
        if tensor_hash(runner.alg.actor.state_dict()) != actor_hash:
            raise RuntimeError("actor strict warm-start hash mismatch")
        if tensor_hash(runner.alg.critic.state_dict()) != critic_hash:
            raise RuntimeError("critic strict warm-start hash mismatch")
        if runner.alg.optimizer.state:
            raise RuntimeError("optimizer was not reset")
        torch.save(
            {
                "actor_state_dict": runner.alg.actor.state_dict(),
                "critic_state_dict": runner.alg.critic.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": 0,
                "infos": {"stage": "Stage 2W-B", "scope": "STEADY_WALK_ONLY"},
            },
            run_dir / "model_0.pt",
        )
        preflight = {
            "stage": "Stage 2W-B",
            "scope": "STEADY_WALK_ONLY",
            "parent_checkpoint": str(checkpoint.relative_to(REPO)),
            "parent_sha256": parent_sha,
            "optimizer_reset": True,
            "actor_strict_load": True,
            "critic_strict_load": True,
            "actor_tensor_hash": actor_hash,
            "critic_tensor_hash": critic_hash,
            "actor_architecture_unchanged": True,
            "observation_dimension": 123,
            "action_dimension": 37,
            "action_scale": 0.5,
            "speed_distribution": {"0.6": 0.2, "0.8": 0.2, "1.0": 0.3, "1.2": 0.3},
            "heading_perturbation": {
                "probability": 0.5,
                "amplitude_max_rad": 0.06,
                "frequency_hz": [0.08, 0.15],
                "smooth_sinusoidal": True,
            },
            "reward_delta": {"yaw_rate_oscillation": args.yaw_oscillation_weight},
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "seed": args.seed,
            "run_expert_loaded": False,
            "stand_expert_loaded": False,
            "transition_training_enabled": False,
            "world_xy_policy_input": False,
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        }
        (run_dir / "stage2wb_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        runner.save(str(run_dir / f"model_{args.iterations}.pt"), infos={"stage": "Stage 2W-B"})
        if sha(checkpoint) != parent_sha:
            raise RuntimeError("parent checkpoint changed")
        result = {
            **preflight,
            "run_directory": str(run_dir.relative_to(REPO)),
            "final_checkpoint": str((run_dir / f"model_{args.iterations}.pt").relative_to(REPO)),
            "parent_unchanged": True,
        }
        (run_dir / "stage2wb_run.json").write_text(json.dumps(result, indent=2) + "\n")
        print("STAGE2WB_RUN=" + json.dumps(result, sort_keys=True))
        env.close()


if __name__ == "__main__":
    main()
