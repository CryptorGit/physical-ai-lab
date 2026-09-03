"""Train one bounded Stage 2R curriculum phase from a strict warm start."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

EXPECTED_PARENT_SHA = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
TASKS = {
    "R0": "Isaac-Velocity-Flat-G1-WalkCentered-R0-v0",
    "R1": "Isaac-Velocity-Flat-G1-WalkCentered-R1-v0",
    "R2": "Isaac-Velocity-Flat-G1-WalkCentered-R2-v0",
    "R3": "Isaac-Velocity-Flat-G1-WalkCentered-R3-v0",
    "R4": "Isaac-Velocity-Flat-G1-WalkCentered-R4-v0",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=tuple(TASKS), required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, required=True)
parser.add_argument("--seed", type=int, default=20260725)
parser.add_argument("--run-name", required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha256(path: Path) -> str:
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
    before_hash = sha256(checkpoint)
    if args.phase == "R0" and before_hash != EXPECTED_PARENT_SHA:
        raise RuntimeError(f"R0 parent SHA mismatch: {before_hash}")
    if not 1 <= args.iterations <= 200:
        raise ValueError("A Stage 2R pilot is bounded to 1..200 iterations")
    if not 2 <= args.num_envs <= 1024:
        raise ValueError("Stage 2R uses 2..1024 environments")

    task = TASKS[args.phase]
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device or agent_cfg.device
    agent_cfg.max_iterations = args.iterations
    env_cfg.sim.device = args.device or env_cfg.sim.device
    log_root = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered"
    run_dir = log_root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with launch_simulation(env_cfg, args):
        raw_env = gym.make(task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parent_actor_hash = tensor_hash(loaded["actor_state_dict"])
        parent_critic_hash = tensor_hash(loaded["critic_state_dict"])
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
            strict=True,
            map_location=agent_cfg.device,
        )
        actor_hash = tensor_hash(runner.alg.actor.state_dict())
        critic_hash = tensor_hash(runner.alg.critic.state_dict())
        if actor_hash != parent_actor_hash or critic_hash != parent_critic_hash:
            raise RuntimeError("Strict warm-start verification failed")
        optimizer_reset = len(runner.alg.optimizer.state) == 0
        if not optimizer_reset:
            raise RuntimeError("Optimizer state was not reset")
        loaded_std = runner.alg.actor.state_dict()["distribution.std_param"].detach().cpu().clone()
        std_policy = "inherit"
        if before_hash == EXPECTED_PARENT_SHA and float(loaded_std.max()) > 0.5:
            with torch.no_grad():
                runner.alg.actor.distribution.std_param.fill_(0.25)
            std_policy = "strict_load_then_reset_trainable_std_to_0.25_due_parent_max_gt_0.5"
        std = runner.alg.actor.state_dict()["distribution.std_param"].detach().cpu()
        torch.save(
            {
                "actor_state_dict": runner.alg.actor.state_dict(),
                "critic_state_dict": runner.alg.critic.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": 0,
                "infos": {"stage2r_phase": args.phase},
            },
            run_dir / "model_0.pt",
        )
        preflight = {
            "stage": "Stage 2R",
            "phase": args.phase,
            "task": task,
            "parent_checkpoint": str(checkpoint.relative_to(REPO)),
            "parent_sha256": before_hash,
            "parent_actor_tensor_sha256": parent_actor_hash,
            "parent_critic_tensor_sha256": parent_critic_hash,
            "actor_strict_load": True,
            "critic_strict_load": True,
            "optimizer_loaded": False,
            "optimizer_reset": optimizer_reset,
            "iteration_loaded": False,
            "exploration_std_loaded_from_parent": True,
            "exploration_std_policy": std_policy,
            "parent_exploration_std_min": float(loaded_std.min()),
            "parent_exploration_std_mean": float(loaded_std.mean()),
            "parent_exploration_std_max": float(loaded_std.max()),
            "exploration_std_min": float(std.min()),
            "exploration_std_mean": float(std.mean()),
            "exploration_std_max": float(std.max()),
            "observation_dimension": int(loaded["actor_state_dict"]["mlp.0.weight"].shape[1]),
            "action_dimension": env.num_actions,
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "seed": args.seed,
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "run_expert_loaded": False,
            "transition_bridge_connected": False,
        }
        (run_dir / "warm_start_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        runner.save(str(run_dir / f"model_{args.iterations}.pt"), infos={"stage2r_phase": args.phase})
        after_hash = sha256(checkpoint)
        if after_hash != before_hash:
            raise RuntimeError("Protected parent checkpoint changed during training")
        result = {
            **preflight,
            "run_directory": str(run_dir.relative_to(REPO)),
            "final_checkpoint": str((run_dir / f"model_{args.iterations}.pt").relative_to(REPO)),
            "parent_sha256_after": after_hash,
            "parent_unchanged": True,
        }
        (run_dir / "stage2r_run.json").write_text(json.dumps(result, indent=2) + "\n")
        print("STAGE2R_RUN=" + json.dumps(result, sort_keys=True))
        env.close()


if __name__ == "__main__":
    main()
