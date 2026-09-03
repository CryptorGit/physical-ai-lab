"""Train one bounded Stage 3 STAND_TO_WALK transition pilot."""

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
parser.add_argument("--stand-checkpoint", required=True)
parser.add_argument("--walk-checkpoint", required=True)
parser.add_argument("--parent", required=True)
parser.add_argument("--num-envs", type=int, choices=(8, 1024), required=True)
parser.add_argument("--iterations", type=int, choices=(2, 100, 150), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--target-alignment-weight", type=float, choices=(-0.10, -0.20), default=-0.10)
parser.add_argument("--source-alignment-weight", type=float, choices=(-0.10, -0.20), default=-0.10)
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
    stand = Path(args.stand_checkpoint).resolve(strict=True)
    walk = Path(args.walk_checkpoint).resolve(strict=True)
    parent = Path(args.parent).resolve(strict=True)
    protected = {str(path): sha(path) for path in (stand, walk)}
    task = "Isaac-Velocity-Flat-G1-StandToWalk-v0"
    cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device or cfg.sim.device
    cfg.actions.joint_pos.stand_checkpoint_path = str(stand)
    cfg.actions.joint_pos.walk_checkpoint_path = str(walk)
    cfg.rewards.target_alignment.weight = args.target_alignment_weight
    cfg.rewards.source_alignment.weight = args.source_alignment_weight
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device or agent_cfg.device
    log_root = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered"
    run_dir = log_root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with launch_simulation(cfg, args):
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
        payload = torch.load(parent, map_location=agent_cfg.device, weights_only=False)
        runner.alg.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        runner.alg.critic.load_state_dict(payload["critic_state_dict"], strict=True)
        if runner.alg.optimizer.state:
            raise RuntimeError("transition optimizer was not reset")
        std_entries = {
            key: {
                "min": float(value.min()),
                "max": float(value.max()),
                "mean": float(value.mean()),
            }
            for key, value in payload["actor_state_dict"].items()
            if "std" in key.lower()
        }
        if std_entries and max(item["max"] for item in std_entries.values()) > 1.5:
            raise RuntimeError("parent exploration std exceeds the pre-frozen safety limit 1.5")
        torch.save(
            {
                "actor_state_dict": runner.alg.actor.state_dict(),
                "critic_state_dict": runner.alg.critic.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": 0,
                "infos": {"stage": "Stage 3", "edge": "STAND_TO_WALK"},
            },
            run_dir / "model_0.pt",
        )
        preflight = {
            "stage": "Stage 3",
            "edge": "STAND_TO_WALK",
            "parent": str(parent.relative_to(REPO)),
            "parent_sha256": sha(parent),
            "stand_checkpoint": str(stand.relative_to(REPO)),
            "walk_checkpoint": str(walk.relative_to(REPO)),
            "protected_hashes": protected,
            "actor_strict_load": True,
            "critic_strict_load": True,
            "optimizer_reset": True,
            "exploration_std": std_entries,
            "exploration_std_policy": "preserve strict-loaded parent; abort above 1.5",
            "actor_tensor_hash": tensor_hash(payload["actor_state_dict"]),
            "observation_dimension": 123,
            "action_dimension": 37,
            "action_scale": 0.5,
            "source_state_owner": "frozen_stage2_model_4246",
            "transition_action_owner": "trainable_transition_actor_only",
            "target_state_owner": "frozen_walk_model_100",
            "runtime_action_blend": False,
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "seed": args.seed,
            "target_alignment_weight": args.target_alignment_weight,
            "source_alignment_weight": args.source_alignment_weight,
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        }
        (run_dir / "stage3_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        final = run_dir / f"model_{args.iterations}.pt"
        runner.save(str(final), infos={"stage": "Stage 3", "edge": "STAND_TO_WALK"})
        for path, before in ((stand, protected[str(stand)]), (walk, protected[str(walk)])):
            if sha(path) != before:
                raise RuntimeError(f"protected checkpoint changed: {path}")
        result = {
            **preflight,
            "run_directory": str(run_dir.relative_to(REPO)),
            "final_checkpoint": str(final.relative_to(REPO)),
            "final_sha256": sha(final),
            "protected_experts_unchanged": True,
        }
        (run_dir / "stage3_run.json").write_text(json.dumps(result, indent=2) + "\n")
        print("STAGE3_RUN=" + json.dumps(result, sort_keys=True))
        env.close()


if __name__ == "__main__":
    main()
