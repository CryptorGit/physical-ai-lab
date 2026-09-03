"""Audit one command dimension in an independent Isaac Lab process."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/stage0_parent_directional_baseline"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=("vx", "vy", "yaw", "gait"), required=True)
parser.add_argument("--checkpoint", required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

INDICES = {"vx": 9, "vy": 10, "yaw": 11, "gait": 123}
VALUES = {"vx": 0.731, "vy": -0.419, "yaw": 0.283, "gait": 1.0}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 2.0
    cfg.seed = 20261300 + list(INDICES).index(args.case)
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        command.external_override.zero_()
        wrapped.reset()
        before = wrapped.get_observations()["policy"].clone()
        before_124 = torch.cat((before, torch.zeros((1, 1), device=before.device)), dim=1)
        gait = torch.zeros(1, device=before.device)
        if args.case == "gait":
            gait.fill_(VALUES["gait"])
        else:
            command.external_override[0, list(INDICES).index(args.case)] = VALUES[args.case]
            # Apply the same command-term update that ManagerBasedEnv performs
            # before the next observation compute, without advancing physics.
            command._update_command()
        after = wrapped.get_observations()["policy"].clone()
        after_124 = torch.cat((after, gait[:, None]), dim=1)
        delta = (after_124 - before_124)[0]
        changed = torch.where(delta.abs() > 1.0e-7)[0].cpu().tolist()
        expected = INDICES[args.case]
        result = {
            "case": args.case,
            "fresh_process": True,
            "pid": __import__("os").getpid(),
            "seed": cfg.seed,
            "original_observation_dimension": int(before.shape[1]),
            "combined_input_dimension": int(after_124.shape[1]),
            "requested_value": VALUES[args.case],
            "changed_indices_zero_based": changed,
            "expected_index_zero_based": expected,
            "expected_value_after": float(after_124[0, expected]),
            "other_command_values_after": {
                name: float(after_124[0, index]) for name, index in INDICES.items() if name != args.case
            },
            "exclusive_change": changed == [expected],
            "checkpoint_sha256": sha256(args.checkpoint),
            "command_tensor": command.external_override[0].cpu().tolist(),
        }
        (OUT / f"_command_case_{args.case}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        wrapped.close()
        if not result["exclusive_change"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
