"""Checkpoint metadata extraction without training or mutation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = payload.get("actor_state_dict", payload.get("model_state_dict", {}))
    weights = [(key, value) for key, value in actor.items() if key.endswith("weight") and value.ndim == 2]
    first, last = weights[0][1], weights[-1][1]
    return {
        "checkpoint_path": str(path),
        "sha256": sha256(path),
        "training_environment_id": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "training_command_range": {"lin_vel_x_mps": [-1.0, 1.0], "lin_vel_y_mps": [-1.0, 1.0], "yaw_rate_radps": [-1.0, 1.0]},
        "training_iteration": int(payload.get("iter", -1)),
        "network_architecture": [int(first.shape[1]), *[int(value.shape[0]) for _, value in weights]],
        "observation_dimension": int(first.shape[1]),
        "action_dimension": int(last.shape[0]),
        "action_distribution": "diagonal Gaussian; deterministic mean used for evaluation",
        "normalization_state": "none",
        "runner_config_path": "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py",
        "strict_load_result": "PENDING_LIVE_RUNNER_AUDIT",
        "provenance": "official Isaac Lab PretrainedCheckpoints/rsl_rl",
    }

