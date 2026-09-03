"""Warm-start the frozen residual actor from an exp_005 Stage-4 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import chain
from pathlib import Path

import torch
from tensordict import TensorDict


OLD_OBSERVATIONS = 123
NEW_OBSERVATIONS = 152
ACTIONS = 37

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from g1_command_skills.models import G1CommandResidualActor  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_models() -> tuple[G1CommandResidualActor, MLPModel]:
    torch.manual_seed(0)
    observations = TensorDict({"policy": torch.zeros(1, NEW_OBSERVATIONS)}, batch_size=[1])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = G1CommandResidualActor(
        observations,
        groups,
        "actor",
        ACTIONS,
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[0],
    )
    critic = MLPModel(
        observations,
        groups,
        "critic",
        1,
        hidden_dims=[256, 128, 128],
        activation="elu",
    )
    return actor, critic


def copy_actor(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> None:
    source_base_keys = sorted(key for key in source if key.startswith("mlp."))
    for source_key in source_base_keys:
        target_key = "base_mlp." + source_key.removeprefix("mlp.")
        if target_key not in target or target[target_key].shape != source[source_key].shape:
            raise ValueError(f"Stage-4 actor mismatch: {source_key} -> {target_key}")
        target[target_key].copy_(source[source_key])
    for key, value in source.items():
        if key.startswith("distribution.") and key in target:
            if target[key].shape != value.shape:
                raise ValueError(f"Actor distribution mismatch at {key}")
            target[key].copy_(value)


def copy_critic(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> str:
    expanded_key = ""
    for key, value in source.items():
        if key not in target:
            raise ValueError(f"Unexpected Stage-4 critic key: {key}")
        if value.shape == target[key].shape:
            target[key].copy_(value)
        elif value.ndim == 2 and value.shape[1] == OLD_OBSERVATIONS and target[key].shape[1] == NEW_OBSERVATIONS:
            target[key].zero_()
            target[key][:, :OLD_OBSERVATIONS].copy_(value)
            expanded_key = key
        else:
            raise ValueError(f"Critic shape mismatch at {key}: {tuple(value.shape)} -> {tuple(target[key].shape)}")
    if not expanded_key:
        raise ValueError("Could not find the critic input layer to expand")
    return expanded_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if source == output:
        raise ValueError("Input and output checkpoint paths must differ")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    actor, critic = make_models()
    actor_state = actor.state_dict()
    critic_state = critic.state_dict()
    copy_actor(checkpoint["actor_state_dict"], actor_state)
    critic_input_key = copy_critic(checkpoint["critic_state_dict"], critic_state)
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)

    # Match RSL-RL's actor+critic optimizer parameter order so resume loading is valid.
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=1.0e-3)
    source_iteration = int(checkpoint.get("iter", -1))
    checkpoint["actor_state_dict"] = actor.state_dict()
    checkpoint["critic_state_dict"] = critic.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["iter"] = 0
    checkpoint["infos"] = {
        "transfer": {
            "source": str(source),
            "source_iteration": source_iteration,
            "architecture": "frozen_stage4_base_plus_skill_local_encoder_adapter_head",
            "base_actor_copy": "exact mlp.* -> base_mlp.*",
            "legacy_observation_columns": [0, OLD_OBSERVATIONS - 1],
            "command_observation_columns": [OLD_OBSERVATIONS, NEW_OBSERVATIONS - 1],
            "command_path_initialization": "independent seeded encoders/adapters; zero residual output layers",
            "initial_action_equivalence": "base action exactly; every residual is zero",
            "critic_expanded_layer": critic_input_key,
            "critic_new_columns_initialization": "zeros",
            "optimizer_reset": True,
            "iteration_reset": 0,
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    manifest = {
        "source": {"path": str(source), "sha256": sha256(source), "iteration": source_iteration},
        "output": {"path": str(output), "sha256": sha256(output), "iteration": 0},
        "architecture": "frozen_stage4_base_plus_skill_local_encoder_adapter_head",
        "base_actor_exact_copy": True,
        "residual_outputs_zero_initialized": list(range(6)),
        "critic_legacy_columns_exact_copy": [0, OLD_OBSERVATIONS - 1],
        "critic_new_command_columns_zero_initialized": [OLD_OBSERVATIONS, NEW_OBSERVATIONS - 1],
        "optimizer_reset": True,
        "critic_expanded_layer": critic_input_key,
    }
    output.with_suffix(".transfer.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
