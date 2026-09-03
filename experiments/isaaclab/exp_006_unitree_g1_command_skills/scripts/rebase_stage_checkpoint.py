"""Reset optimizer and iteration at a skill boundary without changing model weights."""

from __future__ import annotations

import argparse
import sys
from itertools import chain
from pathlib import Path

import torch
from tensordict import TensorDict


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from g1_command_skills.models import G1CommandResidualActor  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402


def make_models(stage: str) -> tuple[G1CommandResidualActor, MLPModel]:
    observations = TensorDict({"policy": torch.zeros(1, 152)}, batch_size=[1])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    trainable = {"run": [0], "turn": [2], "stop": [], "crouch": [3], "sequence": []}[stage]
    actor = G1CommandResidualActor(
        observations,
        groups,
        "actor",
        37,
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=trainable,
        train_stop_correction=stage == "stop",
    )
    critic = MLPModel(observations, groups, "critic", 1, hidden_dims=[256, 128, 128], activation="elu")
    return actor, critic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.input.resolve(strict=True), map_location="cpu", weights_only=False)
    source_iteration = int(checkpoint.get("iter", -1))
    legacy_shared_routes = any(key.startswith("command_encoder.") for key in checkpoint["actor_state_dict"])
    actor, critic = make_models(args.stage)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    checkpoint["actor_state_dict"] = actor.state_dict()
    checkpoint["critic_state_dict"] = critic.state_dict()
    checkpoint["iter"] = 0
    # Architecture migration changes the optimizer parameter list.  Rebuild it
    # instead of retaining stale parameter ids from the shared-route actor.
    learning_rate = 2.5e-4 if args.stage == "stop" else 5.0e-4 if args.stage == "crouch" else 1.0e-3
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=learning_rate)
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    infos = checkpoint.get("infos") or {}
    checkpoint["infos"] = infos
    infos["stage_rebase"] = {
        "source": str(args.input.resolve()),
        "source_iteration": source_iteration,
        "target_stage": args.stage,
        "policy_function_preserved": True,
        "architecture": (
            "frozen_model31_parent_plus_bounded_stop_correction"
            if args.stage == "stop"
            else "frozen_base_plus_skill_local_encoder_adapter_head"
        ),
        "legacy_shared_routes_expanded": legacy_shared_routes,
        "optimizer_reset": True,
        "learning_rate": learning_rate,
        "iteration_reset": 0,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output.resolve())
    print(f"Rebased {args.input.resolve()} -> {args.output.resolve()} for stage {args.stage}")


if __name__ == "__main__":
    main()
