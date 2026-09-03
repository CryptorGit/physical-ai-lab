"""Create a zero-CROUCH parent with a frozen, external standing-base actor."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path

import torch
from tensordict import TensorDict

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from g1_command_skills.models import G1CommandResidualActor  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402


def make_models() -> tuple[G1CommandResidualActor, MLPModel]:
    observations = TensorDict({"policy": torch.zeros(1, 152)}, batch_size=[1])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = G1CommandResidualActor(
        observations, groups, "actor", 37, hidden_dims=[256, 128, 128], activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[3], train_stop_correction=False,
    )
    critic = MLPModel(observations, groups, "critic", 1, hidden_dims=[256, 128, 128], activation="elu")
    return actor, critic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--standing-checkpoint", type=Path, required=True)
    parser.add_argument("--standing-candidate", required=True)
    parser.add_argument("--standing-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate_report = json.loads(args.standing_gate.resolve(strict=True).read_text(encoding="utf-8"))
    if not gate_report.get("eligible_for_crouch", False):
        raise RuntimeError("Standing-base gate is not eligible for CROUCH")
    if gate_report.get("candidate") != args.standing_candidate:
        raise RuntimeError("Standing candidate does not match the supplied gate")

    parent = torch.load(args.parent.resolve(strict=True), map_location="cpu", weights_only=False)
    standing = torch.load(args.standing_checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    actor, critic = make_models()
    fresh = actor.state_dict()
    actor.load_state_dict(parent["actor_state_dict"], strict=True)
    critic.load_state_dict(parent["critic_state_dict"], strict=True)
    state = actor.state_dict()
    standing_state = standing["actor_state_dict"]
    copied = []
    for key, value in standing_state.items():
        if not key.startswith("mlp."):
            continue
        target = f"stand_base_mlp.{key.removeprefix('mlp.')}"
        if target not in state or state[target].shape != value.shape:
            raise RuntimeError(f"Incompatible standing actor tensor: {key} -> {target}")
        state[target] = value.clone()
        copied.append(target)
    if len(copied) != 8:
        raise RuntimeError(f"Expected 8 standing MLP tensors, copied {len(copied)}")
    for prefix in ("skill_command_encoders.3.", "skill_state_adapters.3.", "residual_heads.3."):
        for key, value in fresh.items():
            if key.startswith(prefix):
                state[key] = value.clone()
    actor.load_state_dict(state, strict=True)
    with torch.no_grad():
        probe = TensorDict({"policy": torch.randn(8, 152)}, batch_size=[8])
        probe["policy"][:, 123:].zero_()
        probe["policy"][:, 123 + 3] = 1.0
        probe["policy"][:, 123 + 6 + 3] = 1.0
        probe["policy"][:, 123 + 25] = 1.0
        residual = actor.diagnostic_components(probe)["selected_residual"]
        if not torch.equal(residual, torch.zeros_like(residual)):
            raise RuntimeError("Fresh CROUCH residual is not bitwise zero")

    parent["actor_state_dict"] = actor.state_dict()
    parent["critic_state_dict"] = critic.state_dict()
    parent["iter"] = 0
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=5.0e-4)
    parent["optimizer_state_dict"] = optimizer.state_dict()
    infos = parent.get("infos") or {}
    parent["infos"] = infos
    infos["crouch_standing_option_rebase"] = {
        "parent": str(args.parent.resolve()),
        "standing_checkpoint": str(args.standing_checkpoint.resolve()),
        "standing_candidate": args.standing_candidate,
        "standing_gate": str(args.standing_gate.resolve()),
        "standing_base_frozen": True,
        "crouch_route_reinitialized": True,
        "crouch_initial_residual_bitwise_zero": True,
        "run_turn_stop_routes_preserved": True,
        "optimizer_reset": True,
        "iteration_reset": 0,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(parent, args.output.resolve())
    sidecar = args.output.resolve().with_suffix(args.output.suffix + ".standing_option.json")
    sidecar.write_text(json.dumps(infos["crouch_standing_option_rebase"], indent=2), encoding="utf-8")
    print(f"Created CROUCH standing-option parent: {args.output.resolve()}")


if __name__ == "__main__":
    main()
