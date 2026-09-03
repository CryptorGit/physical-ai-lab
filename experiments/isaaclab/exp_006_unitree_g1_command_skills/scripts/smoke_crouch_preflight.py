"""Pure-torch CROUCH route initialization and five-update freeze audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from g1_command_skills.models import G1CommandResidualActor  # noqa: E402


def make_actor(trainable: bool) -> G1CommandResidualActor:
    obs = TensorDict({"policy": torch.zeros(1, 152)}, batch_size=[1])
    return G1CommandResidualActor(
        obs, {"actor": ["policy"]}, "actor", 37,
        hidden_dims=[256, 128, 128], activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[3] if trainable else [], train_stop_correction=False,
    )


def skill_obs(skill: int, batch: int = 8) -> TensorDict:
    generator = torch.Generator().manual_seed(6100 + skill)
    policy = 0.1 * torch.randn(batch, 152, generator=generator)
    policy[:, 123:].zero_()
    policy[:, 123 + skill] = 1.0
    policy[:, 123 + 6 + skill] = 1.0
    policy[:, 123 + 25] = 1.0
    if skill == 3:
        policy[:, 123 + 16] = -0.12
    return TensorDict({"policy": policy}, batch_size=[batch])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    parent_checkpoint_state = None
    if args.parent_checkpoint:
        parent_checkpoint_state = torch.load(
            args.parent_checkpoint.resolve(strict=True), map_location="cpu", weights_only=False
        )["actor_state_dict"]
    source_keys_preserved = parent_checkpoint_state is None or (
        all(name in state and torch.equal(value, state[name]) for name, value in parent_checkpoint_state.items())
    )
    added_keys = [] if parent_checkpoint_state is None else [name for name in state if name not in parent_checkpoint_state]
    added_compatibility_only = all(
        name.startswith("stop_corrective_") or name == "stop_correction_scale" for name in added_keys
    )
    parent, actor = make_actor(False), make_actor(True)
    parent.load_state_dict(state, strict=True)
    actor.load_state_dict(state, strict=True)
    observations = {name: skill_obs(skill) for name, skill in (("run", 0), ("stop", 1), ("turn", 2), ("crouch", 3))}
    with torch.no_grad():
        parent_actions = {name: parent.diagnostic_components(obs)["action_mean"] for name, obs in observations.items()}
        before = {name: actor.diagnostic_components(obs) for name, obs in observations.items()}
    state_before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    trainable_names = [name for name, parameter in actor.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam((p for p in actor.parameters() if p.requires_grad), lr=5.0e-4)
    target = torch.zeros(8, 37)
    target[:, [0, 1, 11, 12, 15, 16]] = torch.tensor([-0.08, -0.08, 0.12, 0.12, -0.04, -0.04])
    for _ in range(args.updates):
        optimizer.zero_grad()
        residual = actor._mean_and_diagnostics(observations["crouch"]["policy"])[1]["selected_residual"]
        loss = (residual - target).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = {name: actor.diagnostic_components(obs) for name, obs in observations.items()}
    state_after = actor.state_dict()
    changed = [name for name, value in state_before.items() if not torch.equal(value, state_after[name])]
    allowed = ("skill_command_encoders.3.", "skill_state_adapters.3.", "residual_heads.3.")
    scale = actor.crouch_action_scale
    non_target = [index for index in range(37) if index not in actor.crouch_action_indices]
    result = {
        "checkpoint": str(args.checkpoint.resolve()), "updates": args.updates,
        "initial_crouch_output_exact_zero": bool(torch.count_nonzero(before["crouch"]["selected_residual"]) == 0),
        "initial_action_matches_parent_bitwise": all(torch.equal(before[name]["action_mean"], parent_actions[name]) for name in observations),
        "source_actor_tensors_preserved_bitwise": source_keys_preserved,
        "added_frozen_compatibility_tensors_only": added_compatibility_only,
        "added_compatibility_tensors": added_keys,
        "run_action_bitwise_frozen": torch.equal(before["run"]["action_mean"], after["run"]["action_mean"]),
        "turn_action_bitwise_frozen": torch.equal(before["turn"]["action_mean"], after["turn"]["action_mean"]),
        "stop_action_bitwise_frozen": torch.equal(before["stop"]["action_mean"], after["stop"]["action_mean"]),
        "base_and_non_crouch_actor_tensors_bitwise_frozen": all(name.startswith(allowed) for name in changed),
        "crouch_route_changed_after_5_updates": bool(changed),
        "changed_tensors": changed,
        "trainable_parameters_only_crouch_or_std": all(name == "distribution.std_param" or name.startswith(allowed) for name in trainable_names),
        "trainable_parameter_names": trainable_names,
        "crouch_action_indices": list(actor.crouch_action_indices),
        "crouch_action_limits": [float(scale[index]) for index in actor.crouch_action_indices],
        "non_target_action_limits_exact_zero": bool(torch.count_nonzero(scale[non_target]) == 0),
        "maximum_residual_limit": float(scale.max()),
    }
    result["passed"] = all(
        result[key] for key in (
            "initial_crouch_output_exact_zero", "initial_action_matches_parent_bitwise",
            "source_actor_tensors_preserved_bitwise", "added_frozen_compatibility_tensors_only",
            "run_action_bitwise_frozen", "turn_action_bitwise_frozen", "stop_action_bitwise_frozen",
            "base_and_non_crouch_actor_tensors_bitwise_frozen", "crouch_route_changed_after_5_updates",
            "trainable_parameters_only_crouch_or_std", "non_target_action_limits_exact_zero",
        )
    ) and result["maximum_residual_limit"] <= 0.25
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("CROUCH preflight failed")


if __name__ == "__main__":
    main()
