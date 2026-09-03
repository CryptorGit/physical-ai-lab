"""Preflight STOP route isolation, initialization, and braking-curve invariants."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_command_skills.models import G1CommandResidualActor  # noqa: E402


def make_actor(trainable: list[int]) -> G1CommandResidualActor:
    observations = TensorDict({"policy": torch.zeros(4, 152)}, batch_size=[4])
    return G1CommandResidualActor(
        observations,
        {"actor": ["policy"]},
        "actor",
        37,
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=trainable,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--expect-zero-stop-residual", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    actor = make_actor([1])
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    before = copy.deepcopy(actor.state_dict())
    route_obs = TensorDict({"policy": torch.randn(4, 152)}, batch_size=[4])
    route_actions_before = {}
    for skill_id, name in ((0, "RUN"), (2, "TURN")):
        route_obs["policy"][:, 123:] = 0.0
        route_obs["policy"][:, 123 + skill_id] = 1.0
        route_obs["policy"][:, 123 + 6 + skill_id] = 1.0
        route_obs["policy"][:, 123 + 25] = 1.0
        route_actions_before[name] = actor.diagnostic_components(route_obs)["action_mean"].clone()

    stop_parameters = [
        parameter
        for name, parameter in actor.named_parameters()
        if name.startswith(("skill_command_encoders.1.", "skill_state_adapters.1.", "residual_heads.1."))
    ]
    optimizer = torch.optim.SGD(stop_parameters, lr=1.0e-3)
    observations = TensorDict({"policy": torch.randn(4, 152)}, batch_size=[4])
    observations["policy"][:, 123:135] = 0.0
    observations["policy"][:, 123 + 1] = 1.0
    observations["policy"][:, 123 + 6 + 1] = 1.0
    observations["policy"][:, 123 + 25] = 1.0
    loss = actor(observations).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = actor.state_dict()
    route_action_max_differences = {}
    for skill_id, name in ((0, "RUN"), (2, "TURN")):
        route_obs["policy"][:, 123:] = 0.0
        route_obs["policy"][:, 123 + skill_id] = 1.0
        route_obs["policy"][:, 123 + 6 + skill_id] = 1.0
        route_obs["policy"][:, 123 + 25] = 1.0
        action_after = actor.diagnostic_components(route_obs)["action_mean"]
        route_action_max_differences[name] = float((route_actions_before[name] - action_after).abs().max())

    changed = [name for name in before if not torch.equal(before[name], after[name])]
    allowed_prefixes = ("skill_command_encoders.1.", "skill_state_adapters.1.", "residual_heads.1.")
    forbidden_changes = [name for name in changed if not name.startswith(allowed_prefixes)]

    fresh_actor = make_actor([1])
    fresh_stop_last = [
        value for name, value in fresh_actor.state_dict().items()
        if name.startswith("residual_heads.1.") and name.endswith(("weight", "bias"))
    ][-2:]
    fresh_stop_zero = all(torch.count_nonzero(value).item() == 0 for value in fresh_stop_last)

    audit_actor = make_actor([1])
    audit_actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    audit_obs = TensorDict({"policy": torch.randn(4, 152)}, batch_size=[4])
    audit_obs["policy"][:, 123:] = 0.0
    audit_obs["policy"][:, 123 + 1] = 1.0
    audit_obs["policy"][:, 123 + 6 + 1] = 1.0
    audit_obs["policy"][:, 123 + 25] = 1.0
    stop_residual_max = float(audit_actor.diagnostic_components(audit_obs)["selected_residual"].abs().max())

    reference_action_max_differences = {}
    if args.reference_checkpoint:
        reference_checkpoint = torch.load(
            args.reference_checkpoint.resolve(strict=True), map_location="cpu", weights_only=False
        )
        reference_actor = make_actor([])
        reference_actor.load_state_dict(reference_checkpoint["actor_state_dict"], strict=True)
        candidate_actor = make_actor([])
        candidate_actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        for skill_id, name in ((0, "RUN"), (2, "TURN")):
            route_obs["policy"][:, 123:] = 0.0
            route_obs["policy"][:, 123 + skill_id] = 1.0
            route_obs["policy"][:, 123 + 6 + skill_id] = 1.0
            route_obs["policy"][:, 123 + 25] = 1.0
            reference_action = reference_actor.diagnostic_components(route_obs)["action_mean"]
            candidate_action = candidate_actor.diagnostic_components(route_obs)["action_mean"]
            reference_action_max_differences[name] = float((reference_action - candidate_action).abs().max())

    entry_speed = torch.tensor([0.8, 1.0, 1.2, 1.4])
    distance = torch.tensor([1.5, 1.8, 2.2, 2.5])
    deceleration = (entry_speed.square() / (2.0 * (distance - 0.15))).clamp(0.2, 2.0)
    remaining = torch.stack([distance * fraction for fraction in (1.0, 0.75, 0.5, 0.25, 0.0)])
    target = torch.minimum(
        entry_speed.unsqueeze(0),
        torch.sqrt(2.0 * deceleration.unsqueeze(0) * torch.relu(remaining - 0.15)),
    )
    monotonic = bool(torch.all(target[1:] <= target[:-1] + 1.0e-7))
    finite = bool(torch.isfinite(target).all())

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "trainable_parameter_prefixes": list(allowed_prefixes),
        "changed_after_stop_step": changed,
        "forbidden_parameter_changes": forbidden_changes,
        "base_run_turn_routes_unchanged_bitwise": not forbidden_changes,
        "run_turn_action_max_abs_differences": route_action_max_differences,
        "run_turn_actions_unchanged_bitwise": all(value == 0.0 for value in route_action_max_differences.values()),
        "fresh_untrained_stop_residual_zero": fresh_stop_zero,
        "checkpoint_stop_residual_max_abs": stop_residual_max,
        "checkpoint_stop_residual_zero": stop_residual_max == 0.0,
        "zero_stop_residual_required": args.expect_zero_stop_residual,
        "reference_run_turn_action_max_abs_differences": reference_action_max_differences,
        "reference_run_turn_actions_unchanged_bitwise": all(
            value == 0.0 for value in reference_action_max_differences.values()
        ),
        "stage_a_braking_curve_monotonic": monotonic,
        "stage_a_braking_curve_finite": finite,
        "stage_a_entry_speeds_mps": entry_speed.tolist(),
        "stage_a_stopping_distances_m": distance.tolist(),
        "stage_a_braking_targets_mps": target.tolist(),
    }
    result["passed"] = bool(
        result["base_run_turn_routes_unchanged_bitwise"]
        and result["run_turn_actions_unchanged_bitwise"]
        and fresh_stop_zero
        and (result["checkpoint_stop_residual_zero"] or not args.expect_zero_stop_residual)
        and result["reference_run_turn_actions_unchanged_bitwise"]
        and monotonic
        and finite
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("STOP pilot preflight failed")


if __name__ == "__main__":
    main()
