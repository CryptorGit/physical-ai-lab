"""Preflight the frozen-model_31 STOP corrective architecture without simulation."""

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


def make_actor() -> G1CommandResidualActor:
    observations = TensorDict({"policy": torch.zeros(1, 152)}, batch_size=[1])
    return G1CommandResidualActor(
        observations,
        {"actor": ["policy"]},
        "actor",
        37,
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[],
        train_stop_correction=True,
    )


def skill_observation(skill_id: int, batch_size: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3100 + skill_id)
    observation = 0.1 * torch.randn(batch_size, 152, generator=generator)
    command = observation[:, 123:]
    command.zero_()
    command[:, skill_id] = 1.0
    command[:, 6 + skill_id] = 1.0
    command[:, 25] = 1.0
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    parent_state = checkpoint["actor_state_dict"]
    actor = make_actor()
    actor.load_state_dict(parent_state, strict=True)

    observations = {name: skill_observation(skill_id) for name, skill_id in (("run", 0), ("stop", 1), ("turn", 2))}
    with torch.no_grad():
        initial = {name: actor._mean_and_diagnostics(obs)[1] for name, obs in observations.items()}

    parent_prefixes = ("base_mlp.", "skill_command_encoders.", "skill_state_adapters.", "residual_heads.")
    parent_before = {
        name: value.detach().clone()
        for name, value in actor.state_dict().items()
        if name.startswith(parent_prefixes)
    }
    corrective_before = {
        name: value.detach().clone()
        for name, value in actor.state_dict().items()
        if name.startswith("stop_corrective_")
    }
    action_before = {name: values["action_mean"].clone() for name, values in initial.items()}
    stop_parent_before = initial["stop"]["parent_action_mean"].clone()

    trainable_names = [name for name, parameter in actor.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam((parameter for parameter in actor.parameters() if parameter.requires_grad), lr=2.5e-4)
    target = torch.zeros(8, 37)
    target[:, list(actor.stop_correction_action_indices)] = torch.tensor(
        [0.010, -0.010, 0.008, -0.008, 0.006, -0.006, 0.005, -0.004, 0.004]
    )
    for _ in range(args.updates):
        optimizer.zero_grad()
        correction = actor._mean_and_diagnostics(observations["stop"])[1]["selected_stop_correction"]
        loss = (correction - target).square().mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        after = {name: actor._mean_and_diagnostics(obs)[1] for name, obs in observations.items()}
    state_after = actor.state_dict()
    parent_unchanged = all(torch.equal(value, state_after[name]) for name, value in parent_before.items())
    changed = [name for name, value in corrective_before.items() if not torch.equal(value, state_after[name])]

    limit_actor = make_actor()
    limit_actor.load_state_dict(parent_state, strict=True)
    final_linear = [module for module in limit_actor.stop_corrective_head.modules() if isinstance(module, torch.nn.Linear)][-1]
    with torch.no_grad():
        final_linear.bias.fill_(100.0)
        saturated = limit_actor._mean_and_diagnostics(observations["stop"])[1]["selected_stop_correction"]

    scale = actor.stop_correction_scale
    initial_stop_correction = initial["stop"]["selected_stop_correction"]
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "updates": args.updates,
        "initial_corrective_output_exact_zero": bool(torch.count_nonzero(initial_stop_correction) == 0),
        "initial_stop_action_matches_model31_parent_bitwise": bool(
            torch.equal(initial["stop"]["action_mean"], initial["stop"]["parent_action_mean"])
        ),
        "model31_stop_parent_action_unchanged_after_updates": bool(
            torch.equal(stop_parent_before, after["stop"]["parent_action_mean"])
        ),
        "model31_parent_route_tensors_bitwise_frozen": parent_unchanged,
        "run_action_bitwise_frozen": bool(torch.equal(action_before["run"], after["run"]["action_mean"])),
        "turn_action_bitwise_frozen": bool(torch.equal(action_before["turn"], after["turn"]["action_mean"])),
        "non_target_correction_scales_exact_zero": bool(
            torch.count_nonzero(scale[[index for index in range(37) if index not in actor.stop_correction_action_indices]]) == 0
        ),
        "configured_correction_max_abs": float(scale.max()),
        "saturated_correction_max_abs": float(saturated.abs().max()),
        "hip_pitch_limit": float(scale[0]),
        "yaw_roll_limit": float(scale[2]),
        "corrective_route_changed_after_5_updates": bool(changed),
        "changed_corrective_tensors": changed,
        "trainable_parameters_only_corrective_or_std": all(
            name == "distribution.std_param" or name.startswith("stop_corrective_") for name in trainable_names
        ),
        "trainable_parameter_names": trainable_names,
        "parent_action_deviation_logged": "parent_action_deviation" in initial["stop"],
        "corrective_residual_logged": "selected_stop_correction" in initial["stop"],
    }
    result["passed"] = all(
        result[key]
        for key in (
            "initial_corrective_output_exact_zero",
            "initial_stop_action_matches_model31_parent_bitwise",
            "model31_stop_parent_action_unchanged_after_updates",
            "model31_parent_route_tensors_bitwise_frozen",
            "run_action_bitwise_frozen",
            "turn_action_bitwise_frozen",
            "non_target_correction_scales_exact_zero",
            "corrective_route_changed_after_5_updates",
            "trainable_parameters_only_corrective_or_std",
            "parent_action_deviation_logged",
            "corrective_residual_logged",
        )
    ) and result["configured_correction_max_abs"] <= 0.03 and result["saturated_correction_max_abs"] <= 0.03

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("STOP corrective preflight failed")


if __name__ == "__main__":
    main()
