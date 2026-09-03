"""Static/action preflight for the CROUCH standing-base option checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from g1_command_skills.command_observation import (  # noqa: E402
    coherent_crouch_observation, coherent_run_observation, coherent_stop_observation, coherent_turn_observation,
)
from g1_command_skills.models import G1CommandResidualActor  # noqa: E402


def actor() -> G1CommandResidualActor:
    obs = TensorDict({"policy": torch.zeros(1, 152)}, batch_size=[1])
    return G1CommandResidualActor(
        obs, {"actor": ["policy"]}, "actor", 37, hidden_dims=[256, 128, 128], activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0}, trainable_skill_ids=[3],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--standing-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_data = torch.load(args.parent.resolve(strict=True), map_location="cpu", weights_only=False)
    option_data = torch.load(args.checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    standing_data = torch.load(args.standing_checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    parent, option = actor(), actor()
    parent.load_state_dict(parent_data["actor_state_dict"], strict=True)
    option.load_state_dict(option_data["actor_state_dict"], strict=True)
    torch.manual_seed(42)
    observation = TensorDict({"policy": torch.randn(32, 152)}, batch_size=[32])
    variants = {
        "RUN": coherent_run_observation(observation),
        "TURN": coherent_turn_observation(observation, 0.75),
        "STOP": coherent_stop_observation(observation),
        "CROUCH": coherent_crouch_observation(observation, height_drop_m=0.12),
    }
    action_checks = {}
    with torch.no_grad():
        for name in ("RUN", "TURN", "STOP"):
            old = parent.diagnostic_components(variants[name])["action_mean"]
            new = option.diagnostic_components(variants[name])["action_mean"]
            action_checks[name] = {"bitwise_equal": torch.equal(old, new), "max_abs_difference": float((old - new).abs().max())}
        crouch = option.diagnostic_components(variants["CROUCH"])
        crossfade = {}
        for progress in (0.0, 0.5, 1.0):
            variant = variants["CROUCH"].clone()
            variant["policy"][:, 123 + 20] = progress
            crossfade[progress] = option.diagnostic_components(variant)
    state = option.state_dict()
    standing_state = standing_data["actor_state_dict"]
    standing_weight_equal = all(
        torch.equal(state[f"stand_base_mlp.{key.removeprefix('mlp.')}"] , value)
        for key, value in standing_state.items() if key.startswith("mlp.")
    )
    protected_prefixes = (
        "base_mlp.", "skill_command_encoders.0.", "skill_command_encoders.1.",
        "skill_command_encoders.2.", "skill_state_adapters.0.", "skill_state_adapters.1.",
        "skill_state_adapters.2.", "residual_heads.0.", "residual_heads.1.", "residual_heads.2.",
        "stand_base_mlp.",
    )
    parent_state = parent.state_dict()
    protected_equal = all(
        key.startswith("stand_base_mlp.") or torch.equal(value, parent_state[key])
        for key, value in state.items() if key.startswith(protected_prefixes)
    )
    residual = crouch["selected_residual"]
    frozen_before = {
        key: value.clone() for key, value in option.state_dict().items()
        if key.startswith(("base_mlp.", "stand_base_mlp.", "skill_command_encoders.0.",
                           "skill_command_encoders.1.", "skill_command_encoders.2.",
                           "skill_state_adapters.0.", "skill_state_adapters.1.",
                           "skill_state_adapters.2.", "residual_heads.0.", "residual_heads.1.",
                           "residual_heads.2."))
    }
    optimizer = torch.optim.Adam((parameter for parameter in option.parameters() if parameter.requires_grad), lr=5.0e-4)
    target = torch.zeros_like(residual)
    target[:, [0, 1, 11, 12, 15, 16]] = torch.tensor([-0.05, -0.05, 0.08, 0.08, -0.03, -0.03])
    for _ in range(5):
        optimizer.zero_grad()
        current = option._mean_and_diagnostics(variants["CROUCH"]["policy"])[1]["selected_residual"]
        loss = (current - target).square().mean()
        loss.backward()
        optimizer.step()
    frozen_after = option.state_dict()
    frozen_after_updates = all(torch.equal(value, frozen_after[key]) for key, value in frozen_before.items())
    report = {
        "run_turn_stop_actions": action_checks,
        "run_turn_stop_bitwise_preserved": all(value["bitwise_equal"] for value in action_checks.values()),
        "protected_parent_tensors_equal": protected_equal,
        "standing_checkpoint_weights_equal": standing_weight_equal,
        "standing_parameters_frozen": all(not parameter.requires_grad for parameter in option.stand_base_mlp.parameters()),
        "standing_and_existing_routes_frozen_after_5_synthetic_updates": frozen_after_updates,
        "crouch_stand_base_gate_min": float(crouch["stand_base_gate"].min()),
        "crouch_stand_base_gate_max": float(crouch["stand_base_gate"].max()),
        "crouch_selected_base_equals_standing_bitwise": torch.equal(
            crouch["selected_base_action"], crouch["standing_base_action"]
        ),
        "crouch_selected_base_differs_from_running": bool(
            (crouch["selected_base_action"] - crouch["running_base_action"]).abs().max() > 0
        ),
        "crouch_initial_residual_bitwise_zero": torch.equal(residual, torch.zeros_like(residual)),
        "crossfade_endpoint_zero_is_running": torch.equal(
            crossfade[0.0]["selected_base_action"], crossfade[0.0]["running_base_action"]
        ),
        "crossfade_endpoint_one_is_standing": torch.equal(
            crossfade[1.0]["selected_base_action"], crossfade[1.0]["standing_base_action"]
        ),
        "crossfade_midpoint_correct": torch.allclose(
            crossfade[0.5]["selected_base_action"],
            0.5 * (crossfade[0.5]["running_base_action"] + crossfade[0.5]["standing_base_action"]),
            rtol=0.0, atol=1.0e-7,
        ),
        "pass": False,
    }
    report["pass"] = all((
        report["run_turn_stop_bitwise_preserved"], report["protected_parent_tensors_equal"],
        report["standing_checkpoint_weights_equal"], report["standing_parameters_frozen"],
        report["standing_and_existing_routes_frozen_after_5_synthetic_updates"],
        report["crouch_stand_base_gate_min"] == 1.0, report["crouch_stand_base_gate_max"] == 1.0,
        report["crouch_selected_base_equals_standing_bitwise"],
        report["crouch_selected_base_differs_from_running"], report["crouch_initial_residual_bitwise_zero"],
        report["crossfade_endpoint_zero_is_running"], report["crossfade_endpoint_one_is_standing"],
        report["crossfade_midpoint_correct"],
    ))
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
