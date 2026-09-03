"""Verify legacy RUN equivalence and bitwise freezing during TURN updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tensordict import TensorDict


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from g1_command_skills.models import G1CommandResidualActor  # noqa: E402


def linear_stack(state: dict[str, torch.Tensor], prefix: str, x: torch.Tensor, activate_last: bool) -> torch.Tensor:
    indices = sorted(
        int(key.removeprefix(prefix).split(".", 1)[0])
        for key in state
        if key.startswith(prefix) and key.endswith(".weight")
    )
    for position, index in enumerate(indices):
        x = F.linear(x, state[f"{prefix}{index}.weight"], state[f"{prefix}{index}.bias"])
        if position < len(indices) - 1 or activate_last:
            x = F.elu(x)
    return x


def legacy_run_action(state: dict[str, torch.Tensor], observation: torch.Tensor) -> torch.Tensor:
    legacy, command = observation[..., :123], observation[..., 123:]
    base = linear_stack(state, "base_mlp.", legacy, activate_last=False)
    command_code = linear_stack(state, "command_encoder.", command, activate_last=True)
    state_code = linear_stack(state, "state_adapter.", legacy, activate_last=True)
    run_residual = 0.25 * torch.tanh(
        linear_stack(state, "residual_heads.0.", torch.cat((state_code, command_code), dim=-1), False)
    )
    return base + run_residual


def make_actor() -> G1CommandResidualActor:
    obs = TensorDict({"policy": torch.zeros(1, 152)}, batch_size=[1])
    return G1CommandResidualActor(
        obs,
        {"actor": ["policy"]},
        "actor",
        37,
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[2],
    )


def command_observation(skill_id: int, angle: float = 0.0) -> torch.Tensor:
    torch.manual_seed(17)
    observation = torch.randn(8, 152) * 0.1
    command = observation[:, 123:]
    command.zero_()
    command[:, skill_id] = 1.0
    command[:, 6] = 1.0
    command[:, 25] = 1.0
    command[:, 12] = torch.sin(torch.tensor(angle))
    command[:, 13] = torch.cos(torch.tensor(angle))
    command[:, 14] = angle
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--candidate-checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    legacy_state = checkpoint["actor_state_dict"]
    if not any(key.startswith("command_encoder.") for key in legacy_state):
        raise ValueError("Smoke input must be the pre-migration RUN_BEST checkpoint")

    actor = make_actor()
    actor.load_state_dict(legacy_state, strict=True)
    run_obs = command_observation(0)
    turn_left = command_observation(2, torch.pi / 4)
    turn_right = command_observation(2, -torch.pi / 4)
    with torch.no_grad():
        legacy_action = legacy_run_action(legacy_state, run_obs)
        migrated_run_action = actor._mean_and_diagnostics(run_obs)[0]

    run_prefixes = ("skill_command_encoders.0.", "skill_state_adapters.0.", "residual_heads.0.")
    before = {
        name: value.detach().clone()
        for name, value in actor.state_dict().items()
        if name.startswith(run_prefixes)
    }
    before_action = migrated_run_action.detach().clone()
    optimizer = torch.optim.Adam((p for p in actor.parameters() if p.requires_grad), lr=3.0e-3)
    target = torch.linspace(-0.15, 0.15, 37).repeat(8, 1)
    for update in range(args.updates):
        optimizer.zero_grad()
        source = turn_left if update % 2 == 0 else turn_right
        action = actor._mean_and_diagnostics(source)[0]
        loss = (action - target).square().mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        after_action = actor._mean_and_diagnostics(run_obs)[0]
        left_residual = actor._mean_and_diagnostics(turn_left)[1]["residual_actions"][:, 2]
        right_residual = actor._mean_and_diagnostics(turn_right)[1]["residual_actions"][:, 2]
    unchanged = {
        name: bool(torch.equal(value, actor.state_dict()[name])) for name, value in before.items()
    }
    trainable_names = [name for name, parameter in actor.named_parameters() if parameter.requires_grad]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "updates": args.updates,
        "legacy_to_skill_local_run_action_bitwise_equal": bool(torch.equal(legacy_action, migrated_run_action)),
        "run_route_tensors_bitwise_unchanged": all(unchanged.values()),
        "run_route_tensor_count": len(unchanged),
        "run_action_before_after_turn_updates_bitwise_equal": bool(torch.equal(before_action, after_action)),
        "turn_command_residual_l2": float(torch.linalg.vector_norm(left_residual - right_residual, dim=-1).mean()),
        "turn_command_changes_residual": bool(
            torch.linalg.vector_norm(left_residual - right_residual, dim=-1).mean() > 1.0e-6
        ),
        "turn_residual_zero_does_not_enter_run_gate": bool(torch.equal(legacy_action, migrated_run_action)),
        "trainable_parameters_only_turn_route_or_std": all(
            name == "distribution.std_param"
            or name.startswith(("skill_command_encoders.2.", "skill_state_adapters.2.", "residual_heads.2."))
            for name in trainable_names
        ),
        "trainable_parameter_names": trainable_names,
    }
    if args.candidate_checkpoint:
        candidate_checkpoint = torch.load(
            args.candidate_checkpoint.resolve(strict=True), map_location="cpu", weights_only=False
        )
        candidate_actor = make_actor()
        candidate_actor.load_state_dict(candidate_checkpoint["actor_state_dict"], strict=True)
        with torch.no_grad():
            candidate_run_action = candidate_actor._mean_and_diagnostics(run_obs)[0]
        parent_expanded = actor._expand_legacy_shared_routes(legacy_state)
        candidate_state = candidate_actor.state_dict()
        run_route_names = [
            name for name in candidate_state if name.startswith(run_prefixes) or name.startswith("base_mlp.")
        ]
        report["rsl_candidate_checkpoint"] = str(args.candidate_checkpoint.resolve())
        report["rsl_run_tensors_bitwise_unchanged"] = all(
            torch.equal(parent_expanded[name], candidate_state[name]) for name in run_route_names
        )
        report["rsl_run_action_parent_candidate_bitwise_equal"] = bool(
            torch.equal(migrated_run_action, candidate_run_action)
        )
        report["rsl_turn_route_changed"] = any(
            not torch.equal(parent_expanded[name], candidate_state[name])
            for name in candidate_state
            if name.startswith(("skill_command_encoders.2.", "skill_state_adapters.2.", "residual_heads.2."))
        )
    report["passed"] = all(
        report[key]
        for key in (
            "legacy_to_skill_local_run_action_bitwise_equal",
            "run_route_tensors_bitwise_unchanged",
            "run_action_before_after_turn_updates_bitwise_equal",
            "turn_command_changes_residual",
            "turn_residual_zero_does_not_enter_run_gate",
            "trainable_parameters_only_turn_route_or_std",
        )
    )
    if args.candidate_checkpoint:
        report["passed"] = report["passed"] and all(
            report[key]
            for key in (
                "rsl_run_tensors_bitwise_unchanged",
                "rsl_run_action_parent_candidate_bitwise_equal",
                "rsl_turn_route_changed",
            )
        )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("skill-local route freeze smoke failed")


if __name__ == "__main__":
    main()
