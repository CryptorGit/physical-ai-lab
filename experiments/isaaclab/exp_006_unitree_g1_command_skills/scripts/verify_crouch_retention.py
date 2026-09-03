"""Prove frozen RUN/TURN/STOP routes are unchanged in a CROUCH checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from g1_command_skills.command_observation import (  # noqa: E402
    coherent_run_observation, coherent_stop_observation, coherent_turn_observation,
    coherent_crouch_observation, coherent_step_over_observation,
)
from g1_command_skills.models import G1CommandResidualActor  # noqa: E402


PROTECTED_PREFIXES = (
    "base_mlp.", "stand_base_mlp.",
    "skill_command_encoders.0.", "skill_command_encoders.1.", "skill_command_encoders.2.",
    "skill_state_adapters.0.", "skill_state_adapters.1.", "skill_state_adapters.2.",
    "residual_heads.0.", "residual_heads.1.", "residual_heads.2.",
)


def make_actor() -> G1CommandResidualActor:
    observation = TensorDict({"policy": torch.zeros(1, 152)}, batch_size=[1])
    return G1CommandResidualActor(
        observation, {"actor": ["policy"]}, "actor", 37,
        hidden_dims=[256, 128, 128], activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[3],
    )


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}|{tuple(value.shape)}|".encode()
    return hashlib.sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def combined_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if key.startswith(PROTECTED_PREFIXES):
            digest.update(key.encode())
            digest.update(tensor_hash(state[key]).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_path = args.reference.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    baseline_path = args.baseline_gate.resolve(strict=True)
    reference_data = torch.load(reference_path, map_location="cpu", weights_only=False)
    candidate_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reference_state = reference_data["actor_state_dict"]
    candidate_state = candidate_data["actor_state_dict"]
    protected_keys = sorted(key for key in reference_state if key.startswith(PROTECTED_PREFIXES))
    missing_keys = sorted(set(protected_keys) - set(candidate_state))
    mismatched = [
        key for key in protected_keys
        if key in candidate_state and not torch.equal(reference_state[key], candidate_state[key])
    ]
    tensor_hash_verified = not missing_keys and not mismatched

    reference_actor, candidate_actor = make_actor(), make_actor()
    reference_actor.load_state_dict(reference_state, strict=True)
    candidate_actor.load_state_dict(candidate_state, strict=True)
    torch.manual_seed(20260722)
    base = TensorDict({"policy": torch.randn(32, 152)}, batch_size=[32])
    variants = {
        "RUN": coherent_run_observation(base, speed_mps=2.2),
        "TURN_left_45": coherent_turn_observation(base, torch.pi / 4),
        "TURN_right_45": coherent_turn_observation(base, -torch.pi / 4),
        "TURN_left_90": coherent_turn_observation(base, torch.pi / 2),
        "TURN_right_90": coherent_turn_observation(base, -torch.pi / 2),
        "STOP": coherent_stop_observation(base, distance_m=1.75),
    }
    action_checks = {}
    with torch.no_grad():
        for name, observation in variants.items():
            reference_components = reference_actor.diagnostic_components(observation)
            candidate_components = candidate_actor.diagnostic_components(observation)
            old = reference_components["action_mean"]
            new = candidate_components["action_mean"]
            running_endpoint = torch.equal(
                candidate_components["selected_base_action"], candidate_components["running_base_action"]
            )
            stand_gate_zero = torch.equal(
                candidate_components["stand_base_gate"], torch.zeros_like(candidate_components["stand_base_gate"])
            )
            scripted_zero = torch.equal(
                candidate_components["scripted_crouch_offset"],
                torch.zeros_like(candidate_components["scripted_crouch_offset"]),
            )
            action_checks[name] = {
                "bitwise_equal": torch.equal(old, new),
                "max_abs_difference": float((old - new).abs().max()),
                "running_base_endpoint_bitwise": running_endpoint,
                "standing_base_gate_bitwise_zero": stand_gate_zero,
                "scripted_crouch_offset_bitwise_zero": scripted_zero,
            }
    action_equivalence_verified = all(
        check["bitwise_equal"] and check["running_base_endpoint_bitwise"]
        and check["standing_base_gate_bitwise_zero"] and check["scripted_crouch_offset_bitwise_zero"]
        for check in action_checks.values()
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    inherited = {name: baseline["metrics"][name] for name in ("run", "turn")}
    report = {
        "retention_source": "inherited_from_baseline_gate",
        "retention_basis": "bitwise_frozen_actor_route",
        "baseline_gate": str(baseline_path),
        "reference_checkpoint": str(reference_path),
        "checkpoint": str(checkpoint_path),
        "tensor_hash_verified": tensor_hash_verified,
        "action_equivalence_verified": action_equivalence_verified,
        "stop_action_immutability_verified": action_checks["STOP"]["bitwise_equal"],
        "running_base_endpoint_verified": all(v["running_base_endpoint_bitwise"] for v in action_checks.values()),
        "standing_or_crouch_route_leakage_detected": not all(
            v["standing_base_gate_bitwise_zero"] and v["scripted_crouch_offset_bitwise_zero"]
            for v in action_checks.values()
        ),
        "reference_protected_tensor_hash": combined_hash(reference_state),
        "candidate_protected_tensor_hash": combined_hash(candidate_state),
        "protected_tensor_count": len(protected_keys),
        "missing_protected_tensors": missing_keys,
        "mismatched_protected_tensors": mismatched,
        "action_checks": action_checks,
        "inherited_metrics": inherited,
    }
    report["verified"] = bool(
        tensor_hash_verified and action_equivalence_verified
        and report["stop_action_immutability_verified"]
        and report["running_base_endpoint_verified"]
        and not report["standing_or_crouch_route_leakage_detected"]
    )
    with torch.no_grad():
        step = candidate_actor.diagnostic_components(coherent_step_over_observation(base, lead_foot="left"))
        ref_crouch = reference_actor.diagnostic_components(coherent_crouch_observation(base, height_drop_m=0.09, phase=2.0))
        candidate_crouch = candidate_actor.diagnostic_components(coherent_crouch_observation(base, height_drop_m=0.09, phase=2.0))
    report["step_over_preflight"] = {
        "standing_base_selected_bitwise": torch.equal(step["selected_base_action"], step["standing_base_action"]),
        "running_base_mixed": not torch.equal(step["selected_base_action"], step["standing_base_action"]),
        "scripted_step_over_offset_bitwise_zero_guarded": torch.equal(step["scripted_step_over_offset"], torch.zeros_like(step["scripted_step_over_offset"])),
        "crouch_offset_bitwise_zero": torch.equal(step["scripted_crouch_offset"], torch.zeros_like(step["scripted_crouch_offset"])),
        "crouch_action_bitwise_unchanged": torch.equal(ref_crouch["action_mean"], candidate_crouch["action_mean"]),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
