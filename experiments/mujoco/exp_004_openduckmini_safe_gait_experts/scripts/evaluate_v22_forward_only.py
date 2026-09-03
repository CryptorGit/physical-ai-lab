"""Fresh, hardware-free forward-only V22 robustness evaluation.

This intentionally reuses the routed evaluator's production safety path rather
than the lightweight exp003 evaluator.  In particular, the physical forward
command is ``[+0.05, 0, 0]`` while the frozen V22 actor observes ``[+0.10, 0,
0]``; every target still goes through the contract's 50 mrad margin, 2 rad/s
slew guard, and control-first startup audit.

The result is a *forward-only* gate, never a substitute for the repository's
full primitive/compound/transition release qualification.  It opens no serial
or network hardware device.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import evaluate_routed_transitions as routed
from validate_v22_forward_no_fall_gate import validate as validate_forward_no_fall


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    EXPERIMENT_ROOT
    / "artifacts"
    / "single_policy_deployment_v1"
    / "models"
    / "base_v22.onnx"
)
DEFAULT_GENERATED_ROOT = EXPERIMENT_ROOT / "artifacts" / "generated_playground"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed V22 forward-only 20 episodes x 30 seconds gate "
            "through the hardware-safe routed runtime."
        )
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite immutable evidence: {args.output}")
    return args


def _forward_case() -> Any:
    matches = [case for case in routed.PRIMITIVE_CASES if case.name == "forward"]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one frozen forward primitive case")
    case = matches[0]
    if tuple(case.command) != (0.05, 0.0, 0.0):
        raise RuntimeError("frozen physical forward command drifted")
    if tuple(case.policy_observation_command or ()) != (0.1, 0.0, 0.0):
        raise RuntimeError("frozen forward policy-observation command drifted")
    if case.expected_expert != "forward" or case.expected_policy_role != "forward":
        raise RuntimeError("frozen forward routing contract drifted")
    return case


def _policy_paths(policy: Path) -> dict[str, Path]:
    resolved = policy.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing V22 policy: {resolved}")
    return {role: resolved for role in routed.REQUIRED_POLICY_ROLES}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy_paths = _policy_paths(args.policy)
    policy_provenance = routed.validate_policy_provenance(
        policy_paths, diagnostic_unadopted=False
    )
    if not policy_provenance["adoption_eligible"]:
        raise RuntimeError("forward gate requires the frozen base V22 policy")

    generated_root = args.generated_root.resolve()
    asset_evidence = routed.validate_exact_generated_assets(generated_root)
    asset_paths = routed.generated_asset_paths(generated_root)
    # Retain the central evaluator's validation of the historic fixed profiles,
    # then select the no-flag H3 adopted profile bank used by its formal path.
    routed.validate_adopted_reverse_profiles(
        routed.FORMAL_FIXED_BACKWARD_PROFILE,
        routed.FORMAL_FIXED_BACKWARD_LEFT_PROFILE,
        asset_paths["backward_right"],
    )
    selected_reverse_profiles = {
        "straight": routed.FORMAL_CANDIDATE_PROFILE_PATHS["straight"].resolve(),
        "left": routed.FORMAL_CANDIDATE_PROFILE_PATHS["left"].resolve(),
        "right": routed.FORMAL_CANDIDATE_PROFILE_PATHS["right"].resolve(),
    }
    reverse_profile_evidence = routed.validate_formal_candidate_reverse_profiles(
        selected_reverse_profiles["straight"],
        selected_reverse_profiles["left"],
        selected_reverse_profiles["right"],
    )
    mujoco, onnxruntime, runtime, runtime_provenance = routed._load_runtime(
        include_provenance=True
    )
    runtime_data_paths = routed._runtime_data_dependency_paths(
        policy_paths=policy_paths,
        generated_root=generated_root,
        asset_paths=asset_paths,
        asset_evidence=asset_evidence,
        selected_reverse_profiles=selected_reverse_profiles,
        include_phase_entry_evidence=False,
        include_backward_exit_recovery_evidence=False,
    )
    runtime_data_pre = routed.capture_runtime_source_dependency_closure(
        runtime_data_paths
    )

    bank = routed.RoutedPolicyBank(policy_paths, onnxruntime)
    evaluator = runtime.OfficialPolicyEvaluator(
        asset_paths["scene"], policy_paths["stand"], asset_paths["reference"]
    )
    evaluator.backward_residual_scale = 0.0
    evaluator.load_backward_profile(selected_reverse_profiles["straight"])
    evaluator.load_backward_turn_profile(
        1, selected_reverse_profiles["left"]
    )
    evaluator.load_backward_turn_profile(-1, selected_reverse_profiles["right"])
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    model_evidence = routed.validate_model_contract(evaluator)
    simulator = routed.RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=routed.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        target_slew_rate_rad_s=routed.RUNTIME_TARGET_SLEW_RATE_RAD_S,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=(
            reverse_profile_evidence["straight"]["composition"][
                "left_knee_extra_upper_margin_rad"
            ]
        ),
        diagnostic_reverse_entry_phase_indices=None,
        diagnostic_unadopted_backward_exit_recovery=False,
        formal_candidate_default=True,
    )

    forward = _forward_case()
    episodes = routed._independent_suite(
        simulator,
        [forward],
        seed_base=20260808,
        episodes=20,
        seconds=30.0,
        joint_noise_scale=1.0,
        initial_base_speed=0.10,
        warmup_seconds=1.5,
    )
    acceptance = routed.suite_acceptance(
        episodes,
        [forward.name],
        routed.AcceptanceThresholds(),
        require_gait_quality=True,
    )

    external_post = routed.validate_frozen_runtime_source_dependencies()
    own_post = routed.capture_runtime_source_dependency_closure(
        routed.OWN_RUNTIME_SOURCE_PATHS
    )
    binary_post = routed.capture_runtime_source_dependency_closure(
        routed._runtime_binary_dependency_paths(mujoco, onnxruntime),
        expected_sha256=routed.FROZEN_RUNTIME_BINARY_SHA256,
    )
    runtime_data_post = routed.capture_runtime_source_dependency_closure(
        runtime_data_paths
    )
    routed._require_runtime_closure_unchanged(
        "external runtime source closure",
        runtime_provenance["pre_import"]["external_hard_allowlisted_source_closure"],
        external_post,
    )
    routed._require_runtime_closure_unchanged(
        "exp004 source/contract closure",
        runtime_provenance["pre_import"]["exp004_source_and_contract_snapshot"],
        own_post,
    )
    routed._require_runtime_closure_unchanged(
        "runtime binary closure",
        runtime_provenance["pre_import"]["hard_allowlisted_runtime_binary_closure"],
        binary_post,
    )
    routed._require_runtime_closure_unchanged(
        "runtime model/data closure", runtime_data_pre, runtime_data_post
    )

    all_sessions_cpu_only = all(
        providers == ["CPUExecutionProvider"]
        for providers in bank.session_providers.values()
    )
    if not all_sessions_cpu_only:
        raise RuntimeError("forward gate requires CPU-only ONNX sessions")
    payload = {
        "schema_version": 1,
        "gate_id": "V22_FORWARD_ONLY_20X30_HARDWARE_SAFE_ROUTED_RUNTIME_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "forward_only": True,
            "not_a_substitute_for_full_motion_suite": True,
            "hardware_actuation": "NOT_ATTEMPTED",
        },
        "configuration": {
            "episodes": 20,
            "seconds": 30.0,
            "seed_base": 20260808,
            "initial_joint_noise_scale": 1.0,
            "initial_base_speed": 0.10,
            "warmup_seconds": 1.5,
            "physical_command": list(forward.command),
            "policy_observation_command": list(forward.policy_observation_command),
            "leg_target_margin_rad": routed.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            "target_slew_rate_rad_s": routed.RUNTIME_TARGET_SLEW_RATE_RAD_S,
            "backward_residual_scale": 0.0,
            "head_targets_locked_rad": 0.0,
        },
        "frozen_forward_case": asdict(forward),
        "policy_provenance": policy_provenance,
        "model_contract": model_evidence,
        "assets": asset_evidence,
        "reverse_profile_evidence": reverse_profile_evidence,
        "runtime_dependency_provenance": {
            **runtime_provenance,
            "post_evaluation": {
                "external_hard_allowlisted_source_closure": external_post,
                "exp004_source_and_contract_snapshot": own_post,
                "hard_allowlisted_runtime_binary_closure": binary_post,
                "runtime_model_and_data_closure": runtime_data_post,
            },
            "all_onnx_sessions_cpu_only_verified": all_sessions_cpu_only,
            "pre_post_source_and_data_hashes_unchanged": True,
        },
        "policies": bank.manifest(),
        "policy_inference_counts": dict(sorted(bank.inference_counts.items())),
        "episodes": episodes,
        # This is retained verbatim because it is the repository's stricter
        # gait-quality assessment, not the revised forward/no-fall decision.
        "strict_routed_acceptance": acceptance,
        # Kept as a compatibility alias for the immutable verdict validator.
        "acceptance": acceptance,
    }
    forward_no_fall_acceptance = validate_forward_no_fall(payload)
    payload["forward_no_fall_acceptance"] = forward_no_fall_acceptance
    payload["passed"] = bool(forward_no_fall_acceptance["passed"])
    _write_json(args.output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
