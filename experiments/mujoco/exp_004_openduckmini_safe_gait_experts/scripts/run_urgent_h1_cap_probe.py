"""One-candidate launcher for the H1 left-knee diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import diagnose_h1_left_knee_cap_sweep as diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=float, default=0.0175)
    parser.add_argument("--hold", type=int, default=13)
    parser.add_argument("--all20", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite artifact: {args.output}")
    launcher_path = Path(__file__).resolve()
    source_paths = tuple(
        dict.fromkeys((*diagnostic._source_paths(), launcher_path))
    )
    source_closure_pre = diagnostic._closure(source_paths)
    assets = diagnostic.central.generated_asset_paths(
        diagnostic.central.DEFAULT_GENERATED_ROOT.resolve()
    )
    asset_evidence = diagnostic.routed_contract.validate_exact_generated_assets(
        diagnostic.central.DEFAULT_GENERATED_ROOT.resolve()
    )
    mujoco, onnxruntime, runtime, runtime_provenance = diagnostic.central._load_runtime(
        include_provenance=True
    )
    policies = {
        role: diagnostic.BASE_POLICY.resolve()
        for role in diagnostic.routed_contract.REQUIRED_POLICY_ROLES
    }
    bank = diagnostic.central.RoutedPolicyBank(policies, onnxruntime)
    evaluator = runtime.OfficialPolicyEvaluator(
        assets["scene"], diagnostic.BASE_POLICY.resolve(), assets["reference"]
    )
    evaluator.backward_residual_scale = 0.0
    evaluator.load_backward_profile(diagnostic.PROFILE_PATHS["straight"])
    evaluator.load_backward_turn_profile(1, diagnostic.PROFILE_PATHS["left"])
    evaluator.load_backward_turn_profile(-1, diagnostic.PROFILE_PATHS["right"])
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    candidate = diagnostic.CapCandidate(args.cap, args.hold)
    seeds = (
        diagnostic.FORMAL_TRANSITION_SEEDS
        if args.all20
        else diagnostic.FOCUS_TRANSITION_SEEDS
    )
    result = diagnostic._run_candidate(
        candidate,
        seeds,
        evaluator=evaluator,
        bank=bank,
        mujoco=mujoco,
        runtime=runtime,
        seed_scope=(
            "all20_formal_transition_prefix"
            if args.all20
            else "focus5_integrated_selection"
        ),
    )
    keys = (
        "passed",
        "passed_episode_count",
        "segment_count",
        "passed_segment_count",
        "fall_count",
        "qpos_limit_violation_samples",
        "maximum_qpos_excess_rad",
        "maximum_left_knee_qpos_rad",
        "minimum_left_knee_safe_upper_margin_rad",
        "expected_physics_substeps",
        "audited_physics_substeps",
        "target_limit_margin_slew_violation_count",
        "maximum_head_action_or_target_peak",
        "route_violation_segment_count",
        "motion_contact_violation_segment_count",
    )
    failures = []
    for episode in result["episodes"]:
        for segment in episode["segments"]:
            if segment["passed"]:
                continue
            failures.append(
                {
                    "seed": episode["seed"],
                    "segment": segment["name"],
                    "fell": segment["fell"],
                    "completed": segment["completed"],
                    "false_hard_checks": sorted(
                        key
                        for key, passed in segment["hard_checks"].items()
                        if not passed
                    ),
                    "false_standard_checks": sorted(
                        key
                        for key, passed in segment["standard_acceptance"][
                            "checks"
                        ].items()
                        if not passed
                    ),
                    "qpos_limit_violations": segment["physics_substep_audit"][
                        "qpos_limit_violations"
                    ],
                    "left_knee_max_qpos_rad": segment[
                        "physics_substep_audit"
                    ]["joint_qpos_max_rad"]["left_knee"],
                    "minimum_height_m": segment["physics_substep_audit"][
                        "minimum_height_m"
                    ],
                    "minimum_upright": segment["physics_substep_audit"][
                        "minimum_upright"
                    ],
                }
            )
    summary = {key: result[key] for key in keys}
    summary["cap"] = args.cap
    summary["hold"] = args.hold
    summary["failures"] = failures
    if args.output is not None:
        source_closure_post = diagnostic._closure(source_paths)
        exact_scale = bool(
            args.all20
            and result["episode_count"] == 20
            and result["segment_count"] == 180
            and result["expected_physics_substeps"] == 1_450_000
            and result["completed_physics_substeps"] == 1_450_000
            and result["audited_physics_substeps"] == 1_450_000
            and result["contact_samples"] == 1_450_000
        )
        passed = bool(
            result["passed"]
            and exact_scale
            and source_closure_pre == source_closure_post
        )
        output = args.output.resolve()
        payload = {
            "schema_version": 1,
            "artifact_kind": (
                "openduckmini_h2_integrated_straight_and_exit_recovery_"
                "transition_qualification"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                "DIAGNOSTIC_20X9_PASS_NOT_ADOPTED"
                if passed
                else "DIAGNOSTIC_20X9_FAIL"
            ),
            "simulation_only": True,
            "adoption_status": "NOT_ADOPTED_PENDING_CENTRAL_INTEGRATION",
            "hardware_deployment": "PROHIBITED",
            "central_contract_package_or_runtime_modified_by_this_run": False,
            "configuration": {
                "formal_master_seed": diagnostic.FORMAL_MASTER_SEED,
                "transition_seeds": list(seeds),
                "moving_seconds": diagnostic.MOVING_SECONDS,
                "stand_seconds": diagnostic.STAND_SECONDS,
                "warmup_seconds": diagnostic.WARMUP_SECONDS,
                "initial_joint_noise_scale": (
                    diagnostic.INITIAL_JOINT_NOISE_SCALE
                ),
                "initial_base_speed_mps": diagnostic.INITIAL_BASE_SPEED_MPS,
                "reverse_endpoint_mps": -0.05,
                "phase_preincrement_indices": diagnostic.PHASE_ENTRY_INDICES,
                "straight_phase_rate_rad_per_control_tick": (
                    evaluator.backward_phase_rate
                ),
                "profile_extra_upper_margin_rad_all_backward_families": (
                    candidate.profile_extra_upper_margin_rad
                ),
                "profile_left_knee_upper_target_rad": (
                    candidate.profile_upper_target_rad
                ),
                "exit_recovery_extra_upper_margin_rad": (
                    candidate.recovery_extra_upper_margin_rad
                ),
                "exit_recovery_left_knee_upper_target_rad": (
                    candidate.recovery_upper_target_rad
                ),
                "exit_recovery_hold_control_ticks": (
                    candidate.recovery_hold_ticks
                ),
                "exit_recovery_hold_seconds": candidate.hold_seconds,
                "exit_recovery_release": "instant_after_hold",
                "target_margin_rad": (
                    diagnostic.central.RUNTIME_TARGET_SAFETY_MARGIN_RAD
                ),
                "target_slew_rate_rad_per_s": (
                    diagnostic.central.RUNTIME_TARGET_SLEW_RATE_RAD_S
                ),
                "schedule": [
                    {
                        "name": case[0],
                        "physical_command": list(case[1]),
                        "seconds": case[2],
                        "policy_observation_command": (
                            None if case[3] is None else list(case[3])
                        ),
                        "expected_expert": case[4],
                        "expected_policy_role": case[5],
                    }
                    for case in diagnostic.exact_formal_transition_prefix()
                ],
            },
            "selection": {
                "selected_candidate": candidate.contract(),
                "rule": (
                    "keep all backward profile caps at 0.0125; minimize the "
                    "exit-only recovery cap, then hold ticks, subject to every "
                    "substep and central motion/contact/route/target gate"
                ),
                "focus5_comparisons": [
                    {
                        "recovery_extra_upper_margin_rad": 0.01625,
                        "hold_ticks": 13,
                        "passed": False,
                        "failed_seed": 22_260_808,
                        "failed_segment": (
                            "transition_stand_after_reverse_turn_left"
                        ),
                        "qpos_limit_violation_samples": 4,
                        "maximum_qpos_excess_rad": 0.00037463858229119795,
                    },
                    {
                        "recovery_extra_upper_margin_rad": 0.01625,
                        "hold_ticks": 16,
                        "passed": False,
                        "failed_seed": 22_260_808,
                        "failed_segment": (
                            "transition_stand_after_reverse_turn_left"
                        ),
                        "qpos_limit_violation_samples": 4,
                        "maximum_qpos_excess_rad": 0.00032382746548992314,
                    },
                    {
                        "recovery_extra_upper_margin_rad": 0.0175,
                        "hold_ticks": 13,
                        "passed": True,
                        "segment_count": 45,
                        "audited_physics_substeps": 362_500,
                        "qpos_limit_violation_samples": 0,
                        "fall_count": 0,
                        "motion_contact_route_target_slew_head_violations": 0,
                    },
                ],
                "coupled_profile_cap_rejected": {
                    "candidate_extra_upper_margin_rad": 0.01625,
                    "hold_ticks": 13,
                    "reason": (
                        "tightening every backward profile cap fixed qpos but "
                        "introduced a right-family fall in the focus prefix"
                    ),
                    "fall_count": 1,
                    "passed_episode_count": 4,
                    "qpos_limit_violation_samples": 0,
                },
            },
            "qualification": {
                "passed": passed,
                "exact_scale_passed": exact_scale,
                "expected_episode_count": 20,
                "expected_segment_count": 180,
                "expected_physics_substeps": 1_450_000,
                "maximum_qpos_limit_violation_samples": 0,
                "maximum_falls": 0,
                "maximum_motion_contact_route_target_slew_head_violations": 0,
                "result": result,
            },
            "provenance": {
                "launcher": {
                    "path": str(launcher_path),
                    "sha256": diagnostic._sha256(launcher_path),
                },
                "diagnostic_runtime": {
                    "path": str(Path(diagnostic.__file__).resolve()),
                    "sha256": diagnostic._sha256(
                        Path(diagnostic.__file__).resolve()
                    ),
                },
                "policy": {
                    "path": str(diagnostic.BASE_POLICY.resolve()),
                    "sha256": diagnostic._sha256(
                        diagnostic.BASE_POLICY.resolve()
                    ),
                },
                "profiles": {
                    key: {
                        "path": str(path),
                        "sha256": diagnostic._sha256(path),
                    }
                    for key, path in diagnostic.PROFILE_PATHS.items()
                },
                "historical_h1_failure_evidence": {
                    "path": str(diagnostic.SOURCE_H1_ARTIFACT.resolve()),
                    "sha256": diagnostic._sha256(
                        diagnostic.SOURCE_H1_ARTIFACT.resolve()
                    ),
                },
                "straight_selection_evidence": [
                    {
                        "path": str(path.resolve()),
                        "sha256": diagnostic._sha256(path.resolve()),
                    }
                    for path in diagnostic.STRAIGHT_SELECTION_EVIDENCE_PATHS
                ],
                "exact_hardware_safe_assets": asset_evidence,
                "runtime_dependency_provenance": runtime_provenance,
                "onnx_providers": bank.session_providers,
                "policy_inference_counts": dict(
                    sorted(bank.inference_counts.items())
                ),
                "source_closure_pre": source_closure_pre,
                "source_closure_post": source_closure_post,
                "source_closure_unchanged": (
                    source_closure_pre == source_closure_post
                ),
            },
            "handoff": {
                "recommended_for_central_integration": passed,
                "requires_full_frozen_20x30_all_suite_requalification": True,
                "package_remains_blocked": True,
                "hardware_deployment": "PROHIBITED",
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        summary["output"] = str(output)
        summary["output_sha256"] = diagnostic._sha256(output)
        summary["artifact_status"] = payload["status"]
    print(json.dumps(summary, sort_keys=True))
    if args.output is not None and summary["artifact_status"] != (
        "DIAGNOSTIC_20X9_PASS_NOT_ADOPTED"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
