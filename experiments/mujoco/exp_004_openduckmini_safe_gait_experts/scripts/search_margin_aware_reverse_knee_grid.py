"""Local left-knee stabilization grid for a margin-aware reverse profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_root in (EXP_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from safe_gait_experts.routed_evaluation import sha256_file  # noqa: E402
from optimize_margin_aware_reverse import (  # noqa: E402
    DEFAULT_GENERATED_ROOT,
    DEFAULT_POLICY,
    TARGET_MARGIN_RAD,
    TARGET_SLEW_RAD_S,
    MarginAwareReverseEvaluator,
    ProfileParameters,
    load_profile,
)


DEFAULT_PROFILE = (
    EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_candidate_v2.json"
)
DEFAULT_OUTPUT = (
    EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_knee_grid_v1.json"
)
DEFAULT_CANDIDATE_OUTPUT = (
    EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_candidate_v3.json"
)
LEFT_KNEE_PARAMETER_INDEX = 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--initial-joint-noise-scale", type=float, default=1.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.10)
    parser.add_argument("--reset-qpos-inward-margin-rad", type=float, default=0.005)
    parser.add_argument("--minimum-reverse-vx", type=float, default=-0.020)
    parser.add_argument("--target-vx-min", type=float, default=-0.050)
    parser.add_argument("--target-vx-max", type=float, default=-0.040)
    parser.add_argument(
        "--knee-scale-factors",
        nargs="+",
        type=float,
        default=[0.60, 0.70, 0.80, 0.90, 1.00],
    )
    parser.add_argument(
        "--knee-bias-deltas",
        nargs="+",
        type=float,
        default=[-0.040, -0.030, -0.020, -0.010, -0.005, 0.0],
    )
    parser.add_argument(
        "--phase-factors", nargs="+", type=float, default=[0.90, 1.00, 1.10]
    )
    parser.add_argument(
        "--left-knee-extra-upper-margins-rad",
        nargs="+",
        type=float,
        default=[0.005, 0.0075, 0.010],
    )
    parser.add_argument("--routed-finalists", type=int, default=10)
    args = parser.parse_args(argv)
    for path in (args.output, args.candidate_output):
        if path.exists() and not args.overwrite:
            parser.error(f"refusing to overwrite existing output: {path}")
    if args.episodes < 3 or args.seconds <= 0.0:
        parser.error("at least three episodes and positive seconds are required")
    if not 0.0 <= args.warmup_seconds < args.seconds:
        parser.error("warmup must lie inside the episode")
    if args.initial_joint_noise_scale <= 0.0 or args.initial_base_speed <= 0.0:
        parser.error("full perturbation search requires positive noise and push")
    if args.reset_qpos_inward_margin_rad != 0.005:
        parser.error("reset qpos inward margin is frozen at 0.005 rad")
    if args.minimum_reverse_vx >= 0.0:
        parser.error("minimum reverse vx gate must be negative")
    if any(value <= 0.0 or value > 1.0 for value in args.knee_scale_factors):
        parser.error("knee scale factors must lie in (0, 1]")
    if any(value > 0.0 or value < -0.15 for value in args.knee_bias_deltas):
        parser.error("knee bias deltas must lie in [-0.15, 0]")
    if any(value <= 0.0 for value in args.phase_factors):
        parser.error("phase factors must be positive")
    if any(
        value < 0.0 or value > 0.05
        for value in args.left_knee_extra_upper_margins_rad
    ):
        parser.error("left-knee extra margins must lie in [0, 0.05]")
    if args.routed_finalists <= 0:
        parser.error("--routed-finalists must be positive")
    return args


def candidate_variant(
    initial: ProfileParameters,
    knee_scale_factor: float,
    knee_bias_delta: float,
    phase_factor: float,
) -> ProfileParameters:
    scales = initial.amplitude_scales.copy()
    biases = initial.bias_offsets.copy()
    scales[LEFT_KNEE_PARAMETER_INDEX] *= knee_scale_factor
    biases[LEFT_KNEE_PARAMETER_INDEX] += knee_bias_delta
    return ProfileParameters(scales, biases, initial.phase_rate * phase_factor)


def compact_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "cost": evaluation["cost"],
        "all_hard_safety_passed": evaluation["all_hard_safety_passed"],
        "mean_vx": evaluation["mean_vx"],
        "worst_seed_vx": evaluation["worst_seed_vx"],
        "vx_standard_deviation": evaluation["vx_standard_deviation"],
        "episodes": [
            {
                "seed": episode["seed"],
                "fell": episode["fell"],
                "completed_seconds": episode["completed_seconds"],
                "mean_local_velocity_xyz": episode["mean_local_velocity_xyz"],
                "mean_local_angular_velocity_xyz": episode[
                    "mean_local_angular_velocity_xyz"
                ],
                "minimum_height_m": episode["minimum_height_m"],
                "minimum_upright": episode["minimum_upright"],
                "safety_passed": episode["safety_passed"],
                "hard_safety_failures": episode["hard_safety_failures"],
                "qpos_limit_violations": episode["safety_audit"][
                    "qpos_limit_violations"
                ],
                "maximum_qpos_excess_rad": episode["safety_audit"][
                    "maximum_qpos_excess_rad"
                ],
                "target_slew_violations": episode["safety_audit"][
                    "target_slew_violations"
                ],
                "head_target_peak_rad": episode["safety_audit"][
                    "head_target_peak_rad"
                ],
                "physics_substep_qpos_limit_violations": episode[
                    "physics_substep_qpos_limit_violations"
                ],
                "maximum_physics_substep_qpos_excess_rad": episode[
                    "maximum_physics_substep_qpos_excess_rad"
                ],
            }
            for episode in evaluation["episodes"]
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    profile_path = args.initial_profile.resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing initial profile: {profile_path}")
    initial = load_profile(profile_path)
    evaluator = MarginAwareReverseEvaluator(args)
    seeds = tuple(args.seed + index for index in range(args.episodes))
    grid_results: list[dict[str, Any]] = []
    candidates: list[tuple[ProfileParameters, dict[str, Any], dict[str, float]]] = []
    total = (
        len(args.knee_scale_factors)
        * len(args.knee_bias_deltas)
        * len(args.phase_factors)
        * len(args.left_knee_extra_upper_margins_rad)
    )
    completed = 0
    for knee_extra_margin in args.left_knee_extra_upper_margins_rad:
        for scale_factor in args.knee_scale_factors:
            for bias_delta in args.knee_bias_deltas:
                for phase_factor in args.phase_factors:
                    parameters = candidate_variant(
                        initial, scale_factor, bias_delta, phase_factor
                    )
                    evaluation = evaluator.evaluate_candidate(
                        parameters,
                        seeds,
                        args.seconds,
                        args.target_vx_min,
                        args.target_vx_max,
                        left_knee_extra_upper_margin_rad=knee_extra_margin,
                    )
                    variation = {
                        "left_knee_amplitude_scale_factor": float(scale_factor),
                        "left_knee_bias_delta_rad": float(bias_delta),
                        "phase_rate_factor": float(phase_factor),
                        "left_knee_extra_upper_margin_rad": float(
                            knee_extra_margin
                        ),
                    }
                    candidates.append((parameters, evaluation, variation))
                    grid_results.append(
                        {
                            "variation": variation,
                            "parameters": parameters.as_json(),
                            "evaluation": compact_evaluation(evaluation),
                        }
                    )
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"grid={completed}/{total}", flush=True)

    safe_ranked = sorted(
        (
            item
            for item in candidates
            if item[1]["all_hard_safety_passed"]
            and item[1]["worst_seed_vx"] <= args.minimum_reverse_vx
        ),
        key=lambda item: item[1]["cost"],
    )
    routed_finalists = []
    selected: tuple[
        ProfileParameters, dict[str, Any], dict[str, float], dict[str, Any]
    ] | None = None
    for rank, (parameters, evaluation, variation) in enumerate(
        safe_ranked[: args.routed_finalists], start=1
    ):
        routed = evaluator.routed_validation(
            parameters,
            seeds,
            args.seconds,
            (args.target_vx_max, args.target_vx_min),
            left_knee_extra_upper_margin_rad=variation[
                "left_knee_extra_upper_margin_rad"
            ],
        )
        passed = bool(
            routed["all_hard_safety_passed"]
            and routed["all_standard_segment_checks_passed"]
            and routed["worst_seed_command_vx"] <= args.minimum_reverse_vx
        )
        routed_finalists.append(
            {
                "safe_grid_rank": rank,
                "variation": variation,
                "parameters": parameters.as_json(),
                "fast_evaluation": compact_evaluation(evaluation),
                "routed_validation": routed,
                "passed_pilot_gate": passed,
            }
        )
        if passed and selected is None:
            selected = (parameters, evaluation, variation, routed)

    passed = selected is not None
    selected_payload = None
    if selected is not None:
        parameters, evaluation, variation, routed = selected
        selected_payload = {
            "variation": variation,
            "parameters": parameters.as_json(),
            "fast_evaluation": compact_evaluation(evaluation),
            "routed_validation": routed,
        }

    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_margin_aware_reverse_knee_grid",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PILOT_CANDIDATE" if passed else "NO_SAFE_MOTION_CANDIDATE",
        "hardware_deployment": "PROHIBITED",
        "simulation_only": True,
        "configuration": {
            "seconds_per_seed": args.seconds,
            "fixed_perturb_seeds": list(seeds),
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "initial_base_speed_max_mps": args.initial_base_speed,
            "positive_noise_reset_qpos_inward_margin_rad": (
                args.reset_qpos_inward_margin_rad
            ),
            "leg_target_margin_rad": TARGET_MARGIN_RAD,
            "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
            "minimum_reverse_vx_gate": args.minimum_reverse_vx,
            "left_knee_amplitude_scale_factors": args.knee_scale_factors,
            "left_knee_bias_deltas_rad": args.knee_bias_deltas,
            "phase_rate_factors": args.phase_factors,
            "left_knee_extra_upper_margins_rad": (
                args.left_knee_extra_upper_margins_rad
            ),
            "grid_candidate_count": total,
        },
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "initial_profile": {
                "path": str(profile_path),
                "sha256": sha256_file(profile_path),
            },
            "base_policy": {
                "path": str(args.policy.resolve()),
                "sha256": sha256_file(args.policy.resolve()),
            },
            "generated_assets": evaluator.asset_evidence,
            "model_contract": evaluator.model_evidence,
        },
        "selected": selected_payload,
        "routed_finalists": routed_finalists,
        "grid_results": grid_results,
        "adoption": {
            "status": "NOT_ADOPTED_PENDING_5X15" if passed else "NO_CANDIDATE",
            "hardware_deployment": "PROHIBITED",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if selected is not None:
        parameters, _, variation, routed = selected
        candidate = {
            "schema_version": 1,
            "artifact_kind": "openduckmini_reverse_feedforward_profile_candidate",
            "release_id": "optimized_reverse_margin050_slew200_candidate_v3",
            "status": "PILOT_CANDIDATE_NOT_ADOPTED",
            "hardware_deployment": "PROHIBITED",
            "parameters": parameters.as_json(),
            "composition": {
                "backward_residual_scale": 0.0,
                "leg_target_margin_rad": TARGET_MARGIN_RAD,
                "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
                "left_knee_extra_upper_margin_rad": variation[
                    "left_knee_extra_upper_margin_rad"
                ],
                "positive_noise_reset_qpos_inward_margin_rad": 0.005,
                "initial_target_order": (
                    "guarded_control_before_first_physics_decimation"
                ),
                "head_target_indices": [5, 6, 7, 8],
                "head_target_value": 0.0,
            },
            "pilot_evidence": {
                "source": str(args.output.resolve()),
                "source_sha256": sha256_file(args.output.resolve()),
                "variation_from_candidate_v2": variation,
                "seconds_per_seed_command": args.seconds,
                "fixed_perturb_seeds": list(seeds),
                "routed_mean_vx": routed["mean_vx"],
                "routed_worst_vx": routed["worst_seed_command_vx"],
                "routed_mean_vy": routed["mean_vy"],
                "routed_mean_yaw_rate": routed["mean_yaw_rate"],
                "falls": sum(
                    int(episode["segment"]["fell"])
                    for episode in routed["episodes"]
                ),
                "all_hard_safety_passed": routed["all_hard_safety_passed"],
                "all_standard_segment_checks_passed": routed[
                    "all_standard_segment_checks_passed"
                ],
            },
            "adoption": {
                "status": "NOT_ADOPTED_PENDING_5X15_AND_20X30",
                "simulation_only": True,
            },
        }
        args.candidate_output.write_text(
            json.dumps(candidate, indent=2), encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_output": (
                    str(args.candidate_output.resolve()) if passed else None
                ),
                "status": payload["status"],
                "safe_fast_candidate_count": len(safe_ranked),
                "routed_finalist_count": len(routed_finalists),
                "selected_mean_vx": (
                    selected[3]["mean_vx"] if selected is not None else None
                ),
                "selected_worst_vx": (
                    selected[3]["worst_seed_command_vx"]
                    if selected is not None
                    else None
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
