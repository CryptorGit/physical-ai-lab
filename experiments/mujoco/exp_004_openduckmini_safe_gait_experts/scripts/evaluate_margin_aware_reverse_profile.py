"""Evaluate one margin-aware reverse profile through exp_004 routing on CPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
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
    load_profile,
)


DEFAULT_PROFILE = (
    EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_candidate_v1.json"
)
DEFAULT_OUTPUT = (
    EXP_ROOT / "artifacts" / "reverse_margin050_slew200_candidate_v1_2x5x15s.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one candidate at reverse command endpoints with physical-SAFE "
            "resets, fixed perturb seeds, margin=0.050, and slew=2.0."
        )
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--initial-joint-noise-scale", type=float, default=1.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.10)
    parser.add_argument("--reset-qpos-inward-margin-rad", type=float, default=0.005)
    parser.add_argument("--minimum-reverse-vx", type=float, default=-0.020)
    parser.add_argument(
        "--left-knee-extra-upper-margin-rad",
        type=float,
        help=(
            "Optional stricter post-profile left-knee upper target cap. "
            "Defaults to the candidate composition, else zero."
        ),
    )
    parser.add_argument(
        "--command-vx",
        nargs="+",
        type=float,
        default=[-0.040, -0.050],
    )
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.episodes < 3:
        parser.error("--episodes must be at least 3 for perturbation robustness")
    if args.seconds <= 0.0 or not 0.0 <= args.warmup_seconds < args.seconds:
        parser.error("seconds must be positive and warmup inside the episode")
    if (
        args.initial_joint_noise_scale < 0.0
        or args.initial_base_speed < 0.0
        or args.reset_qpos_inward_margin_rad < 0.0
    ):
        parser.error("perturbation magnitudes must be non-negative")
    if not args.command_vx or any(value >= 0.0 for value in args.command_vx):
        parser.error("--command-vx requires one or more negative values")
    if args.minimum_reverse_vx >= 0.0:
        parser.error("--minimum-reverse-vx must be negative")
    if (
        args.left_knee_extra_upper_margin_rad is not None
        and not 0.0 <= args.left_knee_extra_upper_margin_rad <= 0.05
    ):
        parser.error("left-knee extra upper margin must be in [0, 0.05]")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    profile_path = args.profile.resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing candidate profile: {profile_path}")
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    parameters = load_profile(profile_path)
    composition = profile_payload.get("composition", {})
    profile_cap = float(
        composition.get("left_knee_extra_upper_margin_rad", 0.0)
    )
    knee_cap = (
        profile_cap
        if args.left_knee_extra_upper_margin_rad is None
        else float(args.left_knee_extra_upper_margin_rad)
    )
    if not 0.0 <= knee_cap <= 0.05:
        raise ValueError("candidate left-knee extra upper margin is outside [0, 0.05]")
    evaluator = MarginAwareReverseEvaluator(args)
    seeds = tuple(args.seed + index for index in range(args.episodes))
    validation = evaluator.routed_validation(
        parameters,
        seeds,
        args.seconds,
        tuple(args.command_vx),
        left_knee_extra_upper_margin_rad=knee_cap,
    )
    passed = bool(
        validation["all_hard_safety_passed"]
        and validation["all_standard_segment_checks_passed"]
        and validation["reverse_motion_retained"]
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_margin_aware_reverse_routed_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SCREEN_PASS" if passed else "SCREEN_FAIL",
        "hardware_deployment": "PROHIBITED",
        "simulation_only": True,
        "configuration": {
            "episodes_per_command": args.episodes,
            "seconds_per_episode": args.seconds,
            "warmup_seconds": args.warmup_seconds,
            "commands_vx": list(args.command_vx),
            "fixed_perturb_seeds": list(seeds),
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "initial_base_speed_max_mps": args.initial_base_speed,
            "reset_qpos_inward_margin_rad": args.reset_qpos_inward_margin_rad,
            "leg_target_margin_rad": TARGET_MARGIN_RAD,
            "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
            "backward_residual_scale": 0.0,
            "head_target_indices": [5, 6, 7, 8],
            "head_target_value": 0.0,
            "left_knee_extra_upper_margin_rad": knee_cap,
            "physical_safe_reset": True,
        },
        "candidate_profile": {
            "path": str(profile_path),
            "sha256": sha256_file(profile_path),
            "parameters": parameters.as_json(),
        },
        "provenance": {
            "evaluator_script": str(Path(__file__).resolve()),
            "evaluator_script_sha256": sha256_file(Path(__file__).resolve()),
            "optimizer_script": str(
                (SCRIPT_DIR / "optimize_margin_aware_reverse.py").resolve()
            ),
            "optimizer_script_sha256": sha256_file(
                (SCRIPT_DIR / "optimize_margin_aware_reverse.py").resolve()
            ),
            "base_policy": {
                "path": str(args.policy.resolve()),
                "sha256": sha256_file(args.policy.resolve()),
            },
            "generated_assets": evaluator.asset_evidence,
            "model_contract": evaluator.model_evidence,
        },
        "acceptance": {
            "passed": passed,
            "all_hard_safety_passed": validation["all_hard_safety_passed"],
            "all_standard_segment_checks_passed": validation[
                "all_standard_segment_checks_passed"
            ],
            "reverse_motion_retained_threshold_vx": validation[
                "reverse_motion_retained_threshold_vx"
            ],
            "reverse_motion_retained": validation["reverse_motion_retained"],
            "requires_zero_falls_qpos_target_head_and_slew_violations": True,
        },
        "metrics": {
            key: validation[key]
            for key in (
                "episode_count",
                "mean_vx",
                "worst_seed_command_vx",
                "mean_vy",
                "maximum_absolute_mean_vy",
                "mean_yaw_rate",
                "maximum_absolute_mean_yaw_rate",
            )
        },
        "routed_validation": validation,
        "adoption": {
            "status": "NOT_ADOPTED_DIAGNOSTIC_ONLY",
            "simulation_pass_does_not_authorize_hardware": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": payload["status"],
                "episode_count": validation["episode_count"],
                "mean_vx": validation["mean_vx"],
                "worst_seed_command_vx": validation["worst_seed_command_vx"],
                "mean_vy": validation["mean_vy"],
                "mean_yaw_rate": validation["mean_yaw_rate"],
                "all_hard_safety_passed": validation[
                    "all_hard_safety_passed"
                ],
                "reverse_motion_retained": validation["reverse_motion_retained"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
