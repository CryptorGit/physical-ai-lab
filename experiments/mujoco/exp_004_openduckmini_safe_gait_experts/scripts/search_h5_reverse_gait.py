"""Simulation-only black-box search for an H5 reverse target gait.

This search changes only the temporary reverse target profile inside the exact
H5 diagnostic simulator.  It never changes the frozen evaluator, candidate
manifests, adoption state, or hardware path.  The output is a diagnostic
search trace used to choose a profile for later actor-space distillation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
for root in (EXP_ROOT, EXP_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts import evaluate_h5_routed_transitions as h5  # noqa: E402
from scripts import explore_h5_target_program as explore  # noqa: E402


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _sha(path: Path) -> str:
    return h5.sha256_file(path)


def _policy_args(path: Path) -> list[str]:
    return [f"{role}={path}" for role in h5.REQUIRED_POLICY_ROLES]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=_path, required=True)
    parser.add_argument("--generated-root", type=_path, required=True)
    parser.add_argument("--planar-params", type=_path, required=True)
    parser.add_argument("--planar-manifest", type=_path, required=True)
    parser.add_argument("--reverse-params", type=_path, required=True)
    parser.add_argument("--reverse-manifest", type=_path, required=True)
    parser.add_argument("--profile", type=_path, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument(
        "--mode",
        choices=("reference", "legacy_forward_target"),
        default="reference",
        help=(
            "temporary target source; legacy_forward_target reuses the audited "
            "forward actor waveform through the H5 decoder"
        ),
    )
    parser.add_argument(
        "--target-transform",
        choices=(
            "none", "flip_sagittal", "flip_sagittal_knee", "flip_hip_pitch",
            "swap_legs", "swap_legs_flip_sagittal",
        ),
        default="none",
        help="exploration-only transform applied by legacy_forward_target",
    )
    parser.add_argument(
        "--target-leg-gains",
        type=float,
        nargs=10,
        metavar=("Y_L", "R_L", "P_L", "K_L", "A_L", "Y_R", "R_R", "P_R", "K_R", "A_R"),
        help="evaluate one explicit ten-joint target deviation gain vector",
    )
    parser.add_argument(
        "--gain-low",
        type=float,
        default=0.25,
        help="lower bound for random legacy-forward target gains",
    )
    parser.add_argument(
        "--gain-high",
        type=float,
        default=1.20,
        help="upper bound for random legacy-forward target gains",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--phase-rate-scale", type=float, default=2.0)
    parser.add_argument("--phase-offset", type=float, default=3.0)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--initial-joint-noise-scale", type=float, default=0.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument(
        "--bias-radius",
        type=float,
        default=0.08,
        help="Random per-joint bias perturbation radius in radians.",
    )
    parser.add_argument(
        "--profile-multipliers",
        type=float,
        nargs=10,
        metavar=("Y_L", "R_L", "P_L", "K_L", "A_L", "Y_R", "R_R", "P_R", "K_R", "A_R"),
        help="evaluate one explicit ten-joint multiplier vector",
    )
    parser.add_argument(
        "--morphology-first",
        action="store_true",
        help=(
            "screen per-joint profile amplitude multipliers before the legacy "
            "phase/cadence anchor grid; simulation-only"
        ),
    )
    parser.add_argument(
        "--random-morphology-only",
        action="store_true",
        help=(
            "draw bounded per-joint scales and biases while holding the "
            "command-line phase/scale point fixed; simulation-only"
        ),
    )
    parser.add_argument(
        "--random-full-morphology",
        action="store_true",
        help=(
            "also randomize phase rate, phase offset, and target scale in the "
            "random morphology screen; simulation-only"
        ),
    )
    return parser


def _candidate_multipliers(rng: np.random.Generator, count: int) -> np.ndarray:
    """Generate structured plus bounded random per-joint gait multipliers."""

    if count <= 0:
        raise ValueError("candidate count must be positive")
    rows: list[np.ndarray] = []
    identity = np.ones(10, dtype=np.float64)
    rows.append(identity)
    for left_roll in (0.55, 0.70, 0.85, 1.0):
        for right_roll in (0.55, 0.70, 0.85, 1.0):
            row = identity.copy()
            row[1] = left_roll
            row[6] = right_roll
            rows.append(row)
    for yaw in (0.0, 0.25, 0.5, 0.75):
        row = identity.copy()
        row[0] = yaw
        row[5] = yaw
        rows.append(row)
    for pitch in (0.75, 0.9, 1.1, 1.25):
        row = identity.copy()
        row[[2, 7]] = pitch
        rows.append(row)
    for knee in (0.75, 0.9, 1.1, 1.25):
        row = identity.copy()
        row[[3, 8]] = knee
        rows.append(row)
    for ankle in (0.75, 0.9, 1.1, 1.25):
        row = identity.copy()
        row[[4, 9]] = ankle
        rows.append(row)
    while len(rows) < count:
        rows.append(rng.uniform(0.55, 1.15, size=10).astype(np.float64))
    return np.asarray(rows[:count], dtype=np.float64)


def _candidate_specs(
    rng: np.random.Generator,
    namespace: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Create a bounded joint/phase search with deterministic anchor points."""

    if namespace.profile_multipliers is not None:
        return [
            {
                "profile_multipliers": np.asarray(namespace.profile_multipliers, dtype=np.float64),
                "profile_bias_offsets": np.zeros(10, dtype=np.float64),
                "phase_rate_scale": float(namespace.phase_rate_scale),
                "phase_offset": float(namespace.phase_offset),
                "target_scale": float(namespace.target_scale),
            }
        ]
    if namespace.target_leg_gains is not None:
        return [
            {
                "profile_multipliers": np.ones(10, dtype=np.float64),
                "profile_bias_offsets": np.zeros(10, dtype=np.float64),
                "target_leg_gains": np.asarray(namespace.target_leg_gains, dtype=np.float64),
                "phase_rate_scale": float(namespace.phase_rate_scale),
                "phase_offset": float(namespace.phase_offset),
                "target_scale": float(namespace.target_scale),
            }
        ]
    if namespace.mode == "legacy_forward_target":
        if not 0.0 <= float(namespace.gain_low) <= float(namespace.gain_high) <= 2.0:
            raise ValueError("legacy-forward gain bounds must satisfy 0 <= low <= high <= 2")
        identity = np.ones(10, dtype=np.float64)
        gain_rows: list[np.ndarray] = [identity]
        # Deterministic low-dimensional anchors make the search interpretable:
        # first reduce sagittal pitch/knee/ankle authority, then restore one
        # actuator family at a time before drawing independent morphology rows.
        for value in (0.45, 0.60, 0.75, 0.90):
            row = identity.copy()
            row[[2, 3, 4, 7, 8, 9]] = value
            gain_rows.append(row)
        for pair in ((2, 7), (3, 8), (4, 9), (1, 6), (0, 5)):
            for value in (0.45, 0.70, 1.0):
                row = identity.copy()
                row[list(pair)] = value
                gain_rows.append(row)
        while len(gain_rows) < namespace.candidates:
            gain_rows.append(
                rng.uniform(
                    float(namespace.gain_low),
                    float(namespace.gain_high),
                    size=10,
                ).astype(np.float64)
            )
        rates = (0.75, 0.90, 1.00, 1.10, 1.25, 1.40)
        offsets = (0.0, 2.0, 4.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0)
        scales = (0.60, 0.70, 0.80, 0.90, 1.00)
        specs: list[dict[str, Any]] = []
        for index, gains in enumerate(gain_rows[: namespace.candidates]):
            if index == 0:
                rate = float(namespace.phase_rate_scale)
                offset = float(namespace.phase_offset)
                scale = float(namespace.target_scale)
            else:
                rate = float(rates[(index - 1) % len(rates)])
                offset = float(offsets[((index - 1) // len(rates)) % len(offsets)])
                scale = float(scales[((index - 1) // (len(rates) * len(offsets))) % len(scales)])
                if index >= len(rates) * len(offsets):
                    rate = float(rng.uniform(0.70, 1.45))
                    offset = float(rng.uniform(-1.0, 27.0))
                    scale = float(rng.uniform(0.55, 1.05))
            specs.append(
                {
                    "profile_multipliers": np.ones(10, dtype=np.float64),
                    "profile_bias_offsets": np.zeros(10, dtype=np.float64),
                    "target_leg_gains": gains,
                    "phase_rate_scale": rate,
                    "phase_offset": offset,
                    "target_scale": scale,
                }
            )
        return specs
    multipliers = _candidate_multipliers(rng, namespace.candidates)
    identity = np.ones(10, dtype=np.float64)
    specs: list[dict[str, Any]] = []
    if bool(getattr(namespace, "random_morphology_only", False)):
        for index in range(namespace.candidates):
            specs.append(
                {
                    "profile_multipliers": (
                        np.ones(10, dtype=np.float64)
                        if index == 0
                        else rng.uniform(0.40, 1.60, size=10).astype(np.float64)
                    ),
                    "profile_bias_offsets": (
                        np.zeros(10, dtype=np.float64)
                        if index == 0
                        else rng.uniform(
                            -float(namespace.bias_radius),
                            float(namespace.bias_radius),
                            size=10,
                        ).astype(np.float64)
                    ),
                    "phase_rate_scale": (
                        float(namespace.phase_rate_scale)
                        if index == 0 or not getattr(namespace, "random_full_morphology", False)
                        else float(rng.uniform(1.10, 2.60))
                    ),
                    "phase_offset": (
                        float(namespace.phase_offset)
                        if index == 0 or not getattr(namespace, "random_full_morphology", False)
                        else float(rng.uniform(-6.0, 14.0))
                    ),
                    "target_scale": (
                        float(namespace.target_scale)
                        if index == 0 or not getattr(namespace, "random_full_morphology", False)
                        else float(rng.uniform(0.80, 1.40))
                    ),
                }
            )
        return specs
    if bool(getattr(namespace, "morphology_first", False)):
        morphology_rows: list[np.ndarray] = [identity.copy()]
        for index in range(10):
            for value in (0.55, 0.75, 0.90, 1.10, 1.30):
                row = identity.copy()
                row[index] = value
                morphology_rows.append(row)
        for pair in ((2, 7), (3, 8), (4, 9), (1, 6), (0, 5)):
            for value in (0.65, 0.85, 1.15, 1.35):
                row = identity.copy()
                row[list(pair)] = value
                morphology_rows.append(row)
        while len(morphology_rows) < namespace.candidates:
            morphology_rows.append(
                rng.uniform(0.45, 1.45, size=10).astype(np.float64)
            )
        for row in morphology_rows[: namespace.candidates]:
            specs.append(
                {
                    "profile_multipliers": row,
                    "profile_bias_offsets": np.zeros(10, dtype=np.float64),
                    "phase_rate_scale": float(namespace.phase_rate_scale),
                    "phase_offset": float(namespace.phase_offset),
                    "target_scale": float(namespace.target_scale),
                }
            )
        return specs
    # Earlier screens concentrated on the fast end of the profile warp and
    # therefore could not distinguish a bad target geometry from a bad actor.
    # Keep the old fast anchors, but cover the slower cadence band where
    # stance slip and touchdown timing can recover under the fixed 0.04-rad
    # per-tick target slew guard.
    anchor_rates = (0.90, 1.05, 1.20, 1.35, 1.45, 1.60, 1.80, 2.00, 2.15, 2.35)
    anchor_offsets = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    anchor_scales = (0.65, 0.78, 0.88, 0.96, 1.0)
    for rate in anchor_rates:
        for offset in anchor_offsets:
            for scale in anchor_scales:
                if len(specs) >= namespace.candidates:
                    break
                specs.append(
                    {
                        # Isolate phase/cadence/scale first.  Coupling every
                        # phase point to a different morphology multiplier
                        # made the previous screen unable to identify the
                        # dominant target-program failure.
                        "profile_multipliers": identity.copy(),
                        "profile_bias_offsets": np.zeros(10, dtype=np.float64),
                        "phase_rate_scale": float(rate),
                        "phase_offset": float(offset),
                        "target_scale": float(scale),
                    }
                )
            if len(specs) >= namespace.candidates:
                break
        if len(specs) >= namespace.candidates:
            break
    while len(specs) < namespace.candidates:
        specs.append(
            {
                "profile_multipliers": multipliers[len(specs)].copy(),
                "profile_bias_offsets": rng.uniform(
                    -float(namespace.bias_radius),
                    float(namespace.bias_radius),
                    size=10,
                ).astype(np.float64),
                "phase_rate_scale": float(rng.uniform(1.25, 2.55)),
                "phase_offset": float(rng.uniform(-4.0, 10.0)),
                "target_scale": float(rng.uniform(0.65, 1.08)),
            }
        )
    # Preserve the command-line point as an explicit anchor whenever the
    # budget allows it; this makes the search output directly comparable to
    # the earlier fixed-point probe.
    specs[0].update(
        {
            "phase_rate_scale": float(namespace.phase_rate_scale),
            "phase_offset": float(namespace.phase_offset),
            "target_scale": float(namespace.target_scale),
        }
    )
    return specs


def _args(namespace: argparse.Namespace) -> argparse.Namespace:
    values: dict[str, Any] = {
        "policy": _policy_args(namespace.policy),
        "generated_root": namespace.generated_root,
        "seed": namespace.seed,
        "episodes": 1,
        "seconds": namespace.seconds,
        "transition_seconds": namespace.seconds,
        "transition_stand_seconds": 1.0,
        "warmup_seconds": 1.5,
        "initial_joint_noise_scale": 0.0,
        "initial_base_speed": 0.0,
        # The search is used to propose target programs for the unified H5
        # actor.  Keep its observation/command semantics identical to the
        # strict evaluator; the old domain-specific reverse mapper can make a
        # candidate look valid while producing the wrong actor input during
        # distillation or release evaluation.
        "unified_single_weight": True,
        "require_pass": False,
    }
    for domain in ("planar", "reverse"):
        params = getattr(namespace, f"{domain}_params")
        manifest = getattr(namespace, f"{domain}_manifest")
        values[f"h5_{domain}_params"] = params
        values[f"h5_{domain}_params_sha256"] = _sha(params)
        values[f"h5_{domain}_manifest"] = manifest
        values[f"h5_{domain}_manifest_sha256"] = _sha(manifest)
    return argparse.Namespace(**values)


def _score(result: dict[str, Any]) -> float:
    segment = result["segment"]
    metrics = segment.get("metrics", {})
    quality = segment.get("gait_quality_metrics", {})
    fell = bool(segment.get("fell", False))
    projected = float(metrics.get("projected_primary_velocity", 0.0))
    orthogonal = float(metrics.get("absolute_orthogonal_velocity", 1.0))
    yaw = float(metrics.get("yaw_rate_error", 1.0))
    support = float(metrics.get("single_support_rate", 0.0))
    slip = float(quality.get("stance_slip_rms_mps", 1.0))
    slip95 = float(quality.get("stance_slip_p95_mps", 1.0))
    step_imbalance = float(quality.get("step_count_imbalance", 20.0))
    cadence_imbalance = float(quality.get("contact_duty_imbalance", 1.0))
    left_steps = float(quality.get("left_step_count", 0.0))
    right_steps = float(quality.get("right_step_count", 0.0))
    support_rate = float(quality.get("single_support_rate", support))
    coverage = float(quality.get("contact_velocity_coverage", 0.0))
    force_p99 = float(
        quality.get("total_normal_force_p99_fraction_body_weight", 10.0)
    )
    force_mean = float(
        quality.get("steady_mean_total_normal_force_fraction_body_weight", 10.0)
    )
    wrong_way_occupancy = float(
        np.max(np.asarray(quality.get("startup_axis_wrong_way_occupancy", [1.0])))
    )
    t75 = quality.get("linear_t75_s")
    # ``projected_primary_velocity`` is the velocity projected onto the
    # *command direction*.  The routed reverse command is [-0.05, 0, 0], so a
    # physically correct reverse gait has a positive projected speed of about
    # +0.05 m/s even though its local-x velocity is negative.  Keep this
    # convention explicit: using -0.05 here ranks a forward-moving candidate
    # as the best reverse gait.
    direction_error = abs(projected - 0.05)
    score = (
        100.0 * float(fell)
        + 20.0 * direction_error
        + 3.0 * orthogonal
        + 2.0 * yaw
        + 2.0 * abs(support - 0.20)
        + 2.0 * slip
        + 1.0 * slip95
        + 0.01 * step_imbalance
        + 1.0 * cadence_imbalance
    )
    if projected <= 0.0:
        score += 2.0
    if not bool(quality.get("measurement_complete", False)):
        score += 10.0
    # This is a diagnostic ranking function, not a replacement acceptance
    # gate.  The old weights selected fast but unusable profiles (for example
    # 16/5 touchdowns, support 0.166, and force p99 3.48).  Penalize the
    # same strict-quality failure dimensions that determine adoption so the
    # next search spends its budget on a gait-shaped candidate.
    score += 8.0 * max(0.0, 0.25 - support_rate)
    score += 8.0 * max(0.0, support_rate - 0.60)
    score += 12.0 * max(0.0, slip - 0.015)
    score += 6.0 * max(0.0, slip95 - 0.030)
    score += 6.0 * max(0.0, 0.020 - coverage)
    score += 8.0 * max(0.0, 0.95 - coverage)
    score += 2.0 * max(0.0, force_p99 - 3.0)
    score += 0.5 * max(0.0, 0.8 - force_mean)
    score += 0.5 * max(0.0, force_mean - 1.2)
    score += 0.04 * max(0.0, step_imbalance - 3.0)
    score += 0.04 * max(0.0, 6.0 - left_steps)
    score += 0.04 * max(0.0, 6.0 - right_steps)
    score += 2.0 * max(0.0, wrong_way_occupancy - 0.10)
    score += 1.0 if t75 is None else 0.0
    return float(score)


def _run_one(
    namespace: argparse.Namespace,
    simulator: Any,
    bank: Any,
    metadata: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    simulator.evaluator._evaluator.load_backward_profile(namespace.profile)
    evaluator = simulator.evaluator._evaluator
    base_scales = np.asarray(evaluator.backward_gait_scales, dtype=np.float64).copy()
    base_biases = np.asarray(evaluator.backward_gait_biases, dtype=np.float64).copy()
    multipliers = np.asarray(spec["profile_multipliers"], dtype=np.float64)
    bias_offsets = np.asarray(
        spec.get("profile_bias_offsets", np.zeros(10, dtype=np.float64)),
        dtype=np.float64,
    )
    evaluator.backward_gait_scales = base_scales * multipliers
    evaluator.backward_gait_biases = base_biases + bias_offsets
    result = explore._run_on_simulator(
        simulator,
        bank,
        metadata,
        route="reverse",
        seconds=namespace.seconds,
        phase_offset=float(spec["phase_offset"]),
        phase_rate_scale=float(spec["phase_rate_scale"]),
        target_scale=float(spec["target_scale"]),
        mode=str(namespace.mode),
        action_scale=1.0,
        action_smoothing=1.0,
        teacher_table=None,
        actor_residual_scale=0.0,
        joint_noise_scale=float(namespace.initial_joint_noise_scale),
        initial_base_speed=float(namespace.initial_base_speed),
        warmup_seconds=float(namespace.warmup_seconds),
        target_transform=str(namespace.target_transform),
        target_leg_gains=spec.get("target_leg_gains"),
    )
    result["profile_multipliers"] = [float(value) for value in multipliers]
    result["profile_bias_offsets"] = [float(value) for value in bias_offsets]
    result["target_transform"] = str(namespace.target_transform)
    result["target_leg_gains"] = (
        [float(value) for value in spec["target_leg_gains"]]
        if spec.get("target_leg_gains") is not None
        else None
    )
    result["objective_score"] = _score(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    if namespace.candidates <= 0 or namespace.seconds <= 0.0:
        raise ValueError("candidates and seconds must be positive")
    rng = np.random.default_rng(namespace.seed)
    rows = []
    setup = _args(namespace)
    simulator, bank, metadata = h5._build_simulator(setup)
    for spec in _candidate_specs(rng, namespace):
        rows.append(_run_one(namespace, simulator, bank, metadata, spec))
    rows.sort(key=lambda row: float(row["objective_score"]))
    payload = {
        "schema_version": 1,
        "status": "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "evaluator": "openduckmini-exp004-h5-target-program-search",
        "configuration": {
            "seed": namespace.seed,
            "candidates": namespace.candidates,
            "seconds": namespace.seconds,
            "phase_rate_scale": namespace.phase_rate_scale,
            "phase_offset": namespace.phase_offset,
            "target_scale": namespace.target_scale,
            "initial_joint_noise_scale": namespace.initial_joint_noise_scale,
            "initial_base_speed": namespace.initial_base_speed,
            "warmup_seconds": namespace.warmup_seconds,
            "bias_radius": namespace.bias_radius,
            "profile": str(namespace.profile),
            "mode": namespace.mode,
            "target_transform": namespace.target_transform,
            "target_leg_gains": namespace.target_leg_gains,
            "gain_low": namespace.gain_low,
            "gain_high": namespace.gain_high,
        },
        "best": rows[0] if rows else None,
        "results": rows,
    }
    namespace.output.parent.mkdir(parents=True, exist_ok=True)
    namespace.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    best = rows[0]
    positive_nonfall = [
        row
        for row in rows
        if float(row["segment"]["metrics"].get("projected_primary_velocity", 0.0)) > 0.0
        and not bool(row["segment"].get("fell", False))
    ]
    best_positive_nonfall = min(positive_nonfall, key=lambda row: float(row["objective_score"])) if positive_nonfall else None
    print(json.dumps({
        "output": str(namespace.output),
        "best_score": best["objective_score"],
        "best_multipliers": best["profile_multipliers"],
        "best_bias_offsets": best["profile_bias_offsets"],
        "best_target_leg_gains": best.get("target_leg_gains"),
        "quality_passed": best["quality_passed"],
        "fell": best["segment"]["fell"],
        "projected_primary_velocity": best["segment"]["metrics"].get("projected_primary_velocity"),
        "best_positive_nonfall": (
            {
                "score": best_positive_nonfall["objective_score"],
                "projected_primary_velocity": best_positive_nonfall["segment"]["metrics"].get("projected_primary_velocity"),
                "phase_rate_scale": best_positive_nonfall["phase_rate_scale"],
                "phase_offset": best_positive_nonfall["phase_offset"],
                "target_scale": best_positive_nonfall["target_scale"],
                "profile_multipliers": best_positive_nonfall["profile_multipliers"],
                "profile_bias_offsets": best_positive_nonfall["profile_bias_offsets"],
                "target_leg_gains": best_positive_nonfall.get("target_leg_gains"),
            }
            if best_positive_nonfall is not None
            else None
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
