"""Search and pilot-qualify the two atomic reverse-turn profiles.

This utility is deliberately separate from the release evaluator.  It fixes the
straight reverse feedforward to candidate v3, including its 0.0125 rad dynamic
left-knee cap, and searches only the periodic turn-profile parameters.  A
profile file is emitted only after every fixed perturbation seed passes a
15-second routed rollout with the atomic endpoint, control-first actuation, and
all MuJoCo physics substeps audited.

The result is simulation evidence only.  Hardware deployment is always
PROHIBITED, including when both pilot gates pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_root in (EXP_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from router import REVERSE_TURN_ENDPOINTS  # noqa: E402
from safe_gait_experts.routed_evaluation import (  # noqa: E402
    AcceptanceThresholds,
    segment_acceptance,
    sha256_file,
)
from optimize_margin_aware_reverse import (  # noqa: E402
    DEFAULT_GENERATED_ROOT,
    DEFAULT_POLICY,
    TARGET_MARGIN_RAD,
    TARGET_SLEW_RAD_S,
    MarginAwareReverseEvaluator,
    PerJointCapRoutedSimulator,
    ProfileParameters,
    SharedV22PolicyBank,
    load_profile,
)


BASE_PROFILE = (
    EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_candidate_v3.json"
)
DEFAULT_OUTPUT = (
    EXP_ROOT / "artifacts" / "reverse_turn_margin050_slew200_search_v1.json"
)
DEFAULT_CANDIDATE_DIR = EXP_ROOT / "artifacts" / "reverse_turn_candidates_v1"
PILOT_SEEDS = (20260808, 20260809, 20260810)
LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD = 0.0125
RESET_QPOS_MARGIN_RAD = 0.005
PILOT_SECONDS = 15.0
PILOT_WARMUP_SECONDS = 1.5
MINIMUM_PROGRESS_FRACTION = 0.30


@dataclass(frozen=True)
class TurnDefinition:
    key: str
    expert: str
    direction: int
    command: tuple[float, float, float]

    @property
    def minimum_reverse_vx(self) -> float:
        return -MINIMUM_PROGRESS_FRACTION * abs(self.command[0])

    @property
    def minimum_signed_yaw_rate(self) -> float:
        return MINIMUM_PROGRESS_FRACTION * abs(self.command[2])


TURNS = {
    "left": TurnDefinition(
        "left", "reverse_turn_left", 1, REVERSE_TURN_ENDPOINTS["reverse_turn_left"]
    ),
    "right": TurnDefinition(
        "right",
        "reverse_turn_right",
        -1,
        REVERSE_TURN_ENDPOINTS["reverse_turn_right"],
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--base-profile", type=Path, default=BASE_PROFILE)
    parser.add_argument(
        "--left-seed-profile",
        type=Path,
        help="Optional previously screened loaded profile inserted into the left search population.",
    )
    parser.add_argument(
        "--right-seed-profile",
        type=Path,
        help="Optional previously screened loaded profile inserted into the right search population.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--directions", nargs="+", choices=tuple(TURNS), default=list(TURNS))
    parser.add_argument("--search-seconds", type=float, default=3.0)
    parser.add_argument("--search-warmup-seconds", type=float, default=0.6)
    parser.add_argument("--pilot-seconds", type=float, default=PILOT_SECONDS)
    parser.add_argument("--pilot-warmup-seconds", type=float, default=PILOT_WARMUP_SECONDS)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--popsize", type=int, default=2)
    parser.add_argument("--search-seed", type=int, default=20260808)
    parser.add_argument("--perturb-seeds", nargs="+", type=int, default=list(PILOT_SEEDS))
    parser.add_argument("--initial-joint-noise-scale", type=float, default=1.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.10)
    parser.add_argument("--reset-qpos-inward-margin-rad", type=float, default=RESET_QPOS_MARGIN_RAD)
    parser.add_argument("--left-knee-extra-upper-margin-rad", type=float, default=LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument(
        "--require-gait-quality",
        action="store_true",
        help=(
            "Require full strict gait-quality acceptance for every pilot "
            "finalist; without it the historical safety-only pilot remains."
        ),
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.maxiter < 1 or args.popsize < 1 or args.top_candidates < 1:
        parser.error("iteration, population, and finalist counts must be positive")
    if len(args.perturb_seeds) < 3 or len(set(args.perturb_seeds)) != len(args.perturb_seeds):
        parser.error("--perturb-seeds requires at least three distinct seeds")
    if args.search_seconds <= 0.0 or args.pilot_seconds < PILOT_SECONDS:
        parser.error("search seconds must be positive and pilot seconds at least 15")
    if not 0.0 <= args.search_warmup_seconds < args.search_seconds:
        parser.error("search warmup must lie inside the search rollout")
    if not 0.0 <= args.pilot_warmup_seconds < args.pilot_seconds:
        parser.error("pilot warmup must lie inside the pilot rollout")
    if args.initial_joint_noise_scale <= 0.0 or args.initial_base_speed <= 0.0:
        parser.error("pilot search requires non-zero reset noise and base push")
    if args.reset_qpos_inward_margin_rad != RESET_QPOS_MARGIN_RAD:
        parser.error("reset qpos margin is frozen at 0.005 rad")
    if args.left_knee_extra_upper_margin_rad != LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD:
        parser.error("candidate-v3 left-knee extra cap is frozen at 0.0125 rad")
    return args


def validate_base_profile(path: Path) -> tuple[ProfileParameters, dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    composition = payload.get("composition", {})
    required = {
        "leg_target_margin_rad": TARGET_MARGIN_RAD,
        "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
        "left_knee_extra_upper_margin_rad": LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
        "positive_noise_reset_qpos_inward_margin_rad": RESET_QPOS_MARGIN_RAD,
        "backward_residual_scale": 0.0,
    }
    mismatches = {
        key: {"expected": expected, "actual": composition.get(key)}
        for key, expected in required.items()
        if composition.get(key) != expected
    }
    if payload.get("release_id") != "optimized_reverse_margin050_slew200_candidate_v3":
        mismatches["release_id"] = {
            "expected": "optimized_reverse_margin050_slew200_candidate_v3",
            "actual": payload.get("release_id"),
        }
    if mismatches:
        raise ValueError(f"base profile is not the frozen candidate v3: {mismatches}")
    return load_profile(path), payload


def parameters_to_vector(parameters: ProfileParameters) -> np.ndarray:
    return np.concatenate(
        (
            parameters.amplitude_scales,
            parameters.bias_offsets,
            np.asarray((parameters.phase_rate,), dtype=np.float64),
        )
    )


def vector_to_parameters(vector: Sequence[float]) -> ProfileParameters:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (21,) or not np.all(np.isfinite(values)):
        raise ValueError("turn candidate must contain 21 finite values")
    return ProfileParameters(values[:10].copy(), values[10:20].copy(), float(values[20]))


def profile_blend_for_direction(
    base: ProfileParameters, candidate: ProfileParameters, direction: int
) -> ProfileParameters:
    """Return the profile actually composed by ``backward_parameters``."""

    blend = 1.0 if direction > 0 else 0.90
    return ProfileParameters(
        (1.0 - blend) * base.amplitude_scales + blend * candidate.amplitude_scales,
        (1.0 - blend) * base.bias_offsets + blend * candidate.bias_offsets,
        (1.0 - blend) * base.phase_rate + blend * candidate.phase_rate,
    )


def effective_to_loaded_profile(
    base: ProfileParameters, effective: ProfileParameters, direction: int
) -> ProfileParameters:
    """Invert the runtime's right-turn 0.90 blend for a searched effective profile."""

    blend = 1.0 if direction > 0 else 0.90
    return ProfileParameters(
        (effective.amplitude_scales - (1.0 - blend) * base.amplitude_scales) / blend,
        (effective.bias_offsets - (1.0 - blend) * base.bias_offsets) / blend,
        (effective.phase_rate - (1.0 - blend) * base.phase_rate) / blend,
    )


def search_bounds(base: ProfileParameters) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for scale in base.amplitude_scales:
        bounds.append((max(0.0, 0.20 * float(scale)), min(8.0, max(1.0, 2.25 * float(scale)))))
    for bias in base.bias_offsets:
        bounds.append((max(-0.22, float(bias) - 0.12), min(0.22, float(bias) + 0.12)))
    bounds.append((0.70, 4.00))
    return bounds


def initial_population(
    bounds: Sequence[tuple[float, float]],
    base: ProfileParameters,
    popsize: int,
    seed: int,
) -> np.ndarray:
    dimensions = len(bounds)
    size = max(5, popsize * dimensions)
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    rng = np.random.default_rng(seed)
    population = rng.uniform(lower, upper, size=(size, dimensions))
    base_vector = np.clip(parameters_to_vector(base), lower, upper)
    population[0] = base_vector
    row = 1
    # Deterministic one-sided steering probes improve early coverage while all
    # other joints remain at candidate-v3 values.
    for index, delta in (
        (11, +0.04),
        (16, -0.04),
        (17, +0.04),
        (18, -0.04),
        (19, +0.04),
        (4, +0.50),
        (9, +0.50),
        (9, -0.50),
    ):
        if row >= size:
            break
        probe = base_vector.copy()
        probe[index] += delta
        population[row] = np.clip(probe, lower, upper)
        row += 1
    return population


def hard_failure_count(episode: Mapping[str, Any]) -> int:
    count = 0
    for value in episode["hard_safety_failures"].values():
        count += int(value) if isinstance(value, (int, np.integer)) else 1
    return count


def fast_cost(turn: TurnDefinition, episodes: Sequence[Mapping[str, Any]]) -> float:
    costs = []
    for episode in episodes:
        vx, vy, _ = episode["mean_local_velocity_xyz"]
        wz = episode["mean_local_angular_velocity_xyz"][2]
        signed_yaw = float(np.sign(turn.command[2]) * wz)
        motion = (
            2200.0 * (vx - turn.command[0]) ** 2
            + 25000.0 * max(0.0, vx - turn.minimum_reverse_vx) ** 2
            + 1600.0 * (wz - turn.command[2]) ** 2
            + 30000.0
            * max(0.0, turn.minimum_signed_yaw_rate - signed_yaw) ** 2
            + 700.0 * vy**2
            + 25.0 * max(0.0, 0.90 - episode["minimum_upright"])
        )
        hard = 0.0
        if not episode["safety_passed"]:
            hard = 1_000_000.0 + 1000.0 * hard_failure_count(episode)
        costs.append(float(motion + hard))
    return float(np.mean(costs) + 0.75 * np.max(costs))


def run_fast_screen(
    evaluator: MarginAwareReverseEvaluator,
    parameters: ProfileParameters,
    turn: TurnDefinition,
    seeds: Sequence[int],
    seconds: float,
) -> tuple[float, list[dict[str, Any]]]:
    episodes = [
        evaluator.run_fast_seed(
            parameters,
            int(seed),
            seconds,
            left_knee_extra_upper_margin_rad=LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
        )
        for seed in seeds
    ]
    return fast_cost(turn, episodes), episodes


def compact_fast_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": episode["seed"],
        "fell": episode["fell"],
        "completed_seconds": episode["completed_seconds"],
        "mean_local_velocity_xyz": episode["mean_local_velocity_xyz"],
        "mean_local_angular_velocity_xyz": episode["mean_local_angular_velocity_xyz"],
        "minimum_height_m": episode["minimum_height_m"],
        "minimum_upright": episode["minimum_upright"],
        "safety_passed": episode["safety_passed"],
        "hard_safety_failures": episode["hard_safety_failures"],
        "physics_substep_qpos_limit_violations": episode[
            "physics_substep_qpos_limit_violations"
        ],
        "maximum_physics_substep_qpos_excess_rad": episode[
            "maximum_physics_substep_qpos_excess_rad"
        ],
    }


def routed_pilot(
    evaluator: MarginAwareReverseEvaluator,
    base: ProfileParameters,
    candidate: ProfileParameters,
    turn: TurnDefinition,
    seeds: Sequence[int],
    seconds: float,
) -> dict[str, Any]:
    core = evaluator.evaluator
    core.backward_gait_scales = base.amplitude_scales.copy()
    core.backward_gait_biases = base.bias_offsets.copy()
    core.backward_phase_rate = float(base.phase_rate)
    core.backward_residual_scale = 0.0
    core.backward_turn_profiles[turn.direction] = (
        candidate.amplitude_scales.copy(),
        candidate.bias_offsets.copy(),
        float(candidate.phase_rate),
    )
    core.backward_turn_minimum_yaw = 0.0
    core.backward_turn_minimum_blend = 0.0
    core.backward_turn_maximum_blend = 1.0
    bank = SharedV22PolicyBank(core)
    simulator = PerJointCapRoutedSimulator(
        core,
        bank,
        evaluator.mujoco,
        evaluator.runtime,
        leg_target_margin_rad=TARGET_MARGIN_RAD,
        target_slew_rate_rad_s=TARGET_SLEW_RAD_S,
        diagnostic_noncontract_safety=True,
        left_knee_extra_upper_margin_rad=LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
    )
    thresholds = AcceptanceThresholds(
        minimum_signed_linear_progress_fraction=MINIMUM_PROGRESS_FRACTION,
        minimum_signed_yaw_progress_fraction=MINIMUM_PROGRESS_FRACTION,
    )
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        run = simulator.run_schedule(
            (
                (
                    turn.expert,
                    turn.command,
                    seconds,
                    None,
                    turn.expert,
                    "compound",
                ),
            ),
            seed=int(seed),
            joint_noise_scale=evaluator.args.initial_joint_noise_scale,
            initial_base_speed=evaluator.args.initial_base_speed,
            warmup_seconds=evaluator.args.pilot_warmup_seconds,
        )
        segment = run["segments"][0]
        acceptance = segment_acceptance(
            segment,
            thresholds,
            require_gait_quality=bool(evaluator.args.require_gait_quality),
        )
        reset_passed = bool(run["reset_qpos_audit"]["passed"])
        startup = run["control_first_startup_audit"]
        startup_passed = bool(
            startup["passed"]
            and startup["physics_steps_before_control"] == 0
            and startup["exactly_one_guard_call_for_first_tick"]
            and startup["guard_calls_for_first_tick"] == 1
            and startup["head_target_peak_rad"] == 0.0
        )
        audit = segment["safety_audit"]
        substep = segment["physics_substep_audit"]
        explicit_hard_checks = {
            "no_fall": not bool(segment["fell"]),
            "completed": bool(segment["completed"]),
            "reset_qpos_safe": reset_passed,
            "control_first_startup": startup_passed,
            "all_substeps_audited": (
                int(substep["sample_count"])
                == int(segment["completed_physics_substeps"])
                == int(segment["expected_physics_substeps"])
            ),
            "substep_qpos_safe": int(substep["qpos_limit_violations"]) == 0,
            "substep_finite": int(substep["nonfinite_state_samples"]) == 0,
            "substep_height_safe": int(substep["height_fall_samples"]) == 0,
            "substep_upright_safe": int(substep["upright_fall_samples"]) == 0,
            "target_safe": int(audit["applied_target_limit_violations"]) == 0,
            "target_inside_margin": int(audit["desired_target_margin_violations"]) == 0,
            "startup_margin_only": int(audit["unauthorized_applied_target_margin_violations"]) == 0,
            "target_slew_safe": int(audit["target_slew_violations"]) == 0,
            "control_rate_qpos_safe": int(audit["qpos_limit_violations"]) == 0,
            "finite": int(audit["nonfinite_sample_count"]) == 0,
            "head_action_zero": float(audit["applied_head_action_peak"]) == 0.0,
            "head_target_zero": float(audit["head_target_peak_rad"]) == 0.0,
            "atomic_endpoint_exact": bool(acceptance["checks"]["atomic_endpoint_exact"]),
            "expected_expert": bool(acceptance["checks"]["steady_route_expected_expert"]),
            "expected_policy_role": bool(
                acceptance["checks"]["steady_route_expected_policy_role"]
            ),
            "signed_linear_progress_30pct": bool(
                acceptance["checks"]["signed_linear_progress"]
            ),
            "signed_yaw_progress_30pct": bool(
                acceptance["checks"]["signed_yaw_progress"]
            ),
        }
        passed = bool(acceptance["passed"] and all(explicit_hard_checks.values()))
        episodes.append(
            {
                "seed": int(seed),
                "passed": passed,
                "acceptance": acceptance,
                "explicit_hard_checks": explicit_hard_checks,
                "reset_qpos_audit": run["reset_qpos_audit"],
                "control_first_startup_audit": startup,
                "segment": segment,
            }
        )
    return {
        "passed": all(episode["passed"] for episode in episodes),
        "episode_count": len(episodes),
        "seconds_per_seed": float(seconds),
        "fixed_perturb_seeds": [int(seed) for seed in seeds],
        "episodes": episodes,
        "policy_inference_counts": dict(sorted(bank.inference_counts.items())),
    }


def candidate_payload(
    turn: TurnDefinition,
    parameters: ProfileParameters,
    base_path: Path,
    pilot: Mapping[str, Any],
    evidence_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "openduckmini_margin_aware_atomic_reverse_turn_profile_candidate",
        "release_id": f"optimized_reverse_turn_{turn.key}_margin050_slew200_candidate_v1",
        "status": "PILOT_PASS_NOT_ADOPTED_PENDING_5X15_AND_20X30",
        "hardware_deployment": "PROHIBITED",
        "simulation_only": True,
        "direction": turn.key,
        "atomic_command": list(turn.command),
        "parameters": parameters.as_json(),
        "composition": {
            "straight_reverse_base_profile": str(base_path.resolve()),
            "straight_reverse_base_sha256": sha256_file(base_path.resolve()),
            "backward_residual_scale": 0.0,
            "leg_target_margin_rad": TARGET_MARGIN_RAD,
            "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
            "left_knee_extra_upper_margin_rad": LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
            "positive_noise_reset_qpos_inward_margin_rad": RESET_QPOS_MARGIN_RAD,
            "initial_target_order": "guarded_control_before_first_physics_decimation",
            "head_target_indices": [5, 6, 7, 8],
            "head_target_value": 0.0,
        },
        "pilot_evidence": {
            "source": str(evidence_path.resolve()),
            "source_sha256": sha256_file(evidence_path.resolve()),
            "seconds_per_seed": pilot["seconds_per_seed"],
            "fixed_perturb_seeds": pilot["fixed_perturb_seeds"],
            "episode_count": pilot["episode_count"],
            "all_passed": pilot["passed"],
            "minimum_signed_linear_progress_fraction": MINIMUM_PROGRESS_FRACTION,
            "minimum_signed_yaw_progress_fraction": MINIMUM_PROGRESS_FRACTION,
            "all_physics_substeps_audited": True,
        },
        "adoption": {
            "status": "NOT_ADOPTED_PENDING_5X15_AND_20X30",
            "hardware_deployment": "PROHIBITED",
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    # The simulation venv owns SciPy; keeping this import out of module scope
    # lets the repository's lightweight system-Python unit tests validate the
    # frozen contract helpers without installing the optimizer dependency.
    from scipy.optimize import differential_evolution

    args = parse_args(argv)
    base_path = args.base_profile.resolve()
    base, _ = validate_base_profile(base_path)
    args.generated_root = args.generated_root.resolve()
    args.policy = args.policy.resolve()
    args.warmup_seconds = args.search_warmup_seconds
    args.minimum_reverse_vx = -0.009
    evaluator = MarginAwareReverseEvaluator(args)
    seeds = tuple(int(seed) for seed in args.perturb_seeds)
    bounds = search_bounds(base)
    all_results: dict[str, Any] = {}
    emitted_candidates: dict[str, Path] = {}

    for direction_index, direction in enumerate(args.directions):
        turn = TURNS[direction]
        cache: dict[tuple[float, ...], tuple[float, ProfileParameters, list[dict[str, Any]]]] = {}
        ranked: list[tuple[float, ProfileParameters, list[dict[str, Any]]]] = []

        def evaluate_vector(vector: Sequence[float]) -> tuple[float, ProfileParameters, list[dict[str, Any]]]:
            key = tuple(np.round(np.asarray(vector, dtype=np.float64), 8))
            if key not in cache:
                effective = vector_to_parameters(vector)
                cost, episodes = run_fast_screen(
                    evaluator, effective, turn, seeds, args.search_seconds
                )
                cache[key] = (cost, effective, episodes)
                ranked.append(cache[key])
            return cache[key]

        def objective(vector: np.ndarray) -> float:
            return evaluate_vector(vector)[0]

        generation = 0

        def callback(intermediate_result: Any) -> None:
            nonlocal generation
            generation += 1
            cost, _, episodes = evaluate_vector(intermediate_result.x)
            vx = [episode["mean_local_velocity_xyz"][0] for episode in episodes]
            wz = [episode["mean_local_angular_velocity_xyz"][2] for episode in episodes]
            print(
                f"direction={turn.key} generation={generation} cost={cost:.4f} "
                f"worst_vx={max(vx):+.4f} signed_worst_yaw="
                f"{min(np.sign(turn.command[2]) * np.asarray(wz)):+.4f} "
                f"hard_safe={all(episode['safety_passed'] for episode in episodes)}",
                flush=True,
            )

        population = initial_population(
            bounds,
            base,
            args.popsize,
            args.search_seed + direction_index * 1000,
        )
        seed_profile_path = getattr(args, f"{direction}_seed_profile")
        seed_profile_evidence = None
        if seed_profile_path is not None:
            seed_profile_path = seed_profile_path.resolve()
            loaded_seed = load_profile(seed_profile_path)
            effective_seed = profile_blend_for_direction(
                base, loaded_seed, turn.direction
            )
            lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
            upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
            population[1] = np.clip(
                parameters_to_vector(effective_seed), lower, upper
            )
            seed_profile_evidence = {
                "path": str(seed_profile_path),
                "sha256": sha256_file(seed_profile_path),
            }

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=args.search_seed + direction_index * 1000,
            workers=1,
            updating="immediate",
            polish=False,
            callback=callback,
            tol=1e-5,
            init=population,
            x0=parameters_to_vector(base),
        )
        evaluate_vector(result.x)
        unique_ranked = sorted(ranked, key=lambda item: item[0])
        finalists = []
        selected: tuple[ProfileParameters, dict[str, Any]] | None = None
        for rank, (cost, effective, fast_episodes) in enumerate(
            unique_ranked[: args.top_candidates], start=1
        ):
            loaded = effective_to_loaded_profile(base, effective, turn.direction)
            # Amplitude scales and phase are runtime parameters and must remain
            # non-negative after inversion of the right-turn 0.90 blend.
            if np.any(loaded.amplitude_scales < 0.0) or loaded.phase_rate <= 0.0:
                finalists.append(
                    {
                        "rank": rank,
                        "search_cost": cost,
                        "parameters": loaded.as_json(),
                        "pilot": {"passed": False, "rejection": "invalid loaded profile"},
                    }
                )
                continue
            pilot = routed_pilot(
                evaluator, base, loaded, turn, seeds, args.pilot_seconds
            )
            finalists.append(
                {
                    "rank": rank,
                    "search_cost": cost,
                    "effective_parameters": effective.as_json(),
                    "parameters": loaded.as_json(),
                    "fast_episodes": [compact_fast_episode(item) for item in fast_episodes],
                    "pilot": pilot,
                }
            )
            print(
                f"direction={turn.key} finalist={rank} pilot_passed={pilot['passed']}",
                flush=True,
            )
            if pilot["passed"] and selected is None:
                selected = (loaded, pilot)
                break

        all_results[turn.key] = {
            "status": "PILOT_PASS" if selected is not None else "NO_PILOT_PASS",
            "command": list(turn.command),
            "minimum_reverse_vx": turn.minimum_reverse_vx,
            "minimum_signed_yaw_rate": turn.minimum_signed_yaw_rate,
            "optimizer": {
                "success": bool(result.success),
                "message": str(result.message),
                "evaluations": int(result.nfev),
                "generations": generation,
                "seed_profile": seed_profile_evidence,
            },
            "finalists": finalists,
        }
        if selected is not None:
            parameters, pilot = selected
            emitted_candidates[turn.key] = args.candidate_dir / (
                f"optimized_reverse_turn_{turn.key}_margin050_slew200_candidate_v1.json"
            )

    overall_passed = set(emitted_candidates) == set(args.directions)
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_margin_aware_reverse_turn_search",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PILOT_PASS" if overall_passed else "REJECTED_NO_COMPLETE_SAFE_PAIR",
        "hardware_deployment": "PROHIBITED",
        "simulation_only": True,
        "configuration": {
            "directions": list(args.directions),
            "search_seconds_per_seed": args.search_seconds,
            "pilot_seconds_per_seed": args.pilot_seconds,
            "pilot_warmup_seconds": args.pilot_warmup_seconds,
            "fixed_perturb_seeds": list(seeds),
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "initial_base_speed_max_mps": args.initial_base_speed,
            "reset_qpos_inward_margin_rad": RESET_QPOS_MARGIN_RAD,
            "leg_target_margin_rad": TARGET_MARGIN_RAD,
            "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
            "left_knee_extra_upper_margin_rad": LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
            "minimum_signed_linear_progress_fraction": MINIMUM_PROGRESS_FRACTION,
            "minimum_signed_yaw_progress_fraction": MINIMUM_PROGRESS_FRACTION,
            "backward_residual_scale": 0.0,
            "control_first": True,
            "all_physics_substeps_audited": True,
        },
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "base_profile": {
                "path": str(base_path),
                "sha256": sha256_file(base_path),
            },
            "policy": {
                "path": str(args.policy),
                "sha256": sha256_file(args.policy),
            },
            "generated_assets": evaluator.asset_evidence,
            "model_contract": evaluator.model_evidence,
        },
        "directions": all_results,
        "candidate_files": {
            key: str(path.resolve()) for key, path in emitted_candidates.items()
        },
        "adoption": {
            "status": "NOT_ADOPTED_PENDING_5X15_AND_20X30" if overall_passed else "NO_CANDIDATE",
            "hardware_deployment": "PROHIBITED",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Candidate profiles are written only after the immutable evidence artifact
    # exists, so each one can point back to a concrete pilot result.
    if emitted_candidates:
        args.candidate_dir.mkdir(parents=True, exist_ok=True)
        for direction, path in emitted_candidates.items():
            finalist = next(
                item
                for item in all_results[direction]["finalists"]
                if item.get("pilot", {}).get("passed")
            )
            parameters = vector_to_parameters(
                list(finalist["parameters"]["joint_amplitude_scales"])
                + list(finalist["parameters"]["joint_bias_offsets"])
                + [finalist["parameters"]["phase_rate"]]
            )
            path.write_text(
                json.dumps(
                    candidate_payload(
                        TURNS[direction],
                        parameters,
                        base_path,
                        finalist["pilot"],
                        args.output,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": payload["status"],
                "directions": {
                    key: value["status"] for key, value in all_results.items()
                },
                "candidate_files": payload["candidate_files"],
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
