"""Isolate and screen a left-knee recovery cap after backward-family exits.

This diagnostic deliberately does not modify the central routed evaluator or
the package runtime.  It replays the exact five-seed transition prefix that
exposed the left-knee overshoot after ``reverse_turn_left`` and compares small,
time-bounded recovery caps.  Every result is simulation-only and hardware
deployment remains PROHIBITED.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
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

from safe_gait_experts.contract import SAFE_JOINT_LIMITS  # noqa: E402
from safe_gait_experts.routed_evaluation import (  # noqa: E402
    AcceptanceThresholds,
    segment_acceptance,
    sha256_file,
    transition_schedule,
)
from optimize_margin_aware_reverse import (  # noqa: E402
    DEFAULT_GENERATED_ROOT,
    DEFAULT_POLICY,
    MarginAwareReverseEvaluator,
    PerJointCapRoutedSimulator,
    SharedV22PolicyBank,
    TARGET_MARGIN_RAD,
    TARGET_SLEW_RAD_S,
)


CONTROL_DT_S = 0.02
STRAIGHT_ENTRY_PHASE_INDEX = 6.0
TURN_ENTRY_PHASE_INDEX = 4.0
PHASE_ENTRY_INDICES = {
    "reverse": STRAIGHT_ENTRY_PHASE_INDEX,
    "reverse_turn_left": TURN_ENTRY_PHASE_INDEX,
    "reverse_turn_right": TURN_ENTRY_PHASE_INDEX,
}
FIXED_SEEDS = tuple(22_260_808 + index for index in range(5))
STAND_SECONDS = 5.0
MOVING_SECONDS = 15.0
WARMUP_SECONDS = 1.5
INITIAL_JOINT_NOISE_SCALE = 1.0
INITIAL_BASE_SPEED_MPS = 0.10
RESET_QPOS_INWARD_MARGIN_RAD = 0.005
BASE_REVERSE_LEFT_KNEE_EXTRA_MARGIN_RAD = 0.0125
GRADUAL_RELEASE_SECONDS = 0.25
PREFIX_SEGMENT_COUNT = 7
EXPECTED_FAILURE_SEED = 22_260_811
EXPECTED_FAILURE_QPOS_VIOLATIONS = 9
EXPECTED_FAILURE_MAX_EXCESS_RAD = 0.0009194232089873022

DEFAULT_STRAIGHT_PROFILE = (
    EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_candidate_v3.json"
)
DEFAULT_LEFT_PROFILE = (
    EXP_ROOT
    / "artifacts"
    / "reverse_turn_candidates_v1"
    / "optimized_reverse_turn_left_margin050_slew200_candidate_v1.json"
)
DEFAULT_RIGHT_PROFILE = (
    EXP_ROOT
    / "artifacts"
    / "reverse_turn_candidates_v1"
    / "optimized_reverse_turn_right_margin050_slew200_candidate_v1.json"
)
DEFAULT_BASELINE = (
    EXP_ROOT
    / "artifacts"
    / "routed_combined_reverse_bank_phase644_control_first_5x15_v1.json"
)
DEFAULT_OUTPUT = (
    EXP_ROOT / "artifacts" / "backward_exit_recovery_phase644_5seed_prefix_v1.json"
)


@dataclass(frozen=True)
class RecoveryStrategy:
    """A discrete-time inward cap applied after backward feedforward exits."""

    name: str
    cap_rad: float
    hold_seconds: float
    release_seconds: float

    def validate(self) -> None:
        values = (self.cap_rad, self.hold_seconds, self.release_seconds)
        if not all(np.isfinite(values)) or any(value < 0.0 for value in values):
            raise ValueError("recovery cap and durations must be finite and non-negative")
        if self.cap_rad > 0.020:
            raise ValueError("diagnostic recovery cap must not exceed 0.020 rad")
        if self.cap_rad == 0.0 and (self.hold_seconds or self.release_seconds):
            raise ValueError("zero-cap baseline cannot declare recovery duration")
        if self.cap_rad > 0.0 and self.hold_seconds <= 0.0:
            raise ValueError("positive recovery cap requires a positive hold")

    @property
    def hold_ticks(self) -> int:
        return seconds_to_safe_ticks(self.hold_seconds)

    @property
    def release_ticks(self) -> int:
        return seconds_to_safe_ticks(self.release_seconds)

    @property
    def applied_hold_seconds(self) -> float:
        return self.hold_ticks * CONTROL_DT_S

    @property
    def applied_release_seconds(self) -> float:
        return self.release_ticks * CONTROL_DT_S

    @property
    def restriction_area_rad_s(self) -> float:
        # A linear release has half the full-cap area.
        return self.cap_rad * (
            self.applied_hold_seconds + 0.5 * self.applied_release_seconds
        )

    def extra_margin_for_tick(self, tick: int) -> float:
        if tick < 0:
            raise ValueError("recovery tick must be non-negative")
        if tick < self.hold_ticks:
            return self.cap_rad
        release_tick = tick - self.hold_ticks
        if self.release_ticks and release_tick < self.release_ticks:
            remaining_fraction = 1.0 - (release_tick + 1) / self.release_ticks
            return self.cap_rad * max(0.0, remaining_fraction)
        return 0.0

    def as_contract(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "control_dt_s": CONTROL_DT_S,
                "duration_quantization": "ceil(seconds / control_dt)",
                "hold_ticks": self.hold_ticks,
                "release_ticks": self.release_ticks,
                "applied_hold_seconds": self.applied_hold_seconds,
                "applied_release_seconds": self.applied_release_seconds,
                "release_shape": (
                    "instant" if self.release_ticks == 0 else "linear_to_zero"
                ),
                "restriction_area_rad_s": self.restriction_area_rad_s,
            }
        )
        return payload


def seconds_to_safe_ticks(seconds: float, *, dt: float = CONTROL_DT_S) -> int:
    """Quantize a minimum requested duration upward to whole control ticks."""

    seconds_value = float(seconds)
    dt_value = float(dt)
    if not np.isfinite(seconds_value) or seconds_value < 0.0:
        raise ValueError("duration must be finite and non-negative")
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("control dt must be finite and positive")
    if seconds_value == 0.0:
        return 0
    return int(math.ceil(seconds_value / dt_value - 1e-12))


def recovery_strategy_bank() -> tuple[RecoveryStrategy, ...]:
    """Return the frozen comparison bank, including escalation controls."""

    strategies = [RecoveryStrategy("baseline_immediate_release", 0.0, 0.0, 0.0)]
    for cap_rad in (0.0125, 0.015, 0.020):
        cap_label = f"{int(round(cap_rad * 10_000)):04d}"
        for hold_seconds in (0.25, 0.50, 1.00):
            hold_label = f"{int(round(hold_seconds * 1000)):04d}ms"
            strategies.extend(
                (
                    RecoveryStrategy(
                        f"cap{cap_label}_hold{hold_label}_instant",
                        cap_rad,
                        hold_seconds,
                        0.0,
                    ),
                    RecoveryStrategy(
                        f"cap{cap_label}_hold{hold_label}_linear0250ms",
                        cap_rad,
                        hold_seconds,
                        GRADUAL_RELEASE_SECONDS,
                    ),
                )
            )
    for strategy in strategies:
        strategy.validate()
    return tuple(strategies)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the exact phase-6/4/4 transition prefix and compare "
            "left-knee recovery caps after backward-family exits."
        )
    )
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--straight-profile", type=Path, default=DEFAULT_STRAIGHT_PROFILE)
    parser.add_argument("--left-profile", type=Path, default=DEFAULT_LEFT_PROFILE)
    parser.add_argument("--right-profile", type=Path, default=DEFAULT_RIGHT_PROFILE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


class BackwardExitRecoverySimulator(PerJointCapRoutedSimulator):
    """Apply one temporary left-knee cap after every backward-family exit."""

    def __init__(self, *args: Any, recovery_strategy: RecoveryStrategy, **kwargs: Any):
        recovery_strategy.validate()
        super().__init__(*args, **kwargs)
        self.recovery_strategy = recovery_strategy
        self._previous_backward_active = False
        self._recovery_tick: int | None = None
        self._global_target_tick = 0
        self._seed: int | None = None
        self.recovery_events: list[dict[str, Any]] = []

    def begin_diagnostic_run(self, seed: int) -> None:
        self._previous_backward_active = False
        self._recovery_tick = None
        self._global_target_tick = 0
        self._seed = int(seed)

    def _start_recovery(self) -> None:
        self._recovery_tick = 0
        self.recovery_events.append(
            {
                "seed": self._seed,
                "start_global_control_tick": self._global_target_tick,
                "cap_rad": self.recovery_strategy.cap_rad,
                "hold_ticks": self.recovery_strategy.hold_ticks,
                "release_ticks": self.recovery_strategy.release_ticks,
                "positive_cap_tick_count": 0,
                "minimum_extra_margin_rad": None,
                "maximum_extra_margin_rad": 0.0,
            }
        )

    def _policy_target(
        self,
        applied_action: np.ndarray,
        effective_command: np.ndarray,
        phase_index: float,
        default: np.ndarray,
    ) -> np.ndarray:
        backward_active = bool(float(effective_command[0]) < -0.02)
        if self._previous_backward_active and not backward_active:
            self._start_recovery()
        if backward_active:
            self._recovery_tick = None

        targets = super()._policy_target(
            applied_action, effective_command, phase_index, default
        )
        if not backward_active and self._recovery_tick is not None:
            extra_margin = self.recovery_strategy.extra_margin_for_tick(
                self._recovery_tick
            )
            if extra_margin > 0.0:
                recovery_upper = (
                    float(SAFE_JOINT_LIMITS["left_knee"][1])
                    - TARGET_MARGIN_RAD
                    - extra_margin
                )
                targets[self.left_knee_index] = min(
                    targets[self.left_knee_index], recovery_upper
                )
                event = self.recovery_events[-1]
                event["positive_cap_tick_count"] += 1
                prior_minimum = event["minimum_extra_margin_rad"]
                event["minimum_extra_margin_rad"] = (
                    extra_margin
                    if prior_minimum is None
                    else min(float(prior_minimum), extra_margin)
                )
                event["maximum_extra_margin_rad"] = max(
                    float(event["maximum_extra_margin_rad"]), extra_margin
                )
            self._recovery_tick += 1
            if self.recovery_strategy.extra_margin_for_tick(self._recovery_tick) == 0.0:
                self.recovery_events[-1]["end_global_control_tick_exclusive"] = (
                    self._global_target_tick + 1
                )
                self._recovery_tick = None

        self._previous_backward_active = backward_active
        self._global_target_tick += 1
        return targets


def exact_prefix_schedule() -> tuple[tuple[Any, ...], ...]:
    """Materialize the central schedule through left-turn recovery stand."""

    return transition_schedule(MOVING_SECONDS, STAND_SECONDS)[:PREFIX_SEGMENT_COUNT]


def _segment_hard_gate(segment: Mapping[str, Any]) -> dict[str, bool]:
    acceptance = segment_acceptance(segment)
    checks = acceptance["checks"]
    audit = segment["safety_audit"]
    substep = segment["physics_substep_audit"]
    routing = segment["routing"]
    return {
        "completed": bool(segment["completed"]),
        "no_fall": not bool(segment["fell"]),
        "all_physics_substeps_audited": (
            int(substep["sample_count"])
            == int(segment["completed_physics_substeps"])
            == int(segment["expected_physics_substeps"])
        ),
        "substep_qpos_safe": int(substep["qpos_limit_violations"]) == 0,
        "control_qpos_safe": int(audit["qpos_limit_violations"]) == 0,
        "substep_finite": int(substep["nonfinite_state_samples"]) == 0,
        "substep_fall_free": (
            int(substep["height_fall_samples"]) == 0
            and int(substep["upright_fall_samples"]) == 0
        ),
        "targets_safe": (
            int(audit["applied_target_limit_violations"]) == 0
            and int(audit["desired_target_margin_violations"]) == 0
            and int(audit["unauthorized_applied_target_margin_violations"]) == 0
        ),
        "target_slew_safe": int(audit["target_slew_violations"]) == 0,
        "head_locked": (
            float(audit["applied_head_action_peak"]) == 0.0
            and float(audit["head_target_peak_rad"]) == 0.0
        ),
        "route_exact": (
            bool(checks["steady_route_expected_expert"])
            and bool(checks["steady_route_expected_policy_role"])
            and bool(checks["steady_route_sample_count"])
            and bool(checks["prohibited_experts_absent"])
            and bool(checks["atomic_endpoint_exact"])
            and int(routing["command_clip_events"]) == 0
            and bool(checks["reverse_entry_phase_audit"])
        ),
    }


def _compact_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    metrics = segment["metrics"]
    audit = segment["safety_audit"]
    substep = segment["physics_substep_audit"]
    acceptance = segment_acceptance(segment)
    hard_checks = _segment_hard_gate(segment)
    return {
        "name": segment["name"],
        "passed_hard_gate": all(hard_checks.values()),
        "hard_checks": hard_checks,
        "standard_acceptance_passed": bool(acceptance["passed"]),
        "standard_acceptance_checks": acceptance["checks"],
        "fell": bool(segment["fell"]),
        "completed": bool(segment["completed"]),
        "mean_local_velocity_xyz": metrics["mean_local_velocity_xyz"],
        "mean_local_yaw_rate": float(metrics["mean_local_yaw_rate"]),
        "planar_displacement_m": float(metrics["planar_displacement"]),
        "signed_linear_progress_fraction": (
            float(metrics["projected_primary_velocity"])
            / max(float(metrics["commanded_linear_speed"]), 1e-12)
        ),
        "signed_yaw_progress_fraction": (
            np.sign(float(segment["command"][2]))
            * float(metrics["mean_local_yaw_rate"])
            / max(abs(float(segment["command"][2])), 1e-12)
        ),
        "physics_substep_audit": {
            key: substep[key]
            for key in (
                "sample_count",
                "qpos_limit_violations",
                "maximum_qpos_excess_rad",
                "joint_qpos_max_rad",
                "joint_qpos_min_rad",
                "nonfinite_state_samples",
                "height_fall_samples",
                "upright_fall_samples",
                "minimum_height_m",
                "minimum_upright",
            )
        },
        "target_head_slew_audit": {
            key: audit[key]
            for key in (
                "applied_target_limit_violations",
                "desired_target_margin_violations",
                "unauthorized_applied_target_margin_violations",
                "target_slew_violations",
                "qpos_limit_violations",
                "nonfinite_sample_count",
                "applied_head_action_peak",
                "head_target_peak_rad",
                "maximum_target_slew_rate_rad_per_s",
            )
        },
    }


def _run_strategy(
    evaluator: MarginAwareReverseEvaluator,
    strategy: RecoveryStrategy,
) -> dict[str, Any]:
    core = evaluator.evaluator
    bank = SharedV22PolicyBank(core)
    simulator = BackwardExitRecoverySimulator(
        core,
        bank,
        evaluator.mujoco,
        evaluator.runtime,
        leg_target_margin_rad=TARGET_MARGIN_RAD,
        target_slew_rate_rad_s=TARGET_SLEW_RAD_S,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=(
            BASE_REVERSE_LEFT_KNEE_EXTRA_MARGIN_RAD
        ),
        diagnostic_reverse_entry_phase_indices=PHASE_ENTRY_INDICES,
        recovery_strategy=strategy,
    )
    episodes: list[dict[str, Any]] = []
    for seed in FIXED_SEEDS:
        simulator.begin_diagnostic_run(seed)
        event_start = len(simulator.recovery_events)
        run = simulator.run_schedule(
            exact_prefix_schedule(),
            seed=seed,
            joint_noise_scale=INITIAL_JOINT_NOISE_SCALE,
            initial_base_speed=INITIAL_BASE_SPEED_MPS,
            warmup_seconds=WARMUP_SECONDS,
        )
        compact = [_compact_segment(segment) for segment in run["segments"]]
        turn = compact[-2]
        stand = compact[-1]
        all_segment_hard_gates = all(item["passed_hard_gate"] for item in compact)
        turn_progress_passed = bool(
            turn["standard_acceptance_checks"]["signed_linear_progress"]
            and turn["standard_acceptance_checks"]["signed_yaw_progress"]
            and turn["standard_acceptance_checks"]["atomic_endpoint_exact"]
        )
        stand_drift_passed = bool(
            stand["standard_acceptance_checks"]["stop_drift"]
        )
        episode_passed = bool(
            len(compact) == PREFIX_SEGMENT_COUNT
            and all_segment_hard_gates
            and turn_progress_passed
            and stand_drift_passed
        )
        episodes.append(
            {
                "seed": seed,
                "passed": episode_passed,
                "all_prefix_segment_hard_gates_passed": all_segment_hard_gates,
                "preceding_reverse_turn_left_progress_passed": turn_progress_passed,
                "recovery_stand_drift_passed": stand_drift_passed,
                "reset_qpos_audit": run["reset_qpos_audit"],
                "control_first_startup_audit": run["control_first_startup_audit"],
                "recovery_events": simulator.recovery_events[event_start:],
                "segments": compact,
            }
        )

    stand_rows = [episode["segments"][-1] for episode in episodes]
    return {
        "strategy": strategy.as_contract(),
        "passed": all(episode["passed"] for episode in episodes),
        "episode_count": len(episodes),
        "passed_episode_count": sum(episode["passed"] for episode in episodes),
        "total_prefix_qpos_limit_violations": sum(
            segment["physics_substep_audit"]["qpos_limit_violations"]
            for episode in episodes
            for segment in episode["segments"]
        ),
        "recovery_stand_qpos_limit_violations": sum(
            row["physics_substep_audit"]["qpos_limit_violations"]
            for row in stand_rows
        ),
        "maximum_recovery_stand_qpos_excess_rad": max(
            row["physics_substep_audit"]["maximum_qpos_excess_rad"]
            for row in stand_rows
        ),
        "maximum_recovery_stand_planar_drift_m": max(
            row["planar_displacement_m"] for row in stand_rows
        ),
        "minimum_preceding_turn_linear_progress_fraction": min(
            episode["segments"][-2]["signed_linear_progress_fraction"]
            for episode in episodes
        ),
        "minimum_preceding_turn_yaw_progress_fraction": min(
            episode["segments"][-2]["signed_yaw_progress_fraction"]
            for episode in episodes
        ),
        "episodes": episodes,
        "policy_inference_counts": dict(sorted(bank.inference_counts.items())),
    }


def _load_baseline_contract(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    configuration = payload["configuration"]
    expected = {
        "episodes": 5,
        "seconds": MOVING_SECONDS,
        "transition_seconds": MOVING_SECONDS,
        "transition_stand_seconds": STAND_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "initial_joint_noise_scale": INITIAL_JOINT_NOISE_SCALE,
        "initial_base_speed": INITIAL_BASE_SPEED_MPS,
        "leg_target_margin_rad": TARGET_MARGIN_RAD,
        "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
        "left_knee_extra_upper_margin_rad": (
            BASE_REVERSE_LEFT_KNEE_EXTRA_MARGIN_RAD
        ),
        "diagnostic_unadopted_reverse_entry_phase_indices": PHASE_ENTRY_INDICES,
    }
    mismatches = {
        key: {"expected": value, "actual": configuration.get(key)}
        for key, value in expected.items()
        if configuration.get(key) != value
    }
    if mismatches:
        raise ValueError(f"baseline configuration mismatch: {mismatches}")
    episodes = payload["suites"]["transitions"]["episodes"]
    seeds = tuple(int(episode["seed"]) for episode in episodes)
    if seeds != FIXED_SEEDS:
        raise ValueError(f"baseline transition seeds changed: {seeds}")
    failure = episodes[FIXED_SEEDS.index(EXPECTED_FAILURE_SEED)]["segments"][6]
    substep = failure["physics_substep_audit"]
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "configuration_matches_exact_replay": True,
        "fixed_seeds": list(seeds),
        "known_failure": {
            "seed": EXPECTED_FAILURE_SEED,
            "segment": failure["name"],
            "joint": "left_knee",
            "safe_upper_rad": float(SAFE_JOINT_LIMITS["left_knee"][1]),
            "observed_max_qpos_rad": float(
                substep["joint_qpos_max_rad"]["left_knee"]
            ),
            "qpos_limit_violations": int(substep["qpos_limit_violations"]),
            "maximum_qpos_excess_rad": float(substep["maximum_qpos_excess_rad"]),
        },
    }


def _baseline_replay_matches(
    baseline: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    episode = replay["episodes"][FIXED_SEEDS.index(EXPECTED_FAILURE_SEED)]
    stand = episode["segments"][-1]["physics_substep_audit"]
    source = baseline["known_failure"]
    checks = {
        "seed": int(episode["seed"]) == EXPECTED_FAILURE_SEED,
        "segment": episode["segments"][-1]["name"]
        == "transition_stand_after_reverse_turn_left",
        "qpos_limit_violations": int(stand["qpos_limit_violations"])
        == int(source["qpos_limit_violations"])
        == EXPECTED_FAILURE_QPOS_VIOLATIONS,
        "maximum_qpos_excess_rad": float(stand["maximum_qpos_excess_rad"])
        == float(source["maximum_qpos_excess_rad"])
        == EXPECTED_FAILURE_MAX_EXCESS_RAD,
        "left_knee_max_qpos_rad": float(stand["joint_qpos_max_rad"]["left_knee"])
        == float(source["observed_max_qpos_rad"]),
        "other_seed_recovery_stands_qpos_safe": all(
            candidate["segments"][-1]["physics_substep_audit"][
                "qpos_limit_violations"
            ]
            == 0
            for candidate in replay["episodes"]
            if candidate["seed"] != EXPECTED_FAILURE_SEED
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _selection_key(result: Mapping[str, Any]) -> tuple[float, float, float, str]:
    strategy = result["strategy"]
    return (
        float(strategy["cap_rad"]),
        float(strategy["restriction_area_rad_s"]),
        float(strategy["applied_hold_seconds"])
        + float(strategy["applied_release_seconds"]),
        str(strategy["name"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = _load_baseline_contract(args.baseline)
    runtime_args = argparse.Namespace(
        generated_root=args.generated_root.resolve(),
        policy=args.policy.resolve(),
        reset_qpos_inward_margin_rad=RESET_QPOS_INWARD_MARGIN_RAD,
        initial_joint_noise_scale=INITIAL_JOINT_NOISE_SCALE,
        initial_base_speed=INITIAL_BASE_SPEED_MPS,
        warmup_seconds=WARMUP_SECONDS,
    )
    evaluator = MarginAwareReverseEvaluator(runtime_args)
    core = evaluator.evaluator
    core.load_backward_profile(args.straight_profile.resolve())
    core.load_backward_turn_profile(1, args.left_profile.resolve())
    core.load_backward_turn_profile(-1, args.right_profile.resolve())
    core.backward_residual_scale = 0.0

    results: list[dict[str, Any]] = []
    for strategy in recovery_strategy_bank():
        print(f"screening {strategy.name}", flush=True)
        result = _run_strategy(evaluator, strategy)
        results.append(result)
        print(
            json.dumps(
                {
                    "strategy": strategy.name,
                    "passed": result["passed"],
                    "stand_qpos_violations": result[
                        "recovery_stand_qpos_limit_violations"
                    ],
                    "max_stand_qpos_excess_rad": result[
                        "maximum_recovery_stand_qpos_excess_rad"
                    ],
                    "max_stand_drift_m": result[
                        "maximum_recovery_stand_planar_drift_m"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    baseline_reproduction = _baseline_replay_matches(baseline, results[0])
    passing = [result for result in results[1:] if result["passed"]]
    selected = min(passing, key=_selection_key) if passing else None
    passed = bool(baseline_reproduction["passed"] and selected is not None)
    script_path = Path(__file__).resolve()
    output = args.output.resolve()
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_backward_exit_recovery_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RECOVERY_SCREEN_PASS" if passed else "RECOVERY_SCREEN_FAIL",
        "hardware_deployment": "PROHIBITED",
        "simulation_only": True,
        "diagnostic_unadopted": True,
        "central_evaluator_modified_by_this_diagnostic": False,
        "configuration": {
            "fixed_seeds": list(FIXED_SEEDS),
            "schedule": [
                {
                    "name": case[0],
                    "command": list(case[1]),
                    "seconds": case[2],
                    "policy_observation_command": (
                        None if case[3] is None else list(case[3])
                    ),
                    "expected_expert": case[4],
                    "expected_policy_role": case[5],
                }
                for case in exact_prefix_schedule()
            ],
            "warmup_seconds": WARMUP_SECONDS,
            "initial_joint_noise_scale": INITIAL_JOINT_NOISE_SCALE,
            "initial_base_speed_max_mps": INITIAL_BASE_SPEED_MPS,
            "positive_noise_reset_qpos_inward_margin_rad": (
                RESET_QPOS_INWARD_MARGIN_RAD
            ),
            "leg_target_margin_rad": TARGET_MARGIN_RAD,
            "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
            "reverse_profile_left_knee_extra_upper_margin_rad": (
                BASE_REVERSE_LEFT_KNEE_EXTRA_MARGIN_RAD
            ),
            "diagnostic_reverse_entry_phase_indices": PHASE_ENTRY_INDICES,
            "recovery_activation": "effective_vx_lt_-0.02_true_to_false",
            "recovery_scope": "every_backward_feedforward_exit",
            "recovery_composition_order": (
                "policy_or_profile -> temporary_left_knee_cap -> "
                "single_frozen_final_target_guard -> physics"
            ),
        },
        "hard_gate": {
            "five_fixed_seeds": True,
            "all_prefix_physics_substeps_audited": True,
            "maximum_qpos_limit_violations": 0,
            "maximum_falls": 0,
            "maximum_target_limit_margin_or_slew_violations": 0,
            "head_action_and_target_exact_zero": True,
            "maximum_route_violations": 0,
            "stand_drift_threshold_m": AcceptanceThresholds().maximum_stop_drift_m,
            "preceding_reverse_turn_minimum_signed_linear_progress_fraction": (
                AcceptanceThresholds().minimum_signed_linear_progress_fraction
            ),
            "preceding_reverse_turn_minimum_signed_yaw_progress_fraction": (
                AcceptanceThresholds().minimum_signed_yaw_progress_fraction
            ),
            "gates_relaxed": False,
        },
        "baseline_source": baseline,
        "baseline_reproduction": baseline_reproduction,
        "strategy_bank": [strategy.as_contract() for strategy in recovery_strategy_bank()],
        "results": results,
        "selection": {
            "passed": selected is not None,
            "rule": (
                "among all hard-gate passes, minimize cap_rad, then integrated "
                "cap-time restriction, then total recovery duration"
            ),
            "selected_strategy": None if selected is None else selected["strategy"],
            "selected_summary": (
                None
                if selected is None
                else {
                    key: selected[key]
                    for key in (
                        "passed_episode_count",
                        "total_prefix_qpos_limit_violations",
                        "recovery_stand_qpos_limit_violations",
                        "maximum_recovery_stand_qpos_excess_rad",
                        "maximum_recovery_stand_planar_drift_m",
                        "minimum_preceding_turn_linear_progress_fraction",
                        "minimum_preceding_turn_yaw_progress_fraction",
                    )
                }
            ),
        },
        "provenance": {
            "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "policy": {
                "path": str(args.policy.resolve()),
                "sha256": sha256_file(args.policy.resolve()),
            },
            "profiles": {
                label: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
                for label, path in (
                    ("straight", args.straight_profile),
                    ("left", args.left_profile),
                    ("right", args.right_profile),
                )
            },
            "generated_assets": evaluator.asset_evidence,
            "model_contract": evaluator.model_evidence,
        },
        "recommendation": (
            "Do not change the central evaluator/package; no recovery strategy passed."
            if selected is None
            else (
                "Implement the selected post-backward left-knee cap centrally, "
                "then rerun the complete 5x15 and frozen 20x30 suites before adoption."
            )
        ),
        "adoption": {
            "status": "NOT_ADOPTED_PENDING_CENTRAL_IMPLEMENTATION_AND_FULL_REQUALIFICATION",
            "hardware_deployment": "PROHIBITED",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": sha256_file(output),
                "status": payload["status"],
                "baseline_reproduced": baseline_reproduction["passed"],
                "selected_strategy": payload["selection"]["selected_strategy"],
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
