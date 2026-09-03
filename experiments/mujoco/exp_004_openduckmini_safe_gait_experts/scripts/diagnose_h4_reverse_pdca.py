"""Isolated strict-PDCA runner for exp004 straight reverse.

The central evaluator, contract, package, and documentation remain immutable.
This diagnostic reuses their CPU MuJoCo execution and every-substep safety
audits, while adding a stricter motion gate, six-second heading drift, and
stance-foot slip measurements.  Every output is simulation-only and hardware
deployment is always prohibited.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import evaluate_routed_transitions as central  # noqa: E402


CURRENT_PROFILE = (
    EXP_ROOT
    / "artifacts"
    / "optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1.json"
)
LEFT_PROFILE = (
    EXP_ROOT
    / "artifacts"
    / "reverse_turn_candidates_v1"
    / "optimized_reverse_turn_left_margin050_slew200_candidate_v1.json"
)
RIGHT_PROFILE = (
    EXP_ROOT
    / "artifacts"
    / "reverse_turn_candidates_v1"
    / "optimized_reverse_turn_right_margin050_slew200_candidate_v1.json"
)
BASE_POLICY = (
    WORKSPACE / ".openduck_runtime_source_review" / "calibrated_hybrid_policy_v22.onnx"
)
DEFAULT_OUTPUT = EXP_ROOT / "artifacts" / "h4_reverse_pdca_diagnostic.json"

PROFILE_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
CURRENT_PHASE_ENTRY = 7.0
PHYSICAL_COMMAND = (-0.05, 0.0, 0.0)
SPEED_RATIO_RANGE = (0.75, 1.25)
CROSS_ABSOLUTE_LIMIT_MPS = 0.012
CROSS_FRACTION_LIMIT = 0.20
YAW_RATE_LIMIT_RADPS = 0.05
HEADING_WINDOW_SECONDS = 6.0
HEADING_WINDOW_LIMIT_RAD = 0.15
SINGLE_SUPPORT_RANGE = (0.25, 0.60)
FLIGHT_RATE_LIMIT = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, *, name: str, length: int) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class Candidate:
    name: str
    phase_entry_preincrement: float = CURRENT_PHASE_ENTRY
    phase_rate_factor: float = 1.0
    amplitude_factors: tuple[float, ...] = (1.0,) * 10
    bias_deltas_rad: tuple[float, ...] = (0.0,) * 10
    lower_cap_extras_rad: tuple[float, ...] = (0.0,) * 10
    upper_cap_extras_rad: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    backward_residual_scale: float = 0.0
    policy_observation_command: tuple[float, ...] = PHYSICAL_COMMAND

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Candidate":
        allowed = {
            "name",
            "phase_entry_preincrement",
            "phase_rate_factor",
            "amplitude_factors",
            "bias_deltas_rad",
            "lower_cap_extras_rad",
            "upper_cap_extras_rad",
            "backward_residual_scale",
            "policy_observation_command",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown candidate fields: {sorted(unknown)}")
        name = value.get("name")
        if not isinstance(name, str) or not name or any(char.isspace() for char in name):
            raise ValueError("candidate name must be a non-empty token")
        candidate = cls(
            name=name,
            phase_entry_preincrement=float(
                value.get("phase_entry_preincrement", CURRENT_PHASE_ENTRY)
            ),
            phase_rate_factor=float(value.get("phase_rate_factor", 1.0)),
            amplitude_factors=_finite_vector(
                value.get("amplitude_factors", (1.0,) * 10),
                name="amplitude_factors",
                length=10,
            ),
            bias_deltas_rad=_finite_vector(
                value.get("bias_deltas_rad", (0.0,) * 10),
                name="bias_deltas_rad",
                length=10,
            ),
            lower_cap_extras_rad=_finite_vector(
                value.get("lower_cap_extras_rad", (0.0,) * 10),
                name="lower_cap_extras_rad",
                length=10,
            ),
            upper_cap_extras_rad=_finite_vector(
                value.get(
                    "upper_cap_extras_rad",
                    (0.0, 0.0, 0.0, 0.0125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ),
                name="upper_cap_extras_rad",
                length=10,
            ),
            backward_residual_scale=float(value.get("backward_residual_scale", 0.0)),
            policy_observation_command=_finite_vector(
                value.get("policy_observation_command", PHYSICAL_COMMAND),
                name="policy_observation_command",
                length=3,
            ),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        scalars = (
            self.phase_entry_preincrement,
            self.phase_rate_factor,
            self.backward_residual_scale,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("candidate scalars must be finite")
        if not 0.0 <= self.phase_entry_preincrement < 20.0:
            raise ValueError("phase entry must remain in [0, 20)")
        if self.phase_rate_factor <= 0.0:
            raise ValueError("phase-rate factor must be positive")
        if any(value <= 0.0 for value in self.amplitude_factors):
            raise ValueError("amplitude factors must be positive")
        if not 0.0 <= self.backward_residual_scale <= 0.25:
            raise ValueError("backward residual scale must remain in [0, 0.25]")
        if any(value < 0.0 or value > 0.05 for value in self.lower_cap_extras_rad):
            raise ValueError("lower cap extras must remain in [0, 0.05]")
        if any(value < 0.0 or value > 0.05 for value in self.upper_cap_extras_rad):
            raise ValueError("upper cap extras must remain in [0, 0.05]")

    @property
    def candidate_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_candidates(path: Path | None) -> tuple[Candidate, ...]:
    if path is None:
        return (Candidate("h3_exact_baseline"),)
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("candidate-set schema_version must be 1")
    values = payload.get("candidates")
    if not isinstance(values, list) or not values:
        raise ValueError("candidate set must contain a non-empty candidates list")
    candidates = tuple(Candidate.from_mapping(value) for value in values)
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate definitions must be unique")
    return candidates


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-set", type=Path)
    parser.add_argument("--seeds", type=_csv_ints, default=(20_271_810, 20_265_810, 20_260_810))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--initial-joint-noise-scale", type=float, default=1.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.seconds <= 0.0:
        parser.error("seconds must be positive")
    if not 0.0 <= args.warmup_seconds < args.seconds:
        parser.error("warmup must be non-negative and shorter than the run")
    if args.initial_joint_noise_scale < 0.0 or args.initial_base_speed < 0.0:
        parser.error("initial perturbations must be non-negative")
    return args


def advance_routed_phase_candidate(
    phase_index: float,
    *,
    phase_steps: float,
    phase_delta: float,
    current_expert: str,
    previous_expert: str | None,
    effective_command: Sequence[float],
    previous_backward_feedforward_active: bool,
    diagnostic_entry_phase_indices: Mapping[str, float] | None = None,
    phase_entry_status: str = "DIAGNOSTIC_H4_REVERSE_PDCA",
    diagnostic_only: bool = True,
    control_step: int | None = None,
    global_control_tick: int | None = None,
) -> tuple[float, bool, dict[str, Any] | None]:
    """Central phase algorithm with only the frozen phase-value gate relaxed."""

    current = float(phase_index)
    count = float(phase_steps)
    delta = float(phase_delta)
    command = np.asarray(effective_command, dtype=np.float64)
    if (
        not np.isfinite(current)
        or not np.isfinite(count)
        or count <= 0.0
        or not np.isfinite(delta)
        or delta <= 0.0
    ):
        raise ValueError("phase state must be finite and positive where required")
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("effective command must be a finite triplet")
    mapping = None
    if diagnostic_entry_phase_indices is not None:
        if set(diagnostic_entry_phase_indices) != set(central.BACKWARD_FAMILY_EXPERTS):
            raise ValueError("phase mapping must cover the backward family exactly")
        mapping = {key: float(value) for key, value in diagnostic_entry_phase_indices.items()}
        if not np.all(np.isfinite(list(mapping.values()))) or any(
            not 0.0 <= value < count for value in mapping.values()
        ):
            raise ValueError("phase mapping values must remain inside the period")
    active = bool(command[0] < -0.02)
    event = None
    if mapping is not None and active and not previous_backward_feedforward_active and current_expert in mapping:
        before = current
        current = mapping[current_expert]
        first = (current + delta) % count
        event = {
            "control_step": None if control_step is None else int(control_step),
            "global_control_tick": None if global_control_tick is None else int(global_control_tick),
            "previous_expert": previous_expert,
            "current_expert": current_expert,
            "effective_command": command.tolist(),
            "activation_predicate": "effective_vx_lt_negative_0p02_false_to_true",
            "previous_backward_feedforward_active": False,
            "current_backward_feedforward_active": True,
            "global_phase_index_before_reset": before,
            "reset_preincrement_phase_index": current,
            "profile_phase_rate": delta,
            "first_feedforward_phase_index": first,
            "phase_steps": count,
            "status": phase_entry_status,
            "formal_candidate": False,
            "adopted_simulation_only": False,
            "diagnostic_only": bool(diagnostic_only),
        }
    return (current + delta) % count, active, event


class SubstepTrace:
    def __init__(self, evaluator: Any, simulation_dt: float):
        self.evaluator = evaluator
        self.simulation_dt = float(simulation_dt)
        self.samples: list[dict[str, Any]] = []
        self._original = evaluator._feet_contacts
        self._last_time: float | None = None

    def reset(self) -> None:
        self.samples = []
        self._last_time = None

    def install(self) -> None:
        def traced(data: Any) -> np.ndarray:
            contacts = np.asarray(self._original(data), dtype=bool)
            now = float(data.time)
            if self._last_time is None or now > self._last_time + self.simulation_dt * 0.25:
                rotation = np.asarray(
                    data.xmat[self.evaluator.trunk_body_id], dtype=np.float64
                ).reshape(3, 3)
                self.samples.append(
                    {
                        "time": now,
                        "yaw": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                        "contacts": contacts.copy(),
                        "left_foot_xy": np.asarray(
                            data.xpos[self.evaluator.left_foot_body_id, :2],
                            dtype=np.float64,
                        ).copy(),
                        "right_foot_xy": np.asarray(
                            data.xpos[self.evaluator.right_foot_body_id, :2],
                            dtype=np.float64,
                        ).copy(),
                    }
                )
                self._last_time = now
            return contacts

        self.evaluator._feet_contacts = traced

    def restore(self) -> None:
        self.evaluator._feet_contacts = self._original


def heading_window_error_rad(yaws: Sequence[float], window_samples: int) -> float:
    values = np.unwrap(np.asarray(yaws, dtype=np.float64))
    if values.size < 2:
        return 0.0
    width = max(1, int(window_samples))
    if values.size <= width:
        return float(abs(values[-1] - values[0]))
    return float(np.max(np.abs(values[width:] - values[:-width])))


def stance_slip_metrics(samples: Sequence[Mapping[str, Any]], simulation_dt: float) -> dict[str, Any]:
    continuous: list[float] = []
    current_contact: list[float] = []
    by_foot = {"left": [], "right": []}
    for previous, current in zip(samples, samples[1:]):
        dt = float(current["time"]) - float(previous["time"])
        if dt <= simulation_dt * 0.25:
            continue
        for index, side in enumerate(("left", "right")):
            speed = float(
                np.linalg.norm(
                    np.asarray(current[f"{side}_foot_xy"])
                    - np.asarray(previous[f"{side}_foot_xy"])
                )
                / dt
            )
            if bool(np.asarray(current["contacts"])[index]):
                current_contact.append(speed)
            if bool(np.asarray(current["contacts"])[index]) and bool(
                np.asarray(previous["contacts"])[index]
            ):
                continuous.append(speed)
                by_foot[side].append(speed)

    def summary(values: Sequence[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "sample_count": int(array.size),
            "mean_mps": float(array.mean()) if array.size else 0.0,
            "p95_mps": float(np.percentile(array, 95.0)) if array.size else 0.0,
            "maximum_mps": float(array.max()) if array.size else 0.0,
        }

    return {
        "definition": "finite_difference_body_origin_xy_speed_when_contact_is_continuous_across_substeps",
        "continuous_stance": summary(continuous),
        "current_contact_including_touchdown": summary(current_contact),
        "by_foot_continuous_stance": {
            side: summary(values) for side, values in by_foot.items()
        },
    }


def strict_checks(segment: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, bool]:
    metrics = segment["metrics"]
    safety = segment["safety_audit"]
    substeps = segment["physics_substep_audit"]
    routing = segment["routing"]
    commanded = float(metrics["commanded_linear_speed"])
    ratio = float(metrics["projected_primary_velocity"]) / commanded
    cross_limit = min(CROSS_ABSOLUTE_LIMIT_MPS, CROSS_FRACTION_LIMIT * commanded)
    applied_margin = int(safety["applied_target_margin_violations"])
    startup_margin = int(safety["startup_margin_transition_joint_samples"])
    return {
        "completed": bool(segment["completed"]),
        "no_fall": not bool(segment["fell"]),
        "speed_ratio_75_to_125_percent": SPEED_RATIO_RANGE[0] <= ratio <= SPEED_RATIO_RANGE[1],
        "cross_velocity_bounded": float(metrics["absolute_orthogonal_velocity"]) <= cross_limit,
        "uncommanded_yaw_bounded": abs(float(metrics["uncommanded_yaw_rate"])) <= YAW_RATE_LIMIT_RADPS,
        "heading_error_per_6s_bounded": float(trace["maximum_heading_change_per_6s_rad"]) <= HEADING_WINDOW_LIMIT_RAD,
        "single_support_25_to_60_percent": SINGLE_SUPPORT_RANGE[0] <= float(substeps["single_support_rate"]) <= SINGLE_SUPPORT_RANGE[1],
        "flight_at_most_1_percent": float(substeps["flight_rate"]) <= FLIGHT_RATE_LIMIT,
        "all_expected_substeps_audited": int(substeps["sample_count"]) == int(segment["completed_physics_substeps"]) == int(segment["expected_physics_substeps"]),
        # The trace intentionally includes the t=0 state so finite-difference
        # slip has one position before each audited post-mj_step sample.
        "trace_covers_initial_state_and_all_substeps": int(trace["sample_count"])
        == int(substeps["sample_count"]) + 1,
        "substep_qpos_zero": int(substeps["qpos_limit_violations"]) == 0,
        "substep_nonfinite_zero": int(substeps["nonfinite_state_samples"]) == 0,
        "substep_fall_zero": int(substeps["height_fall_samples"]) == 0 and int(substeps["upright_fall_samples"]) == 0,
        "control_qpos_zero": int(safety["qpos_limit_violations"]) == 0,
        "control_nonfinite_zero": int(safety["nonfinite_sample_count"]) == 0,
        "applied_target_limit_zero": int(safety["applied_target_limit_violations"]) == 0,
        "desired_target_margin_zero": int(safety["desired_target_margin_violations"]) == 0,
        "unauthorized_applied_margin_zero": int(safety["unauthorized_applied_target_margin_violations"]) == 0,
        "startup_applied_margin_exactly_authorized": applied_margin == startup_margin,
        "target_slew_zero": int(safety["target_slew_violations"]) == 0,
        "route_prohibited_zero": int(routing["prohibited_expert_steps"]) == 0 and int(routing["steady_state_prohibited_expert_steps"]) == 0,
        "contact_count_exact": int(substeps["contact_sample_count"]) == int(substeps["sample_count"]) and bool(substeps["contact_sample_count_matches_sample_count"]),
    }


def _apply_candidate(evaluator: Any, base: Mapping[str, Any], candidate: Candidate) -> None:
    parameters = base["parameters"]
    evaluator.backward_gait_scales = np.asarray(
        parameters["joint_amplitude_scales"], dtype=np.float64
    ) * np.asarray(candidate.amplitude_factors, dtype=np.float64)
    evaluator.backward_gait_biases = np.asarray(
        parameters["joint_bias_offsets"], dtype=np.float64
    ) + np.asarray(candidate.bias_deltas_rad, dtype=np.float64)
    evaluator.backward_phase_rate = float(parameters["phase_rate"]) * candidate.phase_rate_factor
    evaluator.backward_residual_scale = candidate.backward_residual_scale


def _make_simulator(
    evaluator: Any,
    bank: Any,
    mujoco: Any,
    runtime: Any,
    candidate: Candidate,
) -> Any:
    simulator = central.RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        target_slew_rate_rad_s=central.RUNTIME_TARGET_SLEW_RATE_RAD_S,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=central.FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
        formal_candidate_default=True,
    )
    simulator.reverse_entry_phase_indices = {
        "reverse": candidate.phase_entry_preincrement,
        "reverse_turn_left": 4.0,
        "reverse_turn_right": 4.0,
    }
    simulator.phase_entry_status = "DIAGNOSTIC_H4_REVERSE_PDCA"
    simulator.phase_entry_diagnostic_only = True
    simulator.left_knee_extra_upper_margin_rad = candidate.upper_cap_extras_rad[3]
    simulator.left_knee_profile_upper_target_rad = (
        float(central.SAFE_JOINT_LIMITS["left_knee"][1])
        - central.RUNTIME_TARGET_SAFETY_MARGIN_RAD
        - candidate.upper_cap_extras_rad[3]
    )
    original_policy_target = simulator._policy_target
    profile_indices = np.asarray(evaluator.backward_actuator_indices, dtype=int)
    lower = simulator.target_lower[profile_indices] + np.asarray(
        candidate.lower_cap_extras_rad, dtype=np.float64
    )
    upper = simulator.target_upper[profile_indices] - np.asarray(
        candidate.upper_cap_extras_rad, dtype=np.float64
    )
    if np.any(lower > upper):
        raise ValueError("candidate per-joint caps collapse the target envelope")

    def capped_policy_target(
        applied_action: np.ndarray,
        effective_command: np.ndarray,
        phase_index: float,
        default: np.ndarray,
    ) -> np.ndarray:
        targets = original_policy_target(
            applied_action, effective_command, phase_index, default
        )
        targets[profile_indices] = np.clip(targets[profile_indices], lower, upper)
        return targets

    simulator._policy_target = capped_policy_target
    return simulator


def _trace_summary(trace: SubstepTrace, runtime: Any) -> dict[str, Any]:
    yaws = [float(sample["yaw"]) for sample in trace.samples]
    window_samples = int(round(HEADING_WINDOW_SECONDS / runtime.SIM_DT))
    return {
        "sample_count": len(trace.samples),
        "heading_window_seconds": HEADING_WINDOW_SECONDS,
        "maximum_heading_change_per_6s_rad": heading_window_error_rad(
            yaws, window_samples
        ),
        "start_yaw_rad": yaws[0] if yaws else None,
        "end_yaw_rad": yaws[-1] if yaws else None,
        "stance_slip": stance_slip_metrics(trace.samples, runtime.SIM_DT),
    }


def _run_record(
    simulator: Any,
    trace: SubstepTrace,
    candidate: Candidate,
    runtime: Any,
    *,
    seed: int,
    seconds: float,
    warmup_seconds: float,
    joint_noise_scale: float,
    initial_base_speed: float,
) -> dict[str, Any]:
    reverse = next(case for case in central.PRIMITIVE_CASES if case.name == "reverse")
    schedule = (
        (
            reverse.name,
            reverse.command,
            seconds,
            candidate.policy_observation_command,
            reverse.expected_expert,
            reverse.expected_policy_role,
        ),
    )
    trace.reset()
    run = simulator.run_schedule(
        schedule,
        seed=seed,
        joint_noise_scale=joint_noise_scale,
        initial_base_speed=initial_base_speed,
        warmup_seconds=warmup_seconds,
    )
    segment = run["segments"][0]
    traced = _trace_summary(trace, runtime)
    checks = strict_checks(segment, traced)
    metrics = segment["metrics"]
    quality = segment.get("gait_quality_acceptance")
    if quality is None:
        raise RuntimeError("central gait-quality acceptance is required")
    fall_safety_checks = {
        name: checks[name]
        for name in (
            "completed",
            "no_fall",
            "substep_fall_zero",
            "all_expected_substeps_audited",
        )
    }
    site_slip_checks = {
        name: bool(quality["checks"][name])
        for name in (
            "stance_slip_available",
            "stance_slip_rms",
            "stance_slip_p95",
            "per_stance_cumulative_slip",
        )
    }
    commanded = float(metrics["commanded_linear_speed"])
    return {
        "seed": seed,
        "strict_passed": all(checks.values()),
        "strict_checks": checks,
        "separate_hard_gates": {
            "fall_safety": {
                "passed": all(fall_safety_checks.values()),
                "checks": fall_safety_checks,
            },
            "site_slip": {
                "passed": all(site_slip_checks.values()),
                "checks": site_slip_checks,
            },
        },
        "strict_metrics": {
            "speed_ratio": float(metrics["projected_primary_velocity"]) / commanded,
            "projected_reverse_speed_mps": float(metrics["projected_primary_velocity"]),
            "absolute_cross_velocity_mps": float(metrics["absolute_orthogonal_velocity"]),
            "cross_limit_mps": min(CROSS_ABSOLUTE_LIMIT_MPS, CROSS_FRACTION_LIMIT * commanded),
            "absolute_uncommanded_yaw_rate_radps": abs(float(metrics["uncommanded_yaw_rate"])),
            "single_support_rate": float(segment["physics_substep_audit"]["single_support_rate"]),
            "flight_rate": float(segment["physics_substep_audit"]["flight_rate"]),
            "maximum_heading_change_per_6s_rad": traced["maximum_heading_change_per_6s_rad"],
        },
        "substep_trace_audit": traced,
        "central_run": run,
    }


def _candidate_summary(candidate: Candidate, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strict = [run["strict_metrics"] for run in runs]
    segments = [run["central_run"]["segments"][0] for run in runs]
    quality_metrics = [segment.get("gait_quality_metrics") for segment in segments]
    quality_acceptance = [
        segment.get("gait_quality_acceptance") for segment in segments
    ]
    if any(item is None for item in quality_metrics + quality_acceptance):
        raise RuntimeError("central gait-quality telemetry is required for every run")

    def quality_range(name: str) -> dict[str, float | None]:
        values = [item[name] for item in quality_metrics]
        finite = [float(value) for value in values if value is not None]
        return {
            "minimum": min(finite) if finite else None,
            "maximum": max(finite) if finite else None,
        }

    quality_failed_checks: dict[str, int] = {}
    for result in quality_acceptance:
        for name in result["failures"]:
            quality_failed_checks[name] = quality_failed_checks.get(name, 0) + 1
    fall_safety_pass_count = sum(
        bool(run["separate_hard_gates"]["fall_safety"]["passed"])
        for run in runs
    )
    site_slip_pass_count = sum(
        bool(run["separate_hard_gates"]["site_slip"]["passed"])
        for run in runs
    )
    failed_checks: dict[str, int] = {}
    for run in runs:
        for name, passed in run["strict_checks"].items():
            if not passed:
                failed_checks[name] = failed_checks.get(name, 0) + 1
    return {
        "candidate_id": candidate.candidate_id,
        "name": candidate.name,
        "parameters": asdict(candidate),
        "run_count": len(runs),
        "strict_pass_count": sum(bool(run["strict_passed"]) for run in runs),
        "strict_passed": all(bool(run["strict_passed"]) for run in runs),
        "failed_check_counts": dict(sorted(failed_checks.items())),
        "separate_hard_gates": {
            "fall_safety_pass_count": fall_safety_pass_count,
            "fall_safety_passed": fall_safety_pass_count == len(runs),
            "site_slip_pass_count": site_slip_pass_count,
            "site_slip_passed": site_slip_pass_count == len(runs),
            "both_pass_count": sum(
                bool(run["separate_hard_gates"]["fall_safety"]["passed"])
                and bool(run["separate_hard_gates"]["site_slip"]["passed"])
                for run in runs
            ),
        },
        "minimum_speed_ratio": min(float(item["speed_ratio"]) for item in strict),
        "maximum_speed_ratio": max(float(item["speed_ratio"]) for item in strict),
        "maximum_cross_velocity_mps": max(float(item["absolute_cross_velocity_mps"]) for item in strict),
        "maximum_uncommanded_yaw_rate_radps": max(float(item["absolute_uncommanded_yaw_rate_radps"]) for item in strict),
        "maximum_heading_change_per_6s_rad": max(float(item["maximum_heading_change_per_6s_rad"]) for item in strict),
        "minimum_single_support_rate": min(float(item["single_support_rate"]) for item in strict),
        "maximum_single_support_rate": max(float(item["single_support_rate"]) for item in strict),
        "maximum_flight_rate": max(float(item["flight_rate"]) for item in strict),
        "maximum_continuous_stance_slip_p95_mps": max(
            float(run["substep_trace_audit"]["stance_slip"]["continuous_stance"]["p95_mps"])
            for run in runs
        ),
        "maximum_continuous_stance_slip_mps": max(
            float(run["substep_trace_audit"]["stance_slip"]["continuous_stance"]["maximum_mps"])
            for run in runs
        ),
        "central_gait_quality": {
            "pass_count": sum(bool(item["passed"]) for item in quality_acceptance),
            "passed": all(bool(item["passed"]) for item in quality_acceptance),
            "failed_check_counts": dict(sorted(quality_failed_checks.items())),
            "startup": {
                "linear_t30_s": quality_range("linear_t30_s"),
                "linear_t75_s": quality_range("linear_t75_s"),
                "first_single_support_s": quality_range("first_single_support_s"),
            },
            "support": {
                "single_support_rate": quality_range("single_support_rate"),
                "flight_rate": quality_range("flight_rate"),
                "left_contact_rate": quality_range("left_contact_rate"),
                "right_contact_rate": quality_range("right_contact_rate"),
                "left_step_count": quality_range("left_step_count"),
                "right_step_count": quality_range("right_step_count"),
                "step_count_imbalance": quality_range("step_count_imbalance"),
                "contact_duty_imbalance": quality_range("contact_duty_imbalance"),
                "alternating_touchdown_fraction": quality_range(
                    "alternating_touchdown_fraction"
                ),
            },
            "stance_slip": {
                "rms_mps": quality_range("stance_slip_rms_mps"),
                "p95_mps": quality_range("stance_slip_p95_mps"),
                "maximum_per_stance_cumulative_slip_m": quality_range(
                    "maximum_per_stance_cumulative_slip_m"
                ),
            },
        },
        "fall_count": sum(bool(segment["fell"]) for segment in segments),
        "substep_qpos_violation_count": sum(int(segment["physics_substep_audit"]["qpos_limit_violations"]) for segment in segments),
        "substep_nonfinite_count": sum(int(segment["physics_substep_audit"]["nonfinite_state_samples"]) for segment in segments),
        "expected_substep_count": sum(int(segment["expected_physics_substeps"]) for segment in segments),
        "audited_substep_count": sum(int(segment["physics_substep_audit"]["sample_count"]) for segment in segments),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {output}")
    candidates = load_candidates(args.candidate_set)
    base = json.loads(CURRENT_PROFILE.read_text(encoding="utf-8"))
    asset_paths = central.generated_asset_paths(central.DEFAULT_GENERATED_ROOT.resolve())
    mujoco, onnxruntime, runtime, runtime_provenance = central._load_runtime(
        include_provenance=True
    )
    bank = central.RoutedPolicyBank(
        {role: BASE_POLICY.resolve() for role in central.REQUIRED_POLICY_ROLES},
        onnxruntime,
    )
    evaluator = runtime.OfficialPolicyEvaluator(
        asset_paths["scene"], BASE_POLICY.resolve(), asset_paths["reference"]
    )
    evaluator.load_backward_profile(CURRENT_PROFILE)
    evaluator.load_backward_turn_profile(1, LEFT_PROFILE)
    evaluator.load_backward_turn_profile(-1, RIGHT_PROFILE)
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    trace = SubstepTrace(evaluator, runtime.SIM_DT)
    original_advance = central.advance_routed_phase
    central.advance_routed_phase = advance_routed_phase_candidate
    trace.install()
    records = []
    try:
        for candidate in candidates:
            _apply_candidate(evaluator, base, candidate)
            simulator = _make_simulator(evaluator, bank, mujoco, runtime, candidate)
            runs = [
                _run_record(
                    simulator,
                    trace,
                    candidate,
                    runtime,
                    seed=seed,
                    seconds=args.seconds,
                    warmup_seconds=args.warmup_seconds,
                    joint_noise_scale=args.initial_joint_noise_scale,
                    initial_base_speed=args.initial_base_speed,
                )
                for seed in args.seeds
            ]
            records.append(
                {
                    "summary": _candidate_summary(candidate, runs),
                    "runs": runs,
                }
            )
    finally:
        trace.restore()
        central.advance_routed_phase = original_advance
    ranking = sorted(
        (record["summary"] for record in records),
        key=lambda item: (
            not item["strict_passed"],
            len(item["failed_check_counts"]),
            item["fall_count"],
            max(0.0, SPEED_RATIO_RANGE[0] - item["minimum_speed_ratio"]),
            max(0.0, item["maximum_cross_velocity_mps"] - 0.01),
            item["maximum_heading_change_per_6s_rad"],
        ),
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h4_reverse_strict_pdca",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY_NOT_ADOPTED",
        "hardware_deployment": "PROHIBITED",
        "strict_gate": {
            "physical_command": list(PHYSICAL_COMMAND),
            "speed_ratio_range": list(SPEED_RATIO_RANGE),
            "cross_velocity_limit": "min(0.012_mps, 0.20_times_commanded_speed)",
            "resolved_cross_velocity_limit_mps": 0.01,
            "absolute_uncommanded_yaw_rate_limit_radps": YAW_RATE_LIMIT_RADPS,
            "maximum_heading_change_limit_rad_per_6s": HEADING_WINDOW_LIMIT_RAD,
            "single_support_rate_range": list(SINGLE_SUPPORT_RANGE),
            "maximum_flight_rate": FLIGHT_RATE_LIMIT,
            "all_substep_safety_counters_zero": True,
            "stance_slip": (
                "recorded_here_and_separately_hard_gated_by_the_frozen_"
                "central_gait_quality_checks"
            ),
            "separate_candidate_hard_gates": {
                "fall_safety": (
                    "completed_no_fall_substep_fall_zero_all_substeps_audited"
                ),
                "site_slip": [
                    "stance_slip_available",
                    "stance_slip_rms",
                    "stance_slip_p95",
                    "per_stance_cumulative_slip",
                ],
            },
        },
        "configuration": {
            "candidate_set_path": (
                str(args.candidate_set.resolve()) if args.candidate_set is not None else None
            ),
            "candidate_set_sha256": (
                _sha256(args.candidate_set.resolve())
                if args.candidate_set is not None
                else None
            ),
            "seeds": list(args.seeds),
            "seconds": args.seconds,
            "warmup_seconds": args.warmup_seconds,
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "initial_base_speed": args.initial_base_speed,
            "candidate_count": len(candidates),
        },
        "current_h3_reproduction": {
            "profile_path": str(CURRENT_PROFILE.resolve()),
            "profile_sha256": _sha256(CURRENT_PROFILE),
            "phase_entry_preincrement": CURRENT_PHASE_ENTRY,
            "profile_left_knee_extra_upper_margin_rad": 0.0125,
            "backward_exit_recovery_extra_upper_margin_rad": central.BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "backward_exit_recovery_upper_target_rad": central.BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "dependencies": {
            "isolated_script_sha256_before_output": _sha256(Path(__file__).resolve()),
            "central_evaluator_path": str(Path(central.__file__).resolve()),
            "central_evaluator_sha256": _sha256(Path(central.__file__).resolve()),
            "base_policy_path": str(BASE_POLICY.resolve()),
            "base_policy_sha256": _sha256(BASE_POLICY),
            "left_profile_sha256": _sha256(LEFT_PROFILE),
            "right_profile_sha256": _sha256(RIGHT_PROFILE),
            "runtime": runtime_provenance,
            "onnx_providers": bank.session_providers,
        },
        "ranking": ranking,
        "candidates": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "ranking": ranking,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
