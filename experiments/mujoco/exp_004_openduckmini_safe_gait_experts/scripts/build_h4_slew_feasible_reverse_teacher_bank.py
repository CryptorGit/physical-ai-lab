"""Build deterministic, slew-feasible H4 reverse teacher trajectories.

This is an isolated design tool.  It does not alter the routed evaluator,
training entrypoint, release package, or any adopted reverse profile.  The
output is a simulation/training candidate bank with explicit 14-axis target
tables and a pure validator.

The H3 reverse feed-forward profile advances a 27-bin reference by roughly
1.966 bins every 20 ms (about 3.64 Hz).  H4 instead needs a 1.5--2.0 Hz
teacher whose *desired* target changes are already within the final 2 rad/s
guard.  We therefore:

* reconstruct the exact H3 27-bin leg target trajectory;
* periodically resample it to an even 54-bin table;
* impose half-cycle left/right mirror symmetry;
* retain one or two circular Fourier harmonics; and
* scale excursion only as much as required by the inward target envelope and
  the exact piecewise-linear 0.04 rad/control-tick delta gate.

The reset-to-first-teacher transition is deliberately not called a desired
target delta.  The frozen contract handles that one startup transition with
the final guard from the exact safe reset state.  All cyclic teacher-to-
teacher desired deltas are validated exactly over every interpolation-region
endpoint, not by a coarse time sample.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.contract import (  # noqa: E402
    ACTUATOR_JOINT_ORDER,
    CONTROL_FIRST_STARTUP_DT_S,
    HEAD_JOINTS,
    LEG_TARGET_MARGIN_RAD,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
    TARGET_SLEW_LIMIT_RAD_PER_S,
)


H3_PROFILE = (
    EXP_ROOT
    / "artifacts"
    / "optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1.json"
)
MINIMUM_SPEC = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_retraining_minimum_spec_from_slip_causality_v1.json"
)
REFERENCE_PICKLE = (
    WORKSPACE
    / ".openduck_runtime_source_review"
    / "polynomial_coefficients_calibrated.pkl"
)
DEFAULT_OUTPUT = (
    EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_bank_v1.json"
)

LEG_JOINT_ORDER = (
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
LEG_ACTUATOR_INDICES = tuple(
    ACTUATOR_JOINT_ORDER.index(name) for name in LEG_JOINT_ORDER
)
HEAD_ACTUATOR_INDICES = tuple(
    index
    for index, name in enumerate(ACTUATOR_JOINT_ORDER)
    if name in HEAD_JOINTS
)
REFERENCE_LEG_INDICES = (0, 1, 2, 3, 4, 11, 12, 13, 14, 15)

# A mirrored pose uses opposite model-space signs for hip yaw, roll, and
# pitch.  Knee and ankle signs are shared.  The relation is
# left[k] == sign * right[k + period/2].
PAIR_MIRROR_SIGNS = (-1.0, -1.0, -1.0, 1.0, 1.0)
SOURCE_PHASE_STEPS = 27
SOURCE_ENTRY_PREINCREMENT = 7.0
RESAMPLED_PHASE_STEPS = 54
PHYSICAL_COMMAND = (-0.05, 0.0, 0.0)
TARGET_CADENCES_HZ = (1.50, 1.75, 2.00)
HARMONIC_COUNTS = (1, 2)
EXCURSION_FRACTIONS = (1.0, 0.75)
CADENCE_RANGE_HZ = (1.5, 2.0)
MAXIMUM_TARGET_DELTA_RAD = (
    TARGET_SLEW_LIMIT_RAD_PER_S * CONTROL_FIRST_STARTUP_DT_S
)
# Leave a small numerical reserve below the hard 0.04-rad threshold.
DESIGN_TARGET_DELTA_RAD = MAXIMUM_TARGET_DELTA_RAD - 5.0e-4
VALIDATION_ATOL = 1.0e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_reference_key(key: str) -> tuple[float, float, float]:
    parts = key.split("_")
    if len(parts) < 3:
        raise ValueError(f"invalid polynomial reference key: {key!r}")
    return tuple(float(value) for value in parts[:3])  # type: ignore[return-value]


def load_backward_reference_leg_frames(path: Path = REFERENCE_PICKLE) -> np.ndarray:
    """Reproduce PolyReferenceMotion's nearest-grid -0.1/0/0 lookup."""

    with path.resolve().open("rb") as stream:
        payload = pickle.load(stream)  # trusted, hash-recorded frozen input
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("polynomial reference must contain a non-empty mapping")
    parsed = [(_parse_reference_key(str(key)), value) for key, value in payload.items()]
    axes = tuple(
        sorted({coordinates[axis] for coordinates, _ in parsed})
        for axis in range(3)
    )
    requested = (-0.1, 0.0, 0.0)
    selected = tuple(
        min(axis_values, key=lambda value: abs(value - requested[index]))
        for index, axis_values in enumerate(axes)
    )
    motion = next(value for coordinates, value in parsed if coordinates == selected)
    period = int(motion.get("nb_steps_in_period", 0))
    if period <= 0:
        period = int(float(motion["period"]) * float(motion["fps"]))
    if period != SOURCE_PHASE_STEPS:
        raise ValueError(
            f"expected {SOURCE_PHASE_STEPS} H3 reference bins, found {period}"
        )
    coefficients = list(motion["coefficients"].values())
    frames = np.asarray(
        [
            [
                np.polyval(np.flip(np.asarray(coefficient, dtype=np.float64)), index / period)
                for coefficient in coefficients
            ]
            for index in range(period)
        ],
        dtype=np.float64,
    )
    if frames.ndim != 2 or frames.shape[1] <= max(REFERENCE_LEG_INDICES):
        raise ValueError("reference polynomial does not cover all ten leg joints")
    leg_frames = frames[:, REFERENCE_LEG_INDICES]
    if leg_frames.shape != (SOURCE_PHASE_STEPS, len(LEG_JOINT_ORDER)):
        raise ValueError("unexpected H3 leg reference shape")
    if not np.all(np.isfinite(leg_frames)):
        raise ValueError("H3 leg reference must be finite")
    return leg_frames


def reconstruct_h3_leg_targets(
    profile_path: Path = H3_PROFILE,
    reference_path: Path = REFERENCE_PICKLE,
) -> np.ndarray:
    profile = json.loads(profile_path.resolve().read_text(encoding="utf-8"))
    parameters = profile["parameters"]
    scales = np.asarray(parameters["joint_amplitude_scales"], dtype=np.float64)
    biases = np.asarray(parameters["joint_bias_offsets"], dtype=np.float64)
    if scales.shape != (10,) or biases.shape != (10,):
        raise ValueError("H3 profile must define ten amplitude scales and biases")
    if not np.all(np.isfinite(scales)) or not np.all(np.isfinite(biases)):
        raise ValueError("H3 profile parameters must be finite")
    reference = load_backward_reference_leg_frames(reference_path)
    mean = reference.mean(axis=0)
    return mean + biases + scales * (reference - mean)


def periodic_interpolate(table: np.ndarray, phases: np.ndarray) -> np.ndarray:
    values = np.asarray(table, dtype=np.float64)
    query = np.asarray(phases, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("periodic table must be a two-dimensional cycle")
    period = values.shape[0]
    wrapped = np.mod(query, period)
    lower = np.floor(wrapped).astype(np.int64)
    upper = (lower + 1) % period
    fraction = wrapped - lower
    return (1.0 - fraction[..., None]) * values[lower] + fraction[..., None] * values[upper]


def periodic_resample(table: np.ndarray, output_steps: int) -> np.ndarray:
    if output_steps <= 1:
        raise ValueError("output_steps must exceed one")
    phases = np.arange(output_steps, dtype=np.float64) * table.shape[0] / output_steps
    return periodic_interpolate(table, phases)


def runtime_compatible_inner_bounds(joint_name: str) -> tuple[float, float]:
    """Intersect the H4 margin with the legacy feed-forward preclip.

    The CPU evaluator first applies its historical 90-percent envelope and
    then the exp004 final 0.05-rad target guard.  A teacher intended to reach
    the final guard unchanged must live in the intersection of both.
    """

    safe_lower, safe_upper = SAFE_JOINT_LIMITS[joint_name]
    default = SAFE_INIT_POS[joint_name]
    legacy_lower = default + 0.9 * (safe_lower - default)
    legacy_upper = default + 0.9 * (safe_upper - default)
    lower = max(safe_lower + LEG_TARGET_MARGIN_RAD, legacy_lower)
    upper = min(safe_upper - LEG_TARGET_MARGIN_RAD, legacy_upper)
    if lower > upper:
        raise ValueError(f"runtime-compatible target envelope is empty for {joint_name}")
    return float(lower), float(upper)


def symmetric_inner_envelopes() -> tuple[np.ndarray, np.ndarray]:
    """Return canonical left-leg bounds valid for both mirrored legs."""

    lower = []
    upper = []
    for pair_index, sign in enumerate(PAIR_MIRROR_SIGNS):
        left_name = LEG_JOINT_ORDER[pair_index]
        right_name = LEG_JOINT_ORDER[pair_index + 5]
        left_bounds = runtime_compatible_inner_bounds(left_name)
        right_bounds = runtime_compatible_inner_bounds(right_name)
        mirrored_right = sorted((sign * right_bounds[0], sign * right_bounds[1]))
        lo = max(left_bounds[0], mirrored_right[0])
        hi = min(left_bounds[1], mirrored_right[1])
        if lo > hi:
            raise ValueError(f"mirrored safe envelope is empty for pair {pair_index}")
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower), np.asarray(upper)


def impose_half_cycle_symmetry(resampled_leg_targets: np.ndarray) -> np.ndarray:
    values = np.asarray(resampled_leg_targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 10 or values.shape[0] % 2:
        raise ValueError("symmetry requires an even table with ten leg columns")
    half = values.shape[0] // 2
    signs = np.asarray(PAIR_MIRROR_SIGNS)
    canonical = 0.5 * (values[:, :5] + signs * np.roll(values[:, 5:], -half, axis=0))
    lower, upper = symmetric_inner_envelopes()
    canonical = np.clip(canonical, lower, upper)
    result = np.empty_like(values)
    result[:, :5] = canonical
    result[:, 5:] = signs * np.roll(canonical, half, axis=0)
    return result


def retain_circular_harmonics(values: np.ndarray, harmonic_count: int) -> np.ndarray:
    if harmonic_count < 1:
        raise ValueError("at least one non-DC harmonic is required")
    spectrum = np.fft.rfft(np.asarray(values, dtype=np.float64), axis=0)
    if harmonic_count + 1 < spectrum.shape[0]:
        spectrum[harmonic_count + 1 :] = 0.0
    return np.fft.irfft(spectrum, n=values.shape[0], axis=0)


def exact_cyclic_tick_delta(table: np.ndarray, phase_advance: float) -> dict[str, Any]:
    """Find the exact max for a piecewise-linear periodic target table.

    ``target(x + d) - target(x)`` is piecewise linear.  Its extrema occur
    where either interpolation argument crosses a table knot, so evaluating
    ``x = integer`` and ``x = integer - d`` is exhaustive.
    """

    values = np.asarray(table, dtype=np.float64)
    if phase_advance <= 0.0 or not np.isfinite(phase_advance):
        raise ValueError("phase_advance must be finite and positive")
    knots = np.arange(values.shape[0], dtype=np.float64)
    starts = np.unique(np.concatenate((knots, np.mod(knots - phase_advance, values.shape[0]))))
    before = periodic_interpolate(values, starts)
    after = periodic_interpolate(values, starts + phase_advance)
    absolute = np.abs(after - before)
    flat_index = int(np.argmax(absolute))
    phase_index, joint_index = np.unravel_index(flat_index, absolute.shape)
    return {
        "maximum_rad": float(absolute[phase_index, joint_index]),
        "joint_index": int(joint_index),
        "joint_name": ACTUATOR_JOINT_ORDER[joint_index],
        "start_phase_bins": float(starts[phase_index]),
        "evaluated_region_endpoint_count": int(len(starts)),
        "method": "exact_piecewise_linear_region_endpoints",
    }


def _maximum_excursion_scale(
    center: np.ndarray,
    deviation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    bound = 1.0
    for joint in range(deviation.shape[1]):
        positive = deviation[:, joint] > 0.0
        negative = deviation[:, joint] < 0.0
        if np.any(positive):
            bound = min(
                bound,
                float(np.min((upper[joint] - center[joint]) / deviation[positive, joint])),
            )
        if np.any(negative):
            bound = min(
                bound,
                float(np.min((lower[joint] - center[joint]) / deviation[negative, joint])),
            )
    return max(0.0, bound)


def _expand_leg_table(leg_table: np.ndarray) -> np.ndarray:
    targets = np.zeros((leg_table.shape[0], len(ACTUATOR_JOINT_ORDER)), dtype=np.float64)
    targets[:, LEG_ACTUATOR_INDICES] = leg_table
    return targets


def build_candidate(
    symmetric_source: np.ndarray,
    *,
    cadence_hz: float,
    harmonic_count: int,
    excursion_fraction: float,
) -> dict[str, Any]:
    filtered = retain_circular_harmonics(symmetric_source, harmonic_count)
    canonical = filtered[:, :5]
    lower, upper = symmetric_inner_envelopes()
    center = np.clip(canonical.mean(axis=0), lower, upper)
    deviation = canonical - canonical.mean(axis=0)
    envelope_scale = _maximum_excursion_scale(center, deviation, lower, upper)

    # Construct a unit-scale symmetric 14-axis table to derive the exact
    # target-to-target tick delta.  Its delta is linear in excursion scale.
    unit_canonical = center + envelope_scale * deviation
    unit_leg = np.empty_like(symmetric_source)
    half = len(symmetric_source) // 2
    signs = np.asarray(PAIR_MIRROR_SIGNS)
    unit_leg[:, :5] = unit_canonical
    unit_leg[:, 5:] = signs * np.roll(unit_canonical, half, axis=0)
    phase_advance = cadence_hz * len(unit_leg) * CONTROL_FIRST_STARTUP_DT_S
    unit_tick_delta = exact_cyclic_tick_delta(_expand_leg_table(unit_leg), phase_advance)[
        "maximum_rad"
    ]
    slew_scale = (
        1.0
        if unit_tick_delta <= DESIGN_TARGET_DELTA_RAD
        else DESIGN_TARGET_DELTA_RAD / unit_tick_delta
    )
    retained_scale = envelope_scale * slew_scale * excursion_fraction
    candidate_canonical = center + retained_scale * deviation
    leg_table = np.empty_like(symmetric_source)
    leg_table[:, :5] = candidate_canonical
    leg_table[:, 5:] = signs * np.roll(candidate_canonical, half, axis=0)
    target_table = _expand_leg_table(leg_table)
    equivalent_source_advance = cadence_hz * SOURCE_PHASE_STEPS * CONTROL_FIRST_STARTUP_DT_S
    construction = {
        "cadence_hz": float(cadence_hz),
        "harmonic_count": int(harmonic_count),
        "excursion_fraction_of_maximum": float(excursion_fraction),
        "envelope_feasible_excursion_scale": float(envelope_scale),
        "slew_feasible_excursion_scale": float(slew_scale),
        "retained_source_excursion_scale": float(retained_scale),
        "phase_steps": int(len(target_table)),
        "phase_advance_bins_per_control": float(phase_advance),
        "equivalent_27_bin_phase_advance_per_control": float(equivalent_source_advance),
        "phase_entry_preincrement_bins": float(
            SOURCE_ENTRY_PREINCREMENT / SOURCE_PHASE_STEPS * len(target_table)
        ),
    }
    identifier_payload = {
        "construction": construction,
        "target_table_rad": target_table.tolist(),
    }
    candidate = {
        "candidate_id": _canonical_sha256(identifier_payload)[:16],
        "name": (
            f"h4_reverse_c{cadence_hz:.2f}_h{harmonic_count}_"
            f"e{excursion_fraction:.2f}"
        ).replace(".", "p"),
        "status": "TRAINING_TEACHER_CANDIDATE_NOT_ADOPTED",
        "hardware_deployment": "PROHIBITED",
        "physical_command_mps_radps": list(PHYSICAL_COMMAND),
        "policy_observation_command_mps_radps": list(PHYSICAL_COMMAND),
        "construction": construction,
        "joint_order": list(ACTUATOR_JOINT_ORDER),
        "target_table_rad": target_table.tolist(),
    }
    candidate["validation"] = validate_candidate(candidate)
    return candidate


def validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    construction = candidate.get("construction", {})
    values = np.asarray(candidate.get("target_table_rad"), dtype=np.float64)
    shape_ok = values.shape == (RESAMPLED_PHASE_STEPS, len(ACTUATOR_JOINT_ORDER))
    finite = bool(shape_ok and np.all(np.isfinite(values)))
    cadence = float(construction.get("cadence_hz", float("nan")))
    phase_advance = float(
        construction.get("phase_advance_bins_per_control", float("nan"))
    )
    expected_advance = cadence * RESAMPLED_PHASE_STEPS * CONTROL_FIRST_STARTUP_DT_S
    cadence_ok = bool(
        np.isfinite(cadence)
        and CADENCE_RANGE_HZ[0] - VALIDATION_ATOL
        <= cadence
        <= CADENCE_RANGE_HZ[1] + VALIDATION_ATOL
    )
    phase_ok = bool(
        np.isfinite(phase_advance)
        and abs(phase_advance - expected_advance) <= VALIDATION_ATOL
    )
    if not finite:
        return {
            "passed": False,
            "checks": {
                "shape_54x14": shape_ok,
                "finite": finite,
                "cadence_1p5_to_2p0_hz": cadence_ok,
                "phase_advance_matches_cadence": phase_ok,
                "signed_reverse_command": candidate.get("physical_command_mps_radps")
                == list(PHYSICAL_COMMAND),
            },
            "failures": ["shape_or_finite"],
        }

    lower = np.full(len(ACTUATOR_JOINT_ORDER), -np.inf)
    upper = np.full(len(ACTUATOR_JOINT_ORDER), np.inf)
    for index, name in enumerate(ACTUATOR_JOINT_ORDER):
        if name in HEAD_JOINTS:
            lower[index] = upper[index] = 0.0
        else:
            lower[index] = SAFE_JOINT_LIMITS[name][0] + LEG_TARGET_MARGIN_RAD
            upper[index] = SAFE_JOINT_LIMITS[name][1] - LEG_TARGET_MARGIN_RAD
    envelope_lower_clearance = float(np.min(values - lower))
    envelope_upper_clearance = float(np.min(upper - values))
    envelope_ok = bool(
        np.all(values >= lower - VALIDATION_ATOL)
        and np.all(values <= upper + VALIDATION_ATOL)
    )
    preclip_lower = lower.copy()
    preclip_upper = upper.copy()
    for index, name in enumerate(ACTUATOR_JOINT_ORDER):
        if name not in HEAD_JOINTS:
            preclip_lower[index], preclip_upper[index] = (
                runtime_compatible_inner_bounds(name)
            )
    legacy_preclip_ok = bool(
        np.all(values >= preclip_lower - VALIDATION_ATOL)
        and np.all(values <= preclip_upper + VALIDATION_ATOL)
    )
    head_peak = float(np.max(np.abs(values[:, HEAD_ACTUATOR_INDICES])))
    head_ok = head_peak == 0.0
    half = values.shape[0] // 2
    left = values[:, LEG_ACTUATOR_INDICES[:5]]
    right = values[:, LEG_ACTUATOR_INDICES[5:]]
    symmetry_error = float(
        np.max(
            np.abs(
                left
                - np.asarray(PAIR_MIRROR_SIGNS)
                * np.roll(right, -half, axis=0)
            )
        )
    )
    symmetry_ok = symmetry_error <= VALIDATION_ATOL
    delta = exact_cyclic_tick_delta(values, phase_advance)
    slew_ok = delta["maximum_rad"] <= MAXIMUM_TARGET_DELTA_RAD + VALIDATION_ATOL
    signed_command_ok = (
        candidate.get("physical_command_mps_radps") == list(PHYSICAL_COMMAND)
        and candidate.get("policy_observation_command_mps_radps")
        == list(PHYSICAL_COMMAND)
    )
    checks = {
        "shape_54x14": shape_ok,
        "finite": finite,
        "cadence_1p5_to_2p0_hz": cadence_ok,
        "phase_advance_matches_cadence": phase_ok,
        "inside_safe_inner_envelope": envelope_ok,
        "unchanged_by_legacy_feedforward_preclip": legacy_preclip_ok,
        "head_targets_exactly_zero": head_ok,
        "half_cycle_left_right_symmetry": symmetry_ok,
        "cyclic_pre_guard_target_delta_at_most_0p04_rad": slew_ok,
        "signed_reverse_command": signed_command_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(name for name, passed in checks.items() if not passed),
        "metrics": {
            "cadence_hz": cadence,
            "phase_advance_bins_per_control": phase_advance,
            "maximum_cyclic_pre_guard_target_delta_rad": delta["maximum_rad"],
            "target_delta_reserve_rad": MAXIMUM_TARGET_DELTA_RAD
            - delta["maximum_rad"],
            "maximum_delta_joint": delta["joint_name"],
            "maximum_delta_start_phase_bins": delta["start_phase_bins"],
            "exact_delta_region_endpoint_count": delta[
                "evaluated_region_endpoint_count"
            ],
            "maximum_left_right_symmetry_error_rad": symmetry_error,
            "head_target_peak_rad": head_peak,
            "minimum_lower_envelope_clearance_rad": envelope_lower_clearance,
            "minimum_upper_envelope_clearance_rad": envelope_upper_clearance,
            "minimum_legacy_preclip_lower_clearance_rad": float(
                np.min(values - preclip_lower)
            ),
            "minimum_legacy_preclip_upper_clearance_rad": float(
                np.min(preclip_upper - values)
            ),
        },
    }


def build_bank() -> dict[str, Any]:
    h3_targets = reconstruct_h3_leg_targets()
    resampled = periodic_resample(h3_targets, RESAMPLED_PHASE_STEPS)
    symmetric = impose_half_cycle_symmetry(resampled)
    candidates = [
        build_candidate(
            symmetric,
            cadence_hz=cadence,
            harmonic_count=harmonics,
            excursion_fraction=excursion,
        )
        for cadence in TARGET_CADENCES_HZ
        for harmonics in HARMONIC_COUNTS
        for excursion in EXCURSION_FRACTIONS
    ]
    if not all(candidate["validation"]["passed"] for candidate in candidates):
        raise RuntimeError("generated reverse teacher bank failed its pure validator")
    ranking = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["construction"]["retained_source_excursion_scale"]),
            float(candidate["validation"]["metrics"]["maximum_cyclic_pre_guard_target_delta_rad"]),
            float(candidate["construction"]["cadence_hz"]),
            int(candidate["construction"]["harmonic_count"]),
        ),
    )
    source_rate = float(
        json.loads(H3_PROFILE.read_text(encoding="utf-8"))["parameters"][
            "phase_rate"
        ]
    )
    return {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h4_reverse_slew_feasible_teacher_bank",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PURE_VALIDATED_TRAINING_TEACHERS_NOT_ADOPTED_NOT_RELEASE_EVIDENCE",
        "hardware_deployment": "PROHIBITED",
        "simulation_adoption_allowed": False,
        "release_allowed": False,
        "source": {
            "h3_profile_path": str(H3_PROFILE.relative_to(EXP_ROOT)),
            "h3_profile_sha256": _sha256(H3_PROFILE),
            "minimum_spec_path": str(MINIMUM_SPEC.relative_to(EXP_ROOT)),
            "minimum_spec_sha256": _sha256(MINIMUM_SPEC),
            "reference_path": str(REFERENCE_PICKLE.relative_to(WORKSPACE)),
            "reference_sha256": _sha256(REFERENCE_PICKLE),
            "source_phase_steps": SOURCE_PHASE_STEPS,
            "source_phase_rate_bins_per_control": source_rate,
            "source_implied_cadence_hz": source_rate
            / (SOURCE_PHASE_STEPS * CONTROL_FIRST_STARTUP_DT_S),
        },
        "construction_contract": {
            "physical_command_mps_radps": list(PHYSICAL_COMMAND),
            "policy_observation_command_mps_radps": list(PHYSICAL_COMMAND),
            "resampling": "periodic_linear_27_to_54_bins",
            "symmetry_relation": (
                "left_joint[k] == mirror_sign * "
                "right_joint[(k + phase_steps/2) mod phase_steps]"
            ),
            "left_to_right_mirror_signs": dict(
                zip(LEG_JOINT_ORDER[:5], PAIR_MIRROR_SIGNS)
            ),
            "circular_harmonic_counts": list(HARMONIC_COUNTS),
            "target_cadences_hz": list(TARGET_CADENCES_HZ),
            "accepted_cadence_range_hz": list(CADENCE_RANGE_HZ),
            "control_period_s": CONTROL_FIRST_STARTUP_DT_S,
            "target_slew_limit_rad_per_s": TARGET_SLEW_LIMIT_RAD_PER_S,
            "maximum_cyclic_pre_guard_target_delta_rad": MAXIMUM_TARGET_DELTA_RAD,
            "design_target_delta_rad": DESIGN_TARGET_DELTA_RAD,
            "leg_target_inward_margin_rad": LEG_TARGET_MARGIN_RAD,
            "legacy_feedforward_preclip": (
                "intersected_with_default_plus_0p9_times_joint_range_span"
            ),
            "head_target_rad": 0.0,
            "startup_note": (
                "exact reset-to-first-teacher transition remains the frozen final "
                "guard's control-first startup responsibility"
            ),
        },
        "pure_validation": {
            "candidate_count": len(candidates),
            "pass_count": len(candidates),
            "all_passed": True,
            "validator": "validate_candidate",
        },
        "ranking_candidate_ids": [candidate["candidate_id"] for candidate in ranking],
        "ranking_rule": (
            "maximum_retained_excursion_then_delta_reserve_then_lower_cadence"
        ),
        "candidates": candidates,
        "decision": {
            "best_teacher_candidate_id": ranking[0]["candidate_id"],
            "physical_failure3_screen": "NOT_RUN_BY_BUILDER",
            "training": "NOT_RUN",
            "adoption": "BLOCKED_PENDING_EXACT_HOME_FAILURE3_AND_TRAINING",
            "hardware": "PROHIBITED",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate",
        type=Path,
        help="validate an existing teacher bank without writing a file",
    )
    return parser.parse_args(argv)


def validate_bank(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return {"passed": False, "candidate_count": 0, "failures": ["candidates"]}
    validations = [validate_candidate(candidate) for candidate in candidates]
    return {
        "passed": all(result["passed"] for result in validations),
        "candidate_count": len(candidates),
        "pass_count": sum(bool(result["passed"]) for result in validations),
        "candidate_failures": {
            str(candidate.get("candidate_id", index)): result["failures"]
            for index, (candidate, result) in enumerate(zip(candidates, validations))
            if not result["passed"]
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate is not None:
        payload = json.loads(args.validate.resolve().read_text(encoding="utf-8"))
        result = validate_bank(payload)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite teacher bank: {output}")
    payload = build_bank()
    # The timestamp is informative only and excluded from candidate IDs.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "candidate_count": len(payload["candidates"]),
                "best_teacher_candidate_id": payload["decision"][
                    "best_teacher_candidate_id"
                ],
                "all_pure_validation_passed": payload["pure_validation"][
                    "all_passed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
