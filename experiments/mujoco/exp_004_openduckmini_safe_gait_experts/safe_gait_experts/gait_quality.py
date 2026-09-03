"""Strict, simulator-agnostic gait-quality accumulation and acceptance.

This module deliberately has no MuJoCo or routed-evaluator dependency.  A caller
feeds every physics substep, using elapsed time from motion activation and world
coordinates for the trunk and the representative sole contact points.  The
accumulator only performs deterministic numerical bookkeeping and does not
replace :class:`SafetyAudit`.

The frozen defaults encode the stricter limits adopted after the H3 V2 video
audit.  In particular, signed progress alone is insufficient: startup response,
left/right support, stance slip, periodic joint motion, command fidelity, both
lower and upper tracking bounds, cross drift, and yaw-only drift all have gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Iterable, Mapping, Sequence

import numpy as np


LEFT_FOOT = 0
RIGHT_FOOT = 1
FOOT_NAMES = ("left", "right")
CADENCE_ANALYSIS_BAND_HZ = (0.30, 4.00)
_COMMAND_EPSILON = 1.0e-9


@dataclass(frozen=True)
class GaitQualityThresholds:
    """Frozen strict-quality limits; safety limits live elsewhere."""

    steady_start_s: float = 1.50
    maximum_t30_s: float = 0.40
    maximum_t75_s: float = 1.00
    maximum_first_single_support_s: float = 0.80
    startup_rolling_window_s: float = 0.20
    startup_sustain_s: float = 0.30
    startup_audit_window_s: float = 1.50
    startup_wrong_way_occupancy_window_s: float = 1.00
    maximum_startup_wrong_way_displacement_m: float = 0.010
    maximum_startup_continuous_wrong_way_s: float = 0.20
    maximum_startup_wrong_way_occupancy: float = 0.10
    maximum_startup_pure_cross_excursion_m: float = 0.020
    maximum_startup_yaw_only_planar_excursion_m: float = 0.020
    contact_debounce_s: float = 0.020
    contact_force_on_fraction_body_weight: float = 0.010
    contact_force_off_fraction_body_weight: float = 0.005

    minimum_single_support_rate: float = 0.25
    maximum_single_support_rate: float = 0.60
    maximum_flight_rate: float = 0.01
    minimum_steps_per_foot: int = 2
    minimum_touchdowns_per_foot_for_six_seconds: int = 3
    maximum_step_count_imbalance: int = 1
    maximum_contact_duty_imbalance: float = 0.10
    minimum_alternating_touchdown_fraction: float = 0.80
    maximum_debounce_touchdown_count_span: int = 1
    maximum_debounce_single_support_rate_span: float = 0.05
    maximum_debounce_contact_rate_span: float = 0.05
    maximum_debounce_flight_rate_span: float = 0.005

    minimum_contact_cadence_hz: float = 0.70
    maximum_contact_cadence_hz: float = 2.50
    maximum_touchdown_median_cadence_difference_hz: float = 0.25
    maximum_touchdown_joint_cadence_difference_hz: float = 0.25
    maximum_touchdown_interval_cv: float = 0.30
    maximum_same_foot_touchdown_gap_s: float = 1.50

    maximum_stance_slip_rms_mps: float = 0.015
    maximum_stance_slip_p95_mps: float = 0.030
    maximum_per_stance_cumulative_slip_m: float = 0.020
    minimum_contact_velocity_coverage: float = 0.95
    maximum_contact_velocity_coverage: float = 1.01
    minimum_steady_total_normal_force_fraction_body_weight: float = 0.80
    maximum_steady_total_normal_force_fraction_body_weight: float = 1.20
    minimum_per_foot_normal_impulse_share: float = 0.35
    maximum_per_foot_normal_impulse_share: float = 0.65
    maximum_total_normal_force_p99_fraction_body_weight: float = 3.0

    minimum_periodic_joint_range_rad: float = 0.080
    maximum_periodic_joint_range_rad: float = 0.800
    minimum_periodic_joints_per_leg: int = 2
    minimum_joint_cadence_hz: float = 0.70
    maximum_joint_cadence_hz: float = 2.50
    maximum_left_right_cadence_difference_hz: float = 0.25

    maximum_requested_effective_axis_error: float = 0.005
    minimum_steady_tracking_ratio: float = 0.75
    maximum_steady_tracking_ratio: float = 1.25
    maximum_pure_cross_drift_mps: float = 0.012
    maximum_pure_cross_drift_fraction: float = 0.20
    maximum_compound_cross_drift_mps: float = 0.015
    maximum_compound_cross_drift_fraction: float = 0.25
    maximum_uncommanded_yaw_rate_radps: float = 0.050
    maximum_uncommanded_heading_drift_rad: float = 0.150
    maximum_yaw_only_planar_drift_mps: float = 0.012
    maximum_yaw_only_net_drift_m: float = 0.050

    minimum_pure_endpoint_primary_ratio: float = 0.75
    maximum_pure_endpoint_primary_ratio: float = 1.25
    maximum_pure_endpoint_cross_error_m: float = 0.030
    maximum_pure_endpoint_cross_error_fraction: float = 0.10
    maximum_pure_cross_path_p95_error_m: float = 0.020
    maximum_pure_cross_path_error_m: float = 0.040
    maximum_path_heading_error_rad: float = 0.150
    maximum_cumulative_backtracking_m: float = 0.010
    maximum_compound_endpoint_position_error_m: float = 0.050
    maximum_compound_endpoint_position_error_fraction: float = 0.20
    maximum_compound_path_p95_error_m: float = 0.035
    maximum_compound_path_error_m: float = 0.050
    maximum_yaw_only_path_radius_p95_m: float = 0.030
    maximum_yaw_only_path_radius_m: float = 0.050
    maximum_stand_path_radius_m: float = 0.020
    maximum_stand_heading_excursion_rad: float = 0.050

    def __post_init__(self) -> None:
        positive = (
            "steady_start_s",
            "maximum_t30_s",
            "maximum_t75_s",
            "maximum_first_single_support_s",
            "startup_rolling_window_s",
            "startup_sustain_s",
            "startup_audit_window_s",
            "startup_wrong_way_occupancy_window_s",
            "maximum_startup_wrong_way_displacement_m",
            "maximum_startup_continuous_wrong_way_s",
            "maximum_startup_wrong_way_occupancy",
            "maximum_startup_pure_cross_excursion_m",
            "maximum_startup_yaw_only_planar_excursion_m",
            "maximum_debounce_single_support_rate_span",
            "maximum_debounce_contact_rate_span",
            "maximum_debounce_flight_rate_span",
            "contact_debounce_s",
            "contact_force_on_fraction_body_weight",
            "contact_force_off_fraction_body_weight",
            "maximum_stance_slip_rms_mps",
            "maximum_stance_slip_p95_mps",
            "maximum_per_stance_cumulative_slip_m",
            "minimum_contact_velocity_coverage",
            "maximum_contact_velocity_coverage",
            "minimum_steady_total_normal_force_fraction_body_weight",
            "maximum_steady_total_normal_force_fraction_body_weight",
            "minimum_per_foot_normal_impulse_share",
            "maximum_per_foot_normal_impulse_share",
            "maximum_total_normal_force_p99_fraction_body_weight",
            "minimum_periodic_joint_range_rad",
            "maximum_periodic_joint_range_rad",
            "minimum_joint_cadence_hz",
            "maximum_joint_cadence_hz",
            "maximum_left_right_cadence_difference_hz",
            "maximum_requested_effective_axis_error",
            "minimum_steady_tracking_ratio",
            "maximum_steady_tracking_ratio",
            "maximum_pure_cross_drift_mps",
            "maximum_pure_cross_drift_fraction",
            "maximum_compound_cross_drift_mps",
            "maximum_compound_cross_drift_fraction",
            "maximum_uncommanded_yaw_rate_radps",
            "maximum_uncommanded_heading_drift_rad",
            "maximum_yaw_only_planar_drift_mps",
            "maximum_yaw_only_net_drift_m",
            "minimum_contact_cadence_hz",
            "maximum_contact_cadence_hz",
            "maximum_touchdown_median_cadence_difference_hz",
            "maximum_touchdown_joint_cadence_difference_hz",
            "maximum_touchdown_interval_cv",
            "maximum_same_foot_touchdown_gap_s",
            "minimum_pure_endpoint_primary_ratio",
            "maximum_pure_endpoint_primary_ratio",
            "maximum_pure_endpoint_cross_error_m",
            "maximum_pure_endpoint_cross_error_fraction",
            "maximum_pure_cross_path_p95_error_m",
            "maximum_pure_cross_path_error_m",
            "maximum_path_heading_error_rad",
            "maximum_cumulative_backtracking_m",
            "maximum_compound_endpoint_position_error_m",
            "maximum_compound_endpoint_position_error_fraction",
            "maximum_compound_path_p95_error_m",
            "maximum_compound_path_error_m",
            "maximum_yaw_only_path_radius_p95_m",
            "maximum_yaw_only_path_radius_m",
            "maximum_stand_path_radius_m",
            "maximum_stand_heading_excursion_rad",
        )
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        rates = (
            "minimum_single_support_rate",
            "maximum_single_support_rate",
            "maximum_flight_rate",
            "maximum_contact_duty_imbalance",
            "minimum_alternating_touchdown_fraction",
            "maximum_startup_wrong_way_occupancy",
            "minimum_per_foot_normal_impulse_share",
            "maximum_per_foot_normal_impulse_share",
        )
        for name in rates:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.minimum_single_support_rate > self.maximum_single_support_rate:
            raise ValueError("single-support lower bound exceeds upper bound")
        if (
            self.contact_force_off_fraction_body_weight
            >= self.contact_force_on_fraction_body_weight
        ):
            raise ValueError("contact force off threshold must be below on threshold")
        if self.minimum_steady_tracking_ratio > self.maximum_steady_tracking_ratio:
            raise ValueError("tracking lower bound exceeds upper bound")
        if self.minimum_joint_cadence_hz > self.maximum_joint_cadence_hz:
            raise ValueError("cadence lower bound exceeds upper bound")
        if self.minimum_contact_cadence_hz > self.maximum_contact_cadence_hz:
            raise ValueError("contact cadence lower bound exceeds upper bound")
        if self.minimum_periodic_joint_range_rad > self.maximum_periodic_joint_range_rad:
            raise ValueError("joint-range lower bound exceeds upper bound")
        if self.minimum_steps_per_foot < 1:
            raise ValueError("minimum_steps_per_foot must be positive")
        if self.minimum_touchdowns_per_foot_for_six_seconds < 1:
            raise ValueError("six-second touchdown minimum must be positive")
        if self.maximum_step_count_imbalance < 0:
            raise ValueError("maximum_step_count_imbalance must be non-negative")
        if self.minimum_periodic_joints_per_leg < 1:
            raise ValueError("minimum_periodic_joints_per_leg must be positive")
        if self.maximum_debounce_touchdown_count_span < 0:
            raise ValueError("debounce touchdown span must be non-negative")
        if self.minimum_contact_velocity_coverage > self.maximum_contact_velocity_coverage:
            raise ValueError("contact-velocity coverage lower bound exceeds upper bound")
        if self.minimum_pure_endpoint_primary_ratio > self.maximum_pure_endpoint_primary_ratio:
            raise ValueError("pure endpoint ratio lower bound exceeds upper bound")
        if (
            self.minimum_steady_total_normal_force_fraction_body_weight
            > self.maximum_steady_total_normal_force_fraction_body_weight
        ):
            raise ValueError("normal-force lower bound exceeds upper bound")
        if self.minimum_per_foot_normal_impulse_share > self.maximum_per_foot_normal_impulse_share:
            raise ValueError("normal-impulse share lower bound exceeds upper bound")


FROZEN_GAIT_QUALITY_THRESHOLDS = GaitQualityThresholds()


@dataclass(frozen=True)
class GaitQualitySubstep:
    """One complete physics-substep observation.

    ``time_s`` is elapsed time since the requested motion became active.
    ``foot_contact_points_world_m`` must describe a stable representative point
    on each sole/contact patch, not a camera-space pixel or a trunk-relative
    point.  Tangential slip uses the world x/y displacement of that point while
    the same foot remains in contact.
    """

    time_s: float
    requested_command: Sequence[float]
    effective_command: Sequence[float]
    local_velocity_xyz_mps: Sequence[float]
    local_yaw_rate_radps: float
    trunk_position_world_m: Sequence[float]
    feet_contacts: Sequence[bool]
    foot_contact_points_world_m: Sequence[Sequence[float]]
    leg_joint_positions_rad: Sequence[float]
    feet_normal_force_fraction_body_weight: Sequence[float] | None = None
    foot_contact_tangential_speeds_mps: Sequence[float] | None = None
    trunk_yaw_world_rad: float | None = None
    trunk_pose_measurement_source: str = "world_trunk_position_only"


@dataclass(frozen=True)
class GaitQualityMetrics:
    sample_count: int
    duration_s: float
    steady_sample_count: int
    requested_command_mean: tuple[float, float, float]
    effective_command_mean: tuple[float, float, float]
    requested_effective_axis_error: tuple[float, float, float]
    requested_effective_axis_rms_error: tuple[float, float, float]

    linear_t30_s: float | None
    linear_t75_s: float | None
    yaw_t30_s: float | None
    yaw_t75_s: float | None
    t30_s: float | None
    t75_s: float | None
    first_single_support_s: float | None

    left_contact_rate: float
    right_contact_rate: float
    single_support_rate: float
    double_support_rate: float
    flight_rate: float
    left_step_count: int
    right_step_count: int
    step_count_imbalance: int
    contact_duty_imbalance: float
    alternating_touchdown_fraction: float | None

    left_stance_count: int
    right_stance_count: int
    left_stance_slip_rms_mps: float | None
    right_stance_slip_rms_mps: float | None
    left_stance_slip_p95_mps: float | None
    right_stance_slip_p95_mps: float | None
    left_maximum_per_stance_cumulative_slip_m: float | None
    right_maximum_per_stance_cumulative_slip_m: float | None
    stance_slip_rms_mps: float | None
    stance_slip_p95_mps: float | None
    maximum_per_stance_cumulative_slip_m: float | None

    joint_names: tuple[str, ...]
    joint_ranges_rad: tuple[float, ...]
    left_joint_indices: tuple[int, ...]
    right_joint_indices: tuple[int, ...]
    left_cadence_joint: str
    right_cadence_joint: str
    left_joint_cadence_hz: float | None
    right_joint_cadence_hz: float | None

    steady_mean_local_velocity_xyz_mps: tuple[float, float, float]
    steady_mean_local_yaw_rate_radps: float
    steady_linear_tracking_ratio: float | None
    steady_yaw_tracking_ratio: float | None
    steady_cross_drift_mps: float | None
    steady_cross_drift_fraction: float | None
    uncommanded_yaw_rate_radps: float | None
    uncommanded_heading_drift_rad: float | None
    yaw_only_planar_drift_mps: float | None
    yaw_only_net_drift_m: float | None
    net_trunk_displacement_xy_m: tuple[float, float]
    contact_state_source: str = "boolean_contact"
    contact_force_sample_count: int = 0
    stance_slip_measurement_source: str = "site_finite_difference"
    contact_velocity_sample_count: int = 0
    contact_debounce_sensitivity: Mapping[str, object] = field(default_factory=dict)
    measurement_schema_version: int = 2
    physics_timestep_s: float | None = None
    maximum_timestep_error_s: float | None = None
    trunk_pose_measurement_source: str = "world_trunk_position_only"
    trunk_yaw_sample_count: int = 0

    startup_active_axis_names: tuple[str, ...] = ()
    startup_axis_t30_s: tuple[float | None, ...] = ()
    startup_axis_t75_s: tuple[float | None, ...] = ()
    startup_axis_wrong_way_displacement: tuple[float, ...] = ()
    startup_axis_maximum_continuous_wrong_way_s: tuple[float, ...] = ()
    startup_axis_wrong_way_occupancy: tuple[float, ...] = ()
    startup_pure_cross_excursion_m: float | None = None
    startup_yaw_only_planar_excursion_m: float | None = None

    command_class: str = "unknown"
    ideal_endpoint_xy_m: tuple[float, float] | None = None
    ideal_endpoint_yaw_rad: float | None = None
    actual_endpoint_yaw_rad: float | None = None
    pure_endpoint_primary_ratio: float | None = None
    pure_endpoint_cross_error_m: float | None = None
    pure_endpoint_cross_error_fraction: float | None = None
    pure_cross_path_p95_error_m: float | None = None
    pure_cross_path_maximum_error_m: float | None = None
    trajectory_heading_endpoint_error_rad: float | None = None
    trajectory_heading_maximum_error_rad: float | None = None
    cumulative_backtracking_m: float | None = None
    compound_endpoint_position_error_m: float | None = None
    compound_endpoint_position_error_fraction: float | None = None
    compound_path_p95_error_m: float | None = None
    compound_path_maximum_error_m: float | None = None
    yaw_only_path_radius_p95_m: float | None = None
    yaw_only_path_radius_maximum_m: float | None = None
    stand_path_radius_maximum_m: float | None = None
    stand_heading_excursion_rad: float | None = None

    left_touchdown_timestamps_s: tuple[float, ...] = ()
    right_touchdown_timestamps_s: tuple[float, ...] = ()
    left_touchdown_count_cadence_hz: float | None = None
    right_touchdown_count_cadence_hz: float | None = None
    left_touchdown_median_cadence_hz: float | None = None
    right_touchdown_median_cadence_hz: float | None = None
    left_touchdown_interval_cv: float | None = None
    right_touchdown_interval_cv: float | None = None
    left_maximum_same_foot_touchdown_gap_s: float | None = None
    right_maximum_same_foot_touchdown_gap_s: float | None = None

    contact_velocity_payload_sample_count: int = 0
    left_contact_velocity_sample_count: int = 0
    right_contact_velocity_sample_count: int = 0
    left_stance_contact_interval_count: int = 0
    right_stance_contact_interval_count: int = 0
    left_contact_velocity_expected_sample_count: float = 0.0
    right_contact_velocity_expected_sample_count: float = 0.0
    contact_velocity_coverage: float | None = None
    left_contact_velocity_coverage: float | None = None
    right_contact_velocity_coverage: float | None = None
    steady_mean_total_normal_force_fraction_body_weight: float | None = None
    left_normal_impulse_share: float | None = None
    right_normal_impulse_share: float | None = None
    total_normal_force_p99_fraction_body_weight: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GaitQualityAcceptance:
    passed: bool
    checks: Mapping[str, bool]
    applicable: Mapping[str, bool]
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "applicable": dict(self.applicable),
            "failures": list(self.failures),
        }


def _finite_array(value: Sequence[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _rms(values: Sequence[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _first_ratio_crossing(
    times: np.ndarray, ratios: np.ndarray | None, threshold: float
) -> float | None:
    if ratios is None:
        return None
    indices = np.flatnonzero(ratios >= threshold)
    return None if not len(indices) else float(times[int(indices[0])])


def _maximum_optional(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return max(float(value) for value in values if value is not None)


def _wrap_angle_rad(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _causal_rolling_mean(
    times: np.ndarray, values: np.ndarray, window_s: float
) -> np.ndarray:
    """Return a causal time-weighted-enough mean for dense physics telemetry.

    Physics samples are required to be near-uniform and their exact timing error is
    exported separately.  The index window therefore avoids interpolating or hiding
    an individual wrong-way sample at startup.
    """

    result = np.empty_like(values, dtype=np.float64)
    left = 0
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    for right, time_s in enumerate(times):
        while left < right and times[left] < time_s - window_s:
            left += 1
        result[right] = (cumulative[right + 1] - cumulative[left]) / (
            right - left + 1
        )
    return result


def _sustained_rolling_ratio_crossing(
    times: np.ndarray,
    ratios: np.ndarray | None,
    threshold: float,
    *,
    rolling_window_s: float,
    sustain_s: float,
) -> float | None:
    if ratios is None:
        return None
    rolling = _causal_rolling_mean(times, ratios, rolling_window_s)
    for start in np.flatnonzero(rolling >= threshold):
        end_time = float(times[start]) + sustain_s
        end = int(np.searchsorted(times, end_time, side="left"))
        if end >= len(times):
            continue
        if np.all(rolling[start : end + 1] >= threshold):
            return float(times[start])
    return None


def _wrong_way_run_metrics(
    times: np.ndarray, signed_rates: np.ndarray, window_s: float
) -> tuple[float, float]:
    """Maximum continuous wrong-way duration and time occupancy in a window."""

    if len(times) < 2:
        return 0.0, 0.0
    end = min(float(times[-1]), float(window_s))
    maximum_run = 0.0
    current_run = 0.0
    wrong_duration = 0.0
    total_duration = 0.0
    for index, dt in enumerate(np.diff(times)):
        interval_start = float(times[index])
        if interval_start >= end:
            break
        clipped_dt = min(float(dt), end - interval_start)
        if clipped_dt <= 0.0:
            continue
        total_duration += clipped_dt
        if signed_rates[index] < 0.0:
            current_run += clipped_dt
            wrong_duration += clipped_dt
            maximum_run = max(maximum_run, current_run)
        else:
            current_run = 0.0
    occupancy = 0.0 if total_duration <= 0.0 else wrong_duration / total_duration
    return maximum_run, occupancy


def _touchdown_interval_metrics(
    timestamps: Sequence[float], duration_s: float
) -> tuple[float | None, float | None, float | None, float | None]:
    count_cadence = None if duration_s <= 0.0 else len(timestamps) / duration_s
    if len(timestamps) < 2:
        return count_cadence, None, None, None
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    median_interval = float(np.median(intervals))
    median_cadence = None if median_interval <= 0.0 else 1.0 / median_interval
    mean_interval = float(np.mean(intervals))
    interval_cv = None if mean_interval <= 0.0 else float(np.std(intervals) / mean_interval)
    return count_cadence, median_cadence, interval_cv, float(np.max(intervals))


def _integrate_body_twist_trajectory(
    times: np.ndarray,
    commands: np.ndarray,
    initial_xy: np.ndarray,
    initial_yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the requested planar body twist exactly over every N+1 state."""

    positions = np.empty((len(times), 2), dtype=np.float64)
    yaws = np.empty(len(times), dtype=np.float64)
    positions[0] = initial_xy
    yaws[0] = initial_yaw
    for index, dt_value in enumerate(np.diff(times), start=1):
        dt = float(dt_value)
        twist = 0.5 * (commands[index - 1] + commands[index])
        vx, vy, yaw_rate = (float(value) for value in twist)
        turn = yaw_rate * dt
        if abs(yaw_rate) <= 1.0e-12:
            local_delta = np.asarray((vx * dt, vy * dt), dtype=np.float64)
        else:
            sine = np.sin(turn)
            one_minus_cosine = 1.0 - np.cos(turn)
            local_delta = np.asarray(
                (
                    sine / yaw_rate * vx - one_minus_cosine / yaw_rate * vy,
                    one_minus_cosine / yaw_rate * vx + sine / yaw_rate * vy,
                ),
                dtype=np.float64,
            )
        cosine, sine_yaw = np.cos(yaws[index - 1]), np.sin(yaws[index - 1])
        rotation = np.asarray(((cosine, -sine_yaw), (sine_yaw, cosine)))
        positions[index] = positions[index - 1] + rotation @ local_delta
        yaws[index] = yaws[index - 1] + turn
    return positions, yaws


def _dominant_frequency_hz(times: np.ndarray, values: np.ndarray) -> float | None:
    if len(times) < 4 or len(values) != len(times):
        return None
    duration = float(times[-1] - times[0])
    if duration <= 0.0 or float(np.ptp(values)) <= 1.0e-9:
        return None
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0.0:
        return None
    regular_times = np.arange(times[0], times[-1] + 0.5 * dt, dt)
    regular_values = np.interp(regular_times, times, values)
    index = np.arange(len(regular_values), dtype=np.float64)
    slope, intercept = np.polyfit(index, regular_values, 1)
    detrended = regular_values - (slope * index + intercept)
    windowed = detrended * np.hanning(len(detrended))
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(len(windowed), d=dt)
    low, high = CADENCE_ANALYSIS_BAND_HZ
    band = (frequencies >= low) & (frequencies <= high)
    if not np.any(band):
        return None
    band_power = power[band]
    if not np.any(band_power > 1.0e-18):
        return None
    band_frequencies = frequencies[band]
    peak_floor = 0.20 * float(np.max(band_power))
    local_peaks = [
        index
        for index, value in enumerate(band_power)
        if value >= peak_floor
        and (index == 0 or value >= band_power[index - 1])
        and (index == len(band_power) - 1 or value >= band_power[index + 1])
    ]
    selected = local_peaks[0] if local_peaks else int(np.argmax(band_power))
    return float(band_frequencies[selected])


def _contact_sequence_summary(
    times: np.ndarray, raw_contacts: np.ndarray, debounce_s: float
) -> dict[str, object]:
    """Summarize one causal-persistence setting without mutating live state."""

    filtered = np.empty_like(raw_contacts, dtype=np.bool_)
    states = [bool(raw_contacts[0, 0]), bool(raw_contacts[0, 1])]
    pending: list[bool | None] = [None, None]
    pending_since: list[float | None] = [None, None]
    filtered[0] = states
    for sample_index in range(1, len(times)):
        for foot in (LEFT_FOOT, RIGHT_FOOT):
            raw_state = bool(raw_contacts[sample_index, foot])
            if raw_state == states[foot]:
                pending[foot] = None
                pending_since[foot] = None
            elif pending[foot] != raw_state:
                pending[foot] = raw_state
                pending_since[foot] = float(times[sample_index])
            elif (
                pending_since[foot] is not None
                and float(times[sample_index]) - pending_since[foot] >= debounce_s
            ):
                states[foot] = raw_state
                pending[foot] = None
                pending_since[foot] = None
        filtered[sample_index] = states
    touchdown_counts = tuple(
        int(np.count_nonzero((~filtered[:-1, foot]) & filtered[1:, foot]))
        for foot in (LEFT_FOOT, RIGHT_FOOT)
    )
    touchdown_sequence: list[int] = []
    touchdown_timestamps: list[list[float]] = [[], []]
    for sample_index in range(1, len(filtered)):
        for foot in (LEFT_FOOT, RIGHT_FOOT):
            if not filtered[sample_index - 1, foot] and filtered[sample_index, foot]:
                touchdown_sequence.append(foot)
                touchdown_timestamps[foot].append(float(times[sample_index]))
    alternating = None
    if len(touchdown_sequence) >= 2:
        alternating = sum(
            left != right
            for left, right in zip(touchdown_sequence, touchdown_sequence[1:])
        ) / (len(touchdown_sequence) - 1)
    return {
        "left_touchdowns": touchdown_counts[LEFT_FOOT],
        "right_touchdowns": touchdown_counts[RIGHT_FOOT],
        "left_contact_rate": float(np.mean(filtered[:, LEFT_FOOT])),
        "right_contact_rate": float(np.mean(filtered[:, RIGHT_FOOT])),
        "single_support_rate": float(
            np.mean(np.logical_xor(filtered[:, LEFT_FOOT], filtered[:, RIGHT_FOOT]))
        ),
        "flight_rate": float(np.mean(~np.logical_or(filtered[:, 0], filtered[:, 1]))),
        "contact_duty_imbalance": abs(
            float(np.mean(filtered[:, LEFT_FOOT]))
            - float(np.mean(filtered[:, RIGHT_FOOT]))
        ),
        "step_count_imbalance": abs(
            touchdown_counts[LEFT_FOOT] - touchdown_counts[RIGHT_FOOT]
        ),
        "alternating_touchdown_fraction": alternating,
        "left_touchdown_timestamps_s": touchdown_timestamps[LEFT_FOOT],
        "right_touchdown_timestamps_s": touchdown_timestamps[RIGHT_FOOT],
    }


class GaitQualityAccumulator:
    """Pure numerical accumulator for one constant-command motion segment."""

    def __init__(
        self,
        *,
        joint_names: Sequence[str],
        left_joint_indices: Sequence[int] | None = None,
        right_joint_indices: Sequence[int] | None = None,
        left_cadence_joint_index: int | None = None,
        right_cadence_joint_index: int | None = None,
        steady_start_s: float = FROZEN_GAIT_QUALITY_THRESHOLDS.steady_start_s,
        contact_debounce_s: float = FROZEN_GAIT_QUALITY_THRESHOLDS.contact_debounce_s,
        contact_force_on_fraction_body_weight: float = (
            FROZEN_GAIT_QUALITY_THRESHOLDS.contact_force_on_fraction_body_weight
        ),
        contact_force_off_fraction_body_weight: float = (
            FROZEN_GAIT_QUALITY_THRESHOLDS.contact_force_off_fraction_body_weight
        ),
    ) -> None:
        self.joint_names = tuple(str(name) for name in joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        if not np.isfinite(steady_start_s) or steady_start_s < 0.0:
            raise ValueError("steady_start_s must be finite and non-negative")
        self.steady_start_s = float(steady_start_s)
        if not np.isfinite(contact_debounce_s) or contact_debounce_s <= 0.0:
            raise ValueError("contact_debounce_s must be finite and positive")
        self.contact_debounce_s = float(contact_debounce_s)
        force_on = float(contact_force_on_fraction_body_weight)
        force_off = float(contact_force_off_fraction_body_weight)
        if not np.isfinite(force_on) or not np.isfinite(force_off):
            raise ValueError("contact force thresholds must be finite")
        if force_on <= 0.0 or force_off < 0.0 or force_off >= force_on:
            raise ValueError("contact force thresholds must satisfy 0 <= off < on")
        self.contact_force_on_fraction_body_weight = force_on
        self.contact_force_off_fraction_body_weight = force_off

        inferred_left = tuple(
            index for index, name in enumerate(self.joint_names) if name.startswith("left_")
        )
        inferred_right = tuple(
            index for index, name in enumerate(self.joint_names) if name.startswith("right_")
        )
        self.left_joint_indices = self._validate_indices(
            inferred_left if left_joint_indices is None else left_joint_indices,
            "left_joint_indices",
        )
        self.right_joint_indices = self._validate_indices(
            inferred_right if right_joint_indices is None else right_joint_indices,
            "right_joint_indices",
        )
        if set(self.left_joint_indices) & set(self.right_joint_indices):
            raise ValueError("left and right joint indices must be disjoint")
        if not self.left_joint_indices or not self.right_joint_indices:
            raise ValueError("both legs must have at least one joint")

        if (left_cadence_joint_index is None) != (right_cadence_joint_index is None):
            raise ValueError("cadence joint indices must be both automatic or both explicit")
        self.left_cadence_joint_index = (
            None
            if left_cadence_joint_index is None
            else self._resolve_cadence_index(
                left_cadence_joint_index, self.left_joint_indices, "left_knee"
            )
        )
        self.right_cadence_joint_index = (
            None
            if right_cadence_joint_index is None
            else self._resolve_cadence_index(
                right_cadence_joint_index, self.right_joint_indices, "right_knee"
            )
        )

        self._times: list[float] = []
        self._requested: list[np.ndarray] = []
        self._effective: list[np.ndarray] = []
        self._local_velocity: list[np.ndarray] = []
        self._yaw_rate: list[float] = []
        self._trunk_position: list[np.ndarray] = []
        self._trunk_yaw: list[float] = []
        self._contacts: list[tuple[bool, bool]] = []
        self._raw_contacts: list[tuple[bool, bool]] = []
        self._foot_points: list[np.ndarray] = []
        self._joint_positions: list[np.ndarray] = []
        self._normal_force_fractions: list[np.ndarray] = []

        self._previous_contacts: tuple[bool, bool] | None = None
        self._previous_raw_contacts: tuple[bool, bool] | None = None
        self._previous_time_s: float | None = None
        self._debounced_contacts: list[bool] | None = None
        self._pending_contact_state: list[bool | None] = [None, None]
        self._pending_contact_since_s: list[float | None] = [None, None]
        self._previous_foot_points: np.ndarray | None = None
        self._touchdowns = [0, 0]
        self._touchdown_sequence: list[int] = []
        self._touchdown_times: list[list[float]] = [[], []]
        self._stance_cumulative: list[float | None] = [None, None]
        self._stance_distances: list[list[float]] = [[], []]
        self._stance_slip_speeds: list[list[float]] = [[], []]
        self._first_single_support_s: float | None = None
        self._force_contact_state: list[bool] | None = None
        self._contact_state_source: str | None = None
        self._stance_slip_measurement_source: str | None = None
        self._contact_velocity_sample_count = 0
        self._contact_velocity_payload_sample_count = 0
        self._contact_velocity_sample_counts = [0, 0]
        self._trunk_pose_measurement_source: str | None = None

    def _validate_indices(self, values: Sequence[int], name: str) -> tuple[int, ...]:
        indices = tuple(int(value) for value in values)
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} must contain unique indices")
        if any(index < 0 or index >= len(self.joint_names) for index in indices):
            raise ValueError(f"{name} contains an out-of-range index")
        return indices

    def _resolve_cadence_index(
        self, explicit: int | None, leg_indices: tuple[int, ...], preferred_name: str
    ) -> int:
        if explicit is not None:
            index = int(explicit)
            if index not in leg_indices:
                raise ValueError("cadence joint index must belong to its leg")
            return index
        if preferred_name in self.joint_names:
            index = self.joint_names.index(preferred_name)
            if index in leg_indices:
                return index
        return leg_indices[0]

    def _select_cadence_pair(self, joint_ranges: np.ndarray) -> tuple[int, int]:
        """Select the same strongest periodic joint role on both legs."""

        if self.left_cadence_joint_index is not None:
            assert self.right_cadence_joint_index is not None
            return self.left_cadence_joint_index, self.right_cadence_joint_index
        left_by_role = {
            self.joint_names[index].removeprefix("left_"): index
            for index in self.left_joint_indices
        }
        right_by_role = {
            self.joint_names[index].removeprefix("right_"): index
            for index in self.right_joint_indices
        }
        shared_roles = sorted(set(left_by_role) & set(right_by_role))
        if not shared_roles:
            raise ValueError("left/right legs must share a cadence joint role")
        role = max(
            shared_roles,
            key=lambda name: (
                min(
                    joint_ranges[left_by_role[name]],
                    joint_ranges[right_by_role[name]],
                ),
                joint_ranges[left_by_role[name]] + joint_ranges[right_by_role[name]],
                name,
            ),
        )
        return left_by_role[role], right_by_role[role]

    @property
    def sample_count(self) -> int:
        return len(self._times)

    def export_contact_continuity_state(self) -> dict[str, object]:
        """Export only cross-segment stance state, never motion metrics."""

        if self._previous_contacts is None or self._previous_foot_points is None:
            raise ValueError("contact continuity requires at least one sample")
        previous_time = 0.0 if self._previous_time_s is None else self._previous_time_s
        pending_age = tuple(
            None
            if pending_since is None
            else max(0.0, previous_time - pending_since)
            for pending_since in self._pending_contact_since_s
        )
        return {
            "contacts": tuple(self._previous_contacts),
            "raw_contacts": tuple(self._previous_raw_contacts or self._previous_contacts),
            "foot_points": self._previous_foot_points.copy(),
            "stance_cumulative": tuple(self._stance_cumulative),
            "debounced_contacts": tuple(self._debounced_contacts or self._previous_contacts),
            "force_contact_state": (
                None
                if self._force_contact_state is None
                else tuple(self._force_contact_state)
            ),
            "pending_contact_state": tuple(self._pending_contact_state),
            "pending_contact_age_s": pending_age,
            "previous_touchdown_foot": (
                None if not self._touchdown_sequence else self._touchdown_sequence[-1]
            ),
            "contact_state_source": self._contact_state_source,
            "stance_slip_measurement_source": self._stance_slip_measurement_source,
        }

    def restore_contact_continuity_state(self, state: Mapping[str, object]) -> None:
        """Continue a physical stance across a schedule-segment boundary."""

        if self.sample_count or self._previous_contacts is not None:
            raise ValueError("contact continuity can only seed an empty accumulator")
        contacts = tuple(bool(value) for value in state["contacts"])
        raw_contacts = tuple(bool(value) for value in state["raw_contacts"])
        debounced = tuple(bool(value) for value in state["debounced_contacts"])
        if len(contacts) != 2 or len(raw_contacts) != 2 or len(debounced) != 2:
            raise ValueError("contact continuity states must contain two feet")
        foot_points = _finite_array(state["foot_points"], (2, 3), "foot_points")
        cumulative_values = tuple(state["stance_cumulative"])
        if len(cumulative_values) != 2:
            raise ValueError("stance continuity must contain two cumulative values")
        cumulative = [
            None if value is None else float(value) for value in cumulative_values
        ]
        if any(value is not None and value < 0.0 for value in cumulative):
            raise ValueError("stance cumulative slip must be non-negative")
        pending_values = tuple(state["pending_contact_state"])
        pending_ages = tuple(state["pending_contact_age_s"])
        if len(pending_values) != 2 or len(pending_ages) != 2:
            raise ValueError("pending contact continuity must contain two feet")
        self._previous_contacts = contacts
        self._previous_raw_contacts = raw_contacts
        self._previous_time_s = 0.0
        self._previous_foot_points = foot_points.copy()
        self._stance_cumulative = cumulative
        self._debounced_contacts = list(debounced)
        force_state = state.get("force_contact_state")
        self._force_contact_state = (
            None if force_state is None else [bool(value) for value in force_state]
        )
        self._pending_contact_state = [
            None if value is None else bool(value) for value in pending_values
        ]
        self._pending_contact_since_s = [
            None if age is None else -float(age) for age in pending_ages
        ]
        previous_touchdown = state.get("previous_touchdown_foot")
        if previous_touchdown is not None:
            foot = int(previous_touchdown)
            if foot not in (LEFT_FOOT, RIGHT_FOOT):
                raise ValueError("previous touchdown foot must be left or right")
            self._touchdown_sequence = [foot]
        self._contact_state_source = (
            None
            if state.get("contact_state_source") is None
            else str(state["contact_state_source"])
        )
        self._stance_slip_measurement_source = (
            None
            if state.get("stance_slip_measurement_source") is None
            else str(state["stance_slip_measurement_source"])
        )

    def _debounce_contacts(
        self, raw_contacts: tuple[bool, bool], time_s: float
    ) -> tuple[bool, bool]:
        """Reject contact transitions that do not persist for the frozen window."""

        if self._debounced_contacts is None:
            self._debounced_contacts = [raw_contacts[0], raw_contacts[1]]
            return raw_contacts
        for foot in (LEFT_FOOT, RIGHT_FOOT):
            raw_state = raw_contacts[foot]
            if raw_state == self._debounced_contacts[foot]:
                self._pending_contact_state[foot] = None
                self._pending_contact_since_s[foot] = None
                continue
            if self._pending_contact_state[foot] != raw_state:
                self._pending_contact_state[foot] = raw_state
                self._pending_contact_since_s[foot] = time_s
                continue
            pending_since = self._pending_contact_since_s[foot]
            if (
                pending_since is not None
                and time_s - pending_since >= self.contact_debounce_s
            ):
                self._debounced_contacts[foot] = raw_state
                self._pending_contact_state[foot] = None
                self._pending_contact_since_s[foot] = None
        return (self._debounced_contacts[0], self._debounced_contacts[1])

    def _force_threshold_contacts(self, force_fractions: np.ndarray) -> tuple[bool, bool]:
        """Apply a per-foot Schmitt trigger to normalized contact force."""

        if self._force_contact_state is None:
            self._force_contact_state = [
                bool(value >= self.contact_force_on_fraction_body_weight)
                for value in force_fractions
            ]
        else:
            for foot in (LEFT_FOOT, RIGHT_FOOT):
                threshold = (
                    self.contact_force_off_fraction_body_weight
                    if self._force_contact_state[foot]
                    else self.contact_force_on_fraction_body_weight
                )
                self._force_contact_state[foot] = bool(force_fractions[foot] >= threshold)
        return (self._force_contact_state[0], self._force_contact_state[1])

    def update(self, sample: GaitQualitySubstep) -> None:
        """Consume one physics substep after validating its complete payload."""

        time_s = float(sample.time_s)
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        if self._times and time_s <= self._times[-1]:
            raise ValueError("time_s must be strictly increasing")
        requested = _finite_array(sample.requested_command, (3,), "requested_command")
        effective = _finite_array(sample.effective_command, (3,), "effective_command")
        velocity = _finite_array(
            sample.local_velocity_xyz_mps, (3,), "local_velocity_xyz_mps"
        )
        yaw_rate = float(sample.local_yaw_rate_radps)
        if not np.isfinite(yaw_rate):
            raise ValueError("local_yaw_rate_radps must be finite")
        trunk = _finite_array(
            sample.trunk_position_world_m, (3,), "trunk_position_world_m"
        )
        trunk_yaw = sample.trunk_yaw_world_rad
        if trunk_yaw is None:
            if self._trunk_yaw:
                raise ValueError("trunk yaw availability must remain constant")
            trunk_yaw_value = np.nan
        else:
            trunk_yaw_value = float(trunk_yaw)
            if not np.isfinite(trunk_yaw_value):
                raise ValueError("trunk_yaw_world_rad must be finite")
            if self._times and not self._trunk_yaw:
                raise ValueError("trunk yaw availability must remain constant")
            self._trunk_yaw.append(trunk_yaw_value)
        pose_source = str(sample.trunk_pose_measurement_source)
        if not pose_source:
            raise ValueError("trunk pose measurement source must be non-empty")
        if self._trunk_pose_measurement_source not in (None, pose_source):
            raise ValueError("trunk pose measurement source must remain constant")
        self._trunk_pose_measurement_source = pose_source
        contacts_array = np.asarray(sample.feet_contacts)
        if contacts_array.shape != (2,):
            raise ValueError("feet_contacts must have shape (2,)")
        geometry_contacts = (
            bool(contacts_array[LEFT_FOOT]),
            bool(contacts_array[RIGHT_FOOT]),
        )
        normal_force_fractions: np.ndarray | None = None
        contact_state_source = "boolean_contact"
        if sample.feet_normal_force_fraction_body_weight is not None:
            normal_force_fractions = _finite_array(
                sample.feet_normal_force_fraction_body_weight,
                (2,),
                "feet_normal_force_fraction_body_weight",
            )
            if np.any(normal_force_fractions < 0.0):
                raise ValueError("normalized contact forces must be non-negative")
            raw_contacts = self._force_threshold_contacts(normal_force_fractions)
            contact_state_source = "normal_force_schmitt"
        else:
            raw_contacts = geometry_contacts
        if self._contact_state_source not in (None, contact_state_source):
            raise ValueError("contact state source must remain constant within a segment")
        self._contact_state_source = contact_state_source
        contacts = self._debounce_contacts(raw_contacts, time_s)
        foot_points = _finite_array(
            sample.foot_contact_points_world_m,
            (2, 3),
            "foot_contact_points_world_m",
        )
        joints = _finite_array(
            sample.leg_joint_positions_rad,
            (len(self.joint_names),),
            "leg_joint_positions_rad",
        )
        contact_tangential_speeds: np.ndarray | None = None
        slip_measurement_source = "site_finite_difference"
        if sample.foot_contact_tangential_speeds_mps is not None:
            contact_tangential_speeds = _finite_array(
                sample.foot_contact_tangential_speeds_mps,
                (2,),
                "foot_contact_tangential_speeds_mps",
            )
            if np.any(contact_tangential_speeds < 0.0):
                raise ValueError("contact tangential speeds must be non-negative")
            slip_measurement_source = "force_weighted_contact_point_jacobian"
            self._contact_velocity_payload_sample_count += 1
        if self._stance_slip_measurement_source not in (None, slip_measurement_source):
            raise ValueError("stance slip source must remain constant within a segment")
        self._stance_slip_measurement_source = slip_measurement_source

        if (
            contact_tangential_speeds is not None
            and self.sample_count > 0
        ):
            for foot in (LEFT_FOOT, RIGHT_FOOT):
                if contacts[foot] and raw_contacts[foot]:
                    self._contact_velocity_sample_count += 1
                    self._contact_velocity_sample_counts[foot] += 1

        if self._first_single_support_s is None and contacts[0] != contacts[1]:
            self._first_single_support_s = time_s

        if self._previous_contacts is None:
            for foot in (LEFT_FOOT, RIGHT_FOOT):
                if contacts[foot]:
                    self._stance_cumulative[foot] = 0.0
        else:
            assert self._previous_time_s is not None
            dt = time_s - self._previous_time_s
            if dt < 0.0:
                raise ValueError("contact continuity time cannot move backwards")
            assert self._previous_foot_points is not None
            for foot in (LEFT_FOOT, RIGHT_FOOT):
                was_contact = self._previous_contacts[foot]
                is_contact = contacts[foot]
                if is_contact and not was_contact:
                    self._touchdowns[foot] += 1
                    self._touchdown_sequence.append(foot)
                    self._touchdown_times[foot].append(time_s)
                    self._stance_cumulative[foot] = 0.0
                elif not is_contact and was_contact:
                    cumulative = self._stance_cumulative[foot]
                    self._stance_distances[foot].append(
                        0.0 if cumulative is None else float(cumulative)
                    )
                    self._stance_cumulative[foot] = None
                if dt > 0.0 and (
                    is_contact
                    and was_contact
                    and raw_contacts[foot]
                    and self._previous_raw_contacts is not None
                    and self._previous_raw_contacts[foot]
                ):
                    if contact_tangential_speeds is None:
                        tangential_distance = float(
                            np.linalg.norm(
                                foot_points[foot, :2]
                                - self._previous_foot_points[foot, :2]
                            )
                        )
                        tangential_speed = tangential_distance / dt
                    else:
                        tangential_speed = float(contact_tangential_speeds[foot])
                        tangential_distance = tangential_speed * dt
                    self._stance_slip_speeds[foot].append(tangential_speed)
                    cumulative = self._stance_cumulative[foot]
                    self._stance_cumulative[foot] = (
                        tangential_distance
                        if cumulative is None
                        else cumulative + tangential_distance
                    )

        self._times.append(time_s)
        self._requested.append(requested.copy())
        self._effective.append(effective.copy())
        self._local_velocity.append(velocity.copy())
        self._yaw_rate.append(yaw_rate)
        self._trunk_position.append(trunk.copy())
        self._contacts.append(contacts)
        self._raw_contacts.append(raw_contacts)
        self._foot_points.append(foot_points.copy())
        self._joint_positions.append(joints.copy())
        if normal_force_fractions is not None:
            self._normal_force_fractions.append(normal_force_fractions.copy())
        self._previous_contacts = contacts
        self._previous_raw_contacts = raw_contacts
        self._previous_time_s = time_s
        self._previous_foot_points = foot_points.copy()

    def finalize(self) -> GaitQualityMetrics:
        """Return immutable metrics without mutating the accumulated episode."""

        if self.sample_count < 2:
            raise ValueError("at least two physics substeps are required")
        times = np.asarray(self._times, dtype=np.float64)
        requested = np.asarray(self._requested, dtype=np.float64)
        effective = np.asarray(self._effective, dtype=np.float64)
        velocity = np.asarray(self._local_velocity, dtype=np.float64)
        yaw_rate = np.asarray(self._yaw_rate, dtype=np.float64)
        trunk = np.asarray(self._trunk_position, dtype=np.float64)
        contacts = np.asarray(self._contacts, dtype=np.bool_)
        raw_contacts = np.asarray(self._raw_contacts, dtype=np.bool_)
        joints = np.asarray(self._joint_positions, dtype=np.float64)
        trunk_yaw = (
            np.unwrap(np.asarray(self._trunk_yaw, dtype=np.float64))
            if len(self._trunk_yaw) == self.sample_count
            else None
        )
        timesteps = np.diff(times)
        physics_timestep_s = float(np.median(timesteps))
        maximum_timestep_error_s = float(
            np.max(np.abs(timesteps - physics_timestep_s))
        )
        steady = times >= self.steady_start_s
        if int(np.count_nonzero(steady)) < 2:
            raise ValueError("at least two steady-state substeps are required")

        requested_mean_array = np.mean(requested[steady], axis=0)
        effective_mean_array = np.mean(effective[steady], axis=0)
        command_error = np.abs(effective_mean_array - requested_mean_array)
        command_rms_error = np.sqrt(
            np.mean(np.square(effective[steady] - requested[steady]), axis=0)
        )
        requested_linear_speed = float(np.linalg.norm(requested_mean_array[:2]))
        requested_yaw_rate = float(requested_mean_array[2])

        linear_ratios: np.ndarray | None = None
        if requested_linear_speed > _COMMAND_EPSILON:
            linear_direction = requested_mean_array[:2] / requested_linear_speed
            linear_ratios = velocity[:, :2] @ linear_direction / requested_linear_speed
        yaw_ratios: np.ndarray | None = None
        if abs(requested_yaw_rate) > _COMMAND_EPSILON:
            yaw_ratios = yaw_rate / requested_yaw_rate

        frozen = FROZEN_GAIT_QUALITY_THRESHOLDS
        crossing_kwargs = {
            "rolling_window_s": frozen.startup_rolling_window_s,
            "sustain_s": frozen.startup_sustain_s,
        }
        linear_t30 = _sustained_rolling_ratio_crossing(
            times, linear_ratios, 0.30, **crossing_kwargs
        )
        linear_t75 = _sustained_rolling_ratio_crossing(
            times, linear_ratios, 0.75, **crossing_kwargs
        )
        yaw_t30 = _sustained_rolling_ratio_crossing(
            times, yaw_ratios, 0.30, **crossing_kwargs
        )
        yaw_t75 = _sustained_rolling_ratio_crossing(
            times, yaw_ratios, 0.75, **crossing_kwargs
        )

        active_axis_names: list[str] = []
        active_axis_ratios: list[np.ndarray] = []
        active_axis_rates: list[np.ndarray] = []
        for axis, name in enumerate(("vx", "vy")):
            command_value = float(requested_mean_array[axis])
            if abs(command_value) > _COMMAND_EPSILON:
                active_axis_names.append(name)
                active_axis_ratios.append(velocity[:, axis] / command_value)
                active_axis_rates.append(
                    velocity[:, axis] * np.sign(command_value)
                )
        if abs(requested_yaw_rate) > _COMMAND_EPSILON:
            active_axis_names.append("yaw")
            active_axis_ratios.append(yaw_rate / requested_yaw_rate)
            active_axis_rates.append(yaw_rate * np.sign(requested_yaw_rate))
        startup_axis_t30 = tuple(
            _sustained_rolling_ratio_crossing(
                times, ratio, 0.30, **crossing_kwargs
            )
            for ratio in active_axis_ratios
        )
        startup_axis_t75 = tuple(
            _sustained_rolling_ratio_crossing(
                times, ratio, 0.75, **crossing_kwargs
            )
            for ratio in active_axis_ratios
        )
        active_t30 = list(startup_axis_t30)
        active_t75 = list(startup_axis_t75)

        initial_yaw = 0.0 if trunk_yaw is None else float(trunk_yaw[0])
        cosine, sine = np.cos(initial_yaw), np.sin(initial_yaw)
        world_to_initial_body = np.asarray(((cosine, sine), (-sine, cosine)))
        trunk_relative_body = (world_to_initial_body @ (trunk[:, :2] - trunk[0, :2]).T).T
        startup_mask = times <= frozen.startup_audit_window_s + 1.0e-12
        # Integrate each commanded body-axis rate independently.  Using an
        # initial-world projection here would mix vx/vy during a commanded turn
        # and could hide a genuinely wrong-way compound axis.
        signed_axis_positions = [
            np.concatenate(
                (
                    [0.0],
                    np.cumsum(
                        0.5 * (signed_rate[1:] + signed_rate[:-1]) * timesteps
                    ),
                )
            )
            for signed_rate in active_axis_rates
        ]
        startup_wrong_way_displacement = tuple(
            max(0.0, -float(np.min(position[startup_mask])))
            for position in signed_axis_positions
        )
        startup_maximum_continuous_wrong_way = []
        startup_wrong_way_occupancy = []
        for signed_rate in active_axis_rates:
            maximum_run, _ = _wrong_way_run_metrics(
                times, signed_rate, frozen.startup_audit_window_s
            )
            _, occupancy = _wrong_way_run_metrics(
                times,
                signed_rate,
                frozen.startup_wrong_way_occupancy_window_s,
            )
            startup_maximum_continuous_wrong_way.append(maximum_run)
            startup_wrong_way_occupancy.append(occupancy)

        linear_active = requested_linear_speed > _COMMAND_EPSILON
        yaw_active = abs(requested_yaw_rate) > _COMMAND_EPSILON
        command_class = (
            "stand"
            if not linear_active and not yaw_active
            else "yaw_only"
            if yaw_active and not linear_active
            else "compound"
            if linear_active and yaw_active
            else "pure_translation"
        )
        startup_pure_cross_excursion: float | None = None
        if command_class == "pure_translation":
            direction = requested_mean_array[:2] / requested_linear_speed
            orthogonal = np.asarray((-direction[1], direction[0]))
            startup_pure_cross_excursion = float(
                np.max(np.abs(trunk_relative_body[startup_mask] @ orthogonal))
            )
        startup_yaw_only_planar_excursion = (
            float(
                np.max(
                    np.linalg.norm(
                        trunk[startup_mask, :2] - trunk[0, :2], axis=1
                    )
                )
            )
            if command_class == "yaw_only"
            else None
        )

        mean_velocity = np.mean(velocity[steady], axis=0)
        mean_yaw_rate = float(np.mean(yaw_rate[steady]))
        linear_tracking: float | None = None
        cross_drift: float | None = None
        cross_fraction: float | None = None
        if requested_linear_speed > _COMMAND_EPSILON:
            direction = requested_mean_array[:2] / requested_linear_speed
            linear_tracking = float(mean_velocity[:2] @ direction / requested_linear_speed)
            orthogonal = np.asarray((-direction[1], direction[0]))
            cross_drift = abs(float(mean_velocity[:2] @ orthogonal))
            cross_fraction = cross_drift / requested_linear_speed
        yaw_tracking = (
            None
            if abs(requested_yaw_rate) <= _COMMAND_EPSILON
            else mean_yaw_rate / requested_yaw_rate
        )

        uncommanded_yaw_rate: float | None = None
        uncommanded_heading_drift: float | None = None
        if abs(requested_yaw_rate) <= _COMMAND_EPSILON:
            uncommanded_yaw_rate = abs(mean_yaw_rate)
            dt = np.diff(times)
            integrated_yaw = np.sum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) * dt)
            uncommanded_heading_drift = abs(float(integrated_yaw))

        net_displacement = trunk[-1, :2] - trunk[0, :2]
        yaw_only = (
            requested_linear_speed <= _COMMAND_EPSILON
            and abs(requested_yaw_rate) > _COMMAND_EPSILON
        )
        yaw_only_drift_rate = (
            float(np.linalg.norm(mean_velocity[:2])) if yaw_only else None
        )
        yaw_only_net_drift = (
            float(np.linalg.norm(net_displacement)) if yaw_only else None
        )

        ideal_xy, ideal_yaw = _integrate_body_twist_trajectory(
            times,
            requested,
            trunk[0, :2],
            initial_yaw,
        )
        heading_error = (
            None
            if trunk_yaw is None
            else np.abs(np.asarray(_wrap_angle_rad(trunk_yaw - ideal_yaw)))
        )
        trajectory_heading_endpoint_error = (
            None if heading_error is None else float(heading_error[-1])
        )
        trajectory_heading_maximum_error = (
            None if heading_error is None else float(np.max(heading_error))
        )

        pure_endpoint_primary_ratio: float | None = None
        pure_endpoint_cross_error: float | None = None
        pure_endpoint_cross_fraction: float | None = None
        pure_cross_path_p95: float | None = None
        pure_cross_path_maximum: float | None = None
        cumulative_backtracking: float | None = None
        compound_endpoint_position_error: float | None = None
        compound_endpoint_position_error_fraction: float | None = None
        compound_path_p95: float | None = None
        compound_path_maximum: float | None = None
        yaw_only_path_radius_p95: float | None = None
        yaw_only_path_radius_maximum: float | None = None
        stand_path_radius_maximum: float | None = None
        stand_heading_excursion: float | None = None

        if command_class == "pure_translation":
            direction_body = requested_mean_array[:2] / requested_linear_speed
            orthogonal_body = np.asarray((-direction_body[1], direction_body[0]))
            primary_path = trunk_relative_body @ direction_body
            cross_path = np.abs(trunk_relative_body @ orthogonal_body)
            nominal_primary = requested_linear_speed * float(times[-1] - times[0])
            pure_endpoint_primary_ratio = (
                None
                if nominal_primary <= _COMMAND_EPSILON
                else float(primary_path[-1] / nominal_primary)
            )
            pure_endpoint_cross_error = float(cross_path[-1])
            pure_endpoint_cross_fraction = (
                None
                if nominal_primary <= _COMMAND_EPSILON
                else pure_endpoint_cross_error / nominal_primary
            )
            pure_cross_path_p95 = float(np.percentile(cross_path, 95))
            pure_cross_path_maximum = float(np.max(cross_path))
            cumulative_backtracking = float(
                np.sum(np.maximum(0.0, -np.diff(primary_path)))
            )
        elif command_class == "compound":
            position_error = np.linalg.norm(trunk[:, :2] - ideal_xy, axis=1)
            compound_endpoint_position_error = float(position_error[-1])
            nominal_arc_length = float(
                np.sum(np.linalg.norm(np.diff(ideal_xy, axis=0), axis=1))
            )
            compound_endpoint_position_error_fraction = (
                None
                if nominal_arc_length <= _COMMAND_EPSILON
                else compound_endpoint_position_error / nominal_arc_length
            )
            compound_path_p95 = float(np.percentile(position_error, 95))
            compound_path_maximum = float(np.max(position_error))
            direction_body = requested_mean_array[:2] / requested_linear_speed
            primary_path = trunk_relative_body @ direction_body
            cumulative_backtracking = float(
                np.sum(np.maximum(0.0, -np.diff(primary_path)))
            )
        elif command_class == "yaw_only":
            path_radius = np.linalg.norm(trunk[:, :2] - trunk[0, :2], axis=1)
            yaw_only_path_radius_p95 = float(np.percentile(path_radius, 95))
            yaw_only_path_radius_maximum = float(np.max(path_radius))
        else:
            path_radius = np.linalg.norm(trunk[:, :2] - trunk[0, :2], axis=1)
            stand_path_radius_maximum = float(np.max(path_radius))
            stand_heading_excursion = (
                None
                if trunk_yaw is None
                else float(
                    np.max(
                        np.abs(
                            np.asarray(_wrap_angle_rad(trunk_yaw - trunk_yaw[0]))
                        )
                    )
                )
            )

        left_contact_rate = float(np.mean(contacts[:, LEFT_FOOT]))
        right_contact_rate = float(np.mean(contacts[:, RIGHT_FOOT]))
        single_support = np.logical_xor(contacts[:, LEFT_FOOT], contacts[:, RIGHT_FOOT])
        double_support = np.logical_and(contacts[:, LEFT_FOOT], contacts[:, RIGHT_FOOT])
        flight = np.logical_not(np.logical_or(contacts[:, LEFT_FOOT], contacts[:, RIGHT_FOOT]))
        alternating_fraction: float | None = None
        if len(self._touchdown_sequence) >= 2:
            alternating = sum(
                left != right
                for left, right in zip(
                    self._touchdown_sequence, self._touchdown_sequence[1:]
                )
            )
            alternating_fraction = alternating / (len(self._touchdown_sequence) - 1)

        stance_distances = [list(values) for values in self._stance_distances]
        for foot in (LEFT_FOOT, RIGHT_FOOT):
            if self._previous_contacts is not None and self._previous_contacts[foot]:
                cumulative = self._stance_cumulative[foot]
                stance_distances[foot].append(
                    0.0 if cumulative is None else float(cumulative)
                )
        combined_slip_speeds = (
            self._stance_slip_speeds[LEFT_FOOT]
            + self._stance_slip_speeds[RIGHT_FOOT]
        )
        all_stance_distances = stance_distances[LEFT_FOOT] + stance_distances[RIGHT_FOOT]
        left_slip_speeds = self._stance_slip_speeds[LEFT_FOOT]
        right_slip_speeds = self._stance_slip_speeds[RIGHT_FOOT]

        steady_joints = joints[steady]
        joint_ranges = np.ptp(steady_joints, axis=0)
        left_cadence_joint_index, right_cadence_joint_index = (
            self._select_cadence_pair(joint_ranges)
        )
        left_cadence = _dominant_frequency_hz(
            times[steady], steady_joints[:, left_cadence_joint_index]
        )
        right_cadence = _dominant_frequency_hz(
            times[steady], steady_joints[:, right_cadence_joint_index]
        )
        debounce_sensitivity = {
            f"{int(round(window_s * 1000.0))}ms": _contact_sequence_summary(
                times, raw_contacts, window_s
            )
            for window_s in (0.010, 0.020, 0.030, 0.040)
        }
        duration_s = float(times[-1] - times[0])
        left_touchdown_metrics = _touchdown_interval_metrics(
            self._touchdown_times[LEFT_FOOT], duration_s
        )
        right_touchdown_metrics = _touchdown_interval_metrics(
            self._touchdown_times[RIGHT_FOOT], duration_s
        )

        stable_stance_intervals = (
            contacts[:-1]
            & contacts[1:]
            & raw_contacts[:-1]
            & raw_contacts[1:]
        )
        stance_contact_interval_counts = np.count_nonzero(
            stable_stance_intervals, axis=0
        ).astype(int)
        expected_contact_velocity_samples = np.asarray(
            (left_contact_rate, right_contact_rate), dtype=np.float64
        ) * (self.sample_count - 1)
        per_foot_velocity_coverage = tuple(
            None
            if expected_contact_velocity_samples[foot] <= 0.0
            else self._contact_velocity_sample_counts[foot]
            / float(expected_contact_velocity_samples[foot])
            for foot in (LEFT_FOOT, RIGHT_FOOT)
        )
        total_contact_intervals = float(np.sum(expected_contact_velocity_samples))
        contact_velocity_coverage = (
            None
            if total_contact_intervals <= 0.0
            else self._contact_velocity_sample_count / total_contact_intervals
        )

        steady_mean_total_normal_force: float | None = None
        left_normal_impulse_share: float | None = None
        right_normal_impulse_share: float | None = None
        total_normal_force_p99: float | None = None
        if len(self._normal_force_fractions) == self.sample_count:
            normal_forces = np.asarray(
                self._normal_force_fractions, dtype=np.float64
            )
            total_normal_force = np.sum(normal_forces, axis=1)
            steady_mean_total_normal_force = float(
                np.mean(total_normal_force[steady])
            )
            total_normal_force_p99 = float(np.percentile(total_normal_force, 99))
            impulses = np.asarray(
                [
                    np.trapezoid(normal_forces[:, foot], times)
                    for foot in (LEFT_FOOT, RIGHT_FOOT)
                ],
                dtype=np.float64,
            )
            total_impulse = float(np.sum(impulses))
            if total_impulse > 0.0:
                left_normal_impulse_share = float(impulses[LEFT_FOOT] / total_impulse)
                right_normal_impulse_share = float(impulses[RIGHT_FOOT] / total_impulse)

        return GaitQualityMetrics(
            sample_count=self.sample_count,
            duration_s=duration_s,
            steady_sample_count=int(np.count_nonzero(steady)),
            requested_command_mean=tuple(float(value) for value in requested_mean_array),
            effective_command_mean=tuple(float(value) for value in effective_mean_array),
            requested_effective_axis_error=tuple(float(value) for value in command_error),
            requested_effective_axis_rms_error=tuple(
                float(value) for value in command_rms_error
            ),
            linear_t30_s=linear_t30,
            linear_t75_s=linear_t75,
            yaw_t30_s=yaw_t30,
            yaw_t75_s=yaw_t75,
            t30_s=_maximum_optional(active_t30),
            t75_s=_maximum_optional(active_t75),
            first_single_support_s=self._first_single_support_s,
            left_contact_rate=left_contact_rate,
            right_contact_rate=right_contact_rate,
            single_support_rate=float(np.mean(single_support)),
            double_support_rate=float(np.mean(double_support)),
            flight_rate=float(np.mean(flight)),
            left_step_count=self._touchdowns[LEFT_FOOT],
            right_step_count=self._touchdowns[RIGHT_FOOT],
            step_count_imbalance=abs(
                self._touchdowns[LEFT_FOOT] - self._touchdowns[RIGHT_FOOT]
            ),
            contact_duty_imbalance=abs(left_contact_rate - right_contact_rate),
            alternating_touchdown_fraction=alternating_fraction,
            left_stance_count=len(stance_distances[LEFT_FOOT]),
            right_stance_count=len(stance_distances[RIGHT_FOOT]),
            left_stance_slip_rms_mps=_rms(left_slip_speeds),
            right_stance_slip_rms_mps=_rms(right_slip_speeds),
            left_stance_slip_p95_mps=(
                None
                if not left_slip_speeds
                else float(np.percentile(left_slip_speeds, 95))
            ),
            right_stance_slip_p95_mps=(
                None
                if not right_slip_speeds
                else float(np.percentile(right_slip_speeds, 95))
            ),
            left_maximum_per_stance_cumulative_slip_m=(
                None
                if not stance_distances[LEFT_FOOT]
                else max(stance_distances[LEFT_FOOT])
            ),
            right_maximum_per_stance_cumulative_slip_m=(
                None
                if not stance_distances[RIGHT_FOOT]
                else max(stance_distances[RIGHT_FOOT])
            ),
            stance_slip_rms_mps=_rms(combined_slip_speeds),
            stance_slip_p95_mps=(
                None
                if not combined_slip_speeds
                else float(np.percentile(combined_slip_speeds, 95))
            ),
            maximum_per_stance_cumulative_slip_m=(
                None if not all_stance_distances else max(all_stance_distances)
            ),
            joint_names=self.joint_names,
            joint_ranges_rad=tuple(float(value) for value in joint_ranges),
            left_joint_indices=self.left_joint_indices,
            right_joint_indices=self.right_joint_indices,
            left_cadence_joint=self.joint_names[left_cadence_joint_index],
            right_cadence_joint=self.joint_names[right_cadence_joint_index],
            left_joint_cadence_hz=left_cadence,
            right_joint_cadence_hz=right_cadence,
            steady_mean_local_velocity_xyz_mps=tuple(
                float(value) for value in mean_velocity
            ),
            steady_mean_local_yaw_rate_radps=mean_yaw_rate,
            steady_linear_tracking_ratio=linear_tracking,
            steady_yaw_tracking_ratio=yaw_tracking,
            steady_cross_drift_mps=cross_drift,
            steady_cross_drift_fraction=cross_fraction,
            uncommanded_yaw_rate_radps=uncommanded_yaw_rate,
            uncommanded_heading_drift_rad=uncommanded_heading_drift,
            yaw_only_planar_drift_mps=yaw_only_drift_rate,
            yaw_only_net_drift_m=yaw_only_net_drift,
            net_trunk_displacement_xy_m=(
                float(net_displacement[0]),
                float(net_displacement[1]),
            ),
            contact_state_source=(self._contact_state_source or "boolean_contact"),
            contact_force_sample_count=len(self._normal_force_fractions),
            stance_slip_measurement_source=(
                self._stance_slip_measurement_source or "site_finite_difference"
            ),
            contact_velocity_sample_count=self._contact_velocity_sample_count,
            contact_debounce_sensitivity=debounce_sensitivity,
            physics_timestep_s=physics_timestep_s,
            maximum_timestep_error_s=maximum_timestep_error_s,
            trunk_pose_measurement_source=(
                self._trunk_pose_measurement_source or "world_trunk_position_only"
            ),
            trunk_yaw_sample_count=(
                0 if trunk_yaw is None else int(len(trunk_yaw))
            ),
            startup_active_axis_names=tuple(active_axis_names),
            startup_axis_t30_s=startup_axis_t30,
            startup_axis_t75_s=startup_axis_t75,
            startup_axis_wrong_way_displacement=startup_wrong_way_displacement,
            startup_axis_maximum_continuous_wrong_way_s=tuple(
                float(value) for value in startup_maximum_continuous_wrong_way
            ),
            startup_axis_wrong_way_occupancy=tuple(
                float(value) for value in startup_wrong_way_occupancy
            ),
            startup_pure_cross_excursion_m=startup_pure_cross_excursion,
            startup_yaw_only_planar_excursion_m=(
                startup_yaw_only_planar_excursion
            ),
            command_class=command_class,
            ideal_endpoint_xy_m=(
                float(ideal_xy[-1, 0]), float(ideal_xy[-1, 1])
            ),
            ideal_endpoint_yaw_rad=float(ideal_yaw[-1]),
            actual_endpoint_yaw_rad=(
                None if trunk_yaw is None else float(trunk_yaw[-1])
            ),
            pure_endpoint_primary_ratio=pure_endpoint_primary_ratio,
            pure_endpoint_cross_error_m=pure_endpoint_cross_error,
            pure_endpoint_cross_error_fraction=pure_endpoint_cross_fraction,
            pure_cross_path_p95_error_m=pure_cross_path_p95,
            pure_cross_path_maximum_error_m=pure_cross_path_maximum,
            trajectory_heading_endpoint_error_rad=(
                trajectory_heading_endpoint_error
            ),
            trajectory_heading_maximum_error_rad=(
                trajectory_heading_maximum_error
            ),
            cumulative_backtracking_m=cumulative_backtracking,
            compound_endpoint_position_error_m=(
                compound_endpoint_position_error
            ),
            compound_endpoint_position_error_fraction=(
                compound_endpoint_position_error_fraction
            ),
            compound_path_p95_error_m=compound_path_p95,
            compound_path_maximum_error_m=compound_path_maximum,
            yaw_only_path_radius_p95_m=yaw_only_path_radius_p95,
            yaw_only_path_radius_maximum_m=yaw_only_path_radius_maximum,
            stand_path_radius_maximum_m=stand_path_radius_maximum,
            stand_heading_excursion_rad=stand_heading_excursion,
            left_touchdown_timestamps_s=tuple(
                float(value) for value in self._touchdown_times[LEFT_FOOT]
            ),
            right_touchdown_timestamps_s=tuple(
                float(value) for value in self._touchdown_times[RIGHT_FOOT]
            ),
            left_touchdown_count_cadence_hz=left_touchdown_metrics[0],
            right_touchdown_count_cadence_hz=right_touchdown_metrics[0],
            left_touchdown_median_cadence_hz=left_touchdown_metrics[1],
            right_touchdown_median_cadence_hz=right_touchdown_metrics[1],
            left_touchdown_interval_cv=left_touchdown_metrics[2],
            right_touchdown_interval_cv=right_touchdown_metrics[2],
            left_maximum_same_foot_touchdown_gap_s=left_touchdown_metrics[3],
            right_maximum_same_foot_touchdown_gap_s=right_touchdown_metrics[3],
            contact_velocity_payload_sample_count=(
                self._contact_velocity_payload_sample_count
            ),
            left_contact_velocity_sample_count=(
                self._contact_velocity_sample_counts[LEFT_FOOT]
            ),
            right_contact_velocity_sample_count=(
                self._contact_velocity_sample_counts[RIGHT_FOOT]
            ),
            left_stance_contact_interval_count=int(
                stance_contact_interval_counts[LEFT_FOOT]
            ),
            right_stance_contact_interval_count=int(
                stance_contact_interval_counts[RIGHT_FOOT]
            ),
            left_contact_velocity_expected_sample_count=float(
                expected_contact_velocity_samples[LEFT_FOOT]
            ),
            right_contact_velocity_expected_sample_count=float(
                expected_contact_velocity_samples[RIGHT_FOOT]
            ),
            contact_velocity_coverage=contact_velocity_coverage,
            left_contact_velocity_coverage=per_foot_velocity_coverage[LEFT_FOOT],
            right_contact_velocity_coverage=per_foot_velocity_coverage[RIGHT_FOOT],
            steady_mean_total_normal_force_fraction_body_weight=(
                steady_mean_total_normal_force
            ),
            left_normal_impulse_share=left_normal_impulse_share,
            right_normal_impulse_share=right_normal_impulse_share,
            total_normal_force_p99_fraction_body_weight=(
                total_normal_force_p99
            ),
        )


def gait_quality_acceptance(
    metrics: GaitQualityMetrics,
    thresholds: GaitQualityThresholds = FROZEN_GAIT_QUALITY_THRESHOLDS,
) -> GaitQualityAcceptance:
    """Apply the frozen strict gait-quality contract to accumulated metrics."""

    requested = np.asarray(metrics.requested_command_mean, dtype=np.float64)
    linear_speed = float(np.linalg.norm(requested[:2]))
    yaw_magnitude = abs(float(requested[2]))
    linear_active = linear_speed > _COMMAND_EPSILON
    yaw_active = yaw_magnitude > _COMMAND_EPSILON
    moving = linear_active or yaw_active
    compound = linear_active and yaw_active
    yaw_only = yaw_active and not linear_active
    stand = not moving
    expected_command_class = (
        "stand"
        if stand
        else "yaw_only"
        if yaw_only
        else "compound"
        if compound
        else "pure_translation"
    )

    checks: dict[str, bool] = {}
    applicable: dict[str, bool] = {}

    def add(name: str, value: bool, *, applies: bool = True) -> None:
        applicable[name] = applies
        checks[name] = bool(value) if applies else True

    def within(value: float, lower: float, upper: float) -> bool:
        return lower - 1.0e-9 <= float(value) <= upper + 1.0e-9

    add("measurement_schema", metrics.measurement_schema_version == 2)
    add("command_class", metrics.command_class == expected_command_class)
    add("motion_commanded", moving, applies=moving)
    add(
        "trunk_pose_complete",
        metrics.trunk_yaw_sample_count == metrics.sample_count
        and metrics.trunk_pose_measurement_source
        != "world_trunk_position_only"
        and metrics.actual_endpoint_yaw_rad is not None,
    )
    add(
        "requested_effective_command",
        max(metrics.requested_effective_axis_error)
        <= thresholds.maximum_requested_effective_axis_error
        and max(metrics.requested_effective_axis_rms_error)
        <= thresholds.maximum_requested_effective_axis_error,
    )
    add(
        "startup_t30",
        metrics.t30_s is not None and metrics.t30_s <= thresholds.maximum_t30_s,
        applies=moving,
    )
    add(
        "startup_t75",
        metrics.t75_s is not None and metrics.t75_s <= thresholds.maximum_t75_s,
        applies=moving,
    )
    startup_axis_payload_complete = bool(
        len(metrics.startup_active_axis_names)
        == len(metrics.startup_axis_t30_s)
        == len(metrics.startup_axis_t75_s)
        == len(metrics.startup_axis_wrong_way_displacement)
        == len(metrics.startup_axis_maximum_continuous_wrong_way_s)
        == len(metrics.startup_axis_wrong_way_occupancy)
        and len(metrics.startup_active_axis_names) > 0
    )
    add(
        "startup_axis_payload_complete",
        startup_axis_payload_complete,
        applies=moving,
    )
    add(
        "startup_axis_sustained_t30",
        startup_axis_payload_complete
        and all(
            value is not None and value <= thresholds.maximum_t30_s
            for value in metrics.startup_axis_t30_s
        ),
        applies=moving,
    )
    add(
        "startup_axis_sustained_t75",
        startup_axis_payload_complete
        and all(
            value is not None and value <= thresholds.maximum_t75_s
            for value in metrics.startup_axis_t75_s
        ),
        applies=moving,
    )
    add(
        "startup_wrong_way_displacement",
        startup_axis_payload_complete
        and max(metrics.startup_axis_wrong_way_displacement, default=np.inf)
        <= thresholds.maximum_startup_wrong_way_displacement_m,
        applies=moving,
    )
    add(
        "startup_continuous_wrong_way",
        startup_axis_payload_complete
        and max(
            metrics.startup_axis_maximum_continuous_wrong_way_s,
            default=np.inf,
        )
        <= thresholds.maximum_startup_continuous_wrong_way_s,
        applies=moving,
    )
    add(
        "startup_wrong_way_occupancy",
        startup_axis_payload_complete
        and max(metrics.startup_axis_wrong_way_occupancy, default=np.inf)
        <= thresholds.maximum_startup_wrong_way_occupancy,
        applies=moving,
    )
    add(
        "startup_pure_cross_excursion",
        metrics.startup_pure_cross_excursion_m is not None
        and metrics.startup_pure_cross_excursion_m
        <= thresholds.maximum_startup_pure_cross_excursion_m,
        applies=linear_active and not yaw_active,
    )
    add(
        "startup_yaw_only_planar_excursion",
        metrics.startup_yaw_only_planar_excursion_m is not None
        and metrics.startup_yaw_only_planar_excursion_m
        <= thresholds.maximum_startup_yaw_only_planar_excursion_m,
        applies=yaw_only,
    )
    add(
        "first_single_support",
        metrics.first_single_support_s is not None
        and metrics.first_single_support_s
        <= thresholds.maximum_first_single_support_s,
        applies=moving,
    )
    add(
        "single_support_rate",
        thresholds.minimum_single_support_rate
        <= metrics.single_support_rate
        <= thresholds.maximum_single_support_rate,
        applies=moving,
    )
    add(
        "flight_rate",
        metrics.flight_rate <= thresholds.maximum_flight_rate,
        applies=moving,
    )
    add(
        "left_right_step_count",
        metrics.left_step_count >= thresholds.minimum_steps_per_foot
        and metrics.right_step_count >= thresholds.minimum_steps_per_foot
        and metrics.step_count_imbalance <= thresholds.maximum_step_count_imbalance,
        applies=moving,
    )
    add(
        "left_right_contact_duty",
        metrics.contact_duty_imbalance
        <= thresholds.maximum_contact_duty_imbalance,
        applies=moving,
    )
    add(
        "alternating_touchdowns",
        metrics.alternating_touchdown_fraction is not None
        and metrics.alternating_touchdown_fraction
        >= thresholds.minimum_alternating_touchdown_fraction,
        applies=moving,
    )

    debounce_keys = ("10ms", "20ms", "30ms", "40ms")
    debounce = metrics.contact_debounce_sensitivity
    debounce_payload_complete = bool(
        isinstance(debounce, Mapping) and set(debounce) == set(debounce_keys)
    )
    add("debounce_payload_complete", debounce_payload_complete, applies=moving)
    debounce_windows = (
        [debounce[key] for key in debounce_keys]
        if debounce_payload_complete
        else []
    )
    per_window_debounce_pass = bool(
        debounce_windows
        and all(
            thresholds.minimum_single_support_rate
            <= float(window["single_support_rate"])
            <= thresholds.maximum_single_support_rate
            and float(window["flight_rate"]) <= thresholds.maximum_flight_rate
            and float(window["contact_duty_imbalance"])
            <= thresholds.maximum_contact_duty_imbalance
            and int(window["step_count_imbalance"])
            <= thresholds.maximum_step_count_imbalance
            and window["alternating_touchdown_fraction"] is not None
            and float(window["alternating_touchdown_fraction"])
            >= thresholds.minimum_alternating_touchdown_fraction
            for window in debounce_windows
        )
    )
    add("debounce_all_windows_quality", per_window_debounce_pass, applies=moving)
    if debounce_windows:
        left_touchdown_span = max(
            int(window["left_touchdowns"]) for window in debounce_windows
        ) - min(int(window["left_touchdowns"]) for window in debounce_windows)
        right_touchdown_span = max(
            int(window["right_touchdowns"]) for window in debounce_windows
        ) - min(int(window["right_touchdowns"]) for window in debounce_windows)
        single_support_span = max(
            float(window["single_support_rate"]) for window in debounce_windows
        ) - min(float(window["single_support_rate"]) for window in debounce_windows)
        left_contact_span = max(
            float(window["left_contact_rate"]) for window in debounce_windows
        ) - min(float(window["left_contact_rate"]) for window in debounce_windows)
        right_contact_span = max(
            float(window["right_contact_rate"]) for window in debounce_windows
        ) - min(float(window["right_contact_rate"]) for window in debounce_windows)
        flight_span = max(
            float(window["flight_rate"]) for window in debounce_windows
        ) - min(float(window["flight_rate"]) for window in debounce_windows)
    else:
        left_touchdown_span = right_touchdown_span = np.inf
        single_support_span = left_contact_span = right_contact_span = np.inf
        flight_span = np.inf
    add(
        "debounce_touchdown_count_robustness",
        left_touchdown_span <= thresholds.maximum_debounce_touchdown_count_span
        and right_touchdown_span <= thresholds.maximum_debounce_touchdown_count_span,
        applies=moving,
    )
    add(
        "debounce_support_rate_robustness",
        single_support_span
        <= thresholds.maximum_debounce_single_support_rate_span
        and left_contact_span <= thresholds.maximum_debounce_contact_rate_span
        and right_contact_span <= thresholds.maximum_debounce_contact_rate_span
        and flight_span <= thresholds.maximum_debounce_flight_rate_span,
        applies=moving,
    )

    add(
        "force_payload_complete",
        metrics.contact_state_source == "normal_force_schmitt"
        and metrics.contact_force_sample_count == metrics.sample_count,
    )
    add(
        "contact_velocity_payload_complete",
        metrics.stance_slip_measurement_source
        == "force_weighted_contact_point_jacobian"
        and metrics.contact_velocity_payload_sample_count == metrics.sample_count,
    )
    contact_velocity_coverage_available = bool(
        metrics.contact_velocity_coverage is not None
        and metrics.left_contact_velocity_coverage is not None
        and metrics.right_contact_velocity_coverage is not None
        and metrics.left_contact_velocity_expected_sample_count > 0.0
        and metrics.right_contact_velocity_expected_sample_count > 0.0
        and metrics.contact_velocity_sample_count
        == metrics.left_contact_velocity_sample_count
        + metrics.right_contact_velocity_sample_count
    )
    add("contact_velocity_coverage_available", contact_velocity_coverage_available)
    add(
        "contact_velocity_coverage",
        contact_velocity_coverage_available
        and thresholds.minimum_contact_velocity_coverage
        <= metrics.contact_velocity_coverage
        <= thresholds.maximum_contact_velocity_coverage
        and thresholds.minimum_contact_velocity_coverage
        <= metrics.left_contact_velocity_coverage
        <= thresholds.maximum_contact_velocity_coverage
        and thresholds.minimum_contact_velocity_coverage
        <= metrics.right_contact_velocity_coverage
        <= thresholds.maximum_contact_velocity_coverage,
    )
    force_distribution_available = bool(
        metrics.steady_mean_total_normal_force_fraction_body_weight is not None
        and metrics.left_normal_impulse_share is not None
        and metrics.right_normal_impulse_share is not None
        and metrics.total_normal_force_p99_fraction_body_weight is not None
    )
    add("normal_force_distribution_available", force_distribution_available)
    add(
        "steady_total_normal_force",
        force_distribution_available
        and thresholds.minimum_steady_total_normal_force_fraction_body_weight
        <= metrics.steady_mean_total_normal_force_fraction_body_weight
        <= thresholds.maximum_steady_total_normal_force_fraction_body_weight,
    )
    add(
        "left_right_normal_impulse_share",
        force_distribution_available
        and thresholds.minimum_per_foot_normal_impulse_share
        <= metrics.left_normal_impulse_share
        <= thresholds.maximum_per_foot_normal_impulse_share
        and thresholds.minimum_per_foot_normal_impulse_share
        <= metrics.right_normal_impulse_share
        <= thresholds.maximum_per_foot_normal_impulse_share
        and abs(
            metrics.left_normal_impulse_share
            + metrics.right_normal_impulse_share
            - 1.0
        )
        <= 1.0e-9,
    )
    add(
        "total_normal_force_p99",
        force_distribution_available
        and metrics.total_normal_force_p99_fraction_body_weight
        <= thresholds.maximum_total_normal_force_p99_fraction_body_weight,
    )

    slip_available = (
        metrics.left_stance_count > 0
        and metrics.right_stance_count > 0
        and metrics.left_stance_slip_rms_mps is not None
        and metrics.right_stance_slip_rms_mps is not None
        and metrics.left_stance_slip_p95_mps is not None
        and metrics.right_stance_slip_p95_mps is not None
        and metrics.left_maximum_per_stance_cumulative_slip_m is not None
        and metrics.right_maximum_per_stance_cumulative_slip_m is not None
        and metrics.stance_slip_rms_mps is not None
        and metrics.stance_slip_p95_mps is not None
        and metrics.maximum_per_stance_cumulative_slip_m is not None
    )
    add("stance_slip_available", slip_available, applies=moving)
    add(
        "stance_slip_rms",
        slip_available
        and metrics.stance_slip_rms_mps
        <= thresholds.maximum_stance_slip_rms_mps
        and metrics.left_stance_slip_rms_mps
        <= thresholds.maximum_stance_slip_rms_mps
        and metrics.right_stance_slip_rms_mps
        <= thresholds.maximum_stance_slip_rms_mps,
        applies=moving,
    )
    add(
        "stance_slip_p95",
        slip_available
        and metrics.stance_slip_p95_mps
        <= thresholds.maximum_stance_slip_p95_mps
        and metrics.left_stance_slip_p95_mps
        <= thresholds.maximum_stance_slip_p95_mps
        and metrics.right_stance_slip_p95_mps
        <= thresholds.maximum_stance_slip_p95_mps,
        applies=moving,
    )
    add(
        "per_stance_cumulative_slip",
        slip_available
        and metrics.maximum_per_stance_cumulative_slip_m
        <= thresholds.maximum_per_stance_cumulative_slip_m
        and metrics.left_maximum_per_stance_cumulative_slip_m
        <= thresholds.maximum_per_stance_cumulative_slip_m
        and metrics.right_maximum_per_stance_cumulative_slip_m
        <= thresholds.maximum_per_stance_cumulative_slip_m,
        applies=moving,
    )

    ranges = np.asarray(metrics.joint_ranges_rad, dtype=np.float64)
    left_ranges = ranges[np.asarray(metrics.left_joint_indices, dtype=np.int64)]
    right_ranges = ranges[np.asarray(metrics.right_joint_indices, dtype=np.int64)]
    periodic_range = thresholds.minimum_periodic_joint_range_rad
    left_periodic = int(np.count_nonzero(left_ranges >= periodic_range))
    right_periodic = int(np.count_nonzero(right_ranges >= periodic_range))
    add(
        "joint_periodic_range",
        left_periodic >= thresholds.minimum_periodic_joints_per_leg
        and right_periodic >= thresholds.minimum_periodic_joints_per_leg
        and float(np.max(ranges)) <= thresholds.maximum_periodic_joint_range_rad,
        applies=moving,
    )
    left_cadence_index = metrics.joint_names.index(metrics.left_cadence_joint)
    right_cadence_index = metrics.joint_names.index(metrics.right_cadence_joint)
    add(
        "cadence_joint_range",
        periodic_range <= ranges[left_cadence_index]
        <= thresholds.maximum_periodic_joint_range_rad
        and periodic_range <= ranges[right_cadence_index]
        <= thresholds.maximum_periodic_joint_range_rad,
        applies=moving,
    )
    cadence_available = (
        metrics.left_joint_cadence_hz is not None
        and metrics.right_joint_cadence_hz is not None
    )
    add("joint_cadence_available", cadence_available, applies=moving)
    add(
        "joint_cadence",
        cadence_available
        and within(
            metrics.left_joint_cadence_hz,
            thresholds.minimum_joint_cadence_hz,
            thresholds.maximum_joint_cadence_hz,
        )
        and within(
            metrics.right_joint_cadence_hz,
            thresholds.minimum_joint_cadence_hz,
            thresholds.maximum_joint_cadence_hz,
        ),
        applies=moving,
    )
    add(
        "left_right_joint_cadence",
        cadence_available
        and abs(metrics.left_joint_cadence_hz - metrics.right_joint_cadence_hz)
        <= thresholds.maximum_left_right_cadence_difference_hz,
        applies=moving,
    )
    contact_count_cadence_available = bool(
        metrics.left_touchdown_count_cadence_hz is not None
        and metrics.right_touchdown_count_cadence_hz is not None
        and metrics.duration_s > 0.0
    )
    add(
        "contact_count_cadence_available",
        contact_count_cadence_available,
        applies=moving,
    )
    add(
        "contact_count_cadence",
        contact_count_cadence_available
        and within(
            metrics.left_touchdown_count_cadence_hz,
            thresholds.minimum_contact_cadence_hz,
            thresholds.maximum_contact_cadence_hz,
        )
        and within(
            metrics.right_touchdown_count_cadence_hz,
            thresholds.minimum_contact_cadence_hz,
            thresholds.maximum_contact_cadence_hz,
        ),
        applies=moving,
    )
    count_joint_tolerance = (
        thresholds.maximum_touchdown_joint_cadence_difference_hz
        + (1.0 / metrics.duration_s if metrics.duration_s > 0.0 else np.inf)
    )
    add(
        "contact_count_vs_joint_cadence",
        contact_count_cadence_available
        and cadence_available
        and abs(
            metrics.left_touchdown_count_cadence_hz
            - metrics.left_joint_cadence_hz
        )
        <= count_joint_tolerance
        and abs(
            metrics.right_touchdown_count_cadence_hz
            - metrics.right_joint_cadence_hz
        )
        <= count_joint_tolerance,
        applies=moving,
    )
    timestamp_cadence_available = bool(
        metrics.left_touchdown_median_cadence_hz is not None
        and metrics.right_touchdown_median_cadence_hz is not None
        and metrics.left_touchdown_interval_cv is not None
        and metrics.right_touchdown_interval_cv is not None
        and metrics.left_maximum_same_foot_touchdown_gap_s is not None
        and metrics.right_maximum_same_foot_touchdown_gap_s is not None
        and len(metrics.left_touchdown_timestamps_s) == metrics.left_step_count
        and len(metrics.right_touchdown_timestamps_s) == metrics.right_step_count
    )
    add(
        "touchdown_timestamp_cadence_available",
        timestamp_cadence_available,
        applies=moving,
    )
    add(
        "touchdown_median_cadence",
        timestamp_cadence_available
        and within(
            metrics.left_touchdown_median_cadence_hz,
            thresholds.minimum_contact_cadence_hz,
            thresholds.maximum_contact_cadence_hz,
        )
        and within(
            metrics.right_touchdown_median_cadence_hz,
            thresholds.minimum_contact_cadence_hz,
            thresholds.maximum_contact_cadence_hz,
        )
        and abs(
            metrics.left_touchdown_median_cadence_hz
            - metrics.right_touchdown_median_cadence_hz
        )
        <= thresholds.maximum_touchdown_median_cadence_difference_hz,
        applies=moving,
    )
    add(
        "touchdown_median_vs_joint_cadence",
        timestamp_cadence_available
        and cadence_available
        and abs(
            metrics.left_touchdown_median_cadence_hz
            - metrics.left_joint_cadence_hz
        )
        <= thresholds.maximum_touchdown_joint_cadence_difference_hz
        and abs(
            metrics.right_touchdown_median_cadence_hz
            - metrics.right_joint_cadence_hz
        )
        <= thresholds.maximum_touchdown_joint_cadence_difference_hz,
        applies=moving,
    )
    add(
        "touchdown_interval_regularity",
        timestamp_cadence_available
        and metrics.left_touchdown_interval_cv
        <= thresholds.maximum_touchdown_interval_cv
        and metrics.right_touchdown_interval_cv
        <= thresholds.maximum_touchdown_interval_cv
        and metrics.left_maximum_same_foot_touchdown_gap_s
        <= thresholds.maximum_same_foot_touchdown_gap_s
        and metrics.right_maximum_same_foot_touchdown_gap_s
        <= thresholds.maximum_same_foot_touchdown_gap_s,
        applies=moving,
    )
    add(
        "six_second_touchdown_minimum",
        metrics.left_step_count
        >= thresholds.minimum_touchdowns_per_foot_for_six_seconds
        and metrics.right_step_count
        >= thresholds.minimum_touchdowns_per_foot_for_six_seconds,
        applies=moving and metrics.duration_s >= 6.0 - 1.0e-9,
    )

    add(
        "steady_linear_tracking",
        metrics.steady_linear_tracking_ratio is not None
        and thresholds.minimum_steady_tracking_ratio
        <= metrics.steady_linear_tracking_ratio
        <= thresholds.maximum_steady_tracking_ratio,
        applies=linear_active,
    )
    add(
        "steady_yaw_tracking",
        metrics.steady_yaw_tracking_ratio is not None
        and thresholds.minimum_steady_tracking_ratio
        <= metrics.steady_yaw_tracking_ratio
        <= thresholds.maximum_steady_tracking_ratio,
        applies=yaw_active,
    )
    pure_cross_pass = (
        metrics.steady_cross_drift_mps is not None
        and metrics.steady_cross_drift_fraction is not None
        and metrics.steady_cross_drift_mps
        <= thresholds.maximum_pure_cross_drift_mps
        and metrics.steady_cross_drift_fraction
        <= thresholds.maximum_pure_cross_drift_fraction
    )
    compound_cross_pass = (
        metrics.steady_cross_drift_mps is not None
        and metrics.steady_cross_drift_fraction is not None
        and metrics.steady_cross_drift_mps
        <= thresholds.maximum_compound_cross_drift_mps
        and metrics.steady_cross_drift_fraction
        <= thresholds.maximum_compound_cross_drift_fraction
    )
    add("pure_cross_drift", pure_cross_pass, applies=linear_active and not yaw_active)
    add("compound_cross_drift", compound_cross_pass, applies=compound)
    add(
        "uncommanded_yaw_rate",
        metrics.uncommanded_yaw_rate_radps is not None
        and metrics.uncommanded_yaw_rate_radps
        <= thresholds.maximum_uncommanded_yaw_rate_radps,
        applies=not yaw_active and linear_active,
    )
    add(
        "uncommanded_heading_drift",
        metrics.uncommanded_heading_drift_rad is not None
        and metrics.uncommanded_heading_drift_rad
        <= thresholds.maximum_uncommanded_heading_drift_rad,
        applies=not yaw_active and linear_active,
    )
    add(
        "yaw_only_planar_drift",
        metrics.yaw_only_planar_drift_mps is not None
        and metrics.yaw_only_planar_drift_mps
        <= thresholds.maximum_yaw_only_planar_drift_mps,
        applies=yaw_only,
    )
    add(
        "yaw_only_net_drift",
        metrics.yaw_only_net_drift_m is not None
        and metrics.yaw_only_net_drift_m
        <= thresholds.maximum_yaw_only_net_drift_m,
        applies=yaw_only,
    )
    add(
        "pure_se2_endpoint_primary",
        metrics.pure_endpoint_primary_ratio is not None
        and thresholds.minimum_pure_endpoint_primary_ratio
        <= metrics.pure_endpoint_primary_ratio
        <= thresholds.maximum_pure_endpoint_primary_ratio,
        applies=linear_active and not yaw_active,
    )
    add(
        "pure_se2_endpoint_cross",
        metrics.pure_endpoint_cross_error_m is not None
        and metrics.pure_endpoint_cross_error_fraction is not None
        and metrics.pure_endpoint_cross_error_m
        <= thresholds.maximum_pure_endpoint_cross_error_m
        and metrics.pure_endpoint_cross_error_fraction
        <= thresholds.maximum_pure_endpoint_cross_error_fraction,
        applies=linear_active and not yaw_active,
    )
    add(
        "pure_se2_cross_path",
        metrics.pure_cross_path_p95_error_m is not None
        and metrics.pure_cross_path_maximum_error_m is not None
        and metrics.pure_cross_path_p95_error_m
        <= thresholds.maximum_pure_cross_path_p95_error_m
        and metrics.pure_cross_path_maximum_error_m
        <= thresholds.maximum_pure_cross_path_error_m,
        applies=linear_active and not yaw_active,
    )
    add(
        "pure_se2_heading_path",
        metrics.trajectory_heading_maximum_error_rad is not None
        and metrics.trajectory_heading_maximum_error_rad
        <= thresholds.maximum_path_heading_error_rad,
        applies=linear_active and not yaw_active,
    )
    add(
        "translation_cumulative_backtracking",
        metrics.cumulative_backtracking_m is not None
        and metrics.cumulative_backtracking_m
        <= thresholds.maximum_cumulative_backtracking_m,
        applies=linear_active,
    )
    add(
        "compound_se2_endpoint_position",
        metrics.compound_endpoint_position_error_m is not None
        and metrics.compound_endpoint_position_error_fraction is not None
        and metrics.compound_endpoint_position_error_m
        <= thresholds.maximum_compound_endpoint_position_error_m
        and metrics.compound_endpoint_position_error_fraction
        <= thresholds.maximum_compound_endpoint_position_error_fraction,
        applies=compound,
    )
    add(
        "compound_se2_path",
        metrics.compound_path_p95_error_m is not None
        and metrics.compound_path_maximum_error_m is not None
        and metrics.compound_path_p95_error_m
        <= thresholds.maximum_compound_path_p95_error_m
        and metrics.compound_path_maximum_error_m
        <= thresholds.maximum_compound_path_error_m,
        applies=compound,
    )
    add(
        "compound_se2_heading",
        metrics.trajectory_heading_endpoint_error_rad is not None
        and metrics.trajectory_heading_maximum_error_rad is not None
        and metrics.trajectory_heading_endpoint_error_rad
        <= thresholds.maximum_path_heading_error_rad
        and metrics.trajectory_heading_maximum_error_rad
        <= thresholds.maximum_path_heading_error_rad,
        applies=compound,
    )
    add(
        "yaw_only_se2_radius",
        metrics.yaw_only_path_radius_p95_m is not None
        and metrics.yaw_only_path_radius_maximum_m is not None
        and metrics.yaw_only_path_radius_p95_m
        <= thresholds.maximum_yaw_only_path_radius_p95_m
        and metrics.yaw_only_path_radius_maximum_m
        <= thresholds.maximum_yaw_only_path_radius_m,
        applies=yaw_only,
    )
    add(
        "yaw_only_se2_heading",
        metrics.trajectory_heading_endpoint_error_rad is not None
        and metrics.trajectory_heading_maximum_error_rad is not None
        and metrics.trajectory_heading_endpoint_error_rad
        <= thresholds.maximum_path_heading_error_rad
        and metrics.trajectory_heading_maximum_error_rad
        <= thresholds.maximum_path_heading_error_rad,
        applies=yaw_only,
    )
    add(
        "stand_se2_radius",
        metrics.stand_path_radius_maximum_m is not None
        and metrics.stand_path_radius_maximum_m
        <= thresholds.maximum_stand_path_radius_m,
        applies=stand,
    )
    add(
        "stand_se2_heading",
        metrics.stand_heading_excursion_rad is not None
        and metrics.stand_heading_excursion_rad
        <= thresholds.maximum_stand_heading_excursion_rad,
        applies=stand,
    )

    failures = tuple(name for name, passed in checks.items() if not passed)
    return GaitQualityAcceptance(
        passed=not failures,
        checks=checks,
        applicable=applicable,
        failures=failures,
    )


def gait_quality_metrics_from_mapping(
    payload: Mapping[str, object],
) -> GaitQualityMetrics:
    """Reconstruct a complete H4 metric payload, rejecting omissions/extras.

    This is intentionally strict so routed acceptance cannot silently interpret a
    historical or hand-edited partial mapping as current formal evidence.
    """

    expected = {definition.name for definition in fields(GaitQualityMetrics)}
    actual = set(payload) - {"measurement_complete"}
    if actual != expected:
        raise ValueError(
            "gait-quality metric key mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if payload.get("measurement_complete") is not True:
        raise ValueError("gait-quality measurement is incomplete")
    values = {name: payload[name] for name in expected}
    return GaitQualityMetrics(**values)  # type: ignore[arg-type]


def rederive_gait_quality_acceptance(
    payload: Mapping[str, object],
    thresholds: GaitQualityThresholds = FROZEN_GAIT_QUALITY_THRESHOLDS,
) -> GaitQualityAcceptance:
    """Recompute checks/failures/passed solely from the serialized H4 metrics."""

    return gait_quality_acceptance(
        gait_quality_metrics_from_mapping(payload), thresholds
    )


def accumulate_gait_quality(
    samples: Iterable[GaitQualitySubstep],
    *,
    joint_names: Sequence[str],
    left_joint_indices: Sequence[int] | None = None,
    right_joint_indices: Sequence[int] | None = None,
    left_cadence_joint_index: int | None = None,
    right_cadence_joint_index: int | None = None,
    steady_start_s: float = FROZEN_GAIT_QUALITY_THRESHOLDS.steady_start_s,
    contact_debounce_s: float = FROZEN_GAIT_QUALITY_THRESHOLDS.contact_debounce_s,
    contact_force_on_fraction_body_weight: float = (
        FROZEN_GAIT_QUALITY_THRESHOLDS.contact_force_on_fraction_body_weight
    ),
    contact_force_off_fraction_body_weight: float = (
        FROZEN_GAIT_QUALITY_THRESHOLDS.contact_force_off_fraction_body_weight
    ),
) -> GaitQualityMetrics:
    """Functional convenience wrapper over :class:`GaitQualityAccumulator`."""

    accumulator = GaitQualityAccumulator(
        joint_names=joint_names,
        left_joint_indices=left_joint_indices,
        right_joint_indices=right_joint_indices,
        left_cadence_joint_index=left_cadence_joint_index,
        right_cadence_joint_index=right_cadence_joint_index,
        steady_start_s=steady_start_s,
        contact_debounce_s=contact_debounce_s,
        contact_force_on_fraction_body_weight=(
            contact_force_on_fraction_body_weight
        ),
        contact_force_off_fraction_body_weight=(
            contact_force_off_fraction_body_weight
        ),
    )
    for sample in samples:
        accumulator.update(sample)
    return accumulator.finalize()


__all__ = [
    "FROZEN_GAIT_QUALITY_THRESHOLDS",
    "GaitQualityAcceptance",
    "GaitQualityAccumulator",
    "GaitQualityMetrics",
    "GaitQualitySubstep",
    "GaitQualityThresholds",
    "accumulate_gait_quality",
    "gait_quality_acceptance",
    "gait_quality_metrics_from_mapping",
    "rederive_gait_quality_acceptance",
]
