from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from safe_gait_experts.gait_quality import (
    FROZEN_GAIT_QUALITY_THRESHOLDS,
    GaitQualityAccumulator,
    GaitQualitySubstep,
    GaitQualityThresholds,
    accumulate_gait_quality,
    gait_quality_acceptance,
    gait_quality_metrics_from_mapping,
    rederive_gait_quality_acceptance,
)


JOINT_NAMES = (
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


def _synthetic_samples(
    *,
    command: tuple[float, float, float] = (0.10, 0.0, 0.0),
    duration_s: float = 4.0,
    dt_s: float = 0.01,
    startup_ramp_s: float = 0.70,
    linear_tracking_scale: float = 1.0,
    yaw_tracking_scale: float = 1.0,
    cross_drift_mps: float = 0.0,
    uncommanded_yaw_rate_radps: float = 0.0,
    yaw_only_drift_mps: float = 0.0,
    effective_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    contact_cadence_hz: float = 1.50,
    contact_mode: str = "balanced",
    contact_start_delay_s: float = 0.0,
    stance_slip_mps: float = 0.0,
    joint_range_rad: float = 0.24,
    left_joint_cadence_hz: float = 1.50,
    right_joint_cadence_hz: float = 1.50,
) -> list[GaitQualitySubstep]:
    requested = np.asarray(command, dtype=np.float64)
    effective = requested + np.asarray(effective_offset, dtype=np.float64)
    linear_speed = float(np.linalg.norm(requested[:2]))
    direction = (
        np.asarray((1.0, 0.0))
        if linear_speed == 0.0
        else requested[:2] / linear_speed
    )
    orthogonal = np.asarray((-direction[1], direction[0]))
    times = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    trunk = np.zeros(3, dtype=np.float64)
    yaw = 0.0
    foot_points = np.asarray(((0.0, 0.06, 0.0), (0.0, -0.06, 0.0)))
    previous_contacts: tuple[bool, bool] | None = None
    samples: list[GaitQualitySubstep] = []

    for index, time_s in enumerate(times):
        gain = min(1.0, time_s / startup_ramp_s) if startup_ramp_s > 0.0 else 1.0
        local_xy = requested[:2] * linear_tracking_scale * gain
        if linear_speed > 0.0:
            local_xy = local_xy + orthogonal * cross_drift_mps
        elif abs(requested[2]) > 0.0:
            local_xy = np.asarray((yaw_only_drift_mps, 0.0))
        yaw_rate = (
            requested[2] * yaw_tracking_scale * gain
            if abs(requested[2]) > 0.0
            else uncommanded_yaw_rate_radps
        )

        if time_s < contact_start_delay_s:
            contacts = (True, True)
        elif contact_mode == "balanced":
            phase = ((time_s - contact_start_delay_s) * contact_cadence_hz) % 1.0
            contacts = (phase < 0.75, ((phase + 0.50) % 1.0) < 0.75)
        elif contact_mode == "left_only":
            contacts = (True, False)
        elif contact_mode == "long_stance":
            phase = ((time_s - contact_start_delay_s) / 3.0) % 1.0
            contacts = (phase < 0.90, ((phase + 0.50) % 1.0) < 0.90)
        else:
            raise ValueError(f"unsupported contact_mode: {contact_mode}")

        if previous_contacts is not None:
            for foot in (0, 1):
                if contacts[foot] and previous_contacts[foot]:
                    foot_points[foot, 0] += stance_slip_mps * dt_s
                elif contacts[foot] and not previous_contacts[foot]:
                    foot_points[foot, 0] += 0.06

        if index:
            c, s = np.cos(yaw), np.sin(yaw)
            world_velocity = np.asarray(
                (c * local_xy[0] - s * local_xy[1], s * local_xy[0] + c * local_xy[1])
            )
            trunk[:2] += world_velocity * dt_s
            yaw += yaw_rate * dt_s

        left_phase = 2.0 * np.pi * left_joint_cadence_hz * time_s
        right_phase = 2.0 * np.pi * right_joint_cadence_hz * time_s + np.pi
        amplitude = 0.5 * joint_range_rad
        joints = np.asarray(
            [
                amplitude * np.sin(left_phase + 0.17 * joint_index)
                for joint_index in range(5)
            ]
            + [
                amplitude * np.sin(right_phase + 0.17 * joint_index)
                for joint_index in range(5)
            ]
        )
        samples.append(
            GaitQualitySubstep(
                time_s=float(time_s),
                requested_command=requested.copy(),
                effective_command=effective.copy(),
                local_velocity_xyz_mps=(local_xy[0], local_xy[1], 0.0),
                local_yaw_rate_radps=float(yaw_rate),
                trunk_position_world_m=trunk.copy(),
                feet_contacts=contacts,
                foot_contact_points_world_m=foot_points.copy(),
                leg_joint_positions_rad=joints,
                feet_normal_force_fraction_body_weight=tuple(
                    (1.0 / sum(contacts)) if contact and sum(contacts) else 0.0
                    for contact in contacts
                ),
                foot_contact_tangential_speeds_mps=tuple(
                    stance_slip_mps if contact else 0.0 for contact in contacts
                ),
                trunk_yaw_world_rad=float(yaw),
                trunk_pose_measurement_source=(
                    "synthetic_world_trunk_position_and_yaw"
                ),
            )
        )
        previous_contacts = contacts
    return samples


def _run(**kwargs):
    metrics = accumulate_gait_quality(
        _synthetic_samples(**kwargs), joint_names=JOINT_NAMES
    )
    return metrics, gait_quality_acceptance(metrics)


def test_frozen_threshold_contract_matches_v2_quality_audit() -> None:
    thresholds = FROZEN_GAIT_QUALITY_THRESHOLDS
    assert thresholds == GaitQualityThresholds()
    assert thresholds.maximum_t30_s == 0.40
    assert thresholds.maximum_t75_s == 1.00
    assert thresholds.maximum_first_single_support_s == 0.80
    assert thresholds.contact_debounce_s == 0.020
    assert thresholds.contact_force_on_fraction_body_weight == 0.010
    assert thresholds.contact_force_off_fraction_body_weight == 0.005
    assert (
        thresholds.minimum_single_support_rate,
        thresholds.maximum_single_support_rate,
    ) == (0.25, 0.60)
    assert thresholds.maximum_flight_rate == 0.01
    assert thresholds.minimum_steps_per_foot == 2
    assert thresholds.maximum_step_count_imbalance == 1
    assert thresholds.maximum_contact_duty_imbalance == 0.10
    assert thresholds.maximum_stance_slip_rms_mps == 0.015
    assert thresholds.maximum_stance_slip_p95_mps == 0.030
    assert thresholds.maximum_per_stance_cumulative_slip_m == 0.020
    assert (
        thresholds.minimum_steady_tracking_ratio,
        thresholds.maximum_steady_tracking_ratio,
    ) == (0.75, 1.25)
    assert (
        thresholds.maximum_pure_cross_drift_mps,
        thresholds.maximum_pure_cross_drift_fraction,
    ) == (0.012, 0.20)
    assert (
        thresholds.maximum_compound_cross_drift_mps,
        thresholds.maximum_compound_cross_drift_fraction,
    ) == (0.015, 0.25)
    assert thresholds.maximum_uncommanded_yaw_rate_radps == 0.050
    assert thresholds.maximum_yaw_only_planar_drift_mps == 0.012
    assert thresholds.maximum_yaw_only_net_drift_m == 0.050
    assert (thresholds.startup_rolling_window_s, thresholds.startup_sustain_s) == (
        0.20,
        0.30,
    )
    assert (
        thresholds.maximum_startup_wrong_way_displacement_m,
        thresholds.maximum_startup_continuous_wrong_way_s,
        thresholds.maximum_startup_wrong_way_occupancy,
    ) == (0.010, 0.20, 0.10)
    assert (
        thresholds.maximum_debounce_touchdown_count_span,
        thresholds.maximum_debounce_single_support_rate_span,
        thresholds.maximum_debounce_contact_rate_span,
        thresholds.maximum_debounce_flight_rate_span,
    ) == (1, 0.05, 0.05, 0.005)
    assert (
        thresholds.minimum_contact_velocity_coverage,
        thresholds.maximum_contact_velocity_coverage,
    ) == (0.95, 1.01)
    assert (
        thresholds.minimum_steady_total_normal_force_fraction_body_weight,
        thresholds.maximum_steady_total_normal_force_fraction_body_weight,
        thresholds.maximum_total_normal_force_p99_fraction_body_weight,
    ) == (0.80, 1.20, 3.0)
    assert (
        thresholds.maximum_pure_endpoint_cross_error_m,
        thresholds.maximum_pure_endpoint_cross_error_fraction,
        thresholds.maximum_pure_cross_path_p95_error_m,
        thresholds.maximum_pure_cross_path_error_m,
    ) == (0.030, 0.10, 0.020, 0.040)
    assert (
        thresholds.maximum_compound_endpoint_position_error_m,
        thresholds.maximum_compound_endpoint_position_error_fraction,
        thresholds.maximum_compound_path_p95_error_m,
        thresholds.maximum_compound_path_error_m,
    ) == (0.050, 0.20, 0.035, 0.050)


@pytest.mark.parametrize(
    "command",
    [
        (0.10, 0.0, 0.0),
        (0.08, 0.04, 0.20),
        (0.0, 0.0, -0.30),
    ],
)
def test_strict_synthetic_pure_compound_and_yaw_only_cases_pass(command) -> None:
    kwargs = {"command": command}
    if command[:2] == (0.0, 0.0):
        kwargs["yaw_only_drift_mps"] = 0.005
    metrics, result = _run(**kwargs)
    assert result.passed, result.failures
    assert metrics.t30_s is not None and metrics.t30_s <= 0.40
    assert metrics.t75_s is not None and metrics.t75_s <= 1.00
    assert metrics.first_single_support_s is not None
    assert metrics.left_step_count >= 2
    assert metrics.right_step_count >= 2
    assert metrics.sample_count == 401
    assert metrics.stance_slip_rms_mps == pytest.approx(0.0)
    assert metrics.left_joint_cadence_hz == pytest.approx(
        metrics.right_joint_cadence_hz
    )


def test_contact_debounce_rejects_single_substep_contact_chatter() -> None:
    accumulator = GaitQualityAccumulator(
        joint_names=JOINT_NAMES,
        contact_debounce_s=0.020,
    )
    for index, time_s in enumerate(np.arange(0.0, 2.01, 0.01)):
        contacts = (index != 50, index != 100)
        accumulator.update(
            GaitQualitySubstep(
                time_s=float(time_s),
                requested_command=(0.10, 0.0, 0.0),
                effective_command=(0.10, 0.0, 0.0),
                local_velocity_xyz_mps=(0.10, 0.0, 0.0),
                local_yaw_rate_radps=0.0,
                trunk_position_world_m=(0.10 * time_s, 0.0, 0.20),
                feet_contacts=contacts,
                foot_contact_points_world_m=(
                    (0.0, 0.06, 0.0),
                    (0.0, -0.06, 0.0),
                ),
                leg_joint_positions_rad=np.zeros(len(JOINT_NAMES)),
            )
        )
    metrics = accumulator.finalize()
    assert metrics.left_step_count == 0
    assert metrics.right_step_count == 0
    assert metrics.left_contact_rate == 1.0
    assert metrics.right_contact_rate == 1.0


def test_open_stance_cumulative_slip_continues_across_segment_boundary() -> None:
    first = GaitQualityAccumulator(joint_names=JOINT_NAMES)
    for time_s in np.arange(0.0, 2.01, 0.01):
        first.update(
            GaitQualitySubstep(
                time_s=float(time_s),
                requested_command=(0.10, 0.0, 0.0),
                effective_command=(0.10, 0.0, 0.0),
                local_velocity_xyz_mps=(0.10, 0.0, 0.0),
                local_yaw_rate_radps=0.0,
                trunk_position_world_m=(0.10 * time_s, 0.0, 0.20),
                feet_contacts=(True, False),
                foot_contact_points_world_m=((0.0, 0.06, 0.0), (0.0, -0.06, 0.0)),
                leg_joint_positions_rad=np.zeros(len(JOINT_NAMES)),
                feet_normal_force_fraction_body_weight=(0.20, 0.0),
                foot_contact_tangential_speeds_mps=(0.01, 0.0),
            )
        )
    continuity = first.export_contact_continuity_state()

    second = GaitQualityAccumulator(joint_names=JOINT_NAMES)
    second.restore_contact_continuity_state(continuity)
    for index, time_s in enumerate(np.arange(0.0, 2.01, 0.01)):
        left_contact = index == 0
        second.update(
            GaitQualitySubstep(
                time_s=float(time_s),
                requested_command=(0.10, 0.0, 0.0),
                effective_command=(0.10, 0.0, 0.0),
                local_velocity_xyz_mps=(0.10, 0.0, 0.0),
                local_yaw_rate_radps=0.0,
                trunk_position_world_m=(0.10 * time_s, 0.0, 0.20),
                feet_contacts=(left_contact, True),
                foot_contact_points_world_m=((0.0, 0.06, 0.0), (0.0, -0.06, 0.0)),
                leg_joint_positions_rad=np.zeros(len(JOINT_NAMES)),
                feet_normal_force_fraction_body_weight=(
                    0.20 if left_contact else 0.0,
                    0.20,
                ),
                foot_contact_tangential_speeds_mps=(0.01, 0.0),
            )
        )
    metrics = second.finalize()
    assert metrics.left_maximum_per_stance_cumulative_slip_m == pytest.approx(
        0.020, abs=0.0002
    )
    assert metrics.stance_slip_measurement_source == (
        "force_weighted_contact_point_jacobian"
    )


def test_startup_t30_t75_and_first_support_are_independent_failures() -> None:
    metrics, result = _run(startup_ramp_s=2.0)
    assert metrics.t30_s == pytest.approx(0.70, abs=0.011)
    assert metrics.t75_s == pytest.approx(1.60, abs=0.011)
    assert not result.checks["startup_t30"]
    assert not result.checks["startup_t75"]

    delayed_metrics, delayed = _run(contact_start_delay_s=1.0)
    assert delayed_metrics.first_single_support_s > 0.80
    assert not delayed.checks["first_single_support"]


def test_left_right_contact_steps_duty_and_alternation_are_measured() -> None:
    metrics, result = _run(contact_mode="left_only")
    assert metrics.left_contact_rate == pytest.approx(1.0)
    assert metrics.right_contact_rate == pytest.approx(0.0)
    assert metrics.contact_duty_imbalance == pytest.approx(1.0)
    assert metrics.left_step_count == 0
    assert metrics.right_step_count == 0
    assert not result.checks["single_support_rate"]
    assert not result.checks["left_right_step_count"]
    assert not result.checks["left_right_contact_duty"]
    assert not result.checks["alternating_touchdowns"]


def test_world_tangential_stance_slip_is_accumulated_per_stance() -> None:
    metrics, result = _run(stance_slip_mps=0.040)
    assert metrics.left_stance_slip_rms_mps == pytest.approx(0.040)
    assert metrics.right_stance_slip_rms_mps == pytest.approx(0.040)
    assert metrics.left_stance_slip_p95_mps == pytest.approx(0.040)
    assert metrics.right_stance_slip_p95_mps == pytest.approx(0.040)
    assert metrics.stance_slip_p95_mps == pytest.approx(0.040)
    assert metrics.maximum_per_stance_cumulative_slip_m is not None
    assert not result.checks["stance_slip_rms"]
    assert not result.checks["stance_slip_p95"]

    cumulative_metrics, cumulative_result = _run(
        duration_s=6.0,
        contact_mode="long_stance",
        stance_slip_mps=0.012,
    )
    assert cumulative_metrics.stance_slip_rms_mps == pytest.approx(0.012)
    assert cumulative_metrics.maximum_per_stance_cumulative_slip_m > 0.020
    assert not cumulative_result.checks["per_stance_cumulative_slip"]

    baseline_metrics, _ = _run()
    for field, check, bad_value in (
        ("stance_slip_rms_mps", "stance_slip_rms", 0.0151),
        ("stance_slip_p95_mps", "stance_slip_p95", 0.0301),
        (
            "maximum_per_stance_cumulative_slip_m",
            "per_stance_cumulative_slip",
            0.0201,
        ),
    ):
        modified = replace(baseline_metrics, **{field: bad_value})
        assert not gait_quality_acceptance(modified).checks[check]


def test_joint_periodic_range_and_cadence_have_strict_gates() -> None:
    low_range, low_range_result = _run(joint_range_rad=0.070)
    assert max(low_range.joint_ranges_rad) < 0.080
    assert not low_range_result.checks["joint_periodic_range"]
    assert not low_range_result.checks["cadence_joint_range"]

    low_cadence, low_cadence_result = _run(
        duration_s=6.0,
        left_joint_cadence_hz=0.40,
        right_joint_cadence_hz=0.40,
    )
    assert low_cadence.left_joint_cadence_hz < 0.70
    assert not low_cadence_result.checks["joint_cadence"]

    baseline, _ = _run()
    mismatched = replace(
        baseline, left_joint_cadence_hz=1.0, right_joint_cadence_hz=1.30
    )
    assert not gait_quality_acceptance(mismatched).checks[
        "left_right_joint_cadence"
    ]


def test_requested_effective_command_mismatch_is_not_hidden() -> None:
    metrics, result = _run(effective_offset=(0.0, 0.006, 0.0))
    assert metrics.requested_effective_axis_error[1] == pytest.approx(0.006)
    assert not result.checks["requested_effective_command"]


def test_zero_mean_effective_command_oscillation_is_caught_by_rms_error() -> None:
    samples = _synthetic_samples()
    modified = []
    for index, sample in enumerate(samples):
        offset = 0.006 if index % 2 else -0.006
        effective = np.asarray(sample.effective_command, dtype=np.float64).copy()
        effective[1] += offset
        modified.append(replace(sample, effective_command=effective))
    metrics = accumulate_gait_quality(modified, joint_names=JOINT_NAMES)
    result = gait_quality_acceptance(metrics)
    assert metrics.requested_effective_axis_error[1] < 0.001
    assert metrics.requested_effective_axis_rms_error[1] == pytest.approx(0.006)
    assert not result.checks["requested_effective_command"]


@pytest.mark.parametrize("scale", [0.70, 1.30])
def test_steady_linear_tracking_has_lower_and_upper_bounds(scale: float) -> None:
    metrics, result = _run(linear_tracking_scale=scale)
    assert metrics.steady_linear_tracking_ratio == pytest.approx(scale)
    assert not result.checks["steady_linear_tracking"]


@pytest.mark.parametrize("scale", [0.70, 1.30])
def test_steady_yaw_tracking_has_lower_and_upper_bounds(scale: float) -> None:
    metrics, result = _run(command=(0.0, 0.0, 0.30), yaw_tracking_scale=scale)
    assert metrics.steady_yaw_tracking_ratio == pytest.approx(scale)
    assert not result.checks["steady_yaw_tracking"]


def test_pure_and_compound_cross_drift_have_distinct_gates() -> None:
    pure_metrics, pure = _run(cross_drift_mps=0.013)
    assert pure_metrics.steady_cross_drift_mps == pytest.approx(0.013)
    assert not pure.checks["pure_cross_drift"]
    assert not pure.applicable["compound_cross_drift"]

    compound_metrics, compound = _run(
        command=(0.08, 0.04, 0.20), cross_drift_mps=0.016
    )
    assert compound_metrics.steady_cross_drift_mps == pytest.approx(0.016)
    assert not compound.checks["compound_cross_drift"]
    assert not compound.applicable["pure_cross_drift"]


def test_uncommanded_yaw_rate_and_integrated_heading_are_both_gated() -> None:
    metrics, result = _run(uncommanded_yaw_rate_radps=0.060)
    assert metrics.uncommanded_yaw_rate_radps == pytest.approx(0.060)
    assert metrics.uncommanded_heading_drift_rad == pytest.approx(0.240)
    assert not result.checks["uncommanded_yaw_rate"]
    assert not result.checks["uncommanded_heading_drift"]


def test_yaw_only_drift_rate_and_net_displacement_are_both_gated() -> None:
    metrics, result = _run(
        command=(0.0, 0.0, 0.30), yaw_only_drift_mps=0.020
    )
    assert metrics.yaw_only_planar_drift_mps == pytest.approx(0.020)
    assert metrics.yaw_only_net_drift_m is not None
    assert metrics.yaw_only_net_drift_m > 0.050
    assert not result.checks["yaw_only_planar_drift"]
    assert not result.checks["yaw_only_net_drift"]


def test_accumulator_rejects_incomplete_or_non_substep_ordered_input() -> None:
    accumulator = GaitQualityAccumulator(joint_names=JOINT_NAMES)
    with pytest.raises(ValueError, match="at least two"):
        accumulator.finalize()
    first = _synthetic_samples(duration_s=0.01)[0]
    accumulator.update(first)
    with pytest.raises(ValueError, match="strictly increasing"):
        accumulator.update(first)
    bad = replace(first, leg_joint_positions_rad=np.zeros(len(JOINT_NAMES) - 1))
    second_accumulator = GaitQualityAccumulator(joint_names=JOINT_NAMES)
    with pytest.raises(ValueError, match="leg_joint_positions_rad"):
        second_accumulator.update(bad)


def test_sustained_rolling_startup_rejects_an_early_velocity_spike() -> None:
    samples = _synthetic_samples(startup_ramp_s=2.0)
    spiked = [
        replace(sample, local_velocity_xyz_mps=(0.10, 0.0, 0.0))
        if 0.05 <= sample.time_s <= 0.10
        else sample
        for sample in samples
    ]
    metrics = accumulate_gait_quality(spiked, joint_names=JOINT_NAMES)
    assert metrics.t30_s == pytest.approx(0.71, abs=0.011)
    assert metrics.t75_s == pytest.approx(1.61, abs=0.011)
    assert not gait_quality_acceptance(metrics).checks["startup_axis_sustained_t30"]


def test_startup_wrong_way_is_measured_as_excursion_duration_and_occupancy() -> None:
    modified = []
    for sample in _synthetic_samples():
        if sample.time_s <= 0.20:
            trunk = np.asarray(sample.trunk_position_world_m).copy()
            trunk[0] = -0.10 * sample.time_s
            modified.append(
                replace(
                    sample,
                    local_velocity_xyz_mps=(-0.10, 0.0, 0.0),
                    trunk_position_world_m=trunk,
                )
            )
        else:
            modified.append(sample)
    metrics = accumulate_gait_quality(modified, joint_names=JOINT_NAMES)
    result = gait_quality_acceptance(metrics)
    assert metrics.startup_axis_wrong_way_displacement[0] >= 0.020
    assert metrics.startup_axis_maximum_continuous_wrong_way_s[0] >= 0.20
    assert metrics.startup_axis_wrong_way_occupancy[0] >= 0.20
    assert not result.checks["startup_wrong_way_displacement"]
    assert not result.checks["startup_wrong_way_occupancy"]


def test_se2_path_gates_cover_pure_compound_yaw_only_and_stand() -> None:
    pure, _ = _run(duration_s=6.0)
    pure_bad = replace(
        pure,
        pure_endpoint_primary_ratio=0.74,
        pure_endpoint_cross_error_m=0.031,
        pure_cross_path_p95_error_m=0.021,
        cumulative_backtracking_m=0.011,
    )
    pure_result = gait_quality_acceptance(pure_bad)
    assert not pure_result.checks["pure_se2_endpoint_primary"]
    assert not pure_result.checks["pure_se2_endpoint_cross"]
    assert not pure_result.checks["pure_se2_cross_path"]
    assert not pure_result.checks["translation_cumulative_backtracking"]

    compound, _ = _run(command=(0.08, 0.04, 0.20), duration_s=6.0)
    compound_result = gait_quality_acceptance(
        replace(compound, compound_endpoint_position_error_m=0.051)
    )
    assert not compound_result.checks["compound_se2_endpoint_position"]

    yaw_only, _ = _run(command=(0.0, 0.0, 0.30), duration_s=6.0)
    yaw_result = gait_quality_acceptance(
        replace(yaw_only, yaw_only_path_radius_p95_m=0.031)
    )
    assert not yaw_result.checks["yaw_only_se2_radius"]

    stand, stand_result = _run(command=(0.0, 0.0, 0.0), duration_s=6.0)
    assert stand_result.passed, stand_result.failures
    assert not gait_quality_acceptance(
        replace(stand, stand_path_radius_maximum_m=0.021)
    ).checks["stand_se2_radius"]


def test_debounce_robustness_requires_every_window_and_small_spans() -> None:
    metrics, result = _run(duration_s=6.0)
    assert result.checks["debounce_all_windows_quality"]
    assert result.checks["debounce_touchdown_count_robustness"]
    assert result.checks["debounce_support_rate_robustness"]

    sensitivity = copy.deepcopy(dict(metrics.contact_debounce_sensitivity))
    sensitivity["40ms"]["left_touchdowns"] += 2
    sensitivity["40ms"]["single_support_rate"] = 0.61
    modified = replace(metrics, contact_debounce_sensitivity=sensitivity)
    modified_result = gait_quality_acceptance(modified)
    assert not modified_result.checks["debounce_all_windows_quality"]
    assert not modified_result.checks["debounce_touchdown_count_robustness"]
    assert not modified_result.checks["debounce_support_rate_robustness"]


def test_force_coverage_impulse_p99_and_touchdown_regularity_are_gated() -> None:
    metrics, result = _run(duration_s=6.0)
    assert result.checks["contact_velocity_coverage"]
    assert result.checks["steady_total_normal_force"]
    assert result.checks["left_right_normal_impulse_share"]
    assert result.checks["total_normal_force_p99"]
    assert result.checks["touchdown_interval_regularity"]
    assert result.checks["contact_count_vs_joint_cadence"]

    modified = replace(
        metrics,
        left_contact_velocity_coverage=0.94,
        steady_mean_total_normal_force_fraction_body_weight=1.21,
        left_normal_impulse_share=0.34,
        right_normal_impulse_share=0.66,
        total_normal_force_p99_fraction_body_weight=3.01,
        left_touchdown_interval_cv=0.31,
    )
    modified_result = gait_quality_acceptance(modified)
    for check in (
        "contact_velocity_coverage",
        "steady_total_normal_force",
        "left_right_normal_impulse_share",
        "total_normal_force_p99",
        "touchdown_interval_regularity",
    ):
        assert not modified_result.checks[check]


def test_complete_metric_mapping_rederives_checks_failures_and_passed() -> None:
    metrics, acceptance = _run(duration_s=6.0)
    payload = {"measurement_complete": True, **metrics.as_dict()}
    reconstructed = gait_quality_metrics_from_mapping(payload)
    assert reconstructed.sample_count == metrics.sample_count
    assert rederive_gait_quality_acceptance(payload).as_dict() == acceptance.as_dict()

    incomplete = dict(payload)
    incomplete.pop("pure_endpoint_primary_ratio")
    with pytest.raises(ValueError, match="metric key mismatch"):
        gait_quality_metrics_from_mapping(incomplete)
