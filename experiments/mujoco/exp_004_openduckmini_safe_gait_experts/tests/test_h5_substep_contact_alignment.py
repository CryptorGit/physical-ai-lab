from __future__ import annotations

import numpy as np
import pytest

from safe_gait_experts.gait_quality import _contact_sequence_summary
from safe_gait_experts.h5_substep_contact_alignment import (
    FROZEN_DEBOUNCE_WINDOWS_S,
    frozen_debounce_window_intervals,
    h5_all_substep_quality_losses,
    h5_all_substep_quality_update,
    h5_reverse_return_order_proof,
    h5_v3_t1_fixed_quality_replay,
    h5_v3_t1_fixed_quality_replay_manifest,
    h5_completed_stance_slip_summary,
    h5_multiwindow_debounce_summaries,
    h5_multiwindow_debounce_trace,
    h5_rederive_strict_20ms_slip_segment,
    initialize_h5_multiwindow_debounce,
    update_h5_multiwindow_debounce,
)


def _force_from_boolean(contacts: np.ndarray) -> np.ndarray:
    return np.where(contacts, 0.020, 0.0).astype(np.float64)


def _mixed_contact_pattern() -> tuple[np.ndarray, np.ndarray]:
    """Return a pattern with persistence, a short island, and simultaneity."""

    contacts = np.zeros((64, 2), dtype=bool)
    contacts[2:20, 0] = True
    contacts[7:12, 1] = True  # short enough to vary across debounce windows
    contacts[26:50, 1] = True
    contacts[36:64, 0] = True
    contacts[53:, :] = True  # simultaneous final touchdown
    return np.arange(len(contacts), dtype=np.float64) * 0.002, _force_from_boolean(contacts)


def test_frozen_windows_are_exact_2ms_multiples() -> None:
    assert frozen_debounce_window_intervals() == (5, 10, 15, 20)
    with pytest.raises(ValueError, match="exactly the frozen"):
        frozen_debounce_window_intervals((0.010, 0.020))
    with pytest.raises(ValueError, match="multiple"):
        frozen_debounce_window_intervals(dt_s=0.003)


def test_t1_fixed_quality_replay_manifest_is_pinned_and_well_formed() -> None:
    force, speed = h5_v3_t1_fixed_quality_replay(xp=np)
    manifest = h5_v3_t1_fixed_quality_replay_manifest()

    assert force.shape == speed.shape == (10, 2)
    assert force.dtype == speed.dtype == np.float32
    assert np.all(force == np.float32(0.5))
    assert np.all(speed == np.float32(0.01))
    assert manifest["force_shape"] == manifest["speed_shape"] == [10, 2]
    assert len(manifest["normalized_force_raw_bytes_sha256"]) == 64
    assert len(manifest["tangential_speed_raw_bytes_sha256"]) == 64


def test_all_four_windows_match_strict_evaluator_summary_exactly() -> None:
    times, force = _mixed_contact_pattern()
    actual = h5_multiwindow_debounce_summaries(times, force)
    raw = force >= 0.010
    for window_s in FROZEN_DEBOUNCE_WINDOWS_S:
        key = f"{int(round(window_s * 1000.0))}ms"
        assert actual[key] == _contact_sequence_summary(times, raw, window_s)


def test_schmitt_hysteresis_and_causal_commit_sample() -> None:
    force = np.asarray(
        (
            (0.0, 0.0),
            (0.011, 0.0),  # on threshold
            (0.007, 0.0),  # retains on through off threshold
            (0.004, 0.0),  # turns off
        ),
        dtype=np.float64,
    )
    trace = h5_multiwindow_debounce_trace(force)
    assert np.array_equal(trace.raw_contact[:, 0], np.asarray((False, True, True, False)))

    sustained = np.zeros((8, 2), dtype=np.float64)
    sustained[1:, 0] = 0.020
    contacts = h5_multiwindow_debounce_trace(sustained).qualified_contact[:, :, 0]
    # A change starts at index 1.  10 ms confirms at index 6; the longer
    # windows still have a right-censored pending state at this horizon.
    assert not bool(contacts[5, 0])
    assert bool(contacts[6, 0])
    assert not bool(contacts[-1, 1])
    assert not bool(contacts[-1, 2])
    assert not bool(contacts[-1, 3])


def test_continuation_across_control_boundary_and_terminal_censoring() -> None:
    times, force = _mixed_contact_pattern()
    full = h5_multiwindow_debounce_trace(force, times_s=times)
    state = initialize_h5_multiwindow_debounce(force[0])
    split = 23
    for index in range(1, split):
        state = update_h5_multiwindow_debounce(
            state, force[index], time_s=times[index]
        ).state
    for index in range(split, len(force)):
        state = update_h5_multiwindow_debounce(
            state, force[index], time_s=times[index]
        ).state
    assert np.array_equal(state.raw_contact, full.final_state.raw_contact)
    assert np.array_equal(state.qualified_contact, full.final_state.qualified_contact)
    assert np.array_equal(state.pending_active, full.final_state.pending_active)
    assert np.array_equal(state.pending_since_s, full.final_state.pending_since_s)

    # One opposite sample at the end is pending, not a fabricated touchdown.
    terminal_force = np.asarray(((0.0, 0.0), (0.020, 0.0)), dtype=np.float64)
    terminal = h5_multiwindow_debounce_trace(
        terminal_force, times_s=np.asarray((0.0, 0.002), dtype=np.float64)
    )
    assert not np.any(terminal.qualified_contact[-1])
    assert np.all(terminal.final_state.pending_active[:, 0])


def test_all_substep_losses_use_every_force_qualified_sample() -> None:
    force = np.asarray(
        (
            (0.50, 0.50),
            (0.50, 0.50),
            (2.00, 2.00),
            (0.50, 0.50),
        ),
        dtype=np.float64,
    )
    speed = np.asarray(
        (
            (0.010, 0.020),
            (0.020, 0.040),
            (0.030, 0.060),
            (0.040, 0.080),
        ),
        dtype=np.float64,
    )
    losses = h5_all_substep_quality_losses(force, speed)
    assert int(losses.force_qualified_sample_count) == 8
    assert np.isclose(
        losses.strict20ms_slip_rms_loss,
        np.mean(np.square(speed)) / 0.015**2,
    )
    expected_tail = np.mean(np.square(np.maximum(speed - 0.030, 0.0) / 0.030))
    assert np.isclose(losses.slip_tail_loss, expected_tail)
    expected_force_tail = np.mean(np.square(np.maximum(np.sum(force, axis=1) - 3.0, 0.0) / 3.0))
    assert np.isclose(losses.force_tail_loss, expected_force_tail)


def test_all_substep_update_carries_strict_20ms_debounce_between_control_ticks() -> None:
    # A left touchdown begins at 2 ms and only becomes 20-ms qualified after
    # the control boundary.  Resetting the debounce state per 20-ms tick would
    # incorrectly leave every sample unqualified.
    times = np.arange(20, dtype=np.float64) * 0.002
    force = np.zeros((20, 2), dtype=np.float64)
    force[1:, 0] = 0.020
    speed = np.full((20, 2), 0.040, dtype=np.float64)
    whole = h5_all_substep_quality_update(force, speed, times_s=times)
    first = h5_all_substep_quality_update(
        force[:10], speed[:10], times_s=times[:10]
    )
    second = h5_all_substep_quality_update(
        force[10:],
        speed[10:],
        initial_debounce=first.debounce,
        times_s=times[10:],
    )

    assert int(first.losses.force_qualified_sample_count) == 0
    assert int(second.losses.force_qualified_sample_count) > 0
    assert np.array_equal(
        whole.debounce.raw_contact, second.debounce.raw_contact
    )
    assert np.array_equal(
        whole.debounce.qualified_contact, second.debounce.qualified_contact
    )
    assert np.array_equal(
        whole.debounce.pending_active, second.debounce.pending_active
    )
    assert np.isclose(
        second.losses.strict20ms_slip_rms_loss,
        0.040**2 / 0.015**2,
    )


def test_stance_slip_counts_terminal_qualified_stance_and_uses_p95() -> None:
    times = np.arange(40, dtype=np.float64) * 0.002
    force = np.zeros((40, 2), dtype=np.float64)
    force[:, 0] = 0.020  # terminal qualified stance: scored by evaluator finalization
    force[1:25, 1] = 0.020  # long enough to confirm both entry and exit
    speed = np.zeros((40, 2), dtype=np.float64)
    speed[:, 0] = 0.010
    speed[:, 1] = 0.040
    summary = h5_completed_stance_slip_summary(times, force, speed)
    # The left terminal stance is included in the evaluator metric; the right
    # completed stance is also retained after its confirmed liftoff.
    assert int(summary["slip_sample_count"]) > 0
    assert summary["stance_slip_rms_mps"] is not None
    assert summary["stance_slip_p95_mps"] is not None
    assert summary["maximum_completed_stance_cumulative_slip_m"] is not None
    assert float(summary["maximum_completed_stance_cumulative_slip_m"]) <= 0.040


def test_terminal_stance_metric_does_not_close_continuity_early() -> None:
    first_times = np.arange(15, dtype=np.float64) * 0.002
    first_force = np.zeros((15, 2), dtype=np.float64)
    first_force[:, 0] = 0.020
    first_speed = np.zeros((15, 2), dtype=np.float64)
    first_speed[:, 0] = 0.010
    first_summary, state = h5_rederive_strict_20ms_slip_segment(
        first_times, first_force, first_speed
    )
    expected_distance = 14 * 0.002 * 0.010
    assert np.isclose(
        float(first_summary["maximum_completed_stance_cumulative_slip_m"]),
        expected_distance,
    )
    assert np.isclose(state.stance_cumulative_m[0], expected_distance)

    # A later confirmed liftoff closes the carried stance at the same distance;
    # finalization above did not fabricate a segment-boundary liftoff.
    second_times = np.arange(12, dtype=np.float64) * 0.002
    second_force = np.zeros((12, 2), dtype=np.float64)
    second_speed = np.zeros((12, 2), dtype=np.float64)
    second_summary, _state = h5_rederive_strict_20ms_slip_segment(
        second_times, second_force, second_speed, initial_state=state
    )
    assert np.isclose(
        float(second_summary["maximum_completed_stance_cumulative_slip_m"]),
        expected_distance,
    )


def test_stance_slip_continuity_preserves_cross_segment_distance() -> None:
    first_times = np.arange(20, dtype=np.float64) * 0.002
    first_force = np.zeros((20, 2), dtype=np.float64)
    first_force[:, 0] = 0.020
    first_speed = np.zeros((20, 2), dtype=np.float64)
    first_speed[:, 0] = 0.010
    _first_summary, state = h5_rederive_strict_20ms_slip_segment(
        first_times, first_force, first_speed
    )
    second_times = np.arange(18, dtype=np.float64) * 0.002
    second_force = np.zeros((18, 2), dtype=np.float64)
    second_speed = np.zeros((18, 2), dtype=np.float64)
    summary, _state = h5_rederive_strict_20ms_slip_segment(
        second_times, second_force, second_speed, initial_state=state
    )
    assert summary["maximum_completed_stance_cumulative_slip_m"] is not None
    assert np.isclose(
        float(summary["maximum_completed_stance_cumulative_slip_m"]),
        19 * 0.002 * 0.010,
    )


def test_jax_update_and_all_substep_losses_are_finite_and_jittable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    force = jnp.asarray(
        ((0.0, 0.020), (0.020, 0.020), (0.020, 0.0), (0.0, 0.0)),
        dtype=jnp.float32,
    )
    speed = jnp.full((4, 2), 0.025, dtype=jnp.float32)
    compiled = jax.jit(lambda f, s: h5_all_substep_quality_losses(f, s, xp=jnp))
    losses = compiled(force, speed)
    values = np.asarray(
        (
            losses.strict20ms_slip_rms_loss,
            losses.slip_tail_loss,
            losses.force_tail_loss,
            losses.force_qualified_sample_count,
        )
    )
    assert np.all(np.isfinite(values))
    assert values[-1] >= 0.0


def test_jax_continuous_all_substep_update_matches_numpy() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    force = np.zeros((20, 2), dtype=np.float32)
    force[1:, 0] = 0.020
    speed = np.full((20, 2), 0.040, dtype=np.float32)
    times = np.arange(20, dtype=np.float32) * 0.002
    expected = h5_all_substep_quality_update(force, speed, times_s=times)
    compiled = jax.jit(
        lambda f, s, t: h5_all_substep_quality_update(f, s, times_s=t, xp=jnp)
    )
    actual = compiled(jnp.asarray(force), jnp.asarray(speed), jnp.asarray(times))

    np.testing.assert_allclose(
        np.asarray(actual.losses.strict20ms_slip_rms_loss),
        np.asarray(expected.losses.strict20ms_slip_rms_loss),
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(actual.losses.slip_tail_loss),
        np.asarray(expected.losses.slip_tail_loss),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert int(actual.losses.force_qualified_sample_count) == int(
        expected.losses.force_qualified_sample_count
    )
    assert np.array_equal(
        np.asarray(actual.debounce.qualified_contact),
        np.asarray(expected.debounce.qualified_contact),
    )


def test_reverse_return_order_keeps_threshold_satisfying_motion_preferred() -> None:
    """A zero-slip/nominal-force reverse trace must not reward standing."""

    force = np.full((12, 2), 0.50, dtype=np.float64)
    speed = np.zeros((12, 2), dtype=np.float64)
    losses = h5_all_substep_quality_losses(force, speed)
    proof = h5_reverse_return_order_proof(
        command_vx_mps=-0.050,
        stationary_local_vx_mps=0.0,
        # Exactly the frozen lower signed-progress boundary.
        moving_local_vx_mps=-0.050 * 0.75,
        stationary_losses=losses,
        moving_losses=losses,
        reverse_speed_boundary_scale=-1.0,
        strict20ms_slip_rms_scale=-1.0,
        slip_tail_scale=-1.0,
        force_tail_scale=-1.0,
    )
    assert np.isclose(proof.stationary_speed_boundary_loss, 1.0)
    assert np.isclose(proof.moving_speed_boundary_loss, 0.0)
    assert np.isclose(proof.stationary_substep_cost, 0.0)
    assert np.isclose(proof.moving_substep_cost, 0.0)
    assert bool(proof.moving_strictly_preferred)
