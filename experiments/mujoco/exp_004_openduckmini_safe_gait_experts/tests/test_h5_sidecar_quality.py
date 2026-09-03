from __future__ import annotations

import numpy as np
import pytest

from safe_gait_experts.h5_sidecar_quality import (
    H5_V3_SIDECAR_QUALITY_CONTRACT_ID,
    h5_sidecar_score_control_tick,
    h5_sidecar_weighted_reward_delta,
    initialize_h5_sidecar_debounce_carry,
)
from safe_gait_experts.h5_substep_contact_alignment import (
    h5_all_substep_quality_update,
    initialize_h5_multiwindow_debounce,
)


def _tick(force: np.ndarray, speed: np.ndarray, start_sample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        force.astype(np.float64),
        speed.astype(np.float64),
        np.arange(start_sample, start_sample + len(force), dtype=np.float64) * 0.002,
    )


def test_sidecar_tick_matches_direct_causal_update_across_boundary() -> None:
    reset_force = np.asarray((0.0, 0.0), dtype=np.float64)
    force = np.zeros((20, 2), dtype=np.float64)
    force[1:, 0] = 0.020
    speed = np.full((20, 2), 0.040, dtype=np.float64)
    first_force, first_speed, first_times = _tick(force[:10], speed[:10], 1)
    second_force, second_speed, second_times = _tick(force[10:], speed[10:], 11)

    carry = initialize_h5_sidecar_debounce_carry(reset_force)
    first = h5_sidecar_score_control_tick(
        first_force,
        first_speed,
        times_s=first_times,
        reset_normalized_force=reset_force,
        carry=carry,
        terminal_after_tick=False,
    )
    second = h5_sidecar_score_control_tick(
        second_force,
        second_speed,
        times_s=second_times,
        reset_normalized_force=reset_force,
        carry=first.carry,
        terminal_after_tick=False,
    )

    direct_first = h5_all_substep_quality_update(
        first_force,
        first_speed,
        initial_debounce=initialize_h5_multiwindow_debounce(reset_force),
        times_s=first_times,
    )
    direct_second = h5_all_substep_quality_update(
        second_force,
        second_speed,
        initial_debounce=direct_first.debounce,
        times_s=second_times,
    )
    assert H5_V3_SIDECAR_QUALITY_CONTRACT_ID.endswith("20260812")
    assert int(second.losses.force_qualified_sample_count) == int(
        direct_second.losses.force_qualified_sample_count
    )
    assert np.isclose(
        second.losses.strict20ms_slip_rms_loss,
        direct_second.losses.strict20ms_slip_rms_loss,
    )
    assert np.array_equal(
        second.carry.debounce.qualified_contact,
        direct_second.debounce.qualified_contact,
    )


def test_terminal_resets_only_the_next_tick_and_preserves_inputs() -> None:
    reset_force = np.asarray((0.020, 0.0), dtype=np.float64)
    force = np.full((10, 2), 0.020, dtype=np.float64)
    speed = np.full((10, 2), 0.040, dtype=np.float64)
    times = np.arange(1, 11, dtype=np.float64) * 0.002
    force_before, speed_before, times_before = force.copy(), speed.copy(), times.copy()

    carry = initialize_h5_sidecar_debounce_carry(reset_force)
    terminal = h5_sidecar_score_control_tick(
        force,
        speed,
        times_s=times,
        reset_normalized_force=reset_force,
        carry=carry,
        terminal_after_tick=True,
    )
    next_tick = h5_sidecar_score_control_tick(
        force,
        speed,
        times_s=times + 0.020,
        reset_normalized_force=reset_force,
        carry=terminal.carry,
        terminal_after_tick=False,
    )
    fresh = h5_all_substep_quality_update(
        force,
        speed,
        initial_debounce=initialize_h5_multiwindow_debounce(reset_force),
        times_s=times + 0.020,
    )

    assert not bool(next_tick.carry.reset_before_tick)
    assert np.isclose(
        next_tick.losses.strict20ms_slip_rms_loss,
        fresh.losses.strict20ms_slip_rms_loss,
    )
    assert np.array_equal(force, force_before)
    assert np.array_equal(speed, speed_before)
    assert np.array_equal(times, times_before)


def test_weighted_reward_is_explicit_once_and_rejects_nonfinite_scale() -> None:
    force = np.full((10, 2), 0.020, dtype=np.float64)
    speed = np.full((10, 2), 0.040, dtype=np.float64)
    update = h5_all_substep_quality_update(force, speed)
    actual = h5_sidecar_weighted_reward_delta(
        update.losses,
        strict20ms_slip_rms_scale=-1.0,
        slip_tail_scale=-2.0,
        force_tail_scale=-3.0,
    )
    expected = (
        -update.losses.strict20ms_slip_rms_loss
        - 2.0 * update.losses.slip_tail_loss
        - 3.0 * update.losses.force_tail_loss
    )
    assert np.isclose(actual, expected)
    with pytest.raises(ValueError, match="finite"):
        h5_sidecar_weighted_reward_delta(
            update.losses,
            strict20ms_slip_rms_scale=float("nan"),
            slip_tail_scale=0.0,
            force_tail_scale=0.0,
        )


def test_sidecar_tick_is_jittable_without_physics() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    force = jnp.full((10, 2), 0.020, dtype=jnp.float32)
    speed = jnp.full((10, 2), 0.040, dtype=jnp.float32)
    times = jnp.arange(1, 11, dtype=jnp.float32) * 0.002
    reset_force = jnp.asarray((0.020, 0.0), dtype=jnp.float32)
    carry = initialize_h5_sidecar_debounce_carry(reset_force, xp=jnp)
    compiled = jax.jit(
        lambda c: h5_sidecar_score_control_tick(
            force,
            speed,
            times_s=times,
            reset_normalized_force=reset_force,
            carry=c,
            terminal_after_tick=jnp.asarray(False),
            xp=jnp,
        )
    )
    result = compiled(carry)
    assert np.isfinite(float(result.losses.force_tail_loss))
