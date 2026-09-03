"""Pure H5 quality sidecar contracts, intentionally outside MJX rollout.

This module consumes immutable 2-ms force/slip samples exported by a common
physics collector.  It contains no simulator, policy, PPO, or hardware calls.
In particular it must never be imported from an environment ``step`` method:
the caller owns the execution boundary between physics collection and quality
scoring.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from .h5_substep_contact_alignment import (
    H5AllSubstepQualityLosses,
    H5MultiWindowDebounceState,
    h5_all_substep_quality_update,
    initialize_h5_multiwindow_debounce,
)


H5_V3_SIDECAR_QUALITY_CONTRACT_ID = "H5_V3_SIDECAR_QUALITY_20260812"
"""Causal quality state used only after a completed physics rollout."""


class H5SidecarDebounceCarry(NamedTuple):
    """Quality-only causal carry independent of ``EnvState.info``.

    ``reset_before_tick`` is the terminal result from the previous control
    tick.  It is applied *before* scoring the next exported trajectory, which
    preserves the first reset measurement without fabricating a contact event.
    """

    debounce: H5MultiWindowDebounceState
    reset_before_tick: Any


class H5SidecarTickScore(NamedTuple):
    """Raw H5 losses and the exact carry for the next control tick."""

    losses: H5AllSubstepQualityLosses
    carry: H5SidecarDebounceCarry


def initialize_h5_sidecar_debounce_carry(
    reset_normalized_force: Any,
    *,
    xp: Any = np,
) -> H5SidecarDebounceCarry:
    """Create a carry which will consume the reset measurement on first use."""

    return H5SidecarDebounceCarry(
        initialize_h5_multiwindow_debounce(reset_normalized_force, xp=xp),
        xp.asarray(True),
    )


def _select_debounce(
    *,
    reset_before_tick: Any,
    reset_debounce: H5MultiWindowDebounceState,
    continuing_debounce: H5MultiWindowDebounceState,
    xp: Any,
) -> H5MultiWindowDebounceState:
    """Choose reset/continuing state without a traced Python branch."""

    return H5MultiWindowDebounceState(
        *(
            xp.where(reset_before_tick, reset_value, continuing_value)
            for reset_value, continuing_value in zip(
                reset_debounce, continuing_debounce, strict=True
            )
        )
    )


def h5_sidecar_score_control_tick(
    normalized_normal_force_samples: Any,
    tangential_speed_samples_mps: Any,
    *,
    times_s: Any,
    reset_normalized_force: Any,
    carry: H5SidecarDebounceCarry,
    terminal_after_tick: Any,
    xp: Any = np,
) -> H5SidecarTickScore:
    """Score one immutable control-tick trace with causal reset semantics.

    The caller must supply the terminal result for *this* tick.  It is retained
    only for the next tick; it does not change any score from the completed
    physical transition.  No input object is mutated.
    """

    reset_debounce = initialize_h5_multiwindow_debounce(
        reset_normalized_force, xp=xp
    )
    initial_debounce = _select_debounce(
        reset_before_tick=carry.reset_before_tick,
        reset_debounce=reset_debounce,
        continuing_debounce=carry.debounce,
        xp=xp,
    )
    update = h5_all_substep_quality_update(
        normalized_normal_force_samples,
        tangential_speed_samples_mps,
        initial_debounce=initial_debounce,
        times_s=times_s,
        xp=xp,
    )
    return H5SidecarTickScore(
        update.losses,
        H5SidecarDebounceCarry(
            update.debounce,
            xp.asarray(terminal_after_tick).astype(bool),
        ),
    )


def h5_sidecar_weighted_reward_delta(
    losses: H5AllSubstepQualityLosses,
    *,
    strict20ms_slip_rms_scale: float,
    slip_tail_scale: float,
    force_tail_scale: float,
    xp: Any = np,
) -> Any:
    """Apply the three explicit H5 scales exactly once, after physics ends."""

    scales = (
        float(strict20ms_slip_rms_scale),
        float(slip_tail_scale),
        float(force_tail_scale),
    )
    if not all(np.isfinite(value) for value in scales):
        raise ValueError("H5 sidecar reward scales must be finite")
    return (
        xp.asarray(scales[0]) * losses.strict20ms_slip_rms_loss
        + xp.asarray(scales[1]) * losses.slip_tail_loss
        + xp.asarray(scales[2]) * losses.force_tail_loss
    )
