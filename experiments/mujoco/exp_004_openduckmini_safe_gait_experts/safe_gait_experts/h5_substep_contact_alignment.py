"""Pure H5 preflight utilities for strict 2 ms contact-quality alignment.

This module has no simulator, policy, training, or hardware side effects.  It
implements the strict evaluator's force-Schmitt and causal debounce semantics
over retained 2 ms samples so training telemetry can be compared directly to
the frozen evaluator before any reward is changed.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from .h4_training_alignment import (
    FORCE_CONTACT_OFF_NORMALIZED,
    FORCE_CONTACT_ON_NORMALIZED,
    STRICT_SLIP_RMS_M_S,
    STRICT_SLIP_TAIL_M_S,
    STRICT_REVERSE_SPEED_RATIO_LOWER,
    STRICT_REVERSE_SPEED_RATIO_UPPER,
    STRICT_STANCE_SLIP_BUDGET_M,
    STRICT_TOTAL_NORMAL_FORCE_TAIL_NORMALIZED,
    V4_PHYSICS_SUBSTEP_DT_S,
    force_schmitt_contacts,
)


FROZEN_DEBOUNCE_WINDOWS_S = (0.010, 0.020, 0.030, 0.040)
"""The four windows required by ``GaitQualityAccumulator`` acceptance."""

H5_V3_SE2_SUBSTEP_CONTACT_ALIGNMENT_ID = (
    "H5_V3_SE2_SUBSTEP_CONTACT_ALIGNMENT_20260812"
)
"""One-factor successor contract: SE(2) residuals plus 2-ms contact costs."""

H5_V3_T1_FIXED_QUALITY_REPLAY_ABLATION_ID = (
    "H5_V3_T1_FIXED_QUALITY_REPLAY_ABLATION_20260812"
)
"""No-PPO diagnostic only: removes the second MJX forward replay."""

H5_V3_T1_FIXED_QUALITY_REPLAY_FORCE_SHA256 = (
    "4e66a273ed886b8bcf436a8570b314b83568703762ded9bbe64c1223ecc9ecb0"
)
H5_V3_T1_FIXED_QUALITY_REPLAY_SPEED_SHA256 = (
    "190211e8c1eccb63f171b4d2d6a3e55d5b996ace21d7c5abc0e3648ac470252d"
)


def h5_v3_t1_fixed_quality_replay(*, xp: Any) -> tuple[Any, Any]:
    """Return the sealed 10x2 diagnostic substitute for the quality replay.

    This is intentionally unavailable to PPO.  It retains the H5 loss and
    debounce graph while replacing only the ten post-physics ``mjx.forward``
    calls, so a failed compiled repeat can be attributed without changing the
    authoritative physics scan.
    """

    return (
        xp.full((10, 2), 0.5, dtype=xp.float32),
        xp.full((10, 2), 0.01, dtype=xp.float32),
    )


def h5_v3_t1_fixed_quality_replay_manifest() -> Mapping[str, object]:
    """Return the byte-pinned host manifest for the diagnostic substitute."""

    force = np.full((10, 2), np.float32(0.5), dtype=np.float32)
    speed = np.full((10, 2), np.float32(0.01), dtype=np.float32)

    def digest(array: np.ndarray) -> str:
        import hashlib
        import json

        result = hashlib.sha256()
        result.update(array.dtype.str.encode("ascii"))
        result.update(json.dumps(list(array.shape)).encode("ascii"))
        result.update(np.ascontiguousarray(array).tobytes(order="C"))
        return result.hexdigest()

    force_digest = digest(force)
    speed_digest = digest(speed)
    if (
        force_digest != H5_V3_T1_FIXED_QUALITY_REPLAY_FORCE_SHA256
        or speed_digest != H5_V3_T1_FIXED_QUALITY_REPLAY_SPEED_SHA256
    ):
        raise RuntimeError("H5 T=1 fixed quality replay bytes drifted")
    return {
        "contract_id": H5_V3_T1_FIXED_QUALITY_REPLAY_ABLATION_ID,
        "force_shape": [10, 2],
        "speed_shape": [10, 2],
        "dtype": force.dtype.str,
        "normalized_force_raw_bytes_sha256": force_digest,
        "tangential_speed_raw_bytes_sha256": speed_digest,
    }


def _vector(value: Any, *, width: int, name: str, xp: Any) -> Any:
    array = xp.asarray(value)
    if array.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {array.shape}")
    return array


def _matrix(value: Any, *, width: int, name: str, xp: Any) -> Any:
    array = xp.asarray(value)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{name} must have shape (samples, {width}), got {array.shape}"
        )
    return array


def frozen_debounce_window_intervals(
    windows_s: Sequence[float] = FROZEN_DEBOUNCE_WINDOWS_S,
    *,
    dt_s: float = V4_PHYSICS_SUBSTEP_DT_S,
) -> tuple[int, ...]:
    """Return exact 2 ms interval counts, rejecting nonrepresentable windows."""

    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    values = tuple(float(window) for window in windows_s)
    if values != FROZEN_DEBOUNCE_WINDOWS_S:
        raise ValueError("H5 alignment requires exactly the frozen 10/20/30/40 ms windows")
    intervals: list[int] = []
    for window in values:
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError("debounce windows must be finite and positive")
        quotient = window / dt_s
        rounded = int(round(quotient))
        if rounded <= 0 or not np.isclose(quotient, rounded, atol=1.0e-12, rtol=0.0):
            raise ValueError("every debounce window must be an exact multiple of dt_s")
        intervals.append(rounded)
    return tuple(intervals)


class H5MultiWindowDebounceState(NamedTuple):
    """Causal raw/qualified state for all four frozen debounce windows."""

    raw_contact: Any  # (2,)
    qualified_contact: Any  # (4, 2)
    pending_active: Any  # (4, 2)
    pending_target: Any  # (4, 2)
    pending_since_s: Any  # (4, 2), first raw-opposing sample time


class H5MultiWindowDebounceUpdate(NamedTuple):
    state: H5MultiWindowDebounceState
    confirmed_transition: Any  # (4, 2)
    touchdown_event: Any  # (4, 2)
    liftoff_event: Any  # (4, 2)


class H5MultiWindowDebounceTrace(NamedTuple):
    """Per-2-ms raw Schmitt and debounced contact trajectories."""

    raw_contact: Any  # (samples, 2)
    qualified_contact: Any  # (samples, 4, 2)
    final_state: H5MultiWindowDebounceState


class H5AllSubstepQualityLosses(NamedTuple):
    """Differentiable all-sample proxies using nominal 20 ms qualified contact.

    The values intentionally expose raw quantities instead of silently applying
    reward scales.  They are preflight diagnostics until a separately approved
    training contract explicitly consumes them.
    """

    strict20ms_slip_rms_loss: Any
    slip_tail_loss: Any
    force_tail_loss: Any
    force_qualified_sample_count: Any


class H5AllSubstepQualityUpdate(NamedTuple):
    """One all-substep loss update plus its causal debounce continuation."""

    losses: H5AllSubstepQualityLosses
    debounce: H5MultiWindowDebounceState


class H5ReverseReturnOrderProof(NamedTuple):
    """Local reverse-objective ordering for an ideal strict-contact trace.

    The proof is intentionally local to the reward components that differ
    between a stationary reverse trace and a threshold-satisfying reverse
    trace.  It does not claim that an untrained policy has produced that
    trace; it prevents this contact-quality delta from making standing the
    better outcome in the already-defined reverse speed objective.
    """

    stationary_speed_boundary_loss: Any
    moving_speed_boundary_loss: Any
    stationary_substep_cost: Any
    moving_substep_cost: Any
    stationary_local_return: Any
    moving_local_return: Any
    moving_strictly_preferred: Any


class H5StrictSlipContinuityState(NamedTuple):
    """Persistent 20 ms force/debounce/stance state across schedule segments.

    ``stance_cumulative_m`` uses NaN for an inactive foot.  This state is a
    NumPy-only rederivation aid; it mirrors the evaluator's exported contact
    continuity state and is deliberately not a PPO carry.
    """

    debounce: H5MultiWindowDebounceState
    stance_cumulative_m: np.ndarray


def initialize_h5_multiwindow_debounce(
    normalized_normal_force: Any,
    *,
    xp: Any = np,
) -> H5MultiWindowDebounceState:
    """Initialize from the first raw measurement without inventing an event."""

    force = _vector(
        normalized_normal_force,
        width=2,
        name="normalized_normal_force",
        xp=xp,
    )
    if xp is np and not np.all(np.isfinite(np.asarray(force))):
        raise ValueError("normalized_normal_force must be finite")
    raw = force_schmitt_contacts(force, xp.zeros(2, dtype=bool), xp=xp)
    window_count = len(FROZEN_DEBOUNCE_WINDOWS_S)
    qualified = xp.broadcast_to(raw, (window_count, 2))
    return H5MultiWindowDebounceState(
        raw,
        qualified,
        xp.zeros((window_count, 2), dtype=bool),
        qualified,
        xp.zeros((window_count, 2), dtype=xp.float32),
    )


def update_h5_multiwindow_debounce(
    previous_state: H5MultiWindowDebounceState,
    normalized_normal_force: Any,
    *,
    time_s: Any,
    xp: Any = np,
) -> H5MultiWindowDebounceUpdate:
    """Advance all four strict causal windows by one 2 ms sample.

    This matches ``gait_quality._contact_sequence_summary`` exactly: a raw
    change starts pending at its first sample and commits after the window's
    elapsed *floating-point sampled time*; an unfinished pending state remains
    uncommitted.  The time comparison is deliberate: the strict evaluator uses
    ``time_s - pending_since_s >= debounce_s`` and its binary boundary behavior
    is part of the frozen acceptance contract.
    """

    force = _vector(
        normalized_normal_force,
        width=2,
        name="normalized_normal_force",
        xp=xp,
    )
    if xp is np and not np.all(np.isfinite(np.asarray(force))):
        raise ValueError("normalized_normal_force must be finite")
    raw_previous = _vector(previous_state.raw_contact, width=2, name="raw_contact", xp=xp)
    qualified = xp.asarray(previous_state.qualified_contact).astype(bool)
    pending_active = xp.asarray(previous_state.pending_active).astype(bool)
    pending_target = xp.asarray(previous_state.pending_target).astype(bool)
    pending_since = xp.asarray(previous_state.pending_since_s)
    expected_shape = (len(FROZEN_DEBOUNCE_WINDOWS_S), 2)
    for name, value in (
        ("qualified_contact", qualified),
        ("pending_active", pending_active),
        ("pending_target", pending_target),
        ("pending_since_s", pending_since),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
    if xp is np:
        values = np.asarray(pending_since)
        if (
            not np.all(np.isfinite(values))
        ):
            raise ValueError("pending_since_s must be finite")
    time = xp.asarray(time_s)
    if time.shape != ():
        raise ValueError("time_s must be scalar")
    if xp is np and not np.isfinite(float(time)):
        raise ValueError("time_s must be finite")

    raw = force_schmitt_contacts(force, raw_previous, xp=xp)
    raw_matrix = xp.broadcast_to(raw, expected_shape)
    differs = raw_matrix != qualified
    starts = ~pending_active & differs
    continues = pending_active & differs & (raw_matrix == pending_target)
    windows_s = xp.asarray(FROZEN_DEBOUNCE_WINDOWS_S, dtype=time.dtype)[:, None]
    confirmed = continues & ((time - pending_since) >= windows_s)
    touchdown = confirmed & ~qualified & raw_matrix
    liftoff = confirmed & qualified & ~raw_matrix
    new_qualified = xp.where(confirmed, raw_matrix, qualified)
    still_pending = continues & ~confirmed
    new_pending_active = starts | still_pending
    new_pending_target = xp.where(
        starts,
        raw_matrix,
        xp.where(new_pending_active, pending_target, new_qualified),
    )
    new_pending_since = xp.where(
        starts,
        time,
        xp.where(still_pending, pending_since, 0.0),
    )
    return H5MultiWindowDebounceUpdate(
        H5MultiWindowDebounceState(
            raw,
            new_qualified,
            new_pending_active,
            new_pending_target,
            new_pending_since,
        ),
        confirmed,
        touchdown,
        liftoff,
    )


def h5_multiwindow_debounce_trace(
    normalized_normal_force_samples: Any,
    *,
    times_s: Any | None = None,
    xp: Any = np,
) -> H5MultiWindowDebounceTrace:
    """Return all four qualified traces for ordered 2 ms force samples."""

    force = _matrix(
        normalized_normal_force_samples,
        width=2,
        name="normalized_normal_force_samples",
        xp=xp,
    )
    if force.shape[0] < 1:
        raise ValueError("at least one 2 ms force sample is required")
    if xp is np and not np.all(np.isfinite(np.asarray(force))):
        raise ValueError("normalized_normal_force_samples must be finite")
    times = (
        xp.arange(force.shape[0], dtype=force.dtype) * V4_PHYSICS_SUBSTEP_DT_S
        if times_s is None
        else xp.asarray(times_s)
    )
    if times.shape != (force.shape[0],):
        raise ValueError("times_s must have shape (samples,)")
    if xp is np and (
        not np.all(np.isfinite(np.asarray(times)))
        or (len(times) > 1 and np.any(np.diff(np.asarray(times)) <= 0.0))
    ):
        raise ValueError("times_s must be finite and strictly increasing")
    state = initialize_h5_multiwindow_debounce(force[0], xp=xp)
    raw_history = [state.raw_contact]
    qualified_history = [state.qualified_contact]
    for index in range(1, force.shape[0]):
        update = update_h5_multiwindow_debounce(
            state, force[index], time_s=times[index], xp=xp
        )
        state = update.state
        raw_history.append(state.raw_contact)
        qualified_history.append(state.qualified_contact)
    return H5MultiWindowDebounceTrace(
        xp.stack(raw_history, axis=0),
        xp.stack(qualified_history, axis=0),
        state,
    )


def h5_multiwindow_debounce_summaries(
    times_s: Sequence[float],
    normalized_normal_force_samples: Sequence[Sequence[float]],
) -> Mapping[str, Mapping[str, object]]:
    """Summarize the exact fields used by strict debounce acceptance.

    This NumPy reporting helper purposely preserves the evaluator's simultaneous
    touchdown ordering (left then right) so comparison is exact rather than a
    friendlier but incompatible alternation interpretation.
    """

    times = np.asarray(times_s, dtype=np.float64)
    force = np.asarray(normalized_normal_force_samples, dtype=np.float64)
    if times.ndim != 1 or force.shape != (len(times), 2):
        raise ValueError("times and force samples must have shapes (samples,) and (samples, 2)")
    if len(times) < 1 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(force)):
        raise ValueError("times and force samples must be finite and nonempty")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    trace = h5_multiwindow_debounce_trace(force, times_s=times)
    qualified = np.asarray(trace.qualified_contact, dtype=bool)
    summaries: dict[str, Mapping[str, object]] = {}
    for window_index, window_s in enumerate(FROZEN_DEBOUNCE_WINDOWS_S):
        filtered = qualified[:, window_index, :]
        touchdown_counts = tuple(
            int(np.count_nonzero((~filtered[:-1, foot]) & filtered[1:, foot]))
            for foot in (0, 1)
        )
        sequence: list[int] = []
        timestamps: list[list[float]] = [[], []]
        for sample_index in range(1, len(filtered)):
            for foot in (0, 1):
                if not filtered[sample_index - 1, foot] and filtered[sample_index, foot]:
                    sequence.append(foot)
                    timestamps[foot].append(float(times[sample_index]))
        alternating = (
            None
            if len(sequence) < 2
            else sum(left != right for left, right in zip(sequence, sequence[1:]))
            / (len(sequence) - 1)
        )
        summaries[f"{int(round(window_s * 1000.0))}ms"] = {
            "left_touchdowns": touchdown_counts[0],
            "right_touchdowns": touchdown_counts[1],
            "left_contact_rate": float(np.mean(filtered[:, 0])),
            "right_contact_rate": float(np.mean(filtered[:, 1])),
            "single_support_rate": float(np.mean(np.logical_xor(filtered[:, 0], filtered[:, 1]))),
            "flight_rate": float(np.mean(~np.logical_or(filtered[:, 0], filtered[:, 1]))),
            "contact_duty_imbalance": abs(
                float(np.mean(filtered[:, 0])) - float(np.mean(filtered[:, 1]))
            ),
            "step_count_imbalance": abs(touchdown_counts[0] - touchdown_counts[1]),
            "alternating_touchdown_fraction": alternating,
            "left_touchdown_timestamps_s": timestamps[0],
            "right_touchdown_timestamps_s": timestamps[1],
        }
    return summaries


def h5_all_substep_quality_losses(
    normalized_normal_force_samples: Any,
    tangential_speed_samples_mps: Any,
    *,
    times_s: Any | None = None,
    xp: Any = np,
) -> H5AllSubstepQualityLosses:
    """Return fresh-segment all-2-ms slip/force losses with no reward scale.

    This compatibility wrapper deliberately starts a new debounce state from
    the first supplied measurement.  Training code that spans control ticks
    must use :func:`h5_all_substep_quality_update` and retain its continuation.
    """

    return h5_all_substep_quality_update(
        normalized_normal_force_samples,
        tangential_speed_samples_mps,
        times_s=times_s,
        xp=xp,
    ).losses


def h5_all_substep_quality_update(
    normalized_normal_force_samples: Any,
    tangential_speed_samples_mps: Any,
    *,
    initial_debounce: H5MultiWindowDebounceState | None = None,
    times_s: Any | None = None,
    xp: Any = np,
) -> H5AllSubstepQualityUpdate:
    """Score one ordered 2-ms trajectory with optional debounce continuity.

    The function is simulator-free and JAX-safe.  It evaluates force-qualified
    stance using the strict 20-ms Schmitt/debounce state at every retained
    sample, while force-tail uses every sample.  The returned state is the
    exact causal continuation for the next control tick; no terminal contact
    transition is fabricated.
    """

    force = _matrix(
        normalized_normal_force_samples,
        width=2,
        name="normalized_normal_force_samples",
        xp=xp,
    )
    speed = _matrix(
        tangential_speed_samples_mps,
        width=2,
        name="tangential_speed_samples_mps",
        xp=xp,
    )
    if speed.shape != force.shape:
        raise ValueError("force and tangential speed samples must have equal shape")
    if force.shape[0] < 1:
        raise ValueError("at least one 2 ms sample is required")
    if xp is np and (
        not np.all(np.isfinite(np.asarray(force)))
        or not np.all(np.isfinite(np.asarray(speed)))
        or np.any(np.asarray(force) < 0.0)
        or np.any(np.asarray(speed) < 0.0)
    ):
        raise ValueError("force and tangential speed samples must be finite and non-negative")
    times = (
        xp.arange(force.shape[0], dtype=force.dtype) * V4_PHYSICS_SUBSTEP_DT_S
        if times_s is None
        else xp.asarray(times_s)
    )
    if times.shape != (force.shape[0],):
        raise ValueError("times_s must have shape (samples,)")
    if xp is np and (
        not np.all(np.isfinite(np.asarray(times)))
        or (len(times) > 1 and np.any(np.diff(np.asarray(times)) <= 0.0))
    ):
        raise ValueError("times_s must be finite and strictly increasing")
    state = (
        initialize_h5_multiwindow_debounce(force[0], xp=xp)
        if initial_debounce is None
        else initial_debounce
    )
    nominal_window_index = FROZEN_DEBOUNCE_WINDOWS_S.index(0.020)
    sum_square = xp.zeros((), dtype=speed.dtype)
    sum_tail = xp.zeros((), dtype=speed.dtype)
    sample_count = xp.zeros((), dtype=speed.dtype)
    for index in range(force.shape[0]):
        update = update_h5_multiwindow_debounce(
            state,
            force[index],
            time_s=times[index],
            xp=xp,
        )
        state = update.state
        stance = (
            state.qualified_contact[nominal_window_index] & state.raw_contact
        )
        stance_float = stance.astype(speed.dtype)
        sample_count = sample_count + xp.sum(stance_float)
        sum_square = sum_square + xp.sum(
            stance_float * xp.square(speed[index])
        )
        sum_tail = sum_tail + xp.sum(
            stance_float
            * xp.square(
                xp.maximum(speed[index] - STRICT_SLIP_TAIL_M_S, 0.0)
                / STRICT_SLIP_TAIL_M_S
            )
        )
    denominator = xp.maximum(sample_count, 1.0)
    # Match the established H4 convention: all slip costs are normalized by
    # the frozen strict RMS threshold before being reward-scaled.  Leaving
    # this in m^2/s^2 would make a -1 scale numerically negligible compared
    # with the pre-existing normalized reverse-speed boundary loss.
    normalized_mean_square = sum_square / (
        denominator * (STRICT_SLIP_RMS_M_S**2)
    )
    slip_tail = sum_tail / denominator
    total_force = xp.sum(xp.maximum(force, 0.0), axis=1)
    force_tail = xp.mean(
        xp.square(
            xp.maximum(
                total_force - STRICT_TOTAL_NORMAL_FORCE_TAIL_NORMALIZED,
                0.0,
            )
            / STRICT_TOTAL_NORMAL_FORCE_TAIL_NORMALIZED
        )
    )
    return H5AllSubstepQualityUpdate(
        H5AllSubstepQualityLosses(
            normalized_mean_square,
            slip_tail,
            force_tail,
            sample_count,
        ),
        state,
    )


def h5_reverse_speed_boundary_loss(
    local_vx_mps: Any,
    command_vx_mps: Any,
    *,
    xp: Any = np,
) -> Any:
    """Exactly reproduce the H4 reverse-speed boundary term for one sample."""

    local_vx = xp.asarray(local_vx_mps)
    command_vx = xp.asarray(command_vx_mps)
    if local_vx.shape != () or command_vx.shape != ():
        raise ValueError("local_vx_mps and command_vx_mps must be scalars")
    if xp is np and (
        not np.isfinite(float(local_vx))
        or not np.isfinite(float(command_vx))
        or float(command_vx) >= 0.0
    ):
        raise ValueError("reverse speed inputs must be finite with command_vx_mps < 0")
    denominator = xp.minimum(command_vx, -1.0e-6)
    speed_ratio = local_vx / denominator
    return xp.square(
        xp.maximum(STRICT_REVERSE_SPEED_RATIO_LOWER - speed_ratio, 0.0)
        / STRICT_REVERSE_SPEED_RATIO_LOWER
    ) + xp.square(
        xp.maximum(speed_ratio - STRICT_REVERSE_SPEED_RATIO_UPPER, 0.0)
        / STRICT_REVERSE_SPEED_RATIO_UPPER
    )


def h5_reverse_return_order_proof(
    *,
    command_vx_mps: Any,
    stationary_local_vx_mps: Any,
    moving_local_vx_mps: Any,
    stationary_losses: H5AllSubstepQualityLosses,
    moving_losses: H5AllSubstepQualityLosses,
    reverse_speed_boundary_scale: float,
    strict20ms_slip_rms_scale: float,
    slip_tail_scale: float,
    force_tail_scale: float,
    xp: Any = np,
) -> H5ReverseReturnOrderProof:
    """Prove local reward ordering without simulator or policy side effects.

    Every supplied scale must be non-positive because each raw term is a cost.
    This makes the sign convention auditable and prevents a configuration
    error from turning an unsafe contact tail into a reward.
    """

    scales = (
        float(reverse_speed_boundary_scale),
        float(strict20ms_slip_rms_scale),
        float(slip_tail_scale),
        float(force_tail_scale),
    )
    if not all(np.isfinite(value) and value <= 0.0 for value in scales):
        raise ValueError("reverse and all-substep reward scales must be finite costs")

    stationary_speed = h5_reverse_speed_boundary_loss(
        stationary_local_vx_mps, command_vx_mps, xp=xp
    )
    moving_speed = h5_reverse_speed_boundary_loss(
        moving_local_vx_mps, command_vx_mps, xp=xp
    )

    def substep_cost(losses: H5AllSubstepQualityLosses) -> Any:
        return (
            strict20ms_slip_rms_scale * losses.strict20ms_slip_rms_loss
            + slip_tail_scale * losses.slip_tail_loss
            + force_tail_scale * losses.force_tail_loss
        )

    stationary_contact = substep_cost(stationary_losses)
    moving_contact = substep_cost(moving_losses)
    stationary_return = reverse_speed_boundary_scale * stationary_speed + stationary_contact
    moving_return = reverse_speed_boundary_scale * moving_speed + moving_contact
    return H5ReverseReturnOrderProof(
        stationary_speed,
        moving_speed,
        stationary_contact,
        moving_contact,
        stationary_return,
        moving_return,
        moving_return > stationary_return,
    )


def h5_rederive_strict_20ms_slip_segment(
    times_s: Sequence[float],
    normalized_normal_force_samples: Sequence[Sequence[float]],
    tangential_speed_samples_mps: Sequence[Sequence[float]],
    *,
    initial_state: H5StrictSlipContinuityState | None = None,
) -> tuple[Mapping[str, float | int | None], H5StrictSlipContinuityState]:
    """Reproduce evaluator 20 ms slip accounting, including segment continuity.

    The returned metrics are local to this segment, just as
    ``GaitQualityAccumulator.finalize`` is.  ``initial_state`` carries only
    the prior raw/debounced contact and unclosed stance distance; any final
    pending transition remains pending and is never fabricated into an event.
    """

    times = np.asarray(times_s, dtype=np.float64)
    force = np.asarray(normalized_normal_force_samples, dtype=np.float64)
    speed = np.asarray(tangential_speed_samples_mps, dtype=np.float64)
    if times.ndim != 1 or force.shape != (len(times), 2) or speed.shape != force.shape:
        raise ValueError("times, force, and speed shapes are inconsistent")
    if len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("at least two strictly increasing finite times are required")
    if (
        not np.all(np.isfinite(force))
        or not np.all(np.isfinite(speed))
        or np.any(force < 0.0)
        or np.any(speed < 0.0)
    ):
        raise ValueError("force and speed samples must be finite and non-negative")
    nominal_window_index = FROZEN_DEBOUNCE_WINDOWS_S.index(0.020)
    if initial_state is None:
        debounce_state = initialize_h5_multiwindow_debounce(force[0])
        contact = np.asarray(
            debounce_state.qualified_contact[nominal_window_index], dtype=bool
        )
        cumulative = np.where(contact, 0.0, np.nan).astype(np.float64)
        start_index = 1
        previous_time = float(times[0])
    else:
        debounce_state = initial_state.debounce
        cumulative = np.asarray(initial_state.stance_cumulative_m, dtype=np.float64).copy()
        if cumulative.shape != (2,) or not np.all(np.isfinite(cumulative) | np.isnan(cumulative)):
            raise ValueError("initial stance continuity must be a finite-or-NaN pair")
        start_index = 0
        previous_time = 0.0
    completed: list[float] = []
    samples: list[float] = []
    for sample_index in range(start_index, len(times)):
        prior_qualified = np.asarray(
            debounce_state.qualified_contact[nominal_window_index], dtype=bool
        )
        prior_raw = np.asarray(debounce_state.raw_contact, dtype=bool)
        update = update_h5_multiwindow_debounce(
            debounce_state, force[sample_index], time_s=float(times[sample_index])
        )
        debounce_state = update.state
        current_qualified = np.asarray(
            debounce_state.qualified_contact[nominal_window_index], dtype=bool
        )
        current_raw = np.asarray(debounce_state.raw_contact, dtype=bool)
        dt = float(times[sample_index] - previous_time)
        if dt < 0.0:
            raise ValueError("segment time cannot move backwards")
        for foot in (0, 1):
            was_contact = bool(prior_qualified[foot])
            is_contact = bool(current_qualified[foot])
            if is_contact and not was_contact:
                cumulative[foot] = 0.0
            elif not is_contact and was_contact:
                completed.append(
                    0.0 if np.isnan(cumulative[foot]) else float(cumulative[foot])
                )
                cumulative[foot] = np.nan
            if (
                dt > 0.0
                and is_contact
                and was_contact
                and current_raw[foot]
                and prior_raw[foot]
            ):
                value = float(speed[sample_index, foot])
                samples.append(value)
                cumulative[foot] = (
                    0.0 if np.isnan(cumulative[foot]) else cumulative[foot]
                ) + value * dt
        previous_time = float(times[sample_index])
    values = np.asarray(samples, dtype=np.float64)
    # ``GaitQualityAccumulator.finalize`` scores a stance that is still
    # qualified at the end of this segment using its accumulated distance.
    # That is deliberately distinct from contact-event handling: an unconfirmed
    # terminal contact transition remains right-censored in ``continuation``.
    # Keep the historical output key for the runner, but match the evaluator's
    # terminal-stance-inclusive metric exactly.
    scored_stances = list(completed)
    for foot in (0, 1):
        if bool(debounce_state.qualified_contact[nominal_window_index, foot]):
            scored_stances.append(
                0.0 if np.isnan(cumulative[foot]) else float(cumulative[foot])
            )
    summary: Mapping[str, float | int | None] = {
        "slip_sample_count": int(values.size),
        "stance_slip_rms_mps": None if not values.size else float(np.sqrt(np.mean(np.square(values)))),
        "stance_slip_p95_mps": None if not values.size else float(np.percentile(values, 95)),
        "maximum_completed_stance_cumulative_slip_m": (
            None if not scored_stances else float(max(scored_stances))
        ),
        "strict_slip_rms_limit_mps": STRICT_SLIP_RMS_M_S,
        "strict_stance_cumulative_limit_m": STRICT_STANCE_SLIP_BUDGET_M,
        "force_contact_on_fraction_body_weight": FORCE_CONTACT_ON_NORMALIZED,
        "force_contact_off_fraction_body_weight": FORCE_CONTACT_OFF_NORMALIZED,
    }
    # Match GaitQualityAccumulator.export_contact_continuity_state: next
    # segment time restarts at zero, so pending timestamps are exported as a
    # negative age rather than retaining this segment's absolute clock.
    pending_age = np.where(
        np.asarray(debounce_state.pending_active, dtype=bool),
        np.maximum(
            previous_time - np.asarray(debounce_state.pending_since_s, dtype=np.float64),
            0.0,
        ),
        0.0,
    )
    continuation_debounce = debounce_state._replace(pending_since_s=-pending_age)
    return summary, H5StrictSlipContinuityState(continuation_debounce, cumulative)


def h5_completed_stance_slip_summary(
    times_s: Sequence[float],
    normalized_normal_force_samples: Sequence[Sequence[float]],
    tangential_speed_samples_mps: Sequence[Sequence[float]],
) -> Mapping[str, float | int | None]:
    """Reproduce strict nominal-window slip accounting for one fresh segment."""

    summary, _state = h5_rederive_strict_20ms_slip_segment(
        times_s,
        normalized_normal_force_samples,
        tangential_speed_samples_mps,
    )
    return summary


__all__ = [
    "FROZEN_DEBOUNCE_WINDOWS_S",
    "H5_V3_SE2_SUBSTEP_CONTACT_ALIGNMENT_ID",
    "H5AllSubstepQualityLosses",
    "H5AllSubstepQualityUpdate",
    "H5ReverseReturnOrderProof",
    "H5MultiWindowDebounceState",
    "H5MultiWindowDebounceTrace",
    "H5MultiWindowDebounceUpdate",
    "H5StrictSlipContinuityState",
    "frozen_debounce_window_intervals",
    "h5_all_substep_quality_losses",
    "h5_all_substep_quality_update",
    "h5_reverse_return_order_proof",
    "h5_reverse_speed_boundary_loss",
    "h5_completed_stance_slip_summary",
    "h5_multiwindow_debounce_summaries",
    "h5_multiwindow_debounce_trace",
    "h5_rederive_strict_20ms_slip_segment",
    "initialize_h5_multiwindow_debounce",
    "update_h5_multiwindow_debounce",
]
