"""H4 training/runtime alignment for OpenDuckMini.

This module is intentionally separate from the H3 trainer and release
runtime.  Its pure functions accept either NumPy or ``jax.numpy`` as ``xp``;
the optional environment factory keeps JAX/MJX behind a lazy boundary.

The target order is the runtime order:

1. preserve the physical-SAFE reset target as guard state,
2. clamp the newly composed policy/profile target to the 0.050 rad envelope,
3. slew each leg once by at most 0.040 rad per 20 ms control tick,
4. clamp the applied result to the physical SAFE limits, and
5. force every head target to exact zero before physics.

Hardware deployment remains prohibited.  These helpers only align simulation
training with the already-frozen simulation runtime contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np

from .contract import (
    ACTUATOR_JOINT_ORDER,
    CONTROL_FIRST_STARTUP_DT_S,
    HEAD_JOINTS,
    LEG_TARGET_MARGIN_RAD,
    RESET_NOISE_MARGIN_RAD,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
    TARGET_SLEW_LIMIT_RAD_PER_S,
)
from .h5_command_conditioned_se2 import (
    advance_h5_v3_command_heading,
    h5_v3_command_conditioned_se2_residuals,
)


ACTION_COUNT = len(ACTUATOR_JOINT_ORDER)
HEAD_ACTION_INDICES = tuple(
    index
    for index, name in enumerate(ACTUATOR_JOINT_ORDER)
    if name in HEAD_JOINTS
)
LEG_ACTION_INDICES = tuple(
    index
    for index, name in enumerate(ACTUATOR_JOINT_ORDER)
    if name not in HEAD_JOINTS
)
MAX_TARGET_DELTA_PER_TICK_RAD = (
    TARGET_SLEW_LIMIT_RAD_PER_S * CONTROL_FIRST_STARTUP_DT_S
)
OBSERVATION_MOTOR_TARGET_SLICE = slice(83, 97)
OBSERVATION_POLICY_COMMAND_SLICE = slice(6, 13)
OBSERVATION_IMITATION_PHASE_SLICE = slice(99, 101)
LEGACY_PRIVILEGED_REFERENCE_SLICE = slice(169, 209)
LEGACY_PRIVILEGED_IMITATION_INDEX_SLICE = slice(209, 210)
LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE = slice(210, 212)
LEGACY_ACTOR_OBSERVATION_WIDTH = 101
H4_ACTOR_OBSERVATION_WIDTH = 116
H4_OBSERVATION_PHYSICAL_COMMAND_SLICE = slice(101, 104)
H4_OBSERVATION_SLIP_SPEED_SLICE = slice(114, 116)
FORCE_CONTACT_ON_NORMALIZED = 0.010
FORCE_CONTACT_OFF_NORMALIZED = 0.005
STRICT_SLIP_RMS_M_S = 0.015
STRICT_SLIP_TAIL_M_S = 0.030
STRICT_STANCE_SLIP_BUDGET_M = 0.020
SUPPORT_EMA_HORIZON_SECONDS = 1.5
STRICT_CONTACT_DUTY_IMBALANCE = 0.10
STRICT_FORWARD_CROSS_DRIFT_M_S = 0.012
STRICT_FORWARD_YAW_RATE_RAD_S = 0.050
STRICT_FORWARD_HEADING_DRIFT_RAD = 0.150
STRICT_REVERSE_CROSS_DRIFT_M_S = 0.010
STRICT_REVERSE_YAW_RATE_RAD_S = 0.050
STRICT_REVERSE_HEADING_DRIFT_RAD = 0.150
STRICT_REVERSE_SPEED_RATIO_LOWER = 0.75
STRICT_REVERSE_SPEED_RATIO_UPPER = 1.25
REVERSE_LEFT_PHASE_INDICES = (26, 0, 1)
REVERSE_RIGHT_MID_PHASE_INDICES = (10, 11, 12, 13, 14, 16)
REVERSE_RIGHT_LATE_PHASE_INDICES = (18, 20, 21, 22)
DEFAULT_ALTERNATION_TARGET_SECONDS = 0.30
DEFAULT_ALTERNATION_SIGMA_SECONDS = 0.15
H4_REVERSE_EXACT_ENDPOINT_PROBABILITY = 0.60
H4_REVERSE_STAND_PROBABILITY = 0.10
H4_REVERSE_LOCAL_ANCHOR_PROBABILITY = 0.20
H4_REVERSE_TRANSITION_PROBABILITY = 0.10
H4_REVERSE_LOCAL_ANCHORS_M_S = (-0.06, -0.04)
H4_REVERSE_PRIMARY_ANCHOR_M_S = -0.05
H4_REVERSE_TRANSITION_BAND_M_S = (-0.04, -0.025)
H4_REVERSE_V2_EXACT_ENDPOINT_PROBABILITY = 0.75
H4_REVERSE_V2_STAND_PROBABILITY = 0.05
H4_REVERSE_V2_LOCAL_ANCHOR_PROBABILITY = 0.15
H4_REVERSE_V2_TRANSITION_PROBABILITY = 0.05
H4_FORWARD_EXACT_ENDPOINT_PROBABILITY = 0.60
H4_FORWARD_STAND_PROBABILITY = 0.10
H4_FORWARD_LOCAL_ANCHOR_PROBABILITY = 0.20
H4_FORWARD_TRANSITION_PROBABILITY = 0.10
H4_FORWARD_LOCAL_ANCHORS_M_S = (0.04, 0.06)
H4_FORWARD_PRIMARY_ANCHOR_M_S = 0.05
H4_FORWARD_TRANSITION_BAND_M_S = (0.025, 0.04)
H4_FORWARD_V2_EXACT_ENDPOINT_PROBABILITY = 0.70
H4_FORWARD_V2_STAND_PROBABILITY = 0.05
H4_FORWARD_V2_LOCAL_ANCHOR_PROBABILITY = 0.20
H4_FORWARD_V2_TRANSITION_PROBABILITY = 0.05
STRICT_TOTAL_NORMAL_FORCE_LOWER_NORMALIZED = 0.80
STRICT_TOTAL_NORMAL_FORCE_UPPER_NORMALIZED = 1.20
STRICT_TOTAL_NORMAL_FORCE_BAND_WIDTH_NORMALIZED = 0.20
STRICT_TOTAL_NORMAL_FORCE_TAIL_NORMALIZED = 3.0
CONTACT_PULSE_MINIMUM_RUN_TICKS = 2
V4_PHYSICS_SUBSTEP_DT_S = 0.002
V4_CONTROL_SUBSTEP_COUNT = 10
V4_CONTACT_PERSISTENCE_SECONDS = 0.040
V4_CONTACT_PERSISTENCE_INTERVALS = 20
FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID = (
    "CONTACT_ABORT_TYPE_SEPARATION_ISLAND_ONLY"
)
REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID = (
    "ABSOLUTE_FULL_LEG_TARGETS_WITH_TEACHER_TIMING_ONLY"
)
REVERSE_ITERATION_V6_DIRECTIONAL_SPAN_FRACTION = 0.9
REVERSE_ITERATION_V6_BASE_ACTION_SPAN_RAD = 0.25
H5_TARGET_SPACE_REWARD_CONTRACT_ID = "OPEN_DUCK_MINI_H5_TARGET_REWARD_V2"
H5_SIGNED_PROGRESS_SCALE = 4.0
H5_TRACKING_SIGMA = (0.05, 0.05, 0.20)


def _require_vector_shape(value: Any, width: int, label: str, *, xp: Any) -> Any:
    array = xp.asarray(value)
    if array.shape != (width,):
        raise ValueError(f"{label} must have shape ({width},), got {array.shape}")
    return array


def contract_target_vectors(*, xp: Any = np) -> tuple[Any, Any, Any]:
    """Return physical lower, physical upper, and SAFE_INIT in actuator order."""

    lower: list[float] = []
    upper: list[float] = []
    initial: list[float] = []
    for name in ACTUATOR_JOINT_ORDER:
        initial.append(float(SAFE_INIT_POS[name]))
        if name in HEAD_JOINTS:
            lower.append(0.0)
            upper.append(0.0)
        else:
            bounds = SAFE_JOINT_LIMITS[name]
            lower.append(float(bounds[0]))
            upper.append(float(bounds[1]))
    return xp.asarray(lower), xp.asarray(upper), xp.asarray(initial)


def contract_reset_noise_vector(*, xp: Any = np) -> Any:
    """Return the contract qpos-noise amplitude for all fourteen actuators."""

    values = []
    for name in ACTUATOR_JOINT_ORDER:
        if name in HEAD_JOINTS:
            values.append(0.0)
        elif name.endswith("_knee"):
            values.append(0.05)
        elif name.endswith("_ankle"):
            values.append(0.08)
        else:
            values.append(0.03)
    return xp.asarray(values)


def project_reset_qpos(
    unit_noise: Any,
    *,
    noise_multiplier: float,
    xp: Any = np,
) -> Any:
    """Build a contract reset without teleporting SAFE_INIT into target bounds.

    ``noise_multiplier == 0`` preserves exact SAFE_INIT.  Positive noise uses
    the reset-only 0.005 rad physical margin and always keeps head qpos zero.
    ``unit_noise`` is expected in ``[-1, 1]``; clipping it keeps this function
    fail-safe when a caller supplies a wider distribution.
    """

    unit = _require_vector_shape(unit_noise, ACTION_COUNT, "unit_noise", xp=xp)
    multiplier = float(noise_multiplier)
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("noise_multiplier must be finite and non-negative")
    lower, upper, initial = contract_target_vectors(xp=xp)
    noise = xp.clip(unit, -1.0, 1.0) * contract_reset_noise_vector(xp=xp)
    candidate = initial + multiplier * noise
    margin = RESET_NOISE_MARGIN_RAD if multiplier > 0.0 else 0.0
    leg_mask = xp.asarray(
        [index in LEG_ACTION_INDICES for index in range(ACTION_COUNT)]
    )
    reset = xp.where(
        leg_mask,
        xp.clip(candidate, lower + margin, upper - margin),
        0.0,
    )
    return reset


def margin_clip_targets(desired_targets: Any, *, xp: Any = np) -> Any:
    """Clamp desired leg targets to the runtime margin and lock the head."""

    desired = _require_vector_shape(
        desired_targets, ACTION_COUNT, "desired_targets", xp=xp
    )
    lower, upper, _ = contract_target_vectors(xp=xp)
    leg_mask = xp.asarray(
        [index in LEG_ACTION_INDICES for index in range(ACTION_COUNT)]
    )
    margin_lower = lower + LEG_TARGET_MARGIN_RAD
    margin_upper = upper - LEG_TARGET_MARGIN_RAD
    return xp.where(
        leg_mask,
        xp.clip(desired, margin_lower, margin_upper),
        0.0,
    )


def final_target_guard_step(
    desired_targets: Any,
    previous_applied_targets: Any,
    *,
    xp: Any = np,
) -> Any:
    """Apply the exact runtime final target guard for one control tick.

    The function is side-effect free and JAX compatible.  The previous vector
    may be physical-SAFE but outside the steady-state inward margin; this is
    required for the left-knee SAFE_INIT startup transition.
    """

    previous = _require_vector_shape(
        previous_applied_targets,
        ACTION_COUNT,
        "previous_applied_targets",
        xp=xp,
    )
    lower, upper, _ = contract_target_vectors(xp=xp)
    leg_mask = xp.asarray(
        [index in LEG_ACTION_INDICES for index in range(ACTION_COUNT)]
    )
    margin_clipped = margin_clip_targets(desired_targets, xp=xp)
    delta = xp.clip(
        margin_clipped - previous,
        -MAX_TARGET_DELTA_PER_TICK_RAD,
        MAX_TARGET_DELTA_PER_TICK_RAD,
    )
    applied = xp.where(leg_mask, previous + delta, 0.0)
    return xp.where(leg_mask, xp.clip(applied, lower, upper), 0.0)


class ForwardV6ContactAbortRoutingTelemetry(NamedTuple):
    """Exact v6 reward/diagnostic split for the two abort event types."""

    reward_loss: Any
    island_loss: Any
    off_gap_diagnostic_loss: Any
    off_gap_reward_contribution: Any
    routing_exact: Any


def forward_iteration_v6_contact_abort_island_only_reward_loss(
    island_loss: Any,
    off_gap_loss: Any,
    *,
    xp: Any = np,
) -> Any:
    """Return the v6 contact-abort reward loss, using islands only.

    Both inputs are retained as exact two-foot values.  ``off_gap_loss`` is
    deliberately routed through a same-dtype zero expression so it remains a
    live diagnostic input without contributing to the optimized reward.
    """

    island = _require_vector_shape(island_loss, 2, "island_loss", xp=xp)
    off_gap = _require_vector_shape(off_gap_loss, 2, "off_gap_loss", xp=xp)
    off_gap_reward_contribution = xp.sum(xp.zeros_like(off_gap))
    return xp.sum(island) + off_gap_reward_contribution


def forward_iteration_v6_contact_abort_island_only_telemetry(
    island_loss: Any,
    off_gap_loss: Any,
    *,
    xp: Any = np,
) -> ForwardV6ContactAbortRoutingTelemetry:
    """Return the qualifying v6 routing telemetry without changing state."""

    island = _require_vector_shape(island_loss, 2, "island_loss", xp=xp)
    off_gap = _require_vector_shape(off_gap_loss, 2, "off_gap_loss", xp=xp)
    island_total = xp.sum(island)
    off_gap_total = xp.sum(off_gap)
    off_gap_reward_contribution = xp.sum(xp.zeros_like(off_gap))
    reward_loss = forward_iteration_v6_contact_abort_island_only_reward_loss(
        island, off_gap, xp=xp
    )
    routing_exact = (
        (reward_loss == island_total)
        & (off_gap_reward_contribution == 0)
    )
    return ForwardV6ContactAbortRoutingTelemetry(
        reward_loss,
        island_total,
        off_gap_total,
        off_gap_reward_contribution,
        routing_exact,
    )


class ReverseV6AbsoluteFullLegTargetTelemetry(NamedTuple):
    """Pure telemetry for the calibrated v6 absolute-action decoder."""

    targets: Any
    bounded_action: Any
    action_clip_count: Any
    leg_count: Any
    head_zero_exact: Any
    all_finite: Any


class ReverseV6AbsoluteFullLegWiringAudit(NamedTuple):
    """Independent parity audit for the target actually selected by the hook."""

    rederived_targets: Any
    teacher_target_contribution: Any
    exact: Any
    max_abs_error: Any
    teacher_target_contribution_zero_exact: Any


class ReverseV6StructuralCountInvariants(NamedTuple):
    """Exact decoder coverage and actual guarded call-site count checks."""

    decoder_leg_count_exact: Any
    precomposer_call_count_exact: Any
    final_guard_call_count_exact: Any
    violation: Any


def reverse_iteration_v6_absolute_full_leg_targets(
    action: Any,
    safe_init: Any | None = None,
    safe_lower: Any | None = None,
    safe_upper: Any | None = None,
    *,
    xp: Any = np,
) -> Any:
    """Decode all ten leg actions as official calibrated absolute targets.

    The decoder is the official bounded linear-plus-quintic branch: each
    direction receives 90% of its physical SAFE span around ``SAFE_INIT``;
    the first ``min(0.25 rad, span)`` is linear and the remaining reach is
    introduced by the fifth-power term.  This is the raw absolute proposal.
    The 0.050 rad inward margin and the 0.040 rad/tick guard remain separate,
    exactly-once stages in :func:`make_h4_aligned_environment_class`.
    """

    policy_action = _require_vector_shape(action, ACTION_COUNT, "action", xp=xp)
    contract_lower, contract_upper, contract_init = contract_target_vectors(xp=xp)
    initial = _require_vector_shape(
        contract_init if safe_init is None else safe_init,
        ACTION_COUNT,
        "safe_init",
        xp=xp,
    )
    lower = _require_vector_shape(
        contract_lower if safe_lower is None else safe_lower,
        ACTION_COUNT,
        "safe_lower",
        xp=xp,
    )
    upper = _require_vector_shape(
        contract_upper if safe_upper is None else safe_upper,
        ACTION_COUNT,
        "safe_upper",
        xp=xp,
    )
    if xp is np:
        numeric_action = np.asarray(policy_action)
        numeric_initial = np.asarray(initial)
        numeric_lower = np.asarray(lower)
        numeric_upper = np.asarray(upper)
        leg_indices = np.asarray(LEG_ACTION_INDICES)
        if not all(
            np.all(np.isfinite(values))
            for values in (
                numeric_action,
                numeric_initial,
                numeric_lower,
                numeric_upper,
            )
        ):
            raise ValueError("reverse v6 decoder inputs must be finite")
        if np.any(numeric_upper[leg_indices] <= numeric_lower[leg_indices]):
            raise ValueError("reverse v6 leg SAFE bounds must have positive span")
        if np.any(numeric_initial[leg_indices] < numeric_lower[leg_indices]) or np.any(
            numeric_initial[leg_indices] > numeric_upper[leg_indices]
        ):
            raise ValueError("reverse v6 SAFE_INIT must be inside physical bounds")

    bounded = xp.clip(policy_action, -1.0, 1.0)
    positive_span = REVERSE_ITERATION_V6_DIRECTIONAL_SPAN_FRACTION * (
        upper - initial
    )
    negative_span = REVERSE_ITERATION_V6_DIRECTIONAL_SPAN_FRACTION * (
        initial - lower
    )
    directional_span = xp.where(bounded >= 0.0, positive_span, negative_span)
    base_span = xp.minimum(
        xp.asarray(
            REVERSE_ITERATION_V6_BASE_ACTION_SPAN_RAD,
            dtype=policy_action.dtype,
        ),
        directional_span,
    )
    magnitude = xp.abs(bounded)
    target_magnitude = (
        base_span * magnitude
        + (directional_span - base_span) * magnitude**5
    )
    decoded = initial + xp.sign(bounded) * target_magnitude
    leg_mask = xp.asarray(
        [index in LEG_ACTION_INDICES for index in range(ACTION_COUNT)]
    )
    return xp.where(leg_mask, decoded, xp.zeros_like(decoded))


def reverse_iteration_v6_absolute_full_leg_target_telemetry(
    action: Any,
    safe_init: Any | None = None,
    safe_lower: Any | None = None,
    safe_upper: Any | None = None,
    *,
    xp: Any = np,
) -> ReverseV6AbsoluteFullLegTargetTelemetry:
    """Return decoder output plus exact, stable qualifying observables."""

    policy_action = _require_vector_shape(action, ACTION_COUNT, "action", xp=xp)
    targets = reverse_iteration_v6_absolute_full_leg_targets(
        policy_action,
        safe_init,
        safe_lower,
        safe_upper,
        xp=xp,
    )
    bounded = xp.clip(policy_action, -1.0, 1.0)
    leg_mask = xp.asarray(
        [index in LEG_ACTION_INDICES for index in range(ACTION_COUNT)]
    )
    leg_indices = xp.asarray(LEG_ACTION_INDICES)
    head_indices = xp.asarray(HEAD_ACTION_INDICES)
    return ReverseV6AbsoluteFullLegTargetTelemetry(
        targets,
        bounded,
        xp.sum((policy_action[leg_indices] != bounded[leg_indices]).astype(xp.int32)),
        xp.sum(leg_mask.astype(xp.int32)),
        xp.all(targets[head_indices] == 0),
        xp.all(xp.isfinite(targets)),
    )


def reverse_iteration_v6_absolute_full_leg_target_wiring_audit(
    selected_raw_targets: Any,
    action: Any,
    safe_init: Any | None = None,
    safe_lower: Any | None = None,
    safe_upper: Any | None = None,
    *,
    xp: Any = np,
) -> ReverseV6AbsoluteFullLegWiringAudit:
    """Compare the actual hook selection with an independent decoder replay.

    Any source proposal, teacher target, or residual blend that leaks into the
    selected raw target appears directly as ``teacher_target_contribution`` and
    makes both exactness checks fail.
    """

    selected = _require_vector_shape(
        selected_raw_targets,
        ACTION_COUNT,
        "selected_raw_targets",
        xp=xp,
    )
    rederived = reverse_iteration_v6_absolute_full_leg_targets(
        action,
        safe_init,
        safe_lower,
        safe_upper,
        xp=xp,
    )
    contribution = selected - rederived
    absolute_error = xp.abs(contribution)
    return ReverseV6AbsoluteFullLegWiringAudit(
        rederived,
        contribution,
        xp.all(selected == rederived),
        xp.max(absolute_error),
        xp.all(contribution == 0),
    )


def reverse_iteration_v6_structural_count_invariants(
    decoder_leg_count: Any,
    precomposer_call_count: Any,
    final_guard_call_count: Any,
    *,
    xp: Any = np,
) -> ReverseV6StructuralCountInvariants:
    """Return exact v6 coverage/call booleans and their combined violation."""

    leg_count = xp.asarray(decoder_leg_count, dtype=xp.int32)
    precomposer_count = xp.asarray(precomposer_call_count, dtype=xp.int32)
    final_guard_count = xp.asarray(final_guard_call_count, dtype=xp.int32)
    if any(
        value.shape != ()
        for value in (leg_count, precomposer_count, final_guard_count)
    ):
        raise ValueError("reverse v6 structural counts must be scalars")
    leg_count_exact = leg_count == len(LEG_ACTION_INDICES)
    precomposer_count_exact = precomposer_count == 1
    final_guard_count_exact = final_guard_count == 1
    violation = (
        ~leg_count_exact
        | ~precomposer_count_exact
        | ~final_guard_count_exact
    )
    return ReverseV6StructuralCountInvariants(
        leg_count_exact,
        precomposer_count_exact,
        final_guard_count_exact,
        violation,
    )


def reverse_iteration_v6_teacher_timing_only_reference(
    source_reference: Any,
    reference_leg_indices: Any,
    safe_init: Any | None = None,
    *,
    xp: Any = np,
) -> Any:
    """Preserve teacher schedule channels but neutralize every target slot.

    No teacher target vector is accepted by this API.  The ten reference-joint
    positions become the corresponding SAFE_INIT constants, while phase and
    contact schedule channels in ``source_reference`` remain byte-for-byte
    unchanged.
    """

    reference = xp.asarray(source_reference)
    if reference.ndim != 1:
        raise ValueError("source_reference must be one vector")
    indices = xp.asarray(reference_leg_indices)
    if indices.shape != (len(LEG_ACTION_INDICES),):
        raise ValueError("reference_leg_indices must identify exactly ten slots")
    _, _, contract_init = contract_target_vectors(xp=xp)
    initial = _require_vector_shape(
        contract_init if safe_init is None else safe_init,
        ACTION_COUNT,
        "safe_init",
        xp=xp,
    )
    neutral_leg_targets = initial[xp.asarray(LEG_ACTION_INDICES)]
    if xp is np:
        numeric_indices = np.asarray(indices)
        if (
            not np.issubdtype(numeric_indices.dtype, np.integer)
            or len(np.unique(numeric_indices)) != len(LEG_ACTION_INDICES)
            or np.any(numeric_indices < 0)
            or np.any(numeric_indices >= reference.shape[0])
        ):
            raise ValueError("reference leg indices must be unique in-range integers")
        neutral = np.asarray(reference).copy()
        neutral[numeric_indices] = np.asarray(neutral_leg_targets)
        return neutral
    return reference.at[indices].set(neutral_leg_targets)


@dataclass
class NumpyTargetGuard:
    """Small stateful NumPy reference used for parity and startup audits."""

    previous_targets: np.ndarray
    steps_since_reset: int = 0

    @classmethod
    def from_reset(cls, reset_targets: Sequence[float]) -> "NumpyTargetGuard":
        values = np.asarray(reset_targets, dtype=np.float64)
        if values.shape != (ACTION_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError("reset_targets must be one finite 14-axis vector")
        lower, upper, _ = contract_target_vectors()
        guarded = np.clip(values, lower, upper)
        guarded[np.asarray(HEAD_ACTION_INDICES)] = 0.0
        return cls(guarded, 0)

    def step(self, desired_targets: Sequence[float]) -> np.ndarray:
        applied = np.asarray(
            final_target_guard_step(
                desired_targets, self.previous_targets, xp=np
            ),
            dtype=np.float64,
        )
        self.previous_targets = applied
        self.steps_since_reset += 1
        return applied.copy()

    def control_first_startup(
        self, first_command_policy_targets: Sequence[float]
    ) -> np.ndarray:
        if self.steps_since_reset != 0:
            raise RuntimeError("control-first startup must be the first guard step")
        return self.step(first_command_policy_targets)


def force_schmitt_contacts(
    normalized_normal_force: Any,
    previous_contact: Any,
    *,
    xp: Any = np,
) -> Any:
    """Apply the evaluator's 1.0%/0.5% body-weight force Schmitt trigger."""

    force = _require_vector_shape(
        normalized_normal_force, 2, "normalized_normal_force", xp=xp
    )
    previous = _require_vector_shape(previous_contact, 2, "previous_contact", xp=xp)
    return (force >= FORCE_CONTACT_ON_NORMALIZED) | (
        previous.astype(bool) & (force >= FORCE_CONTACT_OFF_NORMALIZED)
    )


class V4ContactPersistenceState(NamedTuple):
    """Raw Schmitt and causally-qualified contact state for both feet."""

    raw_contact: Any
    qualified_contact: Any
    pending_active: Any
    pending_target: Any
    pending_intervals: Any


class V4ContactPersistenceUpdate(NamedTuple):
    state: V4ContactPersistenceState
    confirmed_transition: Any
    touchdown_event: Any
    liftoff_event: Any
    aborted_contact_island_event: Any
    aborted_off_gap_event: Any
    aborted_span_intervals: Any
    aborted_loss: Any


class V4ContactTelemetryState(NamedTuple):
    """Persistent v4 state shared by consecutive 2 ms observations."""

    persistence: V4ContactPersistenceState
    touchdown_counts: Any
    last_touchdown_foot: Any
    intervals_since_touchdown: Any


class V4ContactTelemetryUpdate(NamedTuple):
    state: V4ContactTelemetryState
    confirmed_transition: Any
    touchdown_event: Any
    liftoff_event: Any
    aborted_contact_island_event: Any
    aborted_off_gap_event: Any
    aborted_span_intervals: Any
    aborted_loss: Any
    alternation_event: Any
    alternation_quality: Any


class V4SubstepTelemetrySummary(NamedTuple):
    """Control-tick aggregate over every measurement-aligned substep."""

    state: V4ContactTelemetryState
    confirmed_transition_count: Any
    touchdown_event_count: Any
    liftoff_event_count: Any
    aborted_contact_island_count: Any
    aborted_off_gap_count: Any
    aborted_contact_island_loss_sum: Any
    aborted_off_gap_loss_sum: Any
    alternation_event_count: Any
    alternation_quality_sum: Any
    normalized_force_finite: Any


class V4SubstepContactQualityTrajectory(NamedTuple):
    """Exact post-physics force/slip samples retained for one control tick."""

    time_s: Any  # (10,)
    normalized_normal_force: Any  # (10, 2)
    tangential_speed_m_s: Any  # (10, 2)


class V4SavedDynamicState(NamedTuple):
    """The only per-substep trajectory leaves retained for v4 telemetry."""

    qpos: Any
    qvel: Any
    act: Any
    ctrl: Any
    time: Any
    qacc_warmstart: Any


class V4TrajectoryParity(NamedTuple):
    exact: Any
    max_abs_error: Any
    leaf_count: int


class V4SourceSemanticPreflight(NamedTuple):
    """Qualifying dynamic6 parity plus nonqualifying derived diagnostics."""

    dynamic6_exact: Any
    dynamic6_max_abs_error: Any
    dynamic6_field_count: int
    derived_cfrc_int_exact: Any
    derived_cfrc_int_max_abs_error: Any
    derived_cfrc_ext_exact: Any
    derived_cfrc_ext_max_abs_error: Any


def v4_dynamic_field_counts_exact(
    dynamic_field_count: Any,
    saved_dynamic_field_count: Any,
    *,
    xp: Any = np,
) -> tuple[Any, Any]:
    """Return device-native exact topology predicates for dynamic6.

    These booleans are emitted per environment step and therefore aggregate
    through PPO with exactly the same reduction as ``episode/length``.  The
    integer count metrics remain diagnostics; qualification never reconstructs
    their totals with an independent host-side multiplication.
    """

    required = len(V4SavedDynamicState._fields)
    return (
        xp.asarray(dynamic_field_count) == required,
        xp.asarray(saved_dynamic_field_count) == required,
    )


def _raise_on_v4_single_authority_violation(
    exact: Any,
    max_abs_error: Any,
    leaf_count: Any,
) -> None:
    """Host callback used as a compiled CPU/CUDA fail-closed assertion."""

    exact_value = bool(np.asarray(exact).item())
    error_value = float(np.asarray(max_abs_error).item())
    leaf_value = int(np.asarray(leaf_count).item())
    if (
        not exact_value
        or error_value != 0.0
        or leaf_value != len(V4SavedDynamicState._fields)
    ):
        raise RuntimeError(
            "forward-v4 single-authority invariant failure: "
            f"exact={exact_value}, max_abs_error={error_value!r}, "
            f"field_count={leaf_value}"
        )


def require_v4_single_authority_invariants(
    parity: V4TrajectoryParity,
    *,
    xp: Any = np,
) -> None:
    """Synchronous reference assertion for exact, nonempty source parity."""

    exact = xp.asarray(parity.exact)
    max_abs_error = xp.asarray(parity.max_abs_error)
    leaf_count = xp.asarray(parity.leaf_count, dtype=xp.int32)
    if exact.shape != () or max_abs_error.shape != () or leaf_count.shape != ():
        raise ValueError("v4 trajectory parity fields must be scalars")
    _raise_on_v4_single_authority_violation(
        exact, max_abs_error, leaf_count
    )


def make_v4_compiled_single_authority_assertion(
    jax: Any,
    xp: Any,
    *,
    failure_callback: Callable[[Any, Any, Any], None] = (
        _raise_on_v4_single_authority_violation
    ),
) -> Callable[[Any, Any, Any], Any]:
    """Build one conditional, batch-aggregating compiled parity assertion.

    A plain ``lax.cond`` inside an environment step is unsafe under ``vmap``:
    batching rewrites it to a select and can execute effectful callbacks for
    every environment.  This custom-vmap rule first reduces the full batch to
    one exact/error/leaf triplet, then executes one scalar ``lax.cond``.  The
    success branch has no callback at all; the failure branch schedules one
    host callback which raises and terminates the compiled CPU/CUDA
    step.  Thus the 1,250-environment PPO path has zero success callbacks and
    at most one callback on a failing batched step.  The callback is unordered
    because JAX 0.5.3 custom-vmap rejects ordered debug effects; the token is
    retained in environment info and consumers synchronize the compiled step,
    so a raised callback exception still fails the step before it can proceed.
    """

    if not callable(failure_callback):
        raise ValueError("failure_callback must be callable")
    if not hasattr(jax, "custom_batching"):
        raise ValueError("jax.custom_batching is required for v4 parity")

    def compiled_assertion(exact, max_abs_error, leaf_count):
        exact_value = xp.asarray(exact).astype(bool)
        error_value = xp.asarray(max_abs_error)
        leaf_value = xp.asarray(leaf_count, dtype=xp.int32)
        violation = (
            ~exact_value
            | (error_value != 0.0)
            | (leaf_value != len(V4SavedDynamicState._fields))
        )

        def fail(operands):
            jax.debug.callback(
                failure_callback,
                *operands,
                ordered=False,
            )
            return xp.zeros((), dtype=xp.int32)

        def pass_without_callback(_operands):
            return xp.zeros((), dtype=xp.int32)

        return jax.lax.cond(
            violation,
            fail,
            pass_without_callback,
            (exact_value, error_value, leaf_value),
        )

    assertion = jax.custom_batching.custom_vmap(compiled_assertion)

    @assertion.def_vmap
    def assertion_vmap(axis_size, in_batched, exact, max_abs_error, leaf_count):
        exact_batched, error_batched, leaf_batched = in_batched
        batch_exact = xp.all(exact, axis=0) if exact_batched else exact
        if error_batched:
            finite_error = xp.where(
                xp.isfinite(max_abs_error),
                xp.abs(max_abs_error),
                xp.asarray(xp.inf, dtype=max_abs_error.dtype),
            )
            batch_error = xp.max(finite_error, axis=0)
        else:
            batch_error = max_abs_error
        batch_leaf_count = (
            xp.where(
                xp.all(
                    leaf_count == len(V4SavedDynamicState._fields), axis=0
                ),
                xp.asarray(len(V4SavedDynamicState._fields), dtype=xp.int32),
                xp.zeros((), dtype=xp.int32),
            )
            if leaf_batched
            else leaf_count
        )
        token = compiled_assertion(
            batch_exact, batch_error, batch_leaf_count
        )
        return xp.broadcast_to(token, (axis_size,)), True

    return assertion


def make_v6_compiled_invariant_assertion(
    jax: Any,
    xp: Any,
    *,
    label: str,
    failure_callback: Callable[[Any], None] | None = None,
) -> Callable[[Any], Any]:
    """Build one fail-closed, batch-aggregating v6 invariant assertion."""

    family_label = str(label).strip()
    if not family_label:
        raise ValueError("v6 assertion label must be non-empty")
    if not hasattr(jax, "custom_batching"):
        raise ValueError("jax.custom_batching is required for v6 assertions")

    if failure_callback is None:

        def raise_violation(violation: Any) -> None:
            if bool(np.asarray(violation).item()):
                raise RuntimeError(f"{family_label} invariant failure")

        callback = raise_violation
    else:
        if not callable(failure_callback):
            raise ValueError("failure_callback must be callable")
        callback = failure_callback

    def compiled_assertion(violation):
        violation_value = xp.asarray(violation).astype(bool)

        def fail(value):
            jax.debug.callback(callback, value, ordered=False)
            return xp.zeros((), dtype=xp.int32)

        def pass_without_callback(_value):
            return xp.zeros((), dtype=xp.int32)

        return jax.lax.cond(
            violation_value,
            fail,
            pass_without_callback,
            violation_value,
        )

    assertion = jax.custom_batching.custom_vmap(compiled_assertion)

    @assertion.def_vmap
    def assertion_vmap(axis_size, in_batched, violation):
        (violation_batched,) = in_batched
        batch_violation = (
            xp.any(violation, axis=0) if violation_batched else violation
        )
        token = compiled_assertion(batch_violation)
        return xp.broadcast_to(token, (axis_size,)), True

    return assertion


def initialize_v4_contact_telemetry(
    normalized_normal_force: Any,
    *,
    xp: Any = np,
) -> V4ContactTelemetryState:
    """Measure the reset baseline without synthesizing a touchdown.

    Reset contact uses the exact ON threshold.  Both the raw Schmitt state and
    the qualified state start from that same measurement, so an already-loaded
    foot is a baseline condition rather than a phantom transition.
    """

    force = _require_vector_shape(
        normalized_normal_force, 2, "normalized_normal_force", xp=xp
    )
    if xp is np and not np.all(np.isfinite(np.asarray(force))):
        raise ValueError("normalized_normal_force must be finite")
    raw = force_schmitt_contacts(force, xp.zeros(2, dtype=bool), xp=xp)
    persistence = V4ContactPersistenceState(
        raw,
        raw,
        xp.zeros(2, dtype=bool),
        raw,
        xp.zeros(2, dtype=xp.int32),
    )
    return V4ContactTelemetryState(
        persistence,
        xp.zeros(2, dtype=xp.int32),
        xp.asarray(-1, dtype=xp.int32),
        xp.zeros((), dtype=xp.int32),
    )


def v4_aborted_transition_loss(
    span_intervals: Any,
    *,
    xp: Any = np,
) -> Any:
    """Return ``((40 ms - span) / 40 ms)^2`` on integer 2 ms spans."""

    intervals = xp.asarray(span_intervals)
    if xp is np:
        numeric = np.asarray(intervals)
        if (
            not np.all(np.isfinite(numeric))
            or np.any(numeric < 0)
            or not np.array_equal(numeric, np.floor(numeric))
        ):
            raise ValueError("span_intervals must be finite non-negative integers")
    remaining = xp.maximum(V4_CONTACT_PERSISTENCE_INTERVALS - intervals, 0)
    return xp.square(
        remaining.astype(xp.float32)
        / float(V4_CONTACT_PERSISTENCE_INTERVALS)
    )


def _validate_numpy_v4_persistence_state(
    state: V4ContactPersistenceState,
) -> None:
    raw = np.asarray(state.raw_contact)
    qualified = np.asarray(state.qualified_contact)
    active = np.asarray(state.pending_active)
    target = np.asarray(state.pending_target)
    intervals = np.asarray(state.pending_intervals)
    for label, value in (
        ("raw_contact", raw),
        ("qualified_contact", qualified),
        ("pending_active", active),
        ("pending_target", target),
        ("pending_intervals", intervals),
    ):
        if value.shape != (2,):
            raise ValueError(f"{label} must have shape (2,), got {value.shape}")
    if (
        not np.all(np.isfinite(intervals))
        or np.any(intervals < 0)
        or not np.array_equal(intervals, np.floor(intervals))
        or np.any(intervals >= V4_CONTACT_PERSISTENCE_INTERVALS)
    ):
        raise ValueError("pending intervals must be integers in [0, 20)")
    if np.any((~active.astype(bool)) & (intervals != 0)):
        raise ValueError("inactive pending transitions must have zero intervals")
    if np.any(active.astype(bool) & (target.astype(bool) == qualified.astype(bool))):
        raise ValueError("active pending target must oppose qualified contact")


def update_v4_contact_persistence(
    previous_state: V4ContactPersistenceState,
    normalized_normal_force: Any,
    *,
    xp: Any = np,
) -> V4ContactPersistenceUpdate:
    """Advance exact Schmitt plus 40 ms causal persistence by one 2 ms sample.

    The first sample opposing the qualified state opens a pending transition at
    elapsed interval zero.  Confirmation therefore occurs after twenty further
    2 ms intervals (twenty-one point observations including the start).  A
    return to the qualified state before then is an aborted contact island or
    off-gap and receives the normalized span loss.
    """

    if xp is np:
        _validate_numpy_v4_persistence_state(previous_state)
    force = _require_vector_shape(
        normalized_normal_force, 2, "normalized_normal_force", xp=xp
    )
    if xp is np and not np.all(np.isfinite(np.asarray(force))):
        raise ValueError("normalized_normal_force must be finite")
    old_raw = _require_vector_shape(
        previous_state.raw_contact, 2, "raw_contact", xp=xp
    ).astype(bool)
    qualified = _require_vector_shape(
        previous_state.qualified_contact, 2, "qualified_contact", xp=xp
    ).astype(bool)
    pending_active = _require_vector_shape(
        previous_state.pending_active, 2, "pending_active", xp=xp
    ).astype(bool)
    pending_target = _require_vector_shape(
        previous_state.pending_target, 2, "pending_target", xp=xp
    ).astype(bool)
    pending_intervals = _require_vector_shape(
        previous_state.pending_intervals, 2, "pending_intervals", xp=xp
    ).astype(xp.int32)

    raw = force_schmitt_contacts(force, old_raw, xp=xp)
    differs = raw != qualified
    starts = ~pending_active & differs
    continues = pending_active & differs & (raw == pending_target)
    reverts = pending_active & ~differs
    elapsed_intervals = pending_intervals + 1
    confirmed = continues & (
        elapsed_intervals >= V4_CONTACT_PERSISTENCE_INTERVALS
    )
    touchdown = confirmed & ~qualified & raw
    liftoff = confirmed & qualified & ~raw
    aborted_island = reverts & ~qualified
    aborted_off_gap = reverts & qualified
    # Pending age measures completed opposite-state intervals.  Therefore a
    # one-point excursion that returns on the next observation has span zero,
    # while stored age ten is the exact 20 ms normalization point.
    aborted_span = xp.where(reverts, pending_intervals, 0).astype(xp.int32)
    aborted_loss = reverts.astype(force.dtype) * v4_aborted_transition_loss(
        aborted_span, xp=xp
    ).astype(force.dtype)

    new_qualified = xp.where(confirmed, raw, qualified)
    still_pending = continues & ~confirmed
    new_pending_active = starts | still_pending
    new_pending_target = xp.where(
        starts,
        raw,
        xp.where(new_pending_active, pending_target, new_qualified),
    )
    new_pending_intervals = xp.where(
        starts,
        0,
        xp.where(still_pending, elapsed_intervals, 0),
    ).astype(xp.int32)
    state = V4ContactPersistenceState(
        raw,
        new_qualified,
        new_pending_active,
        new_pending_target,
        new_pending_intervals,
    )
    return V4ContactPersistenceUpdate(
        state,
        confirmed,
        touchdown,
        liftoff,
        aborted_island,
        aborted_off_gap,
        aborted_span,
        aborted_loss,
    )


def update_v4_contact_telemetry(
    previous_state: V4ContactTelemetryState,
    normalized_normal_force: Any,
    *,
    xp: Any = np,
) -> V4ContactTelemetryUpdate:
    """Advance persistence, confirmed touchdown counts, and alternation."""

    persistence = update_v4_contact_persistence(
        previous_state.persistence, normalized_normal_force, xp=xp
    )
    touchdown_counts = _require_vector_shape(
        previous_state.touchdown_counts, 2, "touchdown_counts", xp=xp
    )
    touchdown_counts = touchdown_counts + persistence.touchdown_event.astype(
        touchdown_counts.dtype
    )

    touchdown = persistence.touchdown_event.astype(bool)
    unique_touchdown = xp.logical_xor(touchdown[0], touchdown[1])
    simultaneous_touchdown = touchdown[0] & touchdown[1]
    current_foot = xp.where(touchdown[0], 0, xp.where(touchdown[1], 1, -1))
    last_foot = xp.asarray(previous_state.last_touchdown_foot).astype(xp.int32)
    intervals = xp.asarray(previous_state.intervals_since_touchdown).astype(
        xp.int32
    )
    initialized = last_foot >= 0
    alternation_event = (
        unique_touchdown & initialized & (current_foot != last_foot)
    )
    elapsed_seconds = (intervals + 1).astype(xp.float32) * V4_PHYSICS_SUBSTEP_DT_S
    alternation_quality = xp.where(
        alternation_event,
        xp.exp(
            -0.5
            * xp.square(
                (elapsed_seconds - DEFAULT_ALTERNATION_TARGET_SECONDS)
                / DEFAULT_ALTERNATION_SIGMA_SECONDS
            )
        ),
        0.0,
    )
    # Simultaneous touchdowns have no causal left/right ordering.  Clear the
    # alternation baseline so neither their arbitrary array order nor the next
    # touchdown can create a phantom reward.
    new_last_foot = xp.where(
        simultaneous_touchdown,
        -1,
        xp.where(unique_touchdown, current_foot, last_foot),
    ).astype(xp.int32)
    new_intervals = xp.where(
        simultaneous_touchdown | unique_touchdown,
        0,
        xp.where(initialized, intervals + 1, 0),
    ).astype(xp.int32)
    state = V4ContactTelemetryState(
        persistence.state,
        touchdown_counts,
        new_last_foot,
        new_intervals,
    )
    return V4ContactTelemetryUpdate(
        state,
        persistence.confirmed_transition,
        persistence.touchdown_event,
        persistence.liftoff_event,
        persistence.aborted_contact_island_event,
        persistence.aborted_off_gap_event,
        persistence.aborted_span_intervals,
        persistence.aborted_loss,
        alternation_event,
        alternation_quality,
    )


def discard_v4_terminal_incomplete(
    state: V4ContactTelemetryState,
    terminal: Any,
    *,
    xp: Any = np,
) -> V4ContactTelemetryState:
    """Right-censor an incomplete terminal transition without a loss/event."""

    is_terminal = xp.asarray(terminal).astype(bool)
    persistence = state.persistence
    cleared = V4ContactPersistenceState(
        persistence.raw_contact,
        persistence.qualified_contact,
        xp.where(is_terminal, xp.zeros(2, dtype=bool), persistence.pending_active),
        xp.where(
            is_terminal,
            persistence.qualified_contact,
            persistence.pending_target,
        ),
        xp.where(
            is_terminal,
            xp.zeros(2, dtype=xp.int32),
            persistence.pending_intervals,
        ),
    )
    return V4ContactTelemetryState(
        cleared,
        state.touchdown_counts,
        state.last_touchdown_foot,
        state.intervals_since_touchdown,
    )


def _empty_v4_substep_summary(
    state: V4ContactTelemetryState,
    *,
    xp: Any,
) -> V4SubstepTelemetrySummary:
    count_dtype = xp.int32
    loss_dtype = xp.float32
    return V4SubstepTelemetrySummary(
        state,
        xp.zeros(2, dtype=count_dtype),
        xp.zeros(2, dtype=count_dtype),
        xp.zeros(2, dtype=count_dtype),
        xp.zeros(2, dtype=count_dtype),
        xp.zeros(2, dtype=count_dtype),
        xp.zeros(2, dtype=loss_dtype),
        xp.zeros(2, dtype=loss_dtype),
        xp.zeros((), dtype=count_dtype),
        xp.zeros((), dtype=loss_dtype),
        xp.asarray(True),
    )


def _accumulate_v4_substep_summary(
    summary: V4SubstepTelemetrySummary,
    update: V4ContactTelemetryUpdate,
    *,
    xp: Any,
) -> V4SubstepTelemetrySummary:
    count_dtype = summary.touchdown_event_count.dtype
    loss_dtype = summary.aborted_contact_island_loss_sum.dtype
    confirmed = update.confirmed_transition.astype(count_dtype)
    touchdown = update.touchdown_event.astype(count_dtype)
    liftoff = update.liftoff_event.astype(count_dtype)
    island = update.aborted_contact_island_event.astype(count_dtype)
    off_gap = update.aborted_off_gap_event.astype(count_dtype)
    return V4SubstepTelemetrySummary(
        update.state,
        summary.confirmed_transition_count + confirmed,
        summary.touchdown_event_count + touchdown,
        summary.liftoff_event_count + liftoff,
        summary.aborted_contact_island_count + island,
        summary.aborted_off_gap_count + off_gap,
        summary.aborted_contact_island_loss_sum
        + island.astype(loss_dtype) * update.aborted_loss.astype(loss_dtype),
        summary.aborted_off_gap_loss_sum
        + off_gap.astype(loss_dtype) * update.aborted_loss.astype(loss_dtype),
        summary.alternation_event_count
        + update.alternation_event.astype(count_dtype),
        summary.alternation_quality_sum
        + update.alternation_quality.astype(loss_dtype),
        summary.normalized_force_finite,
    )


def v4_authoritative_primitive_step(
    model: Any,
    data: Any,
    action: Any,
    *,
    mjx_step: Callable[[Any, Any], Any],
) -> Any:
    """Execute the exact primitive body used by authoritative MJX scan.

    This deliberately performs no nested scan: control replacement immediately
    precedes the same imported ``mjx.step`` primitive used by the source body.
    """

    if not callable(mjx_step):
        raise ValueError("mjx_step must be callable")
    return mjx_step(model, data.replace(ctrl=action))


def save_v4_dynamic_state(data: Any) -> V4SavedDynamicState:
    """Snapshot the six dynamic leaves needed to replay a measured state."""

    try:
        return V4SavedDynamicState(
            data.qpos,
            data.qvel,
            data.act,
            data.ctrl,
            data.time,
            data.qacc_warmstart,
        )
    except AttributeError as exc:
        raise ValueError(
            "v4 dynamic state requires qpos, qvel, act, ctrl, time, and "
            "qacc_warmstart"
        ) from exc


def reconstruct_v4_dynamic_state(
    control_entry_data: Any,
    saved_state: V4SavedDynamicState,
) -> Any:
    """Rebuild one unforwarded sample from immutable control-entry data."""

    if not hasattr(control_entry_data, "replace"):
        raise ValueError("control_entry_data must provide replace()")
    return control_entry_data.replace(
        qpos=saved_state.qpos,
        qvel=saved_state.qvel,
        act=saved_state.act,
        ctrl=saved_state.ctrl,
        time=saved_state.time,
        qacc_warmstart=saved_state.qacc_warmstart,
    )


def _stack_v4_saved_dynamic_states(
    saved_states: Sequence[V4SavedDynamicState],
    *,
    xp: Any,
) -> V4SavedDynamicState:
    if len(saved_states) != V4_CONTROL_SUBSTEP_COUNT:
        raise ValueError("forward v4 requires exactly 10 saved dynamic states")
    return V4SavedDynamicState(
        *(
            xp.stack(tuple(getattr(state, field) for state in saved_states), axis=0)
            for field in V4SavedDynamicState._fields
        )
    )


def _require_v4_saved_dynamic_trajectory(
    saved_states: V4SavedDynamicState,
    *,
    xp: Any,
) -> V4SavedDynamicState:
    arrays = []
    for field in V4SavedDynamicState._fields:
        value = xp.asarray(getattr(saved_states, field))
        if value.ndim < 1 or value.shape[0] != V4_CONTROL_SUBSTEP_COUNT:
            raise ValueError(
                f"saved v4 dynamic field {field} must lead with 10 substeps"
            )
        arrays.append(value)
    return V4SavedDynamicState(*arrays)


def scan_v4_instrumented_physics_trajectory(
    initial_data: Any,
    action: Any,
    *,
    single_physics_step: Callable[[Any, Any], Any],
    n_substeps: int = V4_CONTROL_SUBSTEP_COUNT,
    scan: Callable[..., Any] | None = None,
    xp: Any = np,
) -> tuple[Any, V4SavedDynamicState]:
    """Run the direct physics scan and emit only post-step dynamic6 leaves.

    No forwarding, force measurement, or telemetry update is permitted in
    this scan body.  Its data carry is therefore the same primitive trajectory
    as the authoritative source scan; the additional scan output is the six
    dynamic leaves required by the later measurement replay.
    """

    if int(n_substeps) != V4_CONTROL_SUBSTEP_COUNT:
        raise ValueError("forward v4 requires exactly 10 physics substeps")
    if not callable(single_physics_step):
        raise ValueError("single_physics_step must be callable")

    def body(data: Any, _unused: Any):
        next_data = single_physics_step(data, action)
        return next_data, save_v4_dynamic_state(next_data)

    if scan is None:
        final_data = initial_data
        saved_sequence = []
        for _ in range(V4_CONTROL_SUBSTEP_COUNT):
            final_data, saved = body(final_data, None)
            saved_sequence.append(saved)
        return final_data, _stack_v4_saved_dynamic_states(
            saved_sequence, xp=xp
        )
    # Keep the source wrapper's scan call shape exactly: ``xs=()`` and the
    # static substep count are positional in mujoco_playground.mjx_env.step.
    final_data, saved_states = scan(
        body,
        initial_data,
        (),
        V4_CONTROL_SUBSTEP_COUNT,
    )
    return final_data, _require_v4_saved_dynamic_trajectory(
        saved_states, xp=xp
    )


def scan_v4_saved_state_contact_telemetry(
    control_entry_data: Any,
    saved_states: V4SavedDynamicState,
    initial_state: V4ContactTelemetryState,
    *,
    cohere_measurement_state: Callable[[Any], Any],
    measure_normalized_force: Callable[[Any, Any], Any],
    n_substeps: int = V4_CONTROL_SUBSTEP_COUNT,
    scan: Callable[..., Any] | None = None,
    xp: Any = np,
) -> V4SubstepTelemetrySummary:
    """Replay saved dynamic6 points and measure only after physics completes."""

    if int(n_substeps) != V4_CONTROL_SUBSTEP_COUNT:
        raise ValueError("forward v4 requires exactly 10 telemetry substeps")
    if not callable(cohere_measurement_state) or not callable(
        measure_normalized_force
    ):
        raise ValueError("measurement coherence and force must be callable")
    saved_states = _require_v4_saved_dynamic_trajectory(saved_states, xp=xp)
    initial_summary = _empty_v4_substep_summary(initial_state, xp=xp)

    def body(
        current_summary: V4SubstepTelemetrySummary,
        saved_state: V4SavedDynamicState,
    ):
        replay_data = reconstruct_v4_dynamic_state(
            control_entry_data, saved_state
        )
        coherent_measurement_data = cohere_measurement_state(replay_data)
        normalized_force = measure_normalized_force(
            coherent_measurement_data,
            current_summary.state.persistence.raw_contact,
        )
        normalized_force = _require_vector_shape(
            normalized_force, 2, "v4 telemetry normalized_force", xp=xp
        )
        update = update_v4_contact_telemetry(
            current_summary.state, normalized_force, xp=xp
        )
        next_summary = _accumulate_v4_substep_summary(
            current_summary, update, xp=xp
        )
        next_summary = next_summary._replace(
            normalized_force_finite=(
                current_summary.normalized_force_finite
                & xp.all(xp.isfinite(normalized_force))
            )
        )
        return next_summary, None

    if scan is None:
        summary = initial_summary
        for index in range(V4_CONTROL_SUBSTEP_COUNT):
            saved_state = V4SavedDynamicState(
                *(getattr(saved_states, field)[index] for field in saved_states._fields)
            )
            summary, _ = body(summary, saved_state)
        return summary
    summary, _ = scan(body, initial_summary, saved_states)
    return summary


def scan_v4_saved_state_contact_telemetry_with_quality_trace(
    control_entry_data: Any,
    saved_states: V4SavedDynamicState,
    initial_state: V4ContactTelemetryState,
    *,
    cohere_measurement_state: Callable[[Any], Any],
    measure_force_and_tangential_speed: Callable[[Any, Any], tuple[Any, Any]],
    n_substeps: int = V4_CONTROL_SUBSTEP_COUNT,
    scan: Callable[..., Any] | None = None,
    xp: Any = np,
) -> tuple[V4SubstepTelemetrySummary, V4SubstepContactQualityTrajectory]:
    """Measure existing V4 replay once and retain its force/slip trace.

    This is a collector primitive, not an H5 helper.  It replaces neither the
    authoritative physics scan nor the normal V4 telemetry semantics: every
    saved dynamic6 sample is forwarded exactly once and supplies both the
    telemetry update and the immutable trajectory returned to a later sidecar.
    """

    if int(n_substeps) != V4_CONTROL_SUBSTEP_COUNT:
        raise ValueError("forward v4 requires exactly 10 telemetry substeps")
    if not callable(cohere_measurement_state) or not callable(
        measure_force_and_tangential_speed
    ):
        raise ValueError("measurement coherence and force/slip must be callable")
    saved_states = _require_v4_saved_dynamic_trajectory(saved_states, xp=xp)
    initial_summary = _empty_v4_substep_summary(initial_state, xp=xp)

    def require_trajectory_matrix(value: Any, name: str) -> Any:
        array = xp.asarray(value)
        expected_shape = (V4_CONTROL_SUBSTEP_COUNT, 2)
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
        return array

    def body(
        current_summary: V4SubstepTelemetrySummary,
        saved_state: V4SavedDynamicState,
    ):
        replay_data = reconstruct_v4_dynamic_state(control_entry_data, saved_state)
        coherent_measurement_data = cohere_measurement_state(replay_data)
        normalized_force, tangential_speed = measure_force_and_tangential_speed(
            coherent_measurement_data,
            current_summary.state.persistence.raw_contact,
        )
        normalized_force = _require_vector_shape(
            normalized_force, 2, "v4 telemetry normalized_force", xp=xp
        )
        tangential_speed = _require_vector_shape(
            tangential_speed, 2, "v4 telemetry tangential_speed", xp=xp
        )
        update = update_v4_contact_telemetry(
            current_summary.state, normalized_force, xp=xp
        )
        next_summary = _accumulate_v4_substep_summary(
            current_summary, update, xp=xp
        )
        next_summary = next_summary._replace(
            normalized_force_finite=(
                current_summary.normalized_force_finite
                & xp.all(xp.isfinite(normalized_force))
            )
        )
        return next_summary, V4SubstepContactQualityTrajectory(
            xp.asarray(coherent_measurement_data.time),
            normalized_force,
            tangential_speed,
        )

    if scan is None:
        summary = initial_summary
        records = []
        for index in range(V4_CONTROL_SUBSTEP_COUNT):
            saved_state = V4SavedDynamicState(
                *(getattr(saved_states, field)[index] for field in saved_states._fields)
            )
            summary, record = body(summary, saved_state)
            records.append(record)
        return summary, V4SubstepContactQualityTrajectory(
            *(xp.stack(tuple(getattr(record, field) for record in records), axis=0)
              for field in V4SubstepContactQualityTrajectory._fields)
        )
    summary, trajectory = scan(body, initial_summary, saved_states)
    return summary, V4SubstepContactQualityTrajectory(
        _require_vector_shape(
            trajectory.time_s,
            V4_CONTROL_SUBSTEP_COUNT,
            "v4 collector time_s",
            xp=xp,
        ),
        require_trajectory_matrix(
            trajectory.normalized_normal_force,
            "v4 collector normalized_force trajectory",
        ),
        require_trajectory_matrix(
            trajectory.tangential_speed_m_s,
            "v4 collector tangential_speed trajectory",
        ),
    )


def scan_v4_saved_state_contact_quality_trajectory(
    control_entry_data: Any,
    saved_states: V4SavedDynamicState,
    *,
    cohere_measurement_state: Callable[[Any], Any],
    measure_force_and_tangential_speed: Callable[[Any], tuple[Any, Any]],
    n_substeps: int = V4_CONTROL_SUBSTEP_COUNT,
    scan: Callable[..., Any] | None = None,
    xp: Any = np,
) -> V4SubstepContactQualityTrajectory:
    """Replay saved dynamic6 points and return exact 2-ms force/slip samples.

    Like the existing telemetry scan, this is a post-physics measurement-only
    replay.  It neither changes the authoritative control trajectory nor adds
    another physics step, so callers can consume its output as an opt-in reward
    diagnostic without altering observations, targets, or guard behavior.
    """

    if int(n_substeps) != V4_CONTROL_SUBSTEP_COUNT:
        raise ValueError("forward v4 requires exactly 10 quality substeps")
    if not callable(cohere_measurement_state) or not callable(
        measure_force_and_tangential_speed
    ):
        raise ValueError("measurement coherence and quality sampler must be callable")
    saved_states = _require_v4_saved_dynamic_trajectory(saved_states, xp=xp)

    def require_matrix(value: Any, name: str) -> Any:
        array = xp.asarray(value)
        expected_shape = (V4_CONTROL_SUBSTEP_COUNT, 2)
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )
        return array

    def body(_unused: Any, saved_state: V4SavedDynamicState):
        replay_data = reconstruct_v4_dynamic_state(
            control_entry_data, saved_state
        )
        coherent_data = cohere_measurement_state(replay_data)
        normalized_force, tangential_speed = measure_force_and_tangential_speed(
            coherent_data
        )
        normalized_force = _require_vector_shape(
            normalized_force,
            2,
            "v4 quality normalized_force",
            xp=xp,
        )
        tangential_speed = _require_vector_shape(
            tangential_speed,
            2,
            "v4 quality tangential_speed",
            xp=xp,
        )
        return None, V4SubstepContactQualityTrajectory(
            xp.asarray(coherent_data.time),
            normalized_force,
            tangential_speed,
        )

    if scan is None:
        records = []
        for index in range(V4_CONTROL_SUBSTEP_COUNT):
            saved_state = V4SavedDynamicState(
                *(getattr(saved_states, field)[index] for field in saved_states._fields)
            )
            _unused, record = body(None, saved_state)
            records.append(record)
        return V4SubstepContactQualityTrajectory(
            *(xp.stack(tuple(getattr(record, field) for record in records), axis=0)
              for field in V4SubstepContactQualityTrajectory._fields)
        )
    _unused, trajectory = scan(body, None, saved_states)
    return V4SubstepContactQualityTrajectory(
        _require_vector_shape(
            trajectory.time_s,
            V4_CONTROL_SUBSTEP_COUNT,
            "v4 quality time_s",
            xp=xp,
        ),
        require_matrix(
            trajectory.normalized_normal_force,
            "v4 quality normalized_force trajectory",
        ),
        require_matrix(
            trajectory.tangential_speed_m_s,
            "v4 quality tangential_speed trajectory",
        ),
    )


def scan_v4_contact_telemetry_two_phase_reference(
    initial_data: Any,
    action: Any,
    initial_state: V4ContactTelemetryState,
    *,
    single_physics_step: Callable[[Any, Any], Any],
    cohere_measurement_state: Callable[[Any], Any],
    measure_normalized_force: Callable[[Any, Any], Any],
    n_substeps: int = V4_CONTROL_SUBSTEP_COUNT,
    scan: Callable[..., Any] | None = None,
    xp: Any = np,
) -> tuple[Any, V4SubstepTelemetrySummary]:
    """Two-phase pure reference: finish physics, then replay measurements.

    The caller retains its independently-computed authoritative source output;
    this compatibility helper returns only the reference endpoint and telemetry.
    Unlike the guarded MJX path, it saves the whole generic data pytree so pure
    NumPy/JAX tests need not emulate MJX dynamic fields.  The phases remain
    strictly separated: all physics calls finish before any coherence or force
    measurement call.
    """

    if int(n_substeps) != V4_CONTROL_SUBSTEP_COUNT:
        raise ValueError("forward v4 requires exactly 10 physics substeps")
    if (
        not callable(single_physics_step)
        or not callable(cohere_measurement_state)
        or not callable(measure_normalized_force)
    ):
        raise ValueError(
            "physics step, measurement coherence, and force measurement "
            "must be callable"
        )
    def physics_body(data: Any, _unused: Any):
        next_data = single_physics_step(data, action)
        return next_data, next_data

    def telemetry_body(
        current_summary: V4SubstepTelemetrySummary,
        replay_data: Any,
    ):
        coherent_measurement_data = cohere_measurement_state(replay_data)
        normalized_force = measure_normalized_force(
            coherent_measurement_data,
            current_summary.state.persistence.raw_contact,
        )
        normalized_force = _require_vector_shape(
            normalized_force, 2, "v4 telemetry normalized_force", xp=xp
        )
        update = update_v4_contact_telemetry(
            current_summary.state, normalized_force, xp=xp
        )
        next_summary = _accumulate_v4_substep_summary(
            current_summary, update, xp=xp
        )
        next_summary = next_summary._replace(
            normalized_force_finite=(
                current_summary.normalized_force_finite
                & xp.all(xp.isfinite(normalized_force))
            )
        )
        return next_summary, None

    if scan is None:
        endpoint_data = initial_data
        saved_data = []
        for _ in range(V4_CONTROL_SUBSTEP_COUNT):
            endpoint_data, saved = physics_body(endpoint_data, None)
            saved_data.append(saved)
        summary = _empty_v4_substep_summary(initial_state, xp=xp)
        for replay_data in saved_data:
            summary, _ = telemetry_body(summary, replay_data)
        return endpoint_data, summary
    endpoint_data, saved_data = scan(
        physics_body,
        initial_data,
        xs=None,
        length=V4_CONTROL_SUBSTEP_COUNT,
    )
    summary, _ = scan(
        telemetry_body,
        _empty_v4_substep_summary(initial_state, xp=xp),
        saved_data,
    )
    return endpoint_data, summary


def _audit_v4_full_tree_nonqualifying_diagnostic(
    authoritative_data: Any,
    comparison_data: Any,
    *,
    tree_leaves: Callable[[Any], Sequence[Any]],
    xp: Any = np,
) -> V4TrajectoryParity:
    """Compare every array leaf while leaving the authoritative data untouched."""

    if not callable(tree_leaves):
        raise ValueError("tree_leaves must be callable")
    authoritative_leaves = tuple(tree_leaves(authoritative_data))
    comparison_leaves = tuple(tree_leaves(comparison_data))
    if len(authoritative_leaves) != len(comparison_leaves):
        raise ValueError("nonqualifying diagnostic trajectory trees differ")
    exact = xp.asarray(True)
    max_abs_error = xp.zeros(())
    for authoritative, comparison in zip(authoritative_leaves, comparison_leaves):
        left = xp.asarray(authoritative)
        right = xp.asarray(comparison)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("nonqualifying diagnostic trajectory leaves differ")
        leaf_exact = xp.all(left == right)
        exact = exact & leaf_exact
        if np.issubdtype(np.dtype(left.dtype), np.inexact):
            leaf_error = xp.max(
                xp.abs(left - right), initial=xp.asarray(0.0, dtype=left.dtype)
            )
            max_abs_error = xp.maximum(
                max_abs_error, leaf_error.astype(max_abs_error.dtype)
            )
    return V4TrajectoryParity(exact, max_abs_error, len(authoritative_leaves))


def audit_v4_dynamic6_parity(
    reference_data: Any,
    candidate_data: Any,
    *,
    xp: Any = np,
) -> V4TrajectoryParity:
    """Compare exactly the six authoritative dynamic state fields."""

    reference = save_v4_dynamic_state(reference_data)
    candidate = save_v4_dynamic_state(candidate_data)
    exact = xp.asarray(True)
    max_abs_error = xp.zeros(())
    for field in V4SavedDynamicState._fields:
        left = xp.asarray(getattr(reference, field))
        right = xp.asarray(getattr(candidate, field))
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"v4 dynamic6 field {field} shape/dtype differs")
        exact = exact & xp.all(left == right)
        if np.issubdtype(np.dtype(left.dtype), np.inexact):
            error = xp.max(
                xp.abs(left - right),
                initial=xp.asarray(0.0, dtype=left.dtype),
            )
            max_abs_error = xp.maximum(
                max_abs_error, error.astype(max_abs_error.dtype)
            )
    return V4TrajectoryParity(exact, max_abs_error, len(reference._fields))


def v4_saved_dynamic_trajectory_all_finite(
    saved_states: V4SavedDynamicState,
    *,
    xp: Any = np,
) -> Any:
    """Return one device scalar covering every element of the 10x dynamic6 trace."""

    saved_states = _require_v4_saved_dynamic_trajectory(saved_states, xp=xp)
    finite = xp.asarray(True)
    for field in saved_states._fields:
        finite = finite & xp.all(xp.isfinite(getattr(saved_states, field)))
    return finite


def audit_v4_dynamic_endpoint_self_consistency(
    final_data: Any,
    saved_states: V4SavedDynamicState,
    *,
    xp: Any = np,
) -> V4TrajectoryParity:
    """Require the returned authority to equal saved substep ten dynamic6."""

    saved_states = _require_v4_saved_dynamic_trajectory(saved_states, xp=xp)
    saved_final = V4SavedDynamicState(
        *(getattr(saved_states, field)[-1] for field in saved_states._fields)
    )
    return audit_v4_dynamic6_parity(
        final_data,
        reconstruct_v4_dynamic_state(final_data, saved_final),
        xp=xp,
    )


def audit_v4_source_semantic_reference(
    model: Any,
    initial_data: Any,
    action: Any,
    *,
    source_physics_step: Callable[[Any, Any, Any, int], Any],
    mjx_step: Callable[[Any, Any], Any],
    scan: Callable[..., Any],
    xp: Any,
    n_substeps: int = V4_CONTROL_SUBSTEP_COUNT,
) -> tuple[Any, V4SavedDynamicState, V4SourceSemanticPreflight]:
    """One pre-PPO official-vs-single-authority semantic reference audit.

    Only dynamic6 qualifies.  ``cfrc_int`` and ``cfrc_ext`` are returned as
    observed post-solver diagnostics and never participate in pass/fail.
    """

    if int(n_substeps) != V4_CONTROL_SUBSTEP_COUNT:
        raise ValueError("forward v4 semantic reference requires 10 substeps")
    if not callable(source_physics_step) or not callable(mjx_step):
        raise ValueError("source_physics_step and mjx_step must be callable")
    reference_data = source_physics_step(
        model, initial_data, action, V4_CONTROL_SUBSTEP_COUNT
    )
    candidate_data, saved_states = scan_v4_instrumented_physics_trajectory(
        initial_data,
        action,
        single_physics_step=lambda data, control: (
            v4_authoritative_primitive_step(
                model, data, control, mjx_step=mjx_step
            )
        ),
        n_substeps=V4_CONTROL_SUBSTEP_COUNT,
        scan=scan,
        xp=xp,
    )
    dynamic6 = audit_v4_dynamic6_parity(
        reference_data, candidate_data, xp=xp
    )

    def derived_diagnostic(field: str) -> tuple[Any, Any]:
        reference = xp.asarray(getattr(reference_data._impl, field))
        candidate = xp.asarray(getattr(candidate_data._impl, field))
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise ValueError(f"v4 derived diagnostic {field} shape/dtype differs")
        exact = xp.all(reference == candidate)
        error = xp.max(
            xp.abs(reference - candidate),
            initial=xp.asarray(0.0, dtype=reference.dtype),
        )
        return exact, error

    cfrc_int_exact, cfrc_int_error = derived_diagnostic("cfrc_int")
    cfrc_ext_exact, cfrc_ext_error = derived_diagnostic("cfrc_ext")
    return (
        candidate_data,
        saved_states,
        V4SourceSemanticPreflight(
            dynamic6.exact,
            dynamic6.max_abs_error,
            dynamic6.leaf_count,
            cfrc_int_exact,
            cfrc_int_error,
            cfrc_ext_exact,
            cfrc_ext_error,
        ),
    )


def robot_body_weight_n(
    body_mass_kg: Any,
    robot_body_mask: Any,
    gravity_m_s2: Any,
    *,
    xp: Any = np,
) -> Any:
    """Compute randomized robot weight from current model body masses."""

    mass = xp.asarray(body_mass_kg)
    mask = xp.asarray(robot_body_mask).astype(bool)
    gravity = _require_vector_shape(
        gravity_m_s2, 3, "gravity_m_s2", xp=xp
    )
    if mass.ndim != 1 or mask.shape != mass.shape:
        raise ValueError("body mass and robot mask must be equal-width vectors")
    return xp.sum(xp.where(mask, mass, 0.0)) * xp.linalg.norm(gravity)


class FootContactQuality(NamedTuple):
    normalized_force: Any
    tangential_speed_m_s: Any
    contact: Any
    slip_rms_m_s: Any
    slip_loss_normalized_squared: Any


def aggregate_force_contact_quality(
    contact_normal_force_n: Any,
    contact_tangential_speed_m_s: Any,
    contact_foot_index: Any,
    previous_contact: Any,
    *,
    robot_weight_n: Any,
    xp: Any = np,
) -> FootContactQuality:
    """Aggregate contact-point slip exactly once per foot.

    Contact forces weight the multiple collision points belonging to one foot,
    matching the CPU evaluator's per-foot contact-point velocity reduction.
    The returned RMS proxy then weights both force-qualified stance feet
    equally, matching the metric's per-foot stance samples rather than letting
    the more heavily loaded foot hide the other foot's slip.
    """

    force = xp.asarray(contact_normal_force_n)
    speed = xp.asarray(contact_tangential_speed_m_s)
    foot_index = xp.asarray(contact_foot_index)
    if force.ndim != 1 or speed.shape != force.shape or foot_index.shape != force.shape:
        raise ValueError("contact force, speed, and foot index must be equal 1-D arrays")
    weight = xp.asarray(robot_weight_n)
    if weight.shape != ():
        raise ValueError("robot_weight_n must be scalar")
    # NumPy callers retain eager validation.  JAX callers may supply a traced
    # weight from a domain-randomized model, so do not concretize it here; the
    # environment constructor has already validated its static robot mask and
    # nominal positive mass, and this denominator remains numerically safe.
    if xp is np:
        numeric_weight = float(weight)
        if not np.isfinite(numeric_weight) or numeric_weight <= 0.0:
            raise ValueError("robot_weight_n must be finite and positive")
    positive_force = xp.maximum(force, 0.0)
    memberships = xp.stack((foot_index == 0, foot_index == 1)).astype(force.dtype)
    foot_force = xp.sum(memberships * positive_force[None, :], axis=1)
    weighted_speed = xp.sum(
        memberships * positive_force[None, :] * speed[None, :], axis=1
    )
    foot_speed = xp.where(
        foot_force > 0.0,
        weighted_speed / xp.maximum(foot_force, 1.0e-12),
        0.0,
    )
    normalized_force = foot_force / xp.maximum(weight, 1.0e-12)
    contact = force_schmitt_contacts(
        normalized_force, previous_contact, xp=xp
    )
    stance = contact.astype(foot_speed.dtype)
    stance_count = xp.sum(stance)
    mean_square = xp.sum(stance * xp.square(foot_speed)) / xp.maximum(
        stance_count, 1.0
    )
    slip_rms = xp.sqrt(xp.maximum(mean_square, 0.0))
    normalized_loss = mean_square / (STRICT_SLIP_RMS_M_S**2)
    return FootContactQuality(
        normalized_force,
        foot_speed,
        contact,
        slip_rms,
        normalized_loss,
    )


class AlternationUpdate(NamedTuple):
    last_single_support: Any
    ticks_since_switch: Any
    single_support: Any
    alternation_event: Any
    alternation_quality: Any


def update_alternation_state(
    previous_last_single_support: Any,
    previous_ticks_since_switch: Any,
    contact: Any,
    *,
    xp: Any = np,
) -> AlternationUpdate:
    """Update a chatter-resistant left/right single-support state.

    Double support and flight do not overwrite the last single-support side.
    A valid alternation event occurs only when a new single-support side is
    opposite the last remembered side.  The raw event and dwell-shaped quality
    are returned separately so reward scales remain an external choice.
    """

    feet = _require_vector_shape(contact, 2, "contact", xp=xp).astype(bool)
    left_only = feet[0] & ~feet[1]
    right_only = feet[1] & ~feet[0]
    single = left_only | right_only
    current_side = xp.where(left_only, 0, xp.where(right_only, 1, -1))
    last = xp.asarray(previous_last_single_support)
    ticks = xp.asarray(previous_ticks_since_switch)
    initialized = last >= 0
    event = single & initialized & (current_side != last)
    elapsed = (ticks + 1) * CONTROL_FIRST_STARTUP_DT_S
    quality = xp.where(
        event,
        xp.exp(
            -0.5
            * xp.square(
                (elapsed - DEFAULT_ALTERNATION_TARGET_SECONDS)
                / DEFAULT_ALTERNATION_SIGMA_SECONDS
            )
        ),
        0.0,
    )
    new_last = xp.where(single, current_side, last)
    new_ticks = xp.where(
        event,
        0,
        xp.where(initialized | single, ticks + 1, 0),
    )
    return AlternationUpdate(new_last, new_ticks, single, event, quality)


def update_load_balance_ema(
    previous_ema: Any,
    normalized_normal_force: Any,
    *,
    alpha: float = 0.02,
    xp: Any = np,
) -> tuple[Any, Any]:
    """Return per-foot force EMA and its dimensionless left/right imbalance."""

    previous = _require_vector_shape(previous_ema, 2, "previous_ema", xp=xp)
    current = _require_vector_shape(
        normalized_normal_force, 2, "normalized_normal_force", xp=xp
    )
    smoothing = float(alpha)
    if not np.isfinite(smoothing) or not 0.0 < smoothing <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    ema = (1.0 - smoothing) * previous + smoothing * current
    imbalance = xp.abs(ema[0] - ema[1]) / xp.maximum(xp.sum(ema), 1.0e-6)
    return ema, imbalance


class SupportQualityUpdate(NamedTuple):
    stance_slip_integral_m: Any
    single_support_ema: Any
    contact_duty_ema: Any
    touchdown_counts: Any
    touchdown_event: Any
    slip_tail_loss: Any
    stance_slip_budget_loss: Any
    single_support_band_loss: Any
    contact_duty_balance_loss: Any
    touchdown_count_balance_loss: Any
    flight: Any


def touchdown_count_imbalance_metric(
    touchdown_counts: Any,
    *,
    xp: Any = np,
) -> Any:
    """Project integer touchdown state to the stable float32 metric dtype.

    Forward-v4 deliberately keeps its causal touchdown counters as int32.
    Brax metrics, however, are initialized as float32 and become part of the
    PPO ``lax.scan`` carry.  Keeping the dtype conversion at this reporting
    boundary preserves exact integer event accounting without changing the
    carry type after the first environment step.
    """

    counts = _require_vector_shape(
        touchdown_counts, 2, "touchdown_counts", xp=xp
    )
    return xp.abs(counts[0] - counts[1]).astype(xp.float32)


class TotalNormalForceQuality(NamedTuple):
    total_normal_force_normalized: Any
    band_loss: Any
    tail_loss: Any


def total_normal_force_quality(
    normalized_normal_force: Any,
    *,
    xp: Any = np,
) -> TotalNormalForceQuality:
    """Return losses aligned to the frozen evaluator's force boundaries.

    The evaluator accepts a steady total normal force in ``[0.80, 1.20]``
    body weights and a full-episode p99 no greater than ``3.0`` body weights.
    These raw losses use the same per-foot, randomized-body-weight-normalized
    force that feeds contact classification.  Reward scales remain external.
    """

    force = _require_vector_shape(
        normalized_normal_force, 2, "normalized_normal_force", xp=xp
    )
    total = xp.sum(xp.maximum(force, 0.0))
    lower = xp.square(
        xp.maximum(STRICT_TOTAL_NORMAL_FORCE_LOWER_NORMALIZED - total, 0.0)
        / STRICT_TOTAL_NORMAL_FORCE_BAND_WIDTH_NORMALIZED
    )
    upper = xp.square(
        xp.maximum(total - STRICT_TOTAL_NORMAL_FORCE_UPPER_NORMALIZED, 0.0)
        / STRICT_TOTAL_NORMAL_FORCE_BAND_WIDTH_NORMALIZED
    )
    tail = xp.square(
        xp.maximum(total - STRICT_TOTAL_NORMAL_FORCE_TAIL_NORMALIZED, 0.0)
        / STRICT_TOTAL_NORMAL_FORCE_TAIL_NORMALIZED
    )
    return TotalNormalForceQuality(total, lower + upper, tail)


class ContactPulseUpdate(NamedTuple):
    contact_run_length_ticks: Any
    liftoff_event: Any
    per_foot_loss: Any
    event_mean_loss: Any


def update_contact_pulse_state(
    previous_contact: Any,
    contact: Any,
    previous_contact_run_length_ticks: Any,
    *,
    xp: Any = np,
) -> ContactPulseUpdate:
    """Penalize contact islands that end before the 40 ms debounce horizon.

    At each liftoff, a foot whose completed contact run lasted ``n`` 20 ms
    control ticks contributes ``relu((2-n)/2)^2``.  Loss is averaged over the
    liftoff events on that tick, so one 20 ms island contributes exactly 0.25.
    Off-contact gaps are diagnostic-only and are deliberately not penalized.
    """

    old_contact = _require_vector_shape(
        previous_contact, 2, "previous_contact", xp=xp
    ).astype(bool)
    feet = _require_vector_shape(contact, 2, "contact", xp=xp).astype(bool)
    previous_run = _require_vector_shape(
        previous_contact_run_length_ticks,
        2,
        "previous_contact_run_length_ticks",
        xp=xp,
    )
    if xp is np:
        numeric_run = np.asarray(previous_run)
        if (
            not np.all(np.isfinite(numeric_run))
            or np.any(numeric_run < 0)
            or not np.array_equal(numeric_run, np.floor(numeric_run))
        ):
            raise ValueError("contact run lengths must be finite non-negative integers")
    liftoff = old_contact & ~feet
    loss_dtype = previous_run.dtype
    per_foot_loss = liftoff.astype(loss_dtype) * xp.square(
        xp.maximum(
            CONTACT_PULSE_MINIMUM_RUN_TICKS - previous_run,
            0,
        ).astype(loss_dtype)
        / float(CONTACT_PULSE_MINIMUM_RUN_TICKS)
    )
    event_count = xp.sum(liftoff.astype(loss_dtype))
    event_mean_loss = xp.sum(per_foot_loss) / xp.maximum(event_count, 1)
    new_run = xp.where(
        feet,
        xp.where(old_contact, previous_run + 1, 1),
        0,
    )
    return ContactPulseUpdate(new_run, liftoff, per_foot_loss, event_mean_loss)


def update_support_quality_state(
    previous_contact: Any,
    contact: Any,
    tangential_speed_m_s: Any,
    previous_stance_slip_integral_m: Any,
    previous_single_support_ema: Any,
    previous_contact_duty_ema: Any,
    previous_touchdown_counts: Any,
    *,
    xp: Any = np,
) -> SupportQualityUpdate:
    """Update differentiable per-foot stance and support quality state."""

    old_contact = _require_vector_shape(
        previous_contact, 2, "previous_contact", xp=xp
    ).astype(bool)
    feet = _require_vector_shape(contact, 2, "contact", xp=xp).astype(bool)
    speed = _require_vector_shape(
        tangential_speed_m_s, 2, "tangential_speed_m_s", xp=xp
    )
    previous_integral = _require_vector_shape(
        previous_stance_slip_integral_m,
        2,
        "previous_stance_slip_integral_m",
        xp=xp,
    )
    previous_duty = _require_vector_shape(
        previous_contact_duty_ema, 2, "previous_contact_duty_ema", xp=xp
    )
    touchdown_counts = _require_vector_shape(
        previous_touchdown_counts, 2, "previous_touchdown_counts", xp=xp
    )
    touchdown = feet & ~old_contact
    stance_integral = xp.where(
        feet,
        xp.where(old_contact, previous_integral, 0.0)
        + speed * CONTROL_FIRST_STARTUP_DT_S,
        0.0,
    )
    alpha = 1.0 - np.exp(
        -CONTROL_FIRST_STARTUP_DT_S / SUPPORT_EMA_HORIZON_SECONDS
    )
    single = xp.logical_xor(feet[0], feet[1]).astype(speed.dtype)
    single_ema = (
        (1.0 - alpha) * xp.asarray(previous_single_support_ema)
        + alpha * single
    )
    duty_ema = (
        (1.0 - alpha) * previous_duty + alpha * feet.astype(speed.dtype)
    )
    new_touchdown_counts = touchdown_counts + touchdown.astype(
        touchdown_counts.dtype
    )
    contact_float = feet.astype(speed.dtype)
    slip_tail_loss = xp.mean(
        contact_float
        * xp.square(
            xp.maximum(speed - STRICT_SLIP_TAIL_M_S, 0.0)
            / STRICT_SLIP_TAIL_M_S
        )
    )
    stance_budget_loss = xp.mean(
        xp.square(
            xp.maximum(
                stance_integral - STRICT_STANCE_SLIP_BUDGET_M, 0.0
            )
            / STRICT_STANCE_SLIP_BUDGET_M
        )
    )
    support_band_loss = (
        xp.square(xp.maximum(0.25 - single_ema, 0.0) / 0.25)
        + xp.square(xp.maximum(single_ema - 0.60, 0.0) / 0.40)
    )
    duty_balance_loss = xp.square(
        (duty_ema[0] - duty_ema[1]) / STRICT_CONTACT_DUTY_IMBALANCE
    )
    touchdown_balance_loss = xp.square(
        xp.maximum(
            xp.abs(new_touchdown_counts[0] - new_touchdown_counts[1]) - 1.0,
            0.0,
        )
    )
    flight = (~feet[0] & ~feet[1]).astype(speed.dtype)
    return SupportQualityUpdate(
        stance_integral,
        single_ema,
        duty_ema,
        new_touchdown_counts,
        touchdown,
        slip_tail_loss,
        stance_budget_loss,
        support_band_loss,
        duty_balance_loss,
        touchdown_balance_loss,
        flight,
    )


def quaternion_yaw_wxyz(quaternion: Any, *, xp: Any = np) -> Any:
    """Return wrapped yaw for one MuJoCo ``[w, x, y, z]`` quaternion."""

    quat = _require_vector_shape(quaternion, 4, "quaternion", xp=xp)
    w, x, y, z = quat
    return xp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrapped_angle_difference(angle: Any, reference: Any, *, xp: Any = np) -> Any:
    difference = xp.asarray(angle) - xp.asarray(reference)
    return xp.arctan2(xp.sin(difference), xp.cos(difference))


class ReversePhaseQualityLosses(NamedTuple):
    phase_force_slip: Any
    contact_priority_reversal_lag: Any
    left_phase_active: Any
    right_phase_active: Any


def reverse_phase_conditioned_quality_losses(
    phase_index: Any,
    force_contact: Any,
    tangential_speed_m_s: Any,
    upstream_signed_delta: Any,
    previous_upstream_signed_delta: Any,
    normalized_joint_lag: Any,
    *,
    xp: Any = np,
) -> ReversePhaseQualityLosses:
    """Return reverse-spec slip and target-reversal losses for phase priorities."""

    contact = _require_vector_shape(force_contact, 2, "force_contact", xp=xp).astype(bool)
    slip = _require_vector_shape(
        tangential_speed_m_s, 2, "tangential_speed_m_s", xp=xp
    )
    current_delta = _require_vector_shape(
        upstream_signed_delta, ACTION_COUNT, "upstream_signed_delta", xp=xp
    )
    previous_delta = _require_vector_shape(
        previous_upstream_signed_delta,
        ACTION_COUNT,
        "previous_upstream_signed_delta",
        xp=xp,
    )
    lag = _require_vector_shape(
        normalized_joint_lag, ACTION_COUNT, "normalized_joint_lag", xp=xp
    )
    phase = xp.asarray(phase_index).astype(xp.int32)

    def in_phase(indices: Sequence[int]) -> Any:
        return xp.any(phase == xp.asarray(indices, dtype=xp.int32))

    left_active = in_phase(REVERSE_LEFT_PHASE_INDICES) & contact[0]
    right_mid_active = in_phase(REVERSE_RIGHT_MID_PHASE_INDICES) & contact[1]
    right_late_active = in_phase(REVERSE_RIGHT_LATE_PHASE_INDICES) & contact[1]
    right_active = right_mid_active | right_late_active
    dtype = slip.dtype
    phase_slip = (
        left_active.astype(dtype) * xp.square(slip[0] / STRICT_SLIP_RMS_M_S)
        + right_active.astype(dtype) * xp.square(slip[1] / STRICT_SLIP_RMS_M_S)
    )
    reversal = xp.maximum(
        -(current_delta * previous_delta) / (MAX_TARGET_DELTA_PER_TICK_RAD**2),
        0.0,
    )
    reversal_lag = reversal + lag
    left_priority = xp.mean(reversal_lag[xp.asarray((1, 2, 4))])
    right_mid_priority = xp.mean(
        reversal_lag[xp.asarray((1, 11, 12, 13))]
    )
    right_late_priority = xp.mean(reversal_lag[xp.asarray((1, 12, 13))])
    priority_loss = (
        left_active.astype(dtype) * left_priority
        + right_mid_active.astype(dtype) * right_mid_priority
        + right_late_active.astype(dtype) * right_late_priority
    )
    return ReversePhaseQualityLosses(
        phase_slip,
        priority_loss,
        left_active,
        right_active,
    )


@dataclass(frozen=True)
class H4QualityRewardScales:
    """External reward scales; raw quality observables do not depend on these."""

    force_slip: float = -2.0
    left_force_slip: float = -0.5
    right_force_slip: float = -0.5
    per_foot_slip_tail: float = -1.0
    per_foot_stance_slip_budget: float = -1.0
    single_support: float = 0.0
    single_support_band: float = -1.0
    alternation: float = 4.0
    load_balance: float = -1.0
    touchdown_count_balance: float = -1.0
    flight: float = -10.0
    total_normal_force_band: float = 0.0
    total_normal_force_tail: float = 0.0
    contact_pulse_40ms: float = 0.0
    slew_feasibility: float = -0.25
    target_lag: float = -0.25
    left_target_lag: float = 0.0
    right_target_lag: float = 0.0
    phase17_left_force_slip: float = -1.0
    phase17_left_knee_envelope_excess: float = -0.5
    phase17_opposite_leg_lag: float = -0.75
    forward_cross_drift: float = -2.0
    forward_uncommanded_yaw_rate: float = -1.0
    forward_heading_drift: float = -1.0
    reverse_speed_boundary: float = 0.0
    reverse_cross_drift: float = 0.0
    reverse_uncommanded_yaw_rate: float = 0.0
    reverse_heading_drift: float = 0.0
    reverse_phase_force_slip: float = 0.0
    reverse_contact_priority_reversal_lag: float = 0.0
    h5_all_substep_strict20ms_slip_rms: float = 0.0
    h5_all_substep_slip_tail: float = 0.0
    h5_all_substep_force_tail: float = 0.0

    def as_reward_scale_dict(
        self, *, include_h5_substep_contact_alignment: bool = False
    ) -> dict[str, float]:
        """Return the configured reward keys for the active environment mode.

        Historical H4 authorizations pin the complete reward-key schema.  The
        three H5-only terms therefore exist only in the explicit H5 substep
        treatment; emitting zero-valued keys into every legacy configuration
        would silently invalidate those byte-pinned contracts.
        """

        values = {
            "h4_force_slip": self.force_slip,
            "h4_left_force_slip": self.left_force_slip,
            "h4_right_force_slip": self.right_force_slip,
            "h4_per_foot_slip_tail": self.per_foot_slip_tail,
            "h4_per_foot_stance_slip_budget": self.per_foot_stance_slip_budget,
            "h4_single_support": self.single_support,
            "h4_single_support_band": self.single_support_band,
            "h4_alternation": self.alternation,
            "h4_load_balance": self.load_balance,
            "h4_touchdown_count_balance": self.touchdown_count_balance,
            "h4_flight": self.flight,
            "h4_total_normal_force_band": self.total_normal_force_band,
            "h4_total_normal_force_tail": self.total_normal_force_tail,
            "h4_contact_pulse_40ms": self.contact_pulse_40ms,
            "h4_slew_feasibility": self.slew_feasibility,
            "h4_target_lag": self.target_lag,
            "h4_left_target_lag": self.left_target_lag,
            "h4_right_target_lag": self.right_target_lag,
            "h4_phase17_left_force_slip": self.phase17_left_force_slip,
            "h4_phase17_left_knee_envelope_excess": (
                self.phase17_left_knee_envelope_excess
            ),
            "h4_phase17_opposite_leg_lag": self.phase17_opposite_leg_lag,
            "h4_forward_cross_drift": self.forward_cross_drift,
            "h4_forward_uncommanded_yaw_rate": (
                self.forward_uncommanded_yaw_rate
            ),
            "h4_forward_heading_drift": self.forward_heading_drift,
            "h4_reverse_speed_boundary": self.reverse_speed_boundary,
            "h4_reverse_cross_drift": self.reverse_cross_drift,
            "h4_reverse_uncommanded_yaw_rate": (
                self.reverse_uncommanded_yaw_rate
            ),
            "h4_reverse_heading_drift": self.reverse_heading_drift,
            "h4_reverse_phase_force_slip": self.reverse_phase_force_slip,
            "h4_reverse_contact_priority_reversal_lag": (
                self.reverse_contact_priority_reversal_lag
            ),
        }
        if include_h5_substep_contact_alignment:
            values.update(
                {
                    "h5_all_substep_strict20ms_slip_rms": (
                        self.h5_all_substep_strict20ms_slip_rms
                    ),
                    "h5_all_substep_slip_tail": self.h5_all_substep_slip_tail,
                    "h5_all_substep_force_tail": self.h5_all_substep_force_tail,
                }
            )
        if any(not np.isfinite(float(value)) for value in values.values()):
            raise ValueError("all H4 reward scales must be finite")
        cost_names = (
            "h4_force_slip",
            "h4_left_force_slip",
            "h4_right_force_slip",
            "h4_per_foot_slip_tail",
            "h4_per_foot_stance_slip_budget",
            "h4_single_support_band",
            "h4_load_balance",
            "h4_touchdown_count_balance",
            "h4_flight",
            "h4_total_normal_force_band",
            "h4_total_normal_force_tail",
            "h4_contact_pulse_40ms",
            "h4_slew_feasibility",
            "h4_target_lag",
            "h4_left_target_lag",
            "h4_right_target_lag",
            "h4_phase17_left_force_slip",
            "h4_phase17_left_knee_envelope_excess",
            "h4_phase17_opposite_leg_lag",
            "h4_forward_cross_drift",
            "h4_forward_uncommanded_yaw_rate",
            "h4_forward_heading_drift",
            "h4_reverse_speed_boundary",
            "h4_reverse_cross_drift",
            "h4_reverse_uncommanded_yaw_rate",
            "h4_reverse_heading_drift",
            "h4_reverse_phase_force_slip",
            "h4_reverse_contact_priority_reversal_lag",
        )
        if include_h5_substep_contact_alignment:
            cost_names = (*cost_names, "h5_all_substep_strict20ms_slip_rms", "h5_all_substep_slip_tail", "h5_all_substep_force_tail")
        if any(values[name] > 0.0 for name in cost_names):
            raise ValueError("H4 cost scales must be non-positive")
        if values["h4_single_support"] < 0.0 or values["h4_alternation"] < 0.0:
            raise ValueError("H4 reward scales must be non-negative")
        return values


def make_anchor_command_mapper(
    physical_anchor: Sequence[float],
    policy_observation_anchor: Sequence[float],
    *,
    xp: Any = np,
) -> Callable[[Any], Any]:
    """Build a sign-preserving physical-to-policy command mapper.

    The three-axis policy anchor is scaled by the projection of the physical
    command onto its declared physical anchor.  Stop therefore maps to exact
    stop, while recovery-scale curriculum samples preserve compensating policy
    axes such as the selected forward ``vy`` and yaw offsets.
    """

    physical = np.asarray(physical_anchor, dtype=np.float64)
    policy = np.asarray(policy_observation_anchor, dtype=np.float64)
    if (
        physical.shape != (3,)
        or policy.shape != (3,)
        or not np.all(np.isfinite(physical))
        or not np.all(np.isfinite(policy))
        or float(np.dot(physical, physical)) <= 0.0
    ):
        raise ValueError("command anchors must be finite triplets with physical motion")
    physical_xp = xp.asarray(physical)
    policy_xp = xp.asarray(policy)
    denominator = float(np.dot(physical, physical))

    def mapper(command: Any) -> Any:
        values = _require_vector_shape(command, 7, "physical command", xp=xp)
        ratio = xp.dot(values[:3], physical_xp) / denominator
        locomotion = policy_xp * ratio
        return xp.concatenate((locomotion, xp.zeros(4, dtype=values.dtype)))

    return mapper


def make_h4_reverse_physical_sampler(jax: Any, xp: Any) -> Callable[[Any], Any]:
    """Return the minimum-spec reverse curriculum (60% exact -0.05 anchor)."""

    stand_edge = H4_REVERSE_STAND_PROBABILITY
    exact_edge = stand_edge + H4_REVERSE_EXACT_ENDPOINT_PROBABILITY
    local_edge = exact_edge + H4_REVERSE_LOCAL_ANCHOR_PROBABILITY
    if not np.isclose(
        local_edge + H4_REVERSE_TRANSITION_PROBABILITY, 1.0
    ):
        raise RuntimeError("reverse curriculum probabilities must sum to one")
    local_anchors = xp.asarray(H4_REVERSE_LOCAL_ANCHORS_M_S)

    def sampler(rng: Any) -> Any:
        mode_key, anchor_key, transition_key = jax.random.split(rng, 3)
        mode = jax.random.uniform(mode_key)
        anchor_index = jax.random.randint(
            anchor_key, shape=(), minval=0, maxval=2
        )
        transition = jax.random.uniform(
            transition_key,
            minval=H4_REVERSE_TRANSITION_BAND_M_S[0],
            maxval=H4_REVERSE_TRANSITION_BAND_M_S[1],
        )
        vx = xp.where(
            mode < stand_edge,
            0.0,
            xp.where(
                mode < exact_edge,
                H4_REVERSE_PRIMARY_ANCHOR_M_S,
                xp.where(mode < local_edge, local_anchors[anchor_index], transition),
            ),
        )
        return xp.asarray((vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    return sampler


def make_h4_reverse_v2_physical_sampler(
    jax: Any, xp: Any
) -> Callable[[Any], Any]:
    """Return the explicit iteration-v2 reverse curriculum."""

    stand_edge = H4_REVERSE_V2_STAND_PROBABILITY
    exact_edge = stand_edge + H4_REVERSE_V2_EXACT_ENDPOINT_PROBABILITY
    local_edge = exact_edge + H4_REVERSE_V2_LOCAL_ANCHOR_PROBABILITY
    if not np.isclose(
        local_edge + H4_REVERSE_V2_TRANSITION_PROBABILITY, 1.0
    ):
        raise RuntimeError("reverse v2 curriculum probabilities must sum to one")
    local_anchors = xp.asarray(H4_REVERSE_LOCAL_ANCHORS_M_S)

    def sampler(rng: Any) -> Any:
        mode_key, anchor_key, transition_key = jax.random.split(rng, 3)
        mode = jax.random.uniform(mode_key)
        anchor_index = jax.random.randint(
            anchor_key, shape=(), minval=0, maxval=2
        )
        transition = jax.random.uniform(
            transition_key,
            minval=H4_REVERSE_TRANSITION_BAND_M_S[0],
            maxval=H4_REVERSE_TRANSITION_BAND_M_S[1],
        )
        vx = xp.where(
            mode < stand_edge,
            0.0,
            xp.where(
                mode < exact_edge,
                H4_REVERSE_PRIMARY_ANCHOR_M_S,
                xp.where(mode < local_edge, local_anchors[anchor_index], transition),
            ),
        )
        return xp.asarray((vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    return sampler


def make_h4_forward_physical_sampler(jax: Any, xp: Any) -> Callable[[Any], Any]:
    """Return the forward counterpart of the H4 anchor curriculum.

    Sixty percent of samples are the physical 0.05 m/s endpoint.  Remaining
    samples cover exact stand, the local 0.04/0.06 m/s neighborhood, and a
    low-speed entry band.  Policy-visible compensation is deliberately not
    embedded here; it belongs to ``policy_observation_mapper``.
    """

    stand_edge = H4_FORWARD_STAND_PROBABILITY
    exact_edge = stand_edge + H4_FORWARD_EXACT_ENDPOINT_PROBABILITY
    local_edge = exact_edge + H4_FORWARD_LOCAL_ANCHOR_PROBABILITY
    if not np.isclose(
        local_edge + H4_FORWARD_TRANSITION_PROBABILITY, 1.0
    ):
        raise RuntimeError("forward curriculum probabilities must sum to one")
    local_anchors = xp.asarray(H4_FORWARD_LOCAL_ANCHORS_M_S)

    def sampler(rng: Any) -> Any:
        mode_key, anchor_key, transition_key = jax.random.split(rng, 3)
        mode = jax.random.uniform(mode_key)
        anchor_index = jax.random.randint(
            anchor_key, shape=(), minval=0, maxval=2
        )
        transition = jax.random.uniform(
            transition_key,
            minval=H4_FORWARD_TRANSITION_BAND_M_S[0],
            maxval=H4_FORWARD_TRANSITION_BAND_M_S[1],
        )
        vx = xp.where(
            mode < stand_edge,
            0.0,
            xp.where(
                mode < exact_edge,
                H4_FORWARD_PRIMARY_ANCHOR_M_S,
                xp.where(mode < local_edge, local_anchors[anchor_index], transition),
            ),
        )
        return xp.asarray((vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    return sampler


def make_h4_forward_v2_physical_sampler(
    jax: Any, xp: Any
) -> Callable[[Any], Any]:
    """Return the authorized iteration-v2 forward curriculum.

    This is separate from :func:`make_h4_forward_physical_sampler` so the
    original 60/20/10/10 pilot contract remains available and cannot silently
    change when an old command line is replayed.
    """

    stand_edge = H4_FORWARD_V2_STAND_PROBABILITY
    exact_edge = stand_edge + H4_FORWARD_V2_EXACT_ENDPOINT_PROBABILITY
    local_edge = exact_edge + H4_FORWARD_V2_LOCAL_ANCHOR_PROBABILITY
    if not np.isclose(
        local_edge + H4_FORWARD_V2_TRANSITION_PROBABILITY, 1.0
    ):
        raise RuntimeError("forward v2 curriculum probabilities must sum to one")
    local_anchors = xp.asarray(H4_FORWARD_LOCAL_ANCHORS_M_S)

    def sampler(rng: Any) -> Any:
        mode_key, anchor_key, transition_key = jax.random.split(rng, 3)
        mode = jax.random.uniform(mode_key)
        anchor_index = jax.random.randint(
            anchor_key, shape=(), minval=0, maxval=2
        )
        transition = jax.random.uniform(
            transition_key,
            minval=H4_FORWARD_TRANSITION_BAND_M_S[0],
            maxval=H4_FORWARD_TRANSITION_BAND_M_S[1],
        )
        vx = xp.where(
            mode < stand_edge,
            0.0,
            xp.where(
                mode < exact_edge,
                H4_FORWARD_PRIMARY_ANCHOR_M_S,
                xp.where(mode < local_edge, local_anchors[anchor_index], transition),
            ),
        )
        return xp.asarray((vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    return sampler


def synchronize_observation_motor_targets(
    observation: Mapping[str, Any], applied_targets: Any, *, xp: Any = np
) -> dict[str, Any]:
    """Replace only the policy-visible motor-target slice after guarded physics."""

    targets = _require_vector_shape(
        applied_targets, ACTION_COUNT, "applied_targets", xp=xp
    )
    result = dict(observation)
    for key in ("state", "privileged_state"):
        if key not in result:
            raise ValueError(f"observation is missing {key!r}")
        values = xp.asarray(result[key])
        if values.ndim != 1 or values.shape[0] < OBSERVATION_MOTOR_TARGET_SLICE.stop:
            raise ValueError(f"observation {key!r} is too short")
        if hasattr(values, "at"):
            values = values.at[OBSERVATION_MOTOR_TARGET_SLICE].set(targets)
        else:
            values = np.array(values, copy=True)
            values[OBSERVATION_MOTOR_TARGET_SLICE] = np.asarray(targets)
        result[key] = values
    return result


def synchronize_post_step_command_observations(
    observation: Mapping[str, Any],
    physical_command: Any,
    policy_observation_command: Any,
    *,
    include_h4_actor_observables: bool,
    xp: Any = np,
) -> dict[str, Any]:
    """Synchronize the *next* command after the frozen resample boundary.

    The frozen task constructs its observation and reward before it resamples
    ``info['command']`` at the episode command boundary.  Consequently its
    returned observation can otherwise contain the just-rewarded command
    while ``info['command']`` already contains the next physical command.
    This pure post-step update changes only command-bearing observation slices:
    the legacy seven-channel policy input receives the mapped command, and the
    optional H4 physical-command feature receives the physical target.
    """

    physical = _require_vector_shape(
        physical_command, 7, "physical command", xp=xp
    )
    policy = _require_vector_shape(
        policy_observation_command, 7, "policy observation command", xp=xp
    )
    if not isinstance(include_h4_actor_observables, (bool, np.bool_)):
        raise ValueError("include_h4_actor_observables must be boolean")
    result = dict(observation)
    minimum_width = (
        H4_OBSERVATION_PHYSICAL_COMMAND_SLICE.stop
        if include_h4_actor_observables
        else OBSERVATION_POLICY_COMMAND_SLICE.stop
    )
    for key in ("state", "privileged_state"):
        if key not in result:
            raise ValueError(f"observation is missing {key!r}")
        values = xp.asarray(result[key])
        if values.ndim != 1 or values.shape[0] < minimum_width:
            raise ValueError(f"observation {key!r} is too short")
        if hasattr(values, "at"):
            values = values.at[OBSERVATION_POLICY_COMMAND_SLICE].set(policy)
            if include_h4_actor_observables:
                values = values.at[
                    H4_OBSERVATION_PHYSICAL_COMMAND_SLICE
                ].set(physical[:3])
        else:
            values = np.array(values, copy=True)
            values[OBSERVATION_POLICY_COMMAND_SLICE] = np.asarray(policy)
            if include_h4_actor_observables:
                values[H4_OBSERVATION_PHYSICAL_COMMAND_SLICE] = np.asarray(
                    physical[:3]
                )
        result[key] = values
    return result


def synchronize_post_step_imitation_state(
    observation: Mapping[str, Any],
    current_reference_motion: Any,
    imitation_index: Any,
    imitation_phase: Any,
    *,
    xp: Any = np,
) -> dict[str, Any]:
    """Synchronize actor phase and every critic imitation-tail channel."""

    phase = _require_vector_shape(
        imitation_phase, 2, "imitation_phase", xp=xp
    )
    reference = _require_vector_shape(
        current_reference_motion, 40, "current_reference_motion", xp=xp
    )
    index = xp.asarray(imitation_index)
    if index.shape != ():
        raise ValueError("imitation_index must be scalar")
    result = dict(observation)
    for key in ("state", "privileged_state"):
        if key not in result:
            raise ValueError(f"observation is missing {key!r}")
        values = xp.asarray(result[key])
        if values.ndim != 1 or values.shape[0] < OBSERVATION_IMITATION_PHASE_SLICE.stop:
            raise ValueError(f"observation {key!r} is too short")
        if hasattr(values, "at"):
            values = values.at[OBSERVATION_IMITATION_PHASE_SLICE].set(phase)
        else:
            values = np.array(values, copy=True)
            values[OBSERVATION_IMITATION_PHASE_SLICE] = np.asarray(phase)
        result[key] = values
    privileged = result["privileged_state"]
    if privileged.shape[0] == 212:
        offset = 0
    elif privileged.shape[0] == 227:
        offset = H4_ACTOR_OBSERVATION_WIDTH - LEGACY_ACTOR_OBSERVATION_WIDTH
    else:
        raise ValueError("privileged observation width must be exact 212 or 227")
    reference_slice = slice(
        LEGACY_PRIVILEGED_REFERENCE_SLICE.start + offset,
        LEGACY_PRIVILEGED_REFERENCE_SLICE.stop + offset,
    )
    index_slice = slice(
        LEGACY_PRIVILEGED_IMITATION_INDEX_SLICE.start + offset,
        LEGACY_PRIVILEGED_IMITATION_INDEX_SLICE.stop + offset,
    )
    tail_phase_slice = slice(
        LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE.start + offset,
        LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE.stop + offset,
    )
    if hasattr(privileged, "at"):
        privileged = privileged.at[reference_slice].set(reference)
        privileged = privileged.at[index_slice].set(index[None])
        privileged = privileged.at[tail_phase_slice].set(phase)
    else:
        privileged = np.array(privileged, copy=True)
        privileged[reference_slice] = np.asarray(reference)
        privileged[index_slice] = np.asarray(index)[None]
        privileged[tail_phase_slice] = np.asarray(phase)
    result["privileged_state"] = privileged
    return result


def synchronize_post_step_imitation_phase(
    observation: Mapping[str, Any], imitation_phase: Any, *, xp: Any = np
) -> dict[str, Any]:
    """Deprecated prefix-only helper retained for pure legacy callers."""

    result = dict(observation)
    phase = _require_vector_shape(imitation_phase, 2, "imitation_phase", xp=xp)
    for key in ("state", "privileged_state"):
        values = xp.asarray(result[key])
        if hasattr(values, "at"):
            values = values.at[OBSERVATION_IMITATION_PHASE_SLICE].set(phase)
        else:
            values = np.array(values, copy=True)
            values[OBSERVATION_IMITATION_PHASE_SLICE] = np.asarray(phase)
        result[key] = values
    return result


def _insert_zero_feature_rows(
    kernel: Any,
    *,
    legacy_prefix_width: int,
    extra_width: int,
    xp: Any,
) -> Any:
    values = xp.asarray(kernel)
    if values.ndim != 2 or not 0 <= legacy_prefix_width <= values.shape[0]:
        raise ValueError("checkpoint first-layer kernel has an invalid shape")
    zeros = xp.zeros((extra_width, values.shape[1]), dtype=values.dtype)
    return xp.concatenate(
        (
            values[:legacy_prefix_width],
            zeros,
            values[legacy_prefix_width:],
        ),
        axis=0,
    )


def transplant_v22_checkpoint_to_h4_observation(
    checkpoint_params: Sequence[Any], *, xp: Any = np
) -> tuple[list[Any], dict[str, Any]]:
    """Explicitly expand the v22 101/212 checkpoint to H4 116/227 inputs.

    The actor's old 101 rows are copied and fifteen new rows are zero.  The
    actor's fifteen new physical/contact features are zero.  The critic keeps
    the old privileged tail in place semantically: zero rows are
    inserted after the 101 actor-state prefix rather than appended after all
    212 old rows.  Running-statistic vectors use the same insertion.  New
    feature means are zero, standard deviations are one, and summed variance
    is initialized consistently with the inherited global sample count.

    This helper does not load or save checkpoints.  A training runner must
    invoke it explicitly and pass the returned values as ``restore_params``;
    pointing Brax directly at the old checkpoint is intentionally unsupported
    for a 116-wide actor.
    """

    if len(checkpoint_params) != 3:
        raise ValueError("v22 PPO checkpoint must contain normalizer, actor, critic")
    normalizer, actor, critic = checkpoint_params
    required_normalizer = ("mean", "std", "summed_variance", "count", "replace")
    if any(not hasattr(normalizer, name) for name in required_normalizer):
        raise ValueError("unexpected v22 running-statistics structure")
    for mapping_name in ("mean", "std", "summed_variance"):
        mapping = getattr(normalizer, mapping_name)
        if set(mapping) != {"state", "privileged_state"}:
            raise ValueError(f"normalizer {mapping_name} keys drifted")
    state_mean = xp.asarray(normalizer.mean["state"])
    privileged_mean = xp.asarray(normalizer.mean["privileged_state"])
    if state_mean.shape != (LEGACY_ACTOR_OBSERVATION_WIDTH,):
        raise ValueError("v22 state normalizer width must be 101")
    if privileged_mean.shape != (212,):
        raise ValueError("v22 privileged normalizer width must be 212")
    extra_width = H4_ACTOR_OBSERVATION_WIDTH - LEGACY_ACTOR_OBSERVATION_WIDTH

    def insert_vector(values: Any, fill: Any) -> Any:
        array = xp.asarray(values)
        prefix = LEGACY_ACTOR_OBSERVATION_WIDTH
        return xp.concatenate(
            (array[:prefix], fill, array[prefix:]), axis=0
        )

    count = normalizer.count
    try:
        count_float = float(np.asarray(count.hi)) * (2.0**32) + float(
            np.asarray(count.lo)
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("v22 normalizer count must expose UInt64 hi/lo") from exc
    extra_zero = xp.zeros((extra_width,), dtype=state_mean.dtype)
    extra_one = xp.ones((extra_width,), dtype=xp.asarray(normalizer.std["state"]).dtype)
    source_state_variance = xp.asarray(normalizer.summed_variance["state"])
    source_privileged_variance = xp.asarray(
        normalizer.summed_variance["privileged_state"]
    )
    source_variance_arrays = (
        np.asarray(source_state_variance),
        np.asarray(source_privileged_variance),
    )
    if any(not np.all(np.isfinite(values)) for values in source_variance_arrays):
        raise ValueError("v22 summed variance must be finite")
    legacy_variance_min = min(float(np.min(values)) for values in source_variance_arrays)
    legacy_variance_repair_count = sum(
        int(np.count_nonzero(values < 0.0)) for values in source_variance_arrays
    )
    if legacy_variance_min < -0.01:
        raise ValueError(
            "v22 summed variance is below the audited legacy repair floor -0.01"
        )
    # The frozen v22 normalizer contains two tiny negative round-off residues in
    # its privileged statistic.  They are accepted only at this explicit
    # 101->116 transplant boundary and are repaired before the first update.
    repaired_state_variance = xp.maximum(source_state_variance, 0.0)
    repaired_privileged_variance = xp.maximum(source_privileged_variance, 0.0)
    extra_variance = xp.full(
        (extra_width,),
        count_float,
        dtype=source_state_variance.dtype,
    )
    mean = {
        "state": xp.concatenate((state_mean, extra_zero)),
        "privileged_state": insert_vector(privileged_mean, extra_zero),
    }
    std = {
        "state": xp.concatenate(
            (xp.asarray(normalizer.std["state"]), extra_one)
        ),
        "privileged_state": insert_vector(
            normalizer.std["privileged_state"], extra_one
        ),
    }
    summed_variance = {
        "state": xp.concatenate(
            (
                repaired_state_variance,
                extra_variance,
            )
        ),
        "privileged_state": insert_vector(
            repaired_privileged_variance, extra_variance
        ),
    }
    new_normalizer = normalizer.replace(
        mean=mean, std=std, summed_variance=summed_variance
    )

    try:
        actor_kernel = actor["params"]["hidden_0"]["kernel"]
        critic_kernel = critic["params"]["hidden_0"]["kernel"]
    except (KeyError, TypeError) as exc:
        raise ValueError("v22 actor/critic first-layer path drifted") from exc
    if xp.asarray(actor_kernel).shape[0] != LEGACY_ACTOR_OBSERVATION_WIDTH:
        raise ValueError("v22 actor first-layer input width must be 101")
    if xp.asarray(critic_kernel).shape[0] != 212:
        raise ValueError("v22 critic first-layer input width must be 212")
    new_actor = {
        **actor,
        "params": {
            **actor["params"],
            "hidden_0": {
                **actor["params"]["hidden_0"],
                "kernel": _insert_zero_feature_rows(
                    actor_kernel,
                    legacy_prefix_width=LEGACY_ACTOR_OBSERVATION_WIDTH,
                    extra_width=extra_width,
                    xp=xp,
                ),
            },
        },
    }
    new_critic = {
        **critic,
        "params": {
            **critic["params"],
            "hidden_0": {
                **critic["params"]["hidden_0"],
                "kernel": _insert_zero_feature_rows(
                    critic_kernel,
                    legacy_prefix_width=LEGACY_ACTOR_OBSERVATION_WIDTH,
                    extra_width=extra_width,
                    xp=xp,
                ),
            },
        },
    }
    audit = {
        "source_actor_width": LEGACY_ACTOR_OBSERVATION_WIDTH,
        "target_actor_width": H4_ACTOR_OBSERVATION_WIDTH,
        "source_critic_width": 212,
        "target_critic_width": 227,
        "inserted_feature_count": extra_width,
        "insert_offset": LEGACY_ACTOR_OBSERVATION_WIDTH,
        "actor_old_rows_copied": True,
        "critic_privileged_tail_semantically_preserved": True,
        "new_first_layer_rows_zero_initialized": True,
        "new_normalizer_mean": 0.0,
        "new_normalizer_std": 1.0,
        "legacy_summed_variance_repair_count": legacy_variance_repair_count,
        "legacy_summed_variance_min_before": legacy_variance_min,
        "legacy_summed_variance_clipped_to_zero": bool(
            legacy_variance_repair_count
        ),
        "passed": True,
    }
    return [new_normalizer, new_actor, new_critic], audit


def _audit_checkpoint_observation_structure(
    checkpoint_params: Sequence[Any],
    *,
    actor_observation_width: int,
    allow_legacy_summed_variance_repair: bool = False,
) -> dict[str, Any]:
    """Validate every restore-critical H4 PPO structure before handing it to Brax."""

    if not isinstance(checkpoint_params, (list, tuple)) or len(checkpoint_params) != 3:
        raise ValueError("PPO checkpoint must contain exactly three parameter groups")
    normalizer, actor, critic = checkpoint_params
    expected_actor = int(actor_observation_width)
    expected_critic = expected_actor + 111
    required_normalizer = ("mean", "std", "summed_variance", "count", "replace")
    if any(not hasattr(normalizer, name) for name in required_normalizer):
        raise ValueError("checkpoint running-statistics structure drifted")

    for mapping_name in ("mean", "std", "summed_variance"):
        mapping = getattr(normalizer, mapping_name)
        if not isinstance(mapping, Mapping) or set(mapping) != {
            "state",
            "privileged_state",
        }:
            raise ValueError(f"normalizer {mapping_name} keys drifted")
        for key, width in (
            ("state", expected_actor),
            ("privileged_state", expected_critic),
        ):
            array = np.asarray(mapping[key])
            if array.shape != (width,) or array.dtype != np.dtype(np.float32):
                raise ValueError(
                    f"normalizer {mapping_name}.{key} must be float32 width {width}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"normalizer {mapping_name}.{key} is non-finite")
            if mapping_name == "std" and np.any(array <= 0.0):
                raise ValueError(f"normalizer std.{key} must be strictly positive")
            if mapping_name == "summed_variance" and np.any(array < 0.0):
                legacy_repair_allowed = (
                    allow_legacy_summed_variance_repair
                    and expected_actor == LEGACY_ACTOR_OBSERVATION_WIDTH
                    and expected_critic == 212
                    and float(np.min(array)) >= -0.01
                )
                if not legacy_repair_allowed:
                    raise ValueError(
                        f"normalizer summed_variance.{key} must be nonnegative"
                    )

    try:
        count_hi = np.asarray(normalizer.count.hi)
        count_lo = np.asarray(normalizer.count.lo)
    except AttributeError as exc:
        raise ValueError("normalizer count must expose UInt64 hi/lo") from exc
    for name, value in (("hi", count_hi), ("lo", count_lo)):
        if value.shape != () or value.dtype != np.dtype(np.uint32):
            raise ValueError(f"normalizer count.{name} must be an exact uint32 scalar")
    if int(count_hi) == 0 and int(count_lo) == 0:
        raise ValueError("normalizer count must be positive")

    def audit_network(
        name: str, network: Any, layer_shapes: Sequence[tuple[int, int]]
    ) -> tuple[int, int]:
        if not isinstance(network, Mapping) or set(network) != {"params"}:
            raise ValueError(f"{name} parameter-group structure drifted")
        params = network["params"]
        expected_keys = {f"hidden_{index}" for index in range(4)}
        if not isinstance(params, Mapping) or set(params) != expected_keys:
            raise ValueError(f"{name} layer keys drifted")
        for index, (input_width, output_width) in enumerate(layer_shapes):
            layer_name = f"hidden_{index}"
            layer = params[layer_name]
            if not isinstance(layer, Mapping) or set(layer) != {"kernel", "bias"}:
                raise ValueError(f"{name} {layer_name} structure drifted")
            kernel = np.asarray(layer["kernel"])
            bias = np.asarray(layer["bias"])
            if (
                kernel.shape != (input_width, output_width)
                or bias.shape != (output_width,)
                or kernel.dtype != np.dtype(np.float32)
                or bias.dtype != np.dtype(np.float32)
            ):
                raise ValueError(f"{name} {layer_name} shape/dtype drifted")
            if not np.all(np.isfinite(kernel)) or not np.all(np.isfinite(bias)):
                raise ValueError(f"{name} {layer_name} contains non-finite values")
        return layer_shapes[0][1], len(layer_shapes) * 2

    actor_shapes = (
        (expected_actor, 512),
        (512, 256),
        (256, 128),
        (128, 28),
    )
    critic_shapes = (
        (expected_critic, 512),
        (512, 256),
        (256, 128),
        (128, 1),
    )
    actor_hidden, actor_leaves = audit_network("actor", actor, actor_shapes)
    critic_hidden, critic_leaves = audit_network("critic", critic, critic_shapes)
    return {
        "source_actor_width": expected_actor,
        "target_actor_width": expected_actor,
        "critic_width": expected_critic,
        "normalizer_state_width": expected_actor,
        "normalizer_privileged_width": expected_critic,
        "actor_hidden_width": actor_hidden,
        "critic_hidden_width": critic_hidden,
        "actor_numeric_leaf_count": actor_leaves,
        "critic_numeric_leaf_count": critic_leaves,
        "all_restore_leaves_finite": True,
        "restore_structure_validated": True,
        "passed": True,
    }


def require_checkpoint_observation_compatibility(
    checkpoint_params: Sequence[Any],
    *,
    actor_observation_width: int,
    allow_explicit_v22_transplant: bool = False,
    xp: Any = np,
) -> tuple[list[Any], dict[str, Any]]:
    """Fail closed on observation mismatch unless v22 transplant is explicit."""

    if not isinstance(checkpoint_params, (list, tuple)) or len(checkpoint_params) != 3:
        raise ValueError("PPO checkpoint must contain exactly three parameter groups")
    try:
        source_width = int(
            np.asarray(
                checkpoint_params[1]["params"]["hidden_0"]["kernel"]
            ).shape[0]
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("cannot resolve checkpoint actor input width") from exc
    target_width = int(actor_observation_width)
    if source_width == target_width:
        audit = _audit_checkpoint_observation_structure(
            checkpoint_params, actor_observation_width=target_width
        )
        return list(checkpoint_params), {**audit, "transplant_applied": False}
    if not allow_explicit_v22_transplant:
        raise ValueError(
            f"checkpoint actor width {source_width} != environment width "
            f"{target_width}; explicit v22 transplant is required"
        )
    if source_width != LEGACY_ACTOR_OBSERVATION_WIDTH or target_width != H4_ACTOR_OBSERVATION_WIDTH:
        raise ValueError("only the audited v22 101-to-H4 116 transplant is supported")
    _audit_checkpoint_observation_structure(
        checkpoint_params,
        actor_observation_width=LEGACY_ACTOR_OBSERVATION_WIDTH,
        allow_legacy_summed_variance_repair=True,
    )
    transplanted, audit = transplant_v22_checkpoint_to_h4_observation(
        checkpoint_params, xp=xp
    )
    final_audit = _audit_checkpoint_observation_structure(
        transplanted, actor_observation_width=H4_ACTOR_OBSERVATION_WIDTH
    )
    # Keep the transplant origin (101/212) distinct from the independently
    # audited target widths (116/227).  The target audit contributes its
    # detailed structure fields, while the transplant audit owns source/target
    # provenance.
    return transplanted, {**final_audit, **audit, "transplant_applied": True}


def make_h4_aligned_environment_class(
    *,
    legacy_environment_class: type,
    stack: Mapping[str, Any],
    physical_command_sampler: Callable[[Any], Any],
    policy_observation_mapper: Callable[[Any], Any],
    reward_scales: H4QualityRewardScales = H4QualityRewardScales(),
    reset_noise_multiplier: float = 1.0,
    reverse_teacher_cycle_hz: float = 1.75,
    reverse_teacher_target_table: Any | None = None,
    reverse_teacher_phase_advance_bins: float | None = None,
    reverse_teacher_entry_phase_bins: float = 0.0,
    include_h4_actor_observables: bool = False,
    legacy_reward_config_overrides: Mapping[str, float] | None = None,
    forward_v4_substep_contact: bool = False,
    v4_substep_collector_trace_capture: bool = False,
    forward_iteration_v6_contact_abort_island_only: bool = False,
    reverse_iteration_v6_absolute_full_leg_targets: bool = False,
    h5_absolute_target_routing: bool = False,
    h5_target_domain: str | None = None,
    h5_v3_command_conditioned_se2_alignment: bool = False,
    h5_v3_substep_contact_alignment: bool = False,
    h5_v3_substep_preflight_telemetry: bool = False,
    h5_v3_substep_preflight_fixed_quality_replay: bool = False,
    h5_seed_params: Any | None = None,
    h5_seed_target_table: Any | None = None,
    h5_seed_bc_anneal_control_steps: float = 250.0,
    h5_seed_teacher_mode: str = "table",
    h5_seed_residual_gain: float = 0.0,
    h5_seed_teacher_reverse_command_contract: bool = False,
) -> type:
    """Wrap the frozen Joystick subclass without editing its source tree.

    The factory is lazy: importing this module never imports JAX, MuJoCo, or
    the frozen training checkout.  The returned class intercepts the single
    MJX physics call made by the frozen step method, applies the H4 target
    guard there, then synchronizes both target-bearing observations with the
    target actually used by physics.
    """

    required = {"jax", "jp", "joystick"}
    missing = required - set(stack)
    if missing:
        raise ValueError(f"training stack is missing {sorted(missing)}")
    if not callable(physical_command_sampler) or not callable(
        policy_observation_mapper
    ):
        raise ValueError("physical sampler and policy mapper must be callable")
    multiplier = float(reset_noise_multiplier)
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("reset_noise_multiplier must be finite and non-negative")
    teacher_cycle_hz = float(reverse_teacher_cycle_hz)
    if not np.isfinite(teacher_cycle_hz) or not 1.5 <= teacher_cycle_hz <= 2.0:
        raise ValueError("reverse_teacher_cycle_hz must be in [1.5, 2.0]")
    if not isinstance(include_h4_actor_observables, (bool, np.bool_)):
        raise ValueError("include_h4_actor_observables must be boolean")
    if not isinstance(forward_v4_substep_contact, (bool, np.bool_)):
        raise ValueError("forward_v4_substep_contact must be boolean")
    if not isinstance(v4_substep_collector_trace_capture, (bool, np.bool_)):
        raise ValueError("v4_substep_collector_trace_capture must be boolean")
    if not isinstance(
        forward_iteration_v6_contact_abort_island_only, (bool, np.bool_)
    ):
        raise ValueError(
            "forward_iteration_v6_contact_abort_island_only must be boolean"
        )
    if not isinstance(
        reverse_iteration_v6_absolute_full_leg_targets, (bool, np.bool_)
    ):
        raise ValueError(
            "reverse_iteration_v6_absolute_full_leg_targets must be boolean"
        )
    if not isinstance(h5_absolute_target_routing, (bool, np.bool_)):
        raise ValueError("h5_absolute_target_routing must be boolean")
    if not isinstance(h5_v3_command_conditioned_se2_alignment, (bool, np.bool_)):
        raise ValueError("h5_v3_command_conditioned_se2_alignment must be boolean")
    if not isinstance(h5_v3_substep_contact_alignment, (bool, np.bool_)):
        raise ValueError("h5_v3_substep_contact_alignment must be boolean")
    if not isinstance(h5_v3_substep_preflight_telemetry, (bool, np.bool_)):
        raise ValueError("h5_v3_substep_preflight_telemetry must be boolean")
    if not isinstance(h5_v3_substep_preflight_fixed_quality_replay, (bool, np.bool_)):
        raise ValueError("h5_v3_substep_preflight_fixed_quality_replay must be boolean")
    if h5_absolute_target_routing:
        if h5_target_domain not in {"planar", "reverse", "unified"}:
            raise ValueError(
                "H5 absolute target routing requires a planar, reverse, or unified domain"
            )
    elif h5_target_domain is not None:
        raise ValueError("h5_target_domain requires H5 absolute target routing")
    if h5_v3_command_conditioned_se2_alignment and not (
        h5_absolute_target_routing and h5_target_domain == "unified"
    ):
        raise ValueError(
            "H5 V3 command-conditioned SE(2) alignment requires unified H5 routing"
        )
    if h5_v3_substep_contact_alignment and not (
        h5_v3_command_conditioned_se2_alignment
        and forward_v4_substep_contact
        and h5_absolute_target_routing
        and h5_target_domain == "unified"
    ):
        raise ValueError(
            "H5 V3 substep contact alignment requires V3 SE(2), unified H5 routing, and V4 substep contact"
        )
    if v4_substep_collector_trace_capture and not forward_v4_substep_contact:
        raise ValueError("V4 collector trace capture requires V4 substep contact")
    if v4_substep_collector_trace_capture and h5_v3_substep_contact_alignment:
        raise ValueError(
            "V4 collector trace capture is sidecar-only and cannot share the monolithic H5 step"
        )
    if h5_v3_substep_preflight_telemetry and not h5_v3_substep_contact_alignment:
        raise ValueError(
            "H5 V3 substep preflight telemetry requires substep contact alignment"
        )
    if h5_v3_substep_preflight_fixed_quality_replay and not (
        h5_v3_substep_contact_alignment and h5_v3_substep_preflight_telemetry
    ):
        raise ValueError(
            "H5 fixed quality replay is a substep preflight-only diagnostic"
        )
    if h5_seed_params is not None and not (
        h5_absolute_target_routing
        and h5_target_domain in {"reverse", "unified"}
    ):
        raise ValueError(
            "H5 target-space seed BC is valid only for reverse or unified H5 routing"
        )
    if h5_seed_target_table is not None and h5_seed_params is None:
        raise ValueError("H5 seed target table requires H5 seed parameters")
    if h5_seed_teacher_mode not in {"table", "actor", "adaptive_residual"}:
        raise ValueError(
            "H5 seed teacher mode must be table, actor, or adaptive_residual"
        )
    if h5_seed_params is None and h5_seed_teacher_mode != "table":
        raise ValueError("H5 actor teachers require H5 seed parameters")
    if (
        h5_seed_params is not None
        and h5_seed_teacher_mode == "table"
        and h5_seed_target_table is None
    ):
        raise ValueError("H5 table teacher mode requires a target table")
    if (
        h5_seed_params is not None
        and h5_seed_teacher_mode == "adaptive_residual"
        and h5_seed_target_table is None
    ):
        raise ValueError("H5 adaptive residual teacher requires a target table")
    if (
        not np.isfinite(float(h5_seed_residual_gain))
        or not 0.0 <= float(h5_seed_residual_gain) <= 1.0
    ):
        raise ValueError("H5 seed residual gain must be finite and in [0, 1]")
    h5_seed_target_table_array = None
    if h5_seed_target_table is not None:
        h5_seed_target_table_array = np.asarray(h5_seed_target_table, dtype=np.float32)
        if h5_seed_target_table_array.shape != (54, ACTION_COUNT):
            raise ValueError("H5 seed target table must have shape (54, 14)")
        if not np.all(np.isfinite(h5_seed_target_table_array)):
            raise ValueError("H5 seed target table must be finite")
        if not np.array_equal(
            h5_seed_target_table_array[:, 5:9],
            np.zeros((54, 4), dtype=np.float32),
        ):
            raise ValueError("H5 seed target table head channels must be exact zero")
    if (
        not np.isfinite(float(h5_seed_bc_anneal_control_steps))
        or float(h5_seed_bc_anneal_control_steps) <= 0.0
    ):
        raise ValueError("h5_seed_bc_anneal_control_steps must be positive")
    if h5_absolute_target_routing and (
        forward_iteration_v6_contact_abort_island_only
        or reverse_iteration_v6_absolute_full_leg_targets
    ):
        raise ValueError("H5 target routing is exclusive with H4 iteration-v6 modes")
    if (
        forward_iteration_v6_contact_abort_island_only
        and reverse_iteration_v6_absolute_full_leg_targets
    ):
        raise ValueError("forward and reverse iteration-v6 families are exclusive")
    if (
        forward_iteration_v6_contact_abort_island_only
        and not forward_v4_substep_contact
    ):
        raise ValueError(
            "forward iteration-v6 island-only routing requires v4 substep contact"
        )
    legacy_overrides = dict(legacy_reward_config_overrides or {})
    allowed_legacy_overrides = {
        "target_imitation",
        "contact_imitation",
        "tracking_sigma",
    }
    if set(legacy_overrides) - allowed_legacy_overrides:
        raise ValueError("unsupported legacy reward-config override")
    if any(not np.isfinite(float(value)) for value in legacy_overrides.values()):
        raise ValueError("legacy reward-config overrides must be finite")
    if (
        "target_imitation" in legacy_overrides
        and legacy_overrides["target_imitation"] > 0.0
    ):
        raise ValueError("target imitation must remain a non-positive cost")
    if (
        "contact_imitation" in legacy_overrides
        and legacy_overrides["contact_imitation"] < 0.0
    ):
        raise ValueError("contact imitation must remain a non-negative reward")
    if (
        "tracking_sigma" in legacy_overrides
        and legacy_overrides["tracking_sigma"] <= 0.0
    ):
        raise ValueError("tracking sigma must be positive")
    teacher_table_values: np.ndarray | None = None
    teacher_phase_advance: float | None = None
    teacher_entry_phase = float(reverse_teacher_entry_phase_bins)
    if reverse_teacher_target_table is not None:
        teacher_table_values = np.asarray(
            reverse_teacher_target_table, dtype=np.float64
        )
        if (
            teacher_table_values.ndim != 2
            or teacher_table_values.shape[0] < 2
            or teacher_table_values.shape[1] != ACTION_COUNT
            or not np.all(np.isfinite(teacher_table_values))
        ):
            raise ValueError("reverse teacher table must be finite Nx14 with N >= 2")
        if not np.array_equal(
            teacher_table_values[:, np.asarray(HEAD_ACTION_INDICES)],
            np.zeros((teacher_table_values.shape[0], len(HEAD_ACTION_INDICES))),
        ):
            raise ValueError("reverse teacher table must lock every head target to zero")
        if reverse_teacher_phase_advance_bins is None:
            raise ValueError("reverse teacher table requires phase advance")
        teacher_phase_advance = float(reverse_teacher_phase_advance_bins)
        expected_advance = (
            teacher_cycle_hz
            * teacher_table_values.shape[0]
            * CONTROL_FIRST_STARTUP_DT_S
        )
        if (
            not np.isfinite(teacher_phase_advance)
            or teacher_phase_advance <= 0.0
            or not np.isclose(
                teacher_phase_advance, expected_advance, atol=1.0e-9, rtol=0.0
            )
        ):
            raise ValueError("reverse teacher phase advance/cadence mismatch")
        if not np.isfinite(teacher_entry_phase):
            raise ValueError("reverse teacher entry phase must be finite")
        teacher_entry_phase %= teacher_table_values.shape[0]
    elif reverse_teacher_phase_advance_bins is not None:
        raise ValueError("reverse teacher phase advance requires a target table")
    scale_values = reward_scales.as_reward_scale_dict(
        include_h5_substep_contact_alignment=h5_v3_substep_contact_alignment
    )
    if forward_iteration_v6_contact_abort_island_only and not np.isclose(
        scale_values["h4_contact_pulse_40ms"], -1.0, atol=0.0, rtol=0.0
    ):
        raise ValueError(
            "forward iteration-v6 contact-abort scale must remain exactly -1"
        )
    if reverse_iteration_v6_absolute_full_leg_targets and teacher_table_values is None:
        raise ValueError(
            "reverse iteration-v6 teacher-timing mode requires a teacher table"
        )
    if h5_v3_substep_contact_alignment and not all(
        np.isclose(scale_values[name], -1.0, atol=0.0, rtol=0.0)
        for name in (
            "h5_all_substep_strict20ms_slip_rms",
            "h5_all_substep_slip_tail",
            "h5_all_substep_force_tail",
        )
    ):
        raise ValueError(
            "H5 V3 substep contact alignment requires all three all-substep scales to be exactly -1"
        )
    jax = stack["jax"]
    jp = stack["jp"]
    joystick = stack["joystick"]
    h5_target_decoder = None
    if h5_absolute_target_routing:
        # Import only at the H5 factory boundary.  This keeps the historical
        # H4 module's import path lazy while binding the exact H5 decoder used
        # by the routed evaluator and target-space composition contract.
        from .h5_target_contract import h5_decode_absolute_targets

        h5_target_decoder = h5_decode_absolute_targets
    h5_seed_target_from_observation = None
    if h5_seed_params is not None:
        try:
            seed_normalizer, seed_actor, _seed_critic = h5_seed_params
            seed_mean = jp.asarray(seed_normalizer.mean["state"])
            seed_std = jp.asarray(seed_normalizer.std["state"])
            seed_layers = seed_actor["params"]
            seed_layer_values = tuple(
                (
                    jp.asarray(seed_layers[f"hidden_{index}"]["kernel"]),
                    jp.asarray(seed_layers[f"hidden_{index}"]["bias"]),
                )
                for index in range(4)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("H5 target-space seed actor structure is invalid") from exc

        table_j = (
            jp.asarray(h5_seed_target_table_array)
            if h5_seed_target_table_array is not None
            else None
        )

        def h5_seed_target_from_observation(observation):
            values = jp.asarray(observation)
            teacher_values = values
            if h5_seed_teacher_reverse_command_contract:
                # Distilled reverse teachers were trained with the historical
                # reverse mapper (vx scale 1), while the unified actor uses
                # the explicit vx scale 2.  Adapt only the private teacher
                # query; the student still sees the unified observation.
                teacher_values = teacher_values.at[6].set(
                    teacher_values[6] * jp.asarray(0.5, dtype=teacher_values.dtype)
                )
            table_target = None
            if table_j is not None:
                values = teacher_values
                angle = jp.arctan2(values[100], values[99])
                angle = jp.where(angle < 0.0, angle + 2.0 * jp.pi, angle)
                phase_steps = float(table_j.shape[0]) / 2.0
                phase = angle / (2.0 * jp.pi) * phase_steps
                wrapped = jp.mod(2.0 * phase, float(table_j.shape[0]))
                index = jp.asarray(jp.floor(wrapped), dtype=jp.int32)
                fraction = wrapped - jp.floor(wrapped)
                next_index = jp.mod(index + 1, table_j.shape[0])
                table_target = (
                    (1.0 - fraction) * table_j[index]
                    + fraction * table_j[next_index]
                )
            if h5_seed_teacher_mode == "table":
                return table_target
            hidden = (teacher_values - seed_mean) / seed_std
            for index, (kernel, bias) in enumerate(seed_layer_values):
                hidden = hidden @ kernel + bias
                if index != 3:
                    hidden = jax.nn.silu(hidden)
            seed_action = jp.tanh(hidden[:ACTION_COUNT])
            seed_domain = (
                h5_target_domain
                if h5_target_domain in {"reverse", "unified"}
                else "reverse"
            )
            actor_target = h5_target_decoder(seed_action, domain=seed_domain, xp=jp)
            if h5_seed_teacher_mode == "actor":
                return actor_target
            _lower, _upper, safe_init = contract_target_vectors(xp=jp)
            return margin_clip_targets(
                table_target
                + float(h5_seed_residual_gain) * (actor_target - safe_init),
                xp=jp,
            )

    h5_routing_enabled = bool(h5_absolute_target_routing)
    h5_domain_value = h5_target_domain
    h5_v3_se2_alignment_enabled = bool(h5_v3_command_conditioned_se2_alignment)
    h5_v3_substep_alignment_enabled = bool(h5_v3_substep_contact_alignment)
    v4_substep_collector_trace_capture_enabled = bool(
        v4_substep_collector_trace_capture
    )
    h5_v3_substep_preflight_telemetry_enabled = bool(
        h5_v3_substep_preflight_telemetry
    )
    h5_v3_substep_preflight_fixed_quality_replay_enabled = bool(
        h5_v3_substep_preflight_fixed_quality_replay
    )
    h5_all_substep_quality_update = None
    initialize_h5_multiwindow_debounce = None
    h5_t1_fixed_quality_replay = None
    if h5_v3_substep_alignment_enabled:
        # h5_substep imports H4's strict constants, so resolve this pure helper
        # only after this module has completed its top-level import.
        from .h5_substep_contact_alignment import (
            h5_all_substep_quality_update as _h5_all_substep_quality_update,
            initialize_h5_multiwindow_debounce as _initialize_h5_multiwindow_debounce,
            h5_v3_t1_fixed_quality_replay as _h5_t1_fixed_quality_replay,
        )

        h5_all_substep_quality_update = _h5_all_substep_quality_update
        initialize_h5_multiwindow_debounce = _initialize_h5_multiwindow_debounce
        h5_t1_fixed_quality_replay = _h5_t1_fixed_quality_replay
    v4_compiled_authority_assertion = (
        make_v4_compiled_single_authority_assertion(jax, jp)
        if forward_v4_substep_contact
        else None
    )
    v6_forward_compiled_routing_assertion = (
        make_v6_compiled_invariant_assertion(
            jax,
            jp,
            label=FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID,
        )
        if forward_iteration_v6_contact_abort_island_only
        else None
    )
    v6_reverse_compiled_decoder_assertion = (
        make_v6_compiled_invariant_assertion(
            jax,
            jp,
            label=REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID,
        )
        if reverse_iteration_v6_absolute_full_leg_targets
        else None
    )
    try:
        from mujoco.mjx._src import support as mjx_support
    except ImportError as exc:  # pragma: no cover - only exercised in WSL/MJX
        raise RuntimeError("MJX contact-force support is required for H4") from exc

    class H4AlignedEnvironment(legacy_environment_class):
        """Frozen-source task with exact H4 target and gait-quality semantics."""

        h4_alignment_schema_version = 1
        h4_forward_v4_substep_contact = bool(forward_v4_substep_contact)
        v4_substep_collector_trace_capture = (
            v4_substep_collector_trace_capture_enabled
        )
        h4_forward_iteration_v6_contact_abort_island_only = bool(
            forward_iteration_v6_contact_abort_island_only
        )
        h4_reverse_iteration_v6_absolute_full_leg_targets = bool(
            reverse_iteration_v6_absolute_full_leg_targets
        )
        h5_absolute_target_routing = h5_routing_enabled
        h5_target_domain = h5_domain_value
        h5_v3_command_conditioned_se2_alignment = h5_v3_se2_alignment_enabled
        h5_v3_substep_contact_alignment = h5_v3_substep_alignment_enabled
        h5_target_space_contract_id = (
            "OPEN_DUCK_MINI_H5_TARGET_SPACE_ROUTING_V1"
            if h5_absolute_target_routing
            else None
        )
        h5_targetspace_seed_bc_enabled = h5_seed_params is not None
        h5_targetspace_seed_teacher_mode = str(h5_seed_teacher_mode)
        h5_targetspace_seed_residual_gain = float(h5_seed_residual_gain)
        h5_targetspace_seed_teacher_reverse_command_contract = bool(
            h5_seed_teacher_reverse_command_contract
        )
        h5_targetspace_seed_bc_anneal_control_steps = float(
            h5_seed_bc_anneal_control_steps
        )
        h4_forward_iteration_v6_contract_id = (
            FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID
            if forward_iteration_v6_contact_abort_island_only
            else None
        )
        h4_reverse_iteration_v6_contract_id = (
            REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID
            if reverse_iteration_v6_absolute_full_leg_targets
            else None
        )
        h4_forward_iteration_v6_compiled_assertion_bound = bool(
            forward_iteration_v6_contact_abort_island_only
            and v6_forward_compiled_routing_assertion is not None
        )
        h4_reverse_iteration_v6_compiled_assertion_bound = bool(
            reverse_iteration_v6_absolute_full_leg_targets
            and v6_reverse_compiled_decoder_assertion is not None
        )
        h4_forward_iteration_v6_off_gap_reward_contribution = 0.0
        h4_forward_iteration_v6_contact_pulse_reward_scale = (
            float(scale_values["h4_contact_pulse_40ms"])
            if forward_iteration_v6_contact_abort_island_only
            else None
        )
        h4_reverse_iteration_v6_directional_span_fraction = (
            REVERSE_ITERATION_V6_DIRECTIONAL_SPAN_FRACTION
        )
        h4_reverse_iteration_v6_base_action_span_rad = (
            REVERSE_ITERATION_V6_BASE_ACTION_SPAN_RAD
        )
        h4_reverse_iteration_v6_residual_authority_scale = 0.0
        h4_reverse_iteration_v6_teacher_target_contribution = 0.0

        def __init__(self):
            super().__init__()
            if forward_v4_substep_contact:
                if not np.isclose(
                    float(self._config.sim_dt),
                    V4_PHYSICS_SUBSTEP_DT_S,
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise ValueError("forward v4 requires exact 2 ms simulation dt")
                if not np.isclose(
                    float(self._config.ctrl_dt),
                    V4_PHYSICS_SUBSTEP_DT_S * V4_CONTROL_SUBSTEP_COUNT,
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise ValueError("forward v4 requires exact 20 ms control dt")
            # randint's maxval is exclusive, so [0, 1) is exact zero delay.
            self._config.noise_config.action_min_delay = 0
            self._config.noise_config.action_max_delay = 1
            self._config.max_motor_velocity = TARGET_SLEW_LIMIT_RAD_PER_S
            self._h5_seed_bc_anneal_control_steps = float(
                h5_seed_bc_anneal_control_steps
            )
            scales = self._config.reward_config.scales
            if "target_imitation" in legacy_overrides:
                scales.target_imitation = float(
                    legacy_overrides["target_imitation"]
                )
            if "contact_imitation" in legacy_overrides:
                scales.contact_imitation = float(
                    legacy_overrides["contact_imitation"]
                )
            if "tracking_sigma" in legacy_overrides:
                self._config.reward_config.tracking_sigma = float(
                    legacy_overrides["tracking_sigma"]
                )
            if h5_absolute_target_routing:
                # H5 owns absolute target generation and uses the legacy
                # reference only as an observation/phase prior.  Legacy joint
                # imitation and boolean contact matching otherwise create a
                # static standing optimum. H5 motion is rewarded by
                # active-axis tracking plus bounded signed progress below.
                scales.imitation = 0.0
            if reverse_iteration_v6_absolute_full_leg_targets:
                # v6 retains phase/contact timing as an observation prior, not
                # as an imitation objective.  Both legacy teacher rewards are
                # structurally zero even if the frozen source defaults differ.
                scales.target_imitation = 0.0
                scales.contact_imitation = 0.0
            # Disable the site-origin/boolean-contact proxy.  H4 exposes the
            # force-qualified contact-point term under a distinct name.
            scales.feet_slip = 0.0
            with scales.unlocked():
                for name, value in scale_values.items():
                    scales[name] = float(value)
            root_body_id = int(self._torso_body_id)
            body_parent = np.asarray(self._mj_model.body_parentid, dtype=np.int32)
            robot_body_mask = np.zeros(self._mj_model.nbody, dtype=bool)
            for body_id in range(self._mj_model.nbody):
                cursor = body_id
                while cursor > 0 and cursor != root_body_id:
                    cursor = int(body_parent[cursor])
                robot_body_mask[body_id] = cursor == root_body_id
            self._h4_robot_body_mask = jp.asarray(robot_body_mask)
            nominal_weight = robot_body_weight_n(
                np.asarray(self._mj_model.body_mass),
                robot_body_mask,
                np.asarray(self._mj_model.opt.gravity),
            )
            if float(nominal_weight) <= 0.0:
                raise ValueError("H4 force normalization requires positive weight")
            self._h4_geom_bodyid = jp.asarray(self._mj_model.geom_bodyid)
            self._h4_reverse_teacher_table = (
                None
                if teacher_table_values is None
                else jp.asarray(teacher_table_values)
            )
            _, _, reverse_v6_safe_init = contract_target_vectors(xp=jp)
            self._h4_reverse_v6_safe_init = reverse_v6_safe_init
            if teacher_table_values is None:
                self._h4_reverse_teacher_phase_scale = 1.0
                self._h4_reverse_teacher_entry_source_phase = 0.0
                self._backward_phase_rate = (
                    teacher_cycle_hz
                    * float(self.PRM.nb_steps_in_period)
                    * CONTROL_FIRST_STARTUP_DT_S
                )
            else:
                self._h4_reverse_teacher_phase_scale = (
                    teacher_table_values.shape[0]
                    / float(self.PRM.nb_steps_in_period)
                )
                self._h4_reverse_teacher_entry_source_phase = (
                    teacher_entry_phase / self._h4_reverse_teacher_phase_scale
                )
                self._backward_phase_rate = (
                    teacher_phase_advance
                    / self._h4_reverse_teacher_phase_scale
                )

        def sample_command(self, rng):
            command = physical_command_sampler(rng)
            return _require_vector_shape(command, 7, "sampled physical command", xp=jp)

        def _h4_selected_teacher_actuator_target(self, table_phase):
            if self._h4_reverse_teacher_table is None:
                raise RuntimeError("selected reverse teacher table is not configured")
            table = self._h4_reverse_teacher_table
            wrapped = jp.mod(table_phase, table.shape[0])
            table_index = jp.floor(wrapped).astype(jp.int32)
            table_next = (table_index + 1) % table.shape[0]
            fraction = wrapped - jp.floor(wrapped)
            return (
                (1.0 - fraction) * table[table_index]
                + fraction * table[table_next]
            )

        def _get_optimized_backward_reference(self, phase):
            """Frozen reverse schedule, with v6 target slots made table-invariant."""

            if self._h4_reverse_teacher_table is not None:
                table_phase = jp.mod(
                    phase * self._h4_reverse_teacher_phase_scale,
                    self._h4_reverse_teacher_table.shape[0],
                )
                # The frozen reference contains non-actuator schedule channels
                # (including contact labels) beyond the selected 14-axis
                # teacher table.  Preserve those source channels at the same
                # normalized phase.  v6 neutralizes only the ten target slots;
                # older families retain the selected table target behavior.
                source_period = self.PRM.nb_steps_in_period
                source_phase = jp.mod(phase, source_period)
                source_index = jp.floor(source_phase).astype(jp.int32)
                source_next = (source_index + 1) % source_period
                source_fraction = source_phase - jp.floor(source_phase)
                source_frame = (
                    (1.0 - source_fraction)
                    * self._backward_reference_frames[source_index]
                    + source_fraction
                    * self._backward_reference_frames[source_next]
                )
                if reverse_iteration_v6_absolute_full_leg_targets:
                    return reverse_iteration_v6_teacher_timing_only_reference(
                        source_frame,
                        self._backward_joint_indices,
                        self._h4_reverse_v6_safe_init,
                        xp=jp,
                    )
                actuator_target = self._h4_selected_teacher_actuator_target(
                    table_phase
                )
                return source_frame.at[self._backward_joint_indices].set(
                    actuator_target[jp.asarray(LEG_ACTION_INDICES)]
                )

            period = self.PRM.nb_steps_in_period
            wrapped = jp.mod(phase, period)
            frame_index = jp.floor(wrapped).astype(jp.int32)
            next_index = (frame_index + 1) % period
            fraction = wrapped - jp.floor(wrapped)
            frame = (
                (1.0 - fraction) * self._backward_reference_frames[frame_index]
                + fraction * self._backward_reference_frames[next_index]
            )
            deviation = (
                (1.0 - fraction) * self._backward_leg_deviations[frame_index]
                + fraction * self._backward_leg_deviations[next_index]
            )
            leg_target = (
                self._backward_leg_means
                + self._backward_gait_biases
                + self._backward_gait_scales * deviation
            )
            leg_target = jp.clip(
                leg_target,
                self._backward_leg_lowers,
                self._backward_leg_uppers,
            )
            return frame.at[self._backward_joint_indices].set(leg_target)

        def _get_obs(self, data, info, contact):
            physical = jp.asarray(info["command"])
            policy_command = _require_vector_shape(
                policy_observation_mapper(physical),
                7,
                "policy observation command",
                xp=jp,
            )
            info["h4_physical_command"] = physical
            info["h4_policy_observation_command"] = policy_command
            info["command"] = policy_command
            try:
                observation = super()._get_obs(data, info, contact)
            finally:
                info["command"] = physical
            if not include_h4_actor_observables:
                return observation
            previous_force_contact = info.get(
                "h4_previous_force_contact", jp.zeros(2, dtype=bool)
            )
            quality = self._h4_contact_observables(
                data, previous_force_contact
            )
            legacy_state = observation["state"]
            if legacy_state.shape != (LEGACY_ACTOR_OBSERVATION_WIDTH,):
                raise ValueError("frozen actor observation width drifted from 101")
            extra = jp.concatenate(
                (
                    physical[:3],
                    self.get_local_linvel(data),
                    self.get_gravity(data),
                    quality.normalized_force,
                    quality.contact.astype(legacy_state.dtype),
                    quality.tangential_speed_m_s,
                )
            )
            state_observation = jp.concatenate((legacy_state, extra))
            privileged_tail = observation["privileged_state"][
                LEGACY_ACTOR_OBSERVATION_WIDTH:
            ]
            return {
                "state": state_observation,
                "privileged_state": jp.concatenate(
                    (state_observation, privileged_tail)
                ),
            }

        def reset(self, rng):
            state = super().reset(rng)
            info = dict(state.info)
            if self._h4_reverse_teacher_table is not None:
                is_reverse = info["command"][0] < -0.02
                entry_source_phase = jp.asarray(
                    self._h4_reverse_teacher_entry_source_phase
                )
                info["imitation_i"] = jp.where(
                    is_reverse, entry_source_phase, info["imitation_i"]
                )
                phase_angle = (
                    info["imitation_i"]
                    / float(self.PRM.nb_steps_in_period)
                    * 2.0
                    * jp.pi
                )
                info["imitation_phase"] = jp.asarray(
                    (jp.cos(phase_angle), jp.sin(phase_angle))
                )
                info["current_reference_motion"] = jp.where(
                    is_reverse,
                    self._get_optimized_backward_reference(info["imitation_i"]),
                    info["current_reference_motion"],
                )
            if h5_absolute_target_routing and h5_target_domain in {"reverse", "unified"}:
                is_reverse = info["command"][0] < -0.02
                h5_entry_phase = jp.asarray(7.0, dtype=info["imitation_i"].dtype)
                info["imitation_i"] = jp.where(
                    is_reverse, h5_entry_phase, info["imitation_i"]
                )
                h5_phase_angle = (
                    info["imitation_i"]
                    / float(self.PRM.nb_steps_in_period)
                    * 2.0
                    * jp.pi
                )
                info["imitation_phase"] = jp.asarray(
                    (jp.cos(h5_phase_angle), jp.sin(h5_phase_angle))
                )
            info["rng"], noise_rng = jax.random.split(info["rng"])
            unit_noise = jax.random.uniform(
                noise_rng, (ACTION_COUNT,), minval=-1.0, maxval=1.0
            )
            reset_targets = project_reset_qpos(
                unit_noise, noise_multiplier=multiplier, xp=jp
            )
            qpos = state.data.qpos.at[
                self.get_actuator_joints_qpos_addr()
            ].set(reset_targets)
            data = joystick.mjx_env.init(
                self.mjx_model,
                qpos=qpos,
                qvel=state.data.qvel,
                ctrl=reset_targets,
            )
            info["motor_targets"] = reset_targets
            info["target_limit_violation"] = jp.zeros(())
            info["h4_previous_force_contact"] = jp.zeros(2, dtype=bool)
            info["h4_contact_run_length_ticks"] = jp.zeros(2, dtype=jp.int32)
            info["h4_last_single_support"] = jp.asarray(-1, dtype=jp.int32)
            info["h4_ticks_since_switch"] = jp.zeros((), dtype=jp.int32)
            info["h4_load_ema"] = jp.zeros(2)
            info["h4_stance_slip_integral_m"] = jp.zeros(2)
            info["h4_single_support_ema"] = jp.zeros(())
            info["h4_contact_duty_ema"] = jp.zeros(2)
            info["h4_touchdown_counts"] = jp.zeros(2)
            info["h4_guard_steps"] = jp.zeros((), dtype=jp.int32)
            info["h4_previous_desired_targets"] = reset_targets
            info["h4_pre_guard_raw_targets"] = reset_targets
            info["h4_guard_desired_targets"] = reset_targets
            if h5_seed_target_from_observation is not None:
                info["h5_seed_bc_target"] = jp.zeros(ACTION_COUNT)
                info["h5_seed_bc_weight"] = jp.zeros(())
                info["h5_seed_bc_loss"] = jp.zeros(())
                info["h5_seed_bc_weight_applied"] = jp.zeros(())
            if h5_absolute_target_routing:
                info["h5_decoded_targets"] = jp.zeros(ACTION_COUNT)
                info["h5_target_space_decoder_exact"] = jp.asarray(True)
            initial_reverse_teacher = (
                jp.asarray(False)
                if self._h4_reverse_teacher_table is None
                else info["command"][0] < -0.02
            )
            info["h4_reverse_teacher_precomposer_active"] = (
                initial_reverse_teacher
            )
            if (
                self._h4_reverse_teacher_table is None
                or reverse_iteration_v6_absolute_full_leg_targets
            ):
                initial_upstream_target = reset_targets
            else:
                entry_table_target = self._h4_selected_teacher_actuator_target(
                    teacher_entry_phase
                )
                initial_upstream_target = jp.where(
                    initial_reverse_teacher,
                    entry_table_target,
                    reset_targets,
                )
            info["h4_upstream_margin_targets"] = initial_upstream_target
            info["h4_previous_upstream_margin_targets"] = (
                initial_upstream_target
            )
            info["h4_upstream_signed_delta"] = jp.zeros(ACTION_COUNT)
            info["h4_previous_upstream_signed_delta"] = jp.zeros(ACTION_COUNT)
            info["h4_upstream_max_delta_rad"] = jp.zeros(())
            info["h4_reverse_teacher_precomposer_transition"] = jp.asarray(
                False
            )
            info["h4_reverse_teacher_entry_reset"] = initial_reverse_teacher
            info["h4_slew_feasibility_loss"] = jp.zeros(())
            info["h4_target_lag_loss"] = jp.zeros(())
            info["h4_pre_guard_max_delta_rad"] = jp.zeros(())
            info["h4_slip_rms_m_s"] = jp.zeros(())
            info["h4_per_foot_slip_m_s"] = jp.zeros(2)
            info["h4_normalized_force"] = jp.zeros(2)
            info["h4_force_contact"] = jp.zeros(2, dtype=bool)
            info["h4_alternation_event"] = jp.zeros((), dtype=bool)
            info["h4_alternation_quality"] = jp.zeros(())
            info["h4_load_imbalance"] = jp.zeros(())
            info["h4_force_load_imbalance"] = jp.zeros(())
            info["h4_single_support_band_loss"] = jp.zeros(())
            info["h4_touchdown_count_balance_loss"] = jp.zeros(())
            info["h4_flight"] = jp.zeros(())
            info["h4_total_normal_force_normalized"] = jp.zeros(())
            info["h4_total_normal_force_band_loss"] = jp.zeros(())
            info["h4_total_normal_force_tail_loss"] = jp.zeros(())
            info["h4_contact_pulse_liftoff_event"] = jp.zeros(2, dtype=bool)
            info["h4_contact_pulse_per_foot_loss"] = jp.zeros(2)
            info["h4_contact_pulse_40ms_loss"] = jp.zeros(())
            info["h4_slip_tail_loss"] = jp.zeros(())
            info["h4_stance_slip_budget_loss"] = jp.zeros(())
            info["h4_left_target_lag_loss"] = jp.zeros(())
            info["h4_right_target_lag_loss"] = jp.zeros(())
            info["h4_phase17_left_force_slip_loss"] = jp.zeros(())
            info["h4_phase17_left_knee_excess_loss"] = jp.zeros(())
            info["h4_phase17_opposite_leg_lag_loss"] = jp.zeros(())
            info["h4_forward_cross_drift_loss"] = jp.zeros(())
            info["h4_forward_yaw_rate_loss"] = jp.zeros(())
            info["h4_forward_heading_drift_loss"] = jp.zeros(())
            info["h4_reverse_speed_boundary_loss"] = jp.zeros(())
            info["h4_reverse_cross_drift_loss"] = jp.zeros(())
            info["h4_reverse_yaw_rate_loss"] = jp.zeros(())
            info["h4_reverse_heading_drift_loss"] = jp.zeros(())
            info["h4_reverse_phase_force_slip_loss"] = jp.zeros(())
            info["h4_reverse_contact_priority_reversal_lag_loss"] = jp.zeros(())
            if reverse_iteration_v6_absolute_full_leg_targets:
                info["h4_v6_reverse_decoder_action"] = jp.zeros(ACTION_COUNT)
                info["h4_v6_reverse_decoder_raw_targets"] = (
                    self._h4_reverse_v6_safe_init
                )
                info["h4_v6_reverse_decoder_margin_targets"] = margin_clip_targets(
                    self._h4_reverse_v6_safe_init, xp=jp
                )
                info["h4_v6_reverse_decoder_exact"] = jp.asarray(True)
                info["h4_v6_reverse_decoder_max_abs_error"] = jp.zeros(())
                info["h4_v6_reverse_decoder_leg_count"] = jp.asarray(
                    len(LEG_ACTION_INDICES), dtype=jp.int32
                )
                info["h4_v6_reverse_decoder_leg_count_exact"] = jp.asarray(True)
                info["h4_v6_reverse_decoder_head_zero_exact"] = jp.asarray(True)
                info[
                    "h4_v6_reverse_teacher_target_contribution_zero_exact"
                ] = jp.asarray(True)
                info["h4_v6_reverse_residual_authority_scale"] = jp.zeros(())
                info["h4_v6_reverse_decoder_all_finite"] = jp.asarray(True)
                info[
                    "h4_v6_reverse_decoder_margin_saturation_count"
                ] = jp.zeros((), dtype=jp.int32)
                info["h4_v6_reverse_decoder_action_clip_count"] = jp.zeros(
                    (), dtype=jp.int32
                )
                info["h4_v6_reverse_decoder_guard_lag_max_rad"] = jp.zeros(())
                info["h4_v6_reverse_precomposer_call_count"] = jp.zeros(
                    (), dtype=jp.int32
                )
                info["h4_v6_reverse_precomposer_call_count_exact"] = jp.asarray(
                    True
                )
                info["h4_v6_reverse_final_guard_call_count"] = jp.zeros(
                    (), dtype=jp.int32
                )
                info["h4_v6_reverse_final_guard_call_count_exact"] = jp.asarray(
                    True
                )
                info["h4_v6_reverse_decoder_violation"] = jp.asarray(False)
                info["h4_v6_reverse_decoder_assertion_token"] = jp.zeros(
                    (), dtype=jp.int32
                )
            if forward_v4_substep_contact:
                baseline_quality = self._h4_contact_observables(
                    data, jp.zeros(2, dtype=bool)
                )
                v4_state = initialize_v4_contact_telemetry(
                    baseline_quality.normalized_force, xp=jp
                )
                info["h4_previous_force_contact"] = (
                    v4_state.persistence.raw_contact
                )
                info["h4_v4_contact_telemetry_state"] = v4_state
                info["h4_v4_control_entry_raw_contact"] = (
                    v4_state.persistence.raw_contact
                )
                info["h4_v4_control_entry_qualified_contact"] = (
                    v4_state.persistence.qualified_contact
                )
                info["h4_v4_qualified_contact"] = (
                    v4_state.persistence.qualified_contact
                )
                info["h4_v4_pending_active"] = v4_state.persistence.pending_active
                info["h4_v4_pending_intervals"] = (
                    v4_state.persistence.pending_intervals
                )
                info["h4_touchdown_counts"] = v4_state.touchdown_counts
                info["h4_v4_confirmed_transition_count"] = jp.zeros(
                    2, dtype=jp.int32
                )
                info["h4_v4_touchdown_event_count"] = jp.zeros(
                    2, dtype=jp.int32
                )
                info["h4_v4_liftoff_event_count"] = jp.zeros(
                    2, dtype=jp.int32
                )
                info["h4_v4_aborted_contact_island_count"] = jp.zeros(
                    2, dtype=jp.int32
                )
                info["h4_v4_aborted_off_gap_count"] = jp.zeros(
                    2, dtype=jp.int32
                )
                info["h4_v4_aborted_contact_island_loss_sum"] = jp.zeros(2)
                info["h4_v4_aborted_off_gap_loss_sum"] = jp.zeros(2)
                info["h4_v4_alternation_event_count"] = jp.zeros(
                    (), dtype=jp.int32
                )
                info["h4_v4_alternation_quality_sum"] = jp.zeros(())
                info["h4_v4_single_authority_dynamic6_exact"] = jp.asarray(True)
                info["h4_v4_single_authority_dynamic6_max_abs_error"] = (
                    jp.zeros(())
                )
                info["h4_v4_single_authority_dynamic6_field_count"] = jp.asarray(
                    len(V4SavedDynamicState._fields), dtype=jp.int32
                )
                info["h4_v4_single_authority_dynamic6_field_count_exact"] = (
                    jp.asarray(True)
                )
                info["h4_v4_saved_dynamic6_substep_count"] = jp.asarray(
                    V4_CONTROL_SUBSTEP_COUNT, dtype=jp.int32
                )
                info["h4_v4_saved_dynamic6_field_count"] = jp.asarray(
                    len(V4SavedDynamicState._fields), dtype=jp.int32
                )
                info["h4_v4_saved_dynamic6_field_count_exact"] = jp.asarray(
                    True
                )
                info["h4_v4_saved_dynamic6_all_finite"] = jp.asarray(True)
                info["h4_v4_telemetry_force_shape_valid"] = jp.asarray(True)
                info["h4_v4_telemetry_force_all_finite"] = jp.asarray(True)
                info["h4_v4_single_authority_violation"] = jp.asarray(False)
                info["h4_v4_single_authority_assertion_token"] = jp.zeros(
                    (), dtype=jp.int32
                )
                info["h4_v4_terminal_pending_discarded"] = jp.asarray(False)
                if v4_substep_collector_trace_capture_enabled:
                    info["v4_substep_collector_reset_normalized_force"] = (
                        baseline_quality.normalized_force
                    )
                    info["v4_substep_collector_quality_trace"] = (
                        V4SubstepContactQualityTrajectory(
                            jp.broadcast_to(jp.asarray(data.time), (V4_CONTROL_SUBSTEP_COUNT,)),
                            jp.zeros(
                                (V4_CONTROL_SUBSTEP_COUNT, 2),
                                dtype=baseline_quality.normalized_force.dtype,
                            ),
                            jp.zeros(
                                (V4_CONTROL_SUBSTEP_COUNT, 2),
                                dtype=baseline_quality.tangential_speed_m_s.dtype,
                            ),
                        )
                    )
                if forward_iteration_v6_contact_abort_island_only:
                    info["h4_v6_forward_contact_abort_routing_exact"] = jp.asarray(
                        True
                    )
                    info["h4_v6_forward_contact_abort_island_loss"] = jp.zeros(())
                    info[
                        "h4_v6_forward_contact_abort_off_gap_diagnostic_loss"
                    ] = jp.zeros(())
                    info[
                        "h4_v6_forward_contact_abort_off_gap_reward_contribution"
                    ] = jp.zeros(())
                    info[
                        "h4_v6_forward_contact_abort_pulse_reward_scale"
                    ] = jp.asarray(
                        scale_values["h4_contact_pulse_40ms"], dtype=jp.float32
                    )
                    info[
                        "h4_v6_forward_contact_abort_routing_violation"
                    ] = jp.asarray(False)
                    info[
                        "h4_v6_forward_contact_abort_routing_assertion_token"
                    ] = jp.zeros((), dtype=jp.int32)
            floating_qpos = self.get_floating_base_qpos(data.qpos)
            info["h4_heading_reference_yaw"] = quaternion_yaw_wxyz(
                floating_qpos[3:7], xp=jp
            )
            if h5_v3_command_conditioned_se2_alignment:
                info["h5_v3_integrated_command_heading_yaw"] = info[
                    "h4_heading_reference_yaw"
                ]
                info["h5_v3_se2_cross_velocity_mps"] = jp.zeros(())
                info["h5_v3_se2_yaw_rate_residual_radps"] = jp.zeros(())
                info["h5_v3_se2_heading_residual_rad"] = jp.zeros(())
            if h5_v3_substep_alignment_enabled:
                if initialize_h5_multiwindow_debounce is None:
                    raise RuntimeError("H5 substep debounce initializer is unavailable")
                baseline_h5_quality = self._h4_contact_observables(
                    data, jp.zeros(2, dtype=bool)
                )
                info["h5_v3_substep_debounce"] = (
                    initialize_h5_multiwindow_debounce(
                        baseline_h5_quality.normalized_force, xp=jp
                    )
                )
                info["h5_v3_substep_strict20ms_slip_rms_loss"] = jp.zeros(())
                info["h5_v3_substep_slip_tail_loss"] = jp.zeros(())
                info["h5_v3_substep_force_tail_loss"] = jp.zeros(())
                info["h5_v3_substep_qualified_sample_count"] = jp.zeros(())
                info["h5_v3_substep_samples_finite"] = jp.asarray(True)
            info["h4_forward_heading_delta_rad"] = jp.zeros(())
            info["h4_reward_physical_command"] = jp.asarray(info["command"])
            if h5_absolute_target_routing:
                info["h5_target_space_decoder_exact"] = jp.asarray(True)
            contact = jp.asarray(
                [
                    joystick.geoms_colliding(
                        data, int(geom_id), int(self._floor_geom_id)
                    )
                    for geom_id in self._feet_geom_id
                ]
            )
            obs = self._get_obs(data, info, contact)
            metrics = dict(state.metrics)
            for name in (
                "h4/raw_slip_rms_m_s",
                "h4/raw_left_normalized_force",
                "h4/raw_right_normalized_force",
                "h4/raw_left_slip_m_s",
                "h4/raw_right_slip_m_s",
                "h4/raw_single_support",
                "h4/raw_alternation_event",
                "h4/raw_alternation_quality",
                "h4/raw_load_imbalance",
                "h4/raw_force_load_imbalance",
                "h4/raw_single_support_ema",
                "h4/raw_contact_duty_imbalance",
                "h4/raw_touchdown_count_imbalance",
                "h4/raw_flight",
                "h4/raw_total_normal_force_normalized",
                "h4/raw_total_normal_force_band_loss",
                "h4/raw_total_normal_force_tail_loss",
                "h4/raw_contact_pulse_liftoff_count",
                "h4/raw_contact_pulse_40ms_loss",
                "h4/raw_slip_tail_loss",
                "h4/raw_stance_slip_budget_loss",
                "h4/raw_left_stance_slip_integral_m",
                "h4/raw_right_stance_slip_integral_m",
                "h4/raw_left_target_lag_loss",
                "h4/raw_right_target_lag_loss",
                "h4/raw_phase17_left_force_slip_loss",
                "h4/raw_phase17_left_knee_excess_loss",
                "h4/raw_phase17_opposite_leg_lag_loss",
                "h4/raw_forward_cross_drift_loss",
                "h4/raw_forward_yaw_rate_loss",
                "h4/raw_forward_heading_drift_loss",
                "h4/raw_forward_heading_delta_rad",
                "h4/raw_reverse_speed_boundary_loss",
                "h4/raw_reverse_cross_drift_loss",
                "h4/raw_reverse_yaw_rate_loss",
                "h4/raw_reverse_heading_drift_loss",
                "h4/raw_reverse_phase_force_slip_loss",
                "h4/raw_reverse_contact_priority_reversal_lag_loss",
                "h4/raw_max_target_delta_rad",
                "h4/raw_pre_guard_max_delta_rad",
                "h4/raw_upstream_max_delta_rad",
                "h4/raw_reverse_teacher_precomposer_active",
                "h4/raw_reverse_teacher_entry_reset",
                "h4/raw_slew_feasibility_loss",
                "h4/raw_target_lag_loss",
            ):
                metrics[name] = jp.zeros(())
            if h5_v3_substep_alignment_enabled:
                for name in (
                    "h5/raw_substep_strict20ms_slip_rms_loss",
                    "h5/raw_substep_slip_tail_loss",
                    "h5/raw_substep_force_tail_loss",
                    "h5/raw_substep_qualified_sample_count",
                    "h5/substep_samples_finite",
                ):
                    metrics[name] = jp.zeros(())
            if forward_v4_substep_contact:
                for name in (
                    "h4/v4_single_authority_dynamic6_exact",
                    "h4/v4_single_authority_dynamic6_max_abs_error",
                    "h4/v4_single_authority_dynamic6_field_count",
                    "h4/v4_single_authority_dynamic6_field_count_exact",
                    "h4/v4_saved_dynamic6_substep_count",
                    "h4/v4_saved_dynamic6_field_count",
                    "h4/v4_saved_dynamic6_field_count_exact",
                    "h4/v4_saved_dynamic6_all_finite",
                    "h4/v4_telemetry_force_shape_valid",
                    "h4/v4_telemetry_force_all_finite",
                    "h4/v4_single_authority_violation",
                    "h4/v4_single_authority_assertion_token",
                    "h4/v4_raw_left_contact",
                    "h4/v4_raw_right_contact",
                    "h4/v4_qualified_left_contact",
                    "h4/v4_qualified_right_contact",
                    "h4/v4_confirmed_transition_count",
                    "h4/v4_touchdown_event_count",
                    "h4/v4_aborted_contact_island_count",
                    "h4/v4_aborted_off_gap_count",
                    "h4/v4_aborted_contact_island_loss",
                    "h4/v4_aborted_off_gap_loss",
                    "h4/v4_terminal_pending_discarded",
                ):
                    metrics[name] = jp.zeros(())
            if forward_iteration_v6_contact_abort_island_only:
                metrics["h4/v6_forward_contact_abort_routing_exact"] = jp.ones(())
                metrics["h4/v6_forward_contact_abort_island_loss"] = jp.zeros(())
                metrics[
                    "h4/v6_forward_contact_abort_off_gap_diagnostic_loss"
                ] = jp.zeros(())
                metrics[
                    "h4/v6_forward_contact_abort_off_gap_reward_contribution"
                ] = jp.zeros(())
                metrics[
                    "h4/v6_forward_contact_abort_pulse_reward_scale"
                ] = jp.asarray(
                    scale_values["h4_contact_pulse_40ms"], dtype=jp.float32
                )
                metrics[
                    "h4/v6_forward_contact_abort_routing_violation"
                ] = jp.zeros(())
                metrics[
                    "h4/v6_forward_contact_abort_routing_assertion_token"
                ] = jp.zeros(())
            if reverse_iteration_v6_absolute_full_leg_targets:
                metrics["h4/v6_reverse_decoder_exact"] = jp.ones(())
                metrics["h4/v6_reverse_decoder_max_abs_error"] = jp.zeros(())
                metrics["h4/v6_reverse_decoder_leg_count"] = jp.asarray(
                    len(LEG_ACTION_INDICES), dtype=jp.float32
                )
                metrics["h4/v6_reverse_decoder_leg_count_exact"] = jp.ones(())
                metrics["h4/v6_reverse_decoder_head_zero_exact"] = jp.ones(())
                metrics[
                    "h4/v6_reverse_teacher_target_contribution_zero_exact"
                ] = jp.ones(())
                metrics["h4/v6_reverse_residual_authority_scale"] = jp.zeros(())
                metrics["h4/v6_reverse_decoder_all_finite"] = jp.ones(())
                metrics[
                    "h4/v6_reverse_decoder_margin_saturation_count"
                ] = jp.zeros(())
                metrics["h4/v6_reverse_decoder_action_clip_count"] = jp.zeros(
                    ()
                )
                metrics["h4/v6_reverse_decoder_guard_lag_max_rad"] = jp.zeros(())
                metrics["h4/v6_reverse_precomposer_call_count"] = jp.zeros(
                    ()
                )
                metrics[
                    "h4/v6_reverse_precomposer_call_count_exact"
                ] = jp.ones(())
                metrics["h4/v6_reverse_final_guard_call_count"] = jp.zeros(
                    ()
                )
                metrics[
                    "h4/v6_reverse_final_guard_call_count_exact"
                ] = jp.ones(())
                metrics["h4/v6_reverse_decoder_violation"] = jp.zeros(())
                metrics["h4/v6_reverse_decoder_assertion_token"] = jp.zeros(
                    ()
                )
            if h5_absolute_target_routing:
                metrics["h5/target_space_decoder_exact"] = jp.ones(())
                metrics["h5/target_space_decoder_domain"] = jp.asarray(
                    0 if h5_target_domain == "planar" else 1,
                    dtype=jp.float32,
                )
            return state.replace(data=data, obs=obs, info=info, metrics=metrics)

        def _h4_contact_observables(self, data, previous_contact):
            contacts = data._impl.contact
            slot_count = int(contacts.geom.shape[0])
            geom_pair = contacts.geom
            geom_count = self._h4_geom_bodyid.shape[0]
            geom_ids_valid = jp.all(
                (geom_pair >= 0) & (geom_pair < geom_count), axis=1
            )
            safe_geom_pair = jp.where(geom_ids_valid[:, None], geom_pair, 0)
            body_pair = self._h4_geom_bodyid[safe_geom_pair]
            constraint_valid = contacts.efc_address >= 0
            foot_index = jp.full((slot_count,), -1, dtype=jp.int32)
            for index, foot_geom_id in enumerate(self._feet_geom_id):
                is_pair = geom_ids_valid & (
                    ((safe_geom_pair[:, 0] == int(self._floor_geom_id))
                     & (safe_geom_pair[:, 1] == int(foot_geom_id)))
                    | ((safe_geom_pair[:, 1] == int(self._floor_geom_id))
                       & (safe_geom_pair[:, 0] == int(foot_geom_id)))
                )
                foot_index = jp.where(is_pair, index, foot_index)
            foot_pair_valid = foot_index >= 0
            valid = foot_pair_valid & constraint_valid

            def safe_normal_force(contact_id, enabled):
                return jax.lax.cond(
                    enabled,
                    lambda _: jp.maximum(
                        mjx_support.contact_force(
                            self.mjx_model, data, contact_id
                        )[0],
                        0.0,
                    ),
                    lambda _: jp.zeros((), dtype=data.qvel.dtype),
                    None,
                )

            normal_force = jp.stack(
                [
                    safe_normal_force(contact_id, valid[contact_id])
                    for contact_id in range(slot_count)
                ]
            )

            def point_velocity(point, body_id, enabled):
                return jax.lax.cond(
                    enabled,
                    lambda operands: (
                        data.qvel
                        @ mjx_support.jac(
                            self.mjx_model, data, operands[0], operands[1]
                        )[0]
                    ),
                    lambda _: jp.zeros(3, dtype=data.qvel.dtype),
                    (point, body_id),
                )

            velocity1 = jax.vmap(point_velocity)(
                contacts.pos, body_pair[:, 0], valid
            )
            velocity2 = jax.vmap(point_velocity)(
                contacts.pos, body_pair[:, 1], valid
            )
            relative_velocity = velocity1 - velocity2
            normal = contacts.frame[:, 0, :]
            tangential = relative_velocity - jp.sum(
                relative_velocity * normal, axis=1, keepdims=True
            ) * normal
            tangential_speed = jp.where(
                valid, jp.linalg.norm(tangential, axis=1), 0.0
            )
            dynamic_robot_weight_n = robot_body_weight_n(
                self.mjx_model.body_mass,
                self._h4_robot_body_mask,
                self.mjx_model.opt.gravity,
                xp=jp,
            )
            return aggregate_force_contact_quality(
                jp.where(valid, normal_force, 0.0),
                jp.where(valid, tangential_speed, 0.0),
                foot_index,
                previous_contact,
                robot_weight_n=dynamic_robot_weight_n,
                xp=jp,
            )

        def _get_reward(self, *args, **kwargs):
            data = args[0]
            info = args[2]
            # The frozen source writes its pre-guard proposal after physics.
            # Replace it before any target imitation or metrics are computed.
            info["motor_targets"] = data.ctrl
            rewards = super()._get_reward(*args, **kwargs)
            if h5_routing_enabled:
                # The parent trainer averages inactive axes into the linear
                # score and disables command_progress. H5 uses only active
                # axes and a bounded signed projection, so zero velocity does
                # not tie the commanded reverse gait.
                command = jp.asarray(
                    info.get("h4_reward_physical_command", info["command"])
                )[:3]
                actual = jp.asarray(
                    (
                        self.get_local_linvel(data)[0],
                        self.get_local_linvel(data)[1],
                        self.get_gyro(data)[2],
                    )
                )
                sigma = jp.asarray(H5_TRACKING_SIGMA, dtype=actual.dtype)
                axis_tracking = jp.exp(-jp.square((actual - command) / sigma))
                active_linear = jp.abs(command[:2]) > 0.01
                linear_count = jp.sum(active_linear.astype(actual.dtype))
                active_tracking = jp.sum(
                    jp.where(active_linear, axis_tracking[:2], 0.0)
                ) / jp.maximum(linear_count, 1.0)
                yaw_active = jp.abs(command[2]) > 0.05
                rewards["tracking_ang_vel"] = jp.where(
                    yaw_active, axis_tracking[2], jp.asarray(1.0, dtype=actual.dtype)
                )
                command_norm_sq = jp.sum(jp.square(command))
                signed_progress = jp.where(
                    command_norm_sq > 1.0e-8,
                    jp.dot(actual, command) / command_norm_sq,
                    jp.asarray(0.0, dtype=actual.dtype),
                )
                # The frozen calibrated reward schema is locked and does not
                # expose command_progress. Reuse the existing bounded linear
                # tracking term so the signed return remains schema-safe.
                rewards["tracking_lin_vel"] = active_tracking + (
                    0.5 * H5_SIGNED_PROGRESS_SCALE * jp.clip(
                        signed_progress, -1.0, 1.0
                    )
                )
                if h5_seed_target_from_observation is not None:
                    seed_target = jp.asarray(info["h5_seed_bc_target"])
                    current_target = jp.asarray(info["h5_decoded_targets"])
                    seed_active = (command[0] < -0.02).astype(actual.dtype)
                    seed_weight = jp.asarray(
                        info["h5_seed_bc_weight"], dtype=actual.dtype
                    )
                    target_error = current_target - seed_target
                    seed_loss = jp.mean(
                        jp.square(target_error[jp.asarray(LEG_ACTION_INDICES)])
                        / (0.10**2)
                    )
                    rewards["target_imitation"] = (
                        seed_active * seed_weight * seed_loss
                    )
                    info["h5_seed_bc_loss"] = seed_active * seed_loss
                    info["h5_seed_bc_weight_applied"] = (
                        seed_active * seed_weight
                    )
            previous_force_contact = info["h4_previous_force_contact"]
            quality = self._h4_contact_observables(
                data, previous_force_contact
            )
            if forward_v4_substep_contact:
                v4_state = info["h4_v4_contact_telemetry_state"]
                v4_alternation_count = info["h4_v4_alternation_event_count"]
                alternation = AlternationUpdate(
                    v4_state.last_touchdown_foot,
                    v4_state.intervals_since_touchdown,
                    jp.logical_xor(
                        v4_state.persistence.qualified_contact[0],
                        v4_state.persistence.qualified_contact[1],
                    ),
                    v4_alternation_count > 0,
                    # Every confirmed opposite solo touchdown earns its own
                    # dwell-shaped reward; never dilute multiple events.
                    info["h4_v4_alternation_quality_sum"],
                )
                support_previous_contact = info[
                    "h4_v4_control_entry_raw_contact"
                ]
            else:
                alternation = update_alternation_state(
                    info["h4_last_single_support"],
                    info["h4_ticks_since_switch"],
                    quality.contact,
                    xp=jp,
                )
                support_previous_contact = previous_force_contact
            load_ema, load_imbalance = update_load_balance_ema(
                info["h4_load_ema"], quality.normalized_force, xp=jp
            )
            support = update_support_quality_state(
                support_previous_contact,
                quality.contact,
                quality.tangential_speed_m_s,
                info["h4_stance_slip_integral_m"],
                info["h4_single_support_ema"],
                info["h4_contact_duty_ema"],
                info["h4_touchdown_counts"],
                xp=jp,
            )
            if forward_v4_substep_contact:
                v4_touchdown_counts = v4_state.touchdown_counts
                v4_touchdown_balance_loss = jp.square(
                    jp.maximum(
                        jp.abs(
                            v4_touchdown_counts[0]
                            - v4_touchdown_counts[1]
                        )
                        - 1.0,
                        0.0,
                    )
                )
                support = support._replace(
                    touchdown_counts=v4_touchdown_counts,
                    touchdown_event=(
                        info["h4_v4_touchdown_event_count"] > 0
                    ),
                    touchdown_count_balance_loss=v4_touchdown_balance_loss,
                )
            force_quality = total_normal_force_quality(
                quality.normalized_force, xp=jp
            )
            contact_pulse = update_contact_pulse_state(
                support_previous_contact,
                quality.contact,
                info["h4_contact_run_length_ticks"],
                xp=jp,
            )
            if forward_v4_substep_contact:
                if forward_iteration_v6_contact_abort_island_only:
                    v6_routing = (
                        forward_iteration_v6_contact_abort_island_only_telemetry(
                            info["h4_v4_aborted_contact_island_loss_sum"],
                            info["h4_v4_aborted_off_gap_loss_sum"],
                            xp=jp,
                        )
                    )
                    v4_abort_loss = info[
                        "h4_v4_aborted_contact_island_loss_sum"
                    ] + jp.zeros_like(info["h4_v4_aborted_off_gap_loss_sum"])
                    pulse_scale = jp.asarray(
                        scale_values["h4_contact_pulse_40ms"], dtype=jp.float32
                    )
                    routing_violation = (
                        ~v6_routing.routing_exact
                        | (v6_routing.off_gap_reward_contribution != 0)
                        | (pulse_scale != -1.0)
                    )
                    info["h4_v6_forward_contact_abort_routing_exact"] = (
                        v6_routing.routing_exact
                    )
                    info["h4_v6_forward_contact_abort_island_loss"] = (
                        v6_routing.island_loss
                    )
                    info[
                        "h4_v6_forward_contact_abort_off_gap_diagnostic_loss"
                    ] = v6_routing.off_gap_diagnostic_loss
                    info[
                        "h4_v6_forward_contact_abort_off_gap_reward_contribution"
                    ] = v6_routing.off_gap_reward_contribution
                    info[
                        "h4_v6_forward_contact_abort_pulse_reward_scale"
                    ] = pulse_scale
                    info[
                        "h4_v6_forward_contact_abort_routing_violation"
                    ] = routing_violation
                    if v6_forward_compiled_routing_assertion is None:
                        raise RuntimeError(
                            "forward v6 compiled routing assertion is unavailable"
                        )
                    info[
                        "h4_v6_forward_contact_abort_routing_assertion_token"
                    ] = v6_forward_compiled_routing_assertion(
                        routing_violation
                    )
                else:
                    v4_abort_loss = (
                        info["h4_v4_aborted_contact_island_loss_sum"]
                        + info["h4_v4_aborted_off_gap_loss_sum"]
                    )
                contact_pulse = contact_pulse._replace(
                    liftoff_event=info["h4_v4_liftoff_event_count"] > 0,
                    per_foot_loss=v4_abort_loss,
                    # Sum every foot/event.  Averaging would let paired
                    # chatter dilute the exact per-transition penalty.
                    event_mean_loss=jp.sum(v4_abort_loss),
                )
            desired = info["h4_upstream_margin_targets"]
            normalized_joint_lag = jp.square(
                (desired - data.ctrl) / MAX_TARGET_DELTA_PER_TICK_RAD
            )
            left_leg_lag = jp.mean(
                normalized_joint_lag[jp.asarray(LEG_ACTION_INDICES[:5])]
            )
            right_leg_lag = jp.mean(
                normalized_joint_lag[jp.asarray(LEG_ACTION_INDICES[5:])]
            )
            phase_index = jp.floor(
                jp.mod(info["imitation_i"], self.PRM.nb_steps_in_period)
            ).astype(jp.int32)
            phase_multiplier = jp.where(
                phase_index == 17,
                3.0,
                jp.where((phase_index == 16) | (phase_index == 18), 1.5, 0.0),
            )
            forward_active = (info["command"][0] > 0.02).astype(
                quality.slip_rms_m_s.dtype
            )
            reverse_active = (info["command"][0] < -0.02).astype(
                quality.slip_rms_m_s.dtype
            )
            phase17_activation = (
                forward_active
                * quality.contact[0].astype(quality.slip_rms_m_s.dtype)
                * phase_multiplier
            )
            left_slip_squared = jp.square(
                quality.tangential_speed_m_s[0] / STRICT_SLIP_RMS_M_S
            )
            left_knee_upper = (
                SAFE_JOINT_LIMITS["left_knee"][1] - LEG_TARGET_MARGIN_RAD
            )
            left_knee_excess = jp.square(
                jp.maximum(
                    info["h4_pre_guard_raw_targets"][3] - left_knee_upper,
                    0.0,
                )
                / MAX_TARGET_DELTA_PER_TICK_RAD
            )
            opposite_leg_lag = (
                0.5 * normalized_joint_lag[12]
                + 0.2 * normalized_joint_lag[10]
                + 0.2 * normalized_joint_lag[13]
                + 0.1 * normalized_joint_lag[11]
            )
            local_velocity = self.get_local_linvel(data)
            yaw_rate = self.get_gyro(data)[2]
            current_yaw = quaternion_yaw_wxyz(
                self.get_floating_base_qpos(data.qpos)[3:7], xp=jp
            )
            heading_delta = wrapped_angle_difference(
                current_yaw, info["h4_heading_reference_yaw"], xp=jp
            )
            se2_cross_velocity = local_velocity[1]
            se2_yaw_rate = yaw_rate
            se2_heading_delta = heading_delta
            if h5_v3_command_conditioned_se2_alignment:
                se2_residuals = h5_v3_command_conditioned_se2_residuals(
                    local_velocity,
                    yaw_rate,
                    current_yaw,
                    info["h4_heading_reference_yaw"],
                    info["h5_v3_integrated_command_heading_yaw"],
                    info["h4_reward_physical_command"][:3],
                    xp=jp,
                )
                se2_cross_velocity = se2_residuals.cross_velocity_mps
                se2_yaw_rate = se2_residuals.yaw_rate_radps
                se2_heading_delta = se2_residuals.heading_error_rad
            forward_cross_loss = forward_active * jp.square(
                se2_cross_velocity / STRICT_FORWARD_CROSS_DRIFT_M_S
            )
            forward_yaw_loss = forward_active * jp.square(
                se2_yaw_rate / STRICT_FORWARD_YAW_RATE_RAD_S
            )
            forward_heading_loss = forward_active * jp.square(
                se2_heading_delta / STRICT_FORWARD_HEADING_DRIFT_RAD
            )
            reverse_speed_ratio = local_velocity[0] / jp.minimum(
                info["command"][0], -1.0e-6
            )
            reverse_speed_boundary_loss = reverse_active * (
                jp.square(
                    jp.maximum(
                        STRICT_REVERSE_SPEED_RATIO_LOWER - reverse_speed_ratio,
                        0.0,
                    )
                    / STRICT_REVERSE_SPEED_RATIO_LOWER
                )
                + jp.square(
                    jp.maximum(
                        reverse_speed_ratio - STRICT_REVERSE_SPEED_RATIO_UPPER,
                        0.0,
                    )
                    / STRICT_REVERSE_SPEED_RATIO_UPPER
                )
            )
            reverse_cross_loss = reverse_active * jp.square(
                se2_cross_velocity / STRICT_REVERSE_CROSS_DRIFT_M_S
            )
            reverse_yaw_loss = reverse_active * jp.square(
                se2_yaw_rate / STRICT_REVERSE_YAW_RATE_RAD_S
            )
            reverse_heading_loss = reverse_active * jp.square(
                se2_heading_delta / STRICT_REVERSE_HEADING_DRIFT_RAD
            )
            reverse_phase = reverse_phase_conditioned_quality_losses(
                phase_index,
                quality.contact,
                quality.tangential_speed_m_s,
                info["h4_upstream_signed_delta"],
                info["h4_previous_upstream_signed_delta"],
                normalized_joint_lag,
                xp=jp,
            )
            info["h4_previous_force_contact"] = quality.contact
            info["h4_contact_run_length_ticks"] = (
                contact_pulse.contact_run_length_ticks
            )
            info["h4_last_single_support"] = alternation.last_single_support
            info["h4_ticks_since_switch"] = alternation.ticks_since_switch
            info["h4_load_ema"] = load_ema
            info["h4_stance_slip_integral_m"] = (
                support.stance_slip_integral_m
            )
            info["h4_single_support_ema"] = support.single_support_ema
            info["h4_contact_duty_ema"] = support.contact_duty_ema
            info["h4_touchdown_counts"] = support.touchdown_counts
            info["h4_slip_rms_m_s"] = quality.slip_rms_m_s
            info["h4_per_foot_slip_m_s"] = quality.tangential_speed_m_s
            info["h4_normalized_force"] = quality.normalized_force
            info["h4_force_contact"] = quality.contact
            info["h4_alternation_event"] = alternation.alternation_event
            info["h4_alternation_quality"] = alternation.alternation_quality
            info["h4_force_load_imbalance"] = load_imbalance
            info["h4_load_imbalance"] = support.contact_duty_balance_loss
            info["h4_single_support_band_loss"] = (
                support.single_support_band_loss
            )
            info["h4_touchdown_count_balance_loss"] = (
                support.touchdown_count_balance_loss
            )
            info["h4_flight"] = support.flight
            info["h4_total_normal_force_normalized"] = (
                force_quality.total_normal_force_normalized
            )
            info["h4_total_normal_force_band_loss"] = force_quality.band_loss
            info["h4_total_normal_force_tail_loss"] = force_quality.tail_loss
            info["h4_contact_pulse_liftoff_event"] = (
                contact_pulse.liftoff_event
            )
            info["h4_contact_pulse_per_foot_loss"] = (
                contact_pulse.per_foot_loss
            )
            info["h4_contact_pulse_40ms_loss"] = (
                contact_pulse.event_mean_loss
            )
            info["h4_slip_tail_loss"] = support.slip_tail_loss
            info["h4_stance_slip_budget_loss"] = (
                support.stance_slip_budget_loss
            )
            info["h4_left_target_lag_loss"] = left_leg_lag
            info["h4_right_target_lag_loss"] = right_leg_lag
            info["h4_phase17_left_force_slip_loss"] = (
                phase17_activation * left_slip_squared
            )
            info["h4_phase17_left_knee_excess_loss"] = (
                phase17_activation * left_knee_excess
            )
            info["h4_phase17_opposite_leg_lag_loss"] = (
                phase17_activation * opposite_leg_lag
            )
            info["h4_forward_cross_drift_loss"] = forward_cross_loss
            info["h4_forward_yaw_rate_loss"] = forward_yaw_loss
            info["h4_forward_heading_drift_loss"] = forward_heading_loss
            info["h4_forward_heading_delta_rad"] = heading_delta
            if h5_v3_command_conditioned_se2_alignment:
                info["h5_v3_se2_cross_velocity_mps"] = se2_cross_velocity
                info["h5_v3_se2_yaw_rate_residual_radps"] = se2_yaw_rate
                info["h5_v3_se2_heading_residual_rad"] = se2_heading_delta
            info["h4_reverse_speed_boundary_loss"] = reverse_speed_boundary_loss
            info["h4_reverse_cross_drift_loss"] = reverse_cross_loss
            info["h4_reverse_yaw_rate_loss"] = reverse_yaw_loss
            info["h4_reverse_heading_drift_loss"] = reverse_heading_loss
            info["h4_reverse_phase_force_slip_loss"] = (
                reverse_active * reverse_phase.phase_force_slip
            )
            info["h4_reverse_contact_priority_reversal_lag_loss"] = (
                reverse_active * reverse_phase.contact_priority_reversal_lag
            )
            rewards["feet_slip"] = jp.zeros(())
            rewards["h4_force_slip"] = quality.slip_loss_normalized_squared
            contact_float = quality.contact.astype(quality.slip_rms_m_s.dtype)
            rewards["h4_left_force_slip"] = (
                contact_float[0]
                * jp.square(quality.tangential_speed_m_s[0])
                / (STRICT_SLIP_RMS_M_S**2)
            )
            rewards["h4_right_force_slip"] = (
                contact_float[1]
                * jp.square(quality.tangential_speed_m_s[1])
                / (STRICT_SLIP_RMS_M_S**2)
            )
            rewards["h4_per_foot_slip_tail"] = support.slip_tail_loss
            rewards["h4_per_foot_stance_slip_budget"] = (
                support.stance_slip_budget_loss
            )
            rewards["h4_single_support"] = alternation.single_support.astype(
                quality.slip_rms_m_s.dtype
            )
            rewards["h4_single_support_band"] = (
                support.single_support_band_loss
            )
            rewards["h4_alternation"] = alternation.alternation_quality
            rewards["h4_load_balance"] = support.contact_duty_balance_loss
            rewards["h4_touchdown_count_balance"] = (
                support.touchdown_count_balance_loss
            )
            rewards["h4_flight"] = support.flight
            rewards["h4_total_normal_force_band"] = force_quality.band_loss
            rewards["h4_total_normal_force_tail"] = force_quality.tail_loss
            rewards["h4_contact_pulse_40ms"] = contact_pulse.event_mean_loss
            rewards["h4_slew_feasibility"] = info[
                "h4_slew_feasibility_loss"
            ]
            rewards["h4_target_lag"] = info["h4_target_lag_loss"]
            rewards["h4_left_target_lag"] = left_leg_lag
            rewards["h4_right_target_lag"] = right_leg_lag
            rewards["h4_phase17_left_force_slip"] = (
                info["h4_phase17_left_force_slip_loss"]
            )
            rewards["h4_phase17_left_knee_envelope_excess"] = (
                info["h4_phase17_left_knee_excess_loss"]
            )
            rewards["h4_phase17_opposite_leg_lag"] = (
                info["h4_phase17_opposite_leg_lag_loss"]
            )
            rewards["h4_forward_cross_drift"] = forward_cross_loss
            rewards["h4_forward_uncommanded_yaw_rate"] = forward_yaw_loss
            rewards["h4_forward_heading_drift"] = forward_heading_loss
            rewards["h4_reverse_speed_boundary"] = reverse_speed_boundary_loss
            rewards["h4_reverse_cross_drift"] = reverse_cross_loss
            rewards["h4_reverse_uncommanded_yaw_rate"] = reverse_yaw_loss
            rewards["h4_reverse_heading_drift"] = reverse_heading_loss
            rewards["h4_reverse_phase_force_slip"] = info[
                "h4_reverse_phase_force_slip_loss"
            ]
            rewards["h4_reverse_contact_priority_reversal_lag"] = info[
                "h4_reverse_contact_priority_reversal_lag_loss"
            ]
            if h5_v3_substep_alignment_enabled:
                rewards["h5_all_substep_strict20ms_slip_rms"] = info[
                    "h5_v3_substep_strict20ms_slip_rms_loss"
                ]
                rewards["h5_all_substep_slip_tail"] = info[
                    "h5_v3_substep_slip_tail_loss"
                ]
                rewards["h5_all_substep_force_tail"] = info[
                    "h5_v3_substep_force_tail_loss"
                ]
            return rewards

        def step(self, state, action):
            previous_targets = jp.asarray(state.data.ctrl)
            reward_physical_command = jp.asarray(state.info["command"])
            if h5_v3_command_conditioned_se2_alignment:
                state.info["h5_v3_integrated_command_heading_yaw"] = (
                    advance_h5_v3_command_heading(
                        state.info["h5_v3_integrated_command_heading_yaw"],
                        reward_physical_command[2],
                        float(CONTROL_FIRST_STARTUP_DT_S),
                        xp=jp,
                    )
                )
            if h5_seed_target_from_observation is not None:
                state.info["h5_seed_bc_target"] = h5_seed_target_from_observation(
                    state.obs["state"]
                )
                state.info["h5_seed_bc_weight"] = jp.clip(
                    1.0
                    - jp.asarray(state.info["h4_guard_steps"], dtype=jp.float32)
                    / float(h5_seed_bc_anneal_control_steps),
                    0.0,
                    1.0,
                )
            state.info["h4_reward_physical_command"] = reward_physical_command
            state.info["motor_targets"] = previous_targets
            source_physics_step = joystick.mjx_env.step
            source_motor_speed_limits = joystick.USE_MOTOR_SPEED_LIMITS
            structural_call_count = 0
            reverse_v6_decoder_structural_call_count = 0
            reverse_v6_precomposer_structural_call_count = 0
            reverse_v6_final_guard_structural_call_count = 0
            captured_desired = None

            def guarded_physics_step(model, data, proposed_targets, n_substeps):
                nonlocal structural_call_count, captured_desired
                nonlocal reverse_v6_decoder_structural_call_count
                nonlocal reverse_v6_precomposer_structural_call_count
                nonlocal reverse_v6_final_guard_structural_call_count
                structural_call_count += 1
                if h5_absolute_target_routing:
                    h5_decoded_targets = h5_target_decoder(
                        action, domain=h5_target_domain, xp=jp
                    )
                    selected_raw_targets = h5_decoded_targets
                    state.info["h5_decoded_targets"] = h5_decoded_targets
                    reverse_v6_active = jp.asarray(False)
                    reverse_v6_decoder = None
                    state.info["h5_target_space_decoder_exact"] = jp.asarray(True)
                elif reverse_iteration_v6_absolute_full_leg_targets:
                    reverse_v6_active = state.info["command"][0] < -0.02
                    reverse_v6_decoder_structural_call_count += 1
                    reverse_v6_decoder = (
                        reverse_iteration_v6_absolute_full_leg_target_telemetry(
                            action, xp=jp
                        )
                    )
                    selected_raw_targets = jp.where(
                        reverse_v6_active,
                        reverse_v6_decoder.targets,
                        proposed_targets,
                    )
                else:
                    reverse_v6_active = jp.asarray(False)
                    reverse_v6_decoder = None
                    selected_raw_targets = proposed_targets
                raw_margin_target = margin_clip_targets(
                    selected_raw_targets, xp=jp
                )
                if h5_absolute_target_routing:
                    # H5 may use the selected reverse table as a phase/reference
                    # prior, but it must never become a second target authority.
                    reverse_teacher_active = jp.asarray(False)
                    reverse_v6_decoder = None
                    captured_desired = raw_margin_target
                elif teacher_table_values is None:
                    reverse_teacher_active = jp.asarray(False)
                    captured_desired = raw_margin_target
                else:
                    reverse_teacher_active = state.info["command"][0] < -0.02
                    raw_from_applied = raw_margin_target - data.ctrl
                    hard_delta = jp.clip(
                        raw_from_applied,
                        -MAX_TARGET_DELTA_PER_TICK_RAD,
                        MAX_TARGET_DELTA_PER_TICK_RAD,
                    )
                    soft_delta = MAX_TARGET_DELTA_PER_TICK_RAD * jp.tanh(
                        raw_from_applied / MAX_TARGET_DELTA_PER_TICK_RAD
                    )
                    # Exact hard value with a smooth surrogate derivative.
                    # This preserves <=0.04 rad/tick at trace time without
                    # erasing policy/teacher gradients outside the boundary.
                    differentiable_delta = soft_delta + jax.lax.stop_gradient(
                        hard_delta - soft_delta
                    )
                    if reverse_iteration_v6_absolute_full_leg_targets:
                        reverse_v6_precomposer_structural_call_count += 1
                    precomposed = data.ctrl + differentiable_delta
                    captured_desired = jp.where(
                        reverse_teacher_active,
                        precomposed,
                        raw_margin_target,
                    )
                state.info["h4_pre_guard_raw_targets"] = jp.asarray(
                    selected_raw_targets
                )
                state.info["h4_upstream_margin_targets"] = raw_margin_target
                state.info["h4_guard_desired_targets"] = captured_desired
                state.info["h4_reverse_teacher_precomposer_transition"] = (
                    reverse_teacher_active
                    != state.info["h4_reverse_teacher_precomposer_active"]
                )
                state.info["h4_reverse_teacher_precomposer_active"] = (
                    reverse_teacher_active
                )
                upstream_signed_delta = (
                    raw_margin_target
                    - state.info["h4_previous_upstream_margin_targets"]
                )
                state.info["h4_upstream_signed_delta"] = upstream_signed_delta
                upstream_delta = jp.abs(upstream_signed_delta)
                leg_upstream_delta = upstream_delta[
                    jp.asarray(LEG_ACTION_INDICES)
                ]
                excess_rate = jp.maximum(
                    leg_upstream_delta / CONTROL_FIRST_STARTUP_DT_S
                    - TARGET_SLEW_LIMIT_RAD_PER_S,
                    0.0,
                )
                state.info["h4_slew_feasibility_loss"] = jp.mean(
                    jp.square(excess_rate / TARGET_SLEW_LIMIT_RAD_PER_S)
                )
                state.info["h4_upstream_max_delta_rad"] = jp.max(
                    leg_upstream_delta
                )
                training_visible_delta = jp.abs(captured_desired - data.ctrl)[
                    jp.asarray(LEG_ACTION_INDICES)
                ]
                state.info["h4_pre_guard_max_delta_rad"] = jp.max(
                    training_visible_delta
                )
                if reverse_iteration_v6_absolute_full_leg_targets:
                    reverse_v6_final_guard_structural_call_count += 1
                applied = final_target_guard_step(
                    captured_desired, data.ctrl, xp=jp
                )
                lag = jp.abs(raw_margin_target - applied)[
                    jp.asarray(LEG_ACTION_INDICES)
                ]
                state.info["h4_target_lag_loss"] = jp.mean(
                    jp.square(lag / MAX_TARGET_DELTA_PER_TICK_RAD)
                )
                if reverse_iteration_v6_absolute_full_leg_targets or h5_absolute_target_routing:
                    if reverse_v6_decoder is None:
                        if not h5_absolute_target_routing:
                            raise RuntimeError("reverse v6 decoder telemetry is unavailable")
                    if h5_absolute_target_routing:
                        decoder_wiring = None
                    else:
                        decoder_wiring = reverse_iteration_v6_absolute_full_leg_target_wiring_audit(
                            selected_raw_targets,
                            action,
                            xp=jp,
                        )
                    if h5_absolute_target_routing:
                        # H5 target-space routing has its own exact decoder
                        # contract; the H4 v6 teacher-zero telemetry is not
                        # applicable and must not be fabricated here.
                        pass
                    else:
                        decoder_max_abs_error = decoder_wiring.max_abs_error
                        decoder_exact = decoder_wiring.exact
                        decoder_margin_targets = margin_clip_targets(
                            selected_raw_targets, xp=jp
                        )
                        leg_indices = jp.asarray(LEG_ACTION_INDICES)
                        head_indices = jp.asarray(HEAD_ACTION_INDICES)
                        margin_saturation_count = jp.sum(
                            (
                                selected_raw_targets[leg_indices]
                                != decoder_margin_targets[leg_indices]
                            ).astype(jp.int32)
                        )
                        teacher_target_contribution = (
                            decoder_wiring.teacher_target_contribution
                        )
                        teacher_target_contribution_zero_exact = (
                            decoder_wiring.teacher_target_contribution_zero_exact
                        )
                        residual_authority_scale = jp.zeros(
                            (), dtype=reverse_v6_decoder.targets.dtype
                        )
                        head_zero_exact = (
                            reverse_v6_decoder.head_zero_exact
                            & jp.all(selected_raw_targets[head_indices] == 0)
                            & jp.all(decoder_margin_targets[head_indices] == 0)
                            & jp.all(captured_desired[head_indices] == 0)
                            & jp.all(applied[head_indices] == 0)
                        )
                        decoder_all_finite = (
                            reverse_v6_decoder.all_finite
                            & jp.all(jp.isfinite(selected_raw_targets))
                            & jp.all(jp.isfinite(decoder_wiring.rederived_targets))
                            & jp.all(jp.isfinite(teacher_target_contribution))
                            & jp.all(jp.isfinite(decoder_margin_targets))
                            & jp.all(jp.isfinite(captured_desired))
                            & jp.all(jp.isfinite(applied))
                        )
                        precomposer_call_count = jp.asarray(
                            reverse_v6_precomposer_structural_call_count,
                            dtype=jp.int32,
                        )
                        final_guard_call_count = jp.asarray(
                            reverse_v6_final_guard_structural_call_count,
                            dtype=jp.int32,
                        )
                        structural_invariants = (
                            reverse_iteration_v6_structural_count_invariants(
                                reverse_v6_decoder.leg_count,
                                precomposer_call_count,
                                final_guard_call_count,
                                xp=jp,
                            )
                        )
                        active_decoder_violation = (
                            ~decoder_exact
                            | (decoder_max_abs_error != 0)
                            | ~head_zero_exact
                            | ~teacher_target_contribution_zero_exact
                            | (residual_authority_scale != 0)
                            | ~decoder_all_finite
                            | structural_invariants.violation
                        )
                        decoder_violation = reverse_v6_active & active_decoder_violation
                        state.info["h4_v6_reverse_decoder_action"] = (
                            jp.asarray(action)
                        )
                        state.info["h4_v6_reverse_decoder_raw_targets"] = (
                            selected_raw_targets
                        )
                        state.info["h4_v6_reverse_decoder_margin_targets"] = (
                            decoder_margin_targets
                        )
                        state.info["h4_v6_reverse_decoder_exact"] = jp.where(
                            reverse_v6_active, decoder_exact, True
                        )
                        state.info["h4_v6_reverse_decoder_max_abs_error"] = jp.where(
                            reverse_v6_active, decoder_max_abs_error, 0.0
                        )
                        state.info["h4_v6_reverse_decoder_leg_count"] = (
                            reverse_v6_decoder.leg_count
                        )
                        state.info["h4_v6_reverse_decoder_leg_count_exact"] = (
                            structural_invariants.decoder_leg_count_exact
                        )
                        state.info["h4_v6_reverse_decoder_head_zero_exact"] = jp.where(
                            reverse_v6_active, head_zero_exact, True
                        )
                        state.info[
                            "h4_v6_reverse_teacher_target_contribution_zero_exact"
                        ] = jp.where(
                            reverse_v6_active,
                            teacher_target_contribution_zero_exact,
                            True,
                        )
                        state.info["h4_v6_reverse_residual_authority_scale"] = (
                            residual_authority_scale
                        )
                        state.info["h4_v6_reverse_decoder_all_finite"] = jp.where(
                            reverse_v6_active, decoder_all_finite, True
                        )
                        state.info[
                            "h4_v6_reverse_decoder_margin_saturation_count"
                        ] = jp.where(
                            reverse_v6_active,
                            margin_saturation_count,
                            jp.zeros((), dtype=jp.int32),
                        )
                        state.info["h4_v6_reverse_decoder_action_clip_count"] = (
                            jp.where(
                                reverse_v6_active,
                                reverse_v6_decoder.action_clip_count,
                                jp.zeros((), dtype=jp.int32),
                            )
                        )
                        state.info["h4_v6_reverse_decoder_guard_lag_max_rad"] = (
                            jp.where(reverse_v6_active, jp.max(lag), 0.0)
                        )
                        state.info["h4_v6_reverse_precomposer_call_count"] = (
                            precomposer_call_count
                        )
                        state.info[
                            "h4_v6_reverse_precomposer_call_count_exact"
                        ] = structural_invariants.precomposer_call_count_exact
                        state.info["h4_v6_reverse_final_guard_call_count"] = (
                            final_guard_call_count
                        )
                        state.info[
                            "h4_v6_reverse_final_guard_call_count_exact"
                        ] = structural_invariants.final_guard_call_count_exact
                        state.info["h4_v6_reverse_decoder_violation"] = (
                            decoder_violation
                        )
                        if v6_reverse_compiled_decoder_assertion is None:
                            raise RuntimeError(
                                "reverse v6 compiled decoder assertion is unavailable"
                            )
                        state.info["h4_v6_reverse_decoder_assertion_token"] = (
                            v6_reverse_compiled_decoder_assertion(
                                decoder_violation
                            )
                        )
                if not forward_v4_substep_contact:
                    return source_physics_step(model, data, applied, n_substeps)
                if int(n_substeps) != V4_CONTROL_SUBSTEP_COUNT:
                    raise RuntimeError(
                        "forward v4 requires the frozen 10-substep physics call"
                    )
                v4_initial_state = state.info["h4_v4_contact_telemetry_state"]
                state.info["h4_v4_control_entry_raw_contact"] = (
                    v4_initial_state.persistence.raw_contact
                )
                state.info["h4_v4_control_entry_qualified_contact"] = (
                    v4_initial_state.persistence.qualified_contact
                )

                def authoritative_single_step(authority_data, authority_action):
                    # Match the source primitive body exactly.  Calling the
                    # source wrapper with ``n_substeps=1`` would introduce a
                    # nested length-one scan and can compile to a different
                    # floating-point program on CUDA.
                    return v4_authoritative_primitive_step(
                        model,
                        authority_data,
                        authority_action,
                        mjx_step=joystick.mjx_env.mjx.step,
                    )

                def coherent_replay_measurement_state(replay_data):
                    return joystick.mjx_env.mjx.forward(model, replay_data)

                def replay_force_measurement(coherent_data, previous_raw):
                    return self._h4_contact_observables(
                        coherent_data, previous_raw
                    ).normalized_force

                authority_data, saved_dynamic_states = (
                    scan_v4_instrumented_physics_trajectory(
                        data,
                        applied,
                        single_physics_step=authoritative_single_step,
                        n_substeps=n_substeps,
                        scan=jax.lax.scan,
                        xp=jp,
                    )
                )
                if v4_substep_collector_trace_capture_enabled:
                    def replay_force_and_tangential_speed(
                        coherent_data, previous_raw
                    ):
                        quality = self._h4_contact_observables(
                            coherent_data, previous_raw
                        )
                        return (
                            quality.normalized_force,
                            quality.tangential_speed_m_s,
                        )

                    v4_summary, v4_substep_collector_quality_trace = (
                        scan_v4_saved_state_contact_telemetry_with_quality_trace(
                            data,
                            saved_dynamic_states,
                            v4_initial_state,
                            cohere_measurement_state=coherent_replay_measurement_state,
                            measure_force_and_tangential_speed=(
                                replay_force_and_tangential_speed
                            ),
                            n_substeps=n_substeps,
                            scan=jax.lax.scan,
                            xp=jp,
                        )
                    )
                else:
                    v4_summary = scan_v4_saved_state_contact_telemetry(
                        data,
                        saved_dynamic_states,
                        v4_initial_state,
                        cohere_measurement_state=coherent_replay_measurement_state,
                        measure_normalized_force=replay_force_measurement,
                        n_substeps=n_substeps,
                        scan=jax.lax.scan,
                        xp=jp,
                    )
                if h5_v3_substep_alignment_enabled:
                    if h5_all_substep_quality_update is None:
                        raise RuntimeError("H5 all-substep quality helper is unavailable")
                    if h5_v3_substep_preflight_fixed_quality_replay_enabled:
                        if h5_t1_fixed_quality_replay is None:
                            raise RuntimeError(
                                "H5 T=1 fixed quality replay helper is unavailable"
                            )
                        fixed_force, fixed_speed = h5_t1_fixed_quality_replay(xp=jp)
                        h5_quality_trajectory = V4SubstepContactQualityTrajectory(
                            jp.asarray(saved_dynamic_states.time),
                            fixed_force,
                            fixed_speed,
                        )
                    else:
                        def replay_force_and_tangential_speed(coherent_data):
                            quality = self._h4_contact_observables(
                                coherent_data, jp.zeros(2, dtype=bool)
                            )
                            return (
                                quality.normalized_force,
                                quality.tangential_speed_m_s,
                            )

                        h5_quality_trajectory = (
                            scan_v4_saved_state_contact_quality_trajectory(
                                data,
                                saved_dynamic_states,
                                cohere_measurement_state=coherent_replay_measurement_state,
                                measure_force_and_tangential_speed=(
                                    replay_force_and_tangential_speed
                                ),
                                n_substeps=n_substeps,
                                scan=jax.lax.scan,
                                xp=jp,
                            )
                        )
                    h5_substep_update = h5_all_substep_quality_update(
                        h5_quality_trajectory.normalized_normal_force,
                        h5_quality_trajectory.tangential_speed_m_s,
                        initial_debounce=state.info["h5_v3_substep_debounce"],
                        times_s=h5_quality_trajectory.time_s,
                        xp=jp,
                    )
                    h5_quality_finite = (
                        jp.all(
                            jp.isfinite(
                                h5_quality_trajectory.normalized_normal_force
                            )
                        )
                        & jp.all(
                            jp.isfinite(h5_quality_trajectory.tangential_speed_m_s)
                        )
                        & jp.all(jp.isfinite(h5_quality_trajectory.time_s))
                        & jp.all(
                            h5_quality_trajectory.normalized_normal_force >= 0.0
                        )
                        & jp.all(h5_quality_trajectory.tangential_speed_m_s >= 0.0)
                    )
                parity = audit_v4_dynamic_endpoint_self_consistency(
                    authority_data,
                    saved_dynamic_states,
                    xp=jp,
                )
                field_count = jp.asarray(parity.leaf_count, dtype=jp.int32)
                saved_substep_count = jp.asarray(
                    V4_CONTROL_SUBSTEP_COUNT, dtype=jp.int32
                )
                saved_field_count = jp.asarray(
                    len(V4SavedDynamicState._fields), dtype=jp.int32
                )
                field_count_exact, saved_field_count_exact = (
                    v4_dynamic_field_counts_exact(
                        field_count,
                        saved_field_count,
                        xp=jp,
                    )
                )
                saved_all_finite = v4_saved_dynamic_trajectory_all_finite(
                    saved_dynamic_states, xp=jp
                )
                telemetry_force_shape_valid = jp.asarray(True)
                telemetry_force_all_finite = (
                    v4_summary.normalized_force_finite
                )
                authority_violation = (
                    ~parity.exact
                    | (parity.max_abs_error != 0.0)
                    | ~field_count_exact
                    | (saved_substep_count != V4_CONTROL_SUBSTEP_COUNT)
                    | ~saved_field_count_exact
                    | ~saved_all_finite
                    | ~telemetry_force_shape_valid
                    | ~telemetry_force_all_finite
                )
                if v4_compiled_authority_assertion is None:
                    raise RuntimeError("forward v4 authority assertion is unavailable")
                authority_assertion_token = v4_compiled_authority_assertion(
                    ~authority_violation,
                    parity.max_abs_error,
                    field_count,
                )
                v4_state = v4_summary.state
                state.info["h4_v4_contact_telemetry_state"] = v4_state
                state.info["h4_previous_force_contact"] = (
                    v4_state.persistence.raw_contact
                )
                state.info["h4_v4_qualified_contact"] = (
                    v4_state.persistence.qualified_contact
                )
                state.info["h4_v4_pending_active"] = (
                    v4_state.persistence.pending_active
                )
                state.info["h4_v4_pending_intervals"] = (
                    v4_state.persistence.pending_intervals
                )
                state.info["h4_v4_confirmed_transition_count"] = (
                    v4_summary.confirmed_transition_count
                )
                state.info["h4_v4_touchdown_event_count"] = (
                    v4_summary.touchdown_event_count
                )
                state.info["h4_v4_liftoff_event_count"] = (
                    v4_summary.liftoff_event_count
                )
                state.info["h4_v4_aborted_contact_island_count"] = (
                    v4_summary.aborted_contact_island_count
                )
                state.info["h4_v4_aborted_off_gap_count"] = (
                    v4_summary.aborted_off_gap_count
                )
                state.info["h4_v4_aborted_contact_island_loss_sum"] = (
                    v4_summary.aborted_contact_island_loss_sum
                )
                state.info["h4_v4_aborted_off_gap_loss_sum"] = (
                    v4_summary.aborted_off_gap_loss_sum
                )
                state.info["h4_v4_alternation_event_count"] = (
                    v4_summary.alternation_event_count
                )
                state.info["h4_v4_alternation_quality_sum"] = (
                    v4_summary.alternation_quality_sum
                )
                state.info["h4_v4_single_authority_dynamic6_exact"] = parity.exact
                state.info["h4_v4_single_authority_dynamic6_max_abs_error"] = (
                    parity.max_abs_error
                )
                state.info["h4_v4_single_authority_dynamic6_field_count"] = (
                    field_count
                )
                state.info[
                    "h4_v4_single_authority_dynamic6_field_count_exact"
                ] = field_count_exact
                state.info["h4_v4_saved_dynamic6_substep_count"] = (
                    saved_substep_count
                )
                state.info["h4_v4_saved_dynamic6_field_count"] = (
                    saved_field_count
                )
                state.info["h4_v4_saved_dynamic6_field_count_exact"] = (
                    saved_field_count_exact
                )
                state.info["h4_v4_saved_dynamic6_all_finite"] = (
                    saved_all_finite
                )
                state.info["h4_v4_telemetry_force_shape_valid"] = (
                    telemetry_force_shape_valid
                )
                state.info["h4_v4_telemetry_force_all_finite"] = (
                    telemetry_force_all_finite
                )
                state.info["h4_v4_single_authority_violation"] = (
                    authority_violation
                )
                state.info["h4_v4_single_authority_assertion_token"] = (
                    authority_assertion_token
                )
                if v4_substep_collector_trace_capture_enabled:
                    state.info["v4_substep_collector_quality_trace"] = (
                        v4_substep_collector_quality_trace
                    )
                if h5_v3_substep_alignment_enabled:
                    state.info["h5_v3_substep_debounce"] = h5_substep_update.debounce
                    state.info["h5_v3_substep_strict20ms_slip_rms_loss"] = (
                        h5_substep_update.losses.strict20ms_slip_rms_loss
                    )
                    state.info["h5_v3_substep_slip_tail_loss"] = (
                        h5_substep_update.losses.slip_tail_loss
                    )
                    state.info["h5_v3_substep_force_tail_loss"] = (
                        h5_substep_update.losses.force_tail_loss
                    )
                    state.info["h5_v3_substep_qualified_sample_count"] = (
                        h5_substep_update.losses.force_qualified_sample_count
                    )
                    state.info["h5_v3_substep_samples_finite"] = h5_quality_finite
                    if h5_v3_substep_preflight_telemetry_enabled:
                        # This opt-in trace is for the no-PPO proof only.  It
                        # exposes the retained post-physics samples without
                        # changing the physics, target, observation, or reward
                        # calculations used by a training candidate.
                        state.info["h5_v3_substep_time_s_trace"] = (
                            h5_quality_trajectory.time_s
                        )
                        state.info["h5_v3_substep_normalized_force_trace"] = (
                            h5_quality_trajectory.normalized_normal_force
                        )
                        state.info["h5_v3_substep_tangential_speed_trace_mps"] = (
                            h5_quality_trajectory.tangential_speed_m_s
                        )
                return authority_data

            joystick.mjx_env.step = guarded_physics_step
            joystick.USE_MOTOR_SPEED_LIMITS = False
            try:
                result = super().step(state, action)
            finally:
                joystick.mjx_env.step = source_physics_step
                joystick.USE_MOTOR_SPEED_LIMITS = source_motor_speed_limits
            if structural_call_count != 1:
                raise RuntimeError(
                    "frozen Joystick step must invoke physics exactly once"
                )
            if reverse_iteration_v6_absolute_full_leg_targets and (
                reverse_v6_decoder_structural_call_count,
                reverse_v6_precomposer_structural_call_count,
                reverse_v6_final_guard_structural_call_count,
            ) != (1, 1, 1):
                raise RuntimeError(
                    "reverse v6 decoder/precomposer/final-guard call counts "
                    "must each be exactly one"
                )
            if h5_absolute_target_routing and structural_call_count != 1:
                raise RuntimeError(
                    "H5 target-space decoder must be applied exactly once per step"
                )
            info = result.info
            info["motor_targets"] = result.data.ctrl
            if forward_v4_substep_contact:
                v4_before_terminal = info["h4_v4_contact_telemetry_state"]
                terminal = jp.asarray(result.done).astype(bool)
                info["h4_v4_terminal_pending_discarded"] = terminal & jp.any(
                    v4_before_terminal.persistence.pending_active
                )
                v4_after_terminal = discard_v4_terminal_incomplete(
                    v4_before_terminal, terminal, xp=jp
                )
                info["h4_v4_contact_telemetry_state"] = v4_after_terminal
                info["h4_v4_qualified_contact"] = (
                    v4_after_terminal.persistence.qualified_contact
                )
                info["h4_v4_pending_active"] = (
                    v4_after_terminal.persistence.pending_active
                )
                info["h4_v4_pending_intervals"] = (
                    v4_after_terminal.persistence.pending_intervals
                )
            # Reward was evaluated against the command present at step entry.
            # The frozen source may have resampled info['command'] afterwards;
            # map that final physical value into the next policy observation.
            next_physical_command = jp.asarray(info["command"])
            next_policy_command = _require_vector_shape(
                policy_observation_mapper(next_physical_command),
                7,
                "policy observation command",
                xp=jp,
            )
            info["h4_reward_physical_command"] = reward_physical_command
            info["h4_physical_command"] = next_physical_command
            info["h4_policy_observation_command"] = next_policy_command
            next_reverse_teacher = (
                jp.asarray(False)
                if teacher_table_values is None
                else next_physical_command[0] < -0.02
            )
            current_reverse_teacher = (
                jp.asarray(False)
                if teacher_table_values is None
                else reward_physical_command[0] < -0.02
            )
            entering_reverse_teacher = (
                next_reverse_teacher & ~current_reverse_teacher
            )
            exiting_reverse_teacher = (
                current_reverse_teacher & ~next_reverse_teacher
            )
            if teacher_table_values is not None:
                entry_source_phase = jp.asarray(
                    self._h4_reverse_teacher_entry_source_phase
                )
                info["imitation_i"] = jp.where(
                    entering_reverse_teacher,
                    entry_source_phase,
                    info["imitation_i"],
                )
                phase_angle = (
                    info["imitation_i"]
                    / float(self.PRM.nb_steps_in_period)
                    * 2.0
                    * jp.pi
                )
                reset_phase_observation = jp.asarray(
                    (jp.cos(phase_angle), jp.sin(phase_angle))
                )
                info["imitation_phase"] = jp.where(
                    entering_reverse_teacher,
                    reset_phase_observation,
                    info["imitation_phase"],
                )
                info["current_reference_motion"] = jp.where(
                    entering_reverse_teacher,
                    self._get_optimized_backward_reference(info["imitation_i"]),
                    info["current_reference_motion"],
                )
                if reverse_iteration_v6_absolute_full_leg_targets or h5_absolute_target_routing:
                    # The next actor action is the sole reverse target
                    # authority.  Seed delta telemetry from the applied target
                    # at route entry; never seed it from a teacher value.  H5
                    # also keeps the selected reverse table out of this
                    # target-history seam: it is a phase/reference prior only.
                    entry_upstream_target = result.data.ctrl
                else:
                    entry_upstream_target = (
                        self._h4_selected_teacher_actuator_target(
                            teacher_entry_phase
                        )
                    )
                next_upstream_reference = jp.where(
                    entering_reverse_teacher,
                    entry_upstream_target,
                    jp.where(
                        exiting_reverse_teacher,
                        result.data.ctrl,
                        info["h4_upstream_margin_targets"],
                    ),
                )
            else:
                next_upstream_reference = info["h4_upstream_margin_targets"]
            h5_entering_reverse = jp.asarray(False)
            if h5_absolute_target_routing and h5_target_domain in {"reverse", "unified"}:
                h5_entering_reverse = (
                    (next_physical_command[0] < -0.02)
                    & ~(reward_physical_command[0] < -0.02)
                )
                h5_entry_phase = jp.asarray(7.0, dtype=info["imitation_i"].dtype)
                info["imitation_i"] = jp.where(
                    h5_entering_reverse,
                    h5_entry_phase,
                    info["imitation_i"],
                )
                h5_phase_angle = (
                    info["imitation_i"]
                    / float(self.PRM.nb_steps_in_period)
                    * 2.0
                    * jp.pi
                )
                h5_phase_observation = jp.asarray(
                    (jp.cos(h5_phase_angle), jp.sin(h5_phase_angle))
                )
                info["imitation_phase"] = jp.where(
                    h5_entering_reverse,
                    h5_phase_observation,
                    info["imitation_phase"],
                )
            standard_next_reference = self.PRM.get_reference_motion(
                next_physical_command[0],
                next_physical_command[1],
                next_physical_command[2],
                info["imitation_i"],
            )
            if teacher_table_values is None:
                info["current_reference_motion"] = standard_next_reference
            else:
                info["current_reference_motion"] = jp.where(
                    next_reverse_teacher,
                    self._get_optimized_backward_reference(info["imitation_i"]),
                    standard_next_reference,
                )
            info["h4_previous_upstream_margin_targets"] = (
                next_upstream_reference
            )
            info["h4_previous_upstream_signed_delta"] = jp.where(
                entering_reverse_teacher | exiting_reverse_teacher | h5_entering_reverse,
                jp.zeros(ACTION_COUNT),
                info["h4_upstream_signed_delta"],
            )
            info["h4_reverse_teacher_entry_reset"] = (
                entering_reverse_teacher | h5_entering_reverse
            )
            info["h4_reverse_teacher_precomposer_transition"] = (
                next_reverse_teacher
                != info["h4_reverse_teacher_precomposer_active"]
            )
            info["h4_reverse_teacher_precomposer_active"] = (
                next_reverse_teacher
            )
            command_changed = jp.any(
                jp.abs(next_physical_command[:3] - reward_physical_command[:3])
                > 1.0e-7
            )
            current_yaw = quaternion_yaw_wxyz(
                self.get_floating_base_qpos(result.data.qpos)[3:7], xp=jp
            )
            info["h4_heading_reference_yaw"] = jp.where(
                command_changed,
                current_yaw,
                info["h4_heading_reference_yaw"],
            )
            if h5_v3_command_conditioned_se2_alignment:
                info["h5_v3_integrated_command_heading_yaw"] = jp.where(
                    command_changed,
                    current_yaw,
                    info["h5_v3_integrated_command_heading_yaw"],
                )
            info["h4_guard_steps"] = state.info["h4_guard_steps"] + 1
            info["h4_previous_desired_targets"] = captured_desired
            obs = synchronize_observation_motor_targets(
                result.obs, result.data.ctrl, xp=jp
            )
            obs = synchronize_post_step_command_observations(
                obs,
                next_physical_command,
                next_policy_command,
                include_h4_actor_observables=include_h4_actor_observables,
                xp=jp,
            )
            obs = synchronize_post_step_imitation_state(
                obs,
                info["current_reference_motion"],
                info["imitation_i"],
                info["imitation_phase"],
                xp=jp,
            )
            delta = jp.abs(result.data.ctrl - previous_targets)
            leg_delta = delta[jp.asarray(LEG_ACTION_INDICES)]
            metrics = result.metrics
            metrics["h4/raw_slip_rms_m_s"] = info["h4_slip_rms_m_s"]
            metrics["h4/raw_left_normalized_force"] = info[
                "h4_normalized_force"
            ][0]
            metrics["h4/raw_right_normalized_force"] = info[
                "h4_normalized_force"
            ][1]
            metrics["h4/raw_left_slip_m_s"] = info[
                "h4_per_foot_slip_m_s"
            ][0]
            metrics["h4/raw_right_slip_m_s"] = info[
                "h4_per_foot_slip_m_s"
            ][1]
            metrics["h4/raw_single_support"] = jp.logical_xor(
                info["h4_force_contact"][0], info["h4_force_contact"][1]
            ).astype(jp.float32)
            metrics["h4/raw_alternation_event"] = info[
                "h4_alternation_event"
            ].astype(jp.float32)
            metrics["h4/raw_alternation_quality"] = info[
                "h4_alternation_quality"
            ]
            metrics["h4/raw_load_imbalance"] = info["h4_load_imbalance"]
            metrics["h4/raw_force_load_imbalance"] = info[
                "h4_force_load_imbalance"
            ]
            metrics["h4/raw_single_support_ema"] = info[
                "h4_single_support_ema"
            ]
            metrics["h4/raw_contact_duty_imbalance"] = info[
                "h4_load_imbalance"
            ]
            metrics["h4/raw_touchdown_count_imbalance"] = (
                touchdown_count_imbalance_metric(
                    info["h4_touchdown_counts"], xp=jp
                )
            )
            metrics["h4/raw_flight"] = info["h4_flight"]
            metrics["h4/raw_total_normal_force_normalized"] = info[
                "h4_total_normal_force_normalized"
            ]
            metrics["h4/raw_total_normal_force_band_loss"] = info[
                "h4_total_normal_force_band_loss"
            ]
            metrics["h4/raw_total_normal_force_tail_loss"] = info[
                "h4_total_normal_force_tail_loss"
            ]
            metrics["h4/raw_contact_pulse_liftoff_count"] = jp.sum(
                info["h4_contact_pulse_liftoff_event"].astype(jp.float32)
            )
            metrics["h4/raw_contact_pulse_40ms_loss"] = info[
                "h4_contact_pulse_40ms_loss"
            ]
            metrics["h4/raw_slip_tail_loss"] = info["h4_slip_tail_loss"]
            metrics["h4/raw_stance_slip_budget_loss"] = info[
                "h4_stance_slip_budget_loss"
            ]
            metrics["h4/raw_left_stance_slip_integral_m"] = info[
                "h4_stance_slip_integral_m"
            ][0]
            metrics["h4/raw_right_stance_slip_integral_m"] = info[
                "h4_stance_slip_integral_m"
            ][1]
            metrics["h4/raw_left_target_lag_loss"] = info[
                "h4_left_target_lag_loss"
            ]
            metrics["h4/raw_right_target_lag_loss"] = info[
                "h4_right_target_lag_loss"
            ]
            metrics["h4/raw_phase17_left_force_slip_loss"] = info[
                "h4_phase17_left_force_slip_loss"
            ]
            metrics["h4/raw_phase17_left_knee_excess_loss"] = info[
                "h4_phase17_left_knee_excess_loss"
            ]
            metrics["h4/raw_phase17_opposite_leg_lag_loss"] = info[
                "h4_phase17_opposite_leg_lag_loss"
            ]
            metrics["h4/raw_forward_cross_drift_loss"] = info[
                "h4_forward_cross_drift_loss"
            ]
            metrics["h4/raw_forward_yaw_rate_loss"] = info[
                "h4_forward_yaw_rate_loss"
            ]
            metrics["h4/raw_forward_heading_drift_loss"] = info[
                "h4_forward_heading_drift_loss"
            ]
            metrics["h4/raw_forward_heading_delta_rad"] = info[
                "h4_forward_heading_delta_rad"
            ]
            metrics["h4/raw_reverse_speed_boundary_loss"] = info[
                "h4_reverse_speed_boundary_loss"
            ]
            metrics["h4/raw_reverse_cross_drift_loss"] = info[
                "h4_reverse_cross_drift_loss"
            ]
            metrics["h4/raw_reverse_yaw_rate_loss"] = info[
                "h4_reverse_yaw_rate_loss"
            ]
            metrics["h4/raw_reverse_heading_drift_loss"] = info[
                "h4_reverse_heading_drift_loss"
            ]
            metrics["h4/raw_reverse_phase_force_slip_loss"] = info[
                "h4_reverse_phase_force_slip_loss"
            ]
            metrics[
                "h4/raw_reverse_contact_priority_reversal_lag_loss"
            ] = info["h4_reverse_contact_priority_reversal_lag_loss"]
            metrics["h4/raw_max_target_delta_rad"] = jp.max(leg_delta)
            metrics["h4/raw_pre_guard_max_delta_rad"] = info[
                "h4_pre_guard_max_delta_rad"
            ]
            metrics["h4/raw_upstream_max_delta_rad"] = info[
                "h4_upstream_max_delta_rad"
            ]
            metrics["h4/raw_reverse_teacher_precomposer_active"] = info[
                "h4_reverse_teacher_precomposer_active"
            ].astype(jp.float32)
            metrics["h4/raw_reverse_teacher_entry_reset"] = info[
                "h4_reverse_teacher_entry_reset"
            ].astype(jp.float32)
            metrics["h4/raw_slew_feasibility_loss"] = info[
                "h4_slew_feasibility_loss"
            ]
            metrics["h4/raw_target_lag_loss"] = info["h4_target_lag_loss"]
            if h5_v3_substep_alignment_enabled:
                metrics["h5/raw_substep_strict20ms_slip_rms_loss"] = info[
                    "h5_v3_substep_strict20ms_slip_rms_loss"
                ]
                metrics["h5/raw_substep_slip_tail_loss"] = info[
                    "h5_v3_substep_slip_tail_loss"
                ]
                metrics["h5/raw_substep_force_tail_loss"] = info[
                    "h5_v3_substep_force_tail_loss"
                ]
                metrics["h5/raw_substep_qualified_sample_count"] = info[
                    "h5_v3_substep_qualified_sample_count"
                ]
                metrics["h5/substep_samples_finite"] = info[
                    "h5_v3_substep_samples_finite"
                ].astype(jp.float32)
            if forward_v4_substep_contact:
                metrics["h4/v4_single_authority_dynamic6_exact"] = info[
                    "h4_v4_single_authority_dynamic6_exact"
                ].astype(jp.float32)
                metrics["h4/v4_single_authority_dynamic6_max_abs_error"] = info[
                    "h4_v4_single_authority_dynamic6_max_abs_error"
                ]
                metrics["h4/v4_single_authority_dynamic6_field_count"] = info[
                    "h4_v4_single_authority_dynamic6_field_count"
                ].astype(jp.float32)
                metrics[
                    "h4/v4_single_authority_dynamic6_field_count_exact"
                ] = info[
                    "h4_v4_single_authority_dynamic6_field_count_exact"
                ].astype(jp.float32)
                metrics["h4/v4_saved_dynamic6_substep_count"] = info[
                    "h4_v4_saved_dynamic6_substep_count"
                ].astype(jp.float32)
                metrics["h4/v4_saved_dynamic6_field_count"] = info[
                    "h4_v4_saved_dynamic6_field_count"
                ].astype(jp.float32)
                metrics["h4/v4_saved_dynamic6_field_count_exact"] = info[
                    "h4_v4_saved_dynamic6_field_count_exact"
                ].astype(jp.float32)
                metrics["h4/v4_saved_dynamic6_all_finite"] = info[
                    "h4_v4_saved_dynamic6_all_finite"
                ].astype(jp.float32)
                metrics["h4/v4_telemetry_force_shape_valid"] = info[
                    "h4_v4_telemetry_force_shape_valid"
                ].astype(jp.float32)
                metrics["h4/v4_telemetry_force_all_finite"] = info[
                    "h4_v4_telemetry_force_all_finite"
                ].astype(jp.float32)
                metrics["h4/v4_single_authority_violation"] = info[
                    "h4_v4_single_authority_violation"
                ].astype(jp.float32)
                metrics["h4/v4_single_authority_assertion_token"] = info[
                    "h4_v4_single_authority_assertion_token"
                ].astype(jp.float32)
                metrics["h4/v4_raw_left_contact"] = info[
                    "h4_previous_force_contact"
                ][0].astype(jp.float32)
                metrics["h4/v4_raw_right_contact"] = info[
                    "h4_previous_force_contact"
                ][1].astype(jp.float32)
                metrics["h4/v4_qualified_left_contact"] = info[
                    "h4_v4_qualified_contact"
                ][0].astype(jp.float32)
                metrics["h4/v4_qualified_right_contact"] = info[
                    "h4_v4_qualified_contact"
                ][1].astype(jp.float32)
                metrics["h4/v4_confirmed_transition_count"] = jp.sum(
                    info["h4_v4_confirmed_transition_count"]
                ).astype(jp.float32)
                metrics["h4/v4_touchdown_event_count"] = jp.sum(
                    info["h4_v4_touchdown_event_count"]
                ).astype(jp.float32)
                metrics["h4/v4_aborted_contact_island_count"] = jp.sum(
                    info["h4_v4_aborted_contact_island_count"]
                ).astype(jp.float32)
                metrics["h4/v4_aborted_off_gap_count"] = jp.sum(
                    info["h4_v4_aborted_off_gap_count"]
                ).astype(jp.float32)
                metrics["h4/v4_aborted_contact_island_loss"] = jp.sum(
                    info["h4_v4_aborted_contact_island_loss_sum"]
                )
                metrics["h4/v4_aborted_off_gap_loss"] = jp.sum(
                    info["h4_v4_aborted_off_gap_loss_sum"]
                )
                metrics["h4/v4_terminal_pending_discarded"] = info[
                    "h4_v4_terminal_pending_discarded"
                ].astype(jp.float32)
            if forward_iteration_v6_contact_abort_island_only:
                metrics["h4/v6_forward_contact_abort_routing_exact"] = info[
                    "h4_v6_forward_contact_abort_routing_exact"
                ].astype(jp.float32)
                metrics["h4/v6_forward_contact_abort_island_loss"] = info[
                    "h4_v6_forward_contact_abort_island_loss"
                ]
                metrics[
                    "h4/v6_forward_contact_abort_off_gap_diagnostic_loss"
                ] = info[
                    "h4_v6_forward_contact_abort_off_gap_diagnostic_loss"
                ]
                metrics[
                    "h4/v6_forward_contact_abort_off_gap_reward_contribution"
                ] = info[
                    "h4_v6_forward_contact_abort_off_gap_reward_contribution"
                ]
                metrics[
                    "h4/v6_forward_contact_abort_pulse_reward_scale"
                ] = info["h4_v6_forward_contact_abort_pulse_reward_scale"]
                metrics[
                    "h4/v6_forward_contact_abort_routing_violation"
                ] = info[
                    "h4_v6_forward_contact_abort_routing_violation"
                ].astype(jp.float32)
                metrics[
                    "h4/v6_forward_contact_abort_routing_assertion_token"
                ] = info[
                    "h4_v6_forward_contact_abort_routing_assertion_token"
                ].astype(jp.float32)
            if reverse_iteration_v6_absolute_full_leg_targets:
                metrics["h4/v6_reverse_decoder_exact"] = info[
                    "h4_v6_reverse_decoder_exact"
                ].astype(jp.float32)
                metrics["h4/v6_reverse_decoder_max_abs_error"] = info[
                    "h4_v6_reverse_decoder_max_abs_error"
                ]
                metrics["h4/v6_reverse_decoder_leg_count"] = info[
                    "h4_v6_reverse_decoder_leg_count"
                ].astype(jp.float32)
                metrics["h4/v6_reverse_decoder_leg_count_exact"] = info[
                    "h4_v6_reverse_decoder_leg_count_exact"
                ].astype(jp.float32)
                metrics["h4/v6_reverse_decoder_head_zero_exact"] = info[
                    "h4_v6_reverse_decoder_head_zero_exact"
                ].astype(jp.float32)
                metrics[
                    "h4/v6_reverse_teacher_target_contribution_zero_exact"
                ] = info[
                    "h4_v6_reverse_teacher_target_contribution_zero_exact"
                ].astype(jp.float32)
                metrics["h4/v6_reverse_residual_authority_scale"] = info[
                    "h4_v6_reverse_residual_authority_scale"
                ]
                metrics["h4/v6_reverse_decoder_all_finite"] = info[
                    "h4_v6_reverse_decoder_all_finite"
                ].astype(jp.float32)
                metrics[
                    "h4/v6_reverse_decoder_margin_saturation_count"
                ] = info[
                    "h4_v6_reverse_decoder_margin_saturation_count"
                ].astype(jp.float32)
                metrics["h4/v6_reverse_decoder_action_clip_count"] = info[
                    "h4_v6_reverse_decoder_action_clip_count"
                ].astype(jp.float32)
                metrics["h4/v6_reverse_decoder_guard_lag_max_rad"] = info[
                    "h4_v6_reverse_decoder_guard_lag_max_rad"
                ]
                metrics["h4/v6_reverse_precomposer_call_count"] = info[
                    "h4_v6_reverse_precomposer_call_count"
                ].astype(jp.float32)
                metrics[
                    "h4/v6_reverse_precomposer_call_count_exact"
                ] = info[
                    "h4_v6_reverse_precomposer_call_count_exact"
                ].astype(jp.float32)
                metrics["h4/v6_reverse_final_guard_call_count"] = info[
                    "h4_v6_reverse_final_guard_call_count"
                ].astype(jp.float32)
                metrics[
                    "h4/v6_reverse_final_guard_call_count_exact"
                ] = info[
                    "h4_v6_reverse_final_guard_call_count_exact"
                ].astype(jp.float32)
                metrics["h4/v6_reverse_decoder_violation"] = info[
                    "h4_v6_reverse_decoder_violation"
                ].astype(jp.float32)
                metrics["h4/v6_reverse_decoder_assertion_token"] = info[
                    "h4_v6_reverse_decoder_assertion_token"
                ].astype(jp.float32)
            if h5_absolute_target_routing:
                metrics["h5/target_space_decoder_exact"] = info[
                    "h5_target_space_decoder_exact"
                ].astype(jp.float32)
                metrics["h5/target_space_decoder_domain"] = jp.asarray(
                    0 if h5_target_domain == "planar" else 1,
                    dtype=jp.float32,
                )
            return result.replace(obs=obs, info=info, metrics=metrics)

    H4AlignedEnvironment.__name__ = f"H4Aligned{legacy_environment_class.__name__}"
    return H4AlignedEnvironment


__all__ = [
    "ACTION_COUNT",
    "AlternationUpdate",
    "DEFAULT_ALTERNATION_SIGMA_SECONDS",
    "DEFAULT_ALTERNATION_TARGET_SECONDS",
    "FORCE_CONTACT_OFF_NORMALIZED",
    "FORCE_CONTACT_ON_NORMALIZED",
    "FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID",
    "ForwardV6ContactAbortRoutingTelemetry",
    "FootContactQuality",
    "ReversePhaseQualityLosses",
    "SupportQualityUpdate",
    "H4QualityRewardScales",
    "H4_ACTOR_OBSERVATION_WIDTH",
    "H4_FORWARD_EXACT_ENDPOINT_PROBABILITY",
    "H4_FORWARD_LOCAL_ANCHORS_M_S",
    "H4_FORWARD_PRIMARY_ANCHOR_M_S",
    "H4_FORWARD_TRANSITION_BAND_M_S",
    "H4_FORWARD_V2_EXACT_ENDPOINT_PROBABILITY",
    "H4_FORWARD_V2_STAND_PROBABILITY",
    "H4_FORWARD_V2_LOCAL_ANCHOR_PROBABILITY",
    "H4_FORWARD_V2_TRANSITION_PROBABILITY",
    "H4_REVERSE_EXACT_ENDPOINT_PROBABILITY",
    "H4_REVERSE_LOCAL_ANCHORS_M_S",
    "H4_REVERSE_PRIMARY_ANCHOR_M_S",
    "H4_REVERSE_TRANSITION_BAND_M_S",
    "H4_REVERSE_V2_EXACT_ENDPOINT_PROBABILITY",
    "H4_REVERSE_V2_STAND_PROBABILITY",
    "H4_REVERSE_V2_LOCAL_ANCHOR_PROBABILITY",
    "H4_REVERSE_V2_TRANSITION_PROBABILITY",
    "REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID",
    "REVERSE_ITERATION_V6_BASE_ACTION_SPAN_RAD",
    "REVERSE_ITERATION_V6_DIRECTIONAL_SPAN_FRACTION",
    "ReverseV6AbsoluteFullLegTargetTelemetry",
    "ReverseV6AbsoluteFullLegWiringAudit",
    "ReverseV6StructuralCountInvariants",
    "LEGACY_ACTOR_OBSERVATION_WIDTH",
    "HEAD_ACTION_INDICES",
    "LEG_ACTION_INDICES",
    "MAX_TARGET_DELTA_PER_TICK_RAD",
    "NumpyTargetGuard",
    "OBSERVATION_MOTOR_TARGET_SLICE",
    "OBSERVATION_POLICY_COMMAND_SLICE",
    "H4_OBSERVATION_PHYSICAL_COMMAND_SLICE",
    "H4_OBSERVATION_SLIP_SPEED_SLICE",
    "OBSERVATION_IMITATION_PHASE_SLICE",
    "LEGACY_PRIVILEGED_REFERENCE_SLICE",
    "LEGACY_PRIVILEGED_IMITATION_INDEX_SLICE",
    "LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE",
    "STRICT_SLIP_RMS_M_S",
    "STRICT_TOTAL_NORMAL_FORCE_LOWER_NORMALIZED",
    "STRICT_TOTAL_NORMAL_FORCE_UPPER_NORMALIZED",
    "STRICT_TOTAL_NORMAL_FORCE_BAND_WIDTH_NORMALIZED",
    "STRICT_TOTAL_NORMAL_FORCE_TAIL_NORMALIZED",
    "CONTACT_PULSE_MINIMUM_RUN_TICKS",
    "V4_PHYSICS_SUBSTEP_DT_S",
    "V4_CONTROL_SUBSTEP_COUNT",
    "V4_CONTACT_PERSISTENCE_SECONDS",
    "V4_CONTACT_PERSISTENCE_INTERVALS",
    "V4ContactPersistenceState",
    "V4ContactPersistenceUpdate",
    "V4ContactTelemetryState",
    "V4ContactTelemetryUpdate",
    "V4SubstepContactQualityTrajectory",
    "V4SavedDynamicState",
    "V4SourceSemanticPreflight",
    "V4SubstepTelemetrySummary",
    "V4TrajectoryParity",
    "aggregate_force_contact_quality",
    "audit_v4_dynamic6_parity",
    "audit_v4_dynamic_endpoint_self_consistency",
    "audit_v4_source_semantic_reference",
    "contract_reset_noise_vector",
    "contract_target_vectors",
    "final_target_guard_step",
    "force_schmitt_contacts",
    "forward_iteration_v6_contact_abort_island_only_reward_loss",
    "forward_iteration_v6_contact_abort_island_only_telemetry",
    "initialize_v4_contact_telemetry",
    "robot_body_weight_n",
    "make_h4_reverse_physical_sampler",
    "make_h4_reverse_v2_physical_sampler",
    "make_h4_forward_physical_sampler",
    "make_h4_forward_v2_physical_sampler",
    "make_v4_compiled_single_authority_assertion",
    "make_v6_compiled_invariant_assertion",
    "make_anchor_command_mapper",
    "make_h4_aligned_environment_class",
    "margin_clip_targets",
    "project_reset_qpos",
    "require_checkpoint_observation_compatibility",
    "require_v4_single_authority_invariants",
    "reconstruct_v4_dynamic_state",
    "save_v4_dynamic_state",
    "scan_v4_instrumented_physics_trajectory",
    "scan_v4_saved_state_contact_quality_trajectory",
    "scan_v4_saved_state_contact_telemetry",
    "scan_v4_contact_telemetry_two_phase_reference",
    "synchronize_observation_motor_targets",
    "synchronize_post_step_command_observations",
    "synchronize_post_step_imitation_phase",
    "synchronize_post_step_imitation_state",
    "transplant_v22_checkpoint_to_h4_observation",
    "update_alternation_state",
    "update_load_balance_ema",
    "update_support_quality_state",
    "total_normal_force_quality",
    "update_contact_pulse_state",
    "update_v4_contact_persistence",
    "update_v4_contact_telemetry",
    "v4_aborted_transition_loss",
    "v4_authoritative_primitive_step",
    "v4_saved_dynamic_trajectory_all_finite",
    "discard_v4_terminal_incomplete",
    "quaternion_yaw_wxyz",
    "wrapped_angle_difference",
    "reverse_phase_conditioned_quality_losses",
    "reverse_iteration_v6_absolute_full_leg_target_telemetry",
    "reverse_iteration_v6_absolute_full_leg_target_wiring_audit",
    "reverse_iteration_v6_absolute_full_leg_targets",
    "reverse_iteration_v6_teacher_timing_only_reference",
    "reverse_iteration_v6_structural_count_invariants",
]
