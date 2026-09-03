"""Final OpenDuckMini actuator-target safety guard for exp_004 packages.

The stateful guard consumes *joint targets* after policy/profile composition.
It limits every leg target to 2.0 rad/s, applies a final clamp inside the
packaged SAFE limits with a frozen 0.050 rad inward margin, and forces all head
targets to exact zero.  Hardware deployment remains prohibited; this is a
simulation runtime invariant, not hardware approval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np


ACTUATOR_JOINT_ORDER: Final = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
HEAD_ACTION_INDICES: Final = (5, 6, 7, 8)
LEG_ACTION_INDICES: Final = tuple(
    index
    for index in range(len(ACTUATOR_JOINT_ORDER))
    if index not in HEAD_ACTION_INDICES
)
LEG_JOINT_NAMES: Final = tuple(
    ACTUATOR_JOINT_ORDER[index] for index in LEG_ACTION_INDICES
)
RUNTIME_TARGET_SAFETY_MARGIN_RAD: Final = 0.050
RUNTIME_TARGET_SLEW_RATE_RAD_S: Final = 2.0
PERTURBED_RESET_QPOS_MARGIN_RAD: Final = 0.005
CONTROL_FIRST_STARTUP_DT_S: Final = 0.02
RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD: Final = (
    RUNTIME_TARGET_SLEW_RATE_RAD_S * CONTROL_FIRST_STARTUP_DT_S
)
BACKWARD_EXIT_RECOVERY_STATUS: Final = "ADOPTED_SIMULATION_ONLY"
BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT: Final = True
BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256: Final = (
    "f040a9c6f9783b7d50dd5590389d3c81411e8f3a7fa9dd155e8ac78175d5ff56"
)
BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_SELECTION_EVIDENCE_SHA256: Final = (
    "6f65bef5053da5962442eca3bf46b855a36691aa9bbad84496c9892b36ee0de4"
)
BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_ADOPTION_EVIDENCE_SHA256: Final = (
    "bd7e8a79b32880fa63e54570854682b5b8912f1cdafeed8e80273501dc6ef611"
)
BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256: Final = (
    "1aea58904598cfba8ea4ef572f9473bba647eacc695f7fce3fcaa1b8646391aa"
)
BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256: Final = (
    "090e09cc2d82c1f42112a5f30a85cd93d940213956d6ec902fb4089875fb855a"
)
BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD: Final = 0.0225
BACKWARD_EXIT_RECOVERY_HOLD_TICKS: Final = 13
BACKWARD_EXIT_RECOVERY_HOLD_SECONDS: Final = (
    BACKWARD_EXIT_RECOVERY_HOLD_TICKS * CONTROL_FIRST_STARTUP_DT_S
)
BACKWARD_EXIT_RECOVERY_LEFT_KNEE_SAFE_UPPER_RAD: Final = 0.475534
BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD: Final = (
    BACKWARD_EXIT_RECOVERY_LEFT_KNEE_SAFE_UPPER_RAD
    - RUNTIME_TARGET_SAFETY_MARGIN_RAD
    - BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
)
LEFT_KNEE_ACTION_INDEX: Final = ACTUATOR_JOINT_ORDER.index("left_knee")


def backward_exit_recovery_contract() -> dict[str, object]:
    """Return the simulation-only adopted H3 recovery contract."""

    return {
        "status": BACKWARD_EXIT_RECOVERY_STATUS,
        "enabled_by_default": BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT,
        "formal_candidate_only": False,
        "diagnostic_unadopted_only": False,
        "adoption_eligible": True,
        "simulation_acceptance_eligible": True,
        "candidate_selection_evidence_sha256": (
            BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256
        ),
        "superseded_h2_selection_evidence_sha256": (
            BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_SELECTION_EVIDENCE_SHA256
        ),
        "superseded_h2_adoption_evidence_sha256": (
            BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_ADOPTION_EVIDENCE_SHA256
        ),
        "adoption_evidence_sha256": (
            BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256
        ),
        "safety_component_evidence_sha256": (
            BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256
        ),
        "safety_component_only": False,
        "safety_component_evidence_is_safety_only": True,
        "fast_exit_safety_passed": True,
        "combined_5x15_required": False,
        "combined_5x15_passed": True,
        "requires_formal_20x30_requalification": False,
        "activation": "backward_feedforward_active_true_to_false",
        "exit_tick_is_first_active_tick": True,
        "joint_name": "left_knee",
        "joint_index": LEFT_KNEE_ACTION_INDEX,
        "safe_upper_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_SAFE_UPPER_RAD,
        "base_target_margin_rad": RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        "extra_upper_margin_rad": (
            BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        ),
        "upper_target_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
        "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
        "control_dt_seconds": CONTROL_FIRST_STARTUP_DT_S,
        "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
        "release": "instant_after_hold",
        "backward_reentry": "cancel_remaining_recovery",
        "reset": "clear_active_state_and_history",
        "composition_stage": "after_policy_or_profile_before_final_target_guard",
        "final_guard_calls_per_tick": 1,
        "hardware_deployment": "PROHIBITED",
    }


class BackwardExitRecovery:
    """Adopted simulation-only H3 left-knee composition after backward exits.

    This class is intentionally separate from :class:`FinalTargetSafetyGuard`.
    :meth:`compose` runs before that guard and never performs slew or physical
    limit enforcement itself. It is enabled on the formal candidate path and
    is bound to an independently audited formal 20x30 adoption record.
    Hardware use remains prohibited.
    """

    def __init__(
        self,
        safe_joint_limits_rad: Mapping[str, Sequence[float]],
        *,
        enabled: bool = BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT,
    ) -> None:
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("backward-exit recovery enabled must be boolean")
        if set(safe_joint_limits_rad) != set(LEG_JOINT_NAMES):
            raise ValueError(
                "safe_joint_limits_rad must contain exactly all ten leg joints"
            )
        left_knee_bounds = np.asarray(
            safe_joint_limits_rad["left_knee"], dtype=np.float64
        )
        if (
            left_knee_bounds.shape != (2,)
            or not np.all(np.isfinite(left_knee_bounds))
            or float(left_knee_bounds[1])
            != BACKWARD_EXIT_RECOVERY_LEFT_KNEE_SAFE_UPPER_RAD
        ):
            raise ValueError(
                "backward-exit recovery requires left_knee SAFE upper 0.475534"
            )
        self.enabled = bool(enabled)
        self._reset_count = 0
        self._events: list[dict[str, object]] = []
        self._previous_backward_feedforward_active = False
        self._remaining_ticks = 0
        self._control_tick = 0
        self._exit_event_count = 0
        self._active_tick_count = 0
        self._completed_event_count = 0
        self._reentry_cancel_count = 0
        self._cap_violation_count = 0
        self._maximum_composed_left_knee_target_rad: float | None = None
        self.reset()

    @property
    def remaining_ticks(self) -> int:
        return self._remaining_ticks

    @property
    def previous_backward_feedforward_active(self) -> bool:
        return self._previous_backward_feedforward_active

    def reset(self) -> dict[str, object]:
        """Clear all active/event state and return the post-reset audit."""

        self._reset_count += 1
        self._events = []
        self._previous_backward_feedforward_active = False
        self._remaining_ticks = 0
        self._control_tick = 0
        self._exit_event_count = 0
        self._active_tick_count = 0
        self._completed_event_count = 0
        self._reentry_cancel_count = 0
        self._cap_violation_count = 0
        self._maximum_composed_left_knee_target_rad = None
        return self.audit()

    @staticmethod
    def _validated_targets(joint_targets: Sequence[float]) -> np.ndarray:
        try:
            targets = np.asarray(joint_targets, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "backward-exit recovery targets must contain 14 numeric values"
            ) from exc
        if targets.shape != (len(ACTUATOR_JOINT_ORDER),):
            raise ValueError(
                "backward-exit recovery targets must have shape (14,)"
            )
        if not np.all(np.isfinite(targets)):
            raise ValueError(
                "backward-exit recovery targets must contain only finite values"
            )
        return targets.copy()

    def compose(
        self,
        joint_targets: Sequence[float],
        *,
        backward_feedforward_active: bool,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Compose one target vector and atomically advance recovery state."""

        targets = self._validated_targets(joint_targets)
        if not isinstance(backward_feedforward_active, (bool, np.bool_)):
            raise ValueError("backward_feedforward_active must be boolean")
        active = bool(backward_feedforward_active)
        input_left_knee = float(targets[LEFT_KNEE_ACTION_INDEX])
        exit_event = False
        reentry_cancelled = False

        if self.enabled:
            exit_event = bool(
                self._previous_backward_feedforward_active and not active
            )
            if exit_event:
                self._remaining_ticks = BACKWARD_EXIT_RECOVERY_HOLD_TICKS
                self._exit_event_count += 1
                self._events.append(
                    {
                        "event_index": self._exit_event_count,
                        "start_control_tick": self._control_tick,
                        "status": "ACTIVE",
                        "active_tick_count": 0,
                        "cap_upper_target_rad": (
                            BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
                        ),
                    }
                )
            if active and self._remaining_ticks:
                reentry_cancelled = True
                self._remaining_ticks = 0
                self._reentry_cancel_count += 1
                if self._events:
                    self._events[-1].update(
                        {
                            "status": "CANCELLED_BY_BACKWARD_REENTRY",
                            "cancel_control_tick": self._control_tick,
                        }
                    )

        recovery_active = bool(
            self.enabled and not active and self._remaining_ticks > 0
        )
        recovery_tick_index: int | None = None
        if recovery_active:
            recovery_tick_index = (
                BACKWARD_EXIT_RECOVERY_HOLD_TICKS - self._remaining_ticks + 1
            )
            targets[LEFT_KNEE_ACTION_INDEX] = min(
                targets[LEFT_KNEE_ACTION_INDEX],
                BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
            )
            self._active_tick_count += 1
            self._remaining_ticks -= 1
            event = self._events[-1]
            event["active_tick_count"] = int(event["active_tick_count"]) + 1
            if self._remaining_ticks == 0:
                self._completed_event_count += 1
                event.update(
                    {
                        "status": "COMPLETED",
                        "end_control_tick_exclusive": self._control_tick + 1,
                    }
                )

        composed_left_knee = float(targets[LEFT_KNEE_ACTION_INDEX])
        cap_excess = (
            max(
                0.0,
                composed_left_knee
                - BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
            )
            if recovery_active
            else 0.0
        )
        self._cap_violation_count += int(cap_excess > 0.0)
        if recovery_active:
            current_maximum = self._maximum_composed_left_knee_target_rad
            self._maximum_composed_left_knee_target_rad = (
                composed_left_knee
                if current_maximum is None
                else max(current_maximum, composed_left_knee)
            )

        step_audit: dict[str, object] = {
            "enabled": self.enabled,
            "control_tick": self._control_tick,
            "backward_feedforward_active": active,
            "exit_event": exit_event,
            "reentry_cancelled": reentry_cancelled,
            "recovery_active": recovery_active,
            "recovery_tick_index": recovery_tick_index,
            "remaining_ticks_after_step": self._remaining_ticks,
            "input_left_knee_target_rad": input_left_knee,
            "composed_left_knee_target_rad": composed_left_knee,
            "cap_upper_target_rad": (
                BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
                if recovery_active
                else None
            ),
            "cap_excess_rad": cap_excess,
            "cap_violation": cap_excess > 0.0,
            "passed": cap_excess == 0.0,
        }
        self._previous_backward_feedforward_active = active
        self._control_tick += 1
        return targets, step_audit

    def audit(self) -> dict[str, object]:
        """Return a serializable audit of state and all events since reset."""

        contract = backward_exit_recovery_contract()
        event_tick_sum = sum(
            int(event["active_tick_count"]) for event in self._events
        )
        event_counts_consistent = event_tick_sum == self._active_tick_count
        disabled_path_inert = bool(
            self.enabled
            or (
                self._exit_event_count == 0
                and self._active_tick_count == 0
                and self._completed_event_count == 0
                and self._reentry_cancel_count == 0
                and self._remaining_ticks == 0
                and not self._events
            )
        )
        return {
            "enabled": self.enabled,
            "formal_candidate_only": False,
            "adopted_simulation_only": True,
            "diagnostic_unadopted_only": False,
            "contract": contract,
            "reset_count": self._reset_count,
            "control_tick_count": self._control_tick,
            "exit_event_count": self._exit_event_count,
            "active_tick_count": self._active_tick_count,
            "completed_event_count": self._completed_event_count,
            "reentry_cancel_count": self._reentry_cancel_count,
            "remaining_ticks": self._remaining_ticks,
            "previous_backward_feedforward_active": (
                self._previous_backward_feedforward_active
            ),
            "cap_violation_count": self._cap_violation_count,
            "maximum_composed_left_knee_target_rad": (
                self._maximum_composed_left_knee_target_rad
            ),
            "events": [dict(event) for event in self._events],
            "event_tick_sum_matches_active_tick_count": event_counts_consistent,
            "disabled_path_inert": disabled_path_inert,
            "passed": bool(
                self._reset_count >= 1
                and self._cap_violation_count == 0
                and event_counts_consistent
                and disabled_path_inert
            ),
        }


def _apply_target_bounds(
    joint_targets: Sequence[float],
    safe_joint_limits_rad: Mapping[str, Sequence[float]],
    *,
    inward_margin_rad: float,
) -> np.ndarray:
    try:
        targets = np.asarray(joint_targets, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint_targets must contain 14 numeric values") from exc
    if targets.shape != (len(ACTUATOR_JOINT_ORDER),):
        raise ValueError(
            f"joint_targets must have shape ({len(ACTUATOR_JOINT_ORDER)},), "
            f"got {targets.shape}"
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("joint_targets must contain only finite values")
    if set(safe_joint_limits_rad) != set(LEG_JOINT_NAMES):
        raise ValueError("safe_joint_limits_rad must contain exactly all ten leg joints")
    try:
        margin = float(inward_margin_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("inward target margin must be finite and non-negative") from exc
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("inward target margin must be finite and non-negative")

    guarded = targets.copy()
    for index, joint_name in zip(LEG_ACTION_INDICES, LEG_JOINT_NAMES, strict=True):
        try:
            bounds = np.asarray(safe_joint_limits_rad[joint_name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid SAFE limits for {joint_name}") from exc
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError(f"SAFE limits for {joint_name} must be two finite values")
        lower = float(bounds[0]) + margin
        upper = float(bounds[1]) - margin
        if lower > upper:
            raise ValueError(f"SAFE range for {joint_name} is narrower than twice the margin")
        guarded[index] = np.clip(guarded[index], lower, upper)

    guarded[np.asarray(HEAD_ACTION_INDICES)] = 0.0
    return guarded


def apply_final_target_safety(
    joint_targets: Sequence[float],
    safe_joint_limits_rad: Mapping[str, Sequence[float]],
    *,
    margin_rad: float = RUNTIME_TARGET_SAFETY_MARGIN_RAD,
) -> np.ndarray:
    """Clamp a *desired* vector to the frozen margin and lock the head.

    The stateful guard uses this before slew.  Its applied output receives a
    second physical-SAFE clamp without an inward margin so reset targets can
    move into the steady-state margin at the declared slew rate.
    """

    margin = FinalTargetSafetyGuard._require_frozen_margin(margin_rad)
    return _apply_target_bounds(
        joint_targets,
        safe_joint_limits_rad,
        inward_margin_rad=margin,
    )


def apply_reset_qpos_safety(
    reset_joint_positions: Sequence[float],
    safe_joint_limits_rad: Mapping[str, Sequence[float]],
    *,
    joint_noise_scale: float,
) -> np.ndarray:
    """Guard reset qpos while preserving the exact zero-noise home pose.

    A positive reset-noise scale uses a 0.005 rad inward physical-SAFE margin
    on all leg qpos.  At exactly zero noise, the supplied physical-SAFE home is
    preserved without an artificial margin jump.  Head qpos are always zero.
    """

    try:
        noise_scale = float(joint_noise_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint_noise_scale must be finite and non-negative") from exc
    if not np.isfinite(noise_scale) or noise_scale < 0.0:
        raise ValueError("joint_noise_scale must be finite and non-negative")
    inward_margin = PERTURBED_RESET_QPOS_MARGIN_RAD if noise_scale > 0.0 else 0.0
    return _apply_target_bounds(
        reset_joint_positions,
        safe_joint_limits_rad,
        inward_margin_rad=inward_margin,
    )


class FinalTargetSafetyGuard:
    """Stateful final guard with the frozen per-leg 2.0 rad/s target slew.

    Construct the guard from the currently applied/reset target vector.  Call
    :meth:`step` exactly once per control step after every policy, feedforward,
    residual, correction, or blend operation.  Invalid input never mutates the
    stored state.
    """

    def __init__(
        self,
        safe_joint_limits_rad: Mapping[str, Sequence[float]],
        initial_joint_targets: Sequence[float],
        *,
        margin_rad: float = RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        max_slew_rate_rad_s: float = RUNTIME_TARGET_SLEW_RATE_RAD_S,
    ) -> None:
        self._safe_joint_limits_rad = dict(safe_joint_limits_rad)
        self._margin_rad = self._require_frozen_margin(margin_rad)
        self._max_slew_rate_rad_s = self._require_frozen_slew(
            max_slew_rate_rad_s
        )
        # Reset/home targets may be physical-SAFE but inside the steady-state
        # 0.050 rad margin.  Preserve that state so the first step cannot hide
        # an instantaneous target jump.
        self._previous_targets = _apply_target_bounds(
            initial_joint_targets,
            self._safe_joint_limits_rad,
            inward_margin_rad=0.0,
        )
        self._steps_since_reset = 0

    @staticmethod
    def _require_frozen_margin(value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "margin_rad must remain exactly 0.050 for exp_004"
            ) from exc
        if not np.isfinite(parsed) or parsed != RUNTIME_TARGET_SAFETY_MARGIN_RAD:
            raise ValueError("margin_rad must remain exactly 0.050 for exp_004")
        return parsed

    @staticmethod
    def _require_frozen_slew(value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_slew_rate_rad_s must remain exactly 2.0 for exp_004"
            ) from exc
        if not np.isfinite(parsed) or parsed != RUNTIME_TARGET_SLEW_RATE_RAD_S:
            raise ValueError(
                "max_slew_rate_rad_s must remain exactly 2.0 for exp_004"
            )
        return parsed

    @property
    def previous_targets(self) -> np.ndarray:
        """Return a defensive copy of the last guarded target vector."""

        return self._previous_targets.copy()

    @property
    def steps_since_reset(self) -> int:
        return self._steps_since_reset

    def reset(self, initial_joint_targets: Sequence[float]) -> np.ndarray:
        """Reset slew state to a guarded currently-applied target vector."""

        guarded = _apply_target_bounds(
            initial_joint_targets,
            self._safe_joint_limits_rad,
            inward_margin_rad=0.0,
        )
        self._previous_targets = guarded
        self._steps_since_reset = 0
        return guarded.copy()

    def control_first_startup(
        self,
        first_command_policy_targets: Sequence[float],
        *,
        dt: float = CONTROL_FIRST_STARTUP_DT_S,
    ) -> np.ndarray:
        """Guard the first command-policy target before the first physics step."""

        try:
            dt_value = float(dt)
        except (TypeError, ValueError) as exc:
            raise ValueError("startup dt must remain exactly 0.02 seconds") from exc
        if not np.isfinite(dt_value) or dt_value != CONTROL_FIRST_STARTUP_DT_S:
            raise ValueError("startup dt must remain exactly 0.02 seconds")
        if self._steps_since_reset != 0:
            raise RuntimeError("control-first startup must be the first step after reset")
        return self.step(first_command_policy_targets, dt_value)

    def step(self, desired_joint_targets: Sequence[float], dt: float) -> np.ndarray:
        """Slew and finally clamp one desired target vector, then commit it."""

        try:
            dt_value = float(dt)
        except (TypeError, ValueError) as exc:
            raise ValueError("dt must be a finite positive scalar") from exc
        if not np.isfinite(dt_value) or dt_value != CONTROL_FIRST_STARTUP_DT_S:
            raise ValueError("dt must remain exactly 0.02 seconds for exp_004")

        desired = apply_final_target_safety(
            desired_joint_targets,
            self._safe_joint_limits_rad,
            margin_rad=self._margin_rad,
        )
        max_delta = self._max_slew_rate_rad_s * dt_value
        applied = self._previous_targets.copy()
        leg_indices = np.asarray(LEG_ACTION_INDICES)
        delta = np.clip(
            desired[leg_indices] - self._previous_targets[leg_indices],
            -max_delta,
            max_delta,
        )
        applied[leg_indices] = self._previous_targets[leg_indices] + delta
        # The final applied clamp uses physical SAFE bounds, not the steady
        # margin.  A reset target may start inside SAFE but outside the margin;
        # slew then moves it into the margin without a hidden discontinuity.
        applied = _apply_target_bounds(
            applied,
            self._safe_joint_limits_rad,
            inward_margin_rad=0.0,
        )
        self._previous_targets = applied
        self._steps_since_reset += 1
        return applied.copy()


__all__ = [
    "ACTUATOR_JOINT_ORDER",
    "BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256",
    "BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256",
    "BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256",
    "BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT",
    "BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD",
    "BACKWARD_EXIT_RECOVERY_HOLD_SECONDS",
    "BACKWARD_EXIT_RECOVERY_HOLD_TICKS",
    "BACKWARD_EXIT_RECOVERY_LEFT_KNEE_SAFE_UPPER_RAD",
    "BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD",
    "BACKWARD_EXIT_RECOVERY_STATUS",
    "BackwardExitRecovery",
    "CONTROL_FIRST_STARTUP_DT_S",
    "HEAD_ACTION_INDICES",
    "FinalTargetSafetyGuard",
    "LEG_ACTION_INDICES",
    "LEG_JOINT_NAMES",
    "PERTURBED_RESET_QPOS_MARGIN_RAD",
    "RUNTIME_TARGET_SAFETY_MARGIN_RAD",
    "RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD",
    "RUNTIME_TARGET_SLEW_RATE_RAD_S",
    "backward_exit_recovery_contract",
    "apply_final_target_safety",
    "apply_reset_qpos_safety",
]
