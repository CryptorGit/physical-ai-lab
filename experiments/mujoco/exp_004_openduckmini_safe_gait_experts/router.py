"""Safety-oriented command router for OpenDuckMini gait experts.

The router deliberately does not expose either of the rejected v59/v60
all-direction policies.  It turns a requested ``(vx, vy, yaw_rate)`` command
into one small, direction-specific expert selection.  Classification happens
after clipping, slew-rate limiting, and deadband/hysteresis filtering so an
expert cannot see a command step that bypasses those guards.  The validated
reverse-turn profiles are atomic maneuvers: they are entered and exited only
through stand and always receive their exact evaluated command endpoint.

Positive ``vx`` is forward, positive ``vy`` is left, and positive yaw rate is
a left turn.  Commands and rates use SI units (m/s and rad/s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Mapping, Sequence

import numpy as np


STAND: Final = "stand"
FORWARD: Final = "forward"
REVERSE: Final = "reverse"
REVERSE_TURN_LEFT: Final = "reverse_turn_left"
REVERSE_TURN_RIGHT: Final = "reverse_turn_right"
LATERAL_LEFT: Final = "lateral_left"
LATERAL_RIGHT: Final = "lateral_right"
YAW_LEFT: Final = "yaw_left"
YAW_RIGHT: Final = "yaw_right"
COMPOUND: Final = "compound"

# Frozen simulation envelope from the exp_004 routed calibration.  The reverse
# cap is deliberately asymmetric: -0.050 is the truthful physical endpoint for
# the slower feedforward reverse gait; the legacy -0.075 label left too little
# signed-progress margin in no-reset transitions.
DEFAULT_COMMAND_MIN: Final = (-0.050, -0.16, -1.00)
DEFAULT_COMMAND_MAX: Final = (0.10, 0.16, 1.00)

# Reverse-turn periodic profiles were accepted only at these command points.
# They must not be interpolated with another profile or driven at an
# intermediate command.
REVERSE_TURN_ENDPOINTS: Final[Mapping[str, tuple[float, float, float]]] = (
    MappingProxyType(
        {
            REVERSE_TURN_LEFT: (-0.03, 0.0, 0.20),
            REVERSE_TURN_RIGHT: (-0.04, 0.0, -0.20),
        }
    )
)
ATOMIC_EXPERTS: Final = frozenset(REVERSE_TURN_ENDPOINTS)

ALLOWED_EXPERTS: Final = frozenset(
    {
        STAND,
        FORWARD,
        REVERSE,
        REVERSE_TURN_LEFT,
        REVERSE_TURN_RIGHT,
        LATERAL_LEFT,
        LATERAL_RIGHT,
        YAW_LEFT,
        YAW_RIGHT,
        COMPOUND,
    }
)

# These names are intentionally explicit so packaging/evaluation code can
# assert that a rejected single omnidirectional policy was never selected.
PROHIBITED_EXPERTS: Final = frozenset(
    {
        "v59",
        "v60",
        "all_direction_v59",
        "all_direction_v60",
        "omnidirectional_v59",
        "omnidirectional_v60",
    }
)

# Classification priority, highest first.  Reverse+yaw has a dedicated expert
# and therefore wins over the generic multi-axis rule.  Reverse+lateral is
# still compound because there is no dedicated reverse-strafe expert.
ROUTING_PRIORITY: Final = (
    "stand (no active axes)",
    "reverse_turn_left/right (reverse + yaw, without lateral)",
    "compound (two or more active axes)",
    "reverse/forward (longitudinal only)",
    "lateral_left/right (lateral only)",
    "yaw_left/right (yaw only)",
)

HEAD_LOCK_METADATA: Final[Mapping[str, object]] = MappingProxyType(
    {
        "head_locked": True,
        "head_target_rad": (0.0, 0.0, 0.0, 0.0),
        "head_action_indices": (5, 6, 7, 8),
    }
)


def _finite_triplet(name: str, value: Sequence[float]) -> np.ndarray:
    """Converts *value* to a finite float64 vector without mutating state."""

    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three numeric values") from exc
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


@dataclass(frozen=True)
class RouterConfig:
    """Limits and switching guards used by :class:`SafeGaitRouter`.

    ``deadband`` is the centre of each axis' hysteresis band.  An inactive axis
    enters at ``deadband + hysteresis`` and an active axis exits at
    ``deadband - hysteresis``.  Equality enters at the upper boundary and exits
    at the lower boundary, making boundary behaviour deterministic.
    """

    linear_deadband: float = 0.025
    yaw_deadband: float = 0.08
    linear_hysteresis: float = 0.008
    yaw_hysteresis: float = 0.025
    command_min: tuple[float, float, float] = DEFAULT_COMMAND_MIN
    command_max: tuple[float, float, float] = DEFAULT_COMMAND_MAX
    slew_rate: tuple[float, float, float] = (0.35, 0.35, 2.00)
    blend_duration_s: float = 0.25

    def __post_init__(self) -> None:
        scalars = np.asarray(
            (
                self.linear_deadband,
                self.yaw_deadband,
                self.linear_hysteresis,
                self.yaw_hysteresis,
                self.blend_duration_s,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars < 0.0):
            raise ValueError("deadbands, hysteresis, and blend duration must be finite and non-negative")
        if self.linear_hysteresis > self.linear_deadband:
            raise ValueError("linear_hysteresis cannot exceed linear_deadband")
        if self.yaw_hysteresis > self.yaw_deadband:
            raise ValueError("yaw_hysteresis cannot exceed yaw_deadband")

        minimum = _finite_triplet("command_min", self.command_min)
        maximum = _finite_triplet("command_max", self.command_max)
        rates = _finite_triplet("slew_rate", self.slew_rate)
        if np.any(minimum >= maximum):
            raise ValueError("command_min must be strictly below command_max")
        if np.any(minimum > 0.0) or np.any(maximum < 0.0):
            raise ValueError("the command envelope must contain zero")
        if np.any(rates <= 0.0):
            raise ValueError("slew_rate values must be positive")


@dataclass(frozen=True)
class RouteDecision:
    """One guarded routing decision.

    ``blend_alpha`` is the weight of ``expert``.  While it is below one, the
    caller should mix the previous expert's action using ``1 - blend_alpha``.
    Head action indices must be overwritten with ``head_target_rad`` *after*
    any such blend.
    """

    expert: str
    requested_command: tuple[float, float, float]
    clipped_command: tuple[float, float, float]
    ramped_command: tuple[float, float, float]
    effective_command: tuple[float, float, float]
    active_axes: tuple[bool, bool, bool]
    command_was_clipped: bool
    switched: bool
    blend_from_expert: str
    blend_to_expert: str
    blend_alpha: float
    routing_reason: str
    metadata: Mapping[str, object] = field(default=HEAD_LOCK_METADATA)

    def __post_init__(self) -> None:
        if self.expert not in ALLOWED_EXPERTS:
            raise ValueError(f"unapproved expert selected: {self.expert!r}")
        if self.expert in PROHIBITED_EXPERTS:
            raise ValueError(f"prohibited all-direction expert selected: {self.expert!r}")
        if not 0.0 <= self.blend_alpha <= 1.0:
            raise ValueError("blend_alpha must be in [0, 1]")

    @property
    def head_locked(self) -> bool:
        return bool(self.metadata["head_locked"])


class SafeGaitRouter:
    """Stateful, deterministic router for direction-specific gait experts."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        self.reset()

    def reset(self) -> None:
        """Resets slew, hysteresis, and blending state to a locked-head stand."""

        self._ramped_command = np.zeros(3, dtype=np.float64)
        self._active_axes = np.zeros(3, dtype=np.bool_)
        self._selected_expert = STAND
        self._blend_from_expert = STAND
        self._blend_to_expert = STAND
        self._blend_elapsed_s = self.config.blend_duration_s

    @property
    def selected_expert(self) -> str:
        return self._selected_expert

    @property
    def ramped_command(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._ramped_command)

    def route(self, command: Sequence[float], dt: float) -> RouteDecision:
        """Routes one command sample.

        Invalid commands and invalid ``dt`` values are rejected before any
        internal state is changed.  Values outside the configured operating
        envelope are clipped per bound and reported in the decision.  A
        reverse-turn request is snapped to its accepted endpoint.  Switching
        into, out of, or between reverse-turn profiles passes through an exact
        stand decision and never action-blends the profiles.
        """

        requested = _finite_triplet("command", command)
        try:
            dt_value = float(dt)
        except (TypeError, ValueError) as exc:
            raise ValueError("dt must be a finite positive scalar") from exc
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError("dt must be a finite positive scalar")

        minimum = np.asarray(self.config.command_min, dtype=np.float64)
        maximum = np.asarray(self.config.command_max, dtype=np.float64)
        clipped = np.clip(requested, minimum, maximum)
        was_clipped = not np.array_equal(requested, clipped)

        atomic_target = self._atomic_target(clipped)
        if self._selected_expert in ATOMIC_EXPERTS and (
            atomic_target != self._selected_expert
        ):
            return self._atomic_stand_interlock(
                requested=requested,
                clipped=clipped,
                was_clipped=was_clipped,
                reason="atomic reverse-turn exit passes through stand",
            )
        if atomic_target is not None:
            if self._selected_expert != STAND and (
                self._selected_expert != atomic_target
            ):
                return self._atomic_stand_interlock(
                    requested=requested,
                    clipped=clipped,
                    was_clipped=was_clipped,
                    reason="atomic reverse-turn entry passes through stand",
                )
            return self._atomic_decision(
                expert=atomic_target,
                requested=requested,
                clipped=clipped,
                was_clipped=was_clipped,
            )

        max_delta = np.asarray(self.config.slew_rate, dtype=np.float64) * dt_value
        delta = np.clip(clipped - self._ramped_command, -max_delta, max_delta)
        next_ramped = self._ramped_command + delta

        deadbands = np.asarray(
            (self.config.linear_deadband, self.config.linear_deadband, self.config.yaw_deadband),
            dtype=np.float64,
        )
        hysteresis = np.asarray(
            (
                self.config.linear_hysteresis,
                self.config.linear_hysteresis,
                self.config.yaw_hysteresis,
            ),
            dtype=np.float64,
        )
        magnitudes = np.abs(next_ramped)
        enter_threshold = deadbands + hysteresis
        exit_threshold = deadbands - hysteresis
        # Treat mathematically equal decimal boundaries as equal even when the
        # subtraction above lands one ULP to either side in binary float.
        boundary_tolerance = np.finfo(np.float64).eps * 16.0 * np.maximum(1.0, deadbands)
        at_enter_boundary = np.abs(magnitudes - enter_threshold) <= boundary_tolerance
        at_exit_boundary = np.abs(magnitudes - exit_threshold) <= boundary_tolerance
        enters = (magnitudes > enter_threshold) | at_enter_boundary
        remains_active = (magnitudes > exit_threshold) & ~at_exit_boundary
        next_active = np.where(
            self._active_axes,
            remains_active,
            enters,
        )
        effective = np.where(next_active, next_ramped, 0.0)

        expert, reason = self._classify(effective, next_active)
        if expert not in ALLOWED_EXPERTS or expert in PROHIBITED_EXPERTS:
            raise RuntimeError(f"router produced an unsafe expert name: {expert!r}")

        switched = expert != self._selected_expert
        if switched:
            blend_from = self._selected_expert
            blend_to = expert
            blend_elapsed = 0.0
            blend_alpha = 1.0 if self.config.blend_duration_s == 0.0 else 0.0
        elif self._blend_to_expert == expert and self._blend_elapsed_s < self.config.blend_duration_s:
            blend_from = self._blend_from_expert
            blend_to = expert
            blend_elapsed = min(
                self.config.blend_duration_s,
                self._blend_elapsed_s + dt_value,
            )
            blend_alpha = (
                1.0
                if self.config.blend_duration_s == 0.0
                else blend_elapsed / self.config.blend_duration_s
            )
        else:
            blend_from = expert
            blend_to = expert
            blend_elapsed = self.config.blend_duration_s
            blend_alpha = 1.0

        # Commit only after the complete decision is valid.
        self._ramped_command = next_ramped
        self._active_axes = next_active
        self._selected_expert = expert
        self._blend_from_expert = blend_from
        self._blend_to_expert = blend_to
        self._blend_elapsed_s = blend_elapsed

        return RouteDecision(
            expert=expert,
            requested_command=tuple(float(value) for value in requested),
            clipped_command=tuple(float(value) for value in clipped),
            ramped_command=tuple(float(value) for value in next_ramped),
            effective_command=tuple(float(value) for value in effective),
            active_axes=tuple(bool(value) for value in next_active),
            command_was_clipped=was_clipped,
            switched=switched,
            blend_from_expert=blend_from,
            blend_to_expert=blend_to,
            blend_alpha=float(blend_alpha),
            routing_reason=reason,
            metadata=HEAD_LOCK_METADATA,
        )

    def _atomic_target(self, clipped: np.ndarray) -> str | None:
        """Return a validated reverse-turn intent before continuous ramping."""

        vx, vy, yaw = (float(value) for value in clipped)
        linear_enter = self.config.linear_deadband + self.config.linear_hysteresis
        validated_reverse_turn_vx = min(
            abs(endpoint[0]) for endpoint in REVERSE_TURN_ENDPOINTS.values()
        )
        atomic_vx_threshold = min(linear_enter, validated_reverse_turn_vx)
        yaw_enter = self.config.yaw_deadband + self.config.yaw_hysteresis
        lateral_exit = self.config.linear_deadband - self.config.linear_hysteresis
        if (
            vx > -atomic_vx_threshold
            or abs(vy) > lateral_exit
            or abs(yaw) < yaw_enter
        ):
            return None
        return REVERSE_TURN_LEFT if yaw > 0.0 else REVERSE_TURN_RIGHT

    def _atomic_stand_interlock(
        self,
        *,
        requested: np.ndarray,
        clipped: np.ndarray,
        was_clipped: bool,
        reason: str,
    ) -> RouteDecision:
        """Commit one exact stand sample between atomic profile changes."""

        previous = self._selected_expert
        self._ramped_command = np.zeros(3, dtype=np.float64)
        self._active_axes = np.zeros(3, dtype=np.bool_)
        self._selected_expert = STAND
        self._blend_from_expert = STAND
        self._blend_to_expert = STAND
        self._blend_elapsed_s = self.config.blend_duration_s
        return RouteDecision(
            expert=STAND,
            requested_command=tuple(float(value) for value in requested),
            clipped_command=tuple(float(value) for value in clipped),
            ramped_command=(0.0, 0.0, 0.0),
            effective_command=(0.0, 0.0, 0.0),
            active_axes=(False, False, False),
            command_was_clipped=was_clipped,
            switched=previous != STAND,
            blend_from_expert=STAND,
            blend_to_expert=STAND,
            blend_alpha=1.0,
            routing_reason=reason,
            metadata=HEAD_LOCK_METADATA,
        )

    def _atomic_decision(
        self,
        *,
        expert: str,
        requested: np.ndarray,
        clipped: np.ndarray,
        was_clipped: bool,
    ) -> RouteDecision:
        """Snap an atomic reverse-turn to its evaluated command point."""

        endpoint = np.asarray(REVERSE_TURN_ENDPOINTS[expert], dtype=np.float64)
        previous = self._selected_expert
        self._ramped_command = endpoint.copy()
        self._active_axes = np.asarray((True, False, True), dtype=np.bool_)
        self._selected_expert = expert
        # An atomic profile is never blended with stand or another profile.
        self._blend_from_expert = expert
        self._blend_to_expert = expert
        self._blend_elapsed_s = self.config.blend_duration_s
        return RouteDecision(
            expert=expert,
            requested_command=tuple(float(value) for value in requested),
            clipped_command=tuple(float(value) for value in clipped),
            ramped_command=tuple(float(value) for value in endpoint),
            effective_command=tuple(float(value) for value in endpoint),
            active_axes=(True, False, True),
            command_was_clipped=was_clipped,
            switched=previous != expert,
            blend_from_expert=expert,
            blend_to_expert=expert,
            blend_alpha=1.0,
            routing_reason=(
                "atomic validated reverse-turn endpoint; profile blending prohibited"
            ),
            metadata=HEAD_LOCK_METADATA,
        )

    @staticmethod
    def _classify(command: np.ndarray, active: np.ndarray) -> tuple[str, str]:
        vx, vy, yaw = (float(value) for value in command)
        active_x, active_y, active_yaw = (bool(value) for value in active)
        active_count = int(active_x) + int(active_y) + int(active_yaw)

        if active_count == 0:
            return STAND, ROUTING_PRIORITY[0]

        # Dedicated reverse-turn experts take precedence over compound routing.
        if active_x and vx < 0.0 and active_yaw and not active_y:
            if yaw > 0.0:
                return REVERSE_TURN_LEFT, ROUTING_PRIORITY[1]
            return REVERSE_TURN_RIGHT, ROUTING_PRIORITY[1]

        if active_count >= 2:
            return COMPOUND, ROUTING_PRIORITY[2]

        if active_x:
            if vx < 0.0:
                return REVERSE, ROUTING_PRIORITY[3]
            return FORWARD, ROUTING_PRIORITY[3]
        if active_y:
            if vy > 0.0:
                return LATERAL_LEFT, ROUTING_PRIORITY[4]
            return LATERAL_RIGHT, ROUTING_PRIORITY[4]
        if yaw > 0.0:
            return YAW_LEFT, ROUTING_PRIORITY[5]
        return YAW_RIGHT, ROUTING_PRIORITY[5]


__all__ = [
    "ALLOWED_EXPERTS",
    "ATOMIC_EXPERTS",
    "COMPOUND",
    "DEFAULT_COMMAND_MAX",
    "DEFAULT_COMMAND_MIN",
    "FORWARD",
    "HEAD_LOCK_METADATA",
    "LATERAL_LEFT",
    "LATERAL_RIGHT",
    "PROHIBITED_EXPERTS",
    "REVERSE",
    "REVERSE_TURN_LEFT",
    "REVERSE_TURN_ENDPOINTS",
    "REVERSE_TURN_RIGHT",
    "ROUTING_PRIORITY",
    "RouteDecision",
    "RouterConfig",
    "STAND",
    "SafeGaitRouter",
    "YAW_LEFT",
    "YAW_RIGHT",
]
