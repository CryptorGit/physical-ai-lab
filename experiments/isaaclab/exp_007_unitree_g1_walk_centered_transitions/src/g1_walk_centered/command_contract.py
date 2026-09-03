"""Stage 0 command schema and fail-closed validation.

This module describes commands only.  It does not dispatch an expert or enable a
transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class Posture(str, Enum):
    STAND = "STAND"
    UPRIGHT = "UPRIGHT"
    CROUCH = "CROUCH"


class CommandKind(str, Enum):
    STAND = "STAND"
    WALK = "WALK"
    RUN = "RUN"
    TURN = "TURN"
    STOP = "STOP"
    CROUCH = "CROUCH"
    STAND_UP = "STAND_UP"


@dataclass(frozen=True)
class MotionCommand:
    target_speed_mps: float
    target_heading_w_rad: float
    posture: str = Posture.UPRIGHT.value
    crouch_depth_m: float = 0.0
    target_yaw_rate_radps: float = 0.0


@dataclass(frozen=True)
class CommandRequest:
    kind: CommandKind
    motion: MotionCommand


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    executable_stage0: bool
    reason: str


def validate_command(request: CommandRequest) -> ValidationResult:
    """Validate the v1 contract without authorizing execution."""
    motion = request.motion
    numeric = (
        motion.target_speed_mps,
        motion.target_heading_w_rad,
        motion.crouch_depth_m,
        motion.target_yaw_rate_radps,
    )
    if not all(math.isfinite(value) for value in numeric):
        return ValidationResult(False, False, "NON_FINITE_COMMAND")
    if motion.target_speed_mps < 0.0:
        return ValidationResult(False, False, "NEGATIVE_SPEED")
    if motion.posture not in {member.value for member in Posture}:
        return ValidationResult(False, False, "UNKNOWN_POSTURE")
    if request.kind is CommandKind.CROUCH and not 0.08 <= motion.crouch_depth_m <= 0.10:
        return ValidationResult(False, False, "CROUCH_DEPTH_OUTSIDE_REFERENCED_RANGE")
    if request.kind is not CommandKind.CROUCH and motion.crouch_depth_m != 0.0:
        return ValidationResult(False, False, "CROUCH_DEPTH_WITH_NON_CROUCH_COMMAND")
    # Stage 0 freezes the contract but deliberately authorizes no transition.
    return ValidationResult(True, False, "FORMAL_EVALUATION_PENDING")
