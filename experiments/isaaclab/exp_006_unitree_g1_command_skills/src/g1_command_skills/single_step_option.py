"""Audit-only phase contract for a Stage-2 actor single-step option.

The production STEP_OVER route remains fail-closed.  This module contains no
learned parameters and does not alter either frozen actor; it only defines the
contact-gated option that may be promoted after the audit gates pass.
"""

from __future__ import annotations

from enum import IntEnum


class SingleStepPhase(IntEnum):
    SETTLE = 0
    WAIT_FOR_SUPPORT_PHASE = 1
    COMMAND_STEP = 2
    SWING = 3
    TOUCHDOWN = 4
    DOUBLE_SUPPORT_RECOVERY = 5
    ZERO_COMMAND_STAND = 6
    STAND_HOLD = 7


def minimum_jerk(progress: float) -> float:
    """Return a C2-continuous 0..1 command ramp."""
    value = min(max(float(progress), 0.0), 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


SINGLE_STEP_FAILURE_CLASSES = (
    "single_step_unavailable",
    "desired_lead_phase_unavailable",
    "step_liftoff_failure",
    "step_clearance_failure",
    "step_placement_failure",
    "touchdown_failure",
    "double_support_recovery_failure",
    "zero_command_recovery_failure",
    "obstacle_distance_unreachable",
    "lead_step_collision",
    "trail_step_collision",
)

