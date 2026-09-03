"""Pure objective functions for the v60 bounded-yaw causal pilot.

This module has no training side effects.  The functions mirror the calibrated
Joystick reward implementation and are shared by the offline contract tests and
the independent v60 training harness.
"""

from __future__ import annotations

from typing import Any


COMMAND_PROGRESS_SCALE = 100.0
COMMAND_YAW_ERROR_SCALE = -20.0
TRACKING_ANG_VEL_SCALE = 10.0
TRACKING_ANG_VEL_VARIANCE = 0.04
SIGMA_YAW = TRACKING_ANG_VEL_VARIANCE**0.5


def old_yaw_progress(
    yaw_command: Any, yaw_actual: Any, *, xp: Any
) -> Any:
    """Unscaled historical command_progress yaw contribution."""

    return yaw_command * yaw_actual


def bounded_yaw_progress(
    yaw_command: Any, yaw_actual: Any, *, xp: Any
) -> Any:
    """Unscaled command-centred Gaussian with exact-tracking amplitude."""

    yaw_error = yaw_actual - yaw_command
    amplitude = yaw_command**2
    return amplitude * xp.exp(-((yaw_error / SIGMA_YAW) ** 2))


def tracking_ang_vel_reward(
    yaw_command: Any, yaw_actual: Any, *, xp: Any
) -> Any:
    error = yaw_actual - yaw_command
    return TRACKING_ANG_VEL_SCALE * xp.exp(
        -(error**2) / TRACKING_ANG_VEL_VARIANCE
    )


def command_yaw_error_reward(
    yaw_command: Any, yaw_actual: Any, *, xp: Any
) -> Any:
    error = yaw_actual - yaw_command
    return COMMAND_YAW_ERROR_SCALE * error**2


def yaw_related_total(
    yaw_command: Any,
    yaw_actual: Any,
    *,
    objective: str,
    xp: Any,
) -> Any:
    if objective == "old_unbounded_dot":
        progress = old_yaw_progress(yaw_command, yaw_actual, xp=xp)
    elif objective == "bounded_command_centered_gaussian":
        progress = bounded_yaw_progress(yaw_command, yaw_actual, xp=xp)
    else:
        raise ValueError(f"Unknown yaw objective: {objective}")
    return (
        COMMAND_PROGRESS_SCALE * progress
        + tracking_ang_vel_reward(yaw_command, yaw_actual, xp=xp)
        + command_yaw_error_reward(yaw_command, yaw_actual, xp=xp)
    )
