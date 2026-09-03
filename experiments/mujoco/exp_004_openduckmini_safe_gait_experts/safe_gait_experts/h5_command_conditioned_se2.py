"""Pure H5 V3 command-conditioned SE(2) reward residuals.

The historical H4 forward/reverse drift terms assume zero lateral and yaw
command.  H5 V3 uses one continuous ``[vx, vy, wz]`` command, so those
assumptions contradict its forward-diagonal and reverse-turn anchors.  This
module exposes only the replacement residual definitions; it has no simulator,
policy, reward-scale, or hardware side effect.
"""

from __future__ import annotations

from typing import Any, NamedTuple


H5_V3_COMMAND_CONDITIONED_SE2_ALIGNMENT_ID = (
    "H5_V3_COMMAND_CONDITIONED_SE2_ALIGNMENT_20260812"
)


def _vector(value: Any, *, width: int, label: str, xp: Any) -> Any:
    array = xp.asarray(value)
    if array.shape != (width,):
        raise ValueError(f"{label} must have shape ({width},), got {array.shape}")
    return array


def wrapped_angle_difference(angle: Any, reference: Any, *, xp: Any) -> Any:
    """Return the signed shortest angular residual without a frame mutation."""

    difference = xp.asarray(angle) - xp.asarray(reference)
    return xp.arctan2(xp.sin(difference), xp.cos(difference))


class H5V3SE2Residuals(NamedTuple):
    """Unscaled residuals consumed by existing forward/reverse loss scales."""

    cross_velocity_mps: Any
    yaw_rate_radps: Any
    heading_error_rad: Any
    commanded_yaw_active: Any
    nonzero_lateral_command: Any


def h5_v3_command_conditioned_se2_residuals(
    local_velocity_xyz_mps: Any,
    yaw_rate_radps: Any,
    current_yaw_rad: Any,
    reset_heading_reference_yaw_rad: Any,
    integrated_command_heading_yaw_rad: Any,
    physical_command_vx_vy_wz: Any,
    *,
    xp: Any,
) -> H5V3SE2Residuals:
    """Return H5 V3 SE(2) residuals while preserving pure vx semantics.

    For zero ``vy`` and ``wz`` this returns the legacy H4 residual values
    exactly: local ``vy``, absolute yaw rate, and reset-heading drift.  A
    nonzero lateral command instead measures velocity orthogonal to the
    commanded translation vector; a nonzero yaw command tracks yaw rate and
    integrated heading.  This deliberate branch is what makes pure
    forward/reverse output a byte-for-byte regression target.
    """

    velocity = _vector(
        local_velocity_xyz_mps, width=3, label="local_velocity_xyz_mps", xp=xp
    )
    command = _vector(
        physical_command_vx_vy_wz,
        width=3,
        label="physical_command_vx_vy_wz",
        xp=xp,
    )
    dtype = velocity.dtype
    command_xy_norm = xp.sqrt(xp.sum(xp.square(command[:2])))
    safe_command_xy_norm = xp.maximum(
        command_xy_norm, xp.asarray(1.0e-12, dtype=dtype)
    )
    orthogonal_velocity = (
        -command[1] * velocity[0] + command[0] * velocity[1]
    ) / safe_command_xy_norm
    nonzero_lateral_command = xp.abs(command[1]) > xp.asarray(0.0, dtype=dtype)
    commanded_yaw_active = xp.abs(command[2]) > xp.asarray(0.0, dtype=dtype)
    return H5V3SE2Residuals(
        xp.where(nonzero_lateral_command, orthogonal_velocity, velocity[1]),
        xp.where(commanded_yaw_active, yaw_rate_radps - command[2], yaw_rate_radps),
        xp.where(
            commanded_yaw_active,
            wrapped_angle_difference(
                current_yaw_rad, integrated_command_heading_yaw_rad, xp=xp
            ),
            wrapped_angle_difference(
                current_yaw_rad, reset_heading_reference_yaw_rad, xp=xp
            ),
        ),
        commanded_yaw_active,
        nonzero_lateral_command,
    )


def advance_h5_v3_command_heading(
    previous_heading_yaw_rad: Any,
    command_yaw_rate_radps: Any,
    control_dt_s: float,
    *,
    xp: Any,
) -> Any:
    """Advance the desired world heading for exactly one control interval."""

    if not isinstance(control_dt_s, float) or control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be a positive float")
    return xp.arctan2(
        xp.sin(previous_heading_yaw_rad + command_yaw_rate_radps * control_dt_s),
        xp.cos(previous_heading_yaw_rad + command_yaw_rate_radps * control_dt_s),
    )
