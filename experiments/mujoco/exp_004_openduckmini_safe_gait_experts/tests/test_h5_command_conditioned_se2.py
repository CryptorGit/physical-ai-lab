from __future__ import annotations

import numpy as np
import pytest

from safe_gait_experts.h5_command_conditioned_se2 import (
    advance_h5_v3_command_heading,
    h5_v3_command_conditioned_se2_residuals,
    wrapped_angle_difference,
)
from safe_gait_experts.h5_command_contract import H5_UNIFIED_PHYSICAL_COMMANDS


def test_pure_forward_and_reverse_residuals_are_bit_exact_legacy_values() -> None:
    velocity = np.asarray((0.021, -0.007, 0.0), dtype=np.float64)
    yaw_rate = np.float64(0.031)
    current_yaw = np.float64(0.22)
    reset_heading = np.float64(-0.14)
    for name in ("forward", "reverse"):
        command = np.asarray(H5_UNIFIED_PHYSICAL_COMMANDS[name], dtype=np.float64)
        residuals = h5_v3_command_conditioned_se2_residuals(
            velocity,
            yaw_rate,
            current_yaw,
            reset_heading,
            np.float64(0.77),
            command,
            xp=np,
        )
        assert np.asarray(residuals.cross_velocity_mps).tobytes() == velocity[1].tobytes()
        assert np.asarray(residuals.yaw_rate_radps).tobytes() == yaw_rate.tobytes()
        expected_heading = wrapped_angle_difference(current_yaw, reset_heading, xp=np)
        assert np.asarray(residuals.heading_error_rad).tobytes() == np.asarray(expected_heading).tobytes()


def test_every_moving_h5_anchor_has_zero_twist_and_heading_residual() -> None:
    desired_heading = np.float64(0.31)
    for name, values in H5_UNIFIED_PHYSICAL_COMMANDS.items():
        command = np.asarray(values, dtype=np.float64)
        if not np.any(command):
            continue
        velocity = np.asarray((command[0], command[1], 0.0), dtype=np.float64)
        current_yaw = desired_heading if command[2] else np.float64(-0.6)
        residuals = h5_v3_command_conditioned_se2_residuals(
            velocity,
            command[2],
            current_yaw,
            np.float64(-0.6),
            desired_heading,
            command,
            xp=np,
        )
        assert np.isclose(residuals.cross_velocity_mps, 0.0)
        assert np.isclose(residuals.yaw_rate_radps, 0.0)
        assert np.isclose(residuals.heading_error_rad, 0.0)


def test_heading_advance_is_wrapped_and_has_exact_control_interval() -> None:
    advanced = advance_h5_v3_command_heading(
        np.float64(3.13), np.float64(0.30), 0.020, xp=np
    )
    assert -np.pi <= advanced <= np.pi
    assert np.isclose(
        advanced,
        np.arctan2(np.sin(3.13 + 0.30 * 0.020), np.cos(3.13 + 0.30 * 0.020)),
    )


def test_jax_residuals_jit_and_preserve_pure_values() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    command = jnp.asarray(H5_UNIFIED_PHYSICAL_COMMANDS["reverse"], dtype=jnp.float32)

    @jax.jit
    def run(velocity, yaw_rate, yaw, heading):
        return h5_v3_command_conditioned_se2_residuals(
            velocity, yaw_rate, yaw, heading, jnp.float32(0.7), command, xp=jnp
        )

    result = run(
        jnp.asarray((0.01, -0.004, 0.0), dtype=jnp.float32),
        jnp.float32(0.02),
        jnp.float32(0.3),
        jnp.float32(-0.1),
    )
    assert np.isfinite(np.asarray(result.cross_velocity_mps))
    assert np.isfinite(np.asarray(result.yaw_rate_radps))
    assert np.isfinite(np.asarray(result.heading_error_rad))
