"""Bounded, command-centred tracking objectives for gait experts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


DEFAULT_SIGMA = (0.05, 0.05, 0.20)
DEFAULT_WEIGHT = (1.0, 1.0, 1.0)


def bounded_symmetric_tracking(
    command: Any,
    actual: Any,
    *,
    sigma: Sequence[float] = DEFAULT_SIGMA,
    weight: Sequence[float] = DEFAULT_WEIGHT,
    xp: Any = np,
) -> Any:
    """Return a bounded vx/vy/yaw tracking reward with the correct argmax.

    The three components are weighted Gaussians centred on the command.  For
    strictly positive ``sigma`` and ``weight``, the result is in ``(0, 1]``
    and has its unique global maximum at ``actual == command``.  Passing
    ``xp=jax.numpy`` keeps the function traceable inside an MJX environment.
    """

    if len(sigma) != 3 or len(weight) != 3:
        raise ValueError("sigma and weight must each contain vx, vy, and yaw")
    if any(float(value) <= 0.0 for value in sigma):
        raise ValueError("all sigma values must be positive")
    if any(float(value) <= 0.0 for value in weight):
        raise ValueError("all weight values must be positive")

    command_array = xp.asarray(command)
    actual_array = xp.asarray(actual)
    if command_array.shape[-1:] != (3,) or actual_array.shape[-1:] != (3,):
        raise ValueError("command and actual must end with [vx, vy, yaw_rate]")

    # Do not let an integer command coerce sub-unit sigma values to zero.
    dtype = xp.result_type(command_array.dtype, actual_array.dtype, xp.float32)
    command_array = command_array.astype(dtype)
    actual_array = actual_array.astype(dtype)

    sigma_array = xp.asarray(sigma, dtype=command_array.dtype)
    weight_array = xp.asarray(weight, dtype=command_array.dtype)
    normalized_error = (actual_array - command_array) / sigma_array
    axis_reward = xp.exp(-(normalized_error**2))
    return xp.sum(axis_reward * weight_array, axis=-1) / xp.sum(weight_array)


def bounded_axis_tracking(
    command: Any,
    actual: Any,
    *,
    sigma: Sequence[float] = DEFAULT_SIGMA,
    xp: Any = np,
) -> Any:
    """Return the three unaggregated bounded tracking terms."""

    if len(sigma) != 3 or any(float(value) <= 0.0 for value in sigma):
        raise ValueError("sigma must contain three positive values")
    command_array = xp.asarray(command)
    actual_array = xp.asarray(actual)
    if command_array.shape[-1:] != (3,) or actual_array.shape[-1:] != (3,):
        raise ValueError("command and actual must end with [vx, vy, yaw_rate]")
    dtype = xp.result_type(command_array.dtype, actual_array.dtype, xp.float32)
    command_array = command_array.astype(dtype)
    actual_array = actual_array.astype(dtype)
    sigma_array = xp.asarray(sigma, dtype=command_array.dtype)
    return xp.exp(-(((actual_array - command_array) / sigma_array) ** 2))
