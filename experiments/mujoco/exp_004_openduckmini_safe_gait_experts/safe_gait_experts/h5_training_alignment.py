"""H5 planar-domain training sampler and command mapping.

This module is separate from the byte-pinned H4 alignment module.  It supplies
the command-conditioned curriculum needed by the H5 planar expert without
changing any historical H4 authorization closure.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .h4_training_alignment import _require_vector_shape
from .h5_command_contract import (
    H5_PLANAR_ROUTE_NAMES,
    H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3,
    H5_UNIFIED_PHYSICAL_COMMANDS,
    H5_UNIFIED_ROUTE_NAMES,
    H5_REVERSE_ROUTE_NAMES,
    H5_REVERSE_ROUTE_PROBABILITIES,
    canonical_h5_unified_command_mapper,
    h5_planar_policy_command_xp,
    h5_reverse_policy_command_xp,
    h5_unified_direct_policy_command_xp,
    h5_unified_policy_command_xp,
)


H5_PLANAR_SAMPLER_ROUTE_COUNT = len(H5_PLANAR_ROUTE_NAMES)
H5_PLANAR_POLICY_COMMAND_SCALE = (2.0, 5.0 / 3.0, 2.0)


def make_h5_planar_command_mapper(jax: Any, xp: Any) -> Callable[[Any], Any]:
    """Map physical planar commands using the shared route-aware contract."""

    del jax  # kept in the signature to mirror the JAX environment factory API

    def mapper(command: Any) -> Any:
        values = _require_vector_shape(command, 7, "physical command", xp=xp)
        return h5_planar_policy_command_xp(values, xp=xp)

    return mapper


def make_h5_planar_physical_sampler(jax: Any, xp: Any) -> Callable[[Any], Any]:
    """Sample stand, six primitive axes, and three forward compounds."""

    def sampler(rng: Any) -> Any:
        route_key, jitter_key = jax.random.split(rng)
        route = jax.random.randint(
            route_key, shape=(), minval=0, maxval=H5_PLANAR_SAMPLER_ROUTE_COUNT
        )
        jitter = jax.random.uniform(jitter_key, shape=(), minval=0.94, maxval=1.06)
        zeros = xp.asarray(0.0, dtype=jitter.dtype)
        vx = xp.asarray(
            xp.where(
                route == 1,
                0.05,
                xp.where(
                    (route >= 6) & (route <= 9),
                    0.04,
                    zeros,
                ),
            ),
            dtype=jitter.dtype,
        )
        vy = xp.where(
            route == 2,
            0.06,
            xp.where(
                route == 3,
                -0.06,
                xp.where(route == 8, 0.05, xp.where(route == 9, -0.03, zeros)),
            ),
        )
        yaw = xp.where(
            route == 4,
            0.30,
            xp.where(
                route == 5,
                -0.30,
                xp.where(
                    route == 6,
                    0.30,
                    xp.where(
                        route == 7,
                        -0.22,
                        xp.where(route == 8, 0.17, xp.where(route == 9, -0.15, zeros)),
                    ),
                ),
            ),
        )
        # Do not jitter the exact stand route.  Small bounded variation around
        # moving commands improves command generalization without introducing
        # a second untracked safety path.
        moving = route != 0
        return xp.asarray(
            (
                xp.where(moving, vx * jitter, vx),
                xp.where(moving, vy * jitter, vy),
                xp.where(moving, yaw * jitter, yaw),
                zeros,
                zeros,
                zeros,
                zeros,
            )
        )

    return sampler


def make_h5_unified_physical_sampler(
    jax: Any,
    xp: Any,
    *,
    reverse_route_probability: float | None = None,
) -> Callable[[Any], Any]:
    """Sample one continuous command-conditioned curriculum across all routes.

    Route anchors provide balanced coverage of the difficult sign changes;
    independent bounded axis jitter makes every moving sample continuous rather
    than an 11/13-point lookup table.  The returned seven-wide vector preserves
    the legacy command container used by the MJX task.
    """

    anchors = xp.asarray(
        [H5_UNIFIED_PHYSICAL_COMMANDS[name] for name in H5_UNIFIED_ROUTE_NAMES]
    )
    route_count = len(H5_UNIFIED_ROUTE_NAMES)
    reverse_route_indices = {
        index
        for index, name in enumerate(H5_UNIFIED_ROUTE_NAMES)
        if name in {"reverse", "reverse_turn_left", "reverse_turn_right"}
    }
    if reverse_route_probability is None:
        route_probabilities = np.full(
            route_count, 1.0 / float(route_count), dtype=np.float32
        )
        resolved_reverse_probability = len(reverse_route_indices) / float(route_count)
    else:
        resolved_reverse_probability = float(reverse_route_probability)
        if not np.isfinite(resolved_reverse_probability) or not (
            0.0 < resolved_reverse_probability < 1.0
        ):
            raise ValueError(
                "reverse_route_probability must be finite and strictly between 0 and 1"
            )
        reverse_count = len(reverse_route_indices)
        non_reverse_count = route_count - reverse_count
        route_probabilities = np.asarray(
            [
                resolved_reverse_probability / reverse_count
                if index in reverse_route_indices
                else (1.0 - resolved_reverse_probability) / non_reverse_count
                for index in range(route_count)
            ],
            dtype=np.float32,
        )
    if not np.isclose(float(np.sum(route_probabilities)), 1.0, atol=0.0, rtol=0.0):
        raise RuntimeError("unified H5 route probabilities must sum exactly to one")
    cumulative_probabilities = xp.asarray(
        np.cumsum(route_probabilities), dtype=anchors.dtype
    )

    def sampler(rng: Any) -> Any:
        route_key, jitter_key = jax.random.split(rng)
        draw = jax.random.uniform(
            route_key, shape=(), minval=0.0, maxval=1.0, dtype=anchors.dtype
        )
        route = xp.sum((draw >= cumulative_probabilities).astype(xp.int32))
        route = xp.minimum(route, xp.asarray(route_count - 1, dtype=xp.int32))
        anchor = anchors[route]
        axis_jitter = jax.random.uniform(
            jitter_key, shape=(3,), minval=0.92, maxval=1.08
        )
        moving = xp.any(xp.abs(anchor) > xp.asarray(0.0, dtype=anchor.dtype))
        command3 = xp.where(moving, anchor * axis_jitter, xp.zeros_like(anchor))
        return xp.concatenate((command3, xp.zeros(4, dtype=command3.dtype)))

    return sampler


def make_h5_unified_command_mapper(
    jax: Any,
    xp: Any,
    *,
    mapper_mode: str = "legacy_h4_compensated",
) -> Callable[[Any], Any]:
    """Map the complete physical [vx, vy, wz] command for one actor.

    The default preserves the existing H5 checkpoint contract.  The direct
    mode is deliberately explicit so a retraining candidate cannot silently
    share a manifest with a legacy-coupled actor.
    """

    del jax
    canonical_mapper_mode = canonical_h5_unified_command_mapper(mapper_mode)

    def mapper(command: Any) -> Any:
        values = _require_vector_shape(command, 7, "physical command", xp=xp)
        return (
            h5_unified_direct_policy_command_xp(values, xp=xp)
            if canonical_mapper_mode == H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3
            else h5_unified_policy_command_xp(values, xp=xp)
        )

    return mapper


def make_h5_reverse_command_mapper(jax: Any, xp: Any) -> Callable[[Any], Any]:
    """Map reverse and reverse-turn commands without collapsing yaw to zero."""

    del jax

    def mapper(command: Any) -> Any:
        values = _require_vector_shape(command, 7, "physical command", xp=xp)
        return h5_reverse_policy_command_xp(values, xp=xp)

    return mapper


def make_h5_reverse_physical_sampler(jax: Any, xp: Any) -> Callable[[Any], Any]:
    """Sample reverse anchors with an exact -0.05 endpoint majority."""

    probabilities = xp.asarray(
        [H5_REVERSE_ROUTE_PROBABILITIES[name] for name in H5_REVERSE_ROUTE_NAMES]
    )
    cumulative = xp.cumsum(probabilities)

    def sampler(rng: Any) -> Any:
        route_key = jax.random.split(rng, 1)[0]
        draw = jax.random.uniform(route_key, shape=(), minval=0.0, maxval=1.0)
        route = xp.sum((draw >= cumulative).astype(xp.int32))
        zeros = xp.asarray(0.0, dtype=draw.dtype)
        vx = xp.where(
            route == 1,
            -0.05,
            xp.where(
                route == 2,
                -0.06,
                xp.where(
                    (route == 3) | (route == 5),
                    -0.04,
                    xp.where(route == 4, -0.03, zeros),
                ),
            ),
        )
        yaw = xp.where(route == 4, 0.20, xp.where(route == 5, -0.20, zeros))
        return xp.asarray(
            (
                vx,
                zeros,
                yaw,
                zeros,
                zeros,
                zeros,
                zeros,
            )
        )

    return sampler
