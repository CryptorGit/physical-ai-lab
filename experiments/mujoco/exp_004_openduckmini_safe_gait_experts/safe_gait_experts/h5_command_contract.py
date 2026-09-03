"""Shared H5 command, curriculum, and phase contracts.

The H5 actor observes a command-space representation that is deliberately
different from the physical router command for a few calibrated axes. This
module is the single source of truth for that representation. It is free of
JAX imports so the training sampler and CPU evaluator share the same anchors.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np


# ``H5_COMMAND_CONTRACT_ID`` is the historical shared H5 command contract.
# Published V2 artifacts must stay byte-for-byte replayable, so a direct
# single-policy command map is assigned a new immutable contract below.
H5_COMMAND_CONTRACT_ID = "OPEN_DUCK_MINI_H5_COMMAND_ROUTING_V2"
H5_UNIFIED_COMMAND_CONTRACT_V2_ID = H5_COMMAND_CONTRACT_ID
H5_UNIFIED_COMMAND_CONTRACT_V3_ID = (
    "OPEN_DUCK_MINI_H5_UNIFIED_COMMAND_ROUTING_V3_DIRECT_NORMALIZED"
)
H5_UNIFIED_COMMAND_MAPPER_LEGACY_V2 = "legacy_h4_compensated"
H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3 = "direct_normalized_v3"
# ``direct_normalized`` was used by the first counterfactual.  Keep that CLI
# spelling replayable but canonicalize all new evidence to the V3 name.
H5_UNIFIED_COMMAND_MAPPER_ALIASES = MappingProxyType(
    {
        H5_UNIFIED_COMMAND_MAPPER_LEGACY_V2: H5_UNIFIED_COMMAND_MAPPER_LEGACY_V2,
        "legacy_h4_compensated_v2": H5_UNIFIED_COMMAND_MAPPER_LEGACY_V2,
        "direct_normalized": H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3,
        H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3: H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3,
    }
)
H5_UNIFIED_COMMAND_MAPPER_SUPPORTED_MODES = tuple(
    H5_UNIFIED_COMMAND_MAPPER_ALIASES
)
H5_REVERSE_PHASE_DELTA_BINS_PER_CONTROL = 0.81

H5_PLANAR_ROUTE_NAMES = (
    "stand",
    "forward",
    "lateral_left",
    "lateral_right",
    "yaw_left",
    "yaw_right",
    "forward_turn_left",
    "forward_turn_right",
    "forward_lateral_left_turn",
    "forward_lateral_right_turn",
)

H5_PLANAR_PHYSICAL_COMMANDS = MappingProxyType(
    {
        "stand": (0.0, 0.0, 0.0),
        "forward": (0.05, 0.0, 0.0),
        "lateral_left": (0.0, 0.06, 0.0),
        "lateral_right": (0.0, -0.06, 0.0),
        "yaw_left": (0.0, 0.0, 0.30),
        "yaw_right": (0.0, 0.0, -0.30),
        "forward_turn_left": (0.04, 0.0, 0.30),
        "forward_turn_right": (0.04, 0.0, -0.22),
        "forward_lateral_left_turn": (0.04, 0.05, 0.17),
        "forward_lateral_right_turn": (0.04, -0.03, -0.15),
    }
)

# Formal policy-observation anchors. The compensation values are intentional;
# they are copied from the frozen routed command cases and are not a global
# scale approximation.
H5_PLANAR_POLICY_COMMANDS = MappingProxyType(
    {
        "stand": (0.0, 0.0, 0.0),
        # The calibrated H4 forward actor was trained against this exact
        # policy-visible anchor.  Keeping vy/wz at zero here injects a
        # repeatable diagonal/yaw bias into an otherwise pure +vx command.
        "forward": (0.10, -0.018, -0.170),
        "lateral_left": (0.0, 0.10, 0.0),
        "lateral_right": (0.0, -0.10, 0.0),
        "yaw_left": (0.0, -0.06, 0.60),
        "yaw_right": (0.0, 0.0, -0.80),
        "forward_turn_left": (0.08, 0.0, 0.30),
        "forward_turn_right": (0.08, 0.0, -0.45),
        "forward_lateral_left_turn": (0.06, 0.05, 0.20),
        "forward_lateral_right_turn": (0.06, -0.05, -0.35),
    }
)

H5_REVERSE_ROUTE_NAMES = (
    "stand",
    "reverse",
    "reverse_anchor_minus_060",
    "reverse_anchor_minus_040",
    "reverse_turn_left",
    "reverse_turn_right",
)
H5_REVERSE_ROUTE_PROBABILITIES = MappingProxyType(
    {
        "stand": 0.10,
        "reverse": 0.60,
        "reverse_anchor_minus_060": 0.10,
        "reverse_anchor_minus_040": 0.10,
        "reverse_turn_left": 0.05,
        "reverse_turn_right": 0.05,
    }
)
H5_REVERSE_PHYSICAL_COMMANDS = MappingProxyType(
    {
        "stand": (0.0, 0.0, 0.0),
        "reverse": (-0.05, 0.0, 0.0),
        "reverse_anchor_minus_060": (-0.06, 0.0, 0.0),
        "reverse_anchor_minus_040": (-0.04, 0.0, 0.0),
        "reverse_turn_left": (-0.03, 0.0, 0.20),
        "reverse_turn_right": (-0.04, 0.0, -0.20),
    }
)
H5_REVERSE_POLICY_SCALE = (1.0, 5.0 / 3.0, 2.0)

# Unified-policy contract.  Unlike the historical two-domain H5 diagnostic,
# this command path is one continuous normalized [vx, vy, wz] representation;
# negative vx is not routed to a standing/forward anchor.  The scale is only
# an observation normalization and is recorded in every unified run manifest.
H5_UNIFIED_ROUTE_NAMES = (
    "stand",
    "forward",
    "reverse",
    "lateral_left",
    "lateral_right",
    "yaw_left",
    "yaw_right",
    "forward_turn_left",
    "forward_turn_right",
    "forward_lateral_left_turn",
    "forward_lateral_right_turn",
    "reverse_turn_left",
    "reverse_turn_right",
)
H5_UNIFIED_PHYSICAL_COMMANDS = MappingProxyType(
    {
        "stand": (0.0, 0.0, 0.0),
        "forward": (0.05, 0.0, 0.0),
        "reverse": (-0.05, 0.0, 0.0),
        "lateral_left": (0.0, 0.06, 0.0),
        "lateral_right": (0.0, -0.06, 0.0),
        "yaw_left": (0.0, 0.0, 0.30),
        "yaw_right": (0.0, 0.0, -0.30),
        "forward_turn_left": (0.04, 0.0, 0.30),
        "forward_turn_right": (0.04, 0.0, -0.22),
        "forward_lateral_left_turn": (0.04, 0.05, 0.17),
        "forward_lateral_right_turn": (0.04, -0.03, -0.15),
        "reverse_turn_left": (-0.04, 0.0, 0.20),
        "reverse_turn_right": (-0.04, 0.0, -0.20),
    }
)
# The direct scale remains the unified command normalization.  The two
# positive-vx coupling terms below are the measured legacy forward command
# compensation: at (+0.05, 0, 0) they produce (0.10, -0.018, -0.170), the
# established H4 observation anchor.  ReLU(vx) makes the extension continuous
# at the stand/reverse boundary and leaves reverse's signed vx path unchanged.
H5_UNIFIED_POLICY_SCALE = (2.0, 5.0 / 3.0, 2.0)
H5_UNIFIED_POSITIVE_VX_POLICY_COMPENSATION = (0.36, 3.40)


def canonical_h5_unified_command_mapper(mapper_mode: str) -> str:
    """Return the immutable unified mapper name for an input CLI spelling."""

    try:
        return H5_UNIFIED_COMMAND_MAPPER_ALIASES[str(mapper_mode)]
    except KeyError as error:
        raise ValueError("unsupported H5 unified command mapper") from error


def h5_unified_command_contract_id(mapper_mode: str) -> str:
    """Return the immutable command-contract ID for one unified mapper."""

    canonical = canonical_h5_unified_command_mapper(mapper_mode)
    return (
        H5_UNIFIED_COMMAND_CONTRACT_V3_ID
        if canonical == H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3
        else H5_UNIFIED_COMMAND_CONTRACT_V2_ID
    )


def _command3(command: Any) -> np.ndarray:
    values = np.asarray(command, dtype=np.float64)
    if values.shape == (7,):
        values = values[:3]
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("H5 command must be a finite 3- or 7-wide vector")
    return values


def h5_planar_route_name(command: Any) -> str:
    """Return the nearest formal planar route for a physical command."""

    values = _command3(command)
    anchors = np.asarray(
        [H5_PLANAR_PHYSICAL_COMMANDS[name] for name in H5_PLANAR_ROUTE_NAMES],
        dtype=np.float64,
    )
    axis_scale = np.asarray((0.05, 0.06, 0.30), dtype=np.float64)
    distances = np.sum(np.square((anchors - values) / axis_scale), axis=1)
    return H5_PLANAR_ROUTE_NAMES[int(np.argmin(distances))]


def h5_planar_policy_command(command: Any) -> np.ndarray:
    """Map a physical planar command to the H5 policy observation command."""

    values = _command3(command)
    route = h5_planar_route_name(values)
    physical_anchor = np.asarray(H5_PLANAR_PHYSICAL_COMMANDS[route])
    policy_anchor = np.asarray(H5_PLANAR_POLICY_COMMANDS[route])
    denominator = float(np.dot(physical_anchor, physical_anchor))
    if denominator == 0.0:
        return np.zeros(3, dtype=np.float64)
    ratio = float(np.dot(values, physical_anchor) / denominator)
    return policy_anchor * ratio


def h5_reverse_policy_command(command: Any) -> np.ndarray:
    """Map a physical reverse-domain command with the H5 reverse scale."""

    values = _command3(command)
    return values * np.asarray(H5_REVERSE_POLICY_SCALE, dtype=np.float64)


def h5_unified_policy_command(command: Any) -> np.ndarray:
    """Map any physical planar command into the single-policy command space."""

    values = _command3(command)
    mapped = values * np.asarray(H5_UNIFIED_POLICY_SCALE, dtype=np.float64)
    positive_vx = max(float(values[0]), 0.0)
    lateral_compensation, yaw_compensation = (
        H5_UNIFIED_POSITIVE_VX_POLICY_COMPENSATION
    )
    mapped[1] -= lateral_compensation * positive_vx
    mapped[2] -= yaw_compensation * positive_vx
    return mapped


def h5_unified_direct_policy_command(command: Any) -> np.ndarray:
    """Map physical [vx, vy, wz] with only the frozen normalization scale.

    This is an evaluation/training-contract candidate, not an implicit change
    to :func:`h5_unified_policy_command`.  Keeping it separate makes every
    no-training ablation explicit about whether legacy positive-vx coupling
    was present in the actor observation.
    """

    values = _command3(command)
    return values * np.asarray(H5_UNIFIED_POLICY_SCALE, dtype=np.float64)


def h5_unified_command_contract_manifest(mapper_mode: str) -> dict[str, Any]:
    """Describe one immutable unified-policy command-observation contract."""

    canonical = canonical_h5_unified_command_mapper(mapper_mode)
    direct = canonical == H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3
    return {
        "contract_id": h5_unified_command_contract_id(canonical),
        "mapper": canonical,
        "physical_command_representation": "[vx, vy, wz]",
        "policy_command_representation": "[2*vx, (5/3)*vy, 2*wz]",
        "policy_scale": list(H5_UNIFIED_POLICY_SCALE),
        "axis_separable": direct,
        "positive_vx_cross_axis_compensation": (
            [0.0, 0.0]
            if direct
            else list(H5_UNIFIED_POSITIVE_VX_POLICY_COMPENSATION)
        ),
        "hardware_deployment": "PROHIBITED",
    }


def h5_planar_policy_command_xp(command: Any, *, xp: Any) -> Any:
    """JAX/NumPy-compatible form of :func:`h5_planar_policy_command`."""

    values = command[:3]
    physical = xp.asarray(
        [H5_PLANAR_PHYSICAL_COMMANDS[name] for name in H5_PLANAR_ROUTE_NAMES],
        dtype=values.dtype,
    )
    policy = xp.asarray(
        [H5_PLANAR_POLICY_COMMANDS[name] for name in H5_PLANAR_ROUTE_NAMES],
        dtype=values.dtype,
    )
    axis_scale = xp.asarray((0.05, 0.06, 0.30), dtype=values.dtype)
    distances = xp.sum(xp.square((physical - values[:3]) / axis_scale), axis=1)
    route = xp.argmin(distances)
    physical_anchor = physical[route]
    policy_anchor = policy[route]
    denominator = xp.dot(physical_anchor, physical_anchor)
    ratio = xp.where(
        denominator > 0.0,
        xp.dot(values[:3], physical_anchor) / denominator,
        xp.asarray(0.0, dtype=values.dtype),
    )
    return xp.concatenate(
        (policy_anchor * ratio, xp.zeros(4, dtype=values.dtype))
    )


def h5_reverse_policy_command_xp(command: Any, *, xp: Any) -> Any:
    values = command[:3]
    scale = xp.asarray(H5_REVERSE_POLICY_SCALE, dtype=values.dtype)
    return xp.concatenate((values * scale, xp.zeros(4, dtype=values.dtype)))


def h5_unified_policy_command_xp(command: Any, *, xp: Any) -> Any:
    values = command[:3]
    scale = xp.asarray(H5_UNIFIED_POLICY_SCALE, dtype=values.dtype)
    mapped = values * scale
    positive_vx = xp.maximum(values[0], xp.asarray(0.0, dtype=values.dtype))
    compensation = xp.asarray(
        H5_UNIFIED_POSITIVE_VX_POLICY_COMPENSATION, dtype=values.dtype
    )
    mapped = xp.stack(
        (
            mapped[0],
            mapped[1] - compensation[0] * positive_vx,
            mapped[2] - compensation[1] * positive_vx,
        ),
        axis=0,
    )
    return xp.concatenate((mapped, xp.zeros(4, dtype=values.dtype)))


def h5_unified_direct_policy_command_xp(command: Any, *, xp: Any) -> Any:
    """JAX/NumPy-compatible direct-normalized unified command mapping."""

    values = command[:3]
    scale = xp.asarray(H5_UNIFIED_POLICY_SCALE, dtype=values.dtype)
    return xp.concatenate((values * scale, xp.zeros(4, dtype=values.dtype)))


def h5_command_contract_manifest() -> dict[str, Any]:
    """Return a JSON-safe manifest used by training/evaluation evidence."""

    return {
        "contract_id": H5_COMMAND_CONTRACT_ID,
        "planar_physical_commands": {
            name: list(H5_PLANAR_PHYSICAL_COMMANDS[name])
            for name in H5_PLANAR_ROUTE_NAMES
        },
        "planar_policy_commands": {
            name: list(H5_PLANAR_POLICY_COMMANDS[name])
            for name in H5_PLANAR_ROUTE_NAMES
        },
        "reverse_route_probabilities": dict(H5_REVERSE_ROUTE_PROBABILITIES),
        "reverse_physical_commands": {
            name: list(H5_REVERSE_PHYSICAL_COMMANDS[name])
            for name in H5_REVERSE_ROUTE_NAMES
        },
        "reverse_phase_delta_bins_per_control": H5_REVERSE_PHASE_DELTA_BINS_PER_CONTROL,
        "unified_route_names": list(H5_UNIFIED_ROUTE_NAMES),
        "unified_physical_commands": {
            name: list(H5_UNIFIED_PHYSICAL_COMMANDS[name])
            for name in H5_UNIFIED_ROUTE_NAMES
        },
        "unified_policy_scale": list(H5_UNIFIED_POLICY_SCALE),
        "unified_positive_vx_policy_compensation": list(
            H5_UNIFIED_POSITIVE_VX_POLICY_COMPENSATION
        ),
        "hardware_deployment": "PROHIBITED",
    }
