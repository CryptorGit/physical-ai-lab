"""Final OpenDuckMini actuator-target safety guard for exp_004 packages.

This function consumes *joint targets*, after policy/profile composition and
all other runtime transforms.  It clamps every leg target inside the packaged
SAFE limits with a frozen 0.015 rad inward margin and forces all head targets
to exact zero.  Hardware deployment remains prohibited; this is a simulation
runtime invariant, not hardware approval.
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
RUNTIME_TARGET_SAFETY_MARGIN_RAD: Final = 0.015


def apply_final_target_safety(
    joint_targets: Sequence[float],
    safe_joint_limits_rad: Mapping[str, Sequence[float]],
    *,
    margin_rad: float = RUNTIME_TARGET_SAFETY_MARGIN_RAD,
) -> np.ndarray:
    """Return guarded 14-channel targets without mutating the caller's input.

    ``safe_joint_limits_rad`` must contain exactly the ten leg joints from the
    frozen contract.  The guard is intended to be the last target mutation
    before actuator application.
    """

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
        margin = float(margin_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("margin_rad must remain exactly 0.015 for exp_004") from exc
    if not np.isfinite(margin) or margin != RUNTIME_TARGET_SAFETY_MARGIN_RAD:
        raise ValueError("margin_rad must remain exactly 0.015 for exp_004")

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


__all__ = [
    "ACTUATOR_JOINT_ORDER",
    "HEAD_ACTION_INDICES",
    "LEG_ACTION_INDICES",
    "LEG_JOINT_NAMES",
    "RUNTIME_TARGET_SAFETY_MARGIN_RAD",
    "apply_final_target_safety",
]
