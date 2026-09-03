"""Audit-only LAND_SHALLOW phase controller and symmetric landing offsets."""

from __future__ import annotations

from enum import IntEnum

import torch

from .scripted_crouch import pose_for_depth, minimum_jerk


class LandPhase(IntEnum):
    PREPARE = 0
    AIRBORNE = 1
    FIRST_CONTACT = 2
    IMPACT_ABSORPTION = 3
    DOUBLE_SUPPORT_RECOVERY = 4
    RETURN_TO_STAND = 5
    STAND_HOLD = 6


LAND_FAILURE_CLASSES = (
    "invalid_initial_clearance", "asymmetric_initial_pose", "premature_contact",
    "excessive_contact_asymmetry", "impact_failure", "absorption_failure",
    "double_support_failure", "excessive_rebound", "unstable_recovery",
    "stand_hold_failure", "saturation_failure", "joint_limit_failure",
    "foot_slip_failure", "fall", "unsupported_drop_height",
)


def landing_offset(
    phase: torch.Tensor,
    phase_progress: torch.Tensor,
    preflex_depth_m: torch.Tensor,
    absorption_depth_m: torch.Tensor,
    action_dim: int = 37,
) -> torch.Tensor:
    """Return a symmetric sagittal offset without sharing CROUCH state."""
    blend = minimum_jerk(phase_progress)
    depth = torch.zeros_like(preflex_depth_m)
    depth = torch.where(phase == int(LandPhase.PREPARE), preflex_depth_m * blend, depth)
    depth = torch.where(phase == int(LandPhase.AIRBORNE), preflex_depth_m, depth)
    depth = torch.where(phase == int(LandPhase.FIRST_CONTACT), preflex_depth_m, depth)
    depth = torch.where(
        phase == int(LandPhase.IMPACT_ABSORPTION),
        preflex_depth_m + absorption_depth_m * blend,
        depth,
    )
    depth = torch.where(
        phase == int(LandPhase.DOUBLE_SUPPORT_RECOVERY),
        preflex_depth_m + absorption_depth_m,
        depth,
    )
    depth = torch.where(
        phase == int(LandPhase.RETURN_TO_STAND),
        (preflex_depth_m + absorption_depth_m) * (1.0 - blend),
        depth,
    )
    return pose_for_depth(depth, action_dim=action_dim)
