"""Calibrated, symmetric CROUCH motion primitive for the frozen standing base."""

from __future__ import annotations

import torch

CROUCH_ACTION_INDICES = (0, 1, 11, 12, 15, 16)
# (measured stable pelvis drop [m], hip, knee, ankle normalized-action offset)
CALIBRATED_POSES = (
    (0.0, 0.0, 0.0, 0.0),
    (0.0752927383, -0.20, 0.60, -0.80),
    (0.1010949334, -0.25, 0.80, -1.00),
    # Deeper static poses held for 1.2 s but failed reproducible RETURN.
    # Production therefore stops at the deepest full-sequence-safe pose.
)


def minimum_jerk(progress: torch.Tensor) -> torch.Tensor:
    """C2-continuous scalar blend: 10t^3 - 15t^4 + 6t^5."""
    value = progress.clamp(0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def pose_for_depth(depth_m: torch.Tensor, action_dim: int = 37) -> torch.Tensor:
    """Piecewise-linear lookup from relative pelvis drop to symmetric offsets."""
    depth = depth_m.clamp(0.0, CALIBRATED_POSES[-1][0])
    triplet = torch.zeros((*depth.shape, 3), device=depth.device, dtype=depth.dtype)
    for lower, upper in zip(CALIBRATED_POSES[:-1], CALIBRATED_POSES[1:]):
        lo_depth, hi_depth = lower[0], upper[0]
        selected = (depth >= lo_depth) & (depth <= hi_depth)
        fraction = ((depth - lo_depth) / max(hi_depth - lo_depth, 1.0e-8)).unsqueeze(-1)
        lo = torch.tensor(lower[1:], device=depth.device, dtype=depth.dtype)
        hi = torch.tensor(upper[1:], device=depth.device, dtype=depth.dtype)
        interpolated = lo + fraction * (hi - lo)
        triplet = torch.where(selected.unsqueeze(-1), interpolated, triplet)
    output = torch.zeros((*depth.shape, action_dim), device=depth.device, dtype=depth.dtype)
    output[..., [0, 1]] = triplet[..., 0:1]
    output[..., [11, 12]] = triplet[..., 1:2]
    output[..., [15, 16]] = triplet[..., 2:3]
    return output


def phased_offset(depth_m: torch.Tensor, phase: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    """Return SETTLE/DOWN/HOLD/RETURN/STAND-HOLD primitive action offset."""
    blend = torch.zeros_like(depth_m)
    blend = torch.where(phase == 1, minimum_jerk(progress), blend)
    blend = torch.where(phase == 2, torch.ones_like(blend), blend)
    blend = torch.where(phase == 3, 1.0 - minimum_jerk(progress), blend)
    # Interpolate in measured depth space, then map through the calibrated
    # pose curve. Scaling the deep action vector directly follows an unsafe
    # straight line through joint space during RETURN.
    return pose_for_depth(depth_m * blend)
