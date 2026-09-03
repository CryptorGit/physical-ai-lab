"""Frozen Stage 2 STAND↔WALK command controller; no action overlays."""

from __future__ import annotations

from enum import IntEnum

import torch


class Phase(IntEnum):
    INITIAL_STAND_SETTLE = 0
    STAND_HOLD = 1
    ACCELERATION_RAMP = 2
    WALK_ACQUISITION = 3
    WALK_HOLD = 4
    DECELERATION_RAMP = 5
    DOUBLE_SUPPORT_RECOVERY = 6
    FINAL_STAND_HOLD = 7
    COMPLETE = 8
    FAILED = 9


def minimum_jerk(progress: torch.Tensor) -> torch.Tensor:
    u = progress.clamp(0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def velocity_command(
    phase: torch.Tensor,
    phase_elapsed_s: torch.Tensor,
    target_speed_mps: torch.Tensor,
    ramp_duration_s: torch.Tensor,
    *,
    abrupt: bool = False,
) -> torch.Tensor:
    """Return body vx only. vy and yaw-rate are always zero."""
    result = torch.zeros_like(target_speed_mps)
    ramp = ramp_duration_s.clamp_min(1.0e-6)
    accelerating = phase == int(Phase.ACCELERATION_RAMP)
    walking = (phase == int(Phase.WALK_ACQUISITION)) | (phase == int(Phase.WALK_HOLD))
    decelerating = phase == int(Phase.DECELERATION_RAMP)
    if abrupt:
        result[accelerating | walking] = target_speed_mps[accelerating | walking]
    else:
        result[accelerating] = target_speed_mps[accelerating] * minimum_jerk(
            phase_elapsed_s[accelerating] / ramp[accelerating]
        )
        result[walking] = target_speed_mps[walking]
        result[decelerating] = target_speed_mps[decelerating] * (
            1.0 - minimum_jerk(phase_elapsed_s[decelerating] / ramp[decelerating])
        )
    return result


ROUTING_CONTRACT = {
    "active_expert": "stage2_model_4246_only",
    "run_contribution": "BITWISE_ZERO",
    "transition_bridge_contribution": "BITWISE_ZERO",
    "scripted_offset": "BITWISE_ZERO",
    "target_body_vy": 0.0,
    "target_yaw_rate": 0.0,
}
