"""Stage 2B world-heading hold controller.

The target heading is controller state only. It is never appended to either
expert observation and this module does not implement a TURN command.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FixedHeadingConfig:
    k_heading: float
    k_yaw_rate: float
    yaw_rate_limit_radps: float

    def validate(self) -> None:
        if self.k_heading < 0.0 or self.k_yaw_rate < 0.0:
            raise ValueError("heading gains must be non-negative")
        if not 0.0 < self.yaw_rate_limit_radps <= 1.0:
            raise ValueError("yaw-rate limit must be in (0, 1] rad/s")


def wrap_angle(angle_rad: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))


def fixed_heading_yaw_rate(
    target_heading_w_rad: torch.Tensor,
    current_heading_w_rad: torch.Tensor,
    current_yaw_rate_radps: torch.Tensor,
    config: FixedHeadingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    config.validate()
    error = wrap_angle(target_heading_w_rad - current_heading_w_rad)
    command = config.k_heading * error - config.k_yaw_rate * current_yaw_rate_radps
    return command.clamp(-config.yaw_rate_limit_radps, config.yaw_rate_limit_radps), error
