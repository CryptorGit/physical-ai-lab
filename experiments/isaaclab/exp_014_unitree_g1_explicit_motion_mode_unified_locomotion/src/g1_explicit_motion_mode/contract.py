"""Frozen Exp014ExplicitMotionModeCommandV1 contract.

The adapter appends exactly 17 causal features to the canonical EXP 013 124D
observation.  A mode request takes effect before the velocity ramp advances.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class MotionMode(IntEnum):
    STAND = 0
    WALK = 1
    RUN = 2


def mode_one_hot(mode: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(mode.long(), num_classes=3).to(torch.float32)


def legacy_gait(mode: torch.Tensor) -> torch.Tensor:
    """STAND/WALK map to zero; RUN maps to one."""
    return (mode.long() == int(MotionMode.RUN)).to(torch.float32)


def minimum_jerk(progress: torch.Tensor) -> torch.Tensor:
    x = progress.clamp(0.0, 1.0)
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


@dataclass
class ExplicitMotionModeCommand:
    """Vectorized causal command-history state for one control stream."""

    physical_command: torch.Tensor
    target_mode: torch.Tensor
    previous_target_mode: torch.Tensor
    previous_physical_command: torch.Tensor
    time_since_mode_change_s: torch.Tensor
    ramp_progress: torch.Tensor

    @classmethod
    def zeros(cls, count: int, *, device: torch.device | str = "cpu") -> "ExplicitMotionModeCommand":
        z3 = torch.zeros(count, 3, device=device)
        stand = torch.full((count,), int(MotionMode.STAND), dtype=torch.long, device=device)
        return cls(z3, stand, stand.clone(), z3.clone(), torch.zeros(count, 1, device=device), torch.ones(count, 1, device=device))

    def request(self, mode: torch.Tensor) -> None:
        """Apply requested mode immediately, before any command ramp update."""
        mode = mode.long().to(self.target_mode.device)
        changed = mode != self.target_mode
        self.previous_target_mode = self.target_mode.clone()
        self.target_mode = mode.clone()
        self.time_since_mode_change_s[changed] = 0.0

    def advance(self, physical_command: torch.Tensor, ramp_progress: torch.Tensor, control_dt: float) -> None:
        self.previous_physical_command = self.physical_command.clone()
        self.physical_command = physical_command.clone()
        self.ramp_progress = ramp_progress.reshape(-1, 1).clamp(0.0, 1.0).clone()
        self.time_since_mode_change_s.add_(float(control_dt)).clamp_(max=3.0)

    def appended_features(self) -> torch.Tensor:
        return torch.cat(
            (
                self.physical_command,
                mode_one_hot(self.target_mode).to(self.physical_command),
                mode_one_hot(self.previous_target_mode).to(self.physical_command),
                self.previous_physical_command,
                self.physical_command - self.previous_physical_command,
                self.time_since_mode_change_s / 3.0,
                self.ramp_progress,
            ),
            dim=-1,
        )


def build_observation_141(observation_123: torch.Tensor, state: ExplicitMotionModeCommand) -> torch.Tensor:
    """Build [old 123D, legacy gait, appended 17D] without future information."""
    if observation_123.shape[-1] != 123:
        raise ValueError(f"expected 123D base observation, got {observation_123.shape[-1]}")
    old124 = torch.cat((observation_123, legacy_gait(state.target_mode).reshape(-1, 1).to(observation_123)), dim=-1)
    result = torch.cat((old124, state.appended_features().to(observation_123)), dim=-1)
    if result.shape[-1] != 141:
        raise RuntimeError("COMMAND_CONTRACT_BUG")
    if not torch.isfinite(result).all():
        raise RuntimeError("NaN/Inf")
    return result
