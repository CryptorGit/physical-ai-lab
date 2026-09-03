"""Frozen WALK base plus a continuous speed-gated bounded residual."""

from __future__ import annotations

import torch
from torch import nn


class ContinuousSpeedResidual123(nn.Sequential):
    observation_dim = 123
    action_dim = 37

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(123, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 37),
        )
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class FrozenWalkSpeedResidualController123(nn.Module):
    observation_dim = 123
    action_dim = 37
    action_scale = 0.5

    def __init__(self, walk_base: nn.Module, residual_bounds: torch.Tensor) -> None:
        super().__init__()
        if residual_bounds.shape != (37,) or not torch.isfinite(residual_bounds).all():
            raise ValueError("residual bounds must be finite 37D")
        self.walk_base = walk_base
        self.residual = ContinuousSpeedResidual123()
        self.register_buffer("residual_bounds", residual_bounds.detach().clone())
        self.walk_base.eval()
        for parameter in self.walk_base.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def speed_gate(speed: torch.Tensor) -> torch.Tensor:
        x = ((speed - 1.2) / (2.4 - 1.2)).clamp(0.0, 1.0)
        return 3.0 * x.square() - 2.0 * x.pow(3)

    @staticmethod
    def base_observation(observation: torch.Tensor) -> torch.Tensor:
        result = observation.clone()
        result[..., 9] = torch.minimum(
            result[..., 9], torch.as_tensor(1.2, dtype=result.dtype, device=result.device)
        )
        return result

    def forward_components(self, observation: torch.Tensor):
        if observation.shape[-1] != 123:
            raise ValueError(f"expected 123D observation, got {observation.shape}")
        base_observation = self.base_observation(observation)
        with torch.no_grad():
            base_action = self.walk_base(base_observation)
        gate = self.speed_gate(observation[..., 9])
        raw = self.residual(observation)
        bounded = self.residual_bounds * torch.tanh(raw)
        return base_action, bounded, gate

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        base_action, bounded, gate = self.forward_components(observation)
        # The WALK rows do not execute a floating-point residual addition.
        final_action = base_action.clone()
        active = torch.nonzero(gate > 0.0, as_tuple=False).flatten()
        if len(active):
            final_action[active] = base_action[active] + gate[active, None] * bounded[active]
        if not torch.isfinite(final_action).all():
            raise RuntimeError("non-finite frozen-base residual action")
        return final_action
