"""Single-head continuous-speed WALK/RUN student."""

from __future__ import annotations

import torch
from torch import nn


class UnifiedWalkRunStudent123(nn.Sequential):
    observation_dim = 123
    action_dim = 37
    action_scale = 0.5

    def __init__(self):
        super().__init__(
            nn.Linear(123, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 37),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(f"expected 123D observation, got {observation.shape}")
        action = super().forward(observation)
        if action.shape[-1] != self.action_dim or not torch.isfinite(action).all():
            raise RuntimeError("invalid unified student action")
        return action

    def initialize_from_walk(self, walk_actor: nn.Module) -> None:
        self.load_state_dict(walk_actor.state_dict(), strict=True)
