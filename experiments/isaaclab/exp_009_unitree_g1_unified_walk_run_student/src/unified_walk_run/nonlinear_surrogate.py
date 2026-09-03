"""Frozen nonlinear short-horizon locomotion dynamics surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


PHYSICAL_INDICES = tuple(range(0, 9)) + tuple(range(12, 86))
COMMAND_INDICES = tuple(range(9, 12))
PREVIOUS_ACTION_INDICES = tuple(range(86, 123))
PHYSICAL_DIM = 83
ACTION_DIM = 37


@dataclass
class SurrogatePrediction:
    physical_residual: torch.Tensor
    contacts: torch.Tensor
    support_logits: torch.Tensor
    landing_logits: torch.Tensor
    gait_logits: torch.Tensor


class NonlinearLocomotionDynamicsSurrogate(nn.Module):
    """A fixed residual MLP member; ensembles are constructed by the caller."""

    input_dim = 123 + ACTION_DIM
    physical_dim = PHYSICAL_DIM
    output_dim = PHYSICAL_DIM + 3 + 4 + 3 + 3

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> SurrogatePrediction:
        if observation.shape[-1] != 123 or action.shape[-1] != ACTION_DIM:
            raise ValueError(f"expected (*,123) observation and (*,37) action, got {observation.shape}, {action.shape}")
        value = self.trunk(torch.cat((observation, action), dim=-1))
        if not torch.isfinite(value).all():
            raise RuntimeError("non-finite surrogate prediction")
        return SurrogatePrediction(
            physical_residual=value[..., :83],
            contacts=value[..., 83:86],
            support_logits=value[..., 86:90],
            landing_logits=value[..., 90:93],
            gait_logits=value[..., 93:96],
        )


def reconstruct_observation(
    current_observation: torch.Tensor,
    action: torch.Tensor,
    normalized_residual: torch.Tensor,
    physical_mean: torch.Tensor,
    physical_std: torch.Tensor,
) -> torch.Tensor:
    """Analytically preserve command, install action as history, and predict physical state."""
    next_observation = current_observation.clone()
    physical = current_observation[..., PHYSICAL_INDICES]
    delta = normalized_residual * physical_std + physical_mean
    next_observation[..., PHYSICAL_INDICES] = physical + delta
    next_observation[..., COMMAND_INDICES] = current_observation[..., COMMAND_INDICES]
    next_observation[..., PREVIOUS_ACTION_INDICES] = action
    return next_observation
