"""Independent full-action POST_RUN_WALK actor."""
from __future__ import annotations

import copy
import torch
from torch import nn


class PostRunWalkExpert152(nn.Module):
    """Strict deep copy of the Stage 8C model_10 transition actor.

    The copied module remains a full 37-D policy. It is intentionally neither
    a residual nor a WALK/RUN blend.
    """

    observation_dim = 152
    action_dim = 37
    action_scale = 0.5

    def __init__(self, stage8c_actor: nn.Module):
        super().__init__()
        self.policy = copy.deepcopy(stage8c_actor)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(True)

    def forward(self, observation_152: torch.Tensor) -> torch.Tensor:
        if observation_152.shape[-1] != self.observation_dim:
            raise ValueError(f"expected [...,152], got {tuple(observation_152.shape)}")
        action = self.policy(observation_152)
        if action.shape[-1] != self.action_dim or not torch.isfinite(action).all():
            raise RuntimeError("invalid POST_RUN_WALK action")
        return action

    @staticmethod
    def assert_strict_initialization(
        source_actor: nn.Module,
        copied_actor: "PostRunWalkExpert152",
        observation: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            source = source_actor(observation)
            copied = copied_actor(observation)
        if not torch.equal(source, copied):
            raise RuntimeError("POST_RUN_WALK strict-copy action mismatch")
