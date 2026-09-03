"""Frozen EXP 012 gait-conditioned actor loader."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class FrozenGaitActor(nn.Module):
    """The exact 123D + scalar-gait representation used by Stage 2N/2Q."""

    def __init__(self, checkpoint):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["actor_state_dict"]
        self.first_base_weight = nn.Parameter(state["first_base_weight"], requires_grad=False)
        self.first_gait_column = nn.Parameter(state["first_gait_column"], requires_grad=False)
        self.first_bias = nn.Parameter(state["first_bias"], requires_grad=False)
        self.hidden = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 37),
        )
        self.hidden.load_state_dict(
            OrderedDict(
                (key.removeprefix("hidden."), value)
                for key, value in state.items()
                if key.startswith("hidden.")
            )
        )
        self.register_buffer("log_std_walk", state["distribution.log_std_walk"])
        self.register_buffer("log_std_run", state["distribution.log_std_run"])

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        return self.hidden(first + gait.reshape(-1, 1) * self.first_gait_column.T)

    @property
    def architecture(self):
        return [124, 256, 128, 128, 37]
