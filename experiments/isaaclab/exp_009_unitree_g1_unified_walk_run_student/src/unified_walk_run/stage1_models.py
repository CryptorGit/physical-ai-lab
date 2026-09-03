"""Diagnostic-only Stage 1 models. None is a production controller."""

from __future__ import annotations

import torch
from torch import nn


def mlp(dimensions: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(source, target))
        if index < len(dimensions) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class DiagnosticSingleHead(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int]):
        super().__init__()
        self.input_dim = input_dim
        self.network = mlp([input_dim, *hidden, 37])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.input_dim:
            raise ValueError((observation.shape, self.input_dim))
        return self.network(observation)


class DiagnosticMultiHead(nn.Module):
    """Oracle-regime upper bound with a shared trunk and three action heads."""

    def __init__(self):
        super().__init__()
        self.trunk = mlp([123, 256, 128, 128])
        self.heads = nn.ModuleList([nn.Linear(128, 37) for _ in range(3)])

    def forward(self, observation: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        latent = self.trunk(observation)
        stacked = torch.stack([head(latent) for head in self.heads], dim=1)
        return stacked[torch.arange(len(observation), device=observation.device), regime]

