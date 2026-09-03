"""History-MLP and diagnostic GRU probes."""

from __future__ import annotations

import torch
from torch import nn

from .probes import _train


def train_history_mlp(x, y, config, seed=0):
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
    return _train(model, x, y, seed=seed, **config)


class GRUProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(output[:, -1])


def train_gru(x, y, config, input_dim, hidden_dim=64, seed=0):
    model = GRUProbe(input_dim, hidden_dim)
    reshaped = x.reshape(len(x), -1, input_dim)
    return _train(model, reshaped, y, seed=seed, **config)


def predict_gru(model, x, input_dim):
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(torch.as_tensor(x.reshape(len(x), -1, input_dim), dtype=torch.float32, device=device)).squeeze(-1).sigmoid()
    return output.cpu().numpy()
