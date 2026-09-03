"""Small diagnostic linear and static-MLP probes."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(0, keepdims=True)
        self.std = x.std(0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std


def _train(model, x, y, *, epochs, batch_size, learning_rate, weight_decay, regression=False, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss() if regression else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], device=device))
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        for indices in torch.randperm(len(tx), generator=generator, device=device).split(batch_size):
            output = model(tx[indices]).squeeze(-1)
            loss = criterion(output, ty[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def train_linear(x, y, config, regression=False, seed=0):
    return _train(nn.Linear(x.shape[1], 1), x, y, regression=regression, seed=seed, **config)


def train_mlp(x, y, config, hidden=(128, 64), regression=False, seed=0):
    model = nn.Sequential(nn.Linear(x.shape[1], hidden[0]), nn.ELU(), nn.Linear(hidden[0], hidden[1]), nn.ELU(), nn.Linear(hidden[1], 1))
    return _train(model, x, y, regression=regression, seed=seed, **config)


def predict(model, x, probabilities=True):
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(torch.as_tensor(x, dtype=torch.float32, device=device)).squeeze(-1)
        if probabilities:
            output = output.sigmoid()
    return output.cpu().numpy()


def ridge_predict(x_train, y_train, x_test, alpha=1.0):
    """Closed-form ridge regression with an unpenalized centered intercept."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_mean = float(y_train.mean())
    gram = x_train.T @ x_train
    gram.flat[:: gram.shape[0] + 1] += alpha
    coefficients = np.linalg.solve(gram, x_train.T @ (y_train - y_mean))
    return (x_test @ coefficients + y_mean).astype(np.float32)
