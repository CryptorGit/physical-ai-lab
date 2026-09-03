"""Single Gaussian-head feed-forward students and preserving initializers."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn


ARCHITECTURES = {
    "S0": (256, 128, 128),
    "S1": (512, 512, 256),
    "S2": (1024, 512, 512),
}


class ExplicitModeStudent(nn.Module):
    """One actor and one diagonal Gaussian head; no runtime routing/blending."""

    def __init__(self, hidden_sizes=(256, 128, 128)):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        dims = (141, *self.hidden_sizes, 37)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.log_std = nn.Parameter(torch.zeros(37))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Keep the legacy gait dot-product separate for exact W1B parity. New
        # features are a separate zero-initialized linear contribution.
        layer0 = self.layers[0]
        x = nn.functional.linear(observation[..., :123], layer0.weight[:, :123], layer0.bias)
        x = x + observation[..., 123:124] * layer0.weight[:, 123]
        x = x + nn.functional.linear(observation[..., 124:], layer0.weight[:, 124:], None)
        x = nn.functional.elu(x)
        for layer in self.layers[1:-1]:
            x = nn.functional.elu(layer(x))
        return self.layers[-1](x)

    @property
    def architecture(self) -> list[int]:
        return [141, *self.hidden_sizes, 37]

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _w1b_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload["actor_state_dict"]


def initialize_s0_from_w1b(checkpoint: str | Path) -> ExplicitModeStudent:
    state = _w1b_state(checkpoint)
    model = ExplicitModeStudent(ARCHITECTURES["S0"])
    with torch.no_grad():
        model.layers[0].weight.zero_()
        model.layers[0].weight[:, :123].copy_(state["first_base_weight"])
        model.layers[0].weight[:, 123:124].copy_(state["first_gait_column"])
        model.layers[0].bias.copy_(state["first_bias"])
        hidden = OrderedDict((k.removeprefix("hidden."), v) for k, v in state.items() if k.startswith("hidden."))
        for dst, index in zip(model.layers[1:], (1, 3, 5)):
            dst.load_state_dict({"weight": hidden[f"{index}.weight"], "bias": hidden[f"{index}.bias"]})
        model.log_std.copy_(state["distribution.log_std_walk"])
    return model


def _replication_map(old: int, new: int, device: torch.device) -> torch.Tensor:
    if new < old:
        raise ValueError("Net2Wider cannot shrink a layer")
    return torch.arange(new, device=device) % old


def widen_student(source: ExplicitModeStudent, target_name: str) -> ExplicitModeStudent:
    """Net2Wider replication with outgoing-weight division."""
    target = ExplicitModeStudent(ARCHITECTURES[target_name]).to(next(source.parameters()).device)
    old_widths = source.hidden_sizes
    new_widths = target.hidden_sizes
    maps = [_replication_map(o, n, next(source.parameters()).device) for o, n in zip(old_widths, new_widths)]
    with torch.no_grad():
        # First layer: replicate neurons; all 17 new-feature columns remain as in source.
        target.layers[0].weight.copy_(source.layers[0].weight[maps[0]])
        target.layers[0].bias.copy_(source.layers[0].bias[maps[0]])
        for li in range(1, len(old_widths)):
            in_map, out_map = maps[li - 1], maps[li]
            counts = torch.bincount(in_map, minlength=old_widths[li - 1]).to(source.layers[li].weight.dtype)
            expanded = source.layers[li].weight[out_map][:, in_map] / counts[in_map].reshape(1, -1)
            target.layers[li].weight.copy_(expanded)
            target.layers[li].bias.copy_(source.layers[li].bias[out_map])
        in_map = maps[-1]
        counts = torch.bincount(in_map, minlength=old_widths[-1]).to(source.layers[-1].weight.dtype)
        target.layers[-1].weight.copy_(source.layers[-1].weight[:, in_map] / counts[in_map].reshape(1, -1))
        target.layers[-1].bias.copy_(source.layers[-1].bias)
        target.log_std.copy_(source.log_std)
    return target


def checkpoint_payload(model: ExplicitModeStudent, **metadata) -> dict:
    return {
        "format": "Exp014ExplicitModeStudentV1",
        "actor_state_dict": model.state_dict(),
        "architecture": model.architecture,
        "parameter_count": model.parameter_count,
        "gaussian_head": "single learned diagonal log_std",
        **metadata,
    }
