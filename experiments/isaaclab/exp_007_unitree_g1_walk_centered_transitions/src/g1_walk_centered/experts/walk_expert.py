"""Frozen Stage 2 actor loader and adapter."""

from pathlib import Path

import torch
import torch.nn as nn

from .adapters import to_walk_observation


class LegacyWalkActor(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37))


class WalkExpert:
    trainable = False

    def __init__(self, actor):
        self.actor = actor
        for parameter in actor.parameters():
            parameter.requires_grad_(False)

    def __call__(self, state, command):
        return self.actor(to_walk_observation(state, command))


def load_walk_expert(checkpoint: str | Path, *, device: str | torch.device = "cpu") -> WalkExpert:
    """Load the immutable exp_005 actor without creating an optimizer or critic."""
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    source = payload["actor_state_dict"]
    actor = LegacyWalkActor().to(device)
    actor.load_state_dict(
        {key.removeprefix("mlp."): value for key, value in source.items() if key.startswith("mlp.")},
        strict=True,
    )
    actor.eval()
    return WalkExpert(actor)
