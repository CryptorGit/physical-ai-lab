"""Frozen exp_006 RUN route adapter."""

from pathlib import Path

import torch
from tensordict import TensorDict

from .adapters import to_run_observation


class RunExpert:
    trainable = False

    def __init__(self, actor):
        self.actor = actor
        for parameter in actor.parameters():
            parameter.requires_grad_(False)

    def action_components(self, state, command, *, route="RUN"):
        adapted = to_run_observation(state, command, route=route)
        wrapped = TensorDict({"policy": adapted}, batch_size=list(adapted.shape[:-1]))
        return self.actor.diagnostic_components(wrapped)

    def __call__(self, state, command, *, route="RUN"):
        return self.action_components(state, command, route=route)["action_mean"]


def load_run_expert(checkpoint: str | Path, *, device: str = "cpu") -> RunExpert:
    """Strictly load the frozen 152-D exp_006 residual actor."""
    from g1_command_skills.models.residual_actor import G1CommandResidualActor

    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    observation = TensorDict(
        {"policy": torch.zeros(1, 152, device=device)}, batch_size=[1]
    )
    actor = G1CommandResidualActor(
        observation,
        {"actor": ["policy"]},
        "actor",
        37,
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[],
        crouch_controller="scripted_shallow_v1",
        learned_crouch_residual_enabled=False,
    ).to(device)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.eval()
    return RunExpert(actor)
