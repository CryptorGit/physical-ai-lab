"""Stage 7R trainable 152-D WALK_TO_RUN action infrastructure."""
from __future__ import annotations
import copy
import torch
from torch import nn
from tensordict import TensorDict

RUN_SKILL_ID = 0

class WalkToRunTransitionActor152(nn.Module):
    """A separately-owned RUN-compatible transition actor.

    The complete parent actor is copied. Only the RUN command encoder, RUN
    state adapter, and RUN residual head are trainable; frozen source/target
    experts remain separate runtime objects.
    """
    def __init__(self, parent: nn.Module):
        super().__init__()
        self.actor = copy.deepcopy(parent)
        for p in self.actor.parameters(): p.requires_grad_(False)
        for module in (
            self.actor.skill_command_encoders[RUN_SKILL_ID],
            self.actor.skill_state_adapters[RUN_SKILL_ID],
            self.actor.residual_heads[RUN_SKILL_ID],
        ):
            for p in module.parameters(): p.requires_grad_(True)

    def forward(self, observation_152: torch.Tensor) -> torch.Tensor:
        if observation_152.shape[-1] != 152:
            raise ValueError(f"expected 152 observations, got {observation_152.shape}")
        wrapped = TensorDict({"policy": observation_152}, batch_size=list(observation_152.shape[:-1]))
        action = self.actor(wrapped)
        if action.shape[-1] != 37 or not torch.isfinite(action).all():
            raise RuntimeError("invalid WALK_TO_RUN transition action")
        return action

    def trainable_names(self):
        return [n for n,p in self.named_parameters() if p.requires_grad]

class WalkToRunTransitionAction:
    """Single-owner production action term contract for Stage 7R."""
    observation_dim=152; action_dim=37; action_scale=0.5
    def __init__(self, actor: WalkToRunTransitionActor152):
        self.actor=actor
        self.global_previous_action=None
    def apply(self, observation_152, global_previous_action):
        if not torch.equal(observation_152[...,86:123],global_previous_action):
            raise RuntimeError("global previous-action mismatch")
        self.global_previous_action=global_previous_action
        return self.actor(observation_152)

WalkToRunTransitionActionCfg = dict(
    class_type="WalkToRunTransitionAction", observation_dim=152,
    action_dim=37, action_scale=0.5, runtime_blend=False,
)


class RunToWalkTransitionActor152(WalkToRunTransitionActor152):
    """RUN-initialized, separately-owned 152-D RUN_TO_WALK actor."""

    def forward(self, observation_152: torch.Tensor) -> torch.Tensor:
        action = super().forward(observation_152)
        if not torch.isfinite(action).all():
            raise RuntimeError("invalid RUN_TO_WALK transition action")
        return action


class RunToWalkTransitionAction(WalkToRunTransitionAction):
    """Single-owner production action term for the reverse directed edge."""


RunToWalkTransitionActionCfg = dict(
    class_type="RunToWalkTransitionAction",
    observation_dim=152,
    action_dim=37,
    action_scale=0.5,
    runtime_blend=False,
)
