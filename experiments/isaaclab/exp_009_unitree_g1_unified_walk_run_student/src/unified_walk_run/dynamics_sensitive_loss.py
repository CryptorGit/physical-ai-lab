"""Data-driven short-horizon dynamics-sensitive distillation loss."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DynamicsSensitivityTable(nn.Module):
    """Frozen state-, regime-, and phase-conditioned local dynamics table."""

    def __init__(
        self,
        jacobian: torch.Tensor,
        contact: torch.Tensor,
        critical_indices: torch.Tensor,
        centroids: torch.Tensor,
        observation_scale: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("jacobian", jacobian.float())       # [regime, phase, cluster, feature, critical]
        self.register_buffer("contact", contact.float())         # [regime, phase, cluster, critical]
        self.register_buffer("critical_indices", critical_indices.long())
        self.register_buffer("centroids", centroids.float())     # [regime, phase, cluster, observation]
        self.register_buffer("observation_scale", observation_scale.float())

    def terms(
        self,
        action_error: torch.Tensor,
        observation: torch.Tensor,
        regime: torch.Tensor,
        phase: torch.Tensor,
        huber_delta: float,
    ):
        critical_error = action_error.index_select(-1, self.critical_indices)
        candidate_centroids = self.centroids[regime, phase]
        normalized_error = (observation[:, None] - candidate_centroids) / self.observation_scale
        cluster = normalized_error.square().mean(-1).argmin(-1)
        table_j = self.jacobian[regime, phase, cluster]
        projected = torch.einsum("bfk,bk->bf", table_j, critical_error)
        dynamic = projected.square().mean(-1)
        contact_weight = self.contact[regime, phase, cluster]
        contact = (
            contact_weight
            * F.huber_loss(critical_error, torch.zeros_like(critical_error), delta=huber_delta, reduction="none")
        ).mean(-1)
        return dynamic, contact


def dynamics_sensitive_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observation: torch.Tensor,
    previous_prediction: torch.Tensor,
    previous_target: torch.Tensor,
    regime: torch.Tensor,
    phase: torch.Tensor,
    table: DynamicsSensitivityTable,
    *,
    huber_delta: float,
    action_delta_weight: float,
    lambda_dynamic: float,
    lambda_contact: float,
):
    action = F.huber_loss(prediction, target, delta=huber_delta)
    action_delta = F.huber_loss(
        prediction - previous_prediction,
        target - previous_target,
        delta=huber_delta,
    )
    dynamic_per, contact_per = table.terms(prediction - target, observation, regime, phase, huber_delta)
    dynamic, contact = dynamic_per.mean(), contact_per.mean()
    total = action + action_delta_weight * action_delta + lambda_dynamic * dynamic + lambda_contact * contact
    return total, {
        "action": action.detach(),
        "action_delta": action_delta.detach(),
        "dynamic": dynamic.detach(),
        "contact": contact.detach(),
        "weighted_dynamic": (lambda_dynamic * dynamic).detach(),
        "weighted_contact": (lambda_contact * contact).detach(),
    }
