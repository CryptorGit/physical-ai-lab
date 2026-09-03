"""Differentiable student supervision through a frozen surrogate ensemble.

This module never backpropagates through Isaac Sim.  Gradients flow through
the frozen surrogate computation graph to the student actions only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .nonlinear_surrogate import PHYSICAL_INDICES, reconstruct_observation


@dataclass(frozen=True)
class RolloutLossWeights:
    action: float
    action_delta: float
    rollout_state: float
    contact_gait: float


def calibrate_by_initial_median(
    action_values: torch.Tensor,
    action_delta_values: torch.Tensor,
    rollout_values: torch.Tensor,
    contact_values: torch.Tensor,
) -> RolloutLossWeights:
    """One-shot train-split calibration; validation metrics are not accepted."""

    def scale(values: torch.Tensor, target: float) -> float:
        finite = values.detach()[torch.isfinite(values.detach())]
        if not len(finite):
            raise ValueError("loss calibration received no finite values")
        median = finite.median().clamp_min(1e-8)
        return float(target / median)

    return RolloutLossWeights(
        action=scale(action_values, 1.0),
        action_delta=scale(action_delta_values, 0.25),
        rollout_state=scale(rollout_values, 0.5),
        contact_gait=scale(contact_values, 0.5),
    )


def nonlinear_rollout_supervision(
    *,
    student,
    surrogate_ensemble,
    initial_observation: torch.Tensor,
    teacher_observations: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_contacts: torch.Tensor,
    teacher_support: torch.Tensor,
    teacher_gait: torch.Tensor,
    normalization: dict[str, torch.Tensor],
    uncertainty_threshold: float,
    weights: RolloutLossWeights,
    horizons: tuple[int, ...] = (1, 2, 4, 8),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Free-run the student/surrogate and compare against teacher futures."""

    for member in surrogate_ensemble:
        member.eval()
        for parameter in member.parameters():
            parameter.requires_grad_(False)
    state = initial_observation
    previous_student_action = initial_observation[..., 86:123]
    previous_teacher_action = previous_student_action
    action_terms, delta_terms, rollout_terms, contact_terms = [], [], [], []
    included, excluded = 0, 0
    for step in range(max(horizons)):
        action = student(state)
        teacher_action = teacher_actions[:, step]
        action_terms.append(F.huber_loss(action, teacher_action, delta=0.1, reduction="none").mean(1))
        delta_terms.append(F.huber_loss(
            action - previous_student_action,
            teacher_action - previous_teacher_action,
            delta=0.1,
            reduction="none",
        ).mean(1))
        normalized_observation = (state - normalization["obs_mean"]) / normalization["obs_std"]
        normalized_action = (action - normalization["action_mean"]) / normalization["action_std"]
        predictions = [member(normalized_observation, normalized_action) for member in surrogate_ensemble]
        residual_members = torch.stack([item.physical_residual for item in predictions])
        residual = residual_members.mean(0)
        uncertainty = residual_members.var(0).mean(1)
        valid = uncertainty <= float(uncertainty_threshold)
        state = reconstruct_observation(
            state, action, residual, normalization["delta_mean"], normalization["delta_std"]
        )
        if step + 1 in horizons:
            included += int(valid.sum())
            excluded += int((~valid).sum())
            physical_error = (
                state[..., PHYSICAL_INDICES] - teacher_observations[:, step, PHYSICAL_INDICES]
            ) / normalization["obs_std"][list(PHYSICAL_INDICES)]
            state_term = physical_error.square().mean(1)
            contact_logits = torch.stack([item.contacts for item in predictions]).mean(0)
            support_logits = torch.stack([item.support_logits for item in predictions]).mean(0)
            gait_logits = torch.stack([item.gait_logits for item in predictions]).mean(0)
            contact_term = (
                F.binary_cross_entropy_with_logits(
                    contact_logits, teacher_contacts[:, step].float(), reduction="none"
                ).mean(1)
                + F.cross_entropy(support_logits, teacher_support[:, step].long(), reduction="none")
                + F.cross_entropy(gait_logits, teacher_gait[:, step].long(), reduction="none")
            )
            rollout_terms.append(torch.where(valid, state_term, torch.zeros_like(state_term)))
            contact_terms.append(torch.where(valid, contact_term, torch.zeros_like(contact_term)))
        previous_student_action, previous_teacher_action = action, teacher_action
    action_loss = torch.stack(action_terms).mean()
    delta_loss = torch.stack(delta_terms).mean()
    rollout_loss = torch.stack(rollout_terms).sum() / max(included, 1)
    contact_loss = torch.stack(contact_terms).sum() / max(included, 1)
    total = (
        weights.action * action_loss
        + weights.action_delta * delta_loss
        + weights.rollout_state * rollout_loss
        + weights.contact_gait * contact_loss
    )
    return total, {
        "action": action_loss, "action_delta": delta_loss,
        "rollout_state": rollout_loss, "contact_gait": contact_loss,
        "uncertainty_included_steps": torch.as_tensor(included, device=total.device),
        "uncertainty_excluded_steps": torch.as_tensor(excluded, device=total.device),
    }
