"""Offline-only multi-teacher distillation loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(student_action, teacher_action, student_previous, teacher_previous, *, delta, action_weight, action_delta_weight):
    action = F.huber_loss(student_action, teacher_action, delta=delta)
    action_delta = F.huber_loss(
        student_action - student_previous,
        teacher_action - teacher_previous,
        delta=delta,
    )
    return action_weight * action + action_delta_weight * action_delta, {
        "action_huber": action.detach(),
        "action_delta_huber": action_delta.detach(),
    }
