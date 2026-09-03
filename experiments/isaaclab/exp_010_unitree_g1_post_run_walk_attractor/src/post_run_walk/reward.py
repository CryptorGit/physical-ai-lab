"""Frozen steady-state reward for POST_RUN_WALK Pilot 1."""
from __future__ import annotations

import torch


def post_run_walk_reward(
    *,
    speed: torch.Tensor,
    heading_error: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    stable_support: torch.Tensor,
    excessive_flight: torch.Tensor,
    dangerous_slip: torch.Tensor,
    impact: torch.Tensor,
    saturation: torch.Tensor,
    fall: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms = {
        "speed_tracking": 2.0 * torch.exp(-((speed - 1.2) / 0.30) ** 2),
        "heading_tracking": torch.exp(-((heading_error / 0.12) ** 2)),
        "upright": torch.exp(-((torch.sqrt(roll**2 + pitch**2) / 0.20) ** 2)),
        "stable_support": 0.5 * stable_support.float(),
        "excessive_flight": -1.0 * excessive_flight.float(),
        "dangerous_slip": -2.0 * dangerous_slip.float(),
        "impact": -1.0 * impact.float(),
        "long_dwell_saturation": -1.0 * saturation.float(),
        "fall": -200.0 * fall.float(),
        "action_rate": -0.005 * (action - previous_action).square().mean(dim=-1),
    }
    total = torch.stack(tuple(terms.values())).sum(0)
    if not torch.isfinite(total).all():
        raise RuntimeError("non-finite POST_RUN_WALK reward")
    return total, terms
