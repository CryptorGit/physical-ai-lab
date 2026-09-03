"""Vectorized POST_RUN_WALK state contract."""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class PostRunWalkContractState:
    valid_dwell: torch.Tensor
    acquisition: torch.Tensor
    success: torch.Tensor
    support_switches: torch.Tensor
    previous_support: torch.Tensor
    periodic_run: torch.Tensor
    flight_events: torch.Tensor
    alternating_flight_landings: torch.Tensor
    previous_flight: torch.Tensor
    last_landing_side: torch.Tensor

    @classmethod
    def create(cls, size: int, device: torch.device) -> "PostRunWalkContractState":
        z = lambda: torch.zeros(size, device=device)
        b = lambda: torch.zeros(size, dtype=torch.bool, device=device)
        l = lambda value=0: torch.full((size,), value, dtype=torch.long, device=device)
        return cls(z(), b(), b(), l(), l(), b(), l(), l(), b(), l(-1))


def update_contract(
    state: PostRunWalkContractState,
    *,
    dt: float,
    speed: torch.Tensor,
    heading_error: torch.Tensor,
    contacts: torch.Tensor,
    flight_dwell: torch.Tensor,
    fall: torch.Tensor,
    slip: torch.Tensor,
    impact: torch.Tensor,
    saturation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
    single = (support == 1) | (support == 2)
    changed = single & (support != state.previous_support)
    state.support_switches += changed.long()

    flight = ~contacts.any(1)
    landing = state.previous_flight & ~flight
    side = contacts.long().argmax(1)
    has_last = state.last_landing_side >= 0
    alternating = landing & has_last & (side != state.last_landing_side)
    state.flight_events += landing.long()
    state.alternating_flight_landings += alternating.long()
    state.last_landing_side = torch.where(landing, side, state.last_landing_side)
    state.previous_flight.copy_(flight)

    alt_ratio = state.alternating_flight_landings.float() / (state.flight_events - 1).clamp_min(1).float()
    state.periodic_run |= (state.flight_events >= 4) & (alt_ratio >= 0.8)
    stable_support = contacts.any(1) & (state.support_switches >= 2)
    excessive_flight = flight_dwell > 0.16
    safe = ~(fall | slip | impact | saturation)
    instantaneous = (
        ((speed - 1.2).abs() <= 0.20)
        & (heading_error.abs() <= 0.12)
        & ~state.periodic_run
        & ~excessive_flight
        & stable_support
        & safe
    )
    state.valid_dwell = torch.where(
        instantaneous,
        state.valid_dwell + dt,
        torch.zeros_like(state.valid_dwell),
    )
    state.acquisition |= state.valid_dwell >= 0.4
    state.success |= state.valid_dwell >= 8.0
    state.previous_support.copy_(support)
    return {
        "instantaneous_contract": instantaneous,
        "stable_support": stable_support,
        "periodic_run": state.periodic_run,
        "excessive_flight": excessive_flight,
        "acquisition": state.acquisition,
        "success": state.success,
        "valid_dwell": state.valid_dwell,
    }
