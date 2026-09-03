"""H5 target-space and continuous-transition contract.

H4 exposed a causal integration bug: candidate actions were blended before
their absolute-target decoder, so a route switch could change the meaning of
the command and amplify the first target jump.  H5 makes the order explicit:

    actor action -> canonical SAFE absolute target -> target-space blend
    -> one final margin/slew/head guard -> MuJoCo control

The module is deliberately pure and works with NumPy or ``jax.numpy``.  It is
simulation-only; it does not authorize adoption or hardware use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contract import ACTUATOR_JOINT_ORDER, HEAD_JOINTS
from .h4_training_alignment import (
    H4_ACTOR_OBSERVATION_WIDTH,
    final_target_guard_step,
    reverse_iteration_v6_absolute_full_leg_targets,
)


H5_CONTRACT_ID = "OPEN_DUCK_MINI_H5_TARGET_SPACE_ROUTING_V1"
H5_ACTOR_OBSERVATION_WIDTH = H4_ACTOR_OBSERVATION_WIDTH
H5_ACTION_WIDTH = len(ACTUATOR_JOINT_ORDER)
H5_HEAD_INDICES = tuple(
    index
    for index, name in enumerate(ACTUATOR_JOINT_ORDER)
    if name in HEAD_JOINTS
)
H5_LEG_INDICES = tuple(
    index for index in range(H5_ACTION_WIDTH) if index not in H5_HEAD_INDICES
)
H5_DOMAINS = ("planar", "reverse")
H5_TARGET_DOMAINS = ("planar", "reverse", "unified")

# The route partition is intentionally explicit.  A command-conditioned
# planar expert owns all non-reverse motion; the reverse expert owns straight
# reverse and both reverse turns.  Stand is listed in both domains because a
# domain change must have a deterministic stationary endpoint.
H5_PLANAR_ROUTES = (
    "stand",
    "forward",
    "lateral_left",
    "lateral_right",
    "yaw_left",
    "yaw_right",
    "forward_turn_left",
    "forward_turn_right",
    "forward_lateral_left_turn",
    "forward_lateral_right_turn",
)
H5_REVERSE_ROUTES = ("stand", "reverse", "reverse_turn_left", "reverse_turn_right")


def h5_domain_for_route(route: str) -> str:
    """Return the owning H5 domain for a canonical routed expert."""

    name = str(route)
    if name in H5_REVERSE_ROUTES and name != "stand":
        return "reverse"
    if name in H5_PLANAR_ROUTES or name == "compound":
        return "planar"
    raise ValueError(f"H5 route is not in the explicit domain partition: {name!r}")


def _require_action(value: Any, label: str, *, xp: Any) -> Any:
    array = xp.asarray(value)
    if array.shape != (H5_ACTION_WIDTH,):
        raise ValueError(f"{label} must have shape ({H5_ACTION_WIDTH},), got {array.shape}")
    return array


def _head_mask(*, xp: Any) -> Any:
    return xp.asarray([index in H5_HEAD_INDICES for index in range(H5_ACTION_WIDTH)])


def h5_decode_absolute_targets(action: Any, *, domain: str, xp: Any = np) -> Any:
    """Decode one domain candidate into canonical SAFE absolute targets.

    Both domains use the same bounded physical decoder.  ``domain`` remains a
    required argument so a future domain-specific decoder cannot be introduced
    silently and so provenance can be audited in every call site.
    """

    if str(domain) not in H5_TARGET_DOMAINS:
        raise ValueError(f"unsupported H5 target domain: {domain!r}")
    values = _require_action(action, "action", xp=xp)
    targets = reverse_iteration_v6_absolute_full_leg_targets(values, xp=xp)
    if xp is np:
        numeric = np.asarray(targets)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("H5 decoded targets must be finite")
        if not np.array_equal(numeric[list(H5_HEAD_INDICES)], np.zeros(len(H5_HEAD_INDICES))):
            raise ValueError("H5 decoder must lock all head targets to exact zero")
    return targets


def h5_blend_targets(
    from_targets: Any,
    to_targets: Any,
    alpha: float,
    *,
    xp: Any = np,
) -> Any:
    """Blend already-decoded targets, never raw actor actions."""

    source = _require_action(from_targets, "from_targets", xp=xp)
    target = _require_action(to_targets, "to_targets", xp=xp)
    weight = float(alpha)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("H5 blend alpha must be finite and in [0, 1]")
    blended = (1.0 - weight) * source + weight * target
    return xp.where(_head_mask(xp=xp), xp.zeros_like(blended), blended)


def h5_decode_and_blend(
    from_action: Any,
    to_action: Any,
    *,
    from_domain: str,
    to_domain: str,
    alpha: float,
    xp: Any = np,
) -> Any:
    """Perform the only approved candidate composition order."""

    source = h5_decode_absolute_targets(from_action, domain=from_domain, xp=xp)
    target = h5_decode_absolute_targets(to_action, domain=to_domain, xp=xp)
    return h5_blend_targets(source, target, alpha, xp=xp)


def h5_final_guard_step(
    desired_targets: Any,
    previous_applied_targets: Any,
    *,
    xp: Any = np,
) -> Any:
    """Apply the single shared target margin/slew/head guard."""

    desired = _require_action(desired_targets, "desired_targets", xp=xp)
    previous = _require_action(
        previous_applied_targets, "previous_applied_targets", xp=xp
    )
    return final_target_guard_step(desired, previous, xp=xp)


@dataclass(frozen=True)
class H5TransitionState:
    """State that must survive a routed command change."""

    phase_index: float
    active_domain: str
    previous_applied_targets: tuple[float, ...]
    contact_continuity: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if self.active_domain not in H5_DOMAINS:
            raise ValueError(f"unknown H5 transition domain: {self.active_domain!r}")
        if not np.isfinite(float(self.phase_index)):
            raise ValueError("H5 phase index must be finite")
        targets = np.asarray(self.previous_applied_targets, dtype=np.float64)
        if targets.shape != (H5_ACTION_WIDTH,) or not np.all(np.isfinite(targets)):
            raise ValueError("H5 previous applied targets must be finite and 14-wide")
        if not np.array_equal(targets[list(H5_HEAD_INDICES)], np.zeros(len(H5_HEAD_INDICES))):
            raise ValueError("H5 transition state head targets must be exact zero")


def h5_reverse_entry_state(
    previous: H5TransitionState,
    *,
    reverse_entry_phase_index: float = 7.0,
) -> H5TransitionState:
    """Enter reverse with phase 7.0 while preserving guard/contact history."""

    previous.validate()
    entry = float(reverse_entry_phase_index)
    if not np.isfinite(entry) or entry < 0.0:
        raise ValueError("reverse entry phase must be finite and non-negative")
    return H5TransitionState(
        phase_index=entry,
        active_domain="reverse",
        previous_applied_targets=previous.previous_applied_targets,
        contact_continuity=previous.contact_continuity,
    )


def h5_contract_manifest() -> dict[str, Any]:
    """Return JSON-ready contract metadata for training/evaluation artifacts."""

    return {
        "contract_id": H5_CONTRACT_ID,
        "actor_observation_width": H5_ACTOR_OBSERVATION_WIDTH,
        "action_width": H5_ACTION_WIDTH,
        "head_indices": list(H5_HEAD_INDICES),
        "domains": list(H5_DOMAINS),
        "planar_routes": list(H5_PLANAR_ROUTES),
        "reverse_routes": list(H5_REVERSE_ROUTES),
        "composition_order": [
            "infer_each_candidate",
            "decode_each_to_safe_absolute_targets",
            "blend_target_space",
            "single_final_margin_slew_head_guard",
        ],
        "reverse_first_observation_phase_index": 7.0,
        "guard_history_continuous_across_route_changes": True,
        "contact_history_continuous_across_route_changes": True,
        "hardware_deployment": "PROHIBITED",
    }
