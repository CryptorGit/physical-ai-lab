from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from router import (  # noqa: E402
    ALLOWED_EXPERTS,
    COMPOUND,
    DEFAULT_COMMAND_MAX,
    DEFAULT_COMMAND_MIN,
    FORWARD,
    LATERAL_LEFT,
    LATERAL_RIGHT,
    PROHIBITED_EXPERTS,
    REVERSE,
    REVERSE_TURN_LEFT,
    REVERSE_TURN_ENDPOINTS,
    REVERSE_TURN_RIGHT,
    STAND,
    YAW_LEFT,
    YAW_RIGHT,
    RouterConfig,
    SafeGaitRouter,
)


FAST_CONFIG = RouterConfig(
    linear_deadband=0.03,
    yaw_deadband=0.10,
    linear_hysteresis=0.01,
    yaw_hysteresis=0.02,
    slew_rate=(100.0, 100.0, 100.0),
    blend_duration_s=0.20,
)


def _one(command: tuple[float, float, float]):
    return SafeGaitRouter(FAST_CONFIG).route(command, dt=0.01)


def test_deadband_and_hysteresis_boundaries_are_deterministic() -> None:
    router = SafeGaitRouter(FAST_CONFIG)

    assert router.route((0.039999, 0.0, 0.0), dt=0.01).expert == STAND
    assert router.route((0.040000, 0.0, 0.0), dt=0.01).expert == FORWARD
    # Once active, values strictly above the 0.02 exit boundary remain active.
    assert router.route((0.020001, 0.0, 0.0), dt=0.01).expert == FORWARD
    assert router.route((0.020000, 0.0, 0.0), dt=0.01).expert == STAND

    assert router.route((0.0, 0.0, 0.119999), dt=0.01).expert == STAND
    assert router.route((0.0, 0.0, 0.120000), dt=0.01).expert == YAW_LEFT
    assert router.route((0.0, 0.0, 0.080000), dt=0.01).expert == STAND


@pytest.mark.parametrize(
    ("positive", "negative", "left_expert", "right_expert"),
    [
        ((0.0, 0.10, 0.0), (0.0, -0.10, 0.0), LATERAL_LEFT, LATERAL_RIGHT),
        ((0.0, 0.0, 0.30), (0.0, 0.0, -0.30), YAW_LEFT, YAW_RIGHT),
    ],
)
def test_left_right_routes_are_sign_symmetric(
    positive: tuple[float, float, float],
    negative: tuple[float, float, float],
    left_expert: str,
    right_expert: str,
) -> None:
    left = _one(positive)
    right = _one(negative)

    assert left.expert == left_expert
    assert right.expert == right_expert
    assert np.allclose(left.effective_command, -np.asarray(right.effective_command))


def test_reverse_turn_has_explicit_priority_over_compound() -> None:
    assert _one((-0.10, 0.0, 0.30)).expert == REVERSE_TURN_LEFT
    assert _one((-0.10, 0.0, -0.30)).expert == REVERSE_TURN_RIGHT
    assert _one((-0.10, 0.10, 0.0)).expert == COMPOUND
    assert _one((-0.10, 0.10, 0.30)).expert == COMPOUND
    assert _one((0.10, 0.0, 0.30)).expert == COMPOUND


def test_single_axis_longitudinal_routes() -> None:
    assert _one((0.10, 0.0, 0.0)).expert == FORWARD
    assert _one((-0.10, 0.0, 0.0)).expert == REVERSE


def test_slew_limit_prevents_an_instant_command_step() -> None:
    config = RouterConfig(
        linear_deadband=0.005,
        yaw_deadband=0.01,
        linear_hysteresis=0.0,
        yaw_hysteresis=0.0,
        slew_rate=(0.10, 0.10, 1.0),
    )
    router = SafeGaitRouter(config)

    first = router.route((0.20, 0.0, 0.0), dt=0.10)
    second = router.route((0.20, 0.0, 0.0), dt=0.10)

    assert first.ramped_command == pytest.approx((0.01, 0.0, 0.0))
    assert second.ramped_command == pytest.approx((0.02, 0.0, 0.0))
    assert first.expert == second.expert == FORWARD


def test_forward_to_reverse_must_pass_through_stand_with_finite_slew() -> None:
    config = RouterConfig(
        linear_deadband=0.02,
        yaw_deadband=0.10,
        linear_hysteresis=0.0,
        yaw_hysteresis=0.0,
        slew_rate=(0.10, 1.0, 1.0),
    )
    router = SafeGaitRouter(config)
    for _ in range(10):
        router.route((0.10, 0.0, 0.0), dt=0.10)

    routes = [router.route((-0.10, 0.0, 0.0), dt=0.10).expert for _ in range(20)]

    assert routes[0] == FORWARD
    assert STAND in routes
    assert routes[-1] == REVERSE
    assert routes.index(STAND) < routes.index(REVERSE)


def test_expert_switch_reports_blend_weight_and_head_lock() -> None:
    router = SafeGaitRouter(FAST_CONFIG)

    switched = router.route((0.10, 0.0, 0.0), dt=0.05)
    blending = router.route((0.10, 0.0, 0.0), dt=0.05)
    complete = [router.route((0.10, 0.0, 0.0), dt=0.05) for _ in range(3)][-1]

    assert switched.switched
    assert switched.blend_from_expert == STAND
    assert switched.blend_to_expert == FORWARD
    assert switched.blend_alpha == 0.0
    assert blending.blend_alpha == pytest.approx(0.25)
    assert complete.blend_alpha == 1.0
    assert switched.head_locked
    assert switched.metadata["head_target_rad"] == (0.0, 0.0, 0.0, 0.0)
    assert switched.metadata["head_action_indices"] == (5, 6, 7, 8)


@pytest.mark.parametrize(
    "bad_command",
    [
        (np.nan, 0.0, 0.0),
        (0.0, np.inf, 0.0),
        (0.0, 0.0, -np.inf),
        (0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        ("forward", 0.0, 0.0),
    ],
)
def test_invalid_command_is_rejected_without_mutating_state(bad_command) -> None:
    router = SafeGaitRouter(FAST_CONFIG)
    accepted = router.route((0.10, 0.0, 0.0), dt=0.01)
    state_before = (router.selected_expert, router.ramped_command)

    with pytest.raises(ValueError):
        router.route(bad_command, dt=0.01)

    assert (router.selected_expert, router.ramped_command) == state_before
    assert accepted.expert == FORWARD


@pytest.mark.parametrize("bad_dt", [0.0, -0.01, np.nan, np.inf, "later"])
def test_invalid_dt_is_rejected_without_mutating_state(bad_dt) -> None:
    router = SafeGaitRouter(FAST_CONFIG)
    state_before = (router.selected_expert, router.ramped_command)

    with pytest.raises(ValueError):
        router.route((0.10, 0.0, 0.0), dt=bad_dt)

    assert (router.selected_expert, router.ramped_command) == state_before


def test_out_of_envelope_command_uses_frozen_asymmetric_bounds() -> None:
    positive = _one((9.0, 8.0, 7.0))
    negative = _one((-9.0, -8.0, -7.0))

    assert positive.command_was_clipped and negative.command_was_clipped
    assert positive.clipped_command == pytest.approx(DEFAULT_COMMAND_MAX)
    assert negative.clipped_command == pytest.approx(DEFAULT_COMMAND_MIN)
    assert negative.clipped_command[0] == pytest.approx(-0.050)
    assert positive.expert == negative.expert == COMPOUND


def test_reverse_turn_snaps_to_validated_endpoint_without_blending() -> None:
    decision = _one((-0.050, 0.0, 0.80))

    assert decision.expert == REVERSE_TURN_LEFT
    assert decision.effective_command == pytest.approx(
        REVERSE_TURN_ENDPOINTS[REVERSE_TURN_LEFT]
    )
    assert decision.ramped_command == decision.effective_command
    assert decision.blend_alpha == 1.0
    assert decision.blend_from_expert == decision.blend_to_expert == REVERSE_TURN_LEFT


@pytest.mark.parametrize("expert", [REVERSE_TURN_LEFT, REVERSE_TURN_RIGHT])
def test_exact_validated_reverse_turn_command_is_reachable(expert: str) -> None:
    endpoint = REVERSE_TURN_ENDPOINTS[expert]
    decision = _one(endpoint)

    assert decision.expert == expert
    assert decision.effective_command == pytest.approx(endpoint)


def test_switching_reverse_turn_profiles_passes_through_exact_stand() -> None:
    router = SafeGaitRouter(FAST_CONFIG)
    left = router.route((-0.050, 0.0, 0.50), dt=0.01)
    interlock = router.route((-0.050, 0.0, -0.50), dt=0.01)
    right = router.route((-0.050, 0.0, -0.50), dt=0.01)

    assert left.expert == REVERSE_TURN_LEFT
    assert interlock.expert == STAND
    assert interlock.effective_command == (0.0, 0.0, 0.0)
    assert interlock.blend_alpha == 1.0
    assert right.expert == REVERSE_TURN_RIGHT
    assert right.effective_command == pytest.approx(
        REVERSE_TURN_ENDPOINTS[REVERSE_TURN_RIGHT]
    )
    assert right.blend_alpha == 1.0


def test_entering_reverse_turn_from_nonstand_has_stand_interlock() -> None:
    router = SafeGaitRouter(FAST_CONFIG)
    assert router.route((0.10, 0.0, 0.0), dt=0.01).expert == FORWARD

    interlock = router.route((-0.050, 0.0, 0.20), dt=0.01)
    entered = router.route((-0.050, 0.0, 0.20), dt=0.01)

    assert interlock.expert == STAND
    assert interlock.ramped_command == (0.0, 0.0, 0.0)
    assert entered.expert == REVERSE_TURN_LEFT


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        ((-0.1, -0.1, -1.0), (-0.1, 0.1, 1.0)),
        ((0.01, -0.1, -1.0), (0.1, 0.1, 1.0)),
        ((-0.1, -0.1, -1.0), (-0.01, 0.1, 1.0)),
    ],
)
def test_invalid_asymmetric_envelope_is_rejected(minimum, maximum) -> None:
    with pytest.raises(ValueError, match="command_min|envelope"):
        RouterConfig(command_min=minimum, command_max=maximum)


def test_router_can_never_select_rejected_all_direction_policy() -> None:
    values = (-1.0, -0.10, 0.0, 0.10, 1.0)
    selected = {
        _one((vx, vy, yaw)).expert
        for vx in values
        for vy in values
        for yaw in values
    }

    assert selected <= ALLOWED_EXPERTS
    assert selected.isdisjoint(PROHIBITED_EXPERTS)
    assert not any("v59" in expert or "v60" in expert for expert in selected)


def test_every_route_carries_locked_head_metadata() -> None:
    commands = (
        (0.0, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        (-0.10, 0.0, 0.0),
        (0.0, 0.10, 0.0),
        (0.0, -0.10, 0.0),
        (0.0, 0.0, 0.30),
        (0.0, 0.0, -0.30),
        (0.10, 0.10, 0.30),
    )

    for command in commands:
        decision = _one(command)
        assert decision.metadata["head_locked"] is True
        assert decision.metadata["head_target_rad"] == (0.0, 0.0, 0.0, 0.0)
