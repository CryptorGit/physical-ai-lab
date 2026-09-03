from __future__ import annotations

import numpy as np

from safe_gait_experts.h5_command_contract import (
    H5_COMMAND_CONTRACT_ID,
    H5_PLANAR_PHYSICAL_COMMANDS,
    H5_PLANAR_POLICY_COMMANDS,
    H5_PLANAR_ROUTE_NAMES,
    H5_REVERSE_PHASE_DELTA_BINS_PER_CONTROL,
    H5_REVERSE_ROUTE_PROBABILITIES,
    h5_command_contract_manifest,
    h5_planar_policy_command,
    h5_planar_route_name,
    h5_reverse_policy_command,
)
from scripts.evaluate_h5_routed_transitions import (
    H5SafeGaitRouter,
    _h5_resolve_policy_observation_command,
)


def test_formal_planar_command_anchors_are_shared_with_evaluator_mapping():
    for route in H5_PLANAR_ROUTE_NAMES:
        physical = H5_PLANAR_PHYSICAL_COMMANDS[route]
        expected = np.asarray(H5_PLANAR_POLICY_COMMANDS[route])
        assert h5_planar_route_name(physical) == route
        np.testing.assert_array_equal(h5_planar_policy_command(physical), expected)
        actual, yaw_offset, accepted = _h5_resolve_policy_observation_command(
            route if route != "stand" else "stand",
            np.asarray(physical),
            backward_residual_scale=0.0,
        )
        np.testing.assert_array_equal(actual, expected)
        assert yaw_offset == 0.0
        assert accepted is True


def test_reverse_mapper_preserves_signed_yaw_and_shared_phase_clock():
    np.testing.assert_array_equal(
        h5_reverse_policy_command((-0.03, 0.0, 0.20)),
        np.asarray((-0.03, 0.0, 0.40)),
    )
    np.testing.assert_array_equal(
        h5_reverse_policy_command((-0.04, 0.0, -0.20)),
        np.asarray((-0.04, 0.0, -0.40)),
    )
    assert H5_REVERSE_PHASE_DELTA_BINS_PER_CONTROL == 0.81


def test_reverse_curriculum_contract_has_exact_endpoint_majority_and_discrete_anchors():
    assert sum(H5_REVERSE_ROUTE_PROBABILITIES.values()) == 1.0
    assert H5_REVERSE_ROUTE_PROBABILITIES["reverse"] >= 0.50
    manifest = h5_command_contract_manifest()
    assert manifest["contract_id"] == H5_COMMAND_CONTRACT_ID
    assert manifest["reverse_route_probabilities"]["reverse"] == 0.60
    assert manifest["hardware_deployment"] == "PROHIBITED"


def test_h5_router_reaches_formal_minus_030_lateral_endpoint():
    router = H5SafeGaitRouter()
    decisions = [
        router.route((0.04, -0.03, -0.15), dt=0.02) for _ in range(20)
    ]
    decision = decisions[-1]
    assert decision.effective_command[1] == -0.03
    assert decision.effective_command[0] > 0.025
    assert decision.effective_command[2] < -0.10

