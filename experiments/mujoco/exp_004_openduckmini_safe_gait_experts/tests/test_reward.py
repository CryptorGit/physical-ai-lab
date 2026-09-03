from __future__ import annotations

import numpy as np
import pytest

from safe_gait_experts.reward import (
    bounded_axis_tracking,
    bounded_symmetric_tracking,
)


def test_reward_is_bounded_and_exact_tracking_is_one() -> None:
    command = np.array([0.10, -0.08, 0.60])
    actual = np.array(
        [
            [0.10, -0.08, 0.60],
            [0.00, -0.08, 0.60],
            [0.10, 0.02, 0.60],
            [0.10, -0.08, -0.60],
        ]
    )
    reward = bounded_symmetric_tracking(command, actual)
    assert reward[0] == pytest.approx(1.0)
    assert np.all(reward >= 0.0)
    assert np.all(reward <= 1.0)
    assert np.all(reward[1:] < reward[0])


def test_grid_argmax_is_the_command_for_vx_vy_and_yaw() -> None:
    command = np.array([0.10, -0.10, 0.40])
    deltas = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
    candidates = np.array(
        [command + [dx, dy, dyaw] for dx in deltas for dy in deltas for dyaw in deltas]
    )
    rewards = bounded_symmetric_tracking(command, candidates)
    maximum = candidates[int(np.argmax(rewards))]
    np.testing.assert_allclose(maximum, command, atol=0.0)
    assert np.count_nonzero(np.isclose(rewards, rewards.max(), atol=1e-14)) == 1


def test_reward_is_symmetric_around_command_and_under_sign_flip() -> None:
    command = np.array([0.08, -0.06, 0.35])
    delta = np.array([0.03, -0.02, 0.11])
    positive = bounded_symmetric_tracking(command, command + delta)
    negative = bounded_symmetric_tracking(command, command - delta)
    flipped = bounded_symmetric_tracking(-command, -(command + delta))
    assert positive == pytest.approx(negative, abs=1e-15)
    assert positive == pytest.approx(flipped, abs=1e-15)


def test_axis_terms_have_independent_command_centred_maxima() -> None:
    command = np.array([0.1, 0.1, -0.5])
    exact = bounded_axis_tracking(command, command)
    displaced = bounded_axis_tracking(command, command + [0.01, -0.02, 0.03])
    np.testing.assert_allclose(exact, np.ones(3))
    assert np.all(displaced < exact)


def test_integer_inputs_do_not_quantize_subunit_sigma() -> None:
    assert bounded_symmetric_tracking([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)
    displaced = bounded_symmetric_tracking([0, 0, 0], [1, 0, 0])
    assert np.isfinite(displaced)
    assert displaced < 1.0


@pytest.mark.parametrize(
    "sigma, weight",
    [((0.0, 0.1, 0.2), (1.0, 1.0, 1.0)), ((0.1, 0.1, 0.2), (1.0, -1.0, 1.0))],
)
def test_invalid_reward_parameters_are_rejected(sigma, weight) -> None:
    with pytest.raises(ValueError):
        bounded_symmetric_tracking([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], sigma=sigma, weight=weight)
