from __future__ import annotations

from qmini_population_bwm.executor import ActionExecutor


def test_proposed_and_applied_actions_are_distinct_and_logged() -> None:
    executor = ActionExecutor(lower=(-1.0,) * 10, upper=(1.0,) * 10)
    result = executor.apply((2.0, -2.0) + (0.0,) * 8)
    assert result.action_proposed[:2] == (2.0, -2.0)
    assert result.action_applied[:2] == (1.0, -1.0)
    assert result.saturation_mask[:2] == (True, True)
    assert result.saturation_dwell[:2] == (1, 1)
    second = executor.apply((0.0,) * 10)
    assert second.saturation_dwell[:2] == (0, 0)
