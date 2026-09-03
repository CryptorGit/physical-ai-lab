from __future__ import annotations

import pytest

from scripts.diagnose_backward_exit_recovery import (
    CONTROL_DT_S,
    FIXED_SEEDS,
    PHASE_ENTRY_INDICES,
    RecoveryStrategy,
    exact_prefix_schedule,
    recovery_strategy_bank,
    seconds_to_safe_ticks,
)


def test_exact_replay_contract_is_frozen() -> None:
    assert FIXED_SEEDS == (22260808, 22260809, 22260810, 22260811, 22260812)
    assert PHASE_ENTRY_INDICES == {
        "reverse": 6.0,
        "reverse_turn_left": 4.0,
        "reverse_turn_right": 4.0,
    }
    schedule = exact_prefix_schedule()
    assert len(schedule) == 7
    assert [case[0] for case in schedule] == [
        "transition_stand_0",
        "transition_forward",
        "transition_stand_after_forward",
        "transition_reverse",
        "transition_stand_after_reverse",
        "transition_reverse_turn_left",
        "transition_stand_after_reverse_turn_left",
    ]
    # The reusable diagnostic now follows the central, current endpoint.  The
    # immutable 588f... selection artifact is separately validated as the
    # historical -0.075 component evidence and cannot prove this endpoint.
    assert schedule[3][1] == (-0.050, 0.0, 0.0)
    assert schedule[-2][1] == (-0.03, 0.0, 0.20)
    assert schedule[-2][2] == 15.0
    assert schedule[-1][2] == 5.0


def test_recovery_duration_quantizes_upward() -> None:
    assert CONTROL_DT_S == 0.02
    assert seconds_to_safe_ticks(0.0) == 0
    assert seconds_to_safe_ticks(0.25) == 13
    assert seconds_to_safe_ticks(0.50) == 25
    assert seconds_to_safe_ticks(1.00) == 50


def test_hold_and_linear_release_curve() -> None:
    strategy = RecoveryStrategy("test", 0.0125, 0.25, 0.25)
    strategy.validate()
    assert strategy.hold_ticks == 13
    assert strategy.release_ticks == 13
    assert strategy.extra_margin_for_tick(0) == 0.0125
    assert strategy.extra_margin_for_tick(12) == 0.0125
    assert 0.0 < strategy.extra_margin_for_tick(13) < 0.0125
    assert strategy.extra_margin_for_tick(24) > 0.0
    assert strategy.extra_margin_for_tick(25) == 0.0
    assert strategy.extra_margin_for_tick(26) == 0.0


def test_strategy_bank_contains_requested_comparisons() -> None:
    strategies = recovery_strategy_bank()
    assert strategies[0] == RecoveryStrategy(
        "baseline_immediate_release", 0.0, 0.0, 0.0
    )
    contracts = {
        (strategy.cap_rad, strategy.hold_seconds, strategy.release_seconds)
        for strategy in strategies
    }
    for cap in (0.0125, 0.015, 0.020):
        for hold in (0.25, 0.50, 1.00):
            assert (cap, hold, 0.0) in contracts
            assert (cap, hold, 0.25) in contracts


def test_invalid_recovery_contract_rejected() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        RecoveryStrategy("too_large", 0.021, 0.25, 0.0).validate()
    with pytest.raises(ValueError, match="requires a positive hold"):
        RecoveryStrategy("no_hold", 0.0125, 0.0, 0.25).validate()
