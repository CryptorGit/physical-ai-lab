from __future__ import annotations

from qmini_population_bwm.fatigue import FatigueLedger


def test_fatigue_equation_and_effectiveness() -> None:
    ledger = FatigueLedger(alpha=0.2, beta=0.1, effectiveness_coefficient=0.5)
    result = ledger.step((1.0,) * 5, (0.0,) * 5, dt=0.015)
    assert result.after.left == (0.2,) * 5
    assert result.after.right == (0.0,) * 5
    assert result.effectiveness[:5] == (0.9,) * 5
    assert result.effectiveness[5:] == (1.0,) * 5
    assert ledger.mechanical_work == (0.075, 0.0)


def test_fatigue_is_clipped_and_reset() -> None:
    ledger = FatigueLedger(alpha=2.0, beta=0.0, effectiveness_coefficient=1.0)
    result = ledger.step((1.0,) * 5, (1.0,) * 5, dt=0.1)
    assert result.after.left == (1.0,) * 5
    assert result.after.right == (1.0,) * 5
    ledger.reset()
    assert ledger.state.left == (0.0,) * 5
