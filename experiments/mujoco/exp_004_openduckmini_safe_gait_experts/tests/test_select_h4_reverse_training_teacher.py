from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import build_h4_slew_feasible_reverse_teacher_bank as builder
from scripts import select_h4_reverse_training_teacher as selection


def _payload() -> dict:
    bank = json.loads(selection.DEFAULT_BANK.read_text(encoding="utf-8"))
    screen = json.loads(selection.DEFAULT_SCREEN.read_text(encoding="utf-8"))
    return selection.select_payload(bank, screen)


def test_screen_winner_is_frozen_as_training_only_teacher() -> None:
    payload = _payload()
    assert payload["selection"]["candidate_id"] == "cbe8decf6a7c4e5e"
    assert payload["selection"]["candidate_name"] == "h4_reverse_c1p50_h2_e1p00"
    assert payload["selection"]["full_h4_failure3_passed"] is False
    assert payload["decision"]["adoption"] is False
    assert payload["decision"]["direct_runtime_use"] == "PROHIBITED"
    assert selection.validate_selected(payload)["passed"]


def test_source_compatible_adapter_matches_independent_manual_interpolation() -> None:
    payload = _payload()
    table = np.asarray(payload["teacher"]["target_table_rad"], dtype=np.float64)
    phases = np.asarray((-54.25, -0.1, 0.0, 0.5, 14.0, 53.75, 54.0, 108.125))
    actual = selection.optimized_backward_reference_adapter(payload, phases)

    expected = []
    for phase in phases:
        wrapped = phase % len(table)
        lower = int(np.floor(wrapped))
        upper = (lower + 1) % len(table)
        fraction = wrapped - lower
        expected.append((1.0 - fraction) * table[lower] + fraction * table[upper])
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1.0e-14)


def test_adapter_returns_exact_knots_and_wraps_periodically() -> None:
    payload = _payload()
    table = np.asarray(payload["teacher"]["target_table_rad"], dtype=np.float64)
    knots = np.arange(len(table), dtype=np.float64)
    np.testing.assert_array_equal(
        selection.optimized_backward_reference_adapter(payload, knots), table
    )
    probes = np.linspace(-3.0, 57.0, 31)
    np.testing.assert_allclose(
        selection.optimized_backward_reference_adapter(payload, probes),
        selection.optimized_backward_reference_adapter(payload, probes + len(table)),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_selected_validator_rejects_adapter_contract_drift() -> None:
    payload = _payload()
    drifted = deepcopy(payload)
    drifted["adapter_contract"]["phase_advance_bins_per_control"] += 0.01
    result = selection.validate_selected(drifted)
    assert not result["passed"]
    assert "adapter_phase_advance" in result["failures"]
