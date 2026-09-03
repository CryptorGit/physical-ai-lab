from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import build_h4_slew_feasible_reverse_teacher_bank as teacher
from scripts import screen_h4_slew_feasible_reverse_teachers as screen


class FakeEvaluator:
    def __init__(self) -> None:
        self.phase_steps = 27
        self.backward_reference_frames = np.zeros((27, 16))
        self.backward_leg_means = np.ones(10)
        self.backward_leg_deviations = np.zeros((27, 10))
        self.backward_gait_scales = np.zeros(10)
        self.backward_gait_biases = np.ones(10)
        self.backward_phase_rate = 9.0
        self.backward_residual_scale = 1.0


def test_selected_screen_is_bounded_and_pure_validated() -> None:
    payload = teacher.build_bank()
    selected = screen.selected_candidates(payload)
    assert tuple(candidate["name"] for candidate in selected) == screen.SELECTED_NAMES
    assert len(selected) == 4
    assert all(teacher.validate_candidate(candidate)["passed"] for candidate in selected)


def test_explicit_teacher_injection_reproduces_target_table() -> None:
    candidate = teacher.build_bank()["candidates"][0]
    evaluator = FakeEvaluator()
    screen.inject_teacher_table(evaluator, candidate)
    expected = np.asarray(candidate["target_table_rad"])[
        :, teacher.LEG_ACTUATOR_INDICES
    ]
    assert evaluator.phase_steps == teacher.RESAMPLED_PHASE_STEPS
    np.testing.assert_array_equal(evaluator.backward_leg_means, np.zeros(10))
    np.testing.assert_allclose(evaluator.backward_leg_deviations, expected)
    np.testing.assert_array_equal(evaluator.backward_gait_scales, np.ones(10))
    np.testing.assert_array_equal(evaluator.backward_gait_biases, np.zeros(10))
    assert evaluator.backward_phase_rate == candidate["construction"][
        "phase_advance_bins_per_control"
    ]
    assert evaluator.backward_residual_scale == 0.0


def test_central_hash_pin_matches_current_snapshot() -> None:
    assert screen.verify_central_snapshot() == screen.EXPECTED_CENTRAL_SHA256
    assert screen.CURRENT_CENTRAL_SNAPSHOT_ID == "h4_strict_quality_gate_v2"


def test_historical_snapshot_remains_named_but_cannot_match_current_source() -> None:
    historical = screen.CENTRAL_SNAPSHOT_SHA256[
        screen.HISTORICAL_FORCE_CONTACT_V1_SNAPSHOT_ID
    ]
    assert historical["safe_gait_experts/gait_quality.py"] == (
        "20a5010037f2157a089501e012881cabceed794bc77b1cca8ca5eaf6f7e88b61"
    )
    with pytest.raises(ValueError, match="h4_force_contact_v1_historical"):
        screen.verify_central_snapshot(
            screen.HISTORICAL_FORCE_CONTACT_V1_SNAPSHOT_ID
        )


def test_unknown_central_snapshot_is_rejected_without_fallback() -> None:
    with pytest.raises(ValueError, match="unsupported central H4 snapshot id"):
        screen.verify_central_snapshot("latest")


def test_serialized_bank_selection_matches_generated_bank() -> None:
    payload = json.loads(screen.DEFAULT_BANK.read_text(encoding="utf-8"))
    selected = screen.selected_candidates(payload)
    assert len(selected) == 4
