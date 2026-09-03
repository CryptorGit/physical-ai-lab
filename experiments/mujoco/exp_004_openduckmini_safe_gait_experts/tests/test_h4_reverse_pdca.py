from __future__ import annotations

import numpy as np
import pytest

from scripts import diagnose_h4_reverse_pdca as pdca


def test_exact_h3_baseline_candidate_is_stable() -> None:
    candidate = pdca.Candidate("baseline")
    assert candidate.phase_entry_preincrement == 7.0
    assert candidate.phase_rate_factor == 1.0
    assert candidate.backward_residual_scale == 0.0
    assert candidate.policy_observation_command == (-0.05, 0.0, 0.0)
    assert candidate.upper_cap_extras_rad[3] == 0.0125
    assert candidate.candidate_id == pdca.Candidate("baseline").candidate_id


def test_candidate_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown candidate fields"):
        pdca.Candidate.from_mapping({"name": "x", "mystery": 1})
    with pytest.raises(ValueError, match="residual scale"):
        pdca.Candidate.from_mapping({"name": "x", "backward_residual_scale": 0.26})
    with pytest.raises(ValueError, match="10 finite"):
        pdca.Candidate.from_mapping({"name": "x", "amplitude_factors": [1.0] * 9})


def test_heading_window_uses_unwrapped_six_second_changes() -> None:
    yaws = np.linspace(3.10, 3.25, 7)
    wrapped = (yaws + np.pi) % (2.0 * np.pi) - np.pi
    assert pdca.heading_window_error_rad(wrapped, 6) == pytest.approx(0.15)
    assert pdca.heading_window_error_rad([0.0, 0.04, 0.08], 10) == pytest.approx(0.08)


def test_stance_slip_excludes_touchdown_from_continuous_metric() -> None:
    samples = [
        {
            "time": 0.0,
            "contacts": np.asarray([False, True]),
            "left_foot_xy": np.asarray([0.0, 0.0]),
            "right_foot_xy": np.asarray([0.0, 0.0]),
        },
        {
            "time": 0.002,
            "contacts": np.asarray([True, True]),
            "left_foot_xy": np.asarray([0.002, 0.0]),
            "right_foot_xy": np.asarray([0.0002, 0.0]),
        },
        {
            "time": 0.004,
            "contacts": np.asarray([True, True]),
            "left_foot_xy": np.asarray([0.0022, 0.0]),
            "right_foot_xy": np.asarray([0.0004, 0.0]),
        },
    ]
    result = pdca.stance_slip_metrics(samples, 0.002)
    assert result["current_contact_including_touchdown"]["sample_count"] == 4
    assert result["continuous_stance"]["sample_count"] == 3
    assert result["continuous_stance"]["maximum_mps"] == pytest.approx(0.1)


def test_cross_gate_resolves_to_twenty_percent_for_reverse_endpoint() -> None:
    assert min(pdca.CROSS_ABSOLUTE_LIMIT_MPS, pdca.CROSS_FRACTION_LIMIT * 0.05) == pytest.approx(0.01)
