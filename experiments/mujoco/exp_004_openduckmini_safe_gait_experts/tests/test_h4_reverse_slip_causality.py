from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import diagnose_h4_reverse_slip_causality as causality


EXP_ROOT = Path(__file__).resolve().parents[1]


def test_summary_and_correlation_fail_closed() -> None:
    summary = causality._summary([3.0, 4.0])
    assert summary == pytest.approx(
        {
            "sample_count": 2,
            "mean": 3.5,
            "rms": (12.5) ** 0.5,
            "p95": 3.95,
            "maximum": 4.0,
        }
    )
    assert causality._correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert causality._correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_v2_causal_artifact_uses_runtime_phase_period() -> None:
    path = (
        EXP_ROOT
        / "artifacts"
        / "h4_reverse_slip_causal_decomposition_baseline_vs_sag114_abs060_worst_median_v2.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scope"] == "four requested causal replays; not a candidate grid"
    assert len(payload["records"]) == 2
    for record in payload["records"]:
        assert len(record["runs"]) == 2
        for run in record["runs"]:
            assert run["synchronized_causal_probe"]["profile_phase_steps"] == 27


def test_retraining_spec_is_fail_closed_and_signed() -> None:
    path = (
        EXP_ROOT
        / "artifacts"
        / "h4_reverse_retraining_minimum_spec_from_slip_causality_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "RETRAINING_REQUIRED_NO_PROMOTABLE_CANDIDATE"
    assert payload["hardware_deployment"] == "PROHIBITED"
    assert payload["scope"]["physical_command_mps_radps"] == [-0.05, 0.0, 0.0]
    assert payload["scope"]["candidate_or_model_created"] is False
    teacher = payload["minimum_retraining_specification"]["teacher_and_action_contract"]
    assert teacher["maximum_pre_guard_delta_rad_per_joint_per_control"] == pytest.approx(0.04)
    qualification = payload["qualification_plan"]
    assert qualification["stage_order"] == [
        "failure3_direct_6s",
        "five_suites_by_fifteen_episodes",
        "twenty_suites_by_thirty_episodes",
    ]
    assert qualification["mandatory_contact_sensitivity_windows_ms"] == [10, 20, 30, 40]
    assert payload["decision"]["adoption"] == "BLOCKED"
