from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import diagnose_h1_left_knee_cap_sweep as diagnostic


FINAL_ARTIFACT = (
    diagnostic.EXP_ROOT
    / "artifacts"
    / "h2_integrated_phase744_rate105_recovery0175_hold13_transition20x9_v1.json"
)
FINAL_ARTIFACT_SHA256 = (
    "bfaf052235e15262c34a794896e2c63a62bd1bd934998a77b7f6ea6c54009133"
)
STRAIGHT_PROFILE_SHA256 = (
    "0a3c0849124b397ca1cb60ae0b5f5783a2e545f1a03108846fa8c60cd5d8bb5b"
)


def _strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def test_integrated_candidate_and_transition_prefix_are_exact() -> None:
    candidate = diagnostic.CapCandidate(0.0175, 13)
    assert candidate.contract() == {
        "recovery_extra_upper_margin_rad": 0.0175,
        "recovery_hold_ticks": 13,
        "profile_extra_upper_margin_rad": 0.0125,
        "candidate_id": "139a35f31067a401",
        "left_knee_safe_upper_rad": 0.475534,
        "base_target_margin_rad": 0.05,
        "profile_upper_target_rad": 0.413034,
        "recovery_upper_target_rad": 0.408034,
        "profile_cap_held_at_h1_value": True,
        "recovery_hold_seconds": 0.26,
        "recovery_release": "instant_after_hold",
    }
    assert diagnostic.PHASE_ENTRY_INDICES == {
        "reverse": 7.0,
        "reverse_turn_left": 4.0,
        "reverse_turn_right": 4.0,
    }
    schedule = diagnostic.exact_formal_transition_prefix()
    assert tuple(case[0] for case in schedule) == diagnostic.PREFIX_SEGMENT_NAMES
    assert tuple(case[2] for case in schedule) == (
        5.0,
        30.0,
        5.0,
        30.0,
        5.0,
        30.0,
        5.0,
        30.0,
        5.0,
    )
    assert schedule[3][1] == (-0.05, 0.0, 0.0)
    assert schedule[5][1] == (-0.03, 0.0, 0.20)
    assert schedule[7][1] == (-0.04, 0.0, -0.20)


def test_screen_grid_keeps_profile_cap_decoupled() -> None:
    candidates = diagnostic.screening_candidates()
    assert {
        (item.recovery_extra_upper_margin_rad, item.recovery_hold_ticks)
        for item in candidates
    } == {
        (cap, hold)
        for cap in (0.01625, 0.0175, 0.0200, 0.0225)
        for hold in (13, 16)
    }
    assert all(
        item.profile_extra_upper_margin_rad == 0.0125 for item in candidates
    )
    with pytest.raises(ValueError, match="outside the diagnostic bound"):
        diagnostic.CapCandidate(0.051, 13).validate()
    with pytest.raises(ValueError, match="positive whole control tick"):
        diagnostic.CapCandidate(0.0175, 0).validate()


def test_process_local_runtime_substitution_is_restored() -> None:
    candidate = diagnostic.CapCandidate(0.0175, 13)
    names = (
        "BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD",
        "BACKWARD_EXIT_RECOVERY_HOLD_TICKS",
        "BACKWARD_EXIT_RECOVERY_HOLD_SECONDS",
        "BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD",
    )
    modules = (
        diagnostic.target_safety_runtime,
        diagnostic.routed_contract,
        diagnostic.central,
    )
    before = {
        (module.__name__, name): getattr(module, name)
        for module in modules
        for name in names
        if hasattr(module, name)
    }
    advance_before = diagnostic.central.advance_routed_phase
    with diagnostic._candidate_runtime_constants(candidate):
        assert (
            diagnostic.target_safety_runtime.BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
            == 0.0175
        )
        assert (
            diagnostic.routed_contract.BACKWARD_EXIT_RECOVERY_HOLD_TICKS == 13
        )
        assert (
            diagnostic.central.BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
            == 0.408034
        )
        assert diagnostic.central.advance_routed_phase is diagnostic.advance_integrated_phase
    after = {
        (module.__name__, name): getattr(module, name)
        for module in modules
        for name in names
        if hasattr(module, name)
    }
    assert after == before
    assert diagnostic.central.advance_routed_phase is advance_before


def test_final_integrated_artifact_is_immutable_and_all_gates_pass() -> None:
    assert diagnostic._sha256(FINAL_ARTIFACT) == FINAL_ARTIFACT_SHA256
    payload = _strict_json(FINAL_ARTIFACT)
    result = payload["qualification"]["result"]
    assert payload["status"] == "DIAGNOSTIC_20X9_PASS_NOT_ADOPTED"
    assert payload["hardware_deployment"] == "PROHIBITED"
    assert payload["qualification"]["passed"] is True
    assert payload["qualification"]["exact_scale_passed"] is True
    assert result["episode_count"] == result["passed_episode_count"] == 20
    assert result["segment_count"] == result["passed_segment_count"] == 180
    assert (
        result["expected_physics_substeps"]
        == result["completed_physics_substeps"]
        == result["audited_physics_substeps"]
        == result["contact_samples"]
        == 1_450_000
    )
    assert result["qpos_limit_violation_samples"] == 0
    assert result["maximum_qpos_excess_rad"] == 0.0
    assert result["fall_count"] == 0
    assert result["motion_contact_violation_segment_count"] == 0
    assert result["route_violation_segment_count"] == 0
    assert result["target_limit_margin_slew_violation_count"] == 0
    assert result["maximum_head_action_or_target_peak"] == 0.0
    assert result["phase_entry_event_count"] == 60
    assert result["recovery_exit_event_count"] == 60
    assert result["recovery_active_tick_count"] == 60 * 13
    assert result["minimum_left_knee_safe_upper_margin_rad"] > 0.0015
    assert payload["provenance"]["source_closure_unchanged"] is True
    assert {
        tuple(value)
        for value in payload["provenance"]["onnx_providers"].values()
    } == {
        ("CPUExecutionProvider",)
    }
    assert (
        payload["provenance"]["profiles"]["straight"]["sha256"]
        == STRAIGHT_PROFILE_SHA256
    )


def test_smaller_recovery_candidates_and_coupled_cap_are_rejected() -> None:
    payload = _strict_json(FINAL_ARTIFACT)
    comparisons = payload["selection"]["focus5_comparisons"]
    assert comparisons[0]["recovery_extra_upper_margin_rad"] == 0.01625
    assert comparisons[0]["hold_ticks"] == 13
    assert comparisons[0]["passed"] is False
    assert comparisons[0]["qpos_limit_violation_samples"] == 4
    assert comparisons[1]["recovery_extra_upper_margin_rad"] == 0.01625
    assert comparisons[1]["hold_ticks"] == 16
    assert comparisons[1]["passed"] is False
    assert comparisons[1]["qpos_limit_violation_samples"] == 4
    assert comparisons[2]["recovery_extra_upper_margin_rad"] == 0.0175
    assert comparisons[2]["hold_ticks"] == 13
    assert comparisons[2]["passed"] is True
    coupled = payload["selection"]["coupled_profile_cap_rejected"]
    assert coupled["candidate_extra_upper_margin_rad"] == 0.01625
    assert coupled["fall_count"] == 1
    assert coupled["qpos_limit_violation_samples"] == 0
