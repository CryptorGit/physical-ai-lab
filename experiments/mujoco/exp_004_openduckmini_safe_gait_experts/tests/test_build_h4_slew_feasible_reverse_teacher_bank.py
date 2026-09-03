from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import build_h4_slew_feasible_reverse_teacher_bank as teacher


def test_generated_bank_is_deterministic_and_all_candidates_pass() -> None:
    first = teacher.build_bank()
    second = teacher.build_bank()
    assert first["pure_validation"] == {
        "candidate_count": 12,
        "pass_count": 12,
        "all_passed": True,
        "validator": "validate_candidate",
    }
    assert first["ranking_candidate_ids"] == second["ranking_candidate_ids"]
    assert [candidate["candidate_id"] for candidate in first["candidates"]] == [
        candidate["candidate_id"] for candidate in second["candidates"]
    ]
    assert all(
        candidate["validation"]["passed"] for candidate in first["candidates"]
    )


def test_candidate_contract_has_signed_command_head_zero_and_exact_symmetry() -> None:
    candidate = teacher.build_bank()["candidates"][0]
    result = teacher.validate_candidate(candidate)
    assert result["passed"]
    assert result["checks"]["signed_reverse_command"]
    assert result["metrics"]["head_target_peak_rad"] == 0.0
    assert result["metrics"]["maximum_left_right_symmetry_error_rad"] <= 1.0e-10
    assert (
        result["metrics"]["maximum_cyclic_pre_guard_target_delta_rad"]
        <= teacher.MAXIMUM_TARGET_DELTA_RAD
    )


def test_validator_rejects_head_motion_and_symmetry_break() -> None:
    candidate = teacher.build_bank()["candidates"][0]
    head_motion = deepcopy(candidate)
    head_motion["target_table_rad"][0][5] = 0.001
    head_result = teacher.validate_candidate(head_motion)
    assert not head_result["passed"]
    assert "head_targets_exactly_zero" in head_result["failures"]

    asymmetric = deepcopy(candidate)
    asymmetric["target_table_rad"][0][0] += 0.001
    symmetry_result = teacher.validate_candidate(asymmetric)
    assert not symmetry_result["passed"]
    assert "half_cycle_left_right_symmetry" in symmetry_result["failures"]


def test_validator_rejects_wrong_cadence_and_large_cyclic_delta() -> None:
    candidate = teacher.build_bank()["candidates"][0]
    wrong_cadence = deepcopy(candidate)
    wrong_cadence["construction"]["cadence_hz"] = 2.1
    wrong_cadence["construction"]["phase_advance_bins_per_control"] = (
        2.1
        * teacher.RESAMPLED_PHASE_STEPS
        * teacher.CONTROL_FIRST_STARTUP_DT_S
    )
    cadence_result = teacher.validate_candidate(wrong_cadence)
    assert not cadence_result["passed"]
    assert "cadence_1p5_to_2p0_hz" in cadence_result["failures"]

    large_delta = deepcopy(candidate)
    left_index = teacher.LEG_ACTUATOR_INDICES[3]
    right_index = teacher.LEG_ACTUATOR_INDICES[8]
    lower = teacher.SAFE_JOINT_LIMITS["left_knee"][0] + teacher.LEG_TARGET_MARGIN_RAD
    upper = teacher.SAFE_JOINT_LIMITS["left_knee"][1] - teacher.LEG_TARGET_MARGIN_RAD
    large_delta["target_table_rad"][0][left_index] = lower
    large_delta["target_table_rad"][1][left_index] = upper
    half = teacher.RESAMPLED_PHASE_STEPS // 2
    large_delta["target_table_rad"][half][right_index] = lower
    large_delta["target_table_rad"][half + 1][right_index] = upper
    delta_result = teacher.validate_candidate(large_delta)
    assert not delta_result["passed"]
    assert "cyclic_pre_guard_target_delta_at_most_0p04_rad" in delta_result[
        "failures"
    ]


def test_validate_bank_recomputes_instead_of_trusting_serialized_validation() -> None:
    bank = teacher.build_bank()
    tampered = deepcopy(bank)
    tampered["candidates"][0]["target_table_rad"][0][6] = 0.01
    tampered["candidates"][0]["validation"]["passed"] = True
    result = teacher.validate_bank(tampered)
    assert not result["passed"]
    assert result["pass_count"] == 11
