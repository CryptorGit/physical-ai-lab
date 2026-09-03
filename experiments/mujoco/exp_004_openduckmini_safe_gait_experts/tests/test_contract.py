from __future__ import annotations

import copy
import math

import pytest

from safe_gait_experts.contract import (
    ACTUATOR_JOINT_ORDER,
    BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256,
    BACKWARD_EXIT_RECOVERY_CONTRACT,
    BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT,
    BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
    BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
    BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
    BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
    BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256,
    BACKWARD_EXIT_RECOVERY_STATUS,
    BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256,
    BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_SELECTION_EVIDENCE_SHA256,
    BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_ADOPTION_EVIDENCE_SHA256,
    CONTROL_FIRST_STARTUP_DT_S,
    CONTRACT,
    HEAD_JOINTS,
    LEG_TARGET_MARGIN_RAD,
    RESET_NOISE_MARGIN_RAD,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
    TARGET_SLEW_LIMIT_RAD_PER_S,
    validate_contract,
)


def test_contract_is_complete_and_hardware_prohibited() -> None:
    assert len(ACTUATOR_JOINT_ORDER) == 14
    assert len(set(ACTUATOR_JOINT_ORDER)) == 14
    assert HEAD_JOINTS == {
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
    }
    assert CONTRACT["head_lock"]["enabled"] is True
    assert LEG_TARGET_MARGIN_RAD == 0.050
    assert TARGET_SLEW_LIMIT_RAD_PER_S == 2.0
    assert RESET_NOISE_MARGIN_RAD == 0.005
    assert CONTROL_FIRST_STARTUP_DT_S == 0.02
    assert SAFE_INIT_POS["left_knee"] == pytest.approx(
        SAFE_JOINT_LIMITS["left_knee"][1] - RESET_NOISE_MARGIN_RAD
    )
    assert CONTRACT["target_safety"]["head_target_rad"] == 0.0
    startup = CONTRACT["target_safety"]["control_first_startup"]
    assert startup["required"] is True
    assert startup["control_dt_seconds"] == 0.02
    assert startup["physics_steps_before_guarded_control"] == 0
    assert startup["guard_steps_before_first_physics"] == 1
    assert startup["slew_applications_per_tick"] == 1
    assert startup["maximum_leg_target_delta_per_tick_rad"] == 0.04
    assert startup["desired_targets"] == "first_command_policy_targets"
    assert startup["home_only_precharge"] == "PROHIBITED"
    assert startup["normal_tick_order"][:2] == [
        "observe_route_and_infer_policy",
        "compose_desired_joint_targets",
    ]
    assert startup["required_order"][-2:] == [
        "first_physics_step",
        "first_post_step_sensor_sample",
    ]
    assert CONTRACT["deployment"]["hardware_status"] == "PROHIBITED"
    assert len(CONTRACT["deployment"]["required_gates"]) >= 5


def test_backward_exit_recovery_is_frozen_h3_simulation_only_adoption() -> None:
    recovery = BACKWARD_EXIT_RECOVERY_CONTRACT

    assert BACKWARD_EXIT_RECOVERY_STATUS == "ADOPTED_SIMULATION_ONLY"
    assert BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT is True
    assert recovery["formal_candidate_only"] is False
    assert recovery["diagnostic_unadopted_only"] is False
    assert recovery["adoption_eligible"] is True
    assert recovery["simulation_acceptance_eligible"] is True
    assert recovery["safety_component_only"] is False
    assert recovery["safety_component_evidence_is_safety_only"] is True
    assert recovery["fast_exit_safety_passed"] is True
    assert recovery["combined_5x15_required"] is False
    assert recovery["combined_5x15_passed"] is True
    assert recovery["requires_formal_20x30_requalification"] is False
    assert recovery["safety_component_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256
    )
    assert recovery["candidate_selection_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256
    )
    assert recovery["superseded_h2_selection_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_SELECTION_EVIDENCE_SHA256
    )
    assert recovery["superseded_h2_adoption_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_ADOPTION_EVIDENCE_SHA256
    )
    assert recovery["adoption_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256
    )
    assert recovery["hardware_deployment"] == "PROHIBITED"
    assert BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD == 0.0225
    assert BACKWARD_EXIT_RECOVERY_HOLD_TICKS == 13
    assert BACKWARD_EXIT_RECOVERY_HOLD_SECONDS == 0.26
    assert BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD == 0.403034
    assert recovery["joint_index"] == ACTUATOR_JOINT_ORDER.index("left_knee")
    assert recovery["final_guard_calls_per_tick"] == 1
    assert recovery["upper_target_rad"] == pytest.approx(
        SAFE_JOINT_LIMITS["left_knee"][1]
        - LEG_TARGET_MARGIN_RAD
        - recovery["extra_upper_margin_rad"]
    )
    assert recovery["hold_seconds"] == pytest.approx(
        recovery["hold_control_ticks"] * recovery["control_dt_seconds"]
    )


def test_offsets_are_recomputed_from_authoritative_raw_counts() -> None:
    calibration = CONTRACT["calibration"]
    encoder_steps = CONTRACT["provenance"]["encoder_steps"]
    for name in ACTUATOR_JOINT_ORDER:
        raw = calibration["zero_raw"][name]
        expected = (raw - encoder_steps / 2) * (2 * math.pi / encoder_steps)
        assert calibration["zero_offset_rad"][name] == pytest.approx(
            expected, abs=1e-12
        )
    assert calibration["joint_direction"]["left_knee"] == -1.0
    assert calibration["joint_direction"]["left_ankle"] == -1.0
    assert calibration["joint_direction"]["right_knee"] == 1.0
    assert calibration["joint_direction"]["right_ankle"] == 1.0


def test_safe_init_is_inside_every_leg_limit_and_head_is_zero() -> None:
    for name, (lower, upper) in SAFE_JOINT_LIMITS.items():
        assert lower <= SAFE_INIT_POS[name] <= upper
    for name in HEAD_JOINTS:
        assert SAFE_INIT_POS[name] == 0.0


def test_validator_rejects_hardware_promotion_and_head_noise() -> None:
    promoted = copy.deepcopy(CONTRACT)
    promoted["deployment"]["hardware_status"] = "ALLOWED"
    with pytest.raises(ValueError, match="deployment"):
        validate_contract(promoted)

    noisy_head = copy.deepcopy(CONTRACT)
    noisy_head["head_lock"]["qpos_noise_scale_rad"] = 0.01
    with pytest.raises(ValueError, match="exactly zero"):
        validate_contract(noisy_head)

    noisy_scale = copy.deepcopy(CONTRACT)
    noisy_scale["qpos_noise_scale_rad"]["head"] = 0.01
    with pytest.raises(ValueError, match="exactly zero"):
        validate_contract(noisy_scale)

    missing_margin = copy.deepcopy(CONTRACT)
    missing_margin["target_safety"]["leg_margin_rad"] = 0.0
    with pytest.raises(ValueError, match="margin"):
        validate_contract(missing_margin)

    missing_slew = copy.deepcopy(CONTRACT)
    missing_slew["target_safety"]["target_slew_limit_rad_per_s"] = 0.0
    with pytest.raises(ValueError, match="slew"):
        validate_contract(missing_slew)

    missing_reset_margin = copy.deepcopy(CONTRACT)
    missing_reset_margin["target_safety"]["reset_qpos"]["noise_margin_rad"] = 0.0
    with pytest.raises(ValueError, match="reset_qpos"):
        validate_contract(missing_reset_margin)

    physics_first = copy.deepcopy(CONTRACT)
    physics_first["target_safety"]["control_first_startup"][
        "physics_steps_allowed_before_control"
    ] = 1
    with pytest.raises(ValueError, match="no physics steps"):
        validate_contract(physics_first)


def test_validator_freezes_formal_runtime_target_and_startup_contract() -> None:
    wrong_margin = copy.deepcopy(CONTRACT)
    wrong_margin["target_safety"]["leg_margin_rad"] = 0.04
    with pytest.raises(ValueError, match="exactly 0.050"):
        validate_contract(wrong_margin)

    wrong_slew = copy.deepcopy(CONTRACT)
    wrong_slew["target_safety"]["target_slew_limit_rad_per_s"] = 1.0
    with pytest.raises(ValueError, match="exactly 2.0"):
        validate_contract(wrong_slew)

    wrong_reset = copy.deepcopy(CONTRACT)
    wrong_reset["target_safety"]["reset_qpos"]["noise_margin_rad"] = 0.004
    with pytest.raises(ValueError, match="exactly 0.005"):
        validate_contract(wrong_reset)

    startup_after_physics = copy.deepcopy(CONTRACT)
    startup_after_physics["target_safety"]["control_first_startup"][
        "physics_steps_before_guarded_control"
    ] = 1
    with pytest.raises(ValueError, match="precede all physics"):
        validate_contract(startup_after_physics)

    wrong_startup_dt = copy.deepcopy(CONTRACT)
    wrong_startup_dt["target_safety"]["control_first_startup"][
        "control_dt_seconds"
    ] = 0.01
    with pytest.raises(ValueError, match="exactly 0.02"):
        validate_contract(wrong_startup_dt)

    home_precharge = copy.deepcopy(CONTRACT)
    home_precharge["target_safety"]["control_first_startup"][
        "home_only_precharge"
    ] = "ALLOWED"
    with pytest.raises(ValueError, match="home-only"):
        validate_contract(home_precharge)

    double_slew = copy.deepcopy(CONTRACT)
    double_slew["target_safety"]["control_first_startup"][
        "slew_applications_per_tick"
    ] = 2
    with pytest.raises(ValueError, match="exactly once"):
        validate_contract(double_slew)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "status",
            "FORMAL_CANDIDATE_H3_5X15_PASSED_PENDING_20X30",
            "adoption status",
        ),
        ("enabled_by_default", False, "enabled for H3"),
        ("formal_candidate_only", True, "cannot remain candidate-only"),
        ("diagnostic_unadopted_only", True, "diagnostic history path"),
        ("adoption_eligible", False, "adoption eligibility"),
        (
            "simulation_acceptance_eligible",
            False,
            "simulation eligibility",
        ),
        (
            "candidate_selection_evidence_sha256",
            "0" * 64,
            "candidate evidence",
        ),
        (
            "superseded_h2_selection_evidence_sha256",
            "0" * 64,
            "superseded H2 selection evidence",
        ),
        (
            "safety_component_evidence_sha256",
            "0" * 64,
            "safety-component evidence",
        ),
        (
            "superseded_h2_adoption_evidence_sha256",
            "0" * 64,
            "superseded H2 adoption evidence",
        ),
        ("adoption_evidence_sha256", "0" * 64, "H3 adoption evidence"),
        ("safety_component_only", True, "not safety-component-only"),
        (
            "safety_component_evidence_is_safety_only",
            False,
            "safety component",
        ),
        ("fast_exit_safety_passed", False, "fast-exit safety"),
        ("combined_5x15_required", True, "another combined 5x15"),
        ("combined_5x15_passed", False, "combined 5x15 pass"),
        ("extra_upper_margin_rad", 0.0, "extra upper margin"),
        ("hold_control_ticks", 12, "hold ticks"),
        ("hold_seconds", 0.24, "hold seconds"),
        ("upper_target_rad", 0.425534, "upper target"),
        ("hardware_deployment", "ALLOWED", "hardware_deployment"),
    ],
)
def test_validator_rejects_backward_exit_recovery_demotion_or_drift(
    field: str, value: object, message: str
) -> None:
    mutated = copy.deepcopy(CONTRACT)
    mutated["target_safety"]["backward_exit_recovery"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_contract(mutated)
