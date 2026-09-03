from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import safe_gait_experts.routed_evaluation as routed_evaluation_module

from safe_gait_experts.contract import (
    ACTUATOR_JOINT_ORDER,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
)
from safe_gait_experts.routed_evaluation import (
    BACKWARD_FAMILY_EXPERTS,
    BASE_V22_POLICY_SHA256,
    COMPOUND_CASES,
    FORMAL_ADOPTION_EVIDENCE_PATH,
    FORMAL_ADOPTION_EVIDENCE_SHA256,
    FORMAL_ADOPTION_EVIDENCE_SHA256_ALLOWLIST,
    FORMAL_CANDIDATE_PROFILE_PATHS,
    FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES,
    FORMAL_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS,
    FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256,
    FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256_ALLOWLIST,
    FORMAL_CANDIDATE_STATUS,
    FORMAL_H2_ADOPTED_REVERSE_PROFILE_SHA256_ALLOWLISTS,
    H2_5X15_SELECTION_EVIDENCE_SHA256,
    H2_5X15_SELECTION_STATUS,
    H2_COMPONENT_SELECTION_EVIDENCE_SHA256,
    H2_SUPERSEDED_ADOPTION_EVIDENCE_PATH,
    H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256,
    H2_SUPERSEDED_ADOPTION_STATUS,
    H3_CANDIDATE_SELECTION_STATUS,
    H3_FAST_EXIT_EXPECTED_MOTION_FAILURES,
    H3_FAST_EXIT_SAFETY_STATUS,
    H3_FAST_EXIT_SAFETY_EVIDENCE_PATH,
    H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
    H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256_ALLOWLIST,
    FORMAL_POLICY_SHA256_ALLOWLIST,
    FORMAL_REVERSE_ADOPTION_STATUSES,
    FORMAL_REVERSE_COMMAND_CASE_ADOPTION_EVIDENCE_SHA256_ALLOWLISTS,
    FORMAL_REVERSE_COMMAND_CASE_SAFETY_EVIDENCE_SHA256_ALLOWLISTS,
    FORMAL_REVERSE_EVIDENCE_SHA256_ALLOWLISTS,
    FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES,
    CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
    DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_SHA256,
    DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_SOURCE_REVERSE_ENDPOINT_MPS,
    DIAGNOSTIC_REVERSE_PHASE_ENTRY_SOURCE_REVERSE_ENDPOINT_MPS,
    DIAGNOSTIC_REVERSE_TURN_PROFILE_SHA256,
    FROZEN_GENERATED_DEPENDENCY_ROOT_SHA256,
    FROZEN_GENERATED_ROOT,
    FROZEN_RUNTIME_BINARY_SHA256,
    FROZEN_RUNTIME_DEPENDENCY_ROOT_SHA256,
    FROZEN_RUNTIME_VERSIONS,
    POLICY_COMMAND_DIAGNOSTIC_CASES,
    PhysicsSubstepAudit,
    PRIMITIVE_CASES,
    REQUIRED_POLICY_ROLES,
    REJECTED_POLICY_COMMAND_DIAGNOSTIC_CASES,
    REVERSE_V1_ADOPTION_STATUS,
    REVERSE_V1_MEASURED_FORWARD_VELOCITY_MPS,
    SafetyAudit,
    TRANSITION_CASES,
    advance_routed_phase,
    audit_control_first_startup,
    audit_reset_qpos,
    backward_exit_recovery_state_acceptance,
    blend_and_mask_actions,
    build_target_envelope,
    canonical_policy_role,
    command_case_validation_gate,
    compute_motion_metrics,
    capture_runtime_source_dependency_closure,
    dependency_closure_root_sha256,
    derive_reverse_profile_adoption,
    discover_mjcf_dependency_closure,
    hardware_gate,
    parse_policy_assignments,
    policy_yaw_observation_offset,
    resolve_policy_observation_command,
    segment_acceptance,
    suite_acceptance,
    summarize_backward_exit_recovery_steps,
    transition_schedule,
    validate_adopted_reverse_profiles,
    validate_diagnostic_unadopted_reverse_profile,
    validate_diagnostic_backward_exit_recovery_evidence,
    validate_diagnostic_backward_exit_recovery_execution_bundle,
    validate_diagnostic_unadopted_reverse_turn_profile,
    validate_diagnostic_reverse_entry_phase_indices,
    validate_diagnostic_reverse_phase_entry_evidence,
    validate_exact_generated_assets,
    validate_formal_candidate_reverse_profiles,
    validate_formal_candidate_selection_evidence,
    validate_formal_adoption_evidence,
    validate_h3_fast_exit_safety_evidence,
    validate_frozen_runtime_source_dependencies,
    validate_policy_provenance,
    validate_reverse_profile_schema,
    validate_runtime_versions,
    validate_superseded_h2_adoption_evidence,
)
from router import SafeGaitRouter
from scripts.evaluate_routed_transitions import (
    DiagnosticTargetSafetyGuard,
    RELEASE_QUALIFICATION_CONFIGURATION,
    RoutedPolicyBank,
    RoutedSimulator,
    apply_control_first_startup,
    apply_guarded_control_then_step_physics,
    classify_evaluation_scale,
    parse_args,
    synchronize_telemetry_data,
)
from target_safety import (
    BackwardExitRecovery,
    FinalTargetSafetyGuard,
    RUNTIME_TARGET_SAFETY_MARGIN_RAD,
    apply_final_target_safety,
)


def _assignments() -> list[str]:
    return [f"{role}=policies/{role}.onnx" for role in REQUIRED_POLICY_ROLES]


def _safe_segment(name: str = "forward") -> dict:
    command = np.asarray((0.10, 0.0, 0.0))
    target = apply_final_target_safety(
        np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER]),
        SAFE_JOINT_LIMITS,
    )
    raw_action = np.zeros(14)
    raw_action[5:9] = (4.0, -3.0, 2.0, -1.0)
    applied = raw_action.copy()
    applied[5:9] = 0.0
    audit = SafetyAudit()
    audit.update(
        raw_policy_action=raw_action,
        applied_action=applied,
        preclip_targets=target,
        margin_clipped_targets=target,
        applied_targets=target,
        previous_applied_targets=target,
        joint_qpos=target,
        control_dt=0.02,
    )
    substep_audit = PhysicsSubstepAudit()
    substep_audit.update(
        joint_qpos=target,
        full_qpos=target,
        full_qvel=np.zeros(14),
        height_m=0.18,
        upright=0.98,
        feet_contacts=(True, False),
    )
    substep_payload = substep_audit.to_dict()
    metrics = compute_motion_metrics(
        command,
        np.repeat(command[None, :], 8, axis=0),
        np.zeros(8),
        displacement_xyz=(0.5, 0.0, 0.0),
        minimum_height_m=0.18,
        minimum_upright=0.98,
        mean_effective_command=command,
        single_support_rate=substep_payload["single_support_rate"],
        flight_rate=substep_payload["flight_rate"],
        contact_sample_count=substep_payload["contact_sample_count"],
        contact_rate_sample_source="physics_substeps_after_mj_step",
    )
    recovery = BackwardExitRecovery(SAFE_JOINT_LIMITS, enabled=False)
    _, recovery_step = recovery.compose(
        target, backward_feedforward_active=False
    )
    recovery_segment_audit = summarize_backward_exit_recovery_steps(
        [recovery_step], enabled=False, expected_sample_count=1
    )
    return {
        "name": name,
        "command": command.tolist(),
        "expected_expert": "forward",
        "expected_policy_role": "forward",
        "completed": True,
        "completed_physics_substeps": 1,
        "expected_physics_substeps": 1,
        "fell": False,
        "metrics": metrics,
        "safety_audit": audit.to_dict(),
        "physics_substep_audit": substep_payload,
        "backward_exit_recovery_audit": recovery_segment_audit,
        "routing": {
            "command_clip_events": 0,
            "prohibited_expert_steps": 0,
            "steady_state_steps": 8,
            "steady_state_routed_expert_steps": {"forward": 8},
            "steady_state_policy_role_steps": {"forward": 8},
            "steady_state_prohibited_expert_steps": 0,
            "atomic_endpoint_required": False,
            "atomic_endpoint_mismatch_steps": 0,
            "reverse_entry_phase": {
                "enabled": False,
                "event_count": 0,
                "events": [],
                "passed": True,
            },
        },
    }


def _disabled_recovery_state_audit(control_ticks: int = 1) -> dict:
    recovery = BackwardExitRecovery(SAFE_JOINT_LIMITS, enabled=False)
    targets = np.asarray(
        [SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER]
    )
    for _ in range(control_ticks):
        recovery.compose(targets, backward_feedforward_active=False)
    audit = recovery.audit()
    audit.update(
        {
            "final_guard_call_count": control_ticks,
            "reset_clear_on_schedule_start": True,
            "composition_before_final_guard": True,
            "final_guard_calls_per_control_tick": 1,
        }
    )
    return audit


def test_strict_gait_quality_rejects_missing_and_partial_claims() -> None:
    segment = _safe_segment()
    assert segment_acceptance(segment)["passed"] is True
    missing = segment_acceptance(segment, require_gait_quality=True)
    assert missing["passed"] is False
    assert missing["checks"]["gait_quality_present"] is False

    segment["gait_quality_metrics"] = {
        "measurement_complete": True,
        "sample_count": 2,
        "contact_state_source": "normal_force_schmitt",
        "contact_force_sample_count": 2,
        "stance_slip_measurement_source": (
            "force_weighted_contact_point_jacobian"
        ),
        "contact_velocity_sample_count": 1,
    }
    segment["gait_quality_acceptance"] = {"passed": True}
    partial = segment_acceptance(segment, require_gait_quality=True)
    assert partial["passed"] is False
    assert partial["checks"]["gait_quality_acceptance_rederived"] is False
    assert partial["checks"]["gait_quality_acceptance_untampered"] is False
    assert partial["checks"]["strict_gait_quality"] is False

    segment["gait_quality_acceptance"] = {"passed": False}
    rejected = segment_acceptance(segment, require_gait_quality=True)
    assert rejected["passed"] is False
    assert rejected["checks"]["strict_gait_quality"] is False


def test_routed_acceptance_rederives_and_rejects_a_tampered_quality_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _safe_segment()
    segment.update(
        requested_seconds=0.002,
        completed_seconds=0.002,
        physics_timestep_s=0.002,
    )
    segment["gait_quality_metrics"] = {
        "measurement_complete": True,
        "sample_count": 2,
        "duration_s": 0.002,
        "physics_timestep_s": 0.002,
        "maximum_timestep_error_s": 0.0,
        "contact_state_source": "normal_force_schmitt",
        "contact_force_sample_count": 2,
        "stance_slip_measurement_source": (
            "force_weighted_contact_point_jacobian"
        ),
        "contact_velocity_payload_sample_count": 2,
        "contact_velocity_sample_count": 1,
        "left_contact_velocity_sample_count": 1,
        "right_contact_velocity_sample_count": 0,
        "trunk_pose_measurement_source": (
            "mujoco_shadow_xpos_xmat_after_mj_forward"
        ),
        "trunk_yaw_sample_count": 2,
    }
    rederived = {
        "passed": True,
        "checks": {"frozen_check": True},
        "applicable": {"frozen_check": True},
        "failures": [],
    }
    monkeypatch.setattr(
        routed_evaluation_module,
        "rederive_gait_quality_acceptance",
        lambda metrics: SimpleNamespace(as_dict=lambda: rederived),
    )
    segment["gait_quality_acceptance"] = copy.deepcopy(rederived)

    accepted = segment_acceptance(segment, require_gait_quality=True)
    assert accepted["passed"] is True
    assert accepted["failures"] == []
    assert accepted["rederived_gait_quality_acceptance"] == rederived
    assert accepted["checks"]["gait_quality_acceptance_rederived"] is True
    assert accepted["checks"]["gait_quality_acceptance_untampered"] is True

    segment["gait_quality_metrics"]["duration_s"] = 0.004
    bad_duration = segment_acceptance(segment, require_gait_quality=True)
    assert bad_duration["checks"]["gait_quality_duration_exact"] is False
    segment["gait_quality_metrics"]["duration_s"] = 0.002

    segment["gait_quality_metrics"]["trunk_pose_measurement_source"] = "claimed"
    bad_source = segment_acceptance(segment, require_gait_quality=True)
    assert bad_source["checks"]["gait_quality_trunk_pose_source"] is False
    segment["gait_quality_metrics"]["trunk_pose_measurement_source"] = (
        "mujoco_shadow_xpos_xmat_after_mj_forward"
    )

    segment["gait_quality_acceptance"]["passed"] = False
    tampered = segment_acceptance(segment, require_gait_quality=True)
    assert tampered["passed"] is False
    assert tampered["checks"]["gait_quality_acceptance_untampered"] is False
    assert "gait_quality_acceptance_untampered" in tampered["failures"]


def test_telemetry_snapshot_copies_then_forwards_without_touching_live_data() -> None:
    events: list[str] = []
    source = SimpleNamespace(state=3.0, derived=-1.0)
    telemetry = SimpleNamespace(state=0.0, derived=0.0)

    class FakeMujoco:
        @staticmethod
        def mj_copyData(destination, model, live) -> None:
            del model
            events.append("copy")
            destination.state = live.state

        @staticmethod
        def mj_forward(model, destination) -> None:
            del model
            events.append("forward")
            destination.derived = destination.state * 2.0

    result = synchronize_telemetry_data(FakeMujoco, object(), source, telemetry)
    assert result is telemetry
    assert events == ["copy", "forward"]
    assert telemetry.derived == 6.0
    assert source.derived == -1.0


def test_suite_definitions_use_explicit_adopted_h3_mappings() -> None:
    assert len(PRIMITIVE_CASES) == 7
    assert {case.name for case in PRIMITIVE_CASES} == {
        "stand",
        "forward",
        "reverse",
        "lateral_left",
        "lateral_right",
        "yaw_left",
        "yaw_right",
    }
    reverse = next(case for case in PRIMITIVE_CASES if case.name == "reverse")
    assert reverse.command == (-0.050, 0.0, 0.0)
    assert reverse.validation_status == FORMAL_CANDIDATE_STATUS
    assert reverse.validation_status == "ADOPTED_SIMULATION_ONLY"
    assert reverse.validation_evidence_sha256 == (
        FORMAL_ADOPTION_EVIDENCE_SHA256
    )
    assert reverse.safety_evidence_sha256 == H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
    assert [
        (case.name, case.command, case.policy_observation_command)
        for case in PRIMITIVE_CASES
    ] == [
        ("stand", (0.0, 0.0, 0.0), None),
        ("forward", (0.05, 0.0, 0.0), (0.10, 0.0, 0.0)),
        ("reverse", (-0.050, 0.0, 0.0), None),
        ("lateral_left", (0.0, 0.06, 0.0), (0.0, 0.10, 0.0)),
        ("lateral_right", (0.0, -0.06, 0.0), (0.0, -0.10, 0.0)),
        ("yaw_left", (0.0, 0.0, 0.30), (0.0, -0.06, 0.60)),
        ("yaw_right", (0.0, 0.0, -0.30), (0.0, 0.0, -0.80)),
    ]
    assert [
        (case.name, case.command, case.policy_observation_command)
        for case in COMPOUND_CASES
    ] == [
        ("forward_turn_left", (0.04, 0.0, 0.30), (0.08, 0.0, 0.30)),
        ("forward_turn_right", (0.04, 0.0, -0.22), (0.08, 0.0, -0.45)),
        (
            "forward_lateral_left_turn",
            (0.04, 0.05, 0.17),
            (0.06, 0.05, 0.20),
        ),
        (
            "forward_lateral_right_turn",
            (0.04, -0.03, -0.15),
            (0.06, -0.05, -0.35),
        ),
        ("reverse_turn_left", (-0.03, 0.0, 0.20), None),
        ("reverse_turn_right", (-0.04, 0.0, -0.20), None),
    ]
    assert all(not (case.command[0] < 0.0 and case.command[1] != 0.0) for case in COMPOUND_CASES)

    assert [
        (case.command, case.policy_observation_command)
        for case in POLICY_COMMAND_DIAGNOSTIC_CASES
    ] == [
        ((0.05, 0.0, 0.0), (0.10, 0.0, 0.0)),
        ((0.0, 0.06, 0.0), (0.0, 0.10, 0.0)),
        ((0.0, -0.06, 0.0), (0.0, -0.10, 0.0)),
        ((0.0, 0.0, 0.30), (0.0, -0.06, 0.60)),
        ((0.0, 0.0, -0.30), (0.0, 0.0, -0.90)),
        ((0.0, 0.0, -0.25), (0.0, 0.0, -0.60)),
        ((0.0, 0.0, -0.25), (0.0, 0.0, -0.70)),
        ((0.0, 0.0, -0.30), (0.0, 0.0, -0.80)),
    ]
    assert REJECTED_POLICY_COMMAND_DIAGNOSTIC_CASES == {
        "diagnostic_yaw_right_rejected_policy_minus_090"
    }


def test_every_command_case_has_explicit_route_contract_and_status_gate() -> None:
    cases = (
        *PRIMITIVE_CASES,
        *COMPOUND_CASES,
        *POLICY_COMMAND_DIAGNOSTIC_CASES,
        *TRANSITION_CASES,
    )
    assert all(case.expected_expert for case in cases)
    assert all(case.expected_policy_role for case in cases)
    assert all(
        canonical_policy_role(case.expected_expert) == case.expected_policy_role
        for case in cases
    )

    formal_gate = command_case_validation_gate(
        (*PRIMITIVE_CASES, *COMPOUND_CASES, *TRANSITION_CASES)
    )
    assert formal_gate["passed"] is True
    assert formal_gate["safety_component_passed"] is True
    assert formal_gate["adoption_evidence_passed"] is True
    assert formal_gate["nonadoptable_case_count"] == 0
    assert formal_gate["nonadoptable_cases"] == []
    assert formal_gate["reverse_adoption_evidence_case_count"] == 6
    assert formal_gate["reverse_adoption_evidence_failure_count"] == 0
    assert formal_gate["reverse_safety_component_evidence_case_count"] == 6
    assert formal_gate["reverse_safety_component_evidence_failure_count"] == 0
    assert all(
        item["status"] == FORMAL_CANDIDATE_STATUS
        and item["evidence_sha256"] == FORMAL_ADOPTION_EVIDENCE_SHA256
        and item["evidence_hash_allowlisted"] is True
        for item in formal_gate["reverse_adoption_evidence_bindings"].values()
    )
    assert all(
        item["evidence_sha256"] == H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        and item["evidence_hash_allowlisted"] is True
        for item in formal_gate[
            "reverse_safety_component_evidence_bindings"
        ].values()
    )


def test_transition_durations_are_configurable_and_stand_specific() -> None:
    schedule = transition_schedule(6.5, 2.5)
    assert [scheduled[0] for scheduled in schedule] == [
        case.name for case in TRANSITION_CASES
    ]
    for scheduled, case in zip(schedule, TRANSITION_CASES, strict=True):
        (
            _,
            command,
            duration,
            policy_command,
            expected_expert,
            expected_policy_role,
        ) = scheduled
        assert duration == (2.5 if np.allclose(command, 0.0) else 6.5)
        assert command == case.command
        assert policy_command == case.policy_observation_command
        assert expected_expert == case.expected_expert
        assert expected_policy_role == case.expected_policy_role
    transition_by_name = {case.name: case for case in TRANSITION_CASES}
    for endpoint in (*PRIMITIVE_CASES[1:], *COMPOUND_CASES):
        transition = transition_by_name[f"transition_{endpoint.name}"]
        assert transition.command == endpoint.command
        assert (
            transition.policy_observation_command
            == endpoint.policy_observation_command
        )
        assert transition.validation_status == endpoint.validation_status
    # Every expert/sign change returns to stand without resetting simulation.
    assert np.allclose(schedule[0][1], 0.0)
    assert np.allclose(schedule[-1][1], 0.0)
    moving_indices = [
        index
        for index, (_, command, _, _, _, _) in enumerate(schedule)
        if not np.allclose(command, 0.0)
    ]
    for previous, current in zip(moving_indices, moving_indices[1:]):
        assert any(
            np.allclose(schedule[index][1], 0.0)
            for index in range(previous + 1, current)
        )
    with pytest.raises(ValueError):
        transition_schedule(0.0, 2.0)


def test_policy_assignments_require_every_role_and_reject_old_omnidirectional_names() -> None:
    policies = parse_policy_assignments(_assignments())
    assert tuple(policies) == REQUIRED_POLICY_ROLES
    with pytest.raises(ValueError, match="missing"):
        parse_policy_assignments(_assignments()[:-1])
    with pytest.raises(ValueError, match="prohibited"):
        parse_policy_assignments([*_assignments(), "v59=bad.onnx"])
    assert canonical_policy_role("reverse_turn_left") == "compound"
    assert canonical_policy_role("reverse_turn_right") == "compound"


def test_action_blend_happens_before_the_exact_head_mask() -> None:
    old = np.arange(14, dtype=np.float64)
    new = -old
    raw, applied = blend_and_mask_actions(old, new, 0.25)
    np.testing.assert_allclose(raw, 0.5 * old)
    np.testing.assert_array_equal(applied[5:9], np.zeros(4))
    np.testing.assert_allclose(applied[:5], raw[:5])
    np.testing.assert_allclose(applied[9:], raw[9:])
    with pytest.raises(ValueError):
        blend_and_mask_actions(old, new, 1.1)


def test_route_specific_policy_yaw_observation_offsets_do_not_change_commands() -> None:
    assert policy_yaw_observation_offset(
        "yaw_right", (0.0, 0.0, -0.55), backward_residual_scale=0.0
    ) == -0.30
    assert policy_yaw_observation_offset(
        "compound", (0.08, 0.0, -0.30), backward_residual_scale=0.0
    ) == -0.15
    assert policy_yaw_observation_offset(
        "compound", (0.08, 0.0, 0.30), backward_residual_scale=0.0
    ) == 0.0
    assert policy_yaw_observation_offset(
        "reverse_turn_right", (-0.04, 0.0, -0.20), backward_residual_scale=0.0
    ) == 0.0

    policy_command, offset, overridden = resolve_policy_observation_command(
        "yaw_right",
        (0.0, 0.0, -0.30),
        backward_residual_scale=0.0,
        override=(0.0, 0.0, -0.90),
    )
    np.testing.assert_allclose(policy_command, (0.0, 0.0, -0.90))
    assert offset == 0.0
    assert overridden is True


def test_all_adopted_commands_stay_inside_the_asymmetric_router_envelope() -> None:
    commands = [
        case.command for case in (*PRIMITIVE_CASES, *COMPOUND_CASES, *TRANSITION_CASES)
    ]
    for command in commands:
        router = SafeGaitRouter()
        for _ in range(80):
            decision = router.route(command, 0.02)
            assert decision.command_was_clipped is False, command


def test_safety_audit_separates_raw_head_output_from_applied_head_lock() -> None:
    segment = _safe_segment()
    audit = segment["safety_audit"]
    assert audit["raw_policy_head_action_peak"] == 4.0
    assert audit["applied_head_action_peak"] == 0.0
    assert audit["head_target_peak_rad"] == 0.0
    assert audit["preclip_target_limit_violations"] == 0
    assert audit["applied_target_limit_violations"] == 0
    assert audit["leg_target_margin_rad"] == 0.050
    assert audit["preclip_target_margin_violations"] == 0
    assert audit["applied_target_margin_violations"] == 0
    assert audit["applied_target_max_rad"]["left_knee"] == pytest.approx(
        SAFE_JOINT_LIMITS["left_knee"][1] - RUNTIME_TARGET_SAFETY_MARGIN_RAD
    )
    assert audit["qpos_limit_violations"] == 0
    acceptance = segment_acceptance(segment)
    assert acceptance["checks"]["desired_targets_inside_margin"] is True
    assert acceptance["checks"]["startup_margin_transition_authorized"] is True
    assert acceptance["checks"]["target_slew_safe"] is True
    assert acceptance["passed"] is True


def test_startup_policy_to_margin_slew_is_visible_and_authorized() -> None:
    home = np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER])
    guard = FinalTargetSafetyGuard(SAFE_JOINT_LIMITS, home)
    previous = guard.previous_targets
    policy_target = home.copy()
    left_knee = ACTUATOR_JOINT_ORDER.index("left_knee")
    policy_target[left_knee] = -99.0
    desired = apply_final_target_safety(policy_target, SAFE_JOINT_LIMITS)
    applied = guard.step(policy_target, 0.02)

    assert previous[left_knee] == pytest.approx(0.470534)
    assert applied[left_knee] == pytest.approx(0.430534)

    audit = SafetyAudit()
    audit.update(
        raw_policy_action=np.zeros(14),
        applied_action=np.zeros(14),
        preclip_targets=policy_target,
        margin_clipped_targets=desired,
        applied_targets=applied,
        previous_applied_targets=previous,
        joint_qpos=home,
        control_dt=0.02,
    )
    result = audit.to_dict()
    assert result["applied_target_margin_violations"] == 1
    assert result["startup_margin_transition_joint_samples"] == 1
    assert result["unauthorized_applied_target_margin_violations"] == 0
    assert result["target_slew_violations"] == 0
    assert result["maximum_target_slew_rate_rad_per_s"] == pytest.approx(2.0)


def test_control_first_startup_is_applied_and_audited_before_physics() -> None:
    home = np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER])
    guard = FinalTargetSafetyGuard(SAFE_JOINT_LIMITS, home)
    ctrl = home.copy()
    policy_target = home.copy()
    left_knee = ACTUATOR_JOINT_ORDER.index("left_knee")
    policy_target[left_knee] = -99.0

    applied, audit = apply_control_first_startup(
        guard,
        ctrl,
        policy_target,
        ACTUATOR_JOINT_ORDER,
        control_dt=0.02,
        leg_target_margin_rad=0.05,
        target_slew_rate_rad_s=2.0,
        physics_steps_before_control=0,
    )

    assert guard.steps_since_reset == 1
    assert applied[left_knee] == pytest.approx(0.430534)
    np.testing.assert_array_equal(ctrl, applied)
    assert audit["passed"] is True
    assert audit["mode"] == "control_first"
    assert audit["physics_steps_before_control"] == 0
    assert audit["control_applied_before_first_physics_step"] is True
    assert audit["guard_calls_before_control"] == 0
    assert audit["guard_calls_for_first_tick"] == 1
    assert audit["exactly_one_guard_call_for_first_tick"] is True
    assert audit["home_only_precharge_used"] is False
    assert audit["guarded_output_matches_reconstructed_step"] is True
    assert audit["maximum_applied_target_delta_rad"] == pytest.approx(0.04)
    assert audit["applied_targets_rad"]["left_knee"] == pytest.approx(0.430534)


def test_control_first_startup_audit_rejects_uncontrolled_or_unguarded_start() -> None:
    home = np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER])
    uncontrolled = audit_control_first_startup(
        home,
        home,
        home,
        control_dt=0.02,
        physics_steps_before_control=1,
    )
    assert uncontrolled["passed"] is False
    assert uncontrolled["control_applied_before_first_physics_step"] is False
    assert uncontrolled["guarded_output_matches_reconstructed_step"] is False


def test_physics_substep_audit_keeps_transient_qpos_nonfinite_and_fall_failures() -> None:
    home = np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER])
    audit = PhysicsSubstepAudit()
    common = {
        "full_qvel": np.zeros(20),
        "height_m": 0.18,
        "upright": 0.98,
        "feet_contacts": (True, False),
    }
    audit.update(joint_qpos=home, full_qpos=home, **common)
    transient = home.copy()
    left_knee = ACTUATOR_JOINT_ORDER.index("left_knee")
    transient[left_knee] = SAFE_JOINT_LIMITS["left_knee"][1] + 0.003
    audit.update(joint_qpos=transient, full_qpos=transient, **common)
    audit.update(joint_qpos=home, full_qpos=home, **common)

    result = audit.to_dict()
    assert result["sample_count"] == 3
    assert result["leg_joint_sample_count"] == 30
    assert result["qpos_limit_violations"] == 1
    assert result["maximum_qpos_excess_rad"] == pytest.approx(0.003)
    assert result["joint_qpos_max_rad"]["left_knee"] == pytest.approx(
        SAFE_JOINT_LIMITS["left_knee"][1] + 0.003
    )
    assert result["fall_or_nonfinite_detected"] is False

    bad_velocity = np.zeros(20)
    bad_velocity[0] = np.nan
    audit.update(
        joint_qpos=home,
        full_qpos=home,
        full_qvel=bad_velocity,
        height_m=0.11,
        upright=0.60,
        feet_contacts=(False, False),
    )
    failed = audit.to_dict()
    assert failed["contact_sample_count"] == failed["sample_count"] == 4
    assert failed["single_support_count"] == 3
    assert failed["flight_count"] == 1
    assert failed["single_support_rate"] == pytest.approx(0.75)
    assert failed["flight_rate"] == pytest.approx(0.25)
    assert failed["contact_sample_count_matches_sample_count"] is True
    assert failed["contact_sampling_stage"] == "immediately_after_each_mj_step"
    assert failed["nonfinite_state_samples"] == 1
    assert failed["height_fall_samples"] == 1
    assert failed["upright_fall_samples"] == 1
    assert failed["fall_or_nonfinite_detected"] is True
    assert failed["first_termination_sample"] == 4


def test_guard_ctrl_physics_order_and_first_post_step_qpos_are_fixed() -> None:
    events: list[str] = []
    home = np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER])
    policy_target = home.copy()
    left_knee = ACTUATOR_JOINT_ORDER.index("left_knee")
    policy_target[left_knee] = -99.0
    inner_guard = FinalTargetSafetyGuard(SAFE_JOINT_LIMITS, home)

    class RecordingGuard:
        @property
        def previous_targets(self):
            return inner_guard.previous_targets

        @property
        def steps_since_reset(self):
            return inner_guard.steps_since_reset

        def control_first_startup(self, targets, *, dt):
            events.append("guard")
            return inner_guard.control_first_startup(targets, dt=dt)

        def step(self, targets, dt):
            events.append("guard")
            return inner_guard.step(targets, dt)

    class RecordingCtrl:
        def __init__(self):
            self.values = home.copy()

        def __setitem__(self, key, value):
            events.append("ctrl")
            self.values[key] = value

    class RecordingMujoco:
        @staticmethod
        def mj_step(model, data):
            del model
            events.append("physics")
            data.qpos = data.ctrl.values.copy()

    data = SimpleNamespace(ctrl=RecordingCtrl(), qpos=home.copy())
    (
        previous,
        applied,
        startup_audit,
        completed_substeps,
        terminated,
    ) = apply_guarded_control_then_step_physics(
        RecordingGuard(),
        policy_target,
        mujoco=RecordingMujoco(),
        model=object(),
        data=data,
        decimation=2,
        control_dt=0.02,
        joint_names=ACTUATOR_JOINT_ORDER,
        leg_target_margin_rad=0.05,
        target_slew_rate_rad_s=2.0,
        first_control_tick=True,
        physics_steps_before_control=0,
        physics_substep_callback=lambda: events.append("substep_audit") or False,
    )

    assert events == [
        "guard",
        "ctrl",
        "physics",
        "substep_audit",
        "physics",
        "substep_audit",
    ]
    assert completed_substeps == 2
    assert terminated is False
    assert inner_guard.steps_since_reset == 1
    assert previous[left_knee] == pytest.approx(0.470534)
    assert applied[left_knee] == pytest.approx(0.430534)
    assert data.qpos[left_knee] == pytest.approx(0.430534)
    assert startup_audit is not None and startup_audit["passed"] is True

    events.clear()
    (
        _,
        second_applied,
        second_startup_audit,
        second_completed_substeps,
        second_terminated,
    ) = (
        apply_guarded_control_then_step_physics(
            RecordingGuard(),
            policy_target,
            mujoco=RecordingMujoco(),
            model=object(),
            data=data,
            decimation=2,
            control_dt=0.02,
            joint_names=ACTUATOR_JOINT_ORDER,
            leg_target_margin_rad=0.05,
            target_slew_rate_rad_s=2.0,
            first_control_tick=False,
            physics_steps_before_control=2,
            physics_substep_callback=lambda: events.append("substep_audit")
            or False,
        )
    )
    assert events == [
        "guard",
        "ctrl",
        "physics",
        "substep_audit",
        "physics",
        "substep_audit",
    ]
    assert second_completed_substeps == 2
    assert second_terminated is False
    assert inner_guard.steps_since_reset == 2
    assert second_applied[left_knee] == pytest.approx(0.390534)
    assert data.qpos[left_knee] == pytest.approx(0.390534)
    assert second_startup_audit is None


def test_reset_qpos_audit_separates_exact_home_from_noisy_five_milliradian_margin() -> None:
    home = np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER])
    exact = audit_reset_qpos(home, noise_applied=False)
    assert exact["passed"] is True
    assert exact["exact_safe_init_passed"] is True
    assert exact["applied_inward_margin_rad"] == 0.0

    noisy = home.copy()
    left_knee = ACTUATOR_JOINT_ORDER.index("left_knee")
    noisy[left_knee] = SAFE_JOINT_LIMITS["left_knee"][1] - 0.005
    bounded = audit_reset_qpos(noisy, noise_applied=True)
    assert bounded["passed"] is True
    assert bounded["reset_noise_margin_rad"] == 0.005
    assert bounded["noise_margin_violations"] == 0

    noisy[left_knee] += 0.001
    unsafe = audit_reset_qpos(noisy, noise_applied=True)
    assert unsafe["passed"] is False
    assert unsafe["physical_safe_limit_violations"] == 0
    assert unsafe["noise_margin_violations"] == 1


def test_explicit_diagnostic_guard_allows_noncontract_sweep_without_hidden_step() -> None:
    home = np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER])
    target_lower, target_upper = build_target_envelope(leg_margin_rad=0.05)
    physical_lower, physical_upper = build_target_envelope(leg_margin_rad=0.0)
    guard = DiagnosticTargetSafetyGuard(
        home,
        target_lower,
        target_upper,
        physical_lower,
        physical_upper,
        slew_rate_rad_per_s=2.0,
    )
    np.testing.assert_array_equal(guard.previous_targets, home)
    applied = guard.step(np.full(14, 99.0), 0.02)
    leg_indices = [
        index
        for index, name in enumerate(ACTUATOR_JOINT_ORDER)
        if name not in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
    ]
    assert np.max(np.abs(applied[leg_indices] - home[leg_indices])) <= 0.04 + 1e-12
    np.testing.assert_array_equal(applied[5:9], np.zeros(4))


def test_safety_audit_counts_preclip_applied_and_qpos_limit_failures_separately() -> None:
    target = np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER])
    raw_target = target.copy()
    applied_target = target.copy()
    qpos = target.copy()
    raw_target[3] = 0.60
    applied_target[13] = 0.70
    qpos[4] = -0.60
    audit = SafetyAudit()
    desired_target = apply_final_target_safety(raw_target, SAFE_JOINT_LIMITS)
    audit.update(
        raw_policy_action=np.zeros(14),
        applied_action=np.zeros(14),
        preclip_targets=raw_target,
        margin_clipped_targets=desired_target,
        applied_targets=applied_target,
        previous_applied_targets=target,
        joint_qpos=qpos,
        control_dt=0.02,
    )
    result = audit.to_dict()
    assert result["preclip_target_limit_violations"] == 1
    assert result["applied_target_limit_violations"] == 1
    assert result["preclip_target_margin_violations"] >= 1
    assert result["applied_target_margin_violations"] >= 1
    assert result["target_slew_violations"] >= 1
    assert result["qpos_limit_violations"] == 1
    assert result["maximum_preclip_target_excess_rad"] > 0.0


def test_motion_metrics_are_signed_and_command_centred_for_reverse_and_compound() -> None:
    reverse = compute_motion_metrics(
        (-0.10, 0.0, 0.0),
        np.tile((-0.10, 0.0, 0.0), (4, 1)),
        np.zeros(4),
        displacement_xyz=(-0.4, 0.0, 0.0),
        minimum_height_m=0.18,
        minimum_upright=0.99,
        mean_effective_command=(-0.10, 0.0, 0.0),
        single_support_rate=0.2,
        flight_rate=0.0,
    )
    assert reverse["projected_primary_velocity"] == pytest.approx(0.10)
    assert reverse["primary_velocity_error"] == pytest.approx(0.0)

    command = np.asarray((0.08, 0.06, 0.28))
    compound = compute_motion_metrics(
        command,
        np.tile((0.08, 0.06, 0.0), (4, 1)),
        np.full(4, 0.28),
        displacement_xyz=(0.32, 0.24, 0.0),
        minimum_height_m=0.18,
        minimum_upright=0.99,
        mean_effective_command=command,
        mean_policy_observation_command=(0.10, 0.0, -0.90),
        single_support_rate=0.2,
        flight_rate=0.0,
    )
    assert compound["primary_velocity_error"] == pytest.approx(0.0, abs=1e-15)
    assert compound["absolute_orthogonal_velocity"] == pytest.approx(0.0, abs=1e-15)
    assert compound["yaw_rate_error"] == pytest.approx(0.0)
    assert compound["physical_command"] == command.tolist()
    assert compound["mean_policy_observation_command"] == [0.10, 0.0, -0.90]


def test_acceptance_fails_head_target_fall_and_command_clipping() -> None:
    segment = _safe_segment()
    unsafe = copy.deepcopy(segment)
    unsafe["fell"] = True
    unsafe["safety_audit"]["head_target_peak_rad"] = 0.001
    unsafe["routing"]["command_clip_events"] = 1
    acceptance = segment_acceptance(unsafe)
    assert acceptance["passed"] is False
    assert acceptance["checks"]["no_fall"] is False
    assert acceptance["checks"]["head_target_locked"] is False
    assert acceptance["checks"]["command_not_clipped"] is False

    missing_substeps = copy.deepcopy(segment)
    del missing_substeps["physics_substep_audit"]
    missing_acceptance = segment_acceptance(missing_substeps)
    assert missing_acceptance["checks"]["all_physics_substeps_audited"] is False
    assert missing_acceptance["checks"]["substep_joint_qpos_safe"] is False
    assert missing_acceptance["passed"] is False

    incomplete_substeps = copy.deepcopy(segment)
    incomplete_substeps["completed_physics_substeps"] = 2
    incomplete_acceptance = segment_acceptance(incomplete_substeps)
    assert incomplete_acceptance["checks"]["all_physics_substeps_audited"] is False

    low_substep_upright = copy.deepcopy(segment)
    low_substep_upright["physics_substep_audit"]["minimum_upright"] = 0.80
    low_upright_acceptance = segment_acceptance(low_substep_upright)
    assert low_upright_acceptance["checks"]["minimum_upright"] is True
    assert low_upright_acceptance["checks"]["substep_minimum_upright"] is False
    assert low_upright_acceptance["passed"] is False

    wrong_route = copy.deepcopy(segment)
    wrong_route["routing"]["steady_state_routed_expert_steps"] = {
        "forward": 7,
        "stand": 1,
    }
    wrong_route_acceptance = segment_acceptance(wrong_route)
    assert wrong_route_acceptance["checks"]["steady_route_expected_expert"] is False
    assert wrong_route_acceptance["passed"] is False

    prohibited_route = copy.deepcopy(segment)
    prohibited_route["routing"]["prohibited_expert_steps"] = 1
    prohibited_acceptance = segment_acceptance(prohibited_route)
    assert prohibited_acceptance["checks"]["prohibited_experts_absent"] is False

    atomic_mismatch = copy.deepcopy(segment)
    atomic_mismatch["routing"]["atomic_endpoint_required"] = True
    atomic_mismatch["routing"]["atomic_endpoint_mismatch_steps"] = 1
    atomic_acceptance = segment_acceptance(atomic_mismatch)
    assert atomic_acceptance["checks"]["atomic_endpoint_exact"] is False


def test_acceptance_requires_support_for_motion_and_low_flight_for_all_segments() -> None:
    moving = _safe_segment("forward")
    no_support = copy.deepcopy(moving)
    no_support["metrics"]["single_support_rate"] = 0.049
    no_support["physics_substep_audit"]["single_support_rate"] = 0.049
    acceptance = segment_acceptance(no_support)
    assert acceptance["checks"]["moving_single_support"] is False
    assert acceptance["passed"] is False

    excessive_flight = copy.deepcopy(moving)
    excessive_flight["metrics"]["flight_rate"] = 0.050001
    excessive_flight["physics_substep_audit"]["flight_rate"] = 0.050001
    acceptance = segment_acceptance(excessive_flight)
    assert acceptance["checks"]["flight_rate"] is False
    assert acceptance["passed"] is False

    stand = _safe_segment("stand")
    stand["command"] = [0.0, 0.0, 0.0]
    stand["metrics"]["single_support_rate"] = 0.0
    stand["physics_substep_audit"]["single_support_rate"] = 0.0
    assert segment_acceptance(stand)["checks"]["moving_single_support"] is True

    endpoint_only = _safe_segment("forward")
    endpoint_only["metrics"]["contact_rate_sample_source"] = (
        "control_endpoint_after_decimation"
    )
    acceptance = segment_acceptance(endpoint_only)
    assert acceptance["checks"]["contact_rates_from_all_physics_substeps"] is False
    assert acceptance["passed"] is False


@pytest.mark.parametrize("actual_vx", [0.0, -0.01])
def test_acceptance_rejects_zero_or_wrong_sign_linear_progress(
    actual_vx: float,
) -> None:
    segment = _safe_segment("slow_forward")
    command = np.asarray((0.04, 0.0, 0.0))
    segment["command"] = command.tolist()
    segment["metrics"] = compute_motion_metrics(
        command,
        np.tile((actual_vx, 0.0, 0.0), (8, 1)),
        np.zeros(8),
        displacement_xyz=(0.0, 0.0, 0.0),
        minimum_height_m=0.18,
        minimum_upright=0.98,
        mean_effective_command=command,
        single_support_rate=0.2,
        flight_rate=0.0,
    )
    acceptance = segment_acceptance(segment)
    assert acceptance["checks"]["primary_velocity"] is True
    assert acceptance["checks"]["signed_linear_progress"] is False
    assert acceptance["passed"] is False


@pytest.mark.parametrize("actual_yaw", [0.0, -0.02])
def test_acceptance_rejects_zero_or_wrong_sign_yaw_progress(
    actual_yaw: float,
) -> None:
    segment = _safe_segment("slow_yaw_left")
    command = np.asarray((0.0, 0.0, 0.20))
    segment["command"] = command.tolist()
    segment["metrics"] = compute_motion_metrics(
        command,
        np.zeros((8, 3)),
        np.full(8, actual_yaw),
        displacement_xyz=(0.0, 0.0, 0.0),
        minimum_height_m=0.18,
        minimum_upright=0.98,
        mean_effective_command=command,
        single_support_rate=0.2,
        flight_rate=0.0,
    )
    acceptance = segment_acceptance(segment)
    assert acceptance["checks"]["yaw_rate"] is True
    assert acceptance["checks"]["signed_yaw_progress"] is False
    assert acceptance["passed"] is False


def test_suite_acceptance_requires_exact_segment_order() -> None:
    segment = _safe_segment("forward")
    home = np.asarray([SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER])
    reset_audit = audit_reset_qpos(
        home,
        noise_applied=False,
    )
    guard = FinalTargetSafetyGuard(SAFE_JOINT_LIMITS, home)
    startup_applied = guard.step(home, 0.02)
    startup_audit = audit_control_first_startup(
        home,
        home,
        startup_applied,
        control_dt=0.02,
    )
    episode = {
        "seed": 7,
        "fell": False,
        "reset_qpos_audit": reset_audit,
        "control_first_startup_audit": startup_audit,
        "backward_exit_recovery_audit": _disabled_recovery_state_audit(),
        "segments": [segment],
    }
    assert suite_acceptance([episode], ["forward"])["passed"] is True
    assert suite_acceptance([episode], ["reverse"])["passed"] is False
    missing_startup = copy.deepcopy(episode)
    del missing_startup["control_first_startup_audit"]
    result = suite_acceptance([missing_startup], ["forward"])
    assert result["passed"] is False
    assert result["episode_checks"][0]["control_first_startup_passed"] is False
    missing_recovery = copy.deepcopy(episode)
    del missing_recovery["backward_exit_recovery_audit"]
    result = suite_acceptance([missing_recovery], ["forward"])
    assert result["passed"] is False
    assert result["episode_checks"][0]["backward_exit_recovery_passed"] is False


def test_recovery_audits_are_mandatory_and_fail_closed_on_cap_or_count_drift() -> None:
    segment = _safe_segment()
    assert segment_acceptance(segment)["checks"][
        "backward_exit_recovery_audit"
    ] is True

    missing = copy.deepcopy(segment)
    del missing["backward_exit_recovery_audit"]
    assert segment_acceptance(missing)["checks"][
        "backward_exit_recovery_audit"
    ] is False

    violated = copy.deepcopy(segment)
    violated["backward_exit_recovery_audit"]["cap_violation_count"] = 1
    assert segment_acceptance(violated)["checks"][
        "backward_exit_recovery_audit"
    ] is False

    state = _disabled_recovery_state_audit(2)
    assert backward_exit_recovery_state_acceptance(state)["passed"] is True
    state["final_guard_call_count"] = 1
    result = backward_exit_recovery_state_acceptance(state)
    assert result["passed"] is False
    assert result["checks"][
        "final_guard_exactly_once_per_control_tick"
    ] is False


def test_segment_recovery_summary_records_exit_cap_release_and_reentry_cancel() -> None:
    targets = np.asarray(
        [SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER]
    )
    recovery = BackwardExitRecovery(SAFE_JOINT_LIMITS, enabled=True)
    steps = []
    for active in (True, False, False, True):
        _, step = recovery.compose(
            targets, backward_feedforward_active=active
        )
        steps.append(step)
    summary = summarize_backward_exit_recovery_steps(
        steps, enabled=True, expected_sample_count=4
    )
    assert summary["passed"] is True
    assert summary["exit_event_count"] == 1
    assert summary["active_tick_count"] == 2
    assert summary["reentry_cancel_count"] == 1
    assert summary["cap_violation_count"] == 0
    assert summary["remaining_ticks_after_segment"] == 0


def test_episode_recovery_state_accepts_exact_completed_thirteen_tick_event() -> None:
    targets = np.asarray(
        [SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER]
    )
    recovery = BackwardExitRecovery(SAFE_JOINT_LIMITS, enabled=True)
    recovery.compose(targets, backward_feedforward_active=True)
    for _ in range(13):
        recovery.compose(targets, backward_feedforward_active=False)
    audit = recovery.audit()
    audit.update(
        {
            "final_guard_call_count": audit["control_tick_count"],
            "reset_clear_on_schedule_start": True,
            "composition_before_final_guard": True,
            "final_guard_calls_per_control_tick": 1,
        }
    )
    accepted = backward_exit_recovery_state_acceptance(audit)
    assert accepted["passed"] is True
    assert accepted["checks"]["runtime_contract_exact"] is True
    assert accepted["checks"][
        "event_lifecycles_complete_or_cancelled"
    ] is True

    drifted = copy.deepcopy(audit)
    drifted["events"][0]["active_tick_count"] = 12
    assert backward_exit_recovery_state_acceptance(drifted)["passed"] is False


def test_episode_recovery_state_accepts_audited_backward_reentry_cancel() -> None:
    targets = np.asarray(
        [SAFE_INIT_POS[joint] for joint in ACTUATOR_JOINT_ORDER]
    )
    recovery = BackwardExitRecovery(SAFE_JOINT_LIMITS, enabled=True)
    for active in (True, False, False, True):
        recovery.compose(targets, backward_feedforward_active=active)
    audit = recovery.audit()
    audit.update(
        {
            "final_guard_call_count": audit["control_tick_count"],
            "reset_clear_on_schedule_start": True,
            "composition_before_final_guard": True,
            "final_guard_calls_per_control_tick": 1,
        }
    )
    accepted = backward_exit_recovery_state_acceptance(audit)
    assert accepted["passed"] is True
    assert audit["reentry_cancel_count"] == 1
    assert audit["events"][0]["status"] == (
        "CANCELLED_BY_BACKWARD_REENTRY"
    )


@pytest.mark.parametrize("simulation_passed", [False, True])
def test_hardware_gate_never_promotes_from_simulation(simulation_passed: bool) -> None:
    gate = hardware_gate(simulation_passed)
    assert gate["status"] == "PROHIBITED"
    assert gate["hardware_deployment_allowed"] is False
    assert gate["simulation_acceptance_passed"] is simulation_passed
    assert gate["simulation_pass_does_not_promote_hardware"] is True


def test_cli_accepts_configurable_seed_and_seconds_without_importing_mujoco() -> None:
    argv = []
    for assignment in _assignments():
        argv.extend(("--policy", assignment))
    argv.extend(
        (
            "--seed",
            "123",
            "--episodes",
            "2",
            "--seconds",
            "8",
            "--transition-seconds",
            "5",
            "--transition-stand-seconds",
            "2",
            "--warmup-seconds",
            "1",
        )
    )
    args = parse_args(argv)
    assert args.seed == 123
    assert args.episodes == 2
    assert args.seconds == 8.0
    assert args.transition_seconds == 5.0
    assert args.backward_residual_scale == 0.0
    assert args.leg_target_margin_rad == 0.050
    assert args.target_slew_rate_rad_s == 2.0
    assert args.diagnostic_unadopted_backward_exit_recovery is False
    assert args.formal_candidate_default is True
    assert args.backward_profile.resolve() == FORMAL_CANDIDATE_PROFILE_PATHS[
        "straight"
    ]
    assert args.backward_left_profile.resolve() == FORMAL_CANDIDATE_PROFILE_PATHS[
        "left"
    ]
    assert args.backward_right_profile.resolve() == FORMAL_CANDIDATE_PROFILE_PATHS[
        "right"
    ]
    screening = classify_evaluation_scale(args)
    assert screening["status"] == "SCREENING_CANDIDATE"
    assert screening["release_qualification_eligible"] is False

    with pytest.raises(SystemExit):
        parse_args([*argv, "--leg-target-margin-rad", "0.015"])
    with pytest.raises(SystemExit):
        parse_args([*argv, "--target-slew-rate-rad-s", "1.0"])

    diagnostic = parse_args(
        [
            *argv,
            "--diagnostic-noncontract-safety",
            "--leg-target-margin-rad",
            "0.06",
            "--target-slew-rate-rad-s",
            "2.0",
        ]
    )
    assert diagnostic.diagnostic_noncontract_safety is True
    assert diagnostic.leg_target_margin_rad == 0.06
    assert diagnostic.target_slew_rate_rad_s == 2.0
    assert diagnostic.formal_candidate_default is False

    policy_command_diagnostic = parse_args(
        [*argv, "--policy-command-diagnostic-suite"]
    )
    assert policy_command_diagnostic.policy_command_diagnostic_suite is True
    assert policy_command_diagnostic.formal_candidate_default is False

    unadopted = parse_args(
        [
            *argv,
            "--diagnostic-unadopted-policy",
            "--diagnostic-unadopted-reverse-profile",
            "candidate.json",
            "--diagnostic-unadopted-reverse-left-profile",
            "left.json",
            "--diagnostic-unadopted-reverse-right-profile",
            "right.json",
        ]
    )
    assert unadopted.diagnostic_unadopted_policy is True
    assert unadopted.formal_candidate_default is False
    assert unadopted.diagnostic_unadopted_reverse_profile == Path("candidate.json")
    assert unadopted.diagnostic_unadopted_reverse_left_profile == Path("left.json")
    assert unadopted.diagnostic_unadopted_reverse_right_profile == Path("right.json")
    assert unadopted.diagnostic_reverse_entry_phase_indices is None
    assert classify_evaluation_scale(unadopted)[
        "release_qualification_eligible"
    ] is False
    with pytest.raises(SystemExit):
        parse_args(
            [
                *argv,
                "--diagnostic-unadopted-reverse-left-profile",
                "left.json",
            ]
        )

    phase_bundle_argv = [
        *argv,
        "--diagnostic-unadopted-reverse-profile",
        "candidate.json",
        "--diagnostic-unadopted-reverse-left-profile",
        "left.json",
        "--diagnostic-unadopted-reverse-right-profile",
        "right.json",
        "--diagnostic-unadopted-reverse-entry-phase-index",
        "6.0",
        "--diagnostic-unadopted-reverse-left-entry-phase-index",
        "4.0",
        "--diagnostic-unadopted-reverse-right-entry-phase-index",
        "4.0",
    ]
    phase_args = parse_args(phase_bundle_argv)
    assert phase_args.diagnostic_reverse_entry_phase_indices == dict(
        FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES
    )
    assert classify_evaluation_scale(phase_args)[
        "release_qualification_eligible"
    ] is False
    recovery_argv = [
        *phase_bundle_argv,
        "--diagnostic-unadopted-backward-exit-recovery",
    ]
    recovery_args = parse_args(recovery_argv)
    assert recovery_args.diagnostic_unadopted_backward_exit_recovery is True
    assert classify_evaluation_scale(recovery_args)[
        "release_qualification_eligible"
    ] is False
    with pytest.raises(SystemExit):
        parse_args(
            [*argv, "--diagnostic-unadopted-backward-exit-recovery"]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [*recovery_argv, "--backward-residual-scale", "0.01"]
        )
    with pytest.raises(SystemExit):
        parse_args([*recovery_argv, "--diagnostic-unadopted-policy"])
    with pytest.raises(SystemExit):
        parse_args([*recovery_argv, "--diagnostic-noncontract-safety"])
    with pytest.raises(SystemExit):
        parse_args([*recovery_argv, "--policy-command-diagnostic-suite"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                *argv,
                "--diagnostic-unadopted-reverse-profile",
                "candidate.json",
                "--diagnostic-unadopted-reverse-entry-phase-index",
                "nan",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                *argv,
                "--diagnostic-unadopted-reverse-profile",
                "candidate.json",
                "--diagnostic-unadopted-reverse-left-profile",
                "left.json",
                "--diagnostic-unadopted-reverse-right-profile",
                "right.json",
                "--diagnostic-unadopted-reverse-entry-phase-index",
                "5.0",
                "--diagnostic-unadopted-reverse-left-entry-phase-index",
                "4.0",
                "--diagnostic-unadopted-reverse-right-entry-phase-index",
                "4.0",
            ]
        )

    release_argv = []
    for assignment in _assignments():
        release_argv.extend(("--policy", assignment))
    release_argv.extend(
        (
            "--seed",
            "20260808",
            "--episodes",
            "20",
            "--seconds",
            "30",
            "--transition-seconds",
            "30",
            "--transition-stand-seconds",
            "5",
            "--warmup-seconds",
            "1.5",
            "--initial-joint-noise-scale",
            "1",
            "--initial-base-speed",
            "0.10",
        )
    )
    release = classify_evaluation_scale(parse_args(release_argv))
    assert release["status"] == "RELEASE_QUALIFICATION"
    assert release["release_qualification_eligible"] is True
    assert release["master_seed_matches_recommendation"] is True
    assert release["master_seed_is_hard_gate"] is True
    assert release["expected"] == RELEASE_QUALIFICATION_CONFIGURATION

    release_diagnostic = classify_evaluation_scale(
        parse_args([*release_argv, "--policy-command-diagnostic-suite"])
    )
    assert release_diagnostic["status"] == "SCREENING_CANDIDATE"
    assert release_diagnostic["release_qualification_eligible"] is False

    wrong_seed = classify_evaluation_scale(
        parse_args([*release_argv, "--seed", "7"])
    )
    assert wrong_seed["scale_matches_frozen_contract"] is False
    assert wrong_seed["master_seed_matches_recommendation"] is False
    assert wrong_seed["release_qualification_eligible"] is False


def test_diagnostic_reverse_phase_entry_is_preincrement_once_and_family_stable() -> None:
    mapping = validate_diagnostic_reverse_entry_phase_indices(
        FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES
    )
    assert mapping == {
        "reverse": 6.0,
        "reverse_turn_left": 4.0,
        "reverse_turn_right": 4.0,
    }
    straight_rate = 1.872330366914159
    phase, active, event = advance_routed_phase(
        13.87233036691416,
        phase_steps=27,
        phase_delta=straight_rate,
        current_expert="reverse",
        previous_expert="stand",
        effective_command=(-0.035, 0.0, 0.0),
        previous_backward_feedforward_active=False,
        diagnostic_entry_phase_indices=mapping,
        control_step=4,
        global_control_tick=1004,
    )
    assert active is True
    assert event is not None
    assert event["status"] == "DIAGNOSTIC_UNADOPTED"
    assert event["diagnostic_only"] is True
    assert event["formal_candidate"] is False
    assert event["global_phase_index_before_reset"] == pytest.approx(
        13.87233036691416
    )
    assert event["reset_preincrement_phase_index"] == 6.0
    assert event["first_feedforward_phase_index"] == pytest.approx(
        7.872330366914159
    )
    assert phase == pytest.approx(7.872330366914159)

    # A continuously active reverse -> reverse-turn switch stays on the same
    # phase trajectory and cannot apply the turn entry value a second time.
    next_phase, still_active, second_event = advance_routed_phase(
        phase,
        phase_steps=27,
        phase_delta=1.932802073918527,
        current_expert="reverse_turn_left",
        previous_expert="reverse",
        effective_command=(-0.03, 0.0, 0.20),
        previous_backward_feedforward_active=active,
        diagnostic_entry_phase_indices=mapping,
    )
    assert still_active is True
    assert second_event is None
    assert next_phase == pytest.approx((phase + 1.932802073918527) % 27)

    _, inactive, no_event = advance_routed_phase(
        next_phase,
        phase_steps=27,
        phase_delta=1.0,
        current_expert="stand",
        previous_expert="reverse_turn_left",
        effective_command=(0.0, 0.0, 0.0),
        previous_backward_feedforward_active=True,
        diagnostic_entry_phase_indices=mapping,
    )
    assert inactive is False
    assert no_event is None
    turn_phase, turn_active, turn_event = advance_routed_phase(
        next_phase + 1.0,
        phase_steps=27,
        phase_delta=1.932802073918527,
        current_expert="reverse_turn_left",
        previous_expert="stand",
        effective_command=(-0.03, 0.0, 0.20),
        previous_backward_feedforward_active=False,
        diagnostic_entry_phase_indices=mapping,
    )
    assert turn_active is True
    assert turn_event is not None
    assert turn_event["reset_preincrement_phase_index"] == 4.0
    assert turn_phase == pytest.approx(5.932802073918527)
    assert set(mapping) == BACKWARD_FAMILY_EXPERTS


def test_adopted_h3_phase_entry_is_default_bundle_not_diagnostic() -> None:
    phase, active, event = advance_routed_phase(
        13.0,
        phase_steps=27,
        phase_delta=1.872330366914159,
        current_expert="reverse",
        previous_expert="stand",
        effective_command=(-0.035, 0.0, 0.0),
        previous_backward_feedforward_active=False,
        diagnostic_entry_phase_indices=(
            FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
        ),
        phase_entry_status=FORMAL_CANDIDATE_STATUS,
        diagnostic_only=False,
    )

    assert active is True
    assert phase == pytest.approx(8.872330366914159)
    assert event is not None
    assert event["status"] == FORMAL_CANDIDATE_STATUS
    assert event["formal_candidate"] is False
    assert event["adopted_simulation_only"] is True
    assert event["diagnostic_only"] is False


def test_formal_phase_path_has_no_reset_and_diagnostic_values_are_frozen() -> None:
    phase, active, event = advance_routed_phase(
        13.0,
        phase_steps=27,
        phase_delta=1.5,
        current_expert="reverse",
        previous_expert="stand",
        effective_command=(-0.050, 0.0, 0.0),
        previous_backward_feedforward_active=False,
        diagnostic_entry_phase_indices=None,
    )
    assert phase == pytest.approx(14.5)
    assert active is True
    assert event is None
    with pytest.raises(ValueError, match="remain exactly 6.0"):
        validate_diagnostic_reverse_entry_phase_indices(
            {
                "reverse": 5.0,
                "reverse_turn_left": 4.0,
                "reverse_turn_right": 4.0,
            }
        )
    with pytest.raises(ValueError, match="must be finite"):
        validate_diagnostic_reverse_entry_phase_indices(
            {
                "reverse": float("nan"),
                "reverse_turn_left": 4.0,
                "reverse_turn_right": 4.0,
            }
        )


def test_phase_entry_evidence_and_runtime_source_closure_are_hash_pinned(
    tmp_path: Path,
) -> None:
    phase_evidence = validate_diagnostic_reverse_phase_entry_evidence()
    assert phase_evidence["sha256"] == (
        "c78643b1c4deee8c293c6f27535190e4b8ca8d80f809de7fa72fa2ffc6751742"
    )
    assert phase_evidence["straight_preincrement_phase_index"] == 6.0
    assert phase_evidence["episode_passes"] == 5
    assert phase_evidence["source_reverse_endpoint_mps"] == (
        DIAGNOSTIC_REVERSE_PHASE_ENTRY_SOURCE_REVERSE_ENDPOINT_MPS
    ) == -0.075
    assert phase_evidence["current_formal_reverse_endpoint_mps"] == (
        CURRENT_FORMAL_REVERSE_ENDPOINT_MPS
    ) == -0.050
    assert phase_evidence[
        "source_endpoint_matches_current_formal_endpoint"
    ] is False
    assert phase_evidence["current_endpoint_status"] == (
        "CURRENT_ENDPOINT_REQUALIFICATION_REQUIRED"
    )
    assert phase_evidence["usable_as_current_straight_endpoint_evidence"] is False
    assert phase_evidence["adopted"] is False

    recovery_evidence = validate_diagnostic_backward_exit_recovery_evidence()
    assert recovery_evidence["sha256"] == (
        DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_SHA256
    )
    assert recovery_evidence["selected_strategy"]["hold_ticks"] == 13
    assert recovery_evidence["profile_sha256s"] == {
        "straight": (
            "af7f14c2c4877a088b9320d59625bd37e41677ddc3a3802761df1e982179373e"
        ),
        **dict(DIAGNOSTIC_REVERSE_TURN_PROFILE_SHA256),
    }
    assert recovery_evidence["policy_sha256"] == BASE_V22_POLICY_SHA256
    assert recovery_evidence["selected_upper_target_rad"] == pytest.approx(
        0.413034
    )
    assert recovery_evidence["source_reverse_endpoint_mps"] == (
        DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_SOURCE_REVERSE_ENDPOINT_MPS
    ) == -0.075
    assert recovery_evidence["current_formal_reverse_endpoint_mps"] == -0.050
    assert recovery_evidence[
        "source_endpoint_matches_current_formal_endpoint"
    ] is False
    assert recovery_evidence["current_endpoint_status"] == (
        "CURRENT_ENDPOINT_REQUALIFICATION_REQUIRED"
    )
    assert recovery_evidence["usable_as_current_straight_endpoint_evidence"] is False
    assert recovery_evidence["adopted"] is False
    assert recovery_evidence["adoption_eligible"] is False

    copied_recovery = tmp_path / "copied_recovery.json"
    copied_recovery.write_bytes(
        Path(recovery_evidence["path"]).read_bytes()
    )
    with pytest.raises(ValueError, match="path must remain pinned"):
        validate_diagnostic_backward_exit_recovery_evidence(copied_recovery)

    profiles = {
        label: {"sha256": digest}
        for label, digest in recovery_evidence["profile_sha256s"].items()
    }
    policies = {
        "roles": {
            role: {"sha256": BASE_V22_POLICY_SHA256}
            for role in REQUIRED_POLICY_ROLES
        }
    }
    binding = validate_diagnostic_backward_exit_recovery_execution_bundle(
        recovery_evidence,
        profiles,
        policies,
    )
    assert binding["passed"] is True
    drifted_profiles = copy.deepcopy(profiles)
    drifted_profiles["left"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact hash-pinned"):
        validate_diagnostic_backward_exit_recovery_execution_bundle(
            recovery_evidence,
            drifted_profiles,
            policies,
        )

    external = validate_frozen_runtime_source_dependencies()
    assert external["dependency_count"] == 4
    assert external["root_sha256"] == FROZEN_RUNTIME_DEPENDENCY_ROOT_SHA256
    assert external["all_hashes_verified"] is True

    dependency = tmp_path / "dependency.py"
    dependency.write_bytes(b"stable source")
    expected = hashlib.sha256(dependency.read_bytes()).hexdigest()
    captured = capture_runtime_source_dependency_closure(
        {"dependency": dependency}, expected_sha256={"dependency": expected}
    )
    assert captured["all_hashes_verified"] is True
    dependency.write_bytes(b"mutated source")
    with pytest.raises(ValueError, match="hash mismatch"):
        capture_runtime_source_dependency_closure(
            {"dependency": dependency}, expected_sha256={"dependency": expected}
        )


def test_h3_safety_selection_and_superseded_h2_lineage_are_independently_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = validate_formal_candidate_selection_evidence()
    assert evidence["sha256"] == FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
    assert evidence["status"] == H3_CANDIDATE_SELECTION_STATUS
    assert evidence["source_artifact_status"] == H3_FAST_EXIT_SAFETY_STATUS
    assert evidence["safety_component_evidence_sha256"] == (
        H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
    )
    assert evidence["superseded_h2_selection_evidence_sha256"] == (
        H2_5X15_SELECTION_EVIDENCE_SHA256
    )
    assert evidence["suite_episode_count"] == 15
    assert evidence["segment_pass_count"] == 190
    assert evidence["physics_substep_count"] == 1_100_000
    assert evidence["contact_sample_count"] == 1_100_000
    assert evidence["control_sample_count"] == 110_000
    assert evidence["leg_qpos_sample_count"] == 11_000_000
    assert evidence["minimum_height_m"] == pytest.approx(0.17911993)
    assert evidence["minimum_upright"] == pytest.approx(0.9785479266972336)
    assert evidence["authorized_startup_margin_transition_joint_samples"] == 40
    assert evidence["preclip_target_margin_violations"] == 110_403
    assert evidence["phase_entry_event_count"] == 30
    assert evidence["recovery_exit_event_count"] == 15
    assert evidence["recovery_active_tick_count"] == 195
    assert evidence["profile_left_knee_cap"] == {
        "extra_upper_margin_rad": 0.0125,
        "upper_target_rad": 0.413034,
    }
    assert evidence["backward_exit_recovery"] == {
        "enabled": True,
        "extra_upper_margin_rad": 0.0225,
        "hold_control_ticks": 13,
        "hold_seconds": 0.26,
        "upper_target_rad": 0.403034,
    }
    assert evidence["combined_5x15_passed"] is True
    assert evidence["combined_5x15_required"] is False
    assert evidence["candidate_execution_eligible"] is True
    assert evidence["adoption_eligible"] is False
    assert evidence["simulation_acceptance_eligible"] is False
    assert evidence["hardware_deployment"] == "PROHIBITED"
    assert FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256_ALLOWLIST == {
        FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
    }
    assert all(
        hashes == frozenset({FORMAL_ADOPTION_EVIDENCE_SHA256})
        for hashes in FORMAL_REVERSE_EVIDENCE_SHA256_ALLOWLISTS.values()
    )

    safety = validate_h3_fast_exit_safety_evidence()
    assert safety["path"] == str(H3_FAST_EXIT_SAFETY_EVIDENCE_PATH)
    assert safety["sha256"] == H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
    assert H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256_ALLOWLIST == {
        H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
    }
    assert safety["status"] == H3_FAST_EXIT_SAFETY_STATUS
    assert safety["source_artifact_status"] == "DIAGNOSTIC_FAIL"
    assert safety["source_artifact_passed"] is False
    assert safety["safety_only_component"] is True
    assert safety["safety_subset_passed"] is True
    assert safety["central_suite_acceptance_passed"] is False
    assert safety["episode_count"] == 20
    assert safety["segment_count"] == 500
    assert safety["safety_passed_segment_count"] == 500
    assert safety["central_passed_segment_count"] == 489
    assert safety["physics_substep_count"] == 370_000
    assert safety["contact_sample_count"] == 370_000
    assert safety["motion_failure_count"] == 11
    assert tuple(
        (item["seed"], item["name"], item["check"])
        for item in safety["motion_failures"]
    ) == H3_FAST_EXIT_EXPECTED_MOTION_FAILURES
    assert safety["minimum_left_knee_safe_upper_margin_rad"] == pytest.approx(
        0.0018947227635599528
    )
    assert safety["profile_left_knee_cap"] == {
        "extra_upper_margin_rad": 0.0125,
        "upper_target_rad": 0.413034,
    }
    assert safety["backward_exit_recovery"] == {
        "enabled": True,
        "extra_upper_margin_rad": 0.0225,
        "hold_control_ticks": 13,
        "hold_seconds": 0.26,
        "upper_target_rad": 0.403034,
    }
    assert safety["combined_5x15_required"] is True
    assert safety["adoption_eligible"] is False
    assert safety["simulation_acceptance_eligible"] is False
    assert safety["adoption_evidence"] is False
    assert safety["release_evidence"] is False

    adoption = validate_formal_adoption_evidence()
    assert adoption["path"] == str(FORMAL_ADOPTION_EVIDENCE_PATH)
    assert adoption["sha256"] == FORMAL_ADOPTION_EVIDENCE_SHA256
    assert FORMAL_ADOPTION_EVIDENCE_SHA256_ALLOWLIST == frozenset(
        {FORMAL_ADOPTION_EVIDENCE_SHA256}
    )
    assert adoption["selection_evidence_sha256"] == (
        FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
    )
    assert adoption["safety_component_evidence_sha256"] == (
        H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
    )
    assert adoption["status"] == FORMAL_CANDIDATE_STATUS
    assert adoption["hash_allowlisted_for_adoption"] is True
    assert adoption["suite_episode_count"] == 60
    assert adoption["segment_pass_count"] == 760
    assert adoption["acceptance_check_true_count"] == 28_120
    assert adoption["physics_substep_count"] == 8_150_000
    assert adoption["contact_sample_count"] == 8_150_000
    assert adoption["control_sample_count"] == 815_000
    assert adoption["leg_qpos_sample_count"] == 81_500_000
    assert adoption["reset_startup_recovery_audit_count"] == 280
    assert adoption["phase_entry_event_count"] == 120
    assert adoption["recovery_exit_event_count"] == 60
    assert adoption["recovery_active_tick_count"] == 780
    assert adoption["authorized_startup_margin_transition_joint_samples"] == 147
    assert adoption["preclip_target_margin_violations"] == 819_203
    assert adoption["maximum_left_knee_qpos_rad"] == pytest.approx(
        0.4736497298325716
    )
    assert adoption["minimum_left_knee_safe_upper_margin_rad"] == pytest.approx(
        0.0018842701674284257
    )
    assert adoption["minimum_height_m"] == pytest.approx(0.17911993)
    assert adoption["minimum_upright"] == pytest.approx(0.9777608163890137)
    assert adoption["performance_extrema"] == pytest.approx(
        {
            "minimum_signed_linear_progress_fraction": 0.3595826926676137,
            "minimum_signed_yaw_progress_fraction": 0.4936553773470118,
            "maximum_primary_velocity_error_mps": 0.028009207275213346,
            "maximum_orthogonal_velocity_mps": 0.024010891338336268,
            "maximum_yaw_only_planar_velocity_mps": 0.02668671634496424,
            "maximum_yaw_rate_error_radps": 0.10269293300510374,
            "maximum_uncommanded_yaw_rate_radps": 0.1395255079553353,
            "maximum_stop_drift_m": 0.04004149774890449,
            "minimum_moving_single_support_rate": 0.11593333333333333,
            "maximum_flight_rate": 0.005666666666666667,
        }
    )
    assert adoption["package_release_evidence"] is False
    assert adoption["adoption_eligible"] is True
    assert adoption["simulation_acceptance_eligible"] is True
    assert adoption["hardware_deployment"] == "PROHIBITED"

    superseded = validate_superseded_h2_adoption_evidence()
    assert superseded["path"] == str(H2_SUPERSEDED_ADOPTION_EVIDENCE_PATH)
    assert superseded["sha256"] == H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256
    assert superseded["selection_evidence_sha256"] == (
        H2_5X15_SELECTION_EVIDENCE_SHA256
    )
    assert superseded["status"] == H2_SUPERSEDED_ADOPTION_STATUS
    assert superseded["superseded_lineage_only"] is True
    assert superseded["hash_allowlisted_for_adoption"] is False
    assert superseded["suite_episode_count"] == 60
    assert superseded["segment_pass_count"] == 760
    assert superseded["physics_substep_count"] == 8_150_000
    assert superseded["contact_sample_count"] == 8_150_000
    assert superseded["control_sample_count"] == 815_000
    assert superseded["leg_qpos_sample_count"] == 81_500_000
    assert superseded["phase_entry_event_count"] == 120
    assert superseded["recovery_exit_event_count"] == 60
    assert superseded["recovery_active_tick_count"] == 780
    assert superseded["package_release_evidence"] is False
    assert superseded["adoption_eligible"] is False
    assert superseded["simulation_acceptance_eligible"] is False
    assert superseded["hardware_deployment"] == "PROHIBITED"

    profiles = validate_formal_candidate_reverse_profiles(
        FORMAL_CANDIDATE_PROFILE_PATHS["straight"],
        FORMAL_CANDIDATE_PROFILE_PATHS["left"],
        FORMAL_CANDIDATE_PROFILE_PATHS["right"],
    )
    assert {
        label: record["sha256"] for label, record in profiles.items()
    } == {
        label: next(iter(hashes))
        for label, hashes in FORMAL_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS.items()
    }
    assert all(record["formal_candidate"] is False for record in profiles.values())
    assert all(record["adopted"] is True for record in profiles.values())
    assert all(record["adoption_eligible"] is True for record in profiles.values())
    assert all(
        record["simulation_acceptance_eligible"] is True
        for record in profiles.values()
    )

    copied_adoption = tmp_path / "h3_adoption.json"
    copied_adoption.write_bytes(Path(adoption["path"]).read_bytes())
    with pytest.raises(ValueError, match="path must remain pinned"):
        validate_formal_adoption_evidence(copied_adoption)

    original_bytes = copied_adoption.read_bytes()
    old_value = b"0.9777608163890137"
    new_value = b"0.9777608163890138"
    assert original_bytes.count(old_value) >= 1
    copied_adoption.write_bytes(original_bytes.replace(old_value, new_value, 1))
    assert copied_adoption.stat().st_size == Path(adoption["path"]).stat().st_size
    mutated_adoption_digest = hashlib.sha256(copied_adoption.read_bytes()).hexdigest()
    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            routed_evaluation_module,
            "FORMAL_ADOPTION_EVIDENCE_PATH",
            copied_adoption.resolve(),
        )
        local_patch.setattr(
            routed_evaluation_module,
            "FORMAL_ADOPTION_EVIDENCE_SHA256",
            mutated_adoption_digest,
        )
        local_patch.setattr(
            routed_evaluation_module,
            "FORMAL_ADOPTION_EVIDENCE_SHA256_ALLOWLIST",
            frozenset({mutated_adoption_digest}),
        )
        with pytest.raises(ValueError, match="mismatch"):
            validate_formal_adoption_evidence(copied_adoption)

    copied = tmp_path / "candidate_selection.json"
    copied.write_bytes(Path(evidence["path"]).read_bytes())
    with pytest.raises(ValueError, match="path must remain pinned"):
        validate_formal_candidate_selection_evidence(copied)

    mutated_payload = json.loads(copied.read_text(encoding="utf-8"))
    mutated_payload["simulation_suite_acceptance_passed"] = False
    copied.write_text(json.dumps(mutated_payload), encoding="utf-8")
    mutated_digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    monkeypatch.setattr(
        routed_evaluation_module,
        "FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH",
        copied.resolve(),
    )
    monkeypatch.setattr(
        routed_evaluation_module,
        "FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256",
        mutated_digest,
    )
    monkeypatch.setattr(
        routed_evaluation_module,
        "FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256_ALLOWLIST",
        frozenset({mutated_digest}),
    )
    with pytest.raises(ValueError, match="size mismatch|top-level status mismatch"):
        validate_formal_candidate_selection_evidence(copied)


def test_runtime_versions_and_binary_allowlist_are_exact() -> None:
    validated = validate_runtime_versions(FROZEN_RUNTIME_VERSIONS)
    assert validated["exact_versions_verified"] is True
    assert validated["actual"]["onnxruntime"] == "1.28.0"
    mismatched = dict(FROZEN_RUNTIME_VERSIONS)
    mismatched["numpy"] = "2.5.0"
    with pytest.raises(ValueError, match="runtime version mismatch"):
        validate_runtime_versions(mismatched)
    assert set(FROZEN_RUNTIME_BINARY_SHA256) == {
        "python_executable",
        "libmujoco",
        "libonnxruntime",
        "onnxruntime_pybind",
        "numpy_core",
    }
    assert all(len(digest) == 64 for digest in FROZEN_RUNTIME_BINARY_SHA256.values())


def test_current_generated_exact_safe_assets_verify_without_simulation() -> None:
    exp_root = Path(__file__).resolve().parents[1]
    evidence = validate_exact_generated_assets(
        exp_root / "artifacts" / "generated_playground"
    )
    assert evidence["contract"] == "hardware_safe_simulation_only"
    assert evidence["real_hardware_deployment_allowed"] is False
    assert set(evidence["verified_files"]) == {
        "scene",
        "model",
        "reference",
        "backward_default",
        "backward_right",
    }
    closure = evidence["dependency_closure"]
    assert evidence["generated_root"]["path"] == str(FROZEN_GENERATED_ROOT)
    assert closure["entry_count"] == 29
    assert closure["xml_count"] == 1
    assert closure["mesh_count"] == 28
    assert closure["hfield_count"] == 0
    assert closure["root_sha256"] == FROZEN_GENERATED_DEPENDENCY_ROOT_SHA256
    assert dependency_closure_root_sha256(closure["entries"]) == (
        FROZEN_GENERATED_DEPENDENCY_ROOT_SHA256
    )


def test_current_reverse_profile_hashes_verify_but_v1_is_rejected() -> None:
    exp_root = Path(__file__).resolve().parents[1]
    generated_data = (
        exp_root
        / "artifacts"
        / "generated_playground"
        / "playground"
        / "open_duck_mini_v2"
        / "data"
    )
    evidence = validate_adopted_reverse_profiles(
        exp_root / "artifacts" / "optimized_reverse_exact_safe_v1.json",
        exp_root / "artifacts" / "optimized_reverse_left_exact_safe_v1.json",
        generated_data / "optimized_backward_right_turn_gait.json",
    )
    assert evidence["straight"]["release_id"] == "optimized_reverse_exact_safe_v1"
    assert evidence["left"]["release_id"] == "optimized_reverse_left_exact_safe_v1"
    assert evidence["right"]["release_id"].endswith("_legacy")
    assert evidence["straight"]["composition"][
        "left_knee_extra_upper_margin_rad"
    ] == 0.0
    assert REVERSE_V1_ADOPTION_STATUS == "REJECTED_AWAITING_REOPTIMIZATION"
    assert REVERSE_V1_MEASURED_FORWARD_VELOCITY_MPS == -0.00156


def test_generated_asset_hash_mismatch_is_a_hard_error(tmp_path: Path) -> None:
    package = tmp_path / "playground" / "open_duck_mini_v2"
    xmls = package / "xmls"
    data = package / "data"
    xmls.mkdir(parents=True)
    data.mkdir(parents=True)
    files = {
        "generated_scene": xmls / "scene_flat_terrain_backlash_hardware_safe_calibrated.xml",
        "generated_reference": data / "polynomial_coefficients_calibrated.pkl",
        "legacy_v22_optimized_backward_gait": data / "optimized_backward_gait.json",
        "legacy_v22_optimized_backward_left_turn_gait": data
        / "optimized_backward_left_turn_gait.json",
        "legacy_v22_optimized_backward_right_turn_gait": data
        / "optimized_backward_right_turn_gait.json",
    }
    manifest_files = {}
    for key, path in files.items():
        path.write_bytes(key.encode("utf-8"))
        manifest_files[key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest_files["generated_scene"]["sha256"] = "0" * 64
    (tmp_path / "hardware_safe_manifest.json").write_text(
        json.dumps(
            {
                "contract": "hardware_safe_simulation_only",
                "real_hardware_deployment_allowed": False,
                "files": manifest_files,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generated root must remain pinned"):
        validate_exact_generated_assets(tmp_path)


def test_mjcf_dependency_discovery_is_transitive_and_includes_hfields(
    tmp_path: Path,
) -> None:
    xmls = tmp_path / "xmls"
    assets = xmls / "assets"
    assets.mkdir(parents=True)
    scene = xmls / "scene.xml"
    model = xmls / "model.xml"
    nested = xmls / "nested.xml"
    mesh = assets / "body.stl"
    hfield = assets / "terrain.png"
    scene.write_text(
        '<mujoco><include file="model.xml"/></mujoco>', encoding="utf-8"
    )
    model.write_text(
        '<mujoco><compiler meshdir="assets"/>'
        '<include file="nested.xml"/><asset><mesh file="body.stl"/></asset></mujoco>',
        encoding="utf-8",
    )
    nested.write_text(
        '<mujoco><compiler assetdir="assets"/>'
        '<asset><hfield file="terrain.png"/></asset></mujoco>',
        encoding="utf-8",
    )
    mesh.write_bytes(b"mesh")
    hfield.write_bytes(b"hfield")

    closure = discover_mjcf_dependency_closure(scene, tmp_path)
    assert set(closure) == {
        "xmls/model.xml",
        "xmls/nested.xml",
        "xmls/assets/body.stl",
        "xmls/assets/terrain.png",
    }
    assert closure["xmls/model.xml"]["kind"] == "xml"
    assert closure["xmls/nested.xml"]["kind"] == "xml"
    assert closure["xmls/assets/body.stl"]["kind"] == "mesh"
    assert closure["xmls/assets/terrain.png"]["kind"] == "hfield"
    assert len(dependency_closure_root_sha256(closure)) == 64


def test_all_formal_policy_roles_are_hash_pinned_to_base_v22(
    tmp_path: Path,
) -> None:
    exp_root = Path(__file__).resolve().parents[1]
    base_v22 = (
        exp_root.parents[2]
        / "experiments"
        / "mujoco"
        / "exp_003_openduckmini_calibrated_walk"
        / "artifacts"
        / "calibrated_hybrid_policy_v22.onnx"
    )
    policies = {role: base_v22 for role in REQUIRED_POLICY_ROLES}
    formal = validate_policy_provenance(policies, diagnostic_unadopted=False)
    assert FORMAL_POLICY_SHA256_ALLOWLIST == frozenset({BASE_V22_POLICY_SHA256})
    assert formal["adoption_eligible"] is True
    assert all(
        record["sha256"] == BASE_V22_POLICY_SHA256
        for record in formal["roles"].values()
    )

    arbitrary = tmp_path / "candidate.onnx"
    arbitrary.write_bytes(b"not base v22")
    arbitrary_bank = {role: arbitrary for role in REQUIRED_POLICY_ROLES}
    with pytest.raises(ValueError, match="not frozen base-v22"):
        validate_policy_provenance(
            arbitrary_bank, diagnostic_unadopted=False
        )
    diagnostic = validate_policy_provenance(
        arbitrary_bank, diagnostic_unadopted=True
    )
    assert diagnostic["adoption_eligible"] is False
    assert diagnostic["diagnostic_unadopted"] is True
    assert all(record["adopted"] is False for record in diagnostic["roles"].values())


def test_v3_reverse_candidate_schema_records_hash_finite_values_and_knee_cap(
    tmp_path: Path,
) -> None:
    exp_root = Path(__file__).resolve().parents[1]
    candidate = (
        exp_root / "artifacts" / "optimized_reverse_margin050_slew200_candidate_v3.json"
    )
    evidence = validate_diagnostic_unadopted_reverse_profile(candidate)
    assert evidence["sha256"] == (
        "af7f14c2c4877a088b9320d59625bd37e41677ddc3a3802761df1e982179373e"
    )
    assert evidence["schema_validated"] is True
    assert evidence["all_json_numbers_finite"] is True
    assert evidence["composition"]["left_knee_extra_upper_margin_rad"] == 0.0125
    assert evidence["adopted"] is False
    assert evidence["adoption_eligible"] is False

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    invalid_cap = tmp_path / "invalid_cap.json"
    payload["composition"]["left_knee_extra_upper_margin_rad"] = 0.050001
    invalid_cap.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="extra upper margin"):
        validate_diagnostic_unadopted_reverse_profile(invalid_cap)

    nonfinite = tmp_path / "nonfinite.json"
    payload["composition"]["left_knee_extra_upper_margin_rad"] = 0.0125
    payload["parameters"]["phase_rate"] = float("nan")
    nonfinite.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        validate_reverse_profile_schema(nonfinite, diagnostic_unadopted=True)


def test_atomic_reverse_turn_candidates_pin_direction_command_v3_base_and_cap(
    tmp_path: Path,
) -> None:
    exp_root = Path(__file__).resolve().parents[1]
    straight = validate_diagnostic_unadopted_reverse_profile(
        exp_root
        / "artifacts"
        / "optimized_reverse_margin050_slew200_candidate_v3.json"
    )
    paths = {
        "left": exp_root
        / "artifacts"
        / "reverse_turn_candidates_v1"
        / "optimized_reverse_turn_left_margin050_slew200_candidate_v1.json",
        "right": exp_root
        / "artifacts"
        / "reverse_turn_candidates_v1"
        / "optimized_reverse_turn_right_margin050_slew200_candidate_v1.json",
    }
    expected_hashes = {
        "left": "b36f14dc1bbacfbf998adc00f6e6fe62d1f14a4a8de034b1b0b18ae5bccb8703",
        "right": "e2229527d435d03636c091ca7b435ed3be483b0e74293d28a2ff927995bea16b",
    }
    for direction, path in paths.items():
        evidence = validate_diagnostic_unadopted_reverse_turn_profile(
            path,
            direction=direction,
            straight_base_evidence=straight,
        )
        assert evidence["sha256"] == expected_hashes[direction]
        assert evidence["schema_validated"] is True
        assert evidence["all_json_numbers_finite"] is True
        assert evidence["composition"]["straight_reverse_base_sha256"] == (
            straight["sha256"]
        )
        assert evidence["composition"]["left_knee_extra_upper_margin_rad"] == (
            0.0125
        )
        assert evidence["adopted"] is False

    invalid = json.loads(paths["left"].read_text(encoding="utf-8"))
    invalid["atomic_command"] = [-0.04, 0.0, -0.20]
    invalid_path = tmp_path / "invalid_left.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="atomic_command mismatch"):
        validate_diagnostic_unadopted_reverse_turn_profile(
            invalid_path,
            direction="left",
            straight_base_evidence=straight,
        )

    wrong_base = copy.deepcopy(straight)
    wrong_base["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact diagnostic v3 base"):
        validate_diagnostic_unadopted_reverse_turn_profile(
            paths["left"],
            direction="left",
            straight_base_evidence=wrong_base,
        )

    portable = json.loads(paths["left"].read_text(encoding="utf-8"))
    portable["composition"]["straight_reverse_base_profile"] = (
        "/mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/"
        "exp_004_openduckmini_safe_gait_experts/artifacts/"
        "optimized_reverse_margin050_slew200_candidate_v3.json"
    )
    portable_path = tmp_path / "portable_left.json"
    portable_path.write_text(json.dumps(portable), encoding="utf-8")
    portable_evidence = validate_diagnostic_unadopted_reverse_turn_profile(
        portable_path,
        direction="left",
        straight_base_evidence=straight,
    )
    assert portable_evidence["schema_validated"] is True


def test_reverse_adoption_is_bound_to_the_h3_20x30_evidence() -> None:
    exp_root = Path(__file__).resolve().parents[1]
    generated_data = (
        exp_root
        / "artifacts"
        / "generated_playground"
        / "playground"
        / "open_duck_mini_v2"
        / "data"
    )
    profiles = validate_adopted_reverse_profiles(
        exp_root / "artifacts" / "optimized_reverse_exact_safe_v1.json",
        exp_root / "artifacts" / "optimized_reverse_left_exact_safe_v1.json",
        generated_data / "optimized_backward_right_turn_gait.json",
    )
    current = derive_reverse_profile_adoption(profiles, {})
    assert current["passed"] is False
    assert current["status"] == "FAIL_CLOSED"
    assert all(
        hashes == frozenset({FORMAL_ADOPTION_EVIDENCE_SHA256})
        for hashes in FORMAL_REVERSE_EVIDENCE_SHA256_ALLOWLISTS.values()
    )
    assert all(
        status == "ADOPTED_SIMULATION_ONLY"
        for status in FORMAL_REVERSE_ADOPTION_STATUSES.values()
    )
    assert all(
        hashes == frozenset({FORMAL_ADOPTION_EVIDENCE_SHA256})
        for hashes in (
            FORMAL_REVERSE_COMMAND_CASE_ADOPTION_EVIDENCE_SHA256_ALLOWLISTS.values()
        )
    )
    assert all(
        hashes == frozenset({H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256})
        for hashes in (
            FORMAL_REVERSE_COMMAND_CASE_SAFETY_EVIDENCE_SHA256_ALLOWLISTS.values()
        )
    )

    adopted_profiles = validate_formal_candidate_reverse_profiles(
        FORMAL_CANDIDATE_PROFILE_PATHS["straight"],
        FORMAL_CANDIDATE_PROFILE_PATHS["left"],
        FORMAL_CANDIDATE_PROFILE_PATHS["right"],
    )
    adoption_evidence = validate_formal_adoption_evidence()
    adopted = derive_reverse_profile_adoption(
        adopted_profiles,
        {label: adoption_evidence for label in ("straight", "left", "right")},
    )
    assert adopted["passed"] is True
    assert adopted["status"] == "ADOPTED_SIMULATION_ONLY"

    labels = ("straight", "left", "right")
    future_profiles = {label: {"sha256": f"profile-{label}"} for label in labels}
    future_evidence = {label: {"sha256": f"evidence-{label}"} for label in labels}
    future = derive_reverse_profile_adoption(
        future_profiles,
        future_evidence,
        {label: "ACCEPTED_LOCKED_RUNTIME" for label in labels},
        profile_hash_allowlists={
            label: frozenset({f"profile-{label}"}) for label in labels
        },
        evidence_hash_allowlists={
            label: frozenset({f"evidence-{label}"}) for label in labels
        },
    )
    assert future["passed"] is True
    assert future["status"] == "ADOPTED_SIMULATION_ONLY"


def test_backward_profile_knee_cap_is_applied_before_the_uniform_guard() -> None:
    simulator = object.__new__(RoutedSimulator)
    simulator.runtime = SimpleNamespace(ACTION_SCALE=0.1)
    simulator.joint_ranges = np.repeat([[-1.0, 1.0]], 14, axis=0)
    simulator.left_knee_index = ACTUATOR_JOINT_ORDER.index("left_knee")
    simulator.left_knee_extra_upper_margin_rad = 0.0125
    simulator.left_knee_profile_upper_target_rad = (
        SAFE_JOINT_LIMITS["left_knee"][1] - 0.050 - 0.0125
    )
    simulator.evaluator = SimpleNamespace(
        backward_parameters=lambda yaw: (
            np.ones(10),
            np.zeros(10),
            1.0,
        ),
        _backward_feedforward=lambda *args, **kwargs: np.ones(14),
    )
    targets = simulator._policy_target(
        np.zeros(14),
        np.asarray((-0.050, 0.0, 0.0)),
        0.0,
        np.zeros(14),
    )
    assert targets[simulator.left_knee_index] == pytest.approx(
        SAFE_JOINT_LIMITS["left_knee"][1] - 0.050 - 0.0125
    )
    np.testing.assert_array_equal(targets[5:9], np.zeros(4))


class _FakeIo:
    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(self, path: str):
        self.value = float(sum(path.encode("utf-8")) % 7)

    def get_inputs(self):
        return [_FakeIo("obs", [1, 101])]

    def get_outputs(self):
        return [_FakeIo("action", [1, 14])]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _, feed):
        assert feed["obs"].shape == (1, 101)
        return [np.full((1, 14), self.value, dtype=np.float32)]


class _FakeOrt:
    @staticmethod
    def InferenceSession(path: str, providers):
        assert providers == ["CPUExecutionProvider"]
        return _FakeSession(path)


def test_policy_bank_routes_reverse_turns_to_compound_and_masks_head(tmp_path: Path) -> None:
    paths = {}
    for role in REQUIRED_POLICY_ROLES:
        path = tmp_path / f"{role}.onnx"
        path.write_bytes(role.encode("utf-8"))
        paths[role] = path
    bank = RoutedPolicyBank(paths, _FakeOrt)
    decision = SimpleNamespace(
        blend_from_expert="stand",
        blend_to_expert="reverse_turn_left",
        blend_alpha=0.5,
    )
    raw, applied = bank.infer_route(decision, np.zeros(101, dtype=np.float32))
    assert raw.shape == (14,)
    np.testing.assert_array_equal(applied[5:9], np.zeros(4))
    assert bank.inference_counts["stand"] == 1
    assert bank.inference_counts["compound"] == 1
