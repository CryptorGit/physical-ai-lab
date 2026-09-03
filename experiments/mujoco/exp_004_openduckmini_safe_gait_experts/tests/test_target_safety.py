from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from target_safety import (
    ACTUATOR_JOINT_ORDER,
    BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256,
    BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256,
    BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT,
    BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
    BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
    BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
    BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
    BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256,
    BACKWARD_EXIT_RECOVERY_STATUS,
    BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_ADOPTION_EVIDENCE_SHA256,
    BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_SELECTION_EVIDENCE_SHA256,
    BackwardExitRecovery,
    CONTROL_FIRST_STARTUP_DT_S,
    FinalTargetSafetyGuard,
    HEAD_ACTION_INDICES,
    LEG_ACTION_INDICES,
    PERTURBED_RESET_QPOS_MARGIN_RAD,
    RUNTIME_TARGET_SAFETY_MARGIN_RAD,
    RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD,
    RUNTIME_TARGET_SLEW_RATE_RAD_S,
    apply_final_target_safety,
    apply_reset_qpos_safety,
    backward_exit_recovery_contract,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (EXPERIMENT_ROOT / "contract.json").read_text(encoding="utf-8")
)
SAFE_LIMITS = CONTRACT["safe_joint_limits_rad"]


def test_final_guard_clamps_every_leg_with_margin_and_locks_head() -> None:
    targets = np.full(len(ACTUATOR_JOINT_ORDER), 99.0, dtype=np.float64)
    original = targets.copy()

    guarded = apply_final_target_safety(targets, SAFE_LIMITS)

    assert np.array_equal(targets, original)
    assert np.all(guarded[np.asarray(HEAD_ACTION_INDICES)] == 0.0)
    for index in LEG_ACTION_INDICES:
        name = ACTUATOR_JOINT_ORDER[index]
        expected_upper = SAFE_LIMITS[name][1] - RUNTIME_TARGET_SAFETY_MARGIN_RAD
        assert guarded[index] == pytest.approx(expected_upper)


def test_final_guard_clamps_lower_bounds_with_margin() -> None:
    guarded = apply_final_target_safety(
        np.full(len(ACTUATOR_JOINT_ORDER), -99.0), SAFE_LIMITS
    )

    for index in LEG_ACTION_INDICES:
        name = ACTUATOR_JOINT_ORDER[index]
        expected_lower = SAFE_LIMITS[name][0] + RUNTIME_TARGET_SAFETY_MARGIN_RAD
        assert guarded[index] == pytest.approx(expected_lower)


def test_final_guard_rejects_non_frozen_margin() -> None:
    with pytest.raises(ValueError, match="exactly 0.050"):
        apply_final_target_safety(
            np.zeros(len(ACTUATOR_JOINT_ORDER)), SAFE_LIMITS, margin_rad=0.0
        )


@pytest.mark.parametrize(
    "targets",
    [
        np.zeros(13),
        np.zeros(15),
        np.asarray([np.nan] + [0.0] * 13),
    ],
)
def test_final_guard_rejects_invalid_targets(targets: np.ndarray) -> None:
    with pytest.raises(ValueError, match="joint_targets"):
        apply_final_target_safety(targets, SAFE_LIMITS)


def test_final_guard_requires_exact_leg_limit_set() -> None:
    incomplete = dict(SAFE_LIMITS)
    incomplete.pop("left_knee")

    with pytest.raises(ValueError, match="exactly all ten"):
        apply_final_target_safety(
            np.zeros(len(ACTUATOR_JOINT_ORDER)), incomplete
        )


def _safe_midpoints() -> np.ndarray:
    targets = np.zeros(len(ACTUATOR_JOINT_ORDER), dtype=np.float64)
    for index in LEG_ACTION_INDICES:
        lower, upper = SAFE_LIMITS[ACTUATOR_JOINT_ORDER[index]]
        targets[index] = (lower + upper) / 2.0
    return targets


def test_stateful_guard_limits_every_leg_target_to_two_rad_per_second() -> None:
    initial = _safe_midpoints()
    guard = FinalTargetSafetyGuard(SAFE_LIMITS, initial)
    desired = initial + 1.0

    applied = guard.step(desired, dt=0.02)

    max_delta = RUNTIME_TARGET_SLEW_RATE_RAD_S * 0.02
    assert max_delta == RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD == 0.04
    assert np.all(
        np.abs(applied[np.asarray(LEG_ACTION_INDICES)] - initial[np.asarray(LEG_ACTION_INDICES)])
        <= max_delta + 1e-12
    )
    assert np.all(applied[np.asarray(HEAD_ACTION_INDICES)] == 0.0)
    assert np.array_equal(guard.previous_targets, applied)


def test_stateful_guard_invalid_step_does_not_mutate_state() -> None:
    initial = _safe_midpoints()
    guard = FinalTargetSafetyGuard(SAFE_LIMITS, initial)
    before = guard.previous_targets

    with pytest.raises(ValueError, match="dt"):
        guard.step(initial + 1.0, dt=0.0)
    with pytest.raises(ValueError, match="exactly 0.02"):
        guard.step(initial + 1.0, dt=0.01)

    assert np.array_equal(guard.previous_targets, before)


def test_stateful_guard_rejects_non_frozen_slew_rate() -> None:
    with pytest.raises(ValueError, match="exactly 2.0"):
        FinalTargetSafetyGuard(
            SAFE_LIMITS,
            _safe_midpoints(),
            max_slew_rate_rad_s=3.0,
        )


def test_first_command_policy_target_is_the_only_startup_slew() -> None:
    home_by_name = CONTRACT["safe_init_pos_rad"]
    home = np.asarray(
        [home_by_name[name] for name in ACTUATOR_JOINT_ORDER],
        dtype=np.float64,
    )
    left_knee_index = ACTUATOR_JOINT_ORDER.index("left_knee")
    guard = FinalTargetSafetyGuard(SAFE_LIMITS, home)
    first_command_policy_targets = home.copy()
    first_command_policy_targets[left_knee_index] = 0.0

    assert guard.previous_targets[left_knee_index] == pytest.approx(0.470534)
    first = guard.control_first_startup(
        first_command_policy_targets,
        dt=CONTROL_FIRST_STARTUP_DT_S,
    )

    assert first[left_knee_index] == pytest.approx(0.430534)
    assert guard.steps_since_reset == 1


def test_control_first_startup_requires_exact_dt_and_first_call() -> None:
    initial = _safe_midpoints()
    guard = FinalTargetSafetyGuard(SAFE_LIMITS, initial)

    with pytest.raises(ValueError, match="exactly 0.02"):
        guard.control_first_startup(initial, dt=0.01)
    assert guard.steps_since_reset == 0

    guard.control_first_startup(initial)
    with pytest.raises(RuntimeError, match="first step"):
        guard.control_first_startup(initial)


def test_zero_noise_reset_preserves_exact_home() -> None:
    home_by_name = CONTRACT["safe_init_pos_rad"]
    home = np.asarray(
        [home_by_name[name] for name in ACTUATOR_JOINT_ORDER],
        dtype=np.float64,
    )

    guarded = apply_reset_qpos_safety(
        home,
        SAFE_LIMITS,
        joint_noise_scale=0.0,
    )

    assert np.array_equal(guarded, home)


def test_positive_noise_reset_uses_five_milliradian_inward_margin() -> None:
    noisy = np.full(len(ACTUATOR_JOINT_ORDER), 99.0, dtype=np.float64)

    guarded = apply_reset_qpos_safety(
        noisy,
        SAFE_LIMITS,
        joint_noise_scale=0.01,
    )

    for index in LEG_ACTION_INDICES:
        name = ACTUATOR_JOINT_ORDER[index]
        assert guarded[index] == pytest.approx(
            SAFE_LIMITS[name][1] - PERTURBED_RESET_QPOS_MARGIN_RAD
        )
    assert np.all(guarded[np.asarray(HEAD_ACTION_INDICES)] == 0.0)


@pytest.mark.parametrize("noise_scale", [-0.01, np.nan, np.inf])
def test_reset_guard_rejects_invalid_noise_scale(noise_scale: float) -> None:
    with pytest.raises(ValueError, match="joint_noise_scale"):
        apply_reset_qpos_safety(
            _safe_midpoints(),
            SAFE_LIMITS,
            joint_noise_scale=noise_scale,
        )


def test_backward_exit_recovery_contract_is_exact_adopted_h3() -> None:
    contract = backward_exit_recovery_contract()

    assert BACKWARD_EXIT_RECOVERY_STATUS == "ADOPTED_SIMULATION_ONLY"
    assert BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT is True
    assert BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD == 0.0225
    assert BACKWARD_EXIT_RECOVERY_HOLD_TICKS == 13
    assert BACKWARD_EXIT_RECOVERY_HOLD_SECONDS == 0.26
    assert BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD == pytest.approx(
        0.403034
    )
    assert contract["status"] == BACKWARD_EXIT_RECOVERY_STATUS
    assert contract["enabled_by_default"] is True
    assert contract["formal_candidate_only"] is False
    assert contract["diagnostic_unadopted_only"] is False
    assert contract["adoption_eligible"] is True
    assert contract["simulation_acceptance_eligible"] is True
    assert contract["candidate_selection_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256
    )
    assert contract["superseded_h2_selection_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_SELECTION_EVIDENCE_SHA256
    )
    assert contract["superseded_h2_adoption_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_ADOPTION_EVIDENCE_SHA256
    )
    assert contract["adoption_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256
    )
    assert contract["safety_component_evidence_sha256"] == (
        BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256
    )
    assert contract["safety_component_only"] is False
    assert contract["safety_component_evidence_is_safety_only"] is True
    assert contract["fast_exit_safety_passed"] is True
    assert contract["combined_5x15_required"] is False
    assert contract["combined_5x15_passed"] is True
    assert contract["requires_formal_20x30_requalification"] is False
    assert contract["exit_tick_is_first_active_tick"] is True
    assert contract["release"] == "instant_after_hold"
    assert contract["composition_stage"] == (
        "after_policy_or_profile_before_final_target_guard"
    )
    assert contract["final_guard_calls_per_tick"] == 1


def test_disabled_backward_exit_recovery_is_exactly_inert() -> None:
    recovery = BackwardExitRecovery(SAFE_LIMITS, enabled=False)
    targets = _safe_midpoints()
    original = targets.copy()

    for active in (False, True, True, False, False):
        composed, step = recovery.compose(
            targets, backward_feedforward_active=active
        )
        assert np.array_equal(composed, original)
        assert step["recovery_active"] is False
        assert step["exit_event"] is False

    audit = recovery.audit()
    assert audit["enabled"] is False
    assert audit["disabled_path_inert"] is True
    assert audit["exit_event_count"] == 0
    assert audit["active_tick_count"] == 0
    assert audit["passed"] is True


def test_backward_exit_recovery_holds_exit_tick_plus_twelve_then_releases() -> None:
    recovery = BackwardExitRecovery(SAFE_LIMITS, enabled=True)
    targets = _safe_midpoints()
    knee = ACTUATOR_JOINT_ORDER.index("left_knee")
    targets[knee] = 99.0

    _, inactive = recovery.compose(
        targets, backward_feedforward_active=False
    )
    assert inactive["recovery_active"] is False
    _, entry = recovery.compose(targets, backward_feedforward_active=True)
    assert entry["recovery_active"] is False

    recovery_steps = []
    for _ in range(BACKWARD_EXIT_RECOVERY_HOLD_TICKS):
        composed, step = recovery.compose(
            targets, backward_feedforward_active=False
        )
        recovery_steps.append(step)
        assert composed[knee] == pytest.approx(
            BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        )
        assert step["cap_violation"] is False

    assert recovery_steps[0]["exit_event"] is True
    assert recovery_steps[0]["recovery_tick_index"] == 1
    assert recovery_steps[-1]["recovery_tick_index"] == 13
    assert recovery_steps[-1]["remaining_ticks_after_step"] == 0
    released, released_step = recovery.compose(
        targets, backward_feedforward_active=False
    )
    assert released[knee] == 99.0
    assert released_step["recovery_active"] is False
    audit = recovery.audit()
    assert audit["exit_event_count"] == 1
    assert audit["active_tick_count"] == 13
    assert audit["completed_event_count"] == 1
    assert audit["reentry_cancel_count"] == 0
    assert audit["events"][0]["status"] == "COMPLETED"
    assert audit["events"][0]["active_tick_count"] == 13
    assert audit["passed"] is True


def test_backward_reentry_cancels_remaining_recovery() -> None:
    recovery = BackwardExitRecovery(SAFE_LIMITS, enabled=True)
    targets = _safe_midpoints()
    recovery.compose(targets, backward_feedforward_active=True)
    recovery.compose(targets, backward_feedforward_active=False)
    assert recovery.remaining_ticks == 12

    _, step = recovery.compose(targets, backward_feedforward_active=True)

    assert step["reentry_cancelled"] is True
    assert step["recovery_active"] is False
    assert recovery.remaining_ticks == 0
    audit = recovery.audit()
    assert audit["reentry_cancel_count"] == 1
    assert audit["completed_event_count"] == 0
    assert audit["events"][0]["status"] == "CANCELLED_BY_BACKWARD_REENTRY"
    assert audit["passed"] is True


def test_backward_exit_recovery_reset_clears_state_and_history() -> None:
    recovery = BackwardExitRecovery(SAFE_LIMITS, enabled=True)
    targets = _safe_midpoints()
    recovery.compose(targets, backward_feedforward_active=True)
    recovery.compose(targets, backward_feedforward_active=False)
    assert recovery.remaining_ticks == 12

    reset_audit = recovery.reset()

    assert reset_audit["reset_count"] == 2
    assert reset_audit["control_tick_count"] == 0
    assert reset_audit["remaining_ticks"] == 0
    assert reset_audit["previous_backward_feedforward_active"] is False
    assert reset_audit["events"] == []
    # A standalone inactive tick after reset cannot synthesize an exit.
    _, step = recovery.compose(targets, backward_feedforward_active=False)
    assert step["exit_event"] is False
    assert step["recovery_active"] is False


def test_backward_exit_recovery_invalid_input_does_not_mutate_state() -> None:
    recovery = BackwardExitRecovery(SAFE_LIMITS, enabled=True)
    targets = _safe_midpoints()
    recovery.compose(targets, backward_feedforward_active=True)
    before = recovery.audit()

    with pytest.raises(ValueError, match="shape"):
        recovery.compose(np.zeros(13), backward_feedforward_active=False)
    with pytest.raises(ValueError, match="boolean"):
        recovery.compose(targets, backward_feedforward_active=1)  # type: ignore[arg-type]

    assert recovery.audit() == before
