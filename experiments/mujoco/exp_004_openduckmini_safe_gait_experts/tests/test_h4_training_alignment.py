from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from safe_gait_experts.contract import (
    ACTUATOR_JOINT_ORDER,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
)
from safe_gait_experts.h4_training_alignment import (
    FORCE_CONTACT_OFF_NORMALIZED,
    FORCE_CONTACT_ON_NORMALIZED,
    FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID,
    H4_FORWARD_V2_EXACT_ENDPOINT_PROBABILITY,
    H4_FORWARD_V2_LOCAL_ANCHOR_PROBABILITY,
    H4_FORWARD_V2_STAND_PROBABILITY,
    H4_FORWARD_V2_TRANSITION_PROBABILITY,
    H4_REVERSE_V2_EXACT_ENDPOINT_PROBABILITY,
    H4_REVERSE_V2_LOCAL_ANCHOR_PROBABILITY,
    H4_REVERSE_V2_STAND_PROBABILITY,
    H4_REVERSE_V2_TRANSITION_PROBABILITY,
    H4QualityRewardScales,
    H4_ACTOR_OBSERVATION_WIDTH,
    H4_OBSERVATION_PHYSICAL_COMMAND_SLICE,
    H4_OBSERVATION_SLIP_SPEED_SLICE,
    LEGACY_PRIVILEGED_IMITATION_INDEX_SLICE,
    LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE,
    LEGACY_PRIVILEGED_REFERENCE_SLICE,
    MAX_TARGET_DELTA_PER_TICK_RAD,
    NumpyTargetGuard,
    OBSERVATION_MOTOR_TARGET_SLICE,
    OBSERVATION_POLICY_COMMAND_SLICE,
    V4_CONTACT_PERSISTENCE_INTERVALS,
    V4_CONTROL_SUBSTEP_COUNT,
    V4ContactPersistenceState,
    V4SavedDynamicState,
    V4SourceSemanticPreflight,
    V4TrajectoryParity,
    REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID,
    aggregate_force_contact_quality,
    audit_v4_dynamic6_parity,
    audit_v4_dynamic_endpoint_self_consistency,
    audit_v4_source_semantic_reference,
    contract_reset_noise_vector,
    contract_target_vectors,
    final_target_guard_step,
    force_schmitt_contacts,
    forward_iteration_v6_contact_abort_island_only_reward_loss,
    forward_iteration_v6_contact_abort_island_only_telemetry,
    initialize_v4_contact_telemetry,
    make_anchor_command_mapper,
    make_h4_aligned_environment_class,
    make_h4_forward_v2_physical_sampler,
    make_h4_reverse_v2_physical_sampler,
    make_v4_compiled_single_authority_assertion,
    make_v6_compiled_invariant_assertion,
    margin_clip_targets,
    project_reset_qpos,
    reconstruct_v4_dynamic_state,
    reverse_phase_conditioned_quality_losses,
    reverse_iteration_v6_absolute_full_leg_target_telemetry,
    reverse_iteration_v6_absolute_full_leg_target_wiring_audit,
    reverse_iteration_v6_absolute_full_leg_targets,
    reverse_iteration_v6_teacher_timing_only_reference,
    reverse_iteration_v6_structural_count_invariants,
    require_checkpoint_observation_compatibility,
    require_v4_single_authority_invariants,
    robot_body_weight_n,
    scan_v4_contact_telemetry_two_phase_reference,
    scan_v4_instrumented_physics_trajectory,
    scan_v4_saved_state_contact_quality_trajectory,
    scan_v4_saved_state_contact_telemetry,
    scan_v4_saved_state_contact_telemetry_with_quality_trace,
    synchronize_observation_motor_targets,
    synchronize_post_step_command_observations,
    synchronize_post_step_imitation_state,
    touchdown_count_imbalance_metric,
    save_v4_dynamic_state,
    total_normal_force_quality,
    transplant_v22_checkpoint_to_h4_observation,
    update_alternation_state,
    update_load_balance_ema,
    update_contact_pulse_state,
    update_support_quality_state,
    update_v4_contact_persistence,
    update_v4_contact_telemetry,
    v4_aborted_transition_loss,
    v4_authoritative_primitive_step,
    v4_dynamic_field_counts_exact,
    v4_saved_dynamic_trajectory_all_finite,
    discard_v4_terminal_incomplete,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _runtime_target_safety_module():
    path = EXPERIMENT_ROOT / "target_safety.py"
    spec = importlib.util.spec_from_file_location("h4_runtime_target_safety", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_vectors_expose_left_knee_startup_exception() -> None:
    lower, upper, initial = contract_target_vectors()
    left_knee = ACTUATOR_JOINT_ORDER.index("left_knee")

    assert lower.shape == upper.shape == initial.shape == (14,)
    assert initial[left_knee] == pytest.approx(0.470534)
    assert upper[left_knee] - 0.05 == pytest.approx(0.425534)
    assert initial[left_knee] > upper[left_knee] - 0.05


def test_zero_noise_preserves_exact_safe_init_and_locks_head() -> None:
    reset = project_reset_qpos(np.ones(14), noise_multiplier=0.0)
    expected = np.asarray(
        [SAFE_INIT_POS[name] for name in ACTUATOR_JOINT_ORDER], dtype=np.float64
    )

    np.testing.assert_array_equal(reset, expected)
    np.testing.assert_array_equal(reset[5:9], np.zeros(4))


def test_positive_reset_noise_uses_joint_scales_and_reset_only_margin() -> None:
    scales = contract_reset_noise_vector()
    reset = project_reset_qpos(np.ones(14), noise_multiplier=1.0)

    np.testing.assert_array_equal(scales[5:9], np.zeros(4))
    assert scales[3] == pytest.approx(0.05)
    assert scales[4] == pytest.approx(0.08)
    np.testing.assert_array_equal(reset[5:9], np.zeros(4))
    for index, name in enumerate(ACTUATOR_JOINT_ORDER):
        if 5 <= index < 9:
            continue
        lower, upper = SAFE_JOINT_LIMITS[name]
        assert lower + 0.005 <= reset[index] <= upper - 0.005


@pytest.mark.parametrize(
    ("left_count", "expected_loss", "expected_scaled_v3_reward"),
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 1.0, -4.0),
        (3.0, 4.0, -16.0),
        (5.0, 16.0, -64.0),
    ),
)
def test_touchdown_balance_formula_and_v3_scale_are_exact(
    left_count: float,
    expected_loss: float,
    expected_scaled_v3_reward: float,
) -> None:
    update = update_support_quality_state(
        previous_contact=np.asarray([False, False]),
        contact=np.asarray([False, False]),
        tangential_speed_m_s=np.zeros(2),
        previous_stance_slip_integral_m=np.zeros(2),
        previous_single_support_ema=np.asarray(0.0),
        previous_contact_duty_ema=np.zeros(2),
        previous_touchdown_counts=np.asarray([left_count, 0.0]),
    )
    assert float(update.touchdown_count_balance_loss) == expected_loss
    assert -4.0 * float(update.touchdown_count_balance_loss) == (
        expected_scaled_v3_reward
    )


def test_touchdown_count_metric_keeps_integer_state_and_reports_float32() -> None:
    counts = np.asarray([7, 3], dtype=np.int32)

    metric = touchdown_count_imbalance_metric(counts)

    np.testing.assert_array_equal(counts, np.asarray([7, 3], dtype=np.int32))
    assert counts.dtype == np.int32
    assert np.asarray(metric).dtype == np.float32
    assert float(metric) == 4.0


def test_aligned_environment_wires_float32_touchdown_metric_reset_to_step() -> None:
    metric_name = "h4/raw_touchdown_count_imbalance"
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(make_h4_aligned_environment_class))
    )

    reset_loops = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Tuple):
            continue
        registered_names = {
            element.value
            for element in node.iter.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
        }
        initializes_float_metric = any(
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and ast.unparse(statement.value.func) == "jp.zeros"
            and any(ast.unparse(target) == "metrics[name]" for target in statement.targets)
            for statement in node.body
        )
        if metric_name in registered_names and initializes_float_metric:
            reset_loops.append(node)
    assert len(reset_loops) == 1

    step_assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "metrics"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == metric_name
        ):
            step_assignments.append(node)
    assert len(step_assignments) == 1
    assert isinstance(step_assignments[0].value, ast.Call)
    assert ast.unparse(step_assignments[0].value.func) == (
        "touchdown_count_imbalance_metric"
    )


def test_touchdown_count_metric_stabilizes_jit_vmap_and_ppo_scan_carry() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    metric_name = "h4/raw_touchdown_count_imbalance"

    @jax.jit
    def reset_then_step(counts):
        metrics = {metric_name: jp.zeros((), dtype=jp.float32)}
        metrics[metric_name] = touchdown_count_imbalance_metric(counts, xp=jp)
        return metrics[metric_name]

    stepped = reset_then_step(jp.asarray([2, 0], dtype=jp.int32))
    assert stepped.dtype == jp.float32
    assert float(stepped) == 2.0

    batched_metric = jax.jit(
        jax.vmap(lambda counts: touchdown_count_imbalance_metric(counts, xp=jp))
    )
    for batch_size in (2, 1_250):
        counts = jp.zeros((batch_size, 2), dtype=jp.int32)
        counts = counts.at[:, 0].set(jp.arange(batch_size, dtype=jp.int32) % 5)
        metrics = batched_metric(counts)
        assert metrics.shape == (batch_size,)
        assert metrics.dtype == jp.float32

    def ppo_like_rollout(batch_size):
        initial_counts = jp.zeros((batch_size, 2), dtype=jp.int32)
        initial_metrics = {
            metric_name: jp.zeros((batch_size,), dtype=jp.float32)
        }

        def body(carry, foot_index):
            counts, metrics = carry
            increment = jax.nn.one_hot(
                foot_index,
                2,
                dtype=jp.int32,
            )
            next_counts = counts + increment
            next_metrics = dict(metrics)
            next_metrics[metric_name] = jax.vmap(
                lambda value: touchdown_count_imbalance_metric(value, xp=jp)
            )(next_counts)
            return (next_counts, next_metrics), next_metrics[metric_name]

        return jax.lax.scan(
            body,
            (initial_counts, initial_metrics),
            jp.asarray([0, 1, 0, 1], dtype=jp.int32),
        )

    compiled_rollout = jax.jit(ppo_like_rollout, static_argnums=0)
    (final_counts, final_metrics), history = compiled_rollout(2)
    assert final_counts.dtype == jp.int32
    assert final_metrics[metric_name].dtype == jp.float32
    assert history.dtype == jp.float32
    np.testing.assert_array_equal(
        np.asarray(final_counts), np.asarray([[2, 2]] * 2)
    )
    np.testing.assert_array_equal(np.asarray(history[:, 0]), [1.0, 0.0, 1.0, 0.0])

    raw_integer_difference = jp.abs(
        final_counts[0, 0] - final_counts[0, 1]
    )
    assert raw_integer_difference.dtype == jp.int32


def test_control_first_left_knee_slews_from_safe_init_without_teleport() -> None:
    _, _, initial = contract_target_vectors()
    desired = initial.copy()
    guard = NumpyTargetGuard.from_reset(initial)

    first = guard.control_first_startup(desired)
    second = guard.step(desired)
    left_knee = ACTUATOR_JOINT_ORDER.index("left_knee")

    assert first[left_knee] == pytest.approx(0.430534)
    assert initial[left_knee] - first[left_knee] == pytest.approx(
        MAX_TARGET_DELTA_PER_TICK_RAD
    )
    assert first[left_knee] > SAFE_JOINT_LIMITS["left_knee"][1] - 0.05
    assert second[left_knee] == pytest.approx(0.425534)
    np.testing.assert_array_equal(first[5:9], np.zeros(4))
    assert guard.steps_since_reset == 2


def test_numpy_guard_matches_frozen_runtime_for_random_sequence() -> None:
    runtime = _runtime_target_safety_module()
    rng = np.random.default_rng(20260809)
    _, _, initial = contract_target_vectors()
    reference = runtime.FinalTargetSafetyGuard(
        SAFE_JOINT_LIMITS, initial, margin_rad=0.05, max_slew_rate_rad_s=2.0
    )
    aligned = NumpyTargetGuard.from_reset(initial)

    for step in range(100):
        desired = rng.uniform(-2.0, 2.0, size=14)
        expected = (
            reference.control_first_startup(desired, dt=0.02)
            if step == 0
            else reference.step(desired, 0.02)
        )
        actual = (
            aligned.control_first_startup(desired)
            if step == 0
            else aligned.step(desired)
        )
        np.testing.assert_array_equal(actual, expected)


def test_jax_guard_matches_numpy_and_is_jittable_when_available() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(71)
    _, _, initial = contract_target_vectors()
    desired = rng.normal(size=14)
    expected = final_target_guard_step(desired, initial)
    compiled = jax.jit(
        lambda target, previous: final_target_guard_step(
            target, previous, xp=jp
        )
    )

    actual = np.asarray(compiled(jp.asarray(desired), jp.asarray(initial)))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-8)


def test_force_schmitt_trigger_has_distinct_on_and_off_thresholds() -> None:
    assert FORCE_CONTACT_ON_NORMALIZED > FORCE_CONTACT_OFF_NORMALIZED
    previous = np.asarray([False, True])
    force = np.asarray([0.007, 0.007])

    contact = force_schmitt_contacts(force, previous)

    np.testing.assert_array_equal(contact, np.asarray([False, True]))


def _drive_v4_transition(state, normalized_force, sample_count=21):
    updates = []
    for _ in range(sample_count):
        update = update_v4_contact_telemetry(
            state, np.asarray(normalized_force, dtype=np.float32)
        )
        state = update.state
        updates.append(update)
    return state, updates


def test_v4_factory_opt_in_is_explicit_and_defaults_off() -> None:
    parameter = inspect.signature(make_h4_aligned_environment_class).parameters[
        "forward_v4_substep_contact"
    ]
    assert parameter.default is False


def test_v4_reset_force_is_baseline_without_phantom_touchdown() -> None:
    state = initialize_v4_contact_telemetry(
        np.asarray([FORCE_CONTACT_ON_NORMALIZED, 0.0], dtype=np.float32)
    )

    np.testing.assert_array_equal(state.persistence.raw_contact, [True, False])
    np.testing.assert_array_equal(
        state.persistence.qualified_contact, [True, False]
    )
    np.testing.assert_array_equal(state.persistence.pending_active, [False, False])
    np.testing.assert_array_equal(state.touchdown_counts, [0, 0])
    assert state.touchdown_counts.dtype == np.int32
    assert int(state.last_touchdown_foot) == -1


def test_v4_raw_schmitt_is_separate_from_qualified_contact() -> None:
    state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    opened = update_v4_contact_telemetry(
        state, np.asarray([FORCE_CONTACT_ON_NORMALIZED, 0.0], dtype=np.float32)
    )
    held = update_v4_contact_telemetry(
        opened.state,
        np.asarray(
            [0.5 * (FORCE_CONTACT_ON_NORMALIZED + FORCE_CONTACT_OFF_NORMALIZED), 0.0],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(opened.state.persistence.raw_contact, [True, False])
    np.testing.assert_array_equal(
        opened.state.persistence.qualified_contact, [False, False]
    )
    np.testing.assert_array_equal(held.state.persistence.raw_contact, [True, False])
    np.testing.assert_array_equal(
        held.state.persistence.qualified_contact, [False, False]
    )
    np.testing.assert_array_equal(held.touchdown_event, [False, False])
    np.testing.assert_array_equal(held.state.touchdown_counts, [0, 0])


def test_v4_confirmation_requires_twenty_elapsed_2ms_intervals() -> None:
    state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    force = np.asarray([0.02, 0.0], dtype=np.float32)
    opened = update_v4_contact_telemetry(state, force)
    state = opened.state
    assert int(state.persistence.pending_intervals[0]) == 0

    for expected_age in range(1, V4_CONTACT_PERSISTENCE_INTERVALS):
        update = update_v4_contact_telemetry(state, force)
        state = update.state
        assert int(state.persistence.pending_intervals[0]) == expected_age
        assert not bool(update.touchdown_event[0])

    confirmed = update_v4_contact_telemetry(state, force)
    assert bool(confirmed.touchdown_event[0])
    assert bool(confirmed.confirmed_transition[0])
    assert bool(confirmed.state.persistence.qualified_contact[0])
    assert not bool(confirmed.state.persistence.pending_active[0])
    assert int(confirmed.state.touchdown_counts[0]) == 1
    assert confirmed.state.touchdown_counts.dtype == np.int32


def test_v4_abort_loss_has_exact_zero_twenty_forty_ms_boundaries() -> None:
    np.testing.assert_allclose(
        v4_aborted_transition_loss(np.asarray([0, 10, 20], dtype=np.int32)),
        [1.0, 0.25, 0.0],
        rtol=0.0,
        atol=0.0,
    )

    state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    state = update_v4_contact_telemetry(state, np.asarray([0.02, 0.0])).state
    immediate = update_v4_contact_telemetry(state, np.zeros(2))
    assert bool(immediate.aborted_contact_island_event[0])
    assert int(immediate.aborted_span_intervals[0]) == 0
    assert float(immediate.aborted_loss[0]) == pytest.approx(1.0)

    state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    state = update_v4_contact_telemetry(state, np.asarray([0.02, 0.0])).state
    for _ in range(10):
        state = update_v4_contact_telemetry(state, np.asarray([0.02, 0.0])).state
    twenty_ms = update_v4_contact_telemetry(state, np.zeros(2))
    assert int(twenty_ms.aborted_span_intervals[0]) == 10
    assert float(twenty_ms.aborted_loss[0]) == pytest.approx(0.25)


def test_v4_aborted_off_gap_is_symmetric_and_never_counts_touchdown() -> None:
    state = initialize_v4_contact_telemetry(
        np.asarray([0.02, 0.0], dtype=np.float32)
    )
    opened = update_v4_contact_telemetry(state, np.zeros(2, dtype=np.float32))
    aborted = update_v4_contact_telemetry(
        opened.state, np.asarray([0.02, 0.0], dtype=np.float32)
    )

    assert bool(aborted.aborted_off_gap_event[0])
    assert not bool(aborted.aborted_contact_island_event[0])
    assert float(aborted.aborted_loss[0]) == pytest.approx(1.0)
    np.testing.assert_array_equal(aborted.state.touchdown_counts, [0, 0])
    assert not bool(aborted.alternation_event)


def test_v4_terminal_pending_is_right_censored_without_event_or_count() -> None:
    state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    state = update_v4_contact_telemetry(state, np.asarray([0.02, 0.0])).state
    before_counts = state.touchdown_counts.copy()

    retained = discard_v4_terminal_incomplete(state, False)
    discarded = discard_v4_terminal_incomplete(state, True)

    np.testing.assert_array_equal(retained.persistence.pending_active, [True, False])
    np.testing.assert_array_equal(discarded.persistence.pending_active, [False, False])
    np.testing.assert_array_equal(discarded.persistence.pending_intervals, [0, 0])
    np.testing.assert_array_equal(discarded.touchdown_counts, before_counts)
    np.testing.assert_array_equal(
        discarded.persistence.qualified_contact, [False, False]
    )


def test_v4_simultaneous_touchdowns_count_both_but_cannot_fake_alternation() -> None:
    state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    state, updates = _drive_v4_transition(state, [0.02, 0.02])

    np.testing.assert_array_equal(state.touchdown_counts, [1, 1])
    assert not bool(updates[-1].alternation_event)
    assert int(state.last_touchdown_foot) == -1


def test_v4_touchdown_alternation_rejects_same_foot_and_accepts_opposite() -> None:
    state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    state, first_left = _drive_v4_transition(state, [0.02, 0.0])
    assert not bool(first_left[-1].alternation_event)
    assert int(state.last_touchdown_foot) == 0

    state, _ = _drive_v4_transition(state, [0.0, 0.0])
    state, same_left = _drive_v4_transition(state, [0.02, 0.0])
    assert not bool(same_left[-1].alternation_event)
    assert int(state.last_touchdown_foot) == 0

    state, _ = _drive_v4_transition(state, [0.0, 0.0])
    state, opposite_right = _drive_v4_transition(state, [0.0, 0.02])
    assert bool(opposite_right[-1].alternation_event)
    assert float(opposite_right[-1].alternation_quality) > 0.0
    assert int(state.last_touchdown_foot) == 1
    np.testing.assert_array_equal(state.touchdown_counts, [2, 1])


def _run_numpy_v4_two_phase(force_trace):
    samples = iter(np.asarray(force_trace, dtype=np.float32))
    initial_state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    return scan_v4_contact_telemetry_two_phase_reference(
        np.asarray(0, dtype=np.int32),
        np.asarray(1, dtype=np.int32),
        initial_state,
        single_physics_step=lambda data, action: data + action,
        cohere_measurement_state=lambda data: data,
        measure_normalized_force=lambda _data, _raw: next(samples),
        n_substeps=V4_CONTROL_SUBSTEP_COUNT,
    )


@dataclass(frozen=True)
class _FakeV4DynamicData:
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    time: np.ndarray
    qacc_warmstart: np.ndarray
    static_marker: np.ndarray
    derived_position: np.ndarray

    def replace(self, **changes):
        return replace(self, **changes)


def _fake_v4_control_entry() -> _FakeV4DynamicData:
    return _FakeV4DynamicData(
        qpos=np.zeros(2, dtype=np.float32),
        qvel=np.zeros(1, dtype=np.float32),
        act=np.zeros(0, dtype=np.float32),
        ctrl=np.zeros(1, dtype=np.float32),
        time=np.asarray(0.0, dtype=np.float32),
        qacc_warmstart=np.zeros(1, dtype=np.float32),
        static_marker=np.asarray([73.0], dtype=np.float32),
        derived_position=np.asarray([-99.0], dtype=np.float32),
    )


def test_v4_primitive_body_is_direct_and_nested_scan_wrapper_is_distinct() -> None:
    @dataclass(frozen=True)
    class FakeData:
        ctrl: np.ndarray
        value: int

        def replace(self, **changes):
            return replace(self, **changes)

    primitive_calls = []
    nested_scan_calls = []

    def fake_mjx_step(model, data):
        primitive_calls.append((model, data.ctrl.copy()))
        return data.replace(value=data.value + int(data.ctrl[0]))

    initial = FakeData(np.asarray([0], dtype=np.int32), 4)
    action = np.asarray([3], dtype=np.int32)
    direct = v4_authoritative_primitive_step(
        "model", initial, action, mjx_step=fake_mjx_step
    )

    def nested_source_wrapper(model, data, nested_action, n_substeps):
        nested_scan_calls.append(n_substeps)
        carry = data
        for _ in range(n_substeps):
            carry = v4_authoritative_primitive_step(
                model, carry, nested_action, mjx_step=fake_mjx_step
            )
        return carry

    nested = nested_source_wrapper("model", initial, action, 1)

    assert direct.value == nested.value == 7
    np.testing.assert_array_equal(direct.ctrl, action)
    np.testing.assert_array_equal(nested.ctrl, action)
    assert nested_scan_calls == [1]
    assert len(primitive_calls) == 2
    np.testing.assert_array_equal(primitive_calls[0][1], action)
    np.testing.assert_array_equal(primitive_calls[1][1], action)


def test_v4_guarded_call_site_binds_single_authority_dynamic6_replay() -> None:
    source = textwrap.dedent(inspect.getsource(make_h4_aligned_environment_class))
    module = ast.parse(source)
    guarded = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "guarded_physics_step"
    )
    authority_body = next(
        node
        for node in ast.walk(guarded)
        if isinstance(node, ast.FunctionDef)
        and node.name == "authoritative_single_step"
    )

    authority_calls = [
        node for node in ast.walk(authority_body) if isinstance(node, ast.Call)
    ]
    direct_calls = [
        node
        for node in authority_calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "v4_authoritative_primitive_step"
    ]
    nested_source_calls = [
        node
        for node in authority_calls
        if isinstance(node.func, ast.Name) and node.func.id == "source_physics_step"
    ]
    assert len(direct_calls) == 1
    assert nested_source_calls == []
    direct_keywords = {keyword.arg: keyword.value for keyword in direct_calls[0].keywords}
    assert ast.unparse(direct_keywords["mjx_step"]) == "joystick.mjx_env.mjx.step"

    all_guarded_calls = [
        node for node in ast.walk(guarded) if isinstance(node, ast.Call)
    ]
    authoritative_calls = [
        node
        for node in all_guarded_calls
        if isinstance(node.func, ast.Name) and node.func.id == "source_physics_step"
    ]
    assert len(authoritative_calls) == 1
    assert len(authoritative_calls[0].args) == 4
    assert ast.unparse(authoritative_calls[0].args[3]) == "n_substeps"
    legacy_branch = next(
        node
        for node in ast.walk(guarded)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not forward_v4_substep_contact"
    )
    assert authoritative_calls[0] in list(ast.walk(legacy_branch))

    physics_scan = next(
        node
        for node in all_guarded_calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "scan_v4_instrumented_physics_trajectory"
    )
    physics_keywords = {keyword.arg: keyword.value for keyword in physics_scan.keywords}
    assert ast.unparse(physics_keywords["single_physics_step"]) == (
        "authoritative_single_step"
    )
    assert all(
        not (
            isinstance(node.func, ast.Name)
            and node.func.id
            in {
                "coherent_replay_measurement_state",
                "replay_force_measurement",
                "scan_v4_saved_state_contact_telemetry",
            }
        )
        for node in ast.walk(authority_body)
        if isinstance(node, ast.Call)
    )

    telemetry_scan = next(
        node
        for node in all_guarded_calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "scan_v4_saved_state_contact_telemetry"
    )
    telemetry_keywords = {
        keyword.arg: keyword.value for keyword in telemetry_scan.keywords
    }
    assert ast.unparse(telemetry_keywords["cohere_measurement_state"]) == (
        "coherent_replay_measurement_state"
    )
    assert ast.unparse(telemetry_scan.args[0]) == "data"
    assert ast.unparse(telemetry_scan.args[1]) == "saved_dynamic_states"
    assert physics_scan.lineno < telemetry_scan.lineno
    returned_names = {
        ast.unparse(node.value)
        for node in ast.walk(guarded)
        if isinstance(node, ast.Return) and node.value is not None
    }
    assert "authority_data" in returned_names


def test_v4_scan_jaxprs_reject_nested_length_one_shadow_wrapper() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")

    def primitive(carry):
        return carry + jp.asarray(1, dtype=carry.dtype)

    def authoritative_single_scan(initial):
        return jax.lax.scan(
            lambda carry, _unused: (primitive(carry), None),
            initial,
            xs=None,
            length=V4_CONTROL_SUBSTEP_COUNT,
        )[0]

    def shadow_outer_scan_direct_primitive(initial):
        return jax.lax.scan(
            lambda carry, _unused: (primitive(carry), None),
            initial,
            xs=None,
            length=V4_CONTROL_SUBSTEP_COUNT,
        )[0]

    def rejected_nested_length_one_wrapper(carry):
        return jax.lax.scan(
            lambda inner, _unused: (primitive(inner), None),
            carry,
            xs=None,
            length=1,
        )[0]

    def shadow_outer_scan_nested_wrapper(initial):
        return jax.lax.scan(
            lambda carry, _unused: (
                rejected_nested_length_one_wrapper(carry),
                None,
            ),
            initial,
            xs=None,
            length=V4_CONTROL_SUBSTEP_COUNT,
        )[0]

    def scan_lengths(closed_jaxpr):
        lengths = []

        def visit(value):
            jaxpr = getattr(value, "jaxpr", None)
            if jaxpr is not None:
                visit(jaxpr)
                return
            equations = getattr(value, "eqns", None)
            if equations is not None:
                for equation in equations:
                    if equation.primitive.name == "scan":
                        lengths.append(int(equation.params["length"]))
                    for parameter in equation.params.values():
                        visit(parameter)
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        visit(closed_jaxpr)
        return lengths

    example = jp.asarray(0, dtype=jp.int32)
    authoritative_structure = scan_lengths(
        jax.make_jaxpr(authoritative_single_scan)(example)
    )
    shadow_direct_structure = scan_lengths(
        jax.make_jaxpr(shadow_outer_scan_direct_primitive)(example)
    )
    rejected_nested_structure = scan_lengths(
        jax.make_jaxpr(shadow_outer_scan_nested_wrapper)(example)
    )

    assert authoritative_structure == [V4_CONTROL_SUBSTEP_COUNT]
    assert shadow_direct_structure == [V4_CONTROL_SUBSTEP_COUNT]
    assert rejected_nested_structure == [V4_CONTROL_SUBSTEP_COUNT, 1]


def test_v4_instrumented_physics_uses_exact_source_positional_scan_shape() -> None:
    source = textwrap.dedent(
        inspect.getsource(scan_v4_instrumented_physics_trajectory)
    )
    function = ast.parse(source).body[0]
    compiled_scan = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "scan"
    )

    assert compiled_scan.keywords == []
    assert len(compiled_scan.args) == 4
    assert ast.unparse(compiled_scan.args[0]) == "body"
    assert ast.unparse(compiled_scan.args[1]) == "initial_data"
    assert isinstance(compiled_scan.args[2], ast.Tuple)
    assert compiled_scan.args[2].elts == []
    assert ast.unparse(compiled_scan.args[3]) == "V4_CONTROL_SUBSTEP_COUNT"

    physics_body = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.FunctionDef) and node.name == "body"
    )
    physics_body_calls = {
        node.func.id
        for node in ast.walk(physics_body)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert physics_body_calls == {"single_physics_step", "save_v4_dynamic_state"}


def test_v4_dynamic6_physics_then_saved_state_telemetry_are_separate() -> None:
    entry = _fake_v4_control_entry()
    action = np.asarray([0.25], dtype=np.float32)
    events = []

    def step(data, control):
        events.append("physics")
        return data.replace(
            qpos=data.qpos + np.asarray([1.0, -1.0], dtype=np.float32),
            qvel=data.qvel + np.asarray([0.5], dtype=np.float32),
            ctrl=np.asarray(control, dtype=np.float32),
            time=data.time + np.asarray(0.002, dtype=np.float32),
            qacc_warmstart=(
                data.qacc_warmstart + np.asarray([0.125], dtype=np.float32)
            ),
        )

    endpoint, saved = scan_v4_instrumented_physics_trajectory(
        entry,
        action,
        single_physics_step=step,
    )
    assert events == ["physics"] * V4_CONTROL_SUBSTEP_COUNT

    measured_positions = []

    def cohere(replay_data):
        events.append("forward")
        np.testing.assert_array_equal(replay_data.static_marker, [73.0])
        np.testing.assert_array_equal(replay_data.derived_position, [-99.0])
        return replay_data.replace(derived_position=replay_data.qpos[:1])

    def measure(coherent_data, _previous_raw):
        events.append("measure")
        measured_positions.append(float(coherent_data.derived_position[0]))
        return np.asarray([0.0, 0.0], dtype=np.float32)

    summary = scan_v4_saved_state_contact_telemetry(
        entry,
        saved,
        initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32)),
        cohere_measurement_state=cohere,
        measure_normalized_force=measure,
    )

    assert events[:V4_CONTROL_SUBSTEP_COUNT] == ["physics"] * 10
    assert events[V4_CONTROL_SUBSTEP_COUNT:] == ["forward", "measure"] * 10
    assert measured_positions == list(np.arange(1.0, 11.0))
    np.testing.assert_array_equal(endpoint.qpos, [10.0, -10.0])
    np.testing.assert_array_equal(endpoint.qvel, [5.0])
    np.testing.assert_array_equal(endpoint.ctrl, action)
    assert float(endpoint.time) == pytest.approx(0.020000001)
    np.testing.assert_array_equal(endpoint.qacc_warmstart, [1.25])
    np.testing.assert_array_equal(endpoint.derived_position, [-99.0])
    np.testing.assert_array_equal(entry.qpos, [0.0, 0.0])
    np.testing.assert_array_equal(summary.touchdown_event_count, [0, 0])

    assert saved._fields == (
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "time",
        "qacc_warmstart",
    )
    assert saved.qpos.shape == (10, 2)
    assert saved.qvel.shape == (10, 1)
    assert saved.act.shape == (10, 0)
    assert saved.ctrl.shape == (10, 1)
    assert saved.time.shape == (10,)
    assert saved.qacc_warmstart.shape == (10, 1)
    for field in saved._fields:
        values = np.asarray(getattr(saved, field))
        assert np.all(np.isfinite(values))
        np.testing.assert_array_equal(values[-1], getattr(endpoint, field))
    assert sum(np.asarray(value).nbytes for value in saved) == 240


def test_v4_saved_state_quality_trajectory_is_post_physics_only() -> None:
    entry = _fake_v4_control_entry()
    action = np.asarray([0.25], dtype=np.float32)
    events = []

    def step(data, control):
        events.append("physics")
        return data.replace(
            qpos=data.qpos + np.asarray([1.0, -1.0], dtype=np.float32),
            qvel=data.qvel + np.asarray([0.5], dtype=np.float32),
            ctrl=np.asarray(control, dtype=np.float32),
            time=data.time + np.asarray(0.002, dtype=np.float32),
        )

    endpoint, saved = scan_v4_instrumented_physics_trajectory(
        entry, action, single_physics_step=step
    )

    def cohere(replay_data):
        events.append("forward")
        return replay_data.replace(derived_position=replay_data.qpos[:1])

    def measure(coherent_data):
        events.append("measure")
        return (
            np.asarray((coherent_data.derived_position[0], 0.25), dtype=np.float32),
            np.asarray((coherent_data.qvel[0], 0.5), dtype=np.float32),
        )

    trajectory = scan_v4_saved_state_contact_quality_trajectory(
        entry,
        saved,
        cohere_measurement_state=cohere,
        measure_force_and_tangential_speed=measure,
    )

    assert events[:V4_CONTROL_SUBSTEP_COUNT] == ["physics"] * 10
    assert events[V4_CONTROL_SUBSTEP_COUNT:] == ["forward", "measure"] * 10
    np.testing.assert_allclose(
        trajectory.time_s, np.arange(1, 11, dtype=np.float32) * 0.002
    )
    np.testing.assert_allclose(
        trajectory.normalized_normal_force[:, 0],
        np.arange(1, 11, dtype=np.float32),
    )
    np.testing.assert_allclose(
        trajectory.normalized_normal_force[:, 1], 0.25,
    )
    np.testing.assert_allclose(
        trajectory.tangential_speed_m_s[:, 0],
        np.arange(1, 11, dtype=np.float32) * 0.5,
    )
    np.testing.assert_allclose(trajectory.tangential_speed_m_s[:, 1], 0.5)
    np.testing.assert_array_equal(endpoint.qpos, [10.0, -10.0])


def test_v4_collector_trace_reuses_telemetry_measurement_once() -> None:
    entry = _fake_v4_control_entry()
    action = np.asarray([0.25], dtype=np.float32)
    events = []

    def step(data, control):
        events.append("physics")
        return data.replace(
            qpos=data.qpos + np.asarray([1.0, -1.0], dtype=np.float32),
            qvel=data.qvel + np.asarray([0.5], dtype=np.float32),
            ctrl=np.asarray(control, dtype=np.float32),
            time=data.time + np.asarray(0.002, dtype=np.float32),
        )

    _endpoint, saved = scan_v4_instrumented_physics_trajectory(
        entry, action, single_physics_step=step
    )

    def cohere(replay_data):
        events.append("forward")
        return replay_data.replace(derived_position=replay_data.qpos[:1])

    def measure(coherent_data, _previous_raw):
        events.append("measure")
        return (
            np.asarray((coherent_data.derived_position[0], 0.25), dtype=np.float32),
            np.asarray((coherent_data.qvel[0], 0.5), dtype=np.float32),
        )

    summary, trace = scan_v4_saved_state_contact_telemetry_with_quality_trace(
        entry,
        saved,
        initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32)),
        cohere_measurement_state=cohere,
        measure_force_and_tangential_speed=measure,
    )

    assert events[:V4_CONTROL_SUBSTEP_COUNT] == ["physics"] * 10
    assert events[V4_CONTROL_SUBSTEP_COUNT:] == ["forward", "measure"] * 10
    np.testing.assert_allclose(trace.time_s, np.arange(1, 11, dtype=np.float32) * 0.002)
    np.testing.assert_allclose(trace.normalized_normal_force[:, 0], np.arange(1, 11))
    np.testing.assert_allclose(trace.tangential_speed_m_s[:, 0], np.arange(1, 11) * 0.5)
    np.testing.assert_array_equal(summary.touchdown_event_count, [0, 0])


def test_v4_dynamic6_reconstruction_uses_entry_template_and_exact_field_order() -> None:
    entry = _fake_v4_control_entry()
    saved = V4SavedDynamicState(
        qpos=np.asarray([1.0, 2.0], dtype=np.float32),
        qvel=np.asarray([3.0], dtype=np.float32),
        act=np.asarray([], dtype=np.float32),
        ctrl=np.asarray([4.0], dtype=np.float32),
        time=np.asarray(5.0, dtype=np.float32),
        qacc_warmstart=np.asarray([6.0], dtype=np.float32),
    )
    rebuilt = reconstruct_v4_dynamic_state(entry, saved)

    for field in saved._fields:
        np.testing.assert_array_equal(getattr(rebuilt, field), getattr(saved, field))
    np.testing.assert_array_equal(rebuilt.static_marker, entry.static_marker)
    np.testing.assert_array_equal(rebuilt.derived_position, entry.derived_position)
    resaved = save_v4_dynamic_state(rebuilt)
    for field in saved._fields:
        np.testing.assert_array_equal(getattr(resaved, field), getattr(saved, field))


def test_v4_source_semantic_preflight_qualifies_only_dynamic6() -> None:
    @dataclass(frozen=True)
    class Derived:
        cfrc_int: np.ndarray
        cfrc_ext: np.ndarray

    @dataclass(frozen=True)
    class Data:
        qpos: np.ndarray
        qvel: np.ndarray
        act: np.ndarray
        ctrl: np.ndarray
        time: np.ndarray
        qacc_warmstart: np.ndarray
        _impl: Derived

        def replace(self, **changes):
            return replace(self, **changes)

    initial = Data(
        qpos=np.zeros(1, dtype=np.float32),
        qvel=np.zeros(1, dtype=np.float32),
        act=np.zeros(1, dtype=np.float32),
        ctrl=np.zeros(1, dtype=np.float32),
        time=np.asarray(0.0, dtype=np.float32),
        qacc_warmstart=np.zeros(1, dtype=np.float32),
        _impl=Derived(
            cfrc_int=np.zeros(1, dtype=np.float32),
            cfrc_ext=np.zeros(1, dtype=np.float32),
        ),
    )
    action = np.asarray([0.25], dtype=np.float32)
    source_calls = []

    def primitive(_model, data):
        return data.replace(
            qpos=data.qpos + np.float32(1.0),
            qvel=data.qvel + np.float32(0.5),
            time=data.time + np.float32(0.002),
            qacc_warmstart=data.qacc_warmstart + np.float32(0.125),
        )

    def source(model, data, control, n_substeps):
        source_calls.append(n_substeps)
        for _ in range(n_substeps):
            data = primitive(model, data.replace(ctrl=control))
        # These post-solver force leaves are observed truthfully but do not
        # qualify the single-authority dynamic-state semantic gate.
        return data.replace(
            _impl=Derived(
                cfrc_int=np.asarray([1.0e-6], dtype=np.float32),
                cfrc_ext=np.asarray([-2.0e-6], dtype=np.float32),
            )
        )

    def positional_scan(body, carry, xs, length):
        assert xs == ()
        assert length == V4_CONTROL_SUBSTEP_COUNT
        outputs = []
        for _ in range(length):
            carry, output = body(carry, None)
            outputs.append(output)
        return carry, V4SavedDynamicState(
            *(
                np.stack([getattr(output, field) for output in outputs])
                for field in V4SavedDynamicState._fields
            )
        )

    candidate, saved, report = audit_v4_source_semantic_reference(
        "model",
        initial,
        action,
        source_physics_step=source,
        mjx_step=primitive,
        scan=positional_scan,
        xp=np,
    )

    assert source_calls == [V4_CONTROL_SUBSTEP_COUNT]
    assert report._fields == (
        "dynamic6_exact",
        "dynamic6_max_abs_error",
        "dynamic6_field_count",
        "derived_cfrc_int_exact",
        "derived_cfrc_int_max_abs_error",
        "derived_cfrc_ext_exact",
        "derived_cfrc_ext_max_abs_error",
    )
    assert bool(report.dynamic6_exact)
    assert float(report.dynamic6_max_abs_error) == 0.0
    assert report.dynamic6_field_count == 6
    assert not bool(report.derived_cfrc_int_exact)
    assert float(report.derived_cfrc_int_max_abs_error) == pytest.approx(1.0e-6)
    assert not bool(report.derived_cfrc_ext_exact)
    assert float(report.derived_cfrc_ext_max_abs_error) == pytest.approx(2.0e-6)
    assert bool(audit_v4_dynamic_endpoint_self_consistency(candidate, saved).exact)


def test_v4_old_interleaved_measurement_order_is_rejected_conceptually() -> None:
    old_order = []
    for _ in range(V4_CONTROL_SUBSTEP_COUNT):
        old_order.extend(("physics", "forward", "measure"))
    new_order = ["physics"] * V4_CONTROL_SUBSTEP_COUNT + [
        event
        for _ in range(V4_CONTROL_SUBSTEP_COUNT)
        for event in ("forward", "measure")
    ]

    assert old_order != new_order
    assert old_order[:4] == ["physics", "forward", "measure", "physics"]
    assert new_order[:V4_CONTROL_SUBSTEP_COUNT] == ["physics"] * 10


def test_v4_same_endpoint_intra_tick_traces_remain_distinguishable() -> None:
    quiet_trace = np.zeros((V4_CONTROL_SUBSTEP_COUNT, 2), dtype=np.float32)
    chatter_trace = quiet_trace.copy()
    chatter_trace[0, 0] = 0.02

    quiet_endpoint, quiet = _run_numpy_v4_two_phase(quiet_trace)
    chatter_endpoint, chatter = _run_numpy_v4_two_phase(chatter_trace)

    assert int(quiet_endpoint) == int(chatter_endpoint) == 10
    np.testing.assert_array_equal(
        quiet.state.persistence.raw_contact,
        chatter.state.persistence.raw_contact,
    )
    np.testing.assert_array_equal(
        quiet.state.persistence.qualified_contact,
        chatter.state.persistence.qualified_contact,
    )
    assert int(quiet.aborted_contact_island_count[0]) == 0
    assert int(chatter.aborted_contact_island_count[0]) == 1
    assert float(chatter.aborted_contact_island_loss_sum[0]) == pytest.approx(1.0)


def test_v4_paired_aborts_sum_to_two_without_event_averaging() -> None:
    trace = np.zeros((V4_CONTROL_SUBSTEP_COUNT, 2), dtype=np.float32)
    trace[0] = 0.02
    _, summary = _run_numpy_v4_two_phase(trace)

    np.testing.assert_array_equal(summary.aborted_contact_island_count, [1, 1])
    np.testing.assert_array_equal(
        summary.aborted_contact_island_loss_sum, [1.0, 1.0]
    )
    assert float(np.sum(summary.aborted_contact_island_loss_sum)) == pytest.approx(
        2.0
    )


def test_v4_multiple_qualified_alternations_sum_within_one_tick() -> None:
    initial = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    initial = initial._replace(
        persistence=V4ContactPersistenceState(
            raw_contact=np.asarray([True, True]),
            qualified_contact=np.asarray([False, False]),
            pending_active=np.asarray([True, True]),
            pending_target=np.asarray([True, True]),
            pending_intervals=np.asarray([19, 18], dtype=np.int32),
        ),
        last_touchdown_foot=np.asarray(1, dtype=np.int32),
        intervals_since_touchdown=np.asarray(149, dtype=np.int32),
    )
    shadow_endpoint, summary = scan_v4_contact_telemetry_two_phase_reference(
        np.asarray(0, dtype=np.int32),
        np.asarray(1, dtype=np.int32),
        initial,
        single_physics_step=lambda data, action: data + action,
        cohere_measurement_state=lambda data: data,
        measure_normalized_force=lambda _data, _raw: np.asarray(
            [0.02, 0.02], dtype=np.float32
        ),
    )

    assert int(shadow_endpoint) == V4_CONTROL_SUBSTEP_COUNT
    np.testing.assert_array_equal(summary.touchdown_event_count, [1, 1])
    assert int(summary.alternation_event_count) == 2
    assert float(summary.alternation_quality_sum) > 1.0


@pytest.mark.parametrize("mutated_field", V4SavedDynamicState._fields)
def test_v4_authority_endpoint_matches_every_saved_dynamic6_field_exactly(
    mutated_field,
) -> None:
    # Give ``act`` one element in this mutation test; the production robot may
    # legitimately have either a populated or a zero-width activation vector.
    entry = replace(
        _fake_v4_control_entry(), act=np.zeros(1, dtype=np.float32)
    )
    endpoint, saved = scan_v4_instrumented_physics_trajectory(
        entry,
        np.asarray([0.25], dtype=np.float32),
        single_physics_step=lambda data, action: data.replace(
            qpos=data.qpos + np.asarray([1.0, -1.0], dtype=np.float32),
            ctrl=action,
            time=data.time + np.asarray(0.002, dtype=np.float32),
        ),
    )
    parity = audit_v4_dynamic_endpoint_self_consistency(endpoint, saved)
    assert bool(parity.exact)
    assert float(parity.max_abs_error) == 0.0
    assert parity.leaf_count == 6
    assert bool(v4_saved_dynamic_trajectory_all_finite(saved))

    changed_values = np.asarray(getattr(saved, mutated_field)).copy()
    changed_values.reshape(V4_CONTROL_SUBSTEP_COUNT, -1)[-1, 0] += np.float32(1.0)
    mutated = saved._replace(**{mutated_field: changed_values})
    negative = audit_v4_dynamic_endpoint_self_consistency(
        endpoint, mutated
    )
    assert not bool(negative.exact)
    assert float(negative.max_abs_error) == pytest.approx(1.0)
    assert negative.leaf_count == 6


def test_v4_saved_state_telemetry_rejects_non_two_foot_force_shape() -> None:
    entry = _fake_v4_control_entry()
    _, saved = scan_v4_instrumented_physics_trajectory(
        entry,
        np.asarray([0.0], dtype=np.float32),
        single_physics_step=lambda data, action: data.replace(ctrl=action),
    )
    with pytest.raises(ValueError, match="normalized_force must have shape \\(2,\\)"):
        scan_v4_saved_state_contact_telemetry(
            entry,
            saved,
            initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32)),
            cohere_measurement_state=lambda data: data,
            measure_normalized_force=lambda _data, _raw: np.zeros(
                3, dtype=np.float32
            ),
        )


def test_v4_measurement_forward_fixes_stale_state_without_carry_mutation() -> None:
    @dataclass(frozen=True)
    class FakeDerivedData:
        position: np.int32
        derived_position: np.int32

    def run(cohere):
        measured_positions = []

        def measure(data, _raw):
            measured_positions.append(int(data.derived_position))
            return np.asarray(
                [0.02 if data.derived_position > 0 else 0.0, 0.0],
                dtype=np.float32,
            )

        endpoint, summary = scan_v4_contact_telemetry_two_phase_reference(
            FakeDerivedData(np.int32(0), np.int32(0)),
            np.int32(1),
            initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32)),
            single_physics_step=lambda data, action: replace(
                data, position=np.int32(data.position + action)
            ),
            cohere_measurement_state=cohere,
            measure_normalized_force=measure,
        )
        return endpoint, summary, measured_positions

    stale_endpoint, stale, stale_measurements = run(lambda data: data)
    coherent_endpoint, coherent, coherent_measurements = run(
        lambda data: replace(data, derived_position=data.position)
    )

    assert stale_endpoint == coherent_endpoint == FakeDerivedData(
        np.int32(10), np.int32(0)
    )
    assert stale_measurements == [0] * V4_CONTROL_SUBSTEP_COUNT
    assert coherent_measurements == list(range(1, V4_CONTROL_SUBSTEP_COUNT + 1))
    np.testing.assert_array_equal(stale.state.persistence.raw_contact, [False, False])
    np.testing.assert_array_equal(
        coherent.state.persistence.raw_contact, [True, False]
    )
    # The measurement-only forward must never leak into the trajectory carry.
    assert int(coherent_endpoint.derived_position) == 0


def test_v4_actual_mjx_step_contact_is_stale_until_measurement_forward() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    mujoco = pytest.importorskip("mujoco")
    mjx = pytest.importorskip("mujoco.mjx")
    mjx_support = pytest.importorskip("mujoco.mjx._src.support")
    cpu_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.02" gravity="0 0 0"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 .1"/>
            <body>
              <freejoint/>
              <geom name="ball" type="sphere" size=".1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    model = mjx.put_model(cpu_model)
    data = mjx.make_data(model)
    data = mjx.forward(
        model,
        data.replace(
            qpos=data.qpos.at[2].set(jp.asarray(0.15)),
            qvel=data.qvel.at[2].set(jp.asarray(-5.0)),
        ),
    )
    stepped = mjx.step(model, data)
    coherent = mjx.forward(model, stepped)
    jax.block_until_ready(coherent.qpos)

    # Forward recomputes only derived state: the integrated trajectory is
    # bit-identical, while contact distance/force move to the current qpos.
    np.testing.assert_array_equal(np.asarray(stepped.qpos), np.asarray(coherent.qpos))
    np.testing.assert_array_equal(np.asarray(stepped.qvel), np.asarray(coherent.qvel))
    stale_distance = float(np.asarray(stepped._impl.contact.dist)[0])
    coherent_distance = float(np.asarray(coherent._impl.contact.dist)[0])
    stale_force = float(np.asarray(mjx_support.contact_force(model, stepped, 0))[0])
    coherent_force = float(
        np.asarray(mjx_support.contact_force(model, coherent, 0))[0]
    )
    assert stale_distance > 0.0
    assert coherent_distance < 0.0
    assert stale_force == pytest.approx(0.0)
    assert coherent_force > 0.0


def test_v4_actual_mjx_varied_state_compiled_ten_step_parity_is_exact() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    mujoco = pytest.importorskip("mujoco")
    mjx = pytest.importorskip("mujoco.mjx")
    source_mjx_env = pytest.importorskip("mujoco_playground._src.mjx_env")
    cpu_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 .1"/>
            <body>
              <freejoint/>
              <geom name="ball" type="sphere" size=".1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    model = mjx.put_model(cpu_model)
    base = mjx.make_data(model)
    states = []
    for x, height, vertical_speed in (
        (-0.20, 0.105, -2.0),
        (-0.05, 0.115, -0.5),
        (0.10, 0.150, 0.5),
        (0.25, 0.300, -0.25),
    ):
        qpos = base.qpos.at[0].set(x).at[2].set(height)
        qvel = base.qvel.at[2].set(vertical_speed)
        states.append(mjx.forward(model, base.replace(qpos=qpos, qvel=qvel)))
    batched_states = jax.tree_util.tree_map(
        lambda *values: jp.stack(values), *states
    )
    action = jp.zeros(model.nu, dtype=base.ctrl.dtype)
    assertion = make_v4_compiled_single_authority_assertion(jax, jp)

    def compare(initial_data):
        # This is the actual upstream authoritative wrapper: one length-ten
        # lax.scan whose body replaces ctrl then calls its imported mjx.step.
        authority, saved_dynamic_states, source_preflight = (
            audit_v4_source_semantic_reference(
                model,
                initial_data,
                action,
                source_physics_step=source_mjx_env.step,
                mjx_step=source_mjx_env.mjx.step,
                scan=jax.lax.scan,
                xp=jp,
            )
        )
        summary = scan_v4_saved_state_contact_telemetry(
            initial_data,
            saved_dynamic_states,
            initialize_v4_contact_telemetry(
                jp.zeros(2, dtype=jp.float32), xp=jp
            ),
            cohere_measurement_state=lambda data: source_mjx_env.mjx.forward(
                model, data
            ),
            measure_normalized_force=lambda data, _raw: jp.stack(
                (
                    jp.where(data._impl.contact.dist[0] <= 0.0, 0.02, 0.0),
                    jp.asarray(0.0, dtype=jp.float32),
                )
            ),
            scan=jax.lax.scan,
            xp=jp,
        )
        parity = audit_v4_dynamic_endpoint_self_consistency(
            authority, saved_dynamic_states, xp=jp
        )
        saved_finite = v4_saved_dynamic_trajectory_all_finite(
            saved_dynamic_states, xp=jp
        )
        runtime_exact = (
            parity.exact & saved_finite & summary.normalized_force_finite
        )
        token = assertion(
            runtime_exact, parity.max_abs_error, parity.leaf_count
        )
        return (
            token,
            parity.exact,
            parity.max_abs_error,
            parity.leaf_count,
            saved_finite,
            summary.normalized_force_finite,
            source_preflight.dynamic6_exact,
            source_preflight.dynamic6_max_abs_error,
            source_preflight.dynamic6_field_count,
            source_preflight.derived_cfrc_int_max_abs_error,
            source_preflight.derived_cfrc_ext_max_abs_error,
        )

    (
        tokens,
        exact,
        errors,
        field_counts,
        saved_finite,
        force_finite,
        source_exact,
        source_errors,
        source_field_counts,
        cfrc_int_errors,
        cfrc_ext_errors,
    ) = jax.jit(jax.vmap(compare))(batched_states)
    jax.block_until_ready(tokens)
    np.testing.assert_array_equal(np.asarray(tokens), np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(exact), np.ones(4, dtype=bool))
    np.testing.assert_array_equal(np.asarray(errors), np.zeros(4))
    np.testing.assert_array_equal(np.asarray(field_counts), np.full(4, 6))
    np.testing.assert_array_equal(np.asarray(saved_finite), np.ones(4, dtype=bool))
    np.testing.assert_array_equal(np.asarray(force_finite), np.ones(4, dtype=bool))
    np.testing.assert_array_equal(np.asarray(source_exact), np.ones(4, dtype=bool))
    np.testing.assert_array_equal(np.asarray(source_errors), np.zeros(4))
    np.testing.assert_array_equal(np.asarray(source_field_counts), np.full(4, 6))
    assert np.all(np.isfinite(np.asarray(cfrc_int_errors)))
    assert np.all(np.isfinite(np.asarray(cfrc_ext_errors)))


def test_v4_official_openduck_seed20260809_first_jit_is_exact(
) -> None:
    official_root = Path("/home/user/openduck_training_20260729").resolve()
    official_python = official_root / ".venv" / "bin" / "python"
    if not official_root.is_dir():
        pytest.skip("frozen OpenDuck source root is unavailable")
    generated_root = EXPERIMENT_ROOT / "artifacts" / "generated_playground"
    if not generated_root.is_dir():
        pytest.skip("generated calibrated OpenDuck package is unavailable")
    if not official_python.is_file():
        pytest.skip("frozen OpenDuck Python environment is unavailable")
    try:
        gpu_preflight = subprocess.run(
            ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader"),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("CUDA GPU preflight is unavailable")
    if gpu_preflight.returncode != 0 or not gpu_preflight.stdout.strip():
        pytest.skip("CUDA GPU is unavailable")

    # The sibling pure AST test gives the detailed scan-shape failure, while
    # this assertion binds the production regression to the same no-forward
    # physics body without changing the child compiler context.
    function = ast.parse(
        textwrap.dedent(inspect.getsource(scan_v4_instrumented_physics_trajectory))
    ).body[0]
    physics_body = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.FunctionDef) and node.name == "body"
    )
    physics_calls = {
        node.func.id
        for node in ast.walk(physics_body)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert physics_calls == {"single_physics_step", "save_v4_dynamic_state"}

    # Pytest/plugin startup itself changes the JAX compiler context on 0.5.3.
    # Run the byte-equivalent, already-audited production graph in a fresh
    # official interpreter with XLA selection fixed before startup.
    production_script = r'''
from pathlib import Path
import hashlib
import numpy as np
import scripts.train_expert as trainer
import scripts.train_h4_aligned_expert as runner
from safe_gait_experts.h4_training_alignment import (
    make_anchor_command_mapper,
    make_h4_aligned_environment_class,
    make_h4_forward_v2_physical_sampler,
)

experiment_root = Path.cwd().resolve()
source_root = Path("/home/user/openduck_training_20260729").resolve()
paths = trainer.generated_paths(
    experiment_root / "artifacts" / "generated_playground"
)
trainer._validate_generated_manifest(paths)
stack = trainer._load_training_stack(source_root)
jax, jp = stack["jax"], stack["jp"]
joystick_path = Path(stack["joystick"].__file__).resolve()
resolved_backend = jax.default_backend()
resolved_devices = tuple(jax.devices())
print(
    "LOADER",
    jax.__version__,
    resolved_backend,
    joystick_path,
    hashlib.sha256(joystick_path.read_bytes()).hexdigest(),
    resolved_devices,
    flush=True,
)

constants = stack["constants"]
scene_type = type(constants.FLAT_TERRAIN_BACKLASH_CALIBRATED_XML)
constants.ROOT_PATH = scene_type(paths["package"].as_posix())
constants.FLAT_TERRAIN_BACKLASH_CALIBRATED_XML = scene_type(
    paths["scene"].as_posix()
)

class TeacherArgs:
    backward_gait = None
    backward_left_gait = None
    backward_right_gait = None

teachers = trainer.resolve_teacher_gaits(TeacherArgs(), paths)
legacy_environment = trainer._make_environment_class(
    stack=stack,
    expert="forward",
    paths=paths,
    teacher_gaits=teachers,
    backward_residual_scale=0.12,
)
anchors = runner.FORWARD_ITERATION_V2_ANCHOR_CONFIG
environment_class = make_h4_aligned_environment_class(
    legacy_environment_class=legacy_environment,
    stack=stack,
    physical_command_sampler=make_h4_forward_v2_physical_sampler(jax, jp),
    policy_observation_mapper=make_anchor_command_mapper(
        anchors["physical_primary"],
        anchors["policy_observation_anchor"],
        xp=jp,
    ),
    reward_scales=runner.forward_iteration_v2_reward_scales(),
    reset_noise_multiplier=1.0,
    include_h4_actor_observables=True,
    forward_v4_substep_contact=True,
)
environment = environment_class()
state = environment.reset(jax.random.PRNGKey(20260809))
reset_qpos = np.asarray(state.data.qpos)
reset_ctrl = np.asarray(state.data.ctrl)
print("RESET", reset_qpos.shape, reset_ctrl.tolist(), flush=True)
zero_action = jp.zeros(14, dtype=jp.float32)
next_state = jax.jit(environment.step)(state, zero_action)
next_state.reward.block_until_ready()

import json
result = {
    "backend": resolved_backend,
    "exact": bool(np.asarray(next_state.info["h4_v4_single_authority_dynamic6_exact"])),
    "force_finite": bool(
        np.all(np.isfinite(np.asarray(next_state.info["h4_normalized_force"])))
    ),
    "jax_version": jax.__version__,
    "joystick_path": str(joystick_path),
    "joystick_sha256": hashlib.sha256(joystick_path.read_bytes()).hexdigest(),
    "field_count": int(
        np.asarray(next_state.info["h4_v4_single_authority_dynamic6_field_count"])
    ),
    "field_count_exact": bool(
        np.asarray(
            next_state.info[
                "h4_v4_single_authority_dynamic6_field_count_exact"
            ]
        )
    ),
    "max_abs_error": float(
        np.asarray(next_state.info["h4_v4_single_authority_dynamic6_max_abs_error"])
    ),
    "saved_all_finite": bool(
        np.asarray(next_state.info["h4_v4_saved_dynamic6_all_finite"])
    ),
    "saved_field_count": int(
        np.asarray(next_state.info["h4_v4_saved_dynamic6_field_count"])
    ),
    "saved_field_count_exact": bool(
        np.asarray(next_state.info["h4_v4_saved_dynamic6_field_count_exact"])
    ),
    "saved_substep_count": int(
        np.asarray(next_state.info["h4_v4_saved_dynamic6_substep_count"])
    ),
    "reset_ctrl": reset_ctrl.tolist(),
    "reset_qpos_shape": list(reset_qpos.shape),
    "seed": 20260809,
    "token": int(
        np.asarray(next_state.info["h4_v4_single_authority_assertion_token"])
    ),
    "telemetry_force_all_finite": bool(
        np.asarray(next_state.info["h4_v4_telemetry_force_all_finite"])
    ),
    "telemetry_force_shape_valid": bool(
        np.asarray(next_state.info["h4_v4_telemetry_force_shape_valid"])
    ),
    "violation": bool(
        np.asarray(next_state.info["h4_v4_single_authority_violation"])
    ),
    "zero_action_dtype": str(zero_action.dtype),
}
print("H4_PRODUCTION_JSON=" + json.dumps(result, sort_keys=True), flush=True)
'''
    child_environment = os.environ.copy()
    for key in tuple(child_environment):
        if key.startswith("JAX_") or key.startswith("XLA_"):
            child_environment.pop(key)
    child_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "JAX_PLATFORMS": "cuda,cpu",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(EXPERIMENT_ROOT),
            "PYTHONUNBUFFERED": "1",
            "XLA_FLAGS": "--xla_gpu_autotune_level=4",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    completed = subprocess.run(
        (str(official_python), "-c", production_script),
        cwd=EXPERIMENT_ROOT,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=360,
        check=False,
    )
    assert completed.returncode == 0, (
        f"production subprocess failed\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    marker = "H4_PRODUCTION_JSON="
    payload_lines = [
        line[len(marker) :]
        for line in completed.stdout.splitlines()
        if line.startswith(marker)
    ]
    assert len(payload_lines) == 1, completed.stdout
    result = json.loads(payload_lines[0])
    assert result == {
        "backend": "gpu",
        "exact": True,
        "force_finite": True,
        "jax_version": "0.5.3",
        "joystick_path": str(
            official_root / "playground" / "open_duck_mini_v2" / "joystick.py"
        ),
        "joystick_sha256": (
            "95890569d971725308b5a9c0996bfa5fd9520479f014f325e810aa1db272eb9d"
        ),
        "field_count": 6,
        "field_count_exact": True,
        "max_abs_error": 0.0,
        "saved_all_finite": True,
        "saved_field_count": 6,
        "saved_field_count_exact": True,
        "saved_substep_count": 10,
        "reset_ctrl": [
            -0.013402150943875313,
            -0.000871516764163971,
            -0.18906623125076294,
            0.47053399682044983,
            -0.3062592148780823,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0060554388910532,
            -0.046868473291397095,
            0.23326759040355682,
            0.46772077679634094,
            -0.22927279770374298,
        ],
        "reset_qpos_shape": [31],
        "seed": 20260809,
        "token": 0,
        "telemetry_force_all_finite": True,
        "telemetry_force_shape_valid": True,
        "violation": False,
        "zero_action_dtype": "float32",
    }


@pytest.mark.parametrize(
    "parity",
    (
        V4TrajectoryParity(False, 0.0, 6),
        V4TrajectoryParity(True, 1.0e-9, 6),
        V4TrajectoryParity(True, 0.0, 5),
    ),
)
def test_v4_synchronous_parity_assertion_fails_closed(parity) -> None:
    with pytest.raises(RuntimeError, match="single-authority invariant failure"):
        require_v4_single_authority_invariants(parity)
    require_v4_single_authority_invariants(V4TrajectoryParity(True, 0.0, 6))


def test_v4_compiled_parity_passes_and_injected_leaf_mismatch_aborts() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    assertion = make_v4_compiled_single_authority_assertion(jax, jp)

    @jax.jit
    def checked_authority(delta):
        error = jp.abs(delta)
        token = assertion(delta == 0.0, error, jp.asarray(6, dtype=jp.int32))
        return token, error

    pass_token, pass_error = checked_authority(jp.asarray(0.0))
    jax.block_until_ready(pass_token)
    assert int(pass_token) == 0
    assert float(pass_error) == 0.0
    with pytest.raises(Exception, match="single-authority invariant failure"):
        failed_token, _ = checked_authority(jp.asarray(1.0))
        jax.block_until_ready(failed_token)


@pytest.mark.parametrize(
    ("exact", "error", "leaf_count"),
    (
        (False, 0.0, 6),
        (True, 1.0e-9, 6),
        (True, 0.0, 5),
    ),
)
def test_v4_compiled_parity_rejects_every_failure_axis(
    exact, error, leaf_count
) -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    assertion = make_v4_compiled_single_authority_assertion(jax, jp)
    compiled = jax.jit(assertion)
    with pytest.raises(Exception, match="single-authority invariant failure"):
        token = compiled(
            jp.asarray(exact),
            jp.asarray(error, dtype=jp.float32),
            jp.asarray(leaf_count, dtype=jp.int32),
        )
        jax.block_until_ready(token)


def test_v4_vmap_success_has_zero_callbacks_and_failure_aggregates_once() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    calls = []

    def record_failure(exact, error, leaf_count):
        calls.append(
            (
                bool(np.asarray(exact).item()),
                float(np.asarray(error).item()),
                int(np.asarray(leaf_count).item()),
            )
        )

    assertion = make_v4_compiled_single_authority_assertion(
        jax, jp, failure_callback=record_failure
    )
    compiled_batch = jax.jit(jax.vmap(assertion))
    batch_size = 1_250
    exact = jp.ones(batch_size, dtype=bool)
    error = jp.zeros(batch_size, dtype=jp.float32)
    leaf_count = jp.full(batch_size, 6, dtype=jp.int32)

    success_tokens = compiled_batch(exact, error, leaf_count)
    jax.block_until_ready(success_tokens)
    assert np.asarray(success_tokens).shape == (batch_size,)
    assert calls == []

    failed_exact = exact.at[731].set(False)
    failure_tokens = compiled_batch(failed_exact, error, leaf_count)
    jax.block_until_ready(failure_tokens)
    assert np.asarray(failure_tokens).shape == (batch_size,)
    assert calls == [(False, 0.0, 6)]


def test_v4_field_count_exact_metrics_jit_vmap_and_episode_aggregate() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")

    def aggregate(dynamic_counts, saved_counts, episode_weights):
        dynamic_exact, saved_exact = jax.vmap(
            lambda dynamic, saved: v4_dynamic_field_counts_exact(
                dynamic, saved, xp=jp
            )
        )(dynamic_counts, saved_counts)
        length = jp.sum(episode_weights)
        return (
            length,
            jp.sum(dynamic_exact.astype(jp.float32) * episode_weights),
            jp.sum(saved_exact.astype(jp.float32) * episode_weights),
            dynamic_exact,
            saved_exact,
        )

    compiled = jax.jit(aggregate)
    counts = jp.full((1_250,), 6, dtype=jp.int32)
    weights = jp.linspace(0.01, 0.05, 1_250, dtype=jp.float32)
    length, dynamic_total, saved_total, dynamic_exact, saved_exact = compiled(
        counts, counts, weights
    )
    jax.block_until_ready(saved_total)
    assert np.asarray(dynamic_exact).all()
    assert np.asarray(saved_exact).all()
    assert np.asarray(dynamic_total).tobytes() == np.asarray(length).tobytes()
    assert np.asarray(saved_total).tobytes() == np.asarray(length).tobytes()

    mutated = counts.at[731].set(5)
    (
        mutated_length,
        mutated_dynamic_total,
        mutated_saved_total,
        mutated_dynamic_exact,
        mutated_saved_exact,
    ) = compiled(mutated, counts, weights)
    jax.block_until_ready(mutated_saved_total)
    assert not bool(np.asarray(mutated_dynamic_exact)[731])
    assert bool(np.asarray(mutated_saved_exact).all())
    assert float(np.asarray(mutated_dynamic_total)) < float(
        np.asarray(mutated_length)
    )
    assert np.asarray(mutated_saved_total).tobytes() == np.asarray(
        mutated_length
    ).tobytes()


def test_v4_field_count_exact_metrics_are_statically_coupled_to_actual_counts() -> None:
    dynamic_exact, saved_exact = v4_dynamic_field_counts_exact(
        np.int32(6), np.int32(6)
    )
    assert bool(dynamic_exact) is True
    assert bool(saved_exact) is True
    for dynamic_count, saved_count, expected in (
        (5, 6, (False, True)),
        (6, 5, (True, False)),
        (7, 7, (False, False)),
    ):
        actual = v4_dynamic_field_counts_exact(
            np.int32(dynamic_count), np.int32(saved_count)
        )
        assert tuple(bool(value) for value in actual) == expected

    source = inspect.getsource(make_h4_aligned_environment_class)
    assert "v4_dynamic_field_counts_exact(" in source
    assert "| ~field_count_exact" in source
    assert "| ~saved_field_count_exact" in source
    assert (
        '"h4_v4_single_authority_dynamic6_field_count_exact"'
        in source
    )
    assert '"h4_v4_saved_dynamic6_field_count_exact"' in source


@pytest.mark.parametrize(
    "force_trace",
    (
        np.tile(np.asarray([[0.02, 0.0]], dtype=np.float32), (21, 1)),
        np.asarray([[0.02, 0.0], [0.0, 0.0]], dtype=np.float32),
    ),
)
def test_v4_numpy_jax_positive_and_negative_sequence_parity(force_trace) -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    numpy_state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    numpy_touchdowns = []
    numpy_losses = []
    for force in force_trace:
        update = update_v4_contact_telemetry(numpy_state, force)
        numpy_state = update.state
        numpy_touchdowns.append(update.touchdown_event)
        numpy_losses.append(update.aborted_loss)

    def run_jax(forces):
        initial = initialize_v4_contact_telemetry(
            jp.zeros(2, dtype=jp.float32), xp=jp
        )

        def body(state, force):
            update = update_v4_contact_telemetry(state, force, xp=jp)
            return update.state, (update.touchdown_event, update.aborted_loss)

        return jax.lax.scan(body, initial, forces)

    jax_state, (jax_touchdowns, jax_losses) = jax.jit(run_jax)(
        jp.asarray(force_trace)
    )
    np.testing.assert_array_equal(
        np.asarray(jax_state.persistence.raw_contact),
        numpy_state.persistence.raw_contact,
    )
    np.testing.assert_array_equal(
        np.asarray(jax_state.persistence.qualified_contact),
        numpy_state.persistence.qualified_contact,
    )
    np.testing.assert_array_equal(
        np.asarray(jax_state.touchdown_counts), numpy_state.touchdown_counts
    )
    assert np.asarray(jax_state.touchdown_counts).dtype == np.int32
    np.testing.assert_array_equal(np.asarray(jax_touchdowns), numpy_touchdowns)
    np.testing.assert_allclose(np.asarray(jax_losses), numpy_losses, rtol=0.0, atol=0.0)


def test_v4_invalid_state_force_and_substep_count_fail_closed() -> None:
    state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        update_v4_contact_telemetry(state, np.asarray([np.nan, 0.0]))
    malformed = V4ContactPersistenceState(
        np.asarray([True, False]),
        np.asarray([False, False]),
        np.asarray([True, False]),
        np.asarray([False, False]),
        np.asarray([0, 0], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="oppose"):
        update_v4_contact_persistence(malformed, np.zeros(2))
    with pytest.raises(ValueError, match="exactly 10"):
        scan_v4_contact_telemetry_two_phase_reference(
            np.asarray(0),
            np.asarray(1),
            state,
            single_physics_step=lambda data, action: data + action,
            cohere_measurement_state=lambda data: data,
            measure_normalized_force=lambda _data, _raw: np.zeros(2),
            n_substeps=V4_CONTROL_SUBSTEP_COUNT - 1,
        )


def test_force_weighted_contact_point_slip_is_aggregated_per_foot() -> None:
    quality = aggregate_force_contact_quality(
        contact_normal_force_n=np.asarray([2.0, 1.0, 3.0, 99.0]),
        contact_tangential_speed_m_s=np.asarray([0.01, 0.04, 0.02, 9.0]),
        contact_foot_index=np.asarray([0, 0, 1, -1]),
        previous_contact=np.asarray([False, False]),
        robot_weight_n=20.0,
    )

    np.testing.assert_allclose(quality.normalized_force, [0.15, 0.15])
    np.testing.assert_allclose(
        quality.tangential_speed_m_s, [(2 * 0.01 + 1 * 0.04) / 3, 0.02]
    )
    np.testing.assert_array_equal(quality.contact, [True, True])
    expected_rms = np.sqrt((((0.02) ** 2) + 0.02**2) / 2)
    assert quality.slip_rms_m_s == pytest.approx(expected_rms)


def test_alternation_ignores_double_support_and_rewards_opposite_side() -> None:
    first = update_alternation_state(-1, 0, np.asarray([True, False]))
    double = update_alternation_state(
        first.last_single_support,
        first.ticks_since_switch,
        np.asarray([True, True]),
    )
    opposite = update_alternation_state(
        double.last_single_support,
        14,
        np.asarray([False, True]),
    )

    assert first.last_single_support == 0
    assert not first.alternation_event
    assert double.last_single_support == 0
    assert not double.alternation_event
    assert opposite.last_single_support == 1
    assert opposite.alternation_event
    assert opposite.alternation_quality == pytest.approx(1.0)


def test_load_balance_raw_observable_is_independent_of_reward_scale() -> None:
    ema, imbalance = update_load_balance_ema(
        np.zeros(2), np.asarray([1.0, 0.0]), alpha=1.0
    )
    scales = H4QualityRewardScales(
        force_slip=-3.0,
        left_force_slip=-0.4,
        right_force_slip=-0.6,
        single_support=0.5,
        alternation=7.0,
        load_balance=-2.0,
        slew_feasibility=-0.7,
        target_lag=-0.8,
    ).as_reward_scale_dict()

    np.testing.assert_array_equal(ema, [1.0, 0.0])
    assert imbalance == pytest.approx(1.0)
    assert {
        key: scales[key]
        for key in (
            "h4_force_slip",
            "h4_left_force_slip",
            "h4_right_force_slip",
            "h4_single_support",
            "h4_alternation",
            "h4_load_balance",
            "h4_slew_feasibility",
            "h4_target_lag",
        )
    } == {
        "h4_force_slip": -3.0,
        "h4_left_force_slip": -0.4,
        "h4_right_force_slip": -0.6,
        "h4_single_support": 0.5,
        "h4_alternation": 7.0,
        "h4_load_balance": -2.0,
        "h4_slew_feasibility": -0.7,
        "h4_target_lag": -0.8,
    }


def test_forward_v2_curriculum_has_exact_bounded_bins() -> None:
    assert (
        H4_FORWARD_V2_STAND_PROBABILITY,
        H4_FORWARD_V2_EXACT_ENDPOINT_PROBABILITY,
        H4_FORWARD_V2_LOCAL_ANCHOR_PROBABILITY,
        H4_FORWARD_V2_TRANSITION_PROBABILITY,
    ) == (0.05, 0.70, 0.20, 0.05)
    assert sum(
        (
            H4_FORWARD_V2_STAND_PROBABILITY,
            H4_FORWARD_V2_EXACT_ENDPOINT_PROBABILITY,
            H4_FORWARD_V2_LOCAL_ANCHOR_PROBABILITY,
            H4_FORWARD_V2_TRANSITION_PROBABILITY,
        )
    ) == pytest.approx(1.0)

    class FakeRandom:
        def __init__(self, mode: float, anchor_index: int = 1) -> None:
            self.mode = mode
            self.anchor_index = anchor_index

        @staticmethod
        def split(_rng: object, _count: int) -> tuple[str, str, str]:
            return "mode", "anchor", "transition"

        def uniform(self, key: str, **kwargs: float) -> float:
            if key == "mode":
                return self.mode
            return 0.5 * (kwargs["minval"] + kwargs["maxval"])

        def randint(self, *_args: object, **_kwargs: object) -> int:
            return self.anchor_index

    class FakeJax:
        def __init__(self, mode: float, anchor_index: int = 1) -> None:
            self.random = FakeRandom(mode, anchor_index)

    def vx(mode: float, anchor_index: int = 1) -> float:
        sampler = make_h4_forward_v2_physical_sampler(
            FakeJax(mode, anchor_index), np
        )
        return float(sampler(object())[0])

    assert vx(0.01) == 0.0
    assert vx(0.50) == pytest.approx(0.05)
    assert vx(0.80, 0) == pytest.approx(0.04)
    assert vx(0.80, 1) == pytest.approx(0.06)
    assert vx(0.98) == pytest.approx(0.0325)


def test_total_normal_force_losses_match_frozen_gate_boundaries() -> None:
    lower = total_normal_force_quality(np.asarray([0.4, 0.4]))
    upper = total_normal_force_quality(np.asarray([0.6, 0.6]))
    below = total_normal_force_quality(np.asarray([0.3, 0.3]))
    above = total_normal_force_quality(np.asarray([0.8, 0.8]))
    tail = total_normal_force_quality(np.asarray([1.75, 1.75]))

    assert lower.total_normal_force_normalized == pytest.approx(0.8)
    assert lower.band_loss == pytest.approx(0.0)
    assert upper.band_loss == pytest.approx(0.0)
    assert below.band_loss == pytest.approx(1.0)
    assert above.band_loss == pytest.approx(4.0)
    assert above.tail_loss == pytest.approx(0.0)
    assert tail.tail_loss == pytest.approx((0.5 / 3.0) ** 2)


def test_reverse_v2_curriculum_routes_every_authorized_bin() -> None:
    assert sum(
        (
            H4_REVERSE_V2_STAND_PROBABILITY,
            H4_REVERSE_V2_EXACT_ENDPOINT_PROBABILITY,
            H4_REVERSE_V2_LOCAL_ANCHOR_PROBABILITY,
            H4_REVERSE_V2_TRANSITION_PROBABILITY,
        )
    ) == pytest.approx(1.0)

    class FakeRandom:
        def __init__(self, mode: float, anchor_index: int = 1) -> None:
            self.mode = mode
            self.anchor_index = anchor_index

        @staticmethod
        def split(_rng: object, _count: int) -> tuple[str, str, str]:
            return "mode", "anchor", "transition"

        def uniform(self, key: str, **kwargs: float) -> float:
            if key == "mode":
                return self.mode
            return 0.5 * (kwargs["minval"] + kwargs["maxval"])

        def randint(self, *_args: object, **_kwargs: object) -> int:
            return self.anchor_index

    class FakeJax:
        def __init__(self, mode: float, anchor_index: int = 1) -> None:
            self.random = FakeRandom(mode, anchor_index)

    def vx(mode: float, anchor_index: int = 1) -> float:
        sampler = make_h4_reverse_v2_physical_sampler(
            FakeJax(mode, anchor_index), np
        )
        return float(sampler(object())[0])

    assert vx(0.01) == 0.0
    assert vx(0.50) == pytest.approx(-0.05)
    assert vx(0.85, 0) == pytest.approx(-0.06)
    assert vx(0.85, 1) == pytest.approx(-0.04)
    assert vx(0.98) == pytest.approx(-0.0325)


@pytest.mark.parametrize(
    ("run_ticks", "expected"),
    ((1, 0.25), (2, 0.0), (7, 0.0)),
)
def test_contact_pulse_cost_is_liftoff_only_and_40ms_aligned(
    run_ticks: int, expected: float
) -> None:
    update = update_contact_pulse_state(
        np.asarray([True, False]),
        np.asarray([False, False]),
        np.asarray([run_ticks, 0], dtype=np.int32),
    )

    np.testing.assert_array_equal(update.liftoff_event, [True, False])
    np.testing.assert_array_equal(update.contact_run_length_ticks, [0, 0])
    assert update.event_mean_loss == pytest.approx(expected)


def test_contact_pulse_state_has_no_off_gap_penalty_or_stale_carry() -> None:
    continued = update_contact_pulse_state(
        np.asarray([True, False]),
        np.asarray([True, False]),
        np.asarray([1, 99], dtype=np.int32),
    )
    touchdown = update_contact_pulse_state(
        np.asarray([False, False]),
        np.asarray([True, False]),
        continued.contact_run_length_ticks,
    )

    np.testing.assert_array_equal(continued.contact_run_length_ticks, [2, 0])
    assert continued.event_mean_loss == pytest.approx(0.0)
    np.testing.assert_array_equal(touchdown.contact_run_length_ticks, [1, 0])
    assert touchdown.event_mean_loss == pytest.approx(0.0)


def test_anchor_mapper_separates_physical_reward_and_policy_command() -> None:
    mapper = make_anchor_command_mapper(
        (0.05, 0.0, 0.0), (0.10, -0.018, -0.170)
    )
    physical = np.asarray([0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    original = physical.copy()

    policy = mapper(physical)

    np.testing.assert_array_equal(physical, original)
    np.testing.assert_allclose(policy[:3], [0.05, -0.009, -0.085])
    np.testing.assert_array_equal(policy[3:], np.zeros(4))
    np.testing.assert_array_equal(mapper(np.zeros(7)), np.zeros(7))


def test_observation_sync_changes_only_motor_target_slice() -> None:
    observation = {
        "state": np.arange(101, dtype=np.float64),
        "privileged_state": np.arange(180, dtype=np.float64),
    }
    originals = {key: value.copy() for key, value in observation.items()}
    targets = np.linspace(-0.2, 0.2, 14)

    synchronized = synchronize_observation_motor_targets(observation, targets)

    for key in observation:
        np.testing.assert_array_equal(observation[key], originals[key])
        np.testing.assert_array_equal(
            synchronized[key][OBSERVATION_MOTOR_TARGET_SLICE], targets
        )
        np.testing.assert_array_equal(
            synchronized[key][: OBSERVATION_MOTOR_TARGET_SLICE.start],
            originals[key][: OBSERVATION_MOTOR_TARGET_SLICE.start],
        )
        np.testing.assert_array_equal(
            synchronized[key][OBSERVATION_MOTOR_TARGET_SLICE.stop :],
            originals[key][OBSERVATION_MOTOR_TARGET_SLICE.stop :],
        )


def test_post_step_command_sync_fixes_frozen_resample_boundary_legacy101() -> None:
    observation = {
        "state": np.arange(101, dtype=np.float64),
        "privileged_state": np.arange(212, dtype=np.float64),
    }
    originals = {key: value.copy() for key, value in observation.items()}
    physical = np.asarray((-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    policy = np.asarray((0.05, -0.01, 0.02, 0.0, 0.0, 0.0, 0.0))

    synchronized = synchronize_post_step_command_observations(
        observation,
        physical,
        policy,
        include_h4_actor_observables=False,
    )

    for key in observation:
        np.testing.assert_array_equal(observation[key], originals[key])
        np.testing.assert_array_equal(
            synchronized[key][OBSERVATION_POLICY_COMMAND_SLICE], policy
        )
        mask = np.ones(originals[key].shape, dtype=bool)
        mask[OBSERVATION_POLICY_COMMAND_SLICE] = False
        np.testing.assert_array_equal(synchronized[key][mask], originals[key][mask])


def test_post_step_command_sync_updates_h4_physical_extra_separately() -> None:
    observation = {
        "state": np.zeros(116, dtype=np.float64),
        "privileged_state": np.zeros(227, dtype=np.float64),
    }
    physical = np.asarray((0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    policy = np.asarray((0.10, -0.018, -0.170, 0.0, 0.0, 0.0, 0.0))

    synchronized = synchronize_post_step_command_observations(
        observation,
        physical,
        policy,
        include_h4_actor_observables=True,
    )

    for values in synchronized.values():
        np.testing.assert_array_equal(
            values[OBSERVATION_POLICY_COMMAND_SLICE], policy
        )
        np.testing.assert_array_equal(
            values[H4_OBSERVATION_PHYSICAL_COMMAND_SLICE], physical[:3]
        )


def test_reentry_sync_updates_actor_and_every_shifted_critic_imitation_channel() -> None:
    observation = {
        "state": np.zeros(116, dtype=np.float32),
        "privileged_state": np.zeros(227, dtype=np.float32),
    }
    reference = np.arange(40, dtype=np.float32)
    phase = np.asarray((-0.25, 0.75), dtype=np.float32)
    synchronized = synchronize_post_step_imitation_state(
        observation, reference, np.asarray(7.0, dtype=np.float32), phase
    )
    offset = 15
    np.testing.assert_array_equal(synchronized["state"][99:101], phase)
    np.testing.assert_array_equal(synchronized["privileged_state"][99:101], phase)
    np.testing.assert_array_equal(
        synchronized["privileged_state"][
            LEGACY_PRIVILEGED_REFERENCE_SLICE.start
            + offset : LEGACY_PRIVILEGED_REFERENCE_SLICE.stop
            + offset
        ],
        reference,
    )
    assert synchronized["privileged_state"][
        LEGACY_PRIVILEGED_IMITATION_INDEX_SLICE.start + offset
    ] == pytest.approx(7.0)
    np.testing.assert_array_equal(
        synchronized["privileged_state"][
            LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE.start
            + offset : LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE.stop
            + offset
        ],
        phase,
    )


def test_dynamic_body_weight_normalization_is_mass_scale_invariant() -> None:
    masses = np.asarray((0.0, 1.0, 2.0), dtype=np.float64)
    mask = np.asarray((False, True, True))
    gravity = np.asarray((0.0, 0.0, -9.81))
    base_weight = robot_body_weight_n(masses, mask, gravity)
    scaled_weight = robot_body_weight_n(1.1 * masses, mask, gravity)
    assert (0.15 * base_weight) / base_weight == pytest.approx(
        (0.15 * scaled_weight) / scaled_weight
    )


def test_reverse_phase_quality_uses_force_contact_and_priority_reversal() -> None:
    current = np.zeros(14)
    previous = np.zeros(14)
    current[[1, 11, 12, 13]] = 0.02
    previous[[1, 11, 12, 13]] = -0.02
    losses = reverse_phase_conditioned_quality_losses(
        12,
        np.asarray((False, True)),
        np.asarray((0.0, 0.03)),
        current,
        previous,
        np.zeros(14),
    )
    assert losses.right_phase_active
    assert not losses.left_phase_active
    assert losses.phase_force_slip == pytest.approx(4.0)
    assert losses.contact_priority_reversal_lag > 0.0


@dataclass(frozen=True)
class _FakeCount:
    hi: np.ndarray
    lo: np.ndarray


@dataclass(frozen=True)
class _FakeNormalizer:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    summed_variance: dict[str, np.ndarray]
    count: _FakeCount

    def replace(self, **changes):
        return replace(self, **changes)


def _fake_v22_checkpoint() -> list[object]:
    normalizer = _FakeNormalizer(
        mean={
            "state": np.arange(101, dtype=np.float32),
            "privileged_state": np.arange(212, dtype=np.float32),
        },
        std={
            "state": np.ones(101, dtype=np.float32) * 2,
            "privileged_state": np.ones(212, dtype=np.float32) * 3,
        },
        summed_variance={
            "state": np.ones(101, dtype=np.float32) * 20,
            "privileged_state": np.ones(212, dtype=np.float32) * 30,
        },
        count=_FakeCount(np.asarray(0, dtype=np.uint32), np.asarray(10, dtype=np.uint32)),
    )

    def layer(input_width: int, output_width: int) -> dict[str, np.ndarray]:
        return {
            "kernel": np.zeros((input_width, output_width), dtype=np.float32),
            "bias": np.zeros(output_width, dtype=np.float32),
        }

    actor = {
        "params": {
            "hidden_0": layer(101, 512),
            "hidden_1": layer(512, 256),
            "hidden_2": layer(256, 128),
            "hidden_3": layer(128, 28),
        }
    }
    critic = {
        "params": {
            "hidden_0": layer(212, 512),
            "hidden_1": layer(512, 256),
            "hidden_2": layer(256, 128),
            "hidden_3": layer(128, 1),
        }
    }
    return [normalizer, actor, critic]


def test_v22_checkpoint_transplant_inserts_zero_rows_without_moving_tail() -> None:
    source = _fake_v22_checkpoint()
    transplanted, audit = transplant_v22_checkpoint_to_h4_observation(source)
    normalizer, actor, critic = transplanted
    source_actor = source[1]["params"]["hidden_0"]["kernel"]
    source_critic = source[2]["params"]["hidden_0"]["kernel"]
    actor_kernel = actor["params"]["hidden_0"]["kernel"]
    critic_kernel = critic["params"]["hidden_0"]["kernel"]

    assert actor_kernel.shape == (116, 512)
    assert critic_kernel.shape == (227, 512)
    np.testing.assert_array_equal(actor_kernel[:101], source_actor)
    np.testing.assert_array_equal(actor_kernel[101:116], np.zeros((15, 512)))
    np.testing.assert_array_equal(critic_kernel[:101], source_critic[:101])
    np.testing.assert_array_equal(critic_kernel[101:116], np.zeros((15, 512)))
    np.testing.assert_array_equal(critic_kernel[116:], source_critic[101:])
    assert normalizer.mean["state"].shape == (116,)
    assert normalizer.mean["privileged_state"].shape == (227,)
    np.testing.assert_array_equal(normalizer.mean["state"][101:], np.zeros(15))
    np.testing.assert_array_equal(normalizer.std["state"][101:], np.ones(15))
    np.testing.assert_array_equal(
        normalizer.summed_variance["state"][101:], np.ones(15) * 10
    )
    assert audit["new_first_layer_rows_zero_initialized"] is True


def test_checkpoint_width_mismatch_fails_without_explicit_transplant() -> None:
    source = _fake_v22_checkpoint()

    with pytest.raises(ValueError, match="explicit v22 transplant"):
        require_checkpoint_observation_compatibility(
            source, actor_observation_width=H4_ACTOR_OBSERVATION_WIDTH
        )

    transplanted, audit = require_checkpoint_observation_compatibility(
        source,
        actor_observation_width=H4_ACTOR_OBSERVATION_WIDTH,
        allow_explicit_v22_transplant=True,
    )
    assert transplanted[1]["params"]["hidden_0"]["kernel"].shape[0] == 116
    assert audit["transplant_applied"] is True
    assert audit["source_actor_width"] == 101
    assert audit["source_critic_width"] == 212
    assert audit["target_actor_width"] == 116
    assert audit["target_critic_width"] == 227


def test_v22_transplant_repairs_only_audited_tiny_negative_variance() -> None:
    source = _fake_v22_checkpoint()
    source_normalizer = source[0]
    state_variance = source_normalizer.summed_variance["state"].copy()
    privileged_variance = source_normalizer.summed_variance[
        "privileged_state"
    ].copy()
    state_variance[7] = np.float32(-0.001)
    privileged_variance[174] = np.float32(-0.0023)
    source[0] = replace(
        source_normalizer,
        summed_variance={
            "state": state_variance,
            "privileged_state": privileged_variance,
        },
    )

    transplanted, audit = require_checkpoint_observation_compatibility(
        source,
        actor_observation_width=H4_ACTOR_OBSERVATION_WIDTH,
        allow_explicit_v22_transplant=True,
    )
    repaired = transplanted[0].summed_variance
    assert audit["legacy_summed_variance_repair_count"] == 2
    assert audit["legacy_summed_variance_min_before"] == pytest.approx(-0.0023)
    assert audit["legacy_summed_variance_clipped_to_zero"] is True
    assert repaired["state"][7] == 0.0
    assert repaired["privileged_state"][174 + 15] == 0.0
    assert np.all(repaired["state"] >= 0.0)
    assert np.all(repaired["privileged_state"] >= 0.0)
    np.testing.assert_array_equal(
        transplanted[0].mean["state"][:101], source[0].mean["state"]
    )
    np.testing.assert_array_equal(
        transplanted[0].std["state"][:101], source[0].std["state"]
    )


def test_v22_transplant_rejects_variance_below_legacy_repair_floor() -> None:
    source = _fake_v22_checkpoint()
    privileged_variance = source[0].summed_variance["privileged_state"].copy()
    privileged_variance[4] = np.float32(-0.0101)
    source[0] = replace(
        source[0],
        summed_variance={
            **source[0].summed_variance,
            "privileged_state": privileged_variance,
        },
    )
    with pytest.raises(ValueError, match="summed_variance"):
        require_checkpoint_observation_compatibility(
            source,
            actor_observation_width=H4_ACTOR_OBSERVATION_WIDTH,
            allow_explicit_v22_transplant=True,
        )


def test_forward_v6_contact_abort_routes_islands_only_and_keeps_off_gap() -> None:
    island = np.asarray([0.25, 1.0], dtype=np.float32)
    off_gap = np.asarray([3.0, 5.0], dtype=np.float32)

    reward_loss = forward_iteration_v6_contact_abort_island_only_reward_loss(
        island, off_gap
    )
    telemetry = forward_iteration_v6_contact_abort_island_only_telemetry(
        island, off_gap
    )

    assert np.asarray(reward_loss).dtype == np.float32
    assert float(reward_loss) == pytest.approx(1.25)
    assert float(telemetry.reward_loss) == pytest.approx(1.25)
    assert float(telemetry.island_loss) == pytest.approx(1.25)
    assert float(telemetry.off_gap_diagnostic_loss) == pytest.approx(8.0)
    assert float(telemetry.off_gap_reward_contribution) == 0.0
    assert bool(telemetry.routing_exact)


def test_forward_v6_paired_two_foot_islands_sum_without_dilution() -> None:
    paired = forward_iteration_v6_contact_abort_island_only_telemetry(
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([7.0, 11.0], dtype=np.float32),
    )
    off_gap_only = forward_iteration_v6_contact_abort_island_only_telemetry(
        np.zeros(2, dtype=np.float32),
        np.asarray([2.0, 4.0], dtype=np.float32),
    )

    assert float(paired.reward_loss) == pytest.approx(2.0)
    assert float(paired.off_gap_diagnostic_loss) == pytest.approx(18.0)
    assert float(off_gap_only.reward_loss) == 0.0
    assert float(off_gap_only.off_gap_diagnostic_loss) == pytest.approx(6.0)


def test_forward_v6_terminal_censor_adds_no_synthetic_abort_reward() -> None:
    state = initialize_v4_contact_telemetry(np.zeros(2, dtype=np.float32))
    opened = update_v4_contact_telemetry(
        state, np.asarray([FORCE_CONTACT_ON_NORMALIZED, 0.0], dtype=np.float32)
    )
    assert bool(opened.state.persistence.pending_active[0])

    censored = discard_v4_terminal_incomplete(opened.state, True)
    telemetry = forward_iteration_v6_contact_abort_island_only_telemetry(
        np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
    )

    np.testing.assert_array_equal(censored.persistence.pending_active, [False, False])
    assert float(telemetry.reward_loss) == 0.0
    assert bool(telemetry.routing_exact)


def test_forward_v6_routing_jit_preserves_jax_float32_and_zero_off_gap() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    compiled = jax.jit(
        lambda island, off_gap: (
            forward_iteration_v6_contact_abort_island_only_reward_loss(
                island, off_gap, xp=jp
            ),
            forward_iteration_v6_contact_abort_island_only_telemetry(
                island, off_gap, xp=jp
            ).off_gap_reward_contribution,
        )
    )

    reward, contribution = compiled(
        jp.asarray([0.25, 1.0], dtype=jp.float32),
        jp.asarray([100.0, 200.0], dtype=jp.float32),
    )
    assert reward.dtype == contribution.dtype == jp.float32
    assert float(reward) == pytest.approx(1.25)
    assert float(contribution) == 0.0


def test_v6_factory_flags_are_explicit_fail_closed_and_keep_v4_v5_default() -> None:
    signature = inspect.signature(make_h4_aligned_environment_class).parameters
    assert signature["forward_iteration_v6_contact_abort_island_only"].default is False
    assert signature["reverse_iteration_v6_absolute_full_leg_targets"].default is False
    dummy_stack = {"jax": object(), "jp": np, "joystick": object()}
    common = {
        "legacy_environment_class": object,
        "stack": dummy_stack,
        "physical_command_sampler": lambda _rng: np.zeros(7),
        "policy_observation_mapper": lambda command: command,
    }
    with pytest.raises(ValueError, match="families are exclusive"):
        make_h4_aligned_environment_class(
            **common,
            forward_v4_substep_contact=True,
            forward_iteration_v6_contact_abort_island_only=True,
            reverse_iteration_v6_absolute_full_leg_targets=True,
        )
    with pytest.raises(ValueError, match="requires v4 substep contact"):
        make_h4_aligned_environment_class(
            **common,
            forward_iteration_v6_contact_abort_island_only=True,
        )
    with pytest.raises(ValueError, match="scale must remain exactly -1"):
        make_h4_aligned_environment_class(
            **common,
            reward_scales=H4QualityRewardScales(contact_pulse_40ms=-2.0),
            forward_v4_substep_contact=True,
            forward_iteration_v6_contact_abort_island_only=True,
        )
    with pytest.raises(ValueError, match="requires a teacher table"):
        make_h4_aligned_environment_class(
            **common,
            reverse_iteration_v6_absolute_full_leg_targets=True,
        )


def test_forward_v6_source_keeps_v4_v5_abort_sum_in_disabled_branch() -> None:
    source = textwrap.dedent(inspect.getsource(make_h4_aligned_environment_class))
    assert "if forward_iteration_v6_contact_abort_island_only:" in source
    assert (
        'info["h4_v4_aborted_contact_island_loss_sum"]\n'
        '                        + info["h4_v4_aborted_off_gap_loss_sum"]'
    ) in source
    assert "FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID" in source
    assert FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID == (
        "CONTACT_ABORT_TYPE_SEPARATION_ISLAND_ONLY"
    )


@pytest.mark.parametrize("action_value", (-1.0, -0.5, 0.0, 0.5, 1.0))
def test_reverse_v6_absolute_decoder_matches_official_calibrated_branch_exactly(
    action_value: float,
) -> None:
    lower, upper, initial = contract_target_vectors()
    action = np.full(14, action_value, dtype=np.float64)
    action[5:9] = -action_value
    bounded = np.clip(action, -1.0, 1.0)
    positive_span = 0.9 * (upper - initial)
    negative_span = 0.9 * (initial - lower)
    directional_span = np.where(bounded >= 0.0, positive_span, negative_span)
    base_span = np.minimum(0.25, directional_span)
    magnitude = np.abs(bounded)
    expected = initial + np.sign(bounded) * (
        base_span * magnitude + (directional_span - base_span) * magnitude**5
    )
    expected[5:9] = 0.0

    actual = reverse_iteration_v6_absolute_full_leg_targets(action)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[5:9], np.zeros(4))


def test_reverse_v6_decoder_uses_asymmetric_directional_safe_spans() -> None:
    lower = np.full(14, -1.0)
    upper = np.full(14, 1.0)
    initial = np.zeros(14)
    lower[0], initial[0], upper[0] = -2.0, -0.2, 0.5
    lower[5:9] = initial[5:9] = upper[5:9] = 0.0

    positive = reverse_iteration_v6_absolute_full_leg_targets(
        np.asarray([1.0] + [0.0] * 13), initial, lower, upper
    )
    negative_action = np.zeros(14)
    negative_action[0] = -1.0
    negative = reverse_iteration_v6_absolute_full_leg_targets(
        negative_action, initial, lower, upper
    )

    assert positive[0] == pytest.approx(-0.2 + 0.9 * 0.7)
    assert negative[0] == pytest.approx(-0.2 - 0.9 * 1.8)
    assert abs(negative[0] - initial[0]) > abs(positive[0] - initial[0])


def test_reverse_v6_zero_action_is_exact_safe_init_and_head_is_zero() -> None:
    _, _, initial = contract_target_vectors()
    decoded = reverse_iteration_v6_absolute_full_leg_targets(np.zeros(14))

    np.testing.assert_array_equal(decoded, initial)
    np.testing.assert_array_equal(decoded[5:9], np.zeros(4))


def test_reverse_v6_action_cap_margin_and_single_guard_are_exact() -> None:
    _, _, initial = contract_target_vectors()
    capped = reverse_iteration_v6_absolute_full_leg_targets(np.full(14, 2.0))
    endpoint = reverse_iteration_v6_absolute_full_leg_targets(np.ones(14))
    telemetry = reverse_iteration_v6_absolute_full_leg_target_telemetry(
        np.full(14, 2.0)
    )
    margin_target = margin_clip_targets(endpoint)
    applied = final_target_guard_step(endpoint, initial)

    np.testing.assert_array_equal(capped, endpoint)
    assert int(telemetry.action_clip_count) == 10
    np.testing.assert_array_equal(margin_target[5:9], np.zeros(4))
    assert np.max(np.abs(applied[np.r_[0:5, 9:14]] - initial[np.r_[0:5, 9:14]])) <= (
        MAX_TARGET_DELTA_PER_TICK_RAD + 1.0e-12
    )
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(make_h4_aligned_environment_class))
    )
    guarded = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "guarded_physics_step"
    )
    guard_calls = [
        node
        for node in ast.walk(guarded)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "final_target_guard_step"
    ]
    assert len(guard_calls) == 1
    precomposer_assignments = [
        node
        for node in ast.walk(guarded)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "precomposed" for target in node.targets)
    ]
    hard_precomposer_assignments = [
        node
        for node in ast.walk(guarded)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "hard_delta" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "jp.clip"
    ]
    assert len(precomposer_assignments) == 1
    assert len(hard_precomposer_assignments) == 1


def test_reverse_v6_actual_selected_target_wiring_detects_one_leaf_mutation() -> None:
    action = np.linspace(-0.8, 0.8, 14, dtype=np.float32)
    selected = reverse_iteration_v6_absolute_full_leg_targets(action)
    exact = reverse_iteration_v6_absolute_full_leg_target_wiring_audit(
        selected, action
    )

    assert bool(exact.exact)
    assert float(exact.max_abs_error) == 0.0
    assert bool(exact.teacher_target_contribution_zero_exact)
    np.testing.assert_array_equal(
        exact.teacher_target_contribution, np.zeros(14)
    )

    mutated = np.asarray(selected).copy()
    mutated[3] = np.nextafter(mutated[3], np.float64(np.inf))
    rejected = reverse_iteration_v6_absolute_full_leg_target_wiring_audit(
        mutated, action
    )

    assert not bool(rejected.exact)
    assert float(rejected.max_abs_error) > 0.0
    assert not bool(rejected.teacher_target_contribution_zero_exact)
    assert rejected.teacher_target_contribution[3] != 0.0
    source = textwrap.dedent(inspect.getsource(make_h4_aligned_environment_class))
    assert (
        "reverse_iteration_v6_absolute_full_leg_target_wiring_audit(\n"
        "                            selected_raw_targets,"
    ) in source
    assert "teacher_target_contribution = jp.zeros_like" not in source


def test_reverse_v6_structural_counters_are_adjacent_to_actual_operations() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(make_h4_aligned_environment_class))
    )
    guarded = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "guarded_physics_step"
    )

    def assignments(target_name: str) -> list[ast.Assign]:
        return [
            node
            for node in ast.walk(guarded)
            if isinstance(node, ast.Assign)
            and any(ast.unparse(target) == target_name for target in node.targets)
        ]

    def increments(target_name: str) -> list[ast.AugAssign]:
        return [
            node
            for node in ast.walk(guarded)
            if isinstance(node, ast.AugAssign)
            and ast.unparse(node.target) == target_name
            and isinstance(node.op, ast.Add)
            and ast.unparse(node.value) == "1"
        ]

    bindings = (
        (
            "reverse_v6_decoder_structural_call_count",
            "reverse_v6_decoder",
        ),
        (
            "reverse_v6_precomposer_structural_call_count",
            "precomposed",
        ),
        (
            "reverse_v6_final_guard_structural_call_count",
            "applied",
        ),
    )
    for counter_name, operation_target in bindings:
        counter_increments = increments(counter_name)
        operation_assignments = assignments(operation_target)
        assert len(counter_increments) == 1
        adjacent_operations = [
            operation
            for operation in operation_assignments
            if operation.lineno == counter_increments[0].end_lineno + 1
        ]
        assert len(adjacent_operations) == 1

    structural_assertions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == (
            "reverse_iteration_v6_absolute_full_leg_targets and "
            "(reverse_v6_decoder_structural_call_count, "
            "reverse_v6_precomposer_structural_call_count, "
            "reverse_v6_final_guard_structural_call_count) != (1, 1, 1)"
        )
    ]
    assert len(structural_assertions) == 1


@pytest.mark.parametrize(
    ("leg_count", "precomposer_count", "final_guard_count", "failed_field"),
    (
        (9, 1, 1, "decoder_leg_count_exact"),
        (10, 0, 1, "precomposer_call_count_exact"),
        (10, 1, 2, "final_guard_call_count_exact"),
    ),
)
def test_reverse_v6_structural_exact_booleans_reject_each_count_axis(
    leg_count: int,
    precomposer_count: int,
    final_guard_count: int,
    failed_field: str,
) -> None:
    invariants = reverse_iteration_v6_structural_count_invariants(
        leg_count, precomposer_count, final_guard_count
    )
    assert not bool(getattr(invariants, failed_field))
    assert bool(invariants.violation)

    passing = reverse_iteration_v6_structural_count_invariants(10, 1, 1)
    assert bool(passing.decoder_leg_count_exact)
    assert bool(passing.precomposer_call_count_exact)
    assert bool(passing.final_guard_call_count_exact)
    assert not bool(passing.violation)


def test_reverse_v6_structural_exact_booleans_jit_vmap_feed_callback() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    callbacks: list[bool] = []

    def record(value) -> None:
        callbacks.append(bool(np.asarray(value).item()))

    assertion = make_v6_compiled_invariant_assertion(
        jax, jp, label="V6_STRUCTURAL_COUNTS", failure_callback=record
    )

    def checked(counts):
        invariants = reverse_iteration_v6_structural_count_invariants(
            counts[0], counts[1], counts[2], xp=jp
        )
        token = assertion(invariants.violation)
        return (
            invariants.decoder_leg_count_exact,
            invariants.precomposer_call_count_exact,
            invariants.final_guard_call_count_exact,
            token,
        )

    compiled = jax.jit(jax.vmap(checked))
    passing_counts = jp.tile(jp.asarray([[10, 1, 1]], dtype=jp.int32), (8, 1))
    passing = compiled(passing_counts)
    jax.block_until_ready(passing[-1])
    assert callbacks == []
    assert all(np.asarray(values).all() for values in passing[:3])

    mutated = passing_counts.at[1, 0].set(9).at[3, 1].set(0).at[6, 2].set(2)
    failing = compiled(mutated)
    jax.block_until_ready(failing[-1])
    assert callbacks == [True]
    assert not np.asarray(failing[0]).all()
    assert not np.asarray(failing[1]).all()
    assert not np.asarray(failing[2]).all()


def test_reverse_v6_action_jacobian_nonzero_teacher_target_jacobian_zero() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    action = jp.full((14,), 0.5, dtype=jp.float32).at[5:9].set(0.75)
    action_jacobian = jax.jacrev(
        lambda value: reverse_iteration_v6_absolute_full_leg_targets(
            value, xp=jp
        )
    )(action)
    action_jacobian = np.asarray(action_jacobian)
    leg_indices = np.r_[0:5, 9:14]
    assert np.all(np.diag(action_jacobian)[leg_indices] > 0.0)
    np.testing.assert_array_equal(action_jacobian[5:9], np.zeros((4, 14)))

    reference_leg_indices = jp.arange(10, dtype=jp.int32)
    source_reference = jp.arange(16, dtype=jp.float32)
    teacher_jacobian = jax.jacrev(
        lambda reference: reverse_iteration_v6_teacher_timing_only_reference(
            reference, reference_leg_indices, xp=jp
        )
    )(source_reference)
    teacher_jacobian = np.asarray(teacher_jacobian)
    np.testing.assert_array_equal(teacher_jacobian[:10], np.zeros((10, 16)))
    np.testing.assert_array_equal(teacher_jacobian[10:, 10:], np.eye(6))


def test_reverse_v6_teacher_table_target_counterfactual_is_invariant() -> None:
    indices = np.arange(10, dtype=np.int32)
    source_a = np.concatenate((np.full(10, -7.0), np.arange(6, dtype=float)))
    source_b = np.concatenate((np.full(10, 11.0), np.arange(6, dtype=float)))

    neutral_a = reverse_iteration_v6_teacher_timing_only_reference(source_a, indices)
    neutral_b = reverse_iteration_v6_teacher_timing_only_reference(source_b, indices)

    np.testing.assert_array_equal(neutral_a, neutral_b)
    np.testing.assert_array_equal(neutral_a[10:], np.arange(6, dtype=float))
    source = textwrap.dedent(inspect.getsource(make_h4_aligned_environment_class))
    timing_branch = source.index(
        "if reverse_iteration_v6_absolute_full_leg_targets:\n"
        "                    return reverse_iteration_v6_teacher_timing_only_reference"
    )
    table_target_selection = source.index(
        "actuator_target = self._h4_selected_teacher_actuator_target"
    )
    assert timing_branch < table_target_selection


def test_reverse_v6_factory_source_removes_teacher_and_residual_authority() -> None:
    source = textwrap.dedent(inspect.getsource(make_h4_aligned_environment_class))
    assert "scales.target_imitation = 0.0" in source
    assert "scales.contact_imitation = 0.0" in source
    assert "selected_raw_targets = jp.where(" in source
    assert "reverse_v6_decoder.targets" in source
    assert 'residual_authority_scale = jp.zeros(' in source
    assert "REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID" in source
    assert REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID == (
        "ABSOLUTE_FULL_LEG_TARGETS_WITH_TEACHER_TIMING_ONLY"
    )


def test_v6_compiled_assertion_has_zero_success_and_one_batched_failure_callback() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    callbacks: list[bool] = []

    def record(value) -> None:
        callbacks.append(bool(np.asarray(value).item()))

    assertion = make_v6_compiled_invariant_assertion(
        jax, jp, label="TEST_V6", failure_callback=record
    )
    compiled = jax.jit(jax.vmap(assertion))
    success = compiled(jp.zeros(8, dtype=bool))
    jax.block_until_ready(success)
    assert callbacks == []

    failure = compiled(jp.asarray([False, False, True, False, False, False, False, False]))
    jax.block_until_ready(failure)
    assert callbacks == [True]


@pytest.mark.parametrize(
    "malformation", ("shape", "dtype", "nonfinite", "count", "variance")
)
def test_same_width_h4_checkpoint_audit_fails_closed(malformation: str) -> None:
    checkpoint, _ = transplant_v22_checkpoint_to_h4_observation(
        _fake_v22_checkpoint()
    )
    if malformation == "shape":
        checkpoint[1]["params"]["hidden_2"]["kernel"] = np.zeros(
            (255, 128), dtype=np.float32
        )
    elif malformation == "dtype":
        checkpoint[2]["params"]["hidden_3"]["bias"] = np.zeros(
            1, dtype=np.float64
        )
    elif malformation == "nonfinite":
        checkpoint[1]["params"]["hidden_1"]["bias"][0] = np.nan
    elif malformation == "count":
        checkpoint[0] = replace(
            checkpoint[0],
            count=_FakeCount(np.asarray(-1, dtype=np.int32), np.asarray(10, dtype=np.uint32)),
        )
    else:
        variance = checkpoint[0].summed_variance["privileged_state"].copy()
        variance[174] = np.float32(-1.0e-7)
        checkpoint[0] = replace(
            checkpoint[0],
            summed_variance={
                **checkpoint[0].summed_variance,
                "privileged_state": variance,
            },
        )
    with pytest.raises(ValueError):
        require_checkpoint_observation_compatibility(
            checkpoint, actor_observation_width=116
        )


@pytest.mark.parametrize(
    "scales",
    [
        H4QualityRewardScales(force_slip=1.0),
        H4QualityRewardScales(left_force_slip=1.0),
        H4QualityRewardScales(right_force_slip=1.0),
        H4QualityRewardScales(load_balance=1.0),
        H4QualityRewardScales(slew_feasibility=1.0),
        H4QualityRewardScales(target_lag=1.0),
        H4QualityRewardScales(single_support=-1.0),
        H4QualityRewardScales(alternation=-1.0),
    ],
)
def test_reward_scale_signs_fail_closed(scales: H4QualityRewardScales) -> None:
    with pytest.raises(ValueError):
        scales.as_reward_scale_dict()
