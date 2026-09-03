from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


EXP_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    EXP_ROOT
    / "scripts"
    / "evaluate_h4_training_candidate_reverse_v6_absolute_full_leg_targets_v1.py"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adapter = _load_module("exp004_reverse_v6_evaluator_adapter_test", ADAPTER_PATH)


def _next_float32(value: float) -> np.float32:
    return np.nextafter(
        np.float32(value), np.float32(np.inf), dtype=np.float32
    )


def _exact_args() -> argparse.Namespace:
    return adapter._resolve_process_start_paths(
        argparse.Namespace(
            params=adapter.EXPECTED_PARAMS_PATH,
            params_sha256="1" * 64,
            manifest=adapter.EXPECTED_MANIFEST_PATH,
            manifest_sha256="2" * 64,
            output=adapter.EXPECTED_OUTPUT_PATH,
            trusted_run_root=adapter.TRUSTED_RUN_ROOT,
            adapter_authorization=adapter.ADAPTER_AUTHORIZATION_PATH,
            source_root=adapter.DEFAULT_SOURCE_ROOT,
            generated_root=adapter.DEFAULT_GENERATED_ROOT,
            v22_parent_checkpoint=adapter.DEFAULT_V22_PARENT_CHECKPOINT,
            platform="cpu",
        )
    )


def test_real_authorization_loader_and_every_fixed_source_binding_are_exact() -> None:
    payload = adapter.load_and_validate_adapter_authorization(
        adapter.ADAPTER_AUTHORIZATION_PATH
    )
    assert payload["scope"]["contract_id"] == adapter.ADAPTER_CONTRACT_ID
    assert adapter.sha256_file(adapter.ADAPTER_AUTHORIZATION_PATH) == (
        adapter.PINNED_ADAPTER_AUTHORIZATION_SHA256
    )
    assert adapter._verify_file_bindings(adapter.PINNED_FROZEN_SOURCES) == {
        name: binding[1]
        for name, binding in adapter.PINNED_FROZEN_SOURCES.items()
    }
    authorization_text = adapter.ADAPTER_AUTHORIZATION_PATH.read_text(
        encoding="utf-8"
    )
    assert adapter.sha256_file(adapter.ADAPTER_PATH) not in authorization_text
    assert "adapter_source_sha256" not in authorization_text


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ((), None),
        (("scope", "contract_id"), "WRONG"),
        (("scope", "promotion_eligible"), True),
        (("candidate_binding", "candidate_hashes_pre_authorized"), True),
        (("candidate_binding", "trusted_root_relative_path"), "wrong"),
        (
            ("runtime_contract", "factory_flags", "forward_v4_substep_contact"),
            True,
        ),
        (
            (
                "runtime_contract",
                "legacy_reward_config_exact",
                "target_imitation",
            ),
            0,
        ),
        (
            ("runtime_contract", "backward_residual_scale"),
            float(np.nextafter(0.0, 1.0)),
        ),
        (("runtime_contract", "precomposer_slew_rad_per_tick"), 0.0400000001),
        (("artifact_contract", "physical_command_mps_radps"), [-0.04, 0.0, 0.0]),
        (("artifact_contract", "gait_sample_count"), 3000),
        (("artifact_contract", "control_tick_count"), 300.0),
        (("artifact_contract", "fixed_duration_s"), 6),
        (
            ("frozen_source_bindings", "h4_post_training", "sha256"),
            "0" * 64,
        ),
        (("authorization", "promotion_evidence"), True),
        (("decision", "promotion"), "ALLOWED"),
    ),
)
def test_authorization_semantic_type_and_numeric_drift_fails_closed(
    path: tuple[str, ...], value: Any
) -> None:
    payload = adapter.load_json_strict(adapter.ADAPTER_AUTHORIZATION_PATH)
    if not path:
        payload["schema_version"] = True
        with pytest.raises(ValueError, match="authorization"):
            adapter.validate_adapter_authorization_payload(payload)
        return
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(ValueError, match="authorization"):
        adapter.validate_adapter_authorization_payload(payload)


def test_authorization_schema_missing_extra_and_byte_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = adapter.load_json_strict(adapter.ADAPTER_AUTHORIZATION_PATH)
    payload.pop("decision")
    with pytest.raises(ValueError, match="schema drifted"):
        adapter.validate_adapter_authorization_payload(payload)
    payload = adapter.load_json_strict(adapter.ADAPTER_AUTHORIZATION_PATH)
    payload["extra"] = False
    with pytest.raises(ValueError, match="schema drifted"):
        adapter.validate_adapter_authorization_payload(payload)

    drifted = tmp_path / adapter.ADAPTER_AUTHORIZATION_PATH.name
    data = bytearray(adapter.ADAPTER_AUTHORIZATION_PATH.read_bytes())
    data[-1] = 0x20 if data[-1] != 0x20 else 0x0A
    drifted.write_bytes(data)
    calls = {"json": 0}

    def reject_json(_path: Path) -> Any:
        calls["json"] += 1
        raise AssertionError("JSON parse reached")

    monkeypatch.setattr(adapter, "ADAPTER_AUTHORIZATION_PATH", drifted)
    monkeypatch.setattr(adapter, "load_json_strict", reject_json)
    with pytest.raises(ValueError, match="SHA256 drifted"):
        adapter.load_and_validate_adapter_authorization(drifted)
    assert calls == {"json": 0}


def test_exact_cli_accepts_unpinned_sha_inputs_but_rejects_any_path_or_mode_drift(
    tmp_path: Path,
) -> None:
    args = _exact_args()
    adapter._validate_exact_cli(args, require_output_absent=False)
    assert args.params_sha256 == "1" * 64
    assert args.manifest_sha256 == "2" * 64
    for field, value in (
        ("params", tmp_path / "final_params.pkl"),
        ("manifest", tmp_path / "run_manifest.json"),
        ("output", tmp_path / "h4_integrated_strict_3x6s_v1.json"),
        ("trusted_run_root", tmp_path),
        ("platform", "gpu"),
        ("allow_wiring_diagnostic", True),
    ):
        broken = copy.copy(args)
        setattr(broken, field, Path(value).resolve() if isinstance(value, Path) else value)
        with pytest.raises(ValueError, match="CLI drifted"):
            adapter._validate_exact_cli(broken, require_output_absent=False)
    broken = copy.copy(args)
    broken.params_sha256 = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA256"):
        adapter._validate_exact_cli(broken, require_output_absent=False)

    parser = adapter.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--params",
                str(adapter.EXPECTED_PARAMS_PATH),
                "--params-sha256",
                "1" * 64,
                "--manifest",
                str(adapter.EXPECTED_MANIFEST_PATH),
                "--manifest-sha256",
                "2" * 64,
                "--output",
                str(adapter.EXPECTED_OUTPUT_PATH),
                "--promotion-evidence-output",
                str(tmp_path / "promotion.json"),
            ]
        )


def test_write_new_only_rejects_the_exact_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "h4_integrated_strict_3x6s_v1.json"
    existing.write_text("{}", encoding="utf-8")
    args = _exact_args()
    args.output = existing.resolve()
    monkeypatch.setattr(adapter, "EXPECTED_OUTPUT_PATH", existing)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        adapter._validate_exact_cli(args)


def test_factory_wrapper_supplies_only_reverse_v6_and_exact_legacy_config() -> None:
    captured: dict[str, Any] = {}

    def frozen_factory(**kwargs: Any) -> type:
        captured.update(kwargs)
        return object

    wrapped = adapter._reverse_v6_factory_wrapper(frozen_factory)
    assert wrapped(probe=1) is object
    assert captured == {
        "probe": 1,
        "legacy_reward_config_overrides": {
            "target_imitation": 0.0,
            "contact_imitation": 0.0,
            "tracking_sigma": 0.01,
        },
        "forward_v4_substep_contact": False,
        "forward_iteration_v6_contact_abort_island_only": False,
        "reverse_iteration_v6_absolute_full_leg_targets": True,
    }
    for competing in (
        "forward_v4_substep_contact",
        "forward_iteration_v6_contact_abort_island_only",
        "reverse_iteration_v6_absolute_full_leg_targets",
        "legacy_reward_config_overrides",
    ):
        with pytest.raises(ValueError, match="competing opt-ins"):
            wrapped(**{competing: False})


def _fake_environment() -> Any:
    scales = SimpleNamespace(target_imitation=0.0, contact_imitation=0.0)
    reward = SimpleNamespace(scales=scales, tracking_sigma=0.01)
    noise = SimpleNamespace(action_min_delay=0, action_max_delay=1)
    return SimpleNamespace(
        h4_reverse_iteration_v6_absolute_full_leg_targets=True,
        h4_forward_iteration_v6_contact_abort_island_only=False,
        h4_forward_v4_substep_contact=False,
        h4_reverse_iteration_v6_contract_id=adapter.CORE_CONTRACT_ID,
        h4_reverse_iteration_v6_compiled_assertion_bound=True,
        h4_reverse_iteration_v6_residual_authority_scale=0.0,
        h4_reverse_iteration_v6_teacher_target_contribution=0.0,
        _backward_residual_scale=0.0,
        _config=SimpleNamespace(reward_config=reward, noise_config=noise),
        _h4_reverse_teacher_table=np.zeros((54, 14)),
        PRM=SimpleNamespace(nb_steps_in_period=27),
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("h4_reverse_iteration_v6_absolute_full_leg_targets", False),
        ("h4_forward_iteration_v6_contact_abort_island_only", True),
        ("h4_forward_v4_substep_contact", True),
        ("h4_reverse_iteration_v6_contract_id", "WRONG"),
        ("h4_reverse_iteration_v6_compiled_assertion_bound", False),
        ("_backward_residual_scale", float(np.nextafter(0.0, 1.0))),
    ),
)
def test_environment_contract_rejects_each_factory_near_miss(
    path: str, value: Any
) -> None:
    env = _fake_environment()
    setattr(env, path, value)
    with pytest.raises(RuntimeError, match="environment contract drifted"):
        adapter._validate_reverse_v6_environment(env)


def test_functiontype_call_graph_preserves_all_frozen_module_state() -> None:
    base, post, core = adapter._load_frozen_modules()
    before = adapter._module_contract_snapshot(base, post, core)
    graph = adapter.build_reverse_v6_call_graph(base, post, core)
    adapter._assert_module_contract_unchanged(base, post, core, before)
    assert base.H4_REVERSE_RESIDUAL_SCALE == 0.12
    assert post.H4_REVERSE_RESIDUAL_SCALE == 0.12
    assert graph["central_safety"].__globals__["_h4_control_trace_arrays"] is (
        adapter._v6_control_trace_arrays
    )
    assert graph["safety"].__globals__["rederive_h4_control_contract"] is (
        adapter.rederive_reverse_v6_control_contract
    )
    assert graph["episode_validator"].__globals__[
        "rederive_central_safety_audit_from_control_trace"
    ] is graph["central_safety"]
    assert graph["compatibility_validator"].__globals__[
        "validate_h4_strict_episode"
    ] is graph["episode_validator"]
    assert graph["run_evaluation"].__globals__["_make_environment_and_policy"] is (
        graph["make_environment"]
    )
    assert graph["run_evaluation"].__globals__["_run_episode"] is graph[
        "run_episode"
    ]
    assert graph["run_evaluation"].__globals__["validate_trusted_h4_bundle"] is (
        graph["trusted_bundle"]
    )


def test_cloned_failure_never_mutates_original_global() -> None:
    namespace: dict[str, Any] = {"VALUE": 0.12}
    exec(
        "def probe(fail=False):\n"
        "    if fail:\n"
        "        raise RuntimeError('probe')\n"
        "    return VALUE\n",
        namespace,
    )
    original = namespace["probe"]
    clone = adapter._clone_function(
        original, global_overrides={"VALUE": 0.0}
    )
    assert clone() == 0.0
    with pytest.raises(RuntimeError, match="probe"):
        clone(True)
    assert original() == 0.12
    assert original.__globals__["VALUE"] == 0.12


def test_adapter_never_assigns_outer_joystick_step_or_motor_speed_global() -> None:
    tree = ast.parse(adapter.ADAPTER_PATH.read_text(encoding="utf-8"))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            raw_targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            targets.extend(raw_targets)
        for target in targets:
            try:
                rendered = ast.unparse(target)
            except Exception:  # pragma: no cover - ast.unparse is available
                continue
            if rendered in {
                "joystick.mjx_env.step",
                "joystick.USE_MOTOR_SPEED_LIMITS",
            }:
                forbidden.append(rendered)
    assert forbidden == []


def test_bundle_wrapper_validates_v6_identity_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, post, core = adapter._load_frozen_modules()
    sentinel = object()
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        post,
        "validate_trusted_h4_bundle",
        lambda **_kwargs: calls.append(("post", None)) or sentinel,
    )
    monkeypatch.setattr(
        adapter,
        "_validate_reverse_v6_bundle",
        lambda value: calls.append(("v6", value)),
    )
    graph = adapter.build_reverse_v6_call_graph(base, post, core)
    assert graph["trusted_bundle"]() is sentinel
    assert calls == [("post", None), ("v6", sentinel)]


def _valid_bundle() -> Any:
    authorization = adapter.load_json_strict(
        adapter.REVERSE_V6_AUTHORIZATION_PATH
    )
    expected_legacy = {
        "target_imitation": 0.0,
        "contact_imitation": 0.0,
        "tracking_sigma": 0.01,
        "backward_residual_scale": 0.0,
    }
    auth_record = {
        "path": str(adapter.REVERSE_V6_AUTHORIZATION_PATH.resolve()),
        "sha256": adapter.PINNED_REVERSE_V6_AUTHORIZATION_SHA256,
        "contract_id": adapter.TRAINING_CONTRACT_ID,
        "h4_parent_checkpoint_allowed": False,
        "v4_gain_inherited": False,
        "v5_parent_checkpoint_inherited": False,
        "legacy_reward_config_audit": {
            "expected": copy.deepcopy(expected_legacy),
            "per_environment": {
                "primary": copy.deepcopy(expected_legacy),
                "local": copy.deepcopy(expected_legacy),
            },
            "passed": True,
        },
    }
    flags = {
        name: name == "reverse_iteration_v6_absolute_full_leg_targets"
        for name in (
            "forward_iteration_v2",
            "reverse_iteration_v2",
            "forward_iteration_v3_touchdown_balance",
            "reverse_iteration_v3_no_target_imitation",
            "forward_iteration_v4_contact_event_validity_persistence",
            "reverse_iteration_v4_residual_transfer_gain_024",
            "forward_v5_contact_pulse_abort_scale_only",
            "reverse_iteration_v5_no_contact_imitation",
            "forward_iteration_v6_contact_abort_island_only",
            "reverse_iteration_v6_absolute_full_leg_targets",
        )
    }
    config = {
        **flags,
        "training_contract_id": adapter.TRAINING_CONTRACT_ID,
        "backward_residual_scale": 0.0,
        "iteration_v6_core_source": {
            "path": str(adapter.CORE_PATH.resolve()),
            "sha256": adapter.PINNED_FROZEN_SOURCES[
                "h4_training_alignment"
            ][1],
        },
        "reverse_iteration_v6_absolute_full_leg_targets_authorization": (
            auth_record
        ),
        "selected_reverse_teacher": {
            "path": str(adapter.SELECTED_REVERSE_TEACHER_PATH.resolve()),
            "sha256": adapter.PINNED_SELECTED_REVERSE_TEACHER_SHA256,
            "candidate_id": "cbe8decf6a7c4e5e",
            "candidate_name": "h4_reverse_c1p50_h2_e1p00",
            "cadence_hz": 1.5,
            "phase_advance_bins_per_control": 1.62,
            "entry_phase_bins": 14.0,
            "training_use": "TRAINING_COMPOSITION_COMPONENT_NOT_ADOPTED",
            "persistent_during_training": True,
            "qualification": "FAILED_EXACT_HOME_H4",
            "runtime_parity_requirement": (
                "Any adopted policy must use this identical teacher plus "
                "learned residual composition at runtime."
            ),
        },
        "action_parameterization_contract": copy.deepcopy(
            authorization["action_parameterization_contract"]
        ),
        "teacher_timing_contract": copy.deepcopy(
            authorization["teacher_timing_contract"]
        ),
    }
    return SimpleNamespace(
        config=config,
        run_name=adapter.CANDIDATE_ROOT.name,
        expert="reverse",
        status="COMPLETED",
        activity="PPO_PILOT_TRAINING",
    )


def test_candidate_bundle_runtime_identity_is_recursive_type_exact() -> None:
    adapter._validate_reverse_v6_bundle(_valid_bundle())
    for mutation in (
        "integer_zero",
        "negative_zero",
        "old_mode",
        "legacy_type",
        "selected_entry_int",
        "action_exponent_float",
        "timing_rows_float",
        "timing_entry_int",
    ):
        bundle = _valid_bundle()
        if mutation == "integer_zero":
            bundle.config["backward_residual_scale"] = 0
        elif mutation == "negative_zero":
            bundle.config["backward_residual_scale"] = -0.0
        elif mutation == "old_mode":
            bundle.config["reverse_iteration_v6_absolute_full_leg_targets"] = False
            bundle.config["reverse_iteration_v5_no_contact_imitation"] = True
        elif mutation == "legacy_type":
            bundle.config[
                "reverse_iteration_v6_absolute_full_leg_targets_authorization"
            ]["legacy_reward_config_audit"]["expected"][
                "backward_residual_scale"
            ] = 0
        elif mutation == "selected_entry_int":
            bundle.config["selected_reverse_teacher"]["entry_phase_bins"] = 14
        elif mutation == "action_exponent_float":
            bundle.config["action_parameterization_contract"][
                "nonlinear_exponent"
            ] = 5.0
        elif mutation == "timing_rows_float":
            bundle.config["teacher_timing_contract"][
                "teacher_table_rows"
            ] = 54.0
        else:
            bundle.config["teacher_timing_contract"]["entry_phase_bins"] = 14
        with pytest.raises(ValueError, match="candidate bundle drifted"):
            adapter._validate_reverse_v6_bundle(bundle)


@pytest.mark.parametrize("value", (-1.0, -0.5, 0.0, 0.5, 1.0))
def test_host_absolute_decoder_matches_independent_float32_scalar_formula(
    value: float,
) -> None:
    action = np.full(14, value, dtype=np.float32)
    action[5:9] = np.float32(-value)
    actual = adapter.reverse_v6_decode_float32(action)
    expected = np.zeros(14, dtype=np.float32)
    for index in adapter.LEG_INDICES:
        bounded = np.clip(action[index], np.float32(-1), np.float32(1))
        if bounded >= 0:
            span = np.float32(
                adapter.DIRECTIONAL_SPAN_F32
                * np.float32(
                    adapter.SAFE_UPPER_F32[index]
                    - adapter.SAFE_INIT_F32[index]
                )
            )
        else:
            span = np.float32(
                adapter.DIRECTIONAL_SPAN_F32
                * np.float32(
                    adapter.SAFE_INIT_F32[index]
                    - adapter.SAFE_LOWER_F32[index]
                )
            )
        base = np.minimum(adapter.BASE_SPAN_F32, span)
        magnitude = np.abs(bounded)
        squared = np.float32(magnitude * magnitude)
        fourth = np.float32(squared * squared)
        fifth = np.float32(magnitude * fourth)
        target_magnitude = np.float32(
            np.float32(base * magnitude)
            + np.float32(np.float32(span - base) * fifth)
        )
        expected[index] = np.float32(
            adapter.SAFE_INIT_F32[index]
            + np.float32(np.sign(bounded) * target_magnitude)
        )
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(actual[5:9], np.zeros(4, dtype=np.float32))


def test_host_decoder_matches_jitted_core_cpu_when_jax_is_available() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    if str(EXP_ROOT) not in sys.path:
        sys.path.insert(0, str(EXP_ROOT))
    import safe_gait_experts.h4_training_alignment as core

    compiled = jax.jit(
        lambda value: core.reverse_iteration_v6_absolute_full_leg_targets(
            value, xp=jp
        ),
        backend="cpu",
    )
    rng = np.random.default_rng(20_260_810)
    for _ in range(128):
        action = rng.uniform(-1.5, 1.5, 14).astype(np.float32)
        actual = np.asarray(compiled(jp.asarray(action)))
        expected = adapter.reverse_v6_decode_float32(action)
        np.testing.assert_array_equal(
            actual.view(np.uint32), expected.view(np.uint32)
        )


def test_precomposer_preserves_hard_value_and_smooth_surrogate_derivative() -> None:
    previous = adapter.SAFE_INIT_F32.copy()
    raw_delta = np.asarray(
        [
            -0.080,
            -0.040001,
            -0.040,
            -0.039999,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.039999,
            0.040,
            0.040001,
            0.080,
            0.020,
        ],
        dtype=np.float32,
    )
    target = np.add(previous, raw_delta, dtype=np.float32)
    target[5:9] = 0.0
    value, derivative = (
        adapter.reverse_v6_precomposer_value_and_surrogate_derivative_float32(
            target, previous
        )
    )
    hard_delta = np.clip(
        np.subtract(target, previous, dtype=np.float32),
        -adapter.SLEW_F32,
        adapter.SLEW_F32,
    )
    expected_value = np.add(previous, hard_delta, dtype=np.float32)
    expected_value[5:9] = np.float32(0.0)
    np.testing.assert_array_equal(value, expected_value)
    scaled = np.divide(
        np.subtract(target, previous, dtype=np.float32),
        adapter.SLEW_F32,
        dtype=np.float32,
    )
    expected_derivative = np.float32(1.0) - np.tanh(scaled).astype(np.float32) ** 2
    expected_derivative[5:9] = 0.0
    np.testing.assert_allclose(derivative, expected_derivative, rtol=0, atol=1e-7)
    assert derivative[2] > derivative[1]
    assert derivative[11] < derivative[9]
    assert np.all(derivative[adapter.LEG_INDICES] > 0)


def _valid_episode() -> dict[str, Any]:
    tick = np.arange(adapter.CONTROL_TICKS, dtype=np.float32)[:, None]
    joint = np.arange(adapter.ACTION_WIDTH, dtype=np.float32)[None, :]
    action = np.sin(tick * np.float32(0.071) + joint * np.float32(0.19)).astype(
        np.float32
    )
    action[:, 0] *= np.float32(1.2)
    raw = adapter.reverse_v6_decode_float32(action)
    margin = adapter.reverse_v6_margin_clip_float32(raw)
    previous = np.empty_like(raw)
    desired = np.empty_like(raw)
    applied = np.empty_like(raw)
    prior = adapter.SAFE_INIT_F32.copy()
    for index in range(adapter.CONTROL_TICKS):
        previous[index] = prior
        desired[index] = adapter.reverse_v6_precomposer_float32(
            margin[index], prior
        )
        applied[index] = adapter.reverse_v6_final_guard_float32(
            desired[index], prior
        )
        prior = applied[index]
    applied_action = action.copy()
    applied_action[:, adapter.HEAD_INDICES] = np.float32(0.0)
    phase_before, table_phase = adapter._expected_phase_timeline_float32()
    action_clip_count = np.count_nonzero(
        action[:, adapter.LEG_INDICES]
        != np.clip(
            action[:, adapter.LEG_INDICES],
            np.float32(-1),
            np.float32(1),
        ),
        axis=1,
    ).astype(np.int32)
    margin_saturation_count = np.count_nonzero(
        raw[:, adapter.LEG_INDICES] != margin[:, adapter.LEG_INDICES],
        axis=1,
    ).astype(np.int32)
    guard_lag = np.max(
        np.abs(
            np.subtract(margin, applied, dtype=np.float32)
        )[:, adapter.LEG_INDICES],
        axis=1,
    ).astype(np.float32)
    true = np.ones(adapter.CONTROL_TICKS, dtype=bool)
    false = np.zeros(adapter.CONTROL_TICKS, dtype=bool)
    trace = {
        "source_dtype": "float32",
        "initial_applied_targets": adapter.SAFE_INIT_F32.copy(),
        "raw_action": action,
        "applied_action": applied_action,
        "preclip_targets": raw.copy(),
        "margin_clipped_targets": margin.copy(),
        "applied_targets": applied,
        "previous_targets": previous,
        "joint_qpos": applied.copy(),
        "v6_decoder_action": action.copy(),
        "v6_decoder_raw_targets": raw.copy(),
        "v6_decoder_margin_targets": margin.copy(),
        "v6_upstream_margin_targets": margin.copy(),
        "v6_precomposer_targets": desired,
        "v6_decoder_exact": true.copy(),
        "v6_decoder_max_abs_error": np.zeros(
            adapter.CONTROL_TICKS, dtype=np.float32
        ),
        "v6_decoder_leg_count": np.full(
            adapter.CONTROL_TICKS, 10, dtype=np.int32
        ),
        "v6_decoder_leg_count_exact": true.copy(),
        "v6_decoder_head_zero_exact": true.copy(),
        "v6_teacher_target_contribution_zero_exact": true.copy(),
        "v6_residual_authority_scale": np.zeros(
            adapter.CONTROL_TICKS, dtype=np.float32
        ),
        "v6_decoder_all_finite": true.copy(),
        "v6_decoder_margin_saturation_count": margin_saturation_count,
        "v6_decoder_action_clip_count": action_clip_count,
        "v6_decoder_guard_lag_max_rad": guard_lag,
        "v6_precomposer_call_count": np.ones(
            adapter.CONTROL_TICKS, dtype=np.int32
        ),
        "v6_precomposer_call_count_exact": true.copy(),
        "v6_final_guard_call_count": np.ones(
            adapter.CONTROL_TICKS, dtype=np.int32
        ),
        "v6_final_guard_call_count_exact": true.copy(),
        "v6_decoder_violation": false.copy(),
        "v6_decoder_assertion_token": np.zeros(
            adapter.CONTROL_TICKS, dtype=np.int32
        ),
        "v6_direct_physics_substep_count": np.full(
            adapter.CONTROL_TICKS,
            adapter.PHYSICS_SUBSTEPS_PER_CONTROL,
            dtype=np.int32,
        ),
        "v6_direct_physics_dynamic6_endpoint_exact": true.copy(),
        "v6_direct_physics_dynamic6_endpoint_max_abs_error": np.zeros(
            adapter.CONTROL_TICKS, dtype=np.float32
        ),
        "v6_direct_physics_dynamic6_field_count": np.full(
            adapter.CONTROL_TICKS,
            len(adapter.DYNAMIC6_FIELDS),
            dtype=np.int32,
        ),
        "v6_direct_physics_dynamic6_all_finite": true.copy(),
        "v6_direct_physics_applied_target_exact": true.copy(),
        "v6_direct_physics_snapshot_endpoint_exact": true.copy(),
        "v6_direct_physics_snapshot_endpoint_max_abs_error": np.zeros(
            adapter.CONTROL_TICKS, dtype=np.float32
        ),
        "v6_direct_physics_snapshot_endpoint_field_count": np.full(
            adapter.CONTROL_TICKS,
            len(adapter.SNAPSHOT_ENDPOINT_FIELDS),
            dtype=np.int32,
        ),
        "v6_direct_physics_snapshot_endpoint_all_finite": true.copy(),
        "v6_teacher_source_phase_before": phase_before,
        "v6_teacher_table_phase": table_phase,
    }
    episode = {
        "expert": "reverse",
        "physical_command_mps_radps": list(adapter.STRICT_COMMAND),
        "control_trace": trace,
        "reverse_v6_absolute_decoder_contract": (
            adapter._expected_reverse_v6_episode_contract()
        ),
    }
    episode["safety_audit"] = adapter._legacy_float64_central_control_counts(
        {name: np.asarray(value) for name, value in trace.items() if name != "source_dtype"}
    )
    return episode


def test_full_300x14_host_rederivation_passes_without_tolerance() -> None:
    audit = adapter.rederive_reverse_v6_control_contract(_valid_episode())
    assert audit["sample_count"] == 300
    assert audit["source_dtype"] == "float32"
    assert audit["decoder_diagnostics"]["total_action_clip_count"] > 0
    assert audit["passed"] is True
    assert all(audit["checks"].values())


def test_frozen_central_float64_alias_counts_are_exact_visible_diagnostics() -> None:
    episode = _valid_episode()
    base, post, core = adapter._load_frozen_modules()
    graph = adapter.build_reverse_v6_call_graph(base, post, core)
    central = graph["central_safety"](episode)
    expected_counts = {
        "sample_count": 300,
        "nonfinite_sample_count": 0,
        "preclip_target_limit_violations": 0,
        "applied_target_limit_violations": 0,
        "preclip_target_margin_violations": 674,
        "desired_target_margin_violations": 74,
        "applied_target_margin_violations": 75,
        "unauthorized_applied_target_margin_violations": 74,
        "startup_margin_transition_joint_samples": 1,
        "target_slew_violations": 92,
        "qpos_limit_violations": 0,
    }
    assert {name: central[name] for name in expected_counts} == expected_counts
    episode["safety_audit"] = central
    audit = adapter.rederive_reverse_v6_control_contract(episode)
    assert audit["legacy_float64_control_audit_diagnostic_only"] is True
    assert audit["legacy_float64_control_audit_counts"] == expected_counts
    assert audit["checks"]["legacy_float64_control_audit_counts_exact"] is True
    assert audit["passed"] is True


def test_recorded_schema4_contract_is_recursively_type_and_key_exact() -> None:
    episode = _valid_episode()
    episode["h4_control_contract"] = (
        adapter.rederive_reverse_v6_control_contract(episode)
    )
    assert adapter._recorded_reverse_v6_control_contract_exact(episode)
    for mutation in (
        "schema_float",
        "passed_int",
        "count_float",
        "central_missing",
        "central_extra",
        "top_extra",
    ):
        changed = copy.deepcopy(episode)
        contract = changed["h4_control_contract"]
        if mutation == "schema_float":
            contract["schema_version"] = 4.0
        elif mutation == "passed_int":
            contract["passed"] = 1
        elif mutation == "count_float":
            contract["legacy_float64_control_audit_counts"][
                "sample_count"
            ] = 300.0
        elif mutation == "central_missing":
            contract["legacy_float64_control_audit_counts"].pop(
                "preclip_target_margin_violations"
            )
        elif mutation == "central_extra":
            contract["legacy_float64_control_audit_counts"]["extra"] = 0
        else:
            contract["extra"] = False
        assert not adapter._recorded_reverse_v6_control_contract_exact(changed)


def test_margin_and_precomposer_fields_cannot_be_swapped() -> None:
    episode = _valid_episode()
    original_margin = episode["control_trace"]["margin_clipped_targets"].copy()
    original_precomposer = episode["control_trace"]["v6_precomposer_targets"].copy()
    episode["control_trace"]["margin_clipped_targets"] = original_precomposer
    episode["control_trace"]["v6_precomposer_targets"] = original_margin
    base, post, core = adapter._load_frozen_modules()
    central = adapter.build_reverse_v6_call_graph(base, post, core)[
        "central_safety"
    ](episode)
    episode["safety_audit"] = central
    audit = adapter.rederive_reverse_v6_control_contract(episode)
    assert audit["passed"] is False
    assert audit["checks"]["central_safety_margin_stage_exact_host_float32"] is False
    assert audit["checks"]["single_precomposer_output_exact_host_float32"] is False
    assert central["desired_target_margin_violations"] != 74


def test_one_ulp_outward_float32_margin_injection_fails_schema4() -> None:
    episode = _valid_episode()
    margin = episode["control_trace"]["margin_clipped_targets"]
    upper = np.subtract(
        adapter.SAFE_UPPER_F32,
        adapter.MARGIN_F32,
        dtype=np.float32,
    )
    location = next(
        (row, int(joint))
        for row in range(adapter.CONTROL_TICKS)
        for joint in adapter.LEG_INDICES
        if margin[row, joint] == upper[joint]
    )
    row, joint = location
    margin[row, joint] = np.nextafter(
        margin[row, joint], np.float32(np.inf), dtype=np.float32
    )
    episode["safety_audit"] = adapter._legacy_float64_central_control_counts(
        {
            name: np.asarray(value)
            for name, value in episode["control_trace"].items()
            if name != "source_dtype"
        }
    )
    audit = adapter.rederive_reverse_v6_control_contract(episode)
    assert audit["passed"] is False
    assert audit["checks"]["central_safety_margin_stage_exact_host_float32"] is False


@pytest.mark.parametrize(
    ("field", "kind"),
    (
        ("v6_decoder_action", "matrix_ulp"),
        ("v6_decoder_raw_targets", "matrix_ulp"),
        ("v6_decoder_margin_targets", "matrix_ulp"),
        ("v6_upstream_margin_targets", "matrix_ulp"),
        ("v6_precomposer_targets", "matrix_ulp"),
        ("preclip_targets", "matrix_ulp"),
        ("margin_clipped_targets", "matrix_ulp"),
        ("applied_targets", "matrix_ulp"),
        ("v6_decoder_max_abs_error", "scalar_ulp"),
        ("v6_residual_authority_scale", "scalar_ulp"),
        ("v6_decoder_guard_lag_max_rad", "scalar_ulp"),
        ("v6_direct_physics_dynamic6_endpoint_max_abs_error", "scalar_ulp"),
        ("v6_direct_physics_snapshot_endpoint_max_abs_error", "scalar_ulp"),
        ("v6_teacher_source_phase_before", "scalar_ulp"),
        ("v6_teacher_table_phase", "scalar_ulp"),
        ("v6_decoder_leg_count", "count"),
        ("v6_precomposer_call_count", "count"),
        ("v6_final_guard_call_count", "count"),
        ("v6_decoder_assertion_token", "count"),
        ("v6_direct_physics_substep_count", "count"),
        ("v6_direct_physics_dynamic6_field_count", "count"),
        ("v6_direct_physics_snapshot_endpoint_field_count", "count"),
        ("v6_decoder_exact", "bool"),
        ("v6_decoder_leg_count_exact", "bool"),
        ("v6_decoder_head_zero_exact", "bool"),
        ("v6_teacher_target_contribution_zero_exact", "bool"),
        ("v6_decoder_all_finite", "bool"),
        ("v6_precomposer_call_count_exact", "bool"),
        ("v6_final_guard_call_count_exact", "bool"),
        ("v6_decoder_violation", "bool"),
        ("v6_direct_physics_dynamic6_endpoint_exact", "bool"),
        ("v6_direct_physics_dynamic6_all_finite", "bool"),
        ("v6_direct_physics_applied_target_exact", "bool"),
        ("v6_direct_physics_snapshot_endpoint_exact", "bool"),
        ("v6_direct_physics_snapshot_endpoint_all_finite", "bool"),
    ),
)
def test_each_v6_runtime_axis_mutation_fails_exact_rederivation(
    field: str, kind: str
) -> None:
    episode = _valid_episode()
    values = episode["control_trace"][field]
    if kind == "matrix_ulp":
        values[17, 3] = _next_float32(values[17, 3])
    elif kind == "scalar_ulp":
        values[17] = _next_float32(values[17])
    elif kind == "count":
        values[17] += 1
    else:
        values[17] = not bool(values[17])
    audit = adapter.rederive_reverse_v6_control_contract(episode)
    assert audit["passed"] is False
    assert not all(audit["checks"].values())


def test_signed_zero_numeric_type_missing_and_extra_trace_fields_fail_closed() -> None:
    episode = _valid_episode()
    episode["control_trace"]["v6_residual_authority_scale"][3] = np.float32(-0.0)
    assert (
        adapter.rederive_reverse_v6_control_contract(episode)["passed"] is False
    )

    episode = _valid_episode()
    episode["control_trace"]["v6_decoder_max_abs_error"] = np.zeros(
        adapter.CONTROL_TICKS, dtype=np.int32
    )
    with pytest.raises(ValueError, match="numeric type drifted"):
        adapter.rederive_reverse_v6_control_contract(episode)

    for mutation in ("missing", "extra"):
        episode = _valid_episode()
        if mutation == "missing":
            episode["control_trace"].pop("v6_decoder_exact")
        else:
            episode["control_trace"]["legacy"] = np.zeros(
                adapter.CONTROL_TICKS, dtype=np.float32
            )
        with pytest.raises(ValueError, match="schema drifted"):
            adapter.rederive_reverse_v6_control_contract(episode)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("v6_decoder_leg_count", np.int64(2**32 + 10)),
        ("v6_precomposer_call_count", np.int64(2**32 + 1)),
        ("v6_final_guard_call_count", np.int64(2**32 + 1)),
    ),
)
def test_integer_trace_requires_exact_int32_without_overflow_alias(
    field: str, value: np.int64
) -> None:
    episode = _valid_episode()
    episode["control_trace"][field] = np.full(
        adapter.CONTROL_TICKS, value, dtype=np.int64
    )
    with pytest.raises(ValueError, match="integer trace"):
        adapter.rederive_reverse_v6_control_contract(episode)


def test_json_native_and_json_roundtrip_preserve_valid_integer_trace_contract() -> None:
    _base, post, _core = adapter._load_frozen_modules()
    native = post.json_native(_valid_episode())
    assert adapter.rederive_reverse_v6_control_contract(native)["passed"] is True
    roundtripped = json.loads(json.dumps(native, allow_nan=False))
    assert (
        adapter.rederive_reverse_v6_control_contract(roundtripped)["passed"]
        is True
    )


@pytest.mark.parametrize(
    "alias",
    (
        [10.0] * adapter.CONTROL_TICKS,
        [True] * adapter.CONTROL_TICKS,
        [2**32 + 10] * adapter.CONTROL_TICKS,
        [-(2**31) - 1] * adapter.CONTROL_TICKS,
    ),
)
def test_serialized_integer_trace_rejects_type_and_int32_range_aliases(
    alias: list[Any],
) -> None:
    episode = _valid_episode()
    episode["control_trace"]["v6_decoder_leg_count"] = alias
    with pytest.raises(ValueError, match="integer trace"):
        adapter.rederive_reverse_v6_control_contract(episode)


@pytest.mark.parametrize("alias", (0, True))
@pytest.mark.parametrize(
    ("field", "index"),
    (
        ("initial_applied_targets", (5,)),
        ("v6_residual_authority_scale", (3,)),
        ("v6_decoder_raw_targets", (3, 4)),
    ),
)
def test_serialized_float_trace_rejects_mixed_int_and_bool_leaves(
    field: str, index: tuple[int, ...], alias: Any
) -> None:
    _base, post, _core = adapter._load_frozen_modules()
    episode = post.json_native(_valid_episode())
    target = episode["control_trace"][field]
    if len(index) == 1:
        target[index[0]] = alias
    else:
        target[index[0]][index[1]] = alias
    with pytest.raises(ValueError, match="numeric type drifted"):
        adapter.rederive_reverse_v6_control_contract(episode)


def test_live_float_trace_requires_exact_float32_capture_dtype() -> None:
    episode = _valid_episode()
    episode["control_trace"]["v6_decoder_max_abs_error"] = np.zeros(
        adapter.CONTROL_TICKS, dtype=np.float64
    )
    with pytest.raises(ValueError, match="numeric type drifted"):
        adapter.rederive_reverse_v6_control_contract(episode)


def test_episode_contract_rejects_legacy_claim_and_float_type_drift() -> None:
    episode = _valid_episode()
    episode["reverse_v6_absolute_decoder_contract"][
        "backward_residual_scale"
    ] = 0
    assert (
        adapter.rederive_reverse_v6_control_contract(episode)["passed"] is False
    )
    episode = _valid_episode()
    episode["reverse_composition_contract"] = {"residual_scale": 0.12}
    real = {
        "episodes": [episode, _valid_episode(), _valid_episode()],
        "official_v22_baseline": {
            "episodes": [_valid_episode(), _valid_episode(), _valid_episode()]
        },
    }
    for record in real["episodes"]:
        record.setdefault(
            "reverse_v6_absolute_decoder_contract",
            adapter._expected_reverse_v6_episode_contract(),
        )
    assert "reverse_composition_contract" in real["episodes"][0]


def _evaluation_contract_with_old_field() -> dict[str, Any]:
    return {
        "fixed_seeds": list(adapter.STRICT_SEEDS),
        "physical_command_mps_radps": list(adapter.STRICT_COMMAND),
        "duration_s": adapter.STRICT_DURATION_S,
        "control_timestep_s": adapter.CONTROL_DT_S,
        "physics_timestep_s": adapter.PHYSICS_DT_S,
        "control_tick_count": adapter.CONTROL_TICKS,
        "physics_substep_count": adapter.PHYSICS_SUBSTEPS,
        "gait_sample_count": adapter.GAIT_SAMPLES,
        "gait_quality_semantics": (
            "FULL_CURRENT_P0_RECOMPUTED_FROM_N_PLUS_ONE_SUBSTEP_TRACE"
        ),
        "reset": "EXACT_SAFE_INIT_NO_RESET_NOISE",
        "policy_inference": "BRAX_DETERMINISTIC_NORMAL_TANH_ACTOR116",
        "reverse_composition": (
            "PINNED_PERSISTENT_TEACHER_PLUS_TRAINABLE_RESIDUAL"
        ),
    }


def test_real_artifact_construction_uses_an_exact_nonmutating_skeleton_copy() -> None:
    artifact = {
        "artifact_kind": adapter.DIAGNOSTIC_ARTIFACT_KIND,
        "hardware_deployment": "PROHIBITED",
        "adoption_allowed": False,
        "release_allowed": False,
        "evaluation_contract": _evaluation_contract_with_old_field(),
        "runtime_provenance": {
            "reverse_composition_checks": {
                name: True for name in adapter.V6_ENVIRONMENT_CHECK_KEYS
            },
            "evaluation_source_hashes_pre": {"base": "a" * 64},
            "evaluation_source_hashes_post": {"base": "a" * 64},
        },
    }
    before = copy.deepcopy(artifact)
    converted = adapter._real_v6_artifact_from_frozen_skeleton(artifact)
    assert artifact == before
    assert converted is not artifact
    assert "reverse_composition" not in converted["evaluation_contract"]
    assert converted["evaluation_contract"]["reverse_action_parameterization"] == (
        adapter.REAL_REVERSE_EVALUATION_SEMANTICS
    )
    assert converted["evaluation_contract"][
        "legacy_teacher_plus_residual_runtime_authority"
    ] is False
    assert "reverse_composition_checks" not in converted["runtime_provenance"]
    assert converted["runtime_provenance"][
        "historical_teacher_residual_training_sources_runtime_authority"
    ] is False
    assert converted["promotion_allowed"] is False
    assert adapter._mapping_diff_paths(before, converted) == {
        ("promotion_allowed",),
        ("evaluation_contract", "reverse_composition"),
        ("evaluation_contract", "reverse_action_parameterization"),
        ("evaluation_contract", "reverse_teacher_role"),
        (
            "evaluation_contract",
            "legacy_teacher_plus_residual_runtime_authority",
        ),
        ("runtime_provenance", "reverse_composition_checks"),
        (
            "runtime_provenance",
            "reverse_v6_absolute_decoder_environment_checks",
        ),
        (
            "runtime_provenance",
            "historical_teacher_residual_training_sources_runtime_authority",
        ),
        ("runtime_provenance", "evaluation_source_hashes_current"),
    }
    compatibility = adapter._compatibility_validation_view(converted)
    assert compatibility["evaluation_contract"]["reverse_composition"] == (
        "PINNED_PERSISTENT_TEACHER_PLUS_TRAINABLE_RESIDUAL"
    )
    assert "reverse_composition" not in converted["evaluation_contract"]
    with pytest.raises(ValueError, match="frozen artifact decision skeleton"):
        adapter._real_v6_artifact_from_frozen_skeleton(converted)


@pytest.mark.parametrize(
    ("path", "wrong"),
    (
        (("artifact_kind",), "wrong"),
        (("promotion_allowed",), True),
        (("adoption_allowed",), True),
        (("release_allowed",), True),
        (("standalone_direct_runtime_allowed",), True),
        (("standalone_direct_runtime_allowed",), 0),
        (("standalone_direct_runtime_allowed",), "__DELETE_KEY__"),
        (("hardware_deployment",), "ALLOWED"),
        (("unexpected",), False),
        (
            (
                "evaluation_contract",
                "legacy_teacher_plus_residual_runtime_authority",
            ),
            True,
        ),
        (
            (
                "runtime_provenance",
                "historical_teacher_residual_training_sources_runtime_authority",
            ),
            True,
        ),
    ),
)
def test_final_graph_validator_rejects_without_repairing_caller_payload(
    path: tuple[str, ...], wrong: Any
) -> None:
    base, post, core = adapter._load_frozen_modules()
    graph = adapter.build_reverse_v6_call_graph(base, post, core)
    real_contract = _evaluation_contract_with_old_field()
    real_contract.pop("reverse_composition")
    real_contract.update(
        {
            "reverse_action_parameterization": (
                adapter.REAL_REVERSE_EVALUATION_SEMANTICS
            ),
            "reverse_teacher_role": "PHASE_TIMING_PRIOR_ONLY",
            "legacy_teacher_plus_residual_runtime_authority": False,
        }
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": adapter.DIAGNOSTIC_ARTIFACT_KIND,
        "hardware_deployment": "PROHIBITED",
        "execution_provider": "CPU",
        "promotion_allowed": False,
        "adoption_allowed": False,
        "release_allowed": False,
        "standalone_direct_runtime_allowed": False,
        "created_at_utc": "2026-08-10T00:00:00+00:00",
        "candidate": {},
        "evaluation_contract": real_contract,
        "central_hashes": {},
        "episodes": [],
        "official_v22_baseline": {},
        "summary": {},
        "runtime_provenance": {
            "historical_teacher_residual_training_sources_runtime_authority": (
                False
            ),
        },
    }
    current: Any = payload
    for key in path[:-1]:
        current = current[key]
    if wrong == "__DELETE_KEY__":
        del current[path[-1]]
    else:
        current[path[-1]] = wrong
    before = copy.deepcopy(payload)
    if len(path) == 1:
        assert not adapter._reverse_v6_top_level_contract_exact(payload)
        error = "CPU/decision contract drifted"
    else:
        assert adapter._reverse_v6_top_level_contract_exact(payload)
        error = "real artifact contract failed"
    with pytest.raises(ValueError, match=error):
        graph["artifact_validator"](payload)
    assert payload == before


def test_final_graph_validator_success_path_never_mutates_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, post, core = adapter._load_frozen_modules()
    graph = adapter.build_reverse_v6_call_graph(base, post, core)
    payload = {"sentinel": {"value": 1}}
    before = copy.deepcopy(payload)
    seen: dict[str, Any] = {}

    def strict_validator(value: Any, **kwargs: Any) -> dict[str, bool]:
        seen["value"] = copy.deepcopy(value)
        seen["initial"] = kwargs["_allow_initial_adapter_record_absent"]
        return {"passed": True}

    monkeypatch.setattr(
        adapter, "validate_reverse_v6_strict_artifact", strict_validator
    )
    assert graph["artifact_validator"](payload) == {"passed": True}
    assert payload == before
    assert seen == {"value": before, "initial": False}


def test_real_artifact_top_schema_version_rejects_boolean_alias() -> None:
    payload = {
        "schema_version": True,
        "artifact_kind": adapter.DIAGNOSTIC_ARTIFACT_KIND,
        "hardware_deployment": "PROHIBITED",
        "execution_provider": "CPU",
        "promotion_allowed": False,
        "adoption_allowed": False,
        "release_allowed": False,
    }
    calls = {"compatibility": 0}

    def compatibility(*_args: Any, **_kwargs: Any) -> Any:
        calls["compatibility"] += 1
        raise AssertionError("compatibility validator reached")

    with pytest.raises(ValueError, match="CPU/decision contract drifted"):
        adapter.validate_reverse_v6_strict_artifact(
            payload,
            compatibility_validator=compatibility,
        )
    assert calls == {"compatibility": 0}


@pytest.mark.parametrize(
    ("field", "alias"),
    (("duration_s", 6), ("control_tick_count", 300.0)),
)
def test_real_evaluation_contract_rejects_numeric_type_aliases(
    field: str, alias: Any
) -> None:
    contract = _evaluation_contract_with_old_field()
    contract.pop("reverse_composition")
    contract.update(
        {
            "reverse_action_parameterization": (
                adapter.REAL_REVERSE_EVALUATION_SEMANTICS
            ),
            "reverse_teacher_role": "PHASE_TIMING_PRIOR_ONLY",
            "legacy_teacher_plus_residual_runtime_authority": False,
        }
    )
    assert all(adapter._real_evaluation_contract_checks(contract).values())
    contract[field] = alias
    assert adapter._real_evaluation_contract_checks(contract)[
        "full_schema_type_and_value_exact"
    ] is False


def test_evaluation_source_extension_preserves_every_frozen_binding() -> None:
    base_hashes = {
        "scripts/evaluate_h4_training_candidate.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_candidate_evaluator"
        ][1],
        "safe_gait_experts/h4_post_training.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_post_training"
        ][1],
        "safe_gait_experts/h4_training_alignment.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_training_alignment"
        ][1],
        "scripts/train_h4_aligned_expert.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_runner"
        ][1],
    }
    adapter_hashes = {
        adapter.ADAPTER_SOURCE_KEY: "8" * 64,
        adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY: (
            adapter.PINNED_ADAPTER_AUTHORIZATION_SHA256
        ),
    }
    assert adapter._augment_evaluation_hashes(base_hashes, adapter_hashes) == {
        **base_hashes,
        **adapter_hashes,
    }
    drifted = dict(base_hashes)
    drifted["safe_gait_experts/h4_training_alignment.py"] = "0" * 64
    with pytest.raises(ValueError, match="source provenance drifted"):
        adapter._augment_evaluation_hashes(drifted, adapter_hashes)


def _adapter_provenance_fixture() -> tuple[
    dict[str, Any], dict[str, str], dict[str, str], dict[str, str]
]:
    candidate = {
        "candidate_params": "1" * 64,
        "candidate_manifest": "2" * 64,
        "candidate_config": "3" * 64,
        "candidate_result": "4" * 64,
        "candidate_training_curve": "5" * 64,
    }
    candidate_evaluation = {
        f"artifacts/test-candidate/{name}": digest
        for name, digest in candidate.items()
    }
    evaluation = {
        **adapter._pinned_evaluation_source_bindings(),
        **candidate_evaluation,
        adapter.ADAPTER_SOURCE_KEY: adapter.sha256_file(adapter.ADAPTER_PATH),
        adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY: (
            adapter.PINNED_ADAPTER_AUTHORIZATION_SHA256
        ),
    }
    frozen = {
        name: digest
        for name, (_path, digest) in adapter.PINNED_FROZEN_SOURCES.items()
    }
    record = {
        "schema_version": 1,
        "contract_id": adapter.ADAPTER_CONTRACT_ID,
        "training_contract_id": adapter.TRAINING_CONTRACT_ID,
        "method": (
            "FUNCTIONTYPE_CLONED_GLOBALS_WITH_ADAPTER_OWNED_"
            "V6_TRACE_AND_REDERIVATION"
        ),
        "runtime_factory": {
            "forward_v4_substep_contact": False,
            "forward_iteration_v6_contact_abort_island_only": False,
            "reverse_iteration_v6_absolute_full_leg_targets": True,
        },
        "legacy_reward_config": {
            "target_imitation": 0.0,
            "contact_imitation": 0.0,
            "tracking_sigma": adapter.TRACKING_SIGMA,
        },
        "backward_residual_scale": 0.0,
        "host_rederivation": adapter.CONTROL_TRACE_SEMANTICS,
        "physics_trace_semantics": (
            "DIRECT_MJX_STEP_REPLAY_FROM_CONTROL_ENTRY_WITH_ACTUAL_APPLIED_TARGETS"
        ),
        "physics_substeps_per_control": adapter.PHYSICS_SUBSTEPS_PER_CONTROL,
        "dynamic6_endpoint_bitwise_exact_required": True,
        "snapshot_endpoint_bitwise_exact_required": True,
        "legacy_float64_control_audit_diagnostic_only": True,
        "legacy_float64_control_audit_counts_exact_required": True,
        "adapter_source": {
            "path": adapter.ADAPTER_SOURCE_KEY,
            "sha256_pre": evaluation[adapter.ADAPTER_SOURCE_KEY],
            "sha256_post": evaluation[adapter.ADAPTER_SOURCE_KEY],
            "sha256_current": evaluation[adapter.ADAPTER_SOURCE_KEY],
            "unchanged": True,
        },
        "adapter_authorization": {
            "path": adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY,
            "sha256_pre": evaluation[adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY],
            "sha256_post": evaluation[adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY],
            "sha256_current": evaluation[
                adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY
            ],
            "unchanged": True,
        },
        "frozen_source_hashes_pre": copy.deepcopy(frozen),
        "frozen_source_hashes_post": copy.deepcopy(frozen),
        "candidate_bundle_hashes_pre": copy.deepcopy(candidate),
        "candidate_bundle_hashes_post": copy.deepcopy(candidate),
        "candidate_bundle_hashes_current": copy.deepcopy(candidate),
        "original_module_globals_and_function_references_unchanged": True,
        "promotion_evidence_allowed": False,
        "candidate_adoption_allowed": False,
        "release_allowed": False,
        "hardware_deployment": "PROHIBITED",
    }
    return record, evaluation, candidate, candidate_evaluation


def test_adapter_provenance_cross_binds_current_sources_and_candidate_files() -> None:
    record, evaluation, candidate, candidate_evaluation = (
        _adapter_provenance_fixture()
    )
    assert adapter._adapter_provenance_record_exact(
        record,
        current_evaluation_hashes=evaluation,
        current_candidate_hashes=candidate,
        current_candidate_evaluation_bindings=candidate_evaluation,
    )
    for mutation in (
        "adapter",
        "adapter_consistent_fake",
        "authorization",
        "frozen_evaluation",
        "candidate",
        "candidate_current",
        "candidate_evaluation",
    ):
        changed = copy.deepcopy(record)
        current_evaluation = copy.deepcopy(evaluation)
        current_candidate = copy.deepcopy(candidate)
        current_candidate_evaluation = copy.deepcopy(candidate_evaluation)
        if mutation in {"adapter", "adapter_consistent_fake"}:
            for suffix in ("pre", "post", "current"):
                changed["adapter_source"][f"sha256_{suffix}"] = "0" * 64
            if mutation == "adapter_consistent_fake":
                current_evaluation[adapter.ADAPTER_SOURCE_KEY] = "0" * 64
        elif mutation == "authorization":
            for suffix in ("pre", "post", "current"):
                changed["adapter_authorization"][f"sha256_{suffix}"] = "1" * 64
        elif mutation == "frozen_evaluation":
            frozen_path = next(iter(adapter._pinned_evaluation_source_bindings()))
            current_evaluation[frozen_path] = "6" * 64
        elif mutation == "candidate":
            for stage in ("pre", "post", "current"):
                changed[f"candidate_bundle_hashes_{stage}"][
                    "candidate_params"
                ] = "2" * 64
        elif mutation == "candidate_current":
            current_candidate["candidate_training_curve"] = "9" * 64
        else:
            candidate_path = next(iter(current_candidate_evaluation))
            current_evaluation[candidate_path] = "7" * 64
        assert not adapter._adapter_provenance_record_exact(
            changed,
            current_evaluation_hashes=current_evaluation,
            current_candidate_hashes=current_candidate,
            current_candidate_evaluation_bindings=(
                current_candidate_evaluation
            ),
        )


def test_central_hash_copies_bind_directly_to_live_pinned_sources() -> None:
    expected = adapter._pinned_central_hashes()
    assert adapter._central_hash_bindings_live_and_exact(
        copy.deepcopy(expected), copy.deepcopy(expected), None
    )
    assert adapter._central_hash_bindings_live_and_exact(
        copy.deepcopy(expected), copy.deepcopy(expected), copy.deepcopy(expected)
    )
    central_path = next(iter(expected))
    fake = copy.deepcopy(expected)
    fake[central_path] = "0" * 64
    assert not adapter._central_hash_bindings_live_and_exact(
        copy.deepcopy(fake), copy.deepcopy(fake), None
    )
    extra = {**expected, "extra": "1" * 64}
    assert not adapter._central_hash_bindings_live_and_exact(
        extra, copy.deepcopy(expected), None
    )


def test_adapter_provenance_record_is_stage_aware_and_finally_mandatory() -> None:
    record, evaluation, candidate, candidate_evaluation = (
        _adapter_provenance_fixture()
    )
    key = "reverse_v6_absolute_targets_evaluator_adapter"
    base_evaluation = {"scripts/evaluate_h4_training_candidate.py": "b" * 64}
    base_provenance = {
        "evaluation_source_hashes_pre": copy.deepcopy(base_evaluation),
        "evaluation_source_hashes_post": copy.deepcopy(base_evaluation),
        "evaluation_source_hashes_current": copy.deepcopy(base_evaluation),
    }
    assert adapter._adapter_provenance_stage_exact(
        base_provenance,
        current_evaluation_hashes=base_evaluation,
        current_candidate_hashes=None,
        current_candidate_evaluation_bindings=None,
        allow_initial_record_absent=True,
    )
    assert not adapter._adapter_provenance_stage_exact(
        base_provenance,
        current_evaluation_hashes=None,
        current_candidate_hashes=None,
        current_candidate_evaluation_bindings=None,
        allow_initial_record_absent=False,
    )
    final_without_record = {
        "evaluation_source_hashes_pre": copy.deepcopy(evaluation),
        "evaluation_source_hashes_post": copy.deepcopy(evaluation),
        "evaluation_source_hashes_current": copy.deepcopy(evaluation),
    }
    assert not adapter._adapter_provenance_stage_exact(
        final_without_record,
        current_evaluation_hashes=None,
        current_candidate_hashes=candidate,
        current_candidate_evaluation_bindings=candidate_evaluation,
        allow_initial_record_absent=True,
    )
    partial = copy.deepcopy(final_without_record)
    for stage in ("pre", "post", "current"):
        partial[f"evaluation_source_hashes_{stage}"].pop(
            adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY
        )
    assert not adapter._adapter_provenance_stage_exact(
        partial,
        current_evaluation_hashes=None,
        current_candidate_hashes=candidate,
        current_candidate_evaluation_bindings=candidate_evaluation,
        allow_initial_record_absent=True,
    )
    final_provenance = {**final_without_record, key: record}
    assert adapter._adapter_provenance_stage_exact(
        final_provenance,
        current_evaluation_hashes=evaluation,
        current_candidate_hashes=candidate,
        current_candidate_evaluation_bindings=candidate_evaluation,
        allow_initial_record_absent=False,
    )
    fake_evaluation = copy.deepcopy(evaluation)
    fake_evaluation[adapter.ADAPTER_SOURCE_KEY] = "0" * 64
    fake_record = copy.deepcopy(record)
    for suffix in ("pre", "post", "current"):
        fake_record["adapter_source"][f"sha256_{suffix}"] = "0" * 64
    fake_final = {
        "evaluation_source_hashes_pre": copy.deepcopy(fake_evaluation),
        "evaluation_source_hashes_post": copy.deepcopy(fake_evaluation),
        "evaluation_source_hashes_current": copy.deepcopy(fake_evaluation),
        key: fake_record,
    }
    assert not adapter._adapter_provenance_stage_exact(
        fake_final,
        current_evaluation_hashes=None,
        current_candidate_hashes=candidate,
        current_candidate_evaluation_bindings=candidate_evaluation,
        allow_initial_record_absent=False,
    )

    candidate_path = next(iter(candidate_evaluation))
    fake_candidate_evaluation = copy.deepcopy(evaluation)
    fake_candidate_evaluation[candidate_path] = "7" * 64
    fake_candidate_final = {
        "evaluation_source_hashes_pre": copy.deepcopy(
            fake_candidate_evaluation
        ),
        "evaluation_source_hashes_post": copy.deepcopy(
            fake_candidate_evaluation
        ),
        "evaluation_source_hashes_current": copy.deepcopy(
            fake_candidate_evaluation
        ),
        key: copy.deepcopy(record),
    }
    assert not adapter._adapter_provenance_stage_exact(
        fake_candidate_final,
        current_evaluation_hashes=None,
        current_candidate_hashes=candidate,
        current_candidate_evaluation_bindings=candidate_evaluation,
        allow_initial_record_absent=False,
    )
