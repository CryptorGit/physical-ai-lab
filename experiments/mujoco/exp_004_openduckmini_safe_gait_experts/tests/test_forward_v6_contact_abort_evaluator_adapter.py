from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


EXP_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    EXP_ROOT
    / "scripts"
    / "evaluate_h4_training_candidate_forward_v6_contact_abort_island_only_v1.py"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = _load_module("exp004_forward_v6_strict_adapter_test", ADAPTER_PATH)


class _FakeJax:
    @staticmethod
    def device_get(value: Any) -> Any:
        return value


def _exact_args() -> argparse.Namespace:
    return argparse.Namespace(
        params=adapter.EXPECTED_PARAMS_PATH.resolve(),
        params_sha256="1" * 64,
        manifest=adapter.EXPECTED_MANIFEST_PATH.resolve(),
        manifest_sha256="2" * 64,
        output=adapter.EXPECTED_OUTPUT_PATH.resolve(),
        trusted_run_root=adapter.EXPECTED_RUN_ROOT.resolve(),
        adapter_authorization=adapter.ADAPTER_AUTHORIZATION_PATH.resolve(),
        source_root=adapter.DEFAULT_SOURCE_ROOT.resolve(),
        generated_root=adapter.DEFAULT_GENERATED_ROOT.resolve(),
        v22_parent_checkpoint=adapter.DEFAULT_V22_PARENT_CHECKPOINT.resolve(),
        platform="cpu",
        allow_wiring_diagnostic=False,
        promotion_evidence_output=None,
    )


def _valid_runtime_trace() -> dict[str, np.ndarray]:
    trace: dict[str, np.ndarray] = {}
    true_fields = {
        "h4_forward_v6_adapter_forward_v4_flag",
        "h4_forward_v6_adapter_forward_v6_flag",
        "h4_forward_v6_adapter_dynamic6_endpoint_bitwise_exact",
        "h4_forward_v6_adapter_saved_dynamic6_all_finite",
        "h4_forward_v6_adapter_applied_target_bitwise_exact",
        "h4_forward_v6_adapter_gait_endpoint_bitwise_exact",
        "h4_forward_v6_adapter_snapshot_endpoint_bitwise_exact",
        "h4_forward_v6_adapter_endpoint_fields_all_finite",
        "h4_v4_single_authority_dynamic6_exact",
        "h4_v4_single_authority_dynamic6_field_count_exact",
        "h4_v4_saved_dynamic6_field_count_exact",
        "h4_v4_saved_dynamic6_all_finite",
        "h4_v4_telemetry_force_shape_valid",
        "h4_v4_telemetry_force_all_finite",
        "h4_v6_forward_contact_abort_routing_exact",
    }
    false_fields = {
        "h4_forward_v6_adapter_reverse_v6_flag",
        "h4_forward_v6_adapter_violation",
        "h4_v4_single_authority_violation",
        "h4_v6_forward_contact_abort_routing_violation",
    }
    count_values = {
        "h4_forward_v6_adapter_direct_primitive_substep_count": 10,
        "h4_forward_v6_adapter_dynamic6_field_count": 6,
        "h4_forward_v6_adapter_gait_endpoint_field_count": len(
            adapter.GAIT_ENDPOINT_FIELDS
        ),
        "h4_forward_v6_adapter_snapshot_endpoint_field_count": len(
            adapter.SNAPSHOT_ENDPOINT_FIELDS
        ),
        "h4_v4_single_authority_dynamic6_field_count": 6,
        "h4_v4_saved_dynamic6_substep_count": 10,
        "h4_v4_saved_dynamic6_field_count": 6,
        "h4_forward_v6_adapter_assertion_token": 0,
        "h4_v4_single_authority_assertion_token": 0,
        "h4_v6_forward_contact_abort_routing_assertion_token": 0,
    }
    float_values = {
        "h4_forward_v6_adapter_dynamic6_endpoint_max_abs_error": 0.0,
        "h4_forward_v6_adapter_gait_endpoint_max_abs_error": 0.0,
        "h4_forward_v6_adapter_snapshot_endpoint_max_abs_error": 0.0,
        "h4_v4_single_authority_dynamic6_max_abs_error": 0.0,
        "h4_v6_forward_contact_abort_island_loss": 0.25,
        "h4_v6_forward_contact_abort_off_gap_diagnostic_loss": 0.5,
        "h4_v6_forward_contact_abort_off_gap_reward_contribution": 0.0,
        "h4_v6_forward_contact_abort_pulse_reward_scale": -1.0,
    }
    for name in adapter.QUALIFYING_TRACE_FIELDS:
        if name in true_fields:
            trace[name] = np.ones(adapter.CONTROL_TICK_COUNT, dtype=bool)
        elif name in false_fields:
            trace[name] = np.zeros(adapter.CONTROL_TICK_COUNT, dtype=bool)
        elif name in count_values:
            trace[name] = np.full(
                adapter.CONTROL_TICK_COUNT, count_values[name], dtype=np.int32
            )
        elif name in float_values:
            trace[name] = np.full(
                adapter.CONTROL_TICK_COUNT, float_values[name], dtype=np.float32
            )
        else:  # pragma: no cover - field additions must be classified above
            raise AssertionError(name)
    return trace


def _valid_witness() -> dict[str, Any]:
    return adapter._summarize_forward_v6_trace(
        _valid_runtime_trace(), jax=_FakeJax
    )


def _valid_bundle(authorization: dict[str, Any]) -> Any:
    flags = {
        flag: flag == "forward_iteration_v6_contact_abort_island_only"
        for flag in adapter.ITERATION_MODE_FLAGS
    }
    config = {
        **flags,
        "training_contract_id": adapter.TRAINING_CONTRACT_ID,
        "authorized_iteration_v6_250k_contract_id": adapter.TRAINING_CONTRACT_ID,
        "forward_iteration_v6_contact_abort_island_only_authorization": {
            "path": str(adapter.FORWARD_V6_AUTHORIZATION_PATH.resolve()),
            "sha256": adapter.PINNED_FORWARD_V6_AUTHORIZATION_SHA256,
        },
        "reward_routing_contract": copy.deepcopy(
            authorization["reward_routing_contract"]
        ),
        "reward_scales": {"h4_contact_pulse_40ms": -1.0},
    }
    bundle = SimpleNamespace(
        params_path=adapter.EXPECTED_PARAMS_PATH.resolve(),
        params_sha256="1" * 64,
        manifest_path=adapter.EXPECTED_MANIFEST_PATH.resolve(),
        manifest_sha256="2" * 64,
        config_path=adapter.EXPECTED_CONFIG_PATH.resolve(),
        config_sha256="3" * 64,
        run_name=adapter.EXPECTED_CANDIDATE_ROOT.name,
        expert="forward",
        status="COMPLETED",
        activity="PPO_PILOT_TRAINING",
        config=config,
        manifest={
            **flags,
            "training_contract_id": adapter.TRAINING_CONTRACT_ID,
            "outputs": {
                "final_params": {
                    "path": str(adapter.EXPECTED_PARAMS_PATH.resolve()),
                    "sha256": "1" * 64,
                },
                "result": {
                    "path": str(adapter.EXPECTED_RESULT_PATH.resolve()),
                    "sha256": "4" * 64,
                },
                "training_curve": {
                    "path": str(adapter.EXPECTED_TRAINING_CURVE_PATH.resolve()),
                    "sha256": "5" * 64,
                },
            },
            "resolved_config": {
                "path": str(adapter.EXPECTED_CONFIG_PATH.resolve()),
                "sha256": "3" * 64,
                "canonical_sha256": "6" * 64,
            },
        },
    )
    bundle.candidate_record = lambda: {
        "run_name": bundle.run_name,
        "expert": bundle.expert,
        "activity": bundle.activity,
        "status": bundle.status,
    }
    return bundle


def test_authorization_and_all_frozen_bindings_are_current() -> None:
    payload = adapter.load_and_validate_adapter_authorization(
        adapter.ADAPTER_AUTHORIZATION_PATH
    )
    assert payload["scope"]["contract_id"] == adapter.ADAPTER_CONTRACT_ID
    assert adapter.sha256_file(adapter.ADAPTER_AUTHORIZATION_PATH) == (
        adapter.PINNED_ADAPTER_AUTHORIZATION_SHA256
    )
    assert adapter._verify_file_bindings(adapter.PINNED_FROZEN_SOURCES) == {
        label: digest
        for label, (_path, digest) in adapter.PINNED_FROZEN_SOURCES.items()
    }


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("schema_version",), True),
        (("scope", "contract_id"), "WRONG"),
        (("scope", "promotion_eligible"), True),
        (("exact_candidate_binding", "run_relative_path"), "forward/wrong"),
        (("factory_contract", "forward_v4_substep_contact"), False),
        (
            (
                "factory_contract",
                "reverse_iteration_v6_absolute_full_leg_targets",
            ),
            True,
        ),
        (("direct_primitive_trace_contract", "physics_substeps_per_control"), 9),
        (("direct_primitive_trace_contract", "one_ulp_tolerance_allowed"), True),
        (("runtime_contract", "fixed_seeds"), [1, 2, 3]),
        (("runtime_contract", "physical_command_mps_radps"), [0.04, 0.0, 0.0]),
        (("runtime_contract", "gait_sample_count"), 3000),
        (("runtime_contract", "control_tick_count"), 300.0),
        (("runtime_contract", "duration_s"), 6),
        (("runtime_contract", "forward_v4_violation_count"), False),
        (("runtime_contract", "strict_thresholds_unchanged"), False),
        (
            ("frozen_source_bindings", "h4_post_training", "sha256"),
            "0" * 64,
        ),
        (("authorization", "promotion_evidence"), True),
        (("decision", "hardware"), "ALLOWED"),
    ),
)
def test_authorization_semantic_drift_fails_closed(
    path: tuple[str, ...], value: Any
) -> None:
    payload = adapter.load_json_strict(adapter.ADAPTER_AUTHORIZATION_PATH)
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(ValueError, match="authorization"):
        adapter.validate_adapter_authorization_payload(payload)


@pytest.mark.parametrize(
    ("section", "mutation"),
    (
        ("runtime_contract", "extra"),
        ("runtime_contract", "missing"),
        ("factory_contract", "extra"),
        ("direct_primitive_trace_contract", "missing"),
        ("evidence_contract", "extra"),
    ),
)
def test_authorization_nested_missing_and_extra_keys_fail_closed(
    section: str, mutation: str
) -> None:
    payload = adapter.load_json_strict(adapter.ADAPTER_AUTHORIZATION_PATH)
    if mutation == "extra":
        payload[section]["unexpected"] = False
    else:
        payload[section].pop(next(iter(payload[section])))
    with pytest.raises(ValueError, match="authorization"):
        adapter.validate_adapter_authorization_payload(payload)


def test_cli_is_exact_cpu_completed_run_and_has_no_promotion_surface() -> None:
    args = _exact_args()
    adapter._validate_exact_cli(args, require_output_absent=False)
    for name, wrong in (
        ("params", adapter.EXP_ROOT / "wrong.pkl"),
        ("manifest", adapter.EXP_ROOT / "wrong.json"),
        ("output", adapter.EXP_ROOT / "wrong-output.json"),
        ("trusted_run_root", adapter.EXP_ROOT / "wrong-root"),
        ("params_sha256", "A" * 64),
        ("allow_wiring_diagnostic", True),
        ("promotion_evidence_output", adapter.EXP_ROOT / "promotion.json"),
    ):
        drifted = copy.copy(args)
        setattr(drifted, name, wrong)
        with pytest.raises(ValueError, match="CLI drifted"):
            adapter._validate_exact_cli(drifted, require_output_absent=False)
    with pytest.raises(SystemExit):
        adapter.build_parser().parse_args(
            [
                "--params",
                str(adapter.EXPECTED_PARAMS_PATH),
                "--params-sha256",
                "1" * 64,
                "--manifest",
                str(adapter.EXPECTED_MANIFEST_PATH),
                "--manifest-sha256",
                "2" * 64,
                "--promotion-evidence-output",
                str(adapter.EXPECTED_CANDIDATE_ROOT / "promotion.json"),
            ]
        )


def test_factory_binding_injects_exact_three_flags() -> None:
    calls: list[dict[str, Any]] = []

    def fake_factory(**kwargs: Any) -> type:
        calls.append(dict(kwargs))

        class Environment:
            h4_forward_v4_substep_contact = kwargs["forward_v4_substep_contact"]
            h4_forward_iteration_v6_contact_abort_island_only = kwargs[
                "forward_iteration_v6_contact_abort_island_only"
            ]
            h4_reverse_iteration_v6_absolute_full_leg_targets = kwargs[
                "reverse_iteration_v6_absolute_full_leg_targets"
            ]
            h4_forward_iteration_v6_compiled_assertion_bound = True
            h4_forward_iteration_v6_off_gap_reward_contribution = 0.0
            h4_forward_iteration_v6_contact_pulse_reward_scale = -1.0

        return Environment

    binding = adapter.ForwardV6FactoryBinding(fake_factory)
    result = binding(legacy_environment_class=object, stack={})
    assert result.h4_forward_v4_substep_contact is True
    assert binding.call_count == 1
    assert calls == [
        {
            "legacy_environment_class": object,
            "stack": {},
            "forward_v4_substep_contact": True,
            "forward_iteration_v6_contact_abort_island_only": True,
            "reverse_iteration_v6_absolute_full_leg_targets": False,
        }
    ]
    with pytest.raises(ValueError, match="attempted to supply"):
        binding(forward_v4_substep_contact=False)


@pytest.mark.parametrize(
    ("attribute", "wrong"),
    (
        ("h4_forward_v4_substep_contact", False),
        ("h4_forward_iteration_v6_contact_abort_island_only", False),
        ("h4_reverse_iteration_v6_absolute_full_leg_targets", True),
        ("h4_forward_iteration_v6_compiled_assertion_bound", False),
        ("h4_forward_iteration_v6_off_gap_reward_contribution", 1.0e-45),
        ("h4_forward_iteration_v6_contact_pulse_reward_scale", -2.0),
    ),
)
def test_factory_binding_rejects_flag_and_runtime_contract_drift(
    attribute: str, wrong: Any
) -> None:
    def fake_factory(**_kwargs: Any) -> type:
        values = {
            "h4_forward_v4_substep_contact": True,
            "h4_forward_iteration_v6_contact_abort_island_only": True,
            "h4_reverse_iteration_v6_absolute_full_leg_targets": False,
            "h4_forward_iteration_v6_compiled_assertion_bound": True,
            "h4_forward_iteration_v6_off_gap_reward_contribution": 0.0,
            "h4_forward_iteration_v6_contact_pulse_reward_scale": -1.0,
        }
        values[attribute] = wrong
        return type("Environment", (), values)

    with pytest.raises(RuntimeError, match="factory flag binding"):
        adapter.ForwardV6FactoryBinding(fake_factory)()


def test_strict_mapping_parity_rejects_one_ulp_and_signed_zero() -> None:
    reference = {"field": np.asarray([0.0, 1.0], dtype=np.float32)}
    exact = adapter._strict_mapping_parity(
        reference, copy.deepcopy(reference), ("field",), xp=np
    )
    assert bool(exact.exact)
    assert float(exact.max_abs_error) == 0.0

    one_ulp = {"field": reference["field"].copy()}
    one_ulp["field"][1] = np.nextafter(
        np.float32(1.0), np.float32(np.inf), dtype=np.float32
    )
    mismatch = adapter._strict_mapping_parity(
        reference, one_ulp, ("field",), xp=np
    )
    assert not bool(mismatch.exact)
    assert float(mismatch.max_abs_error) > 0.0

    signed_zero = {"field": reference["field"].copy()}
    signed_zero["field"][0] = np.float32(-0.0)
    mismatch = adapter._strict_mapping_parity(
        reference, signed_zero, ("field",), xp=np
    )
    assert not bool(mismatch.exact)
    assert float(mismatch.max_abs_error) == 0.0


def test_valid_runtime_trace_binds_all_three_authorities() -> None:
    witness = _valid_witness()
    adapter.validate_forward_v6_runtime_witness(witness)
    assert witness["control_tick_count"] == 300
    assert witness["physics_trace_row_count"] == 3000
    assert witness["direct_primitive_witness"][
        "dynamic6_bitwise_exact_control_count"
    ] == 300
    assert witness["direct_primitive_witness"][
        "gait_endpoint_bitwise_exact_control_count"
    ] == 300
    assert witness["actual_forward_v4_authority"]["violation_count"] == 0
    assert witness["forward_v6_reward_routing"][
        "off_gap_reward_contribution_sum"
    ] == 0.0
    assert witness["forward_v6_reward_routing"][
        "contact_pulse_reward_scale_min"
    ] == -1.0


@pytest.mark.parametrize(
    ("field", "wrong"),
    (
        (
            "h4_forward_v6_adapter_dynamic6_endpoint_max_abs_error",
            np.nextafter(np.float32(0.0), np.float32(np.inf), dtype=np.float32),
        ),
        (
            "h4_forward_v6_adapter_gait_endpoint_max_abs_error",
            np.nextafter(np.float32(0.0), np.float32(np.inf), dtype=np.float32),
        ),
        (
            "h4_v4_single_authority_dynamic6_max_abs_error",
            np.nextafter(np.float32(0.0), np.float32(np.inf), dtype=np.float32),
        ),
        ("h4_forward_v6_adapter_direct_primitive_substep_count", 9),
        ("h4_forward_v6_adapter_forward_v6_flag", False),
        ("h4_forward_v6_adapter_reverse_v6_flag", True),
        (
            "h4_v6_forward_contact_abort_off_gap_reward_contribution",
            np.nextafter(np.float32(0.0), np.float32(np.inf), dtype=np.float32),
        ),
        ("h4_v6_forward_contact_abort_pulse_reward_scale", -2.0),
        ("h4_v6_forward_contact_abort_routing_violation", True),
        ("h4_v6_forward_contact_abort_routing_assertion_token", 1),
    ),
)
def test_runtime_trace_one_ulp_count_flag_and_routing_negatives_fail_closed(
    field: str, wrong: Any
) -> None:
    trace = _valid_runtime_trace()
    trace[field][0] = wrong
    with pytest.raises(RuntimeError, match="runtime witness failed"):
        adapter._summarize_forward_v6_trace(trace, jax=_FakeJax)


@pytest.mark.parametrize(
    ("path", "wrong"),
    (
        (("direct_primitive_witness", "dynamic6_max_abs_error"), 5.0e-324),
        (("direct_primitive_witness", "gait_endpoint_max_abs_error"), 5.0e-324),
        (("direct_primitive_witness", "violation_count"), 1),
        (("actual_forward_v4_authority", "assertion_token_sum"), 1),
        (("forward_v6_reward_routing", "off_gap_reward_contribution_sum"), 5.0e-324),
        (("forward_v6_reward_routing", "contact_pulse_reward_scale_max"), -2.0),
        (("factory_flags", "reverse_iteration_v6_absolute_full_leg_targets"), True),
        (("checks", "field_set_exact"), False),
        (("schema_version",), True),
        (("control_tick_count",), 300.0),
        (("direct_primitive_witness", "violation_count"), False),
        (("direct_primitive_witness", "dynamic6_max_abs_error"), 0),
        (("forward_v6_reward_routing", "contact_pulse_reward_scale_min"), -1),
    ),
)
def test_serialized_witness_negatives_fail_closed(
    path: tuple[str, ...], wrong: Any
) -> None:
    witness = _valid_witness()
    current = witness
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = wrong
    with pytest.raises(ValueError, match="runtime witness drifted"):
        adapter.validate_forward_v6_runtime_witness(witness)


@pytest.mark.parametrize(
    ("section", "mutation"),
    (
        ("direct_primitive_witness", "extra"),
        ("direct_primitive_witness", "missing"),
        ("actual_forward_v4_authority", "extra"),
        ("forward_v6_reward_routing", "missing"),
        ("checks", "extra"),
    ),
)
def test_serialized_witness_nested_key_closure_fails_closed(
    section: str, mutation: str
) -> None:
    witness = _valid_witness()
    if mutation == "extra":
        witness[section]["unexpected"] = True
    else:
        witness[section].pop(next(iter(witness[section])))
    with pytest.raises(ValueError, match="runtime witness drifted"):
        adapter.validate_forward_v6_runtime_witness(witness)


def test_source_extension_preserves_all_frozen_hashes_and_rejects_collision() -> None:
    base_hashes = {
        str(path.relative_to(adapter.EXP_ROOT)).replace("\\", "/"): digest
        for path, digest in adapter.PINNED_FROZEN_SOURCES.values()
    }
    base_hashes["extra/source.py"] = "9" * 64
    adapter_hashes = {
        adapter.ADAPTER_SOURCE_KEY: "8" * 64,
        adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY: (
            adapter.PINNED_ADAPTER_AUTHORIZATION_SHA256
        ),
    }
    augmented = adapter._augment_evaluation_hashes(base_hashes, adapter_hashes)
    assert augmented == {**base_hashes, **adapter_hashes}
    assert base_hashes["extra/source.py"] == "9" * 64

    collided = dict(base_hashes)
    collided[adapter.ADAPTER_SOURCE_KEY] = "7" * 64
    with pytest.raises(ValueError, match="collision"):
        adapter._augment_evaluation_hashes(collided, adapter_hashes)

    missing = dict(base_hashes)
    missing.pop("safe_gait_experts/h4_training_alignment.py")
    with pytest.raises(ValueError, match="source provenance"):
        adapter._augment_evaluation_hashes(missing, adapter_hashes)

    drifted = dict(base_hashes)
    drifted["safe_gait_experts/h4_post_training.py"] = "0" * 64
    with pytest.raises(ValueError, match="source provenance"):
        adapter._augment_evaluation_hashes(drifted, adapter_hashes)


def _valid_provenance_record(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, str], Any]:
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    bundle = _valid_bundle(authorization)
    frozen = {
        label: digest
        for label, (_path, digest) in adapter.PINNED_FROZEN_SOURCES.items()
    }
    candidate = adapter._candidate_bundle_snapshot(bundle)
    monkeypatch.setattr(
        adapter,
        "_candidate_file_snapshot",
        lambda actual_bundle: copy.deepcopy(candidate)
        if actual_bundle is bundle
        else (_ for _ in ()).throw(AssertionError("wrong bundle")),
    )
    adapter_hashes = adapter._adapter_source_hashes()
    current = {
        **adapter_hashes,
        **{
            adapter._evaluation_path_key(path): digest
            for path, digest in adapter.PINNED_FROZEN_SOURCES.values()
        },
        **{
            value["path"]: value["sha256"] for value in candidate.values()
        },
    }
    record = adapter._adapter_provenance_record(
        adapter_hashes_pre=adapter_hashes,
        adapter_hashes_post=adapter_hashes,
        adapter_hashes_current=adapter_hashes,
        frozen_hashes_pre=frozen,
        frozen_hashes_post=frozen,
        frozen_hashes_current=frozen,
        candidate_snapshot_pre=candidate,
        candidate_snapshot_post=candidate,
        candidate_snapshot_current=candidate,
        witness_hashes=[str(index) * 64 for index in range(1, 7)],
    )
    return record, current, bundle


@pytest.mark.parametrize(
    ("path", "wrong"),
    (
        (("schema_version",), True),
        (("contract_id",), "WRONG"),
        (("factory_flags", "forward_v4_substep_contact"), False),
        (("factory_flags", "forward_v4_substep_contact"), 1),
        (("direct_primitive_trace", "physics_substeps_per_control"), 9),
        (("direct_primitive_trace", "control_tick_count"), 300.0),
        (("adapter_authorization", "sha256_pre"), "0" * 64),
        (("adapter_source", "sha256_post"), "4" * 64),
        (("frozen_source_hashes_post", "h4_post_training"), "0" * 64),
        (
            ("candidate_bundle_snapshot_post", "candidate_params", "sha256"),
            "0" * 64,
        ),
        (
            ("candidate_bundle_snapshot_pre", "candidate_params", "sha256"),
            "A" * 64,
        ),
        (("six_episode_witness_sha256",), ["1" * 64] * 5),
        (("original_module_globals_and_function_references_unchanged",), False),
        (("promotion_evidence_allowed",), True),
        (("hardware_deployment",), "ALLOWED"),
    ),
)
def test_provenance_negatives_fail_closed(
    path: tuple[str, ...], wrong: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, current_hashes, bundle = _valid_provenance_record(monkeypatch)
    current = record
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = wrong
    with pytest.raises(ValueError, match="provenance"):
        adapter.validate_adapter_provenance_record(
            record,
            current_evaluation_hashes=current_hashes,
            bundle=bundle,
        )


@pytest.mark.parametrize(
    ("section", "mutation"),
    (
        ("factory_flags", "extra"),
        ("direct_primitive_trace", "missing"),
        ("adapter_source", "extra"),
        ("adapter_authorization", "missing"),
        ("candidate_bundle_snapshot_pre", "extra"),
    ),
)
def test_provenance_nested_key_closure_fails_closed(
    section: str, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, current_hashes, bundle = _valid_provenance_record(monkeypatch)
    if mutation == "extra":
        record[section]["unexpected"] = "0" * 64
    else:
        record[section].pop(next(iter(record[section])))
    with pytest.raises(ValueError, match="provenance"):
        adapter.validate_adapter_provenance_record(
            record,
            current_evaluation_hashes=current_hashes,
            bundle=bundle,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_provenance_top_level_missing_and_extra_fail_closed(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, current_hashes, bundle = _valid_provenance_record(monkeypatch)
    if mutation == "missing":
        record.pop("schema_version")
    else:
        record["unexpected"] = False
    with pytest.raises(ValueError, match="top-level closure"):
        adapter.validate_adapter_provenance_record(
            record,
            current_evaluation_hashes=current_hashes,
            bundle=bundle,
        )


def test_provenance_record_is_exact_and_non_promotable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, current_hashes, bundle = _valid_provenance_record(monkeypatch)
    adapter.validate_adapter_provenance_record(
        record,
        current_evaluation_hashes=current_hashes,
        bundle=bundle,
    )
    assert record["promotion_evidence_allowed"] is False
    assert record["candidate_adoption_allowed"] is False
    assert record["release_allowed"] is False
    assert record["hardware_deployment"] == "PROHIBITED"
    assert record["direct_primitive_trace"]["outer_mjx_env_step_trace_used"] is False


def _valid_augmented_provenance_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, str], Any]:
    record, current_hashes, bundle = _valid_provenance_record(monkeypatch)
    runtime_witnesses = []
    for index in range(1, 7):
        witness = _valid_witness()
        witness["qualifying_trace_sha256"] = str(index) * 64
        runtime_witnesses.append(witness)
    central_hashes = adapter._live_pinned_central_hashes()
    payload = {
        "schema_version": 1,
        "artifact_kind": adapter.DIAGNOSTIC_ARTIFACT_KIND,
        "hardware_deployment": "PROHIBITED",
        "adoption_allowed": False,
        "release_allowed": False,
        "standalone_direct_runtime_allowed": False,
        "execution_provider": "CPU",
        "created_at_utc": "2026-08-09T00:00:00+00:00",
        "candidate": {},
        "central_hashes": copy.deepcopy(central_hashes),
        "episodes": [
            {
                "seed": seed,
                "forward_v6_runtime_witness": copy.deepcopy(
                    runtime_witnesses[index]
                ),
            }
            for index, seed in enumerate(adapter.FIXED_FORWARD_SEEDS)
        ],
        "official_v22_baseline": {
            "episodes": [
                {
                    "seed": seed,
                    "forward_v6_runtime_witness": copy.deepcopy(
                        runtime_witnesses[index + 3]
                    ),
                }
                for index, seed in enumerate(adapter.FIXED_FORWARD_SEEDS)
            ]
        },
        "summary": {},
        "promotion_allowed": False,
        "evaluation_contract": adapter._expected_final_evaluation_contract(),
        "runtime_provenance": {
            "evaluation_source_hashes_pre": copy.deepcopy(current_hashes),
            "evaluation_source_hashes_post": copy.deepcopy(current_hashes),
            "evaluation_source_hashes_current": copy.deepcopy(current_hashes),
            "central_hashes": copy.deepcopy(central_hashes),
            "forward_v6_strict_evaluator_adapter": record,
        },
    }
    return payload, current_hashes, bundle


def test_augmented_stage_binds_current_bytes_even_without_caller_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    before = copy.deepcopy(payload)
    adapter._validate_adapter_provenance_stage(
        payload,
        bundle=bundle,
        current_evaluation_hashes=None,
        allow_initial_record_absent=False,
    )
    assert payload == before


@pytest.mark.parametrize(
    ("path", "wrong"),
    (
        (("artifact_kind",), "wrong"),
        (("hardware_deployment",), "ALLOWED"),
        (("execution_provider",), "GPU"),
        (("promotion_allowed",), True),
        (("promotion_allowed",), 0),
        (("adoption_allowed",), True),
        (("release_allowed",), True),
        (("standalone_direct_runtime_allowed",), True),
        (("evaluation_contract", "duration_s"), 6),
        (("evaluation_contract", "control_tick_count"), 300.0),
        (("evaluation_contract", "fixed_seeds"), [True, 20261809, 20262809]),
        (
            (
                "evaluation_contract",
                "forward_v6_adapter",
                "promotion_eligible",
            ),
            True,
        ),
    ),
)
def test_final_kind_decision_hardware_and_contract_mutations_fail_unchanged(
    path: tuple[str, ...], wrong: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = wrong
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="final artifact surface"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )
    assert payload == before


def test_final_top_level_extra_field_fails_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    payload["unexpected"] = False
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="final artifact surface"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )
    assert payload == before


@pytest.mark.parametrize(
    "target",
    ("adapter_source", "adapter_authorization", "frozen", "candidate"),
)
def test_augmented_stage_rejects_equal_fake_digests_across_all_payload_maps(
    target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    provenance = payload["runtime_provenance"]
    record = provenance["forward_v6_strict_evaluator_adapter"]
    fake = "f" * 64
    if target in {"adapter_source", "adapter_authorization"}:
        key = (
            adapter.ADAPTER_SOURCE_KEY
            if target == "adapter_source"
            else adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY
        )
        for stage in ("pre", "post", "current"):
            provenance[f"evaluation_source_hashes_{stage}"][key] = fake
            record[target][f"sha256_{stage}"] = fake
    elif target == "frozen":
        label = "h4_post_training"
        path = adapter._evaluation_path_key(
            adapter.PINNED_FROZEN_SOURCES[label][0]
        )
        for stage in ("pre", "post", "current"):
            provenance[f"evaluation_source_hashes_{stage}"][path] = fake
            record[f"frozen_source_hashes_{stage}"][label] = fake
    else:
        label = "candidate_result"
        path = record["candidate_bundle_snapshot_current"][label]["path"]
        for stage in ("pre", "post", "current"):
            provenance[f"evaluation_source_hashes_{stage}"][path] = fake
            record[f"candidate_bundle_snapshot_{stage}"][label]["sha256"] = fake
    with pytest.raises(ValueError, match="provenance"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_record",
        "missing_candidate",
        "extra_candidate",
        "candidate_path",
        "one_file_drift",
        "partial_adapter_keys",
        "adapter_path",
    ),
)
def test_augmented_stage_provenance_closure_negatives(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    provenance = payload["runtime_provenance"]
    record = provenance["forward_v6_strict_evaluator_adapter"]
    if mutation == "missing_record":
        provenance.pop("forward_v6_strict_evaluator_adapter")
    elif mutation == "missing_candidate":
        for stage in ("pre", "post", "current"):
            record[f"candidate_bundle_snapshot_{stage}"].pop("candidate_result")
    elif mutation == "extra_candidate":
        extra = {"path": "artifacts/unexpected.bin", "sha256": "9" * 64}
        for stage in ("pre", "post", "current"):
            record[f"candidate_bundle_snapshot_{stage}"]["unexpected"] = copy.deepcopy(
                extra
            )
    elif mutation == "candidate_path":
        for stage in ("pre", "post", "current"):
            record[f"candidate_bundle_snapshot_{stage}"]["candidate_result"][
                "path"
            ] = "artifacts/wrong/run_result.json"
    elif mutation == "one_file_drift":
        for stage in ("pre", "post", "current"):
            record[f"candidate_bundle_snapshot_{stage}"]["candidate_result"][
                "sha256"
            ] = "9" * 64
    elif mutation == "partial_adapter_keys":
        provenance["evaluation_source_hashes_current"].pop(
            adapter.ADAPTER_SOURCE_KEY
        )
    else:
        record["adapter_source"]["path"] = "scripts/wrong.py"
    with pytest.raises(ValueError, match="forward-v6"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )


def test_augmented_stage_rejects_caller_map_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    caller = copy.deepcopy(current_hashes)
    caller[adapter.ADAPTER_SOURCE_KEY] = "0" * 64
    with pytest.raises(ValueError, match="caller evaluation hashes"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=caller,
            allow_initial_record_absent=False,
        )


@pytest.mark.parametrize(
    ("group", "index"),
    (("candidate", 1), ("baseline", 2)),
)
def test_augmented_stage_binds_each_actual_episode_witness_hash(
    group: str, index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    episodes = (
        payload["episodes"]
        if group == "candidate"
        else payload["official_v22_baseline"]["episodes"]
    )
    episodes[index]["forward_v6_runtime_witness"][
        "qualifying_trace_sha256"
    ] = "9" * 64
    with pytest.raises(ValueError, match="episode witness provenance"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )


def test_augmented_stage_rejects_swapped_episode_seed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    payload["episodes"][0], payload["episodes"][1] = (
        payload["episodes"][1],
        payload["episodes"][0],
    )
    with pytest.raises(ValueError, match="seed/order"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )


@pytest.mark.parametrize("mutation", ("swapped", "forged"))
def test_augmented_stage_rejects_internally_consistent_wrong_record_hashes(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    record = payload["runtime_provenance"][
        "forward_v6_strict_evaluator_adapter"
    ]
    hashes = record["six_episode_witness_sha256"]
    if mutation == "swapped":
        hashes[0], hashes[3] = hashes[3], hashes[0]
    else:
        hashes[4] = "9" * 64
    record["six_episode_witness_set_sha256"] = adapter.canonical_json_sha256(
        hashes
    )
    with pytest.raises(ValueError, match="episode witness provenance"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )


@pytest.mark.parametrize("mutation", ("fake", "extra"))
def test_augmented_stage_binds_live_central_trio_without_caller_map(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    provenance = payload["runtime_provenance"]
    record = provenance["forward_v6_strict_evaluator_adapter"]
    if mutation == "extra":
        payload["central_hashes"]["unexpected"] = "9" * 64
        provenance["central_hashes"]["unexpected"] = "9" * 64
    else:
        fake = "9" * 64
        payload["central_hashes"]["evaluator"] = fake
        provenance["central_hashes"]["evaluator"] = fake
        frozen_label = adapter.CENTRAL_ARTIFACT_SOURCE_LABELS["evaluator"]
        central_path = adapter._evaluation_path_key(
            adapter.PINNED_FROZEN_SOURCES[frozen_label][0]
        )
        for stage in ("pre", "post", "current"):
            provenance[f"evaluation_source_hashes_{stage}"][central_path] = fake
            record[f"frozen_source_hashes_{stage}"][frozen_label] = fake
    with pytest.raises(ValueError, match="live central provenance"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )


@pytest.mark.parametrize("schema", (True, 1.0, None))
def test_augmented_stage_requires_exact_integer_schema_version(
    schema: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    if schema is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = schema
    with pytest.raises(ValueError, match="schema_version"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )


def test_private_initial_stage_is_the_only_record_omission_path() -> None:
    hashes = {"base/source.py": "1" * 64}
    payload = {
        "schema_version": 1,
        "evaluation_contract": {},
        "runtime_provenance": {
            "evaluation_source_hashes_pre": copy.deepcopy(hashes),
            "evaluation_source_hashes_post": copy.deepcopy(hashes),
            "evaluation_source_hashes_current": copy.deepcopy(hashes),
        },
    }
    adapter._validate_adapter_provenance_stage(
        payload,
        bundle=None,
        current_evaluation_hashes=hashes,
        allow_initial_record_absent=True,
    )
    with pytest.raises(ValueError, match="private initial stage"):
        adapter._validate_adapter_provenance_stage(
            payload,
            bundle=None,
            current_evaluation_hashes=None,
            allow_initial_record_absent=False,
        )


def test_private_initial_current_map_normalization_uses_copy_only() -> None:
    hashes = {"base/source.py": "1" * 64}
    payload = {
        "schema_version": 1,
        "evaluation_contract": {},
        "runtime_provenance": {
            "evaluation_source_hashes_pre": copy.deepcopy(hashes),
            "evaluation_source_hashes_post": copy.deepcopy(hashes),
        },
    }
    before = copy.deepcopy(payload)
    view = adapter._private_initial_artifact_validation_view(
        payload, initial_pending=True
    )
    assert view is not payload
    assert view["runtime_provenance"]["evaluation_source_hashes_current"] == hashes
    assert payload == before
    assert adapter._private_initial_artifact_validation_view(
        payload, initial_pending=False
    ) is payload


def test_augmented_missing_current_map_is_not_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    payload["runtime_provenance"].pop("evaluation_source_hashes_current")
    before = copy.deepcopy(payload)
    view = adapter._private_initial_artifact_validation_view(
        payload, initial_pending=True
    )
    assert view is payload
    with pytest.raises(ValueError, match="source map is missing"):
        adapter._validate_adapter_provenance_stage(
            view,
            bundle=bundle,
            current_evaluation_hashes=None,
            allow_initial_record_absent=True,
        )
    assert payload == before


def test_exact_bundle_validation_and_pre_pickle_wrapper() -> None:
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    bundle = _valid_bundle(authorization)
    adapter._validate_forward_v6_bundle(bundle, authorization)
    calls: list[dict[str, Any]] = []

    def frozen_validator(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return bundle

    wrapper = adapter.ForwardV6BundleValidator(frozen_validator, authorization)
    assert wrapper(
        params_path=adapter.EXPECTED_PARAMS_PATH,
        manifest_path=adapter.EXPECTED_MANIFEST_PATH,
        expected_params_sha256="1" * 64,
        expected_manifest_sha256="2" * 64,
        trusted_run_root=adapter.EXPECTED_RUN_ROOT,
        allow_wiring_diagnostic=False,
    ) is bundle
    assert len(calls) == 1

    with pytest.raises(ValueError, match="trusted run root"):
        wrapper(
            trusted_run_root=adapter.EXP_ROOT / "wrong",
            allow_wiring_diagnostic=False,
        )
    assert len(calls) == 1
    with pytest.raises(ValueError, match="rejects wiring"):
        wrapper(
            trusted_run_root=adapter.EXPECTED_RUN_ROOT,
            allow_wiring_diagnostic=True,
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    ("missing_output", "extra_output", "result_path", "result_sha"),
)
def test_candidate_five_file_bundle_snapshot_closure_fails_closed(
    mutation: str,
) -> None:
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    bundle = _valid_bundle(authorization)
    if mutation == "missing_output":
        bundle.manifest["outputs"].pop("result")
    elif mutation == "extra_output":
        bundle.manifest["outputs"]["unexpected"] = {
            "path": str(adapter.EXPECTED_CANDIDATE_ROOT / "unexpected.bin"),
            "sha256": "9" * 64,
        }
    elif mutation == "result_path":
        bundle.manifest["outputs"]["result"]["path"] = str(
            adapter.EXPECTED_CANDIDATE_ROOT / "wrong.json"
        )
    else:
        bundle.manifest["outputs"]["result"]["sha256"] = "A" * 64
    with pytest.raises(ValueError, match="candidate"):
        adapter._candidate_bundle_snapshot(bundle)


@pytest.mark.parametrize(
    ("record", "key", "wrong"),
    (
        ("config", "training_contract_id", "WRONG"),
        ("config", "forward_iteration_v6_contact_abort_island_only", False),
        ("config", "reverse_iteration_v6_absolute_full_leg_targets", True),
        ("config", "forward_iteration_v6_contact_abort_island_only", 1),
        ("config", "reward_scales", {"h4_contact_pulse_40ms": -1}),
        ("manifest", "forward_iteration_v6_contact_abort_island_only", False),
        ("bundle", "status", "WIRING_PASS"),
        ("bundle", "expert", "reverse"),
    ),
)
def test_bundle_mode_identity_negatives_fail_closed(
    record: str, key: str, wrong: Any
) -> None:
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    bundle = _valid_bundle(authorization)
    if record == "bundle":
        setattr(bundle, key, wrong)
    else:
        getattr(bundle, record)[key] = wrong
    with pytest.raises(ValueError, match="candidate bundle drifted"):
        adapter._validate_forward_v6_bundle(bundle, authorization)


def test_six_episode_witness_closure_rejects_any_drift() -> None:
    episodes = [
        {
            "seed": seed,
            "forward_v6_runtime_witness": _valid_witness(),
        }
        for seed in adapter.FIXED_FORWARD_SEEDS
    ]
    artifact = {
        "episodes": copy.deepcopy(episodes),
        "official_v22_baseline": {"episodes": copy.deepcopy(episodes)},
    }
    assert len(adapter._six_runtime_witnesses(artifact)) == 6
    artifact["official_v22_baseline"]["episodes"][2][
        "forward_v6_runtime_witness"
    ]["direct_primitive_witness"]["gait_endpoint_bitwise_exact_control_count"] = 299
    with pytest.raises(ValueError, match="runtime witness drifted"):
        adapter._six_runtime_witnesses(artifact)


def _validation_only_call_graph(authorization: dict[str, Any]) -> dict[str, Any]:
    def factory(**_kwargs: Any) -> Any:
        return object

    def episode(**_kwargs: Any) -> dict[str, Any]:
        return {}

    def run_evaluation(args: Any) -> dict[str, Any]:
        return validate_h4_strict_artifact(
            args.payload,
            current_evaluation_hashes=args.current_evaluation_hashes,
        )

    def compatibility_validator(
        _payload: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        return {"passed": True}

    def bundle_validator(**_kwargs: Any) -> Any:
        return None

    def placeholder(*_args: Any, **_kwargs: Any) -> Any:
        return None

    post = SimpleNamespace(
        validate_trusted_h4_bundle=bundle_validator,
        validate_h4_strict_episode=placeholder,
        validate_h4_strict_artifact=compatibility_validator,
    )
    alignment = SimpleNamespace(
        make_h4_aligned_environment_class=factory,
        v4_authoritative_primitive_step=placeholder,
        scan_v4_instrumented_physics_trajectory=placeholder,
        reconstruct_v4_dynamic_state=placeholder,
    )
    base = SimpleNamespace(
        _make_environment_and_policy=factory,
        _compiled_rollout_for=placeholder,
        _run_episode=episode,
        run_evaluation=run_evaluation,
        validate_trusted_h4_bundle=post.validate_trusted_h4_bundle,
        validate_h4_strict_artifact=post.validate_h4_strict_artifact,
        make_h4_aligned_environment_class=(
            alignment.make_h4_aligned_environment_class
        ),
    )
    return adapter.build_forward_v6_call_graph(
        base,
        post,
        alignment,
        forward_v6_authorization=authorization,
    )


def test_public_graph_validator_rejects_post_write_witness_semantic_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    graph = _validation_only_call_graph(authorization)
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    payload["official_v22_baseline"]["episodes"][1][
        "forward_v6_runtime_witness"
    ]["direct_primitive_witness"][
        "gait_endpoint_bitwise_exact_control_count"
    ] = 299
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="runtime witness drifted"):
        graph["artifact_validator"](
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
        )
    assert payload == before


def _private_initial_payload() -> tuple[dict[str, Any], dict[str, str]]:
    hashes = {"base/source.py": "1" * 64}
    return (
        {
            "schema_version": 1,
            "evaluation_contract": {},
            "runtime_provenance": {
                "evaluation_source_hashes_pre": copy.deepcopy(hashes),
                "evaluation_source_hashes_post": copy.deepcopy(hashes),
            },
        },
        hashes,
    )


def test_public_graph_rejects_unaugmented_base_artifact_on_fresh_graph() -> None:
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    graph = _validation_only_call_graph(authorization)
    payload, hashes = _private_initial_payload()
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="source map is missing"):
        graph["artifact_validator"](
            payload,
            current_evaluation_hashes=hashes,
        )
    assert payload == before


def test_private_initial_validator_is_copy_only_one_shot_and_not_exposed() -> None:
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    graph = _validation_only_call_graph(authorization)
    payload, hashes = _private_initial_payload()
    before = copy.deepcopy(payload)
    private_validator = graph["run_evaluation"].__globals__[
        "validate_h4_strict_artifact"
    ]
    assert private_validator is not graph["artifact_validator"]
    assert private_validator not in graph.values()
    args = SimpleNamespace(payload=payload, current_evaluation_hashes=hashes)
    assert graph["run_evaluation"](args) == {"passed": True}
    assert payload == before
    with pytest.raises(RuntimeError, match="one-shot"):
        graph["run_evaluation"](args)
    assert payload == before


def test_public_final_graph_success_path_is_input_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    graph = _validation_only_call_graph(authorization)
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    before = copy.deepcopy(payload)
    assert graph["artifact_validator"](payload, bundle=bundle) == {
        "passed": True
    }
    assert payload == before


@pytest.mark.parametrize(
    ("mutation", "wrong"),
    (("value", 0), ("value", False), ("value", ""), ("missing", None)),
)
def test_public_graph_validator_rejects_invalid_created_at_without_mutation(
    mutation: str, wrong: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    graph = _validation_only_call_graph(authorization)
    payload, _current_hashes, bundle = _valid_augmented_provenance_payload(
        monkeypatch
    )
    if mutation == "missing":
        payload.pop("created_at_utc")
    else:
        payload["created_at_utc"] = wrong
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="final artifact surface"):
        graph["artifact_validator"](
            payload,
            bundle=bundle,
            current_evaluation_hashes=None,
        )
    assert payload == before


def test_functiontype_call_graph_and_module_state_are_invariant() -> None:
    base, post, alignment = adapter._load_frozen_modules()
    before = adapter._module_contract_snapshot(base, post, alignment)
    authorization = adapter.load_json_strict(adapter.FORWARD_V6_AUTHORIZATION_PATH)
    graph = adapter.build_forward_v6_call_graph(
        base,
        post,
        alignment,
        forward_v6_authorization=authorization,
    )
    assert graph["make_environment"] is not base._make_environment_and_policy
    assert graph["base_episode"] is not base._run_episode
    assert graph["run_evaluation"] is not base.run_evaluation
    assert graph["make_environment"].__globals__[
        "make_h4_aligned_environment_class"
    ] is graph["factory_binding"]
    assert graph["base_episode"].__globals__["_compiled_rollout_for"] is graph[
        "compiler"
    ]
    assert graph["run_evaluation"].__globals__["_run_episode"] is graph[
        "episode_runner"
    ]
    assert graph["run_evaluation"].__globals__["validate_trusted_h4_bundle"] is graph[
        "bundle_validator"
    ]
    assert graph["run_evaluation"].__globals__[
        "validate_h4_strict_artifact"
    ] is not graph["artifact_validator"]
    assert graph["run_evaluation"].__globals__[
        "validate_h4_strict_artifact"
    ] not in graph.values()
    assert graph["run_evaluation"].__globals__["STRICT_ARTIFACT_KIND"] == (
        adapter.DIAGNOSTIC_ARTIFACT_KIND
    )
    assert graph["compatibility_artifact_validator"].__globals__[
        "STRICT_ARTIFACT_KIND"
    ] == (
        adapter.DIAGNOSTIC_ARTIFACT_KIND
    )
    assert graph["compiler"]._cache == {}
    assert graph["compiler"]._pending == {}
    adapter._assert_module_contract_unchanged(
        base, post, alignment, before
    )


def test_cloned_function_never_mutates_original_globals_even_on_error() -> None:
    namespace: dict[str, Any] = {"VALUE": 1}
    exec(
        "def probe(fail=False):\n"
        "    if fail:\n"
        "        raise RuntimeError('probe')\n"
        "    return VALUE\n",
        namespace,
    )
    original = namespace["probe"]
    clone = adapter._clone_function(original, global_overrides={"VALUE": 2})
    assert clone() == 2
    assert original() == 1
    with pytest.raises(RuntimeError, match="probe"):
        clone(True)
    assert original() == 1
    assert original.__globals__["VALUE"] == 1


def test_read_only_joystick_view_suppresses_only_same_step_restore() -> None:
    def source_step() -> None:
        return None

    mjx_env = SimpleNamespace(step=source_step, mjx="primitive")
    joystick = SimpleNamespace(
        mjx_env=mjx_env,
        mjx="mjx",
        USE_MOTOR_SPEED_LIMITS=False,
    )
    view = adapter.ReadOnlyJoystickView(joystick)
    assert view.mjx_env.step is source_step
    assert view.mjx_env.mjx == "primitive"
    view.mjx_env.step = source_step
    assert view.mjx_env.suppressed_same_value_step_writes == 1
    assert joystick.mjx_env.step is source_step
    with pytest.raises(RuntimeError, match="mutable mjx_env write"):
        view.mjx_env.step = lambda: None
    with pytest.raises(RuntimeError, match="mutable joystick write"):
        view.USE_MOTOR_SPEED_LIMITS = True
    assert joystick.USE_MOTOR_SPEED_LIMITS is False
