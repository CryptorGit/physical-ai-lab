"""Source-closed strict evaluator adapter for the exact forward-v6 candidate.

The frozen H4 evaluator predates the forward-v6 factory flags and its outer
``mjx_env.step`` replacement cannot observe the direct primitive scan owned by
the v4/v6 environment.  This adapter leaves every frozen module unchanged.  It
clones the affected evaluator functions, injects the three exact factory flags,
and supplies an isolated rollout compiler that builds a second, exact ten-step
direct-primitive witness from the same control-entry data and the target used
by the actual environment step.

Every control endpoint must match in finite, same-shape, same-dtype, bitwise
dynamic6 state and in all gait-sample fields.  The actual v4 authority and v6
reward-routing telemetry are also required on every tick.  The output is a
write-new, CPU-only diagnostic strict artifact.  Promotion, adoption, release,
and hardware use are intentionally unsupported.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, ModuleType
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
BASE_EVALUATOR_PATH = EXP_ROOT / "scripts" / "evaluate_h4_training_candidate.py"
POST_TRAINING_PATH = EXP_ROOT / "safe_gait_experts" / "h4_post_training.py"
ALIGNMENT_PATH = EXP_ROOT / "safe_gait_experts" / "h4_training_alignment.py"
RUNNER_PATH = EXP_ROOT / "scripts" / "train_h4_aligned_expert.py"
SMOKE_PATH = EXP_ROOT / "scripts" / "smoke_h4_training_alignment.py"
ADAPTER_PATH = Path(__file__).resolve()
ADAPTER_AUTHORIZATION_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h4_forward_iteration_v6_strict_evaluator_adapter_v1_authorization.json"
)
FORWARD_V6_AUTHORIZATION_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h4_forward_iteration_v6_contact_abort_island_only_authorization.json"
)

ADAPTER_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_"
    "STRICT_EVALUATOR_ADAPTER_V1"
)
TRAINING_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_250K_FROM_V22"
)
DIAGNOSTIC_ARTIFACT_KIND = (
    "openduckmini_h4_forward_iteration_v6_contact_abort_island_only_"
    "strict_evaluation_diagnostic"
)
EXPECTED_RUN_ROOT = (
    EXP_ROOT / "artifacts" / "h4_iteration_v6_training_runs_20260809"
)
EXPECTED_CANDIDATE_ROOT = (
    EXPECTED_RUN_ROOT
    / "forward"
    / "h4_forward_250k_seed20260809_iteration_v6_contact_abort_island_only_level4_v1"
)
EXPECTED_PARAMS_PATH = EXPECTED_CANDIDATE_ROOT / "final_params.pkl"
EXPECTED_MANIFEST_PATH = EXPECTED_CANDIDATE_ROOT / "run_manifest.json"
EXPECTED_CONFIG_PATH = EXPECTED_CANDIDATE_ROOT / "resolved_config.json"
EXPECTED_RESULT_PATH = EXPECTED_CANDIDATE_ROOT / "run_result.json"
EXPECTED_TRAINING_CURVE_PATH = EXPECTED_CANDIDATE_ROOT / "training_curve.csv"
EXPECTED_OUTPUT_PATH = EXPECTED_CANDIDATE_ROOT / "h4_integrated_strict_3x6s_v1.json"
DEFAULT_SOURCE_ROOT = Path("/home/user/openduck_training_20260729")
DEFAULT_GENERATED_ROOT = EXP_ROOT / "artifacts" / "generated_playground"
DEFAULT_V22_PARENT_CHECKPOINT = Path(
    "/home/user/openduck_training_runs/"
    "calibrated_hybrid_yaw_cost_v22_300m/2026_07_29_154427_10485760"
)

PINNED_ADAPTER_AUTHORIZATION_SHA256 = (
    "88e5984f8475a66d879519057a3b1df0af617a6aff6804fbff1588820701f993"
)
PINNED_FORWARD_V6_AUTHORIZATION_SHA256 = (
    "8e8b722e9e3f8f4b3827a7ffd2dee3e3ee5a2d799bfd996e09b066ff71d93a04"
)
PINNED_POST_TRAINING_SHA256 = (
    "3fa23b759de391e963c8d16b74fa5019076a2bd0bc67dac384ace60310653240"
)
PINNED_ALIGNMENT_SHA256 = (
    "5da1d3a8a2c505a5ce4bc6621f76dd3031070cdb467a4cde96b4ed3c23190c02"
)
PINNED_RUNNER_SHA256 = (
    "d6d075ab257494599dec1beebdac523912b30d42dfc712699a9ebed3a131e8ef"
)
PINNED_SMOKE_SHA256 = (
    "87fcc39e5c339d5591db8e74a1bf9e321a8ad998fae332248d034f6a1db2d271"
)
PINNED_BASE_EVALUATOR_SHA256 = (
    "c214d086e6d66f6f9f98c7268481899e4133961dcc5355d738d4cd134a82e6ae"
)
PINNED_CENTRAL_EVALUATOR_SHA256 = (
    "31fb8846fc6267f28d032bca164dee2c872bfb484ebe272850100834bf1b1a9b"
)
PINNED_GAIT_QUALITY_SHA256 = (
    "b28e1ceb4cb6406411150bbad772a78203b8163bf10adcd79a5f31f83da5f2de"
)
PINNED_ROUTED_EVALUATION_SHA256 = (
    "f25ed858fbb5753fdcfd9e76f08396d0f09f95ac4696eb2e50cb5c128b80db09"
)

PINNED_FROZEN_SOURCES: Mapping[str, tuple[Path, str]] = {
    "h4_candidate_evaluator": (
        BASE_EVALUATOR_PATH,
        PINNED_BASE_EVALUATOR_SHA256,
    ),
    "h4_post_training": (POST_TRAINING_PATH, PINNED_POST_TRAINING_SHA256),
    "h4_training_alignment": (ALIGNMENT_PATH, PINNED_ALIGNMENT_SHA256),
    "h4_runner": (RUNNER_PATH, PINNED_RUNNER_SHA256),
    "h4_no_ppo_smoke": (SMOKE_PATH, PINNED_SMOKE_SHA256),
    "central_evaluator": (
        EXP_ROOT / "scripts" / "evaluate_routed_transitions.py",
        PINNED_CENTRAL_EVALUATOR_SHA256,
    ),
    "central_gait_quality": (
        EXP_ROOT / "safe_gait_experts" / "gait_quality.py",
        PINNED_GAIT_QUALITY_SHA256,
    ),
    "central_routed_evaluation": (
        EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py",
        PINNED_ROUTED_EVALUATION_SHA256,
    ),
    "forward_iteration_v6_authorization": (
        FORWARD_V6_AUTHORIZATION_PATH,
        PINNED_FORWARD_V6_AUTHORIZATION_SHA256,
    ),
}

ADAPTER_SOURCE_KEY = (
    "scripts/evaluate_h4_training_candidate_"
    "forward_v6_contact_abort_island_only_v1.py"
)
ADAPTER_AUTHORIZATION_SOURCE_KEY = (
    "artifacts/h4_forward_iteration_v6_strict_evaluator_adapter_v1_"
    "authorization.json"
)
CANDIDATE_FILE_PATHS: Mapping[str, Path] = {
    "candidate_params": EXPECTED_PARAMS_PATH,
    "candidate_manifest": EXPECTED_MANIFEST_PATH,
    "candidate_config": EXPECTED_CONFIG_PATH,
    "candidate_result": EXPECTED_RESULT_PATH,
    "candidate_training_curve": EXPECTED_TRAINING_CURVE_PATH,
}
CENTRAL_ARTIFACT_SOURCE_LABELS: Mapping[str, str] = {
    "evaluator": "central_evaluator",
    "gait_quality": "central_gait_quality",
    "routed_evaluation": "central_routed_evaluation",
}

FIXED_FORWARD_SEEDS = (20_260_809, 20_261_809, 20_262_809)
FIXED_FORWARD_COMMAND = (0.05, 0.0, 0.0)
CONTROL_TICK_COUNT = 300
PHYSICS_SUBSTEPS_PER_CONTROL = 10
PHYSICS_TRACE_ROW_COUNT = 3_000
GAIT_SAMPLE_COUNT = 3_001
DYNAMIC6_FIELD_COUNT = 6
GAIT_ENDPOINT_FIELDS = (
    "joint_qpos",
    "trunk_position",
    "trunk_yaw",
    "local_velocity",
    "local_yaw_rate",
    "contacts",
    "normal_force",
    "tangential_speed",
    "foot_points",
)
SNAPSHOT_ENDPOINT_FIELDS = (
    "joint_qpos",
    "full_qpos",
    "full_qvel",
    "height",
    "upright",
    "trunk_position",
    "trunk_yaw",
    "local_velocity",
    "local_yaw_rate",
    "contacts",
    "normal_force",
    "tangential_speed",
    "foot_points",
)

ITERATION_MODE_FLAGS = (
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

ADAPTER_TRACE_FIELDS = (
    "h4_forward_v6_adapter_forward_v4_flag",
    "h4_forward_v6_adapter_forward_v6_flag",
    "h4_forward_v6_adapter_reverse_v6_flag",
    "h4_forward_v6_adapter_direct_primitive_substep_count",
    "h4_forward_v6_adapter_dynamic6_endpoint_bitwise_exact",
    "h4_forward_v6_adapter_dynamic6_endpoint_max_abs_error",
    "h4_forward_v6_adapter_dynamic6_field_count",
    "h4_forward_v6_adapter_saved_dynamic6_all_finite",
    "h4_forward_v6_adapter_applied_target_bitwise_exact",
    "h4_forward_v6_adapter_gait_endpoint_bitwise_exact",
    "h4_forward_v6_adapter_gait_endpoint_max_abs_error",
    "h4_forward_v6_adapter_gait_endpoint_field_count",
    "h4_forward_v6_adapter_snapshot_endpoint_bitwise_exact",
    "h4_forward_v6_adapter_snapshot_endpoint_max_abs_error",
    "h4_forward_v6_adapter_snapshot_endpoint_field_count",
    "h4_forward_v6_adapter_endpoint_fields_all_finite",
    "h4_forward_v6_adapter_violation",
    "h4_forward_v6_adapter_assertion_token",
)
V4_AUTHORITY_TRACE_FIELDS = (
    "h4_v4_single_authority_dynamic6_exact",
    "h4_v4_single_authority_dynamic6_max_abs_error",
    "h4_v4_single_authority_dynamic6_field_count",
    "h4_v4_single_authority_dynamic6_field_count_exact",
    "h4_v4_saved_dynamic6_substep_count",
    "h4_v4_saved_dynamic6_field_count",
    "h4_v4_saved_dynamic6_field_count_exact",
    "h4_v4_saved_dynamic6_all_finite",
    "h4_v4_telemetry_force_shape_valid",
    "h4_v4_telemetry_force_all_finite",
    "h4_v4_single_authority_violation",
    "h4_v4_single_authority_assertion_token",
)
V6_ROUTING_TRACE_FIELDS = (
    "h4_v6_forward_contact_abort_routing_exact",
    "h4_v6_forward_contact_abort_island_loss",
    "h4_v6_forward_contact_abort_off_gap_diagnostic_loss",
    "h4_v6_forward_contact_abort_off_gap_reward_contribution",
    "h4_v6_forward_contact_abort_pulse_reward_scale",
    "h4_v6_forward_contact_abort_routing_violation",
    "h4_v6_forward_contact_abort_routing_assertion_token",
)
QUALIFYING_TRACE_FIELDS = (
    *ADAPTER_TRACE_FIELDS,
    *V4_AUTHORITY_TRACE_FIELDS,
    *V6_ROUTING_TRACE_FIELDS,
)
RUNTIME_WITNESS_CHECK_KEYS = (
    "field_set_exact",
    "control_trace_shape_exact",
    "finite_fields_exact",
    "boolean_field_dtypes_exact",
    "integer_field_dtypes_exact",
    "float_field_dtypes_exact",
    "required_true_every_tick",
    "required_false_every_tick",
    "direct_primitive_trace_count_exact",
    "dynamic6_field_count_exact",
    "gait_endpoint_field_count_exact",
    "snapshot_endpoint_field_count_exact",
    "adapter_dynamic6_zero_error_exact",
    "adapter_gait_endpoint_zero_error_exact",
    "adapter_snapshot_endpoint_zero_error_exact",
    "v4_dynamic6_zero_error_exact",
    "v4_dynamic6_field_count_exact",
    "v4_saved_substep_count_exact",
    "v4_saved_field_count_exact",
    "v4_assertion_token_zero_exact",
    "v6_off_gap_reward_contribution_zero_exact",
    "v6_contact_pulse_scale_minus_one_exact",
    "v6_assertion_token_zero_exact",
    "adapter_assertion_token_zero_exact",
)


class StrictParity(NamedTuple):
    exact: Any
    max_abs_error: Any
    field_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_strict(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"authorization field is missing: {'.'.join(keys)}")
        current = current[key]
    return current


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be an exact lowercase SHA256")
    return value


def _type_exact_equal(actual: Any, expected: Any) -> bool:
    """Recursively compare JSON-like values without bool/int aliasing."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _type_exact_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _type_exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def validate_adapter_authorization_payload(payload: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "artifact_kind",
        "status",
        "hardware_deployment",
        "authorization",
        "scope",
        "exact_candidate_binding",
        "frozen_source_bindings",
        "factory_contract",
        "direct_primitive_trace_contract",
        "runtime_contract",
        "evidence_contract",
        "decision",
    }
    if set(payload) != expected_top:
        raise ValueError("forward-v6 adapter authorization top-level closure drifted")
    expected_authorization = {
        "exact_candidate_strict_evaluation": True,
        "ppo_training": False,
        "promotion_evidence": False,
        "candidate_adoption": False,
        "package_release": False,
        "hardware": False,
    }
    candidate = _nested(payload, "exact_candidate_binding")
    factory = _nested(payload, "factory_contract")
    trace = _nested(payload, "direct_primitive_trace_contract")
    runtime = _nested(payload, "runtime_contract")
    evidence = _nested(payload, "evidence_contract")
    expected_scope = {
        "contract_id": ADAPTER_CONTRACT_ID,
        "training_contract_id": TRAINING_CONTRACT_ID,
        "expert": "forward",
        "adapter_source_path": ADAPTER_SOURCE_KEY,
        "evaluation_artifact_kind": DIAGNOSTIC_ARTIFACT_KIND,
        "method": (
            "FUNCTIONTYPE_CLONED_GLOBALS_AND_ISOLATED_CALLABLE_STATE_"
            "WITHOUT_MODULE_MUTATION"
        ),
        "promotion_eligible": False,
    }
    expected_candidate = {
        "trusted_run_root_relative_path": (
            "artifacts/h4_iteration_v6_training_runs_20260809"
        ),
        "run_relative_path": (
            "forward/h4_forward_250k_seed20260809_iteration_v6_"
            "contact_abort_island_only_level4_v1"
        ),
        "params_basename": "final_params.pkl",
        "manifest_basename": "run_manifest.json",
        "output_basename": "h4_integrated_strict_3x6s_v1.json",
        "status": "COMPLETED",
        "activity": "PPO_PILOT_TRAINING",
        "candidate_hash_binding": (
            "EXACT_CLI_HASHES_CROSS_BOUND_TO_VALIDATED_COMPLETED_BUNDLE"
        ),
        "overwrite_allowed": False,
    }
    expected_factory = {
        "forward_v4_substep_contact": True,
        "forward_iteration_v6_contact_abort_island_only": True,
        "reverse_iteration_v6_absolute_full_leg_targets": False,
        "flag_injection_scope": "CLONED_EVALUATOR_FACTORY_CALL_ONLY",
        "mutable_module_global_changes_allowed": False,
    }
    expected_trace = {
        "control_tick_count": CONTROL_TICK_COUNT,
        "physics_substeps_per_control": PHYSICS_SUBSTEPS_PER_CONTROL,
        "physics_trace_row_count": PHYSICS_TRACE_ROW_COUNT,
        "control_entry_source": (
            "SAME_IMMUTABLE_CONTROL_ENTRY_DATA_AS_ACTUAL_ENVIRONMENT_STEP"
        ),
        "applied_target_source": "ACTUAL_FORWARD_V6_ENVIRONMENT_STEP_DATA_CTRL",
        "primitive": (
            "joystick.mjx_env.mjx.step(model,data.replace(ctrl=applied_target))"
        ),
        "measurement_replay": "SAVED_DYNAMIC6_RECONSTRUCT_THEN_MJX_FORWARD",
        "endpoint_comparison": (
            "SAME_DTYPE_SHAPE_FINITE_BITWISE_DYNAMIC6_EXACT"
        ),
        "dynamic6_fields": [
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "time",
            "qacc_warmstart",
        ],
        "actual_next_state_gait_endpoint_comparison": (
            "SAME_DTYPE_SHAPE_FINITE_BITWISE_EXACT_AT_EVERY_CONTROL_ENDPOINT"
        ),
        "actual_next_state_gait_endpoint_fields": list(GAIT_ENDPOINT_FIELDS),
        "one_ulp_tolerance_allowed": False,
        "frozen_outer_mjx_env_step_trace_used": False,
    }
    expected_runtime = {
        "fixed_seeds": list(FIXED_FORWARD_SEEDS),
        "physical_command_mps_radps": list(FIXED_FORWARD_COMMAND),
        "control_tick_count": CONTROL_TICK_COUNT,
        "physics_substep_count": PHYSICS_TRACE_ROW_COUNT,
        "gait_sample_count": GAIT_SAMPLE_COUNT,
        "forward_v4_authority_exact_required": True,
        "forward_v4_authority_max_abs_error": 0.0,
        "forward_v4_saved_substep_count": PHYSICS_SUBSTEPS_PER_CONTROL,
        "forward_v4_dynamic_field_count": DYNAMIC6_FIELD_COUNT,
        "forward_v4_violation_count": 0,
        "forward_v4_assertion_token_sum": 0,
        "forward_v6_routing_exact_required": True,
        "forward_v6_off_gap_reward_contribution": 0.0,
        "forward_v6_contact_pulse_reward_scale": -1.0,
        "forward_v6_routing_violation_count": 0,
        "forward_v6_assertion_token_sum": 0,
        "candidate_and_official_v22_episode_count": 6,
        "fixed_seed_count_per_policy": 3,
        "duration_s": 6.0,
        "execution_provider": "CPU",
        "strict_threshold_source": "safe_gait_experts/gait_quality.py",
        "strict_threshold_source_sha256": PINNED_GAIT_QUALITY_SHA256,
        "strict_thresholds_unchanged": True,
    }
    expected_evidence = {
        "write_new_standard_json_only": True,
        "adapter_source_sha256_bound_at_runtime": True,
        "adapter_authorization_sha256_bound_at_runtime": True,
        "frozen_sources_checked_before_pickle_and_after_evaluation": True,
        "candidate_inputs_checked_before_pickle_and_after_evaluation": True,
        "base_evaluation_source_hashes_preserved_and_extended_without_collision": True,
        "six_episode_runtime_witnesses_required": True,
        "promotion_builder_or_output_prohibited": True,
    }
    expected_decision = {
        "promotion": "PROHIBITED",
        "candidate_adoption": "BLOCKED",
        "release": "BLOCKED",
        "hardware": "PROHIBITED",
    }
    expected_sources = {
        label: {
            "path": str(path.relative_to(EXP_ROOT)).replace("\\", "/"),
            "sha256": digest,
        }
        for label, (path, digest) in PINNED_FROZEN_SOURCES.items()
    }
    expected_payload = {
        "schema_version": 1,
        "artifact_kind": (
            "openduckmini_h4_forward_iteration_v6_strict_evaluator_"
            "adapter_authorization"
        ),
        "status": "AUTHORIZED_EXACT_CPU_STRICT_DIAGNOSTIC_ONLY",
        "hardware_deployment": "PROHIBITED",
        "authorization": expected_authorization,
        "scope": expected_scope,
        "exact_candidate_binding": expected_candidate,
        "frozen_source_bindings": expected_sources,
        "factory_contract": expected_factory,
        "direct_primitive_trace_contract": expected_trace,
        "runtime_contract": expected_runtime,
        "evidence_contract": expected_evidence,
        "decision": expected_decision,
    }
    checks = {
        "full_payload_type_exact": _type_exact_equal(payload, expected_payload),
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("artifact_kind")
        == "openduckmini_h4_forward_iteration_v6_strict_evaluator_adapter_authorization",
        "status": payload.get("status")
        == "AUTHORIZED_EXACT_CPU_STRICT_DIAGNOSTIC_ONLY",
        "hardware": payload.get("hardware_deployment") == "PROHIBITED",
        "authorization": _type_exact_equal(
            payload.get("authorization"), expected_authorization
        ),
        "scope_type_exact": _type_exact_equal(payload.get("scope"), expected_scope),
        "candidate_type_exact": _type_exact_equal(candidate, expected_candidate),
        "factory_type_exact": _type_exact_equal(factory, expected_factory),
        "trace_type_exact": _type_exact_equal(trace, expected_trace),
        "runtime_type_exact": _type_exact_equal(runtime, expected_runtime),
        "evidence_type_exact": _type_exact_equal(evidence, expected_evidence),
        "decision_type_exact": _type_exact_equal(
            payload.get("decision"), expected_decision
        ),
        "contract": _nested(payload, "scope", "contract_id")
        == ADAPTER_CONTRACT_ID,
        "training_contract": _nested(payload, "scope", "training_contract_id")
        == TRAINING_CONTRACT_ID,
        "expert": _nested(payload, "scope", "expert") == "forward",
        "adapter_path": _nested(payload, "scope", "adapter_source_path")
        == ADAPTER_SOURCE_KEY,
        "artifact_kind": _nested(payload, "scope", "evaluation_artifact_kind")
        == DIAGNOSTIC_ARTIFACT_KIND,
        "method": _nested(payload, "scope", "method")
        == "FUNCTIONTYPE_CLONED_GLOBALS_AND_ISOLATED_CALLABLE_STATE_WITHOUT_MODULE_MUTATION",
        "promotion_ineligible": _nested(payload, "scope", "promotion_eligible")
        is False,
        "candidate_root": candidate.get("trusted_run_root_relative_path")
        == "artifacts/h4_iteration_v6_training_runs_20260809",
        "candidate_run": candidate.get("run_relative_path")
        == (
            "forward/h4_forward_250k_seed20260809_iteration_v6_"
            "contact_abort_island_only_level4_v1"
        ),
        "params_name": candidate.get("params_basename") == "final_params.pkl",
        "manifest_name": candidate.get("manifest_basename") == "run_manifest.json",
        "output_name": candidate.get("output_basename")
        == "h4_integrated_strict_3x6s_v1.json",
        "candidate_status": candidate.get("status") == "COMPLETED",
        "candidate_activity": candidate.get("activity") == "PPO_PILOT_TRAINING",
        "candidate_hash_binding": candidate.get("candidate_hash_binding")
        == "EXACT_CLI_HASHES_CROSS_BOUND_TO_VALIDATED_COMPLETED_BUNDLE",
        "overwrite": candidate.get("overwrite_allowed") is False,
        "factory_fv4": factory.get("forward_v4_substep_contact") is True,
        "factory_fv6": factory.get(
            "forward_iteration_v6_contact_abort_island_only"
        )
        is True,
        "factory_rv6": factory.get(
            "reverse_iteration_v6_absolute_full_leg_targets"
        )
        is False,
        "factory_scope": factory.get("flag_injection_scope")
        == "CLONED_EVALUATOR_FACTORY_CALL_ONLY",
        "factory_no_mutation": factory.get("mutable_module_global_changes_allowed")
        is False,
        "trace_controls": runtime.get("control_tick_count")
        == CONTROL_TICK_COUNT,
        "trace_substeps": trace.get("physics_substeps_per_control")
        == PHYSICS_SUBSTEPS_PER_CONTROL,
        "trace_rows": trace.get("physics_trace_row_count")
        == PHYSICS_TRACE_ROW_COUNT,
        "trace_dynamic6": trace.get("dynamic6_fields")
        == ["qpos", "qvel", "act", "ctrl", "time", "qacc_warmstart"],
        "trace_gait_fields": trace.get("actual_next_state_gait_endpoint_fields")
        == list(GAIT_ENDPOINT_FIELDS),
        "trace_one_ulp": trace.get("one_ulp_tolerance_allowed") is False,
        "trace_outer_rejected": trace.get("frozen_outer_mjx_env_step_trace_used")
        is False,
        "runtime_seeds": runtime.get("fixed_seeds") == list(FIXED_FORWARD_SEEDS),
        "runtime_command": runtime.get("physical_command_mps_radps")
        == list(FIXED_FORWARD_COMMAND),
        "runtime_substeps": runtime.get("physics_substep_count")
        == PHYSICS_TRACE_ROW_COUNT,
        "runtime_samples": runtime.get("gait_sample_count") == GAIT_SAMPLE_COUNT,
        "runtime_duration": runtime.get("duration_s") == 6.0,
        "runtime_provider": runtime.get("execution_provider") == "CPU",
        "runtime_v4_error": runtime.get("forward_v4_authority_max_abs_error")
        == 0.0,
        "runtime_v4_count": runtime.get("forward_v4_dynamic_field_count")
        == DYNAMIC6_FIELD_COUNT,
        "runtime_v6_contribution": runtime.get(
            "forward_v6_off_gap_reward_contribution"
        )
        == 0.0,
        "runtime_v6_scale": runtime.get(
            "forward_v6_contact_pulse_reward_scale"
        )
        == -1.0,
        "runtime_threshold_path": runtime.get("strict_threshold_source")
        == "safe_gait_experts/gait_quality.py",
        "runtime_threshold_sha": runtime.get("strict_threshold_source_sha256")
        == PINNED_GAIT_QUALITY_SHA256,
        "runtime_threshold_unchanged": runtime.get("strict_thresholds_unchanged")
        is True,
        "evidence_write_new": evidence.get("write_new_standard_json_only") is True,
        "evidence_six": evidence.get("six_episode_runtime_witnesses_required")
        is True,
        "evidence_no_promotion": evidence.get(
            "promotion_builder_or_output_prohibited"
        )
        is True,
        "decision_promotion": _nested(payload, "decision", "promotion")
        == "PROHIBITED",
        "decision_adoption": _nested(payload, "decision", "candidate_adoption")
        == "BLOCKED",
        "decision_release": _nested(payload, "decision", "release") == "BLOCKED",
        "decision_hardware": _nested(payload, "decision", "hardware")
        == "PROHIBITED",
    }
    source_payload = _nested(payload, "frozen_source_bindings")
    if set(source_payload) != set(PINNED_FROZEN_SOURCES):
        checks["frozen_source_labels"] = False
    else:
        checks["frozen_source_labels"] = True
        for label, (path, digest) in PINNED_FROZEN_SOURCES.items():
            record = source_payload[label]
            checks[f"source_{label}"] = record == {
                "path": str(path.relative_to(EXP_ROOT)).replace("\\", "/"),
                "sha256": digest,
            }
    if not all(checks.values()):
        raise ValueError(
            f"forward-v6 adapter authorization semantic binding failed: {checks}"
        )


def load_and_validate_adapter_authorization(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved != ADAPTER_AUTHORIZATION_PATH.resolve():
        raise ValueError("forward-v6 adapter authorization path must remain exact")
    actual_sha = sha256_file(resolved)
    if actual_sha != PINNED_ADAPTER_AUTHORIZATION_SHA256:
        raise ValueError(
            f"forward-v6 adapter authorization SHA256 drifted: {actual_sha}"
        )
    payload = load_json_strict(resolved)
    validate_adapter_authorization_payload(payload)
    return payload


def _verify_file_bindings(
    bindings: Mapping[str, tuple[Path, str]],
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for label, (path, expected_sha) in bindings.items():
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ValueError(f"pinned file is missing: {label}: {resolved}")
        digest = sha256_file(resolved)
        if digest != expected_sha:
            raise ValueError(f"pinned file SHA256 drifted: {label}: {digest}")
        actual[label] = digest
    return actual


def _clone_function(
    function: FunctionType, *, global_overrides: Mapping[str, Any]
) -> FunctionType:
    if not isinstance(function, FunctionType) or function.__closure__ is not None:
        raise TypeError("forward-v6 adapter requires a closure-free Python function")
    cloned_globals = dict(function.__globals__)
    cloned_globals.update(global_overrides)
    clone = FunctionType(
        function.__code__,
        cloned_globals,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=None,
    )
    clone.__kwdefaults__ = (
        dict(function.__kwdefaults__) if function.__kwdefaults__ is not None else None
    )
    clone.__annotations__ = dict(function.__annotations__)
    clone.__dict__.update(function.__dict__)
    clone.__doc__ = function.__doc__
    clone.__qualname__ = function.__qualname__
    return clone


def _strict_leaf_bitwise_equal(left: Any, right: Any, *, xp: Any) -> Any:
    left_array = xp.asarray(left)
    right_array = xp.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        raise ValueError("strict parity leaf shape/dtype differs")
    exact = xp.all(left_array == right_array)
    if np.issubdtype(np.dtype(left_array.dtype), np.inexact):
        # Finite equal IEEE values have identical bits except signed zero.
        # Comparing signbit closes that final gap without a dtype-changing view.
        exact = exact & xp.all(xp.signbit(left_array) == xp.signbit(right_array))
    return exact


def _strict_mapping_parity(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fields: Sequence[str],
    *,
    xp: Any,
) -> StrictParity:
    exact = xp.asarray(True)
    maximum = xp.zeros(())
    for field in fields:
        if field not in reference or field not in candidate:
            raise ValueError(f"strict parity field is missing: {field}")
        left = xp.asarray(reference[field])
        right = xp.asarray(candidate[field])
        exact = exact & _strict_leaf_bitwise_equal(left, right, xp=xp)
        if np.issubdtype(np.dtype(left.dtype), np.inexact):
            error = xp.max(
                xp.abs(left - right),
                initial=xp.asarray(0.0, dtype=left.dtype),
            )
            maximum = xp.maximum(maximum, error.astype(maximum.dtype))
    return StrictParity(exact, maximum, len(tuple(fields)))


def _strict_dynamic6_parity(
    reference_data: Any,
    candidate_data: Any,
    *,
    alignment: ModuleType,
    xp: Any,
) -> StrictParity:
    reference = alignment.save_v4_dynamic_state(reference_data)
    candidate = alignment.save_v4_dynamic_state(candidate_data)
    reference_mapping = {
        field: getattr(reference, field) for field in reference._fields
    }
    candidate_mapping = {
        field: getattr(candidate, field) for field in candidate._fields
    }
    return _strict_mapping_parity(
        reference_mapping,
        candidate_mapping,
        reference._fields,
        xp=xp,
    )


def _all_mapping_fields_finite(
    mapping: Mapping[str, Any], fields: Sequence[str], *, xp: Any
) -> Any:
    result = xp.asarray(True)
    for field in fields:
        array = xp.asarray(mapping[field])
        if np.issubdtype(np.dtype(array.dtype), np.inexact):
            result = result & xp.all(xp.isfinite(array))
    return result


class ForwardV6FactoryBinding:
    """Isolated exact-flag binding for one cloned evaluator call graph."""

    def __init__(self, factory: Callable[..., type]) -> None:
        if not callable(factory):
            raise TypeError("forward-v6 environment factory must be callable")
        self._factory = factory
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> type:
        forbidden = {
            "forward_v4_substep_contact",
            "forward_iteration_v6_contact_abort_island_only",
            "reverse_iteration_v6_absolute_full_leg_targets",
        }
        if forbidden & set(kwargs):
            raise ValueError("cloned evaluator attempted to supply iteration flags")
        self.call_count += 1
        environment_class = self._factory(
            **kwargs,
            forward_v4_substep_contact=True,
            forward_iteration_v6_contact_abort_island_only=True,
            reverse_iteration_v6_absolute_full_leg_targets=False,
        )
        checks = {
            "forward_v4": getattr(
                environment_class, "h4_forward_v4_substep_contact", None
            )
            is True,
            "forward_v6": getattr(
                environment_class,
                "h4_forward_iteration_v6_contact_abort_island_only",
                None,
            )
            is True,
            "reverse_v6": getattr(
                environment_class,
                "h4_reverse_iteration_v6_absolute_full_leg_targets",
                None,
            )
            is False,
            "v6_assertion": getattr(
                environment_class,
                "h4_forward_iteration_v6_compiled_assertion_bound",
                None,
            )
            is True,
            "off_gap_zero": getattr(
                environment_class,
                "h4_forward_iteration_v6_off_gap_reward_contribution",
                None,
            )
            == 0.0,
            "scale_minus_one": getattr(
                environment_class,
                "h4_forward_iteration_v6_contact_pulse_reward_scale",
                None,
            )
            == -1.0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"forward-v6 factory flag binding drifted: {checks}")
        return environment_class


def _validate_forward_v6_bundle(
    bundle: Any, authorization: Mapping[str, Any]
) -> None:
    config = bundle.config
    manifest = bundle.manifest
    expected_flags = {
        flag: flag == "forward_iteration_v6_contact_abort_island_only"
        for flag in ITERATION_MODE_FLAGS
    }
    authorization_record = config.get(
        "forward_iteration_v6_contact_abort_island_only_authorization"
    )
    checks = {
        "params_path": Path(bundle.params_path).resolve()
        == EXPECTED_PARAMS_PATH.resolve(),
        "manifest_path": Path(bundle.manifest_path).resolve()
        == EXPECTED_MANIFEST_PATH.resolve(),
        "run_name": bundle.run_name == EXPECTED_CANDIDATE_ROOT.name,
        "expert": bundle.expert == "forward",
        "status": bundle.status == "COMPLETED",
        "activity": bundle.activity == "PPO_PILOT_TRAINING",
        "config_contract": config.get("training_contract_id")
        == TRAINING_CONTRACT_ID,
        "manifest_contract": manifest.get("training_contract_id")
        == TRAINING_CONTRACT_ID,
        "authorized_contract": config.get(
            "authorized_iteration_v6_250k_contract_id"
        )
        == TRAINING_CONTRACT_ID,
        "config_flags": _type_exact_equal(
            {
                flag: config.get(flag, False) for flag in ITERATION_MODE_FLAGS
            },
            expected_flags,
        ),
        "manifest_flags": _type_exact_equal(
            {
                flag: manifest.get(flag, False) for flag in ITERATION_MODE_FLAGS
            },
            expected_flags,
        ),
        "auth_record": isinstance(authorization_record, Mapping),
        "auth_sha": isinstance(authorization_record, Mapping)
        and authorization_record.get("sha256")
        == PINNED_FORWARD_V6_AUTHORIZATION_SHA256,
        "reward_routing": _type_exact_equal(
            config.get("reward_routing_contract"),
            authorization.get("reward_routing_contract"),
        ),
        "source_scale": type(
            config.get("reward_scales", {}).get("h4_contact_pulse_40ms")
        )
        is float
        and config.get("reward_scales", {}).get("h4_contact_pulse_40ms")
        == -1.0,
    }
    if not all(checks.values()):
        raise ValueError(f"exact forward-v6 candidate bundle drifted: {checks}")


class ForwardV6BundleValidator:
    """Pre-pickle exact candidate wrapper around the frozen validator."""

    def __init__(
        self,
        validator: Callable[..., Any],
        authorization: Mapping[str, Any],
    ) -> None:
        self._validator = validator
        self._authorization = authorization

    def __call__(self, **kwargs: Any) -> Any:
        if kwargs.get("allow_wiring_diagnostic", False) is not False:
            raise ValueError("forward-v6 strict adapter rejects wiring candidates")
        if Path(kwargs.get("trusted_run_root", "")).resolve() != EXPECTED_RUN_ROOT.resolve():
            raise ValueError("forward-v6 trusted run root drifted")
        bundle = self._validator(**kwargs)
        _validate_forward_v6_bundle(bundle, self._authorization)
        return bundle


def _device_positive_zero(value: Any, *, xp: Any) -> Any:
    array = xp.asarray(value)
    result = array == 0
    if np.issubdtype(np.dtype(array.dtype), np.inexact):
        result = result & ~xp.signbit(array)
    return result


def _build_forward_v6_compiled_rollout(
    env: Any,
    policy: Any,
    stack: Mapping[str, Any],
    *,
    base: ModuleType,
    alignment: ModuleType,
) -> tuple[Any, Any]:
    """Build the exact 300x10 rollout without replacing outer mjx_env.step."""

    jax = stack["jax"]
    jp = stack["jp"]
    joystick = stack["joystick"]
    snapshot = base._snapshot_function(env, stack)
    joint_addresses = jp.asarray(env.get_actuator_joints_qpos_addr())
    adapter_assertion = alignment.make_v6_compiled_invariant_assertion(
        jax,
        jp,
        label=ADAPTER_CONTRACT_ID,
    )
    factory_checks = {
        "forward_v4": env.h4_forward_v4_substep_contact is True,
        "forward_v6": env.h4_forward_iteration_v6_contact_abort_island_only
        is True,
        "reverse_v6": env.h4_reverse_iteration_v6_absolute_full_leg_targets
        is False,
    }
    if not all(factory_checks.values()):
        raise RuntimeError(f"forward-v6 runtime factory flags drifted: {factory_checks}")

    def control_step(carry: tuple[Any, Any], _control_index: Any) -> tuple[Any, Any]:
        current_state, inference_key = carry
        control_entry_data = current_state.data
        actor_observation = current_state.obs["state"]
        previous_targets = current_state.data.ctrl
        guard_before = current_state.info["h4_guard_steps"]
        previous_contact = current_state.info["h4_previous_force_contact"]
        inference_key, action_key = jax.random.split(inference_key)
        raw_action, _extras = policy(current_state.obs, action_key)
        applied_action = raw_action.at[5:9].set(0.0)

        source_physics_step = joystick.mjx_env.step
        source_motor_speed_limits = joystick.USE_MOTOR_SPEED_LIMITS
        next_state = env.step(current_state, raw_action)
        if (
            joystick.mjx_env.step is not source_physics_step
            or joystick.USE_MOTOR_SPEED_LIMITS is not source_motor_speed_limits
        ):
            raise RuntimeError("forward-v6 environment failed to restore source globals")
        applied_targets = next_state.data.ctrl

        def authoritative_single_step(data: Any, target: Any) -> Any:
            return alignment.v4_authoritative_primitive_step(
                env.mjx_model,
                data,
                target,
                mjx_step=joystick.mjx_env.mjx.step,
            )

        witness_final, saved_dynamic = (
            alignment.scan_v4_instrumented_physics_trajectory(
                control_entry_data,
                applied_targets,
                single_physics_step=authoritative_single_step,
                n_substeps=PHYSICS_SUBSTEPS_PER_CONTROL,
                scan=jax.lax.scan,
                xp=jp,
            )
        )

        def replay_body(contact: Any, saved_state: Any) -> tuple[Any, Any]:
            replay_data = alignment.reconstruct_v4_dynamic_state(
                control_entry_data, saved_state
            )
            coherent_data = joystick.mjx_env.mjx.forward(
                env.mjx_model, replay_data
            )
            next_contact, row = snapshot(coherent_data, contact)
            return next_contact, {**row, "_contact_before": contact}

        _final_contact, replay_trace = jax.lax.scan(
            replay_body,
            previous_contact,
            saved_dynamic,
        )
        endpoint_previous_contact = replay_trace["_contact_before"][-1]
        witness_endpoint = {
            name: value[-1]
            for name, value in replay_trace.items()
            if name != "_contact_before"
        }
        actual_coherent = joystick.mjx_env.mjx.forward(
            env.mjx_model, next_state.data
        )
        _actual_contact, actual_endpoint = snapshot(
            actual_coherent, endpoint_previous_contact
        )
        physics_trace = {
            name: value
            for name, value in replay_trace.items()
            if name != "_contact_before"
        }

        dynamic6 = _strict_dynamic6_parity(
            next_state.data,
            witness_final,
            alignment=alignment,
            xp=jp,
        )
        gait_endpoint = _strict_mapping_parity(
            actual_endpoint,
            witness_endpoint,
            GAIT_ENDPOINT_FIELDS,
            xp=jp,
        )
        snapshot_endpoint = _strict_mapping_parity(
            actual_endpoint,
            witness_endpoint,
            SNAPSHOT_ENDPOINT_FIELDS,
            xp=jp,
        )
        endpoint_finite = _all_mapping_fields_finite(
            actual_endpoint, SNAPSHOT_ENDPOINT_FIELDS, xp=jp
        ) & _all_mapping_fields_finite(
            witness_endpoint, SNAPSHOT_ENDPOINT_FIELDS, xp=jp
        )
        saved_finite = alignment.v4_saved_dynamic_trajectory_all_finite(
            saved_dynamic, xp=jp
        )
        applied_target_exact = _strict_leaf_bitwise_equal(
            saved_dynamic.ctrl,
            jp.broadcast_to(applied_targets, saved_dynamic.ctrl.shape),
            xp=jp,
        )
        direct_count = jp.asarray(
            saved_dynamic.qpos.shape[0], dtype=jp.int32
        )

        info = next_state.info
        actual_v4_exact = (
            info["h4_v4_single_authority_dynamic6_exact"]
            & _device_positive_zero(
                info["h4_v4_single_authority_dynamic6_max_abs_error"], xp=jp
            )
            & (info["h4_v4_single_authority_dynamic6_field_count"] == 6)
            & info["h4_v4_single_authority_dynamic6_field_count_exact"]
            & (info["h4_v4_saved_dynamic6_substep_count"] == 10)
            & (info["h4_v4_saved_dynamic6_field_count"] == 6)
            & info["h4_v4_saved_dynamic6_field_count_exact"]
            & info["h4_v4_saved_dynamic6_all_finite"]
            & info["h4_v4_telemetry_force_shape_valid"]
            & info["h4_v4_telemetry_force_all_finite"]
            & ~info["h4_v4_single_authority_violation"]
            & (info["h4_v4_single_authority_assertion_token"] == 0)
        )
        actual_v6_exact = (
            info["h4_v6_forward_contact_abort_routing_exact"]
            & jp.isfinite(info["h4_v6_forward_contact_abort_island_loss"])
            & jp.isfinite(
                info["h4_v6_forward_contact_abort_off_gap_diagnostic_loss"]
            )
            & _device_positive_zero(
                info[
                    "h4_v6_forward_contact_abort_off_gap_reward_contribution"
                ],
                xp=jp,
            )
            & (info["h4_v6_forward_contact_abort_pulse_reward_scale"] == -1.0)
            & ~info["h4_v6_forward_contact_abort_routing_violation"]
            & (info["h4_v6_forward_contact_abort_routing_assertion_token"] == 0)
        )
        adapter_violation = (
            ~dynamic6.exact
            | ~_device_positive_zero(dynamic6.max_abs_error, xp=jp)
            | (dynamic6.field_count != DYNAMIC6_FIELD_COUNT)
            | (direct_count != PHYSICS_SUBSTEPS_PER_CONTROL)
            | ~saved_finite
            | ~applied_target_exact
            | ~gait_endpoint.exact
            | ~_device_positive_zero(gait_endpoint.max_abs_error, xp=jp)
            | (gait_endpoint.field_count != len(GAIT_ENDPOINT_FIELDS))
            | ~snapshot_endpoint.exact
            | ~_device_positive_zero(snapshot_endpoint.max_abs_error, xp=jp)
            | (snapshot_endpoint.field_count != len(SNAPSHOT_ENDPOINT_FIELDS))
            | ~endpoint_finite
            | ~actual_v4_exact
            | ~actual_v6_exact
        )
        adapter_assertion_token = adapter_assertion(adapter_violation)

        control_trace = {
            "actor_observation": actor_observation,
            "raw_action": raw_action,
            "applied_action": applied_action,
            "preclip_targets": next_state.info["h4_pre_guard_raw_targets"],
            "margin_clipped_targets": next_state.info["h4_guard_desired_targets"],
            "applied_targets": next_state.data.ctrl,
            "previous_targets": previous_targets,
            "joint_qpos": next_state.data.qpos[joint_addresses],
            "guard_calls": next_state.info["h4_guard_steps"] - guard_before,
            "done": next_state.done,
            "h4_forward_v6_adapter_forward_v4_flag": jp.asarray(True),
            "h4_forward_v6_adapter_forward_v6_flag": jp.asarray(True),
            "h4_forward_v6_adapter_reverse_v6_flag": jp.asarray(False),
            "h4_forward_v6_adapter_direct_primitive_substep_count": direct_count,
            "h4_forward_v6_adapter_dynamic6_endpoint_bitwise_exact": dynamic6.exact,
            "h4_forward_v6_adapter_dynamic6_endpoint_max_abs_error": dynamic6.max_abs_error,
            "h4_forward_v6_adapter_dynamic6_field_count": jp.asarray(
                dynamic6.field_count, dtype=jp.int32
            ),
            "h4_forward_v6_adapter_saved_dynamic6_all_finite": saved_finite,
            "h4_forward_v6_adapter_applied_target_bitwise_exact": applied_target_exact,
            "h4_forward_v6_adapter_gait_endpoint_bitwise_exact": gait_endpoint.exact,
            "h4_forward_v6_adapter_gait_endpoint_max_abs_error": gait_endpoint.max_abs_error,
            "h4_forward_v6_adapter_gait_endpoint_field_count": jp.asarray(
                gait_endpoint.field_count, dtype=jp.int32
            ),
            "h4_forward_v6_adapter_snapshot_endpoint_bitwise_exact": snapshot_endpoint.exact,
            "h4_forward_v6_adapter_snapshot_endpoint_max_abs_error": snapshot_endpoint.max_abs_error,
            "h4_forward_v6_adapter_snapshot_endpoint_field_count": jp.asarray(
                snapshot_endpoint.field_count, dtype=jp.int32
            ),
            "h4_forward_v6_adapter_endpoint_fields_all_finite": endpoint_finite,
            "h4_forward_v6_adapter_violation": adapter_violation,
            "h4_forward_v6_adapter_assertion_token": adapter_assertion_token,
        }
        for name in (*V4_AUTHORITY_TRACE_FIELDS, *V6_ROUTING_TRACE_FIELDS):
            control_trace[name] = next_state.info[name]
        return (next_state, inference_key), {
            "physics": physics_trace,
            "control": control_trace,
        }

    def complete_rollout(initial_state: Any, inference_key: Any) -> tuple[Any, Any]:
        return jax.lax.scan(
            control_step,
            (initial_state, inference_key),
            stack["jp"].arange(CONTROL_TICK_COUNT),
        )

    return snapshot, jax.jit(complete_rollout)


class ForwardV6RolloutCompiler:
    """Per-call-graph cache and trace handoff; no module global is mutated."""

    def __init__(self, base: ModuleType, alignment: ModuleType) -> None:
        self._base = base
        self._alignment = alignment
        self._cache: dict[tuple[int, int], tuple[Any, Any]] = {}
        self._pending: dict[tuple[int, int], Mapping[str, Any]] = {}

    @staticmethod
    def _key(env: Any, policy: Any) -> tuple[int, int]:
        return (id(env), id(policy))

    def __call__(
        self, env: Any, policy: Any, stack: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        key = self._key(env, policy)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        snapshot, compiled = _build_forward_v6_compiled_rollout(
            env,
            policy,
            stack,
            base=self._base,
            alignment=self._alignment,
        )

        def execute(initial_state: Any, inference_key: Any) -> tuple[Any, Any]:
            if key in self._pending:
                raise RuntimeError("unconsumed forward-v6 runtime trace collision")
            result = compiled(initial_state, inference_key)
            device_trace = result[1]
            self._pending[key] = {
                name: device_trace["control"][name]
                for name in QUALIFYING_TRACE_FIELDS
            }
            return result

        result = (snapshot, execute)
        self._cache[key] = result
        return result

    def consume(self, env: Any, policy: Any) -> Mapping[str, Any]:
        key = self._key(env, policy)
        try:
            return self._pending.pop(key)
        except KeyError as exc:
            raise RuntimeError("forward-v6 runtime trace was not captured") from exc

    def discard(self, env: Any, policy: Any) -> None:
        self._pending.pop(self._key(env, policy), None)


class SameValueStepWriteShield:
    """Read-through MJX namespace that suppresses only same-object restores."""

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "suppressed_same_value_step_writes", 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "step" and value is getattr(self._target, "step"):
            object.__setattr__(
                self,
                "suppressed_same_value_step_writes",
                self.suppressed_same_value_step_writes + 1,
            )
            return
        raise RuntimeError(
            f"forward-v6 evaluator attempted a mutable mjx_env write: {name}"
        )


class ReadOnlyJoystickView:
    """Read-through joystick view whose mjx_env cannot mutate the module."""

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(
            self, "mjx_env", SameValueStepWriteShield(target.mjx_env)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __setattr__(self, name: str, _value: Any) -> None:
        raise RuntimeError(
            f"forward-v6 evaluator attempted a mutable joystick write: {name}"
        )


def _trace_sha256(trace: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in QUALIFYING_TRACE_FIELDS:
        array = np.ascontiguousarray(trace[name])
        encoded_name = name.encode("utf-8")
        encoded_dtype = array.dtype.str.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(4, "big"))
        digest.update(encoded_dtype)
        digest.update(len(array.shape).to_bytes(4, "big"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=False))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _all_bitwise_constant(array: np.ndarray, value: Any) -> bool:
    source = np.ascontiguousarray(array)
    expected = np.full(source.shape, value, dtype=source.dtype)
    return source.tobytes(order="C") == expected.tobytes(order="C")


def _summarize_forward_v6_trace(
    device_trace: Mapping[str, Any], *, jax: Any
) -> dict[str, Any]:
    if set(device_trace) != set(QUALIFYING_TRACE_FIELDS):
        raise ValueError("forward-v6 qualifying runtime trace field set drifted")
    host = {
        name: np.asarray(value)
        for name, value in jax.device_get(device_trace).items()
    }
    shapes_exact = all(
        value.shape == (CONTROL_TICK_COUNT,) for value in host.values()
    )
    finite_fields = (
        "h4_forward_v6_adapter_dynamic6_endpoint_max_abs_error",
        "h4_forward_v6_adapter_gait_endpoint_max_abs_error",
        "h4_forward_v6_adapter_snapshot_endpoint_max_abs_error",
        "h4_v4_single_authority_dynamic6_max_abs_error",
        "h4_v6_forward_contact_abort_island_loss",
        "h4_v6_forward_contact_abort_off_gap_diagnostic_loss",
        "h4_v6_forward_contact_abort_off_gap_reward_contribution",
        "h4_v6_forward_contact_abort_pulse_reward_scale",
    )
    all_finite = all(np.all(np.isfinite(host[name])) for name in finite_fields)
    all_true_fields = (
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
    )
    all_false_fields = (
        "h4_forward_v6_adapter_reverse_v6_flag",
        "h4_forward_v6_adapter_violation",
        "h4_v4_single_authority_violation",
        "h4_v6_forward_contact_abort_routing_violation",
    )
    integer_fields = (
        "h4_forward_v6_adapter_direct_primitive_substep_count",
        "h4_forward_v6_adapter_dynamic6_field_count",
        "h4_forward_v6_adapter_gait_endpoint_field_count",
        "h4_forward_v6_adapter_snapshot_endpoint_field_count",
        "h4_forward_v6_adapter_assertion_token",
        "h4_v4_single_authority_dynamic6_field_count",
        "h4_v4_saved_dynamic6_substep_count",
        "h4_v4_saved_dynamic6_field_count",
        "h4_v4_single_authority_assertion_token",
        "h4_v6_forward_contact_abort_routing_assertion_token",
    )
    checks = {
        "field_set_exact": set(host) == set(QUALIFYING_TRACE_FIELDS),
        "control_trace_shape_exact": shapes_exact,
        "finite_fields_exact": all_finite,
        "boolean_field_dtypes_exact": all(
            host[name].dtype.kind == "b"
            for name in (*all_true_fields, *all_false_fields)
        ),
        "integer_field_dtypes_exact": all(
            host[name].dtype.kind in {"i", "u"} for name in integer_fields
        ),
        "float_field_dtypes_exact": all(
            host[name].dtype.kind == "f" for name in finite_fields
        ),
        "required_true_every_tick": all(
            np.asarray(host[name]).dtype.kind == "b" and np.all(host[name])
            for name in all_true_fields
        ),
        "required_false_every_tick": all(
            np.asarray(host[name]).dtype.kind == "b" and not np.any(host[name])
            for name in all_false_fields
        ),
        "direct_primitive_trace_count_exact": _all_bitwise_constant(
            host["h4_forward_v6_adapter_direct_primitive_substep_count"], 10
        ),
        "dynamic6_field_count_exact": _all_bitwise_constant(
            host["h4_forward_v6_adapter_dynamic6_field_count"], 6
        ),
        "gait_endpoint_field_count_exact": _all_bitwise_constant(
            host["h4_forward_v6_adapter_gait_endpoint_field_count"],
            len(GAIT_ENDPOINT_FIELDS),
        ),
        "snapshot_endpoint_field_count_exact": _all_bitwise_constant(
            host["h4_forward_v6_adapter_snapshot_endpoint_field_count"],
            len(SNAPSHOT_ENDPOINT_FIELDS),
        ),
        "adapter_dynamic6_zero_error_exact": _all_bitwise_constant(
            host["h4_forward_v6_adapter_dynamic6_endpoint_max_abs_error"], 0.0
        ),
        "adapter_gait_endpoint_zero_error_exact": _all_bitwise_constant(
            host["h4_forward_v6_adapter_gait_endpoint_max_abs_error"], 0.0
        ),
        "adapter_snapshot_endpoint_zero_error_exact": _all_bitwise_constant(
            host["h4_forward_v6_adapter_snapshot_endpoint_max_abs_error"], 0.0
        ),
        "v4_dynamic6_zero_error_exact": _all_bitwise_constant(
            host["h4_v4_single_authority_dynamic6_max_abs_error"], 0.0
        ),
        "v4_dynamic6_field_count_exact": _all_bitwise_constant(
            host["h4_v4_single_authority_dynamic6_field_count"], 6
        ),
        "v4_saved_substep_count_exact": _all_bitwise_constant(
            host["h4_v4_saved_dynamic6_substep_count"], 10
        ),
        "v4_saved_field_count_exact": _all_bitwise_constant(
            host["h4_v4_saved_dynamic6_field_count"], 6
        ),
        "v4_assertion_token_zero_exact": _all_bitwise_constant(
            host["h4_v4_single_authority_assertion_token"], 0
        ),
        "v6_off_gap_reward_contribution_zero_exact": _all_bitwise_constant(
            host[
                "h4_v6_forward_contact_abort_off_gap_reward_contribution"
            ],
            0.0,
        ),
        "v6_contact_pulse_scale_minus_one_exact": _all_bitwise_constant(
            host["h4_v6_forward_contact_abort_pulse_reward_scale"], -1.0
        ),
        "v6_assertion_token_zero_exact": _all_bitwise_constant(
            host["h4_v6_forward_contact_abort_routing_assertion_token"], 0
        ),
        "adapter_assertion_token_zero_exact": _all_bitwise_constant(
            host["h4_forward_v6_adapter_assertion_token"], 0
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"forward-v6 runtime witness failed: {checks}")
    summary = {
        "schema_version": 1,
        "contract_id": ADAPTER_CONTRACT_ID,
        "source_dtype_contract": "PRESERVED_PER_FIELD",
        "control_tick_count": CONTROL_TICK_COUNT,
        "direct_primitive_substeps_per_control": PHYSICS_SUBSTEPS_PER_CONTROL,
        "physics_trace_row_count": PHYSICS_TRACE_ROW_COUNT,
        "dynamic6_field_count": DYNAMIC6_FIELD_COUNT,
        "gait_endpoint_field_count": len(GAIT_ENDPOINT_FIELDS),
        "snapshot_endpoint_field_count": len(SNAPSHOT_ENDPOINT_FIELDS),
        "qualifying_trace_sha256": _trace_sha256(host),
        "factory_flags": {
            "forward_v4_substep_contact": True,
            "forward_iteration_v6_contact_abort_island_only": True,
            "reverse_iteration_v6_absolute_full_leg_targets": False,
        },
        "direct_primitive_witness": {
            "dynamic6_bitwise_exact_control_count": int(
                np.count_nonzero(
                    host[
                        "h4_forward_v6_adapter_dynamic6_endpoint_bitwise_exact"
                    ]
                )
            ),
            "dynamic6_max_abs_error": float(
                np.max(
                    host[
                        "h4_forward_v6_adapter_dynamic6_endpoint_max_abs_error"
                    ]
                )
            ),
            "applied_target_bitwise_exact_control_count": int(
                np.count_nonzero(
                    host[
                        "h4_forward_v6_adapter_applied_target_bitwise_exact"
                    ]
                )
            ),
            "gait_endpoint_bitwise_exact_control_count": int(
                np.count_nonzero(
                    host[
                        "h4_forward_v6_adapter_gait_endpoint_bitwise_exact"
                    ]
                )
            ),
            "gait_endpoint_max_abs_error": float(
                np.max(
                    host[
                        "h4_forward_v6_adapter_gait_endpoint_max_abs_error"
                    ]
                )
            ),
            "snapshot_endpoint_bitwise_exact_control_count": int(
                np.count_nonzero(
                    host[
                        "h4_forward_v6_adapter_snapshot_endpoint_bitwise_exact"
                    ]
                )
            ),
            "snapshot_endpoint_max_abs_error": float(
                np.max(
                    host[
                        "h4_forward_v6_adapter_snapshot_endpoint_max_abs_error"
                    ]
                )
            ),
            "violation_count": int(
                np.count_nonzero(host["h4_forward_v6_adapter_violation"])
            ),
            "assertion_token_sum": int(
                np.sum(host["h4_forward_v6_adapter_assertion_token"])
            ),
        },
        "actual_forward_v4_authority": {
            "dynamic6_exact_control_count": int(
                np.count_nonzero(
                    host["h4_v4_single_authority_dynamic6_exact"]
                )
            ),
            "dynamic6_max_abs_error": float(
                np.max(host["h4_v4_single_authority_dynamic6_max_abs_error"])
            ),
            "saved_dynamic6_all_finite_control_count": int(
                np.count_nonzero(host["h4_v4_saved_dynamic6_all_finite"])
            ),
            "violation_count": int(
                np.count_nonzero(host["h4_v4_single_authority_violation"])
            ),
            "assertion_token_sum": int(
                np.sum(host["h4_v4_single_authority_assertion_token"])
            ),
        },
        "forward_v6_reward_routing": {
            "routing_exact_control_count": int(
                np.count_nonzero(
                    host["h4_v6_forward_contact_abort_routing_exact"]
                )
            ),
            "island_loss_sum": float(
                np.sum(host["h4_v6_forward_contact_abort_island_loss"])
            ),
            "off_gap_diagnostic_loss_sum": float(
                np.sum(
                    host[
                        "h4_v6_forward_contact_abort_off_gap_diagnostic_loss"
                    ]
                )
            ),
            "off_gap_reward_contribution_sum": float(
                np.sum(
                    host[
                        "h4_v6_forward_contact_abort_off_gap_reward_contribution"
                    ]
                )
            ),
            "contact_pulse_reward_scale_min": float(
                np.min(
                    host["h4_v6_forward_contact_abort_pulse_reward_scale"]
                )
            ),
            "contact_pulse_reward_scale_max": float(
                np.max(
                    host["h4_v6_forward_contact_abort_pulse_reward_scale"]
                )
            ),
            "routing_violation_count": int(
                np.count_nonzero(
                    host["h4_v6_forward_contact_abort_routing_violation"]
                )
            ),
            "assertion_token_sum": int(
                np.sum(
                    host["h4_v6_forward_contact_abort_routing_assertion_token"]
                )
            ),
        },
        "checks": checks,
    }
    validate_forward_v6_runtime_witness(summary)
    return summary


def _exact_positive_zero(value: Any) -> bool:
    return (
        type(value) is float
        and value == 0.0
        and not bool(np.signbit(value))
    )


def _exact_int_value(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _exact_float_value(value: Any, expected: float) -> bool:
    if type(value) is not float or not np.isfinite(value):
        return False
    left = np.asarray(value, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    return left.tobytes() == right.tobytes()


def validate_forward_v6_runtime_witness(witness: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "contract_id",
        "source_dtype_contract",
        "control_tick_count",
        "direct_primitive_substeps_per_control",
        "physics_trace_row_count",
        "dynamic6_field_count",
        "gait_endpoint_field_count",
        "snapshot_endpoint_field_count",
        "qualifying_trace_sha256",
        "factory_flags",
        "direct_primitive_witness",
        "actual_forward_v4_authority",
        "forward_v6_reward_routing",
        "checks",
    }
    if set(witness) != expected_top:
        raise ValueError("forward-v6 runtime witness top-level closure drifted")
    direct = witness.get("direct_primitive_witness")
    v4 = witness.get("actual_forward_v4_authority")
    routing = witness.get("forward_v6_reward_routing")
    checks = witness.get("checks")
    if not all(isinstance(value, Mapping) for value in (direct, v4, routing, checks)):
        raise ValueError("forward-v6 runtime witness sections are incomplete")
    expected_direct_keys = {
        "dynamic6_bitwise_exact_control_count",
        "dynamic6_max_abs_error",
        "applied_target_bitwise_exact_control_count",
        "gait_endpoint_bitwise_exact_control_count",
        "gait_endpoint_max_abs_error",
        "snapshot_endpoint_bitwise_exact_control_count",
        "snapshot_endpoint_max_abs_error",
        "violation_count",
        "assertion_token_sum",
    }
    expected_v4_keys = {
        "dynamic6_exact_control_count",
        "dynamic6_max_abs_error",
        "saved_dynamic6_all_finite_control_count",
        "violation_count",
        "assertion_token_sum",
    }
    expected_routing_keys = {
        "routing_exact_control_count",
        "island_loss_sum",
        "off_gap_diagnostic_loss_sum",
        "off_gap_reward_contribution_sum",
        "contact_pulse_reward_scale_min",
        "contact_pulse_reward_scale_max",
        "routing_violation_count",
        "assertion_token_sum",
    }
    semantic = {
        "direct_key_closure": set(direct) == expected_direct_keys,
        "v4_key_closure": set(v4) == expected_v4_keys,
        "routing_key_closure": set(routing) == expected_routing_keys,
        "check_key_closure": set(checks) == set(RUNTIME_WITNESS_CHECK_KEYS),
        "schema": _exact_int_value(witness.get("schema_version"), 1),
        "contract": witness.get("contract_id") == ADAPTER_CONTRACT_ID,
        "dtype": witness.get("source_dtype_contract") == "PRESERVED_PER_FIELD",
        "controls": _exact_int_value(
            witness.get("control_tick_count"), CONTROL_TICK_COUNT
        ),
        "substeps": _exact_int_value(
            witness.get("direct_primitive_substeps_per_control"), 10
        ),
        "rows": _exact_int_value(
            witness.get("physics_trace_row_count"), PHYSICS_TRACE_ROW_COUNT
        ),
        "dynamic6": _exact_int_value(witness.get("dynamic6_field_count"), 6),
        "gait_fields": _exact_int_value(
            witness.get("gait_endpoint_field_count"), len(GAIT_ENDPOINT_FIELDS)
        ),
        "snapshot_fields": _exact_int_value(
            witness.get("snapshot_endpoint_field_count"),
            len(SNAPSHOT_ENDPOINT_FIELDS),
        ),
        "trace_sha": isinstance(witness.get("qualifying_trace_sha256"), str),
        "flags": _type_exact_equal(
            witness.get("factory_flags"),
            {
                "forward_v4_substep_contact": True,
                "forward_iteration_v6_contact_abort_island_only": True,
                "reverse_iteration_v6_absolute_full_leg_targets": False,
            },
        ),
        "direct_count": _exact_int_value(
            direct.get("dynamic6_bitwise_exact_control_count"), CONTROL_TICK_COUNT
        ),
        "direct_error": _exact_positive_zero(
            direct.get("dynamic6_max_abs_error")
        ),
        "target_count": _exact_int_value(
            direct.get("applied_target_bitwise_exact_control_count"),
            CONTROL_TICK_COUNT,
        ),
        "gait_count": _exact_int_value(
            direct.get("gait_endpoint_bitwise_exact_control_count"),
            CONTROL_TICK_COUNT,
        ),
        "gait_error": _exact_positive_zero(
            direct.get("gait_endpoint_max_abs_error")
        ),
        "snapshot_count": _exact_int_value(
            direct.get("snapshot_endpoint_bitwise_exact_control_count"),
            CONTROL_TICK_COUNT,
        ),
        "snapshot_error": _exact_positive_zero(
            direct.get("snapshot_endpoint_max_abs_error")
        ),
        "direct_violation": _exact_int_value(direct.get("violation_count"), 0),
        "direct_token": _exact_int_value(direct.get("assertion_token_sum"), 0),
        "v4_count": _exact_int_value(
            v4.get("dynamic6_exact_control_count"), CONTROL_TICK_COUNT
        ),
        "v4_error": _exact_positive_zero(v4.get("dynamic6_max_abs_error")),
        "v4_finite": _exact_int_value(
            v4.get("saved_dynamic6_all_finite_control_count"), CONTROL_TICK_COUNT
        ),
        "v4_violation": _exact_int_value(v4.get("violation_count"), 0),
        "v4_token": _exact_int_value(v4.get("assertion_token_sum"), 0),
        "routing_count": _exact_int_value(
            routing.get("routing_exact_control_count"), CONTROL_TICK_COUNT
        ),
        "routing_island_finite": type(routing.get("island_loss_sum")) is float
        and np.isfinite(routing.get("island_loss_sum")),
        "routing_offgap_finite": type(
            routing.get("off_gap_diagnostic_loss_sum")
        )
        is float
        and np.isfinite(routing.get("off_gap_diagnostic_loss_sum")),
        "routing_zero": _exact_positive_zero(
            routing.get("off_gap_reward_contribution_sum")
        ),
        "routing_scale_min": _exact_float_value(
            routing.get("contact_pulse_reward_scale_min"), -1.0
        ),
        "routing_scale_max": _exact_float_value(
            routing.get("contact_pulse_reward_scale_max"), -1.0
        ),
        "routing_violation": _exact_int_value(
            routing.get("routing_violation_count"), 0
        ),
        "routing_token": _exact_int_value(routing.get("assertion_token_sum"), 0),
        "checks": bool(checks)
        and all(type(value) is bool and value is True for value in checks.values()),
    }
    if semantic["trace_sha"]:
        try:
            _require_sha256(witness["qualifying_trace_sha256"], "trace SHA256")
        except ValueError:
            semantic["trace_sha"] = False
    if not all(semantic.values()):
        raise ValueError(f"forward-v6 runtime witness drifted: {semantic}")


class ForwardV6EpisodeRunner:
    """Append a host-validated witness to the frozen episode result."""

    def __init__(
        self,
        base_episode: Callable[..., dict[str, Any]],
        compiler: ForwardV6RolloutCompiler,
    ) -> None:
        self._base_episode = base_episode
        self._compiler = compiler

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        env = kwargs["env"]
        policy = kwargs["policy"]
        stack = kwargs["stack"]
        source_joystick = stack["joystick"]
        source_step = source_joystick.mjx_env.step
        source_motor_speed_limits = source_joystick.USE_MOTOR_SPEED_LIMITS
        joystick_view = ReadOnlyJoystickView(source_joystick)
        isolated_stack = dict(stack)
        isolated_stack["joystick"] = joystick_view
        isolated_kwargs = dict(kwargs)
        isolated_kwargs["stack"] = isolated_stack
        try:
            episode = self._base_episode(**isolated_kwargs)
            trace = self._compiler.consume(env, policy)
        except BaseException:
            self._compiler.discard(env, policy)
            raise
        finally:
            if (
                source_joystick.mjx_env.step is not source_step
                or source_joystick.USE_MOTOR_SPEED_LIMITS is not source_motor_speed_limits
            ):
                raise RuntimeError(
                    "forward-v6 evaluation changed frozen joystick module state"
                )
        if joystick_view.mjx_env.suppressed_same_value_step_writes != 1:
            raise RuntimeError(
                "frozen episode wrapper restore topology drifted from one same-value write"
            )
        episode["forward_v6_runtime_witness"] = _summarize_forward_v6_trace(
            trace, jax=isolated_stack["jax"]
        )
        return episode


def _module_contract_snapshot(
    base: ModuleType, post: ModuleType, alignment: ModuleType
) -> dict[str, Any]:
    return {
        "base_make": base._make_environment_and_policy,
        "base_compiled": base._compiled_rollout_for,
        "base_episode": base._run_episode,
        "base_run": base.run_evaluation,
        "base_bundle_validator": base.validate_trusted_h4_bundle,
        "base_artifact_validator": base.validate_h4_strict_artifact,
        "post_bundle_validator": post.validate_trusted_h4_bundle,
        "post_episode_validator": post.validate_h4_strict_episode,
        "post_artifact_validator": post.validate_h4_strict_artifact,
        "alignment_factory": alignment.make_h4_aligned_environment_class,
        "alignment_primitive": alignment.v4_authoritative_primitive_step,
        "alignment_scan": alignment.scan_v4_instrumented_physics_trajectory,
        "alignment_reconstruct": alignment.reconstruct_v4_dynamic_state,
    }


def _assert_module_contract_unchanged(
    base: ModuleType,
    post: ModuleType,
    alignment: ModuleType,
    expected: Mapping[str, Any],
) -> None:
    if _module_contract_snapshot(base, post, alignment) != dict(expected):
        raise RuntimeError("frozen evaluator/post/alignment module contract changed")


def _private_initial_artifact_validation_view(
    payload: Mapping[str, Any], *, initial_pending: bool
) -> Mapping[str, Any]:
    """Add the base-only current map on a private copy, never on real payload."""

    provenance = payload.get("runtime_provenance")
    adapter_keys = {ADAPTER_SOURCE_KEY, ADAPTER_AUTHORIZATION_SOURCE_KEY}
    source_pre = (
        provenance.get("evaluation_source_hashes_pre")
        if isinstance(provenance, Mapping)
        else None
    )
    source_post = (
        provenance.get("evaluation_source_hashes_post")
        if isinstance(provenance, Mapping)
        else None
    )
    private_initial_shape = bool(
        initial_pending is True
        and isinstance(source_pre, Mapping)
        and isinstance(source_post, Mapping)
        and not (adapter_keys & set(source_pre))
        and not (adapter_keys & set(source_post))
        and "evaluation_source_hashes_current" not in provenance
        and "forward_v6_strict_evaluator_adapter" not in provenance
    )
    if not private_initial_shape:
        return payload
    view = copy.deepcopy(dict(payload))
    view["runtime_provenance"]["evaluation_source_hashes_current"] = dict(
        source_post
    )
    return view


def build_forward_v6_call_graph(
    base: ModuleType,
    post: ModuleType,
    alignment: ModuleType,
    *,
    forward_v6_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    original = _module_contract_snapshot(base, post, alignment)
    identity_checks = {
        "base_bundle_is_post": base.validate_trusted_h4_bundle
        is post.validate_trusted_h4_bundle,
        "base_artifact_is_post": base.validate_h4_strict_artifact
        is post.validate_h4_strict_artifact,
        "base_factory_is_alignment": base.make_h4_aligned_environment_class
        is alignment.make_h4_aligned_environment_class,
    }
    if not all(identity_checks.values()):
        raise RuntimeError(
            f"frozen evaluator import identity drifted: {identity_checks}"
        )
    factory_binding = ForwardV6FactoryBinding(
        alignment.make_h4_aligned_environment_class
    )
    make_environment = _clone_function(
        base._make_environment_and_policy,
        global_overrides={"make_h4_aligned_environment_class": factory_binding},
    )
    compiler = ForwardV6RolloutCompiler(base, alignment)
    base_episode = _clone_function(
        base._run_episode,
        global_overrides={"_compiled_rollout_for": compiler},
    )
    episode_runner = ForwardV6EpisodeRunner(base_episode, compiler)
    compatibility_artifact_validator = _clone_function(
        post.validate_h4_strict_artifact,
        global_overrides={"STRICT_ARTIFACT_KIND": DIAGNOSTIC_ARTIFACT_KIND},
    )
    def artifact_validator(
        payload: Mapping[str, Any],
        *,
        bundle: Any = None,
        current_central_hashes: Mapping[str, str] | None = None,
        current_evaluation_hashes: Mapping[str, str] | None = None,
        require_all_three_pass: bool = False,
        gait_quality_rederive: Any = None,
    ) -> dict[str, Any]:
        audit = compatibility_artifact_validator(
            payload,
            bundle=bundle,
            current_central_hashes=current_central_hashes,
            current_evaluation_hashes=current_evaluation_hashes,
            require_all_three_pass=require_all_three_pass,
            gait_quality_rederive=gait_quality_rederive,
        )
        _validate_adapter_provenance_stage(
            payload,
            bundle=bundle,
            current_evaluation_hashes=current_evaluation_hashes,
            allow_initial_record_absent=False,
        )
        return audit

    initial_adapter_validation_pending = True

    def initial_artifact_validator(
        payload: Mapping[str, Any],
        *,
        bundle: Any = None,
        current_central_hashes: Mapping[str, str] | None = None,
        current_evaluation_hashes: Mapping[str, str] | None = None,
        require_all_three_pass: bool = False,
        gait_quality_rederive: Any = None,
    ) -> dict[str, Any]:
        nonlocal initial_adapter_validation_pending
        if not initial_adapter_validation_pending:
            raise RuntimeError("forward-v6 initial artifact validation is one-shot")
        validation_payload = _private_initial_artifact_validation_view(
            payload,
            initial_pending=True,
        )
        audit = compatibility_artifact_validator(
            validation_payload,
            bundle=bundle,
            current_central_hashes=current_central_hashes,
            current_evaluation_hashes=current_evaluation_hashes,
            require_all_three_pass=require_all_three_pass,
            gait_quality_rederive=gait_quality_rederive,
        )
        _validate_adapter_provenance_stage(
            validation_payload,
            bundle=bundle,
            current_evaluation_hashes=current_evaluation_hashes,
            allow_initial_record_absent=True,
        )
        initial_adapter_validation_pending = False
        return audit

    bundle_validator = ForwardV6BundleValidator(
        post.validate_trusted_h4_bundle,
        forward_v6_authorization,
    )
    run_evaluation = _clone_function(
        base.run_evaluation,
        global_overrides={
            "STRICT_ARTIFACT_KIND": DIAGNOSTIC_ARTIFACT_KIND,
            "validate_trusted_h4_bundle": bundle_validator,
            "_make_environment_and_policy": make_environment,
            "_run_episode": episode_runner,
            "validate_h4_strict_artifact": initial_artifact_validator,
        },
    )
    _assert_module_contract_unchanged(base, post, alignment, original)
    return {
        "factory_binding": factory_binding,
        "compiler": compiler,
        "make_environment": make_environment,
        "base_episode": base_episode,
        "episode_runner": episode_runner,
        "compatibility_artifact_validator": compatibility_artifact_validator,
        "artifact_validator": artifact_validator,
        "bundle_validator": bundle_validator,
        "run_evaluation": run_evaluation,
    }


def _load_frozen_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    if str(EXP_ROOT) not in sys.path:
        sys.path.insert(0, str(EXP_ROOT))
    import safe_gait_experts.h4_post_training as post
    import safe_gait_experts.h4_training_alignment as alignment

    spec = importlib.util.spec_from_file_location(
        "exp004_h4_forward_v6_strict_adapter_base_evaluator_v1",
        BASE_EVALUATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen evaluator: {BASE_EVALUATOR_PATH}")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base, post, alignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only diagnostic strict evaluation of the exact completed "
            "forward-v6 contact-abort island-only candidate."
        )
    )
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--params-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=EXPECTED_OUTPUT_PATH)
    parser.add_argument("--trusted-run-root", type=Path, default=EXPECTED_RUN_ROOT)
    parser.add_argument(
        "--adapter-authorization",
        type=Path,
        default=ADAPTER_AUTHORIZATION_PATH,
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument(
        "--v22-parent-checkpoint",
        type=Path,
        default=DEFAULT_V22_PARENT_CHECKPOINT,
    )
    parser.add_argument("--platform", choices=("cpu",), default="cpu")
    return parser


def _resolve_process_start_paths(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "params",
        "manifest",
        "output",
        "trusted_run_root",
        "adapter_authorization",
        "source_root",
        "generated_root",
        "v22_parent_checkpoint",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).resolve())
    args.allow_wiring_diagnostic = False
    args.promotion_evidence_output = None
    return args


def _validate_exact_cli(
    args: argparse.Namespace, *, require_output_absent: bool = True
) -> None:
    checks = {
        "params_path": args.params == EXPECTED_PARAMS_PATH.resolve(),
        "manifest_path": args.manifest == EXPECTED_MANIFEST_PATH.resolve(),
        "output_path": args.output == EXPECTED_OUTPUT_PATH.resolve(),
        "trusted_root": args.trusted_run_root == EXPECTED_RUN_ROOT.resolve(),
        "adapter_authorization": args.adapter_authorization
        == ADAPTER_AUTHORIZATION_PATH.resolve(),
        "params_sha": isinstance(args.params_sha256, str),
        "manifest_sha": isinstance(args.manifest_sha256, str),
        "platform": args.platform == "cpu",
        "wiring_forbidden": args.allow_wiring_diagnostic is False,
        "promotion_forbidden": args.promotion_evidence_output is None,
    }
    try:
        _require_sha256(args.params_sha256, "params SHA256")
        _require_sha256(args.manifest_sha256, "manifest SHA256")
    except ValueError:
        checks["hash_syntax"] = False
    else:
        checks["hash_syntax"] = True
    if not all(checks.values()):
        raise ValueError(f"exact forward-v6 adapter CLI drifted: {checks}")
    if require_output_absent and args.output.exists():
        raise FileExistsError(f"refusing to overwrite strict diagnostic: {args.output}")


def _evaluation_path_key(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(EXP_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _candidate_bundle_snapshot(bundle: Any) -> dict[str, dict[str, str]]:
    """Derive the exact five-file closure from one validated bundle."""

    manifest = bundle.manifest
    outputs = manifest.get("outputs") if isinstance(manifest, Mapping) else None
    resolved_config = (
        manifest.get("resolved_config") if isinstance(manifest, Mapping) else None
    )
    if (
        not isinstance(outputs, Mapping)
        or set(outputs) != {"final_params", "result", "training_curve"}
        or not isinstance(resolved_config, Mapping)
    ):
        raise ValueError("forward-v6 candidate five-file manifest closure drifted")

    def bound_record(raw: Any, *, label: str) -> tuple[Path, str]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"forward-v6 candidate {label} record is missing")
        raw_path = raw.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"forward-v6 candidate {label} path is invalid")
        return Path(raw_path).resolve(), _require_sha256(
            raw.get("sha256"), f"candidate {label} SHA256"
        )

    output_params_path, output_params_sha = bound_record(
        outputs.get("final_params"), label="params"
    )
    result_path, result_sha = bound_record(outputs.get("result"), label="result")
    curve_path, curve_sha = bound_record(
        outputs.get("training_curve"), label="training curve"
    )
    config_record_path, config_record_sha = bound_record(
        resolved_config, label="config"
    )
    paths = {
        "candidate_params": Path(bundle.params_path).resolve(),
        "candidate_manifest": Path(bundle.manifest_path).resolve(),
        "candidate_config": Path(bundle.config_path).resolve(),
        "candidate_result": result_path,
        "candidate_training_curve": curve_path,
    }
    digests = {
        "candidate_params": _require_sha256(
            bundle.params_sha256, "bundle params SHA256"
        ),
        "candidate_manifest": _require_sha256(
            bundle.manifest_sha256, "bundle manifest SHA256"
        ),
        "candidate_config": _require_sha256(
            bundle.config_sha256, "bundle config SHA256"
        ),
        "candidate_result": result_sha,
        "candidate_training_curve": curve_sha,
    }
    expected_paths = {
        label: Path(path).resolve() for label, path in CANDIDATE_FILE_PATHS.items()
    }
    if (
        paths != expected_paths
        or output_params_path != paths["candidate_params"]
        or output_params_sha != digests["candidate_params"]
        or config_record_path != paths["candidate_config"]
        or config_record_sha != digests["candidate_config"]
    ):
        raise ValueError("forward-v6 candidate five-file bundle binding drifted")
    return {
        label: {
            "path": _evaluation_path_key(paths[label]),
            "sha256": digests[label],
        }
        for label in CANDIDATE_FILE_PATHS
    }


def _candidate_file_snapshot(bundle: Any) -> dict[str, dict[str, str]]:
    """Rehash all five bound files and reject any bundle/current drift."""

    expected = _candidate_bundle_snapshot(bundle)
    current = {
        label: {
            "path": _evaluation_path_key(path),
            "sha256": sha256_file(path),
        }
        for label, path in CANDIDATE_FILE_PATHS.items()
    }
    if not _type_exact_equal(current, expected):
        raise ValueError("forward-v6 candidate five-file current snapshot drifted")
    return current


def _adapter_source_hashes() -> dict[str, str]:
    return {
        ADAPTER_SOURCE_KEY: sha256_file(ADAPTER_PATH),
        ADAPTER_AUTHORIZATION_SOURCE_KEY: sha256_file(
            ADAPTER_AUTHORIZATION_PATH
        ),
    }


def _live_pinned_central_hashes() -> dict[str, str]:
    """Rehash the exact central trio and return its artifact-keyed closure."""

    result: dict[str, str] = {}
    for artifact_label, frozen_label in CENTRAL_ARTIFACT_SOURCE_LABELS.items():
        path, pinned_digest = PINNED_FROZEN_SOURCES[frozen_label]
        current_digest = sha256_file(path)
        if current_digest != pinned_digest:
            raise ValueError(
                f"forward-v6 pinned central source drifted: {artifact_label}"
            )
        result[artifact_label] = current_digest
    return result


def _augment_evaluation_hashes(
    base_hashes: Mapping[str, str], adapter_hashes: Mapping[str, str]
) -> dict[str, str]:
    result = dict(base_hashes)
    for path, digest in adapter_hashes.items():
        if path in result:
            raise ValueError(f"forward-v6 adapter source key collision: {path}")
        result[path] = digest
    expected_existing = {
        str(path.relative_to(EXP_ROOT)).replace("\\", "/"): digest
        for path, digest in PINNED_FROZEN_SOURCES.values()
    }
    missing_or_drifted = {
        path: (result.get(path), digest)
        for path, digest in expected_existing.items()
        if result.get(path) != digest
    }
    if missing_or_drifted:
        raise ValueError(
            f"frozen evaluation source provenance drifted: {missing_or_drifted}"
        )
    return result


def _six_runtime_witnesses(artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidate = artifact.get("episodes")
    baseline_record = artifact.get("official_v22_baseline")
    baseline = (
        baseline_record.get("episodes")
        if isinstance(baseline_record, Mapping)
        else None
    )
    if not isinstance(candidate, list) or not isinstance(baseline, list):
        raise ValueError("forward-v6 diagnostic requires candidate and baseline episodes")
    if len(candidate) != 3 or len(baseline) != 3:
        raise ValueError("forward-v6 diagnostic requires exactly six episodes")
    witnesses: list[Mapping[str, Any]] = []
    groups = (
        ("candidate", candidate),
        ("official_v22_baseline", baseline),
    )
    for group_name, episodes in groups:
        for index, (expected_seed, episode) in enumerate(
            zip(FIXED_FORWARD_SEEDS, episodes, strict=True)
        ):
            if not isinstance(episode, Mapping):
                raise ValueError(
                    f"forward-v6 {group_name} episode {index} is not an object"
                )
            if not _exact_int_value(episode.get("seed"), expected_seed):
                raise ValueError(
                    f"forward-v6 {group_name} episode seed/order drifted at {index}"
                )
            witness = episode.get("forward_v6_runtime_witness")
            if not isinstance(witness, Mapping):
                raise ValueError(
                    f"forward-v6 {group_name} episode {index} witness is missing"
                )
            validate_forward_v6_runtime_witness(witness)
            witnesses.append(witness)
    return witnesses


def _adapter_provenance_record(
    *,
    adapter_hashes_pre: Mapping[str, str],
    adapter_hashes_post: Mapping[str, str],
    adapter_hashes_current: Mapping[str, str],
    frozen_hashes_pre: Mapping[str, str],
    frozen_hashes_post: Mapping[str, str],
    frozen_hashes_current: Mapping[str, str],
    candidate_snapshot_pre: Mapping[str, Mapping[str, str]],
    candidate_snapshot_post: Mapping[str, Mapping[str, str]],
    candidate_snapshot_current: Mapping[str, Mapping[str, str]],
    witness_hashes: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": ADAPTER_CONTRACT_ID,
        "training_contract_id": TRAINING_CONTRACT_ID,
        "method": (
            "FUNCTIONTYPE_CLONED_GLOBALS_AND_ISOLATED_CALLABLE_STATE_"
            "WITHOUT_MODULE_MUTATION"
        ),
        "diagnostic_artifact_kind": DIAGNOSTIC_ARTIFACT_KIND,
        "factory_flags": {
            "forward_v4_substep_contact": True,
            "forward_iteration_v6_contact_abort_island_only": True,
            "reverse_iteration_v6_absolute_full_leg_targets": False,
        },
        "direct_primitive_trace": {
            "control_tick_count": CONTROL_TICK_COUNT,
            "physics_substeps_per_control": PHYSICS_SUBSTEPS_PER_CONTROL,
            "physics_trace_row_count": PHYSICS_TRACE_ROW_COUNT,
            "gait_sample_count": GAIT_SAMPLE_COUNT,
            "dynamic6_bitwise_endpoint_exact_required": True,
            "actual_next_state_gait_endpoint_bitwise_exact_required": True,
            "outer_mjx_env_step_trace_used": False,
        },
        "adapter_source": {
            "path": ADAPTER_SOURCE_KEY,
            "sha256_pre": adapter_hashes_pre[ADAPTER_SOURCE_KEY],
            "sha256_post": adapter_hashes_post[ADAPTER_SOURCE_KEY],
            "sha256_current": adapter_hashes_current[ADAPTER_SOURCE_KEY],
            "unchanged": True,
        },
        "adapter_authorization": {
            "path": ADAPTER_AUTHORIZATION_SOURCE_KEY,
            "sha256_pre": adapter_hashes_pre[ADAPTER_AUTHORIZATION_SOURCE_KEY],
            "sha256_post": adapter_hashes_post[ADAPTER_AUTHORIZATION_SOURCE_KEY],
            "sha256_current": adapter_hashes_current[
                ADAPTER_AUTHORIZATION_SOURCE_KEY
            ],
            "unchanged": True,
        },
        "frozen_source_hashes_pre": dict(frozen_hashes_pre),
        "frozen_source_hashes_post": dict(frozen_hashes_post),
        "frozen_source_hashes_current": dict(frozen_hashes_current),
        "candidate_bundle_snapshot_pre": {
            label: dict(value) for label, value in candidate_snapshot_pre.items()
        },
        "candidate_bundle_snapshot_post": {
            label: dict(value) for label, value in candidate_snapshot_post.items()
        },
        "candidate_bundle_snapshot_current": {
            label: dict(value)
            for label, value in candidate_snapshot_current.items()
        },
        "six_episode_witness_sha256": list(witness_hashes),
        "six_episode_witness_set_sha256": canonical_json_sha256(
            list(witness_hashes)
        ),
        "original_module_globals_and_function_references_unchanged": True,
        "promotion_evidence_allowed": False,
        "candidate_adoption_allowed": False,
        "release_allowed": False,
        "hardware_deployment": "PROHIBITED",
    }


def validate_adapter_provenance_record(
    record: Mapping[str, Any],
    *,
    current_evaluation_hashes: Mapping[str, str],
    bundle: Any,
) -> None:
    """Bind the serialized record to current bytes and the validated bundle."""

    expected_top = {
        "schema_version",
        "contract_id",
        "training_contract_id",
        "method",
        "diagnostic_artifact_kind",
        "factory_flags",
        "direct_primitive_trace",
        "adapter_source",
        "adapter_authorization",
        "frozen_source_hashes_pre",
        "frozen_source_hashes_post",
        "frozen_source_hashes_current",
        "candidate_bundle_snapshot_pre",
        "candidate_bundle_snapshot_post",
        "candidate_bundle_snapshot_current",
        "six_episode_witness_sha256",
        "six_episode_witness_set_sha256",
        "original_module_globals_and_function_references_unchanged",
        "promotion_evidence_allowed",
        "candidate_adoption_allowed",
        "release_allowed",
        "hardware_deployment",
    }
    if set(record) != expected_top:
        raise ValueError("forward-v6 adapter provenance top-level closure drifted")
    witness_hashes = record.get("six_episode_witness_sha256")
    adapter_source = record.get("adapter_source")
    adapter_authorization = record.get("adapter_authorization")
    frozen_pre = record.get("frozen_source_hashes_pre")
    frozen_post = record.get("frozen_source_hashes_post")
    frozen_current = record.get("frozen_source_hashes_current")
    candidate_pre = record.get("candidate_bundle_snapshot_pre")
    candidate_post = record.get("candidate_bundle_snapshot_post")
    candidate_current = record.get("candidate_bundle_snapshot_current")
    expected_flags = {
        "forward_v4_substep_contact": True,
        "forward_iteration_v6_contact_abort_island_only": True,
        "reverse_iteration_v6_absolute_full_leg_targets": False,
    }
    expected_trace = {
        "control_tick_count": CONTROL_TICK_COUNT,
        "physics_substeps_per_control": PHYSICS_SUBSTEPS_PER_CONTROL,
        "physics_trace_row_count": PHYSICS_TRACE_ROW_COUNT,
        "gait_sample_count": GAIT_SAMPLE_COUNT,
        "dynamic6_bitwise_endpoint_exact_required": True,
        "actual_next_state_gait_endpoint_bitwise_exact_required": True,
        "outer_mjx_env_step_trace_used": False,
    }
    expected_frozen = {
        label: digest for label, (_path, digest) in PINNED_FROZEN_SOURCES.items()
    }
    actual_adapter = _adapter_source_hashes()
    actual_frozen = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    actual_candidate = _candidate_file_snapshot(bundle)
    bundle_candidate = _candidate_bundle_snapshot(bundle)
    expected_adapter_source = {
        "path": ADAPTER_SOURCE_KEY,
        "sha256_pre": actual_adapter[ADAPTER_SOURCE_KEY],
        "sha256_post": actual_adapter[ADAPTER_SOURCE_KEY],
        "sha256_current": actual_adapter[ADAPTER_SOURCE_KEY],
        "unchanged": True,
    }
    expected_adapter_authorization = {
        "path": ADAPTER_AUTHORIZATION_SOURCE_KEY,
        "sha256_pre": actual_adapter[ADAPTER_AUTHORIZATION_SOURCE_KEY],
        "sha256_post": actual_adapter[ADAPTER_AUTHORIZATION_SOURCE_KEY],
        "sha256_current": actual_adapter[ADAPTER_AUTHORIZATION_SOURCE_KEY],
        "unchanged": True,
    }
    evaluation_bindings = {
        **actual_adapter,
        **{
            _evaluation_path_key(path): digest
            for path, digest in (
                (source_path, source_sha)
                for source_path, source_sha in PINNED_FROZEN_SOURCES.values()
            )
        },
        **{
            file_record["path"]: file_record["sha256"]
            for file_record in actual_candidate.values()
        },
    }
    checks = {
        "schema": _exact_int_value(record.get("schema_version"), 1),
        "contract": record.get("contract_id") == ADAPTER_CONTRACT_ID,
        "training": record.get("training_contract_id") == TRAINING_CONTRACT_ID,
        "method": record.get("method")
        == "FUNCTIONTYPE_CLONED_GLOBALS_AND_ISOLATED_CALLABLE_STATE_WITHOUT_MODULE_MUTATION",
        "kind": record.get("diagnostic_artifact_kind")
        == DIAGNOSTIC_ARTIFACT_KIND,
        "flags": _type_exact_equal(record.get("factory_flags"), expected_flags),
        "trace": _type_exact_equal(
            record.get("direct_primitive_trace"), expected_trace
        ),
        "adapter_source_current_bytes": _type_exact_equal(
            adapter_source, expected_adapter_source
        ),
        "adapter_authorization_current_bytes": _type_exact_equal(
            adapter_authorization, expected_adapter_authorization
        ),
        "adapter_authorization_pin": actual_adapter.get(
            ADAPTER_AUTHORIZATION_SOURCE_KEY
        )
        == PINNED_ADAPTER_AUTHORIZATION_SHA256,
        "frozen_current_bytes_exact_pins": _type_exact_equal(
            actual_frozen, expected_frozen
        ),
        "frozen_pre_exact": _type_exact_equal(frozen_pre, expected_frozen),
        "frozen_post_exact": _type_exact_equal(frozen_post, expected_frozen),
        "frozen_current_exact": _type_exact_equal(
            frozen_current, expected_frozen
        ),
        "candidate_bundle_current_exact": _type_exact_equal(
            bundle_candidate, actual_candidate
        ),
        "candidate_pre_exact": _type_exact_equal(
            candidate_pre, actual_candidate
        ),
        "candidate_post_exact": _type_exact_equal(
            candidate_post, actual_candidate
        ),
        "candidate_current_exact": _type_exact_equal(
            candidate_current, actual_candidate
        ),
        "evaluation_bindings_current": isinstance(
            current_evaluation_hashes, Mapping
        )
        and all(
            current_evaluation_hashes.get(path) == digest
            for path, digest in evaluation_bindings.items()
        ),
        "witness_count": type(witness_hashes) is list and len(witness_hashes) == 6,
        "witness_set": isinstance(witness_hashes, list)
        and record.get("six_episode_witness_set_sha256")
        == canonical_json_sha256(witness_hashes),
        "module_unchanged": type(record.get(
            "original_module_globals_and_function_references_unchanged"
        )) is bool
        and record.get("original_module_globals_and_function_references_unchanged")
        is True,
        "promotion": type(record.get("promotion_evidence_allowed")) is bool
        and record.get("promotion_evidence_allowed") is False,
        "adoption": type(record.get("candidate_adoption_allowed")) is bool
        and record.get("candidate_adoption_allowed") is False,
        "release": type(record.get("release_allowed")) is bool
        and record.get("release_allowed") is False,
        "hardware": record.get("hardware_deployment") == "PROHIBITED",
    }
    if isinstance(witness_hashes, list):
        checks["witness_hash_syntax"] = all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in witness_hashes
        )
    if not all(checks.values()):
        raise ValueError(f"forward-v6 adapter provenance drifted: {checks}")


def _expected_forward_v6_adapter_evaluation_contract() -> dict[str, Any]:
    return {
        "contract_id": ADAPTER_CONTRACT_ID,
        "fixed_seeds": list(FIXED_FORWARD_SEEDS),
        "physical_command_mps_radps": list(FIXED_FORWARD_COMMAND),
        "control_tick_count": CONTROL_TICK_COUNT,
        "physics_substep_count": PHYSICS_TRACE_ROW_COUNT,
        "gait_sample_count": GAIT_SAMPLE_COUNT,
        "strict_threshold_source": "safe_gait_experts/gait_quality.py",
        "strict_threshold_source_sha256": PINNED_GAIT_QUALITY_SHA256,
        "strict_thresholds_unchanged": True,
        "promotion_eligible": False,
    }


def _expected_final_evaluation_contract() -> dict[str, Any]:
    return {
        "fixed_seeds": list(FIXED_FORWARD_SEEDS),
        "physical_command_mps_radps": list(FIXED_FORWARD_COMMAND),
        "duration_s": 6.0,
        "control_timestep_s": 0.02,
        "physics_timestep_s": 0.002,
        "control_tick_count": CONTROL_TICK_COUNT,
        "physics_substep_count": PHYSICS_TRACE_ROW_COUNT,
        "gait_sample_count": GAIT_SAMPLE_COUNT,
        "gait_quality_semantics": (
            "FULL_CURRENT_P0_RECOMPUTED_FROM_N_PLUS_ONE_SUBSTEP_TRACE"
        ),
        "reset": "EXACT_SAFE_INIT_NO_RESET_NOISE",
        "policy_inference": "BRAX_DETERMINISTIC_NORMAL_TANH_ACTOR116",
        "reverse_composition": None,
        "forward_v6_adapter": _expected_forward_v6_adapter_evaluation_contract(),
    }


def _validate_final_artifact_surface(payload: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "artifact_kind",
        "hardware_deployment",
        "adoption_allowed",
        "release_allowed",
        "standalone_direct_runtime_allowed",
        "execution_provider",
        "created_at_utc",
        "candidate",
        "evaluation_contract",
        "central_hashes",
        "episodes",
        "official_v22_baseline",
        "summary",
        "runtime_provenance",
        "promotion_allowed",
    }
    checks = {
        "top_level_closure": set(payload) == expected_top,
        "schema": _exact_int_value(payload.get("schema_version"), 1),
        "kind": payload.get("artifact_kind") == DIAGNOSTIC_ARTIFACT_KIND,
        "created_at_utc": type(payload.get("created_at_utc")) is str
        and bool(payload.get("created_at_utc")),
        "hardware": payload.get("hardware_deployment") == "PROHIBITED",
        "execution": payload.get("execution_provider") == "CPU",
        "promotion": type(payload.get("promotion_allowed")) is bool
        and payload.get("promotion_allowed") is False,
        "adoption": type(payload.get("adoption_allowed")) is bool
        and payload.get("adoption_allowed") is False,
        "release": type(payload.get("release_allowed")) is bool
        and payload.get("release_allowed") is False,
        "standalone": type(payload.get("standalone_direct_runtime_allowed"))
        is bool
        and payload.get("standalone_direct_runtime_allowed") is False,
        "evaluation_contract": _type_exact_equal(
            payload.get("evaluation_contract"),
            _expected_final_evaluation_contract(),
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"forward-v6 final artifact surface drifted: {checks}")


def _validate_adapter_provenance_stage(
    payload: Mapping[str, Any],
    *,
    bundle: Any,
    current_evaluation_hashes: Mapping[str, str] | None,
    allow_initial_record_absent: bool,
) -> None:
    """Infer initial/final state only from the artifact's three source maps."""

    if not _exact_int_value(payload.get("schema_version"), 1):
        raise ValueError("forward-v6 artifact schema_version must be exact int 1")
    provenance = payload.get("runtime_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("forward-v6 adapter runtime provenance is missing")
    source_maps = []
    for key in (
        "evaluation_source_hashes_pre",
        "evaluation_source_hashes_post",
        "evaluation_source_hashes_current",
    ):
        mapping = provenance.get(key)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"forward-v6 artifact source map is missing: {key}")
        source_maps.append(mapping)
    source_pre, source_post, source_current = source_maps
    if (
        not _type_exact_equal(dict(source_pre), dict(source_post))
        or not _type_exact_equal(dict(source_post), dict(source_current))
    ):
        raise ValueError("forward-v6 artifact pre/post/current source maps differ")
    if current_evaluation_hashes is not None and not _type_exact_equal(
        dict(current_evaluation_hashes), dict(source_current)
    ):
        raise ValueError(
            "forward-v6 caller evaluation hashes differ from artifact current"
        )

    adapter_keys = {ADAPTER_SOURCE_KEY, ADAPTER_AUTHORIZATION_SOURCE_KEY}
    present = [adapter_keys & set(mapping) for mapping in source_maps]
    record = provenance.get("forward_v6_strict_evaluator_adapter")
    if all(not keys for keys in present):
        contract = payload.get("evaluation_contract")
        initial_shape = bool(
            isinstance(contract, Mapping)
            and "forward_v6_adapter" not in contract
            and "promotion_allowed" not in payload
            and record is None
        )
        if allow_initial_record_absent is not True or not initial_shape:
            raise ValueError(
                "forward-v6 adapter provenance may be absent only in the private initial stage"
            )
        return
    if any(keys != adapter_keys for keys in present):
        raise ValueError("forward-v6 adapter source-key stage is partial")
    if bundle is None:
        raise ValueError("forward-v6 augmented provenance requires a trusted bundle")
    if not isinstance(record, Mapping):
        raise ValueError("forward-v6 adapter provenance record is missing")
    _validate_final_artifact_surface(payload)
    live_central_hashes = _live_pinned_central_hashes()
    provenance_central_hashes = provenance.get("central_hashes")
    central_evaluation_bindings = {
        _evaluation_path_key(PINNED_FROZEN_SOURCES[frozen_label][0]): digest
        for artifact_label, frozen_label in CENTRAL_ARTIFACT_SOURCE_LABELS.items()
        for digest in (live_central_hashes[artifact_label],)
    }
    central_binding_checks = {
        "artifact_central_hashes": _type_exact_equal(
            payload.get("central_hashes"), live_central_hashes
        ),
        "provenance_central_hashes": _type_exact_equal(
            provenance_central_hashes, live_central_hashes
        ),
        "evaluation_central_hashes": all(
            _type_exact_equal(source_current.get(path), digest)
            for path, digest in central_evaluation_bindings.items()
        ),
    }
    if not all(central_binding_checks.values()):
        raise ValueError(
            "forward-v6 live central provenance drifted: "
            f"{central_binding_checks}"
        )
    runtime_witnesses = _six_runtime_witnesses(payload)
    ordered_witness_hashes = [
        witness["qualifying_trace_sha256"] for witness in runtime_witnesses
    ]
    witness_binding_checks = {
        "ordered_candidate_then_baseline_hashes": _type_exact_equal(
            record.get("six_episode_witness_sha256"), ordered_witness_hashes
        ),
        "ordered_witness_set_sha256": _type_exact_equal(
            record.get("six_episode_witness_set_sha256"),
            canonical_json_sha256(ordered_witness_hashes),
        ),
    }
    if not all(witness_binding_checks.values()):
        raise ValueError(
            "forward-v6 adapter episode witness provenance drifted: "
            f"{witness_binding_checks}"
        )
    validate_adapter_provenance_record(
        record,
        current_evaluation_hashes=source_current,
        bundle=bundle,
    )


def _augment_and_revalidate(
    *,
    artifact: dict[str, Any],
    bundle: Any,
    central_hashes: Mapping[str, str],
    base_evaluation_hashes: Mapping[str, str],
    adapter_hashes_pre: Mapping[str, str],
    adapter_hashes_post: Mapping[str, str],
    frozen_hashes_pre: Mapping[str, str],
    frozen_hashes_post: Mapping[str, str],
    candidate_snapshot_pre: Mapping[str, Mapping[str, str]],
    candidate_snapshot_post: Mapping[str, Mapping[str, str]],
    validator: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    adapter_hashes_current = _adapter_source_hashes()
    frozen_hashes_current = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    candidate_snapshot_current = _candidate_file_snapshot(bundle)
    if not _type_exact_equal(adapter_hashes_pre, adapter_hashes_post) or not (
        _type_exact_equal(adapter_hashes_post, adapter_hashes_current)
    ):
        raise RuntimeError("forward-v6 adapter/authorization changed during evaluation")
    if not _type_exact_equal(frozen_hashes_pre, frozen_hashes_post) or not (
        _type_exact_equal(frozen_hashes_post, frozen_hashes_current)
    ):
        raise RuntimeError("forward-v6 frozen sources changed during evaluation")
    if not _type_exact_equal(candidate_snapshot_pre, candidate_snapshot_post) or not (
        _type_exact_equal(candidate_snapshot_post, candidate_snapshot_current)
    ):
        raise RuntimeError("forward-v6 candidate inputs changed during evaluation")
    augmented = _augment_evaluation_hashes(
        base_evaluation_hashes, adapter_hashes_post
    )
    witnesses = _six_runtime_witnesses(artifact)
    witness_hashes = [str(witness["qualifying_trace_sha256"]) for witness in witnesses]
    provenance = artifact.get("runtime_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("forward-v6 strict runtime provenance is missing")
    provenance["evaluation_source_hashes_pre"] = dict(augmented)
    provenance["evaluation_source_hashes_post"] = dict(augmented)
    provenance["evaluation_source_hashes_current"] = dict(augmented)
    provenance["pre_post_source_hashes_unchanged"] = True
    record = _adapter_provenance_record(
        adapter_hashes_pre=adapter_hashes_pre,
        adapter_hashes_post=adapter_hashes_post,
        adapter_hashes_current=adapter_hashes_current,
        frozen_hashes_pre=frozen_hashes_pre,
        frozen_hashes_post=frozen_hashes_post,
        frozen_hashes_current=frozen_hashes_current,
        candidate_snapshot_pre=candidate_snapshot_pre,
        candidate_snapshot_post=candidate_snapshot_post,
        candidate_snapshot_current=candidate_snapshot_current,
        witness_hashes=witness_hashes,
    )
    validate_adapter_provenance_record(
        record,
        current_evaluation_hashes=augmented,
        bundle=bundle,
    )
    provenance["forward_v6_strict_evaluator_adapter"] = record
    artifact["artifact_kind"] = DIAGNOSTIC_ARTIFACT_KIND
    artifact["promotion_allowed"] = False
    artifact["adoption_allowed"] = False
    artifact["release_allowed"] = False
    artifact["hardware_deployment"] = "PROHIBITED"
    contract = artifact.get("evaluation_contract")
    if not isinstance(contract, dict):
        raise ValueError("forward-v6 evaluation contract is missing")
    contract["forward_v6_adapter"] = (
        _expected_forward_v6_adapter_evaluation_contract()
    )
    audit = validator(
        artifact,
        bundle=bundle,
        current_central_hashes=central_hashes,
        current_evaluation_hashes=augmented,
        require_all_three_pass=False,
    )
    summary = artifact.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("forward-v6 strict summary is missing")
    summary["recomputed_validation_passed"] = bool(
        audit.get("passing_seed_count") == summary.get("passing_seed_count")
    )
    validator(
        artifact,
        bundle=bundle,
        current_central_hashes=central_hashes,
        current_evaluation_hashes=augmented,
        require_all_three_pass=False,
    )
    _validate_augmented_artifact_contract(
        artifact,
        bundle=bundle,
        current_evaluation_hashes=augmented,
    )
    if (
        not _type_exact_equal(_adapter_source_hashes(), adapter_hashes_current)
        or not _type_exact_equal(
            _verify_file_bindings(PINNED_FROZEN_SOURCES), frozen_hashes_current
        )
        or not _type_exact_equal(
            _candidate_file_snapshot(bundle), candidate_snapshot_current
        )
    ):
        raise RuntimeError("forward-v6 final provenance bytes changed")
    return artifact, augmented


def _validate_augmented_artifact_contract(
    artifact: Mapping[str, Any],
    *,
    bundle: Any,
    current_evaluation_hashes: Mapping[str, str] | None,
) -> None:
    if (
        not _exact_int_value(artifact.get("schema_version"), 1)
        or artifact.get("artifact_kind") != DIAGNOSTIC_ARTIFACT_KIND
        or artifact.get("hardware_deployment") != "PROHIBITED"
        or type(artifact.get("promotion_allowed")) is not bool
        or artifact.get("promotion_allowed") is not False
        or type(artifact.get("adoption_allowed")) is not bool
        or artifact.get("adoption_allowed") is not False
        or type(artifact.get("release_allowed")) is not bool
        or artifact.get("release_allowed") is not False
        or artifact.get("execution_provider") != "CPU"
    ):
        raise ValueError("forward-v6 diagnostic safety surface drifted")
    contract = artifact.get("evaluation_contract")
    expected_adapter_contract = _expected_forward_v6_adapter_evaluation_contract()
    if (
        not isinstance(contract, Mapping)
        or not _type_exact_equal(
            contract.get("fixed_seeds"), list(FIXED_FORWARD_SEEDS)
        )
        or not _type_exact_equal(
            contract.get("physical_command_mps_radps"),
            list(FIXED_FORWARD_COMMAND),
        )
        or not _exact_float_value(contract.get("duration_s"), 6.0)
        or not _exact_int_value(
            contract.get("control_tick_count"), CONTROL_TICK_COUNT
        )
        or not _exact_int_value(
            contract.get("physics_substep_count"), PHYSICS_TRACE_ROW_COUNT
        )
        or not _exact_int_value(
            contract.get("gait_sample_count"), GAIT_SAMPLE_COUNT
        )
        or not _type_exact_equal(
            contract.get("forward_v6_adapter"), expected_adapter_contract
        )
    ):
        raise ValueError("forward-v6 exact evaluation contract drifted")
    _six_runtime_witnesses(artifact)
    _validate_adapter_provenance_stage(
        artifact,
        bundle=bundle,
        current_evaluation_hashes=current_evaluation_hashes,
        allow_initial_record_absent=False,
    )


def run_adapter(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str], Any]:
    _validate_exact_cli(args)
    load_and_validate_adapter_authorization(args.adapter_authorization)
    forward_authorization = load_json_strict(FORWARD_V6_AUTHORIZATION_PATH)
    adapter_hashes_pre = _adapter_source_hashes()
    frozen_hashes_pre = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    base, post, alignment = _load_frozen_modules()
    after_import = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    if after_import != frozen_hashes_pre:
        raise RuntimeError("forward-v6 frozen sources changed while importing")
    graph = build_forward_v6_call_graph(
        base,
        post,
        alignment,
        forward_v6_authorization=forward_authorization,
    )
    bundle_pre = graph["bundle_validator"](
        params_path=args.params,
        manifest_path=args.manifest,
        expected_params_sha256=args.params_sha256,
        expected_manifest_sha256=args.manifest_sha256,
        trusted_run_root=args.trusted_run_root,
        allow_wiring_diagnostic=False,
    )
    candidate_snapshot_pre = _candidate_file_snapshot(bundle_pre)
    original_contract = _module_contract_snapshot(base, post, alignment)
    try:
        artifact, bundle, central_hashes, evaluation_hashes = graph[
            "run_evaluation"
        ](args)
    finally:
        _assert_module_contract_unchanged(
            base, post, alignment, original_contract
        )
    if graph["factory_binding"].call_count != 1:
        raise RuntimeError("forward-v6 evaluator must construct one environment class")
    _validate_forward_v6_bundle(bundle, forward_authorization)
    if not _type_exact_equal(bundle.candidate_record(), bundle_pre.candidate_record()):
        raise RuntimeError("forward-v6 validated bundle identity changed")
    adapter_hashes_post = _adapter_source_hashes()
    frozen_hashes_post = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    candidate_snapshot_post = _candidate_file_snapshot(bundle)
    artifact, augmented = _augment_and_revalidate(
        artifact=artifact,
        bundle=bundle,
        central_hashes=central_hashes,
        base_evaluation_hashes=evaluation_hashes,
        adapter_hashes_pre=adapter_hashes_pre,
        adapter_hashes_post=adapter_hashes_post,
        frozen_hashes_pre=frozen_hashes_pre,
        frozen_hashes_post=frozen_hashes_post,
        candidate_snapshot_pre=candidate_snapshot_pre,
        candidate_snapshot_post=candidate_snapshot_post,
        validator=graph["artifact_validator"],
    )
    if not _type_exact_equal(_adapter_source_hashes(), adapter_hashes_post):
        raise RuntimeError("forward-v6 adapter sources changed before output write")
    if not _type_exact_equal(
        _verify_file_bindings(PINNED_FROZEN_SOURCES), frozen_hashes_post
    ):
        raise RuntimeError("forward-v6 frozen sources changed before output write")
    if not _type_exact_equal(
        _candidate_file_snapshot(bundle), candidate_snapshot_post
    ):
        raise RuntimeError("forward-v6 candidate files changed before output write")
    _assert_module_contract_unchanged(base, post, alignment, original_contract)
    return artifact, augmented, base.write_new_json


def main() -> None:
    args = _resolve_process_start_paths(build_parser().parse_args())
    artifact, evaluation_hashes, write_new_json = run_adapter(args)
    artifact_sha = write_new_json(args.output, artifact)
    result = {
        "diagnostic_strict_artifact": {
            "path": str(args.output),
            "sha256": artifact_sha,
            "artifact_kind": DIAGNOSTIC_ARTIFACT_KIND,
        },
        "adapter_contract_id": ADAPTER_CONTRACT_ID,
        "evaluation_source_hash_count": len(evaluation_hashes),
        "passing_seed_count": artifact["summary"]["passing_seed_count"],
        "all_three_strict_pass": artifact["summary"]["all_three_strict_pass"],
        "promotion_allowed": False,
        "adoption_allowed": False,
        "release_allowed": False,
        "hardware_deployment": "PROHIBITED",
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    if not artifact["summary"]["all_three_strict_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
