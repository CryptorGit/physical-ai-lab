"""Strict CPU evaluator adapter for the reverse iteration-v6 candidate.

The frozen evaluator predates the v6 absolute-target trace schema.  This
adapter therefore preserves its source bytes and module globals, clones the
reusable call graph, and owns only the reverse-v6 environment opt-in, runtime
trace, and independent host rederivation.  It cannot train, promote, adopt,
release, overwrite evidence, or authorize hardware use.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
from types import FunctionType, ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = Path(__file__).resolve()
BASE_EVALUATOR_PATH = EXP_ROOT / "scripts" / "evaluate_h4_training_candidate.py"
POST_TRAINING_PATH = EXP_ROOT / "safe_gait_experts" / "h4_post_training.py"
CORE_PATH = EXP_ROOT / "safe_gait_experts" / "h4_training_alignment.py"
RUNNER_PATH = EXP_ROOT / "scripts" / "train_h4_aligned_expert.py"
REVERSE_V6_AUTHORIZATION_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_iteration_v6_absolute_full_leg_targets_authorization.json"
)
SELECTED_REVERSE_TEACHER_PATH = (
    EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_selected_v1.json"
)
ADAPTER_AUTHORIZATION_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_iteration_v6_absolute_full_leg_targets_strict_evaluator_adapter_v1_authorization.json"
)

ADAPTER_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_"
    "STRICT_EVALUATOR_ADAPTER_V1"
)
TRAINING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_250K_FROM_V22"
)
CORE_CONTRACT_ID = "ABSOLUTE_FULL_LEG_TARGETS_WITH_TEACHER_TIMING_ONLY"
DIAGNOSTIC_ARTIFACT_KIND = (
    "openduckmini_h4_reverse_iteration_v6_absolute_full_leg_targets_"
    "strict_evaluation_diagnostic"
)
REAL_REVERSE_EVALUATION_SEMANTICS = (
    "ABSOLUTE_FULL_LEG_TARGETS_WITH_TEACHER_TIMING_ONLY"
)
CONTROL_TRACE_SEMANTICS = (
    "FLOAT32_ABSOLUTE_DECODER_THEN_MARGIN_THEN_ONE_0P04_PRECOMPOSER_"
    "THEN_ONE_FINAL_GUARD"
)

PINNED_ADAPTER_AUTHORIZATION_SHA256 = (
    "076fe724675e8e20a1fa04ab80a8b0b08775e72b51f99d5d04c05693509b5ff8"
)
PINNED_REVERSE_V6_AUTHORIZATION_SHA256 = (
    "6a10315593761a4d0ed034b331fe14e3f682bf8154a252e6820a5dd4f71038fe"
)
PINNED_SELECTED_REVERSE_TEACHER_SHA256 = (
    "7a24a7c9096a1c4a9dc72ac85ec01c5e0a41acf8214d80cc7e2cf4ccc50ae237"
)

TRUSTED_RUN_ROOT = (
    EXP_ROOT / "artifacts" / "h4_iteration_v6_training_runs_20260809"
)
RUN_RELATIVE_PATH = Path(
    "reverse"
) / "h4_reverse_250k_seed20260810_iteration_v6_absolute_full_leg_targets_level4_v1"
CANDIDATE_ROOT = TRUSTED_RUN_ROOT / RUN_RELATIVE_PATH
EXPECTED_PARAMS_PATH = CANDIDATE_ROOT / "final_params.pkl"
EXPECTED_MANIFEST_PATH = CANDIDATE_ROOT / "run_manifest.json"
EXPECTED_OUTPUT_PATH = CANDIDATE_ROOT / "h4_integrated_strict_3x6s_v1.json"
DEFAULT_SOURCE_ROOT = Path("/home/user/openduck_training_20260729")
DEFAULT_GENERATED_ROOT = EXP_ROOT / "artifacts" / "generated_playground"
DEFAULT_V22_PARENT_CHECKPOINT = Path(
    "/home/user/openduck_training_runs/"
    "calibrated_hybrid_yaw_cost_v22_300m/2026_07_29_154427_10485760"
)

PINNED_FROZEN_SOURCES: dict[str, tuple[Path, str]] = {
    "h4_candidate_evaluator": (
        BASE_EVALUATOR_PATH,
        "c214d086e6d66f6f9f98c7268481899e4133961dcc5355d738d4cd134a82e6ae",
    ),
    "h4_post_training": (
        POST_TRAINING_PATH,
        "3fa23b759de391e963c8d16b74fa5019076a2bd0bc67dac384ace60310653240",
    ),
    "h4_training_alignment": (
        CORE_PATH,
        "5da1d3a8a2c505a5ce4bc6621f76dd3031070cdb467a4cde96b4ed3c23190c02",
    ),
    "h4_runner": (
        RUNNER_PATH,
        "d6d075ab257494599dec1beebdac523912b30d42dfc712699a9ebed3a131e8ef",
    ),
    "central_evaluator": (
        EXP_ROOT / "scripts" / "evaluate_routed_transitions.py",
        "31fb8846fc6267f28d032bca164dee2c872bfb484ebe272850100834bf1b1a9b",
    ),
    "central_gait_quality": (
        EXP_ROOT / "safe_gait_experts" / "gait_quality.py",
        "b28e1ceb4cb6406411150bbad772a78203b8163bf10adcd79a5f31f83da5f2de",
    ),
    "central_routed_evaluation": (
        EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py",
        "f25ed858fbb5753fdcfd9e76f08396d0f09f95ac4696eb2e50cb5c128b80db09",
    ),
    "reverse_iteration_v6_authorization": (
        REVERSE_V6_AUTHORIZATION_PATH,
        PINNED_REVERSE_V6_AUTHORIZATION_SHA256,
    ),
    "selected_reverse_teacher": (
        SELECTED_REVERSE_TEACHER_PATH,
        PINNED_SELECTED_REVERSE_TEACHER_SHA256,
    ),
}

ADAPTER_SOURCE_KEY = (
    "scripts/evaluate_h4_training_candidate_"
    "reverse_v6_absolute_full_leg_targets_v1.py"
)
ADAPTER_AUTHORIZATION_SOURCE_KEY = (
    "artifacts/h4_reverse_iteration_v6_absolute_full_leg_targets_"
    "strict_evaluator_adapter_v1_authorization.json"
)

ACTION_WIDTH = 14
CONTROL_TICKS = 300
PHYSICS_SUBSTEPS = 3000
PHYSICS_SUBSTEPS_PER_CONTROL = 10
GAIT_SAMPLES = 3001
CONTROL_DT_S = 0.02
PHYSICS_DT_S = 0.002
STRICT_DURATION_S = 6.0
STRICT_SEEDS = (20_260_810, 20_265_810, 20_271_810)
STRICT_COMMAND = (-0.05, 0.0, 0.0)
LEG_INDICES = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13))
HEAD_INDICES = np.asarray((5, 6, 7, 8))
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
SAFE_INIT_F32 = np.asarray(
    (
        0.0006879152046783626,
        0.018229752923976605,
        -0.2166932894736842,
        0.470534,
        -0.2696627602339181,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0010318728070175438,
        -0.022357244152046783,
        0.2184130774853801,
        0.474317533625731,
        -0.2737902514619883,
    ),
    dtype=np.float32,
)
SAFE_LOWER_F32 = np.asarray(
    (
        -0.523599,
        -0.436332,
        -0.472466,
        -0.28532,
        -0.45099,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.12732,
        -0.348214,
        -0.193282,
        -0.526155,
        -0.412641,
    ),
    dtype=np.float32,
)
SAFE_UPPER_F32 = np.asarray(
    (
        0.35435,
        0.436332,
        0.240835,
        0.475534,
        0.401903,
        0.0,
        0.0,
        0.0,
        0.0,
        0.523599,
        0.436332,
        1.22173,
        0.562971,
        0.435651,
    ),
    dtype=np.float32,
)
# These float64 arrays intentionally retain the JSON-decimal central SafetyAudit
# thresholds.  They are not decoder bounds: they exist only to reproduce and
# serialize the frozen legacy float64 diagnostic without hiding float32-boundary
# aliases.
CENTRAL_SAFE_LOWER_F64 = np.asarray(
    (
        -0.523599,
        -0.436332,
        -0.472466,
        -0.28532,
        -0.45099,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.12732,
        -0.348214,
        -0.193282,
        -0.526155,
        -0.412641,
    ),
    dtype=np.float64,
)
CENTRAL_SAFE_UPPER_F64 = np.asarray(
    (
        0.35435,
        0.436332,
        0.240835,
        0.475534,
        0.401903,
        0.0,
        0.0,
        0.0,
        0.0,
        0.523599,
        0.436332,
        1.22173,
        0.562971,
        0.435651,
    ),
    dtype=np.float64,
)
MARGIN_F32 = np.float32(0.05)
SLEW_F32 = np.float32(0.04)
CENTRAL_MARGIN_F64 = 0.05
CENTRAL_SLEW_LIMIT_RAD_PER_S_F64 = 2.0
CENTRAL_TOLERANCE_F64 = 1.0e-9
DIRECTIONAL_SPAN_F32 = np.float32(0.9)
BASE_SPAN_F32 = np.float32(0.25)
TRACKING_SIGMA = 0.01
TEACHER_TABLE_ROWS = 54
TEACHER_ENTRY_PHASE_BINS = 14.0
TEACHER_PHASE_ADVANCE_BINS = 1.62
SOURCE_PERIOD_BINS = 27

V6_VECTOR_TRACE_KEYS = {
    "v6_decoder_action",
    "v6_decoder_raw_targets",
    "v6_decoder_margin_targets",
    "v6_upstream_margin_targets",
    "v6_precomposer_targets",
}
V6_FLOAT_TRACE_KEYS = {
    "v6_decoder_max_abs_error",
    "v6_residual_authority_scale",
    "v6_decoder_guard_lag_max_rad",
    "v6_teacher_source_phase_before",
    "v6_teacher_table_phase",
    "v6_direct_physics_dynamic6_endpoint_max_abs_error",
    "v6_direct_physics_snapshot_endpoint_max_abs_error",
}
V6_INTEGER_TRACE_KEYS = {
    "v6_decoder_leg_count",
    "v6_decoder_margin_saturation_count",
    "v6_decoder_action_clip_count",
    "v6_precomposer_call_count",
    "v6_final_guard_call_count",
    "v6_decoder_assertion_token",
    "v6_direct_physics_substep_count",
    "v6_direct_physics_dynamic6_field_count",
    "v6_direct_physics_snapshot_endpoint_field_count",
}
V6_BOOLEAN_TRACE_KEYS = {
    "v6_decoder_exact",
    "v6_decoder_leg_count_exact",
    "v6_decoder_head_zero_exact",
    "v6_teacher_target_contribution_zero_exact",
    "v6_decoder_all_finite",
    "v6_precomposer_call_count_exact",
    "v6_final_guard_call_count_exact",
    "v6_decoder_violation",
    "v6_direct_physics_dynamic6_endpoint_exact",
    "v6_direct_physics_dynamic6_all_finite",
    "v6_direct_physics_applied_target_exact",
    "v6_direct_physics_snapshot_endpoint_exact",
    "v6_direct_physics_snapshot_endpoint_all_finite",
}
DYNAMIC6_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "time",
    "qacc_warmstart",
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
V6_ENVIRONMENT_CHECK_KEYS = {
    "teacher_validation",
    "table_shape",
    "table_finite",
    "head_zero",
    "cadence",
    "advance",
    "entry",
    "v6_kind",
    "v6_contract",
    "v6_family",
    "teacher_timing",
    "action_parameterization",
    "runtime_teacher_rows",
    "runtime_source_period",
    "runtime_residual_scale",
    "runtime_action_delay_exact_zero",
    "runtime_phase_entry",
    "runtime_phase_advance",
    "factory_reverse_v6_exact",
    "factory_forward_flags_false",
    "legacy_reward_target_zero",
    "legacy_reward_contact_zero",
    "legacy_tracking_sigma_0p01",
    "residual_authority_zero",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            raise ValueError(f"required field is missing: {'.'.join(keys)}")
        current = current[key]
    return current


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{label} schema drifted: "
            f"missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be lowercase SHA256")
    return value


def _type_exact_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _type_exact_equal(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _type_exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, float):
        return struct.pack(">d", actual) == struct.pack(">d", expected)
    return actual == expected


def _array_equal_float32_bits(actual: Any, expected: Any) -> bool:
    left = np.asarray(actual, dtype=np.float32)
    right = np.asarray(expected, dtype=np.float32)
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint32), right.view(np.uint32)
    )


def _strict_device_leaf_bitwise_equal(
    actual: Any, expected: Any, *, xp: Any
) -> Any:
    left = xp.asarray(actual)
    right = xp.asarray(expected)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("direct physics parity leaf shape/dtype differs")
    exact = xp.all(left == right)
    if np.issubdtype(np.dtype(left.dtype), np.inexact):
        # Equal finite IEEE values differ in bits only for signed zero.  The
        # signbit comparison closes that final alias without dtype views that
        # are awkward under JIT tracing.
        exact = exact & xp.all(xp.signbit(left) == xp.signbit(right))
    return exact


def _direct_physics_dynamic6_parity(
    actual_data: Any, replay_data: Any, *, xp: Any
) -> tuple[Any, Any, int, Any]:
    exact = xp.asarray(True)
    maximum_error = xp.asarray(0.0, dtype=xp.float32)
    all_finite = xp.asarray(True)
    for field in DYNAMIC6_FIELDS:
        actual = xp.asarray(getattr(actual_data, field))
        replay = xp.asarray(getattr(replay_data, field))
        exact = exact & _strict_device_leaf_bitwise_equal(
            actual, replay, xp=xp
        )
        if np.issubdtype(np.dtype(actual.dtype), np.inexact):
            all_finite = (
                all_finite
                & xp.all(xp.isfinite(actual))
                & xp.all(xp.isfinite(replay))
            )
            field_error = xp.max(
                xp.abs(actual - replay),
                initial=xp.asarray(0.0, dtype=actual.dtype),
            )
            maximum_error = xp.maximum(
                maximum_error, field_error.astype(xp.float32)
            )
    return exact, maximum_error, len(DYNAMIC6_FIELDS), all_finite


def _direct_physics_snapshot_parity(
    actual: Mapping[str, Any], replay: Mapping[str, Any], *, xp: Any
) -> tuple[Any, Any, int, Any]:
    if set(actual) != set(SNAPSHOT_ENDPOINT_FIELDS) or set(replay) != set(
        SNAPSHOT_ENDPOINT_FIELDS
    ):
        raise ValueError("direct physics snapshot endpoint schema differs")
    exact = xp.asarray(True)
    maximum_error = xp.asarray(0.0, dtype=xp.float32)
    all_finite = xp.asarray(True)
    for field in SNAPSHOT_ENDPOINT_FIELDS:
        actual_value = xp.asarray(actual[field])
        replay_value = xp.asarray(replay[field])
        exact = exact & _strict_device_leaf_bitwise_equal(
            actual_value, replay_value, xp=xp
        )
        if np.issubdtype(np.dtype(actual_value.dtype), np.inexact):
            all_finite = (
                all_finite
                & xp.all(xp.isfinite(actual_value))
                & xp.all(xp.isfinite(replay_value))
            )
            field_error = xp.max(
                xp.abs(actual_value - replay_value),
                initial=xp.asarray(0.0, dtype=actual_value.dtype),
            )
            maximum_error = xp.maximum(
                maximum_error, field_error.astype(xp.float32)
            )
    return exact, maximum_error, len(SNAPSHOT_ENDPOINT_FIELDS), all_finite


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


def validate_adapter_authorization_payload(payload: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "artifact_kind",
        "status",
        "hardware_deployment",
        "authorization",
        "scope",
        "candidate_binding",
        "authorization_bindings",
        "frozen_source_bindings",
        "runtime_contract",
        "artifact_contract",
        "provenance_contract",
        "decision",
    }
    _require_exact_keys(payload, expected_top, "adapter authorization")
    expected_authorization = {
        "exact_fixed_3x6s_cpu_evaluation": True,
        "ppo_training": False,
        "promotion_evidence": False,
        "candidate_adoption": False,
        "package_release": False,
        "hardware": False,
    }
    expected_factory = {
        "forward_v4_substep_contact": False,
        "forward_iteration_v6_contact_abort_island_only": False,
        "reverse_iteration_v6_absolute_full_leg_targets": True,
    }
    expected_reward = {
        "target_imitation": 0.0,
        "contact_imitation": 0.0,
        "tracking_sigma": TRACKING_SIGMA,
    }
    expected_scope = {
        "contract_id": ADAPTER_CONTRACT_ID,
        "training_contract_id": TRAINING_CONTRACT_ID,
        "expert": "reverse",
        "adapter_source_path": ADAPTER_SOURCE_KEY,
        "evaluation_artifact_kind": DIAGNOSTIC_ARTIFACT_KIND,
        "method": (
            "FUNCTIONTYPE_CLONED_GLOBALS_WITH_ADAPTER_OWNED_"
            "V6_TRACE_AND_REDERIVATION"
        ),
        "promotion_eligible": False,
    }
    expected_candidate = {
        "trusted_root_relative_path": str(
            TRUSTED_RUN_ROOT.relative_to(EXP_ROOT)
        ).replace("\\", "/"),
        "run_relative_path": str(RUN_RELATIVE_PATH).replace("\\", "/"),
        "expert": "reverse",
        "status": "COMPLETED",
        "activity": "PPO_PILOT_TRAINING",
        "required_mode": "reverse_iteration_v6_absolute_full_leg_targets",
        "final_params_basename": "final_params.pkl",
        "manifest_basename": "run_manifest.json",
        "standard_output_basename": "h4_integrated_strict_3x6s_v1.json",
        "candidate_hashes_pre_authorized": False,
        "params_and_manifest_sha256_are_required_cli_inputs": True,
        "validated_bundle_cross_binding_required_before_pickle": True,
        "overwrite_allowed": False,
    }
    expected_runtime = {
        "factory_flags": expected_factory,
        "legacy_reward_config_exact": expected_reward,
        "backward_residual_scale": 0.0,
        "decoder_semantics": (
            "FLOAT32_ABSOLUTE_FULL_LEG_SAFE_INIT_DIRECTIONAL_LINEAR_PLUS_QUINTIC"
        ),
        "active_leg_indices": LEG_INDICES.tolist(),
        "hard_zero_head_indices": HEAD_INDICES.tolist(),
        "directional_span_fraction": 0.9,
        "near_zero_base_cap_rad": 0.25,
        "nonlinear_exponent": 5,
        "inward_margin_rad": 0.05,
        "precomposer_slew_rad_per_tick": 0.04,
        "precomposer_value_semantics": (
            "HARD_CLIP_WITH_SMOOTH_TANH_SURROGATE_DERIVATIVE"
        ),
        "final_guard_slew_rad_per_tick": 0.04,
        "physics_trace_semantics": (
            "DIRECT_MJX_STEP_REPLAY_FROM_CONTROL_ENTRY_WITH_ACTUAL_APPLIED_TARGETS"
        ),
        "physics_substeps_per_control": PHYSICS_SUBSTEPS_PER_CONTROL,
        "dynamic6_endpoint_bitwise_exact_required": True,
        "snapshot_endpoint_bitwise_exact_required": True,
        "legacy_float64_control_audit_diagnostic_only": True,
        "legacy_float64_control_audit_counts_exact_required": True,
        "decoder_leg_count": 10,
        "precomposer_call_count": 1,
        "final_guard_call_count": 1,
        "teacher_role": "PHASE_TIMING_PRIOR_ONLY",
        "teacher_table_rows": TEACHER_TABLE_ROWS,
        "teacher_entry_phase_preincrement_bins": 14.0,
        "teacher_phase_advance_bins_per_control": 1.62,
        "source_period_bins": SOURCE_PERIOD_BINS,
        "teacher_target_contribution": 0.0,
        "device_exact_booleans_required": True,
        "numeric_tolerance_used": False,
    }
    expected_artifact = {
        "fixed_seeds": list(STRICT_SEEDS),
        "physical_command_mps_radps": list(STRICT_COMMAND),
        "fixed_episode_count": 3,
        "fixed_duration_s": 6.0,
        "control_tick_count": CONTROL_TICKS,
        "physics_substep_count": PHYSICS_SUBSTEPS,
        "gait_sample_count": GAIT_SAMPLES,
        "execution_provider": "CPU",
        "central_threshold_sources_unchanged": True,
        "official_v22_same_process_baseline_required": True,
        "real_contract_may_claim_legacy_teacher_plus_residual": False,
        "write_new_standard_json_only": True,
        "promotion_builder_or_output_prohibited": True,
    }
    expected_provenance = {
        "adapter_authorization_may_pin_adapter_sha256": False,
        "adapter_must_pin_authorization_bytes_before_json_parse": True,
        "frozen_sources_pre_post_exact": True,
        "candidate_bundle_files_pre_post_exact": True,
        "original_module_globals_and_function_references_unchanged": True,
        "adapter_outer_mjx_env_step_replacement_prohibited": True,
        "historical_teacher_residual_sources_may_remain_training_source_closure_only": True,
    }
    expected_decision = {
        "evaluation": "AUTHORIZED_EXACT_CPU_FIXED_3X6S_ONLY",
        "promotion": "PROHIBITED",
        "candidate_adoption": "PROHIBITED",
        "release": "PROHIBITED",
        "hardware": "PROHIBITED",
    }
    checks = {
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("artifact_kind")
        == (
            "openduckmini_h4_reverse_iteration_v6_absolute_full_leg_targets_"
            "strict_evaluator_adapter_authorization"
        ),
        "status": payload.get("status")
        == "AUTHORIZED_EXACT_SIMULATION_STRICT_EVALUATION_ONLY",
        "hardware": payload.get("hardware_deployment") == "PROHIBITED",
        "authorization": _type_exact_equal(
            payload.get("authorization"), expected_authorization
        ),
        "scope_exact": _type_exact_equal(payload.get("scope"), expected_scope),
        "candidate_exact": _type_exact_equal(
            payload.get("candidate_binding"), expected_candidate
        ),
        "runtime_exact": _type_exact_equal(
            payload.get("runtime_contract"), expected_runtime
        ),
        "artifact_exact": _type_exact_equal(
            payload.get("artifact_contract"), expected_artifact
        ),
        "provenance_exact": _type_exact_equal(
            payload.get("provenance_contract"), expected_provenance
        ),
        "decision_exact": _type_exact_equal(
            payload.get("decision"), expected_decision
        ),
        "contract": _nested(payload, "scope", "contract_id")
        == ADAPTER_CONTRACT_ID,
        "training_contract": _nested(payload, "scope", "training_contract_id")
        == TRAINING_CONTRACT_ID,
        "expert": _nested(payload, "scope", "expert") == "reverse",
        "adapter_path": _nested(payload, "scope", "adapter_source_path")
        == ADAPTER_SOURCE_KEY,
        "artifact_kind": _nested(payload, "scope", "evaluation_artifact_kind")
        == DIAGNOSTIC_ARTIFACT_KIND,
        "method": _nested(payload, "scope", "method")
        == (
            "FUNCTIONTYPE_CLONED_GLOBALS_WITH_ADAPTER_OWNED_"
            "V6_TRACE_AND_REDERIVATION"
        ),
        "promotion_ineligible": _nested(payload, "scope", "promotion_eligible")
        is False,
        "trusted_root": _nested(
            payload, "candidate_binding", "trusted_root_relative_path"
        )
        == str(TRUSTED_RUN_ROOT.relative_to(EXP_ROOT)).replace("\\", "/"),
        "run": _nested(payload, "candidate_binding", "run_relative_path")
        == str(RUN_RELATIVE_PATH).replace("\\", "/"),
        "candidate_status": _nested(payload, "candidate_binding", "status")
        == "COMPLETED",
        "candidate_hashes_unpinned": _nested(
            payload, "candidate_binding", "candidate_hashes_pre_authorized"
        )
        is False,
        "cli_sha": _nested(
            payload,
            "candidate_binding",
            "params_and_manifest_sha256_are_required_cli_inputs",
        )
        is True,
        "bundle_before_pickle": _nested(
            payload,
            "candidate_binding",
            "validated_bundle_cross_binding_required_before_pickle",
        )
        is True,
        "factory": _nested(payload, "runtime_contract", "factory_flags")
        == expected_factory,
        "legacy_reward": _nested(
            payload, "runtime_contract", "legacy_reward_config_exact"
        )
        == expected_reward,
        "residual_zero": _nested(
            payload, "runtime_contract", "backward_residual_scale"
        )
        == 0.0,
        "decoder": _nested(payload, "runtime_contract", "decoder_semantics")
        == "FLOAT32_ABSOLUTE_FULL_LEG_SAFE_INIT_DIRECTIONAL_LINEAR_PLUS_QUINTIC",
        "legs": _nested(payload, "runtime_contract", "active_leg_indices")
        == LEG_INDICES.tolist(),
        "head": _nested(payload, "runtime_contract", "hard_zero_head_indices")
        == HEAD_INDICES.tolist(),
        "counts": (
            _nested(payload, "runtime_contract", "decoder_leg_count") == 10
            and _nested(payload, "runtime_contract", "precomposer_call_count")
            == 1
            and _nested(payload, "runtime_contract", "final_guard_call_count")
            == 1
        ),
        "teacher_role": _nested(payload, "runtime_contract", "teacher_role")
        == "PHASE_TIMING_PRIOR_ONLY",
        "fixed_seeds": _nested(payload, "artifact_contract", "fixed_seeds")
        == list(STRICT_SEEDS),
        "no_legacy_claim": _nested(
            payload,
            "artifact_contract",
            "real_contract_may_claim_legacy_teacher_plus_residual",
        )
        is False,
        "write_new": _nested(
            payload, "artifact_contract", "write_new_standard_json_only"
        )
        is True,
        "no_cycle": _nested(
            payload,
            "provenance_contract",
            "adapter_authorization_may_pin_adapter_sha256",
        )
        is False,
        "decision": _nested(payload, "decision", "evaluation")
        == "AUTHORIZED_EXACT_CPU_FIXED_3X6S_ONLY",
    }
    expected_fixed = {
        label: {"path": str(path.relative_to(EXP_ROOT)).replace("\\", "/"), "sha256": digest}
        for label, (path, digest) in PINNED_FROZEN_SOURCES.items()
        if label
        not in {"reverse_iteration_v6_authorization", "selected_reverse_teacher"}
    }
    checks["frozen_sources"] = _type_exact_equal(
        payload.get("frozen_source_bindings"), expected_fixed
    )
    checks["v6_auth"] = _nested(
        payload, "authorization_bindings", "reverse_iteration_v6"
    )
    checks["v6_auth"] = _type_exact_equal(checks["v6_auth"], {
        "path": str(REVERSE_V6_AUTHORIZATION_PATH.relative_to(EXP_ROOT)).replace(
            "\\", "/"
        ),
        "sha256": PINNED_REVERSE_V6_AUTHORIZATION_SHA256,
    })
    teacher_binding = _nested(
        payload, "authorization_bindings", "selected_reverse_teacher"
    )
    checks["teacher"] = _type_exact_equal(teacher_binding, {
        "path": str(SELECTED_REVERSE_TEACHER_PATH.relative_to(EXP_ROOT)).replace(
            "\\", "/"
        ),
        "sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
    })
    expected_bindings = {
        "reverse_iteration_v6": {
            "path": str(
                REVERSE_V6_AUTHORIZATION_PATH.relative_to(EXP_ROOT)
            ).replace("\\", "/"),
            "sha256": PINNED_REVERSE_V6_AUTHORIZATION_SHA256,
        },
        "selected_reverse_teacher": {
            "path": str(
                SELECTED_REVERSE_TEACHER_PATH.relative_to(EXP_ROOT)
            ).replace("\\", "/"),
            "sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
        },
    }
    expected_payload = {
        "schema_version": 1,
        "artifact_kind": (
            "openduckmini_h4_reverse_iteration_v6_absolute_full_leg_targets_"
            "strict_evaluator_adapter_authorization"
        ),
        "status": "AUTHORIZED_EXACT_SIMULATION_STRICT_EVALUATION_ONLY",
        "hardware_deployment": "PROHIBITED",
        "authorization": expected_authorization,
        "scope": expected_scope,
        "candidate_binding": expected_candidate,
        "authorization_bindings": expected_bindings,
        "frozen_source_bindings": expected_fixed,
        "runtime_contract": expected_runtime,
        "artifact_contract": expected_artifact,
        "provenance_contract": expected_provenance,
        "decision": expected_decision,
    }
    checks["entire_payload_type_and_value_exact"] = _type_exact_equal(
        dict(payload), expected_payload
    )
    if not all(checks.values()):
        raise ValueError(f"adapter authorization semantic binding failed: {checks}")


def load_and_validate_adapter_authorization(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved != ADAPTER_AUTHORIZATION_PATH.resolve():
        raise ValueError("adapter authorization path must remain exact")
    actual_sha = sha256_file(resolved)
    if actual_sha != PINNED_ADAPTER_AUTHORIZATION_SHA256:
        raise ValueError(f"adapter authorization SHA256 drifted: {actual_sha}")
    payload = load_json_strict(resolved)
    validate_adapter_authorization_payload(payload)
    return payload


def _float_is_positive_zero(value: Any) -> bool:
    return (
        type(value) is float
        and value == 0.0
        and not bool(np.signbit(np.asarray(value, dtype=np.float64)))
    )


def _validate_reverse_v6_bundle(bundle: Any) -> None:
    config = bundle.config
    flags = (
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
    mode_exact = all(
        config.get(name) is (name == "reverse_iteration_v6_absolute_full_leg_targets")
        for name in flags
    )
    auth = config.get(
        "reverse_iteration_v6_absolute_full_leg_targets_authorization"
    )
    selected = config.get("selected_reverse_teacher")
    expected_core_source = {
        "path": str(CORE_PATH.resolve()),
        "sha256": PINNED_FROZEN_SOURCES["h4_training_alignment"][1],
    }
    expected_selected_teacher = {
        "path": str(SELECTED_REVERSE_TEACHER_PATH.resolve()),
        "sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
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
    }
    checks = {
        "run_name": bundle.run_name == CANDIDATE_ROOT.name,
        "expert": bundle.expert == "reverse",
        "status": bundle.status == "COMPLETED",
        "activity": bundle.activity == "PPO_PILOT_TRAINING",
        "training_contract": config.get("training_contract_id")
        == TRAINING_CONTRACT_ID,
        "mode_exact": mode_exact,
        "residual_positive_zero": _float_is_positive_zero(
            config.get("backward_residual_scale")
        ),
        "h4_parent_prohibited": isinstance(auth, Mapping)
        and auth.get("h4_parent_checkpoint_allowed") is False,
        "v4_gain_not_inherited": isinstance(auth, Mapping)
        and auth.get("v4_gain_inherited") is False,
        "v5_parent_not_inherited": isinstance(auth, Mapping)
        and auth.get("v5_parent_checkpoint_inherited") is False,
        "core_source": _type_exact_equal(
            config.get("iteration_v6_core_source"), expected_core_source
        ),
        "auth_record": isinstance(auth, Mapping)
        and auth.get("path") == str(REVERSE_V6_AUTHORIZATION_PATH.resolve())
        and auth.get("sha256") == PINNED_REVERSE_V6_AUTHORIZATION_SHA256
        and auth.get("contract_id") == TRAINING_CONTRACT_ID,
        "teacher_record": _type_exact_equal(
            selected, expected_selected_teacher
        ),
    }
    authorization = load_json_strict(REVERSE_V6_AUTHORIZATION_PATH)
    checks["action_contract"] = _type_exact_equal(
        config.get("action_parameterization_contract"),
        authorization.get("action_parameterization_contract"),
    )
    checks["teacher_contract"] = _type_exact_equal(
        config.get("teacher_timing_contract"),
        authorization.get("teacher_timing_contract"),
    )
    legacy_audit = auth.get("legacy_reward_config_audit") if isinstance(auth, Mapping) else None
    expected_legacy = {
        "target_imitation": 0.0,
        "contact_imitation": 0.0,
        "tracking_sigma": TRACKING_SIGMA,
        "backward_residual_scale": 0.0,
    }
    per_environment = (
        legacy_audit.get("per_environment")
        if isinstance(legacy_audit, Mapping)
        else None
    )
    checks["legacy_reward_audit"] = bool(
        isinstance(legacy_audit, Mapping)
        and _type_exact_equal(legacy_audit.get("expected"), expected_legacy)
        and isinstance(per_environment, Mapping)
        and bool(per_environment)
        and all(
            _type_exact_equal(record, expected_legacy)
            for record in per_environment.values()
        )
        and legacy_audit.get("passed") is True
    )
    if not all(checks.values()):
        raise ValueError(f"exact reverse-v6 candidate bundle drifted: {checks}")


def _candidate_file_paths(bundle: Any) -> dict[str, Path]:
    return {
        "candidate_params": bundle.params_path,
        "candidate_manifest": bundle.manifest_path,
        "candidate_config": bundle.config_path,
        "candidate_result": Path(bundle.manifest["outputs"]["result"]["path"]),
        "candidate_training_curve": Path(
            bundle.manifest["outputs"]["training_curve"]["path"]
        ),
    }


def _source_map_key(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(EXP_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _candidate_file_snapshot(bundle: Any) -> dict[str, str]:
    paths = _candidate_file_paths(bundle)
    return {name: sha256_file(Path(path).resolve()) for name, path in paths.items()}


def _candidate_evaluation_source_bindings(
    bundle: Any, candidate_hashes: Mapping[str, str]
) -> dict[str, str]:
    paths = _candidate_file_paths(bundle)
    if set(candidate_hashes) != set(paths):
        raise ValueError("candidate snapshot schema drifted")
    return {
        _source_map_key(path): candidate_hashes[name]
        for name, path in paths.items()
    }


def _pinned_evaluation_source_bindings() -> dict[str, str]:
    return {
        _source_map_key(path): digest
        for path, digest in PINNED_FROZEN_SOURCES.values()
    }


def _pinned_central_hashes() -> dict[str, str]:
    return {
        _source_map_key(path): digest
        for label, (path, digest) in PINNED_FROZEN_SOURCES.items()
        if label.startswith("central_")
    }


def _central_hash_bindings_live_and_exact(
    artifact_central_hashes: Any,
    provenance_central_hashes: Any,
    current_central_hashes: Mapping[str, str] | None,
) -> bool:
    expected = _pinned_central_hashes()
    try:
        live_sources = _verify_file_bindings(
            {
                label: binding
                for label, binding in PINNED_FROZEN_SOURCES.items()
                if label.startswith("central_")
            }
        )
    except ValueError:
        return False
    live = {
        _source_map_key(PINNED_FROZEN_SOURCES[label][0]): digest
        for label, digest in live_sources.items()
    }
    return bool(
        _type_exact_equal(artifact_central_hashes, expected)
        and _type_exact_equal(provenance_central_hashes, expected)
        and _type_exact_equal(live, expected)
        and (
            current_central_hashes is None
            or _type_exact_equal(dict(current_central_hashes), expected)
        )
    )


def reverse_v6_decode_float32(action: Any) -> np.ndarray:
    """Independently reproduce the selected float32 absolute decoder."""

    policy = np.asarray(action, dtype=np.float32)
    if policy.shape[-1:] != (ACTION_WIDTH,) or not np.all(np.isfinite(policy)):
        raise ValueError("reverse-v6 decoder action must be finite with width 14")
    bounded = np.clip(policy, np.float32(-1.0), np.float32(1.0))
    positive_span = np.multiply(
        DIRECTIONAL_SPAN_F32,
        np.subtract(SAFE_UPPER_F32, SAFE_INIT_F32, dtype=np.float32),
        dtype=np.float32,
    )
    negative_span = np.multiply(
        DIRECTIONAL_SPAN_F32,
        np.subtract(SAFE_INIT_F32, SAFE_LOWER_F32, dtype=np.float32),
        dtype=np.float32,
    )
    directional = np.where(bounded >= np.float32(0.0), positive_span, negative_span)
    base = np.minimum(BASE_SPAN_F32, directional)
    magnitude = np.abs(bounded)
    # JAX lowers the literal fifth power to integer_pow as
    # magnitude * ((magnitude * magnitude) * (magnitude * magnitude)).
    # Reproduce those three float32 multiply roundings instead of calling the
    # host libm power routine, which differs by an ULP for some inputs.
    magnitude_squared = np.multiply(magnitude, magnitude, dtype=np.float32)
    magnitude_fourth = np.multiply(
        magnitude_squared, magnitude_squared, dtype=np.float32
    )
    fifth = np.multiply(magnitude, magnitude_fourth, dtype=np.float32)
    target_magnitude = np.add(
        np.multiply(base, magnitude, dtype=np.float32),
        np.multiply(
            np.subtract(directional, base, dtype=np.float32),
            fifth,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    decoded = np.add(
        SAFE_INIT_F32,
        np.multiply(np.sign(bounded), target_magnitude, dtype=np.float32),
        dtype=np.float32,
    )
    decoded = np.asarray(decoded, dtype=np.float32)
    decoded[..., HEAD_INDICES] = np.float32(0.0)
    return decoded


def reverse_v6_margin_clip_float32(targets: Any) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float32)
    if values.shape[-1:] != (ACTION_WIDTH,) or not np.all(np.isfinite(values)):
        raise ValueError("reverse-v6 margin input must be finite with width 14")
    lower = np.add(SAFE_LOWER_F32, MARGIN_F32, dtype=np.float32)
    upper = np.subtract(SAFE_UPPER_F32, MARGIN_F32, dtype=np.float32)
    clipped = np.zeros_like(values, dtype=np.float32)
    clipped[..., LEG_INDICES] = np.clip(
        values[..., LEG_INDICES], lower[LEG_INDICES], upper[LEG_INDICES]
    )
    return clipped


def reverse_v6_precomposer_float32(
    margin_targets: Any, previous_targets: Any
) -> np.ndarray:
    value, _surrogate_derivative = (
        reverse_v6_precomposer_value_and_surrogate_derivative_float32(
            margin_targets, previous_targets
        )
    )
    return value


def reverse_v6_precomposer_value_and_surrogate_derivative_float32(
    margin_targets: Any, previous_targets: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return the hard runtime value and smooth tanh surrogate derivative.

    The device program evaluates soft + stop_gradient(hard - soft).  Its
    forward value is the exact hard clip while its derivative is sech-squared.
    Both are represented here so the host contract does not erase the training
    semantics while rederiving the stored hard value.
    """

    margin = np.asarray(margin_targets, dtype=np.float32)
    previous = np.asarray(previous_targets, dtype=np.float32)
    if margin.shape != previous.shape or margin.shape[-1:] != (ACTION_WIDTH,):
        raise ValueError("reverse-v6 precomposer inputs must have equal width-14 shape")
    raw_delta = np.subtract(margin, previous, dtype=np.float32)
    hard_delta = np.clip(raw_delta, -SLEW_F32, SLEW_F32)
    scaled = np.divide(raw_delta, SLEW_F32, dtype=np.float32)
    tanh_scaled = np.tanh(scaled).astype(np.float32)
    surrogate_derivative = np.subtract(
        np.float32(1.0),
        np.multiply(tanh_scaled, tanh_scaled, dtype=np.float32),
        dtype=np.float32,
    )
    desired = np.add(previous, hard_delta, dtype=np.float32)
    desired[..., HEAD_INDICES] = np.float32(0.0)
    surrogate_derivative[..., HEAD_INDICES] = np.float32(0.0)
    return desired, surrogate_derivative


def reverse_v6_final_guard_float32(
    desired_targets: Any, previous_targets: Any
) -> np.ndarray:
    desired = reverse_v6_margin_clip_float32(desired_targets)
    previous = np.asarray(previous_targets, dtype=np.float32)
    if desired.shape != previous.shape:
        raise ValueError("reverse-v6 final-guard inputs must have equal shape")
    delta = np.clip(
        np.subtract(desired, previous, dtype=np.float32), -SLEW_F32, SLEW_F32
    )
    applied = np.zeros_like(previous, dtype=np.float32)
    applied[..., LEG_INDICES] = np.clip(
        np.add(
            previous[..., LEG_INDICES],
            delta[..., LEG_INDICES],
            dtype=np.float32,
        ),
        SAFE_LOWER_F32[LEG_INDICES],
        SAFE_UPPER_F32[LEG_INDICES],
    )
    return applied


def _expected_phase_timeline_float32() -> tuple[np.ndarray, np.ndarray]:
    phase_scale = np.float32(TEACHER_TABLE_ROWS / SOURCE_PERIOD_BINS)
    entry_source = np.float32(TEACHER_ENTRY_PHASE_BINS / float(phase_scale))
    source_rate = np.float32(TEACHER_PHASE_ADVANCE_BINS / float(phase_scale))
    before = np.empty(CONTROL_TICKS, dtype=np.float32)
    after = np.empty(CONTROL_TICKS, dtype=np.float32)
    before[0] = entry_source
    for index in range(CONTROL_TICKS):
        after[index] = np.remainder(
            np.add(before[index], source_rate, dtype=np.float32),
            np.float32(SOURCE_PERIOD_BINS),
            dtype=np.float32,
        )
        if index + 1 < CONTROL_TICKS:
            before[index + 1] = after[index]
    table = np.remainder(
        np.multiply(after, phase_scale, dtype=np.float32),
        np.float32(TEACHER_TABLE_ROWS),
        dtype=np.float32,
    )
    return before, table


def _central_safe_excess_float64(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    return np.maximum(
        np.maximum(lower - values, values - upper), np.float64(0.0)
    )


def _serialized_float_leaves_are_exact(value: Any) -> bool:
    if isinstance(value, list):
        return all(_serialized_float_leaves_are_exact(item) for item in value)
    return type(value) is float


def _float_trace_array_exact(value: Any, *, label: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype != np.float32:
            raise ValueError(f"reverse-v6 float trace {label} numeric type drifted")
        raw = value
    elif isinstance(value, list) and _serialized_float_leaves_are_exact(value):
        raw = np.asarray(value, dtype=np.float64)
        if not np.array_equal(raw, raw.astype(np.float32).astype(np.float64)):
            raise ValueError(f"reverse-v6 float trace {label} numeric type drifted")
    else:
        raise ValueError(f"reverse-v6 float trace {label} numeric type drifted")
    return np.asarray(raw, dtype=np.float32)


def _legacy_float64_central_control_counts(
    trace: Mapping[str, np.ndarray]
) -> dict[str, int]:
    """Reproduce frozen central counts as visible, non-authoritative diagnostics."""

    leg = LEG_INDICES
    lower = CENTRAL_SAFE_LOWER_F64[leg]
    upper = CENTRAL_SAFE_UPPER_F64[leg]
    margin_lower = lower + CENTRAL_MARGIN_F64
    margin_upper = upper - CENTRAL_MARGIN_F64
    raw_target = np.asarray(trace["preclip_targets"], dtype=np.float64)[:, leg]
    desired = np.asarray(
        trace["margin_clipped_targets"], dtype=np.float64
    )[:, leg]
    target = np.asarray(trace["applied_targets"], dtype=np.float64)[:, leg]
    previous = np.asarray(
        trace["previous_targets"], dtype=np.float64
    )[:, leg]
    qpos = np.asarray(trace["joint_qpos"], dtype=np.float64)[:, leg]
    raw_excess = _central_safe_excess_float64(raw_target, lower, upper)
    raw_margin_excess = _central_safe_excess_float64(
        raw_target, margin_lower, margin_upper
    )
    target_excess = _central_safe_excess_float64(target, lower, upper)
    qpos_excess = _central_safe_excess_float64(qpos, lower, upper)
    desired_margin_excess = _central_safe_excess_float64(
        desired, margin_lower, margin_upper
    )
    target_margin_excess = _central_safe_excess_float64(
        target, margin_lower, margin_upper
    )
    previous_margin_excess = _central_safe_excess_float64(
        previous, margin_lower, margin_upper
    )
    slew_rates = np.abs(target - previous) / CONTROL_DT_S
    slew_violation = (
        slew_rates
        > CENTRAL_SLEW_LIMIT_RAD_PER_S_F64 + CENTRAL_TOLERANCE_F64
    )
    startup_mask = target_margin_excess > CENTRAL_TOLERANCE_F64
    authorized_startup = (
        startup_mask
        & (desired_margin_excess <= CENTRAL_TOLERANCE_F64)
        & (target_excess <= CENTRAL_TOLERANCE_F64)
        & ~slew_violation
        & (
            target_margin_excess
            < previous_margin_excess - CENTRAL_TOLERANCE_F64
        )
    )
    return {
        "sample_count": CONTROL_TICKS,
        "nonfinite_sample_count": 0,
        "preclip_target_limit_violations": int(
            np.count_nonzero(raw_excess > CENTRAL_TOLERANCE_F64)
        ),
        "applied_target_limit_violations": int(
            np.count_nonzero(target_excess > CENTRAL_TOLERANCE_F64)
        ),
        "preclip_target_margin_violations": int(
            np.count_nonzero(raw_margin_excess > CENTRAL_TOLERANCE_F64)
        ),
        "desired_target_margin_violations": int(
            np.count_nonzero(desired_margin_excess > CENTRAL_TOLERANCE_F64)
        ),
        "applied_target_margin_violations": int(
            np.count_nonzero(target_margin_excess > CENTRAL_TOLERANCE_F64)
        ),
        "unauthorized_applied_target_margin_violations": int(
            np.count_nonzero(startup_mask & ~authorized_startup)
        ),
        "startup_margin_transition_joint_samples": int(
            np.count_nonzero(authorized_startup)
        ),
        "target_slew_violations": int(np.count_nonzero(slew_violation)),
        "qpos_limit_violations": int(
            np.count_nonzero(qpos_excess > CENTRAL_TOLERANCE_F64)
        ),
    }


def _v6_control_trace_arrays(episode: Mapping[str, Any]) -> dict[str, np.ndarray]:
    trace = episode.get("control_trace")
    if not isinstance(trace, Mapping):
        raise ValueError("reverse-v6 episode control trace is missing")
    matrix_keys = {
        "raw_action",
        "applied_action",
        "preclip_targets",
        "margin_clipped_targets",
        "applied_targets",
        "previous_targets",
        "joint_qpos",
        *V6_VECTOR_TRACE_KEYS,
    }
    expected = {
        "source_dtype",
        "initial_applied_targets",
        *matrix_keys,
        *V6_FLOAT_TRACE_KEYS,
        *V6_INTEGER_TRACE_KEYS,
        *V6_BOOLEAN_TRACE_KEYS,
    }
    _require_exact_keys(trace, expected, "reverse-v6 control trace")
    if trace.get("source_dtype") != "float32":
        raise ValueError("reverse-v6 control trace source dtype drifted")
    try:
        initial = _float_trace_array_exact(
            trace["initial_applied_targets"], label="initial_applied_targets"
        )
    except ValueError as exc:
        raise ValueError("reverse-v6 initial target numeric type drifted") from exc
    result = {"initial_applied_targets": initial}
    for name in matrix_keys | V6_FLOAT_TRACE_KEYS:
        result[name] = _float_trace_array_exact(trace[name], label=name)
    for name in V6_INTEGER_TRACE_KEYS:
        source = trace[name]
        raw = np.asarray(source)
        native_json_integers = (
            isinstance(source, list)
            and len(source) == CONTROL_TICKS
            and all(type(value) is int for value in source)
            and all(
                np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max
                for value in source
            )
        )
        exact_device_int32 = (
            isinstance(source, np.ndarray) and source.dtype == np.int32
        )
        if (
            raw.shape != (CONTROL_TICKS,)
            or not (native_json_integers or exact_device_int32)
        ):
            raise ValueError(f"reverse-v6 integer trace {name} drifted")
        result[name] = raw.astype(np.int32, copy=False)
    for name in V6_BOOLEAN_TRACE_KEYS:
        raw = np.asarray(trace[name])
        if raw.shape != (CONTROL_TICKS,) or raw.dtype != np.bool_:
            raise ValueError(f"reverse-v6 boolean trace {name} drifted")
        result[name] = raw
    if result["initial_applied_targets"].shape != (ACTION_WIDTH,):
        raise ValueError("reverse-v6 initial targets must have width 14")
    if any(result[name].shape != (CONTROL_TICKS, ACTION_WIDTH) for name in matrix_keys):
        raise ValueError("reverse-v6 matrix traces must be exactly 300x14")
    if any(result[name].shape != (CONTROL_TICKS,) for name in V6_FLOAT_TRACE_KEYS):
        raise ValueError("reverse-v6 scalar traces must contain exactly 300 rows")
    if not all(
        np.all(np.isfinite(value))
        for value in result.values()
        if value.dtype != np.bool_
    ):
        raise ValueError("reverse-v6 control trace contains non-finite values")
    return result


def _expected_reverse_v6_episode_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "semantics": CONTROL_TRACE_SEMANTICS,
        "core_contract_id": CORE_CONTRACT_ID,
        "selected_reverse_teacher_sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
        "reverse_iteration_v6_authorization_sha256": (
            PINNED_REVERSE_V6_AUTHORIZATION_SHA256
        ),
        "teacher_role": "PHASE_TIMING_PRIOR_ONLY",
        "teacher_table_rows": TEACHER_TABLE_ROWS,
        "teacher_entry_phase_preincrement_bins": TEACHER_ENTRY_PHASE_BINS,
        "teacher_phase_advance_bins_per_control": TEACHER_PHASE_ADVANCE_BINS,
        "source_period_bins": SOURCE_PERIOD_BINS,
        "teacher_target_contribution": 0.0,
        "backward_residual_scale": 0.0,
        "legacy_reward_config": {
            "target_imitation": 0.0,
            "contact_imitation": 0.0,
            "tracking_sigma": TRACKING_SIGMA,
        },
        "decoder_leg_count": 10,
        "precomposer_call_count": 1,
        "final_guard_call_count": 1,
        "inward_margin_rad": 0.05,
        "slew_rad_per_tick": 0.04,
        "precomposer_value_semantics": (
            "HARD_CLIP_WITH_SMOOTH_TANH_SURROGATE_DERIVATIVE"
        ),
        "physics_trace_semantics": (
            "DIRECT_MJX_STEP_REPLAY_FROM_CONTROL_ENTRY_WITH_ACTUAL_APPLIED_TARGETS"
        ),
        "physics_substeps_per_control": PHYSICS_SUBSTEPS_PER_CONTROL,
        "dynamic6_endpoint_bitwise_exact_required": True,
        "snapshot_endpoint_bitwise_exact_required": True,
        "legacy_float64_control_audit_diagnostic_only": True,
        "legacy_float64_control_audit_counts_exact_required": True,
        "step_entry_physical_command_x_mps": -0.05,
    }


def rederive_reverse_v6_control_contract(
    episode: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute every v6 action-to-target stage from all 300 float32 rows."""

    if episode.get("expert") != "reverse":
        raise ValueError("reverse-v6 rederivation requires reverse expert")
    contract = episode.get("reverse_v6_absolute_decoder_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("reverse-v6 episode contract is missing")
    trace = _v6_control_trace_arrays(episode)
    legacy_float64_counts = _legacy_float64_central_control_counts(trace)
    recorded_central_safety = episode.get("safety_audit")
    recorded_legacy_float64_counts = (
        {
            name: recorded_central_safety.get(name)
            for name in legacy_float64_counts
        }
        if isinstance(recorded_central_safety, Mapping)
        else None
    )
    initial = trace["initial_applied_targets"]
    raw_action = trace["raw_action"]
    applied_action = trace["applied_action"]
    decoder_action = trace["v6_decoder_action"]
    raw_targets = trace["v6_decoder_raw_targets"]
    decoder_margin = trace["v6_decoder_margin_targets"]
    upstream_margin = trace["v6_upstream_margin_targets"]
    preclip = trace["preclip_targets"]
    central_margin = trace["margin_clipped_targets"]
    recorded_desired = trace["v6_precomposer_targets"]
    applied = trace["applied_targets"]
    previous = trace["previous_targets"]
    qpos = trace["joint_qpos"]

    expected_raw = reverse_v6_decode_float32(decoder_action)
    expected_margin = reverse_v6_margin_clip_float32(expected_raw)
    expected_precomposer, expected_surrogate_derivative = (
        reverse_v6_precomposer_value_and_surrogate_derivative_float32(
            expected_margin, previous
        )
    )
    expected_applied = reverse_v6_final_guard_float32(
        expected_precomposer, previous
    )
    expected_applied_action = raw_action.copy()
    expected_applied_action[:, HEAD_INDICES] = np.float32(0.0)
    expected_phase_before, expected_table_phase = (
        _expected_phase_timeline_float32()
    )
    expected_action_clip_count = np.count_nonzero(
        decoder_action[:, LEG_INDICES]
        != np.clip(
            decoder_action[:, LEG_INDICES],
            np.float32(-1.0),
            np.float32(1.0),
        ),
        axis=1,
    ).astype(np.int32)
    expected_margin_saturation_count = np.count_nonzero(
        expected_raw[:, LEG_INDICES] != expected_margin[:, LEG_INDICES], axis=1
    ).astype(np.int32)
    expected_guard_lag = np.max(
        np.abs(
            np.subtract(expected_margin, expected_applied, dtype=np.float32)
        )[:, LEG_INDICES],
        axis=1,
    )

    lower_leg = SAFE_LOWER_F32[LEG_INDICES]
    upper_leg = SAFE_UPPER_F32[LEG_INDICES]
    margin_lower_leg = np.add(lower_leg, MARGIN_F32, dtype=np.float32)
    margin_upper_leg = np.subtract(upper_leg, MARGIN_F32, dtype=np.float32)
    final_guard_margin = reverse_v6_margin_clip_float32(expected_precomposer)
    guard_delta = np.clip(
        np.subtract(final_guard_margin, previous, dtype=np.float32),
        -SLEW_F32,
        SLEW_F32,
    )
    applied_leg = applied[:, LEG_INDICES]
    previous_leg = previous[:, LEG_INDICES]
    qpos_leg = qpos[:, LEG_INDICES]
    applied_outside_margin = (applied_leg < margin_lower_leg) | (
        applied_leg > margin_upper_leg
    )
    precomposer_leg = recorded_desired[:, LEG_INDICES]
    precomposer_outside_margin = (
        (precomposer_leg < margin_lower_leg)
        | (precomposer_leg > margin_upper_leg)
    )
    authorized_startup = np.zeros_like(applied_outside_margin, dtype=bool)
    left_knee_leg_offset = int(np.flatnonzero(LEG_INDICES == 3)[0])
    previous_margin_excess = np.maximum(
        np.maximum(
            margin_lower_leg - previous_leg, previous_leg - margin_upper_leg
        ),
        np.float32(0.0),
    )
    applied_margin_excess = np.maximum(
        np.maximum(
            margin_lower_leg - applied_leg, applied_leg - margin_upper_leg
        ),
        np.float32(0.0),
    )
    authorized_startup[0, left_knee_leg_offset] = bool(
        applied_outside_margin[0, left_knee_leg_offset]
        and not (
            expected_margin[0, 3] < margin_lower_leg[left_knee_leg_offset]
            or expected_margin[0, 3] > margin_upper_leg[left_knee_leg_offset]
        )
        and np.abs(
            np.subtract(
                recorded_desired[0, 3],
                previous[0, 3],
                dtype=np.float32,
            )
        )
        <= SLEW_F32
        and applied_margin_excess[0, left_knee_leg_offset]
        < previous_margin_excess[0, left_knee_leg_offset]
        and np.array_equal(applied[0], expected_applied[0])
    )
    unauthorized_startup = applied_outside_margin & ~authorized_startup
    device_exact_boolean_checks = {
        name: bool(np.all(trace[name] == expected))
        for name, expected in {
            "v6_decoder_exact": True,
            "v6_decoder_leg_count_exact": True,
            "v6_decoder_head_zero_exact": True,
            "v6_teacher_target_contribution_zero_exact": True,
            "v6_decoder_all_finite": True,
            "v6_precomposer_call_count_exact": True,
            "v6_final_guard_call_count_exact": True,
            "v6_decoder_violation": False,
            "v6_direct_physics_dynamic6_endpoint_exact": True,
            "v6_direct_physics_dynamic6_all_finite": True,
            "v6_direct_physics_applied_target_exact": True,
            "v6_direct_physics_snapshot_endpoint_exact": True,
            "v6_direct_physics_snapshot_endpoint_all_finite": True,
        }.items()
    }
    checks = {
        "reverse_v6_episode_contract_exact": _type_exact_equal(
            dict(contract), _expected_reverse_v6_episode_contract()
        ),
        "legacy_float64_control_audit_counts_exact": _type_exact_equal(
            recorded_legacy_float64_counts, legacy_float64_counts
        ),
        "source_dtype_float32": True,
        "reset_safe_init_float32_exact": _array_equal_float32_bits(
            initial, SAFE_INIT_F32
        ),
        "previous_target_timeline_exact": _array_equal_float32_bits(
            previous[0], initial
        )
        and _array_equal_float32_bits(previous[1:], applied[:-1]),
        "decoder_action_exact_raw_policy_action": _array_equal_float32_bits(
            decoder_action, raw_action
        ),
        "applied_action_exact_post_inference_head_mask": _array_equal_float32_bits(
            applied_action, expected_applied_action
        ),
        "decoder_raw_targets_exact_host_float32": _array_equal_float32_bits(
            raw_targets, expected_raw
        ),
        "selected_preclip_targets_exact_decoder_output": _array_equal_float32_bits(
            preclip, expected_raw
        ),
        "decoder_margin_targets_exact_host_float32": _array_equal_float32_bits(
            decoder_margin, expected_margin
        ),
        "runtime_upstream_margin_targets_exact_host_float32": (
            _array_equal_float32_bits(upstream_margin, expected_margin)
        ),
        "central_safety_margin_stage_exact_host_float32": (
            _array_equal_float32_bits(central_margin, expected_margin)
        ),
        "single_precomposer_output_exact_host_float32": _array_equal_float32_bits(
            recorded_desired, expected_precomposer
        ),
        "sole_precomposer_outside_margin_sample_is_explicit_tick0_left_knee": bool(
            np.count_nonzero(precomposer_outside_margin) == 1
            and precomposer_outside_margin[0, left_knee_leg_offset]
            and not np.any(precomposer_outside_margin[1:])
        ),
        "precomposer_smooth_surrogate_derivative_finite": bool(
            np.all(np.isfinite(expected_surrogate_derivative))
            and np.all(
                (expected_surrogate_derivative[:, LEG_INDICES] >= 0)
                & (expected_surrogate_derivative[:, LEG_INDICES] <= 1)
            )
            and np.all(expected_surrogate_derivative[:, HEAD_INDICES] == 0)
        ),
        "single_final_guard_output_exact_host_float32": _array_equal_float32_bits(
            applied, expected_applied
        ),
        "all_head_targets_exact_zero": bool(
            np.all(raw_targets[:, HEAD_INDICES] == 0)
            and np.all(decoder_margin[:, HEAD_INDICES] == 0)
            and np.all(recorded_desired[:, HEAD_INDICES] == 0)
            and np.all(applied[:, HEAD_INDICES] == 0)
        ),
        "decoder_max_abs_error_exact_zero": bool(
            np.all(trace["v6_decoder_max_abs_error"].view(np.uint32) == 0)
        ),
        "residual_authority_scale_exact_zero": bool(
            np.all(trace["v6_residual_authority_scale"].view(np.uint32) == 0)
        ),
        "decoder_leg_count_exact_ten": bool(
            np.all(trace["v6_decoder_leg_count"] == 10)
        ),
        "precomposer_call_count_exact_one": bool(
            np.all(trace["v6_precomposer_call_count"] == 1)
        ),
        "final_guard_call_count_exact_one": bool(
            np.all(trace["v6_final_guard_call_count"] == 1)
        ),
        "direct_physics_substep_count_exact_ten": bool(
            np.all(
                trace["v6_direct_physics_substep_count"]
                == PHYSICS_SUBSTEPS_PER_CONTROL
            )
        ),
        "direct_physics_dynamic6_field_count_exact_six": bool(
            np.all(
                trace["v6_direct_physics_dynamic6_field_count"]
                == len(DYNAMIC6_FIELDS)
            )
        ),
        "direct_physics_dynamic6_max_abs_error_exact_positive_zero": bool(
            np.all(
                trace[
                    "v6_direct_physics_dynamic6_endpoint_max_abs_error"
                ].view(np.uint32)
                == 0
            )
        ),
        "direct_physics_snapshot_endpoint_field_count_exact": bool(
            np.all(
                trace["v6_direct_physics_snapshot_endpoint_field_count"]
                == len(SNAPSHOT_ENDPOINT_FIELDS)
            )
        ),
        "direct_physics_snapshot_endpoint_max_abs_error_exact_positive_zero": bool(
            np.all(
                trace[
                    "v6_direct_physics_snapshot_endpoint_max_abs_error"
                ].view(np.uint32)
                == 0
            )
        ),
        "action_clip_count_exact_host": np.array_equal(
            trace["v6_decoder_action_clip_count"],
            expected_action_clip_count,
        ),
        "margin_saturation_count_exact_host": np.array_equal(
            trace["v6_decoder_margin_saturation_count"],
            expected_margin_saturation_count,
        ),
        "guard_lag_max_exact_host_float32": _array_equal_float32_bits(
            trace["v6_decoder_guard_lag_max_rad"], expected_guard_lag
        ),
        "assertion_token_exact_zero": bool(
            np.all(trace["v6_decoder_assertion_token"].view(np.uint32) == 0)
        ),
        "teacher_source_phase_timeline_exact_float32": _array_equal_float32_bits(
            trace["v6_teacher_source_phase_before"], expected_phase_before
        ),
        "teacher_table_phase_after_preincrement_exact_float32": _array_equal_float32_bits(
            trace["v6_teacher_table_phase"], expected_table_phase
        ),
        "desired_leg_targets_inside_float32_margin": bool(
            np.all(
                (final_guard_margin[:, LEG_INDICES] >= margin_lower_leg)
                & (final_guard_margin[:, LEG_INDICES] <= margin_upper_leg)
            )
        ),
        "guard_internal_float32_delta_within_exact_0p04": bool(
            np.all(np.abs(guard_delta[:, LEG_INDICES]) <= SLEW_F32)
        ),
        "applied_leg_targets_physical_safe": bool(
            np.all(
                (applied_leg >= lower_leg)
                & (applied_leg <= upper_leg)
            )
        ),
        "joint_qpos_physical_safe": bool(
            np.all((qpos_leg >= lower_leg) & (qpos_leg <= upper_leg))
        ),
        "sole_outside_margin_sample_is_authorized_left_knee_startup": bool(
            np.count_nonzero(applied_outside_margin) == 1
            and np.count_nonzero(authorized_startup) == 1
            and np.count_nonzero(unauthorized_startup) == 0
        ),
        "startup_exception_cleared_after_first_tick": bool(
            not np.any(applied_outside_margin[1:])
        ),
        **{
            f"device_{name}_exact": passed
            for name, passed in device_exact_boolean_checks.items()
        },
    }
    checks = {name: bool(value) for name, value in checks.items()}
    guard_output_mismatch_count = int(np.count_nonzero(applied != expected_applied))
    return {
        "schema_version": 4,
        "semantics": "H4_REVERSE_V6_FLOAT32_EXACT_ABSOLUTE_TARGET_CONTROL",
        "control_path": CONTROL_TRACE_SEMANTICS,
        "sample_count": CONTROL_TICKS,
        "source_dtype": "float32",
        "legacy_float64_control_audit_diagnostic_only": True,
        "legacy_float64_control_audit_counts": legacy_float64_counts,
        "exact_slew_cap_rad_per_tick": float(SLEW_F32),
        "desired_margin_violation_count": int(
            np.count_nonzero(
                (final_guard_margin[:, LEG_INDICES] < margin_lower_leg)
                | (final_guard_margin[:, LEG_INDICES] > margin_upper_leg)
            )
        ),
        "float32_slew_violation_count": guard_output_mismatch_count
        + int(np.count_nonzero(np.abs(guard_delta[:, LEG_INDICES]) > SLEW_F32)),
        "guard_output_mismatch_joint_sample_count": guard_output_mismatch_count,
        "guard_internal_delta_violation_joint_sample_count": int(
            np.count_nonzero(np.abs(guard_delta[:, LEG_INDICES]) > SLEW_F32)
        ),
        "applied_outside_margin_joint_sample_count": int(
            np.count_nonzero(applied_outside_margin)
        ),
        "precomposer_outside_margin_joint_sample_count": int(
            np.count_nonzero(precomposer_outside_margin)
        ),
        "authorized_startup_joint_sample_count": int(
            np.count_nonzero(authorized_startup)
        ),
        "unauthorized_startup_joint_sample_count": int(
            np.count_nonzero(unauthorized_startup)
        ),
        "decoder_diagnostics": {
            "maximum_guard_lag_rad": float(np.max(expected_guard_lag)),
            "total_action_clip_count": int(np.sum(expected_action_clip_count)),
            "total_margin_saturation_count": int(
                np.sum(expected_margin_saturation_count)
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _recorded_reverse_v6_control_contract_exact(
    episode: Mapping[str, Any]
) -> bool:
    return _type_exact_equal(
        episode.get("h4_control_contract"),
        rederive_reverse_v6_control_contract(episode),
    )


def _require_bound_file(
    record: Any, *, expected_path: Path, expected_sha256: str, label: str
) -> Path:
    if not isinstance(record, Mapping) or record.get("path") != str(
        expected_path.resolve()
    ) or record.get("sha256") != expected_sha256:
        raise ValueError(f"{label} config binding drifted")
    resolved = expected_path.resolve()
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} source bytes drifted")
    return resolved


def _load_reverse_v6_teacher(bundle: Any) -> dict[str, Any]:
    """Load timing data under v6 authority, never legacy composition authority."""

    selected_path = _require_bound_file(
        bundle.config.get("selected_reverse_teacher"),
        expected_path=SELECTED_REVERSE_TEACHER_PATH,
        expected_sha256=PINNED_SELECTED_REVERSE_TEACHER_SHA256,
        label="selected reverse teacher",
    )
    v6_record = bundle.config.get(
        "reverse_iteration_v6_absolute_full_leg_targets_authorization"
    )
    authorization_path = _require_bound_file(
        v6_record,
        expected_path=REVERSE_V6_AUTHORIZATION_PATH,
        expected_sha256=PINNED_REVERSE_V6_AUTHORIZATION_SHA256,
        label="reverse iteration-v6 authorization",
    )
    selected = load_json_strict(selected_path)
    authorization = load_json_strict(authorization_path)
    teacher = selected.get("teacher")
    adapter = selected.get("adapter_contract")
    if not isinstance(teacher, Mapping) or not isinstance(adapter, Mapping):
        raise ValueError("selected reverse teacher schema drifted")
    table = np.asarray(teacher.get("target_table_rad"), dtype=np.float64)
    checks = {
        "teacher_validation": teacher.get("validation", {}).get("passed")
        is True,
        "table_shape": table.shape == (TEACHER_TABLE_ROWS, ACTION_WIDTH),
        "table_finite": bool(np.all(np.isfinite(table))),
        "head_zero": bool(
            table.shape == (TEACHER_TABLE_ROWS, ACTION_WIDTH)
            and np.array_equal(
                table[:, HEAD_INDICES],
                np.zeros((TEACHER_TABLE_ROWS, len(HEAD_INDICES))),
            )
        ),
        "cadence": _type_exact_equal(adapter.get("cadence_hz"), 1.5),
        "advance": _type_exact_equal(
            adapter.get("phase_advance_bins_per_control"),
            TEACHER_PHASE_ADVANCE_BINS,
        ),
        "entry": _type_exact_equal(
            adapter.get("entry_phase_preincrement_bins"),
            TEACHER_ENTRY_PHASE_BINS,
        ),
        "v6_kind": authorization.get("artifact_kind")
        == (
            "openduckmini_h4_reverse_iteration_v6_"
            "absolute_full_leg_targets_authorization"
        ),
        "v6_contract": authorization.get("scope", {}).get("contract_id")
        == TRAINING_CONTRACT_ID,
        "v6_family": authorization.get("scope", {}).get(
            "selected_change_family"
        )
        == CORE_CONTRACT_ID,
        "teacher_timing": _type_exact_equal(
            authorization.get("teacher_timing_contract"),
            bundle.config.get("teacher_timing_contract"),
        ),
        "action_parameterization": _type_exact_equal(
            authorization.get("action_parameterization_contract"),
            bundle.config.get("action_parameterization_contract"),
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"reverse-v6 teacher timing authority failed: {checks}")
    return {
        "selected_path": selected_path,
        "authorization_path": authorization_path,
        "table": table,
        "cadence_hz": 1.5,
        "phase_advance_bins": TEACHER_PHASE_ADVANCE_BINS,
        "entry_phase_bins": TEACHER_ENTRY_PHASE_BINS,
        "checks": checks,
    }


def _reverse_v6_factory_wrapper(frozen_factory: Any) -> Any:
    """Return a wrapper that supplies the only authorized factory flags."""

    def factory(**kwargs: Any) -> type:
        forbidden = {
            "forward_v4_substep_contact",
            "forward_iteration_v6_contact_abort_island_only",
            "reverse_iteration_v6_absolute_full_leg_targets",
            "legacy_reward_config_overrides",
        }
        if forbidden & set(kwargs):
            raise ValueError("reverse-v6 factory adapter received competing opt-ins")
        return frozen_factory(
            **kwargs,
            legacy_reward_config_overrides={
                "target_imitation": 0.0,
                "contact_imitation": 0.0,
                "tracking_sigma": TRACKING_SIGMA,
            },
            forward_v4_substep_contact=False,
            forward_iteration_v6_contact_abort_island_only=False,
            reverse_iteration_v6_absolute_full_leg_targets=True,
        )

    return factory


def _validate_reverse_v6_environment(env: Any) -> None:
    scales = env._config.reward_config.scales
    checks = {
        "reverse_v6": env.h4_reverse_iteration_v6_absolute_full_leg_targets
        is True,
        "forward_v6": env.h4_forward_iteration_v6_contact_abort_island_only
        is False,
        "forward_v4": env.h4_forward_v4_substep_contact is False,
        "contract": env.h4_reverse_iteration_v6_contract_id == CORE_CONTRACT_ID,
        "compiled_assertion": env.h4_reverse_iteration_v6_compiled_assertion_bound
        is True,
        "residual_class": _float_is_positive_zero(
            env.h4_reverse_iteration_v6_residual_authority_scale
        ),
        "teacher_target_class": _float_is_positive_zero(
            env.h4_reverse_iteration_v6_teacher_target_contribution
        ),
        "runtime_residual": _float_is_positive_zero(
            env._backward_residual_scale
        ),
        "target_reward": _float_is_positive_zero(scales.target_imitation),
        "contact_reward": _float_is_positive_zero(scales.contact_imitation),
        "tracking_sigma": _type_exact_equal(
            env._config.reward_config.tracking_sigma, TRACKING_SIGMA
        ),
        "teacher_rows": env._h4_reverse_teacher_table.shape
        == (TEACHER_TABLE_ROWS, ACTION_WIDTH),
        "source_period": int(env.PRM.nb_steps_in_period)
        == SOURCE_PERIOD_BINS,
        "action_delay": int(env._config.noise_config.action_min_delay) == 0
        and int(env._config.noise_config.action_max_delay) == 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"reverse-v6 environment contract drifted: {checks}")


def _clone_function(
    function: FunctionType, *, global_overrides: Mapping[str, Any]
) -> FunctionType:
    if not isinstance(function, FunctionType) or function.__closure__ is not None:
        raise TypeError("adapter requires a closure-free Python function")
    globals_copy = dict(function.__globals__)
    globals_copy.update(global_overrides)
    clone = FunctionType(
        function.__code__,
        globals_copy,
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


def _module_contract_snapshot(
    base: ModuleType, post: ModuleType, core: ModuleType
) -> dict[str, Any]:
    return {
        "base_residual": base.H4_REVERSE_RESIDUAL_SCALE,
        "post_residual": post.H4_REVERSE_RESIDUAL_SCALE,
        "base_make": base._make_environment_and_policy,
        "base_compiled": base._compiled_rollout_for,
        "base_episode": base._run_episode,
        "base_run": base.run_evaluation,
        "base_control": base.rederive_h4_control_contract,
        "base_safety": base.rederive_h4_safety_acceptance,
        "base_validate": base.validate_h4_strict_artifact,
        "post_trace": post._h4_control_trace_arrays,
        "post_control": post.rederive_h4_control_contract,
        "post_central": post.rederive_central_safety_audit_from_control_trace,
        "post_safety": post.rederive_h4_safety_acceptance,
        "post_episode": post.validate_h4_strict_episode,
        "post_validate": post.validate_h4_strict_artifact,
        "core_factory": core.make_h4_aligned_environment_class,
        "core_decoder": core.reverse_iteration_v6_absolute_full_leg_targets,
        "core_margin": core.margin_clip_targets,
        "core_guard": core.final_target_guard_step,
    }


def _assert_module_contract_unchanged(
    base: ModuleType,
    post: ModuleType,
    core: ModuleType,
    expected: Mapping[str, Any],
) -> None:
    if _module_contract_snapshot(base, post, core) != dict(expected):
        raise RuntimeError("frozen evaluator/post/core module state changed")


def _load_frozen_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    if str(EXP_ROOT) not in sys.path:
        sys.path.insert(0, str(EXP_ROOT))
    import safe_gait_experts.h4_post_training as post
    import safe_gait_experts.h4_training_alignment as core

    spec = importlib.util.spec_from_file_location(
        "exp004_h4_reverse_v6_absolute_adapter_base_evaluator_v1",
        BASE_EVALUATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen evaluator: {BASE_EVALUATOR_PATH}")
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    return base, post, core


_ADAPTER_COMPILED_ROLLOUT_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}


def _compiled_reverse_v6_rollout_for(
    base: ModuleType, env: Any, policy: Any, stack: Mapping[str, Any]
) -> tuple[Any, Any]:
    """Build the frozen 300x10 rollout while retaining every v6 device field."""

    key = (id(env), id(policy))
    cached = _ADAPTER_COMPILED_ROLLOUT_CACHE.get(key)
    if cached is not None:
        return cached
    jax = stack["jax"]
    jp = stack["jp"]
    joystick = stack["joystick"]
    snapshot = base._snapshot_function(env, stack)
    joint_addresses = jp.asarray(env.get_actuator_joints_qpos_addr())

    def control_step(carry: tuple[Any, Any], _index: Any) -> tuple[Any, Any]:
        current_state, inference_key = carry
        control_entry_data = current_state.data
        actor_observation = current_state.obs["state"]
        previous_targets = current_state.data.ctrl
        guard_before = current_state.info["h4_guard_steps"]
        previous_contact = current_state.info["h4_previous_force_contact"]
        source_phase_before = current_state.info["imitation_i"]
        source_phase_after = jp.mod(
            source_phase_before + env._backward_phase_rate,
            env.PRM.nb_steps_in_period,
        )
        table_phase = jp.mod(
            source_phase_after * env._h4_reverse_teacher_phase_scale,
            env._h4_reverse_teacher_table.shape[0],
        )
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
            raise RuntimeError("reverse-v6 environment failed to restore globals")
        applied_targets = next_state.data.ctrl

        def direct_physics_step(
            physics_carry: tuple[Any, Any], _unused: Any
        ) -> tuple[Any, Any]:
            data, contact = physics_carry
            data = data.replace(ctrl=applied_targets)
            data = joystick.mjx_env.mjx.step(env.mjx_model, data)
            next_contact, row = snapshot(data, contact)
            return (data, next_contact), {
                **row,
                "_direct_applied_targets": data.ctrl,
                "_contact_before": contact,
            }

        (replay_final, _replay_contact), replay_with_private = jax.lax.scan(
            direct_physics_step,
            (control_entry_data, previous_contact),
            jp.arange(PHYSICS_SUBSTEPS_PER_CONTROL),
        )
        replay_applied_targets = replay_with_private[
            "_direct_applied_targets"
        ]
        endpoint_previous_contact = replay_with_private["_contact_before"][-1]
        physics_trace = {
            name: value
            for name, value in replay_with_private.items()
            if name not in {"_direct_applied_targets", "_contact_before"}
        }
        (
            dynamic6_exact,
            dynamic6_max_abs_error,
            dynamic6_field_count,
            dynamic6_all_finite,
        ) = _direct_physics_dynamic6_parity(
            next_state.data, replay_final, xp=jp
        )
        applied_target_exact = _strict_device_leaf_bitwise_equal(
            replay_applied_targets,
            jp.broadcast_to(applied_targets, replay_applied_targets.shape),
            xp=jp,
        )
        _actual_endpoint_contact, actual_endpoint = snapshot(
            next_state.data, endpoint_previous_contact
        )
        replay_endpoint = {
            name: physics_trace[name][-1]
            for name in SNAPSHOT_ENDPOINT_FIELDS
        }
        (
            snapshot_endpoint_exact,
            snapshot_endpoint_max_abs_error,
            snapshot_endpoint_field_count,
            snapshot_endpoint_all_finite,
        ) = _direct_physics_snapshot_parity(
            actual_endpoint, replay_endpoint, xp=jp
        )
        info = next_state.info
        control_trace = {
            "actor_observation": actor_observation,
            "raw_action": raw_action,
            "applied_action": applied_action,
            "preclip_targets": info["h4_pre_guard_raw_targets"],
            # The frozen central SafetyAudit field means the actual inward
            # margin stage.  The following v6-specific field separately
            # preserves the once-slew precomposer output, including the sole
            # authorized tick-zero left-knee startup value outside margin.
            "margin_clipped_targets": info[
                "h4_v6_reverse_decoder_margin_targets"
            ],
            "applied_targets": next_state.data.ctrl,
            "previous_targets": previous_targets,
            "joint_qpos": next_state.data.qpos[joint_addresses],
            "guard_calls": info["h4_guard_steps"] - guard_before,
            "done": next_state.done,
            "v6_teacher_source_phase_before": source_phase_before,
            "v6_teacher_table_phase": table_phase,
            "v6_decoder_action": info["h4_v6_reverse_decoder_action"],
            "v6_decoder_raw_targets": info[
                "h4_v6_reverse_decoder_raw_targets"
            ],
            "v6_decoder_margin_targets": info[
                "h4_v6_reverse_decoder_margin_targets"
            ],
            "v6_upstream_margin_targets": info[
                "h4_upstream_margin_targets"
            ],
            "v6_precomposer_targets": info["h4_guard_desired_targets"],
            "v6_decoder_exact": info["h4_v6_reverse_decoder_exact"],
            "v6_decoder_max_abs_error": info[
                "h4_v6_reverse_decoder_max_abs_error"
            ],
            "v6_decoder_leg_count": info["h4_v6_reverse_decoder_leg_count"],
            "v6_decoder_leg_count_exact": info[
                "h4_v6_reverse_decoder_leg_count_exact"
            ],
            "v6_decoder_head_zero_exact": info[
                "h4_v6_reverse_decoder_head_zero_exact"
            ],
            "v6_teacher_target_contribution_zero_exact": info[
                "h4_v6_reverse_teacher_target_contribution_zero_exact"
            ],
            "v6_residual_authority_scale": info[
                "h4_v6_reverse_residual_authority_scale"
            ],
            "v6_decoder_all_finite": info[
                "h4_v6_reverse_decoder_all_finite"
            ],
            "v6_decoder_margin_saturation_count": info[
                "h4_v6_reverse_decoder_margin_saturation_count"
            ],
            "v6_decoder_action_clip_count": info[
                "h4_v6_reverse_decoder_action_clip_count"
            ],
            "v6_decoder_guard_lag_max_rad": info[
                "h4_v6_reverse_decoder_guard_lag_max_rad"
            ],
            "v6_precomposer_call_count": info[
                "h4_v6_reverse_precomposer_call_count"
            ],
            "v6_precomposer_call_count_exact": info[
                "h4_v6_reverse_precomposer_call_count_exact"
            ],
            "v6_final_guard_call_count": info[
                "h4_v6_reverse_final_guard_call_count"
            ],
            "v6_final_guard_call_count_exact": info[
                "h4_v6_reverse_final_guard_call_count_exact"
            ],
            "v6_decoder_violation": info[
                "h4_v6_reverse_decoder_violation"
            ],
            "v6_decoder_assertion_token": info[
                "h4_v6_reverse_decoder_assertion_token"
            ],
            "v6_direct_physics_substep_count": jp.asarray(
                PHYSICS_SUBSTEPS_PER_CONTROL, dtype=jp.int32
            ),
            "v6_direct_physics_dynamic6_endpoint_exact": dynamic6_exact,
            "v6_direct_physics_dynamic6_endpoint_max_abs_error": (
                dynamic6_max_abs_error
            ),
            "v6_direct_physics_dynamic6_field_count": jp.asarray(
                dynamic6_field_count, dtype=jp.int32
            ),
            "v6_direct_physics_dynamic6_all_finite": dynamic6_all_finite,
            "v6_direct_physics_applied_target_exact": applied_target_exact,
            "v6_direct_physics_snapshot_endpoint_exact": (
                snapshot_endpoint_exact
            ),
            "v6_direct_physics_snapshot_endpoint_max_abs_error": (
                snapshot_endpoint_max_abs_error
            ),
            "v6_direct_physics_snapshot_endpoint_field_count": jp.asarray(
                snapshot_endpoint_field_count, dtype=jp.int32
            ),
            "v6_direct_physics_snapshot_endpoint_all_finite": (
                snapshot_endpoint_all_finite
            ),
        }
        return (next_state, inference_key), {
            "physics": physics_trace,
            "control": control_trace,
        }

    def complete_rollout(initial_state: Any, key_value: Any) -> tuple[Any, Any]:
        return jax.lax.scan(
            control_step,
            (initial_state, key_value),
            jp.arange(CONTROL_TICKS),
        )

    result = (snapshot, jax.jit(complete_rollout))
    _ADAPTER_COMPILED_ROLLOUT_CACHE[key] = result
    return result


def _run_reverse_v6_episode(
    *,
    base: ModuleType,
    central_safety_rederive: FunctionType,
    safety_rederive: FunctionType,
    env: Any,
    policy: Any,
    params: Any,
    stack: Mapping[str, Any],
    seed: int,
    expert: str,
) -> dict[str, Any]:
    if expert != "reverse" or seed not in STRICT_SEEDS:
        raise ValueError("reverse-v6 adapter permits only the fixed reverse seeds")
    jax = stack["jax"]
    jp = stack["jp"]
    joystick = stack["joystick"]
    command = STRICT_COMMAND
    state = env.reset(jax.random.PRNGKey(seed))
    state.reward.block_until_ready()
    observation = np.asarray(state.obs["state"], dtype=np.float32)
    if observation.shape != (base.H4_ACTOR_OBSERVATION_WIDTH,) or not np.all(
        np.isfinite(observation)
    ):
        raise RuntimeError("reverse-v6 reset actor observation is invalid")
    initial_targets_source = np.asarray(state.data.ctrl)
    expected_source = SAFE_INIT_F32.astype(initial_targets_source.dtype)
    reset_error = np.abs(
        initial_targets_source.astype(np.float64)
        - expected_source.astype(np.float64)
    )
    reset_audit = {
        "comparison_semantics": "SOURCE_DTYPE_FLOAT32_EXACT",
        "exact_safe_init": bool(
            np.array_equal(initial_targets_source, expected_source)
        ),
        "maximum_safe_init_error_rad": float(np.max(reset_error)),
        "head_qpos_peak_rad": float(
            np.max(np.abs(initial_targets_source[HEAD_INDICES]))
        ),
    }
    control_audit = base.SafetyAudit(base.ACTUATOR_JOINT_ORDER)
    physics_audit = base.PhysicsSubstepAudit(base.ACTUATOR_JOINT_ORDER)
    gait = base.GaitQualityAccumulator(joint_names=base.ACTUATOR_JOINT_ORDER)
    snapshot, compiled_rollout = _compiled_reverse_v6_rollout_for(
        base, env, policy, stack
    )
    _contact, initial_snapshot = snapshot(state.data, jp.zeros(2, dtype=bool))
    initial_host = {
        name: np.asarray(value)
        for name, value in jax.device_get(initial_snapshot).items()
    }
    gait.update(base._gait_sample(initial_host, time_s=0.0, command=command))

    source_physics_step = joystick.mjx_env.step
    source_motor_speed_limits = joystick.USE_MOTOR_SPEED_LIMITS
    try:
        (state, _final_key), device_trace = compiled_rollout(
            state, jax.random.PRNGKey(seed ^ 0x4844)
        )
        state.reward.block_until_ready()
    finally:
        if (
            joystick.mjx_env.step is not source_physics_step
            or joystick.USE_MOTOR_SPEED_LIMITS is not source_motor_speed_limits
        ):
            raise RuntimeError(
                "reverse-v6 evaluation changed frozen joystick module state"
            )
    host_trace = jax.device_get(device_trace)
    physics_trace = {
        name: np.asarray(value).reshape(
            (PHYSICS_SUBSTEPS,) + np.asarray(value).shape[2:]
        )
        for name, value in host_trace["physics"].items()
    }
    control_trace = {
        name: np.asarray(value) for name, value in host_trace["control"].items()
    }
    if any(value.shape[0] != PHYSICS_SUBSTEPS for value in physics_trace.values()):
        raise RuntimeError("reverse-v6 rollout must expose exactly 3,000 physics rows")
    if any(value.shape[0] != CONTROL_TICKS for value in control_trace.values()):
        raise RuntimeError("reverse-v6 rollout must expose exactly 300 control rows")

    fell = False
    for substep_index in range(PHYSICS_SUBSTEPS):
        row = {
            name: value[substep_index] for name, value in physics_trace.items()
        }
        physics_audit.update(
            joint_qpos=row["joint_qpos"],
            full_qpos=row["full_qpos"],
            full_qvel=row["full_qvel"],
            height_m=float(row["height"]),
            upright=float(row["upright"]),
            feet_contacts=row["contacts"],
        )
        gait.update(
            base._gait_sample(
                row,
                time_s=(substep_index + 1) * PHYSICS_DT_S,
                command=command,
            )
        )
        fell = bool(fell or physics_audit.termination_required)

    for index in range(CONTROL_TICKS):
        control_audit.update(
            raw_policy_action=control_trace["raw_action"][index],
            applied_action=control_trace["applied_action"][index],
            preclip_targets=control_trace["preclip_targets"][index],
            margin_clipped_targets=control_trace["margin_clipped_targets"][index],
            applied_targets=control_trace["applied_targets"][index],
            previous_applied_targets=control_trace["previous_targets"][index],
            joint_qpos=control_trace["joint_qpos"][index],
            control_dt=CONTROL_DT_S,
        )
    actor_observations = control_trace["actor_observation"]
    raw_actions = control_trace["raw_action"]
    applied_actions = control_trace["applied_action"]
    guard_calls = control_trace["guard_calls"]
    nonfinite_observation_count = int(
        np.count_nonzero(~np.all(np.isfinite(actor_observations), axis=1))
    )
    nonfinite_action_count = int(
        np.count_nonzero(~np.all(np.isfinite(raw_actions), axis=1))
    )
    if (
        actor_observations.shape
        != (CONTROL_TICKS, base.H4_ACTOR_OBSERVATION_WIDTH)
        or raw_actions.shape != (CONTROL_TICKS, ACTION_WIDTH)
        or nonfinite_observation_count
        or nonfinite_action_count
    ):
        raise RuntimeError("reverse-v6 actor observations/actions are invalid")
    parity = base.compare_policy_outputs(
        base.infer_h4_action_numpy(params, actor_observations[0]),
        raw_actions[0],
    )
    if not parity["passed"]:
        raise RuntimeError(f"reverse-v6 NumPy/Brax actor parity failed: {parity}")
    guard_violations = int(np.count_nonzero(guard_calls != 1))
    fell = bool(fell or np.any(control_trace["done"]))
    gait_metrics = gait.finalize()
    gait_payload = {**gait_metrics.as_dict(), "measurement_complete": True}
    gait_result = base.gait_quality_acceptance(gait_metrics).as_dict()
    safety_payload = control_audit.to_dict()
    physics_payload = physics_audit.to_dict()
    serialized_control_trace = {
        "source_dtype": str(raw_actions.dtype),
        "initial_applied_targets": initial_targets_source,
        **{
            name: control_trace[name]
            for name in (
                "raw_action",
                "applied_action",
                "preclip_targets",
                "margin_clipped_targets",
                "applied_targets",
                "previous_targets",
                "joint_qpos",
                *sorted(V6_VECTOR_TRACE_KEYS),
                *sorted(V6_FLOAT_TRACE_KEYS),
                *sorted(V6_INTEGER_TRACE_KEYS),
                *sorted(V6_BOOLEAN_TRACE_KEYS),
            )
        },
    }
    episode: dict[str, Any] = {
        "seed": seed,
        "segment_id": f"h4_reverse_seed{seed}_6s",
        "expert": "reverse",
        "physical_command_mps_radps": list(command),
        "source_segment_kind": "H4_STRICT_6S",
        "completed": True,
        "fell": fell,
        "duration_s": STRICT_DURATION_S,
        "physics_timestep_s": PHYSICS_DT_S,
        "completed_control_ticks": CONTROL_TICKS,
        "completed_physics_substeps": PHYSICS_SUBSTEPS,
        "reset_audit": reset_audit,
        "control_trace": serialized_control_trace,
        "reverse_v6_absolute_decoder_contract": (
            _expected_reverse_v6_episode_contract()
        ),
        "safety_audit": safety_payload,
        "physics_substep_audit": physics_payload,
        "guard_call_audit": {
            "control_tick_count": CONTROL_TICKS,
            "total_guard_calls": int(np.sum(guard_calls)),
            "guard_call_violation_count": guard_violations,
            "maximum_guard_calls_per_tick": int(np.max(guard_calls)),
        },
        "policy_inference_audit": {
            "input_width": base.H4_ACTOR_OBSERVATION_WIDTH,
            "output_width": ACTION_WIDTH,
            "inference_count": CONTROL_TICKS,
            "nonfinite_observation_count": nonfinite_observation_count,
            "nonfinite_action_count": nonfinite_action_count,
            "post_mask_nonzero_head_count": int(
                np.count_nonzero(applied_actions[:, HEAD_INDICES])
            ),
            "maximum_raw_action_magnitude": float(np.max(np.abs(raw_actions))),
            "first_tick_numpy_brax_parity": [parity],
        },
        "gait_quality_metrics": gait_payload,
        "gait_quality_acceptance": gait_result,
        "metrics": base.legacy_metrics_from_gait_quality(gait_payload),
    }
    episode["h4_control_contract"] = rederive_reverse_v6_control_contract(
        episode
    )
    if central_safety_rederive(episode) != safety_payload:
        raise RuntimeError("central SafetyAudit v6 control-trace rederivation drifted")
    episode["h4_safety_acceptance"] = safety_rederive(episode)
    episode["safety"] = {
        "fall_count": int(fell),
        "qpos_violation_samples": int(
            safety_payload["qpos_limit_violations"]
            + physics_payload["qpos_limit_violations"]
        ),
        "target_violation_samples": int(
            safety_payload["applied_target_limit_violations"]
            + safety_payload["desired_target_margin_violations"]
        ),
        "slew_violation_samples": int(safety_payload["target_slew_violations"]),
        "guard_call_violation_samples": guard_violations,
        "nonfinite_samples": int(
            safety_payload["nonfinite_sample_count"]
            + physics_payload["nonfinite_state_samples"]
        ),
    }
    episode["strict_passed"] = bool(
        episode["h4_safety_acceptance"]["passed"] and gait_result["passed"]
    )
    return episode


def _real_evaluation_contract_checks(contract: Mapping[str, Any]) -> dict[str, bool]:
    expected = {
        "fixed_seeds": list(STRICT_SEEDS),
        "physical_command_mps_radps": list(STRICT_COMMAND),
        "duration_s": STRICT_DURATION_S,
        "control_timestep_s": CONTROL_DT_S,
        "physics_timestep_s": PHYSICS_DT_S,
        "control_tick_count": CONTROL_TICKS,
        "physics_substep_count": PHYSICS_SUBSTEPS,
        "gait_sample_count": GAIT_SAMPLES,
        "gait_quality_semantics": (
            "FULL_CURRENT_P0_RECOMPUTED_FROM_N_PLUS_ONE_SUBSTEP_TRACE"
        ),
        "reset": "EXACT_SAFE_INIT_NO_RESET_NOISE",
        "policy_inference": "BRAX_DETERMINISTIC_NORMAL_TANH_ACTOR116",
        "reverse_action_parameterization": REAL_REVERSE_EVALUATION_SEMANTICS,
        "reverse_teacher_role": "PHASE_TIMING_PRIOR_ONLY",
        "legacy_teacher_plus_residual_runtime_authority": False,
    }
    return {
        "full_schema_type_and_value_exact": _type_exact_equal(
            dict(contract), expected
        ),
        "old_field_absent": "reverse_composition" not in contract,
    }


FINAL_ARTIFACT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "hardware_deployment",
        "promotion_allowed",
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
    }
)


def _reverse_v6_top_level_contract_exact(payload: Mapping[str, Any]) -> bool:
    return bool(
        set(payload) == FINAL_ARTIFACT_TOP_LEVEL_KEYS
        and _type_exact_equal(payload.get("schema_version"), 1)
        and payload.get("artifact_kind") == DIAGNOSTIC_ARTIFACT_KIND
        and payload.get("hardware_deployment") == "PROHIBITED"
        and payload.get("execution_provider") == "CPU"
        and payload.get("promotion_allowed") is False
        and payload.get("adoption_allowed") is False
        and payload.get("release_allowed") is False
        and payload.get("standalone_direct_runtime_allowed") is False
        and type(payload.get("created_at_utc")) is str
        and bool(payload.get("created_at_utc"))
    )


def _adapter_provenance_record_exact(
    record: Any,
    *,
    current_evaluation_hashes: Mapping[str, str],
    current_candidate_hashes: Mapping[str, str],
    current_candidate_evaluation_bindings: Mapping[str, str],
) -> bool:
    if not isinstance(record, Mapping):
        return False
    expected_keys = {
        "schema_version",
        "contract_id",
        "training_contract_id",
        "method",
        "runtime_factory",
        "legacy_reward_config",
        "backward_residual_scale",
        "host_rederivation",
        "physics_trace_semantics",
        "physics_substeps_per_control",
        "dynamic6_endpoint_bitwise_exact_required",
        "snapshot_endpoint_bitwise_exact_required",
        "legacy_float64_control_audit_diagnostic_only",
        "legacy_float64_control_audit_counts_exact_required",
        "adapter_source",
        "adapter_authorization",
        "frozen_source_hashes_pre",
        "frozen_source_hashes_post",
        "candidate_bundle_hashes_pre",
        "candidate_bundle_hashes_post",
        "candidate_bundle_hashes_current",
        "original_module_globals_and_function_references_unchanged",
        "promotion_evidence_allowed",
        "candidate_adoption_allowed",
        "release_allowed",
        "hardware_deployment",
    }
    if set(record) != expected_keys:
        return False
    expected_static = {
        "schema_version": 1,
        "contract_id": ADAPTER_CONTRACT_ID,
        "training_contract_id": TRAINING_CONTRACT_ID,
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
            "tracking_sigma": TRACKING_SIGMA,
        },
        "backward_residual_scale": 0.0,
        "host_rederivation": CONTROL_TRACE_SEMANTICS,
        "physics_trace_semantics": (
            "DIRECT_MJX_STEP_REPLAY_FROM_CONTROL_ENTRY_WITH_ACTUAL_APPLIED_TARGETS"
        ),
        "physics_substeps_per_control": PHYSICS_SUBSTEPS_PER_CONTROL,
        "dynamic6_endpoint_bitwise_exact_required": True,
        "snapshot_endpoint_bitwise_exact_required": True,
        "legacy_float64_control_audit_diagnostic_only": True,
        "legacy_float64_control_audit_counts_exact_required": True,
        "original_module_globals_and_function_references_unchanged": True,
        "promotion_evidence_allowed": False,
        "candidate_adoption_allowed": False,
        "release_allowed": False,
        "hardware_deployment": "PROHIBITED",
    }
    if any(
        not _type_exact_equal(record.get(key), value)
        for key, value in expected_static.items()
    ):
        return False
    for key, path in (
        ("adapter_source", ADAPTER_SOURCE_KEY),
        ("adapter_authorization", ADAPTER_AUTHORIZATION_SOURCE_KEY),
    ):
        source = record.get(key)
        if (
            not isinstance(source, Mapping)
            or set(source)
            != {
                "path",
                "sha256_pre",
                "sha256_post",
                "sha256_current",
                "unchanged",
            }
            or source.get("path") != path
            or source.get("unchanged") is not True
            or source.get("sha256_pre") != source.get("sha256_post")
            or source.get("sha256_post") != source.get("sha256_current")
            or source.get("sha256_current")
            != current_evaluation_hashes.get(path)
        ):
            return False
        try:
            _require_sha256(source.get("sha256_pre"), f"{key} SHA256")
        except ValueError:
            return False
    live_adapter_hashes = _adapter_source_hashes()
    if (
        record["adapter_source"].get("sha256_current")
        != live_adapter_hashes[ADAPTER_SOURCE_KEY]
        or record["adapter_authorization"].get("sha256_current")
        != live_adapter_hashes[ADAPTER_AUTHORIZATION_SOURCE_KEY]
        or live_adapter_hashes[ADAPTER_AUTHORIZATION_SOURCE_KEY]
        != PINNED_ADAPTER_AUTHORIZATION_SHA256
    ):
        return False
    frozen_pre = record.get("frozen_source_hashes_pre")
    frozen_post = record.get("frozen_source_hashes_post")
    expected_frozen = {
        name: digest for name, (_path, digest) in PINNED_FROZEN_SOURCES.items()
    }
    try:
        live_frozen = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    except ValueError:
        return False
    expected_frozen_evaluation = _pinned_evaluation_source_bindings()
    if not _type_exact_equal(frozen_pre, expected_frozen) or not _type_exact_equal(
        frozen_post, expected_frozen
    ) or not _type_exact_equal(live_frozen, expected_frozen) or any(
        current_evaluation_hashes.get(path) != digest
        for path, digest in expected_frozen_evaluation.items()
    ):
        return False
    candidate_pre = record.get("candidate_bundle_hashes_pre")
    candidate_post = record.get("candidate_bundle_hashes_post")
    candidate_current = record.get("candidate_bundle_hashes_current")
    expected_candidate_keys = {
        "candidate_params",
        "candidate_manifest",
        "candidate_config",
        "candidate_result",
        "candidate_training_curve",
    }
    return bool(
        isinstance(candidate_pre, Mapping)
        and isinstance(candidate_post, Mapping)
        and isinstance(candidate_current, Mapping)
        and set(candidate_pre) == expected_candidate_keys
        and set(candidate_post) == expected_candidate_keys
        and set(candidate_current) == expected_candidate_keys
        and _type_exact_equal(candidate_pre, candidate_post)
        and _type_exact_equal(candidate_post, candidate_current)
        and _type_exact_equal(candidate_current, current_candidate_hashes)
        and isinstance(current_candidate_evaluation_bindings, Mapping)
        and len(current_candidate_evaluation_bindings)
        == len(current_candidate_hashes)
        == len(expected_candidate_keys)
        and sorted(current_candidate_evaluation_bindings.values())
        == sorted(current_candidate_hashes.values())
        and all(
            current_evaluation_hashes.get(path) == digest
            for path, digest in current_candidate_evaluation_bindings.items()
        )
        and all(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            for value in candidate_pre.values()
        )
    )


def _adapter_provenance_stage_exact(
    provenance: Mapping[str, Any],
    *,
    current_evaluation_hashes: Mapping[str, str] | None,
    current_candidate_hashes: Mapping[str, str] | None,
    current_candidate_evaluation_bindings: Mapping[str, str] | None,
    allow_initial_record_absent: bool,
) -> bool:
    source_pre = provenance.get("evaluation_source_hashes_pre")
    source_post = provenance.get("evaluation_source_hashes_post")
    source_current = provenance.get("evaluation_source_hashes_current")
    if (
        not isinstance(source_pre, Mapping)
        or not isinstance(source_post, Mapping)
        or not isinstance(source_current, Mapping)
        or not _type_exact_equal(dict(source_pre), dict(source_post))
        or not _type_exact_equal(dict(source_post), dict(source_current))
    ):
        return False
    evaluation = dict(source_current)
    if current_evaluation_hashes is not None and not _type_exact_equal(
        dict(current_evaluation_hashes), evaluation
    ):
        return False
    adapter_keys = {ADAPTER_SOURCE_KEY, ADAPTER_AUTHORIZATION_SOURCE_KEY}
    present_by_stage = [
        adapter_keys & set(mapping)
        for mapping in (source_pre, source_post, source_current)
    ]
    record_present = (
        "reverse_v6_absolute_targets_evaluator_adapter" in provenance
    )
    if all(not present for present in present_by_stage):
        return allow_initial_record_absent and not record_present
    if (
        any(present != adapter_keys for present in present_by_stage)
        or not record_present
        or current_candidate_hashes is None
        or current_candidate_evaluation_bindings is None
    ):
        return False
    return _adapter_provenance_record_exact(
        provenance.get("reverse_v6_absolute_targets_evaluator_adapter"),
        current_evaluation_hashes=evaluation,
        current_candidate_hashes=current_candidate_hashes,
        current_candidate_evaluation_bindings=(
            current_candidate_evaluation_bindings
        ),
    )


def _mapping_diff_paths(
    before: Any, after: Any, *, prefix: tuple[str, ...] = ()
) -> set[tuple[str, ...]]:
    """Return the exact structural/value diff without JSON scalar coercion."""

    if isinstance(before, Mapping) and isinstance(after, Mapping):
        result: set[tuple[str, ...]] = set()
        for key in set(before) | set(after):
            path = (*prefix, str(key))
            if key not in before or key not in after:
                result.add(path)
            else:
                result.update(
                    _mapping_diff_paths(before[key], after[key], prefix=path)
                )
        return result
    return set() if _type_exact_equal(before, after) else {prefix}


def _real_v6_artifact_from_frozen_skeleton(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct one real-v6 artifact copy from the pinned frozen skeleton.

    This is a construction adapter, not a validator repair path.  Every legacy
    input field that is replaced is checked first, only the enumerated paths may
    differ, and the caller-owned skeleton is never mutated.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("reverse-v6 artifact skeleton must be a mapping")
    before = copy.deepcopy(dict(payload))
    if (
        before.get("artifact_kind") != DIAGNOSTIC_ARTIFACT_KIND
        or before.get("hardware_deployment") != "PROHIBITED"
        or before.get("adoption_allowed") is not False
        or before.get("release_allowed") is not False
        or "promotion_allowed" in before
    ):
        raise ValueError("reverse-v6 frozen artifact decision skeleton drifted")
    contract = before.get("evaluation_contract")
    provenance = before.get("runtime_provenance")
    expected_legacy_contract = {
        "fixed_seeds": list(STRICT_SEEDS),
        "physical_command_mps_radps": list(STRICT_COMMAND),
        "duration_s": STRICT_DURATION_S,
        "control_timestep_s": CONTROL_DT_S,
        "physics_timestep_s": PHYSICS_DT_S,
        "control_tick_count": CONTROL_TICKS,
        "physics_substep_count": PHYSICS_SUBSTEPS,
        "gait_sample_count": GAIT_SAMPLES,
        "gait_quality_semantics": (
            "FULL_CURRENT_P0_RECOMPUTED_FROM_N_PLUS_ONE_SUBSTEP_TRACE"
        ),
        "reset": "EXACT_SAFE_INIT_NO_RESET_NOISE",
        "policy_inference": "BRAX_DETERMINISTIC_NORMAL_TANH_ACTOR116",
        "reverse_composition": (
            "PINNED_PERSISTENT_TEACHER_PLUS_TRAINABLE_RESIDUAL"
        ),
    }
    if not isinstance(contract, Mapping) or not _type_exact_equal(
        dict(contract), expected_legacy_contract
    ):
        raise ValueError("reverse-v6 frozen evaluation-contract skeleton drifted")
    if not isinstance(provenance, Mapping):
        raise ValueError("reverse-v6 frozen provenance skeleton is incomplete")
    checks = provenance.get("reverse_composition_checks")
    source_pre = provenance.get("evaluation_source_hashes_pre")
    source_post = provenance.get("evaluation_source_hashes_post")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != V6_ENVIRONMENT_CHECK_KEYS
        or not all(value is True for value in checks.values())
        or not isinstance(source_pre, Mapping)
        or not isinstance(source_post, Mapping)
        or not _type_exact_equal(dict(source_pre), dict(source_post))
        or "evaluation_source_hashes_current" in provenance
        or "reverse_v6_absolute_decoder_environment_checks" in provenance
        or "historical_teacher_residual_training_sources_runtime_authority"
        in provenance
        or "reverse_v6_absolute_targets_evaluator_adapter" in provenance
    ):
        raise ValueError("reverse-v6 frozen provenance skeleton drifted")

    artifact = copy.deepcopy(before)
    artifact["promotion_allowed"] = False
    real_contract = artifact["evaluation_contract"]
    del real_contract["reverse_composition"]
    real_contract["reverse_action_parameterization"] = (
        REAL_REVERSE_EVALUATION_SEMANTICS
    )
    real_contract["reverse_teacher_role"] = "PHASE_TIMING_PRIOR_ONLY"
    real_contract["legacy_teacher_plus_residual_runtime_authority"] = False
    real_provenance = artifact["runtime_provenance"]
    del real_provenance["reverse_composition_checks"]
    real_provenance["reverse_v6_absolute_decoder_environment_checks"] = dict(
        checks
    )
    real_provenance[
        "historical_teacher_residual_training_sources_runtime_authority"
    ] = False
    real_provenance["evaluation_source_hashes_current"] = dict(source_post)

    allowed_differences = {
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
    differences = _mapping_diff_paths(before, artifact)
    if differences != allowed_differences or not _type_exact_equal(
        dict(payload), before
    ):
        raise RuntimeError(
            "reverse-v6 skeleton conversion changed a non-authorized path: "
            f"{sorted(differences ^ allowed_differences)}"
        )
    return artifact


def _compatibility_validation_view(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Make a private view for unchanged generic structure/provenance checks."""

    view = copy.deepcopy(dict(artifact))
    contract = view["evaluation_contract"]
    contract["reverse_composition"] = (
        "PINNED_PERSISTENT_TEACHER_PLUS_TRAINABLE_RESIDUAL"
    )
    provenance = view["runtime_provenance"]
    checks = provenance["reverse_v6_absolute_decoder_environment_checks"]
    provenance["reverse_composition_checks"] = dict(checks)
    return view


def validate_reverse_v6_strict_artifact(
    payload: Mapping[str, Any],
    *,
    compatibility_validator: FunctionType,
    bundle: Any = None,
    current_central_hashes: Mapping[str, str] | None = None,
    current_evaluation_hashes: Mapping[str, str] | None = None,
    require_all_three_pass: bool = False,
    gait_quality_rederive: Any = None,
    _allow_initial_adapter_record_absent: bool = False,
) -> dict[str, Any]:
    """Validate the real v6 artifact, then reuse frozen generic checks privately."""

    if not _reverse_v6_top_level_contract_exact(payload):
        raise ValueError("reverse-v6 artifact CPU/decision contract drifted")
    candidate = payload.get("candidate")
    contract = payload.get("evaluation_contract")
    provenance = payload.get("runtime_provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (candidate, contract, provenance)
    ):
        raise ValueError("reverse-v6 artifact mapping closure is incomplete")
    contract_checks = _real_evaluation_contract_checks(contract)
    expected_candidate = bundle.candidate_record() if bundle is not None else None
    candidate_checks = {
        "expert": candidate.get("expert") == "reverse",
        "status": candidate.get("status") == "COMPLETED",
        "activity": candidate.get("activity") == "PPO_PILOT_TRAINING",
        "bundle_type_and_value_exact": bundle is None
        or _type_exact_equal(dict(candidate), expected_candidate),
    }
    sources = provenance.get("source_and_teacher_hashes")
    v6_source = (
        sources.get("reverse_iteration_v6_authorization", {})
        if isinstance(sources, Mapping)
        else {}
    )
    teacher_source = (
        sources.get("selected_reverse_teacher", {})
        if isinstance(sources, Mapping)
        else {}
    )
    environment_checks = provenance.get(
        "reverse_v6_absolute_decoder_environment_checks"
    )
    payload_evaluation_current = provenance.get(
        "evaluation_source_hashes_current"
    )
    augmented_evaluation_context = bool(
        isinstance(payload_evaluation_current, Mapping)
        and ADAPTER_SOURCE_KEY in payload_evaluation_current
        and ADAPTER_AUTHORIZATION_SOURCE_KEY in payload_evaluation_current
    )
    current_candidate_hashes = (
        _candidate_file_snapshot(bundle)
        if augmented_evaluation_context and bundle is not None
        else None
    )
    current_candidate_evaluation_bindings = (
        _candidate_evaluation_source_bindings(bundle, current_candidate_hashes)
        if current_candidate_hashes is not None and bundle is not None
        else None
    )
    central_payload = payload.get("central_hashes")
    central_provenance = provenance.get("central_hashes")
    provenance_checks = {
        "v6_authorization": _type_exact_equal(
            v6_source,
            {
                "path": str(REVERSE_V6_AUTHORIZATION_PATH.resolve()),
                "sha256": PINNED_REVERSE_V6_AUTHORIZATION_SHA256,
            },
        ),
        "teacher": _type_exact_equal(
            teacher_source,
            {
                "path": str(SELECTED_REVERSE_TEACHER_PATH.resolve()),
                "sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
            },
        ),
        "environment": isinstance(environment_checks, Mapping)
        and set(environment_checks) == V6_ENVIRONMENT_CHECK_KEYS
        and all(value is True for value in environment_checks.values()),
        "historical_not_runtime": provenance.get(
            "historical_teacher_residual_training_sources_runtime_authority"
        )
        is False,
        "no_real_legacy_checks": "reverse_composition_checks" not in provenance,
        "bundle_sources": bundle is None
        or (
            isinstance(sources, Mapping)
            and _type_exact_equal(dict(sources), dict(bundle.source_hashes))
        ),
        "central_hashes_live_and_exact": _central_hash_bindings_live_and_exact(
            central_payload,
            central_provenance,
            current_central_hashes,
        ),
        "adapter_record_stage_and_binding_exact": _adapter_provenance_stage_exact(
            provenance,
            current_evaluation_hashes=current_evaluation_hashes,
            current_candidate_hashes=current_candidate_hashes,
            current_candidate_evaluation_bindings=(
                current_candidate_evaluation_bindings
            ),
            allow_initial_record_absent=(
                _allow_initial_adapter_record_absent is True
            ),
        ),
    }
    if not all(
        (*contract_checks.values(), *candidate_checks.values(), *provenance_checks.values())
    ):
        raise ValueError(
            "reverse-v6 real artifact contract failed: "
            f"contract={contract_checks}, candidate={candidate_checks}, "
            f"provenance={provenance_checks}"
        )
    episodes = payload.get("episodes")
    baseline_record = payload.get("official_v22_baseline")
    baseline = (
        baseline_record.get("episodes")
        if isinstance(baseline_record, Mapping)
        else None
    )
    if (
        not isinstance(episodes, list)
        or len(episodes) != 3
        or not isinstance(baseline, list)
        or len(baseline) != 3
    ):
        raise ValueError("reverse-v6 strict artifact requires exact candidate/baseline 3x6s")
    for index, episode in enumerate([*episodes, *baseline]):
        if (
            not isinstance(episode, Mapping)
            or "reverse_composition_contract" in episode
            or not _type_exact_equal(
                episode.get("reverse_v6_absolute_decoder_contract"),
                _expected_reverse_v6_episode_contract(),
            )
        ):
            raise ValueError(f"reverse-v6 real episode contract drifted: {index}")
        if not _recorded_reverse_v6_control_contract_exact(episode):
            raise ValueError(
                f"reverse-v6 real control audit type/value drifted: {index}"
            )
        trace = episode.get("control_trace")
        if isinstance(trace, Mapping) and any(
            isinstance(name, str) and name.startswith("reverse_") for name in trace
        ):
            raise ValueError(f"legacy reverse target trace leaked into episode {index}")

    compatibility = _compatibility_validation_view(payload)
    audit = compatibility_validator(
        compatibility,
        bundle=bundle,
        current_central_hashes=current_central_hashes,
        current_evaluation_hashes=current_evaluation_hashes,
        require_all_three_pass=require_all_three_pass,
        gait_quality_rederive=gait_quality_rederive,
    )
    return {
        **audit,
        "reverse_v6_real_contract_checks": {
            **contract_checks,
            **candidate_checks,
            **provenance_checks,
        },
    }


def build_reverse_v6_call_graph(
    base: ModuleType, post: ModuleType, core: ModuleType
) -> dict[str, Any]:
    original = _module_contract_snapshot(base, post, core)
    identity = {
        "base_residual_legacy": original["base_residual"] == 0.12,
        "post_residual_legacy": original["post_residual"] == 0.12,
        "base_control_is_post": original["base_control"] is original["post_control"],
        "base_safety_is_post": original["base_safety"] is original["post_safety"],
        "base_validate_is_post": original["base_validate"] is original["post_validate"],
        "base_factory_is_core": base.make_h4_aligned_environment_class
        is original["core_factory"],
    }
    if not all(identity.values()):
        raise RuntimeError(f"frozen evaluator import identity drifted: {identity}")

    central_safety = _clone_function(
        post.rederive_central_safety_audit_from_control_trace,
        global_overrides={"_h4_control_trace_arrays": _v6_control_trace_arrays},
    )
    safety = _clone_function(
        post.rederive_h4_safety_acceptance,
        global_overrides={
            "rederive_h4_control_contract": rederive_reverse_v6_control_contract
        },
    )
    episode_validator = _clone_function(
        post.validate_h4_strict_episode,
        global_overrides={
            "rederive_h4_control_contract": rederive_reverse_v6_control_contract,
            "rederive_central_safety_audit_from_control_trace": central_safety,
            "rederive_h4_safety_acceptance": safety,
        },
    )
    compatibility_validator = _clone_function(
        post.validate_h4_strict_artifact,
        global_overrides={
            "STRICT_ARTIFACT_KIND": DIAGNOSTIC_ARTIFACT_KIND,
            "validate_h4_strict_episode": episode_validator,
        },
    )

    def trusted_bundle(**kwargs: Any) -> Any:
        bundle = post.validate_trusted_h4_bundle(**kwargs)
        _validate_reverse_v6_bundle(bundle)
        return bundle

    factory = _reverse_v6_factory_wrapper(core.make_h4_aligned_environment_class)
    make_clone = _clone_function(
        base._make_environment_and_policy,
        global_overrides={
            "H4_REVERSE_RESIDUAL_SCALE": 0.0,
            "_load_reverse_teacher": _load_reverse_v6_teacher,
            "make_h4_aligned_environment_class": factory,
        },
    )

    def make_environment(**kwargs: Any) -> Any:
        result = make_clone(**kwargs)
        env, policy, stack, trainer, composition, source_paths = result
        _validate_reverse_v6_environment(env)
        if "reverse_composition_authorization" not in source_paths:
            raise RuntimeError("v6 authorization source path was not surfaced")
        source_paths = dict(source_paths)
        source_paths["reverse_iteration_v6_authorization"] = source_paths.pop(
            "reverse_composition_authorization"
        )
        composition = dict(composition)
        composition["checks"] = {
            **dict(composition.get("checks", {})),
            "factory_reverse_v6_exact": True,
            "factory_forward_flags_false": True,
            "legacy_reward_target_zero": True,
            "legacy_reward_contact_zero": True,
            "legacy_tracking_sigma_0p01": True,
            "residual_authority_zero": True,
        }
        return env, policy, stack, trainer, composition, source_paths

    def run_episode(**kwargs: Any) -> dict[str, Any]:
        return _run_reverse_v6_episode(
            base=base,
            central_safety_rederive=central_safety,
            safety_rederive=safety,
            **kwargs,
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
        return validate_reverse_v6_strict_artifact(
            payload,
            compatibility_validator=compatibility_validator,
            bundle=bundle,
            current_central_hashes=current_central_hashes,
            current_evaluation_hashes=current_evaluation_hashes,
            require_all_three_pass=require_all_three_pass,
            gait_quality_rederive=gait_quality_rederive,
            _allow_initial_adapter_record_absent=False,
        )

    initial_validation_pending = True

    def initial_artifact_validator(
        payload: Mapping[str, Any],
        *,
        bundle: Any = None,
        current_central_hashes: Mapping[str, str] | None = None,
        current_evaluation_hashes: Mapping[str, str] | None = None,
        require_all_three_pass: bool = False,
        gait_quality_rederive: Any = None,
    ) -> dict[str, Any]:
        nonlocal initial_validation_pending
        if not initial_validation_pending:
            raise RuntimeError("reverse-v6 initial artifact validation is one-shot")
        audit = validate_reverse_v6_strict_artifact(
            payload,
            compatibility_validator=compatibility_validator,
            bundle=bundle,
            current_central_hashes=current_central_hashes,
            current_evaluation_hashes=current_evaluation_hashes,
            require_all_three_pass=require_all_three_pass,
            gait_quality_rederive=gait_quality_rederive,
            _allow_initial_adapter_record_absent=True,
        )
        initial_validation_pending = False
        return audit

    skeleton_conversion_pending = True

    def construct_real_v6_artifact(value: Any) -> Any:
        nonlocal skeleton_conversion_pending
        if not skeleton_conversion_pending:
            raise RuntimeError("reverse-v6 frozen skeleton conversion is one-shot")
        native = base.json_native(value)
        artifact = _real_v6_artifact_from_frozen_skeleton(native)
        skeleton_conversion_pending = False
        return artifact

    run_evaluation = _clone_function(
        base.run_evaluation,
        global_overrides={
            "STRICT_ARTIFACT_KIND": DIAGNOSTIC_ARTIFACT_KIND,
            "validate_trusted_h4_bundle": trusted_bundle,
            "_make_environment_and_policy": make_environment,
            "_run_episode": run_episode,
            "json_native": construct_real_v6_artifact,
            "validate_h4_strict_artifact": initial_artifact_validator,
        },
    )
    _assert_module_contract_unchanged(base, post, core, original)
    return {
        "trusted_bundle": trusted_bundle,
        "central_safety": central_safety,
        "safety": safety,
        "episode_validator": episode_validator,
        "compatibility_validator": compatibility_validator,
        "artifact_validator": artifact_validator,
        "make_environment": make_environment,
        "run_episode": run_episode,
        "run_evaluation": run_evaluation,
    }


def _adapter_source_hashes() -> dict[str, str]:
    return {
        ADAPTER_SOURCE_KEY: sha256_file(ADAPTER_PATH),
        ADAPTER_AUTHORIZATION_SOURCE_KEY: sha256_file(ADAPTER_AUTHORIZATION_PATH),
    }


def _augment_evaluation_hashes(
    base_hashes: Mapping[str, str], adapter_hashes: Mapping[str, str]
) -> dict[str, str]:
    result = dict(base_hashes)
    for path, digest in adapter_hashes.items():
        if path in result:
            raise ValueError(f"reverse-v6 adapter source collision: {path}")
        result[path] = digest
    expected = {
        "scripts/evaluate_h4_training_candidate.py": PINNED_FROZEN_SOURCES[
            "h4_candidate_evaluator"
        ][1],
        "safe_gait_experts/h4_post_training.py": PINNED_FROZEN_SOURCES[
            "h4_post_training"
        ][1],
        "safe_gait_experts/h4_training_alignment.py": PINNED_FROZEN_SOURCES[
            "h4_training_alignment"
        ][1],
        "scripts/train_h4_aligned_expert.py": PINNED_FROZEN_SOURCES[
            "h4_runner"
        ][1],
    }
    if any(result.get(path) != digest for path, digest in expected.items()):
        raise ValueError("reverse-v6 frozen evaluation source provenance drifted")
    return result


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
    candidate_hashes_pre: Mapping[str, str],
    candidate_hashes_post: Mapping[str, str],
    validator: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    if adapter_hashes_pre != adapter_hashes_post:
        raise RuntimeError("adapter or adapter authorization changed during evaluation")
    if frozen_hashes_pre != frozen_hashes_post:
        raise RuntimeError("frozen evaluator sources changed during evaluation")
    if candidate_hashes_pre != candidate_hashes_post:
        raise RuntimeError("validated reverse-v6 candidate files changed during evaluation")
    augmented = _augment_evaluation_hashes(
        base_evaluation_hashes, adapter_hashes_post
    )
    provenance = artifact.get("runtime_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("reverse-v6 artifact provenance is missing")
    provenance["evaluation_source_hashes_pre"] = dict(augmented)
    provenance["evaluation_source_hashes_post"] = dict(augmented)
    provenance["evaluation_source_hashes_current"] = dict(augmented)
    provenance["pre_post_source_hashes_unchanged"] = True
    provenance["reverse_v6_absolute_targets_evaluator_adapter"] = {
        "schema_version": 1,
        "contract_id": ADAPTER_CONTRACT_ID,
        "training_contract_id": TRAINING_CONTRACT_ID,
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
            "tracking_sigma": TRACKING_SIGMA,
        },
        "backward_residual_scale": 0.0,
        "host_rederivation": CONTROL_TRACE_SEMANTICS,
        "physics_trace_semantics": (
            "DIRECT_MJX_STEP_REPLAY_FROM_CONTROL_ENTRY_WITH_ACTUAL_APPLIED_TARGETS"
        ),
        "physics_substeps_per_control": PHYSICS_SUBSTEPS_PER_CONTROL,
        "dynamic6_endpoint_bitwise_exact_required": True,
        "snapshot_endpoint_bitwise_exact_required": True,
        "legacy_float64_control_audit_diagnostic_only": True,
        "legacy_float64_control_audit_counts_exact_required": True,
        "adapter_source": {
            "path": ADAPTER_SOURCE_KEY,
            "sha256_pre": adapter_hashes_pre[ADAPTER_SOURCE_KEY],
            "sha256_post": adapter_hashes_post[ADAPTER_SOURCE_KEY],
            "sha256_current": adapter_hashes_post[ADAPTER_SOURCE_KEY],
            "unchanged": True,
        },
        "adapter_authorization": {
            "path": ADAPTER_AUTHORIZATION_SOURCE_KEY,
            "sha256_pre": adapter_hashes_pre[ADAPTER_AUTHORIZATION_SOURCE_KEY],
            "sha256_post": adapter_hashes_post[
                ADAPTER_AUTHORIZATION_SOURCE_KEY
            ],
            "sha256_current": adapter_hashes_post[
                ADAPTER_AUTHORIZATION_SOURCE_KEY
            ],
            "unchanged": True,
        },
        "frozen_source_hashes_pre": dict(frozen_hashes_pre),
        "frozen_source_hashes_post": dict(frozen_hashes_post),
        "candidate_bundle_hashes_pre": dict(candidate_hashes_pre),
        "candidate_bundle_hashes_post": dict(candidate_hashes_post),
        "candidate_bundle_hashes_current": dict(candidate_hashes_post),
        "original_module_globals_and_function_references_unchanged": True,
        "promotion_evidence_allowed": False,
        "candidate_adoption_allowed": False,
        "release_allowed": False,
        "hardware_deployment": "PROHIBITED",
    }
    audit = validator(
        artifact,
        bundle=bundle,
        current_central_hashes=central_hashes,
        current_evaluation_hashes=augmented,
        require_all_three_pass=False,
    )
    summary = artifact.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("reverse-v6 artifact summary is missing")
    summary["recomputed_validation_passed"] = bool(
        audit["passing_seed_count"] == summary.get("passing_seed_count")
    )
    validator(
        artifact,
        bundle=bundle,
        current_central_hashes=central_hashes,
        current_evaluation_hashes=augmented,
        require_all_three_pass=False,
    )
    if _candidate_file_snapshot(bundle) != dict(candidate_hashes_post):
        raise RuntimeError(
            "validated reverse-v6 candidate files changed during final validation"
        )
    return artifact, augmented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exact CPU fixed-3x6s reverse-v6 evaluation; training, promotion, "
            "adoption, release, overwrite, and hardware use are unsupported."
        )
    )
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--params-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-run-root", type=Path, default=TRUSTED_RUN_ROOT)
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
    params_sha = _require_sha256(args.params_sha256, "params CLI SHA256")
    manifest_sha = _require_sha256(args.manifest_sha256, "manifest CLI SHA256")
    checks = {
        "params_path": args.params == EXPECTED_PARAMS_PATH.resolve(),
        "manifest_path": args.manifest == EXPECTED_MANIFEST_PATH.resolve(),
        "output_path": args.output == EXPECTED_OUTPUT_PATH.resolve(),
        "trusted_root": args.trusted_run_root == TRUSTED_RUN_ROOT.resolve(),
        "adapter_authorization": args.adapter_authorization
        == ADAPTER_AUTHORIZATION_PATH.resolve(),
        "params_manifest_sha_distinct": params_sha != manifest_sha,
        "platform_cpu": args.platform == "cpu",
        "wiring_forbidden": args.allow_wiring_diagnostic is False,
        "promotion_forbidden": args.promotion_evidence_output is None,
    }
    if not all(checks.values()):
        raise ValueError(f"exact reverse-v6 adapter CLI drifted: {checks}")
    if require_output_absent and args.output.exists():
        raise FileExistsError(f"refusing to overwrite strict output: {args.output}")


def run_adapter(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str], Any]:
    _validate_exact_cli(args)
    load_and_validate_adapter_authorization(args.adapter_authorization)
    adapter_hashes_pre = _adapter_source_hashes()
    frozen_hashes_pre = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    base, post, core = _load_frozen_modules()
    if _verify_file_bindings(PINNED_FROZEN_SOURCES) != frozen_hashes_pre:
        raise RuntimeError("frozen sources changed during module import")
    graph = build_reverse_v6_call_graph(base, post, core)
    original = _module_contract_snapshot(base, post, core)
    bundle_pre = graph["trusted_bundle"](
        params_path=args.params,
        manifest_path=args.manifest,
        expected_params_sha256=args.params_sha256,
        expected_manifest_sha256=args.manifest_sha256,
        trusted_run_root=args.trusted_run_root,
        allow_wiring_diagnostic=False,
    )
    candidate_hashes_pre = _candidate_file_snapshot(bundle_pre)
    try:
        artifact, bundle, central_hashes, evaluation_hashes = graph[
            "run_evaluation"
        ](args)
    finally:
        _assert_module_contract_unchanged(base, post, core, original)
    if bundle.candidate_record() != bundle_pre.candidate_record():
        raise RuntimeError("reverse-v6 bundle identity changed during evaluation")
    candidate_hashes_post = _candidate_file_snapshot(bundle)
    adapter_hashes_post = _adapter_source_hashes()
    frozen_hashes_post = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    artifact, augmented = _augment_and_revalidate(
        artifact=artifact,
        bundle=bundle,
        central_hashes=central_hashes,
        base_evaluation_hashes=evaluation_hashes,
        adapter_hashes_pre=adapter_hashes_pre,
        adapter_hashes_post=adapter_hashes_post,
        frozen_hashes_pre=frozen_hashes_pre,
        frozen_hashes_post=frozen_hashes_post,
        candidate_hashes_pre=candidate_hashes_pre,
        candidate_hashes_post=candidate_hashes_post,
        validator=graph["artifact_validator"],
    )
    if _adapter_source_hashes() != adapter_hashes_post:
        raise RuntimeError("adapter sources changed before output write")
    if _verify_file_bindings(PINNED_FROZEN_SOURCES) != frozen_hashes_post:
        raise RuntimeError("frozen sources changed before output write")
    if _candidate_file_snapshot(bundle) != candidate_hashes_post:
        raise RuntimeError("reverse-v6 candidate files changed before output write")
    _assert_module_contract_unchanged(base, post, core, original)
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
