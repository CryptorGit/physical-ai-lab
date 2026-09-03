"""Fail-closed H4 checkpoint, inference, and strict-evidence contracts.

This module intentionally depends only on the Python standard library and
NumPy.  JAX, Brax, MJX, MuJoCo, and ONNX Runtime are imported lazily by the
entrypoints that use this contract.  In particular, a pickle is never opened
until its immutable runner manifest, resolved configuration, and exact SHA256
bindings have all been checked.

The helpers here do not adopt a policy and never authorize hardware.  They
only establish whether a simulation checkpoint or evidence artifact is
structurally suitable for diagnostic evaluation or later promotion review.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import pickle
from typing import Any, Callable, Mapping, Sequence

import numpy as np


H4_ACTOR_OBSERVATION_WIDTH = 116
H4_CRITIC_OBSERVATION_WIDTH = 227
H4_ACTION_WIDTH = 14
H4_HEAD_ACTION_SLICE = slice(5, 9)
H4_CONTROL_DT_S = 0.020
H4_PHYSICS_DT_S = 0.002
H4_STRICT_DURATION_S = 6.0
H4_STRICT_CONTROL_TICKS = 300
H4_STRICT_PHYSICS_SUBSTEPS = 3_000
H4_STRICT_GAIT_SAMPLES = H4_STRICT_PHYSICS_SUBSTEPS + 1
H4_STRICT_SEEDS: Mapping[str, tuple[int, int, int]] = {
    "forward": (20_260_809, 20_261_809, 20_262_809),
    "reverse": (20_260_810, 20_265_810, 20_271_810),
}
H4_STRICT_COMMANDS: Mapping[str, tuple[float, float, float]] = {
    "forward": (0.05, 0.0, 0.0),
    "reverse": (-0.05, 0.0, 0.0),
}
H4_GAIT_SAMPLE_SOURCE = "mjx_xpos_xmat_after_each_mjx_step"
H4_WIRING_INTERACTIONS = 40
H4_PILOT_INTERACTIONS = 250_000
H4_WIRING_TRAINING_STEPS = 1
H4_PILOT_TRAINING_STEPS = 5
H4_WIRING_OPTIMIZER_UPDATES = 2
H4_PILOT_OPTIMIZER_UPDATES = 400
H4_FORWARD_V4_FULL_TRAINING_PROGRESS_INTERACTIONS = (
    50_000,
    100_000,
    150_000,
    200_000,
    250_000,
)
H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE = (
    "WIRING_COMPILED_ASSERTION_NO_EPISODE_ROWS_EXPECTED"
)
H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE = "FULL_RUNTIME_EPISODE_ROWS_REQUIRED"
H4_GPU_PLATFORM_SELECTOR = "cuda,cpu"
H4_GPU_XLA_FLAGS = "--xla_gpu_autotune_level=4"
H4_GPU_XLA_POLICY = "CORRECTNESS_CHECKED_LEVEL4_DISQUALIFY_MISMATCH"
H4_REVERSE_TEACHER_TABLE_ROWS = 54
H4_REVERSE_TEACHER_ENTRY_PHASE_BINS = 14.0
H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS = 1.62
H4_REVERSE_SOURCE_PERIOD_BINS = 27
H4_REVERSE_RESIDUAL_SCALE = 0.12
H4_REVERSE_ACTION_DELAY_MIN = 0
H4_REVERSE_ACTION_DELAY_MAX_EXCLUSIVE = 1
H4_REVERSE_COMPOSITION_TRACE_SEMANTICS = (
    "PINNED_TEACHER_DELAYED_RESIDUAL_THEN_MARGIN_SLEW_PRECOMPOSER_"
    "THEN_FINAL_MARGIN_GUARD"
)
STRICT_ARTIFACT_KIND = "openduckmini_h4_strict_promotion_evaluation"
PROMOTION_EVIDENCE_KIND = "openduckmini_h4_promotion_evidence"
PINNED_SELECTED_REVERSE_TEACHER_SHA256 = (
    "7a24a7c9096a1c4a9dc72ac85ec01c5e0a41acf8214d80cc7e2cf4ccc50ae237"
)
PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256 = (
    "082405e34b8a46e7d4a9ccf7b8c0729871fee1eb202b4a1ed8c758b2c7a52900"
)
PINNED_FORWARD_MINIMUM_SPEC_SHA256 = (
    "26611630368069e9cbd2516e08d5adb13547a5fa2763173ca04d67751be83428"
)
PINNED_FORWARD_MINIMUM_SPEC_CANONICAL_SHA256 = (
    "dff0b683020e3eec21e221249b27233ef008215fb156996cad314736f7c89d65"
)
PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256 = (
    "b574d4a41b05f54666f3befe41eda9a54b4e12970e6acaa7a9e95c1bf82de7c3"
)
H4_FORWARD_ITERATION_V2_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V2_250K_FROM_V22"
)
H4_FORWARD_ITERATION_V2_SOURCE_LABELS = frozenset(
    {
        "forward_iteration_v2_authorization",
        "forward_iteration_v2_failed_candidate_manifest",
        "forward_iteration_v2_failed_candidate_params",
        "forward_iteration_v2_integrated_strict_evaluation",
    }
)
PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256 = (
    "d364cc752c4702a6edada7fe5fac5ddfbab1926d5520b2bd0e1a20f532d6e3f3"
)
H4_REVERSE_ITERATION_V2_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V2_250K_FROM_V22"
)
H4_REVERSE_ITERATION_V2_SOURCE_LABELS = frozenset(
    {
        "reverse_iteration_v2_authorization",
        "reverse_iteration_v2_failed_candidate_manifest",
        "reverse_iteration_v2_failed_candidate_params",
        "reverse_iteration_v2_integrated_strict_evaluation",
    }
)
PINNED_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_AUTHORIZATION_SHA256 = (
    "93daa0c35f08929c17c6eef799565d327ce362c1c1ebdeaf9aa22ca6cc5d153f"
)
H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_250K_FROM_V22"
)
H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_WIRING_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_WIRING_PREFLIGHT_40_FROM_V22"
)
H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_SOURCE_LABELS = frozenset(
    {
        "forward_iteration_v3_authorization",
        "forward_iteration_v3_failed_candidate_manifest",
        "forward_iteration_v3_failed_candidate_params",
        "forward_iteration_v3_integrated_strict_evaluation",
    }
)
PINNED_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION_SHA256 = (
    "b27d3e12f5619bf008b5034f33e561a8ab8d06c3880a914f1a28781c0a3bb5c7"
)
H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_250K_FROM_V22"
)
H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_WIRING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_WIRING_PREFLIGHT_40_FROM_V22"
)
H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_SOURCE_LABELS = frozenset(
    {
        "reverse_iteration_v3_authorization",
        "reverse_iteration_v3_failed_candidate_manifest",
        "reverse_iteration_v3_failed_candidate_params",
        "reverse_iteration_v3_integrated_strict_evaluation",
    }
)
H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_250K_FROM_V22"
)
H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_WIRING_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_250K_FROM_V22"
)
H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_WIRING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN = 0.24
H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_250K_FROM_V22"
)
H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_WIRING_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_NO_PPO_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_"
    "NO_PPO_PREFLIGHT_FROM_V22"
)
H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_250K_FROM_V22"
)
H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_WIRING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_WIRING_PREFLIGHT_40_FROM_V22"
)
H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_NO_PPO_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_NO_PPO_PREFLIGHT_FROM_V22"
)
H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_250K_FROM_V22"
)
H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_WIRING_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_NO_PPO_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_"
    "NO_PPO_PREFLIGHT_FROM_V22"
)
H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_250K_FROM_V22"
)
H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_WIRING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_NO_PPO_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_"
    "NO_PPO_PREFLIGHT_FROM_V22"
)
PINNED_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_AUTHORIZATION_SHA256 = (
    "8e8b722e9e3f8f4b3827a7ffd2dee3e3ee5a2d799bfd996e09b066ff71d93a04"
)
PINNED_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_AUTHORIZATION_SHA256 = (
    "6a10315593761a4d0ed034b331fe14e3f682bf8154a252e6820a5dd4f71038fe"
)
PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256 = (
    "5da1d3a8a2c505a5ce4bc6621f76dd3031070cdb467a4cde96b4ed3c23190c02"
)
H4_REVERSE_ITERATION_V6_RESIDUAL_AUTHORITY_SCALE = 0.0
H4_REVERSE_ITERATION_V6_ACTIVE_LEG_INDICES = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
H4_REVERSE_ITERATION_V6_HEAD_INDICES = (5, 6, 7, 8)
H4_REVERSE_ITERATION_V6_DIRECTIONAL_SPAN_FRACTION = 0.9
H4_REVERSE_ITERATION_V6_NEAR_ZERO_BASE_CAP_RAD = 0.25
H4_REVERSE_ITERATION_V6_NONLINEAR_EXPONENT = 5
H4_FORWARD_ITERATION_V6_RUNTIME_AUDIT_MODES = {
    "WIRING_PASS": "WIRING_COMPILED_ASSERTION_NO_EPISODE_ROWS_ALLOWED",
    "COMPLETED": "FULL_RUNTIME_EPISODE_ROWS_REQUIRED",
}
H4_REVERSE_ITERATION_V6_RUNTIME_AUDIT_MODES = {
    "WIRING_PASS": "WIRING_COMPILED_ASSERTION_NO_EPISODE_ROWS_ALLOWED",
    "COMPLETED": "FULL_RUNTIME_EPISODE_ROWS_REQUIRED",
}
H4_ITERATION_V6_ARTIFACT_CROSS_BINDING = {
    "all_ten_iteration_mode_booleans_exact": True,
    "authorization_sha_and_contract_id_exact": True,
    "execution_contract_id_cross_bound": True,
    "runtime_requirement_cross_bound": True,
    "core_source_cross_bound": True,
    "authorization_contracts_cross_bound": True,
    "expert_runtime_evidence_cross_bound": True,
    "passed": True,
}


def _json_type_and_value_exact(actual: Any, expected: Any) -> bool:
    """Compare JSON-native values without Python's bool/int coercion."""

    if isinstance(expected, Mapping):
        return bool(
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                _json_type_and_value_exact(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, list):
        return bool(
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _json_type_and_value_exact(item, expected_item)
                for item, expected_item in zip(actual, expected)
            )
        )
    return bool(type(actual) is type(expected) and actual == expected)


def _iteration_v6_artifact_cross_binding_is_exact(value: Any) -> bool:
    """Return whether one runner cross-binding record is the exact bool schema."""

    return _json_type_and_value_exact(
        value, H4_ITERATION_V6_ARTIFACT_CROSS_BINDING
    )
H4_FORWARD_ITERATION_V6_RUNTIME_INFO_KEYS = (
    "h4_v6_forward_contact_abort_routing_exact",
    "h4_v6_forward_contact_abort_island_loss",
    "h4_v6_forward_contact_abort_off_gap_diagnostic_loss",
    "h4_v6_forward_contact_abort_off_gap_reward_contribution",
    "h4_v6_forward_contact_abort_pulse_reward_scale",
    "h4_v6_forward_contact_abort_routing_violation",
    "h4_v6_forward_contact_abort_routing_assertion_token",
)
H4_REVERSE_ITERATION_V6_RUNTIME_INFO_KEYS = (
    "h4_v6_reverse_decoder_action",
    "h4_v6_reverse_decoder_raw_targets",
    "h4_v6_reverse_decoder_margin_targets",
    "h4_v6_reverse_decoder_exact",
    "h4_v6_reverse_decoder_max_abs_error",
    "h4_v6_reverse_decoder_leg_count",
    "h4_v6_reverse_decoder_leg_count_exact",
    "h4_v6_reverse_decoder_head_zero_exact",
    "h4_v6_reverse_teacher_target_contribution_zero_exact",
    "h4_v6_reverse_residual_authority_scale",
    "h4_v6_reverse_decoder_all_finite",
    "h4_v6_reverse_decoder_margin_saturation_count",
    "h4_v6_reverse_decoder_action_clip_count",
    "h4_v6_reverse_decoder_guard_lag_max_rad",
    "h4_v6_reverse_precomposer_call_count",
    "h4_v6_reverse_precomposer_call_count_exact",
    "h4_v6_reverse_final_guard_call_count",
    "h4_v6_reverse_final_guard_call_count_exact",
    "h4_v6_reverse_decoder_violation",
    "h4_v6_reverse_decoder_assertion_token",
)
H4_ITERATION_V4_CAUSAL_SOURCE_PATHS: Mapping[str, str] = {
    "h4_training_alignment": "safe_gait_experts/h4_training_alignment.py",
    "h4_runner": "scripts/train_h4_aligned_expert.py",
    "h4_post_training": "safe_gait_experts/h4_post_training.py",
    "h4_candidate_evaluator": "scripts/evaluate_h4_training_candidate.py",
    "h4_no_ppo_smoke": "scripts/smoke_h4_training_alignment.py",
}


def _iteration_v4_source_labels(prefix: str) -> frozenset[str]:
    return frozenset(
        {
            f"{prefix}authorization",
            f"{prefix}previous_iteration_authorization",
            f"{prefix}failed_candidate_manifest",
            f"{prefix}failed_candidate_params",
            f"{prefix}integrated_strict_evaluation",
            *(f"{prefix}source_{label}" for label in H4_ITERATION_V4_CAUSAL_SOURCE_PATHS),
        }
    )


H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_SOURCE_LABELS = (
    _iteration_v4_source_labels("forward_iteration_v4_")
)
H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_SOURCE_LABELS = (
    _iteration_v4_source_labels("reverse_iteration_v4_")
)
PINNED_REVERSE_MINIMUM_SPEC_SHA256 = (
    "66b12bcbaf8a55cc0477b8872cebb8fe29c2c321b2c2224afd3089c5ecb500a8"
)
PINNED_V22_PARENT_TREE_SHA256 = (
    "fe35e5ee932dc0ba70c1c32f3e410ea469d229e69cab43ed85f34aefe9505f1f"
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 for one regular file."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash one file or a non-empty tree exactly like the frozen trainer."""

    resolved = Path(path).resolve()
    if resolved.is_file():
        return sha256_file(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"cannot hash empty directory: {resolved}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    """Require an exact lowercase SHA256, avoiding ambiguous normalization."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be an exact lowercase SHA256")
    return value


def _validate_json_value(value: Any, location: str = "$") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string JSON object key at {location}")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise ValueError(f"non-JSON value at {location}: {type(value).__name__}")


def json_native(value: Any) -> Any:
    """Convert finite NumPy/dataclass payload leaves to strict JSON values."""

    if isinstance(value, Mapping):
        converted = {
            key: json_native(item)
            for key, item in value.items()
        }
    elif isinstance(value, (list, tuple)):
        converted = [json_native(item) for item in value]
    elif isinstance(value, np.ndarray):
        converted = json_native(value.tolist())
    elif isinstance(value, np.generic):
        converted = json_native(value.item())
    else:
        converted = value
    _validate_json_value(converted)
    return converted


def load_json_strict(path: Path) -> Any:
    """Load JSON while rejecting duplicate keys and non-finite numbers."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is prohibited: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is prohibited: {key}")
            result[key] = value
        return result

    payload = json.loads(
        resolved.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    _validate_json_value(payload)
    return payload


def canonical_json_bytes(value: Any) -> bytes:
    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_new_json(path: Path, value: Any) -> str:
    """Create one immutable JSON artifact and return its SHA256."""

    resolved = Path(path).resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {resolved}")
    _validate_json_value(value)
    encoded = (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("xb") as stream:
        stream.write(encoded)
    return sha256_file(resolved)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _is_finite_number(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _recorded_path(record: Mapping[str, Any], label: str) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be non-empty")
    return Path(raw_path).resolve()


def _is_absolute_path_text(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and value
        and (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
        )
    )


def _verify_file_record(
    record: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> tuple[Path, str]:
    mapping = _require_mapping(record, label)
    path = _recorded_path(mapping, label)
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"{label} path mismatch")
    recorded_sha = require_sha256(mapping.get("sha256"), f"{label}.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != recorded_sha:
        raise ValueError(f"{label} SHA256 mismatch")
    return path, actual_sha


def _validate_forward_v4_training_curve_runtime(
    curve_path: Path, *, wiring_only: bool
) -> dict[str, Any]:
    """Independently rederive v4 runtime evidence from the bound CSV."""

    qualifying_totals = {
        "episode/h4/v4_single_authority_dynamic6_exact": 1.0,
        "episode/h4/v4_single_authority_dynamic6_max_abs_error": 0.0,
        "episode/h4/v4_single_authority_dynamic6_field_count_exact": 1.0,
        "episode/h4/v4_saved_dynamic6_field_count_exact": 1.0,
        "episode/h4/v4_saved_dynamic6_all_finite": 1.0,
        "episode/h4/v4_telemetry_force_shape_valid": 1.0,
        "episode/h4/v4_telemetry_force_all_finite": 1.0,
        "episode/h4/v4_single_authority_violation": 0.0,
        "episode/h4/v4_single_authority_assertion_token": 0.0,
    }
    diagnostic_count_keys = {
        "episode/h4/v4_single_authority_dynamic6_field_count": 6.0,
        "episode/h4/v4_saved_dynamic6_substep_count": 10.0,
        "episode/h4/v4_saved_dynamic6_field_count": 6.0,
    }
    try:
        with Path(curve_path).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise ValueError("forward-v4 training curve could not be read") from exc
    if not rows:
        raise ValueError("forward-v4 training curve has no progress rows")

    progress_interactions: list[int] = []
    episode_progress_interactions: list[int] = []
    non_episode_rows: list[tuple[int, int, dict[str, str | None]]] = []
    episode_row_count = 0
    for index, row in enumerate(rows):
        raw_step = row.get("environment_interactions")
        try:
            numeric_step = float(raw_step) if raw_step not in (None, "") else math.nan
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"forward-v4 training curve row {index} interaction is invalid"
            ) from exc
        if not np.isfinite(numeric_step) or not numeric_step.is_integer():
            raise ValueError(
                f"forward-v4 training curve row {index} interaction is invalid"
            )
        progress_interactions.append(int(numeric_step))

        raw_length = row.get("episode/length")
        if raw_length in (None, ""):
            non_episode_rows.append((index, int(numeric_step), dict(row)))
            continue
        try:
            length = float(raw_length)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"forward-v4 episode row {index} length is invalid"
            ) from exc
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(
                f"forward-v4 episode row {index} length is invalid"
            )
        episode_row_count += 1
        episode_progress_interactions.append(int(numeric_step))
        for key, multiplier in qualifying_totals.items():
            raw_value = row.get(key)
            try:
                value = (
                    float(raw_value) if raw_value not in (None, "") else math.nan
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"forward-v4 episode row {index} {key} is invalid"
                ) from exc
            expected = length if multiplier == 1.0 else 0.0
            if not np.isfinite(value) or value != expected:
                raise ValueError(
                    f"forward-v4 episode row {index} {key} drifted"
                )
        for key in diagnostic_count_keys:
            raw_value = row.get(key)
            try:
                value = (
                    float(raw_value) if raw_value not in (None, "") else math.nan
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"forward-v4 episode row {index} {key} is invalid"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"forward-v4 episode row {index} {key} is not finite"
                )

    if wiring_only:
        progress_exact = bool(
            progress_interactions
            and progress_interactions[-1] == H4_WIRING_INTERACTIONS
            and max(progress_interactions) == H4_WIRING_INTERACTIONS
            and all(0 <= step <= H4_WIRING_INTERACTIONS for step in progress_interactions)
        )
        if not progress_exact:
            raise ValueError(
                "forward-v4 wiring curve did not complete exact 40 interactions"
            )
    else:
        if episode_row_count <= 0:
            raise ValueError(
                "forward-v4 full run has no exact runtime episode rows"
            )
        expected_training_progress = list(
            H4_FORWARD_V4_FULL_TRAINING_PROGRESS_INTERACTIONS
        )
        progress_exact = bool(
            episode_progress_interactions == expected_training_progress
            and progress_interactions[-1] == H4_PILOT_INTERACTIONS
            and max(progress_interactions) == H4_PILOT_INTERACTIONS
            and all(
                0 < step <= H4_PILOT_INTERACTIONS
                for step in progress_interactions
            )
            and len(non_episode_rows) == 1
            and non_episode_rows[0][0] == len(rows) - 1
            and non_episode_rows[0][1] == H4_PILOT_INTERACTIONS
        )
        if not progress_exact:
            raise ValueError(
                "forward-v4 full curve is not exact five monotonic training "
                "rows plus one final-metrics row at 250000 interactions"
            )
        final_metrics_row = non_episode_rows[0][2]
        if any(
            value not in (None, "")
            for key, value in final_metrics_row.items()
            if key.startswith("episode/")
        ):
            raise ValueError(
                "forward-v4 full final-metrics row contains episode metrics"
            )
        final_metrics: dict[str, float] = {}
        for key, raw_value in final_metrics_row.items():
            if key == "environment_interactions" or raw_value in (None, ""):
                continue
            if not key.startswith("training/"):
                raise ValueError(
                    "forward-v4 full final-metrics row contains a non-training field"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"forward-v4 full final metric {key} is invalid"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"forward-v4 full final metric {key} is not finite"
                )
            final_metrics[key] = value
        if not final_metrics:
            raise ValueError("forward-v4 full final-metrics row is empty")
    return {
        "observed_episode_metric_rows": episode_row_count,
        "progress_reached_final_interaction": progress_exact,
        "episode_metric_rows_exact_if_observed": True,
        **(
            {
                "training_progress_interactions": list(
                    episode_progress_interactions
                ),
                "final_metrics_interaction": H4_PILOT_INTERACTIONS,
                "final_metrics": final_metrics,
            }
            if not wiring_only
            else {}
        ),
    }


def _validate_forward_v4_single_authority_closure(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result_payload: Mapping[str, Any],
    outputs: Mapping[str, Any],
    status: str,
) -> None:
    """Validate the canonical forward-v4 authority contract for v4 and descendants."""

    expected_authority_requirement = {
        "dynamic6_exact": True,
        "dynamic6_max_abs_error": 0.0,
        "dynamic6_field_count": 6,
        "dynamic6_field_count_exact": True,
        "saved_dynamic6_substep_count": 10,
        "saved_dynamic6_field_count": 6,
        "saved_dynamic6_field_count_exact": True,
        "saved_dynamic6_all_finite": True,
        "telemetry_force_shape": [2],
        "telemetry_force_shape_valid": True,
        "telemetry_force_all_finite": True,
        "count_totals_qualification_role": (
            "NON_QUALIFYING_FINITE_DIAGNOSTICS_ONLY"
        ),
        "host_count_multiplication_for_qualification": False,
        "numeric_tolerance_used": False,
        "authority_violation_count": 0.0,
        "assertion_token_sum": 0.0,
        "fail_closed_before_output_commit": True,
        "full_nonempty_episode_rows_required": True,
        "wiring_zero_episode_rows_require_compiled_assertion_evidence": True,
    }
    source_preflight = _require_mapping(
        config.get("forward_v4_source_semantic_preflight"),
        "forward-v4 source-semantic preflight",
    )
    authority_runtime = _require_mapping(
        result_payload.get("forward_v4_single_authority_runtime"),
        "forward-v4 single-authority runtime",
    )
    expected_runtime_audit_mode = (
        H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
        if status == "WIRING_PASS"
        else H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
    )
    curve_path, _ = _verify_file_record(
        outputs.get("training_curve"), label="outputs.training_curve"
    )
    curve_runtime = _validate_forward_v4_training_curve_runtime(
        curve_path, wiring_only=(status == "WIRING_PASS")
    )
    if status != "WIRING_PASS":
        result_final_metrics = _require_mapping(
            result_payload.get("final_metrics"),
            "forward-v4 result final metrics",
        )
        normalized_result_final_metrics: dict[str, float] = {}
        for key, value in result_final_metrics.items():
            if (
                not isinstance(key, str)
                or not key.startswith("training/")
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
            ):
                raise ValueError(
                    "forward-v4 result final metrics are not exact finite "
                    "training scalars"
                )
            normalized_result_final_metrics[key] = float(value)
        if (
            not normalized_result_final_metrics
            or curve_runtime.get("final_metrics")
            != normalized_result_final_metrics
        ):
            raise ValueError(
                "forward-v4 bound curve final metrics differ from run result"
            )

    derived = _require_mapping(
        source_preflight.get("derived_diagnostics"),
        "forward-v4 excluded derived diagnostics",
    )
    derived_fields = _require_mapping(
        derived.get("fields"), "forward-v4 excluded diagnostic fields"
    )
    source_provenance = _require_mapping(
        source_preflight.get("source_provenance"),
        "forward-v4 official source provenance",
    )
    probe_input = _require_mapping(
        source_preflight.get("probe_input"),
        "forward-v4 source-semantic probe input",
    )
    derived_exact = True
    for field in ("cfrc_int", "cfrc_ext"):
        diagnostic = _require_mapping(
            derived_fields.get(field),
            f"forward-v4 excluded {field} diagnostic",
        )
        exact = diagnostic.get("exact")
        error = diagnostic.get("max_abs_error")
        derived_exact = bool(
            derived_exact
            and set(diagnostic) == {"exact", "max_abs_error"}
            and isinstance(exact, bool)
            and isinstance(error, (int, float))
            and not isinstance(error, bool)
            and np.isfinite(float(error))
            and float(error) >= 0.0
            and exact is (float(error) == 0.0)
        )
    source_root = PurePosixPath(str(source_provenance.get("source_root", "")))
    expected_source_files = {
        "joystick": (
            "playground/open_duck_mini_v2/joystick.py",
            "95890569d971725308b5a9c0996bfa5fd9520479f014f325e810aa1db272eb9d",
        ),
        "mjx_env": (
            ".venv/lib/python3.12/site-packages/"
            "mujoco_playground/_src/mjx_env.py",
            "c3f1cfe0de036c3ccbba46e8cdd661cb48bfea8f182955298205f17787f53dfe",
        ),
    }
    source_provenance_exact = source_root.is_absolute()
    for label, (relative, expected_sha) in expected_source_files.items():
        record = _require_mapping(
            source_provenance.get(label),
            f"forward-v4 official {label} provenance",
        )
        resolved = PurePosixPath(str(record.get("resolved_path", "")))
        source_provenance_exact = bool(
            source_provenance_exact
            and set(record) == {"resolved_path", "relative_path", "sha256"}
            and record.get("relative_path") == relative
            and record.get("sha256") == expected_sha
            and resolved == source_root / PurePosixPath(relative)
        )

    if status == "WIRING_PASS":
        expected_runtime = {
            "audit_mode": H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE,
            "observed_episode_metric_rows": curve_runtime[
                "observed_episode_metric_rows"
            ],
            "episode_metric_rows_exact_if_observed": True,
            "source_semantic_preflight_passed": True,
            "per_step_compiled_fail_closed_assertion_bound": True,
            "completed_environment_interactions": H4_WIRING_INTERACTIONS,
            "completed_training_steps": H4_WIRING_TRAINING_STEPS,
            "completed_optimizer_updates": H4_WIRING_OPTIMIZER_UPDATES,
            "progress_reached_final_interaction": True,
            "final_params_all_finite": True,
            "final_metrics_all_finite": True,
            "source_and_teacher_unchanged": True,
            "authority_violation_count": 0.0,
            "assertion_token_sum": 0.0,
            "passed": True,
        }
        runtime_boolean_keys = {
            "episode_metric_rows_exact_if_observed",
            "source_semantic_preflight_passed",
            "per_step_compiled_fail_closed_assertion_bound",
            "progress_reached_final_interaction",
            "final_params_all_finite",
            "final_metrics_all_finite",
            "source_and_teacher_unchanged",
            "passed",
        }
    else:
        expected_runtime = {
            "audit_mode": H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE,
            "dynamic6_exact": True,
            "dynamic6_max_abs_error": 0.0,
            "dynamic6_field_count": 6,
            "dynamic6_field_count_exact": True,
            "saved_dynamic6_substep_count": 10,
            "saved_dynamic6_field_count": 6,
            "saved_dynamic6_field_count_exact": True,
            "saved_dynamic6_all_finite": True,
            "telemetry_force_shape": [2],
            "telemetry_force_shape_valid": True,
            "telemetry_force_all_finite": True,
            "observed_episode_metric_rows": curve_runtime[
                "observed_episode_metric_rows"
            ],
            "authority_violation_count": 0.0,
            "assertion_token_sum": 0.0,
            "passed": True,
        }
        runtime_boolean_keys = {
            "dynamic6_exact",
            "dynamic6_field_count_exact",
            "saved_dynamic6_field_count_exact",
            "saved_dynamic6_all_finite",
            "telemetry_force_shape_valid",
            "telemetry_force_all_finite",
            "passed",
        }
    runtime_contract_exact = bool(
        set(authority_runtime) == set(expected_runtime)
        and dict(authority_runtime) == expected_runtime
        and all(
            authority_runtime.get(key) is True for key in runtime_boolean_keys
        )
        and type(authority_runtime.get("observed_episode_metric_rows")) is int
        and authority_runtime.get("audit_mode") == expected_runtime_audit_mode
    )
    if (
        set(source_preflight)
        != {
            "timing",
            "reference_source",
            "candidate_source",
            "source_provenance",
            "probe_input",
            "qualifying_dynamic_state_fields",
            "dynamic6_exact",
            "dynamic6_max_abs_error",
            "dynamic6_field_count",
            "derived_diagnostics",
            "observed_reference_count",
            "passed",
        }
        or source_preflight.get("timing") != "ONCE_BEFORE_PPO_COLLECTION"
        or source_preflight.get("reference_source")
        != "OFFICIAL_MJX_ENV_STEP_WRAPPER_NSUBSTEPS_10"
        or source_preflight.get("candidate_source")
        != "SINGLE_INSTRUMENTED_TEN_SUBSTEP_SCAN_ENDPOINT"
        or set(source_provenance)
        != {
            "source_root",
            "joystick",
            "mjx_env",
            "step_source_sha256",
            "step_source_semantics",
            "all_files_under_requested_source_root",
            "passed",
        }
        or source_provenance_exact is not True
        or source_provenance.get("all_files_under_requested_source_root") is not True
        or source_provenance.get("passed") is not True
        or source_provenance.get("step_source_sha256")
        != "26571e7510b2837dca07f69890dc26a89695dff4caa1fdc6a0d6736bd22da06b"
        or source_provenance.get("step_source_semantics")
        != (
            "LAX_SCAN_XS_EMPTY_LENGTH_NSUBSTEPS_BODY_REPLACE_CTRL_"
            "ACTION_THEN_MJX_STEP_RETURN_FINAL_CARRY"
        )
        or probe_input
        != {
            "seed": 20260809,
            "reset_noise_multiplier": 1.0,
            "initial_state_source": "ENV_RESET_JAX_PRNGKEY_SEED",
            "action_shape": [14],
            "action_dtype": "float32",
            "action_all_zero": True,
        }
        or source_preflight.get("qualifying_dynamic_state_fields")
        != ["qpos", "qvel", "act", "ctrl", "time", "qacc_warmstart"]
        or source_preflight.get("dynamic6_exact") is not True
        or source_preflight.get("dynamic6_max_abs_error") != 0.0
        or source_preflight.get("dynamic6_field_count") != 6
        or source_preflight.get("observed_reference_count") != 1
        or source_preflight.get("passed") is not True
        or set(derived)
        != {
            "qualification_role",
            "fields",
            "all_finite",
            "exclusion_is_semantic_not_tolerance",
            "numeric_tolerance_used",
        }
        or derived.get("qualification_role")
        != "NON_QUALIFYING_OBSERVED_DIAGNOSTICS_ONLY"
        or set(derived_fields) != {"cfrc_int", "cfrc_ext"}
        or derived_exact is not True
        or derived.get("all_finite") is not True
        or derived.get("exclusion_is_semantic_not_tolerance") is not True
        or derived.get("numeric_tolerance_used") is not False
        or runtime_contract_exact is not True
        or config.get("forward_v4_single_authority_runtime_requirement")
        != expected_authority_requirement
        or manifest.get("forward_v4_single_authority_runtime_requirement")
        != expected_authority_requirement
        or result_payload.get("forward_v4_single_authority_runtime_requirement")
        != expected_authority_requirement
        or manifest.get("forward_v4_source_semantic_preflight") != source_preflight
        or result_payload.get("forward_v4_source_semantic_preflight")
        != source_preflight
        or manifest.get("forward_v4_single_authority_runtime")
        != authority_runtime
        or manifest.get("forward_v4_single_authority_runtime_audit_mode")
        != expected_runtime_audit_mode
        or config.get("forward_v4_single_authority_runtime_audit_mode")
        != expected_runtime_audit_mode
        or result_payload.get("forward_v4_single_authority_runtime_audit_mode")
        != expected_runtime_audit_mode
    ):
        raise ValueError("forward-v4 single-authority closure drifted")


def _require_exact_gpu_device_audit(
    value: object, *, label: str, expected_label: str
) -> dict[str, Any]:
    audit = _require_mapping(value, label)
    expected_keys = {
        "label",
        "jax_array_leaf_count",
        "platforms",
        "devices",
        "expected_platform",
        "passed",
    }
    if set(audit) != expected_keys:
        raise ValueError(f"{label} field set drifted")
    platforms = audit.get("platforms")
    devices = audit.get("devices")
    leaf_count = audit.get("jax_array_leaf_count")
    if (
        audit.get("label") != expected_label
        or audit.get("expected_platform") != "gpu"
        or audit.get("passed") is not True
        or not isinstance(leaf_count, int)
        or isinstance(leaf_count, bool)
        or leaf_count <= 0
        or not isinstance(platforms, list)
        or not platforms
        or any(platform != "gpu" for platform in platforms)
        or not isinstance(devices, list)
        or not devices
        or any(not isinstance(device, str) or not device for device in devices)
    ):
        raise ValueError(f"{label} is not an exact successful GPU placement audit")
    return dict(audit)


def _validate_h4_gpu_training_provenance(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-bind the runner's GPU, XLA, callback, and tree-placement proof."""

    if config.get("platform") != "gpu":
        raise ValueError("formal H4 candidate training must resolve from --platform gpu")
    backend = _require_mapping(config.get("backend_resolution"), "GPU backend")
    if set(backend) != {
        "requested_cli_platform",
        "jax_platform_selector",
        "expected_resolved_backend",
        "resolved_default_backend",
        "resolved_device_platforms",
        "resolved_devices",
        "local_cpu_callback_devices",
        "local_cpu_callback_available",
        "passed",
    }:
        raise ValueError("GPU backend resolution field set drifted")
    resolved_devices = backend.get("resolved_devices")
    callback_devices = backend.get("local_cpu_callback_devices")
    if (
        backend.get("requested_cli_platform") != "gpu"
        or backend.get("jax_platform_selector") != H4_GPU_PLATFORM_SELECTOR
        or backend.get("expected_resolved_backend") != "gpu"
        or backend.get("resolved_default_backend") != "gpu"
        or backend.get("resolved_device_platforms") != ["gpu"]
        or not isinstance(resolved_devices, list)
        or not resolved_devices
        or any(not isinstance(device, str) or not device for device in resolved_devices)
        or not isinstance(callback_devices, list)
        or not callback_devices
        or any(not isinstance(device, str) or not device for device in callback_devices)
        or backend.get("local_cpu_callback_available") is not True
        or backend.get("passed") is not True
    ):
        raise ValueError("GPU backend resolution contract failed")

    xla = _require_mapping(config.get("xla_autotune_policy"), "GPU XLA policy")
    if set(xla) != {
        "requested_cli_platform",
        "xla_flags_before",
        "xla_flags_effective",
        "policy",
        "configured_before_training_stack_import",
        "correctness_check_enabled",
        "mismatching_autotune_candidates_disqualified",
        "cpu_mode_did_not_set_xla_flags",
        "passed",
    } or (
        xla.get("requested_cli_platform") != "gpu"
        or xla.get("xla_flags_before") is not None
        or xla.get("xla_flags_effective") != H4_GPU_XLA_FLAGS
        or xla.get("policy") != H4_GPU_XLA_POLICY
        or xla.get("configured_before_training_stack_import") is not True
        or xla.get("correctness_check_enabled") is not True
        or xla.get("mismatching_autotune_candidates_disqualified") is not True
        or xla.get("cpu_mode_did_not_set_xla_flags") is not True
        or xla.get("passed") is not True
    ):
        raise ValueError("GPU correctness-checked level-4 XLA policy failed")

    callback = _require_mapping(
        config.get("debug_callback_preflight"), "GPU callback preflight"
    )
    if set(callback) != {
        "input",
        "callback_observed",
        "result",
        "local_cpu_callback_executed",
        "passed",
    } or (
        callback.get("input") != 2.0
        or callback.get("callback_observed") != 2.0
        or callback.get("result") != 3.0
        or callback.get("local_cpu_callback_executed") is not True
        or callback.get("passed") is not True
    ):
        raise ValueError("GPU local-CPU callback preflight failed")

    pre = _require_mapping(
        config.get("pre_training_device_audits"), "pre-training GPU audits"
    )
    if set(pre) != {"probe_state", "restore_params"}:
        raise ValueError("pre-training GPU audit field set drifted")
    probe = _require_exact_gpu_device_audit(
        pre.get("probe_state"),
        label="pre-training probe-state GPU audit",
        expected_label="pre_training_probe_state",
    )
    restore = _require_exact_gpu_device_audit(
        pre.get("restore_params"),
        label="pre-training restore-params GPU audit",
        expected_label="pre_training_restore_params",
    )
    post = _require_exact_gpu_device_audit(
        result.get("post_training_device_audit"),
        label="post-training params GPU audit",
        expected_label="post_training_params",
    )

    checkpoint = _require_mapping(
        config.get("checkpoint_compatibility"), "training checkpoint audit"
    )
    post_checkpoint = _require_mapping(
        result.get("post_training_checkpoint_audit"),
        "post-training checkpoint audit",
    )
    if (
        checkpoint.get("passed") is not True
        or checkpoint.get("source_actor_width") != 101
        or checkpoint.get("target_actor_width") != H4_ACTOR_OBSERVATION_WIDTH
        or checkpoint.get("source_critic_width") != 212
        or checkpoint.get("target_critic_width") != H4_CRITIC_OBSERVATION_WIDTH
        or checkpoint.get("inserted_feature_count") != 15
        or checkpoint.get("insert_offset") != 101
        or checkpoint.get("actor_new_15_rows_exact_zero") is not True
        or checkpoint.get("critic_new_15_rows_exact_zero") is not True
        or checkpoint.get("all_restore_leaves_finite") is not True
        or post_checkpoint.get("passed") is not True
        or post_checkpoint.get("source_actor_width")
        != H4_ACTOR_OBSERVATION_WIDTH
        or post_checkpoint.get("target_actor_width")
        != H4_ACTOR_OBSERVATION_WIDTH
        or post_checkpoint.get("critic_width") != H4_CRITIC_OBSERVATION_WIDTH
        or post_checkpoint.get("normalizer_state_width")
        != H4_ACTOR_OBSERVATION_WIDTH
        or post_checkpoint.get("normalizer_privileged_width")
        != H4_CRITIC_OBSERVATION_WIDTH
        or post_checkpoint.get("all_restore_leaves_finite") is not True
        or post_checkpoint.get("restore_structure_validated") is not True
        or post_checkpoint.get("transplant_applied") is not False
    ):
        raise ValueError("H4 training checkpoint provenance failed")

    cross_bound = (
        manifest.get("backend_resolution") == backend
        and result.get("backend_resolution") == backend
        and manifest.get("xla_autotune_policy") == xla
        and result.get("xla_autotune_policy") == xla
        and manifest.get("debug_callback_preflight") == callback
        and result.get("debug_callback_preflight") == callback
        and manifest.get("pre_training_device_audits") == pre
        and result.get("pre_training_device_audits") == pre
        and manifest.get("checkpoint_compatibility") == checkpoint
        and result.get("checkpoint_compatibility") == checkpoint
        and manifest.get("jax_devices") == resolved_devices
    )
    if not cross_bound:
        raise ValueError("GPU provenance differs across config/manifest/result")
    return {
        "schema_version": 1,
        "training_execution_provider": "JAX_GPU",
        "platform": "gpu",
        "manifest_jax_devices": list(resolved_devices),
        "backend_resolution": dict(backend),
        "xla_autotune_policy": dict(xla),
        "debug_callback_preflight": dict(callback),
        "pre_training_device_audits": {
            "probe_state": probe,
            "restore_params": restore,
        },
        "post_training_device_audit": post,
        "checkpoint_compatibility": dict(checkpoint),
        "post_training_checkpoint_audit": dict(post_checkpoint),
        "cross_bound_config_manifest_result": True,
        "passed": True,
    }


@dataclass(frozen=True)
class TrustedH4Bundle:
    """Validated immutable runner output, before or after params restoration."""

    params_path: Path
    params_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    config_canonical_sha256: str
    config: Mapping[str, Any]
    source_hashes: Mapping[str, Mapping[str, str]]
    source_hashes_canonical_sha256: str
    status: str
    run_name: str
    expert: str
    activity: str
    training_provenance: Mapping[str, Any]
    training_provenance_sha256: str
    source_closure_audit: Mapping[str, Any] | None = None

    def candidate_record(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "expert": self.expert,
            "activity": self.activity,
            "status": self.status,
            "final_params_path": str(self.params_path),
            "final_params_sha256": self.params_sha256,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "resolved_config_path": str(self.config_path),
            "resolved_config_sha256": self.config_sha256,
            "resolved_config_canonical_sha256": self.config_canonical_sha256,
            "source_and_teacher_hashes_sha256": (
                self.source_hashes_canonical_sha256
            ),
            "training_provenance_sha256": self.training_provenance_sha256,
        }


def _validated_forward_iteration_v2_source_paths(
    *,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Path]:
    """Reconstruct the authorized forward-v2 causal source closure.

    The authorization file is pinned independently.  Its failed-run name,
    causal hashes, and relative strict-evaluation path are the authority for
    the three historical inputs; neither config nor manifest may introduce a
    path or hash.  The exact records must then agree across authorization,
    resolved config, manifest source snapshot, and the current files.
    """

    raw_enabled = config.get("forward_iteration_v2", False)
    if not isinstance(raw_enabled, bool):
        raise ValueError("forward iteration-v2 flag must be boolean")
    recorded_labels = {
        label
        for label in source_hashes
        if label.startswith("forward_iteration_v2_")
    }
    if not raw_enabled:
        if recorded_labels:
            raise ValueError(
                "forward iteration-v2 source label set drifted while disabled"
            )
        if config.get("forward_iteration_v2_authorization") is not None:
            raise ValueError(
                "forward iteration-v2 authorization present while disabled"
            )
        return {}

    if recorded_labels != H4_FORWARD_ITERATION_V2_SOURCE_LABELS:
        raise ValueError(
            "forward iteration-v2 source label set drifted: "
            f"expected={sorted(H4_FORWARD_ITERATION_V2_SOURCE_LABELS)}, "
            f"actual={sorted(recorded_labels)}"
        )
    if (
        config.get("expert") != "forward"
        or config.get("reverse_iteration_v2") is not False
        or config.get("forward_iteration_v3_touchdown_balance", False) is not False
        or config.get("reverse_iteration_v3_no_target_imitation", False) is not False
        or config.get(
            "forward_iteration_v4_contact_event_validity_persistence", False
        ) is not False
        or config.get(
            "reverse_iteration_v4_residual_transfer_gain_024", False
        ) is not False
        or config.get("authorized_iteration_v2_250k_contract_id")
        != H4_FORWARD_ITERATION_V2_CONTRACT_ID
        or config.get("authorized_iteration_v4_250k_contract_id") is not None
    ):
        raise ValueError("forward iteration-v2 activation contract drifted")

    authorization_config = _require_mapping(
        config.get("forward_iteration_v2_authorization"),
        "forward iteration-v2 authorization config",
    )
    expected_authorization_config_keys = {
        "path",
        "sha256",
        "contract_id",
        "status",
        "semantic_audit",
        "bound_causal_inputs",
        "scope",
        "adoption_release_hardware",
    }
    if set(authorization_config) != expected_authorization_config_keys:
        raise ValueError("forward iteration-v2 authorization config field set drifted")
    authorization_path = _recorded_path(
        authorization_config, "forward iteration-v2 authorization config"
    )
    authorization_sha = require_sha256(
        authorization_config.get("sha256"),
        "forward iteration-v2 authorization config SHA256",
    )
    authorization_source = _require_mapping(
        source_hashes.get("forward_iteration_v2_authorization"),
        "forward iteration-v2 authorization source",
    )
    if (
        authorization_path.name != "h4_forward_iteration_v2_authorization.json"
        or authorization_path.parent.name != "artifacts"
        or authorization_path
        != authorization_path.parent.parent
        / "artifacts"
        / "h4_forward_iteration_v2_authorization.json"
        or authorization_sha
        != PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256
        or authorization_source
        != {
            "path": str(authorization_path),
            "sha256": PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256,
        }
        or sha256_file(authorization_path)
        != PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256
    ):
        raise ValueError("forward iteration-v2 authorization path/SHA drifted")

    semantic_audit = _require_mapping(
        authorization_config.get("semantic_audit"),
        "forward iteration-v2 semantic audit",
    )
    expected_semantic_audit_keys = {
        "schema",
        "kind",
        "status",
        "hardware_prohibited",
        "authorization_exact",
        "contract_id",
        "required_flag",
        "training_exact",
        "curriculum_exact",
        "reward_scales_exact",
        "reward_deltas_exact",
        "force_band_exact",
        "force_tail_exact",
        "contact_pulse_exact",
        "strict_gate_unchanged",
        "central_hashes_exact",
        "manifest_binding_exact",
    }
    if (
        set(semantic_audit) != expected_semantic_audit_keys
        or any(value is not True for value in semantic_audit.values())
        or authorization_config.get("contract_id")
        != H4_FORWARD_ITERATION_V2_CONTRACT_ID
        or authorization_config.get("status")
        != "AUTHORIZED_SIMULATION_250K_ONLY"
        or authorization_config.get("scope")
        != "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY"
        or authorization_config.get("adoption_release_hardware") != "PROHIBITED"
    ):
        raise ValueError("forward iteration-v2 authorization audit drifted")

    authorization = _require_mapping(
        load_json_strict(authorization_path), "forward iteration-v2 authorization"
    )
    scope = _require_mapping(
        authorization.get("scope"), "forward iteration-v2 authorization scope"
    )
    permissions = _require_mapping(
        authorization.get("authorization"),
        "forward iteration-v2 authorization permissions",
    )
    training = _require_mapping(
        authorization.get("training_contract"),
        "forward iteration-v2 training contract",
    )
    strict_gate = _require_mapping(
        authorization.get("strict_gate_contract"),
        "forward iteration-v2 strict gate",
    )
    if (
        authorization.get("schema_version") != 1
        or authorization.get("artifact_kind")
        != "openduckmini_h4_forward_iteration_v2_authorization"
        or authorization.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or authorization.get("hardware_deployment") != "PROHIBITED"
        or scope.get("contract_id") != H4_FORWARD_ITERATION_V2_CONTRACT_ID
        or scope.get("expert") != "forward"
        or scope.get("required_cli_flag") != "--forward-iteration-v2"
        or permissions
        != {
            "candidate_adoption": False,
            "hardware": False,
            "release": False,
            "simulation_1m_training": False,
            "simulation_250k_training": True,
        }
        or training.get("actor_observation_width")
        != H4_ACTOR_OBSERVATION_WIDTH
        or training.get("num_timesteps") != H4_PILOT_INTERACTIONS
        or training.get("seed") != H4_STRICT_SEEDS["forward"][0]
        or training.get("initialization")
        != "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT"
        or training.get("pinned_v22_parent_tree_sha256")
        != PINNED_V22_PARENT_TREE_SHA256
        or training.get("h4_parent_checkpoint_allowed") is not False
        or strict_gate.get("promotion_requires_all_three_fixed_six_second_seeds")
        is not True
        or strict_gate.get("thresholds_may_be_weakened") is not False
    ):
        raise ValueError("forward iteration-v2 authorization semantics drifted")

    causal = _require_mapping(
        authorization.get("causal_input"),
        "forward iteration-v2 causal input",
    )
    failed_run_name = causal.get("failed_candidate_run_name")
    if (
        not isinstance(failed_run_name, str)
        or not failed_run_name
        or Path(failed_run_name).name != failed_run_name
    ):
        raise ValueError("forward iteration-v2 failed run name drifted")
    evidence = _require_mapping(
        causal.get("integrated_strict_evaluation"),
        "forward iteration-v2 integrated strict evaluation",
    )
    raw_evidence_path = evidence.get("path")
    if (
        not isinstance(raw_evidence_path, str)
        or not raw_evidence_path
        or _is_absolute_path_text(raw_evidence_path)
    ):
        raise ValueError(
            "forward iteration-v2 evidence path must be one relative path"
        )
    experiment_root = authorization_path.parent.parent.resolve()
    failed_run_dir = (
        experiment_root
        / "artifacts"
        / "h4_training_runs"
        / "forward"
        / failed_run_name
    ).resolve()
    evidence_path = (experiment_root / raw_evidence_path).resolve()
    expected_records = {
        "forward_iteration_v2_authorization": {
            "path": str(authorization_path),
            "sha256": PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256,
        },
        "forward_iteration_v2_failed_candidate_manifest": {
            "path": str((failed_run_dir / "run_manifest.json").resolve()),
            "sha256": require_sha256(
                causal.get("failed_candidate_manifest_sha256"),
                "forward iteration-v2 failed manifest SHA256",
            ),
        },
        "forward_iteration_v2_failed_candidate_params": {
            "path": str((failed_run_dir / "final_params.pkl").resolve()),
            "sha256": require_sha256(
                causal.get("failed_candidate_final_params_sha256"),
                "forward iteration-v2 failed params SHA256",
            ),
        },
        "forward_iteration_v2_integrated_strict_evaluation": {
            "path": str(evidence_path),
            "sha256": require_sha256(
                evidence.get("sha256"),
                "forward iteration-v2 strict evaluation SHA256",
            ),
        },
    }
    if (
        evidence.get("fixed_seed_count") != 3
        or evidence.get("strict_pass_count") != 0
        or evidence.get("recomputed_validation_passed") is not True
    ):
        raise ValueError("forward iteration-v2 causal evaluation drifted")

    bound = _require_mapping(
        authorization_config.get("bound_causal_inputs"),
        "forward iteration-v2 bound causal inputs",
    )
    bound_to_source = {
        "failed_candidate_manifest": (
            "forward_iteration_v2_failed_candidate_manifest"
        ),
        "failed_candidate_params": "forward_iteration_v2_failed_candidate_params",
        "integrated_strict_evaluation": (
            "forward_iteration_v2_integrated_strict_evaluation"
        ),
    }
    if set(bound) != set(bound_to_source):
        raise ValueError("forward iteration-v2 bound causal label set drifted")
    for bound_label, source_label in bound_to_source.items():
        bound_record = _require_mapping(
            bound.get(bound_label),
            f"forward iteration-v2 bound causal input {bound_label}",
        )
        if set(bound_record) != {"path", "sha256"}:
            raise ValueError(
                f"forward iteration-v2 {bound_label} record field set drifted"
            )
        normalized_bound = {
            "path": str(_recorded_path(bound_record, bound_label)),
            "sha256": require_sha256(
                bound_record.get("sha256"),
                f"forward iteration-v2 {bound_label} SHA256",
            ),
        }
        if normalized_bound != expected_records[source_label]:
            raise ValueError(
                f"forward iteration-v2 {bound_label} authorization/config binding drifted"
            )

    for source_label, expected_record in expected_records.items():
        actual_record = _require_mapping(
            source_hashes.get(source_label),
            f"forward iteration-v2 manifest source {source_label}",
        )
        if actual_record != expected_record:
            raise ValueError(
                f"forward iteration-v2 manifest path/SHA drifted for {source_label}"
            )
        if sha256_file(Path(expected_record["path"])) != expected_record["sha256"]:
            raise ValueError(
                f"forward iteration-v2 current source SHA256 drifted for {source_label}"
            )
    return {
        label: Path(record["path"]).resolve()
        for label, record in expected_records.items()
    }


def _validated_reverse_iteration_v2_source_paths(
    *,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Path]:
    """Reconstruct the authorized reverse-v2 causal source closure exactly."""

    raw_enabled = config.get("reverse_iteration_v2", False)
    if not isinstance(raw_enabled, bool):
        raise ValueError("reverse iteration-v2 flag must be boolean")
    recorded_labels = {
        label
        for label in source_hashes
        if label.startswith("reverse_iteration_v2_")
    }
    if not raw_enabled:
        if recorded_labels:
            raise ValueError(
                "reverse iteration-v2 source label set drifted while disabled"
            )
        if config.get("reverse_iteration_v2_authorization") is not None:
            raise ValueError(
                "reverse iteration-v2 authorization present while disabled"
            )
        return {}

    if recorded_labels != H4_REVERSE_ITERATION_V2_SOURCE_LABELS:
        raise ValueError(
            "reverse iteration-v2 source label set drifted: "
            f"expected={sorted(H4_REVERSE_ITERATION_V2_SOURCE_LABELS)}, "
            f"actual={sorted(recorded_labels)}"
        )
    if (
        config.get("expert") != "reverse"
        or config.get("forward_iteration_v2") is not False
        or config.get("forward_iteration_v3_touchdown_balance", False) is not False
        or config.get("reverse_iteration_v3_no_target_imitation", False) is not False
        or config.get(
            "forward_iteration_v4_contact_event_validity_persistence", False
        ) is not False
        or config.get(
            "reverse_iteration_v4_residual_transfer_gain_024", False
        ) is not False
        or config.get("authorized_iteration_v2_250k_contract_id")
        != H4_REVERSE_ITERATION_V2_CONTRACT_ID
        or config.get("authorized_iteration_v4_250k_contract_id") is not None
    ):
        raise ValueError("reverse iteration-v2 activation contract drifted")

    authorization_config = _require_mapping(
        config.get("reverse_iteration_v2_authorization"),
        "reverse iteration-v2 authorization config",
    )
    expected_authorization_config_keys = {
        "path",
        "sha256",
        "contract_id",
        "status",
        "semantic_audit",
        "bound_causal_inputs",
        "legacy_reward_config_audit",
        "scope",
        "adoption_release_hardware",
    }
    if set(authorization_config) != expected_authorization_config_keys:
        raise ValueError("reverse iteration-v2 authorization config field set drifted")
    authorization_path = _recorded_path(
        authorization_config, "reverse iteration-v2 authorization config"
    )
    authorization_sha = require_sha256(
        authorization_config.get("sha256"),
        "reverse iteration-v2 authorization config SHA256",
    )
    authorization_source = _require_mapping(
        source_hashes.get("reverse_iteration_v2_authorization"),
        "reverse iteration-v2 authorization source",
    )
    if (
        authorization_path.name != "h4_reverse_iteration_v2_authorization.json"
        or authorization_path.parent.name != "artifacts"
        or authorization_path
        != authorization_path.parent.parent
        / "artifacts"
        / "h4_reverse_iteration_v2_authorization.json"
        or authorization_sha
        != PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256
        or authorization_source
        != {
            "path": str(authorization_path),
            "sha256": PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256,
        }
        or sha256_file(authorization_path)
        != PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256
    ):
        raise ValueError("reverse iteration-v2 authorization path/SHA drifted")

    semantic_audit = _require_mapping(
        authorization_config.get("semantic_audit"),
        "reverse iteration-v2 semantic audit",
    )
    expected_semantic_audit_keys = {
        "schema",
        "kind",
        "status",
        "hardware_prohibited",
        "authorization_exact",
        "contract_id",
        "required_flag",
        "training_exact",
        "teacher_guard_exact",
        "legacy_reward_exact",
        "tracking_sigma_truthful",
        "legacy_schema3_causal_truth",
        "curriculum_exact",
        "reward_scales_exact",
        "new_force_pulse_disabled",
        "strict_gate_unchanged",
        "central_hashes_exact",
        "manifest_binding_exact",
    }
    if (
        set(semantic_audit) != expected_semantic_audit_keys
        or any(value is not True for value in semantic_audit.values())
        or authorization_config.get("contract_id")
        != H4_REVERSE_ITERATION_V2_CONTRACT_ID
        or authorization_config.get("status")
        != "AUTHORIZED_SIMULATION_250K_ONLY"
        or authorization_config.get("scope")
        != "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY"
        or authorization_config.get("adoption_release_hardware") != "PROHIBITED"
    ):
        raise ValueError("reverse iteration-v2 authorization audit drifted")

    expected_legacy_reward = {
        "target_imitation": -20.0,
        "contact_imitation": 15.0,
        "tracking_sigma": 0.01,
        "backward_residual_scale": H4_REVERSE_RESIDUAL_SCALE,
    }
    legacy_reward_audit = _require_mapping(
        authorization_config.get("legacy_reward_config_audit"),
        "reverse iteration-v2 legacy reward audit",
    )
    if (
        set(legacy_reward_audit) != {"expected", "per_environment", "passed"}
        or legacy_reward_audit.get("expected") != expected_legacy_reward
        or legacy_reward_audit.get("per_environment")
        != {
            "train": expected_legacy_reward,
            "eval": expected_legacy_reward,
        }
        or legacy_reward_audit.get("passed") is not True
    ):
        raise ValueError("reverse iteration-v2 legacy reward config drifted")

    authorization = _require_mapping(
        load_json_strict(authorization_path), "reverse iteration-v2 authorization"
    )
    scope = _require_mapping(
        authorization.get("scope"), "reverse iteration-v2 authorization scope"
    )
    permissions = _require_mapping(
        authorization.get("authorization"),
        "reverse iteration-v2 authorization permissions",
    )
    training = _require_mapping(
        authorization.get("training_contract"),
        "reverse iteration-v2 training contract",
    )
    strict_gate = _require_mapping(
        authorization.get("strict_gate_contract"),
        "reverse iteration-v2 strict gate",
    )
    teacher_guard = _require_mapping(
        authorization.get("teacher_and_guard_contract"),
        "reverse iteration-v2 teacher/guard contract",
    )
    reward = _require_mapping(
        authorization.get("reward_contract"),
        "reverse iteration-v2 reward contract",
    )
    if (
        authorization.get("schema_version") != 1
        or authorization.get("artifact_kind")
        != "openduckmini_h4_reverse_iteration_v2_authorization"
        or authorization.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or authorization.get("hardware_deployment") != "PROHIBITED"
        or scope.get("contract_id") != H4_REVERSE_ITERATION_V2_CONTRACT_ID
        or scope.get("expert") != "reverse"
        or scope.get("required_cli_flag") != "--reverse-iteration-v2"
        or permissions
        != {
            "candidate_adoption": False,
            "hardware": False,
            "release": False,
            "simulation_1m_training": False,
            "simulation_250k_training": True,
        }
        or training.get("actor_observation_width")
        != H4_ACTOR_OBSERVATION_WIDTH
        or training.get("num_timesteps") != H4_PILOT_INTERACTIONS
        or training.get("seed") != H4_STRICT_SEEDS["reverse"][0]
        or training.get("initialization")
        != "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT"
        or training.get("pinned_v22_parent_tree_sha256")
        != PINNED_V22_PARENT_TREE_SHA256
        or training.get("h4_parent_checkpoint_allowed") is not False
        or strict_gate.get("promotion_requires_all_three_fixed_six_second_seeds")
        is not True
        or strict_gate.get("thresholds_may_be_weakened") is not False
        or teacher_guard.get("selected_teacher_sha256")
        != PINNED_SELECTED_REVERSE_TEACHER_SHA256
        or teacher_guard.get("reverse_composition_authorization_sha256")
        != PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
        or teacher_guard.get("reverse_minimum_spec_sha256")
        != PINNED_REVERSE_MINIMUM_SPEC_SHA256
        or teacher_guard.get("backward_residual_scale")
        != H4_REVERSE_RESIDUAL_SCALE
        or teacher_guard.get("entry_phase_bins")
        != H4_REVERSE_TEACHER_ENTRY_PHASE_BINS
        or teacher_guard.get("phase_advance_bins_per_control")
        != H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS
        or teacher_guard.get("target_guard_changed") is not False
        or teacher_guard.get("teacher_composition_changed") is not False
        or reward.get("new_force_and_pulse_scales_explicitly_disabled") is not True
    ):
        raise ValueError("reverse iteration-v2 authorization semantics drifted")

    causal = _require_mapping(
        authorization.get("causal_input"),
        "reverse iteration-v2 causal input",
    )
    failed_run_name = causal.get("failed_candidate_run_name")
    if (
        not isinstance(failed_run_name, str)
        or not failed_run_name
        or Path(failed_run_name).name != failed_run_name
    ):
        raise ValueError("reverse iteration-v2 failed run name drifted")
    evidence = _require_mapping(
        causal.get("integrated_strict_evaluation"),
        "reverse iteration-v2 integrated strict evaluation",
    )
    raw_evidence_path = evidence.get("path")
    if (
        not isinstance(raw_evidence_path, str)
        or not raw_evidence_path
        or _is_absolute_path_text(raw_evidence_path)
    ):
        raise ValueError(
            "reverse iteration-v2 evidence path must be one relative path"
        )
    experiment_root = authorization_path.parent.parent.resolve()
    failed_run_dir = (
        experiment_root
        / "artifacts"
        / "h4_training_runs"
        / "reverse"
        / failed_run_name
    ).resolve()
    evidence_path = (experiment_root / raw_evidence_path).resolve()
    expected_records = {
        "reverse_iteration_v2_authorization": {
            "path": str(authorization_path),
            "sha256": PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256,
        },
        "reverse_iteration_v2_failed_candidate_manifest": {
            "path": str((failed_run_dir / "run_manifest.json").resolve()),
            "sha256": require_sha256(
                causal.get("failed_candidate_manifest_sha256"),
                "reverse iteration-v2 failed manifest SHA256",
            ),
        },
        "reverse_iteration_v2_failed_candidate_params": {
            "path": str((failed_run_dir / "final_params.pkl").resolve()),
            "sha256": require_sha256(
                causal.get("failed_candidate_final_params_sha256"),
                "reverse iteration-v2 failed params SHA256",
            ),
        },
        "reverse_iteration_v2_integrated_strict_evaluation": {
            "path": str(evidence_path),
            "sha256": require_sha256(
                evidence.get("sha256"),
                "reverse iteration-v2 strict evaluation SHA256",
            ),
        },
    }
    speed_range = evidence.get("steady_reverse_speed_ratio_range")
    if (
        evidence.get("fixed_seed_count") != 3
        or evidence.get("strict_pass_count") != 0
        or evidence.get("recomputed_validation_passed") is not True
        or evidence.get("legacy_schema3_composition_trace_complete") is not False
        or evidence.get("safety_trace_used_for_qualification") is not False
        or evidence.get("causal_basis")
        != "GAIT_QUALITY_0_OF_3_AND_STEADY_REVERSE_SPEED_ONLY"
        or not isinstance(speed_range, list)
        or len(speed_range) != 2
        or not all(isinstance(value, (int, float)) for value in speed_range)
        or not (0.0 <= float(speed_range[0]) <= float(speed_range[1]))
    ):
        raise ValueError("reverse iteration-v2 causal evaluation drifted")

    bound = _require_mapping(
        authorization_config.get("bound_causal_inputs"),
        "reverse iteration-v2 bound causal inputs",
    )
    bound_to_source = {
        "failed_candidate_manifest": (
            "reverse_iteration_v2_failed_candidate_manifest"
        ),
        "failed_candidate_params": "reverse_iteration_v2_failed_candidate_params",
        "integrated_strict_evaluation": (
            "reverse_iteration_v2_integrated_strict_evaluation"
        ),
    }
    if set(bound) != set(bound_to_source):
        raise ValueError("reverse iteration-v2 bound causal label set drifted")
    for bound_label, source_label in bound_to_source.items():
        bound_record = _require_mapping(
            bound.get(bound_label),
            f"reverse iteration-v2 bound causal input {bound_label}",
        )
        if set(bound_record) != {"path", "sha256"}:
            raise ValueError(
                f"reverse iteration-v2 {bound_label} record field set drifted"
            )
        normalized_bound = {
            "path": str(_recorded_path(bound_record, bound_label)),
            "sha256": require_sha256(
                bound_record.get("sha256"),
                f"reverse iteration-v2 {bound_label} SHA256",
            ),
        }
        if normalized_bound != expected_records[source_label]:
            raise ValueError(
                f"reverse iteration-v2 {bound_label} authorization/config binding drifted"
            )

    for source_label, expected_record in expected_records.items():
        actual_record = _require_mapping(
            source_hashes.get(source_label),
            f"reverse iteration-v2 manifest source {source_label}",
        )
        if actual_record != expected_record:
            raise ValueError(
                f"reverse iteration-v2 manifest path/SHA drifted for {source_label}"
            )
        if sha256_file(Path(expected_record["path"])) != expected_record["sha256"]:
            raise ValueError(
                f"reverse iteration-v2 current source SHA256 drifted for {source_label}"
            )
    return {
        label: Path(record["path"]).resolve()
        for label, record in expected_records.items()
    }


def _iteration_v3_spec(expert: str) -> dict[str, Any]:
    if expert == "forward":
        return {
            "flag": "forward_iteration_v3_touchdown_balance",
            "auth_key": "forward_iteration_v3_touchdown_balance_authorization",
            "prefix": "forward_iteration_v3_",
            "labels": H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_SOURCE_LABELS,
            "auth_label": "forward_iteration_v3_authorization",
            "auth_filename": "h4_forward_iteration_v3_touchdown_balance_authorization.json",
            "auth_sha": PINNED_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_AUTHORIZATION_SHA256,
            "contract": H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_CONTRACT_ID,
            "wiring_contract": H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_WIRING_CONTRACT_ID,
            "kind": "openduckmini_h4_forward_iteration_v3_touchdown_balance_authorization",
            "required_flag": "--forward-iteration-v3-touchdown-balance",
            "change_family": "TOUCHDOWN_COUNT_BALANCE_SCALE_ONLY",
            "failed_root": "artifacts/h4_iteration_v2_training_runs_20260809/forward",
            "semantic_keys": {
                "schema", "kind", "status", "hardware_prohibited",
                "top_level_fields_exact", "scope_fail_closed",
                "decision_fail_closed",
                "authorization_exact", "contract_id", "required_flag",
                "one_change_family", "training_exact", "curriculum_exact",
                "reward_scales_exact", "single_scale_delta_exact",
                "touchdown_formula_unchanged", "strict_gate_unchanged",
                "central_hashes_exact", "manifest_binding_exact",
            },
        }
    if expert == "reverse":
        return {
            "flag": "reverse_iteration_v3_no_target_imitation",
            "auth_key": "reverse_iteration_v3_no_target_imitation_authorization",
            "prefix": "reverse_iteration_v3_",
            "labels": H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_SOURCE_LABELS,
            "auth_label": "reverse_iteration_v3_authorization",
            "auth_filename": "h4_reverse_iteration_v3_no_target_imitation_authorization.json",
            "auth_sha": PINNED_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION_SHA256,
            "contract": H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_CONTRACT_ID,
            "wiring_contract": H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_WIRING_CONTRACT_ID,
            "kind": "openduckmini_h4_reverse_iteration_v3_no_target_imitation_authorization",
            "required_flag": "--reverse-iteration-v3-no-target-imitation",
            "change_family": "LEGACY_TARGET_IMITATION_SCALE_ONLY",
            "failed_root": "artifacts/h4_iteration_v2_training_runs_20260809/reverse",
            "semantic_keys": {
                "schema", "kind", "status", "hardware_prohibited",
                "top_level_fields_exact", "scope_fail_closed",
                "decision_fail_closed",
                "authorization_exact", "contract_id", "required_flag",
                "one_change_family", "training_exact", "teacher_guard_exact",
                "legacy_single_delta_exact", "curriculum_exact",
                "h4_reward_scales_unchanged", "directional_diagnostic_exact",
                "strict_gate_unchanged", "central_hashes_exact",
                "manifest_binding_exact",
            },
        }
    raise ValueError(f"unsupported H4 iteration-v3 expert: {expert!r}")


def _validated_iteration_v3_source_paths(
    *,
    expert: str,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Path]:
    """Reconstruct one v3 authorization and its three immutable causal inputs."""

    spec = _iteration_v3_spec(expert)
    raw_enabled = config.get(spec["flag"], False)
    if not isinstance(raw_enabled, bool):
        raise ValueError(f"{spec['flag']} must be boolean")
    recorded_labels = {
        label for label in source_hashes if label.startswith(spec["prefix"])
    }
    if not raw_enabled:
        if recorded_labels:
            raise ValueError(f"{spec['flag']} source labels present while disabled")
        if config.get(spec["auth_key"]) is not None:
            raise ValueError(f"{spec['auth_key']} present while disabled")
        return {}
    if recorded_labels != spec["labels"]:
        raise ValueError(
            f"{spec['flag']} source label set drifted: "
            f"expected={sorted(spec['labels'])}, actual={sorted(recorded_labels)}"
        )

    all_mode_flags = {
        "forward_iteration_v2": config.get("forward_iteration_v2", False),
        "reverse_iteration_v2": config.get("reverse_iteration_v2", False),
        "forward_iteration_v3_touchdown_balance": config.get(
            "forward_iteration_v3_touchdown_balance", False
        ),
        "reverse_iteration_v3_no_target_imitation": config.get(
            "reverse_iteration_v3_no_target_imitation", False
        ),
        "forward_iteration_v4_contact_event_validity_persistence": config.get(
            "forward_iteration_v4_contact_event_validity_persistence", False
        ),
        "reverse_iteration_v4_residual_transfer_gain_024": config.get(
            "reverse_iteration_v4_residual_transfer_gain_024", False
        ),
        "forward_v5_contact_pulse_abort_scale_only": config.get(
            "forward_v5_contact_pulse_abort_scale_only", False
        ),
        "reverse_iteration_v5_no_contact_imitation": config.get(
            "reverse_iteration_v5_no_contact_imitation", False
        ),
        "forward_iteration_v6_contact_abort_island_only": config.get(
            "forward_iteration_v6_contact_abort_island_only", False
        ),
        "reverse_iteration_v6_absolute_full_leg_targets": config.get(
            "reverse_iteration_v6_absolute_full_leg_targets", False
        ),
    }
    raw_wiring = config.get("wiring_only")
    if not isinstance(raw_wiring, bool):
        raise ValueError("H4 iteration-v3 wiring flag must be boolean")
    expected_execution_contract = (
        spec["wiring_contract"] if raw_wiring else spec["contract"]
    )
    expected_activity = (
        "PPO_WIRING_TRAINING" if raw_wiring else "PPO_PILOT_TRAINING"
    )
    if (
        config.get("expert") != expert
        or any(not isinstance(value, bool) for value in all_mode_flags.values())
        or sum(value is True for value in all_mode_flags.values()) != 1
        or config.get("authorized_iteration_v2_250k_contract_id") is not None
        or config.get("authorized_iteration_v3_250k_contract_id")
        != spec["contract"]
        or config.get("authorized_iteration_v4_250k_contract_id") is not None
        or config.get("training_contract_id") != expected_execution_contract
        or config.get("activity") != expected_activity
    ):
        raise ValueError(f"{spec['flag']} activation contract drifted")

    auth_config = _require_mapping(
        config.get(spec["auth_key"]), f"{spec['auth_key']} config"
    )
    expected_auth_config_keys = {
        "path", "sha256", "contract_id", "status", "semantic_audit",
        "bound_causal_inputs", "scope", "adoption_release_hardware",
    }
    if expert == "reverse":
        expected_auth_config_keys.add("legacy_reward_config_audit")
    if set(auth_config) != expected_auth_config_keys:
        raise ValueError(f"{spec['auth_key']} config field set drifted")
    auth_path = _recorded_path(auth_config, f"{spec['auth_key']} config")
    auth_source = _require_mapping(
        source_hashes.get(spec["auth_label"]), spec["auth_label"]
    )
    if (
        auth_path.name != spec["auth_filename"]
        or auth_path.parent.name != "artifacts"
        or auth_config.get("sha256") != spec["auth_sha"]
        or auth_config.get("contract_id") != spec["contract"]
        or auth_config.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or auth_config.get("scope")
        != "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY"
        or auth_config.get("adoption_release_hardware") != "PROHIBITED"
        or auth_source
        != {"path": str(auth_path), "sha256": spec["auth_sha"]}
        or sha256_file(auth_path) != spec["auth_sha"]
    ):
        raise ValueError(f"{spec['auth_key']} path/SHA/config binding drifted")
    semantic = _require_mapping(
        auth_config.get("semantic_audit"), f"{spec['auth_key']} semantic audit"
    )
    if set(semantic) != spec["semantic_keys"] or any(
        value is not True for value in semantic.values()
    ):
        raise ValueError(f"{spec['auth_key']} semantic audit drifted")

    authorization = _require_mapping(load_json_strict(auth_path), spec["auth_key"])
    scope = _require_mapping(authorization.get("scope"), "v3 authorization scope")
    training = _require_mapping(
        authorization.get("training_contract"), "v3 training contract"
    )
    permissions = _require_mapping(
        authorization.get("authorization"), "v3 authorization permissions"
    )
    strict_gate = _require_mapping(
        authorization.get("strict_gate_contract"), "v3 strict gate"
    )
    curriculum = _require_mapping(
        authorization.get("curriculum"), "v3 curriculum"
    )
    reward_contract = _require_mapping(
        authorization.get("reward_contract"), "v3 reward contract"
    )
    if (
        authorization.get("schema_version") != 1
        or authorization.get("artifact_kind") != spec["kind"]
        or authorization.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or authorization.get("hardware_deployment") != "PROHIBITED"
        or scope.get("expert") != expert
        or scope.get("contract_id") != spec["contract"]
        or scope.get("required_cli_flag") != spec["required_flag"]
        or scope.get("selected_change_family") != spec["change_family"]
        or permissions != {
            "simulation_250k_training": True,
            "simulation_1m_training": False,
            "candidate_adoption": False,
            "release": False,
            "hardware": False,
        }
        or training.get("initialization")
        != "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT"
        or training.get("pinned_v22_parent_tree_sha256")
        != PINNED_V22_PARENT_TREE_SHA256
        or training.get("actor_observation_width") != H4_ACTOR_OBSERVATION_WIDTH
        or training.get("seed") != H4_STRICT_SEEDS[expert][0]
        or training.get("num_timesteps") != H4_PILOT_INTERACTIONS
        or training.get("num_envs") != 1250
        or training.get("reset_noise_multiplier") != 1.0
        or training.get("h4_parent_checkpoint_allowed") is not False
        or strict_gate.get("thresholds_may_be_weakened") is not False
        or strict_gate.get("promotion_requires_all_three_fixed_six_second_seeds")
        is not True
    ):
        raise ValueError(f"{spec['auth_key']} authorization semantics drifted")
    expected_anchor = {
        "physical_primary": curriculum.get("physical_primary_mps_radps"),
        "policy_observation_anchor": curriculum.get("policy_observation_anchor"),
        "stand_probability": curriculum.get("stand_probability"),
        "exact_primary_probability": curriculum.get("exact_primary_probability"),
        "local_probability": curriculum.get("local_probability"),
        "local_vx_m_s": curriculum.get("local_vx_m_s"),
        "transition_probability": curriculum.get("transition_probability"),
        "transition_vx_m_s": curriculum.get("transition_vx_uniform_m_s"),
    }
    expected_optimizer = {
        "learning_rate": training.get("learning_rate"),
        "entropy_cost": training.get("entropy_cost"),
        "clipping_epsilon": training.get("clipping_epsilon"),
        "discounting": training.get("discounting"),
        "max_grad_norm": training.get("max_grad_norm"),
    }
    ppo = _require_mapping(config.get("ppo"), "v3 resolved PPO config")
    expected_qualification = (
        "WIRING_PREFLIGHT_ONLY_NOT_250K_QUALIFICATION"
        if config.get("wiring_only") is True
        else "AUTHORIZED_250K_PILOT"
    )
    if (
        config.get("seed") != training.get("seed")
        or config.get("anchor_config") != expected_anchor
        or config.get("reward_scales") != reward_contract.get("exact_scales")
        or config.get("reset_noise_multiplier") != 1.0
        or config.get("backward_residual_scale") != H4_REVERSE_RESIDUAL_SCALE
        or config.get("qualification_use") != expected_qualification
        or any(config.get(key) != value for key, value in expected_optimizer.items())
        or any(ppo.get(key) != value for key, value in expected_optimizer.items())
    ):
        raise ValueError(f"{spec['auth_key']} resolved training config drifted")
    if expert == "reverse":
        legacy = _require_mapping(
            authorization.get("legacy_reward_config"), "reverse v3 legacy reward"
        )
        legacy_audit = _require_mapping(
            auth_config.get("legacy_reward_config_audit"),
            "reverse v3 legacy reward audit",
        )
        expected_legacy_environment = {
            **_require_mapping(
                legacy.get("iteration_v3_exact"), "reverse v3 legacy exact"
            ),
            "backward_residual_scale": H4_REVERSE_RESIDUAL_SCALE,
        }
        if (
            legacy_audit.get("expected") != expected_legacy_environment
            or legacy_audit.get("per_environment")
            != {
                "train": expected_legacy_environment,
                "eval": expected_legacy_environment,
            }
            or legacy_audit.get("passed") is not True
        ):
            raise ValueError("reverse v3 actual legacy reward audit drifted")

    causal = _require_mapping(authorization.get("causal_input"), "v3 causal input")
    failed_run_name = causal.get("failed_candidate_run_name")
    if (
        causal.get("failed_candidate_root_relative_path") != spec["failed_root"]
        or not isinstance(failed_run_name, str)
        or not failed_run_name
        or Path(failed_run_name).name != failed_run_name
    ):
        raise ValueError(f"{spec['auth_key']} failed candidate identity drifted")
    evidence_record = _require_mapping(
        causal.get("integrated_strict_evaluation"), "v3 strict evidence record"
    )
    raw_evidence_path = evidence_record.get("path")
    if (
        not isinstance(raw_evidence_path, str)
        or not raw_evidence_path
        or _is_absolute_path_text(raw_evidence_path)
    ):
        raise ValueError("v3 strict evidence path must be relative")
    experiment_root = auth_path.parent.parent.resolve()
    failed_run_dir = (
        experiment_root / spec["failed_root"] / failed_run_name
    ).resolve()
    evidence_path = (experiment_root / raw_evidence_path).resolve()
    expected_records = {
        spec["auth_label"]: {
            "path": str(auth_path), "sha256": spec["auth_sha"]
        },
        f"{spec['prefix']}failed_candidate_manifest": {
            "path": str((failed_run_dir / "run_manifest.json").resolve()),
            "sha256": require_sha256(
                causal.get("failed_candidate_manifest_sha256"),
                "v3 failed candidate manifest SHA256",
            ),
        },
        f"{spec['prefix']}failed_candidate_params": {
            "path": str((failed_run_dir / "final_params.pkl").resolve()),
            "sha256": require_sha256(
                causal.get("failed_candidate_final_params_sha256"),
                "v3 failed candidate params SHA256",
            ),
        },
        f"{spec['prefix']}integrated_strict_evaluation": {
            "path": str(evidence_path),
            "sha256": require_sha256(
                evidence_record.get("sha256"), "v3 strict evidence SHA256"
            ),
        },
    }
    bound = _require_mapping(
        auth_config.get("bound_causal_inputs"), "v3 bound causal inputs"
    )
    bound_to_source = {
        "failed_candidate_manifest": f"{spec['prefix']}failed_candidate_manifest",
        "failed_candidate_params": f"{spec['prefix']}failed_candidate_params",
        "integrated_strict_evaluation": (
            f"{spec['prefix']}integrated_strict_evaluation"
        ),
    }
    if set(bound) != set(bound_to_source):
        raise ValueError("v3 bound causal input label set drifted")
    for bound_label, source_label in bound_to_source.items():
        record = _require_mapping(bound.get(bound_label), bound_label)
        normalized = {
            "path": str(_recorded_path(record, bound_label)),
            "sha256": require_sha256(record.get("sha256"), bound_label),
        }
        if set(record) != {"path", "sha256"} or normalized != expected_records[
            source_label
        ]:
            raise ValueError(f"v3 causal binding drifted for {bound_label}")
    for label, expected in expected_records.items():
        if source_hashes.get(label) != expected:
            raise ValueError(f"v3 manifest source drifted for {label}")
        if sha256_file(Path(expected["path"])) != expected["sha256"]:
            raise ValueError(f"v3 current source drifted for {label}")

    evidence = _require_mapping(load_json_strict(evidence_path), "v3 strict evidence")
    episodes = evidence.get("episodes")
    baseline_episodes = evidence.get("official_v22_baseline", {}).get("episodes")
    if (
        evidence.get("artifact_kind") != STRICT_ARTIFACT_KIND
        or evidence.get("candidate", {}).get("expert") != expert
        or evidence.get("candidate", {}).get("final_params_sha256")
        != causal.get("failed_candidate_final_params_sha256")
        or evidence.get("candidate", {}).get("manifest_sha256")
        != causal.get("failed_candidate_manifest_sha256")
        or evidence.get("evaluation_contract", {}).get("fixed_seeds")
        != list(H4_STRICT_SEEDS[expert])
        or not isinstance(episodes, list)
        or len(episodes) != 3
        or any(
            episode.get("h4_safety_acceptance", {}).get("passed") is not True
            or episode.get("gait_quality_acceptance", {}).get("passed") is not False
            or episode.get("strict_passed") is not False
            for episode in episodes
        )
        or evidence.get("summary", {}).get("passing_seed_count") != 0
        or evidence.get("summary", {}).get("recomputed_validation_passed") is not True
        or evidence.get("official_v22_baseline", {})
        .get("summary", {})
        .get("passing_seed_count") != 0
    ):
        raise ValueError("v3 strict evidence candidate/manifest/seed linkage drifted")
    if expert == "reverse":
        all_six = (
            list(episodes) + list(baseline_episodes)
            if isinstance(baseline_episodes, list) and len(baseline_episodes) == 3
            else []
        )
        evaluation_hashes = evidence.get("runtime_provenance", {}).get(
            "evaluation_source_hashes_pre", {}
        )
        if (
            len(all_six) != 6
            or any(
                item.get("reverse_composition_contract", {}).get("semantics")
                != H4_REVERSE_COMPOSITION_TRACE_SEMANTICS
                or item.get("reverse_composition_contract", {}).get(
                    "selected_reverse_teacher_sha256"
                ) != PINNED_SELECTED_REVERSE_TEACHER_SHA256
                or item.get("reverse_composition_contract", {}).get("residual_scale")
                != H4_REVERSE_RESIDUAL_SCALE
                for item in all_six
            )
            or evaluation_hashes.get("safe_gait_experts/h4_post_training.py")
            != evidence_record.get("causal_h4_post_training_sha256")
            or evaluation_hashes.get("scripts/evaluate_h4_training_candidate.py")
            != evidence_record.get("causal_h4_evaluator_sha256")
        ):
            raise ValueError("reverse v3 schema3 composition provenance drifted")
    return {
        label: Path(record["path"]).resolve()
        for label, record in expected_records.items()
    }


def _validated_forward_iteration_v3_touchdown_balance_source_paths(
    *, config: Mapping[str, Any], source_hashes: Mapping[str, Mapping[str, str]]
) -> dict[str, Path]:
    return _validated_iteration_v3_source_paths(
        expert="forward", config=config, source_hashes=source_hashes
    )


def _validated_reverse_iteration_v3_no_target_imitation_source_paths(
    *, config: Mapping[str, Any], source_hashes: Mapping[str, Mapping[str, str]]
) -> dict[str, Path]:
    return _validated_iteration_v3_source_paths(
        expert="reverse", config=config, source_hashes=source_hashes
    )


def _iteration_v4_spec(expert: str) -> dict[str, Any]:
    """Return the immutable identity of one fourth-iteration experiment."""

    if expert == "forward":
        return {
            "flag": "forward_iteration_v4_contact_event_validity_persistence",
            "auth_key": (
                "forward_iteration_v4_contact_event_validity_persistence_"
                "authorization"
            ),
            "prefix": "forward_iteration_v4_",
            "labels": (
                H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_SOURCE_LABELS
            ),
            "auth_filename": (
                "h4_forward_iteration_v4_contact_event_validity_"
                "persistence_authorization.json"
            ),
            "contract": (
                H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_CONTRACT_ID
            ),
            "wiring_contract": (
                H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_WIRING_CONTRACT_ID
            ),
            "kind": (
                "openduckmini_h4_forward_iteration_v4_contact_event_"
                "validity_persistence_authorization"
            ),
            "required_flag": (
                "--forward-iteration-v4-contact-event-validity-persistence"
            ),
            "change_family": (
                "CONTACT_EVENT_VALIDITY_PERSISTENCE_CORE_OPT_IN_WITH_V2_"
                "REWARD_BASELINE"
            ),
            "purpose": (
                "test a bounded substep contact-event validity and persistence "
                "state-machine after forward iteration v3 failed every unchanged "
                "strict seed"
            ),
            "failed_root": (
                "artifacts/h4_iteration_v3_training_runs_20260809/forward"
            ),
            "failed_run": (
                "h4_forward_250k_seed20260809_iteration_v3_touchdown_"
                "balance_level4_v1"
            ),
            "failed_params_sha": (
                "8946249b3531957166dc13005df7b2f25e50feefe03d78e9657e4724973e5dfa"
            ),
            "failed_manifest_sha": (
                "4dfef12700363ae9274e1e8d9371a3780bf4871a1fe2a03d1e806749cc7deb92"
            ),
            "previous_auth_relative": (
                "artifacts/h4_forward_iteration_v3_touchdown_balance_"
                "authorization.json"
            ),
            "previous_auth_sha": (
                PINNED_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_AUTHORIZATION_SHA256
            ),
            "strict_relative": (
                "artifacts/h4_iteration_v3_training_runs_20260809/forward/"
                "h4_forward_250k_seed20260809_iteration_v3_touchdown_balance_"
                "level4_v1/h4_integrated_strict_3x6s_v1.json"
            ),
            "strict_sha": (
                "3375ad29f0443ac95637c1970b73f355a8ae2ee856903a0a43f79b8c7d74fd0f"
            ),
            "safety_by_seed": {
                20_260_809: True,
                20_261_809: True,
                20_262_809: False,
            },
            "semantic_keys": {
                "top_level_fields_exact", "schema", "kind", "status",
                "hardware_prohibited",
                "authorization_exact", "contract_id", "required_flag",
                "change_family", "scope_fail_closed", "training_exact",
                "curriculum_exact",
                "causal_identity_exact", "source_closure_exact",
                "strict_gate_unchanged", "core_opt_in_exact",
                "v2_reward_baseline_exact", "hypothesis_only",
                "manifest_binding_exact", "decision_fail_closed",
            },
        }
    if expert == "reverse":
        return {
            "flag": "reverse_iteration_v4_residual_transfer_gain_024",
            "auth_key": (
                "reverse_iteration_v4_residual_transfer_gain_024_authorization"
            ),
            "prefix": "reverse_iteration_v4_",
            "labels": H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_SOURCE_LABELS,
            "auth_filename": (
                "h4_reverse_iteration_v4_residual_transfer_gain_024_"
                "authorization.json"
            ),
            "contract": H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_CONTRACT_ID,
            "wiring_contract": (
                H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_WIRING_CONTRACT_ID
            ),
            "kind": (
                "openduckmini_h4_reverse_iteration_v4_residual_transfer_"
                "gain_024_authorization"
            ),
            "required_flag": "--reverse-iteration-v4-residual-transfer-gain-024",
            "change_family": "BACKWARD_RESIDUAL_TRANSFER_GAIN_ONLY",
            "purpose": (
                "test one bounded residual transfer-gain exploration hypothesis "
                "after reverse iteration v3 remained non-propulsive on every "
                "unchanged strict seed"
            ),
            "failed_root": (
                "artifacts/h4_iteration_v3_training_runs_20260809/reverse"
            ),
            "failed_run": (
                "h4_reverse_250k_seed20260810_iteration_v3_no_target_"
                "imitation_level4_v1"
            ),
            "failed_params_sha": (
                "59871b9c35ea34ed3f62b8157d5afe8e2c8277cdc97e763c4a70dfafd8720414"
            ),
            "failed_manifest_sha": (
                "a80801d81118ed557b8b32426307543cd0d298dbc9d57837a6517d8e4b66c67c"
            ),
            "previous_auth_relative": (
                "artifacts/h4_reverse_iteration_v3_no_target_imitation_"
                "authorization.json"
            ),
            "previous_auth_sha": (
                PINNED_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION_SHA256
            ),
            "strict_relative": (
                "artifacts/h4_iteration_v3_training_runs_20260809/reverse/"
                "h4_reverse_250k_seed20260810_iteration_v3_no_target_"
                "imitation_level4_v1/h4_integrated_strict_3x6s_v1.json"
            ),
            "strict_sha": (
                "a52054327ec6c65326f4a869260cc4dd55b3935fe7375cededd3551f8b56ece2"
            ),
            "safety_by_seed": {
                20_260_810: True,
                20_265_810: True,
                20_271_810: True,
            },
            "semantic_keys": {
                "top_level_fields_exact", "schema", "kind", "status",
                "hardware_prohibited",
                "authorization_exact", "contract_id", "required_flag",
                "change_family", "scope_fail_closed", "training_exact",
                "curriculum_exact",
                "causal_identity_exact", "source_closure_exact",
                "strict_gate_unchanged", "teacher_gain_single_delta_exact",
                "legacy_reward_retained_exact", "h4_reward_retained_exact",
                "bounded_exploration_no_saturation_claim",
                "manifest_binding_exact", "decision_fail_closed",
            },
        }
    raise ValueError(f"unsupported H4 iteration-v4 expert: {expert!r}")


def _validated_iteration_v4_source_paths(
    *,
    expert: str,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Path]:
    """Rebuild v4 authority, failed-v3 evidence, and causal-source closure."""

    spec = _iteration_v4_spec(expert)
    raw_enabled = config.get(spec["flag"], False)
    if not isinstance(raw_enabled, bool):
        raise ValueError(f"{spec['flag']} must be boolean")
    recorded_labels = {
        label for label in source_hashes if label.startswith(spec["prefix"])
    }
    if not raw_enabled:
        if recorded_labels:
            raise ValueError(f"{spec['flag']} source labels present while disabled")
        if config.get(spec["auth_key"]) is not None:
            raise ValueError(f"{spec['auth_key']} present while disabled")
        return {}
    if recorded_labels != spec["labels"]:
        raise ValueError(
            f"{spec['flag']} source label set drifted: "
            f"expected={sorted(spec['labels'])}, actual={sorted(recorded_labels)}"
        )

    mode_flags = {
        "forward_iteration_v2": config.get("forward_iteration_v2", False),
        "reverse_iteration_v2": config.get("reverse_iteration_v2", False),
        "forward_iteration_v3_touchdown_balance": config.get(
            "forward_iteration_v3_touchdown_balance", False
        ),
        "reverse_iteration_v3_no_target_imitation": config.get(
            "reverse_iteration_v3_no_target_imitation", False
        ),
        "forward_iteration_v4_contact_event_validity_persistence": config.get(
            "forward_iteration_v4_contact_event_validity_persistence", False
        ),
        "reverse_iteration_v4_residual_transfer_gain_024": config.get(
            "reverse_iteration_v4_residual_transfer_gain_024", False
        ),
        "forward_v5_contact_pulse_abort_scale_only": config.get(
            "forward_v5_contact_pulse_abort_scale_only", False
        ),
        "reverse_iteration_v5_no_contact_imitation": config.get(
            "reverse_iteration_v5_no_contact_imitation", False
        ),
        "forward_iteration_v6_contact_abort_island_only": config.get(
            "forward_iteration_v6_contact_abort_island_only", False
        ),
        "reverse_iteration_v6_absolute_full_leg_targets": config.get(
            "reverse_iteration_v6_absolute_full_leg_targets", False
        ),
    }
    raw_wiring = config.get("wiring_only")
    if any(not isinstance(value, bool) for value in mode_flags.values()):
        raise ValueError("H4 iteration-v4 mode flags must be boolean")
    if not isinstance(raw_wiring, bool):
        raise ValueError("H4 iteration-v4 wiring flag must be boolean")
    expected_execution_contract = (
        spec["wiring_contract"] if raw_wiring else spec["contract"]
    )
    expected_activity = "PPO_WIRING_TRAINING" if raw_wiring else "PPO_PILOT_TRAINING"
    if (
        config.get("expert") != expert
        or sum(value is True for value in mode_flags.values()) != 1
        or config.get("authorized_iteration_v2_250k_contract_id") is not None
        or config.get("authorized_iteration_v3_250k_contract_id") is not None
        or config.get("authorized_iteration_v4_250k_contract_id") != spec["contract"]
        or config.get("training_contract_id") != expected_execution_contract
        or config.get("activity") != expected_activity
        or config.get("forward_v4_substep_contact") is not (expert == "forward")
    ):
        raise ValueError(f"{spec['flag']} activation contract drifted")

    auth_config = _require_mapping(
        config.get(spec["auth_key"]), f"{spec['auth_key']} config"
    )
    expected_auth_keys = {
        "path", "sha256", "contract_id", "status", "semantic_audit",
        "bound_causal_inputs", "bound_causal_sources", "scope",
        "adoption_release_hardware",
    }
    if expert == "reverse":
        expected_auth_keys.add("legacy_reward_config_audit")
    if set(auth_config) != expected_auth_keys:
        raise ValueError(f"{spec['auth_key']} config field set drifted")
    auth_path = _recorded_path(auth_config, f"{spec['auth_key']} config")
    auth_sha = require_sha256(auth_config.get("sha256"), "v4 authorization SHA256")
    auth_label = f"{spec['prefix']}authorization"
    if (
        auth_path.name != spec["auth_filename"]
        or auth_path.parent.name != "artifacts"
        or auth_config.get("contract_id") != spec["contract"]
        or auth_config.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or auth_config.get("scope")
        != "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY"
        or auth_config.get("adoption_release_hardware") != "PROHIBITED"
        or source_hashes.get(auth_label)
        != {"path": str(auth_path), "sha256": auth_sha}
        or sha256_file(auth_path) != auth_sha
    ):
        raise ValueError(f"{spec['auth_key']} path/SHA/config binding drifted")
    semantic = _require_mapping(
        auth_config.get("semantic_audit"), f"{spec['auth_key']} semantic audit"
    )
    if set(semantic) != spec["semantic_keys"] or any(
        value is not True for value in semantic.values()
    ):
        raise ValueError(f"{spec['auth_key']} semantic audit drifted")

    authorization = _require_mapping(load_json_strict(auth_path), spec["auth_key"])
    scope = _require_mapping(authorization.get("scope"), "v4 authorization scope")
    causal = _require_mapping(authorization.get("causal_input"), "v4 causal input")
    strict_record = _require_mapping(
        causal.get("integrated_strict_evaluation"), "v4 strict evidence record"
    )
    previous_record = _require_mapping(
        causal.get("previous_iteration_authorization"),
        "v4 previous authorization record",
    )
    training = _require_mapping(
        authorization.get("training_contract"), "v4 training contract"
    )
    curriculum = _require_mapping(authorization.get("curriculum"), "v4 curriculum")
    reward = _require_mapping(authorization.get("reward_contract"), "v4 reward")
    strict_gate = _require_mapping(
        authorization.get("strict_gate_contract"), "v4 strict gate"
    )
    permissions = _require_mapping(
        authorization.get("authorization"), "v4 permissions"
    )
    manifest_binding = _require_mapping(
        authorization.get("manifest_binding"), "v4 manifest binding"
    )
    decision = _require_mapping(authorization.get("decision"), "v4 decision")
    experiment_root = auth_path.parent.parent.resolve()
    previous_path = (experiment_root / spec["previous_auth_relative"]).resolve()
    strict_path = (experiment_root / spec["strict_relative"]).resolve()
    failed_run_dir = (
        experiment_root / spec["failed_root"] / spec["failed_run"]
    ).resolve()
    expected_inputs = {
        "previous_iteration_authorization": {
            "path": str(previous_path), "sha256": spec["previous_auth_sha"]
        },
        "failed_candidate_params": {
            "path": str((failed_run_dir / "final_params.pkl").resolve()),
            "sha256": spec["failed_params_sha"],
        },
        "failed_candidate_manifest": {
            "path": str((failed_run_dir / "run_manifest.json").resolve()),
            "sha256": spec["failed_manifest_sha"],
        },
        "integrated_strict_evaluation": {
            "path": str(strict_path), "sha256": spec["strict_sha"]
        },
    }
    expected_top_level = {
        "schema_version", "artifact_kind", "status", "hardware_deployment",
        "authorization", "scope", "causal_input", "training_contract",
        "curriculum", "reward_contract", "strict_gate_contract",
        "manifest_binding", "decision", "causal_source_closure",
        *( {"core_contract"} if expert == "forward" else {
            "teacher_and_guard_contract", "legacy_reward_config"
        } ),
    }
    expected_manifest_binding = {
        "authorization_artifact_sha256_required": True,
        "resolved_config_contract_id_required": True,
        **(
            {
                "core_opt_in_required": True,
                "source_semantic_preflight_config_manifest_result_binding_required": True,
                "single_authority_runtime_config_manifest_result_binding_required": True,
            }
            if expert == "forward"
            else {"teacher_guard_legacy_reward_config_required": True}
        ),
        "source_hash_snapshot_pre_and_post_required": True,
        "source_and_authorization_unchanged_required": True,
        "final_params_and_result_sha256_required": True,
    }
    if (
        set(authorization) != expected_top_level
        or
        authorization.get("schema_version") != 1
        or authorization.get("artifact_kind") != spec["kind"]
        or authorization.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or authorization.get("hardware_deployment") != "PROHIBITED"
        or permissions
        != {
            "simulation_250k_training": True,
            "simulation_1m_training": False,
            "candidate_adoption": False,
            "release": False,
            "hardware": False,
        }
        or scope.get("expert") != expert
        or scope.get("contract_id") != spec["contract"]
        or scope.get("required_cli_flag") != spec["required_flag"]
        or scope.get("selected_change_family") != spec["change_family"]
        or scope.get("purpose") != spec["purpose"]
        or scope.get("training_launch_performed_by_this_artifact") is not False
        or causal.get("failed_candidate_root_relative_path") != spec["failed_root"]
        or causal.get("failed_candidate_run_name") != spec["failed_run"]
        or causal.get("failed_candidate_final_params_sha256")
        != spec["failed_params_sha"]
        or causal.get("failed_candidate_manifest_sha256")
        != spec["failed_manifest_sha"]
        or previous_record
        != {"path": spec["previous_auth_relative"], "sha256": spec["previous_auth_sha"]}
        or strict_record.get("path") != spec["strict_relative"]
        or strict_record.get("sha256") != spec["strict_sha"]
        or strict_record.get("fixed_seed_count") != 3
        or strict_record.get("strict_pass_count") != 0
        or strict_record.get("safety_pass_count")
        != sum(spec["safety_by_seed"].values())
        or strict_record.get("official_v22_strict_pass_count") != 0
        or strict_record.get("recomputed_validation_passed") is not True
        or training.get("initialization")
        != "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT"
        or training.get("pinned_v22_parent_tree_sha256")
        != PINNED_V22_PARENT_TREE_SHA256
        or training.get("actor_observation_width") != H4_ACTOR_OBSERVATION_WIDTH
        or training.get("seed") != H4_STRICT_SEEDS[expert][0]
        or training.get("num_timesteps") != H4_PILOT_INTERACTIONS
        or training.get("num_envs") != 1250
        or training.get("reset_noise_multiplier") != 1.0
        or training.get("h4_parent_checkpoint_allowed") is not False
        or strict_gate.get("thresholds_may_be_weakened") is not False
        or strict_gate.get("promotion_requires_all_three_fixed_six_second_seeds")
        is not True
        or strict_gate.get("central_evaluator_sha256")
        != sha256_file(experiment_root / "scripts/evaluate_routed_transitions.py")
        or strict_gate.get("central_gait_quality_sha256")
        != sha256_file(experiment_root / "safe_gait_experts/gait_quality.py")
        or strict_gate.get("central_routed_evaluation_sha256")
        != sha256_file(experiment_root / "safe_gait_experts/routed_evaluation.py")
        or manifest_binding != expected_manifest_binding
        or decision
        != {
            "next_authorized_action": (
                "RUN_ONE_UNIQUE_SIMULATION_250K_AFTER_NO_PPO_WIRING_AND_"
                "PROVENANCE_AUDIT"
            ),
            "training_launch": "NOT_PERFORMED",
            "candidate_adoption": "BLOCKED",
            "release": "BLOCKED",
            "hardware": "PROHIBITED",
        }
    ):
        raise ValueError(f"{spec['auth_key']} authorization semantics drifted")

    previous_authorization = _require_mapping(
        load_json_strict(previous_path), "v3 authorization"
    )
    previous_training = _require_mapping(
        previous_authorization.get("training_contract"), "v3 training contract"
    )
    previous_curriculum = _require_mapping(
        previous_authorization.get("curriculum"), "v3 curriculum"
    )
    previous_reward = _require_mapping(
        previous_authorization.get("reward_contract"), "v3 reward"
    )
    if training != previous_training or curriculum != previous_curriculum:
        raise ValueError("iteration-v4 optimizer/curriculum drifted from iteration-v3")
    expected_reward_scales = dict(
        _require_mapping(previous_reward.get("exact_scales"), "v3 exact scales")
    )
    if expert == "forward":
        expected_reward_scales["h4_touchdown_count_balance"] = -2.0
        expected_core = {
            "factory_argument": "forward_v4_substep_contact",
            "exact_value": True,
            "legacy_default": False,
            "scope": "FORWARD_ITERATION_V4_ONLY",
            "substep_telemetry": {
                "interval_count_per_control_tick": 10,
                "interval_duration_s": 0.002,
                "runtime_authoritative_source": (
                    "SINGLE_INSTRUMENTED_TEN_SUBSTEP_SCAN_ENDPOINT"
                ),
                "instrumented_physics_source": (
                    "TEN_SINGLE_SUBSTEP_SCAN_EXACT_REPLACE_CTRL_THEN_MJX_STEP"
                ),
                "saved_dynamic_state_fields": [
                    "qpos",
                    "qvel",
                    "act",
                    "ctrl",
                    "time",
                    "qacc_warmstart",
                ],
                "saved_dynamic_state_topology": {
                    "substep_count": 10,
                    "field_count": 6,
                },
                "telemetry_source": "POST_PHYSICS_SAVED_DYNAMIC6_REPLAY",
                "measurement_state_coherence": {
                    "operation": (
                        "MJX_FORWARD_TELEMETRY_ONLY_AFTER_INSTRUMENTED_"
                        "PHYSICS_SCAN_COMPLETES"
                    ),
                    "reconstruction_base": (
                        "IMMUTABLE_CONTROL_ENTRY_DATA_PLUS_SAVED_DYNAMIC6"
                    ),
                    "measurement_uses_forwarded_copy": True,
                    "instrumented_scan_carry_and_endpoint_use_unforwarded_integrated_state": True,
                },
            },
            "source_semantic_theorem": {
                "runtime_physics_authority_count": 1,
                "official_source_wrapper_role": "ONCE_ONLY_PRE_PPO_REFERENCE",
                "official_source_wrapper_executed_inside_ppo": False,
                "official_source_provenance": {
                    "both_files_must_resolve_under_requested_source_root": True,
                    "joystick_relative_path": (
                        "playground/open_duck_mini_v2/joystick.py"
                    ),
                    "joystick_sha256": (
                        "95890569d971725308b5a9c0996bfa5fd9520479f014f325e810aa1db272eb9d"
                    ),
                    "mjx_env_relative_path": (
                        ".venv/lib/python3.12/site-packages/"
                        "mujoco_playground/_src/mjx_env.py"
                    ),
                    "mjx_env_sha256": (
                        "c3f1cfe0de036c3ccbba46e8cdd661cb48bfea8f182955298205f17787f53dfe"
                    ),
                    "step_source_sha256": (
                        "26571e7510b2837dca07f69890dc26a89695dff4caa1fdc6a0d6736bd22da06b"
                    ),
                    "step_source_semantics": (
                        "LAX_SCAN_XS_EMPTY_LENGTH_NSUBSTEPS_BODY_REPLACE_CTRL_"
                        "ACTION_THEN_MJX_STEP_RETURN_FINAL_CARRY"
                    ),
                },
                "preflight_probe_contract": {
                    "seed": 20260809,
                    "reset_noise_multiplier": 1.0,
                    "initial_state_source": "ENV_RESET_JAX_PRNGKEY_SEED",
                    "action_shape": [14],
                    "action_dtype": "float32",
                    "action_all_zero": True,
                    "observed_reference_count": 1,
                },
                "qualifying_dynamic_state_fields": [
                    "qpos",
                    "qvel",
                    "act",
                    "ctrl",
                    "time",
                    "qacc_warmstart",
                ],
                "qualifying_exact_required": True,
                "qualifying_max_abs_error_required": 0.0,
                "qualifying_field_count_required": 6,
                "excluded_derived_diagnostics": ["cfrc_int", "cfrc_ext"],
                "excluded_diagnostics_role": (
                    "NON_QUALIFYING_OBSERVED_DIAGNOSTICS_ONLY"
                ),
                "exclusion_is_semantic_not_tolerance": True,
                "numeric_tolerance_used": False,
                "post_physics_telemetry_may_modify_authoritative_endpoint": False,
            },
            "runtime_authority_assertion": {
                "endpoint_vs_saved_final_dynamic6_exact_required": True,
                "max_abs_error_required": 0.0,
                "dynamic_field_count_required": 6,
                "dynamic_field_count_exact_metric_required": True,
                "saved_substep_count_required": 10,
                "saved_dynamic_field_count_required": 6,
                "saved_dynamic_field_count_exact_metric_required": True,
                "saved_dynamic_all_finite_required": True,
                "telemetry_force_shape_required": [2],
                "telemetry_force_all_finite_required": True,
                "episode_field_count_exact_totals_equal_length": True,
                "diagnostic_count_totals_qualification_role": (
                    "NON_QUALIFYING_FINITE_DIAGNOSTICS_ONLY"
                ),
                "host_count_multiplication_for_qualification": False,
                "numeric_tolerance_used": False,
                "compiled_platforms": ["cpu", "cuda"],
                "vmap_batch_aggregation_required": True,
                "success_callback_count_per_vmap_batch": 0,
                "maximum_failure_callback_count_per_vmap_batch": 1,
                "failure_callback": (
                    "CONDITIONAL_UNORDERED_CALLBACK_WITH_RETAINED_TOKEN"
                ),
                "once_only_source_semantic_preflight_required": True,
                "no_ppo_and_ppo_output_closure_required": True,
            },
            "raw_force_schmitt": {
                "on_threshold": 0.01,
                "off_threshold": 0.005,
                "separate_carried_raw_state": True,
            },
            "qualification_state": {
                "qualified_q_integer_interval_horizon": 20,
                "pending_integer_interval_horizon": 20,
                "symmetric_island_gap_sum": True,
                "confirmed_events_only": [
                    "touchdown", "touchdown_count", "alternation"
                ],
            },
            "event_loss": {
                "span_intervals_0_0ms": 1.0,
                "span_intervals_10_20ms": 0.25,
                "span_intervals_20_40ms": 0.0,
                "terminal_pending_event": "RIGHT_CENSORED_NO_LOSS",
            },
            "reset": {
                "measured_on_initializes_contact": True,
                "phantom_event_prohibited": True,
            },
            "state_machine_formula_source_changed": True,
            "all_legacy_and_prior_iteration_paths_unchanged": True,
        }
        if (
            authorization.get("core_contract") != expected_core
            or reward.get("baseline") != "FORWARD_ITERATION_V2_EXACT"
            or reward.get("exact_scales") != expected_reward_scales
            or reward.get("touchdown_count_balance")
            != {
                "iteration_v3_scale": -4.0,
                "iteration_v4_scale": -2.0,
                "iteration_v4_matches_iteration_v2": True,
            }
            or reward.get("all_other_scales_match_iteration_v2") is not True
            or causal.get("hypothesis")
            != {
                "classification": "BOUNDED_CONTACT_EVENT_STATE_HYPOTHESIS_ONLY",
                "statement": (
                    "substep-qualified contact state may prevent transient contact "
                    "samples from manufacturing touchdown events and may align event "
                    "rewards with the strict debounce measurements"
                ),
                "verified_by_existing_evidence": False,
                "diagnostic_does_not_authorize_promotion": True,
            }
            or manifest_binding.get("core_opt_in_required") is not True
        ):
            raise ValueError("forward iteration-v4 single-family contract drifted")
    else:
        previous_legacy = _require_mapping(
            previous_authorization.get("legacy_reward_config"), "v3 legacy reward"
        )
        expected_legacy = dict(
            _require_mapping(previous_legacy.get("iteration_v3_exact"), "v3 legacy exact")
        )
        teacher = _require_mapping(
            authorization.get("teacher_and_guard_contract"), "v4 teacher contract"
        )
        legacy = _require_mapping(
            authorization.get("legacy_reward_config"), "v4 legacy reward"
        )
        hypothesis = _require_mapping(causal.get("hypothesis"), "v4 hypothesis")
        if (
            teacher.get("selected_teacher_sha256")
            != PINNED_SELECTED_REVERSE_TEACHER_SHA256
            or teacher.get("cadence_hz") != 1.5
            or teacher.get("entry_phase_bins") != H4_REVERSE_TEACHER_ENTRY_PHASE_BINS
            or teacher.get("phase_advance_bins_per_control")
            != H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS
            or teacher.get("backward_residual_scale_iteration_v3")
            != H4_REVERSE_RESIDUAL_SCALE
            or teacher.get("backward_residual_scale_iteration_v4")
            != H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN
            or teacher.get("only_delta") != "backward_residual_scale"
            or teacher.get("target_guard_changed") is not False
            or teacher.get("teacher_table_or_phase_changed") is not False
            or teacher.get("reverse_minimum_spec_sha256")
            != PINNED_REVERSE_MINIMUM_SPEC_SHA256
            or teacher.get("reverse_composition_authorization_sha256")
            != PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
            or legacy.get("iteration_v3_exact") != expected_legacy
            or legacy.get("iteration_v4_exact") != expected_legacy
            or legacy.get("identical_to_iteration_v3") is not True
            or reward.get("exact_scales") != expected_reward_scales
            or reward.get("identical_to_iteration_v3") is not True
            or reward.get("new_force_and_pulse_scales_explicitly_disabled") is not True
            or hypothesis.get("classification")
            != "BOUNDED_RESIDUAL_GAIN_EXPLORATION_HYPOTHESIS_ONLY"
            or hypothesis.get("statement")
            != (
                "raising the frozen-teacher residual transfer gain from 0.12 "
                "to 0.24 may expose a larger bounded residual action range for "
                "discovering propulsive reverse contact while the unchanged "
                "final target guard remains authoritative"
            )
            or hypothesis.get("verified_by_existing_evidence") is not False
            or hypothesis.get("action_saturation_observed_or_claimed") is not False
            or hypothesis.get("no_saturation_claim") is not True
            or hypothesis.get("diagnostic_does_not_authorize_promotion") is not True
            or manifest_binding.get("teacher_guard_legacy_reward_config_required")
            is not True
        ):
            raise ValueError("reverse iteration-v4 single-family contract drifted")

    expected_anchor = {
        "physical_primary": curriculum.get("physical_primary_mps_radps"),
        "policy_observation_anchor": curriculum.get("policy_observation_anchor"),
        "stand_probability": curriculum.get("stand_probability"),
        "exact_primary_probability": curriculum.get("exact_primary_probability"),
        "local_probability": curriculum.get("local_probability"),
        "local_vx_m_s": curriculum.get("local_vx_m_s"),
        "transition_probability": curriculum.get("transition_probability"),
        "transition_vx_m_s": curriculum.get("transition_vx_uniform_m_s"),
    }
    expected_optimizer = {
        key: training.get(key)
        for key in (
            "learning_rate", "entropy_cost", "clipping_epsilon",
            "discounting", "max_grad_norm",
        )
    }
    ppo = _require_mapping(config.get("ppo"), "v4 resolved PPO config")
    expected_qualification = (
        "WIRING_PREFLIGHT_ONLY_NOT_250K_QUALIFICATION"
        if raw_wiring
        else "AUTHORIZED_250K_PILOT"
    )
    expected_residual = (
        H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN
        if expert == "reverse"
        else H4_REVERSE_RESIDUAL_SCALE
    )
    if (
        config.get("seed") != training.get("seed")
        or config.get("anchor_config") != expected_anchor
        or config.get("reward_scales") != expected_reward_scales
        or config.get("reset_noise_multiplier") != 1.0
        or config.get("backward_residual_scale") != expected_residual
        or config.get("qualification_use") != expected_qualification
        or any(config.get(key) != value for key, value in expected_optimizer.items())
        or any(ppo.get(key) != value for key, value in expected_optimizer.items())
    ):
        raise ValueError(f"{spec['auth_key']} resolved training config drifted")
    if expert == "reverse":
        legacy_audit = _require_mapping(
            auth_config.get("legacy_reward_config_audit"),
            "reverse v4 legacy reward audit",
        )
        expected_environment = {
            **_require_mapping(
                authorization["legacy_reward_config"].get("iteration_v4_exact"),
                "reverse v4 legacy exact",
            ),
            "backward_residual_scale": H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN,
        }
        if (
            legacy_audit.get("expected") != expected_environment
            or legacy_audit.get("per_environment")
            != {"train": expected_environment, "eval": expected_environment}
            or legacy_audit.get("passed") is not True
        ):
            raise ValueError("reverse iteration-v4 actual legacy audit drifted")

    bound_inputs = _require_mapping(
        auth_config.get("bound_causal_inputs"), "v4 bound causal inputs"
    )
    if set(bound_inputs) != set(expected_inputs):
        raise ValueError("v4 bound causal input label set drifted")
    for label, expected in expected_inputs.items():
        record = _require_mapping(bound_inputs.get(label), label)
        normalized = {
            "path": str(_recorded_path(record, label)),
            "sha256": require_sha256(record.get("sha256"), label),
        }
        source_label = f"{spec['prefix']}{label}"
        if (
            set(record) != {"path", "sha256"}
            or normalized != expected
            or source_hashes.get(source_label) != expected
            or sha256_file(Path(expected["path"])) != expected["sha256"]
        ):
            raise ValueError(f"v4 causal binding drifted for {label}")

    auth_closure = _require_mapping(
        authorization.get("causal_source_closure"), "v4 causal source closure"
    )
    bound_sources = _require_mapping(
        auth_config.get("bound_causal_sources"), "v4 bound causal sources"
    )
    if set(auth_closure) != set(H4_ITERATION_V4_CAUSAL_SOURCE_PATHS) or set(
        bound_sources
    ) != set(H4_ITERATION_V4_CAUSAL_SOURCE_PATHS):
        raise ValueError("v4 causal source closure label set drifted")
    expected_records = {auth_label: {"path": str(auth_path), "sha256": auth_sha}}
    expected_records.update(
        {
            f"{spec['prefix']}{label}": record
            for label, record in expected_inputs.items()
        }
    )
    for label, relative in H4_ITERATION_V4_CAUSAL_SOURCE_PATHS.items():
        auth_record = _require_mapping(auth_closure.get(label), f"auth source {label}")
        config_record = _require_mapping(
            bound_sources.get(label), f"config source {label}"
        )
        expected_path = (experiment_root / relative).resolve()
        expected_sha = require_sha256(auth_record.get("sha256"), label)
        expected = {"path": str(expected_path), "sha256": expected_sha}
        source_label = f"{spec['prefix']}source_{label}"
        if (
            set(auth_record) != {"path", "sha256"}
            or auth_record.get("path") != relative
            or set(config_record) != {"path", "sha256"}
            or {"path": str(_recorded_path(config_record, label)), "sha256": config_record.get("sha256")}
            != expected
            or source_hashes.get(source_label) != expected
            or sha256_file(expected_path) != expected_sha
        ):
            raise ValueError(f"v4 causal source binding drifted for {label}")
        expected_records[source_label] = expected

    evidence = _require_mapping(load_json_strict(strict_path), "v4 strict evidence")
    episodes = evidence.get("episodes")
    if (
        evidence.get("artifact_kind") != STRICT_ARTIFACT_KIND
        or evidence.get("candidate", {}).get("expert") != expert
        or evidence.get("candidate", {}).get("final_params_sha256")
        != spec["failed_params_sha"]
        or evidence.get("candidate", {}).get("manifest_sha256")
        != spec["failed_manifest_sha"]
        or evidence.get("evaluation_contract", {}).get("fixed_seeds")
        != list(H4_STRICT_SEEDS[expert])
        or not isinstance(episodes, list)
        or len(episodes) != 3
        or {episode.get("seed"): episode.get("h4_safety_acceptance", {}).get("passed") for episode in episodes}
        != spec["safety_by_seed"]
        or any(
            episode.get("gait_quality_acceptance", {}).get("passed") is not False
            or episode.get("strict_passed") is not False
            for episode in episodes
        )
        or evidence.get("summary", {}).get("passing_seed_count") != 0
        or evidence.get("summary", {}).get("recomputed_validation_passed") is not True
        or evidence.get("official_v22_baseline", {}).get("summary", {}).get(
            "passing_seed_count"
        ) != 0
    ):
        raise ValueError("v4 strict evidence candidate/manifest/seed linkage drifted")
    return {label: Path(record["path"]).resolve() for label, record in expected_records.items()}


def _validated_forward_iteration_v4_contact_event_validity_persistence_source_paths(
    *, config: Mapping[str, Any], source_hashes: Mapping[str, Mapping[str, str]]
) -> dict[str, Path]:
    return _validated_iteration_v4_source_paths(
        expert="forward", config=config, source_hashes=source_hashes
    )


def _validated_reverse_iteration_v4_residual_transfer_gain_024_source_paths(
    *, config: Mapping[str, Any], source_hashes: Mapping[str, Mapping[str, str]]
) -> dict[str, Path]:
    return _validated_iteration_v4_source_paths(
        expert="reverse", config=config, source_hashes=source_hashes
    )


def _iteration_v5_spec(expert: str) -> dict[str, Any]:
    """Return exact post-training identity for one independent v5 family."""

    if expert == "forward":
        return {
            "flag": "forward_v5_contact_pulse_abort_scale_only",
            "auth_key": (
                "forward_iteration_v5_contact_pulse_abort_scale_only_authorization"
            ),
            "prefix": "forward_iteration_v5_",
            "auth_label": "forward_iteration_v5_authorization",
            "auth_filename": (
                "h4_forward_iteration_v5_contact_pulse_abort_scale_only_"
                "authorization.json"
            ),
            "auth_sha": (
                "c8a197e2b2eeb1b24cce1cace560841bd2620ee6bce5f97506c8c9f7518b210b"
            ),
            "kind": (
                "openduckmini_h4_forward_iteration_v5_contact_pulse_abort_"
                "scale_only_authorization"
            ),
            "family": "CONTACT_PULSE_ABORT_SCALE_ONLY",
            "contract": H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_CONTRACT_ID,
            "wiring_contract": (
                H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_WIRING_CONTRACT_ID
            ),
            "no_ppo_contract": (
                H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_NO_PPO_CONTRACT_ID
            ),
            "required_flag": "--forward-v5-contact-pulse-abort-scale-only",
            "v4_manifest_label": "forward_iteration_v5_v4_run_manifest.json",
            "v4_source_prefix": "forward_iteration_v4_source_",
            "causal_labels": {
                "previous_iteration_authorization": "forward_iteration_v5_v4_authorization",
                "final_params": "forward_iteration_v5_v4_final_params.pkl",
                "manifest": "forward_iteration_v5_v4_run_manifest.json",
                "resolved_config": "forward_iteration_v5_v4_resolved_config.json",
                "run_result": "forward_iteration_v5_v4_run_result.json",
                "training_curve": "forward_iteration_v5_v4_training_curve.csv",
                "integrated_strict_evaluation": (
                    "forward_iteration_v5_v4_h4_integrated_strict_3x6s_v1.json"
                ),
            },
        }
    if expert == "reverse":
        return {
            "flag": "reverse_iteration_v5_no_contact_imitation",
            "auth_key": "reverse_iteration_v5_no_contact_imitation_authorization",
            "prefix": "reverse_iteration_v5_",
            "auth_label": "reverse_iteration_v5_authorization",
            "auth_filename": "h4_reverse_iteration_v5_no_contact_imitation_authorization.json",
            "auth_sha": (
                "1a0da8b77110c92fdaa0a81cdc41879a1a45660456567dfe68f88b2b5deb5976"
            ),
            "kind": (
                "openduckmini_h4_reverse_iteration_v5_no_contact_imitation_"
                "authorization"
            ),
            "family": "LEGACY_CONTACT_IMITATION_SCALE_ONLY",
            "contract": H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_CONTRACT_ID,
            "wiring_contract": H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_WIRING_CONTRACT_ID,
            "no_ppo_contract": H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_NO_PPO_CONTRACT_ID,
            "required_flag": "--reverse-iteration-v5-no-contact-imitation",
            "v4_manifest_label": "reverse_iteration_v5_v4_run_manifest.json",
            "v4_source_prefix": "reverse_iteration_v4_source_",
            "causal_labels": {
                "v3_authorization": "reverse_iteration_v5_v3_authorization",
                "v3_final_params": "reverse_iteration_v5_v3_final_params.pkl",
                "v3_manifest": "reverse_iteration_v5_v3_run_manifest.json",
                "v3_integrated_strict_evaluation": (
                    "reverse_iteration_v5_v3_h4_integrated_strict_3x6s_v1.json"
                ),
                "rejected_v4_authorization": "reverse_iteration_v5_v4_authorization",
                "rejected_v4_final_params": "reverse_iteration_v5_v4_final_params.pkl",
                "rejected_v4_manifest": "reverse_iteration_v5_v4_run_manifest.json",
                "rejected_v4_resolved_config": (
                    "reverse_iteration_v5_v4_resolved_config.json"
                ),
                "rejected_v4_run_result": "reverse_iteration_v5_v4_run_result.json",
                "rejected_v4_training_curve": (
                    "reverse_iteration_v5_v4_training_curve.csv"
                ),
                "rejected_v4_diagnostic": (
                    "reverse_iteration_v5_v4_h4_integrated_strict_3x6s_v1.json"
                ),
                "diagnostic_adapter": "reverse_iteration_v5_diagnostic_adapter",
                "diagnostic_adapter_authorization": (
                    "reverse_iteration_v5_diagnostic_adapter_authorization"
                ),
            },
        }
    raise ValueError(f"unsupported H4 iteration-v5 expert: {expert!r}")


def _validated_iteration_v5_source_paths(
    *,
    expert: str,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Path]:
    """Validate v5 authority, evidence, and historical-v4/current-source closure."""

    spec = _iteration_v5_spec(expert)
    enabled = config.get(spec["flag"], False)
    if not isinstance(enabled, bool):
        raise ValueError(f"{spec['flag']} must be boolean")
    prefixed = {
        label: record
        for label, record in source_hashes.items()
        if label.startswith(spec["prefix"])
    }
    if not enabled:
        if prefixed or config.get(spec["auth_key"]) is not None:
            raise ValueError(f"disabled {expert} iteration-v5 bound source data")
        return {}

    mode_keys = (
        "forward_iteration_v2",
        "reverse_iteration_v2",
        "forward_iteration_v3_touchdown_balance",
        "reverse_iteration_v3_no_target_imitation",
        "forward_iteration_v4_contact_event_validity_persistence",
        "reverse_iteration_v4_residual_transfer_gain_024",
        "forward_v5_contact_pulse_abort_scale_only",
        "reverse_iteration_v5_no_contact_imitation",
    )
    mode_values = {key: config.get(key) for key in mode_keys}
    if (
        any(not isinstance(value, bool) for value in mode_values.values())
        or mode_values.get(spec["flag"]) is not True
        or sum(value is True for value in mode_values.values()) != 1
    ):
        raise ValueError(f"{expert} iteration-v5 eight-mode closure drifted")

    current_labels = {
        f"{spec['prefix']}current_source_{label}": relative
        for label, relative in H4_ITERATION_V4_CAUSAL_SOURCE_PATHS.items()
    }
    expected_labels = {
        spec["auth_label"],
        *spec["causal_labels"].values(),
        *current_labels,
    }
    if set(prefixed) != expected_labels:
        raise ValueError(
            f"{expert} iteration-v5 source label closure drifted: "
            f"{sorted(set(prefixed) ^ expected_labels)}"
        )
    auth_config = _require_mapping(
        config.get(spec["auth_key"]), f"{expert} iteration-v5 config authorization"
    )
    semantic_audit = _require_mapping(
        auth_config.get("semantic_audit"),
        f"{expert} iteration-v5 semantic audit",
    )
    expected_semantic_keys = {
        "top_level_fields_exact",
        "schema",
        "kind",
        "status",
        "hardware_prohibited",
        "authorization_exact",
        "scope_exact",
        "training_exact",
        "curriculum_exact",
        "historical_v4_source_closure_exact",
        "strict_gate_unchanged",
        "manifest_binding_exact",
        "decision_fail_closed",
        *(
            {
                "causal_identity_exact",
                "core_contract_exact",
                "reward_single_delta_exact",
            }
            if expert == "forward"
            else {
                "v3_causal_identity_exact",
                "rejected_v4_causal_identity_exact",
                "teacher_and_guard_exact",
                "legacy_reward_single_delta_exact",
                "h4_reward_unchanged",
                "diagnostic_never_promotion",
            }
        ),
    }
    expected_auth_keys = {
        "path",
        "sha256",
        "contract_id",
        "status",
        "semantic_audit",
        "bound_causal_inputs",
        "bound_historical_v4_sources",
        "scope",
        "adoption_release_hardware",
    }
    if expert == "reverse":
        expected_auth_keys.update(
            {"legacy_reward_config_audit", "rejected_v4_diagnostic_promotion_allowed"}
        )
    auth_source = _require_mapping(
        source_hashes.get(spec["auth_label"]),
        f"{expert} iteration-v5 authorization source",
    )
    auth_path = Path(str(auth_source.get("path", ""))).resolve()
    if (
        set(auth_config) != expected_auth_keys
        or set(semantic_audit) != expected_semantic_keys
        or any(value is not True for value in semantic_audit.values())
        or auth_path.name != spec["auth_filename"]
        or Path(str(auth_config.get("path", ""))).resolve() != auth_path
        or auth_source.get("sha256") != spec["auth_sha"]
        or auth_config.get("sha256") != spec["auth_sha"]
        or sha256_file(auth_path) != spec["auth_sha"]
        or auth_config.get("contract_id") != spec["contract"]
        or auth_config.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or auth_config.get("scope")
        != "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY"
        or auth_config.get("adoption_release_hardware") != "PROHIBITED"
    ):
        raise ValueError(f"{expert} iteration-v5 authorization binding drifted")
    authorization = _require_mapping(
        load_json_strict(auth_path), f"{expert} iteration-v5 authorization"
    )
    scope = _require_mapping(authorization.get("scope"), "v5 authorization scope")
    causal = _require_mapping(
        authorization.get("causal_inputs"), "v5 authorization causal inputs"
    )
    if (
        authorization.get("schema_version") != 1
        or authorization.get("artifact_kind") != spec["kind"]
        or authorization.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or authorization.get("hardware_deployment") != "PROHIBITED"
        or authorization.get("authorization")
        != {
            "simulation_250k_training": True,
            "simulation_1m_training": False,
            "candidate_adoption": False,
            "release": False,
            "hardware": False,
        }
        or scope.get("expert") != expert
        or scope.get("contract_id") != spec["contract"]
        or scope.get("wiring_contract_id") != spec["wiring_contract"]
        or scope.get("no_ppo_contract_id") != spec["no_ppo_contract"]
        or scope.get("required_cli_flag") != spec["required_flag"]
        or scope.get("selected_change_family") != spec["family"]
        or scope.get("training_launch_performed_by_this_artifact") is not False
    ):
        raise ValueError(f"{expert} iteration-v5 authorization semantics drifted")
    expected_execution_id = (
        spec["wiring_contract"] if config.get("wiring_only") is True else spec["contract"]
    )
    if (
        config.get("training_contract_id") != expected_execution_id
        or config.get("authorized_iteration_v5_250k_contract_id") != spec["contract"]
        or config.get("authorized_iteration_v2_250k_contract_id") is not None
        or config.get("authorized_iteration_v3_250k_contract_id") is not None
        or config.get("authorized_iteration_v4_250k_contract_id") is not None
        or config.get("initialization_source") != "V22_BRAX_CHECKPOINT"
        or config.get("trusted_h4_parent") is not None
        or config.get("pinned_v22_parent_tree_sha256")
        != PINNED_V22_PARENT_TREE_SHA256
        or config.get("reset_noise_multiplier") != 1.0
        or config.get("reward_scales")
        != _require_mapping(
            authorization.get("reward_contract"), "v5 reward contract"
        ).get("exact_scales")
    ):
        raise ValueError(f"{expert} iteration-v5 execution config drifted")

    paths: dict[str, Path] = {spec["auth_label"]: auth_path}
    bound_causal = _require_mapping(
        auth_config.get("bound_causal_inputs"), "v5 bound causal inputs"
    )
    if set(bound_causal) != {
        label.removeprefix(f"{spec['prefix']}")
        for label in spec["causal_labels"].values()
    }:
        raise ValueError(f"{expert} iteration-v5 bound causal key set drifted")
    for causal_key, source_label in spec["causal_labels"].items():
        source_record = _require_mapping(
            source_hashes.get(source_label), f"v5 causal source {source_label}"
        )
        bound_key = source_label.removeprefix(spec["prefix"])
        bound_record = _require_mapping(
            bound_causal.get(bound_key), f"v5 config causal binding {bound_key}"
        )
        causal_record = _require_mapping(
            causal.get(causal_key), f"v5 authorization causal {causal_key}"
        )
        source_path = Path(str(source_record.get("path", ""))).resolve()
        if (
            source_record.get("sha256") != causal_record.get("sha256")
            or bound_record.get("sha256") != causal_record.get("sha256")
            or Path(str(bound_record.get("path", ""))).resolve() != source_path
            or sha256_file(source_path) != causal_record.get("sha256")
        ):
            raise ValueError(f"{expert} iteration-v5 causal binding drifted: {causal_key}")
        paths[source_label] = source_path

    historical = _require_mapping(
        authorization.get("historical_v4_source_closure"),
        "v5 historical v4 source closure",
    )
    bound_historical = _require_mapping(
        auth_config.get("bound_historical_v4_sources"),
        "v5 bound historical v4 sources",
    )
    v4_manifest_path = paths[spec["v4_manifest_label"]]
    v4_manifest = _require_mapping(
        load_json_strict(v4_manifest_path), "v5 bound v4 manifest"
    )
    if (
        historical.get("verification_source")
        != "BOUND_V4_MANIFEST_PRE_POST_SNAPSHOT_NOT_CURRENT_FILES"
        or set(bound_historical) != set(H4_ITERATION_V4_CAUSAL_SOURCE_PATHS)
    ):
        raise ValueError(f"{expert} iteration-v5 historical source mode drifted")
    for label, relative in H4_ITERATION_V4_CAUSAL_SOURCE_PATHS.items():
        historical_record = _require_mapping(
            historical.get(label), f"v5 historical source {label}"
        )
        bound_record = _require_mapping(
            bound_historical.get(label), f"v5 bound historical source {label}"
        )
        manifest_label = f"{spec['v4_source_prefix']}{label}"
        pre_record = _require_mapping(
            _require_mapping(
                v4_manifest.get("source_and_teacher_hashes_pre"), "v4 pre hashes"
            ).get(manifest_label),
            f"v4 pre source {manifest_label}",
        )
        post_record = _require_mapping(
            _require_mapping(
                v4_manifest.get("source_and_teacher_hashes_post"), "v4 post hashes"
            ).get(manifest_label),
            f"v4 post source {manifest_label}",
        )
        if (
            historical_record.get("path") != relative
            or historical_record.get("sha256") != pre_record.get("sha256")
            or historical_record.get("sha256") != post_record.get("sha256")
            or bound_record.get("path") != relative
            or bound_record.get("sha256") != historical_record.get("sha256")
            or bound_record.get("manifest_pre") != pre_record
            or bound_record.get("manifest_post") != post_record
        ):
            raise ValueError(f"{expert} iteration-v5 historical source drifted: {label}")

    for source_label, relative in current_labels.items():
        record = _require_mapping(
            source_hashes.get(source_label), f"v5 current source {source_label}"
        )
        path = Path(str(record.get("path", ""))).resolve()
        normalized = PurePosixPath(str(path).replace("\\", "/")).as_posix()
        if (
            not normalized.endswith(relative)
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"{expert} iteration-v5 current source drifted: {source_label}")
        paths[source_label] = path

    reward = _require_mapping(authorization.get("reward_contract"), "v5 reward")
    if expert == "forward":
        delta = _require_mapping(reward.get("only_scale_delta"), "forward v5 delta")
        if (
            config.get("forward_v4_substep_contact") is not True
            or delta
            != {
                "name": "h4_contact_pulse_40ms",
                "iteration_v4_scale": -1.0,
                "iteration_v5_scale": -2.0,
            }
            or reward.get("all_other_scales_match_iteration_v4") is not True
            or reward.get("exact_scales", {}).get("h4_contact_pulse_40ms") != -2.0
        ):
            raise ValueError("forward iteration-v5 single scale/core contract drifted")
    else:
        legacy = _require_mapping(
            authorization.get("legacy_reward_config"), "reverse v5 legacy reward"
        )
        expected_legacy = {
            "target_imitation": 0.0,
            "contact_imitation": 0.0,
            "tracking_sigma": 0.01,
            "backward_residual_scale": 0.12,
        }
        audit = _require_mapping(
            auth_config.get("legacy_reward_config_audit"),
            "reverse v5 legacy reward audit",
        )
        diagnostic = _require_mapping(
            load_json_strict(paths[spec["causal_labels"]["rejected_v4_diagnostic"]]),
            "reverse v4 diagnostic",
        )
        if (
            config.get("forward_v4_substep_contact") is not False
            or config.get("backward_residual_scale") != 0.12
            or legacy.get("iteration_v3_baseline")
            != {"target_imitation": 0.0, "contact_imitation": 15.0, "tracking_sigma": 0.01}
            or legacy.get("iteration_v5_exact")
            != {"target_imitation": 0.0, "contact_imitation": 0.0, "tracking_sigma": 0.01}
            or legacy.get("only_scale_delta")
            != {
                "name": "contact_imitation",
                "iteration_v3_scale": 15.0,
                "iteration_v5_scale": 0.0,
            }
            or audit.get("expected") != expected_legacy
            or audit.get("per_environment")
            != {"train": expected_legacy, "eval": expected_legacy}
            or audit.get("passed") is not True
            or auth_config.get("rejected_v4_diagnostic_promotion_allowed") is not False
            or diagnostic.get("artifact_kind")
            != (
                "openduckmini_h4_reverse_iteration_v4_gain024_strict_"
                "evaluation_diagnostic"
            )
            or diagnostic.get("promotion_allowed") is not False
            or diagnostic.get("summary", {}).get("passing_seed_count") != 0
        ):
            raise ValueError("reverse iteration-v5 legacy/diagnostic contract drifted")
    return paths


def _validated_forward_iteration_v5_contact_pulse_abort_scale_only_source_paths(
    *, config: Mapping[str, Any], source_hashes: Mapping[str, Mapping[str, str]]
) -> dict[str, Path]:
    return _validated_iteration_v5_source_paths(
        expert="forward", config=config, source_hashes=source_hashes
    )


def _validated_reverse_iteration_v5_no_contact_imitation_source_paths(
    *, config: Mapping[str, Any], source_hashes: Mapping[str, Mapping[str, str]]
) -> dict[str, Path]:
    return _validated_iteration_v5_source_paths(
        expert="reverse", config=config, source_hashes=source_hashes
    )


def _iteration_v6_spec(expert: str) -> dict[str, Any]:
    """Return the exact post-training identity for one independent v6 family."""

    if expert == "forward":
        return {
            "flag": "forward_iteration_v6_contact_abort_island_only",
            "auth_key": "forward_iteration_v6_contact_abort_island_only_authorization",
            "prefix": "forward_iteration_v6_",
            "auth_label": "forward_iteration_v6_authorization",
            "auth_filename": "h4_forward_iteration_v6_contact_abort_island_only_authorization.json",
            "auth_sha": PINNED_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_AUTHORIZATION_SHA256,
            "kind": "openduckmini_h4_forward_iteration_v6_contact_abort_island_only_authorization",
            "family": "CONTACT_ABORT_TYPE_SEPARATION_ISLAND_ONLY",
            "contract": H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_CONTRACT_ID,
            "wiring_contract": H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_WIRING_CONTRACT_ID,
            "no_ppo_contract": H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_NO_PPO_CONTRACT_ID,
            "required_flag": "--forward-iteration-v6-contact-abort-island-only",
            "v5_manifest_label": (
                "forward_iteration_v6_rejected_iteration_v5_manifest"
            ),
            "v5_source_prefix": "forward_iteration_v5_current_source_",
            "causal_labels": {
                "iteration_v4_final_params": (
                    "forward_iteration_v6_iteration_v4_final_params"
                ),
                "iteration_v4_manifest": (
                    "forward_iteration_v6_iteration_v4_manifest"
                ),
                "iteration_v4_integrated_strict_evaluation": (
                    "forward_iteration_v6_"
                    "iteration_v4_integrated_strict_evaluation"
                ),
                "rejected_iteration_v5_final_params": (
                    "forward_iteration_v6_rejected_iteration_v5_final_params"
                ),
                "rejected_iteration_v5_manifest": (
                    "forward_iteration_v6_rejected_iteration_v5_manifest"
                ),
                "rejected_iteration_v5_integrated_strict_evaluation": (
                    "forward_iteration_v6_"
                    "rejected_iteration_v5_integrated_strict_evaluation"
                ),
            },
            "causal_roots": {
                "iteration_v4_final_params": "iteration_v4_candidate_root_relative_path",
                "iteration_v4_manifest": "iteration_v4_candidate_root_relative_path",
                "iteration_v4_integrated_strict_evaluation": (
                    "iteration_v4_candidate_root_relative_path"
                ),
                "rejected_iteration_v5_final_params": (
                    "rejected_iteration_v5_candidate_root_relative_path"
                ),
                "rejected_iteration_v5_manifest": (
                    "rejected_iteration_v5_candidate_root_relative_path"
                ),
                "rejected_iteration_v5_integrated_strict_evaluation": (
                    "rejected_iteration_v5_candidate_root_relative_path"
                ),
            },
        }
    if expert == "reverse":
        return {
            "flag": "reverse_iteration_v6_absolute_full_leg_targets",
            "auth_key": "reverse_iteration_v6_absolute_full_leg_targets_authorization",
            "prefix": "reverse_iteration_v6_",
            "auth_label": "reverse_iteration_v6_authorization",
            "auth_filename": "h4_reverse_iteration_v6_absolute_full_leg_targets_authorization.json",
            "auth_sha": PINNED_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_AUTHORIZATION_SHA256,
            "kind": "openduckmini_h4_reverse_iteration_v6_absolute_full_leg_targets_authorization",
            "family": "ABSOLUTE_FULL_LEG_TARGETS_WITH_TEACHER_TIMING_ONLY",
            "contract": H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_CONTRACT_ID,
            "wiring_contract": H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_WIRING_CONTRACT_ID,
            "no_ppo_contract": H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_NO_PPO_CONTRACT_ID,
            "required_flag": "--reverse-iteration-v6-absolute-full-leg-targets",
            "v5_manifest_label": (
                "reverse_iteration_v6_rejected_iteration_v5_manifest"
            ),
            "v5_source_prefix": "reverse_iteration_v5_current_source_",
            "causal_labels": {
                "iteration_v3_integrated_strict_evaluation": (
                    "reverse_iteration_v6_"
                    "iteration_v3_integrated_strict_evaluation"
                ),
                "rejected_iteration_v4_integrated_strict_evaluation": (
                    "reverse_iteration_v6_"
                    "rejected_iteration_v4_integrated_strict_evaluation"
                ),
                "rejected_iteration_v4_diagnostic_adapter": (
                    "reverse_iteration_v6_rejected_iteration_v4_diagnostic_adapter"
                ),
                "rejected_iteration_v4_diagnostic_adapter_authorization": (
                    "reverse_iteration_v6_"
                    "rejected_iteration_v4_diagnostic_adapter_authorization"
                ),
                "rejected_iteration_v5_final_params": (
                    "reverse_iteration_v6_rejected_iteration_v5_final_params"
                ),
                "rejected_iteration_v5_manifest": (
                    "reverse_iteration_v6_rejected_iteration_v5_manifest"
                ),
                "rejected_iteration_v5_integrated_strict_evaluation": (
                    "reverse_iteration_v6_"
                    "rejected_iteration_v5_integrated_strict_evaluation"
                ),
                "selected_reverse_teacher": (
                    "reverse_iteration_v6_selected_reverse_teacher"
                ),
            },
            "causal_roots": {
                "rejected_iteration_v5_final_params": (
                    "rejected_iteration_v5_candidate_root_relative_path"
                ),
                "rejected_iteration_v5_manifest": (
                    "rejected_iteration_v5_candidate_root_relative_path"
                ),
                "rejected_iteration_v5_integrated_strict_evaluation": (
                    "rejected_iteration_v5_candidate_root_relative_path"
                ),
            },
        }
    raise ValueError(f"unsupported H4 iteration-v6 expert: {expert!r}")


def _iteration_v6_expected_semantic_audit_keys(expert: str) -> set[str]:
    del expert
    return {
        "top_level_fields_exact",
        "nested_schema_exact",
        "numeric_types_and_values_exact",
        "causal_identity_exact",
        "training_contract_exact",
        "runtime_contract_exact",
        "passed",
    }


def _iteration_v6_causal_path(
    *,
    experiment_root: Path,
    causal: Mapping[str, Any],
    causal_key: str,
    root_key: str | None,
) -> Path:
    record = _require_mapping(causal.get(causal_key), f"v6 causal {causal_key}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"v6 causal {causal_key}.path is invalid")
    if root_key is None:
        return (experiment_root / raw_path).resolve()
    root_relative = causal.get(root_key)
    if not isinstance(root_relative, str) or not root_relative:
        raise ValueError(f"v6 causal root {root_key} is invalid")
    return (experiment_root / root_relative / raw_path).resolve()


def _validate_iteration_v6_evidence(
    *, expert: str, spec: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    for key, source_label in spec["causal_labels"].items():
        if "integrated_strict_evaluation" not in key:
            continue
        evidence = _require_mapping(
            load_json_strict(paths[source_label]), f"v6 strict evidence {key}"
        )
        summary = _require_mapping(evidence.get("summary"), f"v6 strict summary {key}")
        if (
            summary.get("passing_seed_count") != 0
            or summary.get("recomputed_validation_passed") is not True
        ):
            raise ValueError(f"{expert} iteration-v6 strict evidence drifted: {key}")
        if key == "rejected_iteration_v4_integrated_strict_evaluation":  # gitleaks:allow - symbolic artifact key
            if (
                evidence.get("artifact_kind")
                != "openduckmini_h4_reverse_iteration_v4_gain024_strict_evaluation_diagnostic"
                or evidence.get("promotion_allowed") is not False
            ):
                raise ValueError("reverse iteration-v6 rejected-v4 diagnostic drifted")
    if expert == "reverse":
        adapter_auth = _require_mapping(
            load_json_strict(
                paths[
                    spec["causal_labels"][
                        "rejected_iteration_v4_diagnostic_adapter_authorization"
                    ]
                ]
            ),
            "reverse v6 rejected-v4 adapter authorization",
        )
        if (
            adapter_auth.get("schema_version") != 1
            or adapter_auth.get("hardware_deployment") != "PROHIBITED"
            or adapter_auth.get("authorization", {}).get("promotion_evidence")
            is not False
            or adapter_auth.get("scope", {}).get("promotion_eligible") is not False
            or adapter_auth.get("status")
            != "AUTHORIZED_EXACT_DIAGNOSTIC_STRICT_EVALUATION_ONLY"
        ):
            raise ValueError("reverse iteration-v6 rejected-v4 adapter evidence drifted")


def _validated_iteration_v6_source_paths(
    *,
    expert: str,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Path]:
    """Validate v6 authorization, causal evidence, and source closure."""

    spec = _iteration_v6_spec(expert)
    enabled = config.get(spec["flag"], False)
    if not isinstance(enabled, bool):
        raise ValueError(f"{spec['flag']} must be boolean")
    prefixed = {
        label: record
        for label, record in source_hashes.items()
        if label.startswith(spec["prefix"])
    }
    if not enabled:
        if prefixed or config.get(spec["auth_key"]) is not None:
            raise ValueError(f"disabled {expert} iteration-v6 bound source data")
        return {}

    mode_keys = (
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
    mode_values = {key: config.get(key) for key in mode_keys}
    if (
        any(not isinstance(value, bool) for value in mode_values.values())
        or mode_values.get(spec["flag"]) is not True
        or sum(value is True for value in mode_values.values()) != 1
    ):
        raise ValueError(f"{expert} iteration-v6 ten-mode closure drifted")

    current_labels = {
        f"{spec['prefix']}current_source_{label}": relative
        for label, relative in H4_ITERATION_V4_CAUSAL_SOURCE_PATHS.items()
    }
    expected_labels = {
        spec["auth_label"],
        *spec["causal_labels"].values(),
        *current_labels,
    }
    if set(prefixed) != expected_labels:
        raise ValueError(
            f"{expert} iteration-v6 source label closure drifted: "
            f"{sorted(set(prefixed) ^ expected_labels)}"
        )

    auth_config = _require_mapping(
        config.get(spec["auth_key"]), f"{expert} iteration-v6 config authorization"
    )
    semantic_audit = _require_mapping(
        auth_config.get("semantic_audit"), f"{expert} iteration-v6 semantic audit"
    )
    expected_auth_config_keys = {
        "path",
        "sha256",
        "contract_id",
        "status",
        "semantic_audit",
        "bound_causal_inputs",
        "bound_historical_v5_sources",
        "scope",
        "adoption_release_hardware",
    }
    if expert == "reverse":
        expected_auth_config_keys.update(
            {
                "legacy_reward_config_audit",
                "h4_parent_checkpoint_allowed",
                "v4_gain_inherited",
                "v5_parent_checkpoint_inherited",
            }
        )
    auth_source = _require_mapping(
        source_hashes.get(spec["auth_label"]),
        f"{expert} iteration-v6 authorization source",
    )
    auth_path = Path(str(auth_source.get("path", ""))).resolve()
    if (
        set(auth_config) != expected_auth_config_keys
        or set(semantic_audit) != _iteration_v6_expected_semantic_audit_keys(expert)
        or any(value is not True for value in semantic_audit.values())
        or auth_path.name != spec["auth_filename"]
        or Path(str(auth_config.get("path", ""))).resolve() != auth_path
        or auth_source.get("sha256") != spec["auth_sha"]
        or auth_config.get("sha256") != spec["auth_sha"]
        or sha256_file(auth_path) != spec["auth_sha"]
        or auth_config.get("contract_id") != spec["contract"]
        or auth_config.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or auth_config.get("scope")
        != "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY"
        or auth_config.get("adoption_release_hardware") != "PROHIBITED"
        or (
            expert == "reverse"
            and (
                not _json_type_and_value_exact(
                    auth_config.get("legacy_reward_config_audit"),
                    {
                    "expected": {
                        "target_imitation": 0.0,
                        "contact_imitation": 0.0,
                        "tracking_sigma": 0.01,
                        "backward_residual_scale": 0.0,
                    },
                    "per_environment": {
                        "train": {
                            "target_imitation": 0.0,
                            "contact_imitation": 0.0,
                            "tracking_sigma": 0.01,
                            "backward_residual_scale": 0.0,
                        },
                        "eval": {
                            "target_imitation": 0.0,
                            "contact_imitation": 0.0,
                            "tracking_sigma": 0.01,
                            "backward_residual_scale": 0.0,
                        },
                    },
                        "passed": True,
                    },
                )
                or auth_config.get("h4_parent_checkpoint_allowed") is not False
                or auth_config.get("v4_gain_inherited") is not False
                or auth_config.get("v5_parent_checkpoint_inherited") is not False
            )
        )
    ):
        raise ValueError(f"{expert} iteration-v6 authorization binding drifted")

    authorization = _require_mapping(
        load_json_strict(auth_path), f"{expert} iteration-v6 authorization"
    )
    common_top = {
        "schema_version",
        "artifact_kind",
        "status",
        "hardware_deployment",
        "authorization",
        "scope",
        "causal_inputs",
        "training_contract",
        "curriculum",
        "reward_contract",
        "historical_v5_source_closure",
        "strict_gate_contract",
        "manifest_binding",
        "decision",
    }
    expected_top = common_top | (
        {"core_contract", "reward_routing_contract"}
        if expert == "forward"
        else {
            "action_parameterization_contract",
            "teacher_timing_contract",
            "legacy_reward_config",
        }
    )
    scope = _require_mapping(authorization.get("scope"), "v6 authorization scope")
    if (
        set(authorization) != expected_top
        or authorization.get("schema_version") != 1
        or authorization.get("artifact_kind") != spec["kind"]
        or authorization.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or authorization.get("hardware_deployment") != "PROHIBITED"
        or not _json_type_and_value_exact(
            authorization.get("authorization"),
            {
                "simulation_250k_training": True,
                "simulation_1m_training": False,
                "candidate_adoption": False,
                "release": False,
                "hardware": False,
            },
        )
        or scope.get("expert") != expert
        or scope.get("contract_id") != spec["contract"]
        or scope.get("wiring_contract_id") != spec["wiring_contract"]
        or scope.get("no_ppo_contract_id") != spec["no_ppo_contract"]
        or scope.get("required_cli_flag") != spec["required_flag"]
        or scope.get("selected_change_family") != spec["family"]
        or scope.get("training_launch_performed_by_this_artifact") is not False
    ):
        raise ValueError(f"{expert} iteration-v6 authorization semantics drifted")

    expected_execution_id = (
        spec["wiring_contract"] if config.get("wiring_only") is True else spec["contract"]
    )
    reward = _require_mapping(authorization.get("reward_contract"), "v6 reward")
    if (
        config.get("training_contract_id") != expected_execution_id
        or config.get("authorized_iteration_v6_250k_contract_id") != spec["contract"]
        or any(
            config.get(f"authorized_iteration_v{version}_250k_contract_id") is not None
            for version in (2, 3, 4, 5)
        )
        or config.get("initialization_source") != "V22_BRAX_CHECKPOINT"
        or config.get("trusted_h4_parent") is not None
        or config.get("pinned_v22_parent_tree_sha256")
        != PINNED_V22_PARENT_TREE_SHA256
        or config.get("reset_noise_multiplier") != 1.0
        or not _json_type_and_value_exact(
            config.get("reward_scales"), reward.get("exact_scales")
        )
    ):
        raise ValueError(f"{expert} iteration-v6 execution config drifted")

    if expert == "forward":
        routing = _require_mapping(
            authorization.get("reward_routing_contract"), "forward v6 reward routing"
        )
        expected_routing = {
            "source_scale_name": "h4_contact_pulse_40ms",
            "source_scale_exact": -1.0,
            "qualifying_loss": "aborted_contact_island_loss",
            "qualifying_loss_scale": -1.0,
            "off_gap_loss_retained_as_telemetry": True,
            "off_gap_qualification_role": "NON_QUALIFYING_FINITE_DIAGNOSTICS_ONLY",
            "off_gap_reward_contribution": 0.0,
            "legacy_aggregate_contact_pulse_routing_allowed": False,
            "routing_violation_count_required": 0.0,
            "assertion_token_sum_required": 0.0,
            "fail_closed_before_output_commit": True,
        }
        if (
            config.get("forward_v4_substep_contact") is not True
            or not _json_type_and_value_exact(
                config.get("reward_routing_contract"), routing
            )
            or not _json_type_and_value_exact(routing, expected_routing)
            or reward.get("all_scales_match_iteration_v4") is not True
            or reward.get("rejected_iteration_v5_minus_two_scale_inherited") is not False
            or reward.get("exact_scales", {}).get("h4_contact_pulse_40ms") != -1.0
        ):
            raise ValueError("forward iteration-v6 reward-routing contract drifted")
    else:
        action = _require_mapping(
            authorization.get("action_parameterization_contract"),
            "reverse v6 action parameterization",
        )
        timing = _require_mapping(
            authorization.get("teacher_timing_contract"), "reverse v6 teacher timing"
        )
        legacy = _require_mapping(
            authorization.get("legacy_reward_config"), "reverse v6 legacy reward"
        )
        if (
            config.get("forward_v4_substep_contact") is not False
            or not _json_type_and_value_exact(
                config.get("backward_residual_scale"), 0.0
            )
            or not _json_type_and_value_exact(
                config.get("action_parameterization_contract"), action
            )
            or not _json_type_and_value_exact(
                config.get("teacher_timing_contract"), timing
            )
            or action.get("decoder")
            != "FROZEN_V22_CALIBRATED_ABSOLUTE_FULL_LEG"
            or action.get("input_clip") != [-1.0, 1.0]
            or action.get("active_leg_indices")
            != list(H4_REVERSE_ITERATION_V6_ACTIVE_LEG_INDICES)
            or action.get("hard_zero_head_indices")
            != list(H4_REVERSE_ITERATION_V6_HEAD_INDICES)
            or action.get("directional_span_fraction") != 0.9
            or action.get("near_zero_base_cap_rad") != 0.25
            or action.get("nonlinear_exponent") != 5
            or action.get("per_control_slew_limit_rad") != 0.04
            or action.get("residual_authority_scale") != 0.0
            or action.get("teacher_target_contribution_zero") is not True
            or action.get("runtime_exact_boolean_metrics")
            != [
                "decoder_leg_count_exact",
                "precomposer_call_count_exact",
                "final_guard_call_count_exact",
            ]
            or action.get("raw_count_metrics")
            != [
                "decoder_leg_count",
                "precomposer_call_count",
                "final_guard_call_count",
            ]
            or action.get("raw_count_metrics_qualification_role")
            != "NON_QUALIFYING_FINITE_DIAGNOSTICS_ONLY"
            or action.get("host_count_multiplication_for_qualification") is not False
            or action.get("numeric_tolerance_used") is not False
            or timing.get("selected_teacher_sha256")
            != PINNED_SELECTED_REVERSE_TEACHER_SHA256
            or timing.get("role") != "PHASE_TIMING_PRIOR_ONLY"
            or timing.get("entry_phase_bins") != H4_REVERSE_TEACHER_ENTRY_PHASE_BINS
            or timing.get("phase_advance_bins_per_control")
            != H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS
            or timing.get("teacher_target_contribution") != 0.0
            or timing.get("teacher_imitation_reward_contribution") != 0.0
            or not _json_type_and_value_exact(
                legacy,
                {
                    "iteration_v6_exact": {
                        "target_imitation": 0.0,
                        "contact_imitation": 0.0,
                        "tracking_sigma": 0.01,
                    },
                    "backward_residual_scale": 0.0,
                    "identical_to_iteration_v5_except_residual_authority_removed": True,
                },
            )
            or reward.get("identical_to_iteration_v5") is not True
            or reward.get("target_imitation") != 0.0
            or reward.get("contact_imitation") != 0.0
            or reward.get("teacher_timing_prior_reward") != 0.0
        ):
            raise ValueError("reverse iteration-v6 action/timing/reward contract drifted")

    causal = _require_mapping(
        authorization.get("causal_inputs"), "v6 authorization causal inputs"
    )
    bound_causal = _require_mapping(
        auth_config.get("bound_causal_inputs"), "v6 bound causal inputs"
    )
    expected_bound_keys = {
        label.removeprefix(spec["prefix"])
        for label in spec["causal_labels"].values()
    }
    if set(bound_causal) != expected_bound_keys:
        raise ValueError(f"{expert} iteration-v6 bound causal key set drifted")
    experiment_root = auth_path.parents[1]
    paths: dict[str, Path] = {spec["auth_label"]: auth_path}
    causal_roots = spec["causal_roots"]
    for causal_key, source_label in spec["causal_labels"].items():
        source_record = _require_mapping(
            source_hashes.get(source_label), f"v6 causal source {source_label}"
        )
        bound_key = source_label.removeprefix(spec["prefix"])
        bound_record = _require_mapping(
            bound_causal.get(bound_key), f"v6 config causal binding {bound_key}"
        )
        causal_record = _require_mapping(
            causal.get(causal_key), f"v6 authorization causal {causal_key}"
        )
        expected_path = _iteration_v6_causal_path(
            experiment_root=experiment_root,
            causal=causal,
            causal_key=causal_key,
            root_key=causal_roots.get(causal_key),
        )
        source_path = Path(str(source_record.get("path", ""))).resolve()
        if (
            source_path != expected_path
            or Path(str(bound_record.get("path", ""))).resolve() != expected_path
            or source_record.get("sha256") != causal_record.get("sha256")
            or bound_record.get("sha256") != causal_record.get("sha256")
            or sha256_file(expected_path) != causal_record.get("sha256")
        ):
            raise ValueError(f"{expert} iteration-v6 causal binding drifted: {causal_key}")
        paths[source_label] = expected_path
    _validate_iteration_v6_evidence(expert=expert, spec=spec, paths=paths)

    historical = _require_mapping(
        authorization.get("historical_v5_source_closure"),
        "v6 historical v5 source closure",
    )
    bound_historical = _require_mapping(
        auth_config.get("bound_historical_v5_sources"),
        "v6 bound historical v5 sources",
    )
    if (
        historical.get("verification_source")
        != "BOUND_V5_MANIFEST_PRE_POST_SNAPSHOT_NOT_CURRENT_FILES"
        or set(historical)
        != {"verification_source", *H4_ITERATION_V4_CAUSAL_SOURCE_PATHS}
        or set(bound_historical) != set(H4_ITERATION_V4_CAUSAL_SOURCE_PATHS)
    ):
        raise ValueError(f"{expert} iteration-v6 historical source mode drifted")
    v5_manifest = _require_mapping(
        load_json_strict(paths[spec["v5_manifest_label"]]), "v6 bound v5 manifest"
    )
    v5_pre = _require_mapping(v5_manifest.get("source_and_teacher_hashes_pre"), "v5 pre")
    v5_post = _require_mapping(v5_manifest.get("source_and_teacher_hashes_post"), "v5 post")
    for label, relative in H4_ITERATION_V4_CAUSAL_SOURCE_PATHS.items():
        historical_record = _require_mapping(
            historical.get(label), f"v6 historical source {label}"
        )
        bound_record = _require_mapping(
            bound_historical.get(label), f"v6 bound historical source {label}"
        )
        manifest_label = f"{spec['v5_source_prefix']}{label}"
        pre_record = _require_mapping(v5_pre.get(manifest_label), f"v5 pre {manifest_label}")
        post_record = _require_mapping(
            v5_post.get(manifest_label), f"v5 post {manifest_label}"
        )
        if (
            historical_record != {"path": relative, "sha256": pre_record.get("sha256")}
            or pre_record.get("sha256") != post_record.get("sha256")
            or bound_record.get("path") != relative
            or bound_record.get("sha256") != historical_record.get("sha256")
            or bound_record.get("manifest_pre") != pre_record
            or bound_record.get("manifest_post") != post_record
        ):
            raise ValueError(f"{expert} iteration-v6 historical source drifted: {label}")

    for source_label, relative in current_labels.items():
        record = _require_mapping(
            source_hashes.get(source_label), f"v6 current source {source_label}"
        )
        path = Path(str(record.get("path", ""))).resolve()
        normalized = PurePosixPath(str(path).replace("\\", "/")).as_posix()
        if not normalized.endswith(relative) or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{expert} iteration-v6 current source drifted: {source_label}")
        paths[source_label] = path
    core_source_label = f"{spec['prefix']}current_source_h4_training_alignment"
    core_source_record = _require_mapping(
        source_hashes.get(core_source_label), "v6 current core source"
    )
    core_source = _require_mapping(
        config.get("iteration_v6_core_source"), "v6 config core source"
    )
    if (
        set(core_source) != {"path", "sha256"}
        or dict(core_source) != dict(core_source_record)
        or core_source.get("sha256")
        != PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256
        or Path(str(core_source.get("path", ""))).resolve()
        != paths[core_source_label]
    ):
        raise ValueError(f"{expert} iteration-v6 core source binding drifted")
    return paths


def _validated_forward_iteration_v6_contact_abort_island_only_source_paths(
    *, config: Mapping[str, Any], source_hashes: Mapping[str, Mapping[str, str]]
) -> dict[str, Path]:
    return _validated_iteration_v6_source_paths(
        expert="forward", config=config, source_hashes=source_hashes
    )


def _validated_reverse_iteration_v6_absolute_full_leg_targets_source_paths(
    *, config: Mapping[str, Any], source_hashes: Mapping[str, Mapping[str, str]]
) -> dict[str, Path]:
    return _validated_iteration_v6_source_paths(
        expert="reverse", config=config, source_hashes=source_hashes
    )


def _validate_iteration_v6_core_source_artifact_closure(
    *,
    expert: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result_payload: Mapping[str, Any],
    source_hashes: Mapping[str, Mapping[str, str]],
) -> None:
    """Bind each v6 artifact independently to the current pinned core."""

    spec = _iteration_v6_spec(expert)
    source_label = f"{spec['prefix']}current_source_h4_training_alignment"
    source_record = _require_mapping(
        source_hashes.get(source_label), "v6 current pinned core source"
    )
    source_path = Path(str(source_record.get("path", ""))).resolve()
    expected_record = {
        "path": str(source_path),
        "sha256": PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256,
    }
    if (
        not _json_type_and_value_exact(source_record, expected_record)
        or not source_path.is_file()
        or sha256_file(source_path)
        != PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256
    ):
        raise ValueError(f"{expert} iteration-v6 current core source drifted")

    for artifact_name, artifact in (
        ("config", config),
        ("manifest", manifest),
        ("result", result_payload),
    ):
        artifact_record = artifact.get("iteration_v6_core_source")
        if not _json_type_and_value_exact(artifact_record, expected_record):
            raise ValueError(
                f"{expert} iteration-v6 {artifact_name} core source binding drifted"
            )


def _validate_iteration_v6_artifact_location_closure(
    *,
    expert: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> None:
    """Require every active-v6 claim in its exact config/manifest/result slots."""

    spec = _iteration_v6_spec(expert)
    auth_key = spec["auth_key"]
    runtime_key = (
        "forward_iteration_v6_reward_routing_runtime"
        if expert == "forward"
        else "reverse_iteration_v6_decoder_runtime"
    )
    expected_locations: dict[str, frozenset[str]] = {
        auth_key: frozenset({"config", "manifest"}),
        f"{auth_key}_sha256": frozenset({"result"}),
        f"{runtime_key}_requirement": frozenset(
            {"config", "manifest", "result"}
        ),
        runtime_key: frozenset({"manifest", "result"}),
        "iteration_v6_core_source": frozenset(
            {"config", "manifest", "result"}
        ),
        "iteration_v6_artifact_cross_binding": frozenset(
            {"manifest", "result"}
        ),
    }
    if expert == "forward":
        expected_locations["legacy_reward_config_audit"] = frozenset()
        for key in (
            "reward_routing_contract",
            "forward_v4_source_semantic_preflight",
            "forward_v4_single_authority_runtime_requirement",
            "forward_v4_single_authority_runtime_audit_mode",
        ):
            expected_locations[key] = frozenset(
                {"config", "manifest", "result"}
            )
        expected_locations["forward_v4_single_authority_runtime"] = frozenset(
            {"manifest", "result"}
        )
    else:
        for key in (
            "action_parameterization_contract",
            "teacher_timing_contract",
            "backward_residual_scale",
        ):
            expected_locations[key] = frozenset(
                {"config", "manifest", "result"}
            )
        for key in (
            "legacy_reward_config_audit",
            "h4_parent_checkpoint_allowed",
            "v4_gain_inherited",
            "v5_parent_checkpoint_inherited",
            "teacher_target_contribution_zero",
        ):
            expected_locations[key] = frozenset({"result"})

    artifacts = {
        "config": config,
        "manifest": manifest,
        "result": result_payload,
    }
    controlled_keys = frozenset(expected_locations)
    for artifact_name, artifact in artifacts.items():
        expected_keys = {
            key
            for key, locations in expected_locations.items()
            if artifact_name in locations
        }
        actual_keys = set(artifact) & controlled_keys
        if actual_keys != expected_keys:
            raise ValueError(
                f"{expert} iteration-v6 artifact location closure drifted for "
                f"{artifact_name}: missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )


def _iteration_v6_runtime_requirement(expert: str, status: str) -> dict[str, Any]:
    if status not in {"WIRING_PASS", "COMPLETED"}:
        raise ValueError(f"unsupported H4 iteration-v6 status: {status!r}")
    if expert == "forward":
        return {
            "routing_exact": True,
            "island_loss": "NON_NEGATIVE_FINITE_QUALIFYING_LOSS",
            "off_gap_diagnostic_loss": (
                "NON_NEGATIVE_FINITE_NON_QUALIFYING_ONLY"
            ),
            "off_gap_reward_contribution": 0.0,
            "pulse_reward_scale": -1.0,
            "routing_violation_count": 0.0,
            "assertion_token_sum": 0.0,
            "per_step_compiled_fail_closed_assertion_required": True,
            "fail_closed_before_output_commit": True,
        }
    if expert == "reverse":
        return {
            "decoder_action_shape": [14],
            "decoder_raw_targets_shape": [14],
            "decoder_margin_targets_shape": [14],
            "decoder_exact": True,
            "max_abs_error": 0.0,
            "leg_count": "NON_NEGATIVE_FINITE_DIAGNOSTIC_ONLY",
            "leg_count_exact": True,
            "head_zero_exact": True,
            "teacher_target_contribution_zero_exact": True,
            "residual_authority_scale": 0.0,
            "decoder_all_finite": True,
            "margin_saturation_count": "NON_NEGATIVE_FINITE_DIAGNOSTIC_ONLY",
            "action_clip_count": "NON_NEGATIVE_FINITE_DIAGNOSTIC_ONLY",
            "guard_lag_max_rad": "NON_NEGATIVE_FINITE_DIAGNOSTIC_ONLY",
            "precomposer_call_count": "NON_NEGATIVE_FINITE_DIAGNOSTIC_ONLY",
            "precomposer_call_count_exact": True,
            "final_guard_call_count": "NON_NEGATIVE_FINITE_DIAGNOSTIC_ONLY",
            "final_guard_call_count_exact": True,
            "diagnostic_count_totals_qualification_role": (
                "NON_QUALIFYING_FINITE_DIAGNOSTICS_ONLY"
            ),
            "host_count_multiplication_for_qualification": False,
            "numeric_tolerance_used": False,
            "decoder_violation_count": 0.0,
            "assertion_token_sum": 0.0,
            "per_step_compiled_fail_closed_assertion_required": True,
            "fail_closed_before_output_commit": True,
        }
    raise ValueError(f"unsupported H4 iteration-v6 expert: {expert!r}")


def _iteration_v6_curve_rows(
    curve_path: Path, *, expert: str, wiring_only: bool
) -> dict[str, Any]:
    """Recheck v6 runtime and completion evidence from the bound CSV.

    A completed run is qualified by the CSV itself, not by the manifest's
    completion summary.  It must contain the five exact PPO progress rows and
    one distinct terminal training-metrics row at 250k interactions.  Wiring
    retains its separate 40-interaction/zero-episode contract.
    """

    try:
        with Path(curve_path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise ValueError(f"{expert} iteration-v6 curve could not be read") from exc
    if not rows:
        raise ValueError(f"{expert} iteration-v6 curve has no progress rows")

    def numeric(row: Mapping[str, Any], key: str, row_index: int) -> float:
        raw = row.get(key)
        try:
            value = float(raw) if raw not in (None, "") else math.nan
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{expert} iteration-v6 curve row {row_index} {key} is invalid"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"{expert} iteration-v6 curve row {row_index} {key} is non-finite"
            )
        return value

    totals: dict[str, float] = {}
    active_reverse_samples = 0.0
    progress_interactions: list[int] = []
    episode_progress_interactions: list[int] = []
    episode_rows: list[Mapping[str, Any]] = []
    non_episode_rows: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        raw_step = row.get("environment_interactions")
        try:
            numeric_step = (
                float(raw_step) if raw_step not in (None, "") else math.nan
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{expert} iteration-v6 curve row {index} interaction is invalid"
            ) from exc
        if not np.isfinite(numeric_step) or not numeric_step.is_integer():
            raise ValueError(
                f"{expert} iteration-v6 curve row {index} interaction is invalid"
            )
        step = int(numeric_step)
        progress_interactions.append(step)
        if row.get("episode/length") in (None, ""):
            non_episode_rows.append((index, step, dict(row)))
            continue

        episode_rows.append(row)
        episode_progress_interactions.append(step)
        length = numeric(row, "episode/length", index)
        if length <= 0.0:
            raise ValueError(f"{expert} iteration-v6 episode length is invalid")
        if expert == "forward":
            prefix = "episode/h4/v6_forward_contact_abort_"
            values = {
                suffix: numeric(row, f"{prefix}{suffix}", index)
                for suffix in (
                    "routing_exact",
                    "island_loss",
                    "off_gap_diagnostic_loss",
                    "off_gap_reward_contribution",
                    "pulse_reward_scale",
                    "routing_violation",
                    "routing_assertion_token",
                )
            }
            if (
                values["routing_exact"] != length
                or values["island_loss"] < 0.0
                or values["off_gap_diagnostic_loss"] < 0.0
                or values["off_gap_reward_contribution"] != 0.0
                or values["pulse_reward_scale"] != -length
                or values["routing_violation"] != 0.0
                or values["routing_assertion_token"] != 0.0
            ):
                raise ValueError("forward iteration-v6 curve reward routing drifted")
        else:
            prefix = "episode/h4/v6_reverse_"
            suffixes = (
                "decoder_exact",
                "decoder_max_abs_error",
                "decoder_leg_count",
                "decoder_leg_count_exact",
                "decoder_head_zero_exact",
                "teacher_target_contribution_zero_exact",
                "residual_authority_scale",
                "decoder_all_finite",
                "decoder_margin_saturation_count",
                "decoder_action_clip_count",
                "decoder_guard_lag_max_rad",
                "precomposer_call_count",
                "precomposer_call_count_exact",
                "final_guard_call_count",
                "final_guard_call_count_exact",
                "decoder_violation",
                "decoder_assertion_token",
            )
            values = {
                suffix: numeric(row, f"{prefix}{suffix}", index)
                for suffix in suffixes
            }
            active = length
            if (
                values["decoder_exact"] != active
                or values["decoder_max_abs_error"] != 0.0
                or values["decoder_leg_count"] < 0.0
                or values["decoder_leg_count_exact"] != active
                or values["decoder_head_zero_exact"] != active
                or values["teacher_target_contribution_zero_exact"] != active
                or values["residual_authority_scale"] != 0.0
                or values["decoder_all_finite"] != active
                or values["decoder_margin_saturation_count"] < 0.0
                or values["decoder_action_clip_count"] < 0.0
                or values["decoder_guard_lag_max_rad"] < 0.0
                or values["precomposer_call_count"] < 0.0
                or values["precomposer_call_count_exact"] != active
                or values["final_guard_call_count"] < 0.0
                or values["final_guard_call_count_exact"] != active
                or values["decoder_violation"] != 0.0
                or values["decoder_assertion_token"] != 0.0
            ):
                raise ValueError("reverse iteration-v6 curve decoder routing drifted")
            active_reverse_samples += active
        for suffix, value in values.items():
            totals[suffix] = totals.get(suffix, 0.0) + value

    final_metrics: dict[str, float] | None = None
    if wiring_only:
        progress_exact = bool(
            progress_interactions
            and progress_interactions == sorted(progress_interactions)
            and progress_interactions[-1] == H4_WIRING_INTERACTIONS
            and max(progress_interactions) == H4_WIRING_INTERACTIONS
            and all(
                0 <= step <= H4_WIRING_INTERACTIONS
                for step in progress_interactions
            )
        )
        if not progress_exact:
            raise ValueError(
                f"{expert} iteration-v6 wiring curve did not complete exact "
                "40 interactions"
            )
    else:
        expected_training_progress = list(
            H4_FORWARD_V4_FULL_TRAINING_PROGRESS_INTERACTIONS
        )
        progress_exact = bool(
            episode_progress_interactions == expected_training_progress
            and progress_interactions[-1] == H4_PILOT_INTERACTIONS
            and max(progress_interactions) == H4_PILOT_INTERACTIONS
            and all(
                0 < step <= H4_PILOT_INTERACTIONS
                for step in progress_interactions
            )
            and len(non_episode_rows) == 1
            and non_episode_rows[0][0] == len(rows) - 1
            and non_episode_rows[0][1] == H4_PILOT_INTERACTIONS
        )
        if not progress_exact:
            raise ValueError(
                f"{expert} iteration-v6 full curve is not exact five monotonic "
                "training rows plus one final-metrics row at 250000 interactions"
            )
        final_metrics_row = non_episode_rows[0][2]
        if any(
            value not in (None, "")
            for key, value in final_metrics_row.items()
            if key.startswith("episode/")
        ):
            raise ValueError(
                f"{expert} iteration-v6 final-metrics row contains episode metrics"
            )
        final_metrics = {}
        for key, raw_value in final_metrics_row.items():
            if key == "environment_interactions" or raw_value in (None, ""):
                continue
            if not key.startswith("training/"):
                raise ValueError(
                    f"{expert} iteration-v6 final-metrics row contains a "
                    "non-training field"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{expert} iteration-v6 final metric {key} is invalid"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"{expert} iteration-v6 final metric {key} is non-finite"
                )
            final_metrics[key] = value
        if not final_metrics:
            raise ValueError(
                f"{expert} iteration-v6 final-metrics row is empty"
            )

    return {
        "observed_episode_metric_rows": len(episode_rows),
        "active_reverse_sample_count": active_reverse_samples,
        "metric_totals": totals,
        "progress_reached_final_interaction": progress_exact,
        **(
            {
                "training_progress_interactions": episode_progress_interactions,
                "final_metrics_interaction": H4_PILOT_INTERACTIONS,
                "final_metrics": final_metrics,
            }
            if not wiring_only
            else {}
        ),
    }


def _validate_iteration_v6_runtime_closure(
    *,
    expert: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result_payload: Mapping[str, Any],
    outputs: Mapping[str, Any],
    status: str,
) -> None:
    """Cross-bind exact v6 runtime requirements and full/wiring evidence."""

    if expert == "forward":
        runtime_key = "forward_iteration_v6_reward_routing_runtime"
        inactive_runtime_key = "reverse_iteration_v6_decoder_runtime"
    else:
        runtime_key = "reverse_iteration_v6_decoder_runtime"
        inactive_runtime_key = "forward_iteration_v6_reward_routing_runtime"
    requirement_key = f"{runtime_key}_requirement"
    inactive_requirement_key = f"{inactive_runtime_key}_requirement"
    if any(
        "iteration_v6_runtime" in artifact
        or inactive_runtime_key in artifact
        or inactive_requirement_key in artifact
        for artifact in (config, manifest, result_payload)
    ):
        raise ValueError(f"{expert} iteration-v6 runtime key schema drifted")
    expected_requirement = _iteration_v6_runtime_requirement(expert, status)
    config_requirement = _require_mapping(
        config.get(requirement_key), f"{expert} iteration-v6 runtime requirement"
    )
    manifest_requirement = _require_mapping(
        manifest.get(requirement_key), f"manifest {expert} iteration-v6 requirement"
    )
    result_requirement = _require_mapping(
        result_payload.get(requirement_key), f"result {expert} iteration-v6 requirement"
    )
    manifest_runtime = _require_mapping(
        manifest.get(runtime_key), f"manifest {expert} iteration-v6 runtime"
    )
    result_runtime = _require_mapping(
        result_payload.get(runtime_key), f"result {expert} iteration-v6 runtime"
    )
    if (
        not _json_type_and_value_exact(config_requirement, expected_requirement)
        or not _json_type_and_value_exact(
            manifest_requirement, expected_requirement
        )
        or not _json_type_and_value_exact(result_requirement, expected_requirement)
        or not _json_type_and_value_exact(manifest_runtime, result_runtime)
    ):
        raise ValueError(f"{expert} iteration-v6 runtime cross-binding drifted")

    curve_path, _ = _verify_file_record(
        outputs.get("training_curve"), label="outputs.training_curve"
    )
    curve = _iteration_v6_curve_rows(
        curve_path, expert=expert, wiring_only=(status == "WIRING_PASS")
    )
    if status == "COMPLETED":
        result_final_metrics = _require_mapping(
            result_payload.get("final_metrics"),
            f"{expert} iteration-v6 result final metrics",
        )
        normalized_result_final_metrics: dict[str, float] = {}
        for key, value in result_final_metrics.items():
            if (
                not isinstance(key, str)
                or not key.startswith("training/")
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
            ):
                raise ValueError(
                    f"{expert} iteration-v6 result final metrics are not exact "
                    "finite training scalars"
                )
            normalized_result_final_metrics[key] = float(value)
        if (
            not normalized_result_final_metrics
            or curve.get("final_metrics") != normalized_result_final_metrics
        ):
            raise ValueError(
                f"{expert} iteration-v6 bound curve final metrics differ from "
                "run result"
            )
    expected_runtime = {
        "audit_mode": (
            H4_FORWARD_ITERATION_V6_RUNTIME_AUDIT_MODES[status]
            if expert == "forward"
            else H4_REVERSE_ITERATION_V6_RUNTIME_AUDIT_MODES[status]
        ),
        "expert": expert,
        "observed_episode_metric_rows": curve["observed_episode_metric_rows"],
        "episode_metric_rows_exact_if_observed": True,
        "per_step_compiled_fail_closed_assertion_bound": True,
        "completed_environment_interactions": (
            H4_WIRING_INTERACTIONS
            if status == "WIRING_PASS"
            else H4_PILOT_INTERACTIONS
        ),
        "completed_training_steps": (
            H4_WIRING_TRAINING_STEPS
            if status == "WIRING_PASS"
            else H4_PILOT_TRAINING_STEPS
        ),
        "completed_optimizer_updates": (
            H4_WIRING_OPTIMIZER_UPDATES
            if status == "WIRING_PASS"
            else H4_PILOT_OPTIMIZER_UPDATES
        ),
        "progress_reached_final_interaction": True,
        "final_params_all_finite": True,
        "final_metrics_all_finite": True,
        "source_and_teacher_unchanged": True,
        "passed": True,
    }
    if not _json_type_and_value_exact(result_runtime, expected_runtime):
        raise ValueError(f"{expert} iteration-v6 full/wiring runtime gate drifted")
    if status == "COMPLETED" and curve["observed_episode_metric_rows"] != 5:
        raise ValueError(f"{expert} iteration-v6 full curve row count drifted")


def validate_trusted_h4_bundle(
    *,
    params_path: Path,
    manifest_path: Path,
    expected_params_sha256: str,
    expected_manifest_sha256: str,
    trusted_run_root: Path,
    allow_wiring_diagnostic: bool = False,
) -> TrustedH4Bundle:
    """Validate every external binding before a params pickle may be opened."""

    params_resolved = Path(params_path).resolve()
    manifest_resolved = Path(manifest_path).resolve()
    run_root_resolved = Path(trusted_run_root).resolve()
    if (
        params_resolved.name != "final_params.pkl"
        or manifest_resolved.name != "run_manifest.json"
    ):
        raise ValueError("trusted H4 bundle requires exact runner output basenames")
    expected_params = require_sha256(expected_params_sha256, "expected params SHA256")
    expected_manifest = require_sha256(
        expected_manifest_sha256, "expected manifest SHA256"
    )
    actual_params = sha256_file(params_resolved)
    actual_manifest = sha256_file(manifest_resolved)
    if actual_params != expected_params:
        raise ValueError("trusted H4 params SHA256 mismatch")
    if actual_manifest != expected_manifest:
        raise ValueError("trusted H4 manifest SHA256 mismatch")

    manifest = _require_mapping(load_json_strict(manifest_resolved), "manifest")
    status = manifest.get("status")
    allowed_statuses = {"COMPLETED"}
    if allow_wiring_diagnostic:
        allowed_statuses.add("WIRING_PASS")
    if status not in allowed_statuses:
        raise ValueError(
            f"H4 manifest status {status!r} is not allowed for this operation"
        )
    if manifest.get("schema_version") != 1:
        raise ValueError("H4 manifest schema drifted")
    if manifest.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("H4 manifest must prohibit hardware deployment")
    if manifest.get("source_and_teacher_unchanged") is not True:
        raise ValueError("H4 source/teacher closure is not immutable")

    pre_hashes = _require_mapping(
        manifest.get("source_and_teacher_hashes_pre"), "source hashes pre"
    )
    post_hashes = _require_mapping(
        manifest.get("source_and_teacher_hashes_post"), "source hashes post"
    )
    if not pre_hashes or dict(pre_hashes) != dict(post_hashes):
        raise ValueError("H4 source/teacher pre/post hashes differ or are empty")
    normalized_source_hashes: dict[str, dict[str, str]] = {}
    for label, raw_record in post_hashes.items():
        if not isinstance(label, str) or not label:
            raise ValueError("source hash labels must be non-empty strings")
        record = _require_mapping(raw_record, f"source hash {label}")
        if set(record) != {"path", "sha256"}:
            raise ValueError(
                f"source hash {label} must contain exactly path and sha256"
            )
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"source hash {label}.path must be non-empty")
        if not _is_absolute_path_text(raw_path):
            raise ValueError(f"source hash {label}.path must be absolute")
        normalized_source_hashes[label] = {
            "path": str(Path(raw_path).resolve()),
            "sha256": require_sha256(
                record.get("sha256"), f"source hash {label}.sha256"
            ),
        }
    if dict(post_hashes) != normalized_source_hashes:
        raise ValueError("source hash records must use resolved absolute paths")

    parent = _require_mapping(manifest.get("parent_checkpoint"), "parent checkpoint")
    if (
        parent.get("unchanged") is not True
        or parent.get("sha256_tree_pre") != parent.get("sha256_tree_post")
    ):
        raise ValueError("H4 parent checkpoint did not remain read-only")
    require_sha256(parent.get("sha256_tree_pre"), "parent checkpoint SHA256")

    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    if set(outputs) != {"final_params", "result", "training_curve"}:
        raise ValueError("manifest output closure field set drifted")
    _, bound_params_sha = _verify_file_record(
        outputs.get("final_params"),
        label="outputs.final_params",
        expected_path=params_resolved,
    )
    if bound_params_sha != actual_params:
        raise ValueError("manifest final params binding mismatch")

    resolved_config_record = _require_mapping(
        manifest.get("resolved_config"), "resolved_config"
    )
    if set(resolved_config_record) != {"path", "sha256", "canonical_sha256"}:
        raise ValueError("resolved_config record field set drifted")
    config_path, config_sha = _verify_file_record(
        resolved_config_record, label="resolved_config"
    )
    config = _require_mapping(load_json_strict(config_path), "resolved config")
    config_canonical = canonical_json_sha256(config)
    if resolved_config_record.get("canonical_sha256") != config_canonical:
        raise ValueError("resolved config canonical SHA256 mismatch")

    run_name = manifest.get("run_name")
    expert = manifest.get("expert")
    activity = manifest.get("activity")
    if not isinstance(run_name, str) or not run_name or Path(run_name).name != run_name:
        raise ValueError("H4 manifest run_name is invalid")
    if expert not in H4_STRICT_SEEDS:
        raise ValueError("H4 manifest expert must be forward or reverse")
    if activity not in {"PPO_PILOT_TRAINING", "PPO_WIRING_TRAINING"}:
        raise ValueError("H4 manifest activity is invalid")
    if (
        config.get("schema_version") != 1
        or config.get("hardware_deployment") != "PROHIBITED"
        or config.get("actor_observation_width") != H4_ACTOR_OBSERVATION_WIDTH
        or config.get("observation_mode") != "h4_116_transplant"
        or config.get("run_name", run_name) != run_name
        or config.get("expert") != expert
        or config.get("activity") != activity
    ):
        raise ValueError("resolved config/manifest H4 contract drifted")
    if config.get("ppo", {}).get("normalize_observations") is not True:
        raise ValueError("H4 actor inference requires normalized observations")
    if config.get("network_factory", {}).get("policy_hidden_layer_sizes") not in (
        [512, 256, 128],
        (512, 256, 128),
    ):
        raise ValueError("H4 actor network topology drifted")
    if config.get("network_factory", {}).get("policy_obs_key") != "state":
        raise ValueError("H4 actor observation key drifted")
    all_iteration_flags = {
        "forward_iteration_v2": config.get("forward_iteration_v2", False),
        "reverse_iteration_v2": config.get("reverse_iteration_v2", False),
        "forward_iteration_v3_touchdown_balance": config.get(
            "forward_iteration_v3_touchdown_balance", False
        ),
        "reverse_iteration_v3_no_target_imitation": config.get(
            "reverse_iteration_v3_no_target_imitation", False
        ),
        "forward_iteration_v4_contact_event_validity_persistence": config.get(
            "forward_iteration_v4_contact_event_validity_persistence", False
        ),
        "reverse_iteration_v4_residual_transfer_gain_024": config.get(
            "reverse_iteration_v4_residual_transfer_gain_024", False
        ),
        "forward_v5_contact_pulse_abort_scale_only": config.get(
            "forward_v5_contact_pulse_abort_scale_only", False
        ),
        "reverse_iteration_v5_no_contact_imitation": config.get(
            "reverse_iteration_v5_no_contact_imitation", False
        ),
        "forward_iteration_v6_contact_abort_island_only": config.get(
            "forward_iteration_v6_contact_abort_island_only", False
        ),
        "reverse_iteration_v6_absolute_full_leg_targets": config.get(
            "reverse_iteration_v6_absolute_full_leg_targets", False
        ),
    }
    if any(not isinstance(value, bool) for value in all_iteration_flags.values()):
        raise ValueError("H4 iteration config flags must be boolean")
    if sum(value is True for value in all_iteration_flags.values()) > 1:
        raise ValueError("H4 iteration config flags are mutually exclusive")
    # Historical artifacts may omit fields introduced by later iterations.  An
    # omitted field makes no claim; every field that is present must agree with
    # the config-selected mode and authorization version.  This global closure
    # runs before every version-specific branch and prevents a trusted older
    # bundle from acquiring contradictory later-version metadata.
    identity_result_path, _ = _verify_file_record(
        outputs.get("result"), label="outputs.result"
    )
    identity_result = _require_mapping(
        load_json_strict(identity_result_path), "run result"
    )
    manifest_iteration_flags = {
        key: manifest[key] for key in all_iteration_flags if key in manifest
    }
    result_iteration_flags = {
        key: identity_result[key]
        for key in all_iteration_flags
        if key in identity_result
    }
    if (
        any(
            not isinstance(value, bool)
            for value in (
                *manifest_iteration_flags.values(),
                *result_iteration_flags.values(),
            )
        )
        or any(
            value is not all_iteration_flags[key]
            for key, value in manifest_iteration_flags.items()
        )
        or any(
            value is not all_iteration_flags[key]
            for key, value in result_iteration_flags.items()
        )
    ):
        raise ValueError(
            "H4 iteration config/manifest/result mode identity drifted"
        )
    active_iteration_flag = next(
        (key for key, value in all_iteration_flags.items() if value), None
    )
    iteration_version_by_flag = {
        "forward_iteration_v2": 2,
        "reverse_iteration_v2": 2,
        "forward_iteration_v3_touchdown_balance": 3,
        "reverse_iteration_v3_no_target_imitation": 3,
        "forward_iteration_v4_contact_event_validity_persistence": 4,
        "reverse_iteration_v4_residual_transfer_gain_024": 4,
        "forward_v5_contact_pulse_abort_scale_only": 5,
        "reverse_iteration_v5_no_contact_imitation": 5,
        "forward_iteration_v6_contact_abort_island_only": 6,
        "reverse_iteration_v6_absolute_full_leg_targets": 6,
    }
    active_iteration_version = (
        iteration_version_by_flag[active_iteration_flag]
        if active_iteration_flag is not None
        else None
    )
    authorization_id_keys = tuple(
        f"authorized_iteration_v{version}_250k_contract_id"
        for version in range(2, 7)
    )
    config_authorization_ids = {
        key: config.get(key) for key in authorization_id_keys
    }
    manifest_authorization_ids = {
        key: manifest[key] for key in authorization_id_keys if key in manifest
    }
    result_authorization_ids = {
        key: identity_result[key]
        for key in authorization_id_keys
        if key in identity_result
    }
    selected_authorization_key = (
        f"authorized_iteration_v{active_iteration_version}_250k_contract_id"
        if active_iteration_version is not None
        else None
    )
    if (
        any(
            value is not None and (not isinstance(value, str) or not value)
            for value in config_authorization_ids.values()
        )
        or any(
            value != config_authorization_ids[key]
            for key, value in manifest_authorization_ids.items()
        )
        or any(
            value != config_authorization_ids[key]
            for key, value in result_authorization_ids.items()
        )
        or any(
            (key == selected_authorization_key) != (value is not None)
            for key, value in config_authorization_ids.items()
        )
    ):
        raise ValueError(
            "H4 iteration config/manifest/result authorization identity drifted"
        )
    version_specific_key_owners: dict[str, frozenset[str]] = {}
    authorization_keys_by_flag = {
        "forward_iteration_v2": "forward_iteration_v2_authorization",
        "reverse_iteration_v2": "reverse_iteration_v2_authorization",
        "forward_iteration_v3_touchdown_balance": (
            "forward_iteration_v3_touchdown_balance_authorization"
        ),
        "reverse_iteration_v3_no_target_imitation": (
            "reverse_iteration_v3_no_target_imitation_authorization"
        ),
        "forward_iteration_v4_contact_event_validity_persistence": (
            "forward_iteration_v4_contact_event_validity_persistence_authorization"
        ),
        "reverse_iteration_v4_residual_transfer_gain_024": (
            "reverse_iteration_v4_residual_transfer_gain_024_authorization"
        ),
        "forward_v5_contact_pulse_abort_scale_only": (
            "forward_iteration_v5_contact_pulse_abort_scale_only_authorization"
        ),
        "reverse_iteration_v5_no_contact_imitation": (
            "reverse_iteration_v5_no_contact_imitation_authorization"
        ),
        "forward_iteration_v6_contact_abort_island_only": (
            "forward_iteration_v6_contact_abort_island_only_authorization"
        ),
        "reverse_iteration_v6_absolute_full_leg_targets": (
            "reverse_iteration_v6_absolute_full_leg_targets_authorization"
        ),
    }
    for owner, authorization_key in authorization_keys_by_flag.items():
        owner_set = frozenset({owner})
        version_specific_key_owners[authorization_key] = owner_set
        version_specific_key_owners[f"{authorization_key}_sha256"] = owner_set
    inactive_null_authorization_placeholders = frozenset(
        key
        for authorization_key in authorization_keys_by_flag.values()
        for key in (authorization_key, f"{authorization_key}_sha256")
    )
    forward_authority_owners = frozenset(
        {
            "forward_iteration_v4_contact_event_validity_persistence",
            "forward_v5_contact_pulse_abort_scale_only",
            "forward_iteration_v6_contact_abort_island_only",
        }
    )
    for key in (
        "forward_v4_source_semantic_preflight",
        "forward_v4_single_authority_runtime_requirement",
        "forward_v4_single_authority_runtime",
        "forward_v4_single_authority_runtime_audit_mode",
    ):
        version_specific_key_owners[key] = forward_authority_owners
    forward_v6_owner = frozenset(
        {"forward_iteration_v6_contact_abort_island_only"}
    )
    reverse_v6_owner = frozenset(
        {"reverse_iteration_v6_absolute_full_leg_targets"}
    )
    both_v6_owners = forward_v6_owner | reverse_v6_owner
    for key in (
        "forward_iteration_v6_reward_routing_runtime_requirement",
        "forward_iteration_v6_reward_routing_runtime",
        "reward_routing_contract",
    ):
        version_specific_key_owners[key] = forward_v6_owner
    for key in (
        "reverse_iteration_v6_decoder_runtime_requirement",
        "reverse_iteration_v6_decoder_runtime",
        "action_parameterization_contract",
        "teacher_timing_contract",
        "h4_parent_checkpoint_allowed",
        "v4_gain_inherited",
        "v5_parent_checkpoint_inherited",
        "teacher_target_contribution_zero",
    ):
        version_specific_key_owners[key] = reverse_v6_owner
    for key in ("iteration_v6_core_source", "iteration_v6_runtime"):
        version_specific_key_owners[key] = both_v6_owners
    version_specific_key_owners[
        "iteration_v6_artifact_cross_binding"
    ] = both_v6_owners
    for artifact in (config, manifest, identity_result):
        for key, owners in version_specific_key_owners.items():
            if key in artifact and active_iteration_flag not in owners:
                # The runner retains legacy authorization record/SHA fields as
                # exact JSON null placeholders.  Null carries no authorization
                # claim; every non-null inactive authorization value and every
                # other inactive version-specific key remain a graft.
                if (
                    key in inactive_null_authorization_placeholders
                    and artifact[key] is None
                ):
                    continue
                raise ValueError(
                    "H4 inactive iteration metadata was grafted onto the bundle"
                )
    active_v3_flags = {
        key: all_iteration_flags[key]
        for key in (
            "forward_iteration_v3_touchdown_balance",
            "reverse_iteration_v3_no_target_imitation",
        )
    }
    active_v3_flag = next(
        (key for key, value in active_v3_flags.items() if value), None
    )
    if active_v3_flag is not None:
        v3_spec = _iteration_v3_spec(expert)
        if active_v3_flag != v3_spec["flag"]:
            raise ValueError("H4 iteration-v3 flag/expert binding drifted")
        manifest_auth = _require_mapping(
            manifest.get(v3_spec["auth_key"]), "manifest v3 authorization"
        )
        config_auth = _require_mapping(
            config.get(v3_spec["auth_key"]), "config v3 authorization"
        )
        result_path, _ = _verify_file_record(
            outputs.get("result"), label="outputs.result"
        )
        result_payload = _require_mapping(
            load_json_strict(result_path), "run result"
        )
        result_auth_sha_key = f"{v3_spec['auth_key']}_sha256"
        inactive_v3_flag = (
            "reverse_iteration_v3_no_target_imitation"
            if active_v3_flag == "forward_iteration_v3_touchdown_balance"
            else "forward_iteration_v3_touchdown_balance"
        )
        expected_manifest_auth_keys = {"path", "sha256", "contract_id"}
        if expert == "reverse":
            expected_manifest_auth_keys.add("legacy_reward_config_audit")
        if (
            manifest.get(active_v3_flag) is not True
            or manifest.get(inactive_v3_flag, False) is not False
            or manifest.get("forward_iteration_v2") is not False
            or manifest.get("reverse_iteration_v2") is not False
            or manifest.get("authorized_iteration_v3_250k_contract_id")
            != v3_spec["contract"]
            or manifest.get("authorized_iteration_v2_250k_contract_id") is not None
            or manifest.get("training_contract_id")
            != config.get("training_contract_id")
            or manifest.get("qualification_use") != config.get("qualification_use")
            or set(manifest_auth) != expected_manifest_auth_keys
            or manifest_auth.get("path") != config_auth.get("path")
            or manifest_auth.get("sha256") != v3_spec["auth_sha"]
            or manifest_auth.get("contract_id") != v3_spec["contract"]
            or config.get("initialization_source") != "V22_BRAX_CHECKPOINT"
            or config.get("trusted_h4_parent") is not None
            or config.get("pinned_v22_parent_tree_sha256")
            != PINNED_V22_PARENT_TREE_SHA256
            or config.get("reset_noise_multiplier") != 1.0
            or parent.get("kind") != "V22_BRAX_CHECKPOINT"
            or parent.get("sha256_tree_pre") != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("sha256_tree_post") != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("unchanged") is not True
            or any(label.startswith("h4_parent_") for label in normalized_source_hashes)
            or result_payload.get("expert") != expert
            or result_payload.get("status") != status
            or result_payload.get("training_contract_id")
            != config.get("training_contract_id")
            or result_payload.get("authorized_iteration_v3_250k_contract_id")
            != v3_spec["contract"]
            or result_payload.get("authorized_iteration_v2_250k_contract_id") is not None
            or result_payload.get("qualification_use")
            != config.get("qualification_use")
            or result_payload.get("forward_iteration_v2") is not False
            or result_payload.get("reverse_iteration_v2") is not False
            or result_payload.get(active_v3_flag) is not True
            or result_payload.get(inactive_v3_flag, False) is not False
            or result_payload.get(result_auth_sha_key) != v3_spec["auth_sha"]
        ):
            raise ValueError("H4 iteration-v3 fresh-v22 provenance binding drifted")
        if expert == "reverse" and manifest_auth.get(
            "legacy_reward_config_audit"
        ) != config_auth.get("legacy_reward_config_audit"):
            raise ValueError("reverse iteration-v3 legacy reward audit binding drifted")
    active_v4_flags = {
        key: all_iteration_flags[key]
        for key in (
            "forward_iteration_v4_contact_event_validity_persistence",
            "reverse_iteration_v4_residual_transfer_gain_024",
        )
    }
    active_v4_flag = next(
        (key for key, value in active_v4_flags.items() if value), None
    )
    if active_v4_flag is not None:
        v4_spec = _iteration_v4_spec(expert)
        if active_v4_flag != v4_spec["flag"]:
            raise ValueError("H4 iteration-v4 flag/expert binding drifted")
        manifest_auth = _require_mapping(
            manifest.get(v4_spec["auth_key"]), "manifest v4 authorization"
        )
        config_auth = _require_mapping(
            config.get(v4_spec["auth_key"]), "config v4 authorization"
        )
        result_path, _ = _verify_file_record(
            outputs.get("result"), label="outputs.result"
        )
        result_payload = _require_mapping(
            load_json_strict(result_path), "run result"
        )
        auth_sha = require_sha256(config_auth.get("sha256"), "v4 config auth SHA")
        result_auth_sha_key = f"{v4_spec['auth_key']}_sha256"
        inactive_v4_flag = (
            "reverse_iteration_v4_residual_transfer_gain_024"
            if expert == "forward"
            else "forward_iteration_v4_contact_event_validity_persistence"
        )
        expected_manifest_auth_keys = {"path", "sha256", "contract_id"}
        if expert == "reverse":
            expected_manifest_auth_keys.add("legacy_reward_config_audit")
        expected_substep_opt_in = expert == "forward"
        expected_authority_requirement = {
            "dynamic6_exact": True,
            "dynamic6_max_abs_error": 0.0,
            "dynamic6_field_count": 6,
            "dynamic6_field_count_exact": True,
            "saved_dynamic6_substep_count": 10,
            "saved_dynamic6_field_count": 6,
            "saved_dynamic6_field_count_exact": True,
            "saved_dynamic6_all_finite": True,
            "telemetry_force_shape": [2],
            "telemetry_force_shape_valid": True,
            "telemetry_force_all_finite": True,
            "count_totals_qualification_role": (
                "NON_QUALIFYING_FINITE_DIAGNOSTICS_ONLY"
            ),
            "host_count_multiplication_for_qualification": False,
            "numeric_tolerance_used": False,
            "authority_violation_count": 0.0,
            "assertion_token_sum": 0.0,
            "fail_closed_before_output_commit": True,
            "full_nonempty_episode_rows_required": True,
            "wiring_zero_episode_rows_require_compiled_assertion_evidence": True,
        }
        source_preflight = config.get("forward_v4_source_semantic_preflight")
        authority_runtime = result_payload.get(
            "forward_v4_single_authority_runtime"
        )
        if expert == "forward":
            _validate_forward_v4_single_authority_closure(
                config=config,
                manifest=manifest,
                result_payload=result_payload,
                outputs=outputs,
                status=status,
            )
            source_preflight = _require_mapping(
                source_preflight, "forward-v4 source-semantic preflight"
            )
            authority_runtime = _require_mapping(
                authority_runtime, "forward-v4 single-authority runtime"
            )
            expected_runtime_audit_mode = (
                H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                if status == "WIRING_PASS"
                else H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
            )
            curve_path, _ = _verify_file_record(
                outputs.get("training_curve"), label="outputs.training_curve"
            )
            curve_runtime = _validate_forward_v4_training_curve_runtime(
                curve_path, wiring_only=(status == "WIRING_PASS")
            )
            if status != "WIRING_PASS":
                result_final_metrics = _require_mapping(
                    result_payload.get("final_metrics"),
                    "forward-v4 result final metrics",
                )
                normalized_result_final_metrics: dict[str, float] = {}
                for key, value in result_final_metrics.items():
                    if (
                        not isinstance(key, str)
                        or not key.startswith("training/")
                        or not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not np.isfinite(float(value))
                    ):
                        raise ValueError(
                            "forward-v4 result final metrics are not exact finite "
                            "training scalars"
                        )
                    normalized_result_final_metrics[key] = float(value)
                if (
                    not normalized_result_final_metrics
                    or curve_runtime.get("final_metrics")
                    != normalized_result_final_metrics
                ):
                    raise ValueError(
                        "forward-v4 bound curve final metrics differ from run result"
                    )
            derived = _require_mapping(
                source_preflight.get("derived_diagnostics"),
                "forward-v4 excluded derived diagnostics",
            )
            derived_fields = _require_mapping(
                derived.get("fields"), "forward-v4 excluded diagnostic fields"
            )
            source_provenance = _require_mapping(
                source_preflight.get("source_provenance"),
                "forward-v4 official source provenance",
            )
            probe_input = _require_mapping(
                source_preflight.get("probe_input"),
                "forward-v4 source-semantic probe input",
            )
            derived_exact = True
            for field in ("cfrc_int", "cfrc_ext"):
                diagnostic = _require_mapping(
                    derived_fields.get(field),
                    f"forward-v4 excluded {field} diagnostic",
                )
                exact = diagnostic.get("exact")
                error = diagnostic.get("max_abs_error")
                derived_exact = bool(
                    derived_exact
                    and set(diagnostic) == {"exact", "max_abs_error"}
                    and isinstance(exact, bool)
                    and isinstance(error, (int, float))
                    and not isinstance(error, bool)
                    and np.isfinite(float(error))
                    and float(error) >= 0.0
                    and exact is (float(error) == 0.0)
                )
            source_root = PurePosixPath(
                str(source_provenance.get("source_root", ""))
            )
            expected_source_files = {
                "joystick": (
                    "playground/open_duck_mini_v2/joystick.py",
                    "95890569d971725308b5a9c0996bfa5fd9520479f014f325e810aa1db272eb9d",
                ),
                "mjx_env": (
                    ".venv/lib/python3.12/site-packages/"
                    "mujoco_playground/_src/mjx_env.py",
                    "c3f1cfe0de036c3ccbba46e8cdd661cb48bfea8f182955298205f17787f53dfe",
                ),
            }
            source_provenance_exact = source_root.is_absolute()
            for label, (relative, expected_sha) in expected_source_files.items():
                record = _require_mapping(
                    source_provenance.get(label),
                    f"forward-v4 official {label} provenance",
                )
                resolved = PurePosixPath(str(record.get("resolved_path", "")))
                source_provenance_exact = bool(
                    source_provenance_exact
                    and set(record) == {"resolved_path", "relative_path", "sha256"}
                    and record.get("relative_path") == relative
                    and record.get("sha256") == expected_sha
                    and resolved == source_root / PurePosixPath(relative)
                )
            if status == "WIRING_PASS":
                expected_runtime = {
                    "audit_mode": H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE,
                    "observed_episode_metric_rows": curve_runtime[
                        "observed_episode_metric_rows"
                    ],
                    "episode_metric_rows_exact_if_observed": True,
                    "source_semantic_preflight_passed": True,
                    "per_step_compiled_fail_closed_assertion_bound": True,
                    "completed_environment_interactions": H4_WIRING_INTERACTIONS,
                    "completed_training_steps": H4_WIRING_TRAINING_STEPS,
                    "completed_optimizer_updates": H4_WIRING_OPTIMIZER_UPDATES,
                    "progress_reached_final_interaction": True,
                    "final_params_all_finite": True,
                    "final_metrics_all_finite": True,
                    "source_and_teacher_unchanged": True,
                    "authority_violation_count": 0.0,
                    "assertion_token_sum": 0.0,
                    "passed": True,
                }
                runtime_boolean_keys = {
                    "episode_metric_rows_exact_if_observed",
                    "source_semantic_preflight_passed",
                    "per_step_compiled_fail_closed_assertion_bound",
                    "progress_reached_final_interaction",
                    "final_params_all_finite",
                    "final_metrics_all_finite",
                    "source_and_teacher_unchanged",
                    "passed",
                }
            else:
                expected_runtime = {
                    "audit_mode": H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE,
                    "dynamic6_exact": True,
                    "dynamic6_max_abs_error": 0.0,
                    "dynamic6_field_count": 6,
                    "dynamic6_field_count_exact": True,
                    "saved_dynamic6_substep_count": 10,
                    "saved_dynamic6_field_count": 6,
                    "saved_dynamic6_field_count_exact": True,
                    "saved_dynamic6_all_finite": True,
                    "telemetry_force_shape": [2],
                    "telemetry_force_shape_valid": True,
                    "telemetry_force_all_finite": True,
                    "observed_episode_metric_rows": curve_runtime[
                        "observed_episode_metric_rows"
                    ],
                    "authority_violation_count": 0.0,
                    "assertion_token_sum": 0.0,
                    "passed": True,
                }
                runtime_boolean_keys = {
                    "dynamic6_exact",
                    "dynamic6_field_count_exact",
                    "saved_dynamic6_field_count_exact",
                    "saved_dynamic6_all_finite",
                    "telemetry_force_shape_valid",
                    "telemetry_force_all_finite",
                    "passed",
                }
            runtime_contract_exact = bool(
                set(authority_runtime) == set(expected_runtime)
                and dict(authority_runtime) == expected_runtime
                and all(
                    authority_runtime.get(key) is True
                    for key in runtime_boolean_keys
                )
                and type(authority_runtime.get("observed_episode_metric_rows"))
                is int
                and authority_runtime.get("audit_mode")
                == expected_runtime_audit_mode
            )
            if (
                set(source_preflight)
                != {
                    "timing",
                    "reference_source",
                    "candidate_source",
                    "source_provenance",
                    "probe_input",
                    "qualifying_dynamic_state_fields",
                    "dynamic6_exact",
                    "dynamic6_max_abs_error",
                    "dynamic6_field_count",
                    "derived_diagnostics",
                    "observed_reference_count",
                    "passed",
                }
                or source_preflight.get("timing")
                != "ONCE_BEFORE_PPO_COLLECTION"
                or source_preflight.get("reference_source")
                != "OFFICIAL_MJX_ENV_STEP_WRAPPER_NSUBSTEPS_10"
                or source_preflight.get("candidate_source")
                != "SINGLE_INSTRUMENTED_TEN_SUBSTEP_SCAN_ENDPOINT"
                or set(source_provenance)
                != {
                    "source_root",
                    "joystick",
                    "mjx_env",
                    "step_source_sha256",
                    "step_source_semantics",
                    "all_files_under_requested_source_root",
                    "passed",
                }
                or source_provenance_exact is not True
                or source_provenance.get("all_files_under_requested_source_root")
                is not True
                or source_provenance.get("passed") is not True
                or source_provenance.get("step_source_sha256")
                != "26571e7510b2837dca07f69890dc26a89695dff4caa1fdc6a0d6736bd22da06b"
                or source_provenance.get("step_source_semantics")
                != (
                    "LAX_SCAN_XS_EMPTY_LENGTH_NSUBSTEPS_BODY_REPLACE_CTRL_"
                    "ACTION_THEN_MJX_STEP_RETURN_FINAL_CARRY"
                )
                or probe_input
                != {
                    "seed": 20260809,
                    "reset_noise_multiplier": 1.0,
                    "initial_state_source": "ENV_RESET_JAX_PRNGKEY_SEED",
                    "action_shape": [14],
                    "action_dtype": "float32",
                    "action_all_zero": True,
                }
                or source_preflight.get("qualifying_dynamic_state_fields")
                != [
                    "qpos",
                    "qvel",
                    "act",
                    "ctrl",
                    "time",
                    "qacc_warmstart",
                ]
                or source_preflight.get("dynamic6_exact") is not True
                or source_preflight.get("dynamic6_max_abs_error") != 0.0
                or source_preflight.get("dynamic6_field_count") != 6
                or source_preflight.get("observed_reference_count") != 1
                or source_preflight.get("passed") is not True
                or set(derived)
                != {
                    "qualification_role",
                    "fields",
                    "all_finite",
                    "exclusion_is_semantic_not_tolerance",
                    "numeric_tolerance_used",
                }
                or derived.get("qualification_role")
                != "NON_QUALIFYING_OBSERVED_DIAGNOSTICS_ONLY"
                or set(derived_fields) != {"cfrc_int", "cfrc_ext"}
                or derived_exact is not True
                or derived.get("all_finite") is not True
                or derived.get("exclusion_is_semantic_not_tolerance") is not True
                or derived.get("numeric_tolerance_used") is not False
                or runtime_contract_exact is not True
                or config.get(
                    "forward_v4_single_authority_runtime_requirement"
                )
                != expected_authority_requirement
                or manifest.get(
                    "forward_v4_single_authority_runtime_requirement"
                )
                != expected_authority_requirement
                or result_payload.get(
                    "forward_v4_single_authority_runtime_requirement"
                )
                != expected_authority_requirement
                or manifest.get("forward_v4_source_semantic_preflight")
                != source_preflight
                or result_payload.get("forward_v4_source_semantic_preflight")
                != source_preflight
                or manifest.get("forward_v4_single_authority_runtime")
                != authority_runtime
                or manifest.get(
                    "forward_v4_single_authority_runtime_audit_mode"
                )
                != expected_runtime_audit_mode
                or config.get(
                    "forward_v4_single_authority_runtime_audit_mode"
                )
                != expected_runtime_audit_mode
                or result_payload.get(
                    "forward_v4_single_authority_runtime_audit_mode"
                )
                != expected_runtime_audit_mode
            ):
                raise ValueError("forward-v4 single-authority closure drifted")
        elif any(
            key in payload
            for payload in (config, manifest, result_payload)
            for key in (
                "forward_v4_source_semantic_preflight",
                "forward_v4_single_authority_runtime",
                "forward_v4_single_authority_runtime_requirement",
                "forward_v4_single_authority_runtime_audit_mode",
            )
        ):
            raise ValueError(
                "reverse iteration-v4 must not bind forward single-authority data"
            )
        if (
            manifest.get(active_v4_flag) is not True
            or manifest.get(inactive_v4_flag, False) is not False
            or manifest.get("forward_iteration_v2") is not False
            or manifest.get("reverse_iteration_v2") is not False
            or manifest.get("forward_iteration_v3_touchdown_balance", False)
            is not False
            or manifest.get("reverse_iteration_v3_no_target_imitation", False)
            is not False
            or manifest.get("forward_v4_substep_contact")
            is not expected_substep_opt_in
            or manifest.get("authorized_iteration_v2_250k_contract_id") is not None
            or manifest.get("authorized_iteration_v3_250k_contract_id") is not None
            or manifest.get("authorized_iteration_v4_250k_contract_id")
            != v4_spec["contract"]
            or manifest.get("training_contract_id")
            != config.get("training_contract_id")
            or manifest.get("qualification_use") != config.get("qualification_use")
            or set(manifest_auth) != expected_manifest_auth_keys
            or manifest_auth.get("path") != config_auth.get("path")
            or manifest_auth.get("sha256") != auth_sha
            or manifest_auth.get("contract_id") != v4_spec["contract"]
            or config.get("initialization_source") != "V22_BRAX_CHECKPOINT"
            or config.get("trusted_h4_parent") is not None
            or config.get("pinned_v22_parent_tree_sha256")
            != PINNED_V22_PARENT_TREE_SHA256
            or config.get("reset_noise_multiplier") != 1.0
            or parent.get("kind") != "V22_BRAX_CHECKPOINT"
            or parent.get("sha256_tree_pre") != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("sha256_tree_post") != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("unchanged") is not True
            or any(label.startswith("h4_parent_") for label in normalized_source_hashes)
            or result_payload.get("expert") != expert
            or result_payload.get("status") != status
            or result_payload.get("training_contract_id")
            != config.get("training_contract_id")
            or result_payload.get("authorized_iteration_v2_250k_contract_id")
            is not None
            or result_payload.get("authorized_iteration_v3_250k_contract_id")
            is not None
            or result_payload.get("authorized_iteration_v4_250k_contract_id")
            != v4_spec["contract"]
            or result_payload.get("qualification_use")
            != config.get("qualification_use")
            or result_payload.get("forward_iteration_v2") is not False
            or result_payload.get("reverse_iteration_v2") is not False
            or result_payload.get("forward_iteration_v3_touchdown_balance", False)
            is not False
            or result_payload.get("reverse_iteration_v3_no_target_imitation", False)
            is not False
            or result_payload.get(active_v4_flag) is not True
            or result_payload.get(inactive_v4_flag, False) is not False
            or result_payload.get("forward_v4_substep_contact")
            is not expected_substep_opt_in
            or result_payload.get(result_auth_sha_key) != auth_sha
        ):
            raise ValueError("H4 iteration-v4 fresh-v22 provenance binding drifted")
        if expert == "reverse" and manifest_auth.get(
            "legacy_reward_config_audit"
        ) != config_auth.get("legacy_reward_config_audit"):
            raise ValueError("reverse iteration-v4 legacy reward audit binding drifted")
    active_v5_flags = {
        key: all_iteration_flags[key]
        for key in (
            "forward_v5_contact_pulse_abort_scale_only",
            "reverse_iteration_v5_no_contact_imitation",
        )
    }
    active_v5_flag = next(
        (key for key, value in active_v5_flags.items() if value), None
    )
    if active_v5_flag is not None:
        v5_spec = _iteration_v5_spec(expert)
        if active_v5_flag != v5_spec["flag"]:
            raise ValueError("H4 iteration-v5 flag/expert binding drifted")
        v5_paths = _validated_iteration_v5_source_paths(
            expert=expert,
            config=config,
            source_hashes=normalized_source_hashes,
        )
        del v5_paths
        manifest_auth = _require_mapping(
            manifest.get(v5_spec["auth_key"]), "manifest v5 authorization"
        )
        config_auth = _require_mapping(
            config.get(v5_spec["auth_key"]), "config v5 authorization"
        )
        result_path, _ = _verify_file_record(
            outputs.get("result"), label="outputs.result"
        )
        result_payload = _require_mapping(
            load_json_strict(result_path), "run result"
        )
        # Historical v5 bundles predate v6, so absent v6 flags mean false.
        # A later artifact may not graft a v6 claim onto a trusted v5 config.
        v5_mode_keys = tuple(all_iteration_flags)
        manifest_mode_values = {
            key: manifest.get(key, False) for key in v5_mode_keys
        }
        result_mode_values = {
            key: result_payload.get(key, False) for key in v5_mode_keys
        }
        v6_only_artifact_keys = {
            "authorized_iteration_v6_250k_contract_id",
            "iteration_v6_core_source",
            "forward_iteration_v6_contact_abort_island_only_authorization",
            "reverse_iteration_v6_absolute_full_leg_targets_authorization",
            "forward_iteration_v6_contact_abort_island_only_authorization_sha256",
            "reverse_iteration_v6_absolute_full_leg_targets_authorization_sha256",
            "forward_iteration_v6_reward_routing_runtime_requirement",
            "forward_iteration_v6_reward_routing_runtime",
            "reverse_iteration_v6_decoder_runtime_requirement",
            "reverse_iteration_v6_decoder_runtime",
            "reward_routing_contract",
            "action_parameterization_contract",
            "teacher_timing_contract",
            "iteration_v6_artifact_cross_binding",
        }
        expected_execution_id = (
            v5_spec["wiring_contract"]
            if status == "WIRING_PASS"
            else v5_spec["contract"]
        )
        expected_manifest_auth_keys = {
            "path",
            "sha256",
            "contract_id",
            "bound_historical_v4_sources",
        }
        if expert == "reverse":
            expected_manifest_auth_keys.update(
                {
                    "legacy_reward_config_audit",
                    "rejected_v4_diagnostic_promotion_allowed",
                }
            )
        result_auth_sha_key = f"{v5_spec['auth_key']}_sha256"
        if (
            any(
                not isinstance(value, bool)
                for value in (*manifest_mode_values.values(), *result_mode_values.values())
            )
            or manifest_mode_values.get(active_v5_flag) is not True
            or result_mode_values.get(active_v5_flag) is not True
            or sum(value is True for value in manifest_mode_values.values()) != 1
            or sum(value is True for value in result_mode_values.values()) != 1
            or manifest.get("authorized_iteration_v2_250k_contract_id") is not None
            or manifest.get("authorized_iteration_v3_250k_contract_id") is not None
            or manifest.get("authorized_iteration_v4_250k_contract_id") is not None
            or manifest.get("authorized_iteration_v5_250k_contract_id")
            != v5_spec["contract"]
            or result_payload.get("authorized_iteration_v2_250k_contract_id")
            is not None
            or result_payload.get("authorized_iteration_v3_250k_contract_id")
            is not None
            or result_payload.get("authorized_iteration_v4_250k_contract_id")
            is not None
            or result_payload.get("authorized_iteration_v5_250k_contract_id")
            != v5_spec["contract"]
            or any(
                key in artifact
                for artifact in (config, manifest, result_payload)
                for key in v6_only_artifact_keys
            )
            or manifest.get("training_contract_id") != expected_execution_id
            or config.get("training_contract_id") != expected_execution_id
            or result_payload.get("training_contract_id") != expected_execution_id
            or set(manifest_auth) != expected_manifest_auth_keys
            or manifest_auth.get("path") != config_auth.get("path")
            or manifest_auth.get("sha256") != v5_spec["auth_sha"]
            or manifest_auth.get("contract_id") != v5_spec["contract"]
            or manifest_auth.get("bound_historical_v4_sources")
            != config_auth.get("bound_historical_v4_sources")
            or result_payload.get(result_auth_sha_key) != v5_spec["auth_sha"]
            or config.get("initialization_source") != "V22_BRAX_CHECKPOINT"
            or config.get("trusted_h4_parent") is not None
            or config.get("pinned_v22_parent_tree_sha256")
            != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("kind") != "V22_BRAX_CHECKPOINT"
            or parent.get("sha256_tree_pre") != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("sha256_tree_post") != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("unchanged") is not True
            or result_payload.get("expert") != expert
            or result_payload.get("status") != status
            or manifest.get("qualification_use") != config.get("qualification_use")
            or result_payload.get("qualification_use")
            != config.get("qualification_use")
        ):
            raise ValueError("H4 iteration-v5 config/manifest/result binding drifted")
        expected_substep = expert == "forward"
        if (
            manifest.get("forward_v4_substep_contact") is not expected_substep
            or config.get("forward_v4_substep_contact") is not expected_substep
            or result_payload.get("forward_v4_substep_contact") is not expected_substep
        ):
            raise ValueError("H4 iteration-v5 core opt-in binding drifted")
        if expert == "forward":
            _validate_forward_v4_single_authority_closure(
                config=config,
                manifest=manifest,
                result_payload=result_payload,
                outputs=outputs,
                status=status,
            )
            source_preflight = _require_mapping(
                config.get("forward_v4_source_semantic_preflight"),
                "forward-v5 source-semantic preflight",
            )
            authority_runtime = _require_mapping(
                result_payload.get("forward_v4_single_authority_runtime"),
                "forward-v5 single-authority runtime",
            )
            requirement = _require_mapping(
                config.get("forward_v4_single_authority_runtime_requirement"),
                "forward-v5 single-authority requirement",
            )
            curve_path, _ = _verify_file_record(
                outputs.get("training_curve"), label="outputs.training_curve"
            )
            curve_runtime = _validate_forward_v4_training_curve_runtime(
                curve_path, wiring_only=(status == "WIRING_PASS")
            )
            expected_mode = (
                H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                if status == "WIRING_PASS"
                else H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
            )
            required_runtime_true = {
                "passed",
                "source_semantic_preflight_passed",
                "per_step_compiled_fail_closed_assertion_bound",
                "progress_reached_final_interaction",
                "final_params_all_finite",
                "final_metrics_all_finite",
                "source_and_teacher_unchanged",
            } if status == "WIRING_PASS" else {
                "passed",
                "dynamic6_exact",
                "dynamic6_field_count_exact",
                "saved_dynamic6_field_count_exact",
                "saved_dynamic6_all_finite",
                "telemetry_force_shape_valid",
                "telemetry_force_all_finite",
            }
            if (
                source_preflight.get("passed") is not True
                or source_preflight.get("dynamic6_exact") is not True
                or source_preflight.get("dynamic6_max_abs_error") != 0.0
                or source_preflight.get("dynamic6_field_count") != 6
                or source_preflight.get("observed_reference_count") != 1
                or authority_runtime.get("audit_mode") != expected_mode
                or any(authority_runtime.get(key) is not True for key in required_runtime_true)
                or authority_runtime.get("authority_violation_count") != 0.0
                or authority_runtime.get("assertion_token_sum") != 0.0
                or type(authority_runtime.get("observed_episode_metric_rows")) is not int
                or authority_runtime.get("observed_episode_metric_rows")
                != curve_runtime.get("observed_episode_metric_rows")
                or requirement.get("fail_closed_before_output_commit") is not True
                or requirement.get("numeric_tolerance_used") is not False
                or manifest.get("forward_v4_source_semantic_preflight")
                != source_preflight
                or result_payload.get("forward_v4_source_semantic_preflight")
                != source_preflight
                or manifest.get("forward_v4_single_authority_runtime")
                != authority_runtime
                or manifest.get("forward_v4_single_authority_runtime_requirement")
                != requirement
                or result_payload.get("forward_v4_single_authority_runtime_requirement")
                != requirement
                or config.get("forward_v4_single_authority_runtime_audit_mode")
                != expected_mode
                or manifest.get("forward_v4_single_authority_runtime_audit_mode")
                != expected_mode
                or result_payload.get("forward_v4_single_authority_runtime_audit_mode")
                != expected_mode
            ):
                raise ValueError("forward-v5 v4 runtime/curve gates drifted")
        else:
            legacy_audit = config_auth.get("legacy_reward_config_audit")
            if (
                manifest_auth.get("legacy_reward_config_audit") != legacy_audit
                or result_payload.get("legacy_reward_config_audit") != legacy_audit
                or manifest_auth.get("rejected_v4_diagnostic_promotion_allowed")
                is not False
                or result_payload.get("rejected_v4_diagnostic_promotion_allowed")
                is not False
                or config.get("backward_residual_scale") != H4_REVERSE_RESIDUAL_SCALE
                or result_payload.get("backward_residual_scale")
                != H4_REVERSE_RESIDUAL_SCALE
            ):
                raise ValueError("reverse-v5 residual/legacy/diagnostic binding drifted")
    active_v6_flags = {
        key: all_iteration_flags[key]
        for key in (
            "forward_iteration_v6_contact_abort_island_only",
            "reverse_iteration_v6_absolute_full_leg_targets",
        )
    }
    active_v6_flag = next(
        (key for key, value in active_v6_flags.items() if value), None
    )
    if active_v6_flag is not None:
        v6_spec = _iteration_v6_spec(expert)
        if active_v6_flag != v6_spec["flag"]:
            raise ValueError("H4 iteration-v6 flag/expert binding drifted")
        result_path, _ = _verify_file_record(
            outputs.get("result"), label="outputs.result"
        )
        result_payload = _require_mapping(
            load_json_strict(result_path), "run result"
        )
        _validate_iteration_v6_artifact_location_closure(
            expert=expert,
            config=config,
            manifest=manifest,
            result_payload=result_payload,
        )
        _validated_iteration_v6_source_paths(
            expert=expert,
            config=config,
            source_hashes=normalized_source_hashes,
        )
        manifest_auth = _require_mapping(
            manifest.get(v6_spec["auth_key"]), "manifest v6 authorization"
        )
        config_auth = _require_mapping(
            config.get(v6_spec["auth_key"]), "config v6 authorization"
        )
        _validate_iteration_v6_core_source_artifact_closure(
            expert=expert,
            config=config,
            manifest=manifest,
            result_payload=result_payload,
            source_hashes=normalized_source_hashes,
        )
        if (
            "iteration_v6_artifact_cross_binding" in config
            or not _iteration_v6_artifact_cross_binding_is_exact(
                manifest.get("iteration_v6_artifact_cross_binding")
            )
            or not _iteration_v6_artifact_cross_binding_is_exact(
                result_payload.get("iteration_v6_artifact_cross_binding")
            )
        ):
            raise ValueError(
                "H4 iteration-v6 artifact cross-binding evidence drifted"
            )
        manifest_mode_values = {
            key: manifest.get(key) for key in all_iteration_flags
        }
        result_mode_values = {
            key: result_payload.get(key) for key in all_iteration_flags
        }
        expected_execution_id = (
            v6_spec["wiring_contract"]
            if status == "WIRING_PASS"
            else v6_spec["contract"]
        )
        expected_manifest_auth_keys = {
            "path",
            "sha256",
            "contract_id",
            "bound_historical_v5_sources",
        }
        if expert == "reverse":
            expected_manifest_auth_keys.update(
                {
                    "legacy_reward_config_audit",
                    "h4_parent_checkpoint_allowed",
                    "v4_gain_inherited",
                    "v5_parent_checkpoint_inherited",
                }
            )
        result_auth_sha_key = f"{v6_spec['auth_key']}_sha256"
        if (
            any(
                not isinstance(value, bool)
                for value in (*manifest_mode_values.values(), *result_mode_values.values())
            )
            or manifest_mode_values.get(active_v6_flag) is not True
            or result_mode_values.get(active_v6_flag) is not True
            or sum(value is True for value in manifest_mode_values.values()) != 1
            or sum(value is True for value in result_mode_values.values()) != 1
            or any(
                manifest.get(f"authorized_iteration_v{version}_250k_contract_id")
                is not None
                for version in (2, 3, 4, 5)
            )
            or any(
                result_payload.get(f"authorized_iteration_v{version}_250k_contract_id")
                is not None
                for version in (2, 3, 4, 5)
            )
            or manifest.get("authorized_iteration_v6_250k_contract_id")
            != v6_spec["contract"]
            or result_payload.get("authorized_iteration_v6_250k_contract_id")
            != v6_spec["contract"]
            or manifest.get("training_contract_id") != expected_execution_id
            or config.get("training_contract_id") != expected_execution_id
            or result_payload.get("training_contract_id") != expected_execution_id
            or set(manifest_auth) != expected_manifest_auth_keys
            or manifest_auth.get("path") != config_auth.get("path")
            or manifest_auth.get("sha256") != v6_spec["auth_sha"]
            or manifest_auth.get("contract_id") != v6_spec["contract"]
            or manifest_auth.get("bound_historical_v5_sources")
            != config_auth.get("bound_historical_v5_sources")
            or result_payload.get(result_auth_sha_key) != v6_spec["auth_sha"]
            or config.get("initialization_source") != "V22_BRAX_CHECKPOINT"
            or config.get("trusted_h4_parent") is not None
            or config.get("pinned_v22_parent_tree_sha256")
            != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("kind") != "V22_BRAX_CHECKPOINT"
            or parent.get("sha256_tree_pre") != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("sha256_tree_post") != PINNED_V22_PARENT_TREE_SHA256
            or parent.get("unchanged") is not True
            or result_payload.get("expert") != expert
            or result_payload.get("status") != status
            or manifest.get("qualification_use") != config.get("qualification_use")
            or result_payload.get("qualification_use")
            != config.get("qualification_use")
        ):
            raise ValueError("H4 iteration-v6 config/manifest/result binding drifted")
        expected_substep = expert == "forward"
        if (
            manifest.get("forward_v4_substep_contact") is not expected_substep
            or config.get("forward_v4_substep_contact") is not expected_substep
            or result_payload.get("forward_v4_substep_contact") is not expected_substep
        ):
            raise ValueError("H4 iteration-v6 core opt-in binding drifted")
        if expert == "forward":
            routing = config.get("reward_routing_contract")
            if (
                not _json_type_and_value_exact(
                    manifest.get("reward_routing_contract"), routing
                )
                or not _json_type_and_value_exact(
                    result_payload.get("reward_routing_contract"), routing
                )
            ):
                raise ValueError("forward iteration-v6 reward routing cross-binding drifted")
            _validate_forward_v4_single_authority_closure(
                config=config,
                manifest=manifest,
                result_payload=result_payload,
                outputs=outputs,
                status=status,
            )
        else:
            action = config.get("action_parameterization_contract")
            timing = config.get("teacher_timing_contract")
            if (
                not _json_type_and_value_exact(
                    manifest.get("action_parameterization_contract"), action
                )
                or not _json_type_and_value_exact(
                    result_payload.get("action_parameterization_contract"), action
                )
                or not _json_type_and_value_exact(
                    manifest.get("teacher_timing_contract"), timing
                )
                or not _json_type_and_value_exact(
                    result_payload.get("teacher_timing_contract"), timing
                )
                or not _json_type_and_value_exact(
                    config.get("backward_residual_scale"), 0.0
                )
                or not _json_type_and_value_exact(
                    manifest.get("backward_residual_scale"), 0.0
                )
                or not _json_type_and_value_exact(
                    result_payload.get("backward_residual_scale"), 0.0
                )
                or not _json_type_and_value_exact(
                    manifest_auth.get("legacy_reward_config_audit"),
                    config_auth.get("legacy_reward_config_audit"),
                )
                or not _json_type_and_value_exact(
                    result_payload.get("legacy_reward_config_audit"),
                    config_auth.get("legacy_reward_config_audit"),
                )
                or config_auth.get("h4_parent_checkpoint_allowed") is not False
                or manifest_auth.get("h4_parent_checkpoint_allowed") is not False
                or result_payload.get("h4_parent_checkpoint_allowed") is not False
                or config_auth.get("v4_gain_inherited") is not False
                or manifest_auth.get("v4_gain_inherited") is not False
                or result_payload.get("v4_gain_inherited") is not False
                or config_auth.get("v5_parent_checkpoint_inherited") is not False
                or manifest_auth.get("v5_parent_checkpoint_inherited") is not False
                or result_payload.get("v5_parent_checkpoint_inherited") is not False
                or result_payload.get("teacher_target_contribution_zero") is not True
            ):
                raise ValueError(
                    "reverse iteration-v6 action/timing/residual cross-binding drifted"
                )
        _validate_iteration_v6_runtime_closure(
            expert=expert,
            config=config,
            manifest=manifest,
            result_payload=result_payload,
            outputs=outputs,
            status=status,
        )
    _validated_forward_iteration_v3_touchdown_balance_source_paths(
        config=config,
        source_hashes=normalized_source_hashes,
    )
    _validated_reverse_iteration_v3_no_target_imitation_source_paths(
        config=config,
        source_hashes=normalized_source_hashes,
    )
    _validated_forward_iteration_v4_contact_event_validity_persistence_source_paths(
        config=config,
        source_hashes=normalized_source_hashes,
    )
    _validated_reverse_iteration_v4_residual_transfer_gain_024_source_paths(
        config=config,
        source_hashes=normalized_source_hashes,
    )
    _validated_forward_iteration_v5_contact_pulse_abort_scale_only_source_paths(
        config=config,
        source_hashes=normalized_source_hashes,
    )
    _validated_reverse_iteration_v5_no_contact_imitation_source_paths(
        config=config,
        source_hashes=normalized_source_hashes,
    )
    _validated_forward_iteration_v6_contact_abort_island_only_source_paths(
        config=config,
        source_hashes=normalized_source_hashes,
    )
    _validated_reverse_iteration_v6_absolute_full_leg_targets_source_paths(
        config=config,
        source_hashes=normalized_source_hashes,
    )
    if expert == "reverse":
        selected_config = _require_mapping(
            config.get("selected_reverse_teacher"), "selected reverse teacher"
        )
        composition_config = _require_mapping(
            config.get("reverse_composition_authorization"),
            "reverse composition authorization",
        )
        reverse_spec_config = _require_mapping(
            config.get("reverse_minimum_spec"), "reverse minimum spec"
        )
        if (
            normalized_source_hashes.get("selected_reverse_teacher", {}).get(
                "sha256"
            )
            != PINNED_SELECTED_REVERSE_TEACHER_SHA256
            or selected_config.get("sha256")
            != PINNED_SELECTED_REVERSE_TEACHER_SHA256
            or normalized_source_hashes.get(
                "reverse_composition_authorization", {}
            ).get("sha256")
            != PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
            or composition_config.get("sha256")
            != PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
            or normalized_source_hashes.get("reverse_minimum_spec", {}).get(
                "sha256"
            )
            != PINNED_REVERSE_MINIMUM_SPEC_SHA256
            or reverse_spec_config.get("sha256")
            != PINNED_REVERSE_MINIMUM_SPEC_SHA256
            or config.get("backward_residual_scale")
            != (
                H4_REVERSE_ITERATION_V6_RESIDUAL_AUTHORITY_SCALE
                if active_v6_flag is not None
                else
                H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN
                if active_v4_flag is not None
                else H4_REVERSE_RESIDUAL_SCALE
            )
            or composition_config.get("status")
            != "SIMULATION_TRAINING_COMPOSITION_AUTHORIZED_NOT_ADOPTED"
            or composition_config.get("standalone_direct_runtime_allowed") is not False
            or composition_config.get("adoption_allowed") is not False
            or composition_config.get("release_allowed") is not False
            or composition_config.get("hardware_allowed") is not False
        ):
            raise ValueError("reverse teacher/composition authorization binding drifted")
        _validated_reverse_iteration_v2_source_paths(
            config=config,
            source_hashes=normalized_source_hashes,
        )
    else:
        forward_config = _require_mapping(
            config.get("forward_minimum_spec"), "forward minimum spec"
        )
        if (
            normalized_source_hashes.get("forward_minimum_spec", {}).get("sha256")
            != PINNED_FORWARD_MINIMUM_SPEC_SHA256
            or forward_config.get("sha256") != PINNED_FORWARD_MINIMUM_SPEC_SHA256
            or forward_config.get("canonical_sha256")
            != PINNED_FORWARD_MINIMUM_SPEC_CANONICAL_SHA256
        ):
            raise ValueError("forward minimum retraining spec binding drifted")
        _validated_forward_iteration_v2_source_paths(
            config=config,
            source_hashes=normalized_source_hashes,
        )
    if status == "WIRING_PASS":
        expected_wiring = True
        expected_activity = "PPO_WIRING_TRAINING"
        expected_shape = {
            "num_timesteps": H4_WIRING_INTERACTIONS,
            "num_envs": 2,
            "unroll_length": 20,
            "batch_size": 1,
            "num_minibatches": 2,
            "num_updates_per_batch": 1,
            "num_evals": 1,
        }
        expected_interactions_per_step = H4_WIRING_INTERACTIONS
        expected_training_steps = H4_WIRING_TRAINING_STEPS
        expected_optimizer_updates = H4_WIRING_OPTIMIZER_UPDATES
        if not allow_wiring_diagnostic:
            raise ValueError("WIRING_PASS is diagnostic-only and requires opt-in")
    else:
        expected_wiring = False
        expected_activity = "PPO_PILOT_TRAINING"
        expected_shape = {
            "num_timesteps": H4_PILOT_INTERACTIONS,
            "num_envs": 1250,
            "unroll_length": 20,
            "batch_size": 125,
            "num_minibatches": 20,
            "num_updates_per_batch": 4,
            "num_evals": 2,
        }
        expected_interactions_per_step = 50_000
        expected_training_steps = H4_PILOT_TRAINING_STEPS
        expected_optimizer_updates = H4_PILOT_OPTIMIZER_UPDATES

    shape = _require_mapping(config.get("shape"), "H4 training shape")
    ppo = _require_mapping(config.get("ppo"), "H4 PPO config")
    ppo_shape = {key: ppo.get(key) for key in expected_shape}
    promotion_protocol = _require_mapping(
        config.get("promotion_protocol"), "H4 promotion protocol"
    )
    if (
        manifest.get("wiring_only") is not expected_wiring
        or config.get("wiring_only") is not expected_wiring
        or activity != expected_activity
        or manifest.get("requested_environment_interactions")
        != expected_shape["num_timesteps"]
        or dict(shape) != expected_shape
        or ppo_shape != expected_shape
        or config.get("interactions_per_training_step")
        != expected_interactions_per_step
        or config.get("expected_training_steps") != expected_training_steps
        or config.get("expected_optimizer_updates")
        != expected_optimizer_updates
        or config.get("promotion_evidence") is not None
        or promotion_protocol.get("candidate_stage_interactions")
        != H4_PILOT_INTERACTIONS
        or promotion_protocol.get(
            "candidate_training_steps_of_50000_interactions"
        )
        != H4_PILOT_TRAINING_STEPS
        or promotion_protocol.get("fixed_failure3_seeds")
        != list(H4_STRICT_SEEDS[expert])
        or promotion_protocol.get("promoted_stage_interactions") != 1_000_000
    ):
        raise ValueError(
            "H4 training stage is not exact 40/1/2 wiring or 250k/5/400 pilot"
        )

    result_path, _ = _verify_file_record(outputs.get("result"), label="outputs.result")
    curve_path, _ = _verify_file_record(
        outputs.get("training_curve"), label="outputs.training_curve"
    )
    result = _require_mapping(load_json_strict(result_path), "run result")
    result_params = _require_mapping(result.get("final_params"), "result params")
    final_metrics = _require_mapping(result.get("final_metrics"), "final metrics")
    if (
        result.get("schema_version") != 1
        or result.get("status") != status
        or result.get("hardware_deployment") != "PROHIBITED"
        or result.get("expert") != expert
        or result.get("activity") != activity
        or result.get("environment_interactions")
        != expected_shape["num_timesteps"]
        or result.get("optimizer_updates") != expected_optimizer_updates
        or result_params.get("sha256") != actual_params
        or _recorded_path(result_params, "result params") != params_resolved
        or result.get("final_metrics_all_finite") is not True
        or not isinstance(result.get("final_metrics_nonzero_count"), int)
        or result.get("final_metrics_nonzero_count", 0) <= 0
        or not final_metrics
        or result.get("source_and_teacher_unchanged") is not True
    ):
        raise ValueError("H4 run result/manifest binding drifted")
    training_provenance = _validate_h4_gpu_training_provenance(
        config=config,
        manifest=manifest,
        result=result,
    )
    run_directory = manifest_resolved.parent
    if (
        params_resolved.parent != run_directory
        or config_path.parent != run_directory
        or result_path.parent != run_directory
        or curve_path.parent != run_directory
        or config_path.name != "resolved_config.json"
        or result_path.name != "run_result.json"
        or curve_path.name != "training_curve.csv"
        or run_directory != run_root_resolved / str(expert) / run_name
        or Path(config.get("output_dir", run_directory)).resolve() != run_directory
    ):
        raise ValueError(
            "H4 runner outputs are not one exact trusted local run-directory closure"
        )

    return TrustedH4Bundle(
        params_path=params_resolved,
        params_sha256=actual_params,
        manifest_path=manifest_resolved,
        manifest_sha256=actual_manifest,
        manifest=manifest,
        config_path=config_path,
        config_sha256=config_sha,
        config_canonical_sha256=config_canonical,
        config=config,
        source_hashes=normalized_source_hashes,
        source_hashes_canonical_sha256=canonical_json_sha256(
            normalized_source_hashes
        ),
        status=str(status),
        run_name=run_name,
        expert=expert,
        activity=activity,
        training_provenance=training_provenance,
        training_provenance_sha256=canonical_json_sha256(training_provenance),
    )


def validate_h4_training_source_closure(
    bundle: TrustedH4Bundle,
    expected_paths: Mapping[str, Path],
) -> TrustedH4Bundle:
    """Bind every runner source record to its reconstructed current file.

    The manifest is an immutable record of what training imported.  This
    second gate independently reconstructs that complete label set from the
    current source stack and refuses missing, extra, redirected, or changed
    files.  ``load_trusted_h4_params`` requires the returned closed bundle, so
    a params pickle cannot be restored if a caller forgets this gate.
    """

    if not isinstance(bundle, TrustedH4Bundle):
        raise TypeError("source closure requires a validated H4 bundle")
    if bundle.source_closure_audit is not None:
        raise ValueError("H4 source closure was already validated")
    expected_labels = set(expected_paths)
    recorded_labels = set(bundle.source_hashes)
    if expected_labels != recorded_labels:
        missing = sorted(recorded_labels - expected_labels)
        extra = sorted(expected_labels - recorded_labels)
        raise ValueError(
            "H4 training source label closure mismatch: "
            f"unreconstructed={missing}, unexpected={extra}"
        )
    verified: dict[str, dict[str, str]] = {}
    seen_paths: dict[Path, str] = {}
    authorized_v4_aliases = {
        frozenset(
            {
                "h4_alignment",
                "forward_iteration_v4_source_h4_training_alignment",
            }
        ),
        frozenset(
            {"h4_runner", "forward_iteration_v4_source_h4_runner"}
        ),
        frozenset(
            {
                "h4_alignment",
                "reverse_iteration_v4_source_h4_training_alignment",
            }
        ),
        frozenset(
            {"h4_runner", "reverse_iteration_v4_source_h4_runner"}
        ),
        frozenset(
            {
                "h4_alignment",
                "forward_iteration_v5_current_source_h4_training_alignment",
            }
        ),
        frozenset(
            {"h4_runner", "forward_iteration_v5_current_source_h4_runner"}
        ),
        frozenset(
            {
                "h4_alignment",
                "reverse_iteration_v5_current_source_h4_training_alignment",
            }
        ),
        frozenset(
            {"h4_runner", "reverse_iteration_v5_current_source_h4_runner"}
        ),
        frozenset(
            {
                "h4_alignment",
                "forward_iteration_v6_current_source_h4_training_alignment",
            }
        ),
        frozenset(
            {"h4_runner", "forward_iteration_v6_current_source_h4_runner"}
        ),
        frozenset(
            {
                "h4_alignment",
                "reverse_iteration_v6_current_source_h4_training_alignment",
            }
        ),
        frozenset(
            {"h4_runner", "reverse_iteration_v6_current_source_h4_runner"}
        ),
            frozenset(
                {
                    "selected_reverse_teacher",
                    "reverse_iteration_v6_selected_reverse_teacher",
                }
            ),
    }
    accepted_aliases: list[tuple[str, str]] = []
    for label in sorted(recorded_labels):
        expected_path = Path(expected_paths[label]).resolve()
        if expected_path in seen_paths:
            pair = frozenset({seen_paths[expected_path], label})
            if pair not in authorized_v4_aliases:
                raise ValueError(f"duplicate reconstructed source path: {expected_path}")
            accepted_aliases.append(tuple(sorted(pair)))
        else:
            seen_paths[expected_path] = label
        record = bundle.source_hashes[label]
        recorded_path = Path(record["path"]).resolve()
        if recorded_path != expected_path:
            raise ValueError(f"H4 training source path mismatch for {label}")
        actual_sha = sha256_file(expected_path)
        if actual_sha != record["sha256"]:
            raise ValueError(f"H4 training source SHA256 mismatch for {label}")
        verified[label] = {"path": str(expected_path), "sha256": actual_sha}
    if verified != dict(bundle.source_hashes):
        raise RuntimeError("H4 source closure normalization drifted")
    audit = {
        "exact_label_set": True,
        "exact_paths": True,
        "all_files_sha256_verified": True,
        "duplicate_paths_rejected": True,
        "authorized_v4_source_aliases": sorted(accepted_aliases),
        "source_count": len(verified),
        "canonical_sha256": canonical_json_sha256(verified),
        "passed": True,
    }
    if audit["canonical_sha256"] != bundle.source_hashes_canonical_sha256:
        raise ValueError("H4 training source closure canonical SHA256 mismatch")
    return replace(bundle, source_closure_audit=audit)


def reconstruct_h4_training_source_paths(
    *,
    bundle: TrustedH4Bundle,
    experiment_root: Path,
    legacy_trainer_path: Path,
    alignment_path: Path,
    runner_path: Path,
    reverse_composition_validator_path: Path,
    stack: Mapping[str, Any],
    ppo_checkpoint_path: Path,
    generated_paths: Mapping[str, Path],
    legacy_teacher_gaits: Mapping[str, Path],
) -> dict[str, Path]:
    """Independently reconstruct the exact source map emitted by the runner."""

    root = Path(experiment_root).resolve()

    def module_path(key: str) -> Path:
        module = stack.get(key)
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"training stack module {key} has no source path")
        return Path(raw).resolve()

    def generated_path(key: str) -> Path:
        if key not in generated_paths:
            raise ValueError(f"generated training source {key} is missing")
        return Path(generated_paths[key]).resolve()

    def teacher_path(key: str) -> Path:
        if key not in legacy_teacher_gaits:
            raise ValueError(f"legacy teacher source {key} is missing")
        return Path(legacy_teacher_gaits[key]).resolve()

    def config_record_path(key: str) -> Path:
        record = _require_mapping(bundle.config.get(key), f"config {key}")
        path = _recorded_path(record, f"config {key}")
        source = _require_mapping(bundle.source_hashes.get(key), f"source {key}")
        if path != Path(source.get("path", "")).resolve():
            raise ValueError(f"config/source path binding drifted for {key}")
        if record.get("sha256") != source.get("sha256"):
            raise ValueError(f"config/source SHA256 binding drifted for {key}")
        return path

    paths = {
        "legacy_trainer": Path(legacy_trainer_path).resolve(),
        "h4_alignment": Path(alignment_path).resolve(),
        "h4_runner": Path(runner_path).resolve(),
        "h4_contract_module": root / "safe_gait_experts" / "contract.py",
        "h4_contract_json": root / "contract.json",
        "safe_randomization": root
        / "safe_gait_experts"
        / "safe_randomization.py",
        "bounded_reward": root / "safe_gait_experts" / "reward.py",
        "central_gait_quality": root / "safe_gait_experts" / "gait_quality.py",
        "central_routed_evaluation": root
        / "safe_gait_experts"
        / "routed_evaluation.py",
        "central_evaluator": root / "scripts" / "evaluate_routed_transitions.py",
        "source_joystick": module_path("joystick"),
        "source_constants": module_path("constants"),
        "brax_ppo_train": module_path("ppo"),
        "brax_ppo_networks": module_path("ppo_networks"),
        "brax_ppo_checkpoint": Path(ppo_checkpoint_path).resolve(),
        "mujoco_playground_wrapper": module_path("wrapper"),
        "mujoco_playground_locomotion_params": module_path(
            "locomotion_params"
        ),
        "generated_manifest": generated_path("manifest"),
        "generated_scene": generated_path("scene"),
        "generated_reference": generated_path("reference"),
        "legacy_backward_teacher": teacher_path("backward"),
        "legacy_backward_left_teacher": teacher_path("backward_left"),
        "legacy_backward_right_teacher": teacher_path("backward_right"),
    }
    if bundle.expert == "forward":
        paths["forward_minimum_spec"] = config_record_path(
            "forward_minimum_spec"
        )
        paths.update(
            _validated_forward_iteration_v2_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )
        paths.update(
            _validated_forward_iteration_v3_touchdown_balance_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )
        paths.update(
            _validated_forward_iteration_v4_contact_event_validity_persistence_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )
        paths.update(
            _validated_forward_iteration_v5_contact_pulse_abort_scale_only_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )
        paths.update(
            _validated_forward_iteration_v6_contact_abort_island_only_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )
    else:
        paths["selected_reverse_teacher"] = config_record_path(
            "selected_reverse_teacher"
        )
        paths["reverse_minimum_spec"] = config_record_path(
            "reverse_minimum_spec"
        )
        paths["reverse_composition_authorization"] = config_record_path(
            "reverse_composition_authorization"
        )
        paths["reverse_composition_validator"] = Path(
            reverse_composition_validator_path
        ).resolve()
        paths.update(
            _validated_reverse_iteration_v2_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )
        paths.update(
            _validated_reverse_iteration_v3_no_target_imitation_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )
        paths.update(
            _validated_reverse_iteration_v4_residual_transfer_gain_024_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )
        paths.update(
            _validated_reverse_iteration_v5_no_contact_imitation_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )
        paths.update(
            _validated_reverse_iteration_v6_absolute_full_leg_targets_source_paths(
                config=bundle.config,
                source_hashes=bundle.source_hashes,
            )
        )

    parent = bundle.config.get("trusted_h4_parent")
    if parent is not None:
        parent_record = _require_mapping(parent, "trusted_h4_parent")
        parent_params = Path(parent_record.get("params_path", "")).resolve()
        parent_manifest = Path(parent_record.get("manifest_path", "")).resolve()
        if (
            sha256_file(parent_params)
            != require_sha256(
                parent_record.get("params_sha256"),
                "trusted_h4_parent.params_sha256",
            )
            or sha256_file(parent_manifest)
            != require_sha256(
                parent_record.get("manifest_sha256"),
                "trusted_h4_parent.manifest_sha256",
            )
        ):
            raise ValueError("trusted H4 parent params/manifest SHA256 drifted")
        parent_payload = _require_mapping(
            load_json_strict(parent_manifest), "trusted H4 parent manifest"
        )
        parent_config_record = _require_mapping(
            parent_payload.get("resolved_config"),
            "trusted H4 parent resolved config",
        )
        parent_config = _recorded_path(
            parent_config_record, "trusted H4 parent resolved config"
        )
        if (
            parent_config_record.get("sha256")
            != parent_record.get("resolved_config_sha256")
            or sha256_file(parent_config) != parent_record.get("resolved_config_sha256")
        ):
            raise ValueError("trusted H4 parent resolved config SHA256 drifted")
        paths.update(
            {
                "h4_parent_params": parent_params,
                "h4_parent_manifest": parent_manifest,
                "h4_parent_resolved_config": parent_config,
            }
        )
    return {label: Path(path).resolve() for label, path in paths.items()}


def _numeric_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    label: str,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != dtype:
        raise ValueError(f"{label} must be {dtype} with shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def validate_h4_params(params: Any) -> dict[str, Any]:
    """Validate the exact actor116/critic227 Brax PPO params tree."""

    if not isinstance(params, (list, tuple)) or len(params) != 3:
        raise ValueError("H4 PPO params must contain exactly normalizer, actor, critic")
    normalizer, actor, critic = params
    for attribute in ("mean", "std", "summed_variance", "count"):
        if not hasattr(normalizer, attribute):
            raise ValueError(f"H4 normalizer is missing {attribute}")
    for mapping_name in ("mean", "std", "summed_variance"):
        mapping = getattr(normalizer, mapping_name)
        if not isinstance(mapping, Mapping) or set(mapping) != {
            "state",
            "privileged_state",
        }:
            raise ValueError(f"normalizer {mapping_name} key set drifted")
        for key, width in (
            ("state", H4_ACTOR_OBSERVATION_WIDTH),
            ("privileged_state", H4_CRITIC_OBSERVATION_WIDTH),
        ):
            array = _numeric_array(
                mapping[key],
                shape=(width,),
                label=f"normalizer {mapping_name}.{key}",
            )
            if mapping_name == "std" and np.any(array <= 0.0):
                raise ValueError(f"normalizer std.{key} must be strictly positive")
            if mapping_name == "summed_variance" and np.any(array < 0.0):
                raise ValueError(
                    f"normalizer summed_variance.{key} must be non-negative"
                )
    try:
        count_hi = np.asarray(normalizer.count.hi)
        count_lo = np.asarray(normalizer.count.lo)
    except AttributeError as error:
        raise ValueError("normalizer count must expose hi/lo") from error
    for name, value in (("hi", count_hi), ("lo", count_lo)):
        if value.shape != () or value.dtype.kind != "u":
            raise ValueError(f"normalizer count.{name} must be an unsigned scalar")

    def validate_network(
        network: Any, *, name: str, shapes: Sequence[tuple[int, int]]
    ) -> None:
        if not isinstance(network, Mapping) or set(network) != {"params"}:
            raise ValueError(f"{name} parameter group drifted")
        layers = network["params"]
        expected = {f"hidden_{index}" for index in range(len(shapes))}
        if not isinstance(layers, Mapping) or set(layers) != expected:
            raise ValueError(f"{name} layer set drifted")
        for index, (input_width, output_width) in enumerate(shapes):
            layer_name = f"hidden_{index}"
            layer = layers[layer_name]
            if not isinstance(layer, Mapping) or set(layer) != {"kernel", "bias"}:
                raise ValueError(f"{name}.{layer_name} parameter set drifted")
            _numeric_array(
                layer["kernel"],
                shape=(input_width, output_width),
                label=f"{name}.{layer_name}.kernel",
            )
            _numeric_array(
                layer["bias"],
                shape=(output_width,),
                label=f"{name}.{layer_name}.bias",
            )

    actor_shapes = (
        (H4_ACTOR_OBSERVATION_WIDTH, 512),
        (512, 256),
        (256, 128),
        (128, 28),
    )
    critic_shapes = (
        (H4_CRITIC_OBSERVATION_WIDTH, 512),
        (512, 256),
        (256, 128),
        (128, 1),
    )
    validate_network(actor, name="actor", shapes=actor_shapes)
    validate_network(critic, name="critic", shapes=critic_shapes)
    return {
        "actor_observation_width": H4_ACTOR_OBSERVATION_WIDTH,
        "critic_observation_width": H4_CRITIC_OBSERVATION_WIDTH,
        "action_width": H4_ACTION_WIDTH,
        "normalizer_float32_and_finite": True,
        "actor_float32_and_finite": True,
        "critic_float32_and_finite": True,
        "structure_validated": True,
        "passed": True,
    }


def h4_params_numeric_sha256(params: Any) -> str:
    """Hash the exact validated numeric PPO tree without pickling it."""

    validate_h4_params(params)
    normalizer, actor, critic = params
    digest = hashlib.sha256()

    def update(label: str, value: Any) -> None:
        array = np.asarray(value)
        encoded = label.encode("utf-8")
        dtype = array.dtype.str.encode("ascii")
        shape = canonical_json_bytes(list(array.shape))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(dtype).to_bytes(8, "big"))
        digest.update(dtype)
        digest.update(len(shape).to_bytes(8, "big"))
        digest.update(shape)
        digest.update(np.ascontiguousarray(array).tobytes())

    for mapping_name in ("mean", "std", "summed_variance"):
        mapping = getattr(normalizer, mapping_name)
        for key in ("state", "privileged_state"):
            update(f"normalizer.{mapping_name}.{key}", mapping[key])
    update("normalizer.count.hi", normalizer.count.hi)
    update("normalizer.count.lo", normalizer.count.lo)
    for group_name, group in (("actor", actor), ("critic", critic)):
        for index in range(4):
            for leaf in ("kernel", "bias"):
                update(
                    f"{group_name}.hidden_{index}.{leaf}",
                    group["params"][f"hidden_{index}"][leaf],
                )
    return digest.hexdigest()


def audit_v22_to_h4_transplant(
    source_params: Any, transplanted_params: Any
) -> dict[str, Any]:
    """Independently prove the official 101/212 -> 116/227 zero-row transplant."""

    if (
        not isinstance(source_params, (list, tuple))
        or len(source_params) != 3
        or not isinstance(transplanted_params, (list, tuple))
        or len(transplanted_params) != 3
    ):
        raise ValueError("v22 transplant audit requires two three-group PPO trees")
    old_normalizer, old_actor, old_critic = source_params
    new_normalizer, new_actor, new_critic = transplanted_params
    validate_h4_params(transplanted_params)

    def array(value: Any) -> np.ndarray:
        result = np.asarray(value)
        if not np.all(np.isfinite(result)):
            raise ValueError("v22 transplant audit found non-finite values")
        return result

    old_actor_kernel = array(old_actor["params"]["hidden_0"]["kernel"])
    new_actor_kernel = array(new_actor["params"]["hidden_0"]["kernel"])
    old_critic_kernel = array(old_critic["params"]["hidden_0"]["kernel"])
    new_critic_kernel = array(new_critic["params"]["hidden_0"]["kernel"])
    if old_actor_kernel.shape[0] != 101 or old_critic_kernel.shape[0] != 212:
        raise ValueError("source v22 actor/critic widths are not exact 101/212")
    try:
        count_hi = int(np.asarray(old_normalizer.count.hi))
        count_lo = int(np.asarray(old_normalizer.count.lo))
        new_count_hi = int(np.asarray(new_normalizer.count.hi))
        new_count_lo = int(np.asarray(new_normalizer.count.lo))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("v22 normalizer count structure drifted") from error
    count_float = float(count_hi) * (2.0**32) + float(count_lo)
    old_state_variance = np.maximum(
        array(old_normalizer.summed_variance["state"]), 0.0
    )
    old_privileged_variance = np.maximum(
        array(old_normalizer.summed_variance["privileged_state"]), 0.0
    )

    checks = {
        "source_actor_width_101": old_actor_kernel.shape == (101, 512),
        "source_critic_width_212": old_critic_kernel.shape == (212, 512),
        "actor_old_101_rows_bitwise_equal": np.array_equal(
            new_actor_kernel[:101], old_actor_kernel
        ),
        "actor_new_15_rows_exact_zero": np.array_equal(
            new_actor_kernel[101:116], np.zeros_like(new_actor_kernel[101:116])
        ),
        "critic_actor_prefix_bitwise_equal": np.array_equal(
            new_critic_kernel[:101], old_critic_kernel[:101]
        ),
        "critic_new_15_rows_exact_zero": np.array_equal(
            new_critic_kernel[101:116], np.zeros_like(new_critic_kernel[101:116])
        ),
        "critic_privileged_tail_bitwise_equal": np.array_equal(
            new_critic_kernel[116:], old_critic_kernel[101:]
        ),
        "normalizer_state_mean_old_rows_bitwise_equal": np.array_equal(
            array(new_normalizer.mean["state"][:101]),
            array(old_normalizer.mean["state"]),
        ),
        "normalizer_state_mean_new_rows_exact_zero": np.array_equal(
            array(new_normalizer.mean["state"][101:116]),
            np.zeros(15, dtype=np.float32),
        ),
        "normalizer_privileged_mean_exact_insert": np.array_equal(
            array(new_normalizer.mean["privileged_state"]),
            np.concatenate(
                (
                    array(old_normalizer.mean["privileged_state"][:101]),
                    np.zeros(15, dtype=np.float32),
                    array(old_normalizer.mean["privileged_state"][101:]),
                )
            ),
        ),
        "normalizer_state_std_exact_append": np.array_equal(
            array(new_normalizer.std["state"]),
            np.concatenate(
                (
                    array(old_normalizer.std["state"]),
                    np.ones(15, dtype=np.float32),
                )
            ),
        ),
        "normalizer_privileged_std_exact_insert": np.array_equal(
            array(new_normalizer.std["privileged_state"]),
            np.concatenate(
                (
                    array(old_normalizer.std["privileged_state"][:101]),
                    np.ones(15, dtype=np.float32),
                    array(old_normalizer.std["privileged_state"][101:]),
                )
            ),
        ),
        "normalizer_state_variance_exact_append": np.array_equal(
            array(new_normalizer.summed_variance["state"]),
            np.concatenate(
                (
                    old_state_variance,
                    np.full(15, count_float, dtype=np.float32),
                )
            ),
        ),
        "normalizer_privileged_variance_exact_insert": np.array_equal(
            array(new_normalizer.summed_variance["privileged_state"]),
            np.concatenate(
                (
                    old_privileged_variance[:101],
                    np.full(15, count_float, dtype=np.float32),
                    old_privileged_variance[101:],
                )
            ),
        ),
        "normalizer_count_bitwise_equal": (
            new_count_hi == count_hi and new_count_lo == count_lo
        ),
        "actor_non_input_parameters_bitwise_equal": all(
            np.array_equal(
                array(new_actor["params"][f"hidden_{index}"][leaf]),
                array(old_actor["params"][f"hidden_{index}"][leaf]),
            )
            for index in range(4)
            for leaf in ("kernel", "bias")
            if not (index == 0 and leaf == "kernel")
        ),
        "critic_non_input_parameters_bitwise_equal": all(
            np.array_equal(
                array(new_critic["params"][f"hidden_{index}"][leaf]),
                array(old_critic["params"][f"hidden_{index}"][leaf]),
            )
            for index in range(4)
            for leaf in ("kernel", "bias")
            if not (index == 0 and leaf == "kernel")
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"official v22 zero-row transplant audit failed: {checks}")
    return {
        "method": "OFFICIAL_V22_101_TO_116_ZERO_ROW_TRANSPLANT",
        "source_actor_width": 101,
        "target_actor_width": 116,
        "source_critic_width": 212,
        "target_critic_width": 227,
        "insert_offset": 101,
        "inserted_feature_count": 15,
        "optimizer_updates": 0,
        "checks": checks,
        "passed": True,
    }


def load_trusted_h4_params(
    bundle: TrustedH4Bundle,
    *,
    pickle_loader: Callable[[Any], Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Restore a manifest-bound pickle, then re-check its hash and full tree."""

    if not isinstance(bundle, TrustedH4Bundle):
        raise TypeError("load_trusted_h4_params requires a validated bundle")
    if not bundle.source_closure_audit or bundle.source_closure_audit.get(
        "passed"
    ) is not True:
        raise ValueError(
            "H4 params restore requires reconstructed training source closure"
        )
    before = sha256_file(bundle.params_path)
    if before != bundle.params_sha256:
        raise ValueError("params changed after bundle validation")
    loader = pickle.load if pickle_loader is None else pickle_loader
    with bundle.params_path.open("rb") as stream:
        params = loader(stream)
    after = sha256_file(bundle.params_path)
    if after != before:
        raise RuntimeError("params file mutated while it was being restored")
    audit = validate_h4_params(params)
    return params, {
        **audit,
        "params_sha256_before_restore": before,
        "params_sha256_after_restore": after,
        "pickle_opened_only_after_manifest_validation": True,
        "pickle_opened_only_after_source_closure": True,
        "source_closure_audit": dict(bundle.source_closure_audit),
    }


def _silu_float32(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    positive = x >= 0.0
    result = np.empty_like(x)
    result[positive] = x[positive] / (
        np.float32(1.0) + np.exp(-x[positive]).astype(np.float32)
    )
    exp_x = np.exp(x[~positive]).astype(np.float32)
    result[~positive] = x[~positive] * exp_x / (np.float32(1.0) + exp_x)
    return result


def infer_h4_action_numpy(params: Any, observation: Any) -> np.ndarray:
    """Deterministic Brax NormalTanh mode for the frozen actor116 topology.

    This NumPy implementation is used as an independent parity oracle.  Formal
    simulation calls the Brax inference function; ONNX export must agree with
    both within its declared numerical tolerance.
    """

    validate_h4_params(params)
    obs = np.asarray(observation)
    single = obs.ndim == 1
    if single:
        obs = obs[None, :]
    if obs.ndim != 2 or obs.shape[1] != H4_ACTOR_OBSERVATION_WIDTH:
        raise ValueError("H4 policy observation must have trailing width 116")
    if not np.issubdtype(obs.dtype, np.floating) or not np.all(np.isfinite(obs)):
        raise ValueError("H4 policy observation must be finite floating point")
    normalizer, actor, _critic = params
    mean = np.asarray(normalizer.mean["state"], dtype=np.float32)
    std = np.asarray(normalizer.std["state"], dtype=np.float32)
    hidden = (obs.astype(np.float32) - mean) / std
    layers = actor["params"]
    for index in range(4):
        layer = layers[f"hidden_{index}"]
        hidden = (
            hidden @ np.asarray(layer["kernel"], dtype=np.float32)
            + np.asarray(layer["bias"], dtype=np.float32)
        ).astype(np.float32)
        if index != 3:
            hidden = _silu_float32(hidden)
    action = np.tanh(hidden[:, :H4_ACTION_WIDTH]).astype(np.float32)
    if action.shape != (obs.shape[0], H4_ACTION_WIDTH):
        raise RuntimeError("H4 policy output shape drifted")
    if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0 + 1.0e-6):
        raise ValueError("H4 deterministic action is non-finite or outside tanh range")
    return action[0] if single else action


def mask_h4_head_action(action: Any) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape[-1:] != (H4_ACTION_WIDTH,) or not np.all(np.isfinite(values)):
        raise ValueError("H4 action must be finite with trailing width 14")
    masked = values.copy()
    masked[..., H4_HEAD_ACTION_SLICE] = np.float32(0.0)
    if not np.array_equal(
        masked[..., H4_HEAD_ACTION_SLICE],
        np.zeros_like(masked[..., H4_HEAD_ACTION_SLICE]),
    ):
        raise RuntimeError("H4 head mask failed")
    return masked


def compare_policy_outputs(
    reference: Any,
    candidate: Any,
    *,
    absolute_tolerance: float = 2.0e-5,
    relative_tolerance: float = 2.0e-5,
) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    if expected.shape != actual.shape or expected.shape[-1:] != (H4_ACTION_WIDTH,):
        raise ValueError("policy parity arrays must share trailing width 14")
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        raise ValueError("policy parity arrays must be finite")
    error = np.abs(expected - actual)
    allowed = absolute_tolerance + relative_tolerance * np.abs(expected)
    passed = bool(np.all(error <= allowed))
    return {
        "shape": list(expected.shape),
        "maximum_absolute_error": float(np.max(error, initial=0.0)),
        "mean_absolute_error": float(np.mean(error)) if error.size else 0.0,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "passed": passed,
    }


_PINNED_REVERSE_TEACHER_TABLE_CACHE: np.ndarray | None = None


def _pinned_reverse_teacher_table_float32() -> np.ndarray:
    """Load the one trusted reverse table and bind its bytes before use."""

    global _PINNED_REVERSE_TEACHER_TABLE_CACHE
    if _PINNED_REVERSE_TEACHER_TABLE_CACHE is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "h4_reverse_slew_feasible_teacher_selected_v1.json"
        )
        if sha256_file(path) != PINNED_SELECTED_REVERSE_TEACHER_SHA256:
            raise ValueError("pinned reverse teacher SHA256 drifted")
        payload = _require_mapping(
            load_json_strict(path), "pinned reverse teacher"
        )
        teacher = _require_mapping(
            payload.get("teacher"), "pinned reverse teacher payload"
        )
        table = np.asarray(teacher.get("target_table_rad"), dtype=np.float32)
        if (
            table.shape != (H4_REVERSE_TEACHER_TABLE_ROWS, H4_ACTION_WIDTH)
            or not np.all(np.isfinite(table))
            or not np.array_equal(
                table[:, H4_HEAD_ACTION_SLICE],
                np.zeros(
                    (
                        H4_REVERSE_TEACHER_TABLE_ROWS,
                        H4_HEAD_ACTION_SLICE.stop - H4_HEAD_ACTION_SLICE.start,
                    ),
                    dtype=np.float32,
                ),
            )
        ):
            raise ValueError("pinned reverse teacher table contract drifted")
        table.setflags(write=False)
        _PINNED_REVERSE_TEACHER_TABLE_CACHE = table
    return _PINNED_REVERSE_TEACHER_TABLE_CACHE


def _interpolate_reverse_teacher_float32(
    table: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    wrapped = np.remainder(
        phase, np.float32(H4_REVERSE_TEACHER_TABLE_ROWS), dtype=np.float32
    )
    phase_floor = np.floor(wrapped).astype(np.int32)
    phase_next = (phase_floor + 1) % H4_REVERSE_TEACHER_TABLE_ROWS
    fraction = np.subtract(
        wrapped, np.floor(wrapped), dtype=np.float32
    )[:, None]
    rounded_left_product = np.multiply(
        np.subtract(np.float32(1.0), fraction, dtype=np.float32),
        table[phase_floor],
        dtype=np.float32,
    )
    # CPU XLA contracts the right multiply plus add into one float32 FMA,
    # while the left product is rounded first.  Float64 is exact for the
    # product and sum of float32 operands, so this reproduces that one final
    # rounding without depending on JAX in the evidence validator.
    return np.asarray(
        fraction.astype(np.float64)
        * table[phase_next].astype(np.float64)
        + rounded_left_product.astype(np.float64),
        dtype=np.float32,
    )


def _h4_control_trace_arrays(episode: Mapping[str, Any]) -> dict[str, np.ndarray]:
    trace = _require_mapping(episode.get("control_trace"), "episode control trace")
    if trace.get("source_dtype") != "float32":
        raise ValueError("H4 control trace must preserve source dtype float32")
    matrix_keys = {
        "raw_action",
        "applied_action",
        "preclip_targets",
        "margin_clipped_targets",
        "applied_targets",
        "previous_targets",
        "joint_qpos",
    }
    expected_keys = {
        "source_dtype",
        "initial_applied_targets",
        *matrix_keys,
    }
    reverse = episode.get("expert") == "reverse"
    reverse_matrix_keys = {
        "reverse_teacher_table_targets",
        "reverse_delayed_applied_action",
        "reverse_upstream_margin_targets",
    }
    reverse_vector_keys = {
        "reverse_teacher_source_phase_before",
        "reverse_teacher_table_phase",
    }
    if reverse:
        expected_keys.update(reverse_matrix_keys)
        expected_keys.update(reverse_vector_keys)
        expected_keys.update(
            {"reverse_action_delay_index", "reverse_precomposer_active"}
        )
    if set(trace) != expected_keys:
        if reverse:
            raise ValueError(
                "H4 reverse composition trace incomplete or field set drifted"
            )
        raise ValueError("H4 control trace field set drifted")
    result = {
        "initial_applied_targets": np.asarray(
            trace["initial_applied_targets"], dtype=np.float32
        )
    }
    for name in matrix_keys | (reverse_matrix_keys if reverse else set()):
        result[name] = np.asarray(trace[name], dtype=np.float32)
    for name in reverse_vector_keys if reverse else set():
        result[name] = np.asarray(trace[name], dtype=np.float32)
    if reverse:
        raw_delay = np.asarray(trace["reverse_action_delay_index"])
        if (
            raw_delay.shape != (H4_STRICT_CONTROL_TICKS,)
            or not np.issubdtype(raw_delay.dtype, np.integer)
        ):
            raise ValueError("H4 reverse action delay indices must be 300 integers")
        result["reverse_action_delay_index"] = raw_delay.astype(np.int32)
        raw_active = np.asarray(trace["reverse_precomposer_active"])
        if (
            raw_active.shape != (H4_STRICT_CONTROL_TICKS,)
            or raw_active.dtype != np.bool_
        ):
            raise ValueError("H4 reverse precomposer activity must be 300 booleans")
        result["reverse_precomposer_active"] = raw_active
    if result["initial_applied_targets"].shape != (H4_ACTION_WIDTH,):
        raise ValueError("H4 initial applied targets must have width 14")
    if any(
        result[name].shape != (H4_STRICT_CONTROL_TICKS, H4_ACTION_WIDTH)
        for name in matrix_keys | (reverse_matrix_keys if reverse else set())
    ):
        raise ValueError("H4 control trace rows must have exact shape 300x14")
    if reverse and any(
        result[name].shape != (H4_STRICT_CONTROL_TICKS,)
        for name in reverse_vector_keys
    ):
        raise ValueError("H4 reverse phase traces must have exact shape 300")
    if not all(
        np.all(np.isfinite(value))
        for value in result.values()
        if value.dtype != np.bool_
    ):
        raise ValueError("H4 control trace contains non-finite values")
    return result


def rederive_h4_control_contract(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute float32-exact H4 reset/margin/slew semantics from all rows."""

    from safe_gait_experts.contract import (
        ACTUATOR_JOINT_ORDER,
        HEAD_JOINTS,
        SAFE_INIT_POS,
        SAFE_JOINT_LIMITS,
    )

    trace = _h4_control_trace_arrays(episode)
    initial = trace["initial_applied_targets"]
    raw_action = trace["raw_action"]
    applied_action = trace["applied_action"]
    preclip = trace["preclip_targets"]
    desired = trace["margin_clipped_targets"]
    applied = trace["applied_targets"]
    previous = trace["previous_targets"]
    qpos = trace["joint_qpos"]
    is_reverse = episode.get("expert") == "reverse"
    joint_names = tuple(ACTUATOR_JOINT_ORDER)
    leg_indices = np.asarray(
        [index for index, name in enumerate(joint_names) if name not in HEAD_JOINTS]
    )
    head_indices = np.asarray(
        [index for index, name in enumerate(joint_names) if name in HEAD_JOINTS]
    )
    left_knee_index = joint_names.index("left_knee")
    margin = np.float32(0.050)
    expected_initial = np.asarray(
        [SAFE_INIT_POS[name] for name in joint_names], dtype=np.float32
    )
    cap = np.float32(0.04)
    # Subtracting two independently rounded float32 endpoints is not the
    # operation that the runtime guard constrains.  The guard clips its
    # float32 delta and then performs a float32 add.  Consequently, the
    # float64 difference of the stored endpoints may exceed 0.04 by an ULP
    # even when the applied endpoint is the bit-exact guard output.  Keep the
    # endpoint subtraction as a bounded diagnostic and gate on the exact
    # guard computation below.
    dtype_endpoint_delta = np.subtract(applied, previous, dtype=np.float32)
    endpoint_delta64 = applied.astype(np.float64) - previous.astype(np.float64)
    applied_leg = applied[:, leg_indices]
    previous_leg = previous[:, leg_indices]
    preclip_leg = preclip[:, leg_indices]
    qpos_leg = qpos[:, leg_indices]
    lower_all = np.asarray(
        [
            SAFE_JOINT_LIMITS[name][0]
            if name in SAFE_JOINT_LIMITS
            else SAFE_INIT_POS[name]
            for name in joint_names
        ],
        dtype=np.float32,
    )
    upper_all = np.asarray(
        [
            SAFE_JOINT_LIMITS[name][1]
            if name in SAFE_JOINT_LIMITS
            else SAFE_INIT_POS[name]
            for name in joint_names
        ],
        dtype=np.float32,
    )
    lower_leg = np.asarray(
        [SAFE_JOINT_LIMITS[joint_names[index]][0] for index in leg_indices],
        dtype=np.float32,
    )
    upper_leg = np.asarray(
        [SAFE_JOINT_LIMITS[joint_names[index]][1] for index in leg_indices],
        dtype=np.float32,
    )
    margin_lower_leg = np.add(lower_leg, margin, dtype=np.float32)
    margin_upper_leg = np.subtract(upper_leg, margin, dtype=np.float32)
    expected_upstream_margin = np.zeros_like(desired)
    expected_upstream_margin[:, leg_indices] = np.clip(
        preclip_leg, margin_lower_leg, margin_upper_leg
    )
    reverse_checks: dict[str, bool] = {}
    reverse_diagnostics: dict[str, Any] | None = None
    if is_reverse:
        composition = _require_mapping(
            episode.get("reverse_composition_contract"),
            "episode reverse composition contract",
        )
        expected_composition = {
            "schema_version": 1,
            "semantics": H4_REVERSE_COMPOSITION_TRACE_SEMANTICS,
            "selected_reverse_teacher_sha256": (
                PINNED_SELECTED_REVERSE_TEACHER_SHA256
            ),
            "reverse_composition_authorization_sha256": (
                PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
            ),
            "teacher_table_rows": H4_REVERSE_TEACHER_TABLE_ROWS,
            "teacher_entry_phase_preincrement_bins": (
                H4_REVERSE_TEACHER_ENTRY_PHASE_BINS
            ),
            "teacher_phase_advance_bins_per_control": (
                H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS
            ),
            "source_period_bins": H4_REVERSE_SOURCE_PERIOD_BINS,
            "residual_scale": H4_REVERSE_RESIDUAL_SCALE,
            "action_delay_min": H4_REVERSE_ACTION_DELAY_MIN,
            "action_delay_max_exclusive": (
                H4_REVERSE_ACTION_DELAY_MAX_EXCLUSIVE
            ),
            "step_entry_physical_command_x_mps": -0.05,
        }
        reverse_checks["reverse_composition_contract_exact"] = (
            dict(composition) == expected_composition
        )
        phase_scale = np.float32(
            H4_REVERSE_TEACHER_TABLE_ROWS / H4_REVERSE_SOURCE_PERIOD_BINS
        )
        entry_source_phase = np.float32(
            H4_REVERSE_TEACHER_ENTRY_PHASE_BINS / float(phase_scale)
        )
        source_phase_rate = np.float32(
            H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS / float(phase_scale)
        )
        expected_source_phase_before = np.empty(
            H4_STRICT_CONTROL_TICKS, dtype=np.float32
        )
        expected_source_phase_after = np.empty_like(
            expected_source_phase_before
        )
        expected_source_phase_before[0] = entry_source_phase
        for index in range(H4_STRICT_CONTROL_TICKS):
            expected_source_phase_after[index] = np.remainder(
                np.add(
                    expected_source_phase_before[index],
                    source_phase_rate,
                    dtype=np.float32,
                ),
                np.float32(H4_REVERSE_SOURCE_PERIOD_BINS),
                dtype=np.float32,
            )
            if index + 1 < H4_STRICT_CONTROL_TICKS:
                expected_source_phase_before[index + 1] = (
                    expected_source_phase_after[index]
                )
        expected_table_phase = np.remainder(
            np.multiply(
                expected_source_phase_after, phase_scale, dtype=np.float32
            ),
            np.float32(H4_REVERSE_TEACHER_TABLE_ROWS),
            dtype=np.float32,
        )
        expected_teacher_targets = _interpolate_reverse_teacher_float32(
            _pinned_reverse_teacher_table_float32(), expected_table_phase
        )
        delay_indices = trace["reverse_action_delay_index"]
        expected_delayed_action = applied_action
        bounded_delayed_action = np.clip(
            trace["reverse_delayed_applied_action"],
            np.float32(-1.0),
            np.float32(1.0),
        )
        # XLA contracts teacher + scale * residual into a float32 FMA.
        teacher_plus_residual = np.asarray(
            expected_teacher_targets.astype(np.float64)
            + float(np.float32(H4_REVERSE_RESIDUAL_SCALE))
            * bounded_delayed_action.astype(np.float64),
            dtype=np.float32,
        )
        teacher_plus_residual[:, head_indices] = np.float32(0.0)
        safe_lower = np.add(
            expected_initial,
            np.multiply(
                np.float32(0.9),
                np.subtract(lower_all, expected_initial, dtype=np.float32),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        safe_upper = np.add(
            expected_initial,
            np.multiply(
                np.float32(0.9),
                np.subtract(upper_all, expected_initial, dtype=np.float32),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        expected_reverse_preclip = np.clip(
            teacher_plus_residual, safe_lower, safe_upper
        )
        expected_reverse_preclip = np.clip(
            expected_reverse_preclip, lower_all, upper_all
        )
        expected_reverse_upstream = np.zeros_like(desired)
        expected_reverse_upstream[:, leg_indices] = np.clip(
            expected_reverse_preclip[:, leg_indices],
            margin_lower_leg,
            margin_upper_leg,
        )
        expected_precomposer_delta = np.clip(
            np.subtract(
                expected_reverse_upstream, previous, dtype=np.float32
            ),
            -cap,
            cap,
        )
        expected_precomposer = np.add(
            previous, expected_precomposer_delta, dtype=np.float32
        )
        expected_precomposer[:, head_indices] = np.float32(0.0)
        expected_reverse_active = np.ones(
            H4_STRICT_CONTROL_TICKS, dtype=bool
        )
        expected_recorded_desired = expected_precomposer
        reverse_checks.update(
            {
                "reverse_step_entry_command_activates_precomposer": (
                    episode.get("physical_command_mps_radps")
                    == list(H4_STRICT_COMMANDS["reverse"])
                    and np.array_equal(
                        trace["reverse_precomposer_active"],
                        expected_reverse_active,
                    )
                ),
                "reverse_source_phase_preincrement_timeline_exact": (
                    np.array_equal(
                        trace["reverse_teacher_source_phase_before"],
                        expected_source_phase_before,
                    )
                ),
                "reverse_teacher_table_phase_exact_after_preincrement": (
                    np.array_equal(
                        trace["reverse_teacher_table_phase"],
                        expected_table_phase,
                    )
                ),
                "reverse_teacher_table_targets_exact_pinned_interpolation": (
                    np.array_equal(
                        trace["reverse_teacher_table_targets"],
                        expected_teacher_targets,
                    )
                ),
                "reverse_action_delay_index_exact_zero": bool(
                    np.all(delay_indices == H4_REVERSE_ACTION_DELAY_MIN)
                ),
                "reverse_delayed_action_exact_current_post_head_mask": (
                    np.array_equal(
                        trace["reverse_delayed_applied_action"],
                        expected_delayed_action,
                    )
                ),
                "reverse_teacher_plus_residual_safe_clip_exact": (
                    np.array_equal(preclip, expected_reverse_preclip)
                ),
                "reverse_upstream_margin_clip_exact": np.array_equal(
                    trace["reverse_upstream_margin_targets"],
                    expected_reverse_upstream,
                ),
                "reverse_slew_precomposer_output_exact": np.array_equal(
                    desired, expected_precomposer
                ),
            }
        )
        expected_upstream_margin = expected_reverse_upstream
        reverse_diagnostics = {
            "source_phase_scale": float(phase_scale),
            "entry_source_phase": float(entry_source_phase),
            "source_phase_rate": float(source_phase_rate),
            "precomposer_outside_margin_joint_sample_count": int(
                np.count_nonzero(
                    (
                        expected_precomposer[:, leg_indices]
                        < margin_lower_leg
                    )
                    | (
                        expected_precomposer[:, leg_indices]
                        > margin_upper_leg
                    )
                )
            ),
        }
    else:
        expected_recorded_desired = expected_upstream_margin

    final_guard_margin_targets = np.zeros_like(desired)
    final_guard_margin_targets[:, leg_indices] = np.clip(
        expected_recorded_desired[:, leg_indices],
        margin_lower_leg,
        margin_upper_leg,
    )
    expected_guard_delta = np.clip(
        np.subtract(
            final_guard_margin_targets, previous, dtype=np.float32
        ),
        -cap,
        cap,
    )
    expected_applied = np.zeros_like(applied)
    expected_applied[:, leg_indices] = np.clip(
        np.add(
            previous_leg,
            expected_guard_delta[:, leg_indices],
            dtype=np.float32,
        ),
        lower_leg,
        upper_leg,
    )
    expected_applied_leg = expected_applied[:, leg_indices]
    guard_internal_delta_ok = (
        np.abs(expected_guard_delta[:, leg_indices]) <= cap
    )
    guard_output_matches = applied == expected_applied
    guard_output_mismatch_count = int(np.count_nonzero(~guard_output_matches))
    # A full ULP at each stored float32 endpoint is a deliberately explicit,
    # conservative bound on the rounding visible after converting both
    # endpoints to float64 and subtracting them.  This bound is diagnostic;
    # bit-exact equality with ``expected_applied`` above is authoritative.
    endpoint_rounding_allowance64 = (
        np.abs(np.spacing(expected_applied_leg)).astype(np.float64)
        + np.abs(np.spacing(previous_leg)).astype(np.float64)
    )
    endpoint_abs_delta64 = np.abs(endpoint_delta64[:, leg_indices])
    endpoint_rounding_bound64 = float(cap) + endpoint_rounding_allowance64
    endpoint_within_rounding_bound = (
        endpoint_abs_delta64 <= endpoint_rounding_bound64
    )
    endpoint_over_nominal_cap = endpoint_abs_delta64 > float(cap)
    expected_applied_action = raw_action.copy()
    expected_applied_action[:, head_indices] = np.float32(0.0)
    final_margin_leg = final_guard_margin_targets[:, leg_indices]
    desired_margin_ok = (final_margin_leg >= margin_lower_leg) & (
        final_margin_leg <= margin_upper_leg
    )
    applied_physical_ok = (applied_leg >= lower_leg) & (applied_leg <= upper_leg)
    preclip_physical_ok = (preclip_leg >= lower_leg) & (preclip_leg <= upper_leg)
    qpos_physical_ok = (qpos_leg >= lower_leg) & (qpos_leg <= upper_leg)
    applied_outside_margin = (applied_leg < margin_lower_leg) | (
        applied_leg > margin_upper_leg
    )
    previous_margin_excess = np.maximum(
        np.maximum(margin_lower_leg - previous_leg, previous_leg - margin_upper_leg),
        np.float32(0.0),
    )
    applied_margin_excess = np.maximum(
        np.maximum(margin_lower_leg - applied_leg, applied_leg - margin_upper_leg),
        np.float32(0.0),
    )
    authorized_startup = np.zeros_like(applied_outside_margin, dtype=bool)
    left_knee_leg_index = int(np.flatnonzero(leg_indices == left_knee_index)[0])
    authorized_startup[0, left_knee_leg_index] = bool(
        applied_outside_margin[0, left_knee_leg_index]
        and desired_margin_ok[0, left_knee_leg_index]
        and applied_physical_ok[0, left_knee_leg_index]
        and guard_internal_delta_ok[0, left_knee_leg_index]
        and guard_output_matches[0, left_knee_index]
        and applied_margin_excess[0, left_knee_leg_index]
        < previous_margin_excess[0, left_knee_leg_index]
    )
    unauthorized_startup = applied_outside_margin & ~authorized_startup
    left_knee_margin_upper = margin_upper_leg[left_knee_leg_index]
    initial_gap = np.subtract(
        initial[left_knee_index], left_knee_margin_upper, dtype=np.float32
    )
    first_guard_delta = np.abs(expected_guard_delta[0, left_knee_index])
    first_endpoint_delta = np.abs(dtype_endpoint_delta[0, left_knee_index])
    first_residual = np.subtract(
        applied[0, left_knee_index],
        left_knee_margin_upper,
        dtype=np.float32,
    )
    second_endpoint_delta = np.abs(
        dtype_endpoint_delta[1, left_knee_index]
    )
    second_margin_excess = applied_margin_excess[1, left_knee_leg_index]
    float32_atol = float(np.finfo(np.float32).eps * 4.0)
    checks = {
        "source_dtype_float32": True,
        "reset_safe_init_float32_exact": np.array_equal(initial, expected_initial),
        "previous_target_timeline_exact": np.array_equal(previous[0], initial)
        and np.array_equal(previous[1:], applied[:-1]),
        "applied_action_exact_post_inference_head_mask": np.array_equal(
            applied_action, expected_applied_action
        ),
        "applied_head_action_exact_zero": np.array_equal(
            applied_action[:, head_indices],
            np.zeros_like(applied_action[:, head_indices]),
        ),
        "applied_head_targets_exact_zero": np.array_equal(
            applied[:, head_indices], np.zeros_like(applied[:, head_indices])
        ),
        "preclip_leg_targets_physical_safe": bool(np.all(preclip_physical_ok)),
        "desired_leg_targets_inside_float32_margin": bool(
            np.all(desired_margin_ok)
        ),
        "upstream_targets_exact_float32_margin_clip_output": np.array_equal(
            (
                trace["reverse_upstream_margin_targets"]
                if is_reverse
                else desired
            ),
            expected_upstream_margin,
        ),
        "desired_targets_exact_float32_margin_clip_output": bool(
            is_reverse
            or np.array_equal(desired, expected_upstream_margin)
        ),
        "applied_leg_targets_physical_safe": bool(np.all(applied_physical_ok)),
        "joint_qpos_physical_safe": bool(np.all(qpos_physical_ok)),
        "guard_internal_float32_delta_within_exact_0p04": bool(
            np.all(guard_internal_delta_ok)
        ),
        "applied_targets_exact_float32_guard_output": np.array_equal(
            applied, expected_applied
        ),
        "endpoint_delta_within_explicit_float32_rounding_bound": bool(
            np.all(endpoint_within_rounding_bound)
        ),
        "sole_outside_margin_sample_is_authorized_left_knee_startup": bool(
            np.count_nonzero(applied_outside_margin) == 1
            and np.count_nonzero(authorized_startup) == 1
            and np.count_nonzero(unauthorized_startup) == 0
        ),
        "startup_initial_gap_0p045": abs(float(initial_gap) - 0.045)
        <= float32_atol,
        "startup_first_delta_nominal_0p04": abs(
            float(first_guard_delta) - 0.04
        )
        <= float32_atol,
        "startup_first_residual_0p005": abs(float(first_residual) - 0.005)
        <= float32_atol,
        # Tick two may legitimately move farther inside the margin when the
        # policy command changes.  It must not receive another exception.
        "startup_second_sample_inside_float32_margin": bool(
            not applied_outside_margin[1, left_knee_leg_index]
        ),
        "startup_exception_cleared_after_first_tick": bool(
            not np.any(applied_outside_margin[1:])
        ),
    }
    checks.update(reverse_checks)
    checks = {name: bool(value) for name, value in checks.items()}
    maximum_dtype_delta = np.max(
        np.abs(dtype_endpoint_delta[:, leg_indices]), axis=0
    )
    maximum_endpoint_delta64 = np.max(
        np.abs(endpoint_delta64[:, leg_indices]), axis=0
    )
    return {
        "schema_version": 3,
        "semantics": "H4_FLOAT32_EXACT_CONTROL_FIRST_TARGET_GUARD",
        "control_path": (
            H4_REVERSE_COMPOSITION_TRACE_SEMANTICS
            if is_reverse
            else "DIRECT_MARGIN_THEN_FINAL_GUARD"
        ),
        "sample_count": H4_STRICT_CONTROL_TICKS,
        "source_dtype": "float32",
        "exact_slew_cap_rad_per_tick": float(cap),
        "float64_endpoint_difference_is_diagnostic_only": True,
        "desired_margin_violation_count": int(
            np.count_nonzero(~desired_margin_ok)
        ),
        # Backward-compatible acceptance counter.  It now represents actual
        # violations of the guard computation, never ULP noise from endpoint
        # subtraction.
        "float32_slew_violation_count": guard_output_mismatch_count
        + int(np.count_nonzero(~guard_internal_delta_ok)),
        "guard_output_mismatch_joint_sample_count": (
            guard_output_mismatch_count
        ),
        "guard_internal_delta_violation_joint_sample_count": int(
            np.count_nonzero(~guard_internal_delta_ok)
        ),
        "endpoint_delta_over_nominal_cap_joint_sample_count": int(
            np.count_nonzero(endpoint_over_nominal_cap)
        ),
        "endpoint_delta_over_float32_rounding_bound_joint_sample_count": int(
            np.count_nonzero(~endpoint_within_rounding_bound)
        ),
        "endpoint_delta_is_diagnostic_only": True,
        "applied_outside_margin_joint_sample_count": int(
            np.count_nonzero(applied_outside_margin)
        ),
        "authorized_startup_joint_sample_count": int(
            np.count_nonzero(authorized_startup)
        ),
        "unauthorized_startup_joint_sample_count": int(
            np.count_nonzero(unauthorized_startup)
        ),
        "startup_left_knee": {
            "initial_gap_rad": float(initial_gap),
            "first_guard_delta_rad": float(first_guard_delta),
            "first_endpoint_delta_rad": float(first_endpoint_delta),
            "first_residual_rad": float(first_residual),
            "second_endpoint_delta_rad": float(second_endpoint_delta),
            "second_margin_excess_rad": float(second_margin_excess),
        },
        "reverse_composition": reverse_diagnostics,
        "maximum_float32_delta_rad_by_leg_joint": {
            joint_names[index]: float(maximum_dtype_delta[offset])
            for offset, index in enumerate(leg_indices)
        },
        "maximum_float64_endpoint_delta_rad_by_leg_joint": {
            joint_names[index]: float(maximum_endpoint_delta64[offset])
            for offset, index in enumerate(leg_indices)
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def rederive_central_safety_audit_from_control_trace(
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproduce the frozen central SafetyAudit as a diagnostic projection."""

    from safe_gait_experts.contract import ACTUATOR_JOINT_ORDER
    from safe_gait_experts.routed_evaluation import SafetyAudit

    trace = _h4_control_trace_arrays(episode)
    audit = SafetyAudit(ACTUATOR_JOINT_ORDER)
    for index in range(H4_STRICT_CONTROL_TICKS):
        audit.update(
            raw_policy_action=trace["raw_action"][index],
            applied_action=trace["applied_action"][index],
            preclip_targets=trace["preclip_targets"][index],
            margin_clipped_targets=trace["margin_clipped_targets"][index],
            applied_targets=trace["applied_targets"][index],
            previous_applied_targets=trace["previous_targets"][index],
            joint_qpos=trace["joint_qpos"][index],
            control_dt=H4_CONTROL_DT_S,
        )
    return audit.to_dict()


def rederive_h4_safety_acceptance(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute standalone H4 safety from raw control/substep audit payloads."""

    control = episode.get("safety_audit", {})
    physics = episode.get("physics_substep_audit", {})
    guard = episode.get("guard_call_audit", {})
    inference = episode.get("policy_inference_audit", {})
    reset = episode.get("reset_audit", {})
    control_contract = rederive_h4_control_contract(episode)
    try:
        minimum_height = float(physics.get("minimum_height_m", -np.inf))
        minimum_upright = float(physics.get("minimum_upright", -np.inf))
        physics_dt = float(episode.get("physics_timestep_s", np.nan))
        completed_seconds = float(episode.get("duration_s", np.nan))
    except (TypeError, ValueError, OverflowError):
        minimum_height = minimum_upright = -np.inf
        physics_dt = completed_seconds = np.nan
    checks = {
        "completed": episode.get("completed") is True,
        "fell_false": episode.get("fell") is False,
        "exact_safe_init_reset": control_contract["checks"][
            "reset_safe_init_float32_exact"
        ]
        is True,
        "reset_audit_matches_float32_raw_trace": reset.get(
            "comparison_semantics"
        )
        == "SOURCE_DTYPE_FLOAT32_EXACT"
        and reset.get("exact_safe_init")
        is control_contract["checks"]["reset_safe_init_float32_exact"]
        and float(reset.get("maximum_safe_init_error_rad", np.inf)) == 0.0
        and float(reset.get("head_qpos_peak_rad", np.inf)) <= 1.0e-12,
        "exact_control_tick_count": episode.get("completed_control_ticks")
        == H4_STRICT_CONTROL_TICKS,
        "exact_physics_substep_count": episode.get("completed_physics_substeps")
        == H4_STRICT_PHYSICS_SUBSTEPS,
        "exact_duration": bool(
            np.isfinite(physics_dt)
            and abs(physics_dt - H4_PHYSICS_DT_S) <= 1.0e-12
            and np.isfinite(completed_seconds)
            and abs(completed_seconds - H4_STRICT_DURATION_S)
            <= H4_PHYSICS_DT_S / 2.0 + 1.0e-12
        ),
        "all_control_ticks_audited": control.get("sample_count")
        == H4_STRICT_CONTROL_TICKS,
        "control_nonfinite_zero": control.get("nonfinite_sample_count") == 0,
        "applied_head_action_locked": float(
            control.get("applied_head_action_peak", np.inf)
        )
        <= 1.0e-12,
        "head_target_locked": float(control.get("head_target_peak_rad", np.inf))
        <= 1.0e-12,
        "preclip_targets_safe": control.get("preclip_target_limit_violations")
        == 0,
        "applied_targets_safe": control.get("applied_target_limit_violations")
        == 0,
        "desired_targets_inside_margin": control_contract[
            "desired_margin_violation_count"
        ]
        == 0,
        "unauthorized_margin_transition_zero": control_contract[
            "unauthorized_startup_joint_sample_count"
        ]
        == 0,
        "target_slew_safe": control_contract["float32_slew_violation_count"] == 0,
        "h4_float32_control_contract": control_contract["passed"] is True,
        "control_joint_qpos_safe": control.get("qpos_limit_violations") == 0,
        "all_physics_substeps_audited": physics.get("sample_count")
        == H4_STRICT_PHYSICS_SUBSTEPS,
        "all_contact_substeps_audited": physics.get("contact_sample_count")
        == H4_STRICT_PHYSICS_SUBSTEPS,
        "substep_qpos_safe": physics.get("qpos_limit_violations") == 0,
        "substep_finite": physics.get("nonfinite_state_samples") == 0,
        "substep_no_height_fall": physics.get("height_fall_samples") == 0
        and minimum_height >= 0.12,
        "substep_no_upright_fall": physics.get("upright_fall_samples") == 0
        and minimum_upright >= 0.65,
        "guard_called_once_per_tick": guard.get("control_tick_count")
        == H4_STRICT_CONTROL_TICKS
        and guard.get("total_guard_calls") == H4_STRICT_CONTROL_TICKS
        and guard.get("guard_call_violation_count") == 0
        and guard.get("maximum_guard_calls_per_tick") == 1,
        "actor_input_width_exact": inference.get("input_width")
        == H4_ACTOR_OBSERVATION_WIDTH,
        "actor_output_width_exact": inference.get("output_width") == H4_ACTION_WIDTH,
        "actor_inference_count_exact": inference.get("inference_count")
        == H4_STRICT_CONTROL_TICKS,
        "actor_nonfinite_zero": inference.get("nonfinite_observation_count") == 0
        and inference.get("nonfinite_action_count") == 0,
        "post_inference_head_mask_exact": inference.get(
            "post_mask_nonzero_head_count"
        )
        == 0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {"passed": all(checks.values()), "checks": checks}


def legacy_metrics_from_gait_quality(metrics: Mapping[str, Any]) -> dict[str, Any]:
    def number(name: str) -> float | None:
        value = metrics.get(name)
        return float(value) if value is not None else None

    return {
        "speed_ratio": number("steady_linear_tracking_ratio"),
        "absolute_cross_velocity_mps": number("steady_cross_drift_mps"),
        "absolute_uncommanded_yaw_rate_radps": number(
            "uncommanded_yaw_rate_radps"
        ),
        "maximum_heading_change_per_6s_rad": number(
            "uncommanded_heading_drift_rad"
        ),
        "single_support_rate": number("single_support_rate"),
        "flight_rate": number("flight_rate"),
        "stance_slip_rms_mps": number("stance_slip_rms_mps"),
        "stance_slip_p95_mps": number("stance_slip_p95_mps"),
        "maximum_per_stance_cumulative_slip_m": number(
            "maximum_per_stance_cumulative_slip_m"
        ),
        "alternating_touchdown_fraction": number(
            "alternating_touchdown_fraction"
        ),
        "contact_duty_imbalance": number("contact_duty_imbalance"),
        "left_right_step_count_imbalance": number("step_count_imbalance"),
        "duration_s": number("duration_s"),
    }


def validate_h4_strict_episode(
    episode: Mapping[str, Any],
    *,
    expert: str,
    expected_seed: int,
    gait_quality_rederive: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Recompute every safety/GQ decision and reject summary-only evidence."""

    if expert not in H4_STRICT_SEEDS:
        raise ValueError("strict H4 episode expert is invalid")
    if gait_quality_rederive is None:
        from safe_gait_experts.gait_quality import rederive_gait_quality_acceptance

        gait_quality_rederive = rederive_gait_quality_acceptance
    metrics = _require_mapping(
        episode.get("gait_quality_metrics"), "episode gait_quality_metrics"
    )
    recorded_gait = _require_mapping(
        episode.get("gait_quality_acceptance"), "episode gait_quality_acceptance"
    )
    rederived_object = gait_quality_rederive(metrics)
    rederived_gait = (
        rederived_object.as_dict()
        if hasattr(rederived_object, "as_dict")
        else dict(rederived_object)
    )
    rederived_control_contract = rederive_h4_control_contract(episode)
    recorded_control_contract = _require_mapping(
        episode.get("h4_control_contract"), "episode H4 control contract"
    )
    central_safety_rederived = rederive_central_safety_audit_from_control_trace(
        episode
    )
    safety = rederive_h4_safety_acceptance(episode)
    recorded_safety = _require_mapping(
        episode.get("h4_safety_acceptance"), "episode h4_safety_acceptance"
    )
    command = list(H4_STRICT_COMMANDS[expert])
    segment_id = episode.get("segment_id")
    structural_checks = {
        "seed_exact": episode.get("seed") == expected_seed,
        "segment_id_unique_form": isinstance(segment_id, str)
        and segment_id == f"h4_{expert}_seed{expected_seed}_6s",
        "expert_exact": episode.get("expert") == expert,
        "physical_command_exact": episode.get("physical_command_mps_radps")
        == command,
        "strict_source_segment": episode.get("source_segment_kind")
        == "H4_STRICT_6S",
        "gait_metrics_schema_v2": metrics.get("measurement_schema_version") == 2,
        "gait_measurement_complete": metrics.get("measurement_complete") is True,
        "gait_n_plus_one_samples": metrics.get("sample_count")
        == H4_STRICT_GAIT_SAMPLES,
        "gait_exact_duration": abs(
            float(metrics.get("duration_s", np.inf)) - H4_STRICT_DURATION_S
        )
        <= H4_PHYSICS_DT_S / 2.0 + 1.0e-12,
        "gait_exact_timestep": abs(
            float(metrics.get("physics_timestep_s", np.inf)) - H4_PHYSICS_DT_S
        )
        <= 1.0e-12,
        "gait_timestep_error_bounded": float(
            metrics.get("maximum_timestep_error_s", np.inf)
        )
        <= 1.0e-12,
        "trunk_pose_source_complete": metrics.get("trunk_pose_measurement_source")
        not in (None, "world_trunk_position_only"),
        "trunk_yaw_n_plus_one": metrics.get("trunk_yaw_sample_count")
        == H4_STRICT_GAIT_SAMPLES,
        "force_n_plus_one": metrics.get("contact_force_sample_count")
        == H4_STRICT_GAIT_SAMPLES,
        "contact_velocity_payload_n_plus_one": metrics.get(
            "contact_velocity_payload_sample_count"
        )
        == H4_STRICT_GAIT_SAMPLES,
        "force_contact_source": metrics.get("contact_state_source")
        == "normal_force_schmitt",
        "jacobian_slip_source": metrics.get("stance_slip_measurement_source")
        == "force_weighted_contact_point_jacobian",
        "gait_acceptance_exactly_rederived": dict(recorded_gait)
        == rederived_gait,
        "h4_control_contract_exactly_rederived": dict(recorded_control_contract)
        == rederived_control_contract,
        "central_safety_audit_exactly_rederived": episode.get("safety_audit")
        == central_safety_rederived,
        "safety_acceptance_exactly_rederived": dict(recorded_safety) == safety,
    }
    legacy = legacy_metrics_from_gait_quality(metrics)
    legacy_exact = episode.get("metrics") == legacy
    structural_checks["legacy_projection_exact"] = legacy_exact
    all_structure = all(bool(value) for value in structural_checks.values())
    strict_passed = bool(
        all_structure
        and safety["passed"]
        and rederived_gait.get("passed") is True
    )
    if episode.get("strict_passed") is not strict_passed:
        structural_checks["strict_pass_record_exact"] = False
        strict_passed = False
    else:
        structural_checks["strict_pass_record_exact"] = True
    return {
        "seed": expected_seed,
        "structural_checks": {
            name: bool(value) for name, value in structural_checks.items()
        },
        "safety": safety,
        "gait_quality": rederived_gait,
        "strict_passed": strict_passed,
    }


def current_source_hashes(
    paths: Mapping[str, Path], *, root: Path | None = None
) -> dict[str, str]:
    result: dict[str, str] = {}
    root_resolved = Path(root).resolve() if root is not None else None
    for label, raw_path in paths.items():
        path = Path(raw_path).resolve()
        key = str(path)
        if root_resolved is not None:
            try:
                key = str(path.relative_to(root_resolved)).replace("\\", "/")
            except ValueError:
                key = str(path)
        if label and key in result:
            raise ValueError(f"duplicate source path key: {key}")
        result[key] = sha256_file(path)
    return result


def validate_h4_strict_artifact(
    payload: Mapping[str, Any],
    *,
    bundle: TrustedH4Bundle | None = None,
    current_central_hashes: Mapping[str, str] | None = None,
    current_evaluation_hashes: Mapping[str, str] | None = None,
    require_all_three_pass: bool = False,
    gait_quality_rederive: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Validate a complete fixed-seed strict artifact from first principles."""

    _validate_json_value(dict(payload))
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind") != STRICT_ARTIFACT_KIND
        or payload.get("hardware_deployment") != "PROHIBITED"
        or payload.get("execution_provider") != "CPU"
    ):
        raise ValueError("H4 strict artifact schema/CPU/hardware contract drifted")
    candidate = _require_mapping(payload.get("candidate"), "strict candidate")
    expert = candidate.get("expert")
    candidate_status = candidate.get("status")
    if expert not in H4_STRICT_SEEDS:
        raise ValueError("strict artifact candidate expert is invalid")
    if candidate_status not in {"COMPLETED", "WIRING_PASS"}:
        raise ValueError("strict artifact candidate status is invalid")
    for field_name in (
        "final_params_sha256",
        "manifest_sha256",
        "resolved_config_sha256",
        "source_and_teacher_hashes_sha256",
        "training_provenance_sha256",
    ):
        require_sha256(candidate.get(field_name), f"strict candidate {field_name}")
    if bundle is not None and candidate != bundle.candidate_record():
        raise ValueError("strict artifact candidate does not match trusted bundle")
    central = _require_mapping(payload.get("central_hashes"), "central hashes")
    if current_central_hashes is not None and dict(central) != dict(
        current_central_hashes
    ):
        raise ValueError("strict artifact central semantics hashes are stale")
    for label, digest in central.items():
        require_sha256(digest, f"central hash {label}")
    contract = _require_mapping(
        payload.get("evaluation_contract"), "evaluation contract"
    )
    expected_seeds = H4_STRICT_SEEDS[expert]
    if (
        contract.get("fixed_seeds") != list(expected_seeds)
        or contract.get("physical_command_mps_radps")
        != list(H4_STRICT_COMMANDS[expert])
        or contract.get("duration_s") != H4_STRICT_DURATION_S
        or contract.get("control_timestep_s") != H4_CONTROL_DT_S
        or contract.get("physics_timestep_s") != H4_PHYSICS_DT_S
        or contract.get("control_tick_count") != H4_STRICT_CONTROL_TICKS
        or contract.get("physics_substep_count") != H4_STRICT_PHYSICS_SUBSTEPS
        or contract.get("gait_sample_count") != H4_STRICT_GAIT_SAMPLES
        or contract.get("gait_quality_semantics")
        != "FULL_CURRENT_P0_RECOMPUTED_FROM_N_PLUS_ONE_SUBSTEP_TRACE"
        or contract.get("reverse_composition")
        != (
            "PINNED_PERSISTENT_TEACHER_PLUS_TRAINABLE_RESIDUAL"
            if expert == "reverse"
            else None
        )
    ):
        raise ValueError("strict artifact evaluation contract drifted")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 3:
        raise ValueError("strict artifact requires exactly three episodes")
    episode_audits = []
    seen_ids: set[str] = set()
    for expected_seed, episode in zip(expected_seeds, episodes, strict=True):
        mapping = _require_mapping(episode, "strict episode")
        segment_id = mapping.get("segment_id")
        if not isinstance(segment_id, str) or segment_id in seen_ids:
            raise ValueError("strict artifact segment IDs must be unique")
        seen_ids.add(segment_id)
        episode_audits.append(
            validate_h4_strict_episode(
                mapping,
                expert=expert,
                expected_seed=expected_seed,
                gait_quality_rederive=gait_quality_rederive,
            )
        )
    passing_seeds = [
        audit["seed"] for audit in episode_audits if audit["strict_passed"]
    ]
    summary = _require_mapping(payload.get("summary"), "strict summary")
    if (
        summary.get("passing_seed_count") != len(passing_seeds)
        or summary.get("passing_seeds") != passing_seeds
        or summary.get("all_three_strict_pass") is not (len(passing_seeds) == 3)
    ):
        raise ValueError("strict artifact summary is not exactly recomputed")

    baseline_audits: list[dict[str, Any]] = []
    baseline_passing_seeds: list[int] = []
    baseline_payload = payload.get("official_v22_baseline")
    if candidate_status == "COMPLETED":
        baseline = _require_mapping(
            baseline_payload, "official v22 integrated baseline"
        )
        source = _require_mapping(
            baseline.get("source_checkpoint"), "official v22 source checkpoint"
        )
        source_path = Path(source.get("path", "")).resolve()
        tree_pre = require_sha256(
            source.get("sha256_tree_pre"), "official v22 tree pre"
        )
        tree_post = require_sha256(
            source.get("sha256_tree_post"), "official v22 tree post"
        )
        if (
            source.get("kind") != "OFFICIAL_FROZEN_V22_BRAX_CHECKPOINT"
            or not source_path.is_absolute()
            or tree_pre != PINNED_V22_PARENT_TREE_SHA256
            or tree_post != tree_pre
            or source.get("unchanged") is not True
            or sha256_tree(source_path) != tree_post
        ):
            raise ValueError("official v22 integrated baseline source drifted")
        transplant = _require_mapping(
            baseline.get("transplant_audit"), "official v22 transplant audit"
        )
        transplant_checks = _require_mapping(
            transplant.get("checks"), "official v22 transplant checks"
        )
        if (
            transplant.get("method")
            != "OFFICIAL_V22_101_TO_116_ZERO_ROW_TRANSPLANT"
            or transplant.get("source_actor_width") != 101
            or transplant.get("target_actor_width") != 116
            or transplant.get("source_critic_width") != 212
            or transplant.get("target_critic_width") != 227
            or transplant.get("insert_offset") != 101
            or transplant.get("inserted_feature_count") != 15
            or transplant.get("optimizer_updates") != 0
            or transplant.get("passed") is not True
            or not transplant_checks
            or not all(value is True for value in transplant_checks.values())
        ):
            raise ValueError("official v22 integrated baseline transplant drifted")
        require_sha256(
            baseline.get("transplanted_params_numeric_sha256"),
            "official v22 transplanted params numeric SHA256",
        )
        if (
            baseline.get("evaluation_process")
            != "SAME_PROCESS_ENVIRONMENT_CONTRACT_AND_FIXED_SEEDS_AS_CANDIDATE"
            or baseline.get("optimizer_updates") != 0
            or baseline.get("policy_inference")
            != "BRAX_DETERMINISTIC_NORMAL_TANH_ACTOR116"
        ):
            raise ValueError("official v22 integrated baseline process drifted")
        baseline_episodes = baseline.get("episodes")
        if not isinstance(baseline_episodes, list) or len(baseline_episodes) != 3:
            raise ValueError("official v22 baseline requires exactly three episodes")
        baseline_seen_ids: set[str] = set()
        for expected_seed, episode in zip(
            expected_seeds, baseline_episodes, strict=True
        ):
            mapping = _require_mapping(episode, "official v22 baseline episode")
            segment_id = mapping.get("segment_id")
            if not isinstance(segment_id, str) or segment_id in baseline_seen_ids:
                raise ValueError("official v22 baseline segment IDs must be unique")
            baseline_seen_ids.add(segment_id)
            baseline_audits.append(
                validate_h4_strict_episode(
                    mapping,
                    expert=expert,
                    expected_seed=expected_seed,
                    gait_quality_rederive=gait_quality_rederive,
                )
            )
        baseline_passing_seeds = [
            audit["seed"]
            for audit in baseline_audits
            if audit["strict_passed"]
        ]
        baseline_summary = _require_mapping(
            baseline.get("summary"), "official v22 baseline summary"
        )
        if (
            baseline_summary.get("passing_seed_count")
            != len(baseline_passing_seeds)
            or baseline_summary.get("passing_seeds") != baseline_passing_seeds
            or baseline_summary.get("all_three_strict_pass")
            is not (len(baseline_passing_seeds) == 3)
        ):
            raise ValueError("official v22 baseline summary is not recomputed")
    elif baseline_payload is not None:
        raise ValueError("WIRING_PASS artifact must not claim a promotion baseline")
    provenance = _require_mapping(
        payload.get("runtime_provenance"), "runtime provenance"
    )
    provenance_source_hashes = provenance.get("source_and_teacher_hashes")
    source_hashes_complete = bool(
        isinstance(provenance_source_hashes, Mapping)
        and provenance_source_hashes
        and all(
            isinstance(label, str)
            and bool(label)
            and isinstance(record, Mapping)
            and set(record) == {"path", "sha256"}
            and isinstance(record.get("path"), str)
            and bool(record.get("path"))
            and _is_absolute_path_text(record["path"])
            and isinstance(record.get("sha256"), str)
            and len(record["sha256"]) == 64
            and all(
                character in "0123456789abcdef"
                for character in record["sha256"]
            )
            for label, record in provenance_source_hashes.items()
        )
    )
    source_hashes_candidate_bound = bool(
        source_hashes_complete
        and candidate.get("source_and_teacher_hashes_sha256")
        == canonical_json_sha256(dict(provenance_source_hashes))
    )
    source_hashes_bundle_bound = bool(
        bundle is None
        or (
            source_hashes_complete
            and dict(provenance_source_hashes) == dict(bundle.source_hashes)
        )
    )
    evaluation_pre = provenance.get("evaluation_source_hashes_pre")
    evaluation_post = provenance.get("evaluation_source_hashes_post")
    evaluation_hashes_complete = bool(
        isinstance(evaluation_pre, Mapping)
        and evaluation_pre
        and isinstance(evaluation_post, Mapping)
        and dict(evaluation_pre) == dict(evaluation_post)
        and all(
            isinstance(path, str)
            and bool(path)
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for path, digest in evaluation_pre.items()
        )
    )
    evaluation_hashes_current = bool(
        current_evaluation_hashes is None
        or (
            evaluation_hashes_complete
            and dict(evaluation_pre) == dict(current_evaluation_hashes)
        )
    )
    training_provenance = provenance.get("training_provenance")
    training_provenance_complete = bool(
        isinstance(training_provenance, Mapping)
        and training_provenance.get("schema_version") == 1
        and training_provenance.get("training_execution_provider") == "JAX_GPU"
        and training_provenance.get("platform") == "gpu"
        and training_provenance.get("cross_bound_config_manifest_result") is True
        and training_provenance.get("passed") is True
    )
    training_provenance_candidate_bound = bool(
        training_provenance_complete
        and candidate.get("training_provenance_sha256")
        == canonical_json_sha256(dict(training_provenance))
    )
    training_provenance_bundle_bound = bool(
        bundle is None
        or (
            training_provenance_complete
            and dict(training_provenance) == dict(bundle.training_provenance)
        )
    )
    jax_devices = provenance.get("jax_devices")
    reverse_provenance_checks = provenance.get("reverse_composition_checks")
    if expert == "reverse":
        selected_source = (
            provenance_source_hashes.get("selected_reverse_teacher", {})
            if isinstance(provenance_source_hashes, Mapping)
            else {}
        )
        authorization_source = (
            provenance_source_hashes.get(
                "reverse_composition_authorization", {}
            )
            if isinstance(provenance_source_hashes, Mapping)
            else {}
        )
        reverse_composition_provenance_bound = bool(
            isinstance(reverse_provenance_checks, Mapping)
            and bool(reverse_provenance_checks)
            and all(value is True for value in reverse_provenance_checks.values())
            and selected_source.get("sha256")
            == PINNED_SELECTED_REVERSE_TEACHER_SHA256
            and authorization_source.get("sha256")
            == PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
        )
    else:
        reverse_composition_provenance_bound = (
            reverse_provenance_checks is None
        )
    provenance_checks = {
        "execution_provider_cpu": provenance.get("execution_provider") == "CPU",
        "jax_platform_cpu": provenance.get("jax_default_backend") == "cpu",
        "jax_devices_cpu_only": isinstance(jax_devices, list)
        and bool(jax_devices)
        and all(
            isinstance(device, Mapping) and device.get("platform") == "cpu"
            for device in jax_devices
        ),
        "candidate_manifest_bound": provenance.get("candidate_manifest_sha256")
        == candidate.get("manifest_sha256"),
        "candidate_params_bound": provenance.get("candidate_final_params_sha256")
        == candidate.get("final_params_sha256"),
        "candidate_config_bound": provenance.get("candidate_resolved_config_sha256")
        == candidate.get("resolved_config_sha256"),
        "source_hashes_complete": source_hashes_complete,
        "source_hashes_candidate_bound": source_hashes_candidate_bound,
        "source_hashes_bundle_bound": source_hashes_bundle_bound,
        "central_hashes_bound": provenance.get("central_hashes") == central,
        "evaluation_hashes_complete_and_equal": evaluation_hashes_complete,
        "evaluation_hashes_match_current_sources": evaluation_hashes_current,
        "training_gpu_provenance_complete": training_provenance_complete,
        "training_gpu_provenance_candidate_bound": (
            training_provenance_candidate_bound
        ),
        "training_gpu_provenance_bundle_bound": training_provenance_bundle_bound,
        "reverse_composition_provenance_bound": (
            reverse_composition_provenance_bound
        ),
        "pre_post_sources_unchanged": provenance.get(
            "pre_post_source_hashes_unchanged"
        )
        is True
        and evaluation_hashes_complete,
    }
    if not all(provenance_checks.values()):
        raise ValueError(f"strict artifact runtime provenance failed: {provenance_checks}")
    if require_all_three_pass and len(passing_seeds) != 3:
        raise ValueError("strict artifact does not pass all three fixed seeds")
    return {
        "expert": expert,
        "passing_seeds": passing_seeds,
        "passing_seed_count": len(passing_seeds),
        "episode_audits": episode_audits,
        "official_v22_baseline_episode_audits": baseline_audits,
        "official_v22_baseline_passing_seeds": baseline_passing_seeds,
        "official_v22_baseline_passing_seed_count": len(
            baseline_passing_seeds
        ),
        "provenance_checks": provenance_checks,
        "all_three_strict_pass": len(passing_seeds) == 3,
        "passed": len(passing_seeds) == 3,
    }


def build_promotion_evidence(
    *,
    candidate_artifact_path: Path,
    baseline_artifact_path: Path,
    bundle: TrustedH4Bundle,
    current_central_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Reject legacy/self-reported baseline artifacts unconditionally."""

    del candidate_artifact_path
    del baseline_artifact_path
    del bundle
    del current_central_hashes
    raise ValueError(
        "external/self-reported baseline artifacts are prohibited; use the "
        "same-process integrated official-v22 baseline"
    )


def build_integrated_promotion_evidence(
    *,
    strict_artifact_path: Path,
    bundle: TrustedH4Bundle,
    current_central_hashes: Mapping[str, str],
    current_evaluation_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Promote only from one artifact containing candidate and v22 baseline."""

    if bundle.status != "COMPLETED" or bundle.activity != "PPO_PILOT_TRAINING":
        raise ValueError("only a completed H4 PPO pilot can produce promotion evidence")
    artifact_path = Path(strict_artifact_path).resolve()
    payload = _require_mapping(
        load_json_strict(artifact_path), "integrated H4 strict artifact"
    )
    audit = validate_h4_strict_artifact(
        payload,
        bundle=bundle,
        current_central_hashes=current_central_hashes,
        current_evaluation_hashes=current_evaluation_hashes,
        require_all_three_pass=True,
    )
    checks = {
        "integrated_official_v22_baseline_present": bool(
            payload.get("official_v22_baseline")
        ),
        "candidate_passes_all_three": audit["passing_seed_count"] == 3,
        "baseline_passes_fewer_than_three": audit[
            "official_v22_baseline_passing_seed_count"
        ]
        < 3,
        "candidate_improves_strict_pass_count": audit["passing_seed_count"]
        > audit["official_v22_baseline_passing_seed_count"],
        "same_process_environment_seeds_and_full_p0": True,
        "no_optimizer_update_in_baseline": payload["official_v22_baseline"].get(
            "optimizer_updates"
        )
        == 0,
        "no_external_baseline_artifact": True,
    }
    if not all(checks.values()):
        raise ValueError(f"H4 integrated promotion comparison failed: {checks}")
    return {
        "schema_version": 1,
        "artifact_kind": PROMOTION_EVIDENCE_KIND,
        "hardware_deployment": "PROHIBITED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "promotion_gate": {
            "candidate_run_name": bundle.run_name,
            "candidate_final_params_sha256": bundle.params_sha256,
            "candidate_resolved_config_sha256": bundle.config_sha256,
            "candidate_manifest_sha256": bundle.manifest_sha256,
            "strict_evaluation_artifact_path": str(artifact_path),
            "strict_evaluation_artifact_sha256": sha256_file(artifact_path),
            "baseline_source": "INTEGRATED_OFFICIAL_V22_ZERO_ROW_TRANSPLANT",
            "required_seeds": list(H4_STRICT_SEEDS[bundle.expert]),
            "baseline_strict_pass_count": audit[
                "official_v22_baseline_passing_seed_count"
            ],
            "candidate_strict_pass_count": audit["passing_seed_count"],
            "central_hashes": dict(current_central_hashes),
            "full_p0_gait_quality_recomputed": True,
            "checks": checks,
            "passed": True,
        },
    }


__all__ = [
    "H4_ACTION_WIDTH",
    "H4_ACTOR_OBSERVATION_WIDTH",
    "H4_CONTROL_DT_S",
    "H4_CRITIC_OBSERVATION_WIDTH",
    "H4_GAIT_SAMPLE_SOURCE",
    "H4_HEAD_ACTION_SLICE",
    "H4_PHYSICS_DT_S",
    "H4_STRICT_COMMANDS",
    "H4_STRICT_CONTROL_TICKS",
    "H4_STRICT_DURATION_S",
    "H4_STRICT_GAIT_SAMPLES",
    "H4_STRICT_PHYSICS_SUBSTEPS",
    "H4_STRICT_SEEDS",
    "PROMOTION_EVIDENCE_KIND",
    "H4_FORWARD_ITERATION_V2_CONTRACT_ID",
    "H4_FORWARD_ITERATION_V2_SOURCE_LABELS",
    "H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_CONTRACT_ID",
    "H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_SOURCE_LABELS",
    "H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_WIRING_CONTRACT_ID",
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_CONTRACT_ID",
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_NO_PPO_CONTRACT_ID",
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_WIRING_CONTRACT_ID",
    "H4_FORWARD_ITERATION_V6_RUNTIME_INFO_KEYS",
    "H4_REVERSE_ITERATION_V2_CONTRACT_ID",
    "H4_REVERSE_ITERATION_V2_SOURCE_LABELS",
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN",
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_CONTRACT_ID",
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_SOURCE_LABELS",
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_WIRING_CONTRACT_ID",
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_CONTRACT_ID",
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_NO_PPO_CONTRACT_ID",
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_WIRING_CONTRACT_ID",
    "H4_REVERSE_ITERATION_V6_RUNTIME_INFO_KEYS",
    "H4_REVERSE_ITERATION_V6_RESIDUAL_AUTHORITY_SCALE",
    "PINNED_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_AUTHORIZATION_SHA256",
    "PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256",
    "PINNED_FORWARD_MINIMUM_SPEC_CANONICAL_SHA256",
    "PINNED_FORWARD_MINIMUM_SPEC_SHA256",
    "PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256",
    "PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256",
    "PINNED_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_AUTHORIZATION_SHA256",
    "PINNED_REVERSE_MINIMUM_SPEC_SHA256",
    "PINNED_SELECTED_REVERSE_TEACHER_SHA256",
    "PINNED_V22_PARENT_TREE_SHA256",
    "STRICT_ARTIFACT_KIND",
    "TrustedH4Bundle",
    "audit_v22_to_h4_transplant",
    "build_integrated_promotion_evidence",
    "build_promotion_evidence",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "compare_policy_outputs",
    "current_source_hashes",
    "h4_params_numeric_sha256",
    "infer_h4_action_numpy",
    "json_native",
    "legacy_metrics_from_gait_quality",
    "load_json_strict",
    "load_trusted_h4_params",
    "mask_h4_head_action",
    "reconstruct_h4_training_source_paths",
    "rederive_central_safety_audit_from_control_trace",
    "rederive_h4_control_contract",
    "rederive_h4_safety_acceptance",
    "require_sha256",
    "sha256_file",
    "sha256_tree",
    "validate_h4_params",
    "validate_h4_training_source_closure",
    "validate_h4_strict_artifact",
    "validate_h4_strict_episode",
    "validate_trusted_h4_bundle",
    "write_new_json",
]
