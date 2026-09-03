"""Run the isolated H4-aligned OpenDuckMini PPO wiring or pilot.

This entrypoint never deploys hardware.  Its default and currently authorized
mode is ``--wiring-only``: two CPU environments, one PPO training step, two
minibatch optimizer updates, and exactly forty interactions.  A later simulation pilot requires an explicit
authorization flag; promotion from 250k to 1M additionally requires a hashed
three-seed, six-second strict-improvement artifact.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import pickle
import re
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.contract import ACTUATOR_JOINT_ORDER, SAFE_INIT_POS  # noqa: E402
from safe_gait_experts.h4_post_training import validate_h4_params  # noqa: E402
from safe_gait_experts.h4_training_alignment import (  # noqa: E402
    H4QualityRewardScales,
    H4_ACTOR_OBSERVATION_WIDTH,
    H5_SIGNED_PROGRESS_SCALE,
    H5_TRACKING_SIGMA,
    H4_FORWARD_EXACT_ENDPOINT_PROBABILITY,
    H4_FORWARD_LOCAL_ANCHORS_M_S,
    H4_FORWARD_PRIMARY_ANCHOR_M_S,
    H4_FORWARD_TRANSITION_BAND_M_S,
    H4_FORWARD_V2_EXACT_ENDPOINT_PROBABILITY,
    H4_FORWARD_V2_LOCAL_ANCHOR_PROBABILITY,
    H4_FORWARD_V2_STAND_PROBABILITY,
    H4_FORWARD_V2_TRANSITION_PROBABILITY,
    H4_REVERSE_EXACT_ENDPOINT_PROBABILITY,
    H4_REVERSE_LOCAL_ANCHORS_M_S,
    H4_REVERSE_PRIMARY_ANCHOR_M_S,
    H4_REVERSE_TRANSITION_BAND_M_S,
    H4_REVERSE_V2_EXACT_ENDPOINT_PROBABILITY,
    H4_REVERSE_V2_LOCAL_ANCHOR_PROBABILITY,
    H4_REVERSE_V2_STAND_PROBABILITY,
    H4_REVERSE_V2_TRANSITION_PROBABILITY,
    FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID,
    REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID,
    LEGACY_ACTOR_OBSERVATION_WIDTH,
    MAX_TARGET_DELTA_PER_TICK_RAD,
    V4SourceSemanticPreflight,
    audit_v4_source_semantic_reference,
    make_anchor_command_mapper,
    make_h4_aligned_environment_class,
    make_h4_forward_physical_sampler,
    make_h4_forward_v2_physical_sampler,
    make_h4_reverse_physical_sampler,
    make_h4_reverse_v2_physical_sampler,
    require_checkpoint_observation_compatibility,
    save_v4_dynamic_state,
    v4_authoritative_primitive_step,
)
from safe_gait_experts.h5_training_alignment import (  # noqa: E402
    make_h5_planar_command_mapper,
    make_h5_planar_physical_sampler,
    make_h5_reverse_command_mapper,
    make_h5_reverse_physical_sampler,
    make_h5_unified_command_mapper,
    make_h5_unified_physical_sampler,
)
from safe_gait_experts.h5_command_contract import (  # noqa: E402
    H5_COMMAND_CONTRACT_ID,
    H5_UNIFIED_COMMAND_MAPPER_SUPPORTED_MODES,
    canonical_h5_unified_command_mapper,
    h5_command_contract_manifest,
    h5_unified_command_contract_id,
    h5_unified_command_contract_manifest,
)
from safe_gait_experts.h5_command_conditioned_se2 import (  # noqa: E402
    H5_V3_COMMAND_CONDITIONED_SE2_ALIGNMENT_ID,
)
from safe_gait_experts.h5_substep_contact_alignment import (  # noqa: E402
    H5_V3_SE2_SUBSTEP_CONTACT_ALIGNMENT_ID,
    h5_all_substep_quality_losses,
    h5_all_substep_quality_update,
    h5_reverse_return_order_proof,
    h5_v3_t1_fixed_quality_replay_manifest,
)
from safe_gait_experts.h5_sidecar_quality import (  # noqa: E402
    H5_V3_SIDECAR_QUALITY_CONTRACT_ID,
    h5_sidecar_score_control_tick,
    h5_sidecar_weighted_reward_delta,
    initialize_h5_sidecar_debounce_carry,
)


LEGACY_TRAINER_PATH = EXP_ROOT / "scripts" / "train_expert.py"
ALIGNMENT_MODULE_PATH = (
    EXP_ROOT / "safe_gait_experts" / "h4_training_alignment.py"
)
DEFAULT_H5_V3_COMMAND_CONDITIONED_SE2_ALIGNMENT_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts/h5_v3_command_conditioned_se2_alignment_250k_authorization_20260812.json"
)
PINNED_H5_V3_COMMAND_CONDITIONED_SE2_ALIGNMENT_AUTHORIZATION_SHA256 = (
    "42bb85195d50da19d9f376f7074733090df81ddd210f4d28ce8d533a9b229e79"
)
DEFAULT_H5_V3_SE2_SUBSTEP_CONTACT_PREFLIGHT_OUTPUT = (
    EXP_ROOT
    / "artifacts"
    / "h5_v3_se2_substep_contact_preflight_20260812"
    / "preflight_result.json"
)
DEFAULT_V4_SUBSTEP_COLLECTOR_TRACE_PREFLIGHT_OUTPUT = (
    EXP_ROOT
    / "artifacts"
    / "v4_substep_collector_trace_preflight_20260812"
    / "preflight_result.json"
)
DEFAULT_SELECTED_REVERSE_TEACHER = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_slew_feasible_teacher_selected_v1.json"
)
DEFAULT_FORWARD_MINIMUM_SPEC = (
    EXP_ROOT
    / "artifacts"
    / "h4_forward_retraining_minimum_spec_from_slip_causality_v1.json"
)
DEFAULT_REVERSE_MINIMUM_SPEC = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_retraining_minimum_spec_from_slip_causality_v1.json"
)
DEFAULT_REVERSE_COMPOSITION_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_training_composition_authorization_v1.json"
)
H5_TARGET_SPACE_DIAGNOSTIC_LEGACY_REWARD_CONFIG = {
    # The H5 reverse actor is authoritative in target space.  The old H4
    # target-imitation term compares incompatible residual/action coordinates
    # and drives the actor back to SAFE_INIT, so the diagnostic H5 route keeps
    # the legacy target term disabled while contact and motion terms remain.
    "target_imitation": 0.0,
    "contact_imitation": 0.0,
}
H5_PLANAR_DIAGNOSTIC_LEGACY_REWARD_CONFIG = {
    # The planar H5 actor is command-conditioned across ten routes.  The
    # legacy compound reference is not a valid target authority for lateral
    # and yaw commands; leaving its imitation cost active collapses the new
    # target-space actor toward SAFE_INIT instead of learning contact quality.
    "target_imitation": 0.0,
    "contact_imitation": 0.0,
}
DEFAULT_FORWARD_ITERATION_V2_AUTHORIZATION = (
    EXP_ROOT / "artifacts" / "h4_forward_iteration_v2_authorization.json"
)
DEFAULT_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts"
    / "h4_forward_iteration_v3_touchdown_balance_authorization.json"
)
DEFAULT_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts"
    / "h4_forward_iteration_v4_contact_event_validity_persistence_authorization.json"
)
DEFAULT_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts"
    / "h4_forward_iteration_v5_contact_pulse_abort_scale_only_authorization.json"
)
PINNED_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_AUTHORIZATION_SHA256 = (
    "c8a197e2b2eeb1b24cce1cace560841bd2620ee6bce5f97506c8c9f7518b210b"
)
DEFAULT_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts"
    / "h4_forward_iteration_v6_contact_abort_island_only_authorization.json"
)
PINNED_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_AUTHORIZATION_SHA256 = (
    "8e8b722e9e3f8f4b3827a7ffd2dee3e3ee5a2d799bfd996e09b066ff71d93a04"
)
DEFAULT_REVERSE_ITERATION_V2_AUTHORIZATION = (
    EXP_ROOT / "artifacts" / "h4_reverse_iteration_v2_authorization.json"
)
DEFAULT_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_iteration_v3_no_target_imitation_authorization.json"
)
DEFAULT_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_iteration_v4_residual_transfer_gain_024_authorization.json"
)
DEFAULT_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_iteration_v5_no_contact_imitation_authorization.json"
)
PINNED_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_AUTHORIZATION_SHA256 = (
    "1a0da8b77110c92fdaa0a81cdc41879a1a45660456567dfe68f88b2b5deb5976"
)
DEFAULT_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_AUTHORIZATION = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_iteration_v6_absolute_full_leg_targets_authorization.json"
)
PINNED_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_AUTHORIZATION_SHA256 = (
    "6a10315593761a4d0ed034b331fe14e3f682bf8154a252e6820a5dd4f71038fe"
)
PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256 = (
    "5da1d3a8a2c505a5ce4bc6621f76dd3031070cdb467a4cde96b4ed3c23190c02"
)
REVERSE_COMPOSITION_VALIDATOR_PATH = (
    EXP_ROOT / "scripts" / "validate_h4_reverse_training_composition.py"
)
PINNED_SELECTED_REVERSE_TEACHER_SHA256 = (
    "7a24a7c9096a1c4a9dc72ac85ec01c5e0a41acf8214d80cc7e2cf4ccc50ae237"
)
PINNED_FORWARD_V4_OFFICIAL_JOYSTICK_RELATIVE_PATH = (
    "playground/open_duck_mini_v2/joystick.py"
)
PINNED_FORWARD_V4_OFFICIAL_JOYSTICK_SHA256 = (
    "95890569d971725308b5a9c0996bfa5fd9520479f014f325e810aa1db272eb9d"
)
PINNED_FORWARD_V4_OFFICIAL_MJX_ENV_RELATIVE_PATH = (
    ".venv/lib/python3.12/site-packages/mujoco_playground/_src/mjx_env.py"
)
PINNED_FORWARD_V4_OFFICIAL_MJX_ENV_SHA256 = (
    "c3f1cfe0de036c3ccbba46e8cdd661cb48bfea8f182955298205f17787f53dfe"
)
PINNED_FORWARD_V4_OFFICIAL_STEP_SOURCE_SHA256 = (
    "26571e7510b2837dca07f69890dc26a89695dff4caa1fdc6a0d6736bd22da06b"
)
FORWARD_V4_OFFICIAL_STEP_SOURCE_SEMANTICS = (
    "LAX_SCAN_XS_EMPTY_LENGTH_NSUBSTEPS_BODY_REPLACE_CTRL_ACTION_THEN_"
    "MJX_STEP_RETURN_FINAL_CARRY"
)
PINNED_FORWARD_MINIMUM_SPEC_SHA256 = (
    "26611630368069e9cbd2516e08d5adb13547a5fa2763173ca04d67751be83428"
)
PINNED_FORWARD_MINIMUM_SPEC_CANONICAL_SHA256 = (
    "dff0b683020e3eec21e221249b27233ef008215fb156996cad314736f7c89d65"
)
PINNED_REVERSE_MINIMUM_SPEC_SHA256 = (
    "66b12bcbaf8a55cc0477b8872cebb8fe29c2c321b2c2224afd3089c5ecb500a8"
)
PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256 = (
    "082405e34b8a46e7d4a9ccf7b8c0729871fee1eb202b4a1ed8c758b2c7a52900"
)
PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256 = (
    "b574d4a41b05f54666f3befe41eda9a54b4e12970e6acaa7a9e95c1bf82de7c3"
)
PINNED_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_AUTHORIZATION_SHA256 = (
    "93daa0c35f08929c17c6eef799565d327ce362c1c1ebdeaf9aa22ca6cc5d153f"
)
PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256 = (
    "d364cc752c4702a6edada7fe5fac5ddfbab1926d5520b2bd0e1a20f532d6e3f3"
)
PINNED_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION_SHA256 = (
    "b27d3e12f5619bf008b5034f33e561a8ab8d06c3880a914f1a28781c0a3bb5c7"
)
PINNED_V22_PARENT_TREE_SHA256 = (
    "fe35e5ee932dc0ba70c1c32f3e410ea469d229e69cab43ed85f34aefe9505f1f"
)
DEFAULT_OUTPUT_ROOT = EXP_ROOT / "artifacts" / "h4_training_runs"
DEFAULT_LEARNING_RATE = 3.0e-5
DEFAULT_ENTROPY_COST = 1.0e-3
DEFAULT_PILOT_TIMESTEPS = 250_000
PROMOTED_TIMESTEPS = 1_000_000
WIRING_TIMESTEPS = 40
H5_V3_PRODUCTION_PILOT_SHAPE = (250_000, 1250, 20, 125, 20, 4)
H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE = 2
H5_V3_SUBSTEP_PREFLIGHT_CONTROL_STEPS = 20
H5_V3_SUBSTEP_PREFLIGHT_SHAPE = (40, 2, 20, 1, 2, 1)
V4_SUBSTEP_COLLECTOR_TRACE_PREFLIGHT_BATCH_SIZE = 2
V4_SUBSTEP_COLLECTOR_TRACE_PREFLIGHT_CONTROL_STEPS = 20
ITERATION_V2_WIRING_CONTRACT_IDS: Mapping[str, str] = {
    "forward": "H4_FORWARD_ITERATION_V2_WIRING_PREFLIGHT_40_FROM_V22",
    "reverse": "H4_REVERSE_ITERATION_V2_WIRING_PREFLIGHT_40_FROM_V22",
}
FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_250K_FROM_V22"
)
FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_WIRING_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_WIRING_PREFLIGHT_40_FROM_V22"
)
REVERSE_ITERATION_V3_NO_TARGET_IMITATION_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_250K_FROM_V22"
)
REVERSE_ITERATION_V3_NO_TARGET_IMITATION_WIRING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_WIRING_PREFLIGHT_40_FROM_V22"
)
FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_250K_FROM_V22"
)
FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_WIRING_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_250K_FROM_V22"
)
REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_WIRING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_250K_FROM_V22"
)
FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_WIRING_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_NO_PPO_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_"
    "NO_PPO_PREFLIGHT_FROM_V22"
)
REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_250K_FROM_V22"
)
REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_WIRING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_NO_PPO_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_"
    "NO_PPO_PREFLIGHT_FROM_V22"
)
FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_250K_FROM_V22"
)
FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_WIRING_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_NO_PPO_CONTRACT_ID = (
    "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_"
    "NO_PPO_PREFLIGHT_FROM_V22"
)
REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_250K_FROM_V22"
)
REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_WIRING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_"
    "WIRING_PREFLIGHT_40_FROM_V22"
)
REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_NO_PPO_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_"
    "NO_PPO_PREFLIGHT_FROM_V22"
)
REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN = 0.24
REVERSE_ITERATION_V6_RESIDUAL_AUTHORITY_SCALE = 0.0
PROMOTION_GATE_STATUS = "H4_STRICT_PROMOTION_PRODUCER_NOT_READY"
OLD_REVERSE_TEACHER_FIRST_JUMP_RAD = 0.408689
REQUIRED_FIRST_JUMP_IMPROVEMENT_RAD = 0.05
EXPERT_CHOICES = ("forward", "reverse", "planar", "unified")
OBSERVATION_MODE_CHOICES = ("legacy101", "h4_116_transplant")
CENTRAL_QUALITY_PATHS = (
    EXP_ROOT / "scripts" / "evaluate_routed_transitions.py",
    EXP_ROOT / "safe_gait_experts" / "gait_quality.py",
    EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py",
)
STRICT_EVALUATION_PRODUCER_PATH = (
    EXP_ROOT / "scripts" / "evaluate_h4_training_candidate.py"
)
H4_STRICT_PROMOTION_SEEDS = {
    "forward": (20_260_809, 20_261_809, 20_262_809),
    "reverse": (20_260_810, 20_265_810, 20_271_810),
    "planar": (20_260_811, 20_261_811, 20_262_811),
    "unified": (20_260_812, 20_261_812, 20_262_812),
}
H4_STRICT_DURATION_S = 6.0
H4_STRICT_PHYSICS_SUBSTEPS = 3_000
JAX_BACKEND_SELECTORS = {"cpu": "cpu", "gpu": "cuda,cpu"}
JAX_RESOLVED_BACKENDS = {"cpu": "cpu", "gpu": "gpu"}
GPU_XLA_FLAGS = "--xla_gpu_autotune_level=4"
GPU_XLA_POLICY = "CORRECTNESS_CHECKED_LEVEL4_DISQUALIFY_MISMATCH"


ANCHOR_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "forward": {
        "physical_primary": (0.05, 0.0, 0.0),
        "policy_observation_anchor": (0.10, -0.018, -0.170),
        "stand_probability": 0.10,
        "exact_primary_probability": H4_FORWARD_EXACT_ENDPOINT_PROBABILITY,
        "local_probability": 0.20,
        "local_vx_m_s": H4_FORWARD_LOCAL_ANCHORS_M_S,
        "transition_probability": 0.10,
        "transition_vx_m_s": H4_FORWARD_TRANSITION_BAND_M_S,
    },
    "reverse": {
        "physical_primary": (H4_REVERSE_PRIMARY_ANCHOR_M_S, 0.0, 0.0),
        "policy_observation_anchor": (-0.05, 0.0, 0.0),
        "stand_probability": 0.10,
        "exact_primary_probability": H4_REVERSE_EXACT_ENDPOINT_PROBABILITY,
        "local_probability": 0.20,
        "local_vx_m_s": H4_REVERSE_LOCAL_ANCHORS_M_S,
        "transition_probability": 0.10,
        "transition_vx_m_s": H4_REVERSE_TRANSITION_BAND_M_S,
    },
    # H5 planar is command-conditioned across the full non-reverse route
    # family; its sampler and mapper are supplied by h5_training_alignment.
    "planar": {
        "physical_primary": (0.0, 0.0, 0.0),
        "policy_observation_anchor": (0.0, 0.0, 0.0),
        "stand_probability": 0.10,
        "exact_primary_probability": 0.0,
        "local_probability": 0.0,
        "local_vx_m_s": (),
        "transition_probability": 0.90,
        "transition_vx_m_s": (),
    },
    "unified": {
        "physical_primary": (0.0, 0.0, 0.0),
        "policy_observation_anchor": (0.0, 0.0, 0.0),
        "stand_probability": 0.10,
        "exact_primary_probability": 0.0,
        "local_probability": 0.0,
        "local_vx_m_s": (),
        "transition_probability": 0.90,
        "transition_vx_m_s": (),
    },
}
FORWARD_ITERATION_V2_ANCHOR_CONFIG: Mapping[str, Any] = {
    **ANCHOR_CONFIGS["forward"],
    "stand_probability": H4_FORWARD_V2_STAND_PROBABILITY,
    "exact_primary_probability": H4_FORWARD_V2_EXACT_ENDPOINT_PROBABILITY,
    "local_probability": H4_FORWARD_V2_LOCAL_ANCHOR_PROBABILITY,
    "transition_probability": H4_FORWARD_V2_TRANSITION_PROBABILITY,
}
REVERSE_ITERATION_V2_ANCHOR_CONFIG: Mapping[str, Any] = {
    **ANCHOR_CONFIGS["reverse"],
    "stand_probability": H4_REVERSE_V2_STAND_PROBABILITY,
    "exact_primary_probability": H4_REVERSE_V2_EXACT_ENDPOINT_PROBABILITY,
    "local_probability": H4_REVERSE_V2_LOCAL_ANCHOR_PROBABILITY,
    "transition_probability": H4_REVERSE_V2_TRANSITION_PROBABILITY,
}
REVERSE_ITERATION_V2_LEGACY_REWARD_CONFIG: Mapping[str, float] = {
    "target_imitation": -20.0,
    "contact_imitation": 15.0,
    "tracking_sigma": 0.01,
}
REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG: Mapping[
    str, float
] = {
    **REVERSE_ITERATION_V2_LEGACY_REWARD_CONFIG,
    "target_imitation": 0.0,
}
REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_LEGACY_REWARD_CONFIG: Mapping[
    str, float
] = {
    **REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG,
    "contact_imitation": 0.0,
}
REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_LEGACY_REWARD_CONFIG: Mapping[
    str, float
] = dict(REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_LEGACY_REWARD_CONFIG)


def _load_legacy_trainer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exp004_h4_isolated_legacy_trainer", LEGACY_TRAINER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy trainer: {LEGACY_TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def h5_preflight_raw_array_digest(value: Any) -> str:
    """Hash dtype, shape, and literal array bytes (including NaN payloads)."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("preflight values may not use object dtype")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def h5_preflight_ordered_array_bundle_digest(
    fields: Mapping[str, Any],
) -> str:
    """Hash named raw arrays in caller-specified order for sealed trace I/O."""

    digest = hashlib.sha256()
    for name, value in fields.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(h5_preflight_raw_array_digest(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def stablehlo_location_stripped_sha256(stablehlo: str) -> str:
    """Hash StableHLO while retaining every operation and dropping source locations.

    MLIR location definitions encode host source line/column and generated symbol
    numbering.  Those fields change when a pure host-side preflight is appended,
    even when the compiled collector operations are identical.  This helper never
    removes an operation, operand, type, attribute, or constant.
    """

    retained_lines = []
    for line in stablehlo.splitlines():
        if line.lstrip().startswith("#loc"):
            continue
        retained_lines.append(re.sub(r"\s+loc\(#[^)]+\)", "", line))
    return hashlib.sha256("\n".join(retained_lines).encode("utf-8")).hexdigest()


def stablehlo_semantic_sha256(stablehlo: str) -> str:
    """Hash collector IR while normalizing only JAX callback registry handles.

    A ``jax.debug.callback`` lowers to ``@xla_python_cpu_callback`` with a host
    process-local registry integer embedded both in the immediate ``i64``
    constant and in ``backend_config``. The integer is allocated afresh in each
    process, so it is not part of the collector's physics or assertion
    semantics. The callback target, operand/result types, all other attributes,
    and the rest of StableHLO remain literal. The callback source is separately
    SHA-256-bound in every parent artifact.
    """

    retained_lines = []
    for line in stablehlo.splitlines():
        if line.lstrip().startswith("#loc"):
            continue
        retained_lines.append(re.sub(r"\s+loc\(#[^)]+\)", "", line))

    for index, line in enumerate(retained_lines):
        if (
            index + 1 < len(retained_lines)
            and re.fullmatch(
                r"\s*%[A-Za-z0-9_.]+ = stablehlo\.constant dense<\d+> : tensor<i64>",
                line,
            )
            and "@xla_python_cpu_callback(" in retained_lines[index + 1]
        ):
            retained_lines[index] = re.sub(
                r"dense<\d+>", "dense<__JAX_CALLBACK_HANDLE__>", line
            )

    inside_cpu_callback = False
    for index, line in enumerate(retained_lines):
        if "@xla_python_cpu_callback(" in line:
            inside_cpu_callback = True
        if inside_cpu_callback:
            retained_lines[index] = re.sub(
                r'(backend_config\s*=\s*")\d+(")',
                r"\1__JAX_CALLBACK_HANDLE__\2",
                line,
            )
        if inside_cpu_callback and "-> tuple<>" in line:
            inside_cpu_callback = False
    return hashlib.sha256("\n".join(retained_lines).encode("utf-8")).hexdigest()


def h5_preflight_raw_array_equal(left: Any, right: Any) -> bool:
    """Literal equality, unlike ``array_equal`` which rejects equal NaNs."""

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return bool(
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and np.ascontiguousarray(left_array).tobytes(order="C")
        == np.ascontiguousarray(right_array).tobytes(order="C")
    )


def h5_preflight_raw_tree_digest(jax: Any, value: Any) -> tuple[str, int]:
    """Hash canonical leaf paths, dtype/shape, and literal array bytes."""

    path_leaves, _tree = jax.tree_util.tree_flatten_with_path(value)
    digest = hashlib.sha256()
    for path, leaf in path_leaves:
        path_text = jax.tree_util.keystr(path).encode("utf-8")
        digest.update(len(path_text).to_bytes(8, "big"))
        digest.update(path_text)
        digest.update(bytes.fromhex(h5_preflight_raw_array_digest(leaf)))
    return digest.hexdigest(), len(path_leaves)


def h5_preflight_raw_array_difference(
    left: Any, right: Any
) -> dict[str, Any]:
    """Return exact mismatch evidence; never use it as an acceptance tolerance."""

    left_array = np.ascontiguousarray(np.asarray(left))
    right_array = np.ascontiguousarray(np.asarray(right))
    result: dict[str, Any] = {
        "exact_raw_equal": h5_preflight_raw_array_equal(left_array, right_array),
        "left_dtype": left_array.dtype.str,
        "right_dtype": right_array.dtype.str,
        "left_shape": list(left_array.shape),
        "right_shape": list(right_array.shape),
    }
    if result["exact_raw_equal"]:
        return result
    if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
        return result
    left_bytes = left_array.reshape(-1).view(np.uint8).reshape(
        -1, left_array.dtype.itemsize
    )
    right_bytes = right_array.reshape(-1).view(np.uint8).reshape(
        -1, right_array.dtype.itemsize
    )
    differing_flat = np.flatnonzero(np.any(left_bytes != right_bytes, axis=1))
    if differing_flat.size == 0:
        raise RuntimeError("raw mismatch reported without a differing element")
    flat_index = int(differing_flat[0])
    index = np.unravel_index(flat_index, left_array.shape)
    result.update(
        {
            "first_differing_flat_index": flat_index,
            "first_differing_index": [int(value) for value in index],
            "left_element_raw_hex": left_array.reshape(-1)[flat_index].tobytes().hex(),
            "right_element_raw_hex": right_array.reshape(-1)[flat_index].tobytes().hex(),
            "differing_element_count": int(differing_flat.size),
        }
    )
    if np.issubdtype(left_array.dtype, np.floating):
        finite = np.isfinite(left_array) & np.isfinite(right_array)
        if np.any(finite):
            result["max_abs_difference_finite"] = float(
                np.max(np.abs(left_array[finite] - right_array[finite]))
            )
        if left_array.dtype in (np.dtype("float32"), np.dtype("float64")):
            unsigned_dtype = (
                np.dtype("uint32")
                if left_array.dtype.itemsize == 4
                else np.dtype("uint64")
            )
            bit_width = left_array.dtype.itemsize * 8
            sign_bit = np.array(1 << (bit_width - 1), dtype=unsigned_dtype)
            left_bits = left_array.view(unsigned_dtype)
            right_bits = right_array.view(unsigned_dtype)
            left_ordered = np.where(
                (left_bits & sign_bit) != 0, ~left_bits, left_bits | sign_bit
            ).astype(np.uint64)
            right_ordered = np.where(
                (right_bits & sign_bit) != 0, ~right_bits, right_bits | sign_bit
            ).astype(np.uint64)
            ulp = np.where(
                left_ordered >= right_ordered,
                left_ordered - right_ordered,
                right_ordered - left_ordered,
            )
            result["max_ulp_difference"] = int(np.max(ulp))
    return result


def h5_preflight_raw_tree_equal(
    jax: Any, left: Any, right: Any
) -> tuple[bool, int, str, str]:
    """Compare a PyTree without losing signed zero/dtype/NaN distinctions."""

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    left_digest, left_count = h5_preflight_raw_tree_digest(jax, left)
    right_digest, right_count = h5_preflight_raw_tree_digest(jax, right)
    if left_tree != right_tree or left_count != right_count:
        return False, left_count, left_digest, right_digest
    return (
        all(
            h5_preflight_raw_array_equal(left_leaf, right_leaf)
            for left_leaf, right_leaf in zip(left_leaves, right_leaves)
        ),
        left_count,
        left_digest,
        right_digest,
    )


def h5_preflight_leaf_records(jax: Any, value: Any) -> list[dict[str, Any]]:
    """Serialize raw per-leaf state evidence for a failed T=1 diagnostic."""

    records: list[dict[str, Any]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        array = np.asarray(leaf)
        records.append(
            {
                "path": jax.tree_util.keystr(path),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "raw_bytes_sha256": h5_preflight_raw_array_digest(array),
                "nan_count": int(np.count_nonzero(np.isnan(array)))
                if np.issubdtype(array.dtype, np.inexact)
                else 0,
                "posinf_count": int(np.count_nonzero(np.isposinf(array)))
                if np.issubdtype(array.dtype, np.inexact)
                else 0,
                "neginf_count": int(np.count_nonzero(np.isneginf(array)))
                if np.issubdtype(array.dtype, np.inexact)
                else 0,
            }
        )
    return records


def load_h5_v3_command_conditioned_se2_alignment_authorization(
    path: Path = DEFAULT_H5_V3_COMMAND_CONDITIONED_SE2_ALIGNMENT_AUTHORIZATION,
) -> dict[str, Any]:
    """Load the one-pilot H5 V3 SE(2) authorization fail-closed."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing H5 V3 SE(2) authorization: {resolved}")
    digest = sha256_file(resolved)
    if digest != PINNED_H5_V3_COMMAND_CONDITIONED_SE2_ALIGNMENT_AUTHORIZATION_SHA256:
        raise ValueError("H5 V3 SE(2) authorization SHA256 drifted")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    scope = payload.get("scope")
    if not isinstance(scope, Mapping) or (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind")
        != "openduckmini_h5_v3_command_conditioned_se2_alignment_250k_authorization"
        or payload.get("status") != "AUTHORIZED_SIMULATION_250K_ONLY"
        or payload.get("hardware_deployment") != "PROHIBITED"
        or scope.get("contract_id") != H5_V3_COMMAND_CONDITIONED_SE2_ALIGNMENT_ID
        or scope.get("required_cli_flag")
        != "--h5-v3-command-conditioned-se2-alignment"
        or scope.get("expert") != "unified"
        or scope.get("command_mapper") != "direct_normalized_v3"
        or scope.get("one_new_250k_pilot_only") is not True
    ):
        raise ValueError("H5 V3 SE(2) authorization semantics drifted")
    inputs = payload.get("frozen_inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("H5 V3 SE(2) authorization frozen inputs are absent")
    for label in (
        "current_rejected_candidate",
        "current_v3_resolved_config",
        "measurement_alignment_preflight",
    ):
        source = inputs.get(label)
        if not isinstance(source, Mapping):
            raise ValueError(f"H5 V3 SE(2) authorization source is absent: {label}")
        source_path = EXP_ROOT / str(source.get("path", ""))
        if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
            raise ValueError(f"H5 V3 SE(2) authorization source drifted: {label}")
    preflight_path = EXP_ROOT / str(inputs["measurement_alignment_preflight"]["path"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != inputs["measurement_alignment_preflight"].get(
        "required_status"
    ):
        raise ValueError("H5 V3 SE(2) measurement-alignment preflight is not passing")
    return {
        "path": str(resolved),
        "sha256": digest,
        "contract_id": scope["contract_id"],
        "status": payload["status"],
    }


def require_iteration_v6_core_source(
    path: Path = ALIGNMENT_MODULE_PATH,
) -> dict[str, Any]:
    """Bind the independently audited v6 core before any PPO/pickle work."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing iteration-v6 core source: {resolved}")
    actual_sha = sha256_file(resolved)
    if actual_sha != PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256:
        raise ValueError(
            "iteration-v6 core source SHA256 drifted: "
            f"expected={PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256}, "
            f"actual={actual_sha}"
        )
    return {
        "path": str(resolved),
        "sha256": actual_sha,
    }


def resolve_jax_backend_selector(requested_platform: str) -> str:
    """Map the user-facing platform to the exact frozen-JAX selector.

    JAX 0.5.3 with the CUDA 12 plugin resolves devices as platform ``gpu``,
    but its selector is ``cuda``.  The CPU backend remains explicitly listed
    because XLA/JAX debug callbacks used by the frozen PPO stack require a
    local CPU device.  CUDA is first, so the default/training backend is GPU.
    Passing ``JAX_PLATFORMS=gpu`` can route through the ROCm plugin and fail
    before the first PPO update.
    """

    try:
        return JAX_BACKEND_SELECTORS[requested_platform]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"unsupported requested JAX platform: {requested_platform!r}"
        ) from exc


def configure_xla_autotune_policy(
    requested_platform: str, *, environ: Any = os.environ
) -> dict[str, Any]:
    """Pin correctness-checked GPU autotuning before JAX stack import."""

    resolve_jax_backend_selector(requested_platform)
    existing = environ.get("XLA_FLAGS")
    if existing not in (None, ""):
        raise ValueError(
            "preexisting XLA_FLAGS override is prohibited; expected unset or empty, "
            f"got {existing!r}"
        )
    if requested_platform == "gpu":
        environ["XLA_FLAGS"] = GPU_XLA_FLAGS
        effective = GPU_XLA_FLAGS
        policy = GPU_XLA_POLICY
    else:
        effective = existing or ""
        policy = "CPU_NO_GPU_AUTOTUNE_FLAG"
    return {
        "requested_cli_platform": requested_platform,
        "xla_flags_before": existing,
        "xla_flags_effective": effective,
        "policy": policy,
        "configured_before_training_stack_import": True,
        "correctness_check_enabled": requested_platform == "gpu",
        "mismatching_autotune_candidates_disqualified": requested_platform == "gpu",
        "cpu_mode_did_not_set_xla_flags": requested_platform != "cpu"
        or "XLA_FLAGS" not in environ
        or environ.get("XLA_FLAGS") == "",
        "passed": True,
    }


def validate_resolved_jax_backend(
    jax: Any, *, requested_platform: str, selector: str
) -> dict[str, Any]:
    """Fail closed unless JAX resolved the backend requested by the CLI."""

    expected_selector = resolve_jax_backend_selector(requested_platform)
    if selector != expected_selector:
        raise ValueError(
            f"JAX selector drifted: {selector!r} != {expected_selector!r}"
        )
    expected_backend = JAX_RESOLVED_BACKENDS[requested_platform]
    resolved_backend = str(jax.default_backend())
    devices = tuple(jax.devices(expected_backend))
    device_platforms = tuple(str(device.platform) for device in devices)
    cpu_callback_devices = tuple(jax.devices("cpu"))
    if (
        resolved_backend != expected_backend
        or not devices
        or any(platform != expected_backend for platform in device_platforms)
        or not cpu_callback_devices
        or any(str(device.platform) != "cpu" for device in cpu_callback_devices)
    ):
        raise RuntimeError(
            "JAX backend resolution mismatch: "
            f"requested={requested_platform!r}, selector={selector!r}, "
            f"default={resolved_backend!r}, devices={device_platforms!r}, "
            f"cpu_callbacks={tuple(str(device) for device in cpu_callback_devices)!r}"
        )
    return {
        "requested_cli_platform": requested_platform,
        "jax_platform_selector": selector,
        "expected_resolved_backend": expected_backend,
        "resolved_default_backend": resolved_backend,
        "resolved_device_platforms": list(device_platforms),
        "resolved_devices": [str(device) for device in devices],
        "local_cpu_callback_devices": [
            str(device) for device in cpu_callback_devices
        ],
        "local_cpu_callback_available": True,
        "passed": True,
    }


def run_jax_debug_callback_preflight(jax: Any, jp: Any) -> dict[str, Any]:
    """Execute the callback path that failed when CPU was omitted."""

    observed: list[float] = []

    def receive(value: Any) -> None:
        observed.append(float(np.asarray(value)))

    def callback_probe(value: Any) -> Any:
        jax.debug.callback(receive, value)
        return value + jp.asarray(1.0, dtype=value.dtype)

    result = jax.jit(callback_probe)(jp.asarray(2.0, dtype=jp.float32))
    result.block_until_ready()
    if observed != [2.0] or float(np.asarray(result)) != 3.0:
        raise RuntimeError(
            f"JAX debug callback preflight failed: observed={observed}, result={result}"
        )
    return {
        "input": 2.0,
        "callback_observed": observed[0],
        "result": 3.0,
        "local_cpu_callback_executed": True,
        "passed": True,
    }


FORWARD_V4_DYNAMIC6_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "time",
    "qacc_warmstart",
)
FORWARD_V4_SINGLE_AUTHORITY_INFO_KEYS = {
    "dynamic6_exact": "h4_v4_single_authority_dynamic6_exact",
    "dynamic6_max_abs_error": (
        "h4_v4_single_authority_dynamic6_max_abs_error"
    ),
    "dynamic6_field_count": "h4_v4_single_authority_dynamic6_field_count",
    "dynamic6_field_count_exact": (
        "h4_v4_single_authority_dynamic6_field_count_exact"
    ),
    "saved_dynamic6_substep_count": "h4_v4_saved_dynamic6_substep_count",
    "saved_dynamic6_field_count": "h4_v4_saved_dynamic6_field_count",
    "saved_dynamic6_field_count_exact": (
        "h4_v4_saved_dynamic6_field_count_exact"
    ),
    "saved_dynamic6_all_finite": "h4_v4_saved_dynamic6_all_finite",
    "telemetry_force_shape_valid": "h4_v4_telemetry_force_shape_valid",
    "telemetry_force_all_finite": "h4_v4_telemetry_force_all_finite",
    "authority_violation": "h4_v4_single_authority_violation",
    "assertion_token": "h4_v4_single_authority_assertion_token",
}
FORWARD_V4_SINGLE_AUTHORITY_EPISODE_KEYS = {
    name: f"episode/h4/{key.removeprefix('h4_')}"
    for name, key in FORWARD_V4_SINGLE_AUTHORITY_INFO_KEYS.items()
}
FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT = {
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
FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE = (
    "WIRING_COMPILED_ASSERTION_NO_EPISODE_ROWS_EXPECTED"
)
FORWARD_V4_FULL_RUNTIME_AUDIT_MODE = "FULL_RUNTIME_EPISODE_ROWS_REQUIRED"
FORWARD_V4_WIRING_COMPLETION_REQUIREMENT = {
    "source_semantic_preflight_passed": True,
    "per_step_compiled_fail_closed_assertion_bound": True,
    "completed_environment_interactions": 40,
    "completed_training_steps": 1,
    "completed_optimizer_updates": 2,
    "progress_reached_final_interaction": True,
    "final_params_all_finite": True,
    "final_metrics_all_finite": True,
    "source_and_teacher_unchanged": True,
}


def _finite_scalar(value: Any, *, label: str) -> float:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.number):
        raise RuntimeError(f"{label} must be one numeric scalar")
    result = float(array)
    if not np.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def _strict_bool_scalar(value: Any, *, label: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "b":
        raise RuntimeError(f"{label} must be one boolean scalar")
    return bool(array)


def require_forward_v4_single_authority_samples(
    samples: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Any]:
    """Require every runtime single-authority invariant on every v4 step."""

    if not samples:
        raise RuntimeError(
            f"{label} did not observe forward-v4 single-authority state"
        )
    for index, sample in enumerate(samples):
        if set(sample) != set(FORWARD_V4_SINGLE_AUTHORITY_INFO_KEYS):
            raise RuntimeError(
                f"{label}[{index}] single-authority field set drifted"
            )
        exact = _strict_bool_scalar(
            sample["dynamic6_exact"],
            label=f"{label}[{index}].dynamic6_exact",
        )
        error = _finite_scalar(
            sample["dynamic6_max_abs_error"],
            label=f"{label}[{index}].dynamic6_max_abs_error",
        )
        field_count = _finite_scalar(
            sample["dynamic6_field_count"],
            label=f"{label}[{index}].dynamic6_field_count",
        )
        field_count_exact = _strict_bool_scalar(
            sample["dynamic6_field_count_exact"],
            label=f"{label}[{index}].dynamic6_field_count_exact",
        )
        saved_substeps = _finite_scalar(
            sample["saved_dynamic6_substep_count"],
            label=f"{label}[{index}].saved_dynamic6_substep_count",
        )
        saved_fields = _finite_scalar(
            sample["saved_dynamic6_field_count"],
            label=f"{label}[{index}].saved_dynamic6_field_count",
        )
        saved_field_count_exact = _strict_bool_scalar(
            sample["saved_dynamic6_field_count_exact"],
            label=f"{label}[{index}].saved_dynamic6_field_count_exact",
        )
        saved_finite = _strict_bool_scalar(
            sample["saved_dynamic6_all_finite"],
            label=f"{label}[{index}].saved_dynamic6_all_finite",
        )
        telemetry_shape = _strict_bool_scalar(
            sample["telemetry_force_shape_valid"],
            label=f"{label}[{index}].telemetry_force_shape_valid",
        )
        telemetry_finite = _strict_bool_scalar(
            sample["telemetry_force_all_finite"],
            label=f"{label}[{index}].telemetry_force_all_finite",
        )
        violation = _strict_bool_scalar(
            sample["authority_violation"],
            label=f"{label}[{index}].authority_violation",
        )
        assertion_token = _finite_scalar(
            sample["assertion_token"],
            label=f"{label}[{index}].assertion_token",
        )
        if (
            exact is not True
            or error != 0.0
            or field_count != 6.0
            or field_count_exact is not True
            or saved_substeps != 10.0
            or saved_fields != 6.0
            or saved_field_count_exact is not True
            or saved_finite is not True
            or telemetry_shape is not True
            or telemetry_finite is not True
            or violation is not False
            or assertion_token != 0.0
        ):
            raise RuntimeError(
                f"{label}[{index}] forward-v4 single-authority audit failed: "
                f"{dict(sample)!r}"
            )
    return {
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
        "authority_violation_count": 0,
        "assertion_token_sum": 0.0,
        "observed_step_count": len(samples),
        "passed": True,
    }


def forward_v4_single_authority_sample_from_info(
    info: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        key
        for key in FORWARD_V4_SINGLE_AUTHORITY_INFO_KEYS.values()
        if key not in info
    ]
    if missing:
        raise RuntimeError(
            f"forward-v4 single-authority info is missing {missing}"
        )
    return {
        name: info[key]
        for name, key in FORWARD_V4_SINGLE_AUTHORITY_INFO_KEYS.items()
    }


def require_forward_v4_single_authority_runtime_progress(
    rows: Sequence[Mapping[str, Any]],
    *,
    wiring_only: bool = False,
    wiring_completion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Close the stage-appropriate PPO single-authority evidence contract.

    A full 250k run must expose at least one completed-episode aggregate row.
    The exact 40-interaction wiring preflight can legitimately finish before
    either 20-step environment terminates.  In that stage only, zero episode
    rows are accepted after the source-semantic preflight, the compiled
    per-step fail-closed assertion, the exact 40/1/2 PPO shape, finite final
    state, and immutable source closure have all succeeded.
    """

    if not isinstance(wiring_only, bool):
        raise RuntimeError("forward-v4 wiring_only must be boolean")
    if wiring_only:
        actual_completion = (
            dict(wiring_completion) if wiring_completion is not None else None
        )
        completion_exact = bool(
            actual_completion is not None
            and set(actual_completion)
            == set(FORWARD_V4_WIRING_COMPLETION_REQUIREMENT)
            and actual_completion == FORWARD_V4_WIRING_COMPLETION_REQUIREMENT
            and all(
                actual_completion[key] is expected
                for key, expected in FORWARD_V4_WIRING_COMPLETION_REQUIREMENT.items()
                if isinstance(expected, bool)
            )
            and all(
                type(actual_completion[key]) is int
                for key, expected in FORWARD_V4_WIRING_COMPLETION_REQUIREMENT.items()
                if isinstance(expected, int) and not isinstance(expected, bool)
            )
        )
        if completion_exact is not True:
            raise RuntimeError(
                "forward-v4 wiring completion evidence is not exact: "
                f"{wiring_completion!r}"
            )
    elif wiring_completion is not None:
        raise RuntimeError(
            "forward-v4 full runtime must not use wiring completion evidence"
        )

    episode_rows = [row for row in rows if "episode/length" in row]
    if not episode_rows and not wiring_only:
        raise RuntimeError(
            "PPO produced no forward-v4 single-authority episode rows"
        )
    required = set(FORWARD_V4_SINGLE_AUTHORITY_EPISODE_KEYS.values())
    for index, row in enumerate(episode_rows):
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(
                f"forward-v4 PPO single-authority row {index} is missing {missing}"
            )
        length = _finite_scalar(
            row["episode/length"],
            label=f"forward-v4 single-authority row {index}.length",
        )
        if length <= 0.0:
            raise RuntimeError(
                "forward-v4 PPO single-authority audit failed at row "
                f"{index}: length={length!r}"
            )
        totals = {
            name: _finite_scalar(
                row[key],
                label=f"forward-v4 single-authority row {index}.{name}",
            )
            for name, key in FORWARD_V4_SINGLE_AUTHORITY_EPISODE_KEYS.items()
        }
        expected_totals = {
            "dynamic6_exact": length,
            "dynamic6_max_abs_error": 0.0,
            "dynamic6_field_count_exact": length,
            "saved_dynamic6_field_count_exact": length,
            "saved_dynamic6_all_finite": length,
            "telemetry_force_shape_valid": length,
            "telemetry_force_all_finite": length,
            "authority_violation": 0.0,
            "assertion_token": 0.0,
        }
        qualifying_totals = {
            name: totals[name] for name in expected_totals
        }
        # The three integer-count aggregates above remain finite, recorded
        # diagnostics.  Their exact device-derived predicates and the OR-ed
        # authority violation are the qualifying evidence; recomputing either
        # ``N * length`` or ``total / length`` on the host would reintroduce an
        # independently rounded operation into the launch gate.
        if qualifying_totals != expected_totals:
            raise RuntimeError(
                "forward-v4 PPO single-authority audit failed at row "
                f"{index}: length={length!r}, observed={totals!r}, "
                f"expected_qualifying={expected_totals!r}"
            )
    if wiring_only:
        return {
            "audit_mode": FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE,
            "observed_episode_metric_rows": len(episode_rows),
            "episode_metric_rows_exact_if_observed": True,
            **FORWARD_V4_WIRING_COMPLETION_REQUIREMENT,
            "authority_violation_count": 0.0,
            "assertion_token_sum": 0.0,
            "passed": True,
        }
    return {
        "audit_mode": FORWARD_V4_FULL_RUNTIME_AUDIT_MODE,
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
        "observed_episode_metric_rows": len(episode_rows),
        "authority_violation_count": 0.0,
        "assertion_token_sum": 0.0,
        "passed": True,
    }


def require_forward_v4_official_source_provenance(
    *,
    source_root: Path,
    joystick_module: Any,
    mjx_env_module: Any,
) -> dict[str, Any]:
    """Bind the exact official wrapper files used by the reference gate."""

    resolved_root = Path(source_root).resolve()
    if not resolved_root.is_dir():
        raise RuntimeError(
            f"forward-v4 source-semantic root is missing: {resolved_root}"
        )
    expected = {
        "joystick": (
            joystick_module,
            PINNED_FORWARD_V4_OFFICIAL_JOYSTICK_RELATIVE_PATH,
            PINNED_FORWARD_V4_OFFICIAL_JOYSTICK_SHA256,
        ),
        "mjx_env": (
            mjx_env_module,
            PINNED_FORWARD_V4_OFFICIAL_MJX_ENV_RELATIVE_PATH,
            PINNED_FORWARD_V4_OFFICIAL_MJX_ENV_SHA256,
        ),
    }
    records: dict[str, dict[str, str]] = {}
    for label, (module, expected_relative, expected_sha) in expected.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError(
                f"forward-v4 official {label} module has no source file"
            )
        resolved_path = Path(raw_path).resolve()
        try:
            relative = resolved_path.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"forward-v4 official {label} resolved outside source root"
            ) from exc
        actual_sha = sha256_file(resolved_path)
        if relative != expected_relative or actual_sha != expected_sha:
            raise RuntimeError(
                f"forward-v4 official {label} provenance drifted: "
                f"relative={relative!r}, sha256={actual_sha!r}"
            )
        records[label] = {
            "resolved_path": str(resolved_path),
            "relative_path": relative,
            "sha256": actual_sha,
        }
    try:
        step_source = inspect.getsource(mjx_env_module.step)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "forward-v4 official mjx_env.step source is unavailable"
        ) from exc
    step_source_sha = hashlib.sha256(step_source.encode("utf-8")).hexdigest()
    if step_source_sha != PINNED_FORWARD_V4_OFFICIAL_STEP_SOURCE_SHA256:
        raise RuntimeError(
            "forward-v4 official mjx_env.step source semantics drifted: "
            f"{step_source_sha}"
        )
    return {
        "source_root": str(resolved_root),
        **records,
        "step_source_sha256": step_source_sha,
        "step_source_semantics": FORWARD_V4_OFFICIAL_STEP_SOURCE_SEMANTICS,
        "all_files_under_requested_source_root": True,
        "passed": True,
    }


FORWARD_ITERATION_V6_REWARD_ROUTING_INFO_KEYS = {
    "routing_exact": "h4_v6_forward_contact_abort_routing_exact",
    "island_loss": "h4_v6_forward_contact_abort_island_loss",
    "off_gap_diagnostic_loss": (
        "h4_v6_forward_contact_abort_off_gap_diagnostic_loss"
    ),
    "off_gap_reward_contribution": (
        "h4_v6_forward_contact_abort_off_gap_reward_contribution"
    ),
    "pulse_reward_scale": "h4_v6_forward_contact_abort_pulse_reward_scale",
    "routing_violation": "h4_v6_forward_contact_abort_routing_violation",
    "assertion_token": (
        "h4_v6_forward_contact_abort_routing_assertion_token"
    ),
}
REVERSE_ITERATION_V6_DECODER_INFO_KEYS = {
    "decoder_exact": "h4_v6_reverse_decoder_exact",
    "max_abs_error": "h4_v6_reverse_decoder_max_abs_error",
    "leg_count": "h4_v6_reverse_decoder_leg_count",
    "leg_count_exact": "h4_v6_reverse_decoder_leg_count_exact",
    "head_zero_exact": "h4_v6_reverse_decoder_head_zero_exact",
    "teacher_target_contribution_zero_exact": (
        "h4_v6_reverse_teacher_target_contribution_zero_exact"
    ),
    "residual_authority_scale": "h4_v6_reverse_residual_authority_scale",
    "decoder_all_finite": "h4_v6_reverse_decoder_all_finite",
    "margin_saturation_count": (
        "h4_v6_reverse_decoder_margin_saturation_count"
    ),
    "action_clip_count": "h4_v6_reverse_decoder_action_clip_count",
    "guard_lag_max_rad": "h4_v6_reverse_decoder_guard_lag_max_rad",
    "precomposer_call_count": "h4_v6_reverse_precomposer_call_count",
    "precomposer_call_count_exact": (
        "h4_v6_reverse_precomposer_call_count_exact"
    ),
    "final_guard_call_count": "h4_v6_reverse_final_guard_call_count",
    "final_guard_call_count_exact": (
        "h4_v6_reverse_final_guard_call_count_exact"
    ),
    "decoder_violation": "h4_v6_reverse_decoder_violation",
    "assertion_token": "h4_v6_reverse_decoder_assertion_token",
}
REVERSE_ITERATION_V6_DECODER_VECTOR_INFO_KEYS = {
    "action": "h4_v6_reverse_decoder_action",
    "raw_targets": "h4_v6_reverse_decoder_raw_targets",
    "margin_targets": "h4_v6_reverse_decoder_margin_targets",
}
FORWARD_ITERATION_V6_REWARD_ROUTING_EPISODE_KEYS = {
    name: f"episode/h4/{key.removeprefix('h4_')}"
    for name, key in FORWARD_ITERATION_V6_REWARD_ROUTING_INFO_KEYS.items()
}
REVERSE_ITERATION_V6_DECODER_EPISODE_KEYS = {
    name: f"episode/h4/{key.removeprefix('h4_')}"
    for name, key in REVERSE_ITERATION_V6_DECODER_INFO_KEYS.items()
}
FORWARD_ITERATION_V6_REWARD_ROUTING_RUNTIME_REQUIREMENT = {
    "routing_exact": True,
    "island_loss": "NON_NEGATIVE_FINITE_QUALIFYING_LOSS",
    "off_gap_diagnostic_loss": "NON_NEGATIVE_FINITE_NON_QUALIFYING_ONLY",
    "off_gap_reward_contribution": 0.0,
    "pulse_reward_scale": -1.0,
    "routing_violation_count": 0.0,
    "assertion_token_sum": 0.0,
    "per_step_compiled_fail_closed_assertion_required": True,
    "fail_closed_before_output_commit": True,
}
REVERSE_ITERATION_V6_DECODER_RUNTIME_REQUIREMENT = {
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
ITERATION_V6_WIRING_COMPLETION_REQUIREMENT = {
    "per_step_compiled_fail_closed_assertion_bound": True,
    "completed_environment_interactions": 40,
    "completed_training_steps": 1,
    "completed_optimizer_updates": 2,
    "progress_reached_final_interaction": True,
    "final_params_all_finite": True,
    "final_metrics_all_finite": True,
    "source_and_teacher_unchanged": True,
}
ITERATION_V6_FULL_COMPLETION_REQUIREMENT = {
    "per_step_compiled_fail_closed_assertion_bound": True,
    "completed_environment_interactions": 250000,
    "completed_training_steps": 5,
    "completed_optimizer_updates": 400,
    "progress_reached_final_interaction": True,
    "final_params_all_finite": True,
    "final_metrics_all_finite": True,
    "source_and_teacher_unchanged": True,
}


def _iteration_v6_sample_from_info(
    info: Mapping[str, Any], *, expert: str
) -> dict[str, Any]:
    keys = (
        FORWARD_ITERATION_V6_REWARD_ROUTING_INFO_KEYS
        if expert == "forward"
        else REVERSE_ITERATION_V6_DECODER_INFO_KEYS
        if expert == "reverse"
        else None
    )
    if keys is None:
        raise ValueError(f"unsupported iteration-v6 expert: {expert!r}")
    missing = sorted(key for key in keys.values() if key not in info)
    if missing:
        raise RuntimeError(f"{expert} iteration-v6 info is missing {missing}")
    return {name: info[key] for name, key in keys.items()}


def require_iteration_v6_runtime_samples(
    samples: Sequence[Mapping[str, Any]], *, expert: str, label: str
) -> dict[str, Any]:
    """Require exact device-derived v6 invariants for every observed step."""

    expected_keys = (
        FORWARD_ITERATION_V6_REWARD_ROUTING_INFO_KEYS
        if expert == "forward"
        else REVERSE_ITERATION_V6_DECODER_INFO_KEYS
        if expert == "reverse"
        else None
    )
    if expected_keys is None:
        raise ValueError(f"unsupported iteration-v6 expert: {expert!r}")
    if not samples:
        raise RuntimeError(f"{label} did not observe iteration-v6 runtime state")
    for index, sample in enumerate(samples):
        if set(sample) != set(expected_keys):
            raise RuntimeError(f"{label}[{index}] iteration-v6 field set drifted")
        if expert == "forward":
            exact = _strict_bool_scalar(
                sample["routing_exact"], label=f"{label}[{index}].routing_exact"
            )
            island = _finite_scalar(
                sample["island_loss"], label=f"{label}[{index}].island_loss"
            )
            off_gap = _finite_scalar(
                sample["off_gap_diagnostic_loss"],
                label=f"{label}[{index}].off_gap_diagnostic_loss",
            )
            contribution = _finite_scalar(
                sample["off_gap_reward_contribution"],
                label=f"{label}[{index}].off_gap_reward_contribution",
            )
            scale = _finite_scalar(
                sample["pulse_reward_scale"],
                label=f"{label}[{index}].pulse_reward_scale",
            )
            violation = _strict_bool_scalar(
                sample["routing_violation"],
                label=f"{label}[{index}].routing_violation",
            )
            token = _finite_scalar(
                sample["assertion_token"],
                label=f"{label}[{index}].assertion_token",
            )
            passed = (
                exact is True
                and island >= 0.0
                and off_gap >= 0.0
                and contribution == 0.0
                and scale == -1.0
                and violation is False
                and token == 0.0
            )
        else:
            boolean_expectations = {
                "decoder_exact": True,
                "leg_count_exact": True,
                "head_zero_exact": True,
                "teacher_target_contribution_zero_exact": True,
                "decoder_all_finite": True,
                "decoder_violation": False,
                "precomposer_call_count_exact": True,
                "final_guard_call_count_exact": True,
            }
            booleans = {
                name: _strict_bool_scalar(
                    sample[name], label=f"{label}[{index}].{name}"
                )
                for name in boolean_expectations
            }
            numeric = {
                name: _finite_scalar(
                    sample[name], label=f"{label}[{index}].{name}"
                )
                for name in set(expected_keys) - set(boolean_expectations)
            }
            passed = (
                booleans == boolean_expectations
                and numeric["max_abs_error"] == 0.0
                and numeric["leg_count"] >= 0.0
                and numeric["residual_authority_scale"] == 0.0
                and numeric["margin_saturation_count"] >= 0.0
                and numeric["action_clip_count"] >= 0.0
                and numeric["guard_lag_max_rad"] >= 0.0
                and numeric["precomposer_call_count"] >= 0.0
                and numeric["final_guard_call_count"] >= 0.0
                and numeric["assertion_token"] == 0.0
            )
        if not passed:
            raise RuntimeError(
                f"{label}[{index}] {expert} iteration-v6 runtime audit failed: "
                f"{dict(sample)!r}"
            )
    return {
        "expert": expert,
        "observed_step_count": len(samples),
        "compiled_invariant_assertion_passed": True,
        "passed": True,
    }


def reverse_iteration_v6_decoder_vector_sample_from_info(
    info: Mapping[str, Any],
) -> dict[str, Any]:
    missing = sorted(
        key
        for key in REVERSE_ITERATION_V6_DECODER_VECTOR_INFO_KEYS.values()
        if key not in info
    )
    if missing:
        raise RuntimeError(f"reverse iteration-v6 vector info is missing {missing}")
    return {
        name: info[key]
        for name, key in REVERSE_ITERATION_V6_DECODER_VECTOR_INFO_KEYS.items()
    }


def require_reverse_iteration_v6_decoder_vector_samples(
    samples: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Any]:
    """Require complete finite 14-wide decoder telemetry on every no-PPO step."""

    if not samples:
        raise RuntimeError(f"{label} did not observe reverse-v6 decoder vectors")
    for index, sample in enumerate(samples):
        if set(sample) != set(REVERSE_ITERATION_V6_DECODER_VECTOR_INFO_KEYS):
            raise RuntimeError(f"{label}[{index}] decoder vector field set drifted")
        arrays = {name: np.asarray(value) for name, value in sample.items()}
        if any(
            array.shape != (14,)
            or not np.issubdtype(array.dtype, np.number)
            or not np.all(np.isfinite(array))
            for array in arrays.values()
        ):
            raise RuntimeError(
                f"{label}[{index}] decoder vectors must be finite shape-(14,)"
            )
        if (
            not np.array_equal(arrays["raw_targets"][5:9], np.zeros(4))
            or not np.array_equal(arrays["margin_targets"][5:9], np.zeros(4))
        ):
            raise RuntimeError(
                f"{label}[{index}] decoder target head channels must be exact zero"
            )
    return {
        "action_shape": [14],
        "raw_targets_shape": [14],
        "margin_targets_shape": [14],
        "target_head_channels_exact_zero": True,
        "all_finite": True,
        "observed_step_count": len(samples),
        "passed": True,
    }


def require_iteration_v6_runtime_progress(
    rows: Sequence[Mapping[str, Any]],
    *,
    expert: str,
    wiring_only: bool = False,
    wiring_completion: Mapping[str, Any] | None = None,
    full_completion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Close exact v6 PPO evidence without accepting absent full episodes."""

    episode_keys = (
        FORWARD_ITERATION_V6_REWARD_ROUTING_EPISODE_KEYS
        if expert == "forward"
        else REVERSE_ITERATION_V6_DECODER_EPISODE_KEYS
        if expert == "reverse"
        else None
    )
    if episode_keys is None:
        raise ValueError(f"unsupported iteration-v6 expert: {expert!r}")
    if type(wiring_only) is not bool:
        raise RuntimeError("iteration-v6 wiring_only must be boolean")
    if wiring_only:
        if full_completion is not None:
            raise RuntimeError("iteration-v6 wiring cannot use full completion")
        completion = dict(wiring_completion or {})
        if (
            set(completion) != set(ITERATION_V6_WIRING_COMPLETION_REQUIREMENT)
            or completion != ITERATION_V6_WIRING_COMPLETION_REQUIREMENT
            or any(
                type(completion[key]) is not type(expected)
                for key, expected in ITERATION_V6_WIRING_COMPLETION_REQUIREMENT.items()
            )
        ):
            raise RuntimeError(
                f"{expert} iteration-v6 wiring completion evidence is not exact: "
                f"{wiring_completion!r}"
            )
    else:
        if wiring_completion is not None:
            raise RuntimeError("iteration-v6 full runtime cannot use wiring completion")
        completion = dict(full_completion or {})
        if (
            set(completion) != set(ITERATION_V6_FULL_COMPLETION_REQUIREMENT)
            or completion != ITERATION_V6_FULL_COMPLETION_REQUIREMENT
            or any(
                type(completion[key]) is not type(expected)
                for key, expected in ITERATION_V6_FULL_COMPLETION_REQUIREMENT.items()
            )
        ):
            raise RuntimeError(
                f"{expert} iteration-v6 full completion evidence is not exact: "
                f"{full_completion!r}"
            )

    episode_rows = [row for row in rows if "episode/length" in row]
    if not episode_rows and not wiring_only:
        raise RuntimeError(f"PPO produced no {expert} iteration-v6 episode rows")
    required = set(episode_keys.values())
    for index, row in enumerate(episode_rows):
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(
                f"{expert} iteration-v6 PPO row {index} is missing {missing}"
            )
        length = _finite_scalar(
            row["episode/length"], label=f"{expert} iteration-v6 row {index}.length"
        )
        if length <= 0.0:
            raise RuntimeError(
                f"{expert} iteration-v6 row {index} has invalid length {length!r}"
            )
        totals = {
            name: _finite_scalar(
                row[key], label=f"{expert} iteration-v6 row {index}.{name}"
            )
            for name, key in episode_keys.items()
        }
        if expert == "forward":
            passed = (
                totals["routing_exact"] == length
                and totals["island_loss"] >= 0.0
                and totals["off_gap_diagnostic_loss"] >= 0.0
                and totals["off_gap_reward_contribution"] == 0.0
                and totals["pulse_reward_scale"] == -length
                and totals["routing_violation"] == 0.0
                and totals["assertion_token"] == 0.0
            )
        else:
            passed = (
                totals["decoder_exact"] == length
                and totals["max_abs_error"] == 0.0
                and totals["leg_count"] >= 0.0
                and totals["leg_count_exact"] == length
                and totals["head_zero_exact"] == length
                and totals["teacher_target_contribution_zero_exact"] == length
                and totals["residual_authority_scale"] == 0.0
                and totals["decoder_all_finite"] == length
                and totals["margin_saturation_count"] >= 0.0
                and totals["action_clip_count"] >= 0.0
                and totals["guard_lag_max_rad"] >= 0.0
                and totals["precomposer_call_count"] >= 0.0
                and totals["precomposer_call_count_exact"] == length
                and totals["final_guard_call_count"] >= 0.0
                and totals["final_guard_call_count_exact"] == length
                and totals["decoder_violation"] == 0.0
                and totals["assertion_token"] == 0.0
            )
        if not passed:
            raise RuntimeError(
                f"{expert} iteration-v6 PPO row {index} audit failed: "
                f"length={length!r}, totals={totals!r}"
            )
    return {
        "audit_mode": (
            "WIRING_COMPILED_ASSERTION_NO_EPISODE_ROWS_ALLOWED"
            if wiring_only
            else "FULL_RUNTIME_EPISODE_ROWS_REQUIRED"
        ),
        "expert": expert,
        "observed_episode_metric_rows": len(episode_rows),
        "episode_metric_rows_exact_if_observed": True,
        **(
            ITERATION_V6_WIRING_COMPLETION_REQUIREMENT
            if wiring_only
            else ITERATION_V6_FULL_COMPLETION_REQUIREMENT
        ),
        "passed": True,
    }


ITERATION_MODE_BOOLEAN_FIELDS = (
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


def require_iteration_v6_artifact_cross_binding(
    resolved_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    expert: str,
) -> dict[str, bool]:
    """Require exact v6 authority/config/manifest/result identity before commit."""

    if expert == "forward":
        selected_flag = "forward_iteration_v6_contact_abort_island_only"
        authorization_key = (
            "forward_iteration_v6_contact_abort_island_only_authorization"
        )
        result_sha_key = (
            "forward_iteration_v6_contact_abort_island_only_authorization_sha256"
        )
        requirement_key = (
            "forward_iteration_v6_reward_routing_runtime_requirement"
        )
        runtime_key = "forward_iteration_v6_reward_routing_runtime"
        expected_requirement = FORWARD_ITERATION_V6_REWARD_ROUTING_RUNTIME_REQUIREMENT
        contract_keys = ("reward_routing_contract",)
    elif expert == "reverse":
        selected_flag = "reverse_iteration_v6_absolute_full_leg_targets"
        authorization_key = (
            "reverse_iteration_v6_absolute_full_leg_targets_authorization"
        )
        result_sha_key = (
            "reverse_iteration_v6_absolute_full_leg_targets_authorization_sha256"
        )
        requirement_key = "reverse_iteration_v6_decoder_runtime_requirement"
        runtime_key = "reverse_iteration_v6_decoder_runtime"
        expected_requirement = REVERSE_ITERATION_V6_DECODER_RUNTIME_REQUIREMENT
        contract_keys = (
            "action_parameterization_contract",
            "teacher_timing_contract",
        )
    else:
        raise ValueError(f"unsupported iteration-v6 expert: {expert!r}")
    payload = authorization.get("payload")
    if not isinstance(payload, Mapping):
        raise RuntimeError("iteration-v6 authorization payload is unavailable")
    expected_contract_id = authorization.get("contract_id")
    expected_sha = authorization.get("sha256")
    if not isinstance(expected_contract_id, str) or not isinstance(expected_sha, str):
        raise RuntimeError("iteration-v6 authorization identity is unavailable")
    expected_core_source = {
        "path": str(ALIGNMENT_MODULE_PATH.resolve()),
        "sha256": PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256,
    }
    artifacts = {
        "resolved_config": resolved_config,
        "manifest": manifest,
        "result": result,
    }
    for label, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise RuntimeError(f"iteration-v6 {label} must be one mapping")
        mode_values = {
            key: artifact.get(key) for key in ITERATION_MODE_BOOLEAN_FIELDS
        }
        if (
            any(type(value) is not bool for value in mode_values.values())
            or mode_values[selected_flag] is not True
            or any(
                value is not False
                for key, value in mode_values.items()
                if key != selected_flag
            )
        ):
            raise RuntimeError(
                f"iteration-v6 {label} ten-mode boolean binding drifted: "
                f"{mode_values!r}"
            )
        if artifact.get("authorized_iteration_v6_250k_contract_id") != expected_contract_id:
            raise RuntimeError(
                f"iteration-v6 {label} authorization contract ID drifted"
            )
        if artifact.get(requirement_key) != expected_requirement:
            raise RuntimeError(f"iteration-v6 {label} runtime requirement drifted")
        if artifact.get("iteration_v6_core_source") != expected_core_source:
            raise RuntimeError(f"iteration-v6 {label} core source binding drifted")
        for contract_key in contract_keys:
            if artifact.get(contract_key) != payload.get(contract_key):
                raise RuntimeError(
                    f"iteration-v6 {label} {contract_key} drifted"
                )
        if "iteration_v6_runtime" in artifact:
            raise RuntimeError(
                f"iteration-v6 {label} used a generic runtime evidence key"
            )
    execution_ids = {
        artifact.get("training_contract_id") for artifact in artifacts.values()
    }
    if len(execution_ids) != 1 or None in execution_ids:
        raise RuntimeError("iteration-v6 execution contract cross-binding drifted")
    for label in ("resolved_config", "manifest"):
        nested = artifacts[label].get(authorization_key)
        if (
            not isinstance(nested, Mapping)
            or nested.get("sha256") != expected_sha
            or nested.get("contract_id") != expected_contract_id
        ):
            raise RuntimeError(
                f"iteration-v6 {label} authorization SHA/ID binding drifted"
            )
    if result.get(result_sha_key) != expected_sha:
        raise RuntimeError("iteration-v6 result authorization SHA binding drifted")
    manifest_runtime = manifest.get(runtime_key)
    result_runtime = result.get(runtime_key)
    if (
        not isinstance(manifest_runtime, Mapping)
        or manifest_runtime != result_runtime
        or manifest_runtime.get("expert") != expert
        or manifest_runtime.get("passed") is not True
    ):
        raise RuntimeError("iteration-v6 manifest/result runtime binding drifted")
    return {
        "all_ten_iteration_mode_booleans_exact": True,
        "authorization_sha_and_contract_id_exact": True,
        "execution_contract_id_cross_bound": True,
        "runtime_requirement_cross_bound": True,
        "core_source_cross_bound": True,
        "authorization_contracts_cross_bound": True,
        "expert_runtime_evidence_cross_bound": True,
        "passed": True,
    }


def require_forward_v4_source_semantic_preflight(
    audit: Any,
    *,
    source_provenance: Mapping[str, Any],
    probe_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Close the once-only official-source semantic reference on dynamic6."""

    expected_fields = {
        "dynamic6_exact",
        "dynamic6_max_abs_error",
        "dynamic6_field_count",
        "derived_cfrc_int_exact",
        "derived_cfrc_int_max_abs_error",
        "derived_cfrc_ext_exact",
        "derived_cfrc_ext_max_abs_error",
    }
    if not hasattr(audit, "_fields") or set(audit._fields) != expected_fields:
        raise RuntimeError("forward-v4 source-semantic audit schema drifted")
    dynamic6_exact = _strict_bool_scalar(
        audit.dynamic6_exact, label="source_semantic.dynamic6_exact"
    )
    dynamic6_error = _finite_scalar(
        audit.dynamic6_max_abs_error,
        label="source_semantic.dynamic6_max_abs_error",
    )
    dynamic6_field_count = _finite_scalar(
        audit.dynamic6_field_count,
        label="source_semantic.dynamic6_field_count",
    )
    derived: dict[str, dict[str, Any]] = {}
    for field in ("cfrc_int", "cfrc_ext"):
        exact = _strict_bool_scalar(
            getattr(audit, f"derived_{field}_exact"),
            label=f"source_semantic.derived_{field}_exact",
        )
        error = _finite_scalar(
            getattr(audit, f"derived_{field}_max_abs_error"),
            label=f"source_semantic.derived_{field}_max_abs_error",
        )
        if error < 0.0 or exact is not (error == 0.0):
            raise RuntimeError(
                f"source_semantic derived {field} diagnostic is inconsistent"
            )
        derived[field] = {
            "exact": exact,
            "max_abs_error": error,
        }
    if (
        dynamic6_exact is not True
        or dynamic6_error != 0.0
        or dynamic6_field_count != 6.0
    ):
        raise RuntimeError(
            "forward-v4 source-semantic dynamic6 reference failed: "
            f"exact={dynamic6_exact!r}, max_abs_error={dynamic6_error!r}, "
            f"field_count={dynamic6_field_count!r}"
        )
    provenance_root = PurePosixPath(
        str(source_provenance.get("source_root", ""))
    )
    provenance_exact = provenance_root.is_absolute()
    for label, relative, expected_sha in (
        (
            "joystick",
            PINNED_FORWARD_V4_OFFICIAL_JOYSTICK_RELATIVE_PATH,
            PINNED_FORWARD_V4_OFFICIAL_JOYSTICK_SHA256,
        ),
        (
            "mjx_env",
            PINNED_FORWARD_V4_OFFICIAL_MJX_ENV_RELATIVE_PATH,
            PINNED_FORWARD_V4_OFFICIAL_MJX_ENV_SHA256,
        ),
    ):
        record = source_provenance.get(label)
        if not isinstance(record, Mapping):
            provenance_exact = False
            continue
        provenance_exact = bool(
            provenance_exact
            and set(record) == {"resolved_path", "relative_path", "sha256"}
            and record.get("relative_path") == relative
            and record.get("sha256") == expected_sha
            and PurePosixPath(str(record.get("resolved_path", "")))
            == provenance_root / PurePosixPath(relative)
        )
    if (
        set(source_provenance)
        != {
            "source_root",
            "joystick",
            "mjx_env",
            "step_source_sha256",
            "step_source_semantics",
            "all_files_under_requested_source_root",
            "passed",
        }
        or source_provenance.get("all_files_under_requested_source_root")
        is not True
        or source_provenance.get("passed") is not True
        or source_provenance.get("step_source_sha256")
        != PINNED_FORWARD_V4_OFFICIAL_STEP_SOURCE_SHA256
        or source_provenance.get("step_source_semantics")
        != FORWARD_V4_OFFICIAL_STEP_SOURCE_SEMANTICS
        or provenance_exact is not True
        or set(probe_input)
        != {
            "seed",
            "reset_noise_multiplier",
            "initial_state_source",
            "action_shape",
            "action_dtype",
            "action_all_zero",
        }
        or probe_input.get("seed") != 20260809
        or probe_input.get("reset_noise_multiplier") != 1.0
        or probe_input.get("initial_state_source")
        != "ENV_RESET_JAX_PRNGKEY_SEED"
        or probe_input.get("action_shape") != [14]
        or probe_input.get("action_dtype") != "float32"
        or probe_input.get("action_all_zero") is not True
    ):
        raise RuntimeError(
            "forward-v4 source-semantic provenance/probe contract drifted"
        )
    return {
        "timing": "ONCE_BEFORE_PPO_COLLECTION",
        "reference_source": "OFFICIAL_MJX_ENV_STEP_WRAPPER_NSUBSTEPS_10",
        "candidate_source": "SINGLE_INSTRUMENTED_TEN_SUBSTEP_SCAN_ENDPOINT",
        "source_provenance": dict(source_provenance),
        "probe_input": dict(probe_input),
        "qualifying_dynamic_state_fields": list(FORWARD_V4_DYNAMIC6_FIELDS),
        "dynamic6_exact": True,
        "dynamic6_max_abs_error": 0.0,
        "dynamic6_field_count": 6,
        "derived_diagnostics": {
            "qualification_role": "NON_QUALIFYING_OBSERVED_DIAGNOSTICS_ONLY",
            "fields": derived,
            "all_finite": True,
            "exclusion_is_semantic_not_tolerance": True,
            "numeric_tolerance_used": False,
        },
        "observed_reference_count": 1,
        "passed": True,
    }


def run_forward_v4_source_semantic_preflight(
    jax: Any,
    jp: Any,
    env: Any,
    probe_state: Any,
    *,
    source_physics_step: Any,
    mjx_step: Any,
    source_root: Path,
    joystick_module: Any,
    mjx_env_module: Any,
    seed: int,
    reset_noise_multiplier: float,
) -> dict[str, Any]:
    """Compile one official-vs-instrumented reference before PPO."""

    direct_control = jp.zeros_like(probe_state.data.ctrl)
    probe_input = {
        "seed": int(seed),
        "reset_noise_multiplier": float(reset_noise_multiplier),
        "initial_state_source": "ENV_RESET_JAX_PRNGKEY_SEED",
        "action_shape": list(direct_control.shape),
        "action_dtype": np.dtype(direct_control.dtype).name,
        "action_all_zero": bool(
            np.all(np.asarray(jax.device_get(direct_control)) == 0)
        ),
    }
    source_provenance = require_forward_v4_official_source_provenance(
        source_root=source_root,
        joystick_module=joystick_module,
        mjx_env_module=mjx_env_module,
    )

    def compiled_reference(initial_data, action):
        return audit_v4_source_semantic_reference(
            env.mjx_model,
            initial_data,
            action,
            source_physics_step=source_physics_step,
            mjx_step=mjx_step,
            scan=jax.lax.scan,
            xp=jp,
        )

    _candidate, _saved_dynamic6, audit = jax.jit(compiled_reference)(
        probe_state.data, direct_control
    )
    audit = jax.device_get(audit)
    return require_forward_v4_source_semantic_preflight(
        audit,
        source_provenance=source_provenance,
        probe_input=probe_input,
    )


def audit_jax_tree_placement(
    jax: Any, value: Any, *, expected_platform: str, label: str
) -> dict[str, Any]:
    """Require every concrete JAX array leaf to live on the training backend."""

    array_leaf_count = 0
    device_labels: set[str] = set()
    platforms: set[str] = set()
    for leaf in jax.tree_util.tree_leaves(value):
        leaf_devices_method = getattr(leaf, "devices", None)
        if not callable(leaf_devices_method):
            continue
        leaf_devices = tuple(leaf_devices_method())
        if not leaf_devices:
            raise RuntimeError(f"{label} contains a JAX array without a device")
        array_leaf_count += 1
        for device in leaf_devices:
            device_labels.add(str(device))
            platforms.add(str(device.platform))
    if array_leaf_count == 0 or platforms != {expected_platform}:
        raise RuntimeError(
            f"{label} JAX placement drifted: leaves={array_leaf_count}, "
            f"platforms={sorted(platforms)!r}, expected={expected_platform!r}"
        )
    return {
        "label": label,
        "jax_array_leaf_count": array_leaf_count,
        "platforms": sorted(platforms),
        "devices": sorted(device_labels),
        "expected_platform": expected_platform,
        "passed": True,
    }


def canonical_json_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json_strict(path: Path) -> Any:
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
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )

    def validate_numbers(value: Any, location: str = "$") -> None:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            if not np.isfinite(value):
                raise ValueError(f"non-finite JSON number at {location}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                validate_numbers(item, f"{location}[{index}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                validate_numbers(item, f"{location}.{key}")
            return
        raise ValueError(f"unsupported JSON value at {location}: {type(value).__name__}")

    validate_numbers(payload)
    return payload


def resolve_anchor_config(
    expert: str,
    *,
    physical_anchor_override: Sequence[float] | None = None,
    policy_anchor_override: Sequence[float] | None = None,
    forward_iteration_v2: bool = False,
    forward_iteration_v3_touchdown_balance: bool = False,
    forward_iteration_v4_contact_event_validity_persistence: bool = False,
    forward_v5_contact_pulse_abort_scale_only: bool = False,
    forward_iteration_v6_contact_abort_island_only: bool = False,
    reverse_iteration_v2: bool = False,
    reverse_iteration_v3_no_target_imitation: bool = False,
    reverse_iteration_v4_residual_transfer_gain_024: bool = False,
    reverse_iteration_v5_no_contact_imitation: bool = False,
    reverse_iteration_v6_absolute_full_leg_targets: bool = False,
) -> dict[str, Any]:
    if expert not in ANCHOR_CONFIGS:
        raise ValueError(f"H4 runner supports only {EXPERT_CHOICES}")
    if forward_iteration_v2 and expert != "forward":
        raise ValueError("forward iteration v2 is valid only for the forward expert")
    if forward_iteration_v3_touchdown_balance and expert != "forward":
        raise ValueError(
            "forward iteration v3 touchdown balance is valid only for the forward expert"
        )
    if forward_iteration_v4_contact_event_validity_persistence and expert != "forward":
        raise ValueError(
            "forward iteration v4 contact event validity/persistence is valid "
            "only for the forward expert"
        )
    if forward_v5_contact_pulse_abort_scale_only and expert != "forward":
        raise ValueError(
            "forward v5 contact pulse abort scale only is valid only for the "
            "forward expert"
        )
    if forward_iteration_v6_contact_abort_island_only and expert != "forward":
        raise ValueError(
            "forward iteration v6 contact abort island only is valid only for "
            "the forward expert"
        )
    if reverse_iteration_v2 and expert != "reverse":
        raise ValueError("reverse iteration v2 is valid only for the reverse expert")
    if reverse_iteration_v3_no_target_imitation and expert != "reverse":
        raise ValueError(
            "reverse iteration v3 no target imitation is valid only for the reverse expert"
        )
    if reverse_iteration_v4_residual_transfer_gain_024 and expert != "reverse":
        raise ValueError(
            "reverse iteration v4 residual transfer gain 0.24 is valid only "
            "for the reverse expert"
        )
    if reverse_iteration_v5_no_contact_imitation and expert != "reverse":
        raise ValueError(
            "reverse iteration v5 no contact imitation is valid only for the "
            "reverse expert"
        )
    if reverse_iteration_v6_absolute_full_leg_targets and expert != "reverse":
        raise ValueError(
            "reverse iteration v6 absolute full-leg targets is valid only for "
            "the reverse expert"
        )
    if sum(
        bool(value)
        for value in (
            forward_iteration_v2,
            forward_iteration_v3_touchdown_balance,
            forward_iteration_v4_contact_event_validity_persistence,
            forward_v5_contact_pulse_abort_scale_only,
            forward_iteration_v6_contact_abort_island_only,
            reverse_iteration_v2,
            reverse_iteration_v3_no_target_imitation,
            reverse_iteration_v4_residual_transfer_gain_024,
            reverse_iteration_v5_no_contact_imitation,
            reverse_iteration_v6_absolute_full_leg_targets,
        )
    ) > 1:
        raise ValueError("H4 iteration modes are mutually exclusive")
    result = dict(
        FORWARD_ITERATION_V2_ANCHOR_CONFIG
        if forward_iteration_v2
        or forward_iteration_v3_touchdown_balance
        or forward_iteration_v4_contact_event_validity_persistence
        or forward_v5_contact_pulse_abort_scale_only
        or forward_iteration_v6_contact_abort_island_only
        else REVERSE_ITERATION_V2_ANCHOR_CONFIG
        if reverse_iteration_v2
        or reverse_iteration_v3_no_target_imitation
        or reverse_iteration_v4_residual_transfer_gain_024
        or reverse_iteration_v5_no_contact_imitation
        or reverse_iteration_v6_absolute_full_leg_targets
        else ANCHOR_CONFIGS[expert]
    )
    for key, override in (
        ("physical_primary", physical_anchor_override),
        ("policy_observation_anchor", policy_anchor_override),
    ):
        if override is None:
            continue
        values = tuple(float(value) for value in override)
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"{key} override must be one finite triplet")
        result[key] = values
    physical = np.asarray(result["physical_primary"], dtype=np.float64)
    if expert == "forward" and not np.isclose(physical[0], 0.05):
        raise ValueError("H4 forward physical vx anchor must remain +0.05 m/s")
    if expert == "reverse" and not np.isclose(physical[0], -0.05):
        raise ValueError("H4 reverse physical vx anchor must remain -0.05 m/s")
    if not np.array_equal(physical[1:], np.zeros(2)):
        raise ValueError("straight H4 physical anchor must have zero vy/yaw")
    return result


def resolve_reward_scales(args: argparse.Namespace) -> H4QualityRewardScales:
    defaults = H4QualityRewardScales()
    h5_v3_substep_contact = bool(
        getattr(args, "h5_v3_substep_contact_alignment", False)
    )
    forward_v2 = bool(getattr(args, "forward_iteration_v2", False))
    forward_v3 = bool(
        getattr(args, "forward_iteration_v3_touchdown_balance", False)
    )
    forward_v4 = bool(
        getattr(
            args,
            "forward_iteration_v4_contact_event_validity_persistence",
            False,
        )
    )
    forward_v5 = bool(
        getattr(args, "forward_v5_contact_pulse_abort_scale_only", False)
    )
    forward_v6 = bool(
        getattr(args, "forward_iteration_v6_contact_abort_island_only", False)
    )
    reverse_v2 = bool(getattr(args, "reverse_iteration_v2", False))
    reverse_v3 = bool(
        getattr(args, "reverse_iteration_v3_no_target_imitation", False)
    )
    reverse_v4 = bool(
        getattr(args, "reverse_iteration_v4_residual_transfer_gain_024", False)
    )
    reverse_v5 = bool(
        getattr(args, "reverse_iteration_v5_no_contact_imitation", False)
    )
    reverse_v6 = bool(
        getattr(args, "reverse_iteration_v6_absolute_full_leg_targets", False)
    )
    if h5_v3_substep_contact:
        # This successor is deliberately a single-factor pilot.  Existing
        # reward CLI switches retain their parser defaults, while the three
        # newly measured strict-contact costs are the only nonzero delta.
        expected_cli = {
            "reward_force_slip": defaults.force_slip,
            "reward_left_force_slip": defaults.left_force_slip,
            "reward_right_force_slip": defaults.right_force_slip,
            "reward_per_foot_slip_tail": defaults.per_foot_slip_tail,
            "reward_per_foot_stance_slip_budget": defaults.per_foot_stance_slip_budget,
            "reward_single_support": defaults.single_support,
            "reward_single_support_band": defaults.single_support_band,
            "reward_alternation": defaults.alternation,
            "reward_load_balance": defaults.load_balance,
            "reward_touchdown_count_balance": defaults.touchdown_count_balance,
            "reward_flight": defaults.flight,
            "reward_total_normal_force_band": defaults.total_normal_force_band,
            "reward_total_normal_force_tail": defaults.total_normal_force_tail,
            "reward_contact_pulse_40ms": defaults.contact_pulse_40ms,
            "reward_slew_feasibility": defaults.slew_feasibility,
            "reward_target_lag": None,
            "reward_left_target_lag": None,
            "reward_right_target_lag": None,
            "reward_phase17_left_force_slip": defaults.phase17_left_force_slip,
            "reward_phase17_left_knee_envelope_excess": (
                defaults.phase17_left_knee_envelope_excess
            ),
            "reward_phase17_opposite_leg_lag": defaults.phase17_opposite_leg_lag,
            "reward_forward_cross_drift": defaults.forward_cross_drift,
            "reward_forward_uncommanded_yaw_rate": (
                defaults.forward_uncommanded_yaw_rate
            ),
            "reward_forward_heading_drift": defaults.forward_heading_drift,
            "reward_reverse_speed_boundary": None,
            "reward_reverse_cross_drift": None,
            "reward_reverse_uncommanded_yaw_rate": None,
            "reward_reverse_heading_drift": None,
            "reward_reverse_phase_force_slip": None,
            "reward_reverse_contact_priority_reversal_lag": None,
        }
        drifted = {
            name: getattr(args, name)
            for name, expected in expected_cli.items()
            if (
                getattr(args, name) is not None
                and expected is not None
                and not np.isclose(
                    float(getattr(args, name)), float(expected), rtol=0.0, atol=0.0
                )
            )
            or ((getattr(args, name) is None) != (expected is None))
        }
        if drifted:
            raise ValueError(
                "H5 V3 substep-contact reward scales are authorization-controlled; "
                f"CLI overrides are forbidden: {sorted(drifted)}"
            )
        return h5_v3_se2_substep_contact_reward_scales()
    if (
        forward_v2
        or forward_v3
        or forward_v4
        or forward_v5
        or forward_v6
        or reverse_v2
        or reverse_v3
        or reverse_v4
        or reverse_v5
        or reverse_v6
    ):
        baseline_cli = {
            "reward_force_slip": defaults.force_slip,
            "reward_left_force_slip": defaults.left_force_slip,
            "reward_right_force_slip": defaults.right_force_slip,
            "reward_per_foot_slip_tail": defaults.per_foot_slip_tail,
            "reward_per_foot_stance_slip_budget": defaults.per_foot_stance_slip_budget,
            "reward_single_support": defaults.single_support,
            "reward_single_support_band": defaults.single_support_band,
            "reward_alternation": defaults.alternation,
            "reward_load_balance": defaults.load_balance,
            "reward_touchdown_count_balance": defaults.touchdown_count_balance,
            "reward_flight": defaults.flight,
            "reward_total_normal_force_band": defaults.total_normal_force_band,
            "reward_total_normal_force_tail": defaults.total_normal_force_tail,
            "reward_contact_pulse_40ms": defaults.contact_pulse_40ms,
            "reward_slew_feasibility": defaults.slew_feasibility,
            "reward_target_lag": None,
            "reward_left_target_lag": None,
            "reward_right_target_lag": None,
            "reward_phase17_left_force_slip": defaults.phase17_left_force_slip,
            "reward_phase17_left_knee_envelope_excess": (
                defaults.phase17_left_knee_envelope_excess
            ),
            "reward_phase17_opposite_leg_lag": defaults.phase17_opposite_leg_lag,
            "reward_forward_cross_drift": defaults.forward_cross_drift,
            "reward_forward_uncommanded_yaw_rate": (
                defaults.forward_uncommanded_yaw_rate
            ),
            "reward_forward_heading_drift": defaults.forward_heading_drift,
            "reward_reverse_speed_boundary": None,
            "reward_reverse_cross_drift": None,
            "reward_reverse_uncommanded_yaw_rate": None,
            "reward_reverse_heading_drift": None,
            "reward_reverse_phase_force_slip": None,
            "reward_reverse_contact_priority_reversal_lag": None,
        }
        drifted = {
            name: getattr(args, name)
            for name, expected in baseline_cli.items()
            if (
                getattr(args, name) is not None
                and expected is not None
                and not np.isclose(
                    float(getattr(args, name)), float(expected), rtol=0.0, atol=0.0
                )
            )
            or ((getattr(args, name) is None) != (expected is None))
        }
        if drifted:
            raise ValueError(
                "iteration reward scales are authorization-controlled; "
                f"CLI overrides are forbidden: {sorted(drifted)}"
            )
        if forward_v3:
            return forward_iteration_v3_touchdown_balance_reward_scales()
        if forward_v5:
            return forward_iteration_v5_contact_pulse_abort_scale_only_reward_scales()
        if forward_v2 or forward_v4 or forward_v6:
            return forward_iteration_v2_reward_scales()
        return reverse_iteration_v2_reward_scales()
    target_lag = (
        args.reward_target_lag
        if args.reward_target_lag is not None
        else (
            0.0
            if args.expert in {"reverse", "unified"}
            else defaults.target_lag
        )
    )
    left_target_lag = (
        args.reward_left_target_lag
        if args.reward_left_target_lag is not None
        else (-0.125 if args.expert in {"reverse", "unified"} else 0.0)
    )
    right_target_lag = (
        args.reward_right_target_lag
        if args.reward_right_target_lag is not None
        else (-0.125 if args.expert in {"reverse", "unified"} else 0.0)
    )
    reverse_scale = lambda value, default: (  # noqa: E731
        value
        if value is not None
        else (default if args.expert in {"reverse", "unified"} else 0.0)
    )
    return H4QualityRewardScales(
        force_slip=args.reward_force_slip,
        left_force_slip=args.reward_left_force_slip,
        right_force_slip=args.reward_right_force_slip,
        per_foot_slip_tail=args.reward_per_foot_slip_tail,
        per_foot_stance_slip_budget=args.reward_per_foot_stance_slip_budget,
        single_support=args.reward_single_support,
        single_support_band=args.reward_single_support_band,
        alternation=args.reward_alternation,
        load_balance=args.reward_load_balance,
        touchdown_count_balance=args.reward_touchdown_count_balance,
        flight=args.reward_flight,
        total_normal_force_band=args.reward_total_normal_force_band,
        total_normal_force_tail=args.reward_total_normal_force_tail,
        contact_pulse_40ms=args.reward_contact_pulse_40ms,
        slew_feasibility=args.reward_slew_feasibility,
        target_lag=target_lag,
        left_target_lag=left_target_lag,
        right_target_lag=right_target_lag,
        phase17_left_force_slip=args.reward_phase17_left_force_slip,
        phase17_left_knee_envelope_excess=(
            args.reward_phase17_left_knee_envelope_excess
        ),
        phase17_opposite_leg_lag=args.reward_phase17_opposite_leg_lag,
        forward_cross_drift=args.reward_forward_cross_drift,
        forward_uncommanded_yaw_rate=(
            args.reward_forward_uncommanded_yaw_rate
        ),
        forward_heading_drift=args.reward_forward_heading_drift,
        reverse_speed_boundary=reverse_scale(
            args.reward_reverse_speed_boundary, -1.0
        ),
        reverse_cross_drift=reverse_scale(args.reward_reverse_cross_drift, -2.0),
        reverse_uncommanded_yaw_rate=reverse_scale(
            args.reward_reverse_uncommanded_yaw_rate, -1.0
        ),
        reverse_heading_drift=reverse_scale(
            args.reward_reverse_heading_drift, -1.0
        ),
        reverse_phase_force_slip=reverse_scale(
            args.reward_reverse_phase_force_slip, -1.0
        ),
        reverse_contact_priority_reversal_lag=reverse_scale(
            args.reward_reverse_contact_priority_reversal_lag, -0.75
        ),
    )


def h5_v3_se2_substep_contact_reward_scales() -> H4QualityRewardScales:
    """Return the one-factor H5 successor scale vector, fail-closed by caller."""

    return H4QualityRewardScales(
        target_lag=0.0,
        left_target_lag=-0.125,
        right_target_lag=-0.125,
        reverse_speed_boundary=-1.0,
        reverse_cross_drift=-2.0,
        reverse_uncommanded_yaw_rate=-1.0,
        reverse_heading_drift=-1.0,
        reverse_phase_force_slip=-1.0,
        reverse_contact_priority_reversal_lag=-0.75,
        h5_all_substep_strict20ms_slip_rms=-1.0,
        h5_all_substep_slip_tail=-1.0,
        h5_all_substep_force_tail=-1.0,
    )


def forward_iteration_v2_reward_scales() -> H4QualityRewardScales:
    """Return the independently audited, bounded iteration-v2 scales."""

    return H4QualityRewardScales(
        force_slip=-3.0,
        left_force_slip=-0.75,
        right_force_slip=-0.75,
        per_foot_slip_tail=-1.5,
        touchdown_count_balance=-2.0,
        total_normal_force_band=-1.0,
        total_normal_force_tail=-1.0,
        contact_pulse_40ms=-1.0,
        slew_feasibility=-0.5,
        target_lag=-0.5,
        phase17_left_force_slip=-2.0,
        phase17_left_knee_envelope_excess=-1.0,
        phase17_opposite_leg_lag=-1.5,
        forward_uncommanded_yaw_rate=-1.5,
        forward_heading_drift=-2.0,
    )


def forward_iteration_v3_touchdown_balance_reward_scales() -> H4QualityRewardScales:
    """Return v2 scales with the sole authorized touchdown-balance delta."""

    return replace(
        forward_iteration_v2_reward_scales(),
        touchdown_count_balance=-4.0,
    )


def forward_iteration_v5_contact_pulse_abort_scale_only_reward_scales(
) -> H4QualityRewardScales:
    """Return v4 scales with the sole authorized contact-pulse delta."""

    return replace(
        forward_iteration_v2_reward_scales(),
        contact_pulse_40ms=-2.0,
    )


def forward_iteration_v6_contact_abort_island_only_reward_scales(
) -> H4QualityRewardScales:
    """Return the exact v4 scale vector; v6 changes routing, not scale."""

    scales = forward_iteration_v2_reward_scales()
    if scales.contact_pulse_40ms != -1.0:
        raise RuntimeError("forward v6 contact-pulse scale must remain exactly -1")
    return scales


def reverse_iteration_v2_reward_scales() -> H4QualityRewardScales:
    """Return the independently audited reverse iteration-v2 scales."""

    return H4QualityRewardScales(
        per_foot_stance_slip_budget=-2.0,
        single_support=4.0,
        single_support_band=-4.0,
        alternation=6.0,
        target_lag=0.0,
        left_target_lag=-0.125,
        right_target_lag=-0.125,
        reverse_speed_boundary=-8.0,
        reverse_cross_drift=-2.0,
        reverse_uncommanded_yaw_rate=-1.0,
        reverse_heading_drift=-1.0,
        reverse_phase_force_slip=-1.0,
        reverse_contact_priority_reversal_lag=-0.75,
    )


def load_forward_minimum_spec(path: Path = DEFAULT_FORWARD_MINIMUM_SPEC) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing forward H4 minimum spec: {resolved}")
    actual_sha = sha256_file(resolved)
    payload = _load_json_strict(resolved)
    if actual_sha != PINNED_FORWARD_MINIMUM_SPEC_SHA256:
        raise ValueError("forward H4 minimum spec raw SHA drifted")
    if canonical_json_sha(payload) != PINNED_FORWARD_MINIMUM_SPEC_CANONICAL_SHA256:
        raise ValueError("forward H4 minimum spec canonical SHA drifted")
    if payload.get("artifact_kind") != "openduckmini_h4_forward_retraining_minimum_spec":
        raise ValueError("unexpected forward H4 minimum spec kind")
    if payload.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("forward H4 minimum spec must prohibit hardware")
    separation = payload["curriculum"]["command_separation_contract"]
    if separation["physical_anchor_mps_radps"] != [0.05, 0.0, 0.0]:
        raise ValueError("forward spec physical anchor drifted")
    if separation["policy_observation_anchor"] != [0.1, -0.018, -0.17]:
        raise ValueError("forward spec observation anchor drifted")
    observation_requirements = payload["training_runtime_contract"][
        "observation_requirements"
    ]
    if "left and right contact-point tangential speed" not in observation_requirements:
        raise ValueError("forward spec must require per-foot tangential speed")
    initialization = payload["v22_preserving_fine_tune"]["initialization"]
    declared_width = int(initialization["h4_actor_observation_width"])
    declared_extra = int(initialization["new_h4_observation_rows"])
    if (declared_width, declared_extra) != (116, 15):
        raise ValueError("forward spec must declare exact actor116/new15 compatibility")
    return {
        "path": resolved,
        "sha256": actual_sha,
        "payload": payload,
        "declared_actor_width": declared_width,
        "declared_extra_rows": declared_extra,
        "implementation_actor_width": H4_ACTOR_OBSERVATION_WIDTH,
        "implementation_extra_rows": (
            H4_ACTOR_OBSERVATION_WIDTH - LEGACY_ACTOR_OBSERVATION_WIDTH
        ),
        "stale_width_declaration_detected": False,
    }


def validate_forward_iteration_v2_authorization_payload(
    payload: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate the immutable bounded-v2 authorization semantics."""

    expected_authorization = {
        "simulation_250k_training": True,
        "simulation_1m_training": False,
        "candidate_adoption": False,
        "release": False,
        "hardware": False,
    }
    expected_training = {
        "initialization": "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT",
        "pinned_v22_parent_tree_sha256": PINNED_V22_PARENT_TREE_SHA256,
        "actor_observation_width": 116,
        "new_observation_rows": 15,
        "seed": 20260809,
        "num_timesteps": 250000,
        "num_envs": 1250,
        "learning_rate": 5.0e-5,
        "entropy_cost": 1.0e-3,
        "clipping_epsilon": 0.10,
        "discounting": 0.97,
        "max_grad_norm": 0.5,
        "same_optimizer_and_seed_as_iteration_v1": True,
        "h4_parent_checkpoint_allowed": False,
        "overwrite_allowed": False,
    }
    expected_curriculum = {
        "physical_primary_mps_radps": [0.05, 0.0, 0.0],
        "policy_observation_anchor": [0.10, -0.018, -0.170],
        "exact_primary_probability": 0.70,
        "local_probability": 0.20,
        "local_vx_m_s": [0.04, 0.06],
        "stand_probability": 0.05,
        "transition_probability": 0.05,
        "transition_vx_uniform_m_s": [0.025, 0.04],
        "probability_sum": 1.0,
    }
    expected_deltas = {
        "h4_force_slip": -1.0,
        "h4_left_force_slip": -0.25,
        "h4_right_force_slip": -0.25,
        "h4_per_foot_slip_tail": -0.5,
        "h4_touchdown_count_balance": -1.0,
        "h4_slew_feasibility": -0.25,
        "h4_target_lag": -0.25,
        "h4_phase17_left_force_slip": -1.0,
        "h4_phase17_left_knee_envelope_excess": -0.5,
        "h4_phase17_opposite_leg_lag": -0.75,
        "h4_forward_uncommanded_yaw_rate": -0.5,
        "h4_forward_heading_drift": -1.0,
    }
    new_losses = payload.get("new_measurement_aligned_losses", {})
    band = new_losses.get("total_normal_force_band", {})
    tail = new_losses.get("total_normal_force_tail", {})
    pulse = new_losses.get("contact_pulse_40ms", {})
    strict_gate = payload.get("strict_gate_contract", {})
    checks = {
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("artifact_kind")
        == "openduckmini_h4_forward_iteration_v2_authorization",
        "status": payload.get("status") == "AUTHORIZED_SIMULATION_250K_ONLY",
        "hardware_prohibited": payload.get("hardware_deployment") == "PROHIBITED",
        "authorization_exact": payload.get("authorization") == expected_authorization,
        "contract_id": payload.get("scope", {}).get("contract_id")
        == "H4_FORWARD_ITERATION_V2_250K_FROM_V22",
        "required_flag": payload.get("scope", {}).get("required_cli_flag")
        == "--forward-iteration-v2",
        "training_exact": payload.get("training_contract") == expected_training,
        "curriculum_exact": payload.get("curriculum") == expected_curriculum,
        "reward_scales_exact": payload.get("reward_contract", {}).get("exact_scales")
        == forward_iteration_v2_reward_scales().as_reward_scale_dict(),
        "reward_deltas_exact": payload.get("reward_contract", {}).get(
            "bounded_existing_scale_deltas_from_iteration_v1"
        )
        == expected_deltas,
        "force_band_exact": band.get("accepted_band_body_weight") == [0.8, 1.2]
        and band.get("band_width_body_weight") == 0.2
        and band.get("loss_formula")
        == "relu((0.8-F)/0.2)^2 + relu((F-1.2)/0.2)^2",
        "force_tail_exact": tail.get("strict_p99_boundary_body_weight") == 3.0
        and tail.get("loss_formula") == "relu((F-3.0)/3.0)^2",
        "contact_pulse_exact": pulse.get("control_period_s") == 0.02
        and pulse.get("minimum_contact_run_ticks") == 2
        and pulse.get("activation") == "per-foot liftoff only"
        and pulse.get("per_foot_loss_formula")
        == "liftoff*relu((2-contact_run_ticks)/2)^2"
        and pulse.get("aggregation")
        == "mean_over_liftoff_events_on_the_control_tick"
        and pulse.get("one_tick_contact_island_loss") == 0.25
        and pulse.get("two_or_more_tick_contact_island_loss") == 0.0
        and pulse.get("off_gap_direct_penalty") is False,
        "strict_gate_unchanged": strict_gate.get("thresholds_may_be_weakened") is False,
        "central_hashes_exact": (
            strict_gate.get("central_evaluator_sha256")
            == "31fb8846fc6267f28d032bca164dee2c872bfb484ebe272850100834bf1b1a9b"
            and strict_gate.get("central_gait_quality_sha256")
            == "b28e1ceb4cb6406411150bbad772a78203b8163bf10adcd79a5f31f83da5f2de"
            and strict_gate.get("central_routed_evaluation_sha256")
            == "f25ed858fbb5753fdcfd9e76f08396d0f09f95ac4696eb2e50cb5c128b80db09"
        ),
        "manifest_binding_exact": payload.get("manifest_binding")
        == {
            "authorization_artifact_sha256_required": True,
            "resolved_config_contract_id_required": True,
            "source_hash_snapshot_pre_and_post_required": True,
            "source_and_authorization_unchanged_required": True,
            "final_params_and_result_sha256_required": True,
        },
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"forward iteration-v2 authorization drifted: {failed}")
    return checks


def load_forward_iteration_v2_authorization(
    path: Path = DEFAULT_FORWARD_ITERATION_V2_AUTHORIZATION,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"missing forward iteration-v2 authorization: {resolved}"
        )
    actual_sha = sha256_file(resolved)
    if actual_sha != PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256:
        raise ValueError("forward iteration-v2 authorization SHA drifted")
    payload = _load_json_strict(resolved)
    checks = validate_forward_iteration_v2_authorization_payload(payload)

    causal = payload["causal_input"]
    evidence_path = (EXP_ROOT / causal["integrated_strict_evaluation"]["path"]).resolve()
    candidate_root = (
        EXP_ROOT
        / "artifacts"
        / "h4_training_runs"
        / "forward"
        / causal["failed_candidate_run_name"]
    ).resolve()
    bound_inputs = {
        "failed_candidate_params": (
            candidate_root / "final_params.pkl",
            causal["failed_candidate_final_params_sha256"],
        ),
        "failed_candidate_manifest": (
            candidate_root / "run_manifest.json",
            causal["failed_candidate_manifest_sha256"],
        ),
        "integrated_strict_evaluation": (
            evidence_path,
            causal["integrated_strict_evaluation"]["sha256"],
        ),
    }
    for label, (bound_path, expected_sha) in bound_inputs.items():
        if not bound_path.is_file() or sha256_file(bound_path) != expected_sha:
            raise ValueError(f"forward iteration-v2 causal input drifted: {label}")

    strict_gate = payload["strict_gate_contract"]
    central_expected = (
        strict_gate["central_evaluator_sha256"],
        strict_gate["central_gait_quality_sha256"],
        strict_gate["central_routed_evaluation_sha256"],
    )
    central_actual = tuple(sha256_file(path) for path in CENTRAL_QUALITY_PATHS)
    if central_actual != central_expected:
        raise ValueError("forward iteration-v2 central strict gate hashes drifted")
    return {
        "path": resolved,
        "sha256": actual_sha,
        "payload": payload,
        "semantic_audit": checks,
        "contract_id": payload["scope"]["contract_id"],
        "bound_causal_inputs": {
            label: {"path": str(bound_path), "sha256": expected_sha}
            for label, (bound_path, expected_sha) in bound_inputs.items()
        },
    }


def validate_forward_iteration_v3_touchdown_balance_authorization_payload(
    payload: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate the one-factor bounded forward-v3 authorization."""

    expected_authorization = {
        "simulation_250k_training": True,
        "simulation_1m_training": False,
        "candidate_adoption": False,
        "release": False,
        "hardware": False,
    }
    expected_training = {
        "initialization": "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT",
        "pinned_v22_parent_tree_sha256": PINNED_V22_PARENT_TREE_SHA256,
        "actor_observation_width": 116,
        "new_observation_rows": 15,
        "seed": 20260809,
        "num_timesteps": 250000,
        "num_envs": 1250,
        "learning_rate": 5.0e-5,
        "entropy_cost": 1.0e-3,
        "clipping_epsilon": 0.10,
        "discounting": 0.97,
        "max_grad_norm": 0.5,
        "reset_noise_multiplier": 1.0,
        "same_optimizer_and_seed_as_iteration_v2": True,
        "h4_parent_checkpoint_allowed": False,
        "overwrite_allowed": False,
    }
    expected_curriculum = {
        "physical_primary_mps_radps": [0.05, 0.0, 0.0],
        "policy_observation_anchor": [0.10, -0.018, -0.170],
        "exact_primary_probability": 0.70,
        "local_probability": 0.20,
        "local_vx_m_s": [0.04, 0.06],
        "stand_probability": 0.05,
        "transition_probability": 0.05,
        "transition_vx_uniform_m_s": [0.025, 0.04],
        "probability_sum": 1.0,
    }
    reward = payload.get("reward_contract", {})
    formula = reward.get("touchdown_count_balance_formula_unchanged", {})
    strict_gate = payload.get("strict_gate_contract", {})
    expected_scale_delta = {
        "name": "h4_touchdown_count_balance",
        "iteration_v2_scale": -2.0,
        "iteration_v3_scale": -4.0,
        "delta": -2.0,
    }
    expected_manifest_binding = {
        "authorization_artifact_sha256_required": True,
        "resolved_config_contract_id_required": True,
        "source_hash_snapshot_pre_and_post_required": True,
        "source_and_authorization_unchanged_required": True,
        "final_params_and_result_sha256_required": True,
    }
    v2_scales = forward_iteration_v2_reward_scales().as_reward_scale_dict()
    v3_scales = (
        forward_iteration_v3_touchdown_balance_reward_scales().as_reward_scale_dict()
    )
    changed_scale_names = {
        name for name in v2_scales if v2_scales[name] != v3_scales[name]
    }
    checks = {
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("artifact_kind")
        == "openduckmini_h4_forward_iteration_v3_touchdown_balance_authorization",
        "status": payload.get("status") == "AUTHORIZED_SIMULATION_250K_ONLY",
        "hardware_prohibited": payload.get("hardware_deployment") == "PROHIBITED",
        "authorization_exact": payload.get("authorization") == expected_authorization,
        "contract_id": payload.get("scope", {}).get("contract_id")
        == FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_CONTRACT_ID,
        "required_flag": payload.get("scope", {}).get("required_cli_flag")
        == "--forward-iteration-v3-touchdown-balance",
        "one_change_family": payload.get("scope", {}).get(
            "selected_change_family"
        )
        == "TOUCHDOWN_COUNT_BALANCE_SCALE_ONLY",
        "training_exact": payload.get("training_contract") == expected_training,
        "curriculum_exact": payload.get("curriculum") == expected_curriculum,
        "reward_scales_exact": reward.get("exact_scales") == v3_scales,
        "single_scale_delta_exact": reward.get(
            "only_scale_delta_from_iteration_v2"
        )
        == expected_scale_delta
        and changed_scale_names == {"h4_touchdown_count_balance"}
        and v3_scales["h4_touchdown_count_balance"] == -4.0,
        "touchdown_formula_unchanged": formula
        == {
            "contact_transition": (
                "touchdown_f_t = contact_f_t and not contact_f_t_minus_1"
            ),
            "count_update": "N_f_t = N_f_t_minus_1 + touchdown_f_t",
            "loss_formula": "relu(abs(N_left_t-N_right_t)-1)^2",
            "scale_applied_after_loss": -4.0,
            "formula_source_changed": False,
        },
        "strict_gate_unchanged": strict_gate.get("thresholds_may_be_weakened")
        is False
        and strict_gate.get("promotion_requires_all_three_fixed_six_second_seeds")
        is True,
        "central_hashes_exact": (
            strict_gate.get("central_evaluator_sha256")
            == "31fb8846fc6267f28d032bca164dee2c872bfb484ebe272850100834bf1b1a9b"
            and strict_gate.get("central_gait_quality_sha256")
            == "b28e1ceb4cb6406411150bbad772a78203b8163bf10adcd79a5f31f83da5f2de"
            and strict_gate.get("central_routed_evaluation_sha256")
            == "f25ed858fbb5753fdcfd9e76f08396d0f09f95ac4696eb2e50cb5c128b80db09"
        ),
        "manifest_binding_exact": payload.get("manifest_binding")
        == expected_manifest_binding,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(
            f"forward iteration-v3 touchdown-balance authorization drifted: {failed}"
        )
    return checks


def load_forward_iteration_v3_touchdown_balance_authorization(
    path: Path = DEFAULT_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_AUTHORIZATION,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            "missing forward iteration-v3 touchdown-balance authorization: "
            f"{resolved}"
        )
    actual_sha = sha256_file(resolved)
    if (
        actual_sha
        != PINNED_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_AUTHORIZATION_SHA256
    ):
        raise ValueError(
            "forward iteration-v3 touchdown-balance authorization SHA drifted"
        )
    payload = _load_json_strict(resolved)
    checks = validate_forward_iteration_v3_touchdown_balance_authorization_payload(
        payload
    )
    causal = payload["causal_input"]
    if causal.get("failed_candidate_root_relative_path") != (
        "artifacts/h4_iteration_v2_training_runs_20260809/forward"
    ):
        raise ValueError("forward iteration-v3 failed candidate root drifted")
    candidate_root = (
        EXP_ROOT
        / causal["failed_candidate_root_relative_path"]
        / causal["failed_candidate_run_name"]
    ).resolve()
    evidence_path = (EXP_ROOT / causal["integrated_strict_evaluation"]["path"]).resolve()
    bound_inputs = {
        "failed_candidate_params": (
            candidate_root / "final_params.pkl",
            causal["failed_candidate_final_params_sha256"],
        ),
        "failed_candidate_manifest": (
            candidate_root / "run_manifest.json",
            causal["failed_candidate_manifest_sha256"],
        ),
        "integrated_strict_evaluation": (
            evidence_path,
            causal["integrated_strict_evaluation"]["sha256"],
        ),
    }
    for label, (bound_path, expected_sha) in bound_inputs.items():
        if not bound_path.is_file() or sha256_file(bound_path) != expected_sha:
            raise ValueError(
                f"forward iteration-v3 touchdown-balance causal input drifted: {label}"
            )
    evidence = _load_json_strict(evidence_path)
    episodes = evidence.get("episodes")
    baseline = evidence.get("official_v22_baseline", {})
    expected_failures = set(
        causal["failed_measurement_aligned_checks_in_all_three_seeds"]
    )
    recorded_intersection = (
        set.intersection(
            *(
                set(episode.get("gait_quality_acceptance", {}).get("failures", []))
                for episode in episodes
            )
        )
        if isinstance(episodes, list) and len(episodes) == 3
        else set()
    )
    if (
        evidence.get("artifact_kind")
        != "openduckmini_h4_strict_promotion_evaluation"
        or evidence.get("candidate", {}).get("expert") != "forward"
        or evidence.get("candidate", {}).get("final_params_sha256")
        != causal["failed_candidate_final_params_sha256"]
        or evidence.get("candidate", {}).get("manifest_sha256")
        != causal["failed_candidate_manifest_sha256"]
        or evidence.get("evaluation_contract", {}).get("fixed_seeds")
        != list(H4_STRICT_PROMOTION_SEEDS["forward"])
        or not isinstance(episodes, list)
        or len(episodes) != 3
        or any(episode.get("h4_safety_acceptance", {}).get("passed") is not True for episode in episodes)
        or any(episode.get("gait_quality_acceptance", {}).get("passed") is not False for episode in episodes)
        or any(episode.get("strict_passed") is not False for episode in episodes)
        or recorded_intersection != expected_failures
        or evidence.get("summary", {}).get("passing_seed_count") != 0
        or evidence.get("summary", {}).get("recomputed_validation_passed") is not True
        or baseline.get("summary", {}).get("passing_seed_count") != 0
    ):
        raise ValueError(
            "forward iteration-v3 touchdown-balance causal evaluation drifted"
        )
    strict_gate = payload["strict_gate_contract"]
    central_expected = (
        strict_gate["central_evaluator_sha256"],
        strict_gate["central_gait_quality_sha256"],
        strict_gate["central_routed_evaluation_sha256"],
    )
    if tuple(sha256_file(item) for item in CENTRAL_QUALITY_PATHS) != central_expected:
        raise ValueError(
            "forward iteration-v3 touchdown-balance central strict gate hashes drifted"
        )
    return {
        "path": resolved,
        "sha256": actual_sha,
        "payload": payload,
        "semantic_audit": checks,
        "contract_id": payload["scope"]["contract_id"],
        "bound_causal_inputs": {
            label: {"path": str(bound_path), "sha256": expected_sha}
            for label, (bound_path, expected_sha) in bound_inputs.items()
        },
    }


def load_reverse_minimum_spec(path: Path = DEFAULT_REVERSE_MINIMUM_SPEC) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing reverse H4 minimum spec: {resolved}")
    actual_sha = sha256_file(resolved)
    if actual_sha != PINNED_REVERSE_MINIMUM_SPEC_SHA256:
        raise ValueError("reverse H4 minimum spec raw SHA drifted")
    payload = _load_json_strict(resolved)
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind")
        != "openduckmini_h4_reverse_retraining_minimum_spec"
        or payload.get("hardware_deployment") != "PROHIBITED"
        or payload.get("scope", {}).get("physical_command_mps_radps")
        != [-0.05, 0.0, 0.0]
    ):
        raise ValueError("reverse H4 minimum spec identity/command drifted")
    minimum = payload["minimum_retraining_specification"]
    curriculum = minimum["command_conditions"]["initial_curriculum"]
    if (
        curriculum["exact_endpoint_probability_minimum"] != 0.5
        or curriculum["required_discrete_anchors_mps"] != [-0.06, -0.05, -0.04]
        or curriculum["transition_band_mps"] != [-0.04, -0.025]
    ):
        raise ValueError("reverse H4 curriculum minimum drifted")
    reward = minimum["reward_terms"]
    if (
        reward["strict_boundary_normalized_tracking"]["cross_velocity_limit_mps"]
        != 0.01
        or reward["force_weighted_per_foot_slip"][
            "normal_force_activation_fraction_body_weight"
        ]
        != 0.01
        or reward["desired_target_slew_feasibility"]["limit_rad_per_s"]
        != 2.0
    ):
        raise ValueError("reverse H4 reward boundary contract drifted")
    priorities = minimum["phase_conditioned_priorities"]
    expected_phases = ([26, 0, 1], [10, 11, 12, 13, 14, 16], [18, 20, 21, 22])
    if tuple(item["phase_indices"] for item in priorities) != expected_phases:
        raise ValueError("reverse H4 phase-conditioned priorities drifted")
    return {"path": resolved, "sha256": actual_sha, "payload": payload}


def load_reverse_composition_authorization(
    path: Path = DEFAULT_REVERSE_COMPOSITION_AUTHORIZATION,
) -> dict[str, Any]:
    resolved = path.resolve()
    if sha256_file(resolved) != PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256:
        raise ValueError("reverse composition authorization SHA drifted")
    validator_spec = importlib.util.spec_from_file_location(
        "exp004_h4_reverse_composition_validator",
        REVERSE_COMPOSITION_VALIDATOR_PATH,
    )
    if validator_spec is None or validator_spec.loader is None:
        raise ImportError("cannot load reverse composition semantic validator")
    validator = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator)
    audit = validator.validate_contract(resolved)
    payload = _load_json_strict(resolved)
    authorization = payload["authorization"]
    if (
        authorization.get("simulation_250k_pilot_training") is not True
        or authorization.get("simulation_1m_training") is not False
        or authorization.get("hardware") is not False
        or payload["pinned_components"]["selected_teacher"]["sha256"]
        != PINNED_SELECTED_REVERSE_TEACHER_SHA256
        or payload["pinned_components"]["minimum_retraining_spec"]["sha256"]
        != PINNED_REVERSE_MINIMUM_SPEC_SHA256
    ):
        raise ValueError("reverse composition authorization semantics drifted")
    return {
        "path": resolved,
        "sha256": PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256,
        "payload": payload,
        "semantic_audit": audit,
    }


def validate_reverse_iteration_v2_authorization_payload(
    payload: Mapping[str, Any],
) -> dict[str, bool]:
    expected_training = {
        "initialization": "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT",
        "pinned_v22_parent_tree_sha256": PINNED_V22_PARENT_TREE_SHA256,
        "actor_observation_width": 116,
        "new_observation_rows": 15,
        "seed": 20260810,
        "num_timesteps": 250000,
        "num_envs": 1250,
        "learning_rate": 3.0e-5,
        "entropy_cost": 1.0e-3,
        "clipping_epsilon": 0.10,
        "discounting": 0.97,
        "max_grad_norm": 0.5,
        "h4_parent_checkpoint_allowed": False,
        "overwrite_allowed": False,
    }
    expected_teacher = {
        "selected_teacher_sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
        "cadence_hz": 1.5,
        "entry_phase_bins": 14.0,
        "phase_advance_bins_per_control": 1.62,
        "backward_residual_scale": 0.12,
        "target_guard_changed": False,
        "teacher_composition_changed": False,
        "reverse_minimum_spec_sha256": PINNED_REVERSE_MINIMUM_SPEC_SHA256,
        "reverse_composition_authorization_sha256": (
            PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
        ),
    }
    expected_curriculum = {
        "physical_primary_mps_radps": [-0.05, 0.0, 0.0],
        "policy_observation_anchor": [-0.05, 0.0, 0.0],
        "exact_primary_probability": 0.75,
        "local_probability": 0.15,
        "local_vx_m_s": [-0.06, -0.04],
        "stand_probability": 0.05,
        "transition_probability": 0.05,
        "transition_vx_uniform_m_s": [-0.04, -0.025],
        "probability_sum": 1.0,
    }
    strict_gate = payload.get("strict_gate_contract", {})
    legacy = payload.get("legacy_reward_config", {})
    integrated = payload.get("causal_input", {}).get(
        "integrated_strict_evaluation", {}
    )
    checks = {
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("artifact_kind")
        == "openduckmini_h4_reverse_iteration_v2_authorization",
        "status": payload.get("status") == "AUTHORIZED_SIMULATION_250K_ONLY",
        "hardware_prohibited": payload.get("hardware_deployment") == "PROHIBITED",
        "authorization_exact": payload.get("authorization")
        == {
            "simulation_250k_training": True,
            "simulation_1m_training": False,
            "candidate_adoption": False,
            "release": False,
            "hardware": False,
        },
        "contract_id": payload.get("scope", {}).get("contract_id")
        == "H4_REVERSE_ITERATION_V2_250K_FROM_V22",
        "required_flag": payload.get("scope", {}).get("required_cli_flag")
        == "--reverse-iteration-v2",
        "training_exact": payload.get("training_contract") == expected_training,
        "teacher_guard_exact": payload.get("teacher_and_guard_contract")
        == expected_teacher,
        "legacy_reward_exact": legacy.get("iteration_v2_exact")
        == dict(REVERSE_ITERATION_V2_LEGACY_REWARD_CONFIG),
        "tracking_sigma_truthful": legacy.get("audit_baseline", {}).get(
            "tracking_sigma"
        )
        == 0.01
        and "already 0.01" in legacy.get("source_audit_correction", ""),
        "legacy_schema3_causal_truth": (
            integrated.get("legacy_schema3_composition_trace_complete")
            is False
            and integrated.get("safety_trace_used_for_qualification") is False
            and integrated.get("strict_pass_count") == 0
            and integrated.get("fixed_seed_count") == 3
            and integrated.get("causal_basis")
            == "GAIT_QUALITY_0_OF_3_AND_STEADY_REVERSE_SPEED_ONLY"
            and "not safety qualification evidence"
            in integrated.get("qualification_note", "")
        ),
        "curriculum_exact": payload.get("curriculum") == expected_curriculum,
        "reward_scales_exact": payload.get("reward_contract", {}).get(
            "exact_scales"
        )
        == reverse_iteration_v2_reward_scales().as_reward_scale_dict(),
        "new_force_pulse_disabled": payload.get("reward_contract", {}).get(
            "new_force_and_pulse_scales_explicitly_disabled"
        )
        is True,
        "strict_gate_unchanged": strict_gate.get("thresholds_may_be_weakened") is False,
        "central_hashes_exact": (
            strict_gate.get("central_evaluator_sha256")
            == "31fb8846fc6267f28d032bca164dee2c872bfb484ebe272850100834bf1b1a9b"
            and strict_gate.get("central_gait_quality_sha256")
            == "b28e1ceb4cb6406411150bbad772a78203b8163bf10adcd79a5f31f83da5f2de"
            and strict_gate.get("central_routed_evaluation_sha256")
            == "f25ed858fbb5753fdcfd9e76f08396d0f09f95ac4696eb2e50cb5c128b80db09"
        ),
        "manifest_binding_exact": payload.get("manifest_binding")
        == {
            "authorization_artifact_sha256_required": True,
            "resolved_config_contract_id_required": True,
            "teacher_guard_legacy_reward_config_required": True,
            "source_hash_snapshot_pre_and_post_required": True,
            "source_and_authorization_unchanged_required": True,
            "final_params_and_result_sha256_required": True,
        },
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"reverse iteration-v2 authorization drifted: {failed}")
    return checks


def load_reverse_iteration_v2_authorization(
    path: Path = DEFAULT_REVERSE_ITERATION_V2_AUTHORIZATION,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"missing reverse iteration-v2 authorization: {resolved}"
        )
    actual_sha = sha256_file(resolved)
    if actual_sha != PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256:
        raise ValueError("reverse iteration-v2 authorization SHA drifted")
    payload = _load_json_strict(resolved)
    checks = validate_reverse_iteration_v2_authorization_payload(payload)

    causal = payload["causal_input"]
    evidence_path = (EXP_ROOT / causal["integrated_strict_evaluation"]["path"]).resolve()
    candidate_root = (
        EXP_ROOT
        / "artifacts"
        / "h4_training_runs"
        / "reverse"
        / causal["failed_candidate_run_name"]
    ).resolve()
    bound_inputs = {
        "failed_candidate_params": (
            candidate_root / "final_params.pkl",
            causal["failed_candidate_final_params_sha256"],
        ),
        "failed_candidate_manifest": (
            candidate_root / "run_manifest.json",
            causal["failed_candidate_manifest_sha256"],
        ),
        "integrated_strict_evaluation": (
            evidence_path,
            causal["integrated_strict_evaluation"]["sha256"],
        ),
    }
    for label, (bound_path, expected_sha) in bound_inputs.items():
        if not bound_path.is_file() or sha256_file(bound_path) != expected_sha:
            raise ValueError(f"reverse iteration-v2 causal input drifted: {label}")

    strict_gate = payload["strict_gate_contract"]
    central_expected = (
        strict_gate["central_evaluator_sha256"],
        strict_gate["central_gait_quality_sha256"],
        strict_gate["central_routed_evaluation_sha256"],
    )
    central_actual = tuple(sha256_file(path) for path in CENTRAL_QUALITY_PATHS)
    if central_actual != central_expected:
        raise ValueError("reverse iteration-v2 central strict gate hashes drifted")
    return {
        "path": resolved,
        "sha256": actual_sha,
        "payload": payload,
        "semantic_audit": checks,
        "contract_id": payload["scope"]["contract_id"],
        "bound_causal_inputs": {
            label: {"path": str(bound_path), "sha256": expected_sha}
            for label, (bound_path, expected_sha) in bound_inputs.items()
        },
    }


def validate_reverse_iteration_v3_no_target_imitation_authorization_payload(
    payload: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate the one-factor bounded reverse-v3 authorization."""

    expected_authorization = {
        "simulation_250k_training": True,
        "simulation_1m_training": False,
        "candidate_adoption": False,
        "release": False,
        "hardware": False,
    }
    expected_training = {
        "initialization": "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT",
        "pinned_v22_parent_tree_sha256": PINNED_V22_PARENT_TREE_SHA256,
        "actor_observation_width": 116,
        "new_observation_rows": 15,
        "seed": 20260810,
        "num_timesteps": 250000,
        "num_envs": 1250,
        "learning_rate": 3.0e-5,
        "entropy_cost": 1.0e-3,
        "clipping_epsilon": 0.10,
        "discounting": 0.97,
        "max_grad_norm": 0.5,
        "reset_noise_multiplier": 1.0,
        "same_optimizer_and_seed_as_iteration_v2": True,
        "h4_parent_checkpoint_allowed": False,
        "overwrite_allowed": False,
    }
    expected_teacher = {
        "selected_teacher_sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
        "cadence_hz": 1.5,
        "entry_phase_bins": 14.0,
        "phase_advance_bins_per_control": 1.62,
        "backward_residual_scale": 0.12,
        "target_guard_changed": False,
        "teacher_composition_changed": False,
        "reverse_minimum_spec_sha256": PINNED_REVERSE_MINIMUM_SPEC_SHA256,
        "reverse_composition_authorization_sha256": (
            PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
        ),
    }
    expected_curriculum = {
        "physical_primary_mps_radps": [-0.05, 0.0, 0.0],
        "policy_observation_anchor": [-0.05, 0.0, 0.0],
        "exact_primary_probability": 0.75,
        "local_probability": 0.15,
        "local_vx_m_s": [-0.06, -0.04],
        "stand_probability": 0.05,
        "transition_probability": 0.05,
        "transition_vx_uniform_m_s": [-0.04, -0.025],
        "probability_sum": 1.0,
    }
    expected_legacy = {
        "iteration_v2_exact": dict(REVERSE_ITERATION_V2_LEGACY_REWARD_CONFIG),
        "iteration_v3_exact": dict(
            REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG
        ),
        "only_scale_delta_from_iteration_v2": {
            "name": "target_imitation",
            "iteration_v2_scale": -20.0,
            "iteration_v3_scale": 0.0,
            "delta": 20.0,
        },
    }
    expected_diagnostic = {
        "formal_promotion_gate_unchanged": True,
        "comparison": "STRICT_IMPROVEMENT_OVER_ITERATION_V2_THREE_SEED_MEAN",
        "minimum_mean_steady_linear_tracking_ratio_exclusive": 0.0426695,
        "minimum_mean_single_support_rate_exclusive": 0.0094413,
        "minimum_mean_pure_endpoint_primary_ratio_exclusive": -0.0398896,
        "maximum_mean_cumulative_backtracking_m_exclusive": 0.088842,
        "diagnostic_does_not_authorize_promotion": True,
    }
    strict_gate = payload.get("strict_gate_contract", {})
    reward = payload.get("reward_contract", {})
    checks = {
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("artifact_kind")
        == "openduckmini_h4_reverse_iteration_v3_no_target_imitation_authorization",
        "status": payload.get("status") == "AUTHORIZED_SIMULATION_250K_ONLY",
        "hardware_prohibited": payload.get("hardware_deployment") == "PROHIBITED",
        "authorization_exact": payload.get("authorization") == expected_authorization,
        "contract_id": payload.get("scope", {}).get("contract_id")
        == REVERSE_ITERATION_V3_NO_TARGET_IMITATION_CONTRACT_ID,
        "required_flag": payload.get("scope", {}).get("required_cli_flag")
        == "--reverse-iteration-v3-no-target-imitation",
        "one_change_family": payload.get("scope", {}).get(
            "selected_change_family"
        )
        == "LEGACY_TARGET_IMITATION_SCALE_ONLY",
        "training_exact": payload.get("training_contract") == expected_training,
        "teacher_guard_exact": payload.get("teacher_and_guard_contract")
        == expected_teacher,
        "legacy_single_delta_exact": payload.get("legacy_reward_config")
        == expected_legacy,
        "curriculum_exact": payload.get("curriculum") == expected_curriculum,
        "h4_reward_scales_unchanged": reward.get("exact_scales")
        == reverse_iteration_v2_reward_scales().as_reward_scale_dict()
        and reward.get("identical_to_iteration_v2") is True
        and reward.get("new_force_and_pulse_scales_explicitly_disabled") is True,
        "directional_diagnostic_exact": payload.get("causal_input", {}).get(
            "directional_diagnostic_only"
        )
        == expected_diagnostic,
        "strict_gate_unchanged": strict_gate.get("thresholds_may_be_weakened")
        is False
        and strict_gate.get("promotion_requires_all_three_fixed_six_second_seeds")
        is True,
        "central_hashes_exact": (
            strict_gate.get("central_evaluator_sha256")
            == "31fb8846fc6267f28d032bca164dee2c872bfb484ebe272850100834bf1b1a9b"
            and strict_gate.get("central_gait_quality_sha256")
            == "b28e1ceb4cb6406411150bbad772a78203b8163bf10adcd79a5f31f83da5f2de"
            and strict_gate.get("central_routed_evaluation_sha256")
            == "f25ed858fbb5753fdcfd9e76f08396d0f09f95ac4696eb2e50cb5c128b80db09"
        ),
        "manifest_binding_exact": payload.get("manifest_binding")
        == {
            "authorization_artifact_sha256_required": True,
            "resolved_config_contract_id_required": True,
            "teacher_guard_legacy_reward_config_required": True,
            "source_hash_snapshot_pre_and_post_required": True,
            "source_and_authorization_unchanged_required": True,
            "final_params_and_result_sha256_required": True,
        },
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(
            f"reverse iteration-v3 no-target-imitation authorization drifted: {failed}"
        )
    return checks


def load_reverse_iteration_v3_no_target_imitation_authorization(
    path: Path = DEFAULT_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            "missing reverse iteration-v3 no-target-imitation authorization: "
            f"{resolved}"
        )
    actual_sha = sha256_file(resolved)
    if actual_sha != PINNED_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION_SHA256:
        raise ValueError(
            "reverse iteration-v3 no-target-imitation authorization SHA drifted"
        )
    payload = _load_json_strict(resolved)
    checks = validate_reverse_iteration_v3_no_target_imitation_authorization_payload(
        payload
    )
    causal = payload["causal_input"]
    if causal.get("failed_candidate_root_relative_path") != (
        "artifacts/h4_iteration_v2_training_runs_20260809/reverse"
    ):
        raise ValueError("reverse iteration-v3 failed candidate root drifted")
    candidate_root = (
        EXP_ROOT
        / causal["failed_candidate_root_relative_path"]
        / causal["failed_candidate_run_name"]
    ).resolve()
    evidence_path = (EXP_ROOT / causal["integrated_strict_evaluation"]["path"]).resolve()
    bound_inputs = {
        "failed_candidate_params": (
            candidate_root / "final_params.pkl",
            causal["failed_candidate_final_params_sha256"],
        ),
        "failed_candidate_manifest": (
            candidate_root / "run_manifest.json",
            causal["failed_candidate_manifest_sha256"],
        ),
        "integrated_strict_evaluation": (
            evidence_path,
            causal["integrated_strict_evaluation"]["sha256"],
        ),
    }
    for label, (bound_path, expected_sha) in bound_inputs.items():
        if not bound_path.is_file() or sha256_file(bound_path) != expected_sha:
            raise ValueError(
                f"reverse iteration-v3 no-target-imitation causal input drifted: {label}"
            )
    evidence = _load_json_strict(evidence_path)
    episodes = evidence.get("episodes")
    baseline = evidence.get("official_v22_baseline", {})
    baseline_episodes = baseline.get("episodes")
    expected_failures = set(
        causal["failed_measurement_aligned_checks_in_all_three_seeds"]
    )
    recorded_intersection = (
        set.intersection(
            *(
                set(episode.get("gait_quality_acceptance", {}).get("failures", []))
                for episode in episodes
            )
        )
        if isinstance(episodes, list) and len(episodes) == 3
        else set()
    )
    composition_contract = {
        "schema_version": 1,
        "semantics": (
            "PINNED_TEACHER_DELAYED_RESIDUAL_THEN_MARGIN_SLEW_PRECOMPOSER_"
            "THEN_FINAL_MARGIN_GUARD"
        ),
        "selected_reverse_teacher_sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
        "reverse_composition_authorization_sha256": (
            PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
        ),
        "teacher_table_rows": 54,
        "teacher_entry_phase_preincrement_bins": 14.0,
        "teacher_phase_advance_bins_per_control": 1.62,
        "source_period_bins": 27,
        "residual_scale": 0.12,
        "action_delay_min": 0,
        "action_delay_max_exclusive": 1,
        "step_entry_physical_command_x_mps": -0.05,
    }
    evaluation_hashes = evidence.get("runtime_provenance", {}).get(
        "evaluation_source_hashes_pre", {}
    )
    all_six = (
        list(episodes) + list(baseline_episodes)
        if isinstance(episodes, list)
        and len(episodes) == 3
        and isinstance(baseline_episodes, list)
        and len(baseline_episodes) == 3
        else []
    )
    causal_evaluation = causal["integrated_strict_evaluation"]
    if (
        evidence.get("artifact_kind")
        != "openduckmini_h4_strict_promotion_evaluation"
        or evidence.get("candidate", {}).get("expert") != "reverse"
        or evidence.get("candidate", {}).get("final_params_sha256")
        != causal["failed_candidate_final_params_sha256"]
        or evidence.get("candidate", {}).get("manifest_sha256")
        != causal["failed_candidate_manifest_sha256"]
        or evidence.get("evaluation_contract", {}).get("fixed_seeds")
        != list(H4_STRICT_PROMOTION_SEEDS["reverse"])
        or len(all_six) != 6
        or any(item.get("h4_safety_acceptance", {}).get("passed") is not True for item in all_six)
        or any(item.get("reverse_composition_contract") != composition_contract for item in all_six)
        or any(episode.get("gait_quality_acceptance", {}).get("passed") is not False for episode in episodes)
        or any(episode.get("strict_passed") is not False for episode in episodes)
        or recorded_intersection != expected_failures
        or evidence.get("summary", {}).get("passing_seed_count") != 0
        or evidence.get("summary", {}).get("recomputed_validation_passed") is not True
        or baseline.get("summary", {}).get("passing_seed_count") != 0
        or evaluation_hashes.get("safe_gait_experts/h4_post_training.py")
        != causal_evaluation["causal_h4_post_training_sha256"]
        or evaluation_hashes.get("scripts/evaluate_h4_training_candidate.py")
        != causal_evaluation["causal_h4_evaluator_sha256"]
    ):
        raise ValueError(
            "reverse iteration-v3 no-target-imitation causal evaluation drifted"
        )
    strict_gate = payload["strict_gate_contract"]
    central_expected = (
        strict_gate["central_evaluator_sha256"],
        strict_gate["central_gait_quality_sha256"],
        strict_gate["central_routed_evaluation_sha256"],
    )
    if tuple(sha256_file(item) for item in CENTRAL_QUALITY_PATHS) != central_expected:
        raise ValueError(
            "reverse iteration-v3 no-target-imitation central strict gate hashes drifted"
        )
    return {
        "path": resolved,
        "sha256": actual_sha,
        "payload": payload,
        "semantic_audit": checks,
        "contract_id": payload["scope"]["contract_id"],
        "bound_causal_inputs": {
            label: {"path": str(bound_path), "sha256": expected_sha}
            for label, (bound_path, expected_sha) in bound_inputs.items()
        },
    }


def _iteration_v4_spec(expert: str) -> dict[str, Any]:
    """Return the exact, bounded fourth-iteration contract for one expert."""

    if expert == "forward":
        return {
            "flag": "forward_iteration_v4_contact_event_validity_persistence",
            "auth_filename": (
                "h4_forward_iteration_v4_contact_event_validity_"
                "persistence_authorization.json"
            ),
            "auth_path": (
                DEFAULT_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_AUTHORIZATION
            ),
            "contract_id": (
                FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_CONTRACT_ID
            ),
            "wiring_contract_id": (
                FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_WIRING_CONTRACT_ID
            ),
            "required_flag": (
                "--forward-iteration-v4-contact-event-validity-persistence"
            ),
            "kind": (
                "openduckmini_h4_forward_iteration_v4_contact_event_"
                "validity_persistence_authorization"
            ),
            "change_family": (
                "CONTACT_EVENT_VALIDITY_PERSISTENCE_CORE_OPT_IN_WITH_V2_"
                "REWARD_BASELINE"
            ),
            "failed_root": (
                "artifacts/h4_iteration_v3_training_runs_20260809/forward"
            ),
            "failed_run": (
                "h4_forward_250k_seed20260809_iteration_v3_touchdown_"
                "balance_level4_v1"
            ),
            "params_sha": (
                "8946249b3531957166dc13005df7b2f25e50feefe03d78e9657e4724973e5dfa"
            ),
            "manifest_sha": (
                "4dfef12700363ae9274e1e8d9371a3780bf4871a1fe2a03d1e806749cc7deb92"
            ),
            "previous_auth_path": (
                "artifacts/h4_forward_iteration_v3_touchdown_balance_"
                "authorization.json"
            ),
            "previous_auth_sha": (
                PINNED_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_AUTHORIZATION_SHA256
            ),
            "strict_path": (
                "artifacts/h4_iteration_v3_training_runs_20260809/forward/"
                "h4_forward_250k_seed20260809_iteration_v3_touchdown_balance_"
                "level4_v1/h4_integrated_strict_3x6s_v1.json"
            ),
            "strict_sha": (
                "3375ad29f0443ac95637c1970b73f355a8ae2ee856903a0a43f79b8c7d74fd0f"
            ),
        }
    if expert == "reverse":
        return {
            "flag": "reverse_iteration_v4_residual_transfer_gain_024",
            "auth_filename": (
                "h4_reverse_iteration_v4_residual_transfer_gain_024_"
                "authorization.json"
            ),
            "auth_path": (
                DEFAULT_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_AUTHORIZATION
            ),
            "contract_id": REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_CONTRACT_ID,
            "wiring_contract_id": (
                REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_WIRING_CONTRACT_ID
            ),
            "required_flag": "--reverse-iteration-v4-residual-transfer-gain-024",
            "kind": (
                "openduckmini_h4_reverse_iteration_v4_residual_transfer_"
                "gain_024_authorization"
            ),
            "change_family": "BACKWARD_RESIDUAL_TRANSFER_GAIN_ONLY",
            "failed_root": (
                "artifacts/h4_iteration_v3_training_runs_20260809/reverse"
            ),
            "failed_run": (
                "h4_reverse_250k_seed20260810_iteration_v3_no_target_"
                "imitation_level4_v1"
            ),
            "params_sha": (
                "59871b9c35ea34ed3f62b8157d5afe8e2c8277cdc97e763c4a70dfafd8720414"
            ),
            "manifest_sha": (
                "a80801d81118ed557b8b32426307543cd0d298dbc9d57837a6517d8e4b66c67c"
            ),
            "previous_auth_path": (
                "artifacts/h4_reverse_iteration_v3_no_target_imitation_"
                "authorization.json"
            ),
            "previous_auth_sha": (
                PINNED_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION_SHA256
            ),
            "strict_path": (
                "artifacts/h4_iteration_v3_training_runs_20260809/reverse/"
                "h4_reverse_250k_seed20260810_iteration_v3_no_target_"
                "imitation_level4_v1/h4_integrated_strict_3x6s_v1.json"
            ),
            "strict_sha": (
                "a52054327ec6c65326f4a869260cc4dd55b3935fe7375cededd3551f8b56ece2"
            ),
        }
    raise ValueError(f"unsupported H4 iteration-v4 expert: {expert!r}")


def validate_iteration_v4_authorization_payload(
    payload: Mapping[str, Any], *, expert: str
) -> dict[str, bool]:
    """Fail closed on every semantic field of a v4 simulation authorization."""

    spec = _iteration_v4_spec(expert)
    expected_training = {
        "initialization": "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT",
        "pinned_v22_parent_tree_sha256": PINNED_V22_PARENT_TREE_SHA256,
        "actor_observation_width": 116,
        "new_observation_rows": 15,
        "seed": 20260809 if expert == "forward" else 20260810,
        "num_timesteps": 250000,
        "num_envs": 1250,
        "learning_rate": 5.0e-5 if expert == "forward" else 3.0e-5,
        "entropy_cost": 1.0e-3,
        "clipping_epsilon": 0.10,
        "discounting": 0.97,
        "max_grad_norm": 0.5,
        "reset_noise_multiplier": 1.0,
        "same_optimizer_and_seed_as_iteration_v2": True,
        "h4_parent_checkpoint_allowed": False,
        "overwrite_allowed": False,
    }
    expected_curriculum = (
        {
            "physical_primary_mps_radps": [0.05, 0.0, 0.0],
            "policy_observation_anchor": [0.10, -0.018, -0.170],
            "exact_primary_probability": 0.70,
            "local_probability": 0.20,
            "local_vx_m_s": [0.04, 0.06],
            "stand_probability": 0.05,
            "transition_probability": 0.05,
            "transition_vx_uniform_m_s": [0.025, 0.04],
            "probability_sum": 1.0,
        }
        if expert == "forward"
        else {
            "physical_primary_mps_radps": [-0.05, 0.0, 0.0],
            "policy_observation_anchor": [-0.05, 0.0, 0.0],
            "exact_primary_probability": 0.75,
            "local_probability": 0.15,
            "local_vx_m_s": [-0.06, -0.04],
            "stand_probability": 0.05,
            "transition_probability": 0.05,
            "transition_vx_uniform_m_s": [-0.04, -0.025],
            "probability_sum": 1.0,
        }
    )
    authorization = {
        "simulation_250k_training": True,
        "simulation_1m_training": False,
        "candidate_adoption": False,
        "release": False,
        "hardware": False,
    }
    causal = payload.get("causal_input", {})
    strict_record = causal.get("integrated_strict_evaluation", {})
    previous_record = causal.get("previous_iteration_authorization", {})
    strict_gate = payload.get("strict_gate_contract", {})
    closure = payload.get("causal_source_closure", {})
    expected_closure_paths = {
        "h4_training_alignment": "safe_gait_experts/h4_training_alignment.py",
        "h4_runner": "scripts/train_h4_aligned_expert.py",
        "h4_post_training": "safe_gait_experts/h4_post_training.py",
        "h4_candidate_evaluator": "scripts/evaluate_h4_training_candidate.py",
        "h4_no_ppo_smoke": "scripts/smoke_h4_training_alignment.py",
    }
    closure_exact = set(closure) == set(expected_closure_paths)
    if closure_exact:
        for label, relative in expected_closure_paths.items():
            record = closure[label]
            closure_exact = bool(
                isinstance(record, Mapping)
                and set(record) == {"path", "sha256"}
                and record.get("path") == relative
                and isinstance(record.get("sha256"), str)
                and len(record["sha256"]) == 64
                and all(char in "0123456789abcdef" for char in record["sha256"])
            )
            if not closure_exact:
                break
    checks = {
        "top_level_fields_exact": set(payload)
        == (
            {
                "schema_version",
                "artifact_kind",
                "status",
                "hardware_deployment",
                "authorization",
                "scope",
                "causal_input",
                "training_contract",
                "curriculum",
                "core_contract",
                "reward_contract",
                "strict_gate_contract",
                "manifest_binding",
                "decision",
                "causal_source_closure",
            }
            if expert == "forward"
            else {
                "schema_version",
                "artifact_kind",
                "status",
                "hardware_deployment",
                "authorization",
                "scope",
                "causal_input",
                "training_contract",
                "teacher_and_guard_contract",
                "legacy_reward_config",
                "curriculum",
                "reward_contract",
                "strict_gate_contract",
                "manifest_binding",
                "decision",
                "causal_source_closure",
            }
        ),
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("artifact_kind") == spec["kind"],
        "status": payload.get("status") == "AUTHORIZED_SIMULATION_250K_ONLY",
        "hardware_prohibited": payload.get("hardware_deployment") == "PROHIBITED",
        "authorization_exact": payload.get("authorization") == authorization,
        "contract_id": payload.get("scope", {}).get("contract_id")
        == spec["contract_id"],
        "required_flag": payload.get("scope", {}).get("required_cli_flag")
        == spec["required_flag"],
        "change_family": payload.get("scope", {}).get("selected_change_family")
        == spec["change_family"],
        "scope_fail_closed": (
            set(payload.get("scope", {}))
            == {
                "expert",
                "contract_id",
                "required_cli_flag",
                "purpose",
                "selected_change_family",
                "training_launch_performed_by_this_artifact",
            }
            and payload.get("scope", {}).get("expert") == expert
            and payload.get("scope", {}).get(
                "training_launch_performed_by_this_artifact"
            )
            is False
        ),
        "training_exact": payload.get("training_contract") == expected_training,
        "curriculum_exact": payload.get("curriculum") == expected_curriculum,
        "causal_identity_exact": (
            causal.get("failed_candidate_root_relative_path") == spec["failed_root"]
            and causal.get("failed_candidate_run_name") == spec["failed_run"]
            and causal.get("failed_candidate_final_params_sha256")
            == spec["params_sha"]
            and causal.get("failed_candidate_manifest_sha256")
            == spec["manifest_sha"]
            and previous_record
            == {
                "path": spec["previous_auth_path"],
                "sha256": spec["previous_auth_sha"],
            }
            and strict_record.get("path") == spec["strict_path"]
            and strict_record.get("sha256") == spec["strict_sha"]
            and strict_record.get("fixed_seed_count") == 3
            and strict_record.get("strict_pass_count") == 0
            and strict_record.get("safety_pass_count")
            == (2 if expert == "forward" else 3)
            and strict_record.get("official_v22_strict_pass_count") == 0
            and strict_record.get("recomputed_validation_passed") is True
        ),
        "source_closure_exact": closure_exact,
        "strict_gate_unchanged": (
            strict_gate.get("thresholds_may_be_weakened") is False
            and strict_gate.get(
                "promotion_requires_all_three_fixed_six_second_seeds"
            )
            is True
            and tuple(
                strict_gate.get(name)
                for name in (
                    "central_evaluator_sha256",
                    "central_gait_quality_sha256",
                    "central_routed_evaluation_sha256",
                )
            )
            == tuple(sha256_file(path) for path in CENTRAL_QUALITY_PATHS)
        ),
        "decision_fail_closed": payload.get("decision")
        == {
            "next_authorized_action": (
                "RUN_ONE_UNIQUE_SIMULATION_250K_AFTER_NO_PPO_WIRING_AND_"
                "PROVENANCE_AUDIT"
            ),
            "training_launch": "NOT_PERFORMED",
            "candidate_adoption": "BLOCKED",
            "release": "BLOCKED",
            "hardware": "PROHIBITED",
        },
    }
    if expert == "forward":
        reward = payload.get("reward_contract", {})
        checks.update(
            {
                "core_opt_in_exact": payload.get("core_contract")
                == {
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
                            "TEN_SINGLE_SUBSTEP_SCAN_EXACT_REPLACE_CTRL_"
                            "THEN_MJX_STEP"
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
                        "telemetry_source": (
                            "POST_PHYSICS_SAVED_DYNAMIC6_REPLAY"
                        ),
                        "measurement_state_coherence": {
                            "operation": (
                                "MJX_FORWARD_TELEMETRY_ONLY_AFTER_"
                                "INSTRUMENTED_PHYSICS_SCAN_COMPLETES"
                            ),
                            "reconstruction_base": (
                                "IMMUTABLE_CONTROL_ENTRY_DATA_PLUS_"
                                "SAVED_DYNAMIC6"
                            ),
                            "measurement_uses_forwarded_copy": True,
                            "instrumented_scan_carry_and_endpoint_use_unforwarded_integrated_state": True,
                        },
                    },
                    "source_semantic_theorem": {
                        "runtime_physics_authority_count": 1,
                        "official_source_wrapper_role": (
                            "ONCE_ONLY_PRE_PPO_REFERENCE"
                        ),
                        "official_source_wrapper_executed_inside_ppo": False,
                        "official_source_provenance": {
                            "both_files_must_resolve_under_requested_source_root": True,
                            "joystick_relative_path": (
                                PINNED_FORWARD_V4_OFFICIAL_JOYSTICK_RELATIVE_PATH
                            ),
                            "joystick_sha256": (
                                PINNED_FORWARD_V4_OFFICIAL_JOYSTICK_SHA256
                            ),
                            "mjx_env_relative_path": (
                                PINNED_FORWARD_V4_OFFICIAL_MJX_ENV_RELATIVE_PATH
                            ),
                            "mjx_env_sha256": (
                                PINNED_FORWARD_V4_OFFICIAL_MJX_ENV_SHA256
                            ),
                            "step_source_sha256": (
                                PINNED_FORWARD_V4_OFFICIAL_STEP_SOURCE_SHA256
                            ),
                            "step_source_semantics": (
                                FORWARD_V4_OFFICIAL_STEP_SOURCE_SEMANTICS
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
                        "qualifying_dynamic_state_fields": list(
                            FORWARD_V4_DYNAMIC6_FIELDS
                        ),
                        "qualifying_exact_required": True,
                        "qualifying_max_abs_error_required": 0.0,
                        "qualifying_field_count_required": 6,
                        "excluded_derived_diagnostics": [
                            "cfrc_int",
                            "cfrc_ext",
                        ],
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
                            "touchdown",
                            "touchdown_count",
                            "alternation",
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
                },
                "v2_reward_baseline_exact": (
                    reward.get("baseline") == "FORWARD_ITERATION_V2_EXACT"
                    and reward.get("exact_scales")
                    == forward_iteration_v2_reward_scales().as_reward_scale_dict()
                    and reward.get("touchdown_count_balance")
                    == {
                        "iteration_v3_scale": -4.0,
                        "iteration_v4_scale": -2.0,
                        "iteration_v4_matches_iteration_v2": True,
                    }
                    and reward.get("all_other_scales_match_iteration_v2") is True
                ),
                "hypothesis_only": causal.get("hypothesis")
                == {
                    "classification": "BOUNDED_CONTACT_EVENT_STATE_HYPOTHESIS_ONLY",
                    "statement": (
                        "substep-qualified contact state may prevent transient contact "
                        "samples from manufacturing touchdown events and may align event "
                        "rewards with the strict debounce measurements"
                    ),
                    "verified_by_existing_evidence": False,
                    "diagnostic_does_not_authorize_promotion": True,
                },
                "manifest_binding_exact": payload.get("manifest_binding")
                == {
                    "authorization_artifact_sha256_required": True,
                    "resolved_config_contract_id_required": True,
                    "core_opt_in_required": True,
                    "source_semantic_preflight_config_manifest_result_binding_required": True,
                    "single_authority_runtime_config_manifest_result_binding_required": True,
                    "source_hash_snapshot_pre_and_post_required": True,
                    "source_and_authorization_unchanged_required": True,
                    "final_params_and_result_sha256_required": True,
                },
            }
        )
    else:
        reward = payload.get("reward_contract", {})
        hypothesis = causal.get("hypothesis")
        checks.update(
            {
                "teacher_gain_single_delta_exact": payload.get(
                    "teacher_and_guard_contract"
                )
                == {
                    "selected_teacher_sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
                    "cadence_hz": 1.5,
                    "entry_phase_bins": 14.0,
                    "phase_advance_bins_per_control": 1.62,
                    "backward_residual_scale_iteration_v3": 0.12,
                    "backward_residual_scale_iteration_v4": 0.24,
                    "only_delta": "backward_residual_scale",
                    "target_guard_changed": False,
                    "teacher_table_or_phase_changed": False,
                    "reverse_minimum_spec_sha256": PINNED_REVERSE_MINIMUM_SPEC_SHA256,
                    "reverse_composition_authorization_sha256": (
                        PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
                    ),
                },
                "legacy_reward_retained_exact": payload.get(
                    "legacy_reward_config"
                )
                == {
                    "iteration_v3_exact": dict(
                        REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG
                    ),
                    "iteration_v4_exact": dict(
                        REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG
                    ),
                    "identical_to_iteration_v3": True,
                },
                "h4_reward_retained_exact": (
                    reward.get("exact_scales")
                    == reverse_iteration_v2_reward_scales().as_reward_scale_dict()
                    and reward.get("identical_to_iteration_v3") is True
                    and reward.get("new_force_and_pulse_scales_explicitly_disabled")
                    is True
                ),
                "bounded_exploration_no_saturation_claim": hypothesis
                == {
                    "classification": (
                        "BOUNDED_RESIDUAL_GAIN_EXPLORATION_HYPOTHESIS_ONLY"
                    ),
                    "statement": (
                        "raising the frozen-teacher residual transfer gain from 0.12 "
                        "to 0.24 may expose a larger bounded residual action range for "
                        "discovering propulsive reverse contact while the unchanged "
                        "final target guard remains authoritative"
                    ),
                    "verified_by_existing_evidence": False,
                    "action_saturation_observed_or_claimed": False,
                    "no_saturation_claim": True,
                    "diagnostic_does_not_authorize_promotion": True,
                },
                "manifest_binding_exact": payload.get("manifest_binding")
                == {
                    "authorization_artifact_sha256_required": True,
                    "resolved_config_contract_id_required": True,
                    "teacher_guard_legacy_reward_config_required": True,
                    "source_hash_snapshot_pre_and_post_required": True,
                    "source_and_authorization_unchanged_required": True,
                    "final_params_and_result_sha256_required": True,
                },
            }
        )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"{expert} iteration-v4 authorization drifted: {failed}")
    return checks


def load_iteration_v4_authorization(
    *, expert: str, path: Path | None = None
) -> dict[str, Any]:
    """Load v4 authority and bind its v3 evidence plus final source closure."""

    spec = _iteration_v4_spec(expert)
    resolved = Path(path or spec["auth_path"]).resolve()
    if not resolved.is_file() or resolved.name != spec["auth_filename"]:
        raise FileNotFoundError(f"missing {expert} iteration-v4 authorization: {resolved}")
    payload = _load_json_strict(resolved)
    checks = validate_iteration_v4_authorization_payload(payload, expert=expert)
    causal = payload["causal_input"]
    candidate_root = (EXP_ROOT / spec["failed_root"] / spec["failed_run"]).resolve()
    bound_inputs = {
        "previous_iteration_authorization": (
            (EXP_ROOT / spec["previous_auth_path"]).resolve(),
            spec["previous_auth_sha"],
        ),
        "failed_candidate_params": (
            candidate_root / "final_params.pkl",
            spec["params_sha"],
        ),
        "failed_candidate_manifest": (
            candidate_root / "run_manifest.json",
            spec["manifest_sha"],
        ),
        "integrated_strict_evaluation": (
            (EXP_ROOT / spec["strict_path"]).resolve(),
            spec["strict_sha"],
        ),
    }
    for label, (bound_path, expected_sha) in bound_inputs.items():
        if not bound_path.is_file() or sha256_file(bound_path) != expected_sha:
            raise ValueError(f"{expert} iteration-v4 causal input drifted: {label}")
    source_closure = {
        label: {
            "path": str((EXP_ROOT / record["path"]).resolve()),
            "sha256": record["sha256"],
        }
        for label, record in payload["causal_source_closure"].items()
    }
    for label, record in source_closure.items():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"{expert} iteration-v4 causal source drifted: {label}")
    evidence = _load_json_strict(bound_inputs["integrated_strict_evaluation"][0])
    episodes = evidence.get("episodes")
    baseline = evidence.get("official_v22_baseline", {})
    expected_candidate_safety_by_seed = (
        {
            20_260_809: True,
            20_261_809: True,
            20_262_809: False,
        }
        if expert == "forward"
        else {seed: True for seed in H4_STRICT_PROMOTION_SEEDS["reverse"]}
    )
    actual_candidate_safety_by_seed = (
        {
            episode.get("seed"): episode.get("h4_safety_acceptance", {}).get(
                "passed"
            )
            for episode in episodes
        }
        if isinstance(episodes, list)
        else {}
    )
    if (
        evidence.get("artifact_kind")
        != "openduckmini_h4_strict_promotion_evaluation"
        or evidence.get("candidate", {}).get("expert") != expert
        or evidence.get("candidate", {}).get("final_params_sha256")
        != causal["failed_candidate_final_params_sha256"]
        or evidence.get("candidate", {}).get("manifest_sha256")
        != causal["failed_candidate_manifest_sha256"]
        or evidence.get("evaluation_contract", {}).get("fixed_seeds")
        != list(H4_STRICT_PROMOTION_SEEDS[expert])
        or not isinstance(episodes, list)
        or len(episodes) != 3
        or actual_candidate_safety_by_seed != expected_candidate_safety_by_seed
        or any(
            episode.get("gait_quality_acceptance", {}).get("passed") is not False
            or episode.get("strict_passed") is not False
            for episode in episodes
        )
        or evidence.get("summary", {}).get("passing_seed_count") != 0
        or evidence.get("summary", {}).get("recomputed_validation_passed") is not True
        or baseline.get("summary", {}).get("passing_seed_count") != 0
    ):
        raise ValueError(f"{expert} iteration-v4 causal evaluation drifted")
    return {
        "path": resolved,
        "sha256": sha256_file(resolved),
        "payload": payload,
        "semantic_audit": checks,
        "contract_id": spec["contract_id"],
        "bound_causal_inputs": {
            label: {"path": str(bound_path), "sha256": expected_sha}
            for label, (bound_path, expected_sha) in bound_inputs.items()
        },
        "bound_causal_sources": source_closure,
    }


def load_forward_iteration_v4_contact_event_validity_persistence_authorization(
    path: Path = DEFAULT_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_AUTHORIZATION,
) -> dict[str, Any]:
    return load_iteration_v4_authorization(expert="forward", path=path)


def load_reverse_iteration_v4_residual_transfer_gain_024_authorization(
    path: Path = DEFAULT_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_AUTHORIZATION,
) -> dict[str, Any]:
    return load_iteration_v4_authorization(expert="reverse", path=path)


def _iteration_v5_spec(expert: str) -> dict[str, Any]:
    """Return the exact evidence and execution identity for one v5 family."""

    if expert == "forward":
        return {
            "flag": "forward_v5_contact_pulse_abort_scale_only",
            "required_flag": "--forward-v5-contact-pulse-abort-scale-only",
            "auth_path": DEFAULT_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_AUTHORIZATION,
            "auth_sha": (
                PINNED_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_AUTHORIZATION_SHA256
            ),
            "auth_filename": (
                "h4_forward_iteration_v5_contact_pulse_abort_scale_only_"
                "authorization.json"
            ),
            "kind": (
                "openduckmini_h4_forward_iteration_v5_contact_pulse_abort_"
                "scale_only_authorization"
            ),
            "family": "CONTACT_PULSE_ABORT_SCALE_ONLY",
            "purpose": (
                "test one bounded increase in the existing 40ms contact-pulse "
                "abort penalty after forward v4 remained strict-failing on all "
                "unchanged seeds"
            ),
            "contract_id": FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_CONTRACT_ID,
            "wiring_contract_id": (
                FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_WIRING_CONTRACT_ID
            ),
            "no_ppo_contract_id": (
                FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_NO_PPO_CONTRACT_ID
            ),
            "v4_auth_path": (
                "artifacts/h4_forward_iteration_v4_contact_event_validity_"
                "persistence_authorization.json"
            ),
            "v4_auth_sha": (
                "a808e329af37387466f9229dd587abf5fd90bcea08f1133295bd8551c3115a1e"
            ),
            "v4_root": (
                "artifacts/h4_iteration_v4_training_runs_20260809_bool_exact_v2/"
                "forward/h4_forward_250k_seed20260809_iteration_v4_contact_event_"
                "validity_persistence_level4_v2"
            ),
            "v4_hashes": {
                "final_params.pkl": "581c372c8373ddaa770bf51116a492363d47afbd6e8ffcf0896177558e569f8d",
                "run_manifest.json": "7bc72bfb340903e75a569210a27df7c0b018300cb045ee34ea68dadb6e0b85b0",
                "resolved_config.json": "2c3e224c1f2b893948fc6c40cd61fb93ec0b0284e3dd296a90fa0bcb4c56e0a2",
                "run_result.json": "f2220dccf29a7e4ffdbe72dd4935ad4840d7ef173abab87fe6c667215c138db6",
                "training_curve.csv": "881a87e3f333a77651c923ec13f90ed5ac2269bac0566ae0ecb64db20f29efcb",
                "h4_integrated_strict_3x6s_v1.json": "add95e7e2234ff469871d80a614821e660d6fb99ef6ac29847ed878c02268979",
            },
            "v4_source_prefix": "forward_iteration_v4_source_",
        }
    if expert == "reverse":
        return {
            "flag": "reverse_iteration_v5_no_contact_imitation",
            "required_flag": "--reverse-iteration-v5-no-contact-imitation",
            "auth_path": DEFAULT_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_AUTHORIZATION,
            "auth_sha": (
                PINNED_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_AUTHORIZATION_SHA256
            ),
            "auth_filename": "h4_reverse_iteration_v5_no_contact_imitation_authorization.json",
            "kind": (
                "openduckmini_h4_reverse_iteration_v5_no_contact_imitation_"
                "authorization"
            ),
            "family": "LEGACY_CONTACT_IMITATION_SCALE_ONLY",
            "purpose": (
                "test removal of legacy reverse contact imitation from the v3 "
                "residual-0.12 baseline after v3 and gain-0.24 v4 both remained "
                "non-propulsive"
            ),
            "contract_id": REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_CONTRACT_ID,
            "wiring_contract_id": (
                REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_WIRING_CONTRACT_ID
            ),
            "no_ppo_contract_id": (
                REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_NO_PPO_CONTRACT_ID
            ),
            "v4_auth_path": (
                "artifacts/h4_reverse_iteration_v4_residual_transfer_gain_024_"
                "authorization.json"
            ),
            "v4_auth_sha": (
                "93e3a53d5b601987df7a4efb84de5fb0ae499dc0ea0dc93acdbb074d96510312"
            ),
            "v4_root": (
                "artifacts/h4_iteration_v4_training_runs_20260809_bool_exact_v2/"
                "reverse/h4_reverse_250k_seed20260810_iteration_v4_residual_"
                "transfer_gain_024_level4_v1"
            ),
            "v4_hashes": {
                "final_params.pkl": "b0aa6fb639da3b8cc9e9bed27eee73f5a216e357e99ef030f5de07cd79d9e417",
                "run_manifest.json": "1ee678f682a8487181d7deaa3ada16e56df3e43b74a2f2f19e3e26bd31cac99c",
                "resolved_config.json": "0217cb992b53d25dbc9e7333cc6c9fbbb40e93693f7e11d891bd2b5ee2ca1ef0",
                "run_result.json": "5954054c4f85dae493fed2a04b49ed66ab259af58e7a1896eba32b3430c2a20c",
                "training_curve.csv": "b05f4fc04c0d23ebb7e7a8c335be0c754d72c089fa56b9b4948593df56ea2cda",
                "h4_integrated_strict_3x6s_v1.json": "7c4f45479891696e394e29d3b8272607b7c21801213ed612554d927b2958088d",
            },
            "v4_source_prefix": "reverse_iteration_v4_source_",
            "v3_auth_path": (
                "artifacts/h4_reverse_iteration_v3_no_target_imitation_"
                "authorization.json"
            ),
            "v3_auth_sha": PINNED_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION_SHA256,
            "v3_root": (
                "artifacts/h4_iteration_v3_training_runs_20260809/reverse/"
                "h4_reverse_250k_seed20260810_iteration_v3_no_target_"
                "imitation_level4_v1"
            ),
            "v3_hashes": {
                "final_params.pkl": "59871b9c35ea34ed3f62b8157d5afe8e2c8277cdc97e763c4a70dfafd8720414",
                "run_manifest.json": "a80801d81118ed557b8b32426307543cd0d298dbc9d57837a6517d8e4b66c67c",
                "h4_integrated_strict_3x6s_v1.json": "a52054327ec6c65326f4a869260cc4dd55b3935fe7375cededd3551f8b56ece2",
            },
            "adapter_path": (
                "scripts/evaluate_h4_training_candidate_reverse_v4_gain024_v1.py"
            ),
            "adapter_sha": (
                "4999fc8b48b5bc9e44df7dfd38ede97ee036dd20009289d128a3fe3d02eeb1f5"
            ),
            "adapter_auth_path": (
                "artifacts/h4_reverse_iteration_v4_gain024_strict_evaluator_"
                "adapter_v1_authorization.json"
            ),
            "adapter_auth_sha": (
                "f2531e5962818fdb3b6d9853447c288bb00c52917f6e72ad7b6a0ac5f4f085c0"
            ),
        }
    raise ValueError(f"unsupported H4 iteration-v5 expert: {expert!r}")


def validate_iteration_v5_authorization_payload(
    payload: Mapping[str, Any], *, expert: str
) -> dict[str, bool]:
    """Fail closed on the immutable one-family v5 authorization payload."""

    spec = _iteration_v5_spec(expert)
    expected_top = {
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
        "historical_v4_source_closure",
        "strict_gate_contract",
        "manifest_binding",
        "decision",
        "core_contract" if expert == "forward" else "teacher_and_guard_contract",
    }
    if expert == "reverse":
        expected_top.add("legacy_reward_config")
    expected_scope = {
        "expert": expert,
        "contract_id": spec["contract_id"],
        "wiring_contract_id": spec["wiring_contract_id"],
        "no_ppo_contract_id": spec["no_ppo_contract_id"],
        "required_cli_flag": spec["required_flag"],
        "selected_change_family": spec["family"],
        "purpose": spec["purpose"],
        "training_launch_performed_by_this_artifact": False,
    }
    expected_training = {
        "initialization": "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT",
        "pinned_v22_parent_tree_sha256": PINNED_V22_PARENT_TREE_SHA256,
        "actor_observation_width": 116,
        "new_observation_rows": 15,
        "seed": 20260809 if expert == "forward" else 20260810,
        "num_timesteps": 250000,
        "num_envs": 1250,
        "learning_rate": 5.0e-5 if expert == "forward" else 3.0e-5,
        "entropy_cost": 1.0e-3,
        "clipping_epsilon": 0.10,
        "discounting": 0.97,
        "max_grad_norm": 0.5,
        "reset_noise_multiplier": 1.0,
        "h4_parent_checkpoint_allowed": False,
        "overwrite_allowed": False,
    }
    expected_curriculum = (
        {
            "physical_primary_mps_radps": [0.05, 0.0, 0.0],
            "policy_observation_anchor": [0.10, -0.018, -0.170],
            "exact_primary_probability": 0.70,
            "local_probability": 0.20,
            "local_vx_m_s": [0.04, 0.06],
            "stand_probability": 0.05,
            "transition_probability": 0.05,
            "transition_vx_uniform_m_s": [0.025, 0.04],
            "probability_sum": 1.0,
        }
        if expert == "forward"
        else {
            "physical_primary_mps_radps": [-0.05, 0.0, 0.0],
            "policy_observation_anchor": [-0.05, 0.0, 0.0],
            "exact_primary_probability": 0.75,
            "local_probability": 0.15,
            "local_vx_m_s": [-0.06, -0.04],
            "stand_probability": 0.05,
            "transition_probability": 0.05,
            "transition_vx_uniform_m_s": [-0.04, -0.025],
            "probability_sum": 1.0,
        }
    )
    authorization = {
        "simulation_250k_training": True,
        "simulation_1m_training": False,
        "candidate_adoption": False,
        "release": False,
        "hardware": False,
    }
    if expert == "forward":
        expected_causal = {
            "previous_iteration_authorization": {
                "path": spec["v4_auth_path"],
                "sha256": spec["v4_auth_sha"],
            },
            "candidate_root_relative_path": spec["v4_root"],
            **{
                key.removesuffix(".pkl").removesuffix(".json").removesuffix(".csv")
                if key == "final_params.pkl"
                else {
                    "run_manifest.json": "manifest",
                    "resolved_config.json": "resolved_config",
                    "run_result.json": "run_result",
                    "training_curve.csv": "training_curve",
                }[key]: {
                    "path": key,
                    "sha256": spec["v4_hashes"][key],
                }
                for key in (
                    "final_params.pkl",
                    "run_manifest.json",
                    "resolved_config.json",
                    "run_result.json",
                    "training_curve.csv",
                )
            },
            "integrated_strict_evaluation": {
                "path": "h4_integrated_strict_3x6s_v1.json",
                "sha256": spec["v4_hashes"]["h4_integrated_strict_3x6s_v1.json"],
                "fixed_seed_count": 3,
                "strict_pass_count": 0,
                "safety_pass_count": 3,
                "gait_quality_pass_count": 0,
                "official_v22_strict_pass_count": 0,
                "recomputed_validation_passed": True,
            },
        }
    else:
        expected_causal = {
            "v3_authorization": {
                "path": spec["v3_auth_path"],
                "sha256": spec["v3_auth_sha"],
            },
            "v3_candidate_root_relative_path": spec["v3_root"],
            "v3_final_params": {
                "path": "final_params.pkl",
                "sha256": spec["v3_hashes"]["final_params.pkl"],
            },
            "v3_manifest": {
                "path": "run_manifest.json",
                "sha256": spec["v3_hashes"]["run_manifest.json"],
            },
            "v3_integrated_strict_evaluation": {
                "path": "h4_integrated_strict_3x6s_v1.json",
                "sha256": spec["v3_hashes"]["h4_integrated_strict_3x6s_v1.json"],
                "fixed_seed_count": 3,
                "strict_pass_count": 0,
                "safety_pass_count": 3,
                "gait_quality_pass_count": 0,
                "recomputed_validation_passed": True,
            },
            "rejected_v4_authorization": {
                "path": spec["v4_auth_path"],
                "sha256": spec["v4_auth_sha"],
            },
            "rejected_v4_candidate_root_relative_path": spec["v4_root"],
            "rejected_v4_final_params": {
                "path": "final_params.pkl",
                "sha256": spec["v4_hashes"]["final_params.pkl"],
            },
            "rejected_v4_manifest": {
                "path": "run_manifest.json",
                "sha256": spec["v4_hashes"]["run_manifest.json"],
            },
            "rejected_v4_resolved_config": {
                "path": "resolved_config.json",
                "sha256": spec["v4_hashes"]["resolved_config.json"],
            },
            "rejected_v4_run_result": {
                "path": "run_result.json",
                "sha256": spec["v4_hashes"]["run_result.json"],
            },
            "rejected_v4_training_curve": {
                "path": "training_curve.csv",
                "sha256": spec["v4_hashes"]["training_curve.csv"],
            },
            "rejected_v4_diagnostic": {
                "path": "h4_integrated_strict_3x6s_v1.json",
                "sha256": spec["v4_hashes"]["h4_integrated_strict_3x6s_v1.json"],
                "artifact_kind": (
                    "openduckmini_h4_reverse_iteration_v4_gain024_strict_"
                    "evaluation_diagnostic"
                ),
                "promotion_allowed": False,
                "strict_pass_count": 0,
                "safety_pass_count": 3,
                "gait_quality_pass_count": 0,
            },
            "diagnostic_adapter": {
                "path": spec["adapter_path"],
                "sha256": spec["adapter_sha"],
            },
            "diagnostic_adapter_authorization": {
                "path": spec["adapter_auth_path"],
                "sha256": spec["adapter_auth_sha"],
            },
        }
    closure = payload.get("historical_v4_source_closure", {})
    expected_closure = {
        "verification_source": "BOUND_V4_MANIFEST_PRE_POST_SNAPSHOT_NOT_CURRENT_FILES",
        "h4_training_alignment": {
            "path": "safe_gait_experts/h4_training_alignment.py",
            "sha256": "872a11a817bb068e3a0819c0afca12ae9e7f2dbfcc103c6569b9081b8d5fbebb",
        },
        "h4_runner": {
            "path": "scripts/train_h4_aligned_expert.py",
            "sha256": "b15b9692a72deadd34790d442f4ab4263c3f987255173566a62438e0d380da13",
        },
        "h4_post_training": {
            "path": "safe_gait_experts/h4_post_training.py",
            "sha256": "afdfcf9da43a7a7e5824ce7562c489b5e5e20a32e83af817be9e80d740a27b3f",
        },
        "h4_candidate_evaluator": {
            "path": "scripts/evaluate_h4_training_candidate.py",
            "sha256": "c214d086e6d66f6f9f98c7268481899e4133961dcc5355d738d4cd134a82e6ae",
        },
        "h4_no_ppo_smoke": {
            "path": "scripts/smoke_h4_training_alignment.py",
            "sha256": "410924542bac85f70de3a4055f617a85e93eb841cd403f5280699778ac96710d",
        },
    }
    expected_strict_gate = {
        "thresholds_may_be_weakened": False,
        "central_evaluator_sha256": sha256_file(CENTRAL_QUALITY_PATHS[0]),
        "central_gait_quality_sha256": sha256_file(CENTRAL_QUALITY_PATHS[1]),
        "central_routed_evaluation_sha256": sha256_file(CENTRAL_QUALITY_PATHS[2]),
        "promotion_requires_all_three_fixed_six_second_seeds": True,
    }
    expected_manifest_binding = {
        "authorization_artifact_required": True,
        "all_eight_iteration_mode_booleans_required": True,
        "historical_v4_sources_checked_against_manifest_snapshots": True,
        "current_source_snapshot_pre_and_post_required": True,
        "config_manifest_result_cross_binding_required": True,
        "final_params_result_curve_hashes_required": True,
        **(
            {"forward_v4_runtime_gates_required": True}
            if expert == "forward"
            else {
                "legacy_reward_and_residual_exact_required": True,
                "rejected_v4_diagnostic_never_promotion_evidence": True,
            }
        ),
    }
    expected_decision = {
        "next_authorized_action": (
            "RUN_NO_PPO_THEN_WIRING_THEN_ONE_UNIQUE_SIMULATION_250K_AFTER_"
            "INDEPENDENT_REVIEW"
        ),
        "training_launch": "NOT_PERFORMED",
        "candidate_adoption": "BLOCKED",
        "release": "BLOCKED",
        "hardware": "PROHIBITED",
    }
    checks = {
        "top_level_fields_exact": set(payload) == expected_top,
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("artifact_kind") == spec["kind"],
        "status": payload.get("status") == "AUTHORIZED_SIMULATION_250K_ONLY",
        "hardware_prohibited": payload.get("hardware_deployment") == "PROHIBITED",
        "authorization_exact": payload.get("authorization") == authorization,
        "scope_exact": payload.get("scope") == expected_scope,
        "training_exact": payload.get("training_contract") == expected_training,
        "curriculum_exact": payload.get("curriculum") == expected_curriculum,
        "historical_v4_source_closure_exact": closure == expected_closure,
        "strict_gate_unchanged": payload.get("strict_gate_contract")
        == expected_strict_gate,
        "manifest_binding_exact": payload.get("manifest_binding")
        == expected_manifest_binding,
        "decision_fail_closed": payload.get("decision") == expected_decision,
    }
    causal = payload.get("causal_inputs", {})
    if expert == "forward":
        expected_v5_scales = (
            forward_iteration_v5_contact_pulse_abort_scale_only_reward_scales()
            .as_reward_scale_dict()
        )
        checks.update(
            {
                "causal_identity_exact": (
                    causal == expected_causal
                ),
                "core_contract_exact": payload.get("core_contract")
                == {
                    "forward_v4_substep_contact": True,
                    "v4_source_semantic_preflight_required": True,
                    "v4_single_authority_runtime_required": True,
                    "v4_boolean_exact_and_full_curve_gates_required": True,
                    "core_source_changed": False,
                },
                "reward_single_delta_exact": (
                    payload.get("reward_contract", {}).get("baseline")
                    == "FORWARD_ITERATION_V4_EXACT"
                    and payload.get("reward_contract", {}).get("exact_scales")
                    == expected_v5_scales
                    and payload.get("reward_contract", {}).get("only_scale_delta")
                    == {
                        "name": "h4_contact_pulse_40ms",
                        "iteration_v4_scale": -1.0,
                        "iteration_v5_scale": -2.0,
                    }
                    and payload.get("reward_contract", {}).get(
                        "all_other_scales_match_iteration_v4"
                    )
                    is True
                ),
            }
        )
    else:
        checks.update(
            {
                "v3_causal_identity_exact": (
                    causal == expected_causal
                ),
                "rejected_v4_causal_identity_exact": (
                    causal == expected_causal
                ),
                "teacher_and_guard_exact": payload.get("teacher_and_guard_contract")
                == {
                    "selected_teacher_sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
                    "cadence_hz": 1.5,
                    "entry_phase_bins": 14.0,
                    "phase_advance_bins_per_control": 1.62,
                    "backward_residual_scale": 0.12,
                    "rejected_v4_backward_residual_scale": 0.24,
                    "v4_gain_inherited": False,
                    "target_guard_changed": False,
                    "teacher_table_or_phase_changed": False,
                    "reverse_minimum_spec_sha256": PINNED_REVERSE_MINIMUM_SPEC_SHA256,
                    "reverse_composition_authorization_sha256": (
                        PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
                    ),
                },
                "legacy_reward_single_delta_exact": payload.get("legacy_reward_config")
                == {
                    "iteration_v3_baseline": dict(
                        REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG
                    ),
                    "iteration_v5_exact": dict(
                        REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_LEGACY_REWARD_CONFIG
                    ),
                    "only_scale_delta": {
                        "name": "contact_imitation",
                        "iteration_v3_scale": 15.0,
                        "iteration_v5_scale": 0.0,
                    },
                },
                "h4_reward_unchanged": (
                    payload.get("reward_contract", {}).get("exact_scales")
                    == reverse_iteration_v2_reward_scales().as_reward_scale_dict()
                    and payload.get("reward_contract", {}).get(
                        "identical_to_iteration_v3"
                    )
                    is True
                    and payload.get("reward_contract", {}).get(
                        "forward_v5_reward_family_coupling"
                    )
                    is False
                ),
                "diagnostic_never_promotion": (
                    causal.get("rejected_v4_diagnostic", {}).get("artifact_kind")
                    == (
                        "openduckmini_h4_reverse_iteration_v4_gain024_strict_"
                        "evaluation_diagnostic"
                    )
                    and causal.get("rejected_v4_diagnostic", {}).get(
                        "promotion_allowed"
                    )
                    is False
                ),
            }
        )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"{expert} iteration-v5 authorization drifted: {failed}")
    return checks


def load_iteration_v5_authorization(
    *, expert: str, path: Path | None = None
) -> dict[str, Any]:
    """Load v5 authority and verify v4 history from its immutable run snapshot."""

    spec = _iteration_v5_spec(expert)
    resolved = Path(path or spec["auth_path"]).resolve()
    if not resolved.is_file() or resolved.name != spec["auth_filename"]:
        raise FileNotFoundError(f"missing {expert} iteration-v5 authorization: {resolved}")
    authorization_sha = sha256_file(resolved)
    if authorization_sha != spec["auth_sha"]:
        raise ValueError(
            f"{expert} iteration-v5 authorization SHA256 drifted: "
            f"expected={spec['auth_sha']}, actual={authorization_sha}"
        )
    payload = _load_json_strict(resolved)
    checks = validate_iteration_v5_authorization_payload(payload, expert=expert)
    causal = payload["causal_inputs"]
    v4_root = (EXP_ROOT / spec["v4_root"]).resolve()
    bound_inputs: dict[str, tuple[Path, str]] = {
        "v4_authorization": (
            (EXP_ROOT / spec["v4_auth_path"]).resolve(),
            spec["v4_auth_sha"],
        ),
        **{
            f"v4_{name}": (v4_root / name, expected_sha)
            for name, expected_sha in spec["v4_hashes"].items()
        },
    }
    if expert == "reverse":
        v3_root = (EXP_ROOT / spec["v3_root"]).resolve()
        bound_inputs.update(
            {
                "v3_authorization": (
                    (EXP_ROOT / spec["v3_auth_path"]).resolve(),
                    spec["v3_auth_sha"],
                ),
                **{
                    f"v3_{name}": (v3_root / name, expected_sha)
                    for name, expected_sha in spec["v3_hashes"].items()
                },
                "diagnostic_adapter": (
                    (EXP_ROOT / spec["adapter_path"]).resolve(),
                    spec["adapter_sha"],
                ),
                "diagnostic_adapter_authorization": (
                    (EXP_ROOT / spec["adapter_auth_path"]).resolve(),
                    spec["adapter_auth_sha"],
                ),
            }
        )
    for label, (bound_path, expected_sha) in bound_inputs.items():
        if not bound_path.is_file() or sha256_file(bound_path) != expected_sha:
            raise ValueError(f"{expert} iteration-v5 causal input drifted: {label}")

    v4_payload = _load_json_strict(bound_inputs["v4_authorization"][0])
    validate_iteration_v4_authorization_payload(v4_payload, expert=expert)
    v4_manifest = _load_json_strict(bound_inputs["v4_run_manifest.json"][0])
    historical = payload["historical_v4_source_closure"]
    bound_historical: dict[str, dict[str, Any]] = {}
    for label, auth_record in historical.items():
        if label == "verification_source":
            continue
        manifest_label = f"{spec['v4_source_prefix']}{label}"
        pre_record = v4_manifest.get("source_and_teacher_hashes_pre", {}).get(
            manifest_label
        )
        post_record = v4_manifest.get("source_and_teacher_hashes_post", {}).get(
            manifest_label
        )
        if (
            not isinstance(pre_record, Mapping)
            or not isinstance(post_record, Mapping)
            or pre_record.get("sha256") != auth_record["sha256"]
            or post_record.get("sha256") != auth_record["sha256"]
            or PurePosixPath(str(pre_record.get("path", "")).replace("\\", "/")).as_posix().endswith(
                auth_record["path"]
            )
            is not True
            or PurePosixPath(str(post_record.get("path", "")).replace("\\", "/")).as_posix().endswith(
                auth_record["path"]
            )
            is not True
        ):
            raise ValueError(
                f"{expert} iteration-v5 historical v4 source snapshot drifted: {label}"
            )
        bound_historical[label] = {
            "path": auth_record["path"],
            "sha256": auth_record["sha256"],
            "manifest_pre": dict(pre_record),
            "manifest_post": dict(post_record),
        }

    if expert == "forward":
        strict = _load_json_strict(bound_inputs["v4_h4_integrated_strict_3x6s_v1.json"][0])
        episodes = strict.get("episodes")
        if (
            strict.get("artifact_kind") != "openduckmini_h4_strict_promotion_evaluation"
            or strict.get("candidate", {}).get("expert") != "forward"
            or not isinstance(episodes, list)
            or len(episodes) != 3
            or sum(
                episode.get("h4_safety_acceptance", {}).get("passed") is True
                for episode in episodes
            )
            != 3
            or any(
                episode.get("gait_quality_acceptance", {}).get("passed") is not False
                or episode.get("strict_passed") is not False
                for episode in episodes
            )
            or strict.get("summary", {}).get("passing_seed_count") != 0
            or strict.get("summary", {}).get("recomputed_validation_passed") is not True
        ):
            raise ValueError("forward iteration-v5 strict causal evidence drifted")
    else:
        v3_strict = _load_json_strict(bound_inputs["v3_h4_integrated_strict_3x6s_v1.json"][0])
        diagnostic = _load_json_strict(
            bound_inputs["v4_h4_integrated_strict_3x6s_v1.json"][0]
        )
        if (
            v3_strict.get("artifact_kind")
            != "openduckmini_h4_strict_promotion_evaluation"
            or v3_strict.get("summary", {}).get("passing_seed_count") != 0
            or v3_strict.get("summary", {}).get("recomputed_validation_passed") is not True
            or diagnostic.get("artifact_kind")
            != (
                "openduckmini_h4_reverse_iteration_v4_gain024_strict_"
                "evaluation_diagnostic"
            )
            or diagnostic.get("promotion_allowed") is not False
            or diagnostic.get("summary", {}).get("passing_seed_count") != 0
            or diagnostic.get("summary", {}).get("recomputed_validation_passed") is not True
        ):
            raise ValueError("reverse iteration-v5 causal evidence drifted")
    return {
        "path": resolved,
        "sha256": authorization_sha,
        "payload": payload,
        "semantic_audit": checks,
        "contract_id": spec["contract_id"],
        "bound_causal_inputs": {
            label: {"path": str(bound_path), "sha256": expected_sha}
            for label, (bound_path, expected_sha) in bound_inputs.items()
        },
        "bound_historical_v4_sources": bound_historical,
    }


def load_forward_iteration_v5_contact_pulse_abort_scale_only_authorization(
    path: Path = DEFAULT_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_AUTHORIZATION,
) -> dict[str, Any]:
    return load_iteration_v5_authorization(expert="forward", path=path)


def load_reverse_iteration_v5_no_contact_imitation_authorization(
    path: Path = DEFAULT_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_AUTHORIZATION,
) -> dict[str, Any]:
    return load_iteration_v5_authorization(expert="reverse", path=path)


def _iteration_v6_spec(expert: str) -> dict[str, Any]:
    if expert == "forward":
        return {
            "auth_path": DEFAULT_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_AUTHORIZATION,
            "auth_filename": (
                "h4_forward_iteration_v6_contact_abort_island_only_authorization.json"
            ),
            "auth_sha": PINNED_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_AUTHORIZATION_SHA256,
            "artifact_kind": (
                "openduckmini_h4_forward_iteration_v6_contact_abort_island_only_"
                "authorization"
            ),
            "contract_id": FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_CONTRACT_ID,
            "wiring_contract_id": FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_WIRING_CONTRACT_ID,
            "no_ppo_contract_id": FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_NO_PPO_CONTRACT_ID,
            "flag": "--forward-iteration-v6-contact-abort-island-only",
            "family": "CONTACT_ABORT_TYPE_SEPARATION_ISLAND_ONLY",
            "purpose": (
                "route the unchanged minus-one contact-abort scale only through "
                "aborted contact-island loss while retaining off-gap loss as "
                "finite non-qualifying zero-reward telemetry"
            ),
            "seed": 20260809,
            "learning_rate": 5.0e-5,
        }
    if expert == "reverse":
        return {
            "auth_path": DEFAULT_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_AUTHORIZATION,
            "auth_filename": (
                "h4_reverse_iteration_v6_absolute_full_leg_targets_authorization.json"
            ),
            "auth_sha": PINNED_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_AUTHORIZATION_SHA256,
            "artifact_kind": (
                "openduckmini_h4_reverse_iteration_v6_absolute_full_leg_targets_"
                "authorization"
            ),
            "contract_id": REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_CONTRACT_ID,
            "wiring_contract_id": REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_WIRING_CONTRACT_ID,
            "no_ppo_contract_id": REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_NO_PPO_CONTRACT_ID,
            "flag": "--reverse-iteration-v6-absolute-full-leg-targets",
            "family": "ABSOLUTE_FULL_LEG_TARGETS_WITH_TEACHER_TIMING_ONLY",
            "purpose": (
                "replace teacher-centered residual actions with the frozen v22 "
                "calibrated absolute full-leg decoder while retaining the selected "
                "teacher only as a phase-timing prior"
            ),
            "seed": 20260810,
            "learning_rate": 3.0e-5,
        }
    raise ValueError(f"unsupported H4 iteration-v6 expert: {expert!r}")


def _iteration_v6_historical_source_closure() -> dict[str, Any]:
    return {
        "verification_source": "BOUND_V5_MANIFEST_PRE_POST_SNAPSHOT_NOT_CURRENT_FILES",
        "h4_training_alignment": {
            "path": "safe_gait_experts/h4_training_alignment.py",
            "sha256": "872a11a817bb068e3a0819c0afca12ae9e7f2dbfcc103c6569b9081b8d5fbebb",
        },
        "h4_runner": {
            "path": "scripts/train_h4_aligned_expert.py",
            "sha256": "2fb420d277fbf4d6c4887b8da76fbb804322cc19d3687ee176d7da94acc96d1b",
        },
        "h4_post_training": {
            "path": "safe_gait_experts/h4_post_training.py",
            "sha256": "d6429bebfe9f2afbdb4fd581d779d4a36e2a1b2232087377567db0b0ab8cd4b8",
        },
        "h4_candidate_evaluator": {
            "path": "scripts/evaluate_h4_training_candidate.py",
            "sha256": "c214d086e6d66f6f9f98c7268481899e4133961dcc5355d738d4cd134a82e6ae",
        },
        "h4_no_ppo_smoke": {
            "path": "scripts/smoke_h4_training_alignment.py",
            "sha256": "9d0014de5605b6c2b61f6e31cc01a1257cc05ce31b7da0a285ee513af05c219a",
        },
    }


def _iteration_v6_expected_causal_inputs(expert: str) -> dict[str, Any]:
    if expert == "forward":
        return {
            "iteration_v4_candidate_root_relative_path": (
                "artifacts/h4_iteration_v4_training_runs_20260809_bool_exact_v2/"
                "forward/h4_forward_250k_seed20260809_iteration_v4_contact_event_"
                "validity_persistence_level4_v2"
            ),
            "iteration_v4_final_params": {
                "path": "final_params.pkl",
                "sha256": "581c372c8373ddaa770bf51116a492363d47afbd6e8ffcf0896177558e569f8d",
            },
            "iteration_v4_manifest": {
                "path": "run_manifest.json",
                "sha256": "7bc72bfb340903e75a569210a27df7c0b018300cb045ee34ea68dadb6e0b85b0",
            },
            "iteration_v4_integrated_strict_evaluation": {
                "path": "h4_integrated_strict_3x6s_v1.json",
                "sha256": "add95e7e2234ff469871d80a614821e660d6fb99ef6ac29847ed878c02268979",
                "fixed_seed_count": 3,
                "strict_pass_count": 0,
                "safety_pass_count": 3,
                "gait_quality_pass_count": 0,
                "recomputed_validation_passed": True,
            },
            "rejected_iteration_v5_candidate_root_relative_path": (
                "artifacts/h4_iteration_v5_training_runs_20260809/forward/"
                "h4_forward_250k_seed20260809_iteration_v5_contact_pulse_abort_"
                "scale_only_level4_v1"
            ),
            "rejected_iteration_v5_final_params": {
                "path": "final_params.pkl",
                "sha256": "0b3102d32f334b34995f820c1dd123a9a0a5f8c55fa0370e4dd2ab17029947a1",
            },
            "rejected_iteration_v5_manifest": {
                "path": "run_manifest.json",
                "sha256": "93652d1331c41cb8fd5dc81e81e1a1d471750603e355ee7aa94a6dd017a6def2",
            },
            "rejected_iteration_v5_integrated_strict_evaluation": {
                "path": "h4_integrated_strict_3x6s_v1.json",
                "sha256": "26f0145e01cf5eb81a5f3313b93cd7513551fca89017bd501beceac243064669",
                "fixed_seed_count": 3,
                "strict_pass_count": 0,
                "safety_pass_count": 3,
                "gait_quality_pass_count": 0,
                "recomputed_validation_passed": True,
            },
        }
    if expert == "reverse":
        return {
            "iteration_v3_integrated_strict_evaluation": {
                "path": (
                    "artifacts/h4_iteration_v3_training_runs_20260809/reverse/"
                    "h4_reverse_250k_seed20260810_iteration_v3_no_target_"
                    "imitation_level4_v1/h4_integrated_strict_3x6s_v1.json"
                ),
                "sha256": "a52054327ec6c65326f4a869260cc4dd55b3935fe7375cededd3551f8b56ece2",
                "strict_pass_count": 0,
            },
            "rejected_iteration_v4_integrated_strict_evaluation": {
                "path": (
                    "artifacts/h4_iteration_v4_training_runs_20260809_bool_exact_v2/"
                    "reverse/h4_reverse_250k_seed20260810_iteration_v4_residual_"
                    "transfer_gain_024_level4_v1/h4_integrated_strict_3x6s_v1.json"
                ),
                "sha256": "7c4f45479891696e394e29d3b8272607b7c21801213ed612554d927b2958088d",
                "strict_pass_count": 0,
                "promotion_allowed": False,
            },
            "rejected_iteration_v4_diagnostic_adapter": {
                "path": "scripts/evaluate_h4_training_candidate_reverse_v4_gain024_v1.py",
                "sha256": "4999fc8b48b5bc9e44df7dfd38ede97ee036dd20009289d128a3fe3d02eeb1f5",
            },
            "rejected_iteration_v4_diagnostic_adapter_authorization": {
                "path": (
                    "artifacts/h4_reverse_iteration_v4_gain024_strict_evaluator_"
                    "adapter_v1_authorization.json"
                ),
                "sha256": "f2531e5962818fdb3b6d9853447c288bb00c52917f6e72ad7b6a0ac5f4f085c0",
            },
            "rejected_iteration_v5_candidate_root_relative_path": (
                "artifacts/h4_iteration_v5_training_runs_20260809/reverse/"
                "h4_reverse_250k_seed20260810_iteration_v5_no_contact_"
                "imitation_level4_v1"
            ),
            "rejected_iteration_v5_final_params": {
                "path": "final_params.pkl",
                "sha256": "a8e0764a7f4752d5b9aea15fd26d8c400e1d0808d756100345384915e3b5a625",
            },
            "rejected_iteration_v5_manifest": {
                "path": "run_manifest.json",
                "sha256": "9743ef34d2cf064ef40960a84322fb7a6524baa54eeae0575c52fa83738a2e6d",
            },
            "rejected_iteration_v5_integrated_strict_evaluation": {
                "path": "h4_integrated_strict_3x6s_v1.json",
                "sha256": "25eebbf7365da36a55840bd98d3afb747bd9892b72c9087e274c9ad15ee25b54",
                "strict_pass_count": 0,
            },
            "selected_reverse_teacher": {
                "path": "artifacts/h4_reverse_slew_feasible_teacher_selected_v1.json",
                "sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
            },
        }
    raise ValueError(f"unsupported H4 iteration-v6 expert: {expert!r}")


def _iteration_v6_expected_payload(expert: str) -> dict[str, Any]:
    spec = _iteration_v6_spec(expert)
    base = {
        "schema_version": 1,
        "artifact_kind": spec["artifact_kind"],
        "status": "AUTHORIZED_SIMULATION_250K_ONLY",
        "hardware_deployment": "PROHIBITED",
        "authorization": {
            "simulation_250k_training": True,
            "simulation_1m_training": False,
            "candidate_adoption": False,
            "release": False,
            "hardware": False,
        },
        "scope": {
            "expert": expert,
            "contract_id": spec["contract_id"],
            "wiring_contract_id": spec["wiring_contract_id"],
            "no_ppo_contract_id": spec["no_ppo_contract_id"],
            "required_cli_flag": spec["flag"],
            "selected_change_family": spec["family"],
            "purpose": spec["purpose"],
            "training_launch_performed_by_this_artifact": False,
        },
        "causal_inputs": _iteration_v6_expected_causal_inputs(expert),
        "training_contract": {
            "initialization": "FROZEN_V22_EXPLICIT_ACTOR116_TRANSPLANT",
            "pinned_v22_parent_tree_sha256": PINNED_V22_PARENT_TREE_SHA256,
            "actor_observation_width": 116,
            "new_observation_rows": 15,
            "seed": spec["seed"],
            "num_timesteps": 250000,
            "num_envs": 1250,
            "learning_rate": spec["learning_rate"],
            "entropy_cost": 1.0e-3,
            "clipping_epsilon": 0.10,
            "discounting": 0.97,
            "max_grad_norm": 0.5,
            "reset_noise_multiplier": 1.0,
            "h4_parent_checkpoint_allowed": False,
            "overwrite_allowed": False,
        },
        "curriculum": (
            {
                "physical_primary_mps_radps": [0.05, 0.0, 0.0],
                "policy_observation_anchor": [0.10, -0.018, -0.170],
                "exact_primary_probability": 0.70,
                "local_probability": 0.20,
                "local_vx_m_s": [0.04, 0.06],
                "stand_probability": 0.05,
                "transition_probability": 0.05,
                "transition_vx_uniform_m_s": [0.025, 0.04],
                "probability_sum": 1.0,
            }
            if expert == "forward"
            else {
                "physical_primary_mps_radps": [-0.05, 0.0, 0.0],
                "policy_observation_anchor": [-0.05, 0.0, 0.0],
                "exact_primary_probability": 0.75,
                "local_probability": 0.15,
                "local_vx_m_s": [-0.06, -0.04],
                "stand_probability": 0.05,
                "transition_probability": 0.05,
                "transition_vx_uniform_m_s": [-0.04, -0.025],
                "probability_sum": 1.0,
            }
        ),
        "historical_v5_source_closure": _iteration_v6_historical_source_closure(),
        "strict_gate_contract": {
            "thresholds_may_be_weakened": False,
            "central_evaluator_sha256": "31fb8846fc6267f28d032bca164dee2c872bfb484ebe272850100834bf1b1a9b",
            "central_gait_quality_sha256": "b28e1ceb4cb6406411150bbad772a78203b8163bf10adcd79a5f31f83da5f2de",
            "central_routed_evaluation_sha256": "f25ed858fbb5753fdcfd9e76f08396d0f09f95ac4696eb2e50cb5c128b80db09",
            "promotion_requires_all_three_fixed_six_second_seeds": True,
        },
        "decision": {
            "next_authorized_action": (
                "RUN_NO_PPO_THEN_WIRING_THEN_ONE_UNIQUE_SIMULATION_250K_AFTER_"
                "INDEPENDENT_REVIEW"
            ),
            "training_launch": "NOT_PERFORMED",
            "candidate_adoption": "BLOCKED",
            "release": "BLOCKED",
            "hardware": "PROHIBITED",
        },
    }
    if expert == "forward":
        base.update(
            {
                "core_contract": {
                    "forward_v4_substep_contact": True,
                    "v4_source_semantic_preflight_required": True,
                    "v4_single_authority_runtime_required": True,
                    "v4_boolean_exact_and_full_curve_gates_required": True,
                    "core_source_changed": True,
                },
                "reward_routing_contract": {
                    "source_scale_name": "h4_contact_pulse_40ms",
                    "source_scale_exact": -1.0,
                    "qualifying_loss": "aborted_contact_island_loss",
                    "qualifying_loss_scale": -1.0,
                    "off_gap_loss_retained_as_telemetry": True,
                    "off_gap_qualification_role": (
                        "NON_QUALIFYING_FINITE_DIAGNOSTICS_ONLY"
                    ),
                    "off_gap_reward_contribution": 0.0,
                    "legacy_aggregate_contact_pulse_routing_allowed": False,
                    "routing_violation_count_required": 0.0,
                    "assertion_token_sum_required": 0.0,
                    "fail_closed_before_output_commit": True,
                },
                "reward_contract": {
                    "baseline": "FORWARD_ITERATION_V4_EXACT",
                    "exact_scales": (
                        forward_iteration_v6_contact_abort_island_only_reward_scales()
                        .as_reward_scale_dict()
                    ),
                    "all_scales_match_iteration_v4": True,
                    "rejected_iteration_v5_minus_two_scale_inherited": False,
                },
                "manifest_binding": {
                    "authorization_artifact_required": True,
                    "all_ten_iteration_mode_booleans_required": True,
                    "historical_v5_sources_checked_against_manifest_snapshots": True,
                    "current_source_snapshot_pre_and_post_required": True,
                    "config_manifest_result_cross_binding_required": True,
                    "reward_routing_runtime_exact_required": True,
                    "forward_v4_authority_and_full_curve_gates_required": True,
                    "wiring_and_full_stage_gates_required": True,
                    "final_params_result_curve_hashes_required": True,
                },
            }
        )
    else:
        base.update(
            {
                "action_parameterization_contract": {
                    "decoder": "FROZEN_V22_CALIBRATED_ABSOLUTE_FULL_LEG",
                    "input_clip": [-1.0, 1.0],
                    "active_leg_indices": [0, 1, 2, 3, 4, 9, 10, 11, 12, 13],
                    "hard_zero_head_indices": [5, 6, 7, 8],
                    "safe_init_centered": True,
                    "directional_span_fraction": 0.9,
                    "near_zero_base_cap_rad": 0.25,
                    "nonlinear_exponent": 5,
                    "magnitude_formula": (
                        "min(0.25,span)*abs(a)+(span-min(0.25,span))*abs(a)^5"
                    ),
                    "target_formula": "SAFE_INIT+sign(a)*magnitude",
                    "inner_margin_then_slew_then_final_guard": True,
                    "per_control_slew_limit_rad": 0.04,
                    "residual_authority_scale": 0.0,
                    "teacher_target_contribution_zero": True,
                    "finite_exact_runtime_required": True,
                    "runtime_exact_boolean_metrics": [
                        "decoder_leg_count_exact",
                        "precomposer_call_count_exact",
                        "final_guard_call_count_exact",
                    ],
                    "raw_count_metrics": [
                        "decoder_leg_count",
                        "precomposer_call_count",
                        "final_guard_call_count",
                    ],
                    "raw_count_metrics_qualification_role": (
                        "NON_QUALIFYING_FINITE_DIAGNOSTICS_ONLY"
                    ),
                    "host_count_multiplication_for_qualification": False,
                    "numeric_tolerance_used": False,
                },
                "teacher_timing_contract": {
                    "selected_teacher_sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
                    "role": "PHASE_TIMING_PRIOR_ONLY",
                    "cadence_hz": 1.5,
                    "entry_phase_bins": 14.0,
                    "phase_advance_bins_per_control": 1.62,
                    "teacher_table_rows": 54,
                    "source_period_bins": 27,
                    "teacher_target_contribution": 0.0,
                    "teacher_imitation_reward_contribution": 0.0,
                    "target_guard_changed": False,
                },
                "legacy_reward_config": {
                    "iteration_v6_exact": dict(
                        REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_LEGACY_REWARD_CONFIG
                    ),
                    "backward_residual_scale": 0.0,
                    "identical_to_iteration_v5_except_residual_authority_removed": True,
                },
                "reward_contract": {
                    "exact_scales": reverse_iteration_v2_reward_scales().as_reward_scale_dict(),
                    "identical_to_iteration_v5": True,
                    "target_imitation": 0.0,
                    "contact_imitation": 0.0,
                    "teacher_timing_prior_reward": 0.0,
                },
                "manifest_binding": {
                    "authorization_artifact_required": True,
                    "all_ten_iteration_mode_booleans_required": True,
                    "historical_v5_sources_checked_against_manifest_snapshots": True,
                    "current_source_snapshot_pre_and_post_required": True,
                    "config_manifest_result_cross_binding_required": True,
                    "absolute_decoder_runtime_exact_required": True,
                    "teacher_timing_only_runtime_exact_required": True,
                    "legacy_reward_and_residual_zero_exact_required": True,
                    "wiring_and_full_stage_gates_required": True,
                    "final_params_result_curve_hashes_required": True,
                },
            }
        )
    return base


def _require_exact_json_value(actual: Any, expected: Any, *, location: str) -> None:
    """Compare JSON recursively with exact keys, types, and float bit values."""

    if type(actual) is not type(expected):
        raise ValueError(
            f"iteration-v6 authorization type drift at {location}: "
            f"{type(actual).__name__} != {type(expected).__name__}"
        )
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(
                f"iteration-v6 authorization schema drift at {location}: "
                f"actual={sorted(actual)}, expected={sorted(expected)}"
            )
        for key in expected:
            _require_exact_json_value(
                actual[key], expected[key], location=f"{location}.{key}"
            )
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"iteration-v6 authorization length drift at {location}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact_json_value(
                actual_item, expected_item, location=f"{location}[{index}]"
            )
        return
    if isinstance(expected, float):
        if not np.isfinite(actual) or float(actual).hex() != float(expected).hex():
            raise ValueError(
                f"iteration-v6 authorization float drift at {location}: "
                f"{actual!r} != {expected!r}"
            )
        return
    if actual != expected:
        raise ValueError(
            f"iteration-v6 authorization value drift at {location}: "
            f"{actual!r} != {expected!r}"
        )


def validate_iteration_v6_authorization_payload(
    payload: Mapping[str, Any], *, expert: str
) -> dict[str, bool]:
    expected = _iteration_v6_expected_payload(expert)
    if type(payload) is not dict:
        raise ValueError("iteration-v6 authorization payload must be one JSON object")
    _require_exact_json_value(payload, expected, location="$")
    return {
        "top_level_fields_exact": True,
        "nested_schema_exact": True,
        "numeric_types_and_values_exact": True,
        "causal_identity_exact": True,
        "training_contract_exact": True,
        "runtime_contract_exact": True,
        "passed": True,
    }


def _iteration_v6_causal_file_bindings(expert: str) -> dict[str, tuple[Path, str]]:
    causal = _iteration_v6_expected_causal_inputs(expert)
    if expert == "forward":
        v4_root = EXP_ROOT / causal["iteration_v4_candidate_root_relative_path"]
        v5_root = EXP_ROOT / causal["rejected_iteration_v5_candidate_root_relative_path"]
        records = {
            "iteration_v4_final_params": (v4_root, causal["iteration_v4_final_params"]),
            "iteration_v4_manifest": (v4_root, causal["iteration_v4_manifest"]),
            "iteration_v4_integrated_strict_evaluation": (
                v4_root,
                causal["iteration_v4_integrated_strict_evaluation"],
            ),
            "rejected_iteration_v5_final_params": (
                v5_root,
                causal["rejected_iteration_v5_final_params"],
            ),
            "rejected_iteration_v5_manifest": (
                v5_root,
                causal["rejected_iteration_v5_manifest"],
            ),
            "rejected_iteration_v5_integrated_strict_evaluation": (
                v5_root,
                causal["rejected_iteration_v5_integrated_strict_evaluation"],
            ),
        }
        return {
            label: ((root / record["path"]).resolve(), record["sha256"])
            for label, (root, record) in records.items()
        }
    records = {
        label: record
        for label, record in causal.items()
        if isinstance(record, dict) and "path" in record and "sha256" in record
    }
    v5_root = EXP_ROOT / causal["rejected_iteration_v5_candidate_root_relative_path"]
    return {
        label: (
            (
                v5_root / record["path"]
                if label.startswith("rejected_iteration_v5_")
                else EXP_ROOT / record["path"]
            ).resolve(),
            record["sha256"],
        )
        for label, record in records.items()
    }


def load_iteration_v6_authorization(
    *, expert: str, path: Path | None = None
) -> dict[str, Any]:
    """Load only byte-pinned v6 authority, then close semantics and history."""

    spec = _iteration_v6_spec(expert)
    resolved = Path(path or spec["auth_path"]).resolve()
    if not resolved.is_file() or resolved.name != spec["auth_filename"]:
        raise FileNotFoundError(f"missing {expert} iteration-v6 authorization: {resolved}")
    pinned_sha = str(spec["auth_sha"])
    if (
        len(pinned_sha) != 64
        or any(character not in "0123456789abcdef" for character in pinned_sha)
    ):
        raise RuntimeError(
            f"{expert} iteration-v6 authorization SHA256 pin is unresolved"
        )
    actual_sha = sha256_file(resolved)
    if actual_sha != pinned_sha:
        raise ValueError(
            f"{expert} iteration-v6 authorization SHA256 drifted: "
            f"expected={pinned_sha}, actual={actual_sha}"
        )
    payload = _load_json_strict(resolved)
    semantic_audit = validate_iteration_v6_authorization_payload(
        payload, expert=expert
    )
    bound_inputs = _iteration_v6_causal_file_bindings(expert)
    for label, (bound_path, expected_sha) in bound_inputs.items():
        if not bound_path.is_file() or sha256_file(bound_path) != expected_sha:
            raise ValueError(f"{expert} iteration-v6 causal input drifted: {label}")

    manifest_label = "rejected_iteration_v5_manifest"
    v5_manifest = _load_json_strict(bound_inputs[manifest_label][0])
    historical = payload["historical_v5_source_closure"]
    prefix = f"{expert}_iteration_v5_current_source_"
    bound_historical: dict[str, dict[str, Any]] = {}
    for label, record in historical.items():
        if label == "verification_source":
            continue
        pre_record = v5_manifest.get("source_and_teacher_hashes_pre", {}).get(
            f"{prefix}{label}"
        )
        post_record = v5_manifest.get("source_and_teacher_hashes_post", {}).get(
            f"{prefix}{label}"
        )
        if (
            not isinstance(pre_record, Mapping)
            or not isinstance(post_record, Mapping)
            or pre_record.get("sha256") != record["sha256"]
            or post_record.get("sha256") != record["sha256"]
            or not PurePosixPath(
                str(pre_record.get("path", "")).replace("\\", "/")
            ).as_posix().endswith(record["path"])
            or not PurePosixPath(
                str(post_record.get("path", "")).replace("\\", "/")
            ).as_posix().endswith(record["path"])
        ):
            raise ValueError(
                f"{expert} iteration-v6 historical v5 source snapshot drifted: {label}"
            )
        bound_historical[label] = {
            "path": record["path"],
            "sha256": record["sha256"],
            "manifest_pre": dict(pre_record),
            "manifest_post": dict(post_record),
        }
    return {
        "path": resolved,
        "sha256": actual_sha,
        "payload": payload,
        "semantic_audit": semantic_audit,
        "contract_id": spec["contract_id"],
        "bound_causal_inputs": {
            label: {"path": str(bound_path), "sha256": expected_sha}
            for label, (bound_path, expected_sha) in bound_inputs.items()
        },
        "bound_historical_v5_sources": bound_historical,
    }


def load_forward_iteration_v6_contact_abort_island_only_authorization(
    path: Path = DEFAULT_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_AUTHORIZATION,
) -> dict[str, Any]:
    return load_iteration_v6_authorization(expert="forward", path=path)


def load_reverse_iteration_v6_absolute_full_leg_targets_authorization(
    path: Path = DEFAULT_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_AUTHORIZATION,
) -> dict[str, Any]:
    return load_iteration_v6_authorization(expert="reverse", path=path)


def load_selected_reverse_teacher(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing selected reverse teacher: {resolved}")
    actual_sha = sha256_file(resolved)
    if actual_sha != PINNED_SELECTED_REVERSE_TEACHER_SHA256:
        raise ValueError(
            "selected reverse teacher hash drifted: "
            f"expected={PINNED_SELECTED_REVERSE_TEACHER_SHA256}, actual={actual_sha}"
        )
    payload = _load_json_strict(resolved)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported selected reverse teacher schema")
    if payload.get("artifact_kind") != (
        "openduckmini_h4_reverse_selected_training_teacher"
    ):
        raise ValueError("unexpected selected reverse teacher kind")
    if payload.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("reverse teacher must prohibit hardware deployment")
    decision = payload.get("decision", {})
    if decision.get("training_use") != "ALLOWED_AS_INITIALIZATION_PRIOR_ONLY":
        raise ValueError("reverse teacher is not approved as a training prior")
    if decision.get("adoption") is not False or decision.get("hardware") != "PROHIBITED":
        raise ValueError("reverse teacher adoption/hardware flags drifted")
    teacher = payload.get("teacher", {})
    validation = teacher.get("validation", {})
    if validation.get("passed") is not True or validation.get("failures") != []:
        raise ValueError("selected reverse teacher pure validation did not pass")
    if teacher.get("joint_order") != list(ACTUATOR_JOINT_ORDER):
        raise ValueError("selected reverse teacher actuator order drifted")
    table = np.asarray(teacher.get("target_table_rad"), dtype=np.float64)
    adapter = payload.get("adapter_contract", {})
    phase_steps = int(adapter.get("phase_steps", -1))
    if table.shape != (phase_steps, len(ACTUATOR_JOINT_ORDER)):
        raise ValueError("selected reverse teacher table shape drifted")
    if not np.all(np.isfinite(table)) or not np.array_equal(
        table[:, 5:9], np.zeros((phase_steps, 4))
    ):
        raise ValueError("selected reverse teacher must be finite and head-passive")
    cadence_hz = float(adapter.get("cadence_hz", float("nan")))
    phase_advance = float(
        adapter.get("phase_advance_bins_per_control", float("nan"))
    )
    entry_phase = float(
        adapter.get("entry_phase_preincrement_bins", float("nan"))
    )
    first_phase = float(
        adapter.get("first_reference_phase_after_increment_bins", float("nan"))
    )
    if not 1.5 <= cadence_hz <= 2.0:
        raise ValueError("selected reverse cadence must be in [1.5, 2.0] Hz")
    if not np.isclose(
        phase_advance, cadence_hz * phase_steps * 0.02, atol=1.0e-12
    ):
        raise ValueError("selected reverse phase advance does not match cadence")
    if not np.isclose(first_phase, entry_phase + phase_advance, atol=1.0e-12):
        raise ValueError("selected reverse entry/first phase contract drifted")
    if teacher.get("physical_command_mps_radps") != [-0.05, 0.0, 0.0]:
        raise ValueError("selected teacher physical command must be -0.05 m/s")
    return {
        "path": resolved,
        "sha256": actual_sha,
        "payload": payload,
        "table": table,
        "cadence_hz": cadence_hz,
        "phase_advance_bins": phase_advance,
        "entry_phase_bins": entry_phase,
        "first_phase_bins": first_phase,
        "candidate_id": teacher["candidate_id"],
        "candidate_name": teacher["name"],
    }


def interpolate_periodic_table(table: np.ndarray, phase: float) -> np.ndarray:
    values = np.asarray(table, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("periodic table must be two-dimensional")
    wrapped = float(phase) % len(values)
    index = int(np.floor(wrapped))
    fraction = wrapped - index
    return (1.0 - fraction) * values[index] + fraction * values[(index + 1) % len(values)]


def reverse_teacher_startup_audit(selected: Mapping[str, Any]) -> dict[str, Any]:
    first_target = interpolate_periodic_table(
        selected["table"], float(selected["first_phase_bins"])
    )
    reset = np.asarray(
        [SAFE_INIT_POS[name] for name in ACTUATOR_JOINT_ORDER], dtype=np.float64
    )
    delta = np.abs(first_target - reset)
    upstream_maximum = float(np.max(delta))
    precomposed = reset + np.clip(
        first_target - reset,
        -MAX_TARGET_DELTA_PER_TICK_RAD,
        MAX_TARGET_DELTA_PER_TICK_RAD,
    )
    precomposed_delta = np.abs(precomposed - reset)
    precomposed_maximum = float(np.max(precomposed_delta))
    if precomposed_maximum > MAX_TARGET_DELTA_PER_TICK_RAD + 1.0e-12:
        raise ValueError("reverse teacher precomposer exceeds 0.04 rad/tick")
    return {
        "old_teacher_first_raw_jump_rad": OLD_REVERSE_TEACHER_FIRST_JUMP_RAD,
        "selected_teacher_upstream_table_jump_rad": upstream_maximum,
        "selected_teacher_upstream_table_jump_joint": ACTUATOR_JOINT_ORDER[
            int(np.argmax(delta))
        ],
        "training_visible_precomposed_first_jump_rad": precomposed_maximum,
        "training_visible_precomposed_first_jump_joint": ACTUATOR_JOINT_ORDER[
            int(np.argmax(precomposed_delta))
        ],
        "precomposer_source": "current applied target; no stale reset/re-entry state",
        "final_guard_maximum_applied_delta_rad": MAX_TARGET_DELTA_PER_TICK_RAD,
        "final_guard_calls_per_control": 1,
        "passed": True,
    }


def load_trusted_h4_parent_bundle(
    *,
    params_path: Path | None,
    manifest_path: Path | None,
    expected_params_sha256: str | None,
) -> dict[str, Any] | None:
    supplied = (params_path is not None, manifest_path is not None, expected_params_sha256 is not None)
    if not any(supplied):
        return None
    if not all(supplied):
        raise ValueError(
            "--h4-parent-params, --h4-parent-manifest, and "
            "--h4-parent-params-sha256 must be supplied together"
        )
    assert params_path is not None and manifest_path is not None
    assert expected_params_sha256 is not None
    params_resolved = params_path.resolve()
    manifest_resolved = manifest_path.resolve()
    if not params_resolved.is_file() or not manifest_resolved.is_file():
        raise FileNotFoundError("trusted H4 parent params/manifest is missing")
    expected_sha = expected_params_sha256.lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("--h4-parent-params-sha256 must be exact lowercase SHA256")
    actual_sha = sha256_file(params_resolved)
    if actual_sha != expected_sha:
        raise ValueError("trusted H4 parent params SHA mismatch")
    manifest = _load_json_strict(manifest_resolved)
    if manifest.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("H4 parent manifest must prohibit hardware")
    if manifest.get("status") not in {"COMPLETED", "WIRING_PASS"}:
        raise ValueError("H4 parent manifest is not complete")
    recorded = manifest.get("outputs", {}).get("final_params", {})
    if recorded.get("sha256") != actual_sha:
        raise ValueError("H4 parent manifest does not bind final params SHA")
    if Path(recorded.get("path", "")).resolve() != params_resolved:
        raise ValueError("H4 parent manifest final params path mismatch")
    resolved_config = manifest.get("resolved_config", {})
    config_path = Path(resolved_config.get("path", "")).resolve()
    if not config_path.is_file() or sha256_file(config_path) != resolved_config.get("sha256"):
        raise ValueError("H4 parent resolved config hash/path mismatch")
    config_payload = _load_json_strict(config_path)
    if (
        config_payload.get("hardware_deployment") != "PROHIBITED"
        or config_payload.get("actor_observation_width") != 116
        or config_payload.get("observation_mode") != "h4_116_transplant"
        or config_payload.get("expert") != manifest.get("expert")
        or config_payload.get("activity") != manifest.get("activity")
    ):
        raise ValueError("trusted H4 parent config/manifest contract drifted")
    recorded_canonical = resolved_config.get("canonical_sha256")
    if recorded_canonical != canonical_json_sha(config_payload):
        raise ValueError("trusted H4 parent canonical config SHA mismatch")
    run_name = manifest.get("run_name")
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("trusted H4 parent run name is missing")
    return {
        "params_path": params_resolved,
        "params_sha256": actual_sha,
        "manifest_path": manifest_resolved,
        "manifest_sha256": sha256_file(manifest_resolved),
        "manifest": manifest,
        "run_name": run_name,
        "resolved_config_path": config_path,
        "resolved_config_sha256": resolved_config["sha256"],
        "resolved_config": config_payload,
    }


def load_h5_targetspace_seed_bundle(
    *,
    params_path: Path | None,
    manifest_path: Path | None,
    expected_params_sha256: str | None,
    expected_manifest_sha256: str | None,
    teacher_mode: str = "table",
) -> dict[str, Any] | None:
    """Load an auditable simulation-only H5 target-space seed."""

    supplied = (
        params_path is not None,
        manifest_path is not None,
        expected_params_sha256 is not None,
        expected_manifest_sha256 is not None,
    )
    if not any(supplied):
        return None
    if not all(supplied):
        raise ValueError(
            "--h5-seed-params, --h5-seed-manifest, and both H5 seed SHA256 "
            "arguments must be supplied together"
        )
    assert params_path is not None and manifest_path is not None
    assert expected_params_sha256 is not None and expected_manifest_sha256 is not None
    resolved_params = params_path.resolve()
    resolved_manifest = manifest_path.resolve()
    if not resolved_params.is_file() or not resolved_manifest.is_file():
        raise FileNotFoundError("H5 target-space seed params/manifest is missing")
    params_sha = expected_params_sha256.lower()
    manifest_sha = expected_manifest_sha256.lower()
    for label, value in (
        ("--h5-seed-params-sha256", params_sha),
        ("--h5-seed-manifest-sha256", manifest_sha),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{label} must be exact lowercase SHA256")
    actual_params_sha = sha256_file(resolved_params)
    actual_manifest_sha = sha256_file(resolved_manifest)
    if actual_params_sha != params_sha:
        raise ValueError("H5 target-space seed params SHA mismatch")
    if actual_manifest_sha != manifest_sha:
        raise ValueError("H5 target-space seed manifest SHA mismatch")
    manifest = _load_json_strict(resolved_manifest)
    if (
        manifest.get("status") != "COMPLETED"
        or manifest.get("expert") not in {"reverse", "unified"}
        or manifest.get("qualification_use")
        != "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION"
        or manifest.get("hardware_deployment") != "PROHIBITED"
        or manifest.get("candidate_kind") != "H5_TARGET_SPACE_DISTILLED_SEED"
    ):
        raise ValueError("H5 target-space seed manifest is not a completed diagnostic seed")
    recorded = manifest.get("outputs", {}).get("final_params", {})
    if recorded.get("sha256") != actual_params_sha:
        raise ValueError("H5 target-space seed manifest does not bind params SHA")
    if Path(str(recorded.get("path", ""))).resolve() != resolved_params:
        raise ValueError("H5 target-space seed manifest params path mismatch")
    target_table = None
    raw_target_table = manifest.get("teacher_source", {}).get("target_table_rad")
    if raw_target_table is not None:
        target_table = np.asarray(raw_target_table, dtype=np.float32)
        if target_table.shape != (54, 14) or not np.all(np.isfinite(target_table)):
            raise ValueError("H5 target-space seed target table is invalid")
        if not np.array_equal(target_table[:, 5:9], np.zeros((54, 4), dtype=np.float32)):
            raise ValueError("H5 target-space seed target table head channels drifted")
    with resolved_params.open("rb") as stream:
        params = pickle.load(stream)
    teacher_params = None
    teacher_params_path = None
    teacher_params_sha256 = None
    if teacher_mode == "adaptive_residual":
        rollout = manifest.get("rollout_policy", {})
        teacher_params_path = Path(str(rollout.get("params_path", ""))).resolve()
        teacher_params_sha256 = str(rollout.get("params_sha256", "")).lower()
        if (
            not teacher_params_path.is_file()
            or len(teacher_params_sha256) != 64
            or sha256_file(teacher_params_path) != teacher_params_sha256
        ):
            raise ValueError(
                "adaptive H5 seed requires a hash-bound rollout teacher params file"
            )
        with teacher_params_path.open("rb") as stream:
            teacher_params = pickle.load(stream)
        validate_h4_params(teacher_params)
    return {
        "params_path": resolved_params,
        "params_sha256": actual_params_sha,
        "manifest_path": resolved_manifest,
        "manifest_sha256": actual_manifest_sha,
        "manifest": manifest,
        "params": params,
        "target_table": target_table,
        "teacher_mode": str(teacher_mode),
        "teacher_params": teacher_params,
        "teacher_params_path": teacher_params_path,
        "teacher_params_sha256": teacher_params_sha256,
    }


def _strict_episode_checks(episode: Mapping[str, Any]) -> dict[str, bool]:
    metrics = episode.get("metrics", episode)
    required = {
        "speed_ratio": lambda value: 0.75 <= value <= 1.25,
        "absolute_cross_velocity_mps": lambda value: value <= 0.012,
        "absolute_uncommanded_yaw_rate_radps": lambda value: value <= 0.05,
        "maximum_heading_change_per_6s_rad": lambda value: value <= 0.15,
        "single_support_rate": lambda value: 0.25 <= value <= 0.60,
        "flight_rate": lambda value: value <= 0.01,
        "stance_slip_rms_mps": lambda value: value <= 0.015,
        "stance_slip_p95_mps": lambda value: value <= 0.030,
        "maximum_per_stance_cumulative_slip_m": lambda value: value <= 0.020,
        "alternating_touchdown_fraction": lambda value: value >= 0.80,
        "contact_duty_imbalance": lambda value: value <= 0.10,
        "left_right_step_count_imbalance": lambda value: value <= 1.0,
    }
    checks: dict[str, bool] = {}
    for name, predicate in required.items():
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            checks[name] = False
        else:
            checks[name] = bool(np.isfinite(value) and predicate(value))
    duration = float(episode.get("duration_s", metrics.get("duration_s", 0.0)))
    checks["duration_s"] = bool(np.isfinite(duration) and duration >= 6.0)
    safety = episode.get("safety", {})
    checks.update(
        {
            "fell_false": episode.get("fell") is False,
            "fall_count_zero": safety.get("fall_count") == 0,
            "qpos_violation_samples_zero": safety.get(
                "qpos_violation_samples"
            )
            == 0,
            "target_violation_samples_zero": safety.get(
                "target_violation_samples"
            )
            == 0,
            "slew_violation_samples_zero": safety.get(
                "slew_violation_samples"
            )
            == 0,
            "guard_call_violation_samples_zero": safety.get(
                "guard_call_violation_samples"
            )
            == 0,
            "nonfinite_samples_zero": safety.get("nonfinite_samples") == 0,
        }
    )
    return checks


def validate_promotion_evidence(
    path: Path | None,
    *,
    h4_parent: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    raise RuntimeError(PROMOTION_GATE_STATUS)
    # The schema below remains deliberately unreachable until the canonical
    # raw-trajectory producer and full current gait-quality rederivation land.
    # Keeping 1M closed is safer than accepting an abbreviated scalar summary.
    resolved = path.resolve()
    payload = _load_json_strict(resolved)
    gate = payload.get("promotion_gate", {})
    if h4_parent is None:
        raise ValueError("promotion evidence requires a bound H4 parent run")
    strict_path = Path(gate.get("strict_evaluation_artifact_path", "")).resolve()
    if not strict_path.is_file():
        raise ValueError("promotion strict evaluation artifact is missing")
    strict_sha = sha256_file(strict_path)
    if strict_sha != gate.get("strict_evaluation_artifact_sha256"):
        raise ValueError("promotion strict evaluation artifact SHA mismatch")
    strict_payload = _load_json_strict(strict_path)
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind") != "openduckmini_h4_promotion_evidence"
        or strict_payload.get("schema_version") != 1
        or strict_payload.get("artifact_kind")
        != "openduckmini_h4_strict_promotion_evaluation"
        or strict_payload.get("hardware_deployment") != "PROHIBITED"
        or strict_payload.get("execution_provider") != "CPU"
    ):
        raise ValueError("promotion/strict artifact schema or CPU contract drifted")
    candidate = strict_payload.get("candidate", {})
    parent_expert = h4_parent["manifest"].get("expert")
    candidate_binding_checks = {
        "run_name": candidate.get("run_name") == h4_parent["run_name"],
        "final_params_sha256": candidate.get("final_params_sha256")
        == h4_parent["params_sha256"],
        "resolved_config_sha256": candidate.get("resolved_config_sha256")
        == h4_parent["resolved_config_sha256"],
        "expert": candidate.get("expert") == parent_expert,
        "manifest_sha256": candidate.get("manifest_sha256")
        == h4_parent["manifest_sha256"],
    }
    if not all(candidate_binding_checks.values()):
        raise ValueError(
            f"strict artifact candidate cross-binding failed: {candidate_binding_checks}"
        )
    episodes = strict_payload.get("episodes")
    required_seeds = gate.get("required_seeds")
    if (
        not isinstance(required_seeds, list)
        or len(required_seeds) != 3
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in required_seeds)
        or len(set(required_seeds)) != 3
    ):
        raise ValueError("promotion gate requires exactly three unique integer seeds")
    if not isinstance(episodes, list) or len(episodes) != 3:
        raise ValueError("strict evaluation artifact must contain exactly three episodes")
    expected_command = [0.05, 0.0, 0.0] if parent_expert == "forward" else [-0.05, 0.0, 0.0]
    seen_seeds: list[int] = []
    seen_segments: list[str] = []
    episode_audits: list[dict[str, Any]] = []
    for episode in episodes:
        seed = episode.get("seed")
        segment_id = episode.get("segment_id")
        structural_checks = {
            "seed_is_integer": isinstance(seed, int) and not isinstance(seed, bool),
            "segment_id_nonempty": isinstance(segment_id, str) and bool(segment_id),
            "expert_matches": episode.get("expert") == parent_expert,
            "physical_command_matches": episode.get("physical_command_mps_radps")
            == expected_command,
            "source_segment_is_strict": episode.get("source_segment_kind")
            == "H4_STRICT_6S",
        }
        if structural_checks["seed_is_integer"]:
            seen_seeds.append(seed)
        if structural_checks["segment_id_nonempty"]:
            seen_segments.append(segment_id)
        episode_audits.append(
            {
                "seed": seed,
                "segment_id": segment_id,
                "checks": {**structural_checks, **_strict_episode_checks(episode)},
            }
        )
    if (
        sorted(seen_seeds) != sorted(required_seeds)
        or len(set(seen_seeds)) != 3
        or len(set(seen_segments)) != 3
    ):
        raise ValueError("strict episode seed/segment set does not match the exact gate")
    passing = [audit for audit in episode_audits if all(audit["checks"].values())]
    passing_seeds = {audit["seed"] for audit in passing}
    central_hashes = strict_payload.get("central_hashes", {})
    if gate.get("central_hashes") != central_hashes:
        raise ValueError("promotion and strict artifact central hashes disagree")
    central_checks: dict[str, bool] = {}
    for central_path in CENTRAL_QUALITY_PATHS:
        key = str(central_path.relative_to(EXP_ROOT)).replace("\\", "/")
        central_checks[key] = central_hashes.get(key) == sha256_file(central_path)
    provenance = strict_payload.get("runtime_provenance", {})
    provenance_checks = {
        "execution_provider_cpu": provenance.get("execution_provider") == "CPU",
        "candidate_manifest_sha_bound": provenance.get(
            "candidate_manifest_sha256"
        )
        == h4_parent["manifest_sha256"],
        "source_teacher_hashes_bound": provenance.get(
            "source_and_teacher_hashes"
        )
        == h4_parent["manifest"].get("source_and_teacher_hashes_post"),
    }
    checks = {
        "hardware_prohibited": payload.get("hardware_deployment") == "PROHIBITED",
        "candidate_run_name_bound": gate.get("candidate_run_name") == h4_parent["run_name"],
        "candidate_params_sha_bound": gate.get("candidate_final_params_sha256")
        == h4_parent["params_sha256"],
        "candidate_config_sha_bound": gate.get("candidate_resolved_config_sha256")
        == h4_parent["resolved_config_sha256"],
        "strict_artifact_candidate_bound": all(candidate_binding_checks.values()),
        "strict_artifact_sha_bound": strict_sha
        == gate.get("strict_evaluation_artifact_sha256"),
        "exact_three_recomputed_strict_seeds": passing_seeds
        == set(required_seeds),
        "all_central_hashes_current": all(central_checks.values()),
        "runtime_provenance_bound": all(provenance_checks.values()),
    }
    baseline_count = gate.get("baseline_strict_pass_count")
    checks["candidate_improves_baseline_strict_count"] = (
        isinstance(baseline_count, int)
        and not isinstance(baseline_count, bool)
        and 0 <= baseline_count < 3
        and len(passing_seeds) > baseline_count
    )
    if not all(checks.values()):
        raise ValueError(f"1M promotion evidence failed: {checks}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "strict_evaluation_artifact": {
            "path": str(strict_path),
            "sha256": strict_sha,
            "passing_seed_count": len(passing_seeds),
            "episode_audits": episode_audits,
        },
        "central_hash_checks": central_checks,
        "runtime_provenance_checks": provenance_checks,
        "checks": checks,
    }


def resolve_training_shape(args: argparse.Namespace, trainer: Any) -> Any:
    requested = args.num_timesteps
    if args.wiring_only:
        if requested not in (None, WIRING_TIMESTEPS):
            raise ValueError("--wiring-only permits exactly 40 interactions")
        return trainer.TrainingShape(
            num_timesteps=WIRING_TIMESTEPS,
            num_envs=2,
            unroll_length=20,
            batch_size=1,
            num_minibatches=2,
            num_updates_per_batch=1,
            num_evals=1,
        )
    # This path executes a deliberately small, vectorized GPU witness.  It
    # returns before PPO, checkpoint creation, run-directory creation, or any
    # hardware interface, so it must not inherit the authorization reserved
    # for an actual simulation training run.  The full production PPO tuple is
    # recorded and later authorization-bound separately; it is not evidence
    # about per-environment trajectory parity.
    if (
        getattr(args, "h5_v3_substep_contact_preflight_only", False)
        or getattr(args, "v4_substep_collector_trace_preflight_only", False)
        or getattr(
            args, "v4_authoritative_primitive_batch_parity_preflight_only", False
        )
        or getattr(args, "v4_direct_primitive_isolation_preflight_only", False)
        or getattr(
            args, "v4_host_synchronized_primitive_ladder_preflight_only", False
        )
    ):
        if requested not in (None, DEFAULT_PILOT_TIMESTEPS):
            raise ValueError(
                "substep no-PPO preflight permits no custom "
                "timestep count"
            )
        return trainer.TrainingShape(
            num_timesteps=WIRING_TIMESTEPS,
            num_envs=V4_SUBSTEP_COLLECTOR_TRACE_PREFLIGHT_BATCH_SIZE,
            unroll_length=20,
            batch_size=1,
            num_minibatches=2,
            num_updates_per_batch=1,
            num_evals=1,
        )
    if not args.authorize_simulation_training:
        raise ValueError(
            "non-wiring PPO is blocked until --authorize-simulation-training "
            "is supplied after an explicit training decision"
        )
    timesteps = DEFAULT_PILOT_TIMESTEPS if requested is None else int(requested)
    if timesteps not in (DEFAULT_PILOT_TIMESTEPS, PROMOTED_TIMESTEPS):
        raise ValueError("H4 pilot timesteps must be 250000 or 1000000")
    if timesteps == PROMOTED_TIMESTEPS and not getattr(
        args, "unified_development_run", False
    ):
        raise RuntimeError(PROMOTION_GATE_STATUS)
    if timesteps == PROMOTED_TIMESTEPS and args.expert != "unified":
        raise ValueError("1M development training is valid only for unified")
    if timesteps == PROMOTED_TIMESTEPS and args.promotion_evidence is not None:
        raise ValueError(
            "unified development training cannot supply formal promotion evidence"
        )
    return trainer.TrainingShape(
        num_timesteps=timesteps,
        num_envs=args.num_envs,
        unroll_length=20,
        batch_size=125,
        num_minibatches=20,
        num_updates_per_batch=4,
        num_evals=2,
    )


def resolve_physical_sampler_family(args: argparse.Namespace) -> str:
    """Keep every bounded iteration on its inherited v2 curriculum sampler."""

    if any(
        (
            args.forward_iteration_v2,
            args.forward_iteration_v3_touchdown_balance,
            args.forward_iteration_v4_contact_event_validity_persistence,
            args.forward_v5_contact_pulse_abort_scale_only,
            args.forward_iteration_v6_contact_abort_island_only,
        )
    ):
        return "forward_iteration_v2"
    if any(
        (
            args.reverse_iteration_v2,
            args.reverse_iteration_v3_no_target_imitation,
            args.reverse_iteration_v4_residual_transfer_gain_024,
            args.reverse_iteration_v5_no_contact_imitation,
            args.reverse_iteration_v6_absolute_full_leg_targets,
        )
    ):
        return "reverse_iteration_v2"
    return args.expert


def _default_run_name(
    expert: str,
    seed: int,
    wiring_only: bool,
    *,
    forward_iteration_v2: bool = False,
    forward_iteration_v3_touchdown_balance: bool = False,
    forward_iteration_v4_contact_event_validity_persistence: bool = False,
    forward_v5_contact_pulse_abort_scale_only: bool = False,
    forward_iteration_v6_contact_abort_island_only: bool = False,
    reverse_iteration_v2: bool = False,
    reverse_iteration_v3_no_target_imitation: bool = False,
    reverse_iteration_v4_residual_transfer_gain_024: bool = False,
    reverse_iteration_v5_no_contact_imitation: bool = False,
    reverse_iteration_v6_absolute_full_leg_targets: bool = False,
) -> str:
    mode = "wiring" if wiring_only else "pilot"
    iteration = (
        "_iteration_v6_contact_abort_island_only"
        if forward_iteration_v6_contact_abort_island_only
        else "_iteration_v6_absolute_full_leg_targets"
        if reverse_iteration_v6_absolute_full_leg_targets
        else "_v5_contact_pulse_abort_scale_only"
        if forward_v5_contact_pulse_abort_scale_only
        else "_iteration_v5_no_contact_imitation"
        if reverse_iteration_v5_no_contact_imitation
        else "_iteration_v4_contact_event_validity_persistence"
        if forward_iteration_v4_contact_event_validity_persistence
        else "_iteration_v4_residual_transfer_gain_024"
        if reverse_iteration_v4_residual_transfer_gain_024
        else "_iteration_v3_touchdown_balance"
        if forward_iteration_v3_touchdown_balance
        else "_iteration_v3_no_target_imitation"
        if reverse_iteration_v3_no_target_imitation
        else "_iteration_v2"
        if (forward_iteration_v2 or reverse_iteration_v2)
        else ""
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"h4_{mode}_{expert}{iteration}_seed{seed}_{stamp}"


def resolve_execution_contract_id(
    args: argparse.Namespace,
    *,
    forward_iteration_v2_authorization: Mapping[str, Any] | None,
    forward_iteration_v3_touchdown_balance_authorization: Mapping[
        str, Any
    ] | None = None,
    forward_iteration_v4_contact_event_validity_persistence_authorization: Mapping[
        str, Any
    ] | None = None,
    forward_v5_contact_pulse_abort_scale_only_authorization: Mapping[
        str, Any
    ] | None = None,
    forward_iteration_v6_contact_abort_island_only_authorization: Mapping[
        str, Any
    ] | None = None,
    reverse_iteration_v2_authorization: Mapping[str, Any] | None,
    reverse_iteration_v3_no_target_imitation_authorization: Mapping[
        str, Any
    ] | None = None,
    reverse_iteration_v4_residual_transfer_gain_024_authorization: Mapping[
        str, Any
    ] | None = None,
    reverse_iteration_v5_no_contact_imitation_authorization: Mapping[
        str, Any
    ] | None = None,
    reverse_iteration_v6_absolute_full_leg_targets_authorization: Mapping[
        str, Any
    ] | None = None,
) -> str:
    """Keep diagnostic wiring identity separate from authorized 250k identity."""

    if getattr(args, "diagnostic_reward_exploration", False):
        if getattr(args, "unified_development_run", False):
            return "H4_DIAGNOSTIC_UNIFIED_DEVELOPMENT_1M_NOT_QUALIFICATION"
        return f"H4_DIAGNOSTIC_REWARD_EXPLORATION_250K_{args.expert.upper()}"

    authorization = (
        forward_iteration_v2_authorization
        or forward_iteration_v3_touchdown_balance_authorization
        or forward_iteration_v4_contact_event_validity_persistence_authorization
        or forward_v5_contact_pulse_abort_scale_only_authorization
        or forward_iteration_v6_contact_abort_island_only_authorization
        or reverse_iteration_v2_authorization
        or reverse_iteration_v3_no_target_imitation_authorization
        or reverse_iteration_v4_residual_transfer_gain_024_authorization
        or reverse_iteration_v5_no_contact_imitation_authorization
        or reverse_iteration_v6_absolute_full_leg_targets_authorization
    )
    if authorization is not None:
        if args.wiring_only:
            if args.forward_iteration_v6_contact_abort_island_only:
                return FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_WIRING_CONTRACT_ID
            if args.reverse_iteration_v6_absolute_full_leg_targets:
                return REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_WIRING_CONTRACT_ID
            if args.forward_v5_contact_pulse_abort_scale_only:
                return FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_WIRING_CONTRACT_ID
            if args.reverse_iteration_v5_no_contact_imitation:
                return REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_WIRING_CONTRACT_ID
            if args.forward_iteration_v4_contact_event_validity_persistence:
                return (
                    FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_WIRING_CONTRACT_ID
                )
            if args.reverse_iteration_v4_residual_transfer_gain_024:
                return (
                    REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_WIRING_CONTRACT_ID
                )
            if args.forward_iteration_v3_touchdown_balance:
                return FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_WIRING_CONTRACT_ID
            if args.reverse_iteration_v3_no_target_imitation:
                return REVERSE_ITERATION_V3_NO_TARGET_IMITATION_WIRING_CONTRACT_ID
            return ITERATION_V2_WIRING_CONTRACT_IDS[args.expert]
        return str(authorization["contract_id"])
    return (
        "H4_FORWARD_ITERATION_V1"
        if args.expert == "forward"
        else "H4_REVERSE_ITERATION_V1"
    )


def _validate_run_name(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("--run-name must be one non-empty directory name")
    return value


def claim_unique_run_directory(
    output_root: Path, expert: str, run_name: str
) -> Path:
    """Atomically claim one immutable run directory or refuse overwrite."""

    name = _validate_run_name(run_name)
    run_dir = output_root.resolve() / expert / name
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _hash_snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"hash input {name!r} is missing: {resolved}")
        result[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return result


def _assert_unchanged(
    before: Mapping[str, Mapping[str, str]],
    after: Mapping[str, Mapping[str, str]],
) -> None:
    drift = {
        name: {"before": before[name]["sha256"], "after": after[name]["sha256"]}
        for name in before
        if before[name]["sha256"] != after[name]["sha256"]
    }
    if drift:
        raise RuntimeError(f"source/teacher mutation detected: {drift}")


def _require_finite_final_state(
    jax: Any, params: Any, metrics: Mapping[str, Any]
) -> tuple[dict[str, float], int]:
    scalar_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.size == 1:
            scalar = float(array.reshape(()))
            if not np.isfinite(scalar):
                raise FloatingPointError(f"non-finite final metric: {key}={scalar}")
            scalar_metrics[key] = scalar
    if not scalar_metrics:
        raise FloatingPointError("PPO returned no scalar final metrics")
    for index, leaf in enumerate(jax.tree_util.tree_leaves(params)):
        array = np.asarray(leaf)
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"non-finite final parameter leaf {index}")
    nonzero_metric_count = sum(value != 0.0 for value in scalar_metrics.values())
    if nonzero_metric_count == 0:
        raise FloatingPointError("all scalar PPO wiring metrics are zero")
    return scalar_metrics, nonzero_metric_count


def run_v4_authoritative_primitive_batch_parity_preflight(
    *,
    args: argparse.Namespace,
    capture_env: Any,
    model: Any,
    mjx_step: Any,
    jax: Any,
    jp: Any,
    backend_resolution: Mapping[str, Any],
    runtime_versions: Mapping[str, Any],
    joystick_module: Any,
    mjx_env_module: Any,
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Compare B=1 lane-1 and B=2 authoritative primitive physics exactly.

    This is deliberately narrower than ``env.step``: each control tick runs
    only the official ``mjx.step`` primitive ten times after replacing ``ctrl``.
    The B=1 case is sliced from the same canonical B=2 reset, so any cross-B
    mismatch cannot be attributed to a different reset key or lane state.
    """

    output_path = Path(
        args.v4_authoritative_primitive_batch_parity_preflight_output
    ).resolve()
    if output_path.exists():
        raise FileExistsError(
            "refusing to overwrite V4 primitive batch-parity evidence: "
            f"{output_path}"
        )

    # This is intentionally host-side and opt-in.  It brackets existing
    # blocking boundaries only; it never enters a jitted function or changes
    # the primitive, action, comparison, or batch topology.  A fresh-process
    # GPU timeout is otherwise silent until the final JSON is written.
    progress_observability = (
        os.environ.get("OPENDUCK_V4_PRIMITIVE_PARITY_PROGRESS") == "1"
    )

    def progress(stage: str) -> None:
        if not progress_observability:
            return
        payload = {
            "event": "v4_authoritative_primitive_batch_parity_progress",
            "stage": stage,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
        sys.stderr.flush()

    expected_backend = JAX_RESOLVED_BACKENDS["gpu"]
    if (
        args.platform != "gpu"
        or str(jax.default_backend()) != expected_backend
        or backend_resolution.get("passed") is not True
        or backend_resolution.get("resolved_default_backend") != expected_backend
        or backend_resolution.get("resolved_device_platforms") in (None, [])
        or any(
            platform != expected_backend
            for platform in backend_resolution["resolved_device_platforms"]
        )
    ):
        raise RuntimeError(
            "V4 primitive batch-parity requires a resolved GPU backend: "
            f"{backend_resolution!r}"
        )
    capture_model = getattr(capture_env, "mjx_model", None)
    if capture_model is None or capture_model is not model:
        raise RuntimeError(
            "V4 primitive batch-parity model is not the capture environment model"
        )
    expected_mjx_step = getattr(getattr(mjx_env_module, "mjx", None), "step", None)
    if expected_mjx_step is None or mjx_step is not expected_mjx_step:
        raise RuntimeError(
            "V4 primitive batch-parity must call the upstream mjx_env.mjx.step"
        )
    if getattr(joystick_module, "mjx_env", None) is not mjx_env_module:
        raise RuntimeError("V4 primitive batch-parity joystick/mjx_env binding drifted")
    upstream_step_source = inspect.getsourcefile(mjx_step)
    if not isinstance(upstream_step_source, str) or not upstream_step_source:
        raise RuntimeError("V4 primitive batch-parity upstream mjx.step source is unavailable")
    bound_source_paths = dict(source_paths)
    bound_source_paths["upstream_mjx_step"] = Path(upstream_step_source).resolve()
    source_before = _hash_snapshot(bound_source_paths)
    control_steps = int(args.v4_authoritative_primitive_batch_parity_control_steps)
    if control_steps not in (1, 3):
        raise ValueError("V4 primitive batch-parity supports T=1 or T=3 only")
    batch_size = V4_SUBSTEP_COLLECTOR_TRACE_PREFLIGHT_BATCH_SIZE
    parity_seed = int(args.seed) + 41
    reset_keys = jax.random.split(jax.random.PRNGKey(parity_seed), batch_size)
    capture_reset = jax.jit(jax.vmap(capture_env.reset))

    def block_whole_tree(value: Any) -> Any:
        return jax.block_until_ready(value)

    def deep_device_copy(value: Any) -> Any:
        return jax.tree_util.tree_map(lambda leaf: jp.array(leaf, copy=True), value)

    progress("probe_reset:start")
    action_dtype = np.asarray(block_whole_tree(capture_reset(reset_keys)).data.qpos).dtype
    progress("probe_reset:finish")
    action_template = np.asarray(
        (
            (0.0,) * len(ACTUATOR_JOINT_ORDER),
            (
                0.12,
                -0.08,
                0.05,
                -0.10,
                0.06,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.12,
                0.08,
                -0.05,
                0.10,
                -0.06,
            ),
            (
                -0.09,
                0.06,
                -0.04,
                0.08,
                -0.05,
                0.0,
                0.0,
                0.0,
                0.0,
                0.09,
                -0.06,
                0.04,
                -0.08,
                0.05,
            ),
        ),
        dtype=action_dtype,
    )
    control_range = np.asarray(getattr(model, "actuator_ctrlrange", None))
    expected_action_shape = (3, len(ACTUATOR_JOINT_ORDER))
    if action_template.shape != expected_action_shape:
        raise RuntimeError(
            "V4 primitive batch-parity action shape drifted: "
            f"{action_template.shape!r} != {expected_action_shape!r}"
        )
    if not np.all(np.isfinite(action_template)):
        raise RuntimeError("V4 primitive batch-parity action template is non-finite")
    if not np.array_equal(
        action_template[:, 5:9], np.zeros((3, 4), dtype=action_dtype)
    ):
        raise RuntimeError("V4 primitive batch-parity must keep all head controls zero")
    if control_range.shape != (len(ACTUATOR_JOINT_ORDER), 2) or not np.all(
        np.isfinite(control_range)
    ):
        raise RuntimeError(
            "V4 primitive batch-parity actuator control range is unavailable or invalid"
        )
    used_actions = action_template[:control_steps]
    if np.any(used_actions < control_range[:, 0]) or np.any(
        used_actions > control_range[:, 1]
    ):
        raise RuntimeError(
            "V4 primitive batch-parity action template exceeds actuator control range"
        )
    action_checks = {
        "shape_exact": action_template.shape == expected_action_shape,
        "finite": bool(np.all(np.isfinite(action_template))),
        "head_controls_exactly_zero": bool(
            np.array_equal(
                action_template[:, 5:9], np.zeros((3, 4), dtype=action_dtype)
            )
        ),
        "within_model_actuator_ctrlrange": bool(
            np.all(used_actions >= control_range[:, 0])
            and np.all(used_actions <= control_range[:, 1])
        ),
        "known_bad_signed_zero_raw_comparison_rejected": not h5_preflight_raw_array_equal(
            np.asarray([0.0], dtype=action_dtype),
            np.asarray([-0.0], dtype=action_dtype),
        ),
    }
    if not all(action_checks.values()):
        raise RuntimeError(f"V4 primitive batch-parity action audit failed: {action_checks}")
    actions_b2_host = np.broadcast_to(
        action_template[:control_steps, None, :],
        (control_steps, batch_size, len(ACTUATOR_JOINT_ORDER)),
    ).copy()
    actions_b2 = jp.asarray(actions_b2_host, dtype=action_dtype)
    actions_b1 = actions_b2[:, 1:2]
    trace_count = {"b1": 0, "b2": 0}

    def make_rollout(label: str) -> Any:
        def per_lane_rollout(initial_data: Any, lane_actions: Any) -> Any:
            def control_body(current_data: Any, control_action: Any) -> tuple[Any, Any]:
                def substep_body(data: Any, _unused: Any) -> tuple[Any, Any]:
                    next_data = v4_authoritative_primitive_step(
                        model,
                        data,
                        control_action,
                        mjx_step=mjx_step,
                    )
                    return next_data, save_v4_dynamic_state(next_data)

                next_data, substeps = jax.lax.scan(
                    substep_body,
                    current_data,
                    xs=None,
                    length=10,
                )
                return next_data, (save_v4_dynamic_state(next_data), substeps)

            return jax.lax.scan(control_body, initial_data, lane_actions)

        def batched_rollout(initial_data: Any, batched_actions: Any) -> Any:
            trace_count[label] += 1
            return jax.vmap(per_lane_rollout, in_axes=(0, 1))(
                initial_data, batched_actions
            )

        return jax.jit(batched_rollout)

    b1_rollout = make_rollout("b1")
    b2_rollout = make_rollout("b2")
    progress("canonical_initial_reset:start")
    canonical_initial = block_whole_tree(capture_reset(reset_keys))
    progress("canonical_initial_reset:finish")
    initial_data_b2 = canonical_initial.data
    initial_data_b1 = jax.tree_util.tree_map(lambda leaf: leaf[1:2], initial_data_b2)
    # Warm up the exact two executables before making the compared entry copies.
    progress("warm_b1:start")
    block_whole_tree(b1_rollout(deep_device_copy(initial_data_b1), actions_b1))
    progress("warm_b1:finish")
    progress("warm_b2:start")
    block_whole_tree(b2_rollout(deep_device_copy(initial_data_b2), actions_b2))
    progress("warm_b2:finish")
    progress("canonical_entry_reset:start")
    canonical_entry = block_whole_tree(capture_reset(reset_keys))
    progress("canonical_entry_reset:finish")
    entry_b2 = canonical_entry.data
    entry_b1 = jax.tree_util.tree_map(lambda leaf: leaf[1:2], entry_b2)
    progress("copy_b1_first_input:start")
    b1_first_input = block_whole_tree(deep_device_copy(entry_b1))
    progress("copy_b1_first_input:finish")
    progress("copy_b1_second_input:start")
    b1_second_input = block_whole_tree(deep_device_copy(entry_b1))
    progress("copy_b1_second_input:finish")
    progress("copy_b2_first_input:start")
    b2_first_input = block_whole_tree(deep_device_copy(entry_b2))
    progress("copy_b2_first_input:finish")
    progress("copy_b2_second_input:start")
    b2_second_input = block_whole_tree(deep_device_copy(entry_b2))
    progress("copy_b2_second_input:finish")
    progress("input_raw_equality:start")
    b1_input_equal, b1_input_leaf_count, b1_input_first_hash, b1_input_second_hash = (
        h5_preflight_raw_tree_equal(jax, b1_first_input, b1_second_input)
    )
    b2_input_equal, b2_input_leaf_count, b2_input_first_hash, b2_input_second_hash = (
        h5_preflight_raw_tree_equal(jax, b2_first_input, b2_second_input)
    )
    progress("input_raw_equality:finish")
    if not b1_input_equal or not b2_input_equal:
        raise RuntimeError("V4 primitive batch-parity inputs are not raw-identical")

    progress("b1_first:start")
    b1_first = block_whole_tree(b1_rollout(b1_first_input, actions_b1))
    progress("b1_first:finish")
    progress("b1_second:start")
    b1_second = block_whole_tree(b1_rollout(b1_second_input, actions_b1))
    progress("b1_second:finish")
    progress("b2_first:start")
    b2_first = block_whole_tree(b2_rollout(b2_first_input, actions_b2))
    progress("b2_first:finish")
    progress("b2_second:start")
    b2_second = block_whole_tree(b2_rollout(b2_second_input, actions_b2))
    progress("b2_second:finish")
    progress("repeat_raw_equality:start")
    b1_repeat_equal, b1_repeat_leaf_count, b1_first_hash, b1_second_hash = (
        h5_preflight_raw_tree_equal(jax, b1_first, b1_second)
    )
    b2_repeat_equal, b2_repeat_leaf_count, b2_first_hash, b2_second_hash = (
        h5_preflight_raw_tree_equal(jax, b2_first, b2_second)
    )
    progress("repeat_raw_equality:finish")

    def select_lane(value: Any, lane: int) -> Any:
        return jax.tree_util.tree_map(lambda leaf: leaf[lane], value)

    def raw_tree_difference(first: Any, second: Any, *, limit: int = 20) -> Mapping[str, Any]:
        first_path_leaves, first_tree = jax.tree_util.tree_flatten_with_path(first)
        second_path_leaves, second_tree = jax.tree_util.tree_flatten_with_path(second)
        if first_tree != second_tree:
            return {
                "leaf_count_equal": False,
                "path_order_equal": False,
                "first_leaf_count": len(first_path_leaves),
                "second_leaf_count": len(second_path_leaves),
                "different_leaf_count": None,
                "first_differences": [],
            }
        differences: list[dict[str, Any]] = []
        difference_count = 0
        for (first_path, first_leaf), (second_path, second_leaf) in zip(
            first_path_leaves, second_path_leaves, strict=True
        ):
            if first_path != second_path:
                return {
                    "leaf_count_equal": True,
                    "path_order_equal": False,
                    "first_path": jax.tree_util.keystr(first_path),
                    "second_path": jax.tree_util.keystr(second_path),
                    "different_leaf_count": None,
                    "first_differences": [],
                }
            if not h5_preflight_raw_array_equal(first_leaf, second_leaf):
                difference_count += 1
                if len(differences) < limit:
                    differences.append(
                        {
                            "path": jax.tree_util.keystr(first_path),
                            **h5_preflight_raw_array_difference(first_leaf, second_leaf),
                        }
                    )
        return {
            "leaf_count_equal": True,
            "path_order_equal": True,
            "different_leaf_count": difference_count,
            "first_differences": differences,
        }

    progress("cross_batch_raw_equality:start")
    entry_lane_equal, entry_lane_leaf_count, entry_b1_hash, entry_b2_lane1_hash = (
        h5_preflight_raw_tree_equal(jax, select_lane(entry_b1, 0), select_lane(entry_b2, 1))
    )
    b1_final, b1_history = b1_first
    b2_final, b2_history = b2_first
    final_lane_equal, final_lane_leaf_count, b1_final_hash, b2_lane1_final_hash = (
        h5_preflight_raw_tree_equal(
            jax, select_lane(b1_final, 0), select_lane(b2_final, 1)
        )
    )
    history_lane_equal, history_lane_leaf_count, b1_history_hash, b2_lane1_history_hash = (
        h5_preflight_raw_tree_equal(
            jax, select_lane(b1_history, 0), select_lane(b2_history, 1)
        )
    )
    progress("cross_batch_raw_equality:finish")
    model_raw_tree_sha256, model_leaf_count = h5_preflight_raw_tree_digest(jax, model)
    source_after = _hash_snapshot(bound_source_paths)
    _assert_unchanged(source_before, source_after)
    checks = {
        "platform_is_gpu": args.platform == "gpu",
        "resolved_gpu_backend_exact": (
            backend_resolution.get("resolved_default_backend") == expected_backend
        ),
        "capture_model_identity_exact": capture_model is model,
        "upstream_mjx_step_identity_exact": mjx_step is expected_mjx_step,
        "action_audit_exact": all(action_checks.values()),
        "canonical_b2_reset_input_repeat_raw_equal": b2_input_equal,
        "canonical_lane1_b1_entry_raw_equal": entry_lane_equal,
        "b1_same_arm_full_raw_equal": b1_repeat_equal,
        "b2_same_arm_full_raw_equal": b2_repeat_equal,
        "b1_b2_lane1_final_raw_equal": final_lane_equal,
        "b1_b2_lane1_dynamic_history_raw_equal": history_lane_equal,
        "two_compiled_shapes_trace_once_each": trace_count == {"b1": 1, "b2": 1},
    }
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_v4_authoritative_primitive_batch_parity_no_ppo_diagnostic",
        "status": (
            "V4_AUTHORITATIVE_PRIMITIVE_BATCH_PARITY_PASS_DIAGNOSTIC_NOT_A_TRAINING_CANDIDATE"
            if all(checks.values())
            else "V4_AUTHORITATIVE_PRIMITIVE_BATCH_PARITY_FAIL_DIAGNOSTIC_NOT_A_TRAINING_CANDIDATE"
        ),
        "hardware_deployment": "PROHIBITED",
        "execution": {
            "canonical_reset_batch_size": 2,
            "compared_batch_sizes": [1, 2],
            "canonical_lane": 1,
            "control_steps": control_steps,
            "physics_substeps_per_control": 10,
            "primitive": "v4_authoritative_primitive_step(data.replace(ctrl=action), mjx.step)",
            "reward_or_info_path": "NOT_INVOKED",
            "ppo_or_checkpoint_path": "NOT_INVOKED",
            "backend_resolution": backend_resolution,
            "runtime_versions": dict(runtime_versions),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
            "model_raw_tree_sha256": model_raw_tree_sha256,
            "model_leaf_count": model_leaf_count,
        },
        "reset_keys": {"parity_seed": parity_seed, "canonical_batch_size": 2},
        "actions": {
            "b2_shape": list(actions_b2_host.shape),
            "b2_raw_bytes_sha256": h5_preflight_raw_array_digest(actions_b2_host),
            "b1_raw_bytes_sha256": h5_preflight_raw_array_digest(
                actions_b2_host[:, 1:2]
            ),
            "control_steps_action_rows": action_template[:control_steps].tolist(),
            "actuator_ctrlrange": control_range.tolist(),
            "audit": action_checks,
        },
        "input_repeat": {
            "b1": {
                "raw_equal": b1_input_equal,
                "leaf_count": b1_input_leaf_count,
                "first_raw_tree_sha256": b1_input_first_hash,
                "second_raw_tree_sha256": b1_input_second_hash,
            },
            "b2": {
                "raw_equal": b2_input_equal,
                "leaf_count": b2_input_leaf_count,
                "first_raw_tree_sha256": b2_input_first_hash,
                "second_raw_tree_sha256": b2_input_second_hash,
            },
            "canonical_lane1_b1_raw_equal": entry_lane_equal,
            "canonical_lane1_leaf_count": entry_lane_leaf_count,
            "b1_raw_tree_sha256": entry_b1_hash,
            "b2_lane1_raw_tree_sha256": entry_b2_lane1_hash,
        },
        "same_arm_repeat": {
            "b1_raw_equal": b1_repeat_equal,
            "b1_leaf_count": b1_repeat_leaf_count,
            "b1_first_raw_tree_sha256": b1_first_hash,
            "b1_second_raw_tree_sha256": b1_second_hash,
            "b1_difference": raw_tree_difference(b1_first, b1_second),
            "b2_raw_equal": b2_repeat_equal,
            "b2_leaf_count": b2_repeat_leaf_count,
            "b2_first_raw_tree_sha256": b2_first_hash,
            "b2_second_raw_tree_sha256": b2_second_hash,
            "b2_difference": raw_tree_difference(b2_first, b2_second),
        },
        "cross_batch_lane1": {
            "final_raw_equal": final_lane_equal,
            "final_leaf_count": final_lane_leaf_count,
            "b1_final_raw_tree_sha256": b1_final_hash,
            "b2_lane1_final_raw_tree_sha256": b2_lane1_final_hash,
            "final_difference": raw_tree_difference(
                select_lane(b1_final, 0), select_lane(b2_final, 1)
            ),
            "dynamic_history_raw_equal": history_lane_equal,
            "dynamic_history_leaf_count": history_lane_leaf_count,
            "b1_dynamic_history_raw_tree_sha256": b1_history_hash,
            "b2_lane1_dynamic_history_raw_tree_sha256": b2_lane1_history_hash,
            "dynamic_history_difference": raw_tree_difference(
                select_lane(b1_history, 0), select_lane(b2_history, 1)
            ),
        },
        "compiled_trace_count": trace_count,
        "checks": checks,
        "bound_inputs_pre_and_post": source_before,
        "no_ppo_tripwire": {
            "ppo_train_called": False,
            "checkpoint_written": False,
            "training_run_directory_created": False,
            "preflight_returns_before_ppo_path": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": payload["status"],
        "hardware_deployment": "PROHIBITED",
        "preflight_output": str(output_path),
        "preflight_sha256": sha256_file(output_path),
        "checks": checks,
    }


def run_v4_direct_primitive_isolation_preflight(
    *,
    args: argparse.Namespace,
    capture_env: Any,
    model: Any,
    mjx_step: Any,
    jax: Any,
    jp: Any,
    backend_resolution: Mapping[str, Any],
    runtime_versions: Mapping[str, Any],
    joystick_module: Any,
    mjx_env_module: Any,
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Run direct upstream MJX isolation or a host-synchronized 10-step ladder.

    This is deliberately not a replacement for the authoritative V4 parity
    proof: the production gate remains a ten-substep ``lax.scan`` with its
    original dynamic-history output.  Both diagnostic variants use the same
    canonical B=2 reset, lane-1 slice, upstream ``mjx.step``, exact raw
    comparisons, and B=1/B=2 topology.  The ladder applies ten direct steps
    with a host synchronization and raw comparison at every substep, which
    makes a timeout boundary observable without temporal fusion.  Both return
    before PPO, checkpoints, or any hardware interface.
    """

    host_synchronized_ladder = bool(
        getattr(args, "v4_host_synchronized_primitive_ladder_preflight_only", False)
    )
    requested_output = (
        args.v4_host_synchronized_primitive_ladder_preflight_output
        if host_synchronized_ladder
        else args.v4_direct_primitive_isolation_preflight_output
    )
    if host_synchronized_ladder:
        # ``timeout --signal=USR1`` asks a hung fresh process for a Python
        # stack before its later hard kill.  This observer is host-only: it
        # does not enter JIT or alter an executable, action, batch, or raw
        # comparison.  Failure to register is itself not a physics result.
        try:
            import faulthandler
            import signal

            faulthandler.enable(file=sys.stderr, all_threads=True)
            faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
        except (OSError, RuntimeError, ValueError):
            pass
    requested_output_path = Path(requested_output)
    output_path = (
        requested_output_path.resolve()
        if requested_output_path.is_absolute()
        else (EXP_ROOT / requested_output_path).resolve()
    )
    if output_path.exists():
        raise FileExistsError(
            "refusing to overwrite V4 direct-primitive isolation evidence: "
            f"{output_path}"
        )

    progress_observability = (
        os.environ.get("OPENDUCK_V4_PRIMITIVE_PARITY_PROGRESS") == "1"
    )

    def progress(stage: str) -> None:
        if not progress_observability:
            return
        payload = {
            "event": (
                "v4_host_synchronized_primitive_ladder_progress"
                if host_synchronized_ladder
                else "v4_direct_primitive_isolation_progress"
            ),
            "stage": stage,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
        sys.stderr.flush()

    expected_backend = JAX_RESOLVED_BACKENDS["gpu"]
    if (
        args.platform != "gpu"
        or str(jax.default_backend()) != expected_backend
        or backend_resolution.get("passed") is not True
        or backend_resolution.get("resolved_default_backend") != expected_backend
        or backend_resolution.get("resolved_device_platforms") in (None, [])
        or any(
            platform != expected_backend
            for platform in backend_resolution["resolved_device_platforms"]
        )
    ):
        raise RuntimeError(
            "V4 direct-primitive isolation requires a resolved GPU backend: "
            f"{backend_resolution!r}"
        )
    capture_model = getattr(capture_env, "mjx_model", None)
    if capture_model is None or capture_model is not model:
        raise RuntimeError(
            "V4 direct-primitive isolation model is not the capture environment model"
        )
    expected_mjx_step = getattr(getattr(mjx_env_module, "mjx", None), "step", None)
    if expected_mjx_step is None or mjx_step is not expected_mjx_step:
        raise RuntimeError(
            "V4 direct-primitive isolation must call upstream mjx_env.mjx.step"
        )
    if getattr(joystick_module, "mjx_env", None) is not mjx_env_module:
        raise RuntimeError(
            "V4 direct-primitive isolation joystick/mjx_env binding drifted"
        )
    upstream_step_source = inspect.getsourcefile(mjx_step)
    if not isinstance(upstream_step_source, str) or not upstream_step_source:
        raise RuntimeError(
            "V4 direct-primitive isolation upstream mjx.step source is unavailable"
        )
    bound_source_paths = dict(source_paths)
    bound_source_paths["upstream_mjx_step"] = Path(upstream_step_source).resolve()
    source_before = _hash_snapshot(bound_source_paths)

    batch_size = V4_SUBSTEP_COLLECTOR_TRACE_PREFLIGHT_BATCH_SIZE
    parity_seed = int(args.seed) + 41
    reset_keys = jax.random.split(jax.random.PRNGKey(parity_seed), batch_size)
    capture_reset = jax.jit(jax.vmap(capture_env.reset))

    def block_whole_tree(value: Any) -> Any:
        return jax.block_until_ready(value)

    def deep_device_copy(value: Any) -> Any:
        return jax.tree_util.tree_map(lambda leaf: jp.array(leaf, copy=True), value)

    progress("canonical_entry_reset:start")
    canonical_entry = block_whole_tree(capture_reset(reset_keys))
    progress("canonical_entry_reset:finish")
    entry_b2 = canonical_entry.data
    entry_b1 = jax.tree_util.tree_map(lambda leaf: leaf[1:2], entry_b2)
    action_dtype = np.asarray(entry_b2.qpos).dtype
    control_range = np.asarray(getattr(model, "actuator_ctrlrange", None))
    expected_action_shape = (batch_size, len(ACTUATOR_JOINT_ORDER))
    actions_b2_host = np.zeros(expected_action_shape, dtype=action_dtype)
    if (
        control_range.shape != (len(ACTUATOR_JOINT_ORDER), 2)
        or not np.all(np.isfinite(control_range))
        or not np.all(actions_b2_host >= control_range[:, 0])
        or not np.all(actions_b2_host <= control_range[:, 1])
    ):
        raise RuntimeError(
            "V4 direct-primitive isolation zero action is outside model actuator range"
        )
    actions_b2 = jp.asarray(actions_b2_host, dtype=action_dtype)
    actions_b1 = actions_b2[1:2]

    def direct_step(data: Any, action: Any) -> Any:
        return v4_authoritative_primitive_step(
            model, data, action, mjx_step=mjx_step
        )

    trace_count = {"b1": 0, "b2": 0}

    def make_direct_rollout(label: str) -> Any:
        def batched_direct_rollout(initial_data: Any, batched_actions: Any) -> Any:
            trace_count[label] += 1
            return jax.vmap(direct_step)(initial_data, batched_actions)

        return jax.jit(batched_direct_rollout)

    b1_rollout = make_direct_rollout("b1")
    b2_rollout = make_direct_rollout("b2")
    progress("copy_b1_first_input:start")
    b1_first_input = block_whole_tree(deep_device_copy(entry_b1))
    progress("copy_b1_first_input:finish")
    progress("copy_b1_second_input:start")
    b1_second_input = block_whole_tree(deep_device_copy(entry_b1))
    progress("copy_b1_second_input:finish")
    progress("copy_b2_first_input:start")
    b2_first_input = block_whole_tree(deep_device_copy(entry_b2))
    progress("copy_b2_first_input:finish")
    progress("copy_b2_second_input:start")
    b2_second_input = block_whole_tree(deep_device_copy(entry_b2))
    progress("copy_b2_second_input:finish")
    b1_input_equal, b1_input_leaf_count, b1_input_first_hash, b1_input_second_hash = (
        h5_preflight_raw_tree_equal(jax, b1_first_input, b1_second_input)
    )
    b2_input_equal, b2_input_leaf_count, b2_input_first_hash, b2_input_second_hash = (
        h5_preflight_raw_tree_equal(jax, b2_first_input, b2_second_input)
    )
    if not b1_input_equal or not b2_input_equal:
        raise RuntimeError("V4 direct-primitive isolation inputs are not raw-identical")

    progress("warm_b1:start")
    block_whole_tree(b1_rollout(deep_device_copy(entry_b1), actions_b1))
    progress("warm_b1:finish")
    progress("warm_b2:start")
    block_whole_tree(b2_rollout(deep_device_copy(entry_b2), actions_b2))
    progress("warm_b2:finish")

    def select_lane(value: Any, lane: int) -> Any:
        return jax.tree_util.tree_map(lambda leaf: leaf[lane], value)

    def raw_tree_differences(
        first: Any, second: Any, *, limit: int = 8
    ) -> Mapping[str, Any]:
        """Record exact leaf evidence only after a raw-equality failure."""

        first_leaves, first_tree = jax.tree_util.tree_flatten_with_path(first)
        second_leaves, second_tree = jax.tree_util.tree_flatten_with_path(second)
        if first_tree != second_tree:
            return {
                "tree_structure_equal": False,
                "first_leaf_count": len(first_leaves),
                "second_leaf_count": len(second_leaves),
                "different_leaf_count": None,
                "first_differences": [],
            }
        different_leaf_count = 0
        first_differences: list[dict[str, Any]] = []
        for (first_path, first_leaf), (second_path, second_leaf) in zip(
            first_leaves, second_leaves, strict=True
        ):
            if first_path != second_path:
                return {
                    "tree_structure_equal": False,
                    "first_leaf_count": len(first_leaves),
                    "second_leaf_count": len(second_leaves),
                    "different_leaf_count": None,
                    "first_differences": [
                        {
                            "first_path": jax.tree_util.keystr(first_path),
                            "second_path": jax.tree_util.keystr(second_path),
                        }
                    ],
                }
            if not h5_preflight_raw_array_equal(first_leaf, second_leaf):
                different_leaf_count += 1
                if len(first_differences) < limit:
                    first_differences.append(
                        {
                            "path": jax.tree_util.keystr(first_path),
                            **h5_preflight_raw_array_difference(
                                first_leaf, second_leaf
                            ),
                        }
                    )
        return {
            "tree_structure_equal": True,
            "first_leaf_count": len(first_leaves),
            "second_leaf_count": len(second_leaves),
            "different_leaf_count": different_leaf_count,
            "first_differences": first_differences,
        }

    entry_lane_equal, entry_lane_leaf_count, entry_b1_hash, entry_b2_lane1_hash = (
        h5_preflight_raw_tree_equal(
            jax, select_lane(entry_b1, 0), select_lane(entry_b2, 1)
        )
    )
    ladder_substeps: list[dict[str, Any]] = []
    if host_synchronized_ladder:
        current_device = {
            "b1_first": b1_first_input,
            "b1_second": b1_second_input,
            "b2_first": b2_first_input,
            "b2_second": b2_second_input,
        }
        current_host: dict[str, Any] = {}

        def execute_substep(label: str, *, substep: int) -> None:
            rollout = b1_rollout if label.startswith("b1_") else b2_rollout
            action = actions_b1 if label.startswith("b1_") else actions_b2
            prefix = f"substep_{substep:02d}.{label}"
            progress(f"{prefix}.dispatch.call")
            next_device = rollout(current_device[label], action)
            progress(f"{prefix}.dispatch.returned")
            progress(f"{prefix}.block.start")
            next_device = block_whole_tree(next_device)
            progress(f"{prefix}.block.finish")
            progress(f"{prefix}.device_get.start")
            current_host[label] = jax.device_get(next_device)
            progress(f"{prefix}.device_get.finish")
            current_device[label] = next_device

        for substep in range(10):
            for label in ("b1_first", "b1_second", "b2_first", "b2_second"):
                execute_substep(label, substep=substep)
            b1_step_equal, b1_step_leaf_count, b1_first_step_hash, b1_second_step_hash = (
                h5_preflight_raw_tree_equal(
                    jax, current_host["b1_first"], current_host["b1_second"]
                )
            )
            b2_step_equal, b2_step_leaf_count, b2_first_step_hash, b2_second_step_hash = (
                h5_preflight_raw_tree_equal(
                    jax, current_host["b2_first"], current_host["b2_second"]
                )
            )
            cross_step_equal, cross_step_leaf_count, b1_lane_hash, b2_lane_hash = (
                h5_preflight_raw_tree_equal(
                    jax,
                    select_lane(current_host["b1_first"], 0),
                    select_lane(current_host["b2_first"], 1),
                )
            )
            ladder_substeps.append(
                {
                    "substep": substep,
                    "b1_same_arm_raw_equal": b1_step_equal,
                    "b1_leaf_count": b1_step_leaf_count,
                    "b1_first_raw_tree_sha256": b1_first_step_hash,
                    "b1_second_raw_tree_sha256": b1_second_step_hash,
                    "b1_difference": (
                        raw_tree_differences(
                            current_host["b1_first"], current_host["b1_second"]
                        )
                        if not b1_step_equal
                        else None
                    ),
                    "b2_same_arm_raw_equal": b2_step_equal,
                    "b2_leaf_count": b2_step_leaf_count,
                    "b2_first_raw_tree_sha256": b2_first_step_hash,
                    "b2_second_raw_tree_sha256": b2_second_step_hash,
                    "b2_difference": (
                        raw_tree_differences(
                            current_host["b2_first"], current_host["b2_second"]
                        )
                        if not b2_step_equal
                        else None
                    ),
                    "b1_b2_lane1_raw_equal": cross_step_equal,
                    "cross_batch_leaf_count": cross_step_leaf_count,
                    "b1_lane1_raw_tree_sha256": b1_lane_hash,
                    "b2_lane1_raw_tree_sha256": b2_lane_hash,
                    "b1_b2_lane1_difference": (
                        raw_tree_differences(
                            select_lane(current_host["b1_first"], 0),
                            select_lane(current_host["b2_first"], 1),
                        )
                        if not cross_step_equal
                        else None
                    ),
                }
            )
        b1_first = current_host["b1_first"]
        b1_second = current_host["b1_second"]
        b2_first = current_host["b2_first"]
        b2_second = current_host["b2_second"]
    else:
        progress("b1_first:start")
        b1_first = block_whole_tree(b1_rollout(b1_first_input, actions_b1))
        progress("b1_first:finish")
        progress("b1_second:start")
        b1_second = block_whole_tree(b1_rollout(b1_second_input, actions_b1))
        progress("b1_second:finish")
        progress("b2_first:start")
        b2_first = block_whole_tree(b2_rollout(b2_first_input, actions_b2))
        progress("b2_first:finish")
        progress("b2_second:start")
        b2_second = block_whole_tree(b2_rollout(b2_second_input, actions_b2))
        progress("b2_second:finish")

    b1_repeat_equal, b1_repeat_leaf_count, b1_first_hash, b1_second_hash = (
        h5_preflight_raw_tree_equal(jax, b1_first, b1_second)
    )
    b2_repeat_equal, b2_repeat_leaf_count, b2_first_hash, b2_second_hash = (
        h5_preflight_raw_tree_equal(jax, b2_first, b2_second)
    )
    ladder_per_substep_all_raw_equal = bool(
        host_synchronized_ladder
        and len(ladder_substeps) == 10
        and all(
            record["b1_same_arm_raw_equal"]
            and record["b2_same_arm_raw_equal"]
            and record["b1_b2_lane1_raw_equal"]
            for record in ladder_substeps
        )
    )
    final_lane_equal, final_lane_leaf_count, b1_final_hash, b2_lane1_final_hash = (
        h5_preflight_raw_tree_equal(
            jax, select_lane(b1_first, 0), select_lane(b2_first, 1)
        )
    )
    model_raw_tree_sha256, model_leaf_count = h5_preflight_raw_tree_digest(jax, model)
    source_after = _hash_snapshot(bound_source_paths)
    _assert_unchanged(source_before, source_after)
    checks = {
        "platform_is_gpu": args.platform == "gpu",
        "resolved_gpu_backend_exact": (
            backend_resolution.get("resolved_default_backend") == expected_backend
        ),
        "capture_model_identity_exact": capture_model is model,
        "upstream_mjx_step_identity_exact": mjx_step is expected_mjx_step,
        "canonical_b2_reset_input_repeat_raw_equal": b2_input_equal,
        "canonical_lane1_b1_entry_raw_equal": entry_lane_equal,
        "b1_same_arm_full_raw_equal": b1_repeat_equal,
        "b2_same_arm_full_raw_equal": b2_repeat_equal,
        "b1_b2_lane1_final_raw_equal": final_lane_equal,
        "host_ladder_per_substep_raw_equal": (
            ladder_per_substep_all_raw_equal if host_synchronized_ladder else True
        ),
        "two_compiled_shapes_trace_once_each": trace_count == {"b1": 1, "b2": 1},
    }
    payload = {
        "schema_version": 1,
        "artifact_kind": (
            "openduckmini_v4_host_synchronized_primitive_ladder_no_ppo_diagnostic"
            if host_synchronized_ladder
            else "openduckmini_v4_direct_primitive_isolation_no_ppo_diagnostic"
        ),
        "status": (
            (
                "V4_HOST_SYNCHRONIZED_PRIMITIVE_LADDER_PASS_NOT_A_V4_SCAN_PARITY_PASS"
                if host_synchronized_ladder
                else "V4_DIRECT_PRIMITIVE_ISOLATION_PASS_NOT_A_V4_SCAN_PARITY_PASS"
            )
            if all(checks.values())
            else (
                "V4_HOST_SYNCHRONIZED_PRIMITIVE_LADDER_FAIL_NOT_A_TRAINING_CANDIDATE"
                if host_synchronized_ladder
                else "V4_DIRECT_PRIMITIVE_ISOLATION_FAIL_NOT_A_TRAINING_CANDIDATE"
            )
        ),
        "hardware_deployment": "PROHIBITED",
        "execution": {
            "canonical_reset_batch_size": 2,
            "compared_batch_sizes": [1, 2],
            "canonical_lane": 1,
            "physics_substeps": 10 if host_synchronized_ladder else 1,
            "primitive": "v4_authoritative_primitive_step(data.replace(ctrl=zero_action), mjx.step)",
            "scan": "NOT_INVOKED; host-synchronized direct calls" if host_synchronized_ladder else "NOT_INVOKED",
            "reward_or_info_path": "NOT_INVOKED",
            "ppo_or_checkpoint_path": "NOT_INVOKED",
            "backend_resolution": backend_resolution,
            "runtime_versions": dict(runtime_versions),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
            "model_raw_tree_sha256": model_raw_tree_sha256,
            "model_leaf_count": model_leaf_count,
        },
        "reset_keys": {"parity_seed": parity_seed, "canonical_batch_size": 2},
        "actions": {
            "b2_shape": list(actions_b2_host.shape),
            "b2_raw_bytes_sha256": h5_preflight_raw_array_digest(actions_b2_host),
            "b1_raw_bytes_sha256": h5_preflight_raw_array_digest(
                actions_b2_host[1:2]
            ),
            "all_zero": True,
            "within_model_actuator_ctrlrange": True,
        },
        "input_repeat": {
            "b1": {
                "raw_equal": b1_input_equal,
                "leaf_count": b1_input_leaf_count,
                "first_raw_tree_sha256": b1_input_first_hash,
                "second_raw_tree_sha256": b1_input_second_hash,
            },
            "b2": {
                "raw_equal": b2_input_equal,
                "leaf_count": b2_input_leaf_count,
                "first_raw_tree_sha256": b2_input_first_hash,
                "second_raw_tree_sha256": b2_input_second_hash,
            },
            "canonical_lane1_b1_raw_equal": entry_lane_equal,
            "canonical_lane1_leaf_count": entry_lane_leaf_count,
            "b1_raw_tree_sha256": entry_b1_hash,
            "b2_lane1_raw_tree_sha256": entry_b2_lane1_hash,
        },
        "same_arm_repeat": {
            "b1_raw_equal": b1_repeat_equal,
            "b1_leaf_count": b1_repeat_leaf_count,
            "b1_first_raw_tree_sha256": b1_first_hash,
            "b1_second_raw_tree_sha256": b1_second_hash,
            "b2_raw_equal": b2_repeat_equal,
            "b2_leaf_count": b2_repeat_leaf_count,
            "b2_first_raw_tree_sha256": b2_first_hash,
            "b2_second_raw_tree_sha256": b2_second_hash,
        },
        "cross_batch_lane1": {
            "final_raw_equal": final_lane_equal,
            "final_leaf_count": final_lane_leaf_count,
            "b1_final_raw_tree_sha256": b1_final_hash,
            "b2_lane1_final_raw_tree_sha256": b2_lane1_final_hash,
        },
        "host_synchronized_ladder": {
            "enabled": host_synchronized_ladder,
            "required_substep_count": 10 if host_synchronized_ladder else None,
            "completed_substep_count": len(ladder_substeps),
            "per_substep_all_raw_equal": (
                ladder_per_substep_all_raw_equal if host_synchronized_ladder else None
            ),
            "substeps": ladder_substeps,
        },
        "compiled_trace_count": trace_count,
        "checks": checks,
        "bound_inputs_pre_and_post": source_before,
        "no_ppo_tripwire": {
            "ppo_train_called": False,
            "checkpoint_written": False,
            "training_run_directory_created": False,
            "preflight_returns_before_ppo_path": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": payload["status"],
        "hardware_deployment": "PROHIBITED",
        "preflight_output": str(output_path),
        "preflight_sha256": sha256_file(output_path),
        "checks": checks,
    }


def run_v4_substep_collector_trace_preflight(
    *,
    args: argparse.Namespace,
    capture_env: Any,
    baseline_env: Any,
    jax: Any,
    jp: Any,
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Run the H5-free B=2/T=20 collector boundary gate and return before PPO."""

    output_path = Path(args.v4_substep_collector_trace_preflight_output).resolve()
    if output_path.exists():
        raise FileExistsError(
            "refusing to overwrite V4 collector trace preflight evidence: "
            f"{output_path}"
        )
    source_before = _hash_snapshot(source_paths)
    parity_seed = int(args.seed) + 41
    batch_size = V4_SUBSTEP_COLLECTOR_TRACE_PREFLIGHT_BATCH_SIZE
    control_steps = V4_SUBSTEP_COLLECTOR_TRACE_PREFLIGHT_CONTROL_STEPS
    reset_keys = jax.random.split(jax.random.PRNGKey(parity_seed), batch_size)
    capture_reset = jax.jit(jax.vmap(capture_env.reset))
    baseline_reset = jax.jit(jax.vmap(baseline_env.reset))
    capture_step = jax.vmap(capture_env.step)
    baseline_step = jax.vmap(baseline_env.step)

    def make_rollout(step_fn: Any) -> Any:
        def rollout(initial_state: Any, actions: Any) -> tuple[Any, Any]:
            def body(current_state: Any, current_action: Any) -> tuple[Any, Any]:
                next_state = step_fn(current_state, current_action)
                return next_state, next_state

            return jax.lax.scan(body, initial_state, actions)

        return jax.jit(rollout)

    capture_rollout = make_rollout(capture_step)
    baseline_rollout = make_rollout(baseline_step)

    def block_whole_tree(value: Any) -> Any:
        return jax.block_until_ready(value)

    def deep_device_copy(value: Any) -> Any:
        return jax.tree_util.tree_map(lambda leaf: jp.array(leaf, copy=True), value)

    capture_initial = block_whole_tree(capture_reset(reset_keys))
    action_dtype = np.asarray(capture_initial.data.qpos).dtype
    action_template = np.asarray(
        (
            (0.0,) * len(ACTUATOR_JOINT_ORDER),
            (0.12, -0.08, 0.05, -0.10, 0.06, 0.0, 0.0, 0.0, 0.0,
             -0.12, 0.08, -0.05, 0.10, -0.06),
            (-0.09, 0.06, -0.04, 0.08, -0.05, 0.0, 0.0, 0.0, 0.0,
             0.09, -0.06, 0.04, -0.08, 0.05),
            (0.0,) * len(ACTUATOR_JOINT_ORDER),
        ),
        dtype=action_dtype,
    )
    actions = np.broadcast_to(
        np.tile(action_template, (5, 1))[:, None, :],
        (control_steps, batch_size, len(ACTUATOR_JOINT_ORDER)),
    ).copy()
    actions_device = jp.asarray(actions, dtype=action_dtype)

    joystick_module = getattr(capture_env, "_env", capture_env)
    del joystick_module  # the wrapped class restores joystick globals internally.

    # Compile on a sacrificial copy. Both compared capture executions then enter
    # the same already-lowered collector with identical device inputs.
    _warmup_final, _warmup_history = block_whole_tree(
        capture_rollout(deep_device_copy(capture_initial), actions_device)
    )
    del _warmup_final, _warmup_history
    capture_entry = block_whole_tree(capture_reset(reset_keys))
    capture_first_input = block_whole_tree(deep_device_copy(capture_entry))
    capture_second_input = block_whole_tree(deep_device_copy(capture_entry))
    inputs_equal, input_leaf_count, first_input_hash, second_input_hash = (
        h5_preflight_raw_tree_equal(jax, capture_first_input, capture_second_input)
    )
    if not inputs_equal:
        raise RuntimeError("V4 collector capture inputs are not raw-identical")
    capture_first = block_whole_tree(capture_rollout(capture_first_input, actions_device))
    capture_second = block_whole_tree(capture_rollout(capture_second_input, actions_device))
    baseline_entry = block_whole_tree(baseline_reset(reset_keys))
    baseline_first = block_whole_tree(
        baseline_rollout(deep_device_copy(baseline_entry), actions_device)
    )
    baseline_second = block_whole_tree(
        baseline_rollout(deep_device_copy(baseline_entry), actions_device)
    )

    capture_final, capture_history = capture_first
    repeat_final, repeat_history = capture_second
    baseline_final, baseline_history = baseline_first
    baseline_repeat_final, baseline_repeat_history = baseline_second
    capture_only_info_keys = {
        "v4_substep_collector_reset_normalized_force",
        "v4_substep_collector_quality_trace",
    }

    def physics_visible_core(state: Any) -> Mapping[str, Any]:
        return {
            "data": state.data,
            "obs": state.obs,
            "reward": state.reward,
            "done": state.done,
            "metrics": state.metrics,
            "info": {
                key: value
                for key, value in state.info.items()
                if key not in capture_only_info_keys
            },
        }

    def raw_tree_difference(first: Any, second: Any, *, limit: int = 20) -> Mapping[str, Any]:
        # ``tree_flatten_with_path`` returns ``(path_leaf_pairs, treedef)``;
        # retain paths beside their leaves so a failed no-PPO preflight can
        # serialize the exact first raw mismatch rather than aborting while
        # constructing its diagnostic artifact.
        first_path_leaves, first_tree = jax.tree_util.tree_flatten_with_path(first)
        second_path_leaves, second_tree = jax.tree_util.tree_flatten_with_path(second)
        if first_tree != second_tree:
            return {
                "leaf_count_equal": False,
                "path_order_equal": False,
                "first_leaf_count": len(first_path_leaves),
                "second_leaf_count": len(second_path_leaves),
                "different_leaf_count": None,
                "first_differences": [],
            }
        first_paths = [path for path, _leaf in first_path_leaves]
        first_leaves = [leaf for _path, leaf in first_path_leaves]
        second_paths = [path for path, _leaf in second_path_leaves]
        second_leaves = [leaf for _path, leaf in second_path_leaves]
        if len(first_leaves) != len(second_leaves):
            return {
                "leaf_count_equal": False,
                "first_leaf_count": len(first_leaves),
                "second_leaf_count": len(second_leaves),
                "different_leaf_count": None,
                "first_differences": [],
            }
        differences = []
        difference_count = 0
        for first_path, second_path, first_leaf, second_leaf in zip(
            first_paths, second_paths, first_leaves, second_leaves, strict=True
        ):
            if first_path != second_path:
                return {
                    "leaf_count_equal": True,
                    "path_order_equal": False,
                    "first_path": jax.tree_util.keystr(first_path),
                    "second_path": jax.tree_util.keystr(second_path),
                    "different_leaf_count": None,
                    "first_differences": [],
                }
            if not h5_preflight_raw_array_equal(first_leaf, second_leaf):
                difference_count += 1
                if len(differences) < limit:
                    differences.append(
                        {
                            "path": jax.tree_util.keystr(first_path),
                            **h5_preflight_raw_array_difference(first_leaf, second_leaf),
                        }
                    )
        return {
            "leaf_count_equal": True,
            "path_order_equal": True,
            "different_leaf_count": difference_count,
            "first_differences": differences,
        }

    repeat_equal, repeat_leaf_count, first_full_hash, second_full_hash = (
        h5_preflight_raw_tree_equal(jax, capture_first, capture_second)
    )
    repeat_core_equal, repeat_core_leaf_count, first_core_hash, second_core_hash = (
        h5_preflight_raw_tree_equal(
            jax,
            physics_visible_core(capture_final),
            physics_visible_core(repeat_final),
        )
    )
    initial_core_equal, initial_core_leaf_count, capture_initial_hash, baseline_initial_hash = (
        h5_preflight_raw_tree_equal(
            jax,
            physics_visible_core(capture_entry),
            physics_visible_core(baseline_entry),
        )
    )
    final_core_equal, final_core_leaf_count, capture_final_hash, baseline_final_hash = (
        h5_preflight_raw_tree_equal(
            jax,
            physics_visible_core(capture_final),
            physics_visible_core(baseline_final),
        )
    )
    history_core_equal, history_core_leaf_count, capture_history_hash, baseline_history_hash = (
        h5_preflight_raw_tree_equal(
            jax,
            physics_visible_core(capture_history),
            physics_visible_core(baseline_history),
        )
    )
    trace_first = capture_history.info["v4_substep_collector_quality_trace"]
    trace_second = repeat_history.info["v4_substep_collector_quality_trace"]
    trace_repeat_equal, trace_leaf_count, trace_first_hash, trace_second_hash = (
        h5_preflight_raw_tree_equal(jax, trace_first, trace_second)
    )
    baseline_repeat_core_equal, baseline_repeat_core_leaf_count, baseline_first_core_hash, baseline_second_core_hash = (
        h5_preflight_raw_tree_equal(
            jax,
            physics_visible_core(baseline_final),
            physics_visible_core(baseline_repeat_final),
        )
    )
    capture_hlo = str(
        capture_rollout.lower(capture_entry, actions_device).compiler_ir(
            dialect="stablehlo"
        )
    )
    capture_hlo_sha256 = hashlib.sha256(capture_hlo.encode("utf-8")).hexdigest()
    capture_hlo_location_stripped_sha256 = stablehlo_location_stripped_sha256(
        capture_hlo
    )
    capture_hlo_semantic_sha256 = stablehlo_semantic_sha256(capture_hlo)
    capture_hlo_cpu_callback_count = capture_hlo.count("@xla_python_cpu_callback")
    capture_hlo_dump = None
    stablehlo_dump_path = getattr(
        args, "v4_substep_collector_trace_stablehlo_dump_output", None
    )
    if stablehlo_dump_path is not None:
        resolved_dump_path = Path(stablehlo_dump_path).resolve()
        if resolved_dump_path.exists():
            raise FileExistsError(
                "refusing to overwrite immutable V4 collector StableHLO dump: "
                f"{resolved_dump_path}"
            )
        resolved_dump_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_dump_path.write_text(capture_hlo, encoding="utf-8")
        capture_hlo_dump = {
            "path": str(resolved_dump_path),
            "sha256": sha256_file(resolved_dump_path),
        }
    forbidden_h5_substep_tokens = (
        "h5_all_substep",
        "h5_v3_substep",
        "h5_substep_contact_alignment",
    )
    trace_time = np.asarray(trace_first.time_s)
    trace_force = np.asarray(trace_first.normalized_normal_force)
    trace_speed = np.asarray(trace_first.tangential_speed_m_s)
    sealed_trace_fields = {
        "time_s": trace_time,
        "normalized_normal_force": trace_force,
        "tangential_speed_m_s": trace_speed,
        "terminal_after_tick": np.asarray(capture_history.done).astype(bool),
        "reset_normalized_force": np.asarray(
            capture_entry.info["v4_substep_collector_reset_normalized_force"]
        ),
        "base_reward": np.asarray(capture_history.reward),
    }
    sealed_trace = None
    sealed_trace_path = getattr(
        args, "v4_substep_collector_trace_sealed_output", None
    )
    if sealed_trace_path is not None:
        resolved_sealed_trace_path = Path(sealed_trace_path).resolve()
        if resolved_sealed_trace_path.suffix != ".npz":
            raise ValueError("V4 sealed trace output must use the .npz suffix")
        if resolved_sealed_trace_path.exists():
            raise FileExistsError(
                "refusing to overwrite immutable V4 sealed trace: "
                f"{resolved_sealed_trace_path}"
            )
        resolved_sealed_trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(resolved_sealed_trace_path, **sealed_trace_fields)
        with np.load(resolved_sealed_trace_path, allow_pickle=False) as reloaded:
            if set(reloaded.files) != set(sealed_trace_fields):
                raise RuntimeError("V4 sealed trace keys drifted during serialization")
            if not all(
                h5_preflight_raw_array_equal(reloaded[name], value)
                for name, value in sealed_trace_fields.items()
            ):
                raise RuntimeError("V4 sealed trace bytes drifted during serialization")
        sealed_trace = {
            "path": str(resolved_sealed_trace_path),
            "sha256": sha256_file(resolved_sealed_trace_path),
            "serialization": "numpy.savez allow_pickle=false v1",
            "field_order": list(sealed_trace_fields),
            "field_raw_bytes_sha256": {
                name: h5_preflight_raw_array_digest(value)
                for name, value in sealed_trace_fields.items()
            },
            "ordered_field_bundle_sha256": h5_preflight_ordered_array_bundle_digest(
                sealed_trace_fields
            ),
        }
    capture_info_keys = set(capture_final.info)
    baseline_info_keys = set(baseline_final.info)
    source_after = _hash_snapshot(source_paths)
    _assert_unchanged(source_before, source_after)
    checks = {
        "h5_v3_flags_disabled": not bool(
            getattr(args, "h5_v3_command_conditioned_se2_alignment", False)
            or getattr(args, "h5_v3_substep_contact_alignment", False)
        ),
        "batch_size_exact": batch_size == 2,
        "control_steps_exact": control_steps == 20,
        "physics_substeps_per_control_exact": True,
        "input_repeat_raw_equal": inputs_equal,
        "capture_same_arm_full_raw_equal": repeat_equal,
        "capture_same_arm_core_raw_equal": repeat_core_equal,
        "baseline_same_arm_core_raw_equal": baseline_repeat_core_equal,
        "capture_vs_baseline_initial_core_raw_equal": initial_core_equal,
        "capture_vs_baseline_final_core_raw_equal": final_core_equal,
        "capture_vs_baseline_history_core_raw_equal": history_core_equal,
        "capture_info_extra_keys_exact": (
            capture_info_keys - baseline_info_keys == capture_only_info_keys
            and not (baseline_info_keys - capture_info_keys)
        ),
        "trace_repeat_raw_equal": trace_repeat_equal,
        "trace_time_shape_exact": trace_time.shape == (control_steps, batch_size, 10),
        "trace_force_shape_exact": trace_force.shape == (control_steps, batch_size, 10, 2),
        "trace_speed_shape_exact": trace_speed.shape == (control_steps, batch_size, 10, 2),
        "trace_all_finite": bool(
            np.all(np.isfinite(trace_time))
            and np.all(np.isfinite(trace_force))
            and np.all(np.isfinite(trace_speed))
        ),
        "trace_force_nonnegative": bool(np.all(trace_force >= 0.0)),
        "trace_speed_nonnegative": bool(np.all(trace_speed >= 0.0)),
        "collector_stablehlo_has_no_h5_substep_token": not any(
            token in capture_hlo.lower() for token in forbidden_h5_substep_tokens
        ),
        "collector_stablehlo_exactly_one_fail_closed_cpu_callback": (
            capture_hlo_cpu_callback_count == 1
        ),
    }
    sidecar_preflight = None
    if bool(getattr(args, "h5_sidecar_quality_preflight_only", False)):
        if not all(checks.values()):
            raise RuntimeError(
                "pure H5 sidecar preflight requires the CPU collector boundary "
                "to pass before any quality score is calculated"
            )
        sidecar_preflight = run_h5_sidecar_quality_preflight(
            args=args,
            jax=jax,
            jp=jp,
            capture_entry=capture_entry,
            capture_final=capture_final,
            capture_history=capture_history,
            trace=trace_first,
            trace_raw_tree_sha256=trace_first_hash,
            collector_stablehlo_sha256=capture_hlo_sha256,
            collector_stablehlo_location_stripped_sha256=(
                capture_hlo_location_stripped_sha256
            ),
            collector_stablehlo_semantic_sha256=capture_hlo_semantic_sha256,
            source_paths=source_paths,
        )
        checks["pure_h5_sidecar_no_ppo_preflight"] = bool(
            sidecar_preflight["passed"]
        )
    status = (
        "V4_COLLECTOR_TRACE_RAW_PARITY_PASS_NOT_A_TRAINING_CANDIDATE"
        if all(checks.values())
        else "V4_COLLECTOR_TRACE_RAW_PARITY_FAIL_NOT_A_TRAINING_CANDIDATE"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_v4_substep_collector_trace_no_ppo_preflight",
        "status": status,
        "hardware_deployment": "PROHIBITED",
        "execution": {
            "batch_size": batch_size,
            "control_steps": control_steps,
            "physics_substeps_per_control": 10,
            "collector": "jax.jit(lax.scan(jax.vmap(env.step)))",
            "outer_ppo_or_sidecar_execution": "NOT_INVOKED",
        },
        "reset_keys": {"parity_seed": parity_seed, "split_batch_keys": batch_size},
        "actions": {
            "shape": list(actions.shape),
            "dtype": action_dtype.str,
            "raw_bytes_sha256": h5_preflight_raw_array_digest(actions),
            "head_entries_5_to_8_exact_zero": bool(np.all(actions[:, :, 5:9] == 0.0)),
        },
        "collector_stablehlo": {
            "sha256": capture_hlo_sha256,
            "location_stripped_sha256": capture_hlo_location_stripped_sha256,
            "semantic_sha256": capture_hlo_semantic_sha256,
            "xla_python_cpu_callback_count": capture_hlo_cpu_callback_count,
            "immutable_raw_dump": capture_hlo_dump,
            "forbidden_h5_substep_tokens": list(forbidden_h5_substep_tokens),
            "forbidden_h5_substep_tokens_absent": checks[
                "collector_stablehlo_has_no_h5_substep_token"
            ],
        },
        "input_repeat": {
            "raw_equal": inputs_equal,
            "leaf_count": input_leaf_count,
            "first_raw_tree_sha256": first_input_hash,
            "second_raw_tree_sha256": second_input_hash,
        },
        "same_arm_capture_repeat": {
            "full_raw_equal": repeat_equal,
            "full_leaf_count": repeat_leaf_count,
            "first_full_raw_tree_sha256": first_full_hash,
            "second_full_raw_tree_sha256": second_full_hash,
            "physics_visible_core_raw_equal": repeat_core_equal,
            "core_leaf_count": repeat_core_leaf_count,
            "first_core_raw_tree_sha256": first_core_hash,
            "second_core_raw_tree_sha256": second_core_hash,
            "core_difference": raw_tree_difference(
                physics_visible_core(capture_final),
                physics_visible_core(repeat_final),
            ),
        },
        "capture_vs_baseline": {
            "allowed_capture_only_info_keys": sorted(capture_only_info_keys),
            "initial_core_raw_equal": initial_core_equal,
            "initial_core_leaf_count": initial_core_leaf_count,
            "capture_initial_core_sha256": capture_initial_hash,
            "baseline_initial_core_sha256": baseline_initial_hash,
            "final_core_raw_equal": final_core_equal,
            "final_core_leaf_count": final_core_leaf_count,
            "capture_final_core_sha256": capture_final_hash,
            "baseline_final_core_sha256": baseline_final_hash,
            "history_core_raw_equal": history_core_equal,
            "history_core_leaf_count": history_core_leaf_count,
            "capture_history_core_sha256": capture_history_hash,
            "baseline_history_core_sha256": baseline_history_hash,
            "final_core_difference": raw_tree_difference(
                physics_visible_core(capture_final),
                physics_visible_core(baseline_final),
            ),
            "history_core_difference": raw_tree_difference(
                physics_visible_core(capture_history),
                physics_visible_core(baseline_history),
            ),
            "baseline_same_arm_core_raw_equal": baseline_repeat_core_equal,
            "baseline_same_arm_core_leaf_count": baseline_repeat_core_leaf_count,
            "baseline_first_core_sha256": baseline_first_core_hash,
            "baseline_second_core_sha256": baseline_second_core_hash,
            "baseline_same_arm_core_difference": raw_tree_difference(
                physics_visible_core(baseline_final),
                physics_visible_core(baseline_repeat_final),
            ),
        },
        "trace_repeat": {
            "raw_equal": trace_repeat_equal,
            "leaf_count": trace_leaf_count,
            "first_raw_tree_sha256": trace_first_hash,
            "second_raw_tree_sha256": trace_second_hash,
            "difference": raw_tree_difference(trace_first, trace_second),
            "time_shape": list(trace_time.shape),
            "force_shape": list(trace_force.shape),
            "speed_shape": list(trace_speed.shape),
        },
        "sealed_trace": sealed_trace,
        "checks": checks,
        "pure_h5_sidecar_preflight": sidecar_preflight,
        "bound_inputs_pre_and_post": source_before,
        "no_ppo_tripwire": {
            "ppo_train_called": False,
            "checkpoint_written": False,
            "training_run_directory_created": False,
            "preflight_returns_before_ppo_path": True,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": status,
        "hardware_deployment": "PROHIBITED",
        "preflight_output": str(output_path),
        "preflight_sha256": sha256_file(output_path),
        "checks": checks,
    }


def run_h5_sidecar_quality_preflight(
    *,
    args: argparse.Namespace,
    jax: Any,
    jp: Any,
    capture_entry: Any,
    capture_final: Any,
    capture_history: Any,
    trace: Any,
    trace_raw_tree_sha256: str,
    collector_stablehlo_sha256: str,
    collector_stablehlo_location_stripped_sha256: str,
    collector_stablehlo_semantic_sha256: str,
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Score one CPU collector trace without touching physics, PPO, or rewards.

    This is deliberately an outer, non-JIT Python loop.  The V4 collector has
    already completed before this function receives its immutable output.  The
    only JAX operations here are pure array transformations inside the sidecar
    contract; neither an environment nor an MJX function is available here.
    """

    parent_path = Path(args.h5_sidecar_quality_parent).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"CPU collector parent is missing: {parent_path}")
    parent = _load_json_strict(parent_path)
    parent_sha256 = sha256_file(parent_path)
    parent_checks = parent.get("checks", {})
    parent_sources = parent.get("bound_inputs_pre_and_post", {})
    parent_execution = parent.get("execution", {})
    parent_trace = parent.get("trace_repeat", {})
    parent_no_ppo = parent.get("no_ppo_tripwire", {})
    parent_required_checks = (
        "capture_same_arm_full_raw_equal",
        "capture_same_arm_core_raw_equal",
        "baseline_same_arm_core_raw_equal",
        "capture_vs_baseline_initial_core_raw_equal",
        "capture_vs_baseline_final_core_raw_equal",
        "capture_vs_baseline_history_core_raw_equal",
        "trace_repeat_raw_equal",
        "collector_stablehlo_has_no_h5_substep_token",
        "collector_stablehlo_exactly_one_fail_closed_cpu_callback",
    )
    parent_validation = {
        "status_is_cpu_collector_pass": (
            parent.get("status")
            == "V4_COLLECTOR_TRACE_RAW_PARITY_PASS_NOT_A_TRAINING_CANDIDATE"
        ),
        "hardware_prohibited": parent.get("hardware_deployment") == "PROHIBITED",
        "all_required_collector_checks_pass": all(
            parent_checks.get(name) is True for name in parent_required_checks
        ),
        "shape_is_b2_t20": (
            parent_execution.get("batch_size") == 2
            and parent_execution.get("control_steps") == 20
            and parent_execution.get("physics_substeps_per_control") == 10
        ),
        "parent_was_no_ppo": (
            parent_no_ppo.get("ppo_train_called") is False
            and parent_no_ppo.get("checkpoint_written") is False
            and parent_no_ppo.get("training_run_directory_created") is False
            and parent_no_ppo.get("preflight_returns_before_ppo_path") is True
        ),
        "h4_source_matches_parent": (
            parent_sources.get("h4_training_alignment", {}).get("sha256")
            == sha256_file(source_paths["h4_training_alignment"])
        ),
        "scene_matches_parent": (
            parent_sources.get("generated_scene", {}).get("sha256")
            == sha256_file(source_paths["generated_scene"])
        ),
        "manifest_matches_parent": (
            parent_sources.get("generated_manifest", {}).get("sha256")
            == sha256_file(source_paths["generated_manifest"])
        ),
        "sidecar_source_matches_parent": (
            parent_sources.get("h5_sidecar_quality", {}).get("sha256")
            == sha256_file(source_paths["h5_sidecar_quality"])
        ),
        # The runner may gain a post-collector, pure-sidecar audit.  The
        # collector itself is bound by its StableHLO hash, not by that host-only
        # reporting addition.
        "collector_semantic_stablehlo_matches_parent": (
            parent.get("collector_stablehlo", {}).get("semantic_sha256")
            == collector_stablehlo_semantic_sha256
        ),
        "trace_raw_hash_matches_parent": (
            parent_trace.get("first_raw_tree_sha256") == trace_raw_tree_sha256
        ),
    }
    if not all(parent_validation.values()):
        failed = sorted(name for name, passed in parent_validation.items() if not passed)
        raise RuntimeError(
            "pure H5 sidecar parent is not the reproduced CPU collector boundary: "
            f"{failed}"
        )

    force = jp.asarray(trace.normalized_normal_force)
    speed = jp.asarray(trace.tangential_speed_m_s)
    times = jp.asarray(trace.time_s)
    terminal_observed = jp.asarray(capture_history.done).astype(bool)
    reset_force = jp.asarray(
        capture_entry.info["v4_substep_collector_reset_normalized_force"]
    )
    if (
        force.shape != (20, 2, 10, 2)
        or speed.shape != force.shape
        or times.shape != (20, 2, 10)
        or terminal_observed.shape != (20, 2)
        or reset_force.shape != (2, 2)
    ):
        raise RuntimeError(
            "pure H5 sidecar preflight received a collector trace with a drifted shape"
        )

    input_before = {
        "trace": h5_preflight_raw_tree_digest(jax, trace),
        "reset_force": h5_preflight_raw_array_digest(reset_force),
        "terminal": h5_preflight_raw_array_digest(terminal_observed),
        "physics_state": h5_preflight_raw_tree_digest(jax, capture_final),
        "base_reward": h5_preflight_raw_array_digest(capture_history.reward),
    }

    def initialize_batched_carry(reset: Any) -> Any:
        return jax.vmap(
            lambda lane_reset: initialize_h5_sidecar_debounce_carry(
                lane_reset, xp=jp
            )
        )(reset)

    def score_lanes(
        tick_force: Any,
        tick_speed: Any,
        tick_times: Any,
        carry: Any,
        tick_terminal: Any,
    ) -> Any:
        return jax.vmap(
            lambda lane_force, lane_speed, lane_times, lane_reset, lane_carry, lane_terminal: h5_sidecar_score_control_tick(
                lane_force,
                lane_speed,
                times_s=lane_times,
                reset_normalized_force=lane_reset,
                carry=lane_carry,
                terminal_after_tick=lane_terminal,
                xp=jp,
            )
        )(tick_force, tick_speed, tick_times, reset_force, carry, tick_terminal)

    def score_sequence(
        sequence_force: Any,
        sequence_speed: Any,
        sequence_times: Any,
        sequence_terminal: Any,
        *,
        initial_carry: Any | None = None,
    ) -> tuple[Any, tuple[Any, Any]]:
        carry = (
            initialize_batched_carry(reset_force)
            if initial_carry is None
            else initial_carry
        )
        score_history = []
        delta_history = []
        for index in range(sequence_force.shape[0]):
            score = score_lanes(
                sequence_force[index],
                sequence_speed[index],
                sequence_times[index],
                carry,
                sequence_terminal[index],
            )
            delta = jax.vmap(
                lambda losses: h5_sidecar_weighted_reward_delta(
                    losses,
                    strict20ms_slip_rms_scale=-1.0,
                    slip_tail_scale=-1.0,
                    force_tail_scale=-1.0,
                    xp=jp,
                )
            )(score.losses)
            carry = score.carry
            score_history.append(score)
            delta_history.append(delta)
        return (
            carry,
            (
                jax.tree_util.tree_map(
                    lambda *items: jp.stack(items, axis=0), *score_history
                ),
                jp.stack(delta_history, axis=0),
            ),
        )

    no_terminal = jp.zeros_like(terminal_observed)
    full_carry, full_result = score_sequence(force, speed, times, no_terminal)
    repeat_carry, repeat_result = score_sequence(force, speed, times, no_terminal)
    sidecar_repeat_equal, _sidecar_leaf_count, first_sidecar_hash, second_sidecar_hash = (
        h5_preflight_raw_tree_equal(
            jax, (full_carry, full_result), (repeat_carry, repeat_result)
        )
    )

    split = 9
    prefix_carry, prefix_result = score_sequence(
        force[:split], speed[:split], times[:split], no_terminal[:split]
    )
    suffix_carry, suffix_result = score_sequence(
        force[split:],
        speed[split:],
        times[split:],
        no_terminal[split:],
        initial_carry=prefix_carry,
    )
    split_result = jax.tree_util.tree_map(
        lambda left, right: jp.concatenate((left, right), axis=0),
        prefix_result,
        suffix_result,
    )
    unroll_result_equal, _unroll_leaf_count, _unroll_first_hash, _unroll_second_hash = (
        h5_preflight_raw_tree_equal(jax, full_result, split_result)
    )
    unroll_carry_equal, _carry_leaf_count, _carry_first_hash, _carry_second_hash = (
        h5_preflight_raw_tree_equal(jax, full_carry, suffix_carry)
    )

    direct_debounce_lanes = []
    for lane in range(2):
        direct_debounce_lanes.append(
            h5_all_substep_quality_update(
                force[:, lane].reshape(200, 2),
                speed[:, lane].reshape(200, 2),
                initial_debounce=initialize_h5_sidecar_debounce_carry(
                    reset_force[lane], xp=jp
                ).debounce,
                times_s=times[:, lane].reshape(200),
                xp=jp,
            ).debounce
        )
    direct_debounce = jax.tree_util.tree_map(
        lambda *items: jp.stack(items, axis=0), *direct_debounce_lanes
    )
    direct_continuation_equal, _direct_leaf_count, _direct_first_hash, _direct_second_hash = (
        h5_preflight_raw_tree_equal(jax, full_carry.debounce, direct_debounce)
    )

    forced_terminal = no_terminal.at[4, 0].set(True).at[11, 1].set(True)
    forced_carry, forced_result = score_sequence(
        force, speed, times, forced_terminal
    )
    scores_history, weighted_delta = full_result
    forced_scores_history, _forced_delta = forced_result

    def lane_tick(value: Any, tick: int, lane: int) -> Any:
        return jax.tree_util.tree_map(lambda leaf: leaf[tick, lane], value)

    terminal_tick_loss_equal = all(
        h5_preflight_raw_tree_equal(
            jax,
            lane_tick(scores_history.losses, tick, lane),
            lane_tick(forced_scores_history.losses, tick, lane),
        )[0]
        for tick, lane in ((4, 0), (11, 1))
    )
    terminal_tick_reward_delta_equal = all(
        h5_preflight_raw_array_equal(
            weighted_delta[tick, lane], _forced_delta[tick, lane]
        )
        for tick, lane in ((4, 0), (11, 1))
    )

    def carry_at_tick_lane(carry_history: Any, tick: int, lane: int) -> Any:
        return jax.tree_util.tree_map(lambda leaf: leaf[tick, lane], carry_history)

    def fresh_next_tick_equal(tick: int, lane: int, *, reset_time: bool) -> bool:
        continuation = carry_at_tick_lane(forced_scores_history.carry, tick, lane)
        lane_times = times[tick + 1, lane]
        if reset_time:
            lane_times = jp.arange(10, dtype=lane_times.dtype) * jp.asarray(
                0.002, dtype=lane_times.dtype
            )
        continued = h5_sidecar_score_control_tick(
            force[tick + 1, lane],
            speed[tick + 1, lane],
            times_s=lane_times,
            reset_normalized_force=reset_force[lane],
            carry=continuation,
            terminal_after_tick=False,
            xp=jp,
        )
        fresh = h5_sidecar_score_control_tick(
            force[tick + 1, lane],
            speed[tick + 1, lane],
            times_s=lane_times,
            reset_normalized_force=reset_force[lane],
            carry=initialize_h5_sidecar_debounce_carry(reset_force[lane], xp=jp),
            terminal_after_tick=False,
            xp=jp,
        )
        return h5_preflight_raw_tree_equal(jax, continued, fresh)[0]

    async_terminal_reset_equal = (
        fresh_next_tick_equal(4, 0, reset_time=False)
        and fresh_next_tick_equal(11, 1, reset_time=False)
    )
    terminal_time_reset_equal = (
        fresh_next_tick_equal(4, 0, reset_time=True)
        and fresh_next_tick_equal(11, 1, reset_time=True)
    )

    manual_weighted_delta = (
        -scores_history.losses.strict20ms_slip_rms_loss
        - scores_history.losses.slip_tail_loss
        - scores_history.losses.force_tail_loss
    )
    weighted_delta_once_equal = h5_preflight_raw_array_equal(
        weighted_delta, manual_weighted_delta
    )
    known_times = jp.arange(10, dtype=force.dtype) * jp.asarray(0.002, dtype=force.dtype)
    known_bad_slip = h5_sidecar_score_control_tick(
        jp.full((10, 2), 0.5, dtype=force.dtype),
        jp.full((10, 2), 0.04, dtype=force.dtype),
        times_s=known_times,
        reset_normalized_force=reset_force[0],
        carry=initialize_h5_sidecar_debounce_carry(reset_force[0], xp=jp),
        terminal_after_tick=False,
        xp=jp,
    )
    known_bad_force = h5_sidecar_score_control_tick(
        jp.full((10, 2), 2.0, dtype=force.dtype),
        jp.zeros((10, 2), dtype=force.dtype),
        times_s=known_times,
        reset_normalized_force=reset_force[0],
        carry=initialize_h5_sidecar_debounce_carry(reset_force[0], xp=jp),
        terminal_after_tick=False,
        xp=jp,
    )
    input_after = {
        "trace": h5_preflight_raw_tree_digest(jax, trace),
        "reset_force": h5_preflight_raw_array_digest(reset_force),
        "terminal": h5_preflight_raw_array_digest(terminal_observed),
        "physics_state": h5_preflight_raw_tree_digest(jax, capture_final),
        "base_reward": h5_preflight_raw_array_digest(capture_history.reward),
    }
    sidecar_source_text = source_paths["h5_sidecar_quality"].read_text(encoding="utf-8")
    checks = {
        "parent_hash_bound_and_reproduced": all(parent_validation.values()),
        "sidecar_contract_id_bound": H5_V3_SIDECAR_QUALITY_CONTRACT_ID
        == "H5_V3_SIDECAR_QUALITY_20260812",
        "same_trace_sidecar_repeat_raw_equal": sidecar_repeat_equal,
        "unroll_boundary_result_raw_equal": unroll_result_equal,
        "unroll_boundary_carry_raw_equal": unroll_carry_equal,
        "continuous_200_sample_debounce_raw_equal": direct_continuation_equal,
        "terminal_tick_loss_raw_equal": terminal_tick_loss_equal,
        "terminal_tick_reward_delta_raw_equal": terminal_tick_reward_delta_equal,
        "asynchronous_terminal_next_tick_resets_raw_equal": async_terminal_reset_equal,
        "terminal_time_reset_next_tick_raw_equal": terminal_time_reset_equal,
        "weighted_reward_delta_added_once_raw_equal": weighted_delta_once_equal,
        "known_bad_slip_cost_nonzero": bool(
            np.asarray(known_bad_slip.losses.strict20ms_slip_rms_loss) > 0.0
            and np.asarray(known_bad_slip.losses.slip_tail_loss) > 0.0
        ),
        "known_bad_force_cost_nonzero": bool(
            np.asarray(known_bad_force.losses.force_tail_loss) > 0.0
        ),
        "input_trace_reset_terminal_physics_reward_unchanged": input_before == input_after,
        "sidecar_source_has_no_simulator_or_ppo_call": not any(
            token in sidecar_source_text
            for token in ("mjx.", "env.step", "ppo.", "checkpoint")
        ),
    }
    return {
        "status": (
            "CPU_PURE_H5_SIDECAR_NO_PPO_PASS"
            if all(checks.values())
            else "CPU_PURE_H5_SIDECAR_NO_PPO_FAIL"
        ),
        "passed": all(checks.values()),
        "hardware_deployment": "PROHIBITED",
        "parent": {
            "path": str(parent_path),
            "sha256": parent_sha256,
            "trace_raw_tree_sha256": trace_raw_tree_sha256,
            "collector_stablehlo_raw_sha256": collector_stablehlo_sha256,
            "collector_stablehlo_location_stripped_sha256": (
                collector_stablehlo_location_stripped_sha256
            ),
            "collector_stablehlo_semantic_sha256": collector_stablehlo_semantic_sha256,
            "validation": parent_validation,
        },
        "execution": {
            "sidecar_execution_boundary": "after_collector_before_no_ppo_return",
            "simulator_calls_from_sidecar": 0,
            "ppo_calls": 0,
            "checkpoint_writes": 0,
            "reward_application": "delta_scored_only_not_added_to_env_reward",
            "batch_size": 2,
            "control_steps": 20,
            "substeps_per_control": 10,
        },
        "checks": checks,
        "raw_hashes": {
            "first_sidecar_output": first_sidecar_hash,
            "second_sidecar_output": second_sidecar_hash,
            "trace_before": input_before["trace"],
            "trace_after": input_after["trace"],
            "base_reward_before": input_before["base_reward"],
            "base_reward_after": input_after["base_reward"],
        },
        "known_bad_losses": {
            "slip": {
                field: float(np.asarray(value))
                for field, value in known_bad_slip.losses._asdict().items()
            },
            "force": {
                field: float(np.asarray(value))
                for field, value in known_bad_force.losses._asdict().items()
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    legacy = _load_legacy_trainer()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", required=True, choices=EXPERT_CHOICES)
    parser.add_argument("--wiring-only", action="store_true")
    parser.add_argument("--authorize-simulation-training", action="store_true")
    parser.add_argument(
        "--diagnostic-reward-exploration",
        action="store_true",
        help=(
            "Allow an explicitly simulation-only 250k reward-weight exploration "
            "from frozen v22. This is not an iteration authorization, promotion, "
            "adoption, release, or hardware path."
        ),
    )
    parser.add_argument(
        "--unified-development-run",
        action="store_true",
        help=(
            "Allow an explicitly authorized 1M-interaction unified-policy "
            "development run. This remains diagnostic-only, hardware-prohibited, "
            "and never satisfies the formal H4 promotion gate."
        ),
    )
    parser.add_argument(
        "--forward-iteration-v2",
        action="store_true",
        help=(
            "Select the hash-authorized bounded second forward 250k iteration; "
            "also selects its exact 40-interaction wiring preflight when combined "
            "with --wiring-only and without --authorize-simulation-training."
        ),
    )
    parser.add_argument(
        "--forward-iteration-v3-touchdown-balance",
        action="store_true",
        help=(
            "Select the hash-authorized scale-only third forward 250k iteration; "
            "also selects its exact 40-interaction wiring preflight with "
            "--wiring-only and without --authorize-simulation-training."
        ),
    )
    parser.add_argument(
        "--forward-iteration-v4-contact-event-validity-persistence",
        action="store_true",
        help=(
            "Select the hash-closed fourth forward 250k contact-event state-machine "
            "iteration; also selects its exact 40-interaction wiring preflight with "
            "--wiring-only and without --authorize-simulation-training."
        ),
    )
    parser.add_argument(
        "--forward-v5-contact-pulse-abort-scale-only",
        action="store_true",
        help=(
            "Select the independently authorized fifth forward scale-only "
            "iteration; its sole reward delta is h4_contact_pulse_40ms -1 to -2."
        ),
    )
    parser.add_argument(
        "--forward-iteration-v6-contact-abort-island-only",
        action="store_true",
        help=(
            "Select the sixth forward routing-only iteration: retain v4 "
            "single-authority physics and the exact -1 contact-pulse scale, "
            "but reward only aborted contact islands and keep off-gap aborts "
            "diagnostic-only."
        ),
    )
    parser.add_argument(
        "--reverse-iteration-v2",
        action="store_true",
        help=(
            "Select the hash-authorized bounded second reverse 250k iteration; "
            "also selects its exact 40-interaction wiring preflight when combined "
            "with --wiring-only and without --authorize-simulation-training."
        ),
    )
    parser.add_argument(
        "--reverse-iteration-v3-no-target-imitation",
        action="store_true",
        help=(
            "Select the hash-authorized target-imitation-only third reverse 250k "
            "iteration; also selects its exact 40-interaction wiring preflight "
            "with --wiring-only and without --authorize-simulation-training."
        ),
    )
    parser.add_argument(
        "--reverse-iteration-v4-residual-transfer-gain-024",
        action="store_true",
        help=(
            "Select the hash-closed fourth reverse 250k bounded residual-gain "
            "exploration; also selects its exact 40-interaction wiring preflight "
            "with --wiring-only and without --authorize-simulation-training."
        ),
    )
    parser.add_argument(
        "--reverse-iteration-v5-no-contact-imitation",
        action="store_true",
        help=(
            "Select the independently authorized fifth reverse scale-only "
            "iteration; its sole legacy-reward delta is contact imitation 15 to 0."
        ),
    )
    parser.add_argument(
        "--reverse-iteration-v6-absolute-full-leg-targets",
        action="store_true",
        help=(
            "Select the sixth reverse execution family: the policy decodes "
            "absolute targets for all ten leg actuators, the frozen teacher "
            "provides timing only, and residual authority is exactly zero."
        ),
    )
    parser.add_argument("--num-timesteps", type=int)
    parser.add_argument("--num-envs", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy-cost", type=float, default=DEFAULT_ENTROPY_COST)
    parser.add_argument("--clipping-epsilon", type=float, default=0.10)
    parser.add_argument("--discounting", type=float, default=0.97)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--observation-mode", choices=OBSERVATION_MODE_CHOICES, default="legacy101"
    )
    parser.add_argument("--allow-verified-v22-transplant", action="store_true")
    parser.add_argument("--physical-anchor", type=float, nargs=3)
    parser.add_argument("--policy-observation-anchor", type=float, nargs=3)
    parser.add_argument(
        "--selected-reverse-teacher",
        type=Path,
        default=DEFAULT_SELECTED_REVERSE_TEACHER,
    )
    parser.add_argument("--promotion-evidence", type=Path)
    parser.add_argument("--source-root", type=Path, default=legacy.DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--generated-root", type=Path, default=legacy.DEFAULT_GENERATED_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--parent-checkpoint", type=Path, default=legacy.DEFAULT_PARENT_CHECKPOINT
    )
    parser.add_argument(
        "--h4-parent-params",
        type=Path,
        help="Trusted final_params.pkl from this H4 runner (never an arbitrary pickle).",
    )
    parser.add_argument("--h4-parent-manifest", type=Path)
    parser.add_argument("--h4-parent-params-sha256")
    parser.add_argument(
        "--h5-seed-params",
        type=Path,
        help="Auditable simulation-only H5 target-space distilled seed.",
    )
    parser.add_argument("--h5-seed-manifest", type=Path)
    parser.add_argument("--h5-seed-params-sha256")
    parser.add_argument("--h5-seed-manifest-sha256")
    parser.add_argument(
        "--h5-seed-teacher-mode",
        choices=("table", "actor", "adaptive_residual"),
        default="table",
        help=(
            "Diagnostic H5 seed target source: fixed phase table, distilled "
            "actor, or rollout-teacher actor residual over the fixed table."
        ),
    )
    parser.add_argument(
        "--h5-seed-residual-gain",
        type=float,
        default=0.0,
        help="Adaptive-residual teacher gain in [0, 1]; simulation-only.",
    )
    parser.add_argument(
        "--h5-seed-teacher-reverse-command-contract",
        action="store_true",
        help=(
            "Adapt the private seed-teacher query from unified vx scale 2 "
            "to the historical reverse vx scale 1."
        ),
    )
    parser.add_argument(
        "--h5-seed-initialize-from-params",
        action="store_true",
        help=(
            "Initialize a unified diagnostic run from the supplied H5 seed "
            "params instead of using them only as a private BC teacher."
        ),
    )
    parser.add_argument(
        "--h5-seed-bc-anneal-control-steps",
        type=float,
        default=250.0,
        help=(
            "Control-step horizon over which the reverse H5 target-space seed "
            "behavior-cloning cost decays; positive and simulation-only."
        ),
    )
    parser.add_argument(
        "--h5-seed-bc-scale",
        type=float,
        default=1.0,
        help="Positive magnitude of the diagnostic H5 target-space BC penalty.",
    )
    parser.add_argument(
        "--h5-unified-reverse-route-probability",
        type=float,
        default=None,
        help=(
            "Diagnostic unified sampler probability assigned to reverse and "
            "reverse-turn anchors.  None preserves the balanced 3/13 default."
        ),
    )
    parser.add_argument(
        "--h5-unified-command-mapper",
        choices=H5_UNIFIED_COMMAND_MAPPER_SUPPORTED_MODES,
        default="legacy_h4_compensated",
        help=(
            "Unified actor observation map. direct_normalized_v3 removes legacy "
            "positive-vx cross-axis compensation and must be recorded as a "
            "separate diagnostic training contract."
        ),
    )
    parser.add_argument(
        "--h5-v3-command-conditioned-se2-alignment",
        action="store_true",
        help=(
            "Enable the sealed H5 V3-only command-conditioned SE(2) residual "
            "definition for one simulation 250k pilot."
        ),
    )
    parser.add_argument(
        "--h5-v3-se2-authorization",
        type=Path,
        default=DEFAULT_H5_V3_COMMAND_CONDITIONED_SE2_ALIGNMENT_AUTHORIZATION,
        help="Pinned authorization artifact required with the H5 V3 SE(2) flag.",
    )
    parser.add_argument(
        "--h5-v3-substep-contact-alignment",
        action="store_true",
        help=(
            "Enable the proposed one-factor successor: strict 20-ms-qualified "
            "2-ms substep slip/force costs on top of H5 V3 SE(2) alignment. "
            "PPO remains fail-closed until a separate authorization is supplied."
        ),
    )
    parser.add_argument(
        "--h5-v3-substep-contact-preflight-only",
        action="store_true",
        help=(
            "Construct treatment/control environments and write no-PPO parity "
            "evidence. Requires substep contact alignment and never trains."
        ),
    )
    parser.add_argument(
        "--h5-v3-substep-contact-preflight-output",
        type=Path,
        default=DEFAULT_H5_V3_SE2_SUBSTEP_CONTACT_PREFLIGHT_OUTPUT,
        help="Immutable JSON output path for the no-PPO H5 substep preflight.",
    )
    parser.add_argument(
        "--h5-v3-substep-contact-t1-diagnostic-arm",
        choices=("treatment", "control"),
        help=(
            "Run exactly one B=2/T=1 fresh-process no-PPO diagnostic arm. "
            "This writes failure-attribution evidence, never a training pass."
        ),
    )
    parser.add_argument(
        "--h5-v3-substep-contact-t1-diagnostic-output",
        type=Path,
        help="Immutable JSON output path for one H5 B=2/T=1 diagnostic arm.",
    )
    parser.add_argument(
        "--h5-v3-substep-contact-t1-diagnostic-mode",
        choices=(
            "compiled_b2",
            "vmap_b1_s1",
            "jit_scalar_s1",
            "eager_s1",
        ),
        default="compiled_b2",
        help=(
            "Execution mode for the no-PPO T=1 attribution diagnostic. "
            "All S1 modes use the exact B=2 lane-1 reset key. compiled_b2 "
            "is the production-shaped witness; vmap_b1_s1 and jit_scalar_s1 "
            "form the transform ladder; eager_s1 is the source-semantics "
            "control. No mode can authorize training."
        ),
    )
    parser.add_argument(
        "--h5-v3-substep-contact-t1-fixed-quality-replay-ablation",
        action="store_true",
        help=(
            "No-PPO T=1 diagnosis only: retain H5 bookkeeping but replace "
            "the second 10x mjx.forward quality replay with pinned fixed 10x2 "
            "telemetry. This can never authorize training."
        ),
    )
    parser.add_argument(
        "--v4-substep-collector-trace-preflight-only",
        action="store_true",
        help=(
            "Run the H5-free B=2/T=20 V4 trace-export collector gate and "
            "return before PPO. It exists only to prove a later sidecar can "
            "consume immutable force/slip traces without changing physics."
        ),
    )
    parser.add_argument(
        "--v4-substep-collector-trace-preflight-output",
        type=Path,
        default=DEFAULT_V4_SUBSTEP_COLLECTOR_TRACE_PREFLIGHT_OUTPUT,
        help="Immutable JSON output path for the H5-free V4 collector gate.",
    )
    parser.add_argument(
        "--v4-authoritative-primitive-batch-parity-preflight-only",
        action="store_true",
        help=(
            "GPU no-PPO diagnostic: compare canonical B=2 lane-1 against B=1 "
            "using only ten authoritative MJX primitive substeps per control tick."
        ),
    )
    parser.add_argument(
        "--v4-authoritative-primitive-batch-parity-control-steps",
        type=int,
        choices=(1, 3),
        default=1,
        help="Control ticks for the GPU primitive B=1/B=2 diagnostic (T=1 or T=3).",
    )
    parser.add_argument(
        "--v4-authoritative-primitive-batch-parity-preflight-output",
        type=Path,
        help="Immutable JSON output path for the GPU primitive B=1/B=2 diagnostic.",
    )
    parser.add_argument(
        "--v4-direct-primitive-isolation-preflight-only",
        action="store_true",
        help=(
            "GPU no-PPO diagnostic: run one direct upstream mjx.step from the "
            "canonical B=1/B=2 inputs. It isolates the V4 10x scan gate and "
            "can never substitute for the authoritative scan-parity proof."
        ),
    )
    parser.add_argument(
        "--v4-direct-primitive-isolation-preflight-output",
        type=Path,
        help="Immutable JSON output path for the direct upstream-MJX isolation diagnostic.",
    )
    parser.add_argument(
        "--v4-host-synchronized-primitive-ladder-preflight-only",
        action="store_true",
        help=(
            "GPU no-PPO diagnostic: execute ten direct upstream mjx.step calls "
            "per B=1/B=2 arm with host synchronization and exact raw checks at "
            "every substep. It can never substitute for V4 scan parity."
        ),
    )
    parser.add_argument(
        "--v4-host-synchronized-primitive-ladder-preflight-output",
        type=Path,
        help="Immutable JSON output path for the host-synchronized ten-step ladder.",
    )
    parser.add_argument(
        "--v4-substep-collector-trace-stablehlo-dump-output",
        type=Path,
        help=(
            "Optional immutable raw StableHLO evidence dump for a V4 collector "
            "preflight. It is diagnostic-only and cannot enable PPO."
        ),
    )
    parser.add_argument(
        "--v4-substep-collector-trace-sealed-output",
        type=Path,
        help=(
            "Optional immutable .npz trace export for a later standalone pure "
            "H5 sidecar audit. It remains a no-PPO collector artifact."
        ),
    )
    parser.add_argument(
        "--h5-sidecar-quality-preflight-only",
        action="store_true",
        help=(
            "CPU-only no-PPO follow-on to the V4 collector gate.  It scores the "
            "completed immutable trace outside env.step and proves reset/debounce "
            "and one-time reward-delta semantics; it never authorizes PPO."
        ),
    )
    parser.add_argument(
        "--h5-sidecar-quality-parent",
        type=Path,
        help=(
            "Hash-bound CPU V4 collector PASS artifact whose trace and StableHLO "
            "must be reproduced before pure sidecar scoring is allowed."
        ),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--reset-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--backward-residual-scale", type=float, default=0.12)
    defaults = H4QualityRewardScales()
    parser.add_argument("--reward-force-slip", type=float, default=defaults.force_slip)
    parser.add_argument(
        "--reward-left-force-slip", type=float, default=defaults.left_force_slip
    )
    parser.add_argument(
        "--reward-right-force-slip", type=float, default=defaults.right_force_slip
    )
    parser.add_argument(
        "--reward-per-foot-slip-tail",
        type=float,
        default=defaults.per_foot_slip_tail,
    )
    parser.add_argument(
        "--reward-per-foot-stance-slip-budget",
        type=float,
        default=defaults.per_foot_stance_slip_budget,
    )
    parser.add_argument(
        "--reward-single-support", type=float, default=defaults.single_support
    )
    parser.add_argument(
        "--reward-single-support-band",
        type=float,
        default=defaults.single_support_band,
    )
    parser.add_argument("--reward-alternation", type=float, default=defaults.alternation)
    parser.add_argument("--reward-load-balance", type=float, default=defaults.load_balance)
    parser.add_argument(
        "--reward-touchdown-count-balance",
        type=float,
        default=defaults.touchdown_count_balance,
    )
    parser.add_argument("--reward-flight", type=float, default=defaults.flight)
    parser.add_argument(
        "--reward-total-normal-force-band",
        type=float,
        default=defaults.total_normal_force_band,
    )
    parser.add_argument(
        "--reward-total-normal-force-tail",
        type=float,
        default=defaults.total_normal_force_tail,
    )
    parser.add_argument(
        "--reward-contact-pulse-40ms",
        type=float,
        default=defaults.contact_pulse_40ms,
    )
    parser.add_argument(
        "--reward-slew-feasibility", type=float, default=defaults.slew_feasibility
    )
    parser.add_argument("--reward-target-lag", type=float)
    parser.add_argument("--reward-left-target-lag", type=float)
    parser.add_argument("--reward-right-target-lag", type=float)
    parser.add_argument(
        "--reward-phase17-left-force-slip",
        type=float,
        default=defaults.phase17_left_force_slip,
    )
    parser.add_argument(
        "--reward-phase17-left-knee-envelope-excess",
        type=float,
        default=defaults.phase17_left_knee_envelope_excess,
    )
    parser.add_argument(
        "--reward-phase17-opposite-leg-lag",
        type=float,
        default=defaults.phase17_opposite_leg_lag,
    )
    parser.add_argument(
        "--reward-forward-cross-drift",
        type=float,
        default=defaults.forward_cross_drift,
    )
    parser.add_argument(
        "--reward-forward-uncommanded-yaw-rate",
        type=float,
        default=defaults.forward_uncommanded_yaw_rate,
    )
    parser.add_argument(
        "--reward-forward-heading-drift",
        type=float,
        default=defaults.forward_heading_drift,
    )
    parser.add_argument("--reward-reverse-speed-boundary", type=float)
    parser.add_argument("--reward-reverse-cross-drift", type=float)
    parser.add_argument("--reward-reverse-uncommanded-yaw-rate", type=float)
    parser.add_argument("--reward-reverse-heading-drift", type=float)
    parser.add_argument("--reward-reverse-phase-force-slip", type=float)
    parser.add_argument(
        "--reward-reverse-contact-priority-reversal-lag", type=float
    )
    return parser


def _validate_scalar_configuration(args: argparse.Namespace) -> None:
    iteration_flags = (
        args.forward_iteration_v2,
        args.forward_iteration_v3_touchdown_balance,
        args.forward_iteration_v4_contact_event_validity_persistence,
        args.forward_v5_contact_pulse_abort_scale_only,
        args.forward_iteration_v6_contact_abort_island_only,
        args.reverse_iteration_v2,
        args.reverse_iteration_v3_no_target_imitation,
        args.reverse_iteration_v4_residual_transfer_gain_024,
        args.reverse_iteration_v5_no_contact_imitation,
        args.reverse_iteration_v6_absolute_full_leg_targets,
    )
    if sum(bool(value) for value in iteration_flags) > 1:
        raise ValueError("H4 iteration expert modes are mutually exclusive")
    iteration_mode = any(iteration_flags)
    iteration_wiring = bool(iteration_mode and args.wiring_only)
    diagnostic_reward_exploration = bool(
        getattr(args, "diagnostic_reward_exploration", False)
    )
    h5_seed_supplied = tuple(
        getattr(args, name, None) is not None
        for name in (
            "h5_seed_params",
            "h5_seed_manifest",
            "h5_seed_params_sha256",
            "h5_seed_manifest_sha256",
        )
    )
    if (
        not np.isfinite(float(args.h5_seed_bc_anneal_control_steps))
        or float(args.h5_seed_bc_anneal_control_steps) <= 0.0
    ):
        raise ValueError("--h5-seed-bc-anneal-control-steps must be positive")
    if not np.isfinite(float(args.h5_seed_bc_scale)) or float(args.h5_seed_bc_scale) <= 0.0:
        raise ValueError("--h5-seed-bc-scale must be positive")
    unified_reverse_probability = getattr(
        args, "h5_unified_reverse_route_probability", None
    )
    if unified_reverse_probability is not None:
        if (
            not np.isfinite(float(unified_reverse_probability))
            or not 0.0 < float(unified_reverse_probability) < 1.0
        ):
            raise ValueError(
                "--h5-unified-reverse-route-probability must be strictly between 0 and 1"
            )
        if args.expert != "unified" or not diagnostic_reward_exploration:
            raise ValueError(
                "reverse-biased H5 sampling is valid only for diagnostic unified training"
            )
    unified_command_mapper = canonical_h5_unified_command_mapper(
        str(getattr(args, "h5_unified_command_mapper", "legacy_h4_compensated"))
    )
    h5_v3_substep_contact_alignment = bool(
        getattr(args, "h5_v3_substep_contact_alignment", False)
    )
    h5_v3_substep_contact_preflight_only = bool(
        getattr(args, "h5_v3_substep_contact_preflight_only", False)
    )
    v4_substep_collector_trace_preflight_only = bool(
        getattr(args, "v4_substep_collector_trace_preflight_only", False)
    )
    v4_authoritative_primitive_batch_parity_preflight_only = bool(
        getattr(args, "v4_authoritative_primitive_batch_parity_preflight_only", False)
    )
    v4_direct_primitive_isolation_preflight_only = bool(
        getattr(args, "v4_direct_primitive_isolation_preflight_only", False)
    )
    v4_host_synchronized_primitive_ladder_preflight_only = bool(
        getattr(args, "v4_host_synchronized_primitive_ladder_preflight_only", False)
    )
    h5_sidecar_quality_preflight_only = bool(
        getattr(args, "h5_sidecar_quality_preflight_only", False)
    )
    if h5_v3_substep_contact_preflight_only and not h5_v3_substep_contact_alignment:
        raise ValueError(
            "--h5-v3-substep-contact-preflight-only requires "
            "--h5-v3-substep-contact-alignment"
        )
    if v4_substep_collector_trace_preflight_only:
        if (
            h5_v3_substep_contact_alignment
            or getattr(args, "h5_v3_command_conditioned_se2_alignment", False)
            or h5_v3_substep_contact_preflight_only
            or args.authorize_simulation_training
            or args.wiring_only
        ):
            raise ValueError(
                "V4 collector trace preflight is H5-free and no-PPO; do not "
                "combine it with H5 V3, training authorization, or wiring-only"
            )
        if (
            args.expert != "unified"
            or not diagnostic_reward_exploration
            or unified_command_mapper != "direct_normalized_v3"
        ):
            raise ValueError(
                "V4 collector trace preflight requires diagnostic unified "
                "direct-V3 command routing"
            )
    if v4_authoritative_primitive_batch_parity_preflight_only:
        if (
            v4_substep_collector_trace_preflight_only
            or v4_direct_primitive_isolation_preflight_only
            or v4_host_synchronized_primitive_ladder_preflight_only
            or h5_v3_substep_contact_alignment
            or getattr(args, "h5_v3_command_conditioned_se2_alignment", False)
            or h5_v3_substep_contact_preflight_only
            or h5_sidecar_quality_preflight_only
            or args.authorize_simulation_training
            or args.wiring_only
            or args.run_name is not None
            or unified_reverse_probability is not None
            or any(h5_seed_supplied)
            or getattr(args, "h5_seed_initialize_from_params", False)
            or args.h5_seed_teacher_mode != "table"
            or float(args.h5_seed_residual_gain) != 0.0
        ):
            raise ValueError(
                "V4 primitive batch-parity diagnostic is H5-free, no-PPO, and "
                "cannot combine with another execution mode"
            )
        if args.platform != "gpu":
            raise ValueError("V4 primitive batch-parity diagnostic is GPU-only")
        if (
            args.expert != "unified"
            or not diagnostic_reward_exploration
            or unified_command_mapper != "direct_normalized_v3"
        ):
            raise ValueError(
                "V4 primitive batch-parity diagnostic requires diagnostic unified "
                "direct-V3 command routing"
            )
        if args.v4_authoritative_primitive_batch_parity_preflight_output is None:
            raise ValueError(
                "V4 primitive batch-parity diagnostic requires an immutable output path"
            )
    if v4_direct_primitive_isolation_preflight_only:
        if (
            v4_substep_collector_trace_preflight_only
            or v4_authoritative_primitive_batch_parity_preflight_only
            or v4_host_synchronized_primitive_ladder_preflight_only
            or h5_v3_substep_contact_alignment
            or getattr(args, "h5_v3_command_conditioned_se2_alignment", False)
            or h5_v3_substep_contact_preflight_only
            or h5_sidecar_quality_preflight_only
            or args.authorize_simulation_training
            or args.wiring_only
            or args.run_name is not None
            or unified_reverse_probability is not None
            or any(h5_seed_supplied)
            or getattr(args, "h5_seed_initialize_from_params", False)
            or args.h5_seed_teacher_mode != "table"
            or float(args.h5_seed_residual_gain) != 0.0
        ):
            raise ValueError(
                "V4 direct-primitive isolation is H5-free, no-PPO, and cannot "
                "combine with another execution mode"
            )
        if args.platform != "gpu":
            raise ValueError("V4 direct-primitive isolation is GPU-only")
        if (
            args.expert != "unified"
            or not diagnostic_reward_exploration
            or unified_command_mapper != "direct_normalized_v3"
        ):
            raise ValueError(
                "V4 direct-primitive isolation requires diagnostic unified "
                "direct-V3 command routing"
            )
        if args.v4_direct_primitive_isolation_preflight_output is None:
            raise ValueError(
                "V4 direct-primitive isolation requires an immutable output path"
            )
    if v4_host_synchronized_primitive_ladder_preflight_only:
        if (
            v4_substep_collector_trace_preflight_only
            or v4_authoritative_primitive_batch_parity_preflight_only
            or v4_direct_primitive_isolation_preflight_only
            or h5_v3_substep_contact_alignment
            or getattr(args, "h5_v3_command_conditioned_se2_alignment", False)
            or h5_v3_substep_contact_preflight_only
            or h5_sidecar_quality_preflight_only
            or args.authorize_simulation_training
            or args.wiring_only
            or args.run_name is not None
            or unified_reverse_probability is not None
            or any(h5_seed_supplied)
            or getattr(args, "h5_seed_initialize_from_params", False)
            or args.h5_seed_teacher_mode != "table"
            or float(args.h5_seed_residual_gain) != 0.0
        ):
            raise ValueError(
                "V4 host-synchronized primitive ladder is H5-free, no-PPO, and "
                "cannot combine with another execution mode"
            )
        if args.platform != "gpu":
            raise ValueError("V4 host-synchronized primitive ladder is GPU-only")
        if (
            args.expert != "unified"
            or not diagnostic_reward_exploration
            or unified_command_mapper != "direct_normalized_v3"
        ):
            raise ValueError(
                "V4 host-synchronized primitive ladder requires diagnostic unified "
                "direct-V3 command routing"
            )
        if args.v4_host_synchronized_primitive_ladder_preflight_output is None:
            raise ValueError(
                "V4 host-synchronized primitive ladder requires an immutable "
                "output path"
            )
    elif (
        getattr(args, "v4_substep_collector_trace_stablehlo_dump_output", None)
        is not None
        or getattr(args, "v4_substep_collector_trace_sealed_output", None)
        is not None
    ):
        raise ValueError(
            "V4 StableHLO dumping and sealed trace export are valid only with the "
            "V4 collector trace preflight"
        )
    if h5_sidecar_quality_preflight_only:
        if not v4_substep_collector_trace_preflight_only:
            raise ValueError(
                "pure H5 sidecar preflight requires the V4 collector trace preflight"
            )
        if args.platform != "cpu":
            raise ValueError(
                "pure H5 sidecar preflight is CPU-only; GPU collector parity is unresolved"
            )
        if args.h5_sidecar_quality_parent is None:
            raise ValueError(
                "pure H5 sidecar preflight requires a hash-bound CPU collector parent"
            )
    h5_v3_substep_t1_diagnostic_arm = getattr(
        args, "h5_v3_substep_contact_t1_diagnostic_arm", None
    )
    if h5_v3_substep_t1_diagnostic_arm is not None:
        if not h5_v3_substep_contact_preflight_only:
            raise ValueError("H5 T=1 diagnostic requires no-PPO preflight mode")
        if args.h5_v3_substep_contact_t1_diagnostic_output is None:
            raise ValueError("H5 T=1 diagnostic requires an immutable output path")
    if getattr(args, "h5_v3_substep_contact_t1_fixed_quality_replay_ablation", False):
        if h5_v3_substep_t1_diagnostic_arm is None:
            raise ValueError("H5 fixed quality replay ablation requires a T=1 arm")
    if unified_command_mapper == "direct_normalized_v3" and (
        args.expert != "unified" or not diagnostic_reward_exploration
    ):
        raise ValueError(
            "direct H5 unified command mapping is diagnostic unified training only"
        )
    if getattr(args, "h5_v3_command_conditioned_se2_alignment", False):
        if (
            args.expert != "unified"
            or not diagnostic_reward_exploration
            or unified_command_mapper != "direct_normalized_v3"
            or unified_reverse_probability is not None
            or any(h5_seed_supplied)
            or getattr(args, "h5_seed_initialize_from_params", False)
        ):
            raise ValueError(
                "H5 V3 command-conditioned SE(2) alignment requires clean, "
                "balanced unified direct-V3 training without a teacher or seed"
            )
    if h5_v3_substep_contact_alignment:
        if not getattr(args, "h5_v3_command_conditioned_se2_alignment", False):
            raise ValueError(
                "H5 V3 substep contact alignment requires the H5 V3 SE(2) flag"
            )
        if (
            args.expert != "unified"
            or not diagnostic_reward_exploration
            or unified_command_mapper != "direct_normalized_v3"
            or unified_reverse_probability is not None
            or any(h5_seed_supplied)
            or getattr(args, "h5_seed_initialize_from_params", False)
        ):
            raise ValueError(
                "H5 V3 substep contact alignment requires clean, balanced unified "
                "direct-V3 training without a teacher or seed"
            )
        if h5_v3_substep_contact_preflight_only and (
            args.authorize_simulation_training or args.wiring_only
        ):
            raise ValueError(
                "H5 V3 substep preflight is no-PPO: do not authorize training "
                "or enable wiring-only"
            )
    if (
        not np.isfinite(float(args.h5_seed_residual_gain))
        or not 0.0 <= float(args.h5_seed_residual_gain) <= 1.0
    ):
        raise ValueError("--h5-seed-residual-gain must be finite and in [0, 1]")
    if args.h5_seed_teacher_mode != "table" and not all(h5_seed_supplied):
        raise ValueError(
            "non-table H5 seed teacher modes require an explicit seed bundle"
        )
    if args.h5_seed_teacher_mode == "adaptive_residual" and (
        float(args.h5_seed_residual_gain) <= 0.0
    ):
        raise ValueError(
            "adaptive_residual H5 seed teacher mode requires positive residual gain"
        )
    if getattr(args, "h5_seed_initialize_from_params", False) and not all(
        h5_seed_supplied
    ):
        raise ValueError(
            "--h5-seed-initialize-from-params requires a complete H5 seed bundle"
        )
    if getattr(args, "h5_seed_initialize_from_params", False) and args.expert != "unified":
        raise ValueError(
            "--h5-seed-initialize-from-params is restricted to the unified expert"
        )
    if any(h5_seed_supplied) and not all(h5_seed_supplied):
        raise ValueError(
            "all four H5 target-space seed params/manifest/hash arguments are required"
        )
    if all(h5_seed_supplied) and args.expert not in {"reverse", "unified"}:
        raise ValueError("H5 target-space seed is valid only for reverse or unified")
    if args.expert in {"planar", "unified"} and not diagnostic_reward_exploration:
        raise ValueError(
            "H5 planar/unified training is diagnostic-only until a dedicated H5 "
            "authorization is reviewed"
        )
    if diagnostic_reward_exploration:
        unified_development_run = bool(
            getattr(args, "unified_development_run", False)
        )
        if unified_development_run and args.expert != "unified":
            raise ValueError(
                "--unified-development-run is valid only for the unified expert"
            )
        if (
            not h5_v3_substep_contact_preflight_only
            and not v4_substep_collector_trace_preflight_only
            and not v4_authoritative_primitive_batch_parity_preflight_only
            and not v4_direct_primitive_isolation_preflight_only
            and not v4_host_synchronized_primitive_ladder_preflight_only
            and (not args.authorize_simulation_training or args.wiring_only)
        ):
            raise ValueError(
                "diagnostic reward exploration requires an authorized, non-wiring "
                "simulation run"
            )
        if iteration_mode:
            raise ValueError(
                "diagnostic reward exploration cannot be combined with an H4 "
                "iteration authorization"
            )
        allowed_diagnostic_timesteps = (
            (None, DEFAULT_PILOT_TIMESTEPS, PROMOTED_TIMESTEPS)
            if unified_development_run
            else (None, DEFAULT_PILOT_TIMESTEPS)
        )
        if args.num_timesteps not in allowed_diagnostic_timesteps:
            raise ValueError(
                "diagnostic reward exploration is one 250k pilot; unified "
                "development additionally permits one explicit 1M run"
            )
        if any(
            value is not None
            for value in (
                args.h4_parent_params,
                args.h4_parent_manifest,
                args.h4_parent_params_sha256,
                args.promotion_evidence,
            )
        ):
            raise ValueError(
                "diagnostic reward exploration must initialize from frozen v22 "
                "without H4-parent or promotion inputs"
            )
        if all(h5_seed_supplied) and args.observation_mode != "h4_116_transplant":
            raise ValueError("H5 target-space seed requires exact 116-wide observations")
    if args.forward_iteration_v2:
        if args.expert != "forward":
            raise ValueError("--forward-iteration-v2 is valid only for forward")
        if args.wiring_only and args.authorize_simulation_training:
            raise ValueError(
                "forward iteration-v2 wiring preflight must not use "
                "--authorize-simulation-training"
            )
        if not args.wiring_only and not args.authorize_simulation_training:
            raise ValueError(
                "forward iteration-v2 250k requires "
                "--authorize-simulation-training"
            )
        allowed_timesteps = (
            (None, WIRING_TIMESTEPS)
            if args.wiring_only
            else (None, DEFAULT_PILOT_TIMESTEPS)
        )
        if args.num_timesteps not in allowed_timesteps:
            raise ValueError(
                "forward iteration v2 is exactly 40 wiring interactions or "
                "one separate 250k pilot"
            )
        if args.seed != 20260809:
            raise ValueError("forward iteration v2 requires exact seed 20260809")
        if any(
            value is not None
            for value in (
                args.h4_parent_params,
                args.h4_parent_manifest,
                args.h4_parent_params_sha256,
                args.promotion_evidence,
            )
        ):
            raise ValueError(
                "forward iteration v2 must initialize from frozen v22 without "
                "H4-parent or promotion inputs"
            )
    if args.reverse_iteration_v2:
        if args.expert != "reverse":
            raise ValueError("--reverse-iteration-v2 is valid only for reverse")
        if args.wiring_only and args.authorize_simulation_training:
            raise ValueError(
                "reverse iteration-v2 wiring preflight must not use "
                "--authorize-simulation-training"
            )
        if not args.wiring_only and not args.authorize_simulation_training:
            raise ValueError(
                "reverse iteration-v2 250k requires "
                "--authorize-simulation-training"
            )
        allowed_timesteps = (
            (None, WIRING_TIMESTEPS)
            if args.wiring_only
            else (None, DEFAULT_PILOT_TIMESTEPS)
        )
        if args.num_timesteps not in allowed_timesteps:
            raise ValueError(
                "reverse iteration v2 is exactly 40 wiring interactions or "
                "one separate 250k pilot"
            )
        if args.seed != 20260810:
            raise ValueError("reverse iteration v2 requires exact seed 20260810")
        if any(
            value is not None
            for value in (
                args.h4_parent_params,
                args.h4_parent_manifest,
                args.h4_parent_params_sha256,
                args.promotion_evidence,
            )
        ):
            raise ValueError(
                "reverse iteration v2 must initialize from frozen v22 without "
                "H4-parent or promotion inputs"
            )
    for enabled, expert, flag_name, expected_seed, label in (
        (
            args.forward_iteration_v3_touchdown_balance,
            "forward",
            "--forward-iteration-v3-touchdown-balance",
            20260809,
            "forward iteration v3 touchdown balance",
        ),
        (
            args.reverse_iteration_v3_no_target_imitation,
            "reverse",
            "--reverse-iteration-v3-no-target-imitation",
            20260810,
            "reverse iteration v3 no target imitation",
        ),
        (
            args.forward_iteration_v4_contact_event_validity_persistence,
            "forward",
            "--forward-iteration-v4-contact-event-validity-persistence",
            20260809,
            "forward iteration v4 contact event validity/persistence",
        ),
        (
            args.reverse_iteration_v4_residual_transfer_gain_024,
            "reverse",
            "--reverse-iteration-v4-residual-transfer-gain-024",
            20260810,
            "reverse iteration v4 residual transfer gain 0.24",
        ),
        (
            args.forward_v5_contact_pulse_abort_scale_only,
            "forward",
            "--forward-v5-contact-pulse-abort-scale-only",
            20260809,
            "forward v5 contact pulse abort scale only",
        ),
        (
            args.forward_iteration_v6_contact_abort_island_only,
            "forward",
            "--forward-iteration-v6-contact-abort-island-only",
            20260809,
            "forward iteration v6 contact abort island only",
        ),
        (
            args.reverse_iteration_v5_no_contact_imitation,
            "reverse",
            "--reverse-iteration-v5-no-contact-imitation",
            20260810,
            "reverse iteration v5 no contact imitation",
        ),
        (
            args.reverse_iteration_v6_absolute_full_leg_targets,
            "reverse",
            "--reverse-iteration-v6-absolute-full-leg-targets",
            20260810,
            "reverse iteration v6 absolute full-leg targets",
        ),
    ):
        if not enabled:
            continue
        if args.expert != expert:
            raise ValueError(f"{flag_name} is valid only for {expert}")
        if args.wiring_only and args.authorize_simulation_training:
            raise ValueError(
                f"{label} wiring preflight must not use "
                "--authorize-simulation-training"
            )
        if not args.wiring_only and not args.authorize_simulation_training:
            raise ValueError(
                f"{label} 250k requires --authorize-simulation-training"
            )
        allowed_timesteps = (
            (None, WIRING_TIMESTEPS)
            if args.wiring_only
            else (None, DEFAULT_PILOT_TIMESTEPS)
        )
        if args.num_timesteps not in allowed_timesteps:
            raise ValueError(
                f"{label} is exactly 40 wiring interactions or one separate 250k pilot"
            )
        if args.seed != expected_seed:
            raise ValueError(f"{label} requires exact seed {expected_seed}")
        if not np.isclose(
            args.reset_noise_multiplier, 1.0, rtol=0.0, atol=0.0
        ):
            raise ValueError(f"{label} requires exact reset-noise multiplier 1.0")
        if any(
            value is not None
            for value in (
                args.h4_parent_params,
                args.h4_parent_manifest,
                args.h4_parent_params_sha256,
                args.promotion_evidence,
            )
        ):
            raise ValueError(
                f"{label} must initialize from frozen v22 without H4-parent or "
                "promotion inputs"
            )
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be finite and positive")
    if not np.isfinite(args.entropy_cost) or args.entropy_cost < 0.0:
        raise ValueError("--entropy-cost must be finite and non-negative")
    if not np.isfinite(args.clipping_epsilon) or not 0.0 < args.clipping_epsilon <= 0.3:
        raise ValueError("--clipping-epsilon must be in (0, 0.3]")
    if not np.isfinite(args.discounting) or not 0.0 < args.discounting <= 1.0:
        raise ValueError("--discounting must be in (0, 1]")
    if not np.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be finite and positive")
    if not np.isfinite(args.reset_noise_multiplier) or args.reset_noise_multiplier < 0.0:
        raise ValueError("--reset-noise-multiplier must be finite and non-negative")
    if args.reverse_iteration_v6_absolute_full_leg_targets:
        if not np.isclose(
            args.backward_residual_scale,
            REVERSE_ITERATION_V6_RESIDUAL_AUTHORITY_SCALE,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "reverse iteration v6 requires exact residual authority scale 0"
            )
    elif (
        not np.isfinite(args.backward_residual_scale)
        or not 0.0 < args.backward_residual_scale <= 0.25
    ):
        raise ValueError("--backward-residual-scale must be in (0, 0.25]")
    if (
        args.expert == "reverse"
        and not args.reverse_iteration_v4_residual_transfer_gain_024
        and not args.reverse_iteration_v6_absolute_full_leg_targets
        and args.backward_residual_scale > 0.12
    ):
        raise ValueError("authorized reverse composition caps residual scale at 0.12")
    if not isinstance(args.num_envs, int) or isinstance(args.num_envs, bool) or args.num_envs <= 0:
        raise ValueError("--num-envs must be a positive integer")
    if args.observation_mode == "h4_116_transplant" and not (
        args.allow_verified_v22_transplant
    ):
        raise ValueError(
            "116-wide observations require --allow-verified-v22-transplant"
        )
    if args.observation_mode == "legacy101" and args.allow_verified_v22_transplant:
        raise ValueError("transplant flag is invalid for legacy101 observations")
    if iteration_mode and args.observation_mode != "h4_116_transplant":
        raise ValueError("H4 iteration mode requires exact actor116 observations")
    if args.authorize_simulation_training and args.observation_mode != "h4_116_transplant":
        raise ValueError(
            "H4 pilot training requires verified 116-wide observations; "
            "legacy101 is wiring-only"
        )
    if args.h4_parent_params is not None and args.observation_mode != "h4_116_transplant":
        raise ValueError("trusted H4 parent params require 116-wide observations")
    if all(h5_seed_supplied) and args.observation_mode != "h4_116_transplant":
        raise ValueError("H5 target-space seed requires 116-wide observations")
    if all(h5_seed_supplied) and not diagnostic_reward_exploration:
        raise ValueError("H5 target-space seed is restricted to diagnostic reward exploration")
    if (
        args.authorize_simulation_training or iteration_wiring
    ) and args.expert == "forward":
        expected = {
            "learning_rate": 5.0e-5,
            "entropy_cost": 1.0e-3,
            "clipping_epsilon": 0.10,
            "discounting": 0.97,
            "max_grad_norm": 0.5,
        }
        actual = {name: float(getattr(args, name)) for name in expected}
        if any(
            not np.isclose(actual[name], value, rtol=0.0, atol=0.0)
            for name, value in expected.items()
        ):
            raise ValueError(
                f"forward H4 pilot optimizer contract drifted: {actual} != {expected}"
            )
    if (
        args.authorize_simulation_training or iteration_wiring
    ) and args.expert == "reverse":
        expected_residual_scale = (
            REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN
            if args.reverse_iteration_v4_residual_transfer_gain_024
            else REVERSE_ITERATION_V6_RESIDUAL_AUTHORITY_SCALE
            if args.reverse_iteration_v6_absolute_full_leg_targets
            else 0.12
        )
        if not np.isclose(
            args.backward_residual_scale,
            expected_residual_scale,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "reverse H4 execution requires exact residual scale "
                f"{expected_residual_scale}"
            )
        expected = {
            "learning_rate": 3.0e-5,
            "entropy_cost": 1.0e-3,
            "clipping_epsilon": 0.10,
            "discounting": 0.97,
            "max_grad_norm": 0.5,
        }
        actual = {name: float(getattr(args, name)) for name in expected}
        optimizer_drift = any(
            not np.isclose(actual[name], value, rtol=0.0, atol=0.0)
            for name, value in expected.items()
        )
        if optimizer_drift and not getattr(
            args, "diagnostic_reward_exploration", False
        ):
            raise ValueError(
                f"reverse H4 pilot optimizer contract drifted: {actual} != {expected}"
            )


def _validate_authorized_training_contract(
    args: argparse.Namespace,
    *,
    shape: Any,
    reward_scale_dict: Mapping[str, float],
    anchors: Mapping[str, Any],
    forward_spec: Mapping[str, Any] | None,
    forward_iteration_v2_authorization: Mapping[str, Any] | None,
    forward_iteration_v3_touchdown_balance_authorization: Mapping[
        str, Any
    ] | None = None,
    forward_iteration_v4_contact_event_validity_persistence_authorization: Mapping[
        str, Any
    ] | None = None,
    forward_v5_contact_pulse_abort_scale_only_authorization: Mapping[
        str, Any
    ] | None = None,
    forward_iteration_v6_contact_abort_island_only_authorization: Mapping[
        str, Any
    ] | None = None,
    reverse_spec: Mapping[str, Any] | None,
    reverse_authorization: Mapping[str, Any] | None,
    reverse_iteration_v2_authorization: Mapping[str, Any] | None,
    reverse_iteration_v3_no_target_imitation_authorization: Mapping[
        str, Any
    ] | None = None,
    reverse_iteration_v4_residual_transfer_gain_024_authorization: Mapping[
        str, Any
    ] | None = None,
    reverse_iteration_v5_no_contact_imitation_authorization: Mapping[
        str, Any
    ] | None = None,
    reverse_iteration_v6_absolute_full_leg_targets_authorization: Mapping[
        str, Any
    ] | None = None,
) -> None:
    """Prevent CLI overrides from weakening a pilot or iteration wiring contract."""

    iteration_mode = bool(
        args.forward_iteration_v2
        or args.forward_iteration_v3_touchdown_balance
        or args.forward_iteration_v4_contact_event_validity_persistence
        or args.forward_v5_contact_pulse_abort_scale_only
        or args.forward_iteration_v6_contact_abort_island_only
        or args.reverse_iteration_v2
        or args.reverse_iteration_v3_no_target_imitation
        or args.reverse_iteration_v4_residual_transfer_gain_024
        or args.reverse_iteration_v5_no_contact_imitation
        or args.reverse_iteration_v6_absolute_full_leg_targets
    )
    iteration_wiring = bool(iteration_mode and args.wiring_only)
    if not args.authorize_simulation_training and not iteration_wiring:
        return
    if args.wiring_only and args.authorize_simulation_training:
        raise ValueError("authorized pilot mode and --wiring-only are mutually exclusive")
    if args.observation_mode != "h4_116_transplant":
        raise ValueError("H4 iteration execution requires actor116 observations")
    if iteration_wiring:
        wiring_shape_checks = {
            "num_timesteps_40": shape.num_timesteps == WIRING_TIMESTEPS,
            "num_envs_2": shape.num_envs == 2,
            "interactions_per_training_step_40": (
                shape.interactions_per_training_step == WIRING_TIMESTEPS
            ),
            "expected_training_steps_1": shape.expected_training_steps == 1,
            "expected_optimizer_updates_2": (
                shape.expected_optimizer_updates == 2
            ),
        }
        if not all(wiring_shape_checks.values()):
            raise ValueError(
                "H4 iteration wiring shape drifted: "
                f"{wiring_shape_checks}"
            )
    elif shape.num_envs != 1250:
        raise ValueError("H4 250k pilot requires exactly 1250 environments")
    if (
        forward_iteration_v6_contact_abort_island_only_authorization is not None
        and not args.forward_iteration_v6_contact_abort_island_only
    ):
        raise ValueError("forward iteration-v6 authorization cannot bind another mode")
    if (
        reverse_iteration_v6_absolute_full_leg_targets_authorization is not None
        and not args.reverse_iteration_v6_absolute_full_leg_targets
    ):
        raise ValueError("reverse iteration-v6 authorization cannot bind another mode")
    if args.expert == "forward":
        if forward_spec is None or forward_spec["stale_width_declaration_detected"]:
            raise ValueError(
                "forward minimum spec is not training-ready: it must declare exact "
                "actor116/new15 compatibility"
            )
        if args.forward_iteration_v2:
            if forward_iteration_v2_authorization is None:
                raise ValueError(
                    "forward iteration v2 requires its pinned authorization"
                )
            expected_timesteps = (
                WIRING_TIMESTEPS
                if iteration_wiring
                else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "forward iteration v2 requires exactly "
                    f"{expected_timesteps} interactions in this execution mode"
                )
            expected_rewards = (
                forward_iteration_v2_reward_scales().as_reward_scale_dict()
            )
            expected_anchors = FORWARD_ITERATION_V2_ANCHOR_CONFIG
            if forward_iteration_v3_touchdown_balance_authorization is not None:
                raise ValueError(
                    "forward iteration-v3 authorization cannot bind a v2 run"
                )
            if (
                forward_iteration_v4_contact_event_validity_persistence_authorization
                is not None
            ):
                raise ValueError(
                    "forward iteration-v4 authorization cannot bind a v2 run"
                )
            if forward_v5_contact_pulse_abort_scale_only_authorization is not None:
                raise ValueError("forward iteration-v5 authorization cannot bind a v2 run")
        elif args.forward_iteration_v3_touchdown_balance:
            if forward_iteration_v3_touchdown_balance_authorization is None:
                raise ValueError(
                    "forward iteration v3 touchdown balance requires its pinned "
                    "authorization"
                )
            if forward_iteration_v2_authorization is not None:
                raise ValueError(
                    "forward iteration-v2 authorization cannot bind a v3 run"
                )
            if (
                forward_iteration_v4_contact_event_validity_persistence_authorization
                is not None
            ):
                raise ValueError(
                    "forward iteration-v4 authorization cannot bind a v3 run"
                )
            if forward_v5_contact_pulse_abort_scale_only_authorization is not None:
                raise ValueError("forward iteration-v5 authorization cannot bind a v3 run")
            expected_timesteps = (
                WIRING_TIMESTEPS if iteration_wiring else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "forward iteration v3 touchdown balance requires exactly "
                    f"{expected_timesteps} interactions in this execution mode"
                )
            expected_rewards = (
                forward_iteration_v3_touchdown_balance_reward_scales().as_reward_scale_dict()
            )
            expected_anchors = FORWARD_ITERATION_V2_ANCHOR_CONFIG
        elif args.forward_iteration_v4_contact_event_validity_persistence:
            if (
                forward_iteration_v4_contact_event_validity_persistence_authorization
                is None
            ):
                raise ValueError(
                    "forward iteration v4 contact event validity/persistence "
                    "requires its closed authorization"
                )
            if (
                forward_iteration_v2_authorization is not None
                or forward_iteration_v3_touchdown_balance_authorization is not None
                or forward_v5_contact_pulse_abort_scale_only_authorization is not None
            ):
                raise ValueError(
                    "prior forward iteration authorization cannot bind a v4 run"
                )
            expected_timesteps = (
                WIRING_TIMESTEPS if iteration_wiring else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "forward iteration v4 contact event validity/persistence "
                    f"requires exactly {expected_timesteps} interactions"
                )
            expected_rewards = (
                forward_iteration_v2_reward_scales().as_reward_scale_dict()
            )
            expected_anchors = FORWARD_ITERATION_V2_ANCHOR_CONFIG
        elif args.forward_v5_contact_pulse_abort_scale_only:
            if forward_v5_contact_pulse_abort_scale_only_authorization is None:
                raise ValueError(
                    "forward v5 contact pulse abort scale only requires its "
                    "immutable authorization"
                )
            if (
                forward_iteration_v2_authorization is not None
                or forward_iteration_v3_touchdown_balance_authorization is not None
                or forward_iteration_v4_contact_event_validity_persistence_authorization
                is not None
            ):
                raise ValueError(
                    "prior forward iteration authorization cannot bind a v5 run"
                )
            expected_timesteps = (
                WIRING_TIMESTEPS if iteration_wiring else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "forward v5 contact pulse abort scale only requires exactly "
                    f"{expected_timesteps} interactions"
                )
            expected_rewards = (
                forward_iteration_v5_contact_pulse_abort_scale_only_reward_scales()
                .as_reward_scale_dict()
            )
            expected_anchors = FORWARD_ITERATION_V2_ANCHOR_CONFIG
        elif args.forward_iteration_v6_contact_abort_island_only:
            if forward_iteration_v6_contact_abort_island_only_authorization is None:
                raise ValueError(
                    "forward iteration v6 contact abort island only requires its "
                    "byte-pinned authorization"
                )
            if any(
                value is not None
                for value in (
                    forward_iteration_v2_authorization,
                    forward_iteration_v3_touchdown_balance_authorization,
                    forward_iteration_v4_contact_event_validity_persistence_authorization,
                    forward_v5_contact_pulse_abort_scale_only_authorization,
                )
            ):
                raise ValueError(
                    "prior forward iteration authorization cannot bind a v6 run"
                )
            expected_timesteps = (
                WIRING_TIMESTEPS if iteration_wiring else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "forward iteration v6 contact abort island only requires exactly "
                    f"{expected_timesteps} interactions"
                )
            expected_rewards = (
                forward_iteration_v6_contact_abort_island_only_reward_scales()
                .as_reward_scale_dict()
            )
            expected_anchors = FORWARD_ITERATION_V2_ANCHOR_CONFIG
        else:
            if (
                forward_iteration_v2_authorization is not None
                or forward_iteration_v3_touchdown_balance_authorization is not None
                or forward_iteration_v4_contact_event_validity_persistence_authorization
                is not None
                or forward_v5_contact_pulse_abort_scale_only_authorization is not None
            ):
                raise ValueError(
                    "forward iteration authorization cannot bind a v1 run"
                )
            expected_rewards = H4QualityRewardScales().as_reward_scale_dict()
            expected_anchors = ANCHOR_CONFIGS[args.expert]
    elif args.expert == "reverse":
        if reverse_spec is None or reverse_authorization is None:
            raise ValueError(
                "reverse minimum spec and composition authorization are required"
            )
        if args.reverse_iteration_v2:
            if reverse_iteration_v2_authorization is None:
                raise ValueError(
                    "reverse iteration v2 requires its pinned authorization"
                )
            expected_timesteps = (
                WIRING_TIMESTEPS
                if iteration_wiring
                else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "reverse iteration v2 requires exactly "
                    f"{expected_timesteps} interactions in this execution mode"
                )
            expected_rewards = (
                reverse_iteration_v2_reward_scales().as_reward_scale_dict()
            )
            expected_anchors = REVERSE_ITERATION_V2_ANCHOR_CONFIG
            if reverse_iteration_v3_no_target_imitation_authorization is not None:
                raise ValueError(
                    "reverse iteration-v3 authorization cannot bind a v2 run"
                )
            if reverse_iteration_v4_residual_transfer_gain_024_authorization is not None:
                raise ValueError(
                    "reverse iteration-v4 authorization cannot bind a v2 run"
                )
            if reverse_iteration_v5_no_contact_imitation_authorization is not None:
                raise ValueError("reverse iteration-v5 authorization cannot bind a v2 run")
        elif args.reverse_iteration_v3_no_target_imitation:
            if reverse_iteration_v3_no_target_imitation_authorization is None:
                raise ValueError(
                    "reverse iteration v3 no target imitation requires its pinned "
                    "authorization"
                )
            if reverse_iteration_v2_authorization is not None:
                raise ValueError(
                    "reverse iteration-v2 authorization cannot bind a v3 run"
                )
            if reverse_iteration_v4_residual_transfer_gain_024_authorization is not None:
                raise ValueError(
                    "reverse iteration-v4 authorization cannot bind a v3 run"
                )
            if reverse_iteration_v5_no_contact_imitation_authorization is not None:
                raise ValueError("reverse iteration-v5 authorization cannot bind a v3 run")
            expected_timesteps = (
                WIRING_TIMESTEPS if iteration_wiring else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "reverse iteration v3 no target imitation requires exactly "
                    f"{expected_timesteps} interactions in this execution mode"
                )
            expected_rewards = (
                reverse_iteration_v2_reward_scales().as_reward_scale_dict()
            )
            expected_anchors = REVERSE_ITERATION_V2_ANCHOR_CONFIG
        elif args.reverse_iteration_v4_residual_transfer_gain_024:
            if reverse_iteration_v4_residual_transfer_gain_024_authorization is None:
                raise ValueError(
                    "reverse iteration v4 residual transfer gain 0.24 requires "
                    "its closed authorization"
                )
            if (
                reverse_iteration_v2_authorization is not None
                or reverse_iteration_v3_no_target_imitation_authorization is not None
                or reverse_iteration_v5_no_contact_imitation_authorization is not None
            ):
                raise ValueError(
                    "prior reverse iteration authorization cannot bind a v4 run"
                )
            expected_timesteps = (
                WIRING_TIMESTEPS if iteration_wiring else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "reverse iteration v4 residual transfer gain 0.24 requires "
                    f"exactly {expected_timesteps} interactions"
                )
            expected_rewards = (
                reverse_iteration_v2_reward_scales().as_reward_scale_dict()
            )
            expected_anchors = REVERSE_ITERATION_V2_ANCHOR_CONFIG
        elif args.reverse_iteration_v5_no_contact_imitation:
            if reverse_iteration_v5_no_contact_imitation_authorization is None:
                raise ValueError(
                    "reverse iteration v5 no contact imitation requires its "
                    "immutable authorization"
                )
            if (
                reverse_iteration_v2_authorization is not None
                or reverse_iteration_v3_no_target_imitation_authorization is not None
                or reverse_iteration_v4_residual_transfer_gain_024_authorization
                is not None
            ):
                raise ValueError(
                    "prior reverse iteration authorization cannot bind a v5 run"
                )
            expected_timesteps = (
                WIRING_TIMESTEPS if iteration_wiring else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "reverse iteration v5 no contact imitation requires exactly "
                    f"{expected_timesteps} interactions"
                )
            expected_rewards = reverse_iteration_v2_reward_scales().as_reward_scale_dict()
            expected_anchors = REVERSE_ITERATION_V2_ANCHOR_CONFIG
        elif args.reverse_iteration_v6_absolute_full_leg_targets:
            if reverse_iteration_v6_absolute_full_leg_targets_authorization is None:
                raise ValueError(
                    "reverse iteration v6 absolute full-leg targets requires its "
                    "byte-pinned authorization"
                )
            if any(
                value is not None
                for value in (
                    reverse_iteration_v2_authorization,
                    reverse_iteration_v3_no_target_imitation_authorization,
                    reverse_iteration_v4_residual_transfer_gain_024_authorization,
                    reverse_iteration_v5_no_contact_imitation_authorization,
                )
            ):
                raise ValueError(
                    "prior reverse iteration authorization cannot bind a v6 run"
                )
            expected_timesteps = (
                WIRING_TIMESTEPS if iteration_wiring else DEFAULT_PILOT_TIMESTEPS
            )
            if shape.num_timesteps != expected_timesteps:
                raise ValueError(
                    "reverse iteration v6 absolute full-leg targets requires exactly "
                    f"{expected_timesteps} interactions"
                )
            expected_rewards = reverse_iteration_v2_reward_scales().as_reward_scale_dict()
            expected_anchors = REVERSE_ITERATION_V2_ANCHOR_CONFIG
        else:
            if (
                reverse_iteration_v2_authorization is not None
                or reverse_iteration_v3_no_target_imitation_authorization is not None
                or reverse_iteration_v4_residual_transfer_gain_024_authorization
                is not None
                or reverse_iteration_v5_no_contact_imitation_authorization is not None
            ):
                raise ValueError(
                    "reverse iteration authorization cannot bind a v1 run"
                )
            expected_rewards = H4QualityRewardScales(
                target_lag=0.0,
                left_target_lag=-0.125,
                right_target_lag=-0.125,
                reverse_speed_boundary=-1.0,
                reverse_cross_drift=-2.0,
                reverse_uncommanded_yaw_rate=-1.0,
                reverse_heading_drift=-1.0,
                reverse_phase_force_slip=-1.0,
                reverse_contact_priority_reversal_lag=-0.75,
            ).as_reward_scale_dict()
            expected_anchors = ANCHOR_CONFIGS[args.expert]
    else:
        if not getattr(args, "diagnostic_reward_exploration", False):
            raise ValueError(
                "H5 planar execution is available only as diagnostic reward exploration"
            )
        if any(
            value is not None
            for value in (
                forward_iteration_v2_authorization,
                forward_iteration_v3_touchdown_balance_authorization,
                forward_iteration_v4_contact_event_validity_persistence_authorization,
                forward_v5_contact_pulse_abort_scale_only_authorization,
                forward_iteration_v6_contact_abort_island_only_authorization,
                reverse_iteration_v2_authorization,
                reverse_iteration_v3_no_target_imitation_authorization,
                reverse_iteration_v4_residual_transfer_gain_024_authorization,
                reverse_iteration_v5_no_contact_imitation_authorization,
                reverse_iteration_v6_absolute_full_leg_targets_authorization,
            )
        ):
            raise ValueError("H5 planar diagnostic cannot bind H4 iteration authorization")
        expected_rewards = H4QualityRewardScales().as_reward_scale_dict()
        expected_anchors = ANCHOR_CONFIGS[args.expert]
    anchor_checks = {
        key: np.array_equal(np.asarray(anchors[key]), np.asarray(value))
        for key, value in expected_anchors.items()
    }
    if not all(anchor_checks.values()):
        raise ValueError(f"H4 pilot anchor contract drifted: {anchor_checks}")
    reward_checks = {
        key: key in reward_scale_dict
        and np.isclose(float(reward_scale_dict[key]), float(value), rtol=0.0, atol=0.0)
        for key, value in expected_rewards.items()
    }
    reward_checks["no_extra_scales"] = set(reward_scale_dict) == set(expected_rewards)
    if not all(reward_checks.values()) and not getattr(
        args, "diagnostic_reward_exploration", False
    ):
        raise ValueError(f"H4 pilot reward contract drifted: {reward_checks}")
    if not np.isclose(args.reset_noise_multiplier, 1.0, rtol=0.0, atol=0.0):
        raise ValueError("H4 pilot reset-noise multiplier must be exactly 1.0")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.learning_rate is None:
        args.learning_rate = (
            5.0e-5
            if args.expert in {"forward", "planar", "unified"}
            else DEFAULT_LEARNING_RATE
        )
    _validate_scalar_configuration(args)
    h5_v3_substep_contact_alignment = bool(
        getattr(args, "h5_v3_substep_contact_alignment", False)
    )
    h5_v3_substep_contact_preflight_only = bool(
        getattr(args, "h5_v3_substep_contact_preflight_only", False)
    )
    v4_substep_collector_trace_preflight_only = bool(
        getattr(args, "v4_substep_collector_trace_preflight_only", False)
    )
    v4_authoritative_primitive_batch_parity_preflight_only = bool(
        getattr(args, "v4_authoritative_primitive_batch_parity_preflight_only", False)
    )
    v4_direct_primitive_isolation_preflight_only = bool(
        getattr(args, "v4_direct_primitive_isolation_preflight_only", False)
    )
    v4_host_synchronized_primitive_ladder_preflight_only = bool(
        getattr(args, "v4_host_synchronized_primitive_ladder_preflight_only", False)
    )
    if h5_v3_substep_contact_alignment and not h5_v3_substep_contact_preflight_only:
        raise ValueError(
            "H5 V3 substep contact PPO is disabled until its separate, "
            "hash-bound authorization is created from the no-PPO preflight"
        )
    h5_v3_se2_authorization = (
        load_h5_v3_command_conditioned_se2_alignment_authorization(
            Path(args.h5_v3_se2_authorization)
        )
        if (
            getattr(args, "h5_v3_command_conditioned_se2_alignment", False)
            and not h5_v3_substep_contact_alignment
        )
        else None
    )
    iteration_v6_core_source = (
        require_iteration_v6_core_source()
        if (
            args.forward_iteration_v6_contact_abort_island_only
            or args.reverse_iteration_v6_absolute_full_leg_targets
        )
        else None
    )
    trainer = _load_legacy_trainer()
    shape = resolve_training_shape(args, trainer)
    if h5_v3_se2_authorization is not None or h5_v3_substep_contact_alignment:
        h5_v3_expected_shape = (
            H5_V3_SUBSTEP_PREFLIGHT_SHAPE
            if h5_v3_substep_contact_preflight_only
            else H5_V3_PRODUCTION_PILOT_SHAPE
        )
        h5_v3_actual_shape = (
            shape.num_timesteps,
            shape.num_envs,
            shape.unroll_length,
            shape.batch_size,
            shape.num_minibatches,
            shape.num_updates_per_batch,
        )
        if h5_v3_actual_shape != h5_v3_expected_shape or (
            float(args.learning_rate),
            float(args.entropy_cost),
            float(args.clipping_epsilon),
            float(args.discounting),
            float(args.max_grad_norm),
        ) != (5.0e-5, 0.001, 0.1, 0.97, 0.5):
            raise ValueError("H5 V3 SE(2) optimizer/shape contract drifted")
    reward_scales = resolve_reward_scales(args)
    reward_scale_dict = reward_scales.as_reward_scale_dict(
        include_h5_substep_contact_alignment=h5_v3_substep_contact_alignment
    )
    anchors = resolve_anchor_config(
        args.expert,
        physical_anchor_override=args.physical_anchor,
        policy_anchor_override=args.policy_observation_anchor,
        forward_iteration_v2=args.forward_iteration_v2,
        forward_iteration_v3_touchdown_balance=(
            args.forward_iteration_v3_touchdown_balance
        ),
        forward_iteration_v4_contact_event_validity_persistence=(
            args.forward_iteration_v4_contact_event_validity_persistence
        ),
        forward_v5_contact_pulse_abort_scale_only=(
            args.forward_v5_contact_pulse_abort_scale_only
        ),
        forward_iteration_v6_contact_abort_island_only=(
            args.forward_iteration_v6_contact_abort_island_only
        ),
        reverse_iteration_v2=args.reverse_iteration_v2,
        reverse_iteration_v3_no_target_imitation=(
            args.reverse_iteration_v3_no_target_imitation
        ),
        reverse_iteration_v4_residual_transfer_gain_024=(
            args.reverse_iteration_v4_residual_transfer_gain_024
        ),
        reverse_iteration_v5_no_contact_imitation=(
            args.reverse_iteration_v5_no_contact_imitation
        ),
        reverse_iteration_v6_absolute_full_leg_targets=(
            args.reverse_iteration_v6_absolute_full_leg_targets
        ),
    )
    h4_parent = load_trusted_h4_parent_bundle(
        params_path=args.h4_parent_params,
        manifest_path=args.h4_parent_manifest,
        expected_params_sha256=args.h4_parent_params_sha256,
    )
    h5_seed = load_h5_targetspace_seed_bundle(
        params_path=getattr(args, "h5_seed_params", None),
        manifest_path=getattr(args, "h5_seed_manifest", None),
        expected_params_sha256=getattr(args, "h5_seed_params_sha256", None),
        expected_manifest_sha256=getattr(args, "h5_seed_manifest_sha256", None),
        teacher_mode=getattr(args, "h5_seed_teacher_mode", "table"),
    )
    h5_teacher_only = (
        h5_seed is not None
        and args.expert == "unified"
        and not getattr(args, "h5_seed_initialize_from_params", False)
    )
    if h4_parent is not None and h5_seed is not None:
        raise ValueError("H4 parent and H5 target-space seed are mutually exclusive")
    if h4_parent is not None and h4_parent["manifest"].get("expert") != args.expert:
        raise ValueError("trusted H4 parent expert does not match requested expert")
    if (
        not args.wiring_only
        and shape.num_timesteps == DEFAULT_PILOT_TIMESTEPS
        and h4_parent is not None
    ):
        raise ValueError("250k H4 pilot must initialize from frozen v22, not H4 params")
    promotion = validate_promotion_evidence(
        args.promotion_evidence, h4_parent=h4_parent
    )
    if (
        shape.num_timesteps == PROMOTED_TIMESTEPS
        and not getattr(args, "unified_development_run", False)
        and promotion is None
    ):
        raise ValueError("1M promotion evidence is required")
    if (
        shape.num_timesteps == PROMOTED_TIMESTEPS
        and not getattr(args, "unified_development_run", False)
        and h4_parent is None
    ):
        raise ValueError("1M promotion requires a trusted 250k H4 parent")
    if (
        shape.num_timesteps == PROMOTED_TIMESTEPS
        and not getattr(args, "unified_development_run", False)
    ):
        assert h4_parent is not None
        parent_manifest = h4_parent["manifest"]
        if (
            parent_manifest.get("status") != "COMPLETED"
            or parent_manifest.get("requested_environment_interactions")
            != DEFAULT_PILOT_TIMESTEPS
        ):
            raise ValueError("1M promotion parent must be a completed 250k H4 pilot")
    selected = (
        load_selected_reverse_teacher(args.selected_reverse_teacher)
        if args.expert == "reverse"
        else None
    )
    forward_spec = load_forward_minimum_spec() if args.expert == "forward" else None
    forward_iteration_v2_authorization = (
        load_forward_iteration_v2_authorization()
        if args.forward_iteration_v2
        else None
    )
    forward_iteration_v3_touchdown_balance_authorization = (
        load_forward_iteration_v3_touchdown_balance_authorization()
        if args.forward_iteration_v3_touchdown_balance
        else None
    )
    forward_iteration_v4_contact_event_validity_persistence_authorization = (
        load_forward_iteration_v4_contact_event_validity_persistence_authorization()
        if args.forward_iteration_v4_contact_event_validity_persistence
        else None
    )
    forward_v5_contact_pulse_abort_scale_only_authorization = (
        load_forward_iteration_v5_contact_pulse_abort_scale_only_authorization()
        if args.forward_v5_contact_pulse_abort_scale_only
        else None
    )
    forward_iteration_v6_contact_abort_island_only_authorization = (
        load_forward_iteration_v6_contact_abort_island_only_authorization()
        if args.forward_iteration_v6_contact_abort_island_only
        else None
    )
    reverse_spec = load_reverse_minimum_spec() if args.expert == "reverse" else None
    reverse_authorization = (
        load_reverse_composition_authorization()
        if args.expert == "reverse"
        else None
    )
    reverse_iteration_v2_authorization = (
        load_reverse_iteration_v2_authorization()
        if args.reverse_iteration_v2
        else None
    )
    reverse_iteration_v3_no_target_imitation_authorization = (
        load_reverse_iteration_v3_no_target_imitation_authorization()
        if args.reverse_iteration_v3_no_target_imitation
        else None
    )
    reverse_iteration_v4_residual_transfer_gain_024_authorization = (
        load_reverse_iteration_v4_residual_transfer_gain_024_authorization()
        if args.reverse_iteration_v4_residual_transfer_gain_024
        else None
    )
    reverse_iteration_v5_no_contact_imitation_authorization = (
        load_reverse_iteration_v5_no_contact_imitation_authorization()
        if args.reverse_iteration_v5_no_contact_imitation
        else None
    )
    reverse_iteration_v6_absolute_full_leg_targets_authorization = (
        load_reverse_iteration_v6_absolute_full_leg_targets_authorization()
        if args.reverse_iteration_v6_absolute_full_leg_targets
        else None
    )
    startup_audit = reverse_teacher_startup_audit(selected) if selected else None
    _validate_authorized_training_contract(
        args,
        shape=shape,
        reward_scale_dict=reward_scale_dict,
        anchors=anchors,
        forward_spec=forward_spec,
        forward_iteration_v2_authorization=(
            forward_iteration_v2_authorization
        ),
        forward_iteration_v3_touchdown_balance_authorization=(
            forward_iteration_v3_touchdown_balance_authorization
        ),
        forward_iteration_v4_contact_event_validity_persistence_authorization=(
            forward_iteration_v4_contact_event_validity_persistence_authorization
        ),
        forward_v5_contact_pulse_abort_scale_only_authorization=(
            forward_v5_contact_pulse_abort_scale_only_authorization
        ),
        forward_iteration_v6_contact_abort_island_only_authorization=(
            forward_iteration_v6_contact_abort_island_only_authorization
        ),
        reverse_spec=reverse_spec,
        reverse_authorization=reverse_authorization,
        reverse_iteration_v2_authorization=(
            reverse_iteration_v2_authorization
        ),
        reverse_iteration_v3_no_target_imitation_authorization=(
            reverse_iteration_v3_no_target_imitation_authorization
        ),
        reverse_iteration_v4_residual_transfer_gain_024_authorization=(
            reverse_iteration_v4_residual_transfer_gain_024_authorization
        ),
        reverse_iteration_v5_no_contact_imitation_authorization=(
            reverse_iteration_v5_no_contact_imitation_authorization
        ),
        reverse_iteration_v6_absolute_full_leg_targets_authorization=(
            reverse_iteration_v6_absolute_full_leg_targets_authorization
        ),
    )

    source_root = args.source_root.resolve()
    generated_root = args.generated_root.resolve()
    output_root = args.output_root.resolve()
    parent_checkpoint = args.parent_checkpoint.resolve()
    if h4_parent is None and not parent_checkpoint.exists():
        if h5_seed is None or h5_teacher_only:
            raise FileNotFoundError(f"parent checkpoint is missing: {parent_checkpoint}")
    paths = trainer.generated_paths(generated_root)
    trainer._validate_generated_manifest(paths)
    pinned_v22_tree_sha = None
    if h4_parent is None and (h5_seed is None or h5_teacher_only):
        pinned_v22_tree_sha = trainer.sha256_tree(parent_checkpoint)
        if pinned_v22_tree_sha != PINNED_V22_PARENT_TREE_SHA256:
            raise ValueError(
                "frozen v22 parent checkpoint tree SHA drifted: "
                f"{pinned_v22_tree_sha}"
            )
    initialization_artifact_path = (
        h5_seed["params_path"]
        if h5_seed is not None and not h5_teacher_only
        else h4_parent["params_path"]
        if h4_parent is not None
        else parent_checkpoint
    )
    initialization_artifact_sha256 = (
        h5_seed["params_sha256"]
        if h5_seed is not None and not h5_teacher_only
        else h4_parent["params_sha256"]
        if h4_parent is not None
        else pinned_v22_tree_sha
    )

    class TeacherArgs:
        backward_gait = None
        backward_left_gait = None
        backward_right_gait = None

    legacy_teacher_gaits = trainer.resolve_teacher_gaits(TeacherArgs(), paths)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    backend_selector = resolve_jax_backend_selector(args.platform)
    os.environ["JAX_PLATFORMS"] = backend_selector
    xla_autotune_policy = configure_xla_autotune_policy(args.platform)
    stack = trainer._load_training_stack(source_root)

    backend_resolution = validate_resolved_jax_backend(
        stack["jax"],
        requested_platform=args.platform,
        selector=backend_selector,
    )
    debug_callback_preflight = run_jax_debug_callback_preflight(
        stack["jax"], stack["jp"]
    )

    constants = stack["constants"]
    scene_type = type(constants.FLAT_TERRAIN_BACKLASH_CALIBRATED_XML)
    constants.ROOT_PATH = scene_type(paths["package"].as_posix())
    constants.FLAT_TERRAIN_BACKLASH_CALIBRATED_XML = scene_type(
        paths["scene"].as_posix()
    )
    resolved_scene = Path(constants.task_to_xml(trainer.TASK).as_posix()).resolve()
    if resolved_scene != paths["scene"].resolve():
        raise RuntimeError("generated H4 scene redirect failed")

    jax = stack["jax"]
    jp = stack["jp"]
    physical_sampler_family = resolve_physical_sampler_family(args)
    if physical_sampler_family == "forward_iteration_v2":
        physical_sampler = make_h4_forward_v2_physical_sampler(jax, jp)
    elif physical_sampler_family == "reverse_iteration_v2":
        physical_sampler = make_h4_reverse_v2_physical_sampler(jax, jp)
    elif physical_sampler_family == "forward":
        physical_sampler = make_h4_forward_physical_sampler(jax, jp)
    elif physical_sampler_family == "planar":
        physical_sampler = make_h5_planar_physical_sampler(jax, jp)
    elif physical_sampler_family == "unified":
        physical_sampler = make_h5_unified_physical_sampler(
            jax,
            jp,
            reverse_route_probability=getattr(
                args, "h5_unified_reverse_route_probability", None
            ),
        )
    elif args.expert == "reverse" and getattr(
        args, "diagnostic_reward_exploration", False
    ):
        physical_sampler = make_h5_reverse_physical_sampler(jax, jp)
    else:
        physical_sampler = make_h4_reverse_physical_sampler(jax, jp)
    mapper = (
        make_h5_planar_command_mapper(jax, jp)
        if args.expert == "planar"
        else make_h5_unified_command_mapper(
            jax,
            jp,
            mapper_mode=canonical_h5_unified_command_mapper(
                str(args.h5_unified_command_mapper)
            ),
        )
        if args.expert == "unified"
        else make_h5_reverse_command_mapper(jax, jp)
        if args.expert == "reverse"
        and getattr(args, "diagnostic_reward_exploration", False)
        else make_anchor_command_mapper(
            anchors["physical_primary"], anchors["policy_observation_anchor"], xp=jp
        )
    )
    LegacyEnvironment = trainer._make_environment_class(
        stack=stack,
        expert=("compound" if args.expert in {"planar", "unified"} else args.expert),
        paths=paths,
        teacher_gaits=legacy_teacher_gaits,
        backward_residual_scale=args.backward_residual_scale,
    )
    include_h4_observables = args.observation_mode == "h4_116_transplant"
    h5_legacy_reward_config = None
    if getattr(args, "diagnostic_reward_exploration", False) and args.expert == "planar":
        h5_legacy_reward_config = dict(H5_PLANAR_DIAGNOSTIC_LEGACY_REWARD_CONFIG)
    elif getattr(args, "diagnostic_reward_exploration", False) and args.expert == "unified":
        h5_legacy_reward_config = dict(
            H5_TARGET_SPACE_DIAGNOSTIC_LEGACY_REWARD_CONFIG
            if h5_seed is not None
            else H5_PLANAR_DIAGNOSTIC_LEGACY_REWARD_CONFIG
        )
        if h5_seed is not None:
            h5_legacy_reward_config["target_imitation"] = -float(
                args.h5_seed_bc_scale
            )
    elif getattr(args, "diagnostic_reward_exploration", False) and args.expert == "reverse":
        h5_legacy_reward_config = dict(H5_TARGET_SPACE_DIAGNOSTIC_LEGACY_REWARD_CONFIG)
        if h5_seed is not None:
            # The component is a positive squared target-space error; the
            # legacy negative scale turns it into an annealed BC cost.
            h5_legacy_reward_config["target_imitation"] = -float(
                args.h5_seed_bc_scale
            )
    elif args.reverse_iteration_v6_absolute_full_leg_targets:
        h5_legacy_reward_config = REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_LEGACY_REWARD_CONFIG
    elif args.reverse_iteration_v5_no_contact_imitation:
        h5_legacy_reward_config = REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_LEGACY_REWARD_CONFIG
    elif args.reverse_iteration_v3_no_target_imitation or args.reverse_iteration_v4_residual_transfer_gain_024:
        h5_legacy_reward_config = REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG
    elif args.reverse_iteration_v2:
        h5_legacy_reward_config = REVERSE_ITERATION_V2_LEGACY_REWARD_CONFIG
    environment_kwargs = dict(
        legacy_environment_class=LegacyEnvironment,
        stack=stack,
        physical_command_sampler=physical_sampler,
        policy_observation_mapper=mapper,
        reward_scales=reward_scales,
        reset_noise_multiplier=args.reset_noise_multiplier,
         reverse_teacher_cycle_hz=(
             1.5
             if getattr(args, "diagnostic_reward_exploration", False)
             and args.expert in {"reverse", "unified"}
             else selected["cadence_hz"]
             if selected
             else 1.75
         ),
        reverse_teacher_target_table=(selected["table"] if selected else None),
        reverse_teacher_phase_advance_bins=(
            selected["phase_advance_bins"] if selected else None
        ),
        reverse_teacher_entry_phase_bins=(
            selected["entry_phase_bins"] if selected else 0.0
        ),
        include_h4_actor_observables=include_h4_observables,
        forward_v4_substep_contact=bool(
            args.forward_iteration_v4_contact_event_validity_persistence
            or args.forward_v5_contact_pulse_abort_scale_only
            or args.forward_iteration_v6_contact_abort_island_only
            or (
                getattr(args, "diagnostic_reward_exploration", False)
                and args.expert in {"planar", "reverse", "unified"}
            )
        ),
        v4_substep_collector_trace_capture=(
            v4_substep_collector_trace_preflight_only
        ),
        forward_iteration_v6_contact_abort_island_only=bool(
            args.forward_iteration_v6_contact_abort_island_only
        ),
        reverse_iteration_v6_absolute_full_leg_targets=bool(
            args.reverse_iteration_v6_absolute_full_leg_targets
        ),
        h5_absolute_target_routing=bool(
            getattr(args, "diagnostic_reward_exploration", False)
            and args.expert in {"planar", "reverse", "unified"}
        ),
        h5_target_domain=(
            "planar"
            if args.expert == "planar"
            else "reverse"
            if args.expert == "reverse"
            and getattr(args, "diagnostic_reward_exploration", False)
            else "unified"
            if args.expert == "unified"
            and getattr(args, "diagnostic_reward_exploration", False)
            else None
        ),
        h5_v3_command_conditioned_se2_alignment=bool(
            getattr(args, "h5_v3_command_conditioned_se2_alignment", False)
        ),
        h5_v3_substep_contact_alignment=h5_v3_substep_contact_alignment,
        h5_v3_substep_preflight_telemetry=(
            h5_v3_substep_contact_preflight_only
        ),
        h5_v3_substep_preflight_fixed_quality_replay=bool(
            getattr(
                args,
                "h5_v3_substep_contact_t1_fixed_quality_replay_ablation",
                False,
            )
        ),
        legacy_reward_config_overrides=h5_legacy_reward_config,
        h5_seed_params=(
            h5_seed["teacher_params"]
            if h5_seed is not None
            and getattr(args, "h5_seed_teacher_mode", "table")
            == "adaptive_residual"
            else h5_seed["params"]
            if h5_seed is not None
            else None
        ),
        h5_seed_target_table=(
            h5_seed["target_table"] if h5_seed is not None else None
        ),
        h5_seed_bc_anneal_control_steps=float(
            args.h5_seed_bc_anneal_control_steps
        ),
        h5_seed_teacher_mode=getattr(args, "h5_seed_teacher_mode", "table"),
        h5_seed_residual_gain=float(getattr(args, "h5_seed_residual_gain", 0.0)),
        h5_seed_teacher_reverse_command_contract=bool(
            getattr(args, "h5_seed_teacher_reverse_command_contract", False)
        ),
    )
    Environment = make_h4_aligned_environment_class(**environment_kwargs)
    t1_diagnostic_arm = getattr(
        args, "h5_v3_substep_contact_t1_diagnostic_arm", None
    )
    # A fresh-process T=1 control diagnostic must not instantiate the
    # treatment class at all; otherwise the global physics monkeypatch can
    # contaminate the very isolation test intended to detect it.
    env = None if t1_diagnostic_arm == "control" else Environment()
    if v4_authoritative_primitive_batch_parity_preflight_only:
        if env is None:
            raise RuntimeError("V4 primitive batch-parity environment is unavailable")
        joystick_module = stack["joystick"]
        mjx_env_module = getattr(joystick_module, "mjx_env", None)
        mjx_model = getattr(env, "mjx_model", None)
        if mjx_env_module is None or mjx_model is None:
            raise RuntimeError(
                "V4 primitive batch-parity environment lacks MJX provenance"
            )
        return run_v4_authoritative_primitive_batch_parity_preflight(
            args=args,
            capture_env=env,
            model=mjx_model,
            mjx_step=mjx_env_module.mjx.step,
            jax=jax,
            jp=jp,
            backend_resolution=backend_resolution,
            runtime_versions=trainer._runtime_versions(),
            joystick_module=joystick_module,
            mjx_env_module=mjx_env_module,
            source_paths={
                "h4_training_alignment": ALIGNMENT_MODULE_PATH,
                "runner": Path(__file__).resolve(),
                "generated_scene": paths["scene"],
                "generated_manifest": paths["manifest"],
                "source_joystick": Path(joystick_module.__file__).resolve(),
                "source_mjx_env": Path(mjx_env_module.__file__).resolve(),
            },
        )
    if (
        v4_direct_primitive_isolation_preflight_only
        or v4_host_synchronized_primitive_ladder_preflight_only
    ):
        if env is None:
            raise RuntimeError("V4 direct-primitive diagnostic environment is unavailable")
        joystick_module = stack["joystick"]
        mjx_env_module = getattr(joystick_module, "mjx_env", None)
        mjx_model = getattr(env, "mjx_model", None)
        if mjx_env_module is None or mjx_model is None:
            raise RuntimeError(
                "V4 direct-primitive diagnostic environment lacks MJX provenance"
            )
        return run_v4_direct_primitive_isolation_preflight(
            args=args,
            capture_env=env,
            model=mjx_model,
            mjx_step=mjx_env_module.mjx.step,
            jax=jax,
            jp=jp,
            backend_resolution=backend_resolution,
            runtime_versions=trainer._runtime_versions(),
            joystick_module=joystick_module,
            mjx_env_module=mjx_env_module,
            source_paths={
                "h4_training_alignment": ALIGNMENT_MODULE_PATH,
                "runner": Path(__file__).resolve(),
                "generated_scene": paths["scene"],
                "generated_manifest": paths["manifest"],
                "source_joystick": Path(joystick_module.__file__).resolve(),
                "source_mjx_env": Path(mjx_env_module.__file__).resolve(),
            },
        )
    eval_env = (
        None
        if t1_diagnostic_arm is not None
        else Environment()
    )
    h5_v3_se2_reward_only_parity_preflight = None
    if h5_v3_se2_authorization is not None:
        baseline_environment_kwargs = dict(environment_kwargs)
        baseline_environment_kwargs["h5_v3_command_conditioned_se2_alignment"] = False
        baseline_env = make_h4_aligned_environment_class(
            **baseline_environment_kwargs
        )()
        parity_state = env.reset(jax.random.PRNGKey(args.seed + 17))
        baseline_state = baseline_env.reset(jax.random.PRNGKey(args.seed + 17))
        checks = {
            "reset_state_observation": np.array_equal(
                np.asarray(parity_state.obs["state"]),
                np.asarray(baseline_state.obs["state"]),
            ),
            "reset_privileged_observation": np.array_equal(
                np.asarray(parity_state.obs["privileged_state"]),
                np.asarray(baseline_state.obs["privileged_state"]),
            ),
            "reset_qpos": np.array_equal(
                np.asarray(parity_state.data.qpos), np.asarray(baseline_state.data.qpos)
            ),
            "reset_qvel": np.array_equal(
                np.asarray(parity_state.data.qvel), np.asarray(baseline_state.data.qvel)
            ),
        }
        parity_action = jp.zeros(
            len(ACTUATOR_JOINT_ORDER), dtype=parity_state.data.qpos.dtype
        )
        for step_index in range(4):
            parity_state = env.step(parity_state, parity_action)
            baseline_state = baseline_env.step(baseline_state, parity_action)
            for label, aligned_value, baseline_value in (
                ("state_observation", parity_state.obs["state"], baseline_state.obs["state"]),
                (
                    "privileged_observation",
                    parity_state.obs["privileged_state"],
                    baseline_state.obs["privileged_state"],
                ),
                ("target_ctrl", parity_state.data.ctrl, baseline_state.data.ctrl),
                ("qpos", parity_state.data.qpos, baseline_state.data.qpos),
                ("qvel", parity_state.data.qvel, baseline_state.data.qvel),
                ("done", parity_state.done, baseline_state.done),
            ):
                checks[f"step_{step_index}_{label}"] = np.array_equal(
                    np.asarray(aligned_value), np.asarray(baseline_value)
                )
        if not all(checks.values()):
            raise RuntimeError("H5 V3 SE(2) reward-only parity preflight failed")
        h5_v3_se2_reward_only_parity_preflight = {
            "status": "PASS",
            "control_steps": 4,
            "same_reset_seed": args.seed + 17,
            "same_action": "exact_zero_14wide",
            "checked": checks,
            "reward_not_compared": True,
            "interpretation": (
                "SE(2) alignment changes only reward residual bookkeeping; "
                "observation, target, physics state, and done are bit-exact."
            ),
        }
    h5_v3_substep_contact_reward_only_parity_preflight = None
    if h5_v3_substep_contact_alignment:
        # Build a control with exactly the same V3 SE(2), direct mapper,
        # sampler, decoder, guard and physics.  Only the proposed reward
        # bookkeeping and its three scales are removed.
        baseline_environment_kwargs = dict(environment_kwargs)
        baseline_environment_kwargs["h5_v3_substep_contact_alignment"] = False
        baseline_environment_kwargs["h5_v3_substep_preflight_telemetry"] = False
        baseline_environment_kwargs[
            "h5_v3_substep_preflight_fixed_quality_replay"
        ] = False
        baseline_environment_kwargs["reward_scales"] = replace(
            reward_scales,
            h5_all_substep_strict20ms_slip_rms=0.0,
            h5_all_substep_slip_tail=0.0,
            h5_all_substep_force_tail=0.0,
        )
        baseline_env = (
            None
            if t1_diagnostic_arm == "treatment"
            else make_h4_aligned_environment_class(
                **baseline_environment_kwargs
            )()
        )
        if t1_diagnostic_arm is not None:
            parity_seed = args.seed + 29
            t1_diagnostic_mode = str(
                getattr(
                    args,
                    "h5_v3_substep_contact_t1_diagnostic_mode",
                    "compiled_b2",
                )
            )
            if t1_diagnostic_mode not in {
                "compiled_b2",
                "vmap_b1_s1",
                "jit_scalar_s1",
                "eager_s1",
            }:
                raise ValueError("unsupported H5 T=1 diagnostic mode")
            fixed_quality_replay_ablation = bool(
                getattr(
                    args,
                    "h5_v3_substep_contact_t1_fixed_quality_replay_ablation",
                    False,
                )
            )
            fixed_quality_replay_manifest = (
                h5_v3_t1_fixed_quality_replay_manifest()
                if fixed_quality_replay_ablation
                else None
            )
            t1_batch_size = (
                H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE
                if t1_diagnostic_mode == "compiled_b2"
                else 1
            )
            t1_uses_vmap = t1_diagnostic_mode in {"compiled_b2", "vmap_b1_s1"}
            t1_is_compiled = t1_diagnostic_mode != "eager_s1"
            exact_b2_keys = jax.random.split(
                jax.random.PRNGKey(parity_seed),
                H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE,
            )
            source_paths = {
                "h4_training_alignment": ALIGNMENT_MODULE_PATH,
                "h5_substep_contact_alignment": (
                    EXP_ROOT / "safe_gait_experts/h5_substep_contact_alignment.py"
                ),
                "h5_command_conditioned_se2": (
                    EXP_ROOT / "safe_gait_experts/h5_command_conditioned_se2.py"
                ),
                "runner": Path(__file__).resolve(),
                "generated_scene": paths["scene"],
                "generated_manifest": paths["manifest"],
            }
            source_before = _hash_snapshot(source_paths)
            diagnostic_env = env if t1_diagnostic_arm == "treatment" else baseline_env
            if diagnostic_env is None:
                raise RuntimeError("T=1 diagnostic arm environment is unavailable")
            joystick_module = stack["joystick"]
            source_physics_step = joystick_module.mjx_env.step
            source_motor_speed_limits = joystick_module.USE_MOTOR_SPEED_LIMITS
            restoration_checks: dict[str, bool] = {}

            def assert_source_globals_restored(label: str) -> None:
                restored = bool(
                    joystick_module.mjx_env.step is source_physics_step
                    and joystick_module.USE_MOTOR_SPEED_LIMITS
                    is source_motor_speed_limits
                )
                restoration_checks[label] = restored
                if not restored:
                    raise RuntimeError(
                        "H5 T=1 diagnostic observed an un-restored joystick global "
                        f"at {label}"
                    )

            def block_whole_tree(value: Any) -> Any:
                # Blocking only qpos lets an independent post-physics replay
                # remain in flight.  The whole state is acceptance evidence.
                return jax.block_until_ready(value)

            def select_exact_b2_lanes(batched_state: Any) -> Any:
                if t1_diagnostic_mode == "compiled_b2":
                    return batched_state
                if t1_diagnostic_mode == "vmap_b1_s1":
                    return jax.tree_util.tree_map(
                        lambda value: value[1:2], batched_state
                    )
                return jax.tree_util.tree_map(
                    lambda value: value[1], batched_state
                )

            compiled_step_trace_invocation_count = 0
            def counted_single_step(single_state: Any, single_action: Any) -> Any:
                nonlocal compiled_step_trace_invocation_count
                compiled_step_trace_invocation_count += 1
                return diagnostic_env.step(single_state, single_action)

            if t1_uses_vmap:
                diagnostic_step = jax.jit(jax.vmap(counted_single_step))
            elif t1_is_compiled:
                diagnostic_step = jax.jit(counted_single_step)
            else:
                diagnostic_step = diagnostic_env.step
            canonical_b2_reset = jax.jit(jax.vmap(diagnostic_env.reset))
            assert_source_globals_restored("before_initial_reset")
            canonical_initial_state = block_whole_tree(
                canonical_b2_reset(exact_b2_keys)
            )
            initial_state = block_whole_tree(
                select_exact_b2_lanes(canonical_initial_state)
            )
            assert_source_globals_restored("after_initial_reset")
            action_dtype = np.asarray(initial_state.data.qpos).dtype
            action = np.zeros(
                (
                    *((t1_batch_size,) if t1_uses_vmap else ()),
                    len(ACTUATOR_JOINT_ORDER),
                ),
                dtype=action_dtype,
            )
            action_device = jp.asarray(action, dtype=action_dtype)

            def deep_device_copy(state: Any) -> Any:
                return jax.tree_util.tree_map(
                    lambda value: jp.array(value, copy=True), state
                )

            if t1_is_compiled:
                # Compile/trace on a sacrificial immutable input.  The two
                # states compared below therefore both enter the same
                # already-compiled executable rather than conflating
                # trace-time Python mutation with runtime determinism.
                assert_source_globals_restored("before_warmup_step")
                warmup_result = block_whole_tree(
                    diagnostic_step(deep_device_copy(initial_state), action_device)
                )
                assert_source_globals_restored("after_warmup_step")
            assert_source_globals_restored("before_entry_reset")
            canonical_entry_state = block_whole_tree(
                canonical_b2_reset(exact_b2_keys)
            )
            entry_state = block_whole_tree(
                select_exact_b2_lanes(canonical_entry_state)
            )
            assert_source_globals_restored("after_entry_reset")
            # ``step`` writes the Python ``info`` mapping while JIT traces.
            # Clone both inputs *before* either call so the repeat test cannot
            # accidentally feed trace-mutated host metadata to its second arm.
            first_input = block_whole_tree(deep_device_copy(entry_state))
            second_input = block_whole_tree(deep_device_copy(entry_state))
            inputs_equal, input_leaf_count, first_input_hash, second_input_hash = (
                h5_preflight_raw_tree_equal(jax, first_input, second_input)
            )
            if not inputs_equal:
                raise RuntimeError("H5 T=1 diagnostic inputs are not raw-identical")
            assert_source_globals_restored("before_first_step")
            first_state = block_whole_tree(diagnostic_step(first_input, action_device))
            assert_source_globals_restored("after_first_step")
            assert_source_globals_restored("before_second_step")
            second_state = block_whole_tree(diagnostic_step(second_input, action_device))
            assert_source_globals_restored("after_second_step")
            if (
                t1_is_compiled
                and compiled_step_trace_invocation_count != 1
            ):
                raise RuntimeError(
                    "H5 T=1 compiled diagnostic retraced: expected one Python "
                    f"step trace, observed {compiled_step_trace_invocation_count}"
                )
            same_arm_equal, same_arm_leaf_count, first_state_hash, second_state_hash = (
                h5_preflight_raw_tree_equal(jax, first_state, second_state)
            )
            dynamic_six = {}
            for field in ("qpos", "qvel", "act", "ctrl", "time", "qacc_warmstart"):
                if hasattr(first_state.data, field) and hasattr(second_state.data, field):
                    first_value = getattr(first_state.data, field)
                    second_value = getattr(second_state.data, field)
                    dynamic_six[field] = {
                        "first_raw_bytes_sha256": h5_preflight_raw_array_digest(first_value),
                        "second_raw_bytes_sha256": h5_preflight_raw_array_digest(second_value),
                        "exact_raw_equal": h5_preflight_raw_array_equal(
                            first_value, second_value
                        ),
                        "difference": h5_preflight_raw_array_difference(
                            first_value, second_value
                        ),
                    }
            source_after = _hash_snapshot(source_paths)
            _assert_unchanged(source_before, source_after)
            output_path = Path(
                args.h5_v3_substep_contact_t1_diagnostic_output
            ).resolve()
            if output_path.exists():
                raise FileExistsError(
                    "refusing to overwrite H5 T=1 diagnostic evidence: "
                    f"{output_path}"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "artifact_kind": "openduckmini_h5_v3_se2_substep_contact_t1_arm_diagnostic",
                "status": (
                    "T1_ARM_REPEAT_RAW_EQUAL_NOT_A_TRAINING_CANDIDATE"
                    if same_arm_equal
                    else "T1_ARM_REPEAT_RAW_MISMATCH_NOT_A_TRAINING_CANDIDATE"
                ),
                "hardware_deployment": "PROHIBITED",
                "arm": t1_diagnostic_arm,
                "quality_replay": {
                    "mode": (
                        "FIXED_HASH_BOUND_ABLATION_NOT_A_TRAINING_CANDIDATE"
                        if fixed_quality_replay_ablation
                        else "MEASURED_SECOND_MJX_FORWARD_REPLAY"
                    ),
                    "fixed_manifest": fixed_quality_replay_manifest,
                },
                "execution": {
                    "mode": t1_diagnostic_mode,
                    "batch_size": t1_batch_size,
                    "uses_vmap": t1_uses_vmap,
                    "exact_b2_lane_key_selection": (
                        "BOTH_LANES_0_1"
                        if t1_diagnostic_mode == "compiled_b2"
                        else "LANE_1_ONLY"
                    ),
                    "control_steps": 1,
                    "physics_substeps_per_control": 10,
                    "fresh_process_required_for_cross_arm_comparison": True,
                    "training_authorization": "PROHIBITED",
                    "seed": args.seed,
                    "parity_seed": parity_seed,
                    "action_dtype": action_dtype.str,
                    "action_shape": list(action.shape),
                    "action_raw_bytes_sha256": h5_preflight_raw_array_digest(action),
                    "whole_pytree_blocked_before_hashing": True,
                    "compiled_step_trace_invocation_count": (
                        compiled_step_trace_invocation_count
                        if t1_is_compiled
                        else None
                    ),
                    "compiled_step_trace_count_exactly_one": (
                        compiled_step_trace_invocation_count == 1
                        if t1_is_compiled
                        else None
                    ),
                },
                "joystick_global_restoration": {
                    "all_checks_passed": all(restoration_checks.values()),
                    "checks": restoration_checks,
                },
                "input_repeat": {
                    "full_state_raw_equal": inputs_equal,
                    "full_state_leaf_count": input_leaf_count,
                    "first_input_raw_tree_sha256": first_input_hash,
                    "second_input_raw_tree_sha256": second_input_hash,
                },
                "canonical_b2_entry_state": {
                    "source": "jax.jit(jax.vmap(env.reset))(split(seed, 2))",
                    "full_state_raw_tree_sha256": h5_preflight_raw_tree_digest(
                        jax, canonical_entry_state
                    )[0],
                    "mjx_data_raw_tree_sha256": h5_preflight_raw_tree_digest(
                        jax, canonical_entry_state.data
                    )[0],
                    "selected": (
                        "BOTH_LANES_0_1"
                        if t1_diagnostic_mode == "compiled_b2"
                        else "LANE_1_ONLY"
                    ),
                },
                "same_arm_repeat": {
                    "full_state_raw_equal": same_arm_equal,
                    "full_state_leaf_count": same_arm_leaf_count,
                    "first_state_raw_tree_sha256": first_state_hash,
                    "second_state_raw_tree_sha256": second_state_hash,
                },
                "entry_state": {
                    "full_state_raw_tree_sha256": h5_preflight_raw_tree_digest(
                        jax, entry_state
                    )[0],
                    "mjx_data_raw_tree_sha256": h5_preflight_raw_tree_digest(
                        jax, entry_state.data
                    )[0],
                    "observations_raw_tree_sha256": h5_preflight_raw_tree_digest(
                        jax, entry_state.obs
                    )[0],
                },
                "authoritative_dynamic_six": dynamic_six,
                "first_state_mjx_data_leaves": h5_preflight_leaf_records(
                    jax, first_state.data
                ),
                "second_state_mjx_data_leaves": h5_preflight_leaf_records(
                    jax, second_state.data
                ),
                "first_policy_visible": {
                    "observations_raw_tree_sha256": h5_preflight_raw_tree_digest(
                        jax, first_state.obs
                    )[0],
                    "done_raw_bytes_sha256": h5_preflight_raw_array_digest(
                        first_state.done
                    ),
                    "motor_targets_raw_bytes_sha256": h5_preflight_raw_array_digest(
                        first_state.info["motor_targets"]
                    ),
                    "command_raw_bytes_sha256": h5_preflight_raw_array_digest(
                        first_state.info["command"]
                    ),
                },
                "second_policy_visible": {
                    "observations_raw_tree_sha256": h5_preflight_raw_tree_digest(
                        jax, second_state.obs
                    )[0],
                    "done_raw_bytes_sha256": h5_preflight_raw_array_digest(
                        second_state.done
                    ),
                    "motor_targets_raw_bytes_sha256": h5_preflight_raw_array_digest(
                        second_state.info["motor_targets"]
                    ),
                    "command_raw_bytes_sha256": h5_preflight_raw_array_digest(
                        second_state.info["command"]
                    ),
                },
                "bound_inputs_pre_and_post": source_before,
            }
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return {
                "status": payload["status"],
                "hardware_deployment": "PROHIBITED",
                "diagnostic_output": str(output_path),
                "diagnostic_sha256": sha256_file(output_path),
            }
        parity_seed = args.seed + 29
        if env is None or baseline_env is None:
            raise RuntimeError("full H5 preflight requires treatment and control environments")
        parity_keys = jax.random.split(
            jax.random.PRNGKey(parity_seed), H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE
        )
        treatment_reset = jax.jit(jax.vmap(env.reset))
        control_reset = jax.jit(jax.vmap(baseline_env.reset))
        treatment_step = jax.jit(jax.vmap(env.step))
        control_step = jax.jit(jax.vmap(baseline_env.step))
        initial_treatment_state = treatment_reset(parity_keys)
        action_dtype = np.asarray(initial_treatment_state.data.qpos).dtype
        action_template = np.asarray(
            (
                (0.0,) * len(ACTUATOR_JOINT_ORDER),
                (0.12, -0.08, 0.05, -0.10, 0.06, 0.0, 0.0, 0.0, 0.0,
                 -0.12, 0.08, -0.05, 0.10, -0.06),
                (-0.09, 0.06, -0.04, 0.08, -0.05, 0.0, 0.0, 0.0, 0.0,
                 0.09, -0.06, 0.04, -0.08, 0.05),
                (0.0,) * len(ACTUATOR_JOINT_ORDER),
            ),
            dtype=action_dtype,
        )
        parity_actions = np.broadcast_to(
            np.tile(action_template, (5, 1))[:, None, :],
            (
                H5_V3_SUBSTEP_PREFLIGHT_CONTROL_STEPS,
                H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE,
                len(ACTUATOR_JOINT_ORDER),
            ),
        ).copy()
        one_ulp_mutation = np.array(parity_actions, copy=True)
        one_ulp_mutation[0, 0, 0] = np.nextafter(
            one_ulp_mutation[0, 0, 0], np.inf
        )
        checks: dict[str, bool] = {
            "execution_batch_size_exact": H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE == 2,
            "execution_control_steps_exact": len(parity_actions)
            == H5_V3_SUBSTEP_PREFLIGHT_CONTROL_STEPS,
            "execution_substeps_per_control_exact": True,
            "actions_have_exact_head_zeros": bool(
                np.all(parity_actions[:, :, 5:9] == 0.0)
            ),
            "one_ulp_action_mutation_detected": not h5_preflight_raw_array_equal(
                parity_actions, one_ulp_mutation
            ),
        }
        treatment_info_base_extra = {
            "h5_v3_substep_debounce",
            "h5_v3_substep_strict20ms_slip_rms_loss",
            "h5_v3_substep_slip_tail_loss",
            "h5_v3_substep_force_tail_loss",
            "h5_v3_substep_qualified_sample_count",
            "h5_v3_substep_samples_finite",
        }
        treatment_info_trace_extra = {
            "h5_v3_substep_time_s_trace",
            "h5_v3_substep_normalized_force_trace",
            "h5_v3_substep_tangential_speed_trace_mps",
        }
        treatment_metric_extra = {
            "h5/raw_substep_strict20ms_slip_rms_loss",
            "h5/raw_substep_slip_tail_loss",
            "h5/raw_substep_force_tail_loss",
            "h5/raw_substep_qualified_sample_count",
            "h5/substep_samples_finite",
        }
        mapping_key_audits: dict[str, dict[str, list[str]]] = {}

        def compare_mapping(
            prefix: str,
            treatment_mapping: Mapping[str, Any],
            control_mapping: Mapping[str, Any],
            *,
            allowed_treatment_extra: set[str],
            excluded: set[str] | None = None,
        ) -> dict[str, str]:
            ignored = set() if excluded is None else excluded
            treatment_keys = set(treatment_mapping) - ignored
            control_keys = set(control_mapping) - ignored
            mapping_key_audits[prefix] = {
                "treatment_only": sorted(treatment_keys - control_keys),
                "control_only": sorted(control_keys - treatment_keys),
                "allowed_treatment_only": sorted(allowed_treatment_extra),
            }
            checks[f"{prefix}_treatment_extra_allowlist"] = (
                treatment_keys - control_keys == allowed_treatment_extra
            )
            checks[f"{prefix}_control_has_no_extra"] = not (control_keys - treatment_keys)
            hashes: dict[str, str] = {}
            for key in sorted(treatment_keys & control_keys):
                equal, _leaves, treatment_hash, _control_hash = (
                    h5_preflight_raw_tree_equal(
                        jax, treatment_mapping[key], control_mapping[key]
                    )
                )
                checks[f"{prefix}_{key}_raw_bytes"] = equal
                hashes[key] = treatment_hash
            return hashes

        def compare_state(
            prefix: str,
            treatment_state: Any,
            control_state: Any,
            *,
            expect_trace_telemetry: bool,
        ) -> dict[str, Any]:
            data_equal, data_leaves, treatment_data_hash, control_data_hash = (
                h5_preflight_raw_tree_equal(
                    jax, treatment_state.data, control_state.data
                )
            )
            obs_equal, _obs_leaves, treatment_obs_hash, control_obs_hash = (
                h5_preflight_raw_tree_equal(
                    jax, treatment_state.obs, control_state.obs
                )
            )
            checks[f"{prefix}_full_mjx_data_raw_bytes"] = data_equal
            checks[f"{prefix}_observations_raw_bytes"] = obs_equal
            checks[f"{prefix}_done_raw_bytes"] = h5_preflight_raw_array_equal(
                treatment_state.done, control_state.done
            )
            for label, treatment_value, control_value in (
                ("qpos", treatment_state.data.qpos, control_state.data.qpos),
                ("qvel", treatment_state.data.qvel, control_state.data.qvel),
                ("ctrl", treatment_state.data.ctrl, control_state.data.ctrl),
                (
                    "guard_motor_targets",
                    treatment_state.info["motor_targets"],
                    control_state.info["motor_targets"],
                ),
                (
                    "guard_desired_targets",
                    treatment_state.info["h4_guard_desired_targets"],
                    control_state.info["h4_guard_desired_targets"],
                ),
            ):
                checks[f"{prefix}_{label}_raw_bytes"] = h5_preflight_raw_array_equal(
                    treatment_value, control_value
                )
            info_hashes = compare_mapping(
                f"{prefix}_info",
                treatment_state.info,
                control_state.info,
                allowed_treatment_extra=(
                    treatment_info_base_extra
                    | (
                        treatment_info_trace_extra
                        if expect_trace_telemetry
                        else set()
                    )
                ),
                excluded={"rewards"},
            )
            reward_excluded_treatment_metrics = {
                key: value
                for key, value in treatment_state.metrics.items()
                if not key.startswith(("cost/", "reward/"))
            }
            reward_excluded_control_metrics = {
                key: value
                for key, value in control_state.metrics.items()
                if not key.startswith(("cost/", "reward/"))
            }
            metric_hashes = compare_mapping(
                f"{prefix}_metrics",
                reward_excluded_treatment_metrics,
                reward_excluded_control_metrics,
                allowed_treatment_extra=treatment_metric_extra,
            )
            return {
                "data": treatment_data_hash,
                "control_data": control_data_hash,
                "observations": treatment_obs_hash,
                "control_observations": control_obs_hash,
                "data_leaf_count": data_leaves,
                "info_shared": info_hashes,
                "metrics_shared": metric_hashes,
            }

        def rederive_treatment_tick(entry_state: Any, result_state: Any) -> dict[str, Any]:
            times = np.asarray(result_state.info["h5_v3_substep_time_s_trace"])
            force = np.asarray(result_state.info["h5_v3_substep_normalized_force_trace"])
            speed = np.asarray(
                result_state.info["h5_v3_substep_tangential_speed_trace_mps"]
            )
            checks["treatment_trace_shape_exact"] = (
                times.shape
                == (H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE, 10)
                and force.shape == (H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE, 10, 2)
                and speed.shape == (H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE, 10, 2)
            )
            checks["treatment_trace_2ms_spacing"] = bool(
                np.allclose(np.diff(times, axis=1), 0.002, rtol=0.0, atol=1.0e-7)
            )
            checks["treatment_trace_finite_nonnegative"] = bool(
                np.all(np.isfinite(times))
                and np.all(np.isfinite(force))
                and np.all(np.isfinite(speed))
                and np.all(force >= 0.0)
                and np.all(speed >= 0.0)
            )
            checks["treatment_qualified_counts_nonnegative"] = bool(
                np.all(
                    np.asarray(
                        result_state.info["h5_v3_substep_qualified_sample_count"]
                    )
                    >= 0.0
                )
            )
            checks["treatment_samples_finite"] = bool(
                np.all(np.asarray(result_state.info["h5_v3_substep_samples_finite"]))
            )
            actual_losses = (
                np.asarray(result_state.info["h5_v3_substep_strict20ms_slip_rms_loss"]),
                np.asarray(result_state.info["h5_v3_substep_slip_tail_loss"]),
                np.asarray(result_state.info["h5_v3_substep_force_tail_loss"]),
                np.asarray(result_state.info["h5_v3_substep_qualified_sample_count"]),
            )
            rederived: list[tuple[float, float, float, float]] = []
            for batch_index in range(H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE):
                entry_debounce = jax.tree_util.tree_map(
                    lambda value: np.asarray(value)[batch_index],
                    entry_state.info["h5_v3_substep_debounce"],
                )
                losses = h5_all_substep_quality_update(
                    force[batch_index],
                    speed[batch_index],
                    initial_debounce=entry_debounce,
                    times_s=times[batch_index],
                ).losses
                rederived.append(
                    (
                        float(losses.strict20ms_slip_rms_loss),
                        float(losses.slip_tail_loss),
                        float(losses.force_tail_loss),
                        float(losses.force_qualified_sample_count),
                    )
                )
            checks["treatment_losses_independently_rederived"] = bool(
                np.allclose(
                    np.asarray(rederived), np.stack(actual_losses, axis=1),
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
            )
            return {
                "time_sha256": h5_preflight_raw_array_digest(times),
                "force_sha256": h5_preflight_raw_array_digest(force),
                "speed_sha256": h5_preflight_raw_array_digest(speed),
            }

        def execute_witness(order: str) -> dict[str, Any]:
            treatment_state = treatment_reset(parity_keys)
            control_state = control_reset(parity_keys)
            hashes = [
                compare_state(
                    f"{order}_reset",
                    treatment_state,
                    control_state,
                    expect_trace_telemetry=False,
                )
            ]
            trace_hashes: list[dict[str, Any]] = []
            for step_index, action in enumerate(parity_actions):
                entry_state = treatment_state
                action_device = jp.asarray(action, dtype=action_dtype)
                checks[f"{order}_step_{step_index}_supplied_action_raw_bytes"] = (
                    h5_preflight_raw_array_equal(action, action_device)
                )
                if order == "treatment_then_control":
                    treatment_state = treatment_step(treatment_state, action_device)
                    control_state = control_step(control_state, action_device)
                else:
                    control_state = control_step(control_state, action_device)
                    treatment_state = treatment_step(treatment_state, action_device)
                hashes.append(
                    compare_state(
                        f"{order}_step_{step_index}",
                        treatment_state,
                        control_state,
                        expect_trace_telemetry=True,
                    )
                )
                trace_hashes.append(rederive_treatment_tick(entry_state, treatment_state))
            return {"trajectory_hashes": hashes, "trace_hashes": trace_hashes}

        historical_evaluator_path = (
            EXP_ROOT
            / "artifacts/h5_v3_substep_alignment_preflight_20260812_v2"
            / "preflight_result.json"
        )
        preflight_bound_paths = {
            "h4_training_alignment": ALIGNMENT_MODULE_PATH,
            "h5_substep_contact_alignment": (
                EXP_ROOT / "safe_gait_experts/h5_substep_contact_alignment.py"
            ),
            "h5_command_conditioned_se2": (
                EXP_ROOT / "safe_gait_experts/h5_command_conditioned_se2.py"
            ),
            "runner": Path(__file__).resolve(),
            "generated_scene": paths["scene"],
            "generated_manifest": paths["manifest"],
            "historical_broad_evaluator": historical_evaluator_path,
        }
        source_hashes_before = _hash_snapshot(preflight_bound_paths)
        forward_witness = execute_witness("treatment_then_control")
        reversed_witness = execute_witness("control_then_treatment")
        source_hashes_after = _hash_snapshot(preflight_bound_paths)
        _assert_unchanged(source_hashes_before, source_hashes_after)
        checks["bound_inputs_unchanged_pre_to_post"] = True
        checks["execution_order_independent_treatment_trajectory"] = (
            forward_witness["trajectory_hashes"] == reversed_witness["trajectory_hashes"]
        )
        checks["execution_order_independent_trace_telemetry"] = (
            forward_witness["trace_hashes"] == reversed_witness["trace_hashes"]
        )
        checks["all_400_environment_substep_records_observed"] = (
            H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE
            * H5_V3_SUBSTEP_PREFLIGHT_CONTROL_STEPS
            * 10
            == 400
        )
        known_force = np.full((10, 2), 0.50, dtype=np.float64)
        known_bad_speed = np.full((10, 2), 0.040, dtype=np.float64)
        ideal_speed = np.zeros((10, 2), dtype=np.float64)
        strict_boundary_speed = np.full((10, 2), 0.015, dtype=np.float64)
        known_bad_losses = h5_all_substep_quality_losses(known_force, known_bad_speed)
        ideal_losses = h5_all_substep_quality_losses(known_force, ideal_speed)
        strict_boundary_losses = h5_all_substep_quality_losses(
            known_force, strict_boundary_speed
        )
        high_force_losses = h5_all_substep_quality_losses(
            np.full((10, 2), 2.0, dtype=np.float64), ideal_speed
        )
        no_contact_high_speed_losses = h5_all_substep_quality_losses(
            np.zeros((10, 2), dtype=np.float64), known_bad_speed
        )
        try:
            h5_all_substep_quality_losses(
                np.full((10, 2), np.nan, dtype=np.float64), ideal_speed
            )
        except ValueError:
            nonfinite_rejected = True
        else:
            nonfinite_rejected = False
        reward_dict = reward_scales.as_reward_scale_dict(
            include_h5_substep_contact_alignment=True
        )
        reverse_return_proof = h5_reverse_return_order_proof(
            command_vx_mps=-0.050,
            stationary_local_vx_mps=0.0,
            moving_local_vx_mps=-0.050 * 0.75,
            stationary_losses=strict_boundary_losses,
            moving_losses=strict_boundary_losses,
            reverse_speed_boundary_scale=reward_dict["h4_reverse_speed_boundary"],
            strict20ms_slip_rms_scale=reward_dict[
                "h5_all_substep_strict20ms_slip_rms"
            ],
            slip_tail_scale=reward_dict["h5_all_substep_slip_tail"],
            force_tail_scale=reward_dict["h5_all_substep_force_tail"],
        )
        tracking_lin_scale = float(env._config.reward_config.scales.tracking_lin_vel)
        tracking_ang_scale = float(env._config.reward_config.scales.tracking_ang_vel)

        def reverse_tracking_return(local_vx_mps: float) -> float:
            command_vx_mps = -0.050
            active_tracking = float(
                np.exp(-np.square((local_vx_mps - command_vx_mps) / H5_TRACKING_SIGMA[0]))
            )
            signed_progress = local_vx_mps / command_vx_mps
            return active_tracking + 0.5 * H5_SIGNED_PROGRESS_SCALE * float(
                np.clip(signed_progress, -1.0, 1.0)
            )

        strict_contact_cost = (
            reward_dict["h5_all_substep_strict20ms_slip_rms"]
            * float(strict_boundary_losses.strict20ms_slip_rms_loss)
            + reward_dict["h5_all_substep_slip_tail"]
            * float(strict_boundary_losses.slip_tail_loss)
            + reward_dict["h5_all_substep_force_tail"]
            * float(strict_boundary_losses.force_tail_loss)
        )
        stationary_full_reverse_return = (
            tracking_lin_scale * reverse_tracking_return(0.0)
            + tracking_ang_scale
            + reward_dict["h4_reverse_speed_boundary"]
            * float(reverse_return_proof.stationary_speed_boundary_loss)
            + strict_contact_cost
        )
        moving_full_reverse_return = (
            tracking_lin_scale * reverse_tracking_return(-0.050 * 0.75)
            + tracking_ang_scale
            + reward_dict["h4_reverse_speed_boundary"]
            * float(reverse_return_proof.moving_speed_boundary_loss)
            + strict_contact_cost
        )
        reverse_return_margin = moving_full_reverse_return - stationary_full_reverse_return
        checks["known_bad_strict20ms_slip_cost_nonzero"] = bool(
            np.asarray(known_bad_losses.strict20ms_slip_rms_loss) > 0.0
        )
        checks["known_bad_slip_tail_cost_nonzero"] = bool(
            np.asarray(known_bad_losses.slip_tail_loss) > 0.0
        )
        checks["known_bad_force_tail_cost_nonzero"] = bool(
            np.asarray(high_force_losses.force_tail_loss) > 0.0
        )
        checks["ideal_contact_losses_zero"] = bool(
            np.asarray(ideal_losses.strict20ms_slip_rms_loss) == 0.0
            and np.asarray(ideal_losses.slip_tail_loss) == 0.0
            and np.asarray(ideal_losses.force_tail_loss) == 0.0
        )
        checks["no_contact_high_speed_has_zero_slip_cost"] = bool(
            np.asarray(no_contact_high_speed_losses.strict20ms_slip_rms_loss) == 0.0
            and np.asarray(no_contact_high_speed_losses.slip_tail_loss) == 0.0
        )
        checks["nonfinite_quality_input_fails_closed"] = nonfinite_rejected
        checks["strict_boundary_slip_rms_is_one"] = bool(
            np.isclose(
                float(strict_boundary_losses.strict20ms_slip_rms_loss),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        checks["reverse_threshold_motion_preferred_to_stationary"] = bool(
            np.asarray(reverse_return_proof.moving_strictly_preferred)
        )
        checks["full_weighted_reverse_margin_exceeds_0p05"] = bool(
            np.isfinite(reverse_return_margin) and reverse_return_margin > 0.05
        )
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            key_drift = {
                prefix: audit
                for prefix, audit in mapping_key_audits.items()
                if audit["treatment_only"] != audit["allowed_treatment_only"]
                or audit["control_only"]
            }
            raise RuntimeError(
                "H5 V3 substep contact reward-only preflight failed: "
                f"failed_count={len(failed)}, first_failed={failed[:40]}, "
                f"mapping_key_drift={key_drift}"
            )
        h5_v3_substep_contact_reward_only_parity_preflight = {
            "status": "PASS_NOT_A_TRAINING_CANDIDATE",
            "contract_id": H5_V3_SE2_SUBSTEP_CONTACT_ALIGNMENT_ID,
            "execution_shape": {
                "batch_size": H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE,
                "control_steps": len(parity_actions),
                "physics_substeps_per_control": 10,
                "environment_substep_records": 400,
                "compiled_gpu_witness": "jax.jit(jax.vmap(env.step))",
            },
            "production_pilot_shape_not_executed": {
                "num_timesteps": H5_V3_PRODUCTION_PILOT_SHAPE[0],
                "num_envs": H5_V3_PRODUCTION_PILOT_SHAPE[1],
                "unroll_length": H5_V3_PRODUCTION_PILOT_SHAPE[2],
                "batch_size": H5_V3_PRODUCTION_PILOT_SHAPE[3],
                "num_minibatches": H5_V3_PRODUCTION_PILOT_SHAPE[4],
                "num_updates_per_batch": H5_V3_PRODUCTION_PILOT_SHAPE[5],
            },
            "reset_keys": {
                "production_seed": args.seed,
                "parity_seed": parity_seed,
                "split_batch_keys": H5_V3_SUBSTEP_PREFLIGHT_BATCH_SIZE,
            },
            "actions": {
                "template_values": action_template.tolist(),
                "repetitions": 5,
                "dtype": action_dtype.str,
                "shape": list(parity_actions.shape),
                "raw_bytes_sha256": h5_preflight_raw_array_digest(parity_actions),
                "four_head_entries_5_to_8_exact_zero": True,
            },
            "treatment_reward_scales": {
                key: reward_dict[key]
                for key in (
                    "h5_all_substep_strict20ms_slip_rms",
                    "h5_all_substep_slip_tail",
                    "h5_all_substep_force_tail",
                )
            },
            "control_reward_scales": {
                "h5_all_substep_strict20ms_slip_rms": 0.0,
                "h5_all_substep_slip_tail": 0.0,
                "h5_all_substep_force_tail": 0.0,
            },
            "execution_order_witnesses": {
                "treatment_then_control": forward_witness,
                "control_then_treatment": reversed_witness,
            },
            "positive_controls": {
                "known_bad_high_slip": {
                    "strict20ms_slip_rms_loss": float(
                        known_bad_losses.strict20ms_slip_rms_loss
                    ),
                    "slip_tail_loss": float(known_bad_losses.slip_tail_loss),
                },
                "high_force_tail": float(high_force_losses.force_tail_loss),
                "ideal_losses": {
                    "strict20ms_slip_rms_loss": float(
                        ideal_losses.strict20ms_slip_rms_loss
                    ),
                    "slip_tail_loss": float(ideal_losses.slip_tail_loss),
                    "force_tail_loss": float(ideal_losses.force_tail_loss),
                },
                "no_contact_high_speed": {
                    "strict20ms_slip_rms_loss": float(
                        no_contact_high_speed_losses.strict20ms_slip_rms_loss
                    ),
                    "slip_tail_loss": float(
                        no_contact_high_speed_losses.slip_tail_loss
                    ),
                },
                "nonfinite_input_rejected": nonfinite_rejected,
            },
            "strict_boundary_contact_losses": {
                "strict20ms_slip_rms_loss": float(
                    strict_boundary_losses.strict20ms_slip_rms_loss
                ),
                "slip_tail_loss": float(strict_boundary_losses.slip_tail_loss),
                "force_tail_loss": float(strict_boundary_losses.force_tail_loss),
            },
            "known_bad_losses": {
                "strict20ms_slip_rms_loss": float(
                    known_bad_losses.strict20ms_slip_rms_loss
                ),
                "slip_tail_loss": float(known_bad_losses.slip_tail_loss),
                "force_tail_loss": float(known_bad_losses.force_tail_loss),
                "qualified_sample_count": int(
                    known_bad_losses.force_qualified_sample_count
                ),
            },
            "reverse_return_order": {
                field: (
                    bool(value)
                    if field == "moving_strictly_preferred"
                    else float(value)
                )
                for field, value in reverse_return_proof._asdict().items()
            },
            "full_weighted_reverse_return": {
                "tracking_lin_vel_scale": tracking_lin_scale,
                "tracking_ang_vel_scale_shared": tracking_ang_scale,
                "stationary_return": stationary_full_reverse_return,
                "moving_return": moving_full_reverse_return,
                "moving_minus_stationary": reverse_return_margin,
                "required_margin": 0.05,
                "shared_noncontact_terms": "explicitly equal by synthetic construction",
            },
            "checked": checks,
            "interpretation": (
                "The B=2 GPU witness compares raw dtype/shape/bytes, including NaN "
                "payloads and signed zero, for every MJX-data leaf and all shared "
                "observation, control-relevant info, guard, and metric fields. "
                "Treatment-only telemetry is an explicit allowlist and reward is "
                "excluded from trajectory equality because it is the sole intended "
                "behavioral difference."
            ),
        }
        if h5_v3_substep_contact_preflight_only:
            output_path = Path(args.h5_v3_substep_contact_preflight_output).resolve()
            if output_path.exists():
                raise FileExistsError(
                    "refusing to overwrite H5 V3 substep preflight evidence: "
                    f"{output_path}"
                )
            import brax
            import jaxlib
            import mujoco

            preflight_config = {
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in sorted(vars(args).items())
                },
                "execution_shape": H5_V3_SUBSTEP_PREFLIGHT_SHAPE,
                "production_pilot_shape_not_executed": H5_V3_PRODUCTION_PILOT_SHAPE,
                "parity_seed": parity_seed,
                "action_raw_bytes_sha256": h5_preflight_raw_array_digest(
                    parity_actions
                ),
            }
            preflight_config_sha256 = hashlib.sha256(
                json.dumps(
                    preflight_config, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 2,
                "artifact_kind": "openduckmini_h5_v3_se2_substep_contact_no_ppo_preflight",
                "status": h5_v3_substep_contact_reward_only_parity_preflight["status"],
                "hardware_deployment": "PROHIBITED",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "bound_inputs_pre_and_post": source_hashes_before,
                "preflight_config": preflight_config,
                "preflight_config_sha256": preflight_config_sha256,
                "runtime": {
                    "jax": str(getattr(jax, "__version__", "unknown")),
                    "jaxlib": str(getattr(jaxlib, "__version__", "unknown")),
                    "brax": str(getattr(brax, "__version__", "unknown")),
                    "mujoco": str(getattr(mujoco, "__version__", "unknown")),
                    "backend_resolution": backend_resolution,
                    "xla_autotune_policy": xla_autotune_policy,
                    "xla_flags": os.environ.get("XLA_FLAGS", ""),
                    "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
                },
                "no_ppo_tripwire": {
                    "ppo_train_called": False,
                    "checkpoint_written": False,
                    "training_run_directory_created": False,
                    "preflight_returns_before_ppo_path": True,
                },
                "preflight": h5_v3_substep_contact_reward_only_parity_preflight,
            }
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return {
                "status": payload["status"],
                "hardware_deployment": "PROHIBITED",
                "preflight_output": str(output_path),
                "preflight_sha256": sha256_file(output_path),
                "preflight": h5_v3_substep_contact_reward_only_parity_preflight,
            }
    if v4_substep_collector_trace_preflight_only:
        if env is None:
            raise RuntimeError("V4 collector trace capture environment is unavailable")
        baseline_environment_kwargs = dict(environment_kwargs)
        baseline_environment_kwargs["v4_substep_collector_trace_capture"] = False
        baseline_env = make_h4_aligned_environment_class(
            **baseline_environment_kwargs
        )()
        return run_v4_substep_collector_trace_preflight(
            args=args,
            capture_env=env,
            baseline_env=baseline_env,
            jax=jax,
            jp=jp,
            source_paths={
                "h4_training_alignment": ALIGNMENT_MODULE_PATH,
                "h5_sidecar_quality": (
                    EXP_ROOT / "safe_gait_experts/h5_sidecar_quality.py"
                ),
                "h5_substep_contact_alignment": (
                    EXP_ROOT / "safe_gait_experts/h5_substep_contact_alignment.py"
                ),
                "runner": Path(__file__).resolve(),
                "generated_scene": paths["scene"],
                "generated_manifest": paths["manifest"],
            },
        )
    if (
        v4_authoritative_primitive_batch_parity_preflight_only
        or v4_direct_primitive_isolation_preflight_only
        or v4_host_synchronized_primitive_ladder_preflight_only
    ):
        raise AssertionError(
            "unhandled V4 primitive no-PPO diagnostic reached the "
            "PPO boundary"
        )
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    reverse_iteration_v2_legacy_reward_audit = None
    reverse_iteration_v3_no_target_imitation_legacy_reward_audit = None
    reverse_iteration_v4_residual_transfer_gain_024_legacy_reward_audit = None
    reverse_iteration_v5_no_contact_imitation_legacy_reward_audit = None
    reverse_iteration_v6_absolute_full_leg_targets_legacy_reward_audit = None
    if (
        args.reverse_iteration_v2
        or args.reverse_iteration_v3_no_target_imitation
        or args.reverse_iteration_v4_residual_transfer_gain_024
        or args.reverse_iteration_v5_no_contact_imitation
        or args.reverse_iteration_v6_absolute_full_leg_targets
    ):
        expected_legacy_reward_config = (
            REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_LEGACY_REWARD_CONFIG
            if args.reverse_iteration_v6_absolute_full_leg_targets
            else REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_LEGACY_REWARD_CONFIG
            if args.reverse_iteration_v5_no_contact_imitation
            else REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG
            if (
                args.reverse_iteration_v3_no_target_imitation
                or args.reverse_iteration_v4_residual_transfer_gain_024
            )
            else REVERSE_ITERATION_V2_LEGACY_REWARD_CONFIG
        )
        per_environment: dict[str, dict[str, float]] = {}
        for label, instance in (("train", env), ("eval", eval_env)):
            actual = {
                "target_imitation": float(
                    instance._config.reward_config.scales.target_imitation
                ),
                "contact_imitation": float(
                    instance._config.reward_config.scales.contact_imitation
                ),
                "tracking_sigma": float(
                    instance._config.reward_config.tracking_sigma
                ),
                "backward_residual_scale": float(
                    instance._backward_residual_scale
                ),
            }
            expected = {
                **expected_legacy_reward_config,
                "backward_residual_scale": (
                    REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN
                    if args.reverse_iteration_v4_residual_transfer_gain_024
                    else REVERSE_ITERATION_V6_RESIDUAL_AUTHORITY_SCALE
                    if args.reverse_iteration_v6_absolute_full_leg_targets
                    else 0.12
                ),
            }
            if any(
                not np.isclose(actual[name], value, rtol=0.0, atol=0.0)
                for name, value in expected.items()
            ):
                raise RuntimeError(
                    f"{label} reverse iteration legacy reward/teacher "
                    f"contract drifted: {actual} != {expected}"
                )
            per_environment[label] = actual
        legacy_reward_audit = {
            "expected": expected,
            "per_environment": per_environment,
            "passed": True,
        }
        if args.reverse_iteration_v6_absolute_full_leg_targets:
            reverse_iteration_v6_absolute_full_leg_targets_legacy_reward_audit = (
                legacy_reward_audit
            )
        elif args.reverse_iteration_v5_no_contact_imitation:
            reverse_iteration_v5_no_contact_imitation_legacy_reward_audit = (
                legacy_reward_audit
            )
        elif args.reverse_iteration_v4_residual_transfer_gain_024:
            reverse_iteration_v4_residual_transfer_gain_024_legacy_reward_audit = (
                legacy_reward_audit
            )
        elif args.reverse_iteration_v3_no_target_imitation:
            reverse_iteration_v3_no_target_imitation_legacy_reward_audit = (
                legacy_reward_audit
            )
        else:
            reverse_iteration_v2_legacy_reward_audit = legacy_reward_audit
    expected_width = (
        H4_ACTOR_OBSERVATION_WIDTH
        if include_h4_observables
        else LEGACY_ACTOR_OBSERVATION_WIDTH
    )
    probe_state = env.reset(jax.random.PRNGKey(args.seed))
    actual_width = int(probe_state.obs["state"].shape[0])
    if actual_width != expected_width:
        raise RuntimeError(
            f"environment observation width {actual_width} != {expected_width}"
        )
    if not (
        env._config.noise_config.action_min_delay == 0
        and env._config.noise_config.action_max_delay == 1
    ):
        raise RuntimeError("H4 action delay must be exact zero")
    if selected is not None:
        actual_advance = float(
            env._backward_phase_rate * env._h4_reverse_teacher_phase_scale
        )
        if not np.isclose(
            actual_advance, selected["phase_advance_bins"], atol=1.0e-12
        ):
            raise RuntimeError("selected reverse teacher phase wiring mismatch")

    forward_v4_source_semantic_preflight = None
    if (
        args.forward_iteration_v4_contact_event_validity_persistence
        or args.forward_v5_contact_pulse_abort_scale_only
        or args.forward_iteration_v6_contact_abort_island_only
    ):
        forward_v4_source_semantic_preflight = (
            run_forward_v4_source_semantic_preflight(
                jax,
                jp,
                env,
                probe_state,
                source_physics_step=stack["joystick"].mjx_env.step,
                mjx_step=stack["joystick"].mjx_env.mjx.step,
                source_root=source_root,
                joystick_module=stack["joystick"],
                mjx_env_module=stack["joystick"].mjx_env,
                seed=args.seed,
                reset_noise_multiplier=args.reset_noise_multiplier,
            )
        )

    if h5_seed is not None and not h5_teacher_only:
        loaded_checkpoint = h5_seed["params"]
        initialization_source = "H5_TARGETSPACE_DISTILLED_SEED"
    elif h4_parent is None:
        loaded_checkpoint = ppo_checkpoint.load(str(parent_checkpoint))
        initialization_source = "V22_BRAX_CHECKPOINT"
    else:
        with h4_parent["params_path"].open("rb") as stream:
            loaded_checkpoint = pickle.load(stream)
        initialization_source = "TRUSTED_H4_FINAL_PARAMS_PICKLE"
    restore_params, checkpoint_audit = require_checkpoint_observation_compatibility(
        loaded_checkpoint,
        actor_observation_width=expected_width,
        allow_explicit_v22_transplant=(
            include_h4_observables
            and h4_parent is None
            and (h5_seed is None or h5_teacher_only)
        ),
        xp=jp,
    )
    expected_training_platform = JAX_RESOLVED_BACKENDS[args.platform]
    pre_training_device_audits = {
        "probe_state": audit_jax_tree_placement(
            jax,
            probe_state,
            expected_platform=expected_training_platform,
            label="pre_training_probe_state",
        ),
        "restore_params": audit_jax_tree_placement(
            jax,
            restore_params,
            expected_platform=expected_training_platform,
            label="pre_training_restore_params",
        ),
    }
    if (
        include_h4_observables
        and h4_parent is None
        and (h5_seed is None or h5_teacher_only)
        and not checkpoint_audit.get("transplant_applied")
    ):
        raise RuntimeError("116-wide checkpoint transplant was not applied")
    if (
        include_h4_observables
        and h4_parent is None
        and (h5_seed is None or h5_teacher_only)
    ):
        old_normalizer = loaded_checkpoint[0]
        new_normalizer = restore_params[0]
        old_actor = np.asarray(
            loaded_checkpoint[1]["params"]["hidden_0"]["kernel"]
        )
        new_actor = np.asarray(
            restore_params[1]["params"]["hidden_0"]["kernel"]
        )
        old_critic = np.asarray(
            loaded_checkpoint[2]["params"]["hidden_0"]["kernel"]
        )
        new_critic = np.asarray(
            restore_params[2]["params"]["hidden_0"]["kernel"]
        )
        expected_state_variance = np.maximum(
            np.asarray(old_normalizer.summed_variance["state"]), 0.0
        )
        expected_privileged_variance = np.maximum(
            np.asarray(old_normalizer.summed_variance["privileged_state"]), 0.0
        )
        bitwise_checks = {
            "actor_old_101_rows_bitwise_equal": np.array_equal(
                new_actor[:101], old_actor
            ),
            "actor_new_15_rows_exact_zero": np.array_equal(
                new_actor[101:116], np.zeros_like(new_actor[101:116])
            ),
            "critic_actor_prefix_bitwise_equal": np.array_equal(
                new_critic[:101], old_critic[:101]
            ),
            "critic_new_15_rows_exact_zero": np.array_equal(
                new_critic[101:116], np.zeros_like(new_critic[101:116])
            ),
            "critic_privileged_tail_bitwise_equal": np.array_equal(
                new_critic[116:], old_critic[101:]
            ),
            "normalizer_state_mean_old_rows_bitwise_equal": np.array_equal(
                np.asarray(new_normalizer.mean["state"][:101]),
                np.asarray(old_normalizer.mean["state"]),
            ),
            "normalizer_privileged_mean_old_rows_bitwise_equal": (
                np.array_equal(
                    np.asarray(new_normalizer.mean["privileged_state"][:101]),
                    np.asarray(old_normalizer.mean["privileged_state"][:101]),
                )
                and np.array_equal(
                    np.asarray(new_normalizer.mean["privileged_state"][116:]),
                    np.asarray(old_normalizer.mean["privileged_state"][101:]),
                )
            ),
            "normalizer_state_std_old_rows_bitwise_equal": np.array_equal(
                np.asarray(new_normalizer.std["state"][:101]),
                np.asarray(old_normalizer.std["state"]),
            ),
            "normalizer_privileged_std_old_rows_bitwise_equal": (
                np.array_equal(
                    np.asarray(new_normalizer.std["privileged_state"][:101]),
                    np.asarray(old_normalizer.std["privileged_state"][:101]),
                )
                and np.array_equal(
                    np.asarray(new_normalizer.std["privileged_state"][116:]),
                    np.asarray(old_normalizer.std["privileged_state"][101:]),
                )
            ),
            "normalizer_state_variance_old_rows_exact_repair": np.array_equal(
                np.asarray(new_normalizer.summed_variance["state"][:101]),
                expected_state_variance,
            ),
            "normalizer_privileged_variance_old_rows_exact_repair": (
                np.array_equal(
                    np.asarray(
                        new_normalizer.summed_variance["privileged_state"][:101]
                    ),
                    expected_privileged_variance[:101],
                )
                and np.array_equal(
                    np.asarray(
                        new_normalizer.summed_variance["privileged_state"][116:]
                    ),
                    expected_privileged_variance[101:],
                )
            ),
        }
        if not all(bitwise_checks.values()):
            raise RuntimeError(f"v22 transplant bitwise verification failed: {bitwise_checks}")
        checkpoint_audit = {**checkpoint_audit, **bitwise_checks}

    source_paths = {
        "legacy_trainer": LEGACY_TRAINER_PATH,
        "h4_alignment": ALIGNMENT_MODULE_PATH,
        "h5_training_alignment": EXP_ROOT
        / "safe_gait_experts"
        / "h5_training_alignment.py",
        "h5_command_contract": EXP_ROOT
        / "safe_gait_experts"
        / "h5_command_contract.py",
        "h5_target_contract": EXP_ROOT
        / "safe_gait_experts"
        / "h5_target_contract.py",
        "h4_runner": Path(__file__).resolve(),
        "h4_contract_module": EXP_ROOT / "safe_gait_experts" / "contract.py",
        "h4_contract_json": EXP_ROOT / "contract.json",
        "safe_randomization": EXP_ROOT
        / "safe_gait_experts"
        / "safe_randomization.py",
        "bounded_reward": EXP_ROOT / "safe_gait_experts" / "reward.py",
        "central_gait_quality": CENTRAL_QUALITY_PATHS[1],
        "central_routed_evaluation": CENTRAL_QUALITY_PATHS[2],
        "central_evaluator": CENTRAL_QUALITY_PATHS[0],
        "source_joystick": Path(stack["joystick"].__file__).resolve(),
        "source_constants": Path(stack["constants"].__file__).resolve(),
        "brax_ppo_train": Path(stack["ppo"].__file__).resolve(),
        "brax_ppo_networks": Path(stack["ppo_networks"].__file__).resolve(),
        "brax_ppo_checkpoint": Path(ppo_checkpoint.__file__).resolve(),
        "mujoco_playground_wrapper": Path(stack["wrapper"].__file__).resolve(),
        "mujoco_playground_locomotion_params": Path(
            stack["locomotion_params"].__file__
        ).resolve(),
        "generated_manifest": paths["manifest"],
        "generated_scene": paths["scene"],
        "generated_reference": paths["reference"],
        "legacy_backward_teacher": legacy_teacher_gaits["backward"],
        "legacy_backward_left_teacher": legacy_teacher_gaits["backward_left"],
        "legacy_backward_right_teacher": legacy_teacher_gaits["backward_right"],
    }
    if selected is not None:
        source_paths["selected_reverse_teacher"] = selected["path"]
    if forward_spec is not None:
        source_paths["forward_minimum_spec"] = forward_spec["path"]
    if forward_iteration_v2_authorization is not None:
        source_paths["forward_iteration_v2_authorization"] = (
            forward_iteration_v2_authorization["path"]
        )
        for label, binding in forward_iteration_v2_authorization[
            "bound_causal_inputs"
        ].items():
            source_paths[f"forward_iteration_v2_{label}"] = Path(
                binding["path"]
            )
    if forward_iteration_v3_touchdown_balance_authorization is not None:
        source_paths["forward_iteration_v3_authorization"] = (
            forward_iteration_v3_touchdown_balance_authorization["path"]
        )
        for label, binding in forward_iteration_v3_touchdown_balance_authorization[
            "bound_causal_inputs"
        ].items():
            source_paths[f"forward_iteration_v3_{label}"] = Path(binding["path"])
    if forward_iteration_v4_contact_event_validity_persistence_authorization is not None:
        source_paths["forward_iteration_v4_authorization"] = (
            forward_iteration_v4_contact_event_validity_persistence_authorization[
                "path"
            ]
        )
        for label, binding in (
            forward_iteration_v4_contact_event_validity_persistence_authorization[
                "bound_causal_inputs"
            ].items()
        ):
            source_paths[f"forward_iteration_v4_{label}"] = Path(binding["path"])
        for label, binding in (
            forward_iteration_v4_contact_event_validity_persistence_authorization[
                "bound_causal_sources"
            ].items()
        ):
            source_paths[f"forward_iteration_v4_source_{label}"] = Path(
                binding["path"]
            )
    if forward_v5_contact_pulse_abort_scale_only_authorization is not None:
        source_paths["forward_iteration_v5_authorization"] = (
            forward_v5_contact_pulse_abort_scale_only_authorization["path"]
        )
        for label, binding in forward_v5_contact_pulse_abort_scale_only_authorization[
            "bound_causal_inputs"
        ].items():
            source_paths[f"forward_iteration_v5_{label}"] = Path(binding["path"])
        for label, relative in {
            "h4_training_alignment": "safe_gait_experts/h4_training_alignment.py",
            "h4_runner": "scripts/train_h4_aligned_expert.py",
            "h4_post_training": "safe_gait_experts/h4_post_training.py",
            "h4_candidate_evaluator": "scripts/evaluate_h4_training_candidate.py",
            "h4_no_ppo_smoke": "scripts/smoke_h4_training_alignment.py",
        }.items():
            source_paths[f"forward_iteration_v5_current_source_{label}"] = (
                EXP_ROOT / relative
            )
    if forward_iteration_v6_contact_abort_island_only_authorization is not None:
        source_paths["forward_iteration_v6_authorization"] = (
            forward_iteration_v6_contact_abort_island_only_authorization["path"]
        )
        for label, binding in forward_iteration_v6_contact_abort_island_only_authorization[
            "bound_causal_inputs"
        ].items():
            source_paths[f"forward_iteration_v6_{label}"] = Path(binding["path"])
        for label, relative in {
            "h4_training_alignment": "safe_gait_experts/h4_training_alignment.py",
            "h4_runner": "scripts/train_h4_aligned_expert.py",
            "h4_post_training": "safe_gait_experts/h4_post_training.py",
            "h4_candidate_evaluator": "scripts/evaluate_h4_training_candidate.py",
            "h4_no_ppo_smoke": "scripts/smoke_h4_training_alignment.py",
        }.items():
            source_paths[f"forward_iteration_v6_current_source_{label}"] = (
                EXP_ROOT / relative
            )
    if reverse_spec is not None:
        source_paths["reverse_minimum_spec"] = reverse_spec["path"]
    if reverse_authorization is not None:
        source_paths["reverse_composition_authorization"] = reverse_authorization[
            "path"
        ]
        source_paths["reverse_composition_validator"] = (
            REVERSE_COMPOSITION_VALIDATOR_PATH
        )
    if reverse_iteration_v2_authorization is not None:
        source_paths["reverse_iteration_v2_authorization"] = (
            reverse_iteration_v2_authorization["path"]
        )
        for label, binding in reverse_iteration_v2_authorization[
            "bound_causal_inputs"
        ].items():
            source_paths[f"reverse_iteration_v2_{label}"] = Path(
                binding["path"]
            )
    if reverse_iteration_v3_no_target_imitation_authorization is not None:
        source_paths["reverse_iteration_v3_authorization"] = (
            reverse_iteration_v3_no_target_imitation_authorization["path"]
        )
        for label, binding in reverse_iteration_v3_no_target_imitation_authorization[
            "bound_causal_inputs"
        ].items():
            source_paths[f"reverse_iteration_v3_{label}"] = Path(binding["path"])
    if reverse_iteration_v4_residual_transfer_gain_024_authorization is not None:
        source_paths["reverse_iteration_v4_authorization"] = (
            reverse_iteration_v4_residual_transfer_gain_024_authorization["path"]
        )
        for label, binding in (
            reverse_iteration_v4_residual_transfer_gain_024_authorization[
                "bound_causal_inputs"
            ].items()
        ):
            source_paths[f"reverse_iteration_v4_{label}"] = Path(binding["path"])
        for label, binding in (
            reverse_iteration_v4_residual_transfer_gain_024_authorization[
                "bound_causal_sources"
            ].items()
        ):
            source_paths[f"reverse_iteration_v4_source_{label}"] = Path(
                binding["path"]
            )
    if reverse_iteration_v5_no_contact_imitation_authorization is not None:
        source_paths["reverse_iteration_v5_authorization"] = (
            reverse_iteration_v5_no_contact_imitation_authorization["path"]
        )
        for label, binding in reverse_iteration_v5_no_contact_imitation_authorization[
            "bound_causal_inputs"
        ].items():
            source_paths[f"reverse_iteration_v5_{label}"] = Path(binding["path"])
        for label, relative in {
            "h4_training_alignment": "safe_gait_experts/h4_training_alignment.py",
            "h4_runner": "scripts/train_h4_aligned_expert.py",
            "h4_post_training": "safe_gait_experts/h4_post_training.py",
            "h4_candidate_evaluator": "scripts/evaluate_h4_training_candidate.py",
            "h4_no_ppo_smoke": "scripts/smoke_h4_training_alignment.py",
        }.items():
            source_paths[f"reverse_iteration_v5_current_source_{label}"] = (
                EXP_ROOT / relative
            )
    if reverse_iteration_v6_absolute_full_leg_targets_authorization is not None:
        source_paths["reverse_iteration_v6_authorization"] = (
            reverse_iteration_v6_absolute_full_leg_targets_authorization["path"]
        )
        for label, binding in reverse_iteration_v6_absolute_full_leg_targets_authorization[
            "bound_causal_inputs"
        ].items():
            source_paths[f"reverse_iteration_v6_{label}"] = Path(binding["path"])
        for label, relative in {
            "h4_training_alignment": "safe_gait_experts/h4_training_alignment.py",
            "h4_runner": "scripts/train_h4_aligned_expert.py",
            "h4_post_training": "safe_gait_experts/h4_post_training.py",
            "h4_candidate_evaluator": "scripts/evaluate_h4_training_candidate.py",
            "h4_no_ppo_smoke": "scripts/smoke_h4_training_alignment.py",
        }.items():
            source_paths[f"reverse_iteration_v6_current_source_{label}"] = (
                EXP_ROOT / relative
            )
    if h4_parent is not None:
        source_paths["h4_parent_params"] = h4_parent["params_path"]
        source_paths["h4_parent_manifest"] = h4_parent["manifest_path"]
        source_paths["h4_parent_resolved_config"] = h4_parent[
            "resolved_config_path"
        ]
    if h5_seed is not None:
        source_paths["h5_targetspace_seed_params"] = h5_seed["params_path"]
        source_paths["h5_targetspace_seed_manifest"] = h5_seed["manifest_path"]
        if h5_seed.get("teacher_params_path") is not None:
            source_paths["h5_targetspace_rollout_teacher_params"] = h5_seed[
                "teacher_params_path"
            ]
    pre_hashes = _hash_snapshot(source_paths)
    parent_checkpoint_sha_pre = initialization_artifact_sha256

    run_name = _validate_run_name(
        args.run_name
        or _default_run_name(
            args.expert,
            args.seed,
            args.wiring_only,
            forward_iteration_v2=args.forward_iteration_v2,
            forward_iteration_v3_touchdown_balance=(
                args.forward_iteration_v3_touchdown_balance
            ),
            forward_iteration_v4_contact_event_validity_persistence=(
                args.forward_iteration_v4_contact_event_validity_persistence
            ),
            forward_v5_contact_pulse_abort_scale_only=(
                args.forward_v5_contact_pulse_abort_scale_only
            ),
            forward_iteration_v6_contact_abort_island_only=(
                args.forward_iteration_v6_contact_abort_island_only
            ),
            reverse_iteration_v2=args.reverse_iteration_v2,
            reverse_iteration_v3_no_target_imitation=(
                args.reverse_iteration_v3_no_target_imitation
            ),
            reverse_iteration_v4_residual_transfer_gain_024=(
                args.reverse_iteration_v4_residual_transfer_gain_024
            ),
            reverse_iteration_v5_no_contact_imitation=(
                args.reverse_iteration_v5_no_contact_imitation
            ),
            reverse_iteration_v6_absolute_full_leg_targets=(
                args.reverse_iteration_v6_absolute_full_leg_targets
            ),
        )
    )
    run_dir = claim_unique_run_directory(output_root, args.expert, run_name)

    locomotion_params = stack["locomotion_params"]
    ppo_config = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    network_config = ppo_config.network_factory.to_dict()
    network_factory = functools.partial(
        stack["ppo_networks"].make_ppo_networks, **network_config
    )
    training = ppo_config.to_dict()
    training.pop("network_factory")
    training.update(
        {
            **asdict(shape),
            "seed": args.seed,
            "learning_rate": float(args.learning_rate),
            "entropy_cost": float(args.entropy_cost),
            "clipping_epsilon": float(args.clipping_epsilon),
            "discounting": float(args.discounting),
            "max_grad_norm": float(args.max_grad_norm),
            "run_evals": False,
            "restore_value_fn": True,
            "num_resets_per_eval": 0,
            "save_checkpoint_path": None,
            "log_training_metrics": True,
            "restore_checkpoint_path": (
                str(parent_checkpoint)
                if not include_h4_observables
                and h4_parent is None
                and (h5_seed is None or h5_teacher_only)
                else None
            ),
            "restore_params": (
                restore_params
                if include_h4_observables or h4_parent is not None
                else None
            ),
        }
    )
    exact_runtime_checks = {
        "episode_length_1000": training.get("episode_length") == 1000,
        "action_repeat_1": training.get("action_repeat") == 1,
        "normalize_observations_true": training.get("normalize_observations") is True,
        "run_evals_false": training.get("run_evals") is False,
        "restore_value_fn_true": training.get("restore_value_fn") is True,
        "num_resets_per_eval_zero": training.get("num_resets_per_eval") == 0,
        "save_checkpoint_path_none": training.get("save_checkpoint_path") is None,
    }
    if not all(exact_runtime_checks.values()):
        raise ValueError(f"H4 PPO runtime contract drifted: {exact_runtime_checks}")
    serializable_training = {
        key: value
        for key, value in training.items()
        if key != "restore_params"
    }
    execution_contract_id = resolve_execution_contract_id(
        args,
        forward_iteration_v2_authorization=(
            forward_iteration_v2_authorization
        ),
        forward_iteration_v3_touchdown_balance_authorization=(
            forward_iteration_v3_touchdown_balance_authorization
        ),
        forward_iteration_v4_contact_event_validity_persistence_authorization=(
            forward_iteration_v4_contact_event_validity_persistence_authorization
        ),
        forward_v5_contact_pulse_abort_scale_only_authorization=(
            forward_v5_contact_pulse_abort_scale_only_authorization
        ),
        forward_iteration_v6_contact_abort_island_only_authorization=(
            forward_iteration_v6_contact_abort_island_only_authorization
        ),
        reverse_iteration_v2_authorization=(
            reverse_iteration_v2_authorization
        ),
        reverse_iteration_v3_no_target_imitation_authorization=(
            reverse_iteration_v3_no_target_imitation_authorization
        ),
        reverse_iteration_v4_residual_transfer_gain_024_authorization=(
            reverse_iteration_v4_residual_transfer_gain_024_authorization
        ),
        reverse_iteration_v5_no_contact_imitation_authorization=(
            reverse_iteration_v5_no_contact_imitation_authorization
        ),
        reverse_iteration_v6_absolute_full_leg_targets_authorization=(
            reverse_iteration_v6_absolute_full_leg_targets_authorization
        ),
    )
    authorized_iteration_v2_250k_contract_id = (
        forward_iteration_v2_authorization["contract_id"]
        if forward_iteration_v2_authorization
        else reverse_iteration_v2_authorization["contract_id"]
        if reverse_iteration_v2_authorization
        else None
    )
    authorized_iteration_v3_250k_contract_id = (
        forward_iteration_v3_touchdown_balance_authorization["contract_id"]
        if forward_iteration_v3_touchdown_balance_authorization
        else reverse_iteration_v3_no_target_imitation_authorization["contract_id"]
        if reverse_iteration_v3_no_target_imitation_authorization
        else None
    )
    authorized_iteration_v4_250k_contract_id = (
        forward_iteration_v4_contact_event_validity_persistence_authorization[
            "contract_id"
        ]
        if forward_iteration_v4_contact_event_validity_persistence_authorization
        else reverse_iteration_v4_residual_transfer_gain_024_authorization[
            "contract_id"
        ]
        if reverse_iteration_v4_residual_transfer_gain_024_authorization
        else None
    )
    authorized_iteration_v5_250k_contract_id = (
        forward_v5_contact_pulse_abort_scale_only_authorization["contract_id"]
        if forward_v5_contact_pulse_abort_scale_only_authorization
        else reverse_iteration_v5_no_contact_imitation_authorization["contract_id"]
        if reverse_iteration_v5_no_contact_imitation_authorization
        else None
    )
    authorized_iteration_v6_250k_contract_id = (
        forward_iteration_v6_contact_abort_island_only_authorization["contract_id"]
        if forward_iteration_v6_contact_abort_island_only_authorization
        else reverse_iteration_v6_absolute_full_leg_targets_authorization["contract_id"]
        if reverse_iteration_v6_absolute_full_leg_targets_authorization
        else None
    )
    authorized_iteration_250k_contract_id = (
        authorized_iteration_v2_250k_contract_id
        or authorized_iteration_v3_250k_contract_id
        or authorized_iteration_v4_250k_contract_id
        or authorized_iteration_v5_250k_contract_id
        or authorized_iteration_v6_250k_contract_id
    )
    resolved_config = {
        "schema_version": 1,
        "hardware_deployment": "PROHIBITED",
        "expert": args.expert,
        "training_contract_id": execution_contract_id,
        "authorized_iteration_v2_250k_contract_id": (
            authorized_iteration_v2_250k_contract_id
        ),
        "authorized_iteration_v3_250k_contract_id": (
            authorized_iteration_v3_250k_contract_id
        ),
        "authorized_iteration_v4_250k_contract_id": (
            authorized_iteration_v4_250k_contract_id
        ),
        "authorized_iteration_v5_250k_contract_id": (
            authorized_iteration_v5_250k_contract_id
        ),
        "authorized_iteration_v6_250k_contract_id": (
            authorized_iteration_v6_250k_contract_id
        ),
        "qualification_use": (
            "WIRING_PREFLIGHT_ONLY_NOT_250K_QUALIFICATION"
            if args.wiring_only and authorized_iteration_250k_contract_id
            else "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION"
            if getattr(args, "diagnostic_reward_exploration", False)
            else "AUTHORIZED_250K_PILOT"
            if authorized_iteration_250k_contract_id
            else "LEGACY_ITERATION_V1_EXECUTION"
        ),
        "diagnostic_reward_exploration": bool(
            getattr(args, "diagnostic_reward_exploration", False)
        ),
        "unified_development_run": bool(
            getattr(args, "unified_development_run", False)
        ),
        "h5_command_contract": (
            h5_unified_command_contract_manifest(
                str(args.h5_unified_command_mapper)
            )
            if getattr(args, "diagnostic_reward_exploration", False)
            and args.expert == "unified"
            else h5_command_contract_manifest()
            if getattr(args, "diagnostic_reward_exploration", False)
            and args.expert in {"planar", "reverse", "unified"}
            else None
        ),
        "h5_command_contract_id": (
            h5_unified_command_contract_id(str(args.h5_unified_command_mapper))
            if getattr(args, "diagnostic_reward_exploration", False)
            and args.expert == "unified"
            else H5_COMMAND_CONTRACT_ID
            if getattr(args, "diagnostic_reward_exploration", False)
            and args.expert in {"planar", "reverse", "unified"}
            else None
        ),
        "h5_unified_reverse_route_probability": (
            float(args.h5_unified_reverse_route_probability)
            if getattr(args, "h5_unified_reverse_route_probability", None)
            is not None
            else None
        ),
        "h5_unified_command_mapper": (
            canonical_h5_unified_command_mapper(
                str(args.h5_unified_command_mapper)
            )
            if getattr(args, "diagnostic_reward_exploration", False)
            and args.expert == "unified"
            else None
        ),
        "h5_v3_command_conditioned_se2_alignment": bool(
            getattr(args, "h5_v3_command_conditioned_se2_alignment", False)
        ),
        "h5_v3_command_conditioned_se2_authorization": h5_v3_se2_authorization,
        "h5_v3_command_conditioned_se2_reward_only_parity_preflight": (
            h5_v3_se2_reward_only_parity_preflight
        ),
        "h5_targetspace_seed": (
            {
                "params_path": str(h5_seed["params_path"]),
                "params_sha256": h5_seed["params_sha256"],
                "manifest_path": str(h5_seed["manifest_path"]),
                "manifest_sha256": h5_seed["manifest_sha256"],
                "candidate_kind": h5_seed["manifest"].get("candidate_kind"),
                "runtime_target_authority": "ACTOR_ONLY_NO_TEACHER_COMPOSITION",
                "bc_target_authority": (
                    "FIXED_54_ROW_PHASE_TABLE"
                    if h5_seed.get("target_table") is not None
                    else "FROZEN_SEED_ACTOR"
                ),
                "bc_scale": float(args.h5_seed_bc_scale),
                "bc_anneal_control_steps": float(
                    args.h5_seed_bc_anneal_control_steps
                ),
                "teacher_mode": str(args.h5_seed_teacher_mode),
                "residual_gain": float(args.h5_seed_residual_gain),
                "teacher_reverse_command_contract": bool(
                    args.h5_seed_teacher_reverse_command_contract
                ),
                "initialize_from_params": bool(
                    getattr(args, "h5_seed_initialize_from_params", False)
                ),
                "rollout_teacher_params_path": (
                    str(h5_seed["teacher_params_path"])
                    if h5_seed.get("teacher_params_path") is not None
                    else None
                ),
                "rollout_teacher_params_sha256": h5_seed.get(
                    "teacher_params_sha256"
                ),
            }
            if h5_seed is not None
            else None
        ),
        "forward_iteration_v2": bool(args.forward_iteration_v2),
        "reverse_iteration_v2": bool(args.reverse_iteration_v2),
        "forward_iteration_v3_touchdown_balance": bool(
            args.forward_iteration_v3_touchdown_balance
        ),
        "reverse_iteration_v3_no_target_imitation": bool(
            args.reverse_iteration_v3_no_target_imitation
        ),
        "forward_iteration_v4_contact_event_validity_persistence": bool(
            args.forward_iteration_v4_contact_event_validity_persistence
        ),
        "reverse_iteration_v4_residual_transfer_gain_024": bool(
            args.reverse_iteration_v4_residual_transfer_gain_024
        ),
        "forward_v5_contact_pulse_abort_scale_only": bool(
            args.forward_v5_contact_pulse_abort_scale_only
        ),
        "reverse_iteration_v5_no_contact_imitation": bool(
            args.reverse_iteration_v5_no_contact_imitation
        ),
        "forward_iteration_v6_contact_abort_island_only": bool(
            args.forward_iteration_v6_contact_abort_island_only
        ),
        "reverse_iteration_v6_absolute_full_leg_targets": bool(
            args.reverse_iteration_v6_absolute_full_leg_targets
        ),
        "forward_v4_substep_contact": bool(
            args.forward_iteration_v4_contact_event_validity_persistence
            or args.forward_v5_contact_pulse_abort_scale_only
            or args.forward_iteration_v6_contact_abort_island_only
            or (
                getattr(args, "diagnostic_reward_exploration", False)
                and args.expert in {"planar", "reverse", "unified"}
            )
        ),
        "seed": args.seed,
        "platform": args.platform,
        "backend_resolution": backend_resolution,
        "xla_autotune_policy": xla_autotune_policy,
        "debug_callback_preflight": debug_callback_preflight,
        "pre_training_device_audits": pre_training_device_audits,
        "wiring_only": bool(args.wiring_only),
        "activity": (
            "PPO_WIRING_TRAINING" if args.wiring_only else "PPO_PILOT_TRAINING"
        ),
        "shape": asdict(shape),
        "interactions_per_training_step": shape.interactions_per_training_step,
        "expected_training_steps": shape.expected_training_steps,
        "expected_optimizer_updates": shape.expected_optimizer_updates,
        "learning_rate": float(args.learning_rate),
        "entropy_cost": float(args.entropy_cost),
        "clipping_epsilon": float(args.clipping_epsilon),
        "discounting": float(args.discounting),
        "max_grad_norm": float(args.max_grad_norm),
        "observation_mode": args.observation_mode,
        "actor_observation_width": expected_width,
        "checkpoint_compatibility": checkpoint_audit,
        "initialization_source": initialization_source,
        "anchor_config": anchors,
        "physical_reward_command_is_separate": True,
        "policy_observation_mapping_is_separate": True,
        "reward_scales": reward_scale_dict,
        "reset_noise_multiplier": args.reset_noise_multiplier,
        "backward_residual_scale": args.backward_residual_scale,
        "selected_reverse_teacher": (
            {
                "path": str(selected["path"]),
                "sha256": selected["sha256"],
                "candidate_id": selected["candidate_id"],
                "candidate_name": selected["candidate_name"],
                "cadence_hz": selected["cadence_hz"],
                "phase_advance_bins_per_control": selected[
                    "phase_advance_bins"
                ],
                "entry_phase_bins": selected["entry_phase_bins"],
                "training_use": "TRAINING_COMPOSITION_COMPONENT_NOT_ADOPTED",
                "persistent_during_training": True,
                "qualification": "FAILED_EXACT_HOME_H4",
                "runtime_parity_requirement": (
                    "Any adopted policy must use this identical teacher plus "
                    "learned residual composition at runtime."
                ),
            }
            if selected
            else None
        ),
        "forward_minimum_spec": (
            {
                "path": str(forward_spec["path"]),
                "sha256": forward_spec["sha256"],
                "canonical_sha256": PINNED_FORWARD_MINIMUM_SPEC_CANONICAL_SHA256,
                "status": forward_spec["payload"]["status"],
                "declared_actor_width": forward_spec["declared_actor_width"],
                "implementation_actor_width": forward_spec[
                    "implementation_actor_width"
                ],
                "stale_width_declaration_detected": forward_spec[
                    "stale_width_declaration_detected"
                ],
            }
            if forward_spec
            else None
        ),
        "forward_iteration_v2_authorization": (
            {
                "path": str(forward_iteration_v2_authorization["path"]),
                "sha256": forward_iteration_v2_authorization["sha256"],
                "contract_id": forward_iteration_v2_authorization[
                    "contract_id"
                ],
                "status": forward_iteration_v2_authorization["payload"][
                    "status"
                ],
                "semantic_audit": forward_iteration_v2_authorization[
                    "semantic_audit"
                ],
                "bound_causal_inputs": forward_iteration_v2_authorization[
                    "bound_causal_inputs"
                ],
                "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                "adoption_release_hardware": "PROHIBITED",
            }
            if forward_iteration_v2_authorization
            else None
        ),
        "reverse_minimum_spec": (
            {
                "path": str(reverse_spec["path"]),
                "sha256": reverse_spec["sha256"],
                "status": reverse_spec["payload"]["status"],
                "phase_conditioned_reward_wiring": True,
                "strict_boundary_normalized_tracking_wiring": True,
            }
            if reverse_spec
            else None
        ),
        "reverse_composition_authorization": (
            {
                "path": str(reverse_authorization["path"]),
                "sha256": reverse_authorization["sha256"],
                "status": reverse_authorization["payload"]["status"],
                "semantic_audit": reverse_authorization["semantic_audit"],
                "scope": "SIMULATION_250K_AND_LIKE_FOR_LIKE_EVALUATION_ONLY",
                "adoption_release_hardware": "PROHIBITED",
                "standalone_direct_runtime_allowed": False,
                "adoption_allowed": False,
                "release_allowed": False,
                "hardware_allowed": False,
            }
            if reverse_authorization
            else None
        ),
        "reverse_iteration_v2_authorization": (
            {
                "path": str(reverse_iteration_v2_authorization["path"]),
                "sha256": reverse_iteration_v2_authorization["sha256"],
                "contract_id": reverse_iteration_v2_authorization[
                    "contract_id"
                ],
                "status": reverse_iteration_v2_authorization["payload"][
                    "status"
                ],
                "semantic_audit": reverse_iteration_v2_authorization[
                    "semantic_audit"
                ],
                "bound_causal_inputs": reverse_iteration_v2_authorization[
                    "bound_causal_inputs"
                ],
                "legacy_reward_config_audit": (
                    reverse_iteration_v2_legacy_reward_audit
                ),
                "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                "adoption_release_hardware": "PROHIBITED",
            }
            if reverse_iteration_v2_authorization
            else None
        ),
        "reverse_teacher_startup_audit": startup_audit,
        "promotion_evidence": promotion,
        "promotion_protocol": {
            "candidate_stage_interactions": DEFAULT_PILOT_TIMESTEPS,
            "candidate_training_steps_of_50000_interactions": 5,
            "gate_status": PROMOTION_GATE_STATUS,
            "gate": "closed until canonical full-P0 raw-trajectory rederivation exists",
            "fixed_failure3_seeds": list(H4_STRICT_PROMOTION_SEEDS[args.expert]),
            "promoted_stage_interactions": PROMOTED_TIMESTEPS,
        },
        "parent_checkpoint": (
            str(initialization_artifact_path)
            if h4_parent is None or h5_seed is not None
            else None
        ),
        "pinned_v22_parent_tree_sha256": (
            PINNED_V22_PARENT_TREE_SHA256
            if h4_parent is None and (h5_seed is None or h5_teacher_only)
            else None
        ),
        "trusted_h4_parent": (
            {
                "params_path": str(h4_parent["params_path"]),
                "params_sha256": h4_parent["params_sha256"],
                "manifest_path": str(h4_parent["manifest_path"]),
                "manifest_sha256": h4_parent["manifest_sha256"],
                "run_name": h4_parent["run_name"],
                "resolved_config_sha256": h4_parent[
                    "resolved_config_sha256"
                ],
            }
            if h4_parent
            else None
        ),
        "output_dir": str(run_dir),
        "network_factory": network_config,
        "ppo": serializable_training,
        "save_policy": "unique final params only; no intermediate checkpoints",
        "control_first_startup_exception": {
            "joint": "left_knee",
            "safe_init_rad": 0.470534,
            "steady_target_envelope_upper_rad": 0.425534,
            "initial_raw_gap_rad": 0.045,
            "first_applied_delta_rad": 0.04,
            "second_applied_delta_rad": 0.005,
            "status": "SOLE_AUTHORIZED_RESET_BOUNDARY_FEASIBILITY_EXCEPTION",
            "teleport": False,
        },
        "runtime_versions": trainer._runtime_versions(),
    }
    if args.forward_iteration_v3_touchdown_balance:
        resolved_config.update(
            {
                "authorized_iteration_v3_250k_contract_id": (
                    authorized_iteration_v3_250k_contract_id
                ),
                "forward_iteration_v3_touchdown_balance": True,
                "forward_iteration_v3_touchdown_balance_authorization": {
                    "path": str(
                        forward_iteration_v3_touchdown_balance_authorization["path"]
                    ),
                    "sha256": forward_iteration_v3_touchdown_balance_authorization[
                        "sha256"
                    ],
                    "contract_id": (
                        forward_iteration_v3_touchdown_balance_authorization[
                            "contract_id"
                        ]
                    ),
                    "status": forward_iteration_v3_touchdown_balance_authorization[
                        "payload"
                    ]["status"],
                    "semantic_audit": (
                        forward_iteration_v3_touchdown_balance_authorization[
                            "semantic_audit"
                        ]
                    ),
                    "bound_causal_inputs": (
                        forward_iteration_v3_touchdown_balance_authorization[
                            "bound_causal_inputs"
                        ]
                    ),
                    "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                    "adoption_release_hardware": "PROHIBITED",
                },
            }
        )
    elif args.reverse_iteration_v3_no_target_imitation:
        resolved_config.update(
            {
                "authorized_iteration_v3_250k_contract_id": (
                    authorized_iteration_v3_250k_contract_id
                ),
                "reverse_iteration_v3_no_target_imitation": True,
                "reverse_iteration_v3_no_target_imitation_authorization": {
                    "path": str(
                        reverse_iteration_v3_no_target_imitation_authorization[
                            "path"
                        ]
                    ),
                    "sha256": reverse_iteration_v3_no_target_imitation_authorization[
                        "sha256"
                    ],
                    "contract_id": (
                        reverse_iteration_v3_no_target_imitation_authorization[
                            "contract_id"
                        ]
                    ),
                    "status": reverse_iteration_v3_no_target_imitation_authorization[
                        "payload"
                    ]["status"],
                    "semantic_audit": (
                        reverse_iteration_v3_no_target_imitation_authorization[
                            "semantic_audit"
                        ]
                    ),
                    "bound_causal_inputs": (
                        reverse_iteration_v3_no_target_imitation_authorization[
                            "bound_causal_inputs"
                        ]
                    ),
                    "legacy_reward_config_audit": (
                        reverse_iteration_v3_no_target_imitation_legacy_reward_audit
                    ),
                    "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                    "adoption_release_hardware": "PROHIBITED",
                },
            }
        )
    if args.forward_iteration_v4_contact_event_validity_persistence:
        resolved_config.update(
            {
                "authorized_iteration_v4_250k_contract_id": (
                    authorized_iteration_v4_250k_contract_id
                ),
                "forward_iteration_v4_contact_event_validity_persistence": True,
                "forward_v4_substep_contact": True,
                "forward_v4_source_semantic_preflight": (
                    forward_v4_source_semantic_preflight
                ),
                "forward_v4_single_authority_runtime_requirement": dict(
                    FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT
                ),
                "forward_v4_single_authority_runtime_audit_mode": (
                    FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                    if args.wiring_only
                    else FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                ),
                "forward_iteration_v4_contact_event_validity_persistence_authorization": {
                    "path": str(
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "path"
                        ]
                    ),
                    "sha256": (
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "sha256"
                        ]
                    ),
                    "contract_id": (
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "contract_id"
                        ]
                    ),
                    "status": (
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "payload"
                        ]["status"]
                    ),
                    "semantic_audit": (
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "semantic_audit"
                        ]
                    ),
                    "bound_causal_inputs": (
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "bound_causal_inputs"
                        ]
                    ),
                    "bound_causal_sources": (
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "bound_causal_sources"
                        ]
                    ),
                    "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                    "adoption_release_hardware": "PROHIBITED",
                },
            }
        )
    elif args.reverse_iteration_v4_residual_transfer_gain_024:
        resolved_config.update(
            {
                "authorized_iteration_v4_250k_contract_id": (
                    authorized_iteration_v4_250k_contract_id
                ),
                "reverse_iteration_v4_residual_transfer_gain_024": True,
                "reverse_iteration_v4_residual_transfer_gain_024_authorization": {
                    "path": str(
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "path"
                        ]
                    ),
                    "sha256": (
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "sha256"
                        ]
                    ),
                    "contract_id": (
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "contract_id"
                        ]
                    ),
                    "status": (
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "payload"
                        ]["status"]
                    ),
                    "semantic_audit": (
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "semantic_audit"
                        ]
                    ),
                    "bound_causal_inputs": (
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "bound_causal_inputs"
                        ]
                    ),
                    "bound_causal_sources": (
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "bound_causal_sources"
                        ]
                    ),
                    "legacy_reward_config_audit": (
                        reverse_iteration_v4_residual_transfer_gain_024_legacy_reward_audit
                    ),
                    "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                    "adoption_release_hardware": "PROHIBITED",
                },
            }
        )
    if args.forward_v5_contact_pulse_abort_scale_only:
        resolved_config.update(
            {
                "authorized_iteration_v5_250k_contract_id": (
                    authorized_iteration_v5_250k_contract_id
                ),
                "forward_v5_contact_pulse_abort_scale_only": True,
                "forward_v4_substep_contact": True,
                "forward_v4_source_semantic_preflight": (
                    forward_v4_source_semantic_preflight
                ),
                "forward_v4_single_authority_runtime_requirement": dict(
                    FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT
                ),
                "forward_v4_single_authority_runtime_audit_mode": (
                    FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                    if args.wiring_only
                    else FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                ),
                "forward_iteration_v5_contact_pulse_abort_scale_only_authorization": {
                    "path": str(
                        forward_v5_contact_pulse_abort_scale_only_authorization["path"]
                    ),
                    "sha256": forward_v5_contact_pulse_abort_scale_only_authorization[
                        "sha256"
                    ],
                    "contract_id": (
                        forward_v5_contact_pulse_abort_scale_only_authorization[
                            "contract_id"
                        ]
                    ),
                    "status": forward_v5_contact_pulse_abort_scale_only_authorization[
                        "payload"
                    ]["status"],
                    "semantic_audit": (
                        forward_v5_contact_pulse_abort_scale_only_authorization[
                            "semantic_audit"
                        ]
                    ),
                    "bound_causal_inputs": (
                        forward_v5_contact_pulse_abort_scale_only_authorization[
                            "bound_causal_inputs"
                        ]
                    ),
                    "bound_historical_v4_sources": (
                        forward_v5_contact_pulse_abort_scale_only_authorization[
                            "bound_historical_v4_sources"
                        ]
                    ),
                    "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                    "adoption_release_hardware": "PROHIBITED",
                },
            }
        )
    elif args.reverse_iteration_v5_no_contact_imitation:
        resolved_config.update(
            {
                "authorized_iteration_v5_250k_contract_id": (
                    authorized_iteration_v5_250k_contract_id
                ),
                "reverse_iteration_v5_no_contact_imitation": True,
                "reverse_iteration_v5_no_contact_imitation_authorization": {
                    "path": str(
                        reverse_iteration_v5_no_contact_imitation_authorization["path"]
                    ),
                    "sha256": reverse_iteration_v5_no_contact_imitation_authorization[
                        "sha256"
                    ],
                    "contract_id": reverse_iteration_v5_no_contact_imitation_authorization[
                        "contract_id"
                    ],
                    "status": reverse_iteration_v5_no_contact_imitation_authorization[
                        "payload"
                    ]["status"],
                    "semantic_audit": reverse_iteration_v5_no_contact_imitation_authorization[
                        "semantic_audit"
                    ],
                    "bound_causal_inputs": reverse_iteration_v5_no_contact_imitation_authorization[
                        "bound_causal_inputs"
                    ],
                    "bound_historical_v4_sources": reverse_iteration_v5_no_contact_imitation_authorization[
                        "bound_historical_v4_sources"
                    ],
                    "legacy_reward_config_audit": (
                        reverse_iteration_v5_no_contact_imitation_legacy_reward_audit
                    ),
                    "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                    "adoption_release_hardware": "PROHIBITED",
                    "rejected_v4_diagnostic_promotion_allowed": False,
                },
            }
        )
    if args.forward_iteration_v6_contact_abort_island_only:
        resolved_config.update(
            {
                "authorized_iteration_v6_250k_contract_id": (
                    authorized_iteration_v6_250k_contract_id
                ),
                "iteration_v6_core_source": dict(iteration_v6_core_source),
                "forward_iteration_v6_contact_abort_island_only": True,
                "forward_v4_substep_contact": True,
                "forward_v4_source_semantic_preflight": (
                    forward_v4_source_semantic_preflight
                ),
                "forward_v4_single_authority_runtime_requirement": dict(
                    FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT
                ),
                "forward_v4_single_authority_runtime_audit_mode": (
                    FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                    if args.wiring_only
                    else FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                ),
                "forward_iteration_v6_reward_routing_runtime_requirement": dict(
                    FORWARD_ITERATION_V6_REWARD_ROUTING_RUNTIME_REQUIREMENT
                ),
                "reward_routing_contract": dict(
                    forward_iteration_v6_contact_abort_island_only_authorization[
                        "payload"
                    ]["reward_routing_contract"]
                ),
                "forward_iteration_v6_contact_abort_island_only_authorization": {
                    "path": str(
                        forward_iteration_v6_contact_abort_island_only_authorization[
                            "path"
                        ]
                    ),
                    "sha256": forward_iteration_v6_contact_abort_island_only_authorization[
                        "sha256"
                    ],
                    "contract_id": forward_iteration_v6_contact_abort_island_only_authorization[
                        "contract_id"
                    ],
                    "status": forward_iteration_v6_contact_abort_island_only_authorization[
                        "payload"
                    ]["status"],
                    "semantic_audit": forward_iteration_v6_contact_abort_island_only_authorization[
                        "semantic_audit"
                    ],
                    "bound_causal_inputs": forward_iteration_v6_contact_abort_island_only_authorization[
                        "bound_causal_inputs"
                    ],
                    "bound_historical_v5_sources": forward_iteration_v6_contact_abort_island_only_authorization[
                        "bound_historical_v5_sources"
                    ],
                    "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                    "adoption_release_hardware": "PROHIBITED",
                },
            }
        )
    elif args.reverse_iteration_v6_absolute_full_leg_targets:
        resolved_config.update(
            {
                "authorized_iteration_v6_250k_contract_id": (
                    authorized_iteration_v6_250k_contract_id
                ),
                "iteration_v6_core_source": dict(iteration_v6_core_source),
                "reverse_iteration_v6_absolute_full_leg_targets": True,
                "reverse_iteration_v6_decoder_runtime_requirement": dict(
                    REVERSE_ITERATION_V6_DECODER_RUNTIME_REQUIREMENT
                ),
                "action_parameterization_contract": dict(
                    reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "payload"
                    ]["action_parameterization_contract"]
                ),
                "teacher_timing_contract": dict(
                    reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "payload"
                    ]["teacher_timing_contract"]
                ),
                "reverse_iteration_v6_absolute_full_leg_targets_authorization": {
                    "path": str(
                        reverse_iteration_v6_absolute_full_leg_targets_authorization[
                            "path"
                        ]
                    ),
                    "sha256": reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "sha256"
                    ],
                    "contract_id": reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "contract_id"
                    ],
                    "status": reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "payload"
                    ]["status"],
                    "semantic_audit": reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "semantic_audit"
                    ],
                    "bound_causal_inputs": reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "bound_causal_inputs"
                    ],
                    "bound_historical_v5_sources": reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "bound_historical_v5_sources"
                    ],
                    "legacy_reward_config_audit": (
                        reverse_iteration_v6_absolute_full_leg_targets_legacy_reward_audit
                    ),
                    "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                    "adoption_release_hardware": "PROHIBITED",
                    "h4_parent_checkpoint_allowed": False,
                    "v4_gain_inherited": False,
                    "v5_parent_checkpoint_inherited": False,
                },
            }
        )
    config_path = run_dir / "resolved_config.json"
    config_path.write_text(
        json.dumps(resolved_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "STARTED",
        "hardware_deployment": "PROHIBITED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "expert": args.expert,
        "training_contract_id": resolved_config["training_contract_id"],
        "authorized_iteration_v2_250k_contract_id": (
            authorized_iteration_v2_250k_contract_id
        ),
        "authorized_iteration_v3_250k_contract_id": (
            authorized_iteration_v3_250k_contract_id
        ),
        "authorized_iteration_v4_250k_contract_id": (
            authorized_iteration_v4_250k_contract_id
        ),
        "authorized_iteration_v5_250k_contract_id": (
            authorized_iteration_v5_250k_contract_id
        ),
        "authorized_iteration_v6_250k_contract_id": (
            authorized_iteration_v6_250k_contract_id
        ),
        "qualification_use": resolved_config["qualification_use"],
        "diagnostic_reward_exploration": bool(
            getattr(args, "diagnostic_reward_exploration", False)
        ),
        "unified_development_run": bool(
            getattr(args, "unified_development_run", False)
        ),
        "forward_iteration_v2": bool(args.forward_iteration_v2),
        "reverse_iteration_v2": bool(args.reverse_iteration_v2),
        "forward_iteration_v3_touchdown_balance": bool(
            args.forward_iteration_v3_touchdown_balance
        ),
        "reverse_iteration_v3_no_target_imitation": bool(
            args.reverse_iteration_v3_no_target_imitation
        ),
        "forward_iteration_v4_contact_event_validity_persistence": bool(
            args.forward_iteration_v4_contact_event_validity_persistence
        ),
        "reverse_iteration_v4_residual_transfer_gain_024": bool(
            args.reverse_iteration_v4_residual_transfer_gain_024
        ),
        "forward_v5_contact_pulse_abort_scale_only": bool(
            args.forward_v5_contact_pulse_abort_scale_only
        ),
        "reverse_iteration_v5_no_contact_imitation": bool(
            args.reverse_iteration_v5_no_contact_imitation
        ),
        "forward_iteration_v6_contact_abort_island_only": bool(
            args.forward_iteration_v6_contact_abort_island_only
        ),
        "reverse_iteration_v6_absolute_full_leg_targets": bool(
            args.reverse_iteration_v6_absolute_full_leg_targets
        ),
        "forward_v4_substep_contact": bool(
            args.forward_iteration_v4_contact_event_validity_persistence
            or args.forward_v5_contact_pulse_abort_scale_only
            or args.forward_iteration_v6_contact_abort_island_only
            or (
                getattr(args, "diagnostic_reward_exploration", False)
                and args.expert in {"planar", "reverse", "unified"}
            )
        ),
        "forward_iteration_v2_authorization": (
            {
                "path": str(forward_iteration_v2_authorization["path"]),
                "sha256": forward_iteration_v2_authorization["sha256"],
                "contract_id": forward_iteration_v2_authorization[
                    "contract_id"
                ],
            }
            if forward_iteration_v2_authorization
            else None
        ),
        "reverse_iteration_v2_authorization": (
            {
                "path": str(reverse_iteration_v2_authorization["path"]),
                "sha256": reverse_iteration_v2_authorization["sha256"],
                "contract_id": reverse_iteration_v2_authorization[
                    "contract_id"
                ],
                "legacy_reward_config_audit": (
                    reverse_iteration_v2_legacy_reward_audit
                ),
            }
            if reverse_iteration_v2_authorization
            else None
        ),
        "wiring_only": bool(args.wiring_only),
        "activity": (
            "PPO_WIRING_TRAINING" if args.wiring_only else "PPO_PILOT_TRAINING"
        ),
        "requested_environment_interactions": shape.num_timesteps,
        "source_and_teacher_hashes_pre": pre_hashes,
        "source_and_teacher_hashes_post": None,
        "source_and_teacher_unchanged": None,
        "parent_checkpoint": {
            "kind": initialization_source,
            "path": str(initialization_artifact_path),
            "sha256_tree_pre": parent_checkpoint_sha_pre,
            "sha256_tree_post": None,
            "unchanged": None,
        },
        "resolved_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "canonical_sha256": canonical_json_sha(resolved_config),
        },
        "jax_devices": [str(device) for device in jax.devices()],
        "backend_resolution": backend_resolution,
        "xla_autotune_policy": xla_autotune_policy,
        "debug_callback_preflight": debug_callback_preflight,
        "pre_training_device_audits": pre_training_device_audits,
        "versions": trainer._runtime_versions(),
        "checkpoint_compatibility": checkpoint_audit,
        "notes": [
            "Simulation-only training; hardware deployment is prohibited.",
            (
                "This run performs PPO optimizer updates for wiring validation; "
                "it is not a no-training environment smoke."
                if args.wiring_only
                else "This run performs explicitly authorized PPO pilot training."
            ),
            "Physical reward commands remain distinct from mapped policy observations.",
            "No intermediate checkpoints are written.",
        ],
    }
    if args.forward_iteration_v3_touchdown_balance:
        manifest.update(
            {
                "authorized_iteration_v3_250k_contract_id": (
                    authorized_iteration_v3_250k_contract_id
                ),
                "forward_iteration_v3_touchdown_balance": True,
                "forward_iteration_v3_touchdown_balance_authorization": {
                    "path": str(
                        forward_iteration_v3_touchdown_balance_authorization["path"]
                    ),
                    "sha256": forward_iteration_v3_touchdown_balance_authorization[
                        "sha256"
                    ],
                    "contract_id": (
                        forward_iteration_v3_touchdown_balance_authorization[
                            "contract_id"
                        ]
                    ),
                },
            }
        )
    elif args.reverse_iteration_v3_no_target_imitation:
        manifest.update(
            {
                "authorized_iteration_v3_250k_contract_id": (
                    authorized_iteration_v3_250k_contract_id
                ),
                "reverse_iteration_v3_no_target_imitation": True,
                "reverse_iteration_v3_no_target_imitation_authorization": {
                    "path": str(
                        reverse_iteration_v3_no_target_imitation_authorization[
                            "path"
                        ]
                    ),
                    "sha256": reverse_iteration_v3_no_target_imitation_authorization[
                        "sha256"
                    ],
                    "contract_id": (
                        reverse_iteration_v3_no_target_imitation_authorization[
                            "contract_id"
                        ]
                    ),
                    "legacy_reward_config_audit": (
                        reverse_iteration_v3_no_target_imitation_legacy_reward_audit
                    ),
                },
            }
        )
    if args.forward_iteration_v4_contact_event_validity_persistence:
        manifest.update(
            {
                "authorized_iteration_v4_250k_contract_id": (
                    authorized_iteration_v4_250k_contract_id
                ),
                "forward_iteration_v4_contact_event_validity_persistence": True,
                "forward_v4_substep_contact": True,
                "forward_v4_source_semantic_preflight": (
                    forward_v4_source_semantic_preflight
                ),
                "forward_v4_single_authority_runtime_requirement": dict(
                    FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT
                ),
                "forward_v4_single_authority_runtime_audit_mode": (
                    FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                    if args.wiring_only
                    else FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                ),
                "forward_iteration_v4_contact_event_validity_persistence_authorization": {
                    "path": str(
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "path"
                        ]
                    ),
                    "sha256": (
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "sha256"
                        ]
                    ),
                    "contract_id": (
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "contract_id"
                        ]
                    ),
                },
            }
        )
    elif args.reverse_iteration_v4_residual_transfer_gain_024:
        manifest.update(
            {
                "authorized_iteration_v4_250k_contract_id": (
                    authorized_iteration_v4_250k_contract_id
                ),
                "reverse_iteration_v4_residual_transfer_gain_024": True,
                "reverse_iteration_v4_residual_transfer_gain_024_authorization": {
                    "path": str(
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "path"
                        ]
                    ),
                    "sha256": (
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "sha256"
                        ]
                    ),
                    "contract_id": (
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "contract_id"
                        ]
                    ),
                    "legacy_reward_config_audit": (
                        reverse_iteration_v4_residual_transfer_gain_024_legacy_reward_audit
                    ),
                },
            }
        )
    if args.forward_v5_contact_pulse_abort_scale_only:
        manifest.update(
            {
                "authorized_iteration_v5_250k_contract_id": (
                    authorized_iteration_v5_250k_contract_id
                ),
                "forward_v5_contact_pulse_abort_scale_only": True,
                "forward_v4_substep_contact": True,
                "forward_v4_source_semantic_preflight": (
                    forward_v4_source_semantic_preflight
                ),
                "forward_v4_single_authority_runtime_requirement": dict(
                    FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT
                ),
                "forward_v4_single_authority_runtime_audit_mode": (
                    FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                    if args.wiring_only
                    else FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                ),
                "forward_iteration_v5_contact_pulse_abort_scale_only_authorization": {
                    "path": str(
                        forward_v5_contact_pulse_abort_scale_only_authorization["path"]
                    ),
                    "sha256": forward_v5_contact_pulse_abort_scale_only_authorization[
                        "sha256"
                    ],
                    "contract_id": forward_v5_contact_pulse_abort_scale_only_authorization[
                        "contract_id"
                    ],
                    "bound_historical_v4_sources": (
                        forward_v5_contact_pulse_abort_scale_only_authorization[
                            "bound_historical_v4_sources"
                        ]
                    ),
                },
            }
        )
    elif args.reverse_iteration_v5_no_contact_imitation:
        manifest.update(
            {
                "authorized_iteration_v5_250k_contract_id": (
                    authorized_iteration_v5_250k_contract_id
                ),
                "reverse_iteration_v5_no_contact_imitation": True,
                "reverse_iteration_v5_no_contact_imitation_authorization": {
                    "path": str(
                        reverse_iteration_v5_no_contact_imitation_authorization["path"]
                    ),
                    "sha256": reverse_iteration_v5_no_contact_imitation_authorization[
                        "sha256"
                    ],
                    "contract_id": reverse_iteration_v5_no_contact_imitation_authorization[
                        "contract_id"
                    ],
                    "bound_historical_v4_sources": (
                        reverse_iteration_v5_no_contact_imitation_authorization[
                            "bound_historical_v4_sources"
                        ]
                    ),
                    "legacy_reward_config_audit": (
                        reverse_iteration_v5_no_contact_imitation_legacy_reward_audit
                    ),
                    "rejected_v4_diagnostic_promotion_allowed": False,
                },
            }
        )
    if args.forward_iteration_v6_contact_abort_island_only:
        manifest.update(
            {
                "authorized_iteration_v6_250k_contract_id": (
                    authorized_iteration_v6_250k_contract_id
                ),
                "iteration_v6_core_source": dict(iteration_v6_core_source),
                "forward_iteration_v6_contact_abort_island_only": True,
                "forward_v4_substep_contact": True,
                "forward_v4_source_semantic_preflight": (
                    forward_v4_source_semantic_preflight
                ),
                "forward_v4_single_authority_runtime_requirement": dict(
                    FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT
                ),
                "forward_v4_single_authority_runtime_audit_mode": (
                    FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                    if args.wiring_only
                    else FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                ),
                "forward_iteration_v6_reward_routing_runtime_requirement": dict(
                    FORWARD_ITERATION_V6_REWARD_ROUTING_RUNTIME_REQUIREMENT
                ),
                "reward_routing_contract": dict(
                    forward_iteration_v6_contact_abort_island_only_authorization[
                        "payload"
                    ]["reward_routing_contract"]
                ),
                "forward_iteration_v6_contact_abort_island_only_authorization": {
                    "path": str(
                        forward_iteration_v6_contact_abort_island_only_authorization[
                            "path"
                        ]
                    ),
                    "sha256": forward_iteration_v6_contact_abort_island_only_authorization[
                        "sha256"
                    ],
                    "contract_id": forward_iteration_v6_contact_abort_island_only_authorization[
                        "contract_id"
                    ],
                    "bound_historical_v5_sources": (
                        forward_iteration_v6_contact_abort_island_only_authorization[
                            "bound_historical_v5_sources"
                        ]
                    ),
                },
            }
        )
    elif args.reverse_iteration_v6_absolute_full_leg_targets:
        manifest.update(
            {
                "authorized_iteration_v6_250k_contract_id": (
                    authorized_iteration_v6_250k_contract_id
                ),
                "iteration_v6_core_source": dict(iteration_v6_core_source),
                "reverse_iteration_v6_absolute_full_leg_targets": True,
                "reverse_iteration_v6_decoder_runtime_requirement": dict(
                    REVERSE_ITERATION_V6_DECODER_RUNTIME_REQUIREMENT
                ),
                "action_parameterization_contract": dict(
                    reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "payload"
                    ]["action_parameterization_contract"]
                ),
                "teacher_timing_contract": dict(
                    reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "payload"
                    ]["teacher_timing_contract"]
                ),
                "reverse_iteration_v6_absolute_full_leg_targets_authorization": {
                    "path": str(
                        reverse_iteration_v6_absolute_full_leg_targets_authorization[
                            "path"
                        ]
                    ),
                    "sha256": reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "sha256"
                    ],
                    "contract_id": reverse_iteration_v6_absolute_full_leg_targets_authorization[
                        "contract_id"
                    ],
                    "bound_historical_v5_sources": (
                        reverse_iteration_v6_absolute_full_leg_targets_authorization[
                            "bound_historical_v5_sources"
                        ]
                    ),
                    "legacy_reward_config_audit": (
                        reverse_iteration_v6_absolute_full_leg_targets_legacy_reward_audit
                    ),
                    "h4_parent_checkpoint_allowed": False,
                    "v4_gain_inherited": False,
                    "v5_parent_checkpoint_inherited": False,
                },
            }
        )
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    curve_path = run_dir / "training_curve.csv"
    curve_rows: list[dict[str, Any]] = []
    forward_v4_single_authority_progress_rows: list[dict[str, Any]] = []
    iteration_v6_runtime_progress_rows: list[dict[str, Any]] = []

    def progress(step: int, metrics: Mapping[str, Any]) -> None:
        row: dict[str, Any] = {"environment_interactions": int(step)}
        row.update(trainer._scalar_metrics(metrics))
        if (
            (
                args.forward_iteration_v4_contact_event_validity_persistence
                or args.forward_v5_contact_pulse_abort_scale_only
                or args.forward_iteration_v6_contact_abort_island_only
            )
            and "episode/length" in row
        ):
            forward_v4_single_authority_progress_rows.append(dict(row))
            require_forward_v4_single_authority_runtime_progress(
                forward_v4_single_authority_progress_rows,
            )
        if (
            (
                args.forward_iteration_v6_contact_abort_island_only
                or args.reverse_iteration_v6_absolute_full_leg_targets
            )
            and "episode/length" in row
        ):
            iteration_v6_runtime_progress_rows.append(dict(row))
        curve_rows.append(row)
        fields = sorted({key for item in curve_rows for key in item})
        with curve_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(curve_rows)

    try:
        make_policy, params, metrics = stack["ppo"].train(
            environment=env,
            eval_env=eval_env,
            network_factory=network_factory,
            randomization_fn=trainer.make_domain_randomizer(env.mj_model),
            wrap_env_fn=stack["wrapper"].wrap_for_brax_training,
            progress_fn=progress,
            **training,
        )
        del make_policy
        final_metrics, nonzero_metric_count = _require_finite_final_state(
            jax, params, metrics
        )
        _, post_training_checkpoint_audit = require_checkpoint_observation_compatibility(
            params,
            actor_observation_width=expected_width,
        )
        post_training_device_audit = audit_jax_tree_placement(
            jax,
            params,
            expected_platform=expected_training_platform,
            label="post_training_params",
        )
        parent_checkpoint_sha_post = (
            trainer.sha256_tree(initialization_artifact_path)
            if h4_parent is None and (h5_seed is None or h5_teacher_only)
            else sha256_file(initialization_artifact_path)
        )
        if parent_checkpoint_sha_post != parent_checkpoint_sha_pre:
            raise RuntimeError("read-only parent checkpoint hash changed during PPO")
        precommit_post_hashes = _hash_snapshot(source_paths)
        _assert_unchanged(pre_hashes, precommit_post_hashes)
        forward_v4_single_authority_runtime = None
        if (
            args.forward_iteration_v4_contact_event_validity_persistence
            or args.forward_v5_contact_pulse_abort_scale_only
            or args.forward_iteration_v6_contact_abort_island_only
        ):
            wiring_completion = None
            if args.wiring_only:
                progress_interactions = [
                    int(row["environment_interactions"])
                    for row in curve_rows
                    if "environment_interactions" in row
                ]
                progress_reached_final_interaction = bool(
                    progress_interactions
                    and progress_interactions[-1] == shape.num_timesteps
                    and max(progress_interactions) == shape.num_timesteps
                    and all(
                        0 <= step <= shape.num_timesteps
                        for step in progress_interactions
                    )
                )
                wiring_completion = {
                    "source_semantic_preflight_passed": bool(
                        forward_v4_source_semantic_preflight is not None
                        and forward_v4_source_semantic_preflight.get("passed") is True
                    ),
                    "per_step_compiled_fail_closed_assertion_bound": bool(
                        getattr(env, "h4_forward_v4_substep_contact", False)
                        is True
                        and getattr(
                            eval_env, "h4_forward_v4_substep_contact", False
                        )
                        is True
                    ),
                    "completed_environment_interactions": shape.num_timesteps,
                    "completed_training_steps": shape.expected_training_steps,
                    "completed_optimizer_updates": (
                        shape.expected_optimizer_updates
                    ),
                    "progress_reached_final_interaction": (
                        progress_reached_final_interaction
                    ),
                    "final_params_all_finite": True,
                    "final_metrics_all_finite": True,
                    "source_and_teacher_unchanged": (
                        precommit_post_hashes == pre_hashes
                    ),
                }
            forward_v4_single_authority_runtime = (
                require_forward_v4_single_authority_runtime_progress(
                    forward_v4_single_authority_progress_rows,
                    wiring_only=bool(args.wiring_only),
                    wiring_completion=wiring_completion,
                )
            )
        iteration_v6_runtime = None
        if (
            args.forward_iteration_v6_contact_abort_island_only
            or args.reverse_iteration_v6_absolute_full_leg_targets
        ):
            progress_interactions = [
                int(row["environment_interactions"])
                for row in curve_rows
                if "environment_interactions" in row
            ]
            expected_flag = (
                "h4_forward_iteration_v6_contact_abort_island_only"
                if args.expert == "forward"
                else "h4_reverse_iteration_v6_absolute_full_leg_targets"
            )
            expected_contract_attr = (
                "h4_forward_iteration_v6_contract_id"
                if args.expert == "forward"
                else "h4_reverse_iteration_v6_contract_id"
            )
            expected_core_contract = (
                FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID
                if args.expert == "forward"
                else REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID
            )
            v6_completion = {
                "per_step_compiled_fail_closed_assertion_bound": bool(
                    getattr(env, expected_flag, False) is True
                    and getattr(eval_env, expected_flag, False) is True
                    and getattr(env, expected_contract_attr, None)
                    == expected_core_contract
                    and getattr(eval_env, expected_contract_attr, None)
                    == expected_core_contract
                ),
                "completed_environment_interactions": shape.num_timesteps,
                "completed_training_steps": shape.expected_training_steps,
                "completed_optimizer_updates": shape.expected_optimizer_updates,
                "progress_reached_final_interaction": bool(
                    progress_interactions
                    and progress_interactions[-1] == shape.num_timesteps
                    and max(progress_interactions) == shape.num_timesteps
                    and all(
                        0 <= step <= shape.num_timesteps
                        for step in progress_interactions
                    )
                ),
                "final_params_all_finite": True,
                "final_metrics_all_finite": True,
                "source_and_teacher_unchanged": (
                    precommit_post_hashes == pre_hashes
                ),
            }
            iteration_v6_runtime = require_iteration_v6_runtime_progress(
                iteration_v6_runtime_progress_rows,
                expert=args.expert,
                wiring_only=bool(args.wiring_only),
                wiring_completion=(v6_completion if args.wiring_only else None),
                full_completion=(None if args.wiring_only else v6_completion),
            )
        params_path = run_dir / "final_params.pkl"
        trainer._save_params(params_path, jax, params)
        post_hashes = _hash_snapshot(source_paths)
        _assert_unchanged(pre_hashes, post_hashes)
        result = {
            "schema_version": 1,
            "status": "WIRING_PASS" if args.wiring_only else "COMPLETED",
            "hardware_deployment": "PROHIBITED",
            "activity": (
                "PPO_WIRING_TRAINING"
                if args.wiring_only
                else "PPO_PILOT_TRAINING"
            ),
            "expert": args.expert,
            "training_contract_id": resolved_config["training_contract_id"],
            "authorized_iteration_v2_250k_contract_id": (
                authorized_iteration_v2_250k_contract_id
            ),
            "authorized_iteration_v3_250k_contract_id": (
                authorized_iteration_v3_250k_contract_id
            ),
            "authorized_iteration_v4_250k_contract_id": (
                authorized_iteration_v4_250k_contract_id
            ),
            "authorized_iteration_v5_250k_contract_id": (
                authorized_iteration_v5_250k_contract_id
            ),
            "authorized_iteration_v6_250k_contract_id": (
                authorized_iteration_v6_250k_contract_id
            ),
            "qualification_use": resolved_config["qualification_use"],
            "forward_iteration_v2": bool(args.forward_iteration_v2),
            "reverse_iteration_v2": bool(args.reverse_iteration_v2),
            "forward_iteration_v3_touchdown_balance": bool(
                args.forward_iteration_v3_touchdown_balance
            ),
            "reverse_iteration_v3_no_target_imitation": bool(
                args.reverse_iteration_v3_no_target_imitation
            ),
            "forward_iteration_v4_contact_event_validity_persistence": bool(
                args.forward_iteration_v4_contact_event_validity_persistence
            ),
            "reverse_iteration_v4_residual_transfer_gain_024": bool(
                args.reverse_iteration_v4_residual_transfer_gain_024
            ),
            "forward_v5_contact_pulse_abort_scale_only": bool(
                args.forward_v5_contact_pulse_abort_scale_only
            ),
            "reverse_iteration_v5_no_contact_imitation": bool(
                args.reverse_iteration_v5_no_contact_imitation
            ),
            "forward_iteration_v6_contact_abort_island_only": bool(
                args.forward_iteration_v6_contact_abort_island_only
            ),
            "reverse_iteration_v6_absolute_full_leg_targets": bool(
                args.reverse_iteration_v6_absolute_full_leg_targets
            ),
            "forward_v4_substep_contact": bool(
                args.forward_iteration_v4_contact_event_validity_persistence
                or args.forward_v5_contact_pulse_abort_scale_only
                or args.forward_iteration_v6_contact_abort_island_only
                or (
                    getattr(args, "diagnostic_reward_exploration", False)
                    and args.expert in {"planar", "reverse", "unified"}
                )
            ),
            "forward_iteration_v2_authorization_sha256": (
                forward_iteration_v2_authorization["sha256"]
                if forward_iteration_v2_authorization
                else None
            ),
            "reverse_iteration_v2_authorization_sha256": (
                reverse_iteration_v2_authorization["sha256"]
                if reverse_iteration_v2_authorization
                else None
            ),
            "environment_interactions": shape.num_timesteps,
            "optimizer_updates": shape.expected_optimizer_updates,
            "final_metrics": final_metrics,
            "final_metrics_all_finite": True,
            "final_metrics_nonzero_count": nonzero_metric_count,
            "backend_resolution": backend_resolution,
            "xla_autotune_policy": xla_autotune_policy,
            "debug_callback_preflight": debug_callback_preflight,
            "pre_training_device_audits": pre_training_device_audits,
            "post_training_device_audit": post_training_device_audit,
            "checkpoint_compatibility": checkpoint_audit,
            "post_training_checkpoint_audit": post_training_checkpoint_audit,
            "final_params": {
                "path": str(params_path),
                "sha256": sha256_file(params_path),
            },
            "source_and_teacher_unchanged": True,
        }
        if args.forward_iteration_v3_touchdown_balance:
            result.update(
                {
                    "authorized_iteration_v3_250k_contract_id": (
                        authorized_iteration_v3_250k_contract_id
                    ),
                    "forward_iteration_v3_touchdown_balance": True,
                    "forward_iteration_v3_touchdown_balance_authorization_sha256": (
                        forward_iteration_v3_touchdown_balance_authorization[
                            "sha256"
                        ]
                    ),
                }
            )
        elif args.reverse_iteration_v3_no_target_imitation:
            result.update(
                {
                    "authorized_iteration_v3_250k_contract_id": (
                        authorized_iteration_v3_250k_contract_id
                    ),
                    "reverse_iteration_v3_no_target_imitation": True,
                    "reverse_iteration_v3_no_target_imitation_authorization_sha256": (
                        reverse_iteration_v3_no_target_imitation_authorization[
                            "sha256"
                        ]
                    ),
                }
            )
        if args.forward_iteration_v4_contact_event_validity_persistence:
            result.update(
                {
                    "authorized_iteration_v4_250k_contract_id": (
                        authorized_iteration_v4_250k_contract_id
                    ),
                    "forward_iteration_v4_contact_event_validity_persistence": True,
                    "forward_v4_substep_contact": True,
                    "forward_v4_source_semantic_preflight": (
                        forward_v4_source_semantic_preflight
                    ),
                    "forward_v4_single_authority_runtime_requirement": dict(
                        FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT
                    ),
                    "forward_v4_single_authority_runtime_audit_mode": (
                        FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                        if args.wiring_only
                        else FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                    ),
                    "forward_v4_single_authority_runtime": (
                        forward_v4_single_authority_runtime
                    ),
                    "forward_iteration_v4_contact_event_validity_persistence_authorization_sha256": (
                        forward_iteration_v4_contact_event_validity_persistence_authorization[
                            "sha256"
                        ]
                    ),
                }
            )
        elif args.reverse_iteration_v4_residual_transfer_gain_024:
            result.update(
                {
                    "authorized_iteration_v4_250k_contract_id": (
                        authorized_iteration_v4_250k_contract_id
                    ),
                    "reverse_iteration_v4_residual_transfer_gain_024": True,
                    "reverse_iteration_v4_residual_transfer_gain_024_authorization_sha256": (
                        reverse_iteration_v4_residual_transfer_gain_024_authorization[
                            "sha256"
                        ]
                    ),
                }
            )
        if args.forward_v5_contact_pulse_abort_scale_only:
            result.update(
                {
                    "authorized_iteration_v5_250k_contract_id": (
                        authorized_iteration_v5_250k_contract_id
                    ),
                    "forward_v5_contact_pulse_abort_scale_only": True,
                    "forward_v4_substep_contact": True,
                    "forward_v4_source_semantic_preflight": (
                        forward_v4_source_semantic_preflight
                    ),
                    "forward_v4_single_authority_runtime_requirement": dict(
                        FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT
                    ),
                    "forward_v4_single_authority_runtime_audit_mode": (
                        FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                        if args.wiring_only
                        else FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                    ),
                    "forward_v4_single_authority_runtime": (
                        forward_v4_single_authority_runtime
                    ),
                    "forward_iteration_v5_contact_pulse_abort_scale_only_authorization_sha256": (
                        forward_v5_contact_pulse_abort_scale_only_authorization[
                            "sha256"
                        ]
                    ),
                }
            )
        elif args.reverse_iteration_v5_no_contact_imitation:
            result.update(
                {
                    "authorized_iteration_v5_250k_contract_id": (
                        authorized_iteration_v5_250k_contract_id
                    ),
                    "reverse_iteration_v5_no_contact_imitation": True,
                    "reverse_iteration_v5_no_contact_imitation_authorization_sha256": (
                        reverse_iteration_v5_no_contact_imitation_authorization[
                            "sha256"
                        ]
                    ),
                    "legacy_reward_config_audit": (
                        reverse_iteration_v5_no_contact_imitation_legacy_reward_audit
                    ),
                    "backward_residual_scale": 0.12,
                    "rejected_v4_diagnostic_promotion_allowed": False,
                }
            )
        if args.forward_iteration_v6_contact_abort_island_only:
            result.update(
                {
                    "authorized_iteration_v6_250k_contract_id": (
                        authorized_iteration_v6_250k_contract_id
                    ),
                    "iteration_v6_core_source": dict(iteration_v6_core_source),
                    "forward_iteration_v6_contact_abort_island_only": True,
                    "forward_v4_substep_contact": True,
                    "forward_v4_source_semantic_preflight": (
                        forward_v4_source_semantic_preflight
                    ),
                    "forward_v4_single_authority_runtime_requirement": dict(
                        FORWARD_V4_SINGLE_AUTHORITY_RUNTIME_REQUIREMENT
                    ),
                    "forward_v4_single_authority_runtime_audit_mode": (
                        FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                        if args.wiring_only
                        else FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                    ),
                    "forward_v4_single_authority_runtime": (
                        forward_v4_single_authority_runtime
                    ),
                    "forward_iteration_v6_reward_routing_runtime_requirement": dict(
                        FORWARD_ITERATION_V6_REWARD_ROUTING_RUNTIME_REQUIREMENT
                    ),
                    "reward_routing_contract": dict(
                        forward_iteration_v6_contact_abort_island_only_authorization[
                            "payload"
                        ]["reward_routing_contract"]
                    ),
                    "forward_iteration_v6_reward_routing_runtime": (
                        iteration_v6_runtime
                    ),
                    "forward_iteration_v6_contact_abort_island_only_authorization_sha256": (
                        forward_iteration_v6_contact_abort_island_only_authorization[
                            "sha256"
                        ]
                    ),
                }
            )
        elif args.reverse_iteration_v6_absolute_full_leg_targets:
            result.update(
                {
                    "authorized_iteration_v6_250k_contract_id": (
                        authorized_iteration_v6_250k_contract_id
                    ),
                    "iteration_v6_core_source": dict(iteration_v6_core_source),
                    "reverse_iteration_v6_absolute_full_leg_targets": True,
                    "reverse_iteration_v6_decoder_runtime_requirement": dict(
                        REVERSE_ITERATION_V6_DECODER_RUNTIME_REQUIREMENT
                    ),
                    "action_parameterization_contract": dict(
                        reverse_iteration_v6_absolute_full_leg_targets_authorization[
                            "payload"
                        ]["action_parameterization_contract"]
                    ),
                    "teacher_timing_contract": dict(
                        reverse_iteration_v6_absolute_full_leg_targets_authorization[
                            "payload"
                        ]["teacher_timing_contract"]
                    ),
                    "reverse_iteration_v6_decoder_runtime": iteration_v6_runtime,
                    "reverse_iteration_v6_absolute_full_leg_targets_authorization_sha256": (
                        reverse_iteration_v6_absolute_full_leg_targets_authorization[
                            "sha256"
                        ]
                    ),
                    "legacy_reward_config_audit": (
                        reverse_iteration_v6_absolute_full_leg_targets_legacy_reward_audit
                    ),
                    "backward_residual_scale": 0.0,
                    "teacher_target_contribution_zero": True,
                    "h4_parent_checkpoint_allowed": False,
                    "v4_gain_inherited": False,
                    "v5_parent_checkpoint_inherited": False,
                }
            )
        iteration_v6_artifact_cross_binding = None
        if (
            args.forward_iteration_v6_contact_abort_island_only
            or args.reverse_iteration_v6_absolute_full_leg_targets
        ):
            runtime_key = (
                "forward_iteration_v6_reward_routing_runtime"
                if args.expert == "forward"
                else "reverse_iteration_v6_decoder_runtime"
            )
            authorization = (
                forward_iteration_v6_contact_abort_island_only_authorization
                if args.expert == "forward"
                else reverse_iteration_v6_absolute_full_leg_targets_authorization
            )
            manifest_for_cross_binding = {
                **manifest,
                runtime_key: iteration_v6_runtime,
            }
            iteration_v6_artifact_cross_binding = (
                require_iteration_v6_artifact_cross_binding(
                    resolved_config,
                    manifest_for_cross_binding,
                    result,
                    authorization,
                    expert=args.expert,
                )
            )
            result["iteration_v6_artifact_cross_binding"] = (
                iteration_v6_artifact_cross_binding
            )
        result_path = run_dir / "run_result.json"
        result_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        manifest.update(
            {
                "status": result["status"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_and_teacher_hashes_post": post_hashes,
                "source_and_teacher_unchanged": True,
                **(
                    {
                        "forward_v4_single_authority_runtime": (
                            forward_v4_single_authority_runtime
                        )
                    }
                    if (
                        args.forward_iteration_v4_contact_event_validity_persistence
                        or args.forward_v5_contact_pulse_abort_scale_only
                        or args.forward_iteration_v6_contact_abort_island_only
                    )
                    else {}
                ),
                **(
                    {
                        "iteration_v6_artifact_cross_binding": (
                            iteration_v6_artifact_cross_binding
                        )
                    }
                    if iteration_v6_artifact_cross_binding is not None
                    else {}
                ),
                **(
                    {
                        "forward_iteration_v6_reward_routing_runtime": (
                            iteration_v6_runtime
                        )
                    }
                    if args.forward_iteration_v6_contact_abort_island_only
                    else {
                        "reverse_iteration_v6_decoder_runtime": (
                            iteration_v6_runtime
                        )
                    }
                    if args.reverse_iteration_v6_absolute_full_leg_targets
                    else {}
                ),
                "parent_checkpoint": {
                    "kind": initialization_source,
                    "path": str(initialization_artifact_path),
                    "sha256_tree_pre": parent_checkpoint_sha_pre,
                    "sha256_tree_post": parent_checkpoint_sha_post,
                    "unchanged": True,
                },
                "outputs": {
                    "final_params": result["final_params"],
                    "result": {
                        "path": str(result_path),
                        "sha256": sha256_file(result_path),
                    },
                    "training_curve": (
                        {
                            "path": str(curve_path),
                            "sha256": sha256_file(curve_path),
                        }
                        if curve_path.exists()
                        else None
                    ),
                },
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return {**result, "run_dir": str(run_dir)}
    except Exception as exc:
        post_hashes = _hash_snapshot(source_paths)
        unchanged = pre_hashes == post_hashes
        parent_checkpoint_sha_post = (
            trainer.sha256_tree(initialization_artifact_path)
            if h4_parent is None and (h5_seed is None or h5_teacher_only)
            else sha256_file(initialization_artifact_path)
        )
        manifest.update(
            {
                "status": "FAILED",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "source_and_teacher_hashes_post": post_hashes,
                "source_and_teacher_unchanged": unchanged,
                "parent_checkpoint": {
                    "kind": initialization_source,
                    "path": str(initialization_artifact_path),
                    "sha256_tree_pre": parent_checkpoint_sha_pre,
                    "sha256_tree_post": parent_checkpoint_sha_post,
                    "unchanged": (
                        parent_checkpoint_sha_pre == parent_checkpoint_sha_post
                    ),
                },
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
