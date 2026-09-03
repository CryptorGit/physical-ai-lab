"""Build and verify self-contained SafeGaitRouter distribution packages.

The package graph is deliberately closed: every ``SafeGaitRouter`` outcome is
mapped to a model declared in the manifest, unknown expert names are rejected,
and there is no dynamic model lookup.  The rejected v59/v60 lineages therefore
cannot become reachable through a filename, fallback, or unchecked route.

This module only packages simulation artifacts.  It cannot promote a policy to
hardware use and validates ``hardware_deployment == \"PROHIBITED\"`` on every
read and write.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from router import (
    ALLOWED_EXPERTS,
    DEFAULT_COMMAND_MAX,
    DEFAULT_COMMAND_MIN,
    PROHIBITED_EXPERTS,
    REVERSE,
    REVERSE_TURN_LEFT,
    REVERSE_TURN_ENDPOINTS,
    REVERSE_TURN_RIGHT,
    YAW_RIGHT,
)
from safe_gait_experts.contract import (
    BACKWARD_EXIT_RECOVERY_STATUS,
    validate_contract,
)
from target_safety import (
    BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT,
    BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
    BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
    BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
    BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
    CONTROL_FIRST_STARTUP_DT_S,
    PERTURBED_RESET_QPOS_MARGIN_RAD,
    RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD,
    RUNTIME_TARGET_SAFETY_MARGIN_RAD,
    RUNTIME_TARGET_SLEW_RATE_RAD_S,
)


SCHEMA_VERSION = 1
MANIFEST_KIND = "openduckmini_safe_gait_router_package"
EXPERIMENT = "exp_004_openduckmini_safe_gait_experts"
MANIFEST_FILENAME = "package_manifest.json"
FORMAL_EVALUATOR_ID = "openduckmini-exp004-routed-transition-v1"
FORMAL_RELEASE_MASTER_SEED = 20260808
FORMAL_RELEASE_STATUS = "ADOPTED_SIMULATION_ONLY"
FORMAL_RELEASE_EVIDENCE_SHA256 = (
    "95819b5bc1d0827a5ad779542a6f98c4aaebacf5f55a8303c0b5a14fba501674"
)
FORMAL_RELEASE_EVIDENCE_SIZE_BYTES = 18_611_839
# H3 remains immutable historical evidence, but it predates the strict H4 gait
# quality contract.  No package can be built until a new independent release
# qualification artifact is reviewed and pinned here.
FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST: frozenset[str] = frozenset()
FORMAL_RELEASE_SCALE = {
    "episodes": 20,
    "seconds": 30.0,
    "transition_seconds": 30.0,
    "transition_stand_seconds": 5.0,
    "warmup_seconds": 1.5,
    "initial_joint_noise_scale": 1.0,
    "initial_base_speed": 0.10,
    "master_seed": FORMAL_RELEASE_MASTER_SEED,
}
REVERSE_PHASE_ENTRY_RELEASE_STATUS = FORMAL_RELEASE_STATUS

BASE_V22_MODEL_ID = "base_v22"
REVERSE_MODEL_ID = "reverse_exp004"
BASE_V22_RELEASE_ID = "openduckmini_calibrated_hybrid_v22"
BASE_V22_SHA256 = (
    "f7a2731330cd3be52858989b021423a5f363cc4a8f9850512281da745a7617c0"
)
YAW_RIGHT_POLICY_OFFSET = -0.30
REVERSE_RESIDUAL_SCALE = 0.0
REVERSE_PROFILE_RELEASE_ID = (
    "optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1"
)
REVERSE_TURN_LEFT_PROFILE_RELEASE_ID = (
    "optimized_reverse_turn_left_margin050_slew200_candidate_v1"
)
REVERSE_TURN_RIGHT_PROFILE_RELEASE_ID = (
    "optimized_reverse_turn_right_margin050_slew200_candidate_v1"
)
REVERSE_PROFILE_SHA256 = (
    "0a3c0849124b397ca1cb60ae0b5f5783a2e545f1a03108846fa8c60cd5d8bb5b"
)
REVERSE_TURN_LEFT_PROFILE_SHA256 = (
    "b36f14dc1bbacfbf998adc00f6e6fe62d1f14a4a8de034b1b0b18ae5bccb8703"
)
REVERSE_TURN_RIGHT_PROFILE_SHA256 = (
    "e2229527d435d03636c091ca7b435ed3be483b0e74293d28a2ff927995bea16b"
)
REJECTED_LINEAGES = ("v59", "v60")

_H3_ADOPTION_EVIDENCE_SHA256 = (
    "1aea58904598cfba8ea4ef572f9473bba647eacc695f7fce3fcaa1b8646391aa"
)
_H3_SELECTION_EVIDENCE_SHA256 = (
    "f040a9c6f9783b7d50dd5590389d3c81411e8f3a7fa9dd155e8ac78175d5ff56"
)
_H3_SAFETY_COMPONENT_EVIDENCE_SHA256 = (
    "090e09cc2d82c1f42112a5f30a85cd93d940213956d6ec902fb4089875fb855a"
)
_H3_RELEASE_SOURCE_CLOSURE_SHA256 = (
    "5f8c97fc8cc0dda465f228d82b59cee9630a716e94e998f7a53ff8b6cf9ba833"
)
_H3_RELEASE_RUNTIME_DATA_CLOSURE_SHA256 = (
    "b7efb0ad2d1de3255cd498d3feebf327aac6dab03cad7e96b2e8f2b533c33182"
)

_PACKAGE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_LAYOUT = {
    "router": PurePosixPath("runtime/router.py"),
    "target_safety": PurePosixPath("runtime/target_safety.py"),
    "contract": PurePosixPath("contracts/contract.json"),
    "formal_release_evidence": PurePosixPath(
        "evidence/formal_release_qualification.json"
    ),
    "scene": PurePosixPath(
        "simulation/xmls/scene_flat_terrain_backlash_hardware_safe_calibrated.xml"
    ),
    "reference": PurePosixPath("simulation/reference.pkl"),
    "reverse_profile": PurePosixPath(
        "corrections/optimized_backward_gait.json"
    ),
    "reverse_turn_left_profile": PurePosixPath(
        "corrections/optimized_backward_left_turn_gait.json"
    ),
    "reverse_turn_right_profile": PurePosixPath(
        "corrections/optimized_backward_right_turn_gait.json"
    ),
    "base_v22_onnx": PurePosixPath("models/base_v22.onnx"),
    "reverse_onnx": PurePosixPath("models/reverse_exp004.onnx"),
    "reverse_export_report": PurePosixPath(
        "models/reverse_exp004.onnx.json"
    ),
}


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_source_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"{label} must not be empty: {resolved}")
    return resolved


def _required_mapping(
    parent: Mapping[str, Any], key: str, label: str
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"formal release evidence is missing {label}")
    return value


def _strict_json_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_validate_h3_release_audits(payload: Mapping[str, Any]) -> None:
    """Re-derive the complete H3 release audit from its nested records.

    The SHA-256 allowlist is the immutable identity gate.  This second,
    independent semantic gate prevents a future allowlist edit from promoting
    a malformed record whose top-level booleans happen to say ``passed``.
    Synthetic unit-test fixtures omit the H3 lineage record and continue to
    exercise the generic package format separately.
    """

    configuration = _required_mapping(
        payload, "configuration", "configuration"
    )
    if (
        configuration.get("formal_candidate_default") is not True
        or configuration.get("formal_candidate_status") is not None
        or configuration.get("formal_adopted_default") is not True
        or configuration.get("formal_adopted_status")
        != FORMAL_RELEASE_STATUS
        or configuration.get("backward_residual_scale") != 0.0
        or configuration.get("leg_target_margin_rad")
        != RUNTIME_TARGET_SAFETY_MARGIN_RAD
        or configuration.get("target_slew_rate_rad_per_s")
        != RUNTIME_TARGET_SLEW_RATE_RAD_S
        or configuration.get("reset_noise_margin_rad")
        != PERTURBED_RESET_QPOS_MARGIN_RAD
        or configuration.get("left_knee_extra_upper_margin_rad") != 0.0125
        or configuration.get("left_knee_profile_upper_target_rad")
        != 0.413034
        or configuration.get("backward_exit_recovery_enabled") is not True
        or any(
            configuration.get(key)
            for key in (
                "diagnostic_unadopted_policy",
                "diagnostic_unadopted_backward_exit_recovery",
                "diagnostic_noncontract_safety",
                "policy_command_diagnostic_suite",
            )
        )
        or any(
            configuration.get(key) is not None
            for key in (
                "diagnostic_unadopted_reverse_profile",
                "diagnostic_unadopted_reverse_left_profile",
                "diagnostic_unadopted_reverse_right_profile",
                "diagnostic_unadopted_reverse_entry_phase_indices",
            )
        )
        or dict(
            configuration.get("executed_reverse_entry_phase_indices", {})
        )
        != {
            "reverse": 7.0,
            "reverse_turn_left": 4.0,
            "reverse_turn_right": 4.0,
        }
    ):
        raise ValueError("formal H3 release configuration mismatch")

    adoption_evidence = _required_mapping(
        payload, "formal_adoption_evidence", "formal_adoption_evidence"
    )
    selection_evidence = _required_mapping(
        payload,
        "formal_candidate_selection_evidence",
        "formal_candidate_selection_evidence",
    )
    safety_evidence = _required_mapping(
        payload,
        "h3_fast_exit_safety_component_evidence",
        "h3_fast_exit_safety_component_evidence",
    )
    if (
        adoption_evidence.get("sha256") != _H3_ADOPTION_EVIDENCE_SHA256
        or adoption_evidence.get("status") != FORMAL_RELEASE_STATUS
        or adoption_evidence.get("adoption_eligible") is not True
        or adoption_evidence.get("simulation_acceptance_eligible") is not True
        or adoption_evidence.get("package_release_evidence") is not False
        or selection_evidence.get("sha256")
        != _H3_SELECTION_EVIDENCE_SHA256
        or selection_evidence.get("release_evidence") is not False
        or safety_evidence.get("sha256")
        != _H3_SAFETY_COMPONENT_EVIDENCE_SHA256
        or safety_evidence.get("safety_only_component") is not True
        or safety_evidence.get("release_evidence") is not False
        or any(
            record.get("hardware_deployment") != "PROHIBITED"
            for record in (
                adoption_evidence,
                selection_evidence,
                safety_evidence,
            )
        )
    ):
        raise ValueError("formal H3 release lineage evidence mismatch")

    suites = _required_mapping(payload, "suites", "suites")
    expected_suites = {
        "primitives": (list(range(20_260_808, 20_260_828)), 140),
        "compounds": (list(range(21_260_808, 21_260_828)), 120),
        "transitions": (list(range(22_260_808, 22_260_828)), 500),
    }
    if set(suites) != set(expected_suites):
        raise ValueError("formal H3 release suite set mismatch")

    episodes: list[Mapping[str, Any]] = []
    segments: list[Mapping[str, Any]] = []
    accepted_segments: list[Mapping[str, Any]] = []
    reset_audits: list[Mapping[str, Any]] = []
    startup_audits: list[Mapping[str, Any]] = []
    recovery_state_audits: list[Mapping[str, Any]] = []
    for suite_name, (expected_seeds, expected_segment_count) in (
        expected_suites.items()
    ):
        suite = suites[suite_name]
        if not isinstance(suite, Mapping):
            raise ValueError(f"formal H3 release {suite_name} is invalid")
        suite_episodes = suite.get("episodes")
        acceptance = suite.get("acceptance")
        checks = (
            acceptance.get("episode_checks")
            if isinstance(acceptance, Mapping)
            else None
        )
        if (
            not isinstance(suite_episodes, list)
            or not isinstance(checks, list)
            or [episode.get("seed") for episode in suite_episodes]
            != expected_seeds
            or [check.get("seed") for check in checks] != expected_seeds
            or acceptance.get("passed") is not True
        ):
            raise ValueError(
                f"formal H3 release {suite_name} episode acceptance mismatch"
            )
        suite_segments = [
            segment
            for episode in suite_episodes
            if isinstance(episode, Mapping)
            for segment in episode.get("segments", ())
            if isinstance(segment, Mapping)
        ]
        suite_accepted = [
            segment
            for check in checks
            if isinstance(check, Mapping)
            for segment in check.get("segments", ())
            if isinstance(segment, Mapping)
        ]
        if (
            len(suite_episodes) != 20
            or len(suite_segments) != expected_segment_count
            or len(suite_accepted) != expected_segment_count
            or not all(
                record.get("passed") is True
                and isinstance(record.get("checks"), Mapping)
                and len(record["checks"]) == 37
                and all(value is True for value in record["checks"].values())
                for record in suite_accepted
            )
        ):
            raise ValueError(
                f"formal H3 release {suite_name} segment acceptance mismatch"
            )
        for check in checks:
            reset_audits.extend(
                item
                for item in check.get("reset_qpos_audits", ())
                if isinstance(item, Mapping)
            )
            startup_audits.extend(
                item
                for item in check.get("control_first_startup_audits", ())
                if isinstance(item, Mapping)
            )
            recovery_state_audits.extend(
                item
                for item in check.get("backward_exit_recovery_audits", ())
                if isinstance(item, Mapping)
            )
        episodes.extend(suite_episodes)
        segments.extend(suite_segments)
        accepted_segments.extend(suite_accepted)

    if (
        len(episodes) != 60
        or len(segments) != 760
        or len(accepted_segments) != 760
        or sum(len(record["checks"]) for record in accepted_segments)
        != 28_120
        or not all(
            episode.get("fell") is False
            and episode.get("completed_segment_count")
            == episode.get("requested_segment_count")
            for episode in episodes
        )
        or not all(
            segment.get("completed") is True
            and segment.get("fell") is False
            and segment.get("completed_physics_substeps")
            == segment.get("expected_physics_substeps")
            for segment in segments
        )
    ):
        raise ValueError("formal H3 release completion totals mismatch")

    safety_zero_fields = (
        "applied_target_limit_violations",
        "desired_target_margin_violations",
        "nonfinite_sample_count",
        "preclip_target_limit_violations",
        "qpos_limit_violations",
        "target_slew_violations",
        "unauthorized_applied_target_margin_violations",
        "maximum_applied_target_excess_rad",
        "maximum_desired_target_margin_excess_rad",
        "maximum_preclip_target_excess_rad",
        "maximum_qpos_excess_rad",
        "applied_head_action_peak",
        "head_target_peak_rad",
    )
    physics_zero_fields = (
        "height_fall_samples",
        "upright_fall_samples",
        "nonfinite_full_qpos_samples",
        "nonfinite_full_qvel_samples",
        "nonfinite_leg_qpos_samples",
        "nonfinite_pose_samples",
        "nonfinite_state_samples",
        "qpos_limit_violations",
        "maximum_qpos_excess_rad",
    )
    physics_samples = 0
    contact_samples = 0
    leg_samples = 0
    control_samples = 0
    applied_margin_samples = 0
    startup_margin_samples = 0
    preclip_margin_samples = 0
    minimum_height = float("inf")
    minimum_upright = float("inf")
    maximum_left_knee = float("-inf")
    phase_audits: list[Mapping[str, Any]] = []
    recovery_audits: list[Mapping[str, Any]] = []
    for segment in segments:
        safety = segment.get("safety_audit")
        physics = segment.get("physics_substep_audit")
        routing = segment.get("routing")
        recovery = segment.get("backward_exit_recovery_audit")
        if not all(
            isinstance(value, Mapping)
            for value in (safety, physics, routing, recovery)
        ):
            raise ValueError("formal H3 release segment audit is incomplete")
        phase = routing.get("reverse_entry_phase")
        joint_maximum = physics.get("joint_qpos_max_rad")
        if (
            any(safety.get(field) != 0 for field in safety_zero_fields)
            or any(physics.get(field) != 0 for field in physics_zero_fields)
            or safety.get("applied_target_margin_violations")
            != safety.get("startup_margin_transition_joint_samples")
            or safety.get("leg_target_margin_rad")
            != RUNTIME_TARGET_SAFETY_MARGIN_RAD
            or safety.get("target_slew_limit_rad_per_s")
            != RUNTIME_TARGET_SLEW_RATE_RAD_S
            or safety.get("maximum_target_slew_rate_rad_per_s")
            > RUNTIME_TARGET_SLEW_RATE_RAD_S + 2e-15
            or physics.get("fall_or_nonfinite_detected") is not False
            or physics.get("contact_sample_count_matches_sample_count")
            is not True
            or physics.get("contact_sample_count")
            != physics.get("sample_count")
            or routing.get("command_clip_events") != 0
            or routing.get("prohibited_expert_steps") != 0
            or routing.get("atomic_endpoint_mismatch_steps") != 0
            or not isinstance(phase, Mapping)
            or phase.get("passed") is not True
            or recovery.get("passed") is not True
            or recovery.get("cap_violation_count") != 0
            or not isinstance(joint_maximum, Mapping)
        ):
            raise ValueError("formal H3 release safety audit failed")
        physics_samples += int(physics["sample_count"])
        contact_samples += int(physics["contact_sample_count"])
        leg_samples += int(physics["leg_joint_sample_count"])
        control_samples += int(safety["sample_count"])
        applied_margin_samples += int(
            safety["applied_target_margin_violations"]
        )
        startup_margin_samples += int(
            safety["startup_margin_transition_joint_samples"]
        )
        preclip_margin_samples += int(
            safety["preclip_target_margin_violations"]
        )
        minimum_height = min(minimum_height, float(physics["minimum_height_m"]))
        minimum_upright = min(
            minimum_upright, float(physics["minimum_upright"])
        )
        maximum_left_knee = max(
            maximum_left_knee, float(joint_maximum["left_knee"])
        )
        phase_audits.append(phase)
        recovery_audits.append(recovery)

    if (
        physics_samples != 8_150_000
        or contact_samples != 8_150_000
        or leg_samples != 81_500_000
        or control_samples != 815_000
        or applied_margin_samples != 147
        or startup_margin_samples != 147
        or preclip_margin_samples != 819_203
        or minimum_height != 0.17911993
        or minimum_upright != 0.9777608163890137
        or maximum_left_knee != 0.4736497298325716
        or 0.475534 - maximum_left_knee != 0.0018842701674284257
    ):
        raise ValueError("formal H3 release audited aggregate mismatch")

    phase_events = [
        event
        for audit in phase_audits
        for event in audit.get("events", ())
        if isinstance(event, Mapping)
    ]
    phase_indices = {
        "reverse": 7.0,
        "reverse_turn_left": 4.0,
        "reverse_turn_right": 4.0,
    }
    phase_counts = {
        expert: sum(
            event.get("current_expert") == expert for event in phase_events
        )
        for expert in phase_indices
    }
    if (
        len(phase_events) != 120
        or phase_counts != {expert: 40 for expert in phase_indices}
        or any(
            event.get("reset_preincrement_phase_index")
            != phase_indices.get(str(event.get("current_expert")))
            for event in phase_events
        )
        or sum(
            int(audit.get("exit_event_count", -1))
            for audit in recovery_audits
        )
        != 60
        or sum(
            int(audit.get("active_tick_count", -1))
            for audit in recovery_audits
        )
        != 780
        or sum(
            int(audit.get("cap_violation_count", -1))
            for audit in recovery_audits
        )
        != 0
        or sum(
            int(audit.get("sample_count", -1))
            for audit in recovery_audits
        )
        != 815_000
        or sum(
            int(audit.get("final_guard_call_count", -1))
            for audit in recovery_audits
        )
        != 815_000
    ):
        raise ValueError("formal H3 release phase/recovery totals mismatch")

    if (
        len(reset_audits) != 280
        or len(startup_audits) != 280
        or len(recovery_state_audits) != 280
        or not all(audit.get("passed") is True for audit in reset_audits)
        or not all(audit.get("passed") is True for audit in startup_audits)
        or not all(
            audit.get("passed") is True for audit in recovery_state_audits
        )
        or sum(
            int(audit.get("physical_safe_limit_violations", -1))
            for audit in reset_audits
        )
        != 0
        or sum(
            int(audit.get("noise_margin_violations", -1))
            for audit in reset_audits
        )
        != 0
        or any(audit.get("head_qpos_peak_rad") != 0.0 for audit in reset_audits)
        or any(
            audit.get("control_applied_before_first_physics_step") is not True
            or audit.get("exactly_one_guard_call_for_first_tick") is not True
            or audit.get("physics_steps_before_control") != 0
            or audit.get("guard_calls_for_first_tick") != 1
            for audit in startup_audits
        )
        or sum(
            int(audit.get("exit_event_count", -1))
            for audit in recovery_state_audits
        )
        != 60
        or sum(
            int(audit.get("active_tick_count", -1))
            for audit in recovery_state_audits
        )
        != 780
        or sum(
            int(audit.get("completed_event_count", -1))
            for audit in recovery_state_audits
        )
        != 60
        or sum(
            int(audit.get("cap_violation_count", -1))
            for audit in recovery_state_audits
        )
        != 0
        or sum(
            int(audit.get("remaining_ticks", -1))
            for audit in recovery_state_audits
        )
        != 0
        or sum(
            int(audit.get("control_tick_count", -1))
            for audit in recovery_state_audits
        )
        != 815_000
        or sum(
            int(audit.get("final_guard_call_count", -1))
            for audit in recovery_state_audits
        )
        != 815_000
    ):
        raise ValueError("formal H3 release physical-state audit mismatch")

    provenance = _required_mapping(
        payload,
        "runtime_dependency_provenance",
        "runtime_dependency_provenance",
    )
    pre_import = _required_mapping(
        provenance, "pre_import", "runtime_dependency_provenance.pre_import"
    )
    post_evaluation = _required_mapping(
        provenance,
        "post_evaluation",
        "runtime_dependency_provenance.post_evaluation",
    )
    closure_contract = {
        "exp004_source_and_contract_snapshot": (
            _H3_RELEASE_SOURCE_CLOSURE_SHA256,
            9,
        ),
        "external_hard_allowlisted_source_closure": (
            "a40d4920049b349334b6d5567859fc2f7533d8fe6648f5127f0c7e4ce54dc098",
            4,
        ),
        "hard_allowlisted_runtime_binary_closure": (
            "4e382762ffe85e33ba4839969088fed6f27cae1b35f6e0247a7d1b18937abe5f",
            5,
        ),
    }
    for label, (root_hash, dependency_count) in closure_contract.items():
        before = _required_mapping(pre_import, label, f"pre_import.{label}")
        after = _required_mapping(
            post_evaluation, label, f"post_evaluation.{label}"
        )
        if (
            dict(before) != dict(after)
            or before.get("root_sha256") != root_hash
            or before.get("dependency_count") != dependency_count
            or before.get("all_hashes_verified") is not True
        ):
            raise ValueError(f"formal H3 release provenance mismatch: {label}")
    runtime_data_before = _required_mapping(
        provenance,
        "runtime_model_and_data_pre_evaluation",
        "runtime_dependency_provenance.runtime_model_and_data_pre_evaluation",
    )
    runtime_data_after = _required_mapping(
        post_evaluation,
        "runtime_model_and_data_closure",
        "post_evaluation.runtime_model_and_data_closure",
    )
    environment = _required_mapping(
        provenance,
        "runtime_environment",
        "runtime_dependency_provenance.runtime_environment",
    )
    providers = _required_mapping(
        provenance,
        "onnx_session_execution_providers",
        "runtime_dependency_provenance.onnx_session_execution_providers",
    )
    if (
        dict(runtime_data_before) != dict(runtime_data_after)
        or runtime_data_before.get("root_sha256")
        != _H3_RELEASE_RUNTIME_DATA_CLOSURE_SHA256
        or runtime_data_before.get("dependency_count") != 57
        or runtime_data_before.get("all_hashes_verified") is not True
        or environment.get("exact_versions_verified") is not True
        or dict(environment.get("actual", {}))
        != {
            "python": "3.12.3",
            "numpy": "2.5.1",
            "mujoco": "3.11.0",
            "onnxruntime": "1.28.0",
        }
        or environment.get("onnxruntime_build_commit_verified")
        != "45de2a8b06"
        or set(providers)
        != {
            "stand",
            "forward",
            "reverse",
            "lateral_left",
            "lateral_right",
            "yaw_left",
            "yaw_right",
            "compound",
        }
        or any(value != ["CPUExecutionProvider"] for value in providers.values())
    ):
        raise ValueError("formal H3 release runtime provenance mismatch")


def load_and_validate_formal_release_evidence(
    path: Path,
    *,
    expected_base_v22_sha256: str,
    expected_profile_sha256: Mapping[str, str],
    expected_scene_sha256: str,
    expected_reference_sha256: str,
) -> dict[str, Any]:
    """Load one allowlisted 20x30 release record and bind its exact inputs.

    Only the independently audited post-adoption H3 20x30 record is frozen in
    the package-release allowlist.  The selection, safety-only, adoption, and
    superseded H2 records cannot be reused as release evidence.
    """

    resolved = _require_source_file(path, "formal release evidence")
    digest = sha256_file(resolved)
    if digest not in FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST:
        raise ValueError(
            "formal release evidence SHA-256 is not in the frozen adoption "
            "allowlist"
        )
    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("formal release evidence is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("formal release evidence must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("formal release evidence schema must remain version 1")
    if payload.get("evaluator_id") != FORMAL_EVALUATOR_ID:
        raise ValueError("formal release evidence evaluator_id mismatch")
    if payload.get("evaluation_mode") != "RELEASE_QUALIFICATION":
        raise ValueError("formal evidence must be a release qualification run")

    qualification = _required_mapping(
        payload, "release_qualification", "release_qualification"
    )
    if (
        qualification.get("status") != "RELEASE_QUALIFICATION"
        or qualification.get("release_qualification_eligible") is not True
        or qualification.get("scale_matches_frozen_contract") is not True
        or qualification.get("diagnostic_mode_disabled") is not True
        or qualification.get("master_seed_matches_recommendation") is not True
    ):
        raise ValueError("formal release qualification gate did not pass")
    actual_scale = _required_mapping(
        qualification, "actual", "release_qualification.actual"
    )
    for key, expected in FORMAL_RELEASE_SCALE.items():
        if actual_scale.get(key) != expected:
            raise ValueError(
                f"formal release scale requires {key}={expected!r}"
            )
    expected_scale = _required_mapping(
        qualification, "expected", "release_qualification.expected"
    )
    if expected_scale.get("recommended_master_seed") != FORMAL_RELEASE_MASTER_SEED:
        raise ValueError(
            "formal release evidence must recommend master seed 20260808"
        )
    configuration = _required_mapping(payload, "configuration", "configuration")
    if configuration.get("seed") != FORMAL_RELEASE_MASTER_SEED:
        raise ValueError("formal release evidence must execute master seed 20260808")

    if payload.get("simulation_suite_acceptance_passed") is not True:
        raise ValueError("all formal simulation suites must pass")
    if payload.get("simulation_acceptance_passed") is not True:
        raise ValueError("formal simulation acceptance must pass")
    adoption = _required_mapping(payload, "adoption_contract", "adoption_contract")
    if adoption.get("passed") is not True:
        raise ValueError("formal adoption contract must pass")
    command_mapping = _required_mapping(
        payload, "command_mapping_contract", "command_mapping_contract"
    )
    command_status_gate = _required_mapping(
        command_mapping,
        "validation_status_gate",
        "command_mapping_contract.validation_status_gate",
    )
    if command_status_gate.get("passed") is not True:
        raise ValueError("all formal CommandCase validation statuses must pass")

    suites = _required_mapping(payload, "suites", "suites")
    if set(suites) != {"primitives", "compounds", "transitions"}:
        raise ValueError("formal evidence must contain exactly all three suites")
    for suite_name, suite in suites.items():
        if not isinstance(suite, Mapping):
            raise ValueError(f"formal suite {suite_name} must be an object")
        acceptance = _required_mapping(
            suite, "acceptance", f"suites.{suite_name}.acceptance"
        )
        if acceptance.get("passed") is not True:
            raise ValueError(f"formal suite {suite_name} did not pass")

    provenance = _required_mapping(
        payload, "runtime_dependency_provenance", "runtime_dependency_provenance"
    )
    if (
        provenance.get("verified") is not True
        or provenance.get("pre_post_source_and_data_hashes_unchanged") is not True
        or provenance.get("all_onnx_sessions_cpu_only_verified") is not True
    ):
        raise ValueError("formal runtime provenance verification did not pass")

    policy = _required_mapping(payload, "policy_provenance", "policy_provenance")
    if (
        policy.get("mode") != "FORMAL_BASE_V22_ONLY"
        or policy.get("adoption_eligible") is not True
        or policy.get("all_roles_allowlisted") is not True
        or policy.get("diagnostic_unadopted") is not False
    ):
        raise ValueError("formal policy provenance is not adoption eligible")
    policy_roles = _required_mapping(
        policy, "roles", "policy_provenance.roles"
    )
    required_policy_roles = {
        "stand",
        "forward",
        "reverse",
        "lateral_left",
        "lateral_right",
        "yaw_left",
        "yaw_right",
        "compound",
    }
    if set(policy_roles) != required_policy_roles:
        raise ValueError("formal evidence must bind exactly eight policy roles")
    for role, record in policy_roles.items():
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != expected_base_v22_sha256
            or record.get("formal_base_v22_allowlisted") is not True
            or record.get("adopted") is not True
        ):
            raise ValueError(f"formal policy role {role} is not bound to base v22")

    reverse_adoption = _required_mapping(
        payload, "reverse_profile_adoption", "reverse_profile_adoption"
    )
    if (
        reverse_adoption.get("status") != FORMAL_RELEASE_STATUS
        or reverse_adoption.get("passed") is not True
    ):
        raise ValueError("reverse profile bank is not formally adopted")
    reverse_roles = _required_mapping(
        reverse_adoption, "roles", "reverse_profile_adoption.roles"
    )
    if set(reverse_roles) != {"straight", "left", "right"}:
        raise ValueError("formal evidence must bind all three reverse profiles")
    for role, expected_hash in expected_profile_sha256.items():
        record = reverse_roles.get(role)
        if (
            not isinstance(record, Mapping)
            or record.get("profile_sha256") != expected_hash
            or record.get("profile_hash_allowlisted") is not True
            or record.get("evidence_hash_allowlisted") is not True
            or record.get("status_nonblocked") is not True
            or record.get("passed") is not True
        ):
            raise ValueError(f"formal reverse profile {role} binding failed")

    assets = _required_mapping(
        payload, "exact_hardware_safe_assets", "exact_hardware_safe_assets"
    )
    if assets.get("real_hardware_deployment_allowed") is not False:
        raise ValueError("formal assets must remain simulation-only")
    verified_files = _required_mapping(
        assets, "verified_files", "exact_hardware_safe_assets.verified_files"
    )
    for file_id, expected_hash in (
        ("scene", expected_scene_sha256),
        ("reference", expected_reference_sha256),
    ):
        record = verified_files.get(file_id)
        if not isinstance(record, Mapping) or record.get("sha256") != expected_hash:
            raise ValueError(f"formal evidence {file_id} hash binding failed")

    phase = _required_mapping(
        payload,
        "formal_reverse_phase_entry_contract",
        "formal_reverse_phase_entry_contract",
    )
    if (
        phase.get("status") != FORMAL_RELEASE_STATUS
        or phase.get("enabled_by_default") is not True
        or phase.get("diagnostic_only") is not False
        or phase.get("adoption_eligible") is not True
        or phase.get("current_endpoint_requalified") is not True
        or phase.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError("reverse phase-entry behavior is not formally adopted")

    recovery = _required_mapping(
        payload,
        "formal_backward_exit_recovery_contract",
        "formal_backward_exit_recovery_contract",
    )
    if (
        recovery.get("status") != FORMAL_RELEASE_STATUS
        or recovery.get("enabled_by_default") is not True
        or recovery.get("diagnostic_unadopted_only") is not False
        or recovery.get("adoption_eligible") is not True
        or recovery.get("simulation_acceptance_eligible") is not True
        or recovery.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError("backward-exit recovery is not formally adopted")
    runtime_recovery = _required_mapping(
        recovery,
        "runtime_contract",
        "formal_backward_exit_recovery_contract.runtime_contract",
    )
    if (
        runtime_recovery.get("extra_upper_margin_rad")
        != BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        or runtime_recovery.get("upper_target_rad")
        != BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        or runtime_recovery.get("hold_control_ticks")
        != BACKWARD_EXIT_RECOVERY_HOLD_TICKS
        or runtime_recovery.get("hold_seconds")
        != BACKWARD_EXIT_RECOVERY_HOLD_SECONDS
        or runtime_recovery.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError("formal backward-exit recovery runtime values drifted")
    execution_binding = _required_mapping(
        recovery,
        "execution_bundle_binding",
        "formal_backward_exit_recovery_contract.execution_bundle_binding",
    )
    if (
        execution_binding.get("passed") is not True
        or execution_binding.get("profile_sha256s")
        != dict(expected_profile_sha256)
        or execution_binding.get("policy_sha256")
        != expected_base_v22_sha256
        or set(execution_binding.get("policy_roles", ()))
        != required_policy_roles
    ):
        raise ValueError(
            "formal backward-exit recovery execution bundle binding failed"
        )

    hardware = _required_mapping(payload, "hardware_gate", "hardware_gate")
    if (
        hardware.get("status") != "PROHIBITED"
        or hardware.get("hardware_deployment_allowed") is not False
        or hardware.get("simulation_pass_does_not_promote_hardware") is not True
    ):
        raise ValueError("formal evidence must keep hardware PROHIBITED")

    if digest == FORMAL_RELEASE_EVIDENCE_SHA256:
        if resolved.stat().st_size != FORMAL_RELEASE_EVIDENCE_SIZE_BYTES:
            raise ValueError("formal H3 release evidence size mismatch")
        _strict_validate_h3_release_audits(payload)
    elif "formal_adoption_evidence" in payload:
        # A test/future H3-shaped record cannot bypass nested semantics merely
        # because its digest was temporarily added to an allowlist.
        _strict_validate_h3_release_audits(payload)

    return {
        "path": resolved,
        "sha256": digest,
        "payload": dict(payload),
    }


def _contains_rejected_lineage(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in REJECTED_LINEAGES)


def _validate_package_id(package_id: str) -> None:
    if not _PACKAGE_ID_PATTERN.fullmatch(package_id):
        raise ValueError(
            "package_id must use lowercase letters, numbers, '.', '_' or '-'"
        )
    if _contains_rejected_lineage(package_id):
        raise ValueError("package_id must not name a rejected v59/v60 lineage")


def _relative_path(value: str, label: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError(f"{label} must use portable '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must be a package-relative path")
    if path.parts[0] in ("", "."):
        raise ValueError(f"{label} must be normalized")
    return path


def _file_record(package_root: Path, relative_path: PurePosixPath) -> dict[str, Any]:
    local_path = package_root.joinpath(*relative_path.parts)
    return {
        "path": relative_path.as_posix(),
        "sha256": sha256_file(local_path),
        "size_bytes": local_path.stat().st_size,
    }


def _copy_into_package(
    source: Path, package_root: Path, relative_path: PurePosixPath
) -> None:
    destination = package_root.joinpath(*relative_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _scene_dependency_sources(
    scene: Path,
) -> dict[str, tuple[Path, PurePosixPath]]:
    """Resolve the local MJCF include and every referenced mesh.

    The generated scene is only portable when its included robot XML and mesh
    closure travel with it.  Paths are intentionally restricted to descendants
    of the scene directory; external and traversing references are rejected.
    """

    scene_dir = scene.parent.resolve()
    try:
        scene_root = ET.parse(scene).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"scene is not valid XML: {scene}") from exc
    includes = [element.get("file", "") for element in scene_root.iter("include")]
    if len(includes) != 1:
        raise ValueError("scene must contain exactly one robot XML include")
    include_relative = _relative_path(includes[0], "scene include")
    include_source = (scene_dir / Path(*include_relative.parts)).resolve()
    if scene_dir not in include_source.parents:
        raise ValueError("scene include must remain under the scene directory")
    include_source = _require_source_file(include_source, "scene robot include")

    try:
        robot_root = ET.parse(include_source).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"included robot model is not valid XML: {include_source}") from exc
    compiler = robot_root.find("compiler")
    mesh_dir_text = "" if compiler is None else compiler.get("meshdir", "")
    mesh_dir = _relative_path(mesh_dir_text, "robot compiler meshdir")
    mesh_sources: list[tuple[Path, PurePosixPath]] = []
    for mesh in robot_root.iter("mesh"):
        file_text = mesh.get("file", "")
        mesh_relative = _relative_path(file_text, "robot mesh file")
        source = (include_source.parent / Path(*mesh_dir.parts) / Path(*mesh_relative.parts)).resolve()
        if scene_dir not in source.parents:
            raise ValueError("robot mesh must remain under the scene directory")
        source = _require_source_file(source, "robot mesh")
        destination = (
            _LAYOUT["scene"].parent / include_relative.parent / mesh_dir / mesh_relative
        )
        mesh_sources.append((source, destination))
    if not mesh_sources:
        raise ValueError("included robot model declares no mesh assets")

    dependencies: dict[str, tuple[Path, PurePosixPath]] = {
        "scene_robot_model": (
            include_source,
            _LAYOUT["scene"].parent / include_relative,
        )
    }
    for index, (source, destination) in enumerate(
        sorted(mesh_sources, key=lambda item: item[1].as_posix())
    ):
        dependencies[f"scene_mesh_{index:03d}"] = (source, destination)
    return dependencies


def _validate_packaged_scene_closure(
    package_root: Path,
    files: Mapping[str, Any],
    scene_asset: Mapping[str, Any],
) -> None:
    """Verify that the packaged MJCF has no missing include or mesh files."""

    dependency_ids = list(scene_asset.get("dependency_file_ids", ()))
    if not dependency_ids or len(dependency_ids) != len(set(dependency_ids)):
        raise ValueError("scene dependency file list is missing or duplicated")
    if not set(dependency_ids) <= set(files):
        raise ValueError("scene dependency file list references missing files")

    scene_record = files.get(str(scene_asset.get("file_id", "")))
    if not isinstance(scene_record, Mapping):
        raise ValueError("scene file record is missing")
    scene_path = package_root.joinpath(
        *_relative_path(str(scene_record.get("path", "")), "scene path").parts
    ).resolve()
    try:
        scene_root = ET.parse(scene_path).getroot()
    except ET.ParseError as exc:
        raise ValueError("packaged scene is not valid XML") from exc
    includes = [element.get("file", "") for element in scene_root.iter("include")]
    if len(includes) != 1:
        raise ValueError("packaged scene must contain exactly one robot include")
    include_relative = _relative_path(includes[0], "packaged scene include")
    robot_path = scene_path.parent.joinpath(*include_relative.parts).resolve()

    dependency_paths: dict[Path, str] = {}
    for file_id in dependency_ids:
        record = files[file_id]
        relative = _relative_path(str(record.get("path", "")), f"files.{file_id}")
        dependency_paths[package_root.joinpath(*relative.parts).resolve()] = file_id
    if robot_path not in dependency_paths:
        raise ValueError("packaged robot include is outside the declared scene closure")
    try:
        robot_root = ET.parse(robot_path).getroot()
    except ET.ParseError as exc:
        raise ValueError("packaged robot include is not valid XML") from exc
    compiler = robot_root.find("compiler")
    mesh_dir_text = "" if compiler is None else compiler.get("meshdir", "")
    mesh_dir = _relative_path(mesh_dir_text, "packaged robot meshdir")
    expected_paths = {robot_path}
    for mesh in robot_root.iter("mesh"):
        mesh_relative = _relative_path(
            mesh.get("file", ""), "packaged robot mesh file"
        )
        expected_paths.add(
            robot_path.parent.joinpath(*mesh_dir.parts, *mesh_relative.parts).resolve()
        )
    if expected_paths != set(dependency_paths):
        raise ValueError("packaged scene dependency closure is incomplete or extraneous")


def _load_reverse_export_report(
    report_path: Path, reverse_sha256: str
) -> Mapping[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("reverse export report must prohibit hardware deployment")
    onnx = report.get("onnx")
    if not isinstance(onnx, Mapping) or onnx.get("sha256") != reverse_sha256:
        raise ValueError("reverse export report ONNX hash mismatch")
    interface = report.get("interface")
    if not isinstance(interface, Mapping):
        raise ValueError("reverse export report is missing interface metadata")
    if interface.get("input_name") != "obs":
        raise ValueError("reverse ONNX input must be named 'obs'")
    if list(interface.get("input_shape", ())) != [1, 101]:
        raise ValueError("reverse ONNX input shape must be [1, 101]")
    if list(interface.get("output_shape", ())) != [1, 14]:
        raise ValueError("reverse ONNX output shape must be [1, 14]")
    return report


def _routes(reverse_included: bool) -> dict[str, dict[str, Any]]:
    routes = {
        expert: {"model_id": BASE_V22_MODEL_ID, "correction_ids": []}
        for expert in sorted(ALLOWED_EXPERTS)
    }
    routes[REVERSE]["correction_ids"] = ["reverse_profile"]
    routes[REVERSE_TURN_LEFT]["correction_ids"] = [
        "reverse_turn_left_profile"
    ]
    routes[REVERSE_TURN_RIGHT]["correction_ids"] = [
        "reverse_turn_right_profile"
    ]
    routes[YAW_RIGHT]["correction_ids"] = [
        "yaw_right_policy_command_offset"
    ]
    if reverse_included:
        # The learned 1M reverse residual did not beat feedforward-only v1.
        # It may be carried for audit/reproducibility, but scale zero makes it
        # causally inert and base v22 remains the only executed policy.
        routes[REVERSE]["residual_model_id"] = REVERSE_MODEL_ID
        routes[REVERSE]["residual_scale"] = REVERSE_RESIDUAL_SCALE
    return routes


def _build_manifest(
    *,
    package_id: str,
    package_root: Path,
    layout: Mapping[str, PurePosixPath],
    scene_dependency_file_ids: list[str],
    reverse_included: bool,
    reverse_report_included: bool,
    formal_release_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    file_ids = [
        "router",
        "target_safety",
        "contract",
        "formal_release_evidence",
        "scene",
        "reference",
        "reverse_profile",
        "reverse_turn_left_profile",
        "reverse_turn_right_profile",
        "base_v22_onnx",
        *scene_dependency_file_ids,
    ]
    if reverse_included:
        file_ids.append("reverse_onnx")
    if reverse_report_included:
        file_ids.append("reverse_export_report")
    files = {
        file_id: _file_record(package_root, layout[file_id])
        for file_id in file_ids
    }

    models: dict[str, Any] = {
        BASE_V22_MODEL_ID: {
            "release_id": BASE_V22_RELEASE_ID,
            "source_role": "frozen_base_v22",
            "file_id": "base_v22_onnx",
            "interface": {
                "input_name": "obs",
                "observation_count": 101,
                "action_count": 14,
                "head_action_indices_masked_after_inference": [5, 6, 7, 8],
            },
        }
    }
    if reverse_included:
        models[REVERSE_MODEL_ID] = {
            "release_id": "exp004_reverse_optional_disabled",
            "source_role": "optional_reverse_residual_audit_only",
            "source_experiment": EXPERIMENT,
            "file_id": "reverse_onnx",
            "execution_status": "DISABLED",
            "evaluation_status": "REJECTED_NOT_ADOPTED",
            "residual_scale": REVERSE_RESIDUAL_SCALE,
            "interface_verification": (
                "export_report_verified"
                if reverse_report_included
                else "sha256_only_export_report_not_supplied"
            ),
            "interface": {
                "input_name": "obs",
                "observation_count": 101,
                "action_count": 14,
                "head_action_indices_masked_after_inference": [5, 6, 7, 8],
            },
        }
        if reverse_report_included:
            models[REVERSE_MODEL_ID]["export_report_file_id"] = (
                "reverse_export_report"
            )

    routes = _routes(reverse_included)
    reachable = sorted({route["model_id"] for route in routes.values()})
    evidence_payload = _required_mapping(
        formal_release_evidence, "payload", "validated evidence payload"
    )
    evidence_sha256 = str(formal_release_evidence["sha256"])
    reverse_adoption = _required_mapping(
        evidence_payload, "reverse_profile_adoption", "reverse_profile_adoption"
    )
    reverse_adoption_roles = _required_mapping(
        reverse_adoption, "roles", "reverse_profile_adoption.roles"
    )
    release_evidence_summary = {
        "file_id": "formal_release_evidence",
        "sha256": evidence_sha256,
        "evaluator_id": FORMAL_EVALUATOR_ID,
        "master_seed": FORMAL_RELEASE_MASTER_SEED,
        "episodes": 20,
        "seconds_per_episode": 30.0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": MANIFEST_KIND,
        "package_id": package_id,
        "experiment": EXPERIMENT,
        "controller": {
            "type": "SafeGaitRouter",
            "module_file_id": "router",
            "class_name": "SafeGaitRouter",
            "allowed_experts": sorted(ALLOWED_EXPERTS),
            "prohibited_experts": sorted(PROHIBITED_EXPERTS),
            "dynamic_model_lookup": False,
            "unknown_expert_behavior": "REJECT",
            "command_envelope": {
                "units": ["meter/second", "meter/second", "radian/second"],
                "axes": ["vx", "vy", "yaw_rate"],
                "minimum": list(DEFAULT_COMMAND_MIN),
                "maximum": list(DEFAULT_COMMAND_MAX),
                "enforcement": "clip_before_slew_and_routing",
            },
            "atomic_maneuvers": {
                REVERSE_TURN_LEFT: {
                    "command": list(REVERSE_TURN_ENDPOINTS[REVERSE_TURN_LEFT]),
                    "enter_via": "stand",
                    "exit_via": "stand",
                    "profile_interpolation": "PROHIBITED",
                    "action_blending": "PROHIBITED",
                },
                REVERSE_TURN_RIGHT: {
                    "command": list(REVERSE_TURN_ENDPOINTS[REVERSE_TURN_RIGHT]),
                    "enter_via": "stand",
                    "exit_via": "stand",
                    "profile_interpolation": "PROHIBITED",
                    "action_blending": "PROHIBITED",
                },
            },
            "runtime_target_safety": {
                "required": True,
                "module_file_id": "target_safety",
                "class_name": "FinalTargetSafetyGuard",
                "step_method": "step",
                "desired_margin_clamp_function": "apply_final_target_safety",
                "state_initialization": "currently_applied_or_reset_joint_targets",
                "runtime_target_safety_margin_rad": (
                    RUNTIME_TARGET_SAFETY_MARGIN_RAD
                ),
                "max_target_slew_rate_rad_s": RUNTIME_TARGET_SLEW_RATE_RAD_S,
                "control_dt_seconds": CONTROL_FIRST_STARTUP_DT_S,
                "maximum_leg_target_delta_per_tick_rad": (
                    RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD
                ),
                "slew_applications_per_tick": 1,
                "applies_to": "all_leg_joint_targets",
                "limit_source": "safety_contract.safe_joint_limits_rad",
                "stage": "stateful_final_guard_before_actuator_application",
                "desired_target_operation": (
                    "clip(desired, lower + margin, upper - margin)"
                ),
                "final_applied_operation": "clip(applied, lower, upper)",
                "transition_margin_behavior": (
                    "physical-safe reset targets outside the inward margin are "
                    "allowed only while slewing toward a margin-clamped desired target"
                ),
                "processing_order": [
                    "observe_route_and_infer_policy",
                    "compose_desired_joint_targets",
                    "clamp_desired_leg_targets_to_inward_margin",
                    "slew_leg_targets_exactly_once_from_previous_applied_state",
                    "final_leg_target_physical_safe_clamp",
                    "apply_guarded_targets_to_actuators",
                    "physics_step",
                    "post_step_sensor_and_audit",
                ],
                "post_clamp_limit_violations_required": 0,
            },
            "runtime_reset_safety": {
                "required": True,
                "module_file_id": "target_safety",
                "function_name": "apply_reset_qpos_safety",
                "limit_source": "safety_contract.safe_joint_limits_rad",
                "zero_noise_behavior": "preserve_exact_physical_safe_home",
                "positive_noise_behavior": "clip_leg_qpos_to_inward_margin",
                "positive_noise_condition": "joint_noise_scale > 0",
                "perturbed_reset_qpos_margin_rad": (
                    PERTURBED_RESET_QPOS_MARGIN_RAD
                ),
                "head_qpos_after_guard_rad": 0.0,
            },
            "runtime_startup_safety": {
                "required": True,
                "mode": "control_first",
                "module_file_id": "target_safety",
                "class_name": "FinalTargetSafetyGuard",
                "method_name": "control_first_startup",
                "control_dt_seconds": CONTROL_FIRST_STARTUP_DT_S,
                "desired_targets": "first_command_policy_targets",
                "home_only_precharge": "PROHIBITED",
                "guard_steps_before_first_physics": 1,
                "slew_applications_per_tick": 1,
                "maximum_leg_target_delta_per_tick_rad": (
                    RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD
                ),
                "required_order": [
                    "reset_qpos_and_guard_state",
                    "observe_route_and_infer_first_command_policy",
                    "compose_first_desired_joint_targets",
                    (
                        "guard.control_first_startup(first_desired_targets, "
                        "dt=0.02)"
                    ),
                    "apply_guarded_targets_to_actuators",
                    "first_physics_step",
                    "first_post_step_sensor_sample",
                ],
                "physics_steps_before_guarded_control": 0,
            },
            "runtime_backward_exit_recovery": {
                "status": BACKWARD_EXIT_RECOVERY_STATUS,
                "module_file_id": "target_safety",
                "class_name": "BackwardExitRecovery",
                "contract_function": "backward_exit_recovery_contract",
                "enabled_by_default": (
                    BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT
                ),
                "extra_upper_margin_rad": (
                    BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
                ),
                "upper_target_rad": (
                    BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
                ),
                "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
                "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
                "hardware_deployment": "PROHIBITED",
            },
        },
        "models": models,
        "corrections": {
            "yaw_right_policy_command_offset": {
                "kind": "policy_observation_command_offset",
                "applies_to_experts": [YAW_RIGHT],
                "stage": "before_policy_inference",
                "axis": "yaw_rate",
                "operation": "add",
                "value": YAW_RIGHT_POLICY_OFFSET,
                "units": "radian/second",
                "requested_command_unchanged": True,
            },
            "reverse_profile": {
                "kind": "feedforward_periodic_reference_profile",
                "release_id": REVERSE_PROFILE_RELEASE_ID,
                "applies_to_experts": [REVERSE],
                "file_id": "reverse_profile",
                "base_model_id": BASE_V22_MODEL_ID,
                "residual_scale": REVERSE_RESIDUAL_SCALE,
            },
            "reverse_turn_left_profile": {
                "kind": "atomic_periodic_reference_profile",
                "release_id": REVERSE_TURN_LEFT_PROFILE_RELEASE_ID,
                "applies_to_experts": [REVERSE_TURN_LEFT],
                "file_id": "reverse_turn_left_profile",
                "validated_command": list(
                    REVERSE_TURN_ENDPOINTS[REVERSE_TURN_LEFT]
                ),
            },
            "reverse_turn_right_profile": {
                "kind": "atomic_periodic_reference_profile",
                "release_id": REVERSE_TURN_RIGHT_PROFILE_RELEASE_ID,
                "applies_to_experts": [REVERSE_TURN_RIGHT],
                "file_id": "reverse_turn_right_profile",
                "validated_command": list(
                    REVERSE_TURN_ENDPOINTS[REVERSE_TURN_RIGHT]
                ),
            },
        },
        "routes": routes,
        "assets": {
            "scene": {
                "file_id": "scene",
                "sha256": files["scene"]["sha256"],
                "dependency_file_ids": scene_dependency_file_ids,
            },
            "reference": {
                "file_id": "reference",
                "sha256": files["reference"]["sha256"],
            },
            "safety_contract": {
                "file_id": "contract",
                "sha256": files["contract"]["sha256"],
            },
            "formal_release_evidence": {
                **release_evidence_summary,
            },
        },
        "integrity": {"algorithm": "sha256", "files": files},
        "safety": {
            "hardware_deployment": "PROHIBITED",
            "simulation_use_only": True,
            "head_action_indices": [5, 6, 7, 8],
            "head_action_value_after_inference": 0.0,
            "rejected_lineages": list(REJECTED_LINEAGES),
            "reachable_model_ids": reachable,
            "rejected_lineages_reachable": False,
            "reverse_release_decision": {
                "status": reverse_adoption["status"],
                "base_model_id": BASE_V22_MODEL_ID,
                "feedforward_profile_release_id": REVERSE_PROFILE_RELEASE_ID,
                "residual_scale": REVERSE_RESIDUAL_SCALE,
                "validated_command": [-0.050, 0.0, 0.0],
                "formal_evidence": release_evidence_summary,
                "profile_evidence": dict(reverse_adoption_roles["straight"]),
            },
            "reverse_turn_release_decisions": {
                REVERSE_TURN_LEFT: {
                    "status": reverse_adoption["status"],
                    "profile_release_id": REVERSE_TURN_LEFT_PROFILE_RELEASE_ID,
                    "validated_command": list(
                        REVERSE_TURN_ENDPOINTS[REVERSE_TURN_LEFT]
                    ),
                    "formal_evidence": release_evidence_summary,
                    "profile_evidence": dict(reverse_adoption_roles["left"]),
                },
                REVERSE_TURN_RIGHT: {
                    "status": reverse_adoption["status"],
                    "profile_release_id": REVERSE_TURN_RIGHT_PROFILE_RELEASE_ID,
                    "validated_command": list(
                        REVERSE_TURN_ENDPOINTS[REVERSE_TURN_RIGHT]
                    ),
                    "formal_evidence": release_evidence_summary,
                    "profile_evidence": dict(reverse_adoption_roles["right"]),
                },
            },
            "formal_release_gate": {
                "status": FORMAL_RELEASE_STATUS,
                "formal_evidence": release_evidence_summary,
                "phase_entry_status": FORMAL_RELEASE_STATUS,
                "backward_exit_recovery_status": FORMAL_RELEASE_STATUS,
                "hardware_deployment": "PROHIBITED",
            },
            "rejected_candidates": {
                "learned_reverse_1m_res012": {
                    "status": "REJECTED",
                    "mean_vx": -0.0163,
                    "mean_lateral_velocity": 0.0464,
                    "trainer_mean_episode_length": 73.94,
                    "trainer_episode_horizon": 1000,
                },
                "legacy_reverse_turn_left_m005_p020": {
                    "status": "REJECTED",
                    "falls": 1,
                    "episodes": 20,
                },
            },
            "runtime_target_safety_margin_rad": (
                RUNTIME_TARGET_SAFETY_MARGIN_RAD
            ),
            "runtime_target_slew_rate_rad_s": RUNTIME_TARGET_SLEW_RATE_RAD_S,
            "runtime_final_leg_target_clamp_required": True,
            "perturbed_reset_qpos_margin_rad": PERTURBED_RESET_QPOS_MARGIN_RAD,
            "zero_noise_exact_home_required": True,
            "control_first_startup_required": True,
            "control_first_startup_dt_seconds": CONTROL_FIRST_STARTUP_DT_S,
            "maximum_leg_target_delta_per_tick_rad": (
                RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD
            ),
            "slew_applications_per_tick": 1,
            "home_only_precharge": "PROHIBITED",
        },
    }


def validate_package_manifest(
    manifest: Mapping[str, Any],
    package_root: Path,
    *,
    expected_base_v22_sha256: str = BASE_V22_SHA256,
) -> None:
    """Validate graph closure, safety metadata, and every packaged file hash."""

    package_root = package_root.resolve()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported package manifest schema")
    if manifest.get("manifest_kind") != MANIFEST_KIND:
        raise ValueError("unexpected package manifest kind")
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("package must belong to exp_004")
    _validate_package_id(str(manifest.get("package_id", "")))

    safety = manifest.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("package safety metadata is missing")
    if safety.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("hardware deployment must remain PROHIBITED")
    if safety.get("simulation_use_only") is not True:
        raise ValueError("package must be marked simulation-only")
    if safety.get("rejected_lineages_reachable") is not False:
        raise ValueError("rejected lineages must be explicitly unreachable")
    if tuple(safety.get("rejected_lineages", ())) != REJECTED_LINEAGES:
        raise ValueError("v59/v60 rejected lineage list must be frozen")

    controller = manifest.get("controller")
    if not isinstance(controller, Mapping):
        raise ValueError("controller metadata is missing")
    if controller.get("type") != "SafeGaitRouter":
        raise ValueError("controller must be SafeGaitRouter")
    if controller.get("class_name") != "SafeGaitRouter":
        raise ValueError("controller class must be SafeGaitRouter")
    if controller.get("dynamic_model_lookup") is not False:
        raise ValueError("dynamic model lookup is prohibited")
    if controller.get("unknown_expert_behavior") != "REJECT":
        raise ValueError("unknown expert names must be rejected")
    if set(controller.get("allowed_experts", ())) != set(ALLOWED_EXPERTS):
        raise ValueError("controller allowed expert set mismatch")
    if set(controller.get("prohibited_experts", ())) != set(
        PROHIBITED_EXPERTS
    ):
        raise ValueError("controller prohibited expert set mismatch")
    envelope = controller.get("command_envelope")
    if not isinstance(envelope, Mapping):
        raise ValueError("controller command envelope is missing")
    if list(envelope.get("minimum", ())) != list(DEFAULT_COMMAND_MIN):
        raise ValueError("controller vx minimum must remain exactly -0.050")
    if list(envelope.get("maximum", ())) != list(DEFAULT_COMMAND_MAX):
        raise ValueError("controller command maximum mismatch")
    if envelope.get("enforcement") != "clip_before_slew_and_routing":
        raise ValueError("controller command envelope enforcement mismatch")
    atomic = controller.get("atomic_maneuvers")
    if not isinstance(atomic, Mapping) or set(atomic) != {
        REVERSE_TURN_LEFT,
        REVERSE_TURN_RIGHT,
    }:
        raise ValueError("atomic reverse-turn contract is missing")
    for expert in (REVERSE_TURN_LEFT, REVERSE_TURN_RIGHT):
        maneuver = atomic[expert]
        if not isinstance(maneuver, Mapping):
            raise ValueError(f"atomic maneuver {expert} must be a mapping")
        if list(maneuver.get("command", ())) != list(
            REVERSE_TURN_ENDPOINTS[expert]
        ):
            raise ValueError(f"atomic maneuver {expert} command mismatch")
        if maneuver.get("enter_via") != "stand" or maneuver.get(
            "exit_via"
        ) != "stand":
            raise ValueError("reverse-turn maneuvers must transition through stand")
        if maneuver.get("profile_interpolation") != "PROHIBITED" or maneuver.get(
            "action_blending"
        ) != "PROHIBITED":
            raise ValueError("reverse-turn profile interpolation is prohibited")
    target_safety = controller.get("runtime_target_safety")
    if not isinstance(target_safety, Mapping):
        raise ValueError("runtime final target safety contract is missing")
    if (
        target_safety.get("required") is not True
        or target_safety.get("module_file_id") != "target_safety"
        or target_safety.get("class_name") != "FinalTargetSafetyGuard"
        or target_safety.get("step_method") != "step"
        or target_safety.get("desired_margin_clamp_function")
        != "apply_final_target_safety"
        or target_safety.get("state_initialization")
        != "currently_applied_or_reset_joint_targets"
        or float(
            target_safety.get("runtime_target_safety_margin_rad", -1.0)
        )
        != RUNTIME_TARGET_SAFETY_MARGIN_RAD
        or float(target_safety.get("max_target_slew_rate_rad_s", -1.0))
        != RUNTIME_TARGET_SLEW_RATE_RAD_S
        or float(target_safety.get("control_dt_seconds", -1.0))
        != CONTROL_FIRST_STARTUP_DT_S
        or float(
            target_safety.get("maximum_leg_target_delta_per_tick_rad", -1.0)
        )
        != RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD
        or int(target_safety.get("slew_applications_per_tick", -1)) != 1
        or target_safety.get("applies_to") != "all_leg_joint_targets"
        or target_safety.get("limit_source")
        != "safety_contract.safe_joint_limits_rad"
        or target_safety.get("stage")
        != "stateful_final_guard_before_actuator_application"
        or target_safety.get("desired_target_operation")
        != "clip(desired, lower + margin, upper - margin)"
        or target_safety.get("final_applied_operation")
        != "clip(applied, lower, upper)"
        or target_safety.get("transition_margin_behavior")
        != (
            "physical-safe reset targets outside the inward margin are "
            "allowed only while slewing toward a margin-clamped desired target"
        )
        or list(target_safety.get("processing_order", ()))
        != [
            "observe_route_and_infer_policy",
            "compose_desired_joint_targets",
            "clamp_desired_leg_targets_to_inward_margin",
            "slew_leg_targets_exactly_once_from_previous_applied_state",
            "final_leg_target_physical_safe_clamp",
            "apply_guarded_targets_to_actuators",
            "physics_step",
            "post_step_sensor_and_audit",
        ]
        or int(target_safety.get("post_clamp_limit_violations_required", -1))
        != 0
    ):
        raise ValueError(
            "runtime target guard must use frozen 2.0 rad/s slew and 0.050 rad margin"
        )
    if (
        float(safety.get("runtime_target_safety_margin_rad", -1.0))
        != RUNTIME_TARGET_SAFETY_MARGIN_RAD
        or float(safety.get("runtime_target_slew_rate_rad_s", -1.0))
        != RUNTIME_TARGET_SLEW_RATE_RAD_S
        or float(safety.get("maximum_leg_target_delta_per_tick_rad", -1.0))
        != RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD
        or int(safety.get("slew_applications_per_tick", -1)) != 1
        or safety.get("home_only_precharge") != "PROHIBITED"
        or safety.get("runtime_final_leg_target_clamp_required") is not True
    ):
        raise ValueError("safety metadata must require the final leg target clamp")
    reset_safety = controller.get("runtime_reset_safety")
    if not isinstance(reset_safety, Mapping):
        raise ValueError("runtime reset qpos safety contract is missing")
    if (
        reset_safety.get("required") is not True
        or reset_safety.get("module_file_id") != "target_safety"
        or reset_safety.get("function_name") != "apply_reset_qpos_safety"
        or reset_safety.get("limit_source")
        != "safety_contract.safe_joint_limits_rad"
        or reset_safety.get("zero_noise_behavior")
        != "preserve_exact_physical_safe_home"
        or reset_safety.get("positive_noise_behavior")
        != "clip_leg_qpos_to_inward_margin"
        or reset_safety.get("positive_noise_condition") != "joint_noise_scale > 0"
        or float(reset_safety.get("perturbed_reset_qpos_margin_rad", -1.0))
        != PERTURBED_RESET_QPOS_MARGIN_RAD
        or float(reset_safety.get("head_qpos_after_guard_rad", -1.0)) != 0.0
    ):
        raise ValueError(
            "perturbed reset qpos must use 0.005 rad margin and preserve zero-noise home"
        )
    if (
        float(safety.get("perturbed_reset_qpos_margin_rad", -1.0))
        != PERTURBED_RESET_QPOS_MARGIN_RAD
        or safety.get("zero_noise_exact_home_required") is not True
    ):
        raise ValueError("safety metadata must freeze reset qpos behavior")
    startup_safety = controller.get("runtime_startup_safety")
    if not isinstance(startup_safety, Mapping):
        raise ValueError("control-first startup contract is missing")
    if (
        startup_safety.get("required") is not True
        or startup_safety.get("mode") != "control_first"
        or startup_safety.get("module_file_id") != "target_safety"
        or startup_safety.get("class_name") != "FinalTargetSafetyGuard"
        or startup_safety.get("method_name") != "control_first_startup"
        or float(startup_safety.get("control_dt_seconds", -1.0))
        != CONTROL_FIRST_STARTUP_DT_S
        or startup_safety.get("desired_targets")
        != "first_command_policy_targets"
        or startup_safety.get("home_only_precharge") != "PROHIBITED"
        or int(startup_safety.get("guard_steps_before_first_physics", -1)) != 1
        or int(startup_safety.get("slew_applications_per_tick", -1)) != 1
        or float(
            startup_safety.get("maximum_leg_target_delta_per_tick_rad", -1.0)
        )
        != RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD
        or list(startup_safety.get("required_order", ()))
        != [
            "reset_qpos_and_guard_state",
            "observe_route_and_infer_first_command_policy",
            "compose_first_desired_joint_targets",
            "guard.control_first_startup(first_desired_targets, dt=0.02)",
            "apply_guarded_targets_to_actuators",
            "first_physics_step",
            "first_post_step_sensor_sample",
        ]
        or int(startup_safety.get("physics_steps_before_guarded_control", -1))
        != 0
    ):
        raise ValueError("control-first startup must precede physics at dt=0.02")
    if (
        safety.get("control_first_startup_required") is not True
        or float(safety.get("control_first_startup_dt_seconds", -1.0))
        != CONTROL_FIRST_STARTUP_DT_S
        or float(safety.get("maximum_leg_target_delta_per_tick_rad", -1.0))
        != RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD
        or int(safety.get("slew_applications_per_tick", -1)) != 1
        or safety.get("home_only_precharge") != "PROHIBITED"
    ):
        raise ValueError("safety metadata must require control-first startup")
    recovery = controller.get("runtime_backward_exit_recovery")
    if not isinstance(recovery, Mapping):
        raise ValueError("runtime backward-exit recovery contract is missing")
    if (
        recovery.get("status") != FORMAL_RELEASE_STATUS
        or recovery.get("enabled_by_default") is not True
        or recovery.get("module_file_id") != "target_safety"
        or recovery.get("class_name") != "BackwardExitRecovery"
        or recovery.get("contract_function")
        != "backward_exit_recovery_contract"
        or recovery.get("extra_upper_margin_rad")
        != BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        or recovery.get("upper_target_rad")
        != BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        or recovery.get("hold_control_ticks")
        != BACKWARD_EXIT_RECOVERY_HOLD_TICKS
        or recovery.get("hold_seconds") != BACKWARD_EXIT_RECOVERY_HOLD_SECONDS
        or recovery.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError(
            "backward-exit recovery is not a formally adopted runtime default"
        )

    models = manifest.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("models mapping is missing")
    allowed_model_ids = {BASE_V22_MODEL_ID, REVERSE_MODEL_ID}
    model_ids = set(models)
    if BASE_V22_MODEL_ID not in model_ids or not model_ids <= allowed_model_ids:
        raise ValueError("model set must be base_v22 plus optional reverse_exp004")
    base = models[BASE_V22_MODEL_ID]
    if base.get("release_id") != BASE_V22_RELEASE_ID:
        raise ValueError("base model must be the frozen v22 release")
    for model_id, model in models.items():
        if _contains_rejected_lineage(model_id):
            raise ValueError("v59/v60 model IDs are prohibited")
        for key in ("release_id", "source_role", "file_id"):
            if _contains_rejected_lineage(str(model.get(key, ""))):
                raise ValueError("v59/v60 model lineage references are prohibited")
    if REVERSE_MODEL_ID in models:
        reverse = models[REVERSE_MODEL_ID]
        if reverse.get("source_experiment") != EXPERIMENT:
            raise ValueError("optional reverse model must come from exp_004")
        if reverse.get("execution_status") != "DISABLED":
            raise ValueError("optional reverse ONNX must remain disabled")
        if reverse.get("evaluation_status") != "REJECTED_NOT_ADOPTED":
            raise ValueError("optional reverse ONNX must remain non-adopted")
        if float(reverse.get("residual_scale", -1.0)) != REVERSE_RESIDUAL_SCALE:
            raise ValueError("optional reverse residual scale must be zero")

    routes = manifest.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != set(ALLOWED_EXPERTS):
        raise ValueError("routes must cover every SafeGaitRouter expert exactly")
    reachable: set[str] = set()
    for expert, route in routes.items():
        if not isinstance(route, Mapping):
            raise ValueError(f"route {expert} must be a mapping")
        model_id = str(route.get("model_id", ""))
        if model_id not in models:
            raise ValueError(f"route {expert} references undeclared model {model_id}")
        if _contains_rejected_lineage(model_id):
            raise ValueError(f"route {expert} reaches a rejected lineage")
        reachable.add(model_id)
    if routes[REVERSE].get("model_id") != BASE_V22_MODEL_ID:
        raise ValueError("reverse route must execute frozen base v22")
    residual_model_id = routes[REVERSE].get("residual_model_id")
    if REVERSE_MODEL_ID in models:
        if residual_model_id != REVERSE_MODEL_ID:
            raise ValueError("optional reverse residual audit link is missing")
        if float(routes[REVERSE].get("residual_scale", -1.0)) != 0.0:
            raise ValueError("optional reverse residual must be causally inert")
    elif residual_model_id is not None or "residual_scale" in routes[REVERSE]:
        raise ValueError("reverse route names an absent residual model")
    for expert in (REVERSE_TURN_LEFT, REVERSE_TURN_RIGHT):
        if routes[expert].get("model_id") != BASE_V22_MODEL_ID:
            raise ValueError("reverse-turn routes must remain on base v22")

    corrections = manifest.get("corrections")
    if not isinstance(corrections, Mapping):
        raise ValueError("corrections mapping is missing")
    yaw_correction = corrections.get("yaw_right_policy_command_offset")
    if not isinstance(yaw_correction, Mapping):
        raise ValueError("yaw-right policy correction is missing")
    if yaw_correction.get("applies_to_experts") != [YAW_RIGHT]:
        raise ValueError("yaw-right correction must apply only to yaw_right")
    if yaw_correction.get("operation") != "add" or float(
        yaw_correction.get("value", 0.0)
    ) != YAW_RIGHT_POLICY_OFFSET:
        raise ValueError("yaw-right policy offset must be exactly -0.30")
    if yaw_correction.get("requested_command_unchanged") is not True:
        raise ValueError("yaw-right correction must not mutate requested command")
    reverse_correction = corrections.get("reverse_profile")
    if not isinstance(reverse_correction, Mapping):
        raise ValueError("reverse v1 feedforward profile is missing")
    if (
        reverse_correction.get("release_id") != REVERSE_PROFILE_RELEASE_ID
        or reverse_correction.get("base_model_id") != BASE_V22_MODEL_ID
        or float(reverse_correction.get("residual_scale", -1.0)) != 0.0
    ):
        raise ValueError("reverse must use v22 + exact-safe-v1 + residual zero")
    for expert, correction_id, release_id in (
        (
            REVERSE_TURN_LEFT,
            "reverse_turn_left_profile",
            REVERSE_TURN_LEFT_PROFILE_RELEASE_ID,
        ),
        (
            REVERSE_TURN_RIGHT,
            "reverse_turn_right_profile",
            REVERSE_TURN_RIGHT_PROFILE_RELEASE_ID,
        ),
    ):
        correction = corrections.get(correction_id)
        if not isinstance(correction, Mapping):
            raise ValueError(f"{correction_id} is missing")
        if correction.get("release_id") != release_id or list(
            correction.get("validated_command", ())
        ) != list(REVERSE_TURN_ENDPOINTS[expert]):
            raise ValueError(f"{expert} profile/command contract mismatch")
    for expert, route in routes.items():
        correction_ids = list(route.get("correction_ids", ()))
        unknown = set(correction_ids) - set(corrections)
        if unknown:
            raise ValueError(f"route {expert} has unknown corrections: {unknown}")
        carries_yaw_offset = "yaw_right_policy_command_offset" in correction_ids
        if carries_yaw_offset != (expert == YAW_RIGHT):
            raise ValueError("yaw-right policy offset route scope mismatch")

    recorded_reachable = set(safety.get("reachable_model_ids", ()))
    if recorded_reachable != reachable:
        raise ValueError("reachable model closure does not match route graph")
    if recorded_reachable != {BASE_V22_MODEL_ID}:
        raise ValueError("base v22 must be the only executable model")
    formal_gate = safety.get("formal_release_gate")
    if not isinstance(formal_gate, Mapping):
        raise ValueError("formal package release gate is missing")
    formal_evidence_summary = formal_gate.get("formal_evidence")
    if (
        formal_gate.get("status") != FORMAL_RELEASE_STATUS
        or formal_gate.get("phase_entry_status") != FORMAL_RELEASE_STATUS
        or formal_gate.get("backward_exit_recovery_status")
        != FORMAL_RELEASE_STATUS
        or formal_gate.get("hardware_deployment") != "PROHIBITED"
        or not isinstance(formal_evidence_summary, Mapping)
        or formal_evidence_summary.get("file_id")
        != "formal_release_evidence"
        or formal_evidence_summary.get("evaluator_id") != FORMAL_EVALUATOR_ID
        or formal_evidence_summary.get("master_seed")
        != FORMAL_RELEASE_MASTER_SEED
        or formal_evidence_summary.get("episodes") != 20
        or formal_evidence_summary.get("seconds_per_episode") != 30.0
    ):
        raise ValueError("formal package release evidence binding is invalid")
    reverse_release = safety.get("reverse_release_decision")
    if not isinstance(reverse_release, Mapping):
        raise ValueError("reverse release decision is missing")
    if (
        reverse_release.get("status") != FORMAL_RELEASE_STATUS
        or reverse_release.get("feedforward_profile_release_id")
        != REVERSE_PROFILE_RELEASE_ID
        or float(reverse_release.get("residual_scale", -1.0)) != 0.0
        or list(reverse_release.get("validated_command", ()))
        != [-0.050, 0.0, 0.0]
        or reverse_release.get("formal_evidence") != formal_evidence_summary
        or not isinstance(reverse_release.get("profile_evidence"), Mapping)
        or reverse_release["profile_evidence"].get("passed") is not True
    ):
        raise ValueError("reverse release evidence contradicts the adopted result")
    turn_releases = safety.get("reverse_turn_release_decisions")
    if not isinstance(turn_releases, Mapping):
        raise ValueError("reverse-turn release decisions are missing")
    for expert in (REVERSE_TURN_LEFT, REVERSE_TURN_RIGHT):
        release = turn_releases.get(expert)
        if (
            not isinstance(release, Mapping)
            or release.get("status") != "ADOPTED_SIMULATION_ONLY"
            or list(release.get("validated_command", ()))
            != list(REVERSE_TURN_ENDPOINTS[expert])
            or release.get("formal_evidence") != formal_evidence_summary
            or not isinstance(release.get("profile_evidence"), Mapping)
            or release["profile_evidence"].get("passed") is not True
        ):
            raise ValueError(f"{expert} release evidence is not accepted")
    rejected_candidates = safety.get("rejected_candidates")
    if not isinstance(rejected_candidates, Mapping):
        raise ValueError("rejected reverse candidates are missing")
    for candidate in (
        "learned_reverse_1m_res012",
        "legacy_reverse_turn_left_m005_p020",
    ):
        decision = rejected_candidates.get(candidate)
        if not isinstance(decision, Mapping) or decision.get("status") != "REJECTED":
            raise ValueError(f"rejected candidate {candidate} became reachable")

    integrity = manifest.get("integrity")
    if not isinstance(integrity, Mapping) or integrity.get("algorithm") != "sha256":
        raise ValueError("integrity algorithm must be sha256")
    files = integrity.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("integrity file table is missing")
    for file_id, record in files.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"file record {file_id} must be a mapping")
        relative = _relative_path(str(record.get("path", "")), f"files.{file_id}")
        if _contains_rejected_lineage(relative.as_posix()):
            raise ValueError("package paths must not name v59/v60")
        digest = str(record.get("sha256", ""))
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"invalid SHA-256 for {file_id}")
        local_path = package_root.joinpath(*relative.parts)
        if not local_path.is_file():
            raise ValueError(f"packaged file is missing: {relative.as_posix()}")
        resolved_local = local_path.resolve()
        if package_root != resolved_local and package_root not in resolved_local.parents:
            raise ValueError(f"packaged file escapes package root: {file_id}")
        if local_path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"packaged file size mismatch: {file_id}")
        if sha256_file(local_path) != digest:
            raise ValueError(f"packaged file SHA-256 mismatch: {file_id}")

    required_file_ids = {
        "router",
        "target_safety",
        "contract",
        "formal_release_evidence",
        "scene",
        "reference",
        "reverse_profile",
        "reverse_turn_left_profile",
        "reverse_turn_right_profile",
        "base_v22_onnx",
    }
    if not required_file_ids <= set(files):
        raise ValueError("required package files are missing")
    if (REVERSE_MODEL_ID in models) != ("reverse_onnx" in files):
        raise ValueError("optional reverse model/file presence mismatch")
    if files["base_v22_onnx"]["sha256"] != expected_base_v22_sha256:
        raise ValueError("base v22 ONNX does not match the frozen release hash")
    expected_profile_hashes = {
        "reverse_profile": REVERSE_PROFILE_SHA256,
        "reverse_turn_left_profile": REVERSE_TURN_LEFT_PROFILE_SHA256,
        "reverse_turn_right_profile": REVERSE_TURN_RIGHT_PROFILE_SHA256,
    }
    for file_id, expected_hash in expected_profile_hashes.items():
        if files[file_id]["sha256"] != expected_hash:
            raise ValueError(f"{file_id} does not match its adopted frozen release")

    referenced_file_ids = {
        str(controller["module_file_id"]),
        str(target_safety["module_file_id"]),
        str(reset_safety["module_file_id"]),
        str(startup_safety["module_file_id"]),
        str(recovery["module_file_id"]),
        *(str(model["file_id"]) for model in models.values()),
    }
    for model in models.values():
        if isinstance(model, Mapping) and "export_report_file_id" in model:
            referenced_file_ids.add(str(model["export_report_file_id"]))
    for correction in corrections.values():
        if isinstance(correction, Mapping) and "file_id" in correction:
            referenced_file_ids.add(str(correction["file_id"]))
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError("asset metadata is missing")
    for asset_name in (
        "scene",
        "reference",
        "safety_contract",
        "formal_release_evidence",
    ):
        asset = assets.get(asset_name)
        if not isinstance(asset, Mapping):
            raise ValueError(f"asset {asset_name} is missing")
        file_id = str(asset.get("file_id", ""))
        referenced_file_ids.add(file_id)
        if file_id not in files or asset.get("sha256") != files[file_id]["sha256"]:
            raise ValueError(f"asset {asset_name} hash mismatch")
        if asset_name == "scene":
            dependency_ids = list(asset.get("dependency_file_ids", ()))
            referenced_file_ids.update(str(item) for item in dependency_ids)
        if asset_name == "formal_release_evidence":
            if dict(asset) != dict(formal_evidence_summary):
                raise ValueError(
                    "formal evidence asset contradicts the package release gate"
                )
    if not referenced_file_ids <= set(files):
        raise ValueError("manifest references undeclared package files")
    if referenced_file_ids != set(files):
        raise ValueError("integrity table contains unreferenced package files")
    _validate_packaged_scene_closure(package_root, files, assets["scene"])

    evidence_file_id = str(formal_evidence_summary["file_id"])
    if formal_evidence_summary.get("sha256") != files[evidence_file_id]["sha256"]:
        raise ValueError("formal evidence summary SHA-256 mismatch")
    evidence_path = package_root.joinpath(
        *_relative_path(
            files[evidence_file_id]["path"], "formal evidence path"
        ).parts
    )
    validated_evidence = load_and_validate_formal_release_evidence(
        evidence_path,
        expected_base_v22_sha256=files["base_v22_onnx"]["sha256"],
        expected_profile_sha256={
            "straight": files["reverse_profile"]["sha256"],
            "left": files["reverse_turn_left_profile"]["sha256"],
            "right": files["reverse_turn_right_profile"]["sha256"],
        },
        expected_scene_sha256=files["scene"]["sha256"],
        expected_reference_sha256=files["reference"]["sha256"],
    )
    if validated_evidence["sha256"] != formal_evidence_summary["sha256"]:
        raise ValueError("formal evidence loader binding mismatch")

    contract_path = package_root.joinpath(
        *_relative_path(files["contract"]["path"], "contract path").parts
    )
    packaged_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(packaged_contract)
    if packaged_contract["deployment"]["hardware_status"] != "PROHIBITED":
        raise ValueError("packaged safety contract permits hardware deployment")
    contract_target_safety = packaged_contract["target_safety"]
    contract_reset_safety = contract_target_safety["reset_qpos"]
    contract_startup_safety = contract_target_safety["control_first_startup"]
    contract_recovery = contract_target_safety["backward_exit_recovery"]
    if (
        contract_recovery.get("status") != FORMAL_RELEASE_STATUS
        or contract_recovery.get("enabled_by_default") is not True
        or contract_recovery.get("diagnostic_unadopted_only") is not False
        or contract_recovery.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError(
            "packaged backward-exit recovery contract is not formally adopted"
        )
    if (
        float(contract_target_safety["leg_margin_rad"])
        != float(target_safety["runtime_target_safety_margin_rad"])
        or float(contract_target_safety["target_slew_limit_rad_per_s"])
        != float(target_safety["max_target_slew_rate_rad_s"])
        or float(contract_startup_safety["control_dt_seconds"])
        != float(target_safety["control_dt_seconds"])
        or float(contract_startup_safety["maximum_leg_target_delta_per_tick_rad"])
        != float(target_safety["maximum_leg_target_delta_per_tick_rad"])
        or int(contract_startup_safety["slew_applications_per_tick"])
        != int(target_safety["slew_applications_per_tick"])
        or list(contract_startup_safety["normal_tick_order"])
        != list(target_safety["processing_order"])
        or float(contract_target_safety["head_target_rad"]) != 0.0
    ):
        raise ValueError("manifest target guard contradicts the packaged contract")
    if (
        float(contract_reset_safety["noise_margin_rad"])
        != float(reset_safety["perturbed_reset_qpos_margin_rad"])
        or contract_reset_safety["zero_noise_rule"]
        != "preserve exact SAFE_INIT without teleporting to the inward target envelope"
    ):
        raise ValueError("manifest reset guard contradicts the packaged contract")
    if (
        contract_startup_safety.get("required") is not True
        or float(contract_startup_safety["control_dt_seconds"])
        != float(startup_safety["control_dt_seconds"])
        or int(contract_startup_safety["physics_steps_before_guarded_control"])
        != int(startup_safety["physics_steps_before_guarded_control"])
        or int(contract_startup_safety["guard_steps_before_first_physics"])
        != int(startup_safety["guard_steps_before_first_physics"])
        or int(contract_startup_safety["slew_applications_per_tick"])
        != int(startup_safety["slew_applications_per_tick"])
        or float(contract_startup_safety["maximum_leg_target_delta_per_tick_rad"])
        != float(startup_safety["maximum_leg_target_delta_per_tick_rad"])
        or contract_startup_safety["desired_targets"]
        != startup_safety["desired_targets"]
        or contract_startup_safety["home_only_precharge"]
        != startup_safety["home_only_precharge"]
        or list(contract_startup_safety["required_order"])
        != list(startup_safety["required_order"])
    ):
        raise ValueError("manifest control-first startup contradicts the contract")
    for joint_name, bounds in packaged_contract["safe_joint_limits_rad"].items():
        if float(bounds[1]) - float(bounds[0]) <= 2.0 * RUNTIME_TARGET_SAFETY_MARGIN_RAD:
            raise ValueError(
                f"safe range for {joint_name} cannot accommodate target margin"
            )


def build_router_package(
    output_dir: Path,
    *,
    package_id: str,
    base_v22_onnx: Path,
    scene: Path,
    reference: Path,
    reverse_profile: Path,
    reverse_turn_left_profile: Path,
    reverse_turn_right_profile: Path,
    router_source: Path,
    target_safety_source: Path,
    contract_source: Path,
    formal_release_evidence: Path | None = None,
    reverse_onnx: Path | None = None,
    reverse_export_report: Path | None = None,
    expected_base_v22_sha256: str = BASE_V22_SHA256,
) -> Path:
    """Create a new package directory and return its manifest path.

    Existing output directories are never reused or overwritten.  Construction
    happens in a sibling staging directory and is renamed into place only after
    full manifest validation succeeds.
    """

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite package: {output_dir}")
    _validate_package_id(package_id)
    if not _SHA256_PATTERN.fullmatch(expected_base_v22_sha256):
        raise ValueError("expected_base_v22_sha256 must be a lowercase SHA-256")
    if formal_release_evidence is None:
        raise ValueError(
            "package release is fail-closed: an allowlisted formal 20x30 "
            "release evidence file is required"
        )

    sources = {
        "router": _require_source_file(router_source, "router source"),
        "target_safety": _require_source_file(
            target_safety_source, "target safety source"
        ),
        "contract": _require_source_file(contract_source, "contract source"),
        "formal_release_evidence": _require_source_file(
            formal_release_evidence, "formal release evidence"
        ),
        "scene": _require_source_file(scene, "scene"),
        "reference": _require_source_file(reference, "reference"),
        "reverse_profile": _require_source_file(
            reverse_profile, "reverse profile"
        ),
        "reverse_turn_left_profile": _require_source_file(
            reverse_turn_left_profile, "reverse-turn-left profile"
        ),
        "reverse_turn_right_profile": _require_source_file(
            reverse_turn_right_profile, "reverse-turn-right profile"
        ),
        "base_v22_onnx": _require_source_file(base_v22_onnx, "base v22 ONNX"),
    }
    layout = dict(_LAYOUT)
    scene_dependencies = _scene_dependency_sources(sources["scene"])
    for file_id, (source, destination) in scene_dependencies.items():
        sources[file_id] = source
        layout[file_id] = destination
    if sources["base_v22_onnx"].suffix.lower() != ".onnx":
        raise ValueError("base v22 model must be an ONNX file")
    if sha256_file(sources["base_v22_onnx"]) != expected_base_v22_sha256:
        raise ValueError("base v22 ONNX does not match the frozen release hash")

    validated_formal_evidence = load_and_validate_formal_release_evidence(
        sources["formal_release_evidence"],
        expected_base_v22_sha256=expected_base_v22_sha256,
        expected_profile_sha256={
            "straight": sha256_file(sources["reverse_profile"]),
            "left": sha256_file(sources["reverse_turn_left_profile"]),
            "right": sha256_file(sources["reverse_turn_right_profile"]),
        },
        expected_scene_sha256=sha256_file(sources["scene"]),
        expected_reference_sha256=sha256_file(sources["reference"]),
    )
    release_blockers = []
    if REVERSE_PHASE_ENTRY_RELEASE_STATUS != FORMAL_RELEASE_STATUS:
        release_blockers.append(
            "reverse phase-entry remains " + REVERSE_PHASE_ENTRY_RELEASE_STATUS
        )
    if BACKWARD_EXIT_RECOVERY_STATUS != FORMAL_RELEASE_STATUS:
        release_blockers.append(
            "backward-exit recovery remains " + BACKWARD_EXIT_RECOVERY_STATUS
        )
    if BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT is not True:
        release_blockers.append("backward-exit recovery is disabled by default")
    if release_blockers:
        raise ValueError(
            "package release is fail-closed: " + "; ".join(release_blockers)
        )

    reverse_included = reverse_onnx is not None
    report_included = False
    if reverse_onnx is not None:
        reverse_path = _require_source_file(reverse_onnx, "reverse ONNX")
        if reverse_path.suffix.lower() != ".onnx":
            raise ValueError("optional reverse model must be an ONNX file")
        if _contains_rejected_lineage(reverse_path.as_posix()):
            raise ValueError("v59/v60 artifacts cannot be packaged as reverse")
        sources["reverse_onnx"] = reverse_path
        reverse_digest = sha256_file(reverse_path)
        report_path = reverse_export_report
        if report_path is None:
            candidate = reverse_path.with_suffix(reverse_path.suffix + ".json")
            if candidate.is_file():
                report_path = candidate
        if report_path is not None:
            resolved_report = _require_source_file(
                report_path, "reverse export report"
            )
            if _contains_rejected_lineage(resolved_report.as_posix()):
                raise ValueError("v59/v60 export reports cannot be packaged")
            _load_reverse_export_report(resolved_report, reverse_digest)
            sources["reverse_export_report"] = resolved_report
            report_included = True
    elif reverse_export_report is not None:
        raise ValueError("reverse_export_report requires reverse_onnx")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    ).resolve()
    try:
        for file_id, source in sources.items():
            _copy_into_package(source, staging, layout[file_id])
        manifest = _build_manifest(
            package_id=package_id,
            package_root=staging,
            layout=layout,
            scene_dependency_file_ids=list(scene_dependencies),
            reverse_included=reverse_included,
            reverse_report_included=report_included,
            formal_release_evidence=validated_formal_evidence,
        )
        validate_package_manifest(
            manifest,
            staging,
            expected_base_v22_sha256=expected_base_v22_sha256,
        )
        manifest_path = staging / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_package_manifest(
            decoded,
            staging,
            expected_base_v22_sha256=expected_base_v22_sha256,
        )
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir / MANIFEST_FILENAME


def load_and_validate_package(
    package_root: Path,
    *,
    expected_base_v22_sha256: str = BASE_V22_SHA256,
) -> dict[str, Any]:
    """Load ``package_manifest.json`` and validate the complete package."""

    package_root = package_root.resolve()
    manifest_path = package_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_package_manifest(
        manifest,
        package_root,
        expected_base_v22_sha256=expected_base_v22_sha256,
    )
    return manifest


__all__ = [
    "BASE_V22_MODEL_ID",
    "BASE_V22_RELEASE_ID",
    "BASE_V22_SHA256",
    "CONTROL_FIRST_STARTUP_DT_S",
    "EXPERIMENT",
    "FORMAL_EVALUATOR_ID",
    "FORMAL_RELEASE_EVIDENCE_SHA256",
    "FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST",
    "FORMAL_RELEASE_EVIDENCE_SIZE_BYTES",
    "FORMAL_RELEASE_MASTER_SEED",
    "FORMAL_RELEASE_SCALE",
    "FORMAL_RELEASE_STATUS",
    "MANIFEST_FILENAME",
    "PERTURBED_RESET_QPOS_MARGIN_RAD",
    "REJECTED_LINEAGES",
    "RUNTIME_TARGET_SAFETY_MARGIN_RAD",
    "RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD",
    "RUNTIME_TARGET_SLEW_RATE_RAD_S",
    "REVERSE_MODEL_ID",
    "YAW_RIGHT_POLICY_OFFSET",
    "build_router_package",
    "load_and_validate_package",
    "load_and_validate_formal_release_evidence",
    "sha256_file",
    "validate_package_manifest",
]
