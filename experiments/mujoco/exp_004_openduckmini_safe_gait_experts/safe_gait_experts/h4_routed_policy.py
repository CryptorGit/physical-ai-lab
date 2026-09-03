"""H4 actor bridge for the exp_004 routed simulation evaluator.

The legacy routed evaluator consumes the frozen 101-wide v22 observation.  H4
actors consume that exact prefix plus the 15 physical observables used by the
training transplant.  This module keeps the two contracts explicit: legacy
roles receive the prefix, while H4 roles receive all 116 values.

This is simulation-only infrastructure.  It has no hardware or release path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import SAFE_INIT_POS
from .h4_post_training import (
    H4_ACTOR_OBSERVATION_WIDTH,
    H4_HEAD_ACTION_SLICE,
    H4_STRICT_COMMANDS,
    H4_ACTION_WIDTH,
    canonical_json_sha256,
    h4_params_numeric_sha256,
    infer_h4_action_numpy,
    mask_h4_head_action,
    sha256_file,
    sha256_tree,
    validate_h4_params,
)
from .h4_training_alignment import (
    LEG_ACTION_INDICES,
    reverse_iteration_v6_absolute_full_leg_targets,
)
from .routed_evaluation import (
    blend_and_mask_actions,
    canonical_policy_role,
)


H4_EXTRA_FEATURE_COUNT = 15
H4_REVERSE_SOURCE_PHASE_ENTRY = 7.0
H4_REVERSE_SOURCE_PHASE_DELTA = 0.81
H4_REVERSE_TEACHER_ENTRY_PHASE_BINS = 14.0
H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS = 1.62
H4_REVERSE_TEACHER_TABLE_ROWS = 54
H4_REVERSE_SOURCE_PERIOD_BINS = 27


@dataclass(frozen=True)
class H4CandidateSpec:
    role: str
    params_path: Path
    params_sha256: str
    manifest_path: Path
    manifest_sha256: str
    trusted_run_root: Path
    authorization_path: Path
    authorization_sha256: str


@dataclass(frozen=True)
class ValidatedH4Candidate:
    spec: H4CandidateSpec
    params: Any
    manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    result: Mapping[str, Any]
    source_hashes: Mapping[str, Mapping[str, str]]
    params_numeric_sha256: str
    validation: Mapping[str, Any]


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _verify_record(
    record: Mapping[str, Any], *, label: str, expected_path: Path | None = None
) -> tuple[Path, str]:
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} record must contain exactly path and sha256")
    path = Path(str(record["path"])).resolve()
    digest = _require_sha256(record["sha256"], f"{label}.sha256")
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{label} path binding drifted")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label} file binding drifted: {path}")
    return path, digest


def _verify_source_closure(
    pre: Mapping[str, Any], post: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    if not pre or dict(pre) != dict(post):
        raise ValueError("H4 source/teacher pre/post closure drifted")
    normalized: dict[str, dict[str, str]] = {}
    for label, raw in post.items():
        if not isinstance(label, str) or not isinstance(raw, Mapping):
            raise ValueError("H4 source closure record is malformed")
        if set(raw) != {"path", "sha256"}:
            raise ValueError(f"H4 source closure field drifted: {label}")
        path = Path(str(raw["path"])).resolve()
        digest = _require_sha256(raw["sha256"], f"source {label}.sha256")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"H4 source closure file drifted: {label} {path}")
        normalized[label] = {"path": str(path), "sha256": digest}
    if dict(post) != normalized:
        raise ValueError("H4 source closure paths are not resolved absolute paths")
    return normalized


def validate_h4_candidate(spec: H4CandidateSpec) -> ValidatedH4Candidate:
    """Validate a completed H4 run before opening its params pickle.

    The reverse-v6 runner predates one manifest mirror field.  The actual
    config/result values remain required at zero; the missing manifest mirror
    is recorded as an explicit compatibility derivation instead of changing
    the historical JSON or its hash.
    """

    role = canonical_policy_role(spec.role)
    if role not in {"forward", "reverse"}:
        raise ValueError("H4 candidates may only replace forward or reverse")
    params_path = spec.params_path.resolve()
    manifest_path = spec.manifest_path.resolve()
    if params_path.name != "final_params.pkl" or manifest_path.name != "run_manifest.json":
        raise ValueError("H4 candidate basenames are not exact")
    params_sha = _require_sha256(spec.params_sha256, "params SHA256")
    manifest_sha = _require_sha256(spec.manifest_sha256, "manifest SHA256")
    if sha256_file(params_path) != params_sha:
        raise ValueError("H4 candidate params hash mismatch")
    if sha256_file(manifest_path) != manifest_sha:
        raise ValueError("H4 candidate manifest hash mismatch")

    manifest = _load_json(manifest_path)
    if manifest.get("status") != "COMPLETED":
        raise ValueError("H4 routed integration requires COMPLETED training")
    if manifest.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("H4 candidate hardware deployment is not prohibited")
    if manifest.get("expert") != role or manifest.get("activity") != "PPO_PILOT_TRAINING":
        raise ValueError("H4 candidate identity drifted")
    if manifest.get("source_and_teacher_unchanged") is not True:
        raise ValueError("H4 candidate source/teacher closure is not immutable")
    if manifest.get("run_name") != manifest_path.parent.name:
        raise ValueError("H4 candidate run_name is not bound to its directory")
    if spec.trusted_run_root.resolve() not in manifest_path.parents:
        raise ValueError("H4 candidate is outside its trusted run root")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("H4 candidate outputs are missing")
    _verify_record(outputs["final_params"], label="outputs.final_params", expected_path=params_path)
    result_path, _ = _verify_record(outputs["result"], label="outputs.result")
    _verify_record(outputs["training_curve"], label="outputs.training_curve")
    resolved_config_record = manifest["resolved_config"]
    if not isinstance(resolved_config_record, Mapping) or set(
        resolved_config_record
    ) != {"path", "sha256", "canonical_sha256"}:
        raise ValueError("resolved_config record field set drifted")
    config_path, config_sha = _verify_record(
        {
            "path": resolved_config_record["path"],
            "sha256": resolved_config_record["sha256"],
        },
        label="resolved_config",
    )
    config = _load_json(config_path)
    result = _load_json(result_path)
    if resolved_config_record.get("canonical_sha256") != canonical_json_sha256(config):
        raise ValueError("H4 resolved config canonical hash drifted")

    source_hashes = _verify_source_closure(
        manifest["source_and_teacher_hashes_pre"],
        manifest["source_and_teacher_hashes_post"],
    )
    parent = manifest.get("parent_checkpoint")
    if not isinstance(parent, Mapping) or parent.get("unchanged") is not True:
        raise ValueError("H4 parent checkpoint is not read-only")
    parent_path = Path(str(parent.get("path", ""))).resolve()
    parent_pre = _require_sha256(parent.get("sha256_tree_pre"), "parent tree pre")
    parent_post = _require_sha256(parent.get("sha256_tree_post"), "parent tree post")
    if parent_pre != parent_post or not parent_path.is_dir() or sha256_tree(parent_path) != parent_pre:
        raise ValueError("H4 parent checkpoint tree drifted")

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
    active_flag = (
        "forward_iteration_v6_contact_abort_island_only"
        if role == "forward"
        else "reverse_iteration_v6_absolute_full_leg_targets"
    )
    for artifact_name, artifact in (("config", config), ("manifest", manifest), ("result", result)):
        if any(not isinstance(artifact.get(flag), bool) for flag in flags):
            raise ValueError(f"{artifact_name} H4 iteration flags are malformed")
        if artifact.get(active_flag) is not True or sum(artifact.get(flag) is True for flag in flags) != 1:
            raise ValueError(f"{artifact_name} H4 iteration flag selection drifted")
    expected_contract = (
        "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_250K_FROM_V22"
        if role == "forward"
        else "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_250K_FROM_V22"
    )
    # The reverse training runner's contract spelling is authoritative in the
    # generated files; keep this explicit and fail closed on any other value.
    expected_contract = (
        "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_250K_FROM_V22"
        if role == "forward"
        else "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_250K_FROM_V22"
    )
    if any(artifact.get("training_contract_id") != expected_contract for artifact in (config, manifest, result)):
        raise ValueError("H4 training contract identity drifted")
    if any(artifact.get("actor_observation_width") != H4_ACTOR_OBSERVATION_WIDTH for artifact in (config, manifest, result) if "actor_observation_width" in artifact):
        raise ValueError("H4 actor observation width drifted")
    if config.get("observation_mode") != "h4_116_transplant":
        raise ValueError("H4 observation mode drifted")
    if config.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("H4 config hardware deployment is not prohibited")
    if role == "reverse" and any(
        artifact.get("backward_residual_scale", 0.0) != 0.0
        for artifact in (config, manifest, result)
    ):
        raise ValueError("reverse H4 residual authority is nonzero")

    auth_key = f"{active_flag}_authorization"
    auth_record = manifest.get(auth_key)
    if not isinstance(auth_record, Mapping) or auth_record.get("sha256") != spec.authorization_sha256:
        raise ValueError("H4 iteration authorization binding drifted")
    auth_path = spec.authorization_path.resolve()
    if not auth_path.is_file() or sha256_file(auth_path) != spec.authorization_sha256:
        raise ValueError("H4 iteration authorization bytes drifted")

    # All file and source bindings are closed before unpickling.
    with params_path.open("rb") as stream:
        params = pickle.load(stream)
    params_audit = validate_h4_params(params)
    params_numeric_sha = h4_params_numeric_sha256(params)
    if role == "reverse":
        timing = config.get("teacher_timing_contract")
        if not isinstance(timing, Mapping) or timing.get("teacher_table_rows") != H4_REVERSE_TEACHER_TABLE_ROWS or timing.get("source_period_bins") != H4_REVERSE_SOURCE_PERIOD_BINS or timing.get("entry_phase_bins") != H4_REVERSE_TEACHER_ENTRY_PHASE_BINS or timing.get("phase_advance_bins_per_control") != H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS:
            raise ValueError("reverse H4 teacher timing contract drifted")

    validation = {
        "passed": True,
        "params_sha256": params_sha,
        "manifest_sha256": manifest_sha,
        "resolved_config_sha256": config_sha,
        "params_numeric_sha256": params_numeric_sha,
        "params_structure": params_audit,
        "source_closure_entry_count": len(source_hashes),
        "parent_checkpoint_tree_verified": True,
        "pickle_opened_only_after_file_and_source_closure": True,
        "reverse_manifest_backward_residual_derivation": (
            "CONFIG_RESULT_ZERO_USED_FOR_MISSING_MANIFEST_MIRROR"
            if role == "reverse" and "backward_residual_scale" not in manifest
            else "NONE"
        ),
    }
    return ValidatedH4Candidate(
        spec=spec,
        params=params,
        manifest=manifest,
        config=config,
        result=result,
        source_hashes=source_hashes,
        params_numeric_sha256=params_numeric_sha,
        validation=validation,
    )


class H4RoutedPolicyBank:
    """Route H4 forward/reverse actors beside the frozen 101-wide bank."""

    def __init__(self, base_bank: Any, candidates: Mapping[str, ValidatedH4Candidate]):
        self.base_bank = base_bank
        self.candidates = {
            canonical_policy_role(role): candidate for role, candidate in candidates.items()
        }
        if set(self.candidates) != {"forward", "reverse"}:
            raise ValueError("H4 routed bank requires exactly forward and reverse")
        self.h4_inference_counts: Counter[str] = Counter()
        self.active_decision: Any | None = None

    @property
    def session_providers(self) -> Mapping[str, list[str]]:
        return self.base_bank.session_providers

    @property
    def paths(self) -> Mapping[str, Path]:
        return self.base_bank.paths

    @property
    def inference_counts(self) -> Counter[str]:
        return self.base_bank.inference_counts

    def infer(self, role: str, observation: np.ndarray) -> np.ndarray:
        canonical = canonical_policy_role(role)
        values = np.asarray(observation, dtype=np.float32)
        if canonical in self.candidates:
            if values.shape != (H4_ACTOR_OBSERVATION_WIDTH,):
                raise ValueError("H4 route requires observation width 116")
            action = infer_h4_action_numpy(self.candidates[canonical].params, values)
            action = mask_h4_head_action(action)
            self.h4_inference_counts[canonical] += 1
            return np.asarray(action, dtype=np.float64)
        if values.shape != (H4_ACTOR_OBSERVATION_WIDTH,):
            raise ValueError("routed observation width must be 116 in H4 mode")
        return self.base_bank.infer(canonical, values[:101])

    def infer_route(self, decision: Any, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.active_decision = decision
        from_action = self.infer(decision.blend_from_expert, observation)
        to_action = (
            from_action
            if decision.blend_to_expert == decision.blend_from_expert
            else self.infer(decision.blend_to_expert, observation)
        )
        return blend_and_mask_actions(from_action, to_action, decision.blend_alpha)

    def manifest(self) -> dict[str, Any]:
        return {
            "legacy_onnx": self.base_bank.manifest(),
            "h4_candidates": {
                role: {
                    "params_path": str(candidate.spec.params_path.resolve()),
                    "params_sha256": candidate.spec.params_sha256,
                    "manifest_path": str(candidate.spec.manifest_path.resolve()),
                    "manifest_sha256": candidate.spec.manifest_sha256,
                    "params_numeric_sha256": candidate.params_numeric_sha256,
                    "validation": dict(candidate.validation),
                }
                for role, candidate in sorted(self.candidates.items())
            },
            "execution": {
                "h4_actor_observation_width": H4_ACTOR_OBSERVATION_WIDTH,
                "legacy_prefix_width": 101,
                "physical_extra_width": H4_EXTRA_FEATURE_COUNT,
                "cpu_onnx_roles_only": True,
                "hardware_deployment": "PROHIBITED",
            },
        }

    @staticmethod
    def append_physical_observables(
        *,
        legacy_observation: Sequence[float],
        physical_command: Sequence[float],
        local_linvel: Sequence[float],
        trunk_rotation: Sequence[float],
        normalized_force_and_tangential_speed: tuple[Sequence[float], Sequence[float]],
        feet_contacts: Sequence[float],
    ) -> np.ndarray:
        legacy = np.asarray(legacy_observation, dtype=np.float32)
        command = np.asarray(physical_command, dtype=np.float32)
        velocity = np.asarray(local_linvel, dtype=np.float32)
        rotation = np.asarray(trunk_rotation, dtype=np.float32).reshape(3, 3)
        normal_force, tangential = (
            np.asarray(value, dtype=np.float32)
            for value in normalized_force_and_tangential_speed
        )
        contacts = np.asarray(feet_contacts, dtype=np.float32)
        if legacy.shape != (101,) or command.shape != (3,) or velocity.shape != (3,) or normal_force.shape != (2,) or tangential.shape != (2,) or contacts.shape != (2,):
            raise ValueError("H4 observation feature shapes drifted")
        gravity = rotation.T @ np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        extra = np.concatenate(
            (command, velocity, gravity, normal_force, contacts, tangential)
        ).astype(np.float32)
        if extra.shape != (H4_EXTRA_FEATURE_COUNT,) or not np.all(np.isfinite(extra)):
            raise ValueError("H4 physical observables must be finite with width 15")
        result = np.concatenate((legacy, extra)).astype(np.float32)
        if result.shape != (H4_ACTOR_OBSERVATION_WIDTH,):
            raise RuntimeError("H4 routed observation width drifted")
        return result

    @staticmethod
    def uses_absolute_reverse_decoder(role: str, effective_command: Sequence[float]) -> bool:
        command = np.asarray(effective_command, dtype=np.float64)
        return canonical_policy_role(role) == "reverse" and command.shape == (3,) and float(command[0]) < -0.02

    @staticmethod
    def absolute_reverse_targets(
        action: Sequence[float], *, default: Sequence[float], joint_ranges: np.ndarray
    ) -> np.ndarray:
        """Decode the reverse-v6 action against the training SAFE contract.

        ``default`` and ``joint_ranges`` remain explicit inputs so the routed
        seam can validate the loaded scene, but they must not define the
        decoder domain.  Training uses the canonical SAFE_INIT/SAFE_JOINT_LIMITS
        vectors from ``h4_training_alignment``; using MuJoCo's raw joint range
        here silently changes the nonlinear absolute-target map.
        """
        values = np.asarray(action, dtype=np.float32)
        lower = np.asarray(joint_ranges[:, 0], dtype=np.float32)
        upper = np.asarray(joint_ranges[:, 1], dtype=np.float32)
        initial = np.asarray(default, dtype=np.float32)
        if values.shape != (H4_ACTION_WIDTH,) or lower.shape != (H4_ACTION_WIDTH,) or upper.shape != (H4_ACTION_WIDTH,) or initial.shape != (H4_ACTION_WIDTH,):
            raise ValueError("reverse v6 decoder vector shape drifted")
        targets = reverse_iteration_v6_absolute_full_leg_targets(
            values,
            xp=np,
        )
        targets = np.asarray(targets, dtype=np.float64)
        if not np.all(np.isfinite(targets)) or not np.array_equal(targets[5:9], np.zeros(4)):
            raise ValueError("reverse v6 decoder produced invalid head targets")
        return targets

    @staticmethod
    def handles_phase(role: str) -> bool:
        return canonical_policy_role(role) == "reverse"

    @staticmethod
    def phase_delta(role: str, effective_command: Sequence[float]) -> float:
        if not H4RoutedPolicyBank.handles_phase(role):
            raise ValueError("H4 phase requested for a non-reverse role")
        command = np.asarray(effective_command, dtype=np.float64)
        if command.shape != (3,) or float(command[0]) >= -0.02:
            raise ValueError("H4 reverse phase requires an active reverse command")
        return H4_REVERSE_SOURCE_PHASE_DELTA

    @staticmethod
    def advance_phase(
        *,
        phase_index: float,
        phase_steps: float,
        phase_delta: float,
        current_expert: str,
        previous_expert: str | None,
        effective_command: Sequence[float],
        previous_backward_feedforward_active: bool,
        control_step: int,
        global_control_tick: int,
    ) -> tuple[float, bool, dict[str, Any] | None]:
        current = float(phase_index)
        count = float(phase_steps)
        command = np.asarray(effective_command, dtype=np.float64)
        active = bool(command[0] < -0.02)
        entering = active and not bool(previous_backward_feedforward_active)
        event = None
        if entering:
            current = H4_REVERSE_SOURCE_PHASE_ENTRY
            event = {
                "control_step": int(control_step),
                "global_control_tick": int(global_control_tick),
                "previous_expert": previous_expert,
                "current_expert": current_expert,
                "effective_command": command.tolist(),
                "activation_predicate": "effective_vx_lt_negative_0p02_false_to_true",
                "phase_semantics": "H4_SOURCE_PERIOD_PHASE_USED_FOR_LEGACY_COS_SIN",
                "global_phase_index_before_reset": float(phase_index),
                "reset_preincrement_phase_index": current,
                # Retain the frozen audit field: it denotes the next phase
                # after the entry observation, not the phase used by the
                # first actor call.
                "first_feedforward_phase_index": (
                    current + float(phase_delta)
                ) % count,
                "first_observation_phase_index": current,
                "next_observation_phase_index": (
                    current + float(phase_delta)
                ) % count,
                "teacher_entry_phase_bins": H4_REVERSE_TEACHER_ENTRY_PHASE_BINS,
                "teacher_phase_scale": H4_REVERSE_TEACHER_TABLE_ROWS / H4_REVERSE_SOURCE_PERIOD_BINS,
                "profile_phase_rate": float(phase_delta),
                "phase_steps": count,
                "previous_backward_feedforward_active": False,
                "current_backward_feedforward_active": True,
                "formal_candidate": False,
                "adopted_simulation_only": False,
                "diagnostic_only": True,
                "status": "H4_SIMULATION_DIAGNOSTIC",
            }
            # The training reset exposes the entry phase on the first actor
            # observation.  Advance only after that control tick so the first
            # policy input is phase 7.0, not phase 7.81.
            return current, active, event
        return (current + float(phase_delta)) % count, active, event


__all__ = [
    "H4CandidateSpec",
    "H4_EXTRA_FEATURE_COUNT",
    "H4RoutedPolicyBank",
    "H4_REVERSE_SOURCE_PHASE_DELTA",
    "H4_REVERSE_SOURCE_PHASE_ENTRY",
    "ValidatedH4Candidate",
    "validate_h4_candidate",
]
