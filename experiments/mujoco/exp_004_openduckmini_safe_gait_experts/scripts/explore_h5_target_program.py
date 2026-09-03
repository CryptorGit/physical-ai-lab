"""Diagnostic H5 target-program probes under the exact routed simulator.

This is an exploration harness only.  It never writes adoption evidence and
always labels its output as hardware-prohibited.  The purpose is to separate
target-generation quality from actor-learning quality while keeping the H5
target-space decoder and the frozen final guard in the loop.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
PLAYGROUND_ROOT = EXP_ROOT.parents[2] / ".openduck_playground_source_review"
for root in (EXP_ROOT, SCRIPT_ROOT, PLAYGROUND_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from safe_gait_experts.contract import (  # noqa: E402
    ACTUATOR_JOINT_ORDER,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
)
from safe_gait_experts.h5_target_contract import (  # noqa: E402
    H5_ACTION_WIDTH,
    h5_decode_absolute_targets,
)
from safe_gait_experts.h5_command_contract import (  # noqa: E402
    h5_unified_policy_command,
)
from safe_gait_experts.routed_evaluation import (  # noqa: E402
    AcceptanceThresholds,
    segment_acceptance,
)
from safe_gait_experts.h4_training_alignment import (  # noqa: E402
    OBSERVATION_POLICY_COMMAND_SLICE,
    margin_clip_targets,
)
from playground.common.poly_reference_motion_numpy import (  # noqa: E402
    PolyReferenceMotion,
)
from scripts import evaluate_h5_routed_transitions as h5  # noqa: E402
from scripts import evaluate_routed_transitions as routed  # noqa: E402


LEG_INDICES = tuple(
    index for index, name in enumerate(ACTUATOR_JOINT_ORDER) if name not in {
        "neck_pitch", "head_pitch", "head_yaw", "head_roll"
    }
)
HEAD_INDICES = (5, 6, 7, 8)
DEFAULT_H5_PARAMS = {
    "planar": EXP_ROOT / "artifacts" / "h5_training_runs_diagnostic_20260811" / "planar" / "h5_planar_250k_diag_multiroute_v5_targetspace" / "final_params.pkl",
    "reverse": EXP_ROOT / "artifacts" / "h5_training_runs_diagnostic_20260811" / "reverse" / "reverse" / "h5_reverse_250k_diag_multiroute_v5_targetspace_teacherprior_phase081_imitation20" / "final_params.pkl",
}
DEFAULT_H5_MANIFESTS = {
    "planar": DEFAULT_H5_PARAMS["planar"].parent / "run_manifest.json",
    "reverse": DEFAULT_H5_PARAMS["reverse"].parent / "run_manifest.json",
}
DEFAULT_POLYNOMIAL_REFERENCE = (
    EXP_ROOT.parents[2]
    / ".openduck_runtime_source_review"
    / "polynomial_coefficients_calibrated.pkl"
)
RECORDED_H3_TRACE_SHA256 = (
    "32fa783e4c589b0beb85afe3ba0a8b738ad3cef7025c88890ec40e7dc0bdb2f2"
)
_POLYNOMIAL_REFERENCE: PolyReferenceMotion | None = None


def _sha(path: Path) -> str:
    return h5.sha256_file(path)


def _policy_args() -> list[str]:
    base = (
        EXP_ROOT
        / "artifacts"
        / "router_packages"
        / "exp004-safe-gait-router-h3-release-20260808-v1"
        / "models"
        / "base_v22.onnx"
    )
    return [f"{role}={base}" for role in routed.REQUIRED_POLICY_ROLES]


def _args(
    params: dict[str, Path],
    manifests: dict[str, Path],
    *,
    strict_actor_only: bool = False,
) -> argparse.Namespace:
    values: dict[str, Any] = {
        "policy": _policy_args(),
        "generated_root": EXP_ROOT / "artifacts" / "generated_playground",
        "seed": 20260808,
        "episodes": 1,
        "seconds": 6.0,
        "transition_seconds": 6.0,
        "transition_stand_seconds": 2.0,
        "warmup_seconds": 1.5,
        "initial_joint_noise_scale": 0.0,
        "initial_base_speed": 0.0,
        # Target-program probes feed the same unified actor/deployment
        # contract as the strict evaluator.  Keeping the historical
        # domain-specific mapper here would make a reverse teacher appear
        # valid in exploration while using a different policy observation
        # during distillation and release evaluation.
        "unified_single_weight": True,
        "strict_actor_only": bool(strict_actor_only),
        "require_pass": False,
    }
    for domain in ("planar", "reverse"):
        values[f"h5_{domain}_params"] = params[domain]
        values[f"h5_{domain}_params_sha256"] = _sha(params[domain])
        values[f"h5_{domain}_manifest"] = manifests[domain]
        values[f"h5_{domain}_manifest_sha256"] = _sha(manifests[domain])
    return argparse.Namespace(**values)


def _inverse_target(targets: Sequence[float]) -> np.ndarray:
    """Invert the H5 linear-plus-quintic absolute target decoder."""

    desired = np.asarray(targets, dtype=np.float64)
    if desired.shape != (H5_ACTION_WIDTH,):
        raise ValueError("target must be 14-wide")
    initial = np.asarray([float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER])
    lower = np.asarray([
        0.0 if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
        else float(SAFE_JOINT_LIMITS[name][0])
        for name in ACTUATOR_JOINT_ORDER
    ])
    upper = np.asarray([
        0.0 if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
        else float(SAFE_JOINT_LIMITS[name][1])
        for name in ACTUATOR_JOINT_ORDER
    ])
    actions = np.zeros(H5_ACTION_WIDTH, dtype=np.float64)
    for index in LEG_INDICES:
        delta = float(desired[index] - initial[index])
        span = 0.9 * (upper[index] - initial[index] if delta >= 0.0 else initial[index] - lower[index])
        base = min(0.25, span)
        magnitude = abs(delta)
        if magnitude <= 0.0:
            actions[index] = 0.0
            continue
        if magnitude >= span:
            normalized = 1.0
        else:
            # Monotone bisection is robust at the small negative spans used by
            # the calibrated left-side joints.
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                value = base * mid + (span - base) * mid**5
                if value < magnitude:
                    lo = mid
                else:
                    hi = mid
            normalized = 0.5 * (lo + hi)
        actions[index] = np.sign(delta) * normalized
    return actions


def _load_recorded_h3_target_trace(
    path: Path,
    *,
    case: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load fixed H3 targets for a no-tuning diagnostic replay only."""

    source = path.resolve()
    source_sha256 = _sha(source)
    if source_sha256 != RECORDED_H3_TRACE_SHA256:
        raise ValueError("recorded H3 trace hash is not the pinned diagnostic source")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("recorded H3 trace schema must be version 1")
    if payload.get("artifact_kind") != "openduckmini_h3_exact_home_bootstrap_diagnostic":
        raise ValueError("recorded H3 trace artifact kind is not accepted")
    if payload.get("status") != "DIAGNOSTIC_ONLY_NOT_ADOPTED":
        raise ValueError("recorded H3 trace must be diagnostic-only evidence")
    if payload.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("recorded H3 trace must be hardware-prohibited")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("recorded H3 trace records must be a list")
    matching = [record for record in records if record.get("case") == case]
    if len(matching) != 1:
        raise ValueError(f"recorded H3 trace must contain exactly one {case!r} case")
    exact_home = matching[0].get("exact_home")
    if not isinstance(exact_home, dict):
        raise ValueError("recorded H3 trace case lacks exact_home data")
    expected_command = {
        "forward": [0.05, 0.0, 0.0],
        "yaw_left": [0.0, 0.0, 0.30],
    }[case]
    configuration = payload.get("configuration")
    summary = exact_home.get("summary")
    source_segment = exact_home.get("segment")
    if (
        not isinstance(configuration, dict)
        or configuration.get("seconds") != 5.0
        or configuration.get("warmup_seconds") != 1.5
        or not isinstance(summary, dict)
        or summary.get("central_acceptance", {}).get("passed") is not True
        or not isinstance(source_segment, dict)
        or source_segment.get("command") != expected_command
        or exact_home.get("joint_noise_scale") != 0.0
        or exact_home.get("initial_base_speed") != 0.0
        or exact_home.get("physical_base_velocity_injected") is not False
        or exact_home.get("reset_qpos_audit", {}).get("passed") is not True
        or exact_home.get("control_first_startup_audit", {}).get("passed") is not True
    ):
        raise ValueError("recorded H3 trace exact-home provenance is incomplete")
    rows = exact_home.get("trace")
    if not isinstance(rows, list) or not rows:
        raise ValueError("recorded H3 trace exact_home.trace must be nonempty")
    try:
        targets = np.asarray(
            [row["applied_targets_rad"] for row in rows], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("recorded H3 trace has invalid applied targets") from error
    if targets.ndim != 2 or targets.shape[1] != H5_ACTION_WIDTH:
        raise ValueError("recorded H3 applied targets must have shape [ticks, 14]")
    if targets.shape[0] != 250:
        raise ValueError("recorded H3 trace must have exactly 250 control ticks")
    if not np.all(np.isfinite(targets)):
        raise ValueError("recorded H3 applied targets must be finite")
    if not np.array_equal(targets[:, list(HEAD_INDICES)], np.zeros((250, 4))):
        raise ValueError("recorded H3 head targets must be exactly zero")
    control_ticks = [row.get("control_tick") for row in rows]
    if control_ticks != list(range(250)):
        raise ValueError("recorded H3 control ticks must be consecutive without replay edits")
    actions = np.asarray([_inverse_target(row) for row in targets], dtype=np.float64)
    reconstructed = np.asarray(
        [h5_decode_absolute_targets(row, domain="planar") for row in actions],
        dtype=np.float64,
    )
    max_roundtrip_error = float(np.max(np.abs(reconstructed - targets)))
    if max_roundtrip_error > 1.0e-12:
        raise RuntimeError("H3 trace is not exactly reachable through H5 target decoding")
    if np.any(np.abs(actions) > 1.0):
        raise RuntimeError("H3 trace requires H5 action values outside [-1, 1]")
    return actions, targets, {
        "source_path": str(source),
        "source_sha256": source_sha256,
        "artifact_status": payload["status"],
        "hardware_deployment": payload["hardware_deployment"],
        "case": case,
        "control_ticks": int(actions.shape[0]),
        "control_dt_seconds": 0.02,
        "action_abs_peak": float(np.max(np.abs(actions))),
        "max_h5_decode_roundtrip_error_rad": max_roundtrip_error,
    }


def _phase_from_observation(observation: np.ndarray, phase_steps: float) -> float:
    angle = float(np.arctan2(float(observation[100]), float(observation[99])))
    if angle < 0.0:
        angle += 2.0 * np.pi
    return angle / (2.0 * np.pi) * float(phase_steps)


def _polynomial_target(
    command: Sequence[float], phase: float, *, target_scale: float
) -> np.ndarray:
    """Return a calibrated polynomial frame in H5 actuator target order."""

    global _POLYNOMIAL_REFERENCE
    if _POLYNOMIAL_REFERENCE is None:
        _POLYNOMIAL_REFERENCE = PolyReferenceMotion(
            str(DEFAULT_POLYNOMIAL_REFERENCE.resolve())
        )
    reference = _POLYNOMIAL_REFERENCE
    period = int(reference.nb_steps_in_period)
    wrapped = float(phase) % period
    index = int(np.floor(wrapped))
    fraction = wrapped - np.floor(wrapped)
    frame = np.asarray(
        reference.get_reference_motion(
            float(command[0]), float(command[1]), float(command[2]), index
        ),
        dtype=np.float64,
    )
    next_frame = np.asarray(
        reference.get_reference_motion(
            float(command[0]), float(command[1]), float(command[2]),
            (index + 1) % period,
        ),
        dtype=np.float64,
    )
    frame = (1.0 - fraction) * frame + fraction * next_frame
    initial = np.asarray(
        [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER]
    )
    target = initial.copy()
    target[:5] = frame[:5]
    target[9:] = frame[11:16]
    target = initial + float(target_scale) * (target - initial)
    target[list(HEAD_INDICES)] = 0.0
    return np.clip(
        target,
        np.asarray([
            0.0 if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
            else float(SAFE_JOINT_LIMITS[name][0])
            for name in ACTUATOR_JOINT_ORDER
        ]),
        np.asarray([
            0.0 if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
            else float(SAFE_JOINT_LIMITS[name][1])
            for name in ACTUATOR_JOINT_ORDER
        ]),
    )


def _legacy_forward_base_target(
    simulator: Any,
    observation: np.ndarray,
    *,
    phase_offset: float,
    phase_rate_scale: float,
) -> np.ndarray:
    """Probe the audited V22 forward waveform through the H5 decoder.

    This is intentionally a diagnostic target source.  It answers whether
    the existing forward waveform can be reused as a geometric seed for a
    signed reverse gait before any actor distillation is attempted.  The
    physical command remains reverse; only the private teacher query uses the
    established forward observation anchor.
    """

    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (116,):
        raise ValueError("legacy forward probe requires a 116-wide observation")
    policy_observation = np.asarray(values[:101], dtype=np.float32).copy()
    phase = _phase_from_observation(values, simulator.evaluator.phase_steps)
    warped_angle = (
        (phase * float(phase_rate_scale) + float(phase_offset))
        / float(simulator.evaluator.phase_steps)
        * 2.0
        * np.pi
    )
    policy_observation[99:101] = np.asarray(
        (np.cos(warped_angle), np.sin(warped_angle)), dtype=np.float32
    )
    mapped = h5_unified_policy_command((0.05, 0.0, 0.0))
    policy_observation[OBSERVATION_POLICY_COMMAND_SLICE] = np.asarray(
        (*mapped, 0.0, 0.0, 0.0, 0.0), dtype=np.float32
    )
    action = simulator.bank.base_bank.infer(
        "forward", policy_observation
    )
    return np.asarray(
        h5_decode_absolute_targets(action, domain="planar"), dtype=np.float64
    )


def _transform_forward_target(target: np.ndarray, transform: str) -> np.ndarray:
    """Apply one explicit sagittal mirror hypothesis to a forward target."""

    initial = np.asarray(
        [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER],
        dtype=np.float64,
    )
    result = np.asarray(target, dtype=np.float64).copy()
    delta = result - initial
    if transform == "none":
        return result
    if transform == "flip_sagittal":
        result[[2, 4, 11, 13]] = initial[[2, 4, 11, 13]] - delta[[2, 4, 11, 13]]
        return result
    if transform == "flip_sagittal_knee":
        indices = [2, 3, 4, 11, 12, 13]
        result[indices] = initial[indices] - delta[indices]
        return result
    if transform == "flip_hip_pitch":
        indices = [2, 11]
        result[indices] = initial[indices] - delta[indices]
        return result
    if transform == "swap_legs":
        left = [0, 1, 2, 3, 4]
        right = [9, 10, 11, 12, 13]
        result[left] = initial[left] + delta[right]
        result[right] = initial[right] + delta[left]
        return result
    if transform == "swap_legs_flip_sagittal":
        left = [0, 1, 2, 3, 4]
        right = [9, 10, 11, 12, 13]
        result[left] = initial[left] + delta[right]
        result[right] = initial[right] + delta[left]
        indices = [2, 4, 11, 13]
        result[indices] = initial[indices] - (result[indices] - initial[indices])
        return result
    raise ValueError(f"unsupported forward target transform: {transform!r}")


def _apply_target_leg_gains(
    target: np.ndarray,
    gains: Sequence[float] | None,
) -> np.ndarray:
    """Scale target deviations per leg actuator around the audited SAFE_INIT.

    This is an exploration-only morphology knob for the mirrored-forward
    target source.  It changes neither the decoder nor the final safety guard;
    all returned targets still pass through the same inverse decoder and the
    runtime margin/slew guard.
    """

    if gains is None:
        return np.asarray(target, dtype=np.float64)
    values = np.asarray(gains, dtype=np.float64)
    if values.shape != (10,) or not np.all(np.isfinite(values)):
        raise ValueError("target leg gains must be exactly ten finite values")
    if np.any(values < -2.0) or np.any(values > 2.0):
        raise ValueError("target leg gains must be in [-2, 2]")
    initial = np.asarray(
        [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER],
        dtype=np.float64,
    )
    result = initial + (np.asarray(target, dtype=np.float64) - initial)
    leg_indices = np.asarray([0, 1, 2, 3, 4, 9, 10, 11, 12, 13], dtype=np.int32)
    result[leg_indices] = initial[leg_indices] + values * (
        result[leg_indices] - initial[leg_indices]
    )
    result[list(HEAD_INDICES)] = 0.0
    return result


def _profile_target_with_phase_offsets(
    simulator: Any,
    phase: float,
    initial: np.ndarray,
    joint_ranges: np.ndarray,
    gait_scales: np.ndarray,
    gait_biases: np.ndarray,
    phase_offsets: Sequence[float],
) -> np.ndarray:
    """Sample the calibrated profile with diagnostic per-joint phase offsets."""

    evaluator = simulator.evaluator
    offsets = np.asarray(phase_offsets, dtype=np.float64)
    if offsets.shape != (10,) or not np.all(np.isfinite(offsets)):
        raise ValueError("profile phase offsets must be a finite 10-vector")
    deviations = np.asarray(evaluator.backward_leg_deviations, dtype=np.float64)
    means = np.asarray(evaluator.backward_leg_means, dtype=np.float64)
    scales = np.asarray(gait_scales, dtype=np.float64)
    biases = np.asarray(gait_biases, dtype=np.float64)
    if deviations.ndim != 2 or deviations.shape[1] != 10:
        raise ValueError("backward profile deviations must be an N x 10 table")
    if means.shape != (10,) or scales.shape != (10,) or biases.shape != (10,):
        raise ValueError("backward profile vectors must be ten-wide")
    period = int(deviations.shape[0])
    target = np.asarray(initial, dtype=np.float64).copy()
    for column, actuator_index in enumerate(evaluator.backward_actuator_indices):
        wrapped = (float(phase) + float(offsets[column])) % period
        frame_index = int(np.floor(wrapped))
        fraction = wrapped - float(frame_index)
        next_index = (frame_index + 1) % period
        deviation = (
            (1.0 - fraction) * deviations[frame_index, column]
            + fraction * deviations[next_index, column]
        )
        target[int(actuator_index)] = (
            means[column] + biases[column] + scales[column] * deviation
        )
    safe_lower = initial + 0.9 * (joint_ranges[:, 0] - initial)
    safe_upper = initial + 0.9 * (joint_ranges[:, 1] - initial)
    return np.clip(target, safe_lower, safe_upper)


def _program_targets(
    simulator: Any,
    role: str,
    observation: np.ndarray,
    *,
    phase_offset: float,
    phase_rate_scale: float,
    target_scale: float,
    mode: str,
    action_scale: float,
    teacher_table: np.ndarray | None = None,
    actor_residual_scale: float = 1.0,
    target_transform: str = "none",
    target_leg_gains: Sequence[float] | None = None,
    profile_phase_offsets: Sequence[float] | None = None,
) -> np.ndarray:
    phase = _phase_from_observation(observation, simulator.evaluator.phase_steps)
    # The reference itself is the canonical 27-frame geometry.  The phase
    # offset/rate knobs are diagnostic; the acceptance thresholds remain fixed.
    effective_phase = (phase * phase_rate_scale + phase_offset) % simulator.evaluator.phase_steps
    command = np.asarray(observation[101:104], dtype=np.float64)
    if mode == "polynomial":
        return _inverse_target(
            _polynomial_target(command, effective_phase, target_scale=target_scale)
        )
    if mode == "polynomial_reverse_teacher":
        # The calibrated polynomial reference was identified with the
        # historical reverse command magnitude (-0.10 m/s), while H5's
        # physical diagnostic route is deliberately capped at -0.05 m/s.
        # Probe that command-contract hypothesis without changing the
        # physical command, router, or deployment path.
        teacher_command = (
            (-0.10, 0.0, 0.0) if float(command[0]) < -0.02 else tuple(command)
        )
        return _inverse_target(
            _polynomial_target(
                teacher_command, effective_phase, target_scale=target_scale
            )
        )
    if mode == "polynomial_actor":
        initial = np.asarray(
            [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER]
        )
        target = _polynomial_target(
            command, effective_phase, target_scale=target_scale
        )
        actor_action = original_infer(role, observation)
        domain = "reverse" if command[0] < -0.02 else "planar"
        actor_target = np.asarray(
            h5_decode_absolute_targets(actor_action, domain=domain),
            dtype=np.float64,
        )
        target = target + float(actor_residual_scale) * (actor_target - initial)
        target[list(HEAD_INDICES)] = 0.0
        return _inverse_target(target)
    if mode == "legacy_forward_target":
        target = _legacy_forward_base_target(
            simulator,
            observation,
            phase_offset=phase_offset,
            phase_rate_scale=phase_rate_scale,
        )
        target = _transform_forward_target(target, target_transform)
        target = _apply_target_leg_gains(target, target_leg_gains)
        initial = np.asarray(
            [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER],
            dtype=np.float64,
        )
        target = initial + float(target_scale) * (target - initial)
        target[list(HEAD_INDICES)] = 0.0
        return _inverse_target(target)
    if command[0] < -0.02 and mode in {
        "reference", "teacher", "teacher_actor", "h3_profile_actor_residual"
    }:
        initial = np.asarray(
            [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER]
        )
        if mode == "h3_profile_actor_residual":
            # Reproduce the old H3 composition in the H5 target-space
            # diagnostic: the calibrated backward feed-forward supplies the
            # absolute leg geometry and the original V22 actor contributes a
            # bounded residual.  This is intentionally a probe, not a
            # deployment path; the residual must later be distilled into the
            # single unified actor before it can be considered a release.
            scales, biases, _ = simulator.evaluator.backward_parameters(
                float(command[2])
            )
            base_action = simulator.bank.base_bank.infer(
                role, np.asarray(observation[:101], dtype=np.float32)
            )
            target = simulator.evaluator._backward_feedforward(
                effective_phase,
                initial,
                np.asarray([
                    [0.0, 0.0]
                    if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
                    else SAFE_JOINT_LIMITS[name]
                    for name in ACTUATOR_JOINT_ORDER
                ], dtype=np.float64),
                np.clip(np.asarray(base_action, dtype=np.float64), -1.0, 1.0),
                gait_scales=scales,
                gait_biases=biases,
                leg_residual_factor=0.50,
                head_residual_factor=0.0,
            )
            # This branch returns before the common target scaling below.
            # Apply the diagnostic amplitude knob here so H3 residual sweeps
            # actually evaluate distinct target programs.
            target = initial + float(target_scale) * (target - initial)
            target[list(HEAD_INDICES)] = 0.0
            target = np.asarray(margin_clip_targets(target, xp=np), dtype=np.float64)
            if not np.all(np.isfinite(target)):
                raise ValueError("scaled H3 residual target is non-finite")
            return _inverse_target(target)
        use_teacher_table = mode == "teacher" or (
            mode == "teacher_actor" and teacher_table is not None
        )
        if use_teacher_table:
            if teacher_table is None:
                raise ValueError(
                    f"{mode} mode requires --teacher-table with a finite 54x14 table"
                )
            if teacher_table.shape != (54, H5_ACTION_WIDTH):
                raise ValueError("teacher table must be exactly 54x14")
            table_phase = effective_phase * (teacher_table.shape[0] / simulator.evaluator.phase_steps)
            wrapped = table_phase % teacher_table.shape[0]
            index = int(np.floor(wrapped))
            fraction = wrapped - index
            target = (
                (1.0 - fraction) * teacher_table[index]
                + fraction * teacher_table[(index + 1) % teacher_table.shape[0]]
            )
        else:
            scales, biases, _ = simulator.evaluator.backward_parameters(float(command[2]))
            joint_ranges = np.asarray([
                    [0.0, 0.0]
                    if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
                    else SAFE_JOINT_LIMITS[name]
                    for name in ACTUATOR_JOINT_ORDER
                ], dtype=np.float64)
            if profile_phase_offsets is None:
                target = simulator.evaluator._backward_feedforward(
                    effective_phase,
                    initial,
                    joint_ranges,
                    np.zeros(H5_ACTION_WIDTH, dtype=np.float64),
                    gait_scales=scales,
                    gait_biases=biases,
                    leg_residual_factor=0.0,
                    head_residual_factor=0.0,
                )
            else:
                target = _profile_target_with_phase_offsets(
                    simulator,
                    effective_phase,
                    initial,
                    joint_ranges,
                    np.asarray(scales, dtype=np.float64),
                    np.asarray(biases, dtype=np.float64),
                    profile_phase_offsets,
                )
        target = initial + target_scale * (np.asarray(target) - initial)
        if mode == "teacher_actor":
            actor_action = original_infer(role, observation)
            actor_target = np.asarray(
                h5_decode_absolute_targets(actor_action, domain="reverse"),
                dtype=np.float64,
            )
            target = target + float(actor_residual_scale) * (actor_target - initial)
        target[list(HEAD_INDICES)] = 0.0
        return _inverse_target(target)
    if mode == "base":
        policy_observation = np.asarray(observation[:101], dtype=np.float32).copy()
        if phase_rate_scale != 1.0 or phase_offset != 0.0:
            warped_angle = (
                phase * phase_rate_scale + phase_offset
            ) / float(simulator.evaluator.phase_steps) * 2.0 * np.pi
            policy_observation[99:101] = np.asarray(
                (np.cos(warped_angle), np.sin(warped_angle)), dtype=np.float32
            )
        action = simulator.bank.base_bank.infer(role, policy_observation)
    else:
        action = original_infer(role, observation)
    return np.clip(np.asarray(action, dtype=np.float64) * action_scale, -1.0, 1.0)


original_infer: Any = None


def _apply_profile_modifiers(
    simulator: Any,
    *,
    profile_multipliers: Sequence[float] | None,
    profile_bias_offsets: Sequence[float] | None,
) -> None:
    if profile_multipliers is None and profile_bias_offsets is None:
        return
    multipliers = (
        np.ones(10, dtype=np.float64)
        if profile_multipliers is None
        else np.asarray(profile_multipliers, dtype=np.float64)
    )
    offsets = (
        np.zeros(10, dtype=np.float64)
        if profile_bias_offsets is None
        else np.asarray(profile_bias_offsets, dtype=np.float64)
    )
    if (
        multipliers.shape != (10,)
        or offsets.shape != (10,)
        or not np.all(np.isfinite(multipliers))
        or not np.all(np.isfinite(offsets))
    ):
        raise ValueError("profile multipliers and offsets must be finite 10-vectors")
    evaluator = simulator.evaluator._evaluator
    evaluator.backward_gait_scales = (
        np.asarray(evaluator.backward_gait_scales, dtype=np.float64) * multipliers
    )
    evaluator.backward_gait_biases = (
        np.asarray(evaluator.backward_gait_biases, dtype=np.float64) + offsets
    )


def run_probe(
    *,
    params: dict[str, Path],
    manifests: dict[str, Path],
    route: str,
    seconds: float,
    phase_offset: float,
    phase_rate_scale: float,
    target_scale: float,
    profile: Path | None = None,
    mode: str = "reference",
    action_scale: float = 1.0,
    action_smoothing: float = 1.0,
    target_smoothing: float = 1.0,
    teacher_table: np.ndarray | None = None,
    actor_residual_scale: float = 1.0,
    backward_residual_scale: float = 0.0,
    target_transform: str = "none",
    target_leg_gains: Sequence[float] | None = None,
    profile_phase_offsets: Sequence[float] | None = None,
    profile_multipliers: Sequence[float] | None = None,
    profile_bias_offsets: Sequence[float] | None = None,
    replay_actions: np.ndarray | None = None,
    replay_targets: np.ndarray | None = None,
    replay_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    global original_infer
    args = _args(params, manifests, strict_actor_only=replay_actions is not None)
    args.seconds = float(seconds)
    simulator, bank, metadata = h5._build_simulator(args)
    if not np.isfinite(float(backward_residual_scale)) or not 0.0 <= float(
        backward_residual_scale
    ) <= 0.25:
        raise ValueError("backward_residual_scale must be finite and in [0, 0.25]")
    simulator.evaluator._evaluator.backward_residual_scale = float(
        backward_residual_scale
    )
    if profile is not None:
        simulator.evaluator._evaluator.load_backward_profile(profile.resolve())
    if profile_multipliers is not None or profile_bias_offsets is not None:
        if profile is None:
            raise ValueError("profile modifiers require --profile")
        _apply_profile_modifiers(
            simulator,
            profile_multipliers=profile_multipliers,
            profile_bias_offsets=profile_bias_offsets,
        )
    return _run_on_simulator(
        simulator,
        bank,
        metadata,
        route=route,
        seconds=seconds,
        phase_offset=phase_offset,
        phase_rate_scale=phase_rate_scale,
        target_scale=target_scale,
        mode=mode,
        action_scale=action_scale,
        action_smoothing=action_smoothing,
        teacher_table=teacher_table,
        actor_residual_scale=actor_residual_scale,
        target_transform=target_transform,
        target_leg_gains=target_leg_gains,
        profile_phase_offsets=profile_phase_offsets,
        profile_multipliers=profile_multipliers,
        profile_bias_offsets=profile_bias_offsets,
        target_smoothing=target_smoothing,
        replay_actions=replay_actions,
        replay_targets=replay_targets,
        replay_source=replay_source,
    )


def _run_on_simulator(
    simulator: Any,
    bank: Any,
    metadata: dict[str, Any],
    *,
    route: str,
    seconds: float,
    phase_offset: float,
    phase_rate_scale: float,
    target_scale: float,
    mode: str,
    action_scale: float,
    action_smoothing: float,
    teacher_table: np.ndarray | None,
    actor_residual_scale: float,
    target_transform: str = "none",
    target_leg_gains: Sequence[float] | None = None,
    profile_phase_offsets: Sequence[float] | None = None,
    profile_multipliers: Sequence[float] | None = None,
    profile_bias_offsets: Sequence[float] | None = None,
    target_smoothing: float = 1.0,
    joint_noise_scale: float = 0.0,
    initial_base_speed: float = 0.0,
    warmup_seconds: float = 1.5,
    replay_actions: np.ndarray | None = None,
    replay_targets: np.ndarray | None = None,
    replay_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    global original_infer
    if not np.isfinite(float(target_smoothing)) or not 0.0 < float(target_smoothing) <= 1.0:
        raise ValueError("target_smoothing must be finite and in (0, 1]")
    if (replay_actions is None) != (replay_source is None) or (
        replay_actions is None
    ) != (replay_targets is None):
        raise ValueError("replay actions, targets, and source must be supplied together")
    if replay_actions is not None:
        replay_actions = np.asarray(replay_actions, dtype=np.float64)
        replay_targets = np.asarray(replay_targets, dtype=np.float64)
        if replay_actions.ndim != 2 or replay_actions.shape[1] != H5_ACTION_WIDTH:
            raise ValueError("replay actions must have shape [ticks, 14]")
        if replay_targets.shape != replay_actions.shape:
            raise ValueError("replay targets must match replay action shape")
        if not np.all(np.isfinite(replay_actions)) or np.any(np.abs(replay_actions) > 1.0):
            raise ValueError("replay actions must be finite and inside [-1, 1]")
        if not np.all(np.isfinite(replay_targets)):
            raise ValueError("replay targets must be finite")
        expected_seconds = float(replay_actions.shape[0]) * float(simulator.runtime.CONTROL_DT)
        if not np.isclose(float(seconds), expected_seconds, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "replay seconds must exactly match recorded control ticks at H5 control_dt"
            )
    if not hasattr(bank, "_exploration_original_infer"):
        bank._exploration_original_infer = bank.infer
    original_infer = bank._exploration_original_infer
    previous_action = np.zeros(H5_ACTION_WIDTH, dtype=np.float64)
    previous_target = np.asarray(
        [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER],
        dtype=np.float64,
    )
    action_samples: list[np.ndarray] = []
    replay_tick_trace: list[dict[str, np.ndarray]] = []
    restore_infer_route: Any | None = None
    restore_guarded_control: Any | None = None

    def infer(role: str, observation: np.ndarray) -> np.ndarray:
        nonlocal previous_action, previous_target
        action = _program_targets(
            simulator,
            role,
            observation,
            phase_offset=phase_offset,
            phase_rate_scale=phase_rate_scale,
            target_scale=target_scale,
            mode=mode,
            action_scale=action_scale,
            teacher_table=teacher_table,
            actor_residual_scale=actor_residual_scale,
            target_transform=target_transform,
            target_leg_gains=target_leg_gains,
            profile_phase_offsets=profile_phase_offsets,
        )
        if action_smoothing < 1.0:
            action = (
                action_smoothing * action
                + (1.0 - action_smoothing) * previous_action
            )
        if target_smoothing < 1.0:
            domain = "reverse" if float(observation[101]) < -0.02 else "planar"
            desired_target = np.asarray(
                h5_decode_absolute_targets(action, domain=domain),
                dtype=np.float64,
            )
            target = (
                target_smoothing * desired_target
                + (1.0 - target_smoothing) * previous_target
            )
            target[list(HEAD_INDICES)] = 0.0
            previous_target = target.copy()
            action = _inverse_target(target)
        else:
            domain = "reverse" if float(observation[101]) < -0.02 else "planar"
            previous_target = np.asarray(
                h5_decode_absolute_targets(action, domain=domain),
                dtype=np.float64,
            )
        previous_action = np.asarray(action, dtype=np.float64).copy()
        action_samples.append(previous_action.copy())
        return action

    if replay_actions is None:
        bank.infer = infer  # type: ignore[method-assign]
    else:
        original_infer_route = bank.infer_route
        original_guarded_control = h5.h4._RUN_SCHEDULE_GLOBALS[
            "apply_guarded_control_then_step_physics"
        ]
        restore_infer_route = original_infer_route
        restore_guarded_control = original_guarded_control

        def replay_infer_route(
            decision: Any, observation: np.ndarray, *args: Any, **kwargs: Any
        ) -> tuple[np.ndarray, np.ndarray]:
            replay_index = len(action_samples)
            if replay_index >= replay_actions.shape[0]:
                raise RuntimeError("H5 replay requested more ticks than the recorded trace")
            action = replay_actions[replay_index].copy()
            action_samples.append(action.copy())
            original_infer = bank.infer

            def replay_infer(_role: str, _observation: np.ndarray) -> np.ndarray:
                return action.copy()

            bank.infer = replay_infer  # type: ignore[method-assign]
            try:
                return original_infer_route(decision, observation, *args, **kwargs)
            finally:
                bank.infer = original_infer  # type: ignore[method-assign]

        bank.infer_route = replay_infer_route  # type: ignore[method-assign]

        def traced_guarded_control(
            target_guard: Any, candidate_targets: np.ndarray, *args: Any, **kwargs: Any
        ) -> tuple[Any, Any, Any, Any, Any]:
            result = original_guarded_control(
                target_guard, candidate_targets, *args, **kwargs
            )
            previous, applied, *_rest = result
            replay_tick_trace.append(
                {
                    "candidate": np.asarray(candidate_targets, dtype=np.float64).copy(),
                    "previous": np.asarray(previous, dtype=np.float64).copy(),
                    "applied": np.asarray(applied, dtype=np.float64).copy(),
                }
            )
            return result

        h5.h4._RUN_SCHEDULE_GLOBALS[
            "apply_guarded_control_then_step_physics"
        ] = traced_guarded_control
    command = {
        "forward": (0.05, 0.0, 0.0),
        "lateral_left": (0.0, 0.06, 0.0),
        "lateral_right": (0.0, -0.06, 0.0),
        "yaw_left": (0.0, 0.0, 0.30),
        "yaw_right": (0.0, 0.0, -0.30),
        "forward_turn_left": (0.04, 0.0, 0.30),
        "forward_turn_right": (0.04, 0.0, -0.22),
        "forward_lateral_left_turn": (0.04, 0.05, 0.17),
        "forward_lateral_right_turn": (0.04, -0.03, -0.15),
        "reverse": (-0.050, 0.0, 0.0),
        "reverse_turn_left": (-0.03, 0.0, 0.20),
        "reverse_turn_right": (-0.04, 0.0, -0.20),
    }[route]
    expected_policy_role = routed.canonical_policy_role(route)
    try:
        run = simulator.run_schedule(
            ((route, command, float(seconds), None, route, expected_policy_role),),
            seed=20260808,
            joint_noise_scale=float(joint_noise_scale),
            initial_base_speed=float(initial_base_speed),
            warmup_seconds=float(warmup_seconds),
        )
    finally:
        if restore_infer_route is not None:
            bank.infer_route = restore_infer_route  # type: ignore[method-assign]
        if restore_guarded_control is not None:
            h5.h4._RUN_SCHEDULE_GLOBALS[
                "apply_guarded_control_then_step_physics"
            ] = restore_guarded_control
    segment = run["segments"][0]
    quality = segment.get("gait_quality_acceptance", {})
    action_array = (
        np.asarray(action_samples, dtype=np.float64)
        if action_samples
        else np.zeros((0, H5_ACTION_WIDTH), dtype=np.float64)
    )
    strict_acceptance = segment_acceptance(
        segment,
        AcceptanceThresholds(),
        require_gait_quality=True,
    )
    rederived_quality = strict_acceptance.get("rederived_gait_quality_acceptance")
    if not isinstance(rederived_quality, Mapping):
        raise RuntimeError("strict acceptance did not rederive gait quality")
    replay_fidelity: dict[str, Any] | None = None
    bank_manifest = bank.manifest()
    if replay_source is not None:
        captured_count = len(replay_tick_trace)
        if captured_count == replay_targets.shape[0]:
            captured_candidate = np.asarray(
                [row["candidate"] for row in replay_tick_trace], dtype=np.float64
            )
            captured_applied = np.asarray(
                [row["applied"] for row in replay_tick_trace], dtype=np.float64
            )
            candidate_error = float(np.max(np.abs(captured_candidate - replay_targets)))
            applied_error = float(np.max(np.abs(captured_applied - replay_targets)))
        else:
            candidate_error = None
            applied_error = None
        replay_fidelity = {
            "captured_control_ticks": captured_count,
            "expected_control_ticks": int(replay_targets.shape[0]),
            "candidate_target_max_abs_error_rad": candidate_error,
            "applied_target_max_abs_error_rad": applied_error,
            "candidate_target_exact": bool(
                candidate_error is not None and candidate_error <= 1.0e-12
            ),
            "applied_target_exact": bool(
                applied_error is not None and applied_error <= 1.0e-12
            ),
            "one_row_per_control_tick": bool(
                captured_count == replay_targets.shape[0]
                and action_array.shape[0] == replay_targets.shape[0]
            ),
        }
    result = {
        "route": route,
        "command": list(command),
        "seconds": float(seconds),
        "phase_offset": float(phase_offset),
        "phase_rate_scale": float(phase_rate_scale),
        "target_scale": float(target_scale),
        "target_leg_gains": (
            [float(value) for value in target_leg_gains]
            if target_leg_gains is not None
            else None
        ),
        "profile_multipliers": (
            [float(value) for value in profile_multipliers]
            if profile_multipliers is not None
            else None
        ),
        "profile_bias_offsets": (
            [float(value) for value in profile_bias_offsets]
            if profile_bias_offsets is not None
            else None
        ),
        "profile_phase_offsets": (
            [float(value) for value in profile_phase_offsets]
            if profile_phase_offsets is not None
            else None
        ),
        "mode": mode,
        "action_scale": float(action_scale),
        "action_smoothing": float(action_smoothing),
        "target_smoothing": float(target_smoothing),
        "actor_residual_scale": float(actor_residual_scale),
        "action_summary": {
            "sample_count": int(action_array.shape[0]),
            "mean": action_array.mean(axis=0).tolist()
            if action_array.size
            else [0.0] * H5_ACTION_WIDTH,
            "std": action_array.std(axis=0).tolist()
            if action_array.size
            else [0.0] * H5_ACTION_WIDTH,
            "minimum": action_array.min(axis=0).tolist()
            if action_array.size
            else [0.0] * H5_ACTION_WIDTH,
            "maximum": action_array.max(axis=0).tolist()
            if action_array.size
            else [0.0] * H5_ACTION_WIDTH,
        },
        "hardware_deployment": "PROHIBITED",
        "qualification": "SEED_SCREEN_ONLY",
        "adoption_allowed": False,
        "release_allowed": False,
        "metadata": metadata,
        "strict_segment_acceptance": strict_acceptance,
        "passed": bool(strict_acceptance.get("passed", False)),
        "quality_passed": bool(rederived_quality.get("passed", False)),
        "quality_failures": list(rederived_quality.get("failures", [])),
        "segment": segment,
    }
    if replay_source is not None:
        result["recorded_h3_target_replay"] = {
            **dict(replay_source),
            "expected_control_ticks": int(replay_actions.shape[0]),
            "inferred_control_ticks": int(action_array.shape[0]),
            "exhausted_exactly": bool(action_array.shape[0] == replay_actions.shape[0]),
            "strict_actor_only": bool(
                metadata.get("legacy_execution", {}).get("strict_actor_only")
            ),
            "legacy_fallback_count": int(
                bank_manifest.get("legacy_fallback", {}).get("count", -1)
            ),
            "no_target_or_action_tuning": bool(
                mode == "recorded_h3_trace"
                and target_scale == 1.0
                and action_scale == 1.0
                and action_smoothing == 1.0
                and target_smoothing == 1.0
            ),
        }
        result["replay_fidelity"] = replay_fidelity
        result["seed_screen_status"] = (
            "H5_SEED_CANDIDATE"
            if (
                result["passed"]
                and replay_fidelity["candidate_target_exact"]
                and replay_fidelity["applied_target_exact"]
                and replay_fidelity["one_row_per_control_tick"]
                and result["recorded_h3_target_replay"]["strict_actor_only"]
                and result["recorded_h3_target_replay"]["legacy_fallback_count"] == 0
            )
            else "REJECTED_AS_H5_SEED"
        )
    return result


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        default="reverse",
        choices=(
            "forward", "lateral_left", "lateral_right", "yaw_left", "yaw_right",
            "forward_turn_left", "forward_turn_right",
            "forward_lateral_left_turn", "forward_lateral_right_turn",
            "reverse", "reverse_turn_left", "reverse_turn_right",
        ),
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--phase-offset", type=float, default=0.0)
    parser.add_argument("--phase-offsets", type=float, nargs="+")
    parser.add_argument("--phase-rate-scale", type=float, default=1.0)
    parser.add_argument("--phase-rate-scales", type=float, nargs="+")
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--target-scales", type=float, nargs="+")
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--mode",
        choices=(
            "polynomial", "polynomial_reverse_teacher", "polynomial_actor", "reference", "teacher",
            "teacher_actor", "h3_profile_actor_residual", "legacy_forward_target",
            "actor", "base", "recorded_h3_trace",
        ),
        default="reference",
    )
    parser.add_argument("--teacher-table", type=Path)
    parser.add_argument(
        "--replay-h3-trace",
        type=Path,
        help=(
            "Diagnostic-only H3 trace to replay unchanged through the H5 target "
            "decoder, frozen guard, and simulator. Requires recorded_h3_trace mode."
        ),
    )
    parser.add_argument(
        "--replay-h3-case",
        choices=("forward", "yaw_left"),
        help="Exact-home case to select from --replay-h3-trace.",
    )
    parser.add_argument(
        "--target-transform",
        choices=(
            "none", "flip_sagittal", "flip_sagittal_knee", "flip_hip_pitch",
            "swap_legs", "swap_legs_flip_sagittal",
        ),
        default="none",
    )
    parser.add_argument(
        "--target-leg-gains",
        type=float,
        nargs=10,
        metavar=("Y_L", "R_L", "P_L", "K_L", "A_L", "Y_R", "R_R", "P_R", "K_R", "A_R"),
        help=(
            "Exploration-only per-leg deviation gains around SAFE_INIT for "
            "legacy_forward_target; each value is constrained to [-2, 2]."
        ),
    )
    parser.add_argument(
        "--profile-multipliers",
        type=float,
        nargs=10,
        metavar=("Y_L", "R_L", "P_L", "K_L", "A_L", "Y_R", "R_R", "P_R", "K_R", "A_R"),
        help=(
            "Exploration-only per-joint multipliers applied to the loaded "
            "backward profile scales. Requires --profile."
        ),
    )
    parser.add_argument(
        "--profile-bias-offsets",
        type=float,
        nargs=10,
        metavar=("Y_L", "R_L", "P_L", "K_L", "A_L", "Y_R", "R_R", "P_R", "K_R", "A_R"),
        help=(
            "Exploration-only per-joint radian offsets applied to the loaded "
            "backward profile biases. Requires --profile."
        ),
    )
    parser.add_argument(
        "--profile-phase-offsets",
        type=float,
        nargs=10,
        metavar=("Y_L", "R_L", "P_L", "K_L", "A_L", "Y_R", "R_R", "P_R", "K_R", "A_R"),
        help=(
            "Exploration-only per-joint phase offsets in profile-frame units; "
            "requires --profile and reference-like mode."
        ),
    )
    parser.add_argument("--actor-residual-scale", type=float, default=1.0)
    parser.add_argument("--actor-residual-scales", type=float, nargs="+")
    parser.add_argument("--backward-residual-scale", type=float, default=0.0)
    parser.add_argument("--planar-params", type=Path)
    parser.add_argument("--reverse-params", type=Path)
    parser.add_argument("--planar-manifest", type=Path)
    parser.add_argument("--reverse-manifest", type=Path)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--action-scales", type=float, nargs="+")
    parser.add_argument("--action-smoothing", type=float, default=1.0)
    parser.add_argument("--action-smoothings", type=float, nargs="+")
    parser.add_argument("--target-smoothing", type=float, default=1.0)
    parser.add_argument("--target-smoothings", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cli(argv)
    params = dict(DEFAULT_H5_PARAMS)
    manifests = dict(DEFAULT_H5_MANIFESTS)
    if args.planar_params is not None:
        params["planar"] = args.planar_params.resolve()
    if args.reverse_params is not None:
        params["reverse"] = args.reverse_params.resolve()
    if args.planar_manifest is not None:
        manifests["planar"] = args.planar_manifest.resolve()
    if args.reverse_manifest is not None:
        manifests["reverse"] = args.reverse_manifest.resolve()
    teacher_table = None
    if args.teacher_table is not None:
        payload = json.loads(args.teacher_table.resolve().read_text(encoding="utf-8"))
        teacher_payload = payload.get("teacher") or payload.get("teacher_source")
        if not isinstance(teacher_payload, dict):
            raise ValueError(
                "--teacher-table must contain teacher or teacher_source metadata"
            )
        teacher_table = np.asarray(
            teacher_payload["target_table_rad"], dtype=np.float64
        )
        if teacher_table.shape != (54, H5_ACTION_WIDTH) or not np.all(np.isfinite(teacher_table)):
            raise ValueError("--teacher-table must contain a finite teacher.target_table_rad 54x14")
    replay_actions = None
    replay_targets = None
    replay_source = None
    if args.replay_h3_trace is not None:
        if args.mode != "recorded_h3_trace" or args.replay_h3_case is None:
            raise ValueError(
                "--replay-h3-trace requires --mode recorded_h3_trace and --replay-h3-case"
            )
        expected_route = {"forward": "forward", "yaw_left": "yaw_left"}[args.replay_h3_case]
        if args.route != expected_route:
            raise ValueError("H3 replay case and H5 route must match exactly")
        if args.teacher_table is not None or args.profile is not None:
            raise ValueError("recorded H3 replay cannot combine a teacher table or profile")
        replay_actions, replay_targets, replay_source = _load_recorded_h3_target_trace(
            args.replay_h3_trace, case=args.replay_h3_case
        )
    elif args.mode == "recorded_h3_trace":
        raise ValueError("recorded_h3_trace mode requires --replay-h3-trace")
    scales = args.target_scales or [args.target_scale]
    offsets = args.phase_offsets or [args.phase_offset]
    action_scales = args.action_scales or [args.action_scale]
    smoothings = args.action_smoothings or [args.action_smoothing]
    target_smoothings = args.target_smoothings or [args.target_smoothing]
    residual_scales = args.actor_residual_scales or [args.actor_residual_scale]
    rates = args.phase_rate_scales or [args.phase_rate_scale]
    if replay_actions is not None:
        if (
            len(scales) != 1
            or len(offsets) != 1
            or len(action_scales) != 1
            or len(smoothings) != 1
            or len(target_smoothings) != 1
            or len(rates) != 1
            or len(residual_scales) != 1
            or scales[0] != 1.0
            or action_scales[0] != 1.0
            or smoothings[0] != 1.0
            or target_smoothings[0] != 1.0
            or offsets[0] != 0.0
            or rates[0] != 1.0
            or residual_scales[0] != 1.0
            or args.backward_residual_scale != 0.0
            or args.target_transform != "none"
            or args.target_leg_gains is not None
            or args.profile_phase_offsets is not None
            or args.profile_multipliers is not None
            or args.profile_bias_offsets is not None
        ):
            raise ValueError("recorded H3 replay forbids all target-program tuning")
        args.seconds = replay_actions.shape[0] * 0.02
    if len(scales) == 1 and len(offsets) == 1 and len(action_scales) == 1 and len(smoothings) == 1 and len(target_smoothings) == 1 and len(rates) == 1 and len(residual_scales) == 1:
        result = run_probe(
            params=params,
            manifests=manifests,
            route=args.route,
            seconds=args.seconds,
            phase_offset=offsets[0],
            phase_rate_scale=rates[0],
            target_scale=scales[0],
            profile=args.profile,
            mode=args.mode,
            action_scale=action_scales[0],
            action_smoothing=smoothings[0],
            target_smoothing=target_smoothings[0],
            teacher_table=teacher_table,
            actor_residual_scale=residual_scales[0],
            backward_residual_scale=args.backward_residual_scale,
            target_transform=args.target_transform,
            target_leg_gains=args.target_leg_gains,
            profile_phase_offsets=args.profile_phase_offsets,
            profile_multipliers=args.profile_multipliers,
            profile_bias_offsets=args.profile_bias_offsets,
            replay_actions=replay_actions,
            replay_targets=replay_targets,
            replay_source=replay_source,
        )
        results = [result]
    else:
        setup = _args(params, manifests)
        setup.seconds = float(args.seconds)
        simulator, bank, metadata = h5._build_simulator(setup)
        if args.profile is not None:
            simulator.evaluator._evaluator.load_backward_profile(args.profile.resolve())
        if args.profile_multipliers is not None or args.profile_bias_offsets is not None:
            if args.profile is None:
                raise ValueError("profile modifiers require --profile")
            _apply_profile_modifiers(
                simulator,
                profile_multipliers=args.profile_multipliers,
                profile_bias_offsets=args.profile_bias_offsets,
            )
        if not np.isfinite(float(args.backward_residual_scale)) or not 0.0 <= float(
            args.backward_residual_scale
        ) <= 0.25:
            raise ValueError("backward_residual_scale must be finite and in [0, 0.25]")
        simulator.evaluator._evaluator.backward_residual_scale = float(
            args.backward_residual_scale
        )
        results = []
        for offset in offsets:
            for rate in rates:
                for scale in scales:
                    for action_scale in action_scales:
                        for smoothing in smoothings:
                            for target_smoothing in target_smoothings:
                                for residual_scale in residual_scales:
                                    results.append(
                                        _run_on_simulator(
                                        simulator,
                                        bank,
                                        metadata,
                                        route=args.route,
                                        seconds=args.seconds,
                                        phase_offset=offset,
                                        phase_rate_scale=rate,
                                        target_scale=scale,
                                        mode=args.mode,
                                        action_scale=action_scale,
                                        action_smoothing=smoothing,
                                        target_smoothing=target_smoothing,
                                        teacher_table=teacher_table,
                                        actor_residual_scale=residual_scale,
                                        target_transform=args.target_transform,
                                        target_leg_gains=args.target_leg_gains,
                                        profile_phase_offsets=args.profile_phase_offsets,
                                        profile_multipliers=args.profile_multipliers,
                                        profile_bias_offsets=args.profile_bias_offsets,
                                        replay_actions=replay_actions,
                                        replay_targets=replay_targets,
                                        replay_source=replay_source,
                                        )
                                    )
        result = results[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: Any = result if len(results) == 1 else {"probes": results}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "route": args.route,
        "probes": [
            {
                "target_scale": item["target_scale"],
                "passed": item["passed"],
                "quality_passed": item["quality_passed"],
                "quality_failures": item["quality_failures"],
            }
            for item in results
        ],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
