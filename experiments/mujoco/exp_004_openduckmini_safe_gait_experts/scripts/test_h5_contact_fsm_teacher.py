"""Evaluate a contact-mode/Jacobian-IK reverse teacher in exact MuJoCo.

This experiment follows the rejection of the independent and structured CEM
teachers.  It makes stance-foot position an explicit state constraint and
generates the swing-foot trajectory from a two-mode finite-state machine:

    left stance / right swing -> right stance / left swing

The teacher has no actor fallback and never writes adoption or deployment
evidence.  Its purpose is to determine whether direct stance-foot anchoring
can reduce the measured slip enough to justify later distillation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
if str(EXP_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(EXP_ROOT / "scripts"))

from safe_gait_experts.contract import (  # noqa: E402
    ACTUATOR_JOINT_ORDER,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
)
from safe_gait_experts.gait_quality import (  # noqa: E402
    GaitQualityAccumulator,
    GaitQualitySubstep,
    gait_quality_acceptance,
)
from safe_gait_experts.h5_target_contract import h5_decode_absolute_targets  # noqa: E402
from scripts import evaluate_h4_routed_transitions as h4  # noqa: E402
from scripts import evaluate_h5_routed_transitions as h5  # noqa: E402
from scripts import evaluate_routed_transitions as routed  # noqa: E402
from scripts import build_h5_mpc_reverse_teacher as base  # noqa: E402


SEED = 20260810
PHYSICAL_COMMAND = np.asarray((-0.05, 0.0, 0.0), dtype=np.float64)
MODE_TICKS = 15
STEP_LENGTH_M = 0.015
SWING_LIFT_M = 0.025
IK_GAIN_STANCE = 0.85
IK_GAIN_SWING = 0.65
IK_DAMPING = 0.004
IK_MAX_DELTA_RAD = 0.08
DEFAULT_SECONDS = 6.0
DEFAULT_OUTPUT = EXP_ROOT / "artifacts" / "h5_contact_fsm_teacher_20260811"
HEAD_NAMES = {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
LEG_NAMES = [name for name in ACTUATOR_JOINT_ORDER if name not in HEAD_NAMES]
LEG_INDEX = {name: index for index, name in enumerate(LEG_NAMES)}
FULL_INDEX = {name: index for index, name in enumerate(ACTUATOR_JOINT_ORDER)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _build_simulator() -> tuple[Any, Any, dict[str, Any]]:
    args = base._build_args(
        base.DEFAULT_PARAMS,
        base.DEFAULT_PLANAR_MANIFEST,
        base.DEFAULT_REVERSE_MANIFEST,
        base.DEFAULT_GENERATED_ROOT,
    )
    return h5._build_simulator(args)


def _site_position(simulator: Any, data: Any, foot: int) -> np.ndarray:
    site_id = (
        simulator.left_foot_site_id if foot == 0 else simulator.right_foot_site_id
    )
    return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()


def _jacobian_position(simulator: Any, data: Any, foot: int) -> np.ndarray:
    site_id = (
        simulator.left_foot_site_id if foot == 0 else simulator.right_foot_site_id
    )
    jacp = np.zeros((3, int(simulator.model.nv)), dtype=np.float64)
    jacr = np.zeros((3, int(simulator.model.nv)), dtype=np.float64)
    simulator.mujoco.mj_jacSite(simulator.model, data, jacp, jacr, site_id)
    del jacr
    names = (
        ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle")
        if foot == 0
        else ("right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
    )
    addresses = [
        int(simulator.evaluator.actuator_qpos_addr[FULL_INDEX[name]]) for name in names
    ]
    return jacp[:, addresses]


def _ik_target(
    simulator: Any,
    data: Any,
    *,
    stance_foot: int,
    stance_anchor: np.ndarray,
    swing_anchor: np.ndarray,
    swing_phase: float,
    previous_target: np.ndarray,
) -> np.ndarray:
    """Solve two damped foot-position corrections in the current qpos frame."""

    current = np.asarray(
        data.qpos[simulator.evaluator.actuator_qpos_addr], dtype=np.float64
    ).copy()
    desired = current.copy()
    swing_foot = 1 - stance_foot
    phase = float(np.clip(swing_phase, 0.0, 1.0))
    smooth = phase * phase * (3.0 - 2.0 * phase)
    swing_target = swing_anchor.copy()
    swing_target[0] -= STEP_LENGTH_M * smooth
    swing_target[2] += SWING_LIFT_M * np.sin(np.pi * phase)
    foot_targets = {stance_foot: stance_anchor, swing_foot: swing_target}
    for foot, target_position in foot_targets.items():
        current_position = _site_position(simulator, data, foot)
        error = np.asarray(target_position, dtype=np.float64) - current_position
        jacobian = _jacobian_position(simulator, data, foot)
        damping = IK_DAMPING * np.eye(3, dtype=np.float64)
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping,
            error,
        )
        gain = IK_GAIN_STANCE if foot == stance_foot else IK_GAIN_SWING
        dq = np.clip(gain * dq, -IK_MAX_DELTA_RAD, IK_MAX_DELTA_RAD)
        if foot == 0:
            indices = np.asarray([FULL_INDEX[name] for name in (
                "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"
            )], dtype=np.int32)
        else:
            indices = np.asarray([FULL_INDEX[name] for name in (
                "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"
            )], dtype=np.int32)
        desired[indices] = current[indices] + dq
    desired[base.HEAD_INDICES] = 0.0
    # The final stateful guard owns the actual target.  This preclip only
    # prevents the IK proposal from requesting an impossible inward envelope.
    desired = routed.apply_final_target_safety(
        desired,
        SAFE_JOINT_LIMITS,
        margin_rad=base.TARGET_MARGIN_RAD,
    )
    del previous_target
    return desired


def _quality_update(
    simulator: Any,
    data: Any,
    quality: GaitQualityAccumulator,
    physics: Any,
    *,
    elapsed_time: float,
    previous_position: np.ndarray,
) -> None:
    del previous_position
    rotation = np.asarray(
        data.xmat[simulator.evaluator.trunk_body_id], dtype=np.float64
    ).reshape(3, 3)
    position = np.asarray(
        data.xpos[simulator.evaluator.trunk_body_id], dtype=np.float64
    )
    contact = np.asarray(simulator.evaluator._feet_contacts(data), dtype=bool)
    normal_force, tangential_speed = simulator._quality_contact_kinematics(data)
    qpos = np.asarray(
        data.qpos[simulator.evaluator.actuator_qpos_addr], dtype=np.float64
    )
    quality.update(
        GaitQualitySubstep(
            time_s=float(elapsed_time),
            requested_command=PHYSICAL_COMMAND,
            effective_command=PHYSICAL_COMMAND,
            local_velocity_xyz_mps=simulator.evaluator._sensor(data, "local_linvel"),
            local_yaw_rate_radps=float(simulator.evaluator._sensor(data, "gyro")[2]),
            trunk_position_world_m=position,
            feet_contacts=contact,
            foot_contact_points_world_m=np.asarray(
                [_site_position(simulator, data, 0), _site_position(simulator, data, 1)],
                dtype=np.float64,
            ),
            leg_joint_positions_rad=qpos,
            feet_normal_force_fraction_body_weight=normal_force,
            foot_contact_tangential_speeds_mps=tangential_speed,
            trunk_yaw_world_rad=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
            trunk_pose_measurement_source="mujoco_shadow_xpos_xmat_after_mj_step",
        )
    )
    physics.update(
        joint_qpos=qpos,
        full_qpos=np.asarray(data.qpos, dtype=np.float64),
        full_qvel=np.asarray(data.qvel, dtype=np.float64),
        height_m=float(position[2]),
        upright=float(rotation[2, 2]),
        feet_contacts=contact,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    simulator, bank, build_metadata = _build_simulator()
    data, reset_audit = simulator._initial_data(int(args.seed), 0.0, 0.0)
    previous_target = np.asarray(
        data.qpos[simulator.evaluator.actuator_qpos_addr], dtype=np.float64
    ).copy()
    guard = routed.FinalTargetSafetyGuard(
        SAFE_JOINT_LIMITS,
        previous_target,
        margin_rad=base.TARGET_MARGIN_RAD,
        max_slew_rate_rad_s=2.0,
    )
    previous_target = guard.previous_targets
    quality = GaitQualityAccumulator(joint_names=simulator.joint_names)
    physics = routed.PhysicsSubstepAudit(simulator.joint_names)
    default = np.asarray(simulator.model.keyframe("home").ctrl, dtype=np.float64)
    action_history = [np.zeros(14, dtype=np.float32) for _ in range(3)]
    control_ticks = int(round(float(args.seconds) / base.CONTROL_DT_S))
    previous_position = np.asarray(
        data.xpos[simulator.evaluator.trunk_body_id], dtype=np.float64
    ).copy()
    stance_foot = 0
    stance_anchor = _site_position(simulator, data, stance_foot)
    swing_anchor = _site_position(simulator, data, 1 - stance_foot)
    fell = False
    max_guard_delta = 0.0
    target_records: list[np.ndarray] = []
    action_records: list[np.ndarray] = []
    phase_records: list[float] = []
    contact_records: list[np.ndarray] = []
    velocity_records: list[np.ndarray] = []
    force_records: list[np.ndarray] = []
    slip_records: list[np.ndarray] = []

    for tick in range(control_ticks):
        local_tick = tick % (2 * MODE_TICKS)
        desired_stance = 0 if local_tick < MODE_TICKS else 1
        if desired_stance != stance_foot:
            stance_foot = desired_stance
            stance_anchor = _site_position(simulator, data, stance_foot)
            swing_anchor = _site_position(simulator, data, 1 - stance_foot)
        swing_phase = (local_tick % MODE_TICKS) / float(MODE_TICKS)
        desired = _ik_target(
            simulator,
            data,
            stance_foot=stance_foot,
            stance_anchor=stance_anchor,
            swing_anchor=swing_anchor,
            swing_phase=swing_phase,
            previous_target=previous_target,
        )
        before = guard.previous_targets
        applied = guard.step(desired, base.CONTROL_DT_S)
        max_guard_delta = max(
            max_guard_delta,
            float(np.max(np.abs(applied[base.LEG_INDICES] - before[base.LEG_INDICES]))),
        )
        data.ctrl[:] = applied
        for _ in range(int(simulator.runtime.DECIMATION)):
            simulator.mujoco.mj_step(simulator.model, data)
            _quality_update(
                simulator,
                data,
                quality,
                physics,
                elapsed_time=float(data.time),
                previous_position=previous_position,
            )
        previous_target = applied.copy()
        action = base._inverse_decoder(applied)
        action_history = [action.astype(np.float32), action_history[0], action_history[1]]
        target_records.append(applied.copy())
        action_records.append(action.copy())
        phase_records.append(float(swing_phase))
        contact = np.asarray(simulator.evaluator._feet_contacts(data), dtype=bool)
        force, slip = simulator._quality_contact_kinematics(data)
        contact_records.append(contact)
        velocity_records.append(
            np.asarray(simulator.evaluator._sensor(data, "local_linvel"), dtype=np.float64)
        )
        force_records.append(force)
        slip_records.append(slip)
        position = np.asarray(
            data.xpos[simulator.evaluator.trunk_body_id], dtype=np.float64
        )
        rotation = np.asarray(
            data.xmat[simulator.evaluator.trunk_body_id], dtype=np.float64
        ).reshape(3, 3)
        previous_position = position.copy()
        if physics.termination_required or float(position[2]) < 0.12 or float(rotation[2, 2]) < 0.65:
            fell = True
            break

    metrics = quality.finalize()
    acceptance = gait_quality_acceptance(metrics)
    physics_payload = physics.to_dict()
    targets = np.asarray(target_records, dtype=np.float32)
    output = {
        "schema_version": 1,
        "evaluator_id": "openduckmini-exp004-h5-contact-fsm-teacher-v1",
        "evaluation_mode": "DIAGNOSTIC_CONTACT_FSM_TEACHER_NOT_QUALIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_deployment": "PROHIBITED",
        "configuration": {
            "seed": int(args.seed),
            "seconds": float(args.seconds),
            "control_dt_s": base.CONTROL_DT_S,
            "mode_ticks": MODE_TICKS,
            "mode_half_cycle_s": MODE_TICKS * base.CONTROL_DT_S,
            "step_length_m": STEP_LENGTH_M,
            "swing_lift_m": SWING_LIFT_M,
            "ik_gain_stance": IK_GAIN_STANCE,
            "ik_gain_swing": IK_GAIN_SWING,
            "ik_damping": IK_DAMPING,
            "ik_max_delta_rad": IK_MAX_DELTA_RAD,
            "target_guard": {
                "margin_rad": base.TARGET_MARGIN_RAD,
                "max_delta_per_tick_rad": base.TARGET_DELTA_RAD,
                "owner": "target_safety.FinalTargetSafetyGuard",
            },
            "physical_command": PHYSICAL_COMMAND.tolist(),
            "fsm": "left_stance_right_swing_then_right_stance_left_swing",
        },
        "provenance": {
            "teacher_script": str(Path(__file__).resolve()),
            "teacher_script_sha256": h5.sha256_file(Path(__file__).resolve()),
            "base_mpc_script": str(Path(base.__file__).resolve()),
            "base_mpc_script_sha256": h5.sha256_file(Path(base.__file__).resolve()),
            "h5_build_metadata": build_metadata,
            "policy_bank": bank.manifest(),
        },
        "run": {
            "reset_qpos_audit": reset_audit,
            "completed_ticks": len(target_records),
            "expected_ticks": control_ticks,
            "completed": len(target_records) == control_ticks and not fell,
            "fell": fell,
            "max_guard_delta_rad": max_guard_delta,
            "physics_substep_audit": physics_payload,
            "gait_quality_metrics": metrics.as_dict(),
            "gait_quality_acceptance": acceptance.as_dict(),
        },
        "adoption_allowed": False,
        "release_allowed": False,
    }
    output["next_step"] = (
        "REJECT_CONTACT_FSM_TEACHER_CONFIGURATION"
        if not bool(acceptance.passed and output["run"]["completed"])
        else "REQUIRES_STRICT_MULTI_SEED_AND_DISTILLATION_SCREEN"
    )
    arrays = {
        "target_actions": np.asarray(action_records, dtype=np.float32),
        "applied_targets": targets,
        "phase": np.asarray(phase_records, dtype=np.float32),
        "contacts": np.asarray(contact_records, dtype=np.uint8),
        "local_velocity": np.asarray(velocity_records, dtype=np.float32),
        "normal_force": np.asarray(force_records, dtype=np.float32),
        "tangential_speed": np.asarray(slip_records, dtype=np.float32),
    }
    return output, arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seconds <= 0.0:
        raise ValueError("seconds must be positive")
    output_prefix = args.output_prefix.expanduser().resolve()
    json_path = output_prefix.with_suffix(".json")
    npz_path = output_prefix.with_suffix(".npz")
    if json_path.exists() or npz_path.exists():
        raise FileExistsError(f"refusing to overwrite {json_path} or {npz_path}")
    payload, arrays = run(args)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    payload["outputs"] = {
        "npz": {"path": str(npz_path), "sha256": h5.sha256_file(npz_path)},
        "json": {"path": str(json_path)},
    }
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "npz": str(npz_path),
                "completed": payload["run"]["completed"],
                "gait_quality_passed": payload["run"]["gait_quality_acceptance"]["passed"],
                "next_step": payload["next_step"],
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
