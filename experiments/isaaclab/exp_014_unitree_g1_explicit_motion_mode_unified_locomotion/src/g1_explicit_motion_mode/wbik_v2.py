"""Versioned prescribed-floating-base WBIK for Phase 2-D26V.

The D26 WBIK V1 module is intentionally left untouched.  V2 keeps its
37-joint/action interface and deterministic SVD/null-space/active-set
contract, while treating the six-dimensional root twist as an external,
prescribed reference.  Root columns therefore contribute to foot and CoM
twists but are never returned as an action variable.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


_V1_SPEC = importlib.util.spec_from_file_location(
    "exp014_d26v_wbik_v1_primitives", Path(__file__).with_name("wbik.py")
)
if _V1_SPEC is None or _V1_SPEC.loader is None:
    raise RuntimeError("D26 WBIK V1 primitives are unavailable")
_V1 = importlib.util.module_from_spec(_V1_SPEC)
sys.modules[_V1_SPEC.name] = _V1
_V1_SPEC.loader.exec_module(_V1)


@dataclass(frozen=True)
class WBIKV2Config:
    damping: float = 1.0e-4
    svd_tolerance: float = 1.0e-8
    max_active_set_iterations: int = 37
    dt: float = 0.02
    velocity_ratio_limit: float = 0.80
    action_bound: float = 1.0
    nominal_gain: float = 0.02
    action_rate_gain: float = 0.01


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    return _V1.quat_to_matrix(quat)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    return _V1.so3_log(rotation)


def com_jacobian(
    body_jacobians: torch.Tensor,
    masses: torch.Tensor,
    body_com_pos: torch.Tensor,
    body_origin_pos: torch.Tensor,
) -> torch.Tensor:
    """Reuse the protected D26 point-corrected CoM Jacobian implementation."""
    return _V1.com_jacobian(body_jacobians, masses, body_com_pos, body_origin_pos)


def action_to_q(action: torch.Tensor, default_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _V1.action_to_q(action, default_q, scale)


def q_to_action(q: torch.Tensor, default_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _V1.q_to_action(q, default_q, scale)


def _pinv(jacobian: torch.Tensor, config: WBIKV2Config) -> torch.Tensor:
    return _V1.deterministic_pseudoinverse(jacobian, config.damping, config.svd_tolerance)


def _hierarchical(
    jacobians: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    config: WBIKV2Config,
) -> tuple[torch.Tensor, list[float], list[int]]:
    """Deterministic sequential null-space projection over free joints.

    The null-space is carried as an orthonormal basis rather than as a
    damped projector matrix.  A damped inverse is still used for each task,
    but repeatedly multiplying damped projectors can create tiny singular
    directions; feeding those directions to the next inverse amplifies them
    catastrophically when the regularization tasks are reached.  Rebuilding
    the basis from the deterministic SVD keeps the same priority contract
    while making the active-set solve numerically bounded.
    """
    if not jacobians:
        raise ValueError("at least one task is required")
    n = int(jacobians[0].shape[-1])
    dq = torch.zeros(n, dtype=jacobians[0].dtype, device=jacobians[0].device)
    null_basis = torch.eye(n, dtype=dq.dtype, device=dq.device)
    residuals: list[float] = []
    ranks: list[int] = []
    for jac, target in zip(jacobians, targets):
        projected = jac @ null_basis
        pinv = _pinv(projected, config)
        residual = target - jac @ dq
        dq = dq + null_basis @ (pinv @ residual)
        if projected.numel():
            _, singular_values, vh = torch.linalg.svd(projected, full_matrices=True)
            rank = int((singular_values > config.svd_tolerance).sum().detach().cpu())
            if rank < null_basis.shape[1]:
                null_basis = null_basis @ vh[rank:].transpose(-2, -1)
            else:
                null_basis = null_basis[:, :0]
        else:
            rank = 0
        ranks.append(rank)
        residuals.append(float(torch.linalg.vector_norm(residual).detach().cpu()))
    return dq, residuals, ranks


def _pose_twist(
    current_position: torch.Tensor,
    current_rotation: torch.Tensor,
    target_position: torch.Tensor,
    target_rotation: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_error = target_position - current_position
    rotation_error = so3_log(target_rotation @ current_rotation.transpose(-2, -1))
    return torch.cat((position_error / dt, rotation_error / dt)), position_error, rotation_error


def _task_projection(
    jacobian: torch.Tensor,
    root_twist: torch.Tensor,
    target_twist: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root_jac = jacobian[..., :6]
    joint_jac = jacobian[..., 6:]
    root_contribution = root_jac @ root_twist
    return joint_jac, target_twist - root_contribution, root_contribution


def _active_set_solve(
    jacobians: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    q: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    config: WBIKV2Config,
) -> dict:
    """Freeze only position-limit violators; never hide velocity/action violations."""
    n = int(q.numel())
    fixed = torch.zeros(n, dtype=torch.bool, device=q.device)
    fixed_values = q.clone()
    diagnostics: list[dict] = []
    dq = torch.zeros_like(q)
    residuals: list[float] = []
    ranks: list[int] = []
    for iteration in range(config.max_active_set_iterations):
        free = ~fixed
        local_jacobians = [jac[..., free] for jac in jacobians]
        local_dq, residuals, ranks = _hierarchical(local_jacobians, targets, config)
        dq = torch.zeros_like(q)
        if bool(fixed.any()):
            dq[fixed] = (fixed_values[fixed] - q[fixed]) / config.dt
        dq[free] = local_dq
        q_trial = q + dq * config.dt
        violation = ((q_trial < q_min - 1.0e-9) | (q_trial > q_max + 1.0e-9)) & free
        if not bool(violation.any()):
            q_des = q_trial
            diagnostics.append({"iteration": iteration + 1, "new_fixed": []})
            return {
                "status": "PASS",
                "q_des": q_des,
                "dq_des": dq,
                "fixed_joints": fixed.nonzero().flatten().tolist(),
                "iterations": iteration + 1,
                "residuals": residuals,
                "ranks": ranks,
                "diagnostics": diagnostics,
                "position_limit_violation": 0,
            }
        for index in violation.nonzero().flatten().tolist():
            fixed_values[index] = torch.minimum(torch.maximum(q_trial[index], q_min[index]), q_max[index])
        fixed |= violation
        diagnostics.append({"iteration": iteration + 1, "new_fixed": violation.nonzero().flatten().tolist()})
        if bool(fixed.all()):
            break
    q_des = torch.where(fixed, fixed_values, q + dq * config.dt)
    return {
        "status": "ACTIVE_SET_NONCONVERGENCE",
        "q_des": q_des,
        "dq_des": (q_des - q) / config.dt,
        "fixed_joints": fixed.nonzero().flatten().tolist(),
        "iterations": config.max_active_set_iterations,
        "residuals": residuals,
        "ranks": ranks,
        "diagnostics": diagnostics,
        "position_limit_violation": int(bool(fixed.any())),
    }


def solve_prescribed_floating_base(
    *,
    root_pose: torch.Tensor,
    root_velocity: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    body_position: torch.Tensor,
    body_quaternion: torch.Tensor,
    body_jacobians: torch.Tensor,
    body_com_position: torch.Tensor,
    body_masses: torch.Tensor,
    com_position: torch.Tensor,
    reference: dict[str, torch.Tensor],
    stance_body_index: int,
    swing_body_index: int,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    velocity_limits: torch.Tensor,
    default_q: torch.Tensor,
    action_scale: torch.Tensor,
    config: WBIKV2Config = WBIKV2Config(),
) -> dict:
    """Solve one D26V kinematic control step with an external root reference."""
    dtype = torch.float64
    q = joint_position.to(dtype)
    root_pose = root_pose.to(dtype)
    root_velocity = root_velocity.to(dtype)
    body_position = body_position.to(dtype)
    body_quaternion = body_quaternion.to(dtype)
    body_jacobians = body_jacobians.to(dtype)
    body_com_position = body_com_position.to(dtype)
    body_masses = body_masses.to(dtype)
    com_position = com_position.to(dtype)
    root_ref_pose = reference["root_pose"].to(dtype)
    root_ref_velocity = reference["root_velocity"].to(dtype)
    root_twist = root_ref_velocity
    root_rotation = quat_to_matrix(root_pose[3:])
    root_ref_rotation = quat_to_matrix(root_ref_pose[3:])

    stance_twist, stance_position_error, stance_rotation_error = _pose_twist(
        body_position[stance_body_index],
        quat_to_matrix(body_quaternion[stance_body_index]),
        reference["stance_position"].to(dtype),
        reference["stance_rotation"].to(dtype),
        config.dt,
    )
    swing_twist, swing_position_error, swing_rotation_error = _pose_twist(
        body_position[swing_body_index],
        quat_to_matrix(body_quaternion[swing_body_index]),
        reference["swing_position"].to(dtype),
        reference["swing_rotation"].to(dtype),
        config.dt,
    )
    stance_j, stance_target, stance_root_contribution = _task_projection(
        body_jacobians[stance_body_index], root_twist, stance_twist
    )
    swing_j, swing_target, swing_root_contribution = _task_projection(
        body_jacobians[swing_body_index], root_twist, swing_twist
    )

    jcom_full = com_jacobian(body_jacobians, body_masses, body_com_position, body_position)
    jcom_joint = jcom_full[..., 6:]
    com_root_contribution = jcom_full[..., :6] @ root_twist
    com_position_error = reference["com_position"].to(dtype) - com_position
    com_target_velocity = reference["com_velocity"].to(dtype) + com_position_error / config.dt
    com_target = com_target_velocity - com_root_contribution

    # Priority 2 uses the registered torso body index and two deterministic
    # regularizers.  Pelvis/root world translation is deliberately not added
    # as a second task: prescribed root pose is the sole world translation
    # contract, and pelvis orientation is audited below.
    torso_index = 4
    torso_j = body_jacobians[torso_index, 3:6, 6:]
    torso_root = body_jacobians[torso_index, 3:6, :6] @ root_twist
    torso_rotation = quat_to_matrix(body_quaternion[torso_index])
    torso_target_rotation = reference.get("torso_rotation", torso_rotation).to(dtype)
    torso_error = so3_log(torso_target_rotation @ torso_rotation.transpose(-2, -1))
    torso_target = torso_error / config.dt - torso_root
    nominal_target = config.nominal_gain * (reference.get("nominal_q", q).to(dtype) - q) / config.dt
    action_rate_target = -config.action_rate_gain * torch.zeros_like(q)

    solved = _active_set_solve(
        [stance_j, jcom_joint, swing_j, torso_j, torch.eye(37, dtype=dtype), torch.eye(37, dtype=dtype)],
        [stance_target, com_target, swing_target, torso_target, nominal_target, action_rate_target],
        q,
        q_min.to(dtype),
        q_max.to(dtype),
        config,
    )
    dq = solved["dq_des"]
    q_des = solved["q_des"]
    action = q_to_action(q_des, default_q.to(dtype), action_scale.to(dtype))
    velocity_ratio = dq.abs() / velocity_limits.to(dtype).abs().clamp_min(1.0e-12)
    velocity_violation = bool((velocity_ratio > config.velocity_ratio_limit + 1.0e-9).any())
    action_violation = bool((action.abs() > config.action_bound + 1.0e-9).any())
    finite = bool(torch.isfinite(q_des).all() and torch.isfinite(dq).all() and torch.isfinite(action).all())

    # One-step predicted task errors are reported in physical units.  This is
    # also the audit used by the offline rollout; no clipping is applied.
    stance_pred = stance_j @ dq + stance_root_contribution
    swing_pred = swing_j @ dq + swing_root_contribution
    com_pred = jcom_full @ torch.cat((root_twist, dq))
    stance_position_after = stance_position_error - stance_pred[:3] * config.dt
    swing_position_after = swing_position_error - swing_pred[:3] * config.dt
    com_after = com_position_error - com_pred * config.dt
    root_position_error = root_ref_pose[:3] - root_pose[:3]
    root_orientation_error = so3_log(root_ref_rotation @ root_rotation.transpose(-2, -1))
    status = solved["status"]
    if status == "PASS" and not finite:
        status = "NUMERICAL_FAILURE"
    elif status == "PASS" and velocity_violation:
        status = "JOINT_VELOCITY_INFEASIBLE"
    elif status == "PASS" and action_violation:
        status = "ACTION_BOUND_INFEASIBLE"
    return {
        "status": status,
        "q_des": q_des,
        "dq_des": dq,
        "normalized_action": action,
        "root_contribution": {
            "root_twist": root_twist,
            "stance_foot_twist": stance_root_contribution,
            "swing_foot_twist": swing_root_contribution,
            "com_velocity": com_root_contribution,
        },
        "task_errors": {
            "stance_position_m": torch.linalg.vector_norm(stance_position_after),
            "stance_rotation_rad": torch.linalg.vector_norm(stance_rotation_error - stance_pred[3:] * config.dt),
            "swing_position_m": torch.linalg.vector_norm(swing_position_after),
            "swing_rotation_rad": torch.linalg.vector_norm(swing_rotation_error - swing_pred[3:] * config.dt),
            "com_horizontal_m": torch.linalg.vector_norm(com_after[:2]),
            "com_xyz_m": torch.linalg.vector_norm(com_after),
            "root_reference_position_m": torch.linalg.vector_norm(root_position_error),
            "root_reference_orientation_rad": torch.linalg.vector_norm(root_orientation_error),
            "pelvis_roll_pitch_rad": torch.linalg.vector_norm(root_orientation_error[:2]),
            "torso_orientation_rad": torch.linalg.vector_norm(torso_error),
        },
        "constraint_margins": {
            "joint_position_lower_rad": q_des - q_min.to(dtype),
            "joint_position_upper_rad": q_max.to(dtype) - q_des,
            "planned_joint_velocity_ratio_max": velocity_ratio.max(),
            "planned_joint_velocity_ratio": velocity_ratio,
            "action_lower": config.action_bound + action,
            "action_upper": config.action_bound - action,
            "action_min_margin": config.action_bound - action.abs().max(),
            "joint_limit_violation": int(bool(((q_des < q_min.to(dtype) - 1.0e-9) | (q_des > q_max.to(dtype) + 1.0e-9)).any())),
            "action_bound_violation": int(action_violation),
            "joint_velocity_violation": int(velocity_violation),
        },
        "solver_diagnostics": {
            "iterations": solved["iterations"],
            "fixed_joints": solved["fixed_joints"],
            "residuals": solved["residuals"],
            "ranks": solved["ranks"],
            "active_set": solved["diagnostics"],
            "finite": finite,
            "root_columns_used": [0, 1, 2, 3, 4, 5],
            "joint_columns_solved": list(range(6, 43)),
        },
    }


__all__ = [
    "WBIKV2Config",
    "solve_prescribed_floating_base",
    "quat_to_matrix",
    "so3_log",
    "com_jacobian",
    "action_to_q",
    "q_to_action",
]
