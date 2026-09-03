"""Versioned centroidal-momentum-aware WBIK primitives for Phase 2-D28.

This module is intentionally separate from :mod:`wbik_v2`.  V2A remains the
protected prescribed-floating-base implementation.  V3 adds only the
centroidal momentum map and a deterministic minimum-norm ``H_z`` residual
solve; it does not clip actions, alter limits, or write simulator state.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


_V2_SPEC = importlib.util.spec_from_file_location(
    "exp014_d28_wbik_v2_read_only", Path(__file__).with_name("wbik_v2.py")
)
if _V2_SPEC is None or _V2_SPEC.loader is None:
    raise RuntimeError("protected D26 WBIK V2A is unavailable")
_V2 = importlib.util.module_from_spec(_V2_SPEC)
sys.modules[_V2_SPEC.name] = _V2
_V2_SPEC.loader.exec_module(_V2)


@dataclass(frozen=True)
class WBIKV3Config:
    """Fixed V3 numerical contract.

    The group coefficients are the preregistered minimum-norm metric, not
    task weights.  They are intentionally constants so a physics outcome
    cannot tune participation after the fact.
    """

    damping: float = 1.0e-4
    svd_tolerance: float = 1.0e-8
    dt: float = 0.02
    momentum_gain: float = 1.0
    momentum_damping: float = 1.0e-4
    legs_weight: float = 1.0
    waist_weight: float = 0.35
    arms_weight: float = 0.20
    wrist_hand_weight: float = 0.05


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """XYZW quaternion to a world rotation matrix."""

    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _inertia_matrix(inertias: torch.Tensor) -> torch.Tensor:
    if inertias.shape[-2:] == (3, 3):
        return inertias
    if inertias.shape[-1] != 9:
        raise ValueError(f"expected inertia shape [...,9] or [...,3,3], got {tuple(inertias.shape)}")
    return inertias.reshape(inertias.shape[:-1] + (3, 3))


def _as_batched(value: torch.Tensor, ndim_without_batch: int) -> tuple[torch.Tensor, bool]:
    if value.ndim == ndim_without_batch:
        return value.unsqueeze(0), True
    return value, False


def centroidal_momentum_matrix(
    body_jacobians: torch.Tensor,
    body_masses: torch.Tensor,
    body_com_position: torch.Tensor,
    body_origin_position: torch.Tensor,
    body_com_quaternion: torch.Tensor,
    inertias_local: torch.Tensor,
    com_position: torch.Tensor,
) -> torch.Tensor:
    """Return ``A`` such that ``H = A [v_root, dq]``.

    Isaac Lab's Jacobian convention is linear rows ``0:3`` and angular rows
    ``3:6``.  The body Jacobian is corrected from body origin to body-local
    CoM before the orbital term is accumulated.  Inertia tensors are rotated
    from the body-local CoM frame into world coordinates.
    """

    jac, squeezed = _as_batched(body_jacobians.to(torch.float64), 3)
    masses, _ = _as_batched(body_masses.to(torch.float64), 1)
    body_com, _ = _as_batched(body_com_position.to(torch.float64), 2)
    body_origin, _ = _as_batched(body_origin_position.to(torch.float64), 2)
    body_quat, _ = _as_batched(body_com_quaternion.to(torch.float64), 2)
    inertia, _ = _as_batched(_inertia_matrix(inertias_local.to(torch.float64)), 3)
    com, _ = _as_batched(com_position.to(torch.float64), 1)

    jv = jac[..., :3, :]
    jw = jac[..., 3:6, :]
    offset = body_com - body_origin
    # v_com = v_origin + omega x offset = v_origin - offset x omega.
    jv_com = jv - torch.matmul(_skew(offset), jw)
    rotation = quat_to_matrix(body_quat)
    inertia_world = rotation @ inertia @ rotation.transpose(-2, -1)
    lever = body_com - com[:, None, :]
    mass = masses[..., None, None]
    body_map = inertia_world @ jw + mass * (_skew(lever) @ jv_com)
    result = body_map.sum(dim=1)
    return result[0] if squeezed else result


def whole_body_momentum(
    body_masses: torch.Tensor,
    body_com_position: torch.Tensor,
    body_com_linear_velocity: torch.Tensor,
    body_com_angular_velocity: torch.Tensor,
    body_com_quaternion: torch.Tensor,
    inertias_local: torch.Tensor,
    com_position: torch.Tensor,
    com_velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct ``H`` and per-body contributions about the whole-body CoM."""

    masses, squeezed = _as_batched(body_masses.to(torch.float64), 1)
    pos, _ = _as_batched(body_com_position.to(torch.float64), 2)
    lin, _ = _as_batched(body_com_linear_velocity.to(torch.float64), 2)
    ang, _ = _as_batched(body_com_angular_velocity.to(torch.float64), 2)
    quat, _ = _as_batched(body_com_quaternion.to(torch.float64), 2)
    inertia, _ = _as_batched(_inertia_matrix(inertias_local.to(torch.float64)), 3)
    com, _ = _as_batched(com_position.to(torch.float64), 1)
    com_vel, _ = _as_batched(com_velocity.to(torch.float64), 1)
    rotation = quat_to_matrix(quat)
    inertia_world = rotation @ inertia @ rotation.transpose(-2, -1)
    rotational = (inertia_world @ ang.unsqueeze(-1)).squeeze(-1)
    orbital = masses[..., None] * torch.cross(pos - com[:, None, :], lin - com_vel[:, None, :], dim=-1)
    contributions = rotational + orbital
    total = contributions.sum(dim=1)
    if squeezed:
        return total[0], contributions[0]
    return total, contributions


def joint_group(joint_name: str) -> str:
    name = joint_name.lower()
    if "wrist" in name or any(token in name for token in ("_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_")):
        return "left wrist/hand" if name.startswith("left_") else "right wrist/hand"
    if "shoulder" in name or "elbow" in name:
        return "left arm" if name.startswith("left_") else "right arm"
    if "torso" in name or "waist" in name:
        return "waist"
    return "left leg" if name.startswith("left_") else "right leg"


def joint_participation_weights(joint_names: list[str], config: WBIKV3Config = WBIKV3Config()) -> torch.Tensor:
    values = []
    for name in joint_names:
        group = joint_group(name)
        values.append(
            config.legs_weight if group in ("left leg", "right leg") else
            config.waist_weight if group == "waist" else
            config.arms_weight if group in ("left arm", "right arm") else
            config.wrist_hand_weight
        )
    return torch.tensor(values, dtype=torch.float64)


def weighted_minimum_norm(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    weights: torch.Tensor,
    damping: float = 1.0e-4,
) -> torch.Tensor:
    """Solve ``min dq' W dq`` subject to the linear residual in least squares."""

    j = jacobian.to(torch.float64)
    r = residual.to(torch.float64)
    w = weights.to(torch.float64).clamp_min(1.0e-12)
    winv = torch.diag(1.0 / w.square())
    regularized = j @ winv @ j.transpose(-2, -1) + damping * torch.eye(j.shape[-2], dtype=j.dtype, device=j.device)
    return winv @ j.transpose(-2, -1) @ torch.linalg.solve(regularized, r)


def momentum_joint_residual(
    momentum_matrix: torch.Tensor,
    prescribed_root_twist: torch.Tensor,
    target_hz: torch.Tensor | float,
    joint_weights: torch.Tensor,
    config: WBIKV3Config = WBIKV3Config(),
) -> dict[str, torch.Tensor]:
    """Solve the fixed ``H_z_target - A_root*v_root`` residual by ``A_joint``."""

    matrix = momentum_matrix.to(torch.float64)
    root_twist = prescribed_root_twist.to(torch.float64)
    target = torch.as_tensor(target_hz, dtype=torch.float64, device=matrix.device).reshape(())
    residual = target - matrix[2, :6] @ root_twist
    delta = weighted_minimum_norm(matrix[2:3, 6:], residual.reshape(1), joint_weights, config.momentum_damping)
    return {
        "hz_residual": residual,
        "joint_delta": config.momentum_gain * delta,
        "root_term": matrix[2, :6] @ root_twist,
        "target_hz": target,
    }


def minimum_jerk(start: torch.Tensor | float, end: torch.Tensor | float, progress: torch.Tensor | float) -> torch.Tensor:
    p = torch.as_tensor(progress, dtype=torch.float64)
    u = p.clamp(0.0, 1.0)
    scalar = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return torch.as_tensor(start, dtype=torch.float64) + scalar * (torch.as_tensor(end, dtype=torch.float64) - torch.as_tensor(start, dtype=torch.float64))


__all__ = [
    "WBIKV3Config",
    "centroidal_momentum_matrix",
    "whole_body_momentum",
    "joint_group",
    "joint_participation_weights",
    "weighted_minimum_norm",
    "momentum_joint_residual",
    "minimum_jerk",
    "quat_to_matrix",
]
