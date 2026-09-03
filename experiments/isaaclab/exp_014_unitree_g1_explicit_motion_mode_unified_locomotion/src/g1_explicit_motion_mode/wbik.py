"""Deterministic, kinematic-only hierarchical whole-body IK for D26.

This module intentionally has no dynamics or mass-matrix dependency.  It is a
small, auditable interface used by the offline planner; physics execution is a
separate, later phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class WBIKConfig:
    damping: float = 1.0e-4
    svd_tolerance: float = 1.0e-8
    max_active_set_iterations: int = 37
    dt: float = 0.02


def deterministic_pseudoinverse(jacobian: torch.Tensor, damping: float = 1.0e-4, tolerance: float = 1.0e-8) -> torch.Tensor:
    """SVD pseudoinverse with fixed ordering and tolerance."""
    if jacobian.numel() == 0:
        return torch.zeros((jacobian.shape[-1], jacobian.shape[-2]), dtype=jacobian.dtype, device=jacobian.device)
    u, s, vh = torch.linalg.svd(jacobian, full_matrices=False)
    gain = torch.where(s > tolerance, s / (s.square() + damping * damping), torch.zeros_like(s))
    return (vh.transpose(-2, -1) * gain.unsqueeze(-2)) @ u.transpose(-2, -1)


def hierarchical_dls(jacobians: Sequence[torch.Tensor], targets: Sequence[torch.Tensor], config: WBIKConfig = WBIKConfig(), dq_previous: torch.Tensor | None = None) -> tuple[torch.Tensor, list[float], list[float]]:
    """Sequential null-space projection, in the order supplied by caller."""
    if not jacobians:
        raise ValueError("at least one task is required")
    n = int(jacobians[0].shape[-1])
    dq = torch.zeros(n, dtype=jacobians[0].dtype, device=jacobians[0].device) if dq_previous is None else dq_previous.clone()
    null = torch.eye(n, dtype=dq.dtype, device=dq.device)
    residuals: list[float] = []
    ranks: list[float] = []
    for jac, target in zip(jacobians, targets):
        projected = jac @ null
        pinv = deterministic_pseudoinverse(projected, config.damping, config.svd_tolerance)
        residual = target - jac @ dq
        dq = dq + pinv @ residual
        null = null - pinv @ projected
        residuals.append(float(residual.norm().detach().cpu()))
        ranks.append(float((torch.linalg.svdvals(projected) > config.svd_tolerance).sum().detach().cpu()) if projected.numel() else 0.0)
    return dq, residuals, ranks


def active_set_hierarchical(
    jacobians: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    q: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    dq_max: torch.Tensor,
    config: WBIKConfig = WBIKConfig(),
    solve_callback: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict:
    """Solve and explicitly freeze violating joints rather than hiding them in a clip.

    ``solve_callback`` may provide a task solve over the free columns.  The
    default path uses the deterministic hierarchy above and is sufficient for
    the D26 linearized feasibility audit.
    """
    n = q.numel()
    fixed = torch.zeros(n, dtype=torch.bool, device=q.device)
    dq = torch.zeros_like(q)
    diagnostics = []
    for iteration in range(config.max_active_set_iterations):
        if solve_callback is not None:
            proposed = solve_callback(fixed)
        else:
            local = [j[:, ~fixed] for j in jacobians]
            local_dq, residuals, ranks = hierarchical_dls(local, targets, config)
            proposed = torch.zeros_like(q)
            proposed[~fixed] = local_dq
            diagnostics = [float(x) for x in residuals]
        proposed = torch.where(fixed, dq, proposed)
        lo = (q_min - q) / config.dt
        hi = (q_max - q) / config.dt
        proposed = torch.minimum(torch.maximum(proposed, -dq_max), dq_max)
        violation = ((q + proposed * config.dt) < q_min - 1.0e-9) | ((q + proposed * config.dt) > q_max + 1.0e-9)
        new = violation & ~fixed
        if not bool(new.any()):
            q_des = q + proposed * config.dt
            return {"status": "PASS", "q_des": q_des, "dq_des": proposed, "fixed_joints": fixed.nonzero().flatten().tolist(), "iterations": iteration + 1, "residuals": diagnostics, "constraint_violation": 0}
        proposed = torch.minimum(torch.maximum(proposed, lo), hi)
        dq = torch.where(new, proposed, dq)
        fixed |= new
    return {"status": "IK_CONSTRAINT_INFEASIBLE", "q_des": q + dq * config.dt, "dq_des": dq, "fixed_joints": fixed.nonzero().flatten().tolist(), "iterations": config.max_active_set_iterations, "residuals": diagnostics, "constraint_violation": int(fixed.sum())}


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """XYZW quaternion to rotation matrix."""
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1).reshape(q.shape[:-1] + (3, 3))


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Numerically finite SO(3) logarithm (rotation vector)."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    vee = torch.stack((rotation[..., 2, 1] - rotation[..., 1, 2], rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 1, 0] - rotation[..., 0, 1]), dim=-1) * 0.5
    scale = torch.where(theta < 1.0e-5, 1.0 + theta.square() / 6.0, theta / torch.sin(theta).clamp_min(1.0e-6))
    out = vee * scale.unsqueeze(-1)
    # A finite axis fallback near pi, where the skew part vanishes.
    near_pi = theta > 3.13
    if bool(near_pi.any()):
        diag = torch.diagonal(rotation, dim1=-2, dim2=-1)
        axis = torch.sqrt(((diag + 1.0).clamp_min(0.0)) * 0.5)
        out = torch.where(near_pi.unsqueeze(-1), axis * theta.unsqueeze(-1), out)
    return torch.nan_to_num(out)


def com_jacobian(body_jacobians: torch.Tensor, masses: torch.Tensor, body_com_pos: torch.Tensor, body_origin_pos: torch.Tensor) -> torch.Tensor:
    """Mass-weighted CoM Jacobian with local CoM point correction.

    IsaacLab's body Jacobian convention is linear rows 0:3 and angular rows
    3:6.  The correction is ``Jv + Jw x r`` for the body-local CoM offset.
    """
    jv = body_jacobians[..., :3, :]
    jw = body_jacobians[..., 3:6, :]
    r = (body_com_pos - body_origin_pos).unsqueeze(-1)
    correction = torch.cross(jw.transpose(-2, -1), r.transpose(-2, -1), dim=-1).transpose(-2, -1)
    point = jv + correction
    weight = masses / masses.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return (point * weight[..., None, None]).sum(dim=-3)


def action_to_q(action: torch.Tensor, default_q: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None = None) -> torch.Tensor:
    return default_q + action * scale + (offset if offset is not None else 0.0)


def q_to_action(q: torch.Tensor, default_q: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None = None) -> torch.Tensor:
    return (q - default_q - (offset if offset is not None else 0.0)) / scale.clamp_min(1.0e-12)
