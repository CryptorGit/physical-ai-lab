"""Causal contact-conditioned tangential-relative-slip reward."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp
from isaaclab.managers import ManagerTermBase

FOOT_LABELS = ("fl", "fr", "rl", "rr")
FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def robust_score(speed: torch.Tensor, v_free: float = 0.20, v_scale: float = 0.30) -> torch.Tensor:
    x = torch.relu((speed - v_free) / v_scale)
    rho = torch.where(x <= 1.0, 0.5 * x.square(), x - 0.5)
    return rho.clamp_max(5.0)


def aggregate_foot_scores(
    foot_speed: torch.Tensor,
    normal_force: torch.Tensor,
    stable: torch.Tensor,
    f_ref: float = 100.0,
) -> torch.Tensor:
    weights = (normal_force / f_ref).clamp(0.0, 1.0) * stable
    numerator = (weights * robust_score(foot_speed)).sum(dim=1)
    return numerator / weights.sum(dim=1).clamp_min(1.0e-12)


def run_unit_tests() -> dict:
    cases = {}

    def record(name, actual, expected, tolerance=1.0e-7):
        error = float(torch.as_tensor(actual).sub(torch.as_tensor(expected)).abs().max())
        cases[name] = {
            "actual": torch.as_tensor(actual).tolist(),
            "expected": torch.as_tensor(expected).tolist(),
            "max_abs_error": error,
            "pass": error <= tolerance,
        }

    speed = torch.tensor([[0.0, 0.20, 0.35, 0.50]])
    force = torch.tensor([[100.0, 100.0, 100.0, 100.0]])
    stable = torch.tensor([[False, True, True, True]])
    record("free_margin", robust_score(torch.tensor([0.20])), [0.0])
    record("quadratic_huber", robust_score(torch.tensor([0.35])), [0.125])
    record("huber_knee", robust_score(torch.tensor([0.50])), [0.5])
    record("outlier_cap", robust_score(torch.tensor([10.0])), [5.0])
    record("causal_mask", aggregate_foot_scores(speed, force, stable), [(0.0 + 0.125 + 0.5) / 3.0])
    record(
        "force_weight",
        aggregate_foot_scores(
            torch.tensor([[0.50, 0.50]]),
            torch.tensor([[100.0, 50.0]]),
            torch.tensor([[True, True]]),
        ),
        [0.5],
    )
    record(
        "no_contact_zero",
        aggregate_foot_scores(
            torch.ones(1, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 4, dtype=torch.bool),
        ),
        [0.0],
    )
    age = torch.zeros(1, dtype=torch.long)
    activation = []
    for contact in (True, True, True, False):
        mask = torch.tensor([contact])
        age = torch.where(mask, age + 1, torch.zeros_like(age))
        activation.append(bool((mask & (age >= 3)).item()))
    cases["contact_age"] = {
        "actual": activation,
        "expected": [False, False, True, False],
        "pass": activation == [False, False, True, False],
    }
    finite = all(torch.isfinite(robust_score(torch.linspace(0, 100, 1000))))
    return {
        "formula": "causal stable-contact force-weighted Huber score",
        "cases": cases,
        "all_pass": all(item["pass"] for item in cases.values()) and bool(finite),
        "nan_inf": 0 if finite else 1,
        "legacy_anchor_displacement_used": False,
        "foot_link_origin_velocity_used": False,
    }


class go2_contact_tangential_slip(ManagerTermBase):
    """Return ``-L_slip`` and expose detached telemetry for audits."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.body_ids = [int(self.robot.find_bodies(name)[0][0]) for name in FOOT_NAMES]
        self.sensor = env.scene.sensors["stage11_contact"]
        self.contact_age = torch.zeros(
            env.num_envs, 4, dtype=torch.long, device=env.device
        )
        self.last_raw_score = torch.zeros(env.num_envs, device=env.device)
        self.last_foot_speed = torch.zeros(env.num_envs, 4, device=env.device)
        self.last_normal_force = torch.zeros(env.num_envs, 4, device=env.device)
        self.last_stable = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        self.last_friction_utilization = torch.zeros(env.num_envs, 4, device=env.device)
        self.missing_telemetry = 0
        sensor_names = tuple(self.sensor.body_names)
        self.association_errors = 0 if sensor_names == FOOT_NAMES else 1
        if self.association_errors:
            raise RuntimeError(
                f"contact body order mismatch: expected {FOOT_NAMES}, got {sensor_names}"
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.contact_age[ids] = 0
        self.last_raw_score[ids] = 0.0

    def _all_feet(self):
        sensor = self.sensor
        fn_raw, point_raw, normal_raw, _, count_raw, start_raw = (
            sensor.contact_view.get_contact_data(dt=sensor._sim_physics_dt)
        )
        fn_flat = wp.to_torch(fn_raw).reshape(-1)
        points_flat = wp.to_torch(point_raw).reshape(-1, 3)
        normals_flat = wp.to_torch(normal_raw).reshape(-1, 3)
        counts = wp.to_torch(count_raw).reshape(self.num_envs, 4, -1)[:, :, 0].long()
        starts = wp.to_torch(start_raw).reshape(self.num_envs, 4, -1)[:, :, 0].long()
        width = 16
        offsets = torch.arange(width, device=self.device)
        indices = (starts[..., None] + offsets).clamp(0, max(0, len(points_flat) - 1))
        mask = offsets[None, None] < counts[..., None]
        points = points_flat[indices]
        normals = normals_flat[indices]
        normal_force = fn_flat[indices] * mask
        unit = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        position = self.robot.data.body_pos_w.torch[:, self.body_ids]
        linear = self.robot.data.body_lin_vel_w.torch[:, self.body_ids]
        angular = self.robot.data.body_ang_vel_w.torch[:, self.body_ids]
        radius = points - position[:, :, None]
        surface = linear[:, :, None] + torch.linalg.cross(
            angular[:, :, None].expand_as(radius), radius
        )
        tangent = surface - (surface * unit).sum(-1, keepdim=True) * unit
        speed = tangent.norm(dim=-1)
        force_sum = normal_force.sum(dim=-1)
        weighted_speed = (speed * normal_force).sum(dim=-1) / force_sum.clamp_min(1.0e-12)
        return weighted_speed, force_sum

    def diagnostic_friction_utilization(self) -> torch.Tensor:
        """Query friction only for sparse preflight/formal diagnostics."""
        friction_raw, _, count_raw, start_raw = self.sensor.contact_view.get_friction_data(
            dt=self.sensor._sim_physics_dt
        )
        friction_flat = wp.to_torch(friction_raw).reshape(-1, 3)
        counts = wp.to_torch(count_raw).reshape(self.num_envs, 4, -1)[:, :, 0].long()
        starts = wp.to_torch(start_raw).reshape(self.num_envs, 4, -1)[:, :, 0].long()
        offsets = torch.arange(16, device=self.device)
        indices = (starts[..., None] + offsets).clamp(0, max(0, len(friction_flat) - 1))
        mask = offsets[None, None] < counts[..., None]
        force = (friction_flat[indices] * mask[..., None]).sum(dim=-2).norm(dim=-1)
        utilization = force / (0.6 * self.last_normal_force).clamp_min(1.0e-12)
        self.last_friction_utilization = utilization.detach()
        return utilization

    def __call__(self, env) -> torch.Tensor:
        foot_speed, normal_force = self._all_feet()
        contact = normal_force > 5.0
        self.contact_age = torch.where(contact, self.contact_age + 1, torch.zeros_like(self.contact_age))
        stable = contact & (self.contact_age >= 3)
        score = aggregate_foot_scores(foot_speed, normal_force, stable)
        if not torch.isfinite(score).all():
            raise FloatingPointError("non-finite tangential-slip reward")
        self.last_raw_score = score.detach()
        self.last_foot_speed = foot_speed.detach()
        self.last_normal_force = normal_force.detach()
        self.last_stable = stable.detach()
        # Friction is diagnostic-only and intentionally excluded from the online
        # reward hot path.  Formal evaluators query RigidContactView separately.
        self.last_friction_utilization.zero_()
        return -score
