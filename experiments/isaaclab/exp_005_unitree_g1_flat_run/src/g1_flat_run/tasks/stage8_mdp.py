"""Stage 8 quality-gated excess-slip shaping from the Stage 6 policy."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from .stage3_mdp import FocusedRunVelocityCommand
from .stage6_mdp import Stage6VelocityCommand


class Stage8VelocityCommand(Stage6VelocityCommand):
    """Concentrate on 4.40--4.50 m/s without sampling above 4.50 m/s."""

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super(FocusedRunVelocityCommand, self)._resample_command(env_ids)
        count = len(env_ids)
        if count == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        draw = torch.rand(count, device=self.device)
        focused = torch.empty(count, device=self.device).uniform_(4.40, 4.50)
        support = torch.empty(count, device=self.device).uniform_(4.30, 4.50)
        self.vel_command_b[ids, 0] = torch.where(draw < 0.80, focused, support)


def quality_saturated_track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    acceptable_error_mps: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Use the official kernel and stop its marginal gain inside a tight band."""
    asset = env.scene[asset_cfg.name]
    velocity_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3]
    )
    error_sq = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - velocity_yaw[:, :2]),
        dim=1,
    )
    minimum_error_sq = torch.full_like(error_sq, acceptable_error_mps**2)
    return torch.exp(-torch.maximum(error_sq, minimum_error_sq) / std**2)


def quality_gated_excess_slip(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_command_speed_mps: float = 4.40,
    max_tracking_error_mps: float = 0.25,
    slip_threshold_mps: float = 0.50,
) -> torch.Tensor:
    """Penalize per-foot slip excess only after acceptable tracking is reached.

    The contact and horizontal-foot-speed definitions exactly match the
    official ``feet_slide`` term.  Squared hinge excess is evaluated per foot,
    so one slipping side cannot be hidden by averaging with the other side.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .amax(dim=1)
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    foot_speed = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    command = env.command_manager.get_command(command_name)
    velocity_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3]
    )
    tracking_error = torch.linalg.vector_norm(command[:, :2] - velocity_yaw[:, :2], dim=1)
    quality_gate = (command[:, 0] >= min_command_speed_mps) & (
        tracking_error <= max_tracking_error_mps
    )
    normalized_excess = torch.relu(foot_speed - slip_threshold_mps) / slip_threshold_mps
    per_foot_cost = normalized_excess.square() * contacts
    return per_foot_cost.sum(dim=1) * quality_gate
