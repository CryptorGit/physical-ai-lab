"""W2 dual-command observation and reward functions.

The command term exposes the calibrated actor command through the normal
CommandManager API.  Reward targets deliberately bypass that API and read the
physical command buffer so positive-yaw calibration never changes reward
semantics.
"""
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def actor_velocity_command(env, command_name: str = "base_velocity") -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return term.actor_command_b


def track_lin_vel_xy_physical(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target = env.command_manager.get_term(command_name).physical_command_b[:, :2]
    error = torch.sum(torch.square(target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_physical(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target = env.command_manager.get_term(command_name).physical_command_b[:, 2]
    error = torch.square(target - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)
