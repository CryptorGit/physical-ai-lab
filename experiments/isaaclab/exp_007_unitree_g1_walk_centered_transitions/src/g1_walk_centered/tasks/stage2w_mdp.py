"""Steady-WALK-only command and reward terms for Stage 2W."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def minimum_jerk(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


class IndependentWalkCommand(UniformVelocityCommand):
    """Sample only 0.6--1.2 m/s WALK and ramp once after reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ramp_duration_s = float(getattr(cfg, "ramp_duration_s", 1.5))
        self.k_heading = float(getattr(cfg, "k_heading", 0.8))
        self.k_yaw_rate = float(getattr(cfg, "k_yaw_rate", 0.10))
        self.yaw_limit = float(getattr(cfg, "yaw_rate_limit", 0.30))
        self.low_pass_alpha = float(getattr(cfg, "low_pass_alpha", 0.15))
        self.slew_limit = float(getattr(cfg, "yaw_rate_slew_limit", 0.01))
        self.heading_mode = str(getattr(cfg, "heading_mode", "FixedTarget"))
        self.target_speed = torch.full((self.num_envs,), 0.6, device=self.device)
        self.target_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.path_origin_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.filtered_yaw_command = torch.zeros(self.num_envs, device=self.device)
        self.raw_yaw_command = torch.zeros(self.num_envs, device=self.device)
        self.heading_controller_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        speeds = torch.tensor((0.6, 0.8, 1.0, 1.2), device=self.device)
        self.target_speed[ids] = speeds[torch.randint(0, 4, (len(ids),), device=self.device)]
        self.target_heading_w[ids] = self.robot.data.heading_w.torch[ids]
        self.path_origin_xy[ids] = self.robot.data.root_pos_w.torch[ids, :2]
        self.filtered_yaw_command[ids] = 0.0
        self.raw_yaw_command[ids] = 0.0
        self.heading_controller_saturated[ids] = False
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = False

    def _update_command(self):
        elapsed = self._env.episode_length_buf.float() * self._env.step_dt
        vx = self.target_speed * minimum_jerk(elapsed / self.ramp_duration_s)
        error = torch.atan2(
            torch.sin(self.target_heading_w - self.robot.data.heading_w.torch),
            torch.cos(self.target_heading_w - self.robot.data.heading_w.torch),
        )
        unclamped = self.k_heading * error - self.k_yaw_rate * self.robot.data.root_ang_vel_b.torch[:, 2]
        if self.heading_mode == "ZeroYaw":
            self.raw_yaw_command.zero_()
            self.filtered_yaw_command.zero_()
            self.heading_controller_saturated.zero_()
        else:
            self.raw_yaw_command[:] = unclamped.clamp(-self.yaw_limit, self.yaw_limit)
            self.heading_controller_saturated[:] = unclamped.abs() >= self.yaw_limit
            low_pass = self.filtered_yaw_command + self.low_pass_alpha * (
                self.raw_yaw_command - self.filtered_yaw_command
            )
            self.filtered_yaw_command += (low_pass - self.filtered_yaw_command).clamp(
                -self.slew_limit, self.slew_limit
            )
        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = self.filtered_yaw_command


def heading_error_l2(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    error = torch.atan2(
        torch.sin(term.target_heading_w - term.robot.data.heading_w.torch),
        torch.cos(term.target_heading_w - term.robot.data.heading_w.torch),
    )
    return error.square()


def lateral_velocity_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_b.torch[:, 1].square()


def cross_track_error_l2(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    displacement = term.robot.data.root_pos_w.torch[:, :2] - term.path_origin_xy
    normal = torch.stack((-torch.sin(term.target_heading_w), torch.cos(term.target_heading_w)), dim=1)
    return (displacement * normal).sum(dim=1).square()


class AnklePitchEffortHinge(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.joint_ids, _ = self.robot.find_joints(".*_ankle_pitch_joint")

    def __call__(self, env, asset_cfg: SceneEntityCfg, threshold: float = 0.95) -> torch.Tensor:
        del env, asset_cfg
        effort = self.robot.data.applied_torque.torch[:, self.joint_ids].abs()
        limit = self.robot.data.joint_effort_limits.torch[:, self.joint_ids].abs().clamp_min(1.0e-6)
        return torch.relu(effort / limit - threshold).square().sum(dim=1)
