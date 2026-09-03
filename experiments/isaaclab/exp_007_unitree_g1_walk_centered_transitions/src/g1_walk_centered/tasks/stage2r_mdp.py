"""Commands and reward terms for Stage 2R unified STAND/WALK retraining."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def _minimum_jerk(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


class UnifiedStandWalkCommand(UniformVelocityCommand):
    """Episode command generator with an externally held, smoothed world heading.

    R1 holds a sampled persistent state. R2-R4 use minimum-jerk transition
    schedules. The target world heading remains controller state and is never
    appended to the 123-dimensional policy observation.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.phase = str(getattr(cfg, "stage2r_phase", "R1"))
        self.k_heading = float(getattr(cfg, "k_heading", 0.4))
        self.k_yaw_rate = float(getattr(cfg, "k_yaw_rate", 0.10))
        self.yaw_limit = float(getattr(cfg, "yaw_rate_limit", 0.30))
        self.low_pass_alpha = float(getattr(cfg, "low_pass_alpha", 0.15))
        self.slew_limit = float(getattr(cfg, "yaw_rate_slew_limit", 0.01))
        self.target_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.filtered_yaw_command = torch.zeros(self.num_envs, device=self.device)
        self.target_speed = torch.zeros(self.num_envs, device=self.device)
        self.acceleration_start_s = torch.zeros(self.num_envs, device=self.device)
        self.acceleration_duration_s = torch.full((self.num_envs,), 2.0, device=self.device)
        self.deceleration_start_s = torch.full((self.num_envs,), 1.0e9, device=self.device)
        self.deceleration_duration_s = torch.full((self.num_envs,), 2.0, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = len(ids)
        r = torch.rand(count, device=self.device)
        speed_index = torch.randint(0, 4, (count,), device=self.device)
        speeds = torch.tensor((0.6, 0.8, 1.0, 1.2), device=self.device)
        sampled = speeds[speed_index]
        if self.phase in ("R0", "R1"):
            self.target_speed[ids] = torch.where(r < 0.40, 0.0, sampled)
            self.acceleration_start_s[ids] = 0.0
        else:
            self.target_speed[ids] = sampled
            self.acceleration_start_s[ids] = 0.8 + 0.7 * torch.rand(count, device=self.device)
        self.acceleration_duration_s[ids] = 1.6 + 0.8 * torch.rand(count, device=self.device)
        walk_hold = 2.5 + 2.0 * torch.rand(count, device=self.device)
        self.deceleration_start_s[ids] = (
            self.acceleration_start_s[ids] + self.acceleration_duration_s[ids] + walk_hold
        )
        self.deceleration_duration_s[ids] = 1.6 + 0.8 * torch.rand(count, device=self.device)
        self.target_heading_w[ids] = self.robot.data.heading_w.torch[ids]
        self.filtered_yaw_command[ids] = 0.0
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = self.target_speed[ids] == 0.0

    def _update_command(self):
        elapsed = self._env.episode_length_buf.float() * self._env.step_dt
        if self.phase in ("R0", "R1"):
            vx = self.target_speed
        else:
            accel_u = (elapsed - self.acceleration_start_s) / self.acceleration_duration_s
            vx = self.target_speed * _minimum_jerk(accel_u)
            if self.phase in ("R3", "R4"):
                decel_u = (elapsed - self.deceleration_start_s) / self.deceleration_duration_s
                vx = torch.where(
                    elapsed >= self.deceleration_start_s,
                    self.target_speed * (1.0 - _minimum_jerk(decel_u)),
                    vx,
                )
            if self.phase == "R2":
                vx = torch.where(elapsed < self.acceleration_start_s, 0.0, vx)

        error = torch.atan2(
            torch.sin(self.target_heading_w - self.robot.data.heading_w.torch),
            torch.cos(self.target_heading_w - self.robot.data.heading_w.torch),
        )
        raw = self.k_heading * error - self.k_yaw_rate * self.robot.data.root_ang_vel_b.torch[:, 2]
        raw = raw.clamp(-self.yaw_limit, self.yaw_limit)
        low_passed = self.filtered_yaw_command + self.low_pass_alpha * (raw - self.filtered_yaw_command)
        delta = (low_passed - self.filtered_yaw_command).clamp(-self.slew_limit, self.slew_limit)
        self.filtered_yaw_command.add_(delta).clamp_(-self.yaw_limit, self.yaw_limit)
        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = self.filtered_yaw_command


def stand_horizontal_speed_l2(env, command_name: str, threshold: float = 0.05) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    active = command[:, 0].abs() <= threshold
    speed_sq = env.scene["robot"].data.root_lin_vel_b.torch[:, :2].square().sum(dim=1)
    return speed_sq * active


def stand_yaw_rate_l2(env, command_name: str, threshold: float = 0.05) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    active = command[:, 0].abs() <= threshold
    return env.scene["robot"].data.root_ang_vel_b.torch[:, 2].square() * active


def stand_flight_penalty(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
    return ((contacts.sum(dim=1) == 0) & (command[:, 0].abs() <= threshold)).float()


def stand_double_support_reward(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
    return ((contacts.sum(dim=1) == 2) & (command[:, 0].abs() <= threshold)).float()


class AnklePitchEffortHinge(ManagerTermBase):
    """Smooth squared hinge above a fraction of ankle-pitch effort limits."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        self.robot: Articulation = env.scene[asset_cfg.name]
        self.joint_ids, _ = self.robot.find_joints(".*_ankle_pitch_joint")

    def __call__(self, env, asset_cfg: SceneEntityCfg, threshold: float = 0.95) -> torch.Tensor:
        del env, asset_cfg
        effort = self.robot.data.applied_torque.torch[:, self.joint_ids].abs()
        limit = self.robot.data.joint_effort_limits.torch[:, self.joint_ids].abs().clamp_min(1.0e-6)
        return torch.relu(effort / limit - threshold).square().sum(dim=1)
