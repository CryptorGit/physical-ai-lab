"""Stage 6 robust landing-load and actuator-saturation shaping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .stage5_mdp import Stage5ProgressiveVelocityCommand

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


class Stage6VelocityCommand(Stage5ProgressiveVelocityCommand):
    """Sample only the validated 3.8--4.0 m/s band without widening it."""

    focus_low = 3.8


class RobustLandingPenalty(ManagerTermBase):
    """Penalize smooth threshold excess and pre-contact downward foot speed.

    Contact force uses the mean of the three most recent physics samples.  The
    touchdown-speed component uses the foot velocity from the previous control
    step, before the contact impulse changes it.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.foot_ids = cfg.params["sensor_cfg"].body_ids
        if len(self.foot_ids) != 2:
            raise ValueError("RobustLandingPenalty requires exactly two feet")
        self.previous_contacts = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.previous_vertical_velocity = torch.zeros((self.num_envs, 2), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.previous_contacts[env_ids] = False
        self.previous_vertical_velocity[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        force_threshold_n: float = 1000.0,
        force_scale_n: float = 1000.0,
        downward_speed_threshold_mps: float = 3.0,
        downward_speed_scale_mps: float = 0.5,
        impact_component: float = 1.0,
        velocity_component: float = 1.0,
    ) -> torch.Tensor:
        del env, asset_cfg, sensor_cfg
        force_history = self.contact_sensor.data.net_forces_w_history.torch[:, :, self.foot_ids, 2].abs()
        contacts = force_history.amax(dim=1) > 1.0
        first_contact = contacts & ~self.previous_contacts
        short_mean_force = force_history.mean(dim=1)
        force_excess = torch.relu(short_mean_force - force_threshold_n) / force_scale_n
        impact_cost = (force_excess.square() * first_contact).sum(dim=1)

        downward_speed = torch.relu(-self.previous_vertical_velocity)
        speed_excess = torch.relu(downward_speed - downward_speed_threshold_mps) / downward_speed_scale_mps
        velocity_cost = (speed_excess.square() * first_contact).sum(dim=1)

        self.previous_contacts.copy_(contacts)
        self.previous_vertical_velocity.copy_(self.robot.data.body_lin_vel_w.torch[:, self.foot_ids, 2])
        return impact_component * impact_cost + velocity_component * velocity_cost


class JointSaturationPenalty(ManagerTermBase):
    """Penalize time spent above a soft 95% actuator-limit threshold."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        quantity: str,
        threshold: float = 0.95,
        scale: float = 0.05,
    ) -> torch.Tensor:
        del env
        if quantity == "velocity":
            value = self.robot.data.joint_vel.torch.abs()
            limit = self.robot.data.joint_vel_limits.torch.abs()
        elif quantity == "torque":
            value = self.robot.data.applied_torque.torch.abs()
            limit = self.robot.data.joint_effort_limits.torch.abs()
        else:
            raise ValueError(f"Unknown saturation quantity: {quantity}")
        value = value[:, asset_cfg.joint_ids]
        limit = limit[:, asset_cfg.joint_ids]
        ratio = value / torch.clamp(limit, min=1.0e-6)
        return (torch.relu(ratio - threshold) / scale).square().sum(dim=1)


class LandingImpactSymmetryPenalty(ManagerTermBase):
    """Apply a small event cost to the difference between recent L/R impacts."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.foot_ids = cfg.params["sensor_cfg"].body_ids
        self.previous_contacts = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.last_impact = torch.zeros((self.num_envs, 2), device=self.device)
        self.has_impact = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.previous_contacts[env_ids] = False
        self.last_impact[env_ids] = 0.0
        self.has_impact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        normalization_n: float = 3500.0,
    ) -> torch.Tensor:
        del env, sensor_cfg
        force_history = self.contact_sensor.data.net_forces_w_history.torch[:, :, self.foot_ids, 2].abs()
        contacts = force_history.amax(dim=1) > 1.0
        first_contact = contacts & ~self.previous_contacts
        short_mean_force = force_history.mean(dim=1)
        self.last_impact = torch.where(first_contact, short_mean_force, self.last_impact)
        self.has_impact |= first_contact
        compare = first_contact.any(dim=1) & self.has_impact.all(dim=1)
        difference = (self.last_impact[:, 0] - self.last_impact[:, 1]).abs() / normalization_n
        self.previous_contacts.copy_(contacts)
        return difference.square() * compare
