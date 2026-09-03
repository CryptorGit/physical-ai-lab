"""Stage 2W-B command distribution and the single diagnostic reward delta."""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .stage2w_mdp import IndependentWalkCommand


class StabilizedWalkCommand(IndependentWalkCommand):
    """Weighted WALK speeds plus smooth, low-frequency heading perturbations."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.base_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.perturb_amplitude = torch.zeros(self.num_envs, device=self.device)
        self.perturb_frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self.perturb_phase = torch.zeros(self.num_envs, device=self.device)
        self.perturb_probability = float(getattr(cfg, "heading_perturb_probability", 0.5))
        self.perturb_amplitude_max = float(getattr(cfg, "heading_perturb_amplitude_max_rad", 0.06))
        self.perturb_frequency_range = tuple(getattr(cfg, "heading_perturb_frequency_hz", (0.08, 0.15)))

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        speeds = torch.tensor((0.6, 0.8, 1.0, 1.2), device=self.device)
        weights = torch.tensor((0.2, 0.2, 0.3, 0.3), device=self.device)
        self.target_speed[ids] = speeds[torch.multinomial(weights, len(ids), replacement=True)]
        self.base_heading_w[ids] = self.robot.data.heading_w.torch[ids]
        self.target_heading_w[ids] = self.base_heading_w[ids]
        self.path_origin_xy[ids] = self.robot.data.root_pos_w.torch[ids, :2]
        active = torch.rand(len(ids), device=self.device) < self.perturb_probability
        amplitude = (
            torch.rand(len(ids), device=self.device) * 2.0 - 1.0
        ) * self.perturb_amplitude_max
        self.perturb_amplitude[ids] = torch.where(active, amplitude, torch.zeros_like(amplitude))
        low, high = self.perturb_frequency_range
        self.perturb_frequency_hz[ids] = low + torch.rand(len(ids), device=self.device) * (high - low)
        self.perturb_phase[ids] = torch.rand(len(ids), device=self.device) * (2.0 * pi)
        self.filtered_yaw_command[ids] = 0.0
        self.raw_yaw_command[ids] = 0.0
        self.heading_controller_saturated[ids] = False
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = False

    def _update_command(self):
        elapsed = self._env.episode_length_buf.float() * self._env.step_dt
        phase = 2.0 * pi * self.perturb_frequency_hz * elapsed + self.perturb_phase
        initial = torch.sin(self.perturb_phase)
        offset = self.perturb_amplitude * (torch.sin(phase) - initial)
        self.target_heading_w[:] = self.base_heading_w + offset
        super()._update_command()


class YawRateOscillationPenalty(ManagerTermBase):
    """Penalize step-to-step yaw-rate changes without prescribing gait phase."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.previous.zero_()
        else:
            self.previous[env_ids] = 0.0

    def __call__(self, env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        del env, asset_cfg
        current = self.robot.data.root_ang_vel_b.torch[:, 2]
        delta = current - self.previous
        self.previous[:] = current
        return delta.square()
