"""Continuous body-frame translation curriculum for Phase W1A."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from g1_single_policy.command_curriculum import G1BidirectionalVelocityCommand


class W1AContinuousTranslationCommand(G1BidirectionalVelocityCommand):
    """One continuous-angle WALK curriculum; yaw is identically zero."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.training_iteration = 0
        self.sampled_theta = torch.zeros(self.num_envs, device=self.device)
        self.sampled_speed = torch.zeros(self.num_envs, device=self.device)
        self.external_override_enabled = False

    @property
    def phase(self) -> str:
        if self.training_iteration <= 40:
            return "A_FORWARD_HALF_PLANE"
        if self.training_iteration <= 100:
            return "B_LOW_SPEED_ALL_DIRECTION"
        if self.training_iteration <= 160:
            return "C_DIRECTIONAL_SPEED_ENVELOPE"
        return "D_BALANCED_CONSOLIDATION"

    def set_training_iteration(self, iteration: int) -> None:
        self.training_iteration = int(iteration)

    def _signed_uniform(self, count: int, lower_deg: float, upper_deg: float) -> torch.Tensor:
        magnitude = torch.empty(count, device=self.device).uniform_(
            math.radians(lower_deg), math.radians(upper_deg)
        )
        sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            torch.full((count,), -1.0, device=self.device),
            torch.ones(count, device=self.device),
        )
        return magnitude * sign

    def _mixture_angles(self, count: int, weights: tuple[float, ...], phase: str) -> torch.Tensor:
        bucket = torch.multinomial(
            torch.tensor(weights, device=self.device), count, replacement=True
        )
        theta = torch.zeros(count, device=self.device)
        if phase == "A":
            # 30% exact forward anchor, 30% continuous front diagonal,
            # 40% continuous near-lateral; left/right sampling is symmetric.
            mask = bucket == 1
            theta[mask] = self._signed_uniform(int(mask.sum()), 15.0, 60.0)
            mask = bucket == 2
            theta[mask] = self._signed_uniform(int(mask.sum()), 60.0, 90.0)
        elif phase == "B":
            # Continuous values within each requested weighted sector.
            mask = bucket == 0
            theta[mask] = self._signed_uniform(int(mask.sum()), 0.0, 45.0)
            mask = bucket == 1
            theta[mask] = self._signed_uniform(int(mask.sum()), 45.0, 105.0)
            mask = bucket == 2
            theta[mask] = self._signed_uniform(int(mask.sum()), 105.0, 157.5)
            mask = bucket == 3
            theta[mask] = self._signed_uniform(int(mask.sum()), 157.5, 180.0)
        else:
            # 15/20/25/25/15: forward, front diagonal, lateral,
            # rear diagonal, backward.
            bounds = ((0.0, 22.5), (22.5, 67.5), (67.5, 112.5),
                      (112.5, 157.5), (157.5, 180.0))
            for index, (lower, upper) in enumerate(bounds):
                mask = bucket == index
                theta[mask] = self._signed_uniform(int(mask.sum()), lower, upper)
        return theta

    @staticmethod
    def _directional_max(theta: torch.Tensor) -> torch.Tensor:
        degrees = torch.rad2deg(theta.abs())
        return torch.where(
            degrees <= 45.0,
            torch.full_like(degrees, 1.2),
            torch.where(
                degrees <= 90.0,
                torch.full_like(degrees, 0.8),
                torch.where(
                    degrees <= 135.0,
                    torch.full_like(degrees, 0.7),
                    torch.full_like(degrees, 0.6),
                ),
            ),
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        count = ids.numel()
        phase = self.phase
        if phase == "A_FORWARD_HALF_PLANE":
            theta = self._mixture_angles(count, (0.30, 0.30, 0.40), "A")
            max_speed = torch.where(
                torch.rad2deg(theta.abs()) <= 45.0,
                torch.full_like(theta, 1.2),
                torch.full_like(theta, 0.6),
            )
            speed = 0.3 + torch.rand(count, device=self.device) * (max_speed - 0.3)
        elif phase == "B_LOW_SPEED_ALL_DIRECTION":
            theta = self._mixture_angles(count, (0.20, 0.30, 0.30, 0.20), "B")
            speed = torch.empty(count, device=self.device).uniform_(0.2, 0.6)
        elif phase == "C_DIRECTIONAL_SPEED_ENVELOPE":
            theta = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
            max_speed = self._directional_max(theta)
            speed = 0.2 + torch.rand(count, device=self.device) * (max_speed - 0.2)
        else:
            theta = self._mixture_angles(count, (0.15, 0.20, 0.25, 0.25, 0.15), "D")
            max_speed = self._directional_max(theta)
            speed = 0.2 + torch.rand(count, device=self.device) * (max_speed - 0.2)
        self.sampled_theta[ids] = theta
        self.sampled_speed[ids] = speed
        self.vel_command_b[ids, 0] = speed * torch.cos(theta)
        self.vel_command_b[ids, 1] = speed * torch.sin(theta)
        self.vel_command_b[ids, 2] = 0.0
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False

    def _update_command(self) -> None:
        if self.external_override_enabled:
            self.vel_command_b.copy_(self.external_override)
        # Otherwise the command is held until CommandManager resamples it.
        self.vel_command_b[:, 2] = 0.0

