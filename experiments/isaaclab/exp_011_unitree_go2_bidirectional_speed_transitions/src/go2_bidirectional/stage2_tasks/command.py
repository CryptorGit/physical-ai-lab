"""Symmetric fixed-cohort 0--2 m/s command curriculum."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand

ZERO_HOLD = 0
STEADY_SPEED = 1
ACCELERATION = 2
DECELERATION = 3

COHORT_NAMES = ("ZERO_HOLD", "STEADY_SPEED", "ACCELERATION", "DECELERATION")
STEADY_SPEEDS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
ACCELERATION_PAIRS = ((0.0, 0.6), (0.0, 1.2), (0.6, 1.2), (0.6, 2.0), (1.2, 2.0))
DECELERATION_PAIRS = ((2.0, 1.2), (2.0, 0.6), (1.2, 0.6), (1.2, 0.0), (0.6, 0.0))


def minimum_jerk(tau: torch.Tensor) -> torch.Tensor:
    tau = tau.clamp(0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


class SymmetricBidirectionalVelocityCommand(UniformVelocityCommand):
    """Keep each environment in one cohort and schedule commands inside episodes."""

    ramp_duration_s = 1.5

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids = torch.arange(self.num_envs, device=self.device)
        self.cohort = ids.remainder(4)
        self.source_speed = torch.zeros(self.num_envs, device=self.device)
        self.target_speed = torch.zeros(self.num_envs, device=self.device)
        self.source_hold_s = torch.zeros(self.num_envs, device=self.device)
        self.elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self.pair_index = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.steady_index = torch.full_like(self.pair_index, -1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        self.elapsed_s[ids] = 0.0
        self.source_hold_s[ids] = 2.0 + torch.rand(ids.numel(), device=self.device)
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False

        zero_ids = ids[self.cohort[ids] == ZERO_HOLD]
        self.source_speed[zero_ids] = 0.0
        self.target_speed[zero_ids] = 0.0

        steady_ids = ids[self.cohort[ids] == STEADY_SPEED]
        if steady_ids.numel():
            choice = torch.randint(len(STEADY_SPEEDS), (steady_ids.numel(),), device=self.device)
            values = torch.tensor(STEADY_SPEEDS, device=self.device)[choice]
            self.steady_index[steady_ids] = choice
            self.source_speed[steady_ids] = values
            self.target_speed[steady_ids] = values

        for cohort_id, pairs in ((ACCELERATION, ACCELERATION_PAIRS), (DECELERATION, DECELERATION_PAIRS)):
            cohort_ids = ids[self.cohort[ids] == cohort_id]
            if cohort_ids.numel():
                choice = torch.randint(len(pairs), (cohort_ids.numel(),), device=self.device)
                pair_tensor = torch.tensor(pairs, device=self.device)
                self.pair_index[cohort_ids] = choice
                self.source_speed[cohort_ids] = pair_tensor[choice, 0]
                self.target_speed[cohort_ids] = pair_tensor[choice, 1]

    def _update_command(self) -> None:
        self.elapsed_s += float(self._env.step_dt)
        command = self.source_speed.clone()
        transition = (self.cohort == ACCELERATION) | (self.cohort == DECELERATION)
        ramp_tau = (self.elapsed_s - self.source_hold_s) / self.ramp_duration_s
        profile = minimum_jerk(ramp_tau)
        command[transition] = (
            self.source_speed[transition]
            + (self.target_speed[transition] - self.source_speed[transition]) * profile[transition]
        )
        self.vel_command_b[:, 0] = command.clamp(0.0, 2.0)
        self.vel_command_b[:, 1:] = 0.0
