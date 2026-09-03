"""Fixed-cohort low-speed gait-stabilization command curriculum."""

from __future__ import annotations

from collections.abc import Sequence
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand

ZERO_HOLD, LOW_SPEED_STEADY, LOW_SPEED_TRANSITION, CAPABILITY_ANCHOR = range(4)
COHORT_NAMES = ("ZERO_HOLD", "LOW_SPEED_STEADY", "LOW_SPEED_TRANSITION", "CAPABILITY_ANCHOR")
LOW_SPEEDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
LOW_PAIRS = (
    (0.0, 0.2), (0.0, 0.4), (0.0, 0.6), (0.2, 0.4), (0.2, 0.6), (0.4, 0.6),
    (0.6, 0.4), (0.6, 0.2), (0.6, 0.0), (0.4, 0.2), (0.4, 0.0), (0.2, 0.0),
)
ANCHORS = (
    (1.2, 1.2), (2.0, 2.0), (0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0),
)


def minimum_jerk(tau: torch.Tensor) -> torch.Tensor:
    tau = tau.clamp(0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


class LowSpeedGaitVelocityCommand(UniformVelocityCommand):
    """Keep every environment in one 15/35/30/20 cohort for its lifetime."""

    ramp_duration_s = 1.5

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids = torch.arange(self.num_envs, device=self.device)
        percentile = torch.div(ids * 100, self.num_envs, rounding_mode="floor")
        self.cohort = torch.where(
            percentile < 15, ZERO_HOLD,
            torch.where(percentile < 50, LOW_SPEED_STEADY,
                        torch.where(percentile < 80, LOW_SPEED_TRANSITION, CAPABILITY_ANCHOR)),
        )
        self.source_speed = torch.zeros(self.num_envs, device=self.device)
        self.target_speed = torch.zeros(self.num_envs, device=self.device)
        self.source_hold_s = torch.zeros(self.num_envs, device=self.device)
        self.elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self.condition_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._balanced_cursor = {
            LOW_SPEED_STEADY: 0, LOW_SPEED_TRANSITION: 0, CAPABILITY_ANCHOR: 0
        }

    def _balanced_choice(self, count: int, choices: int, cohort_id: int) -> torch.Tensor:
        start = self._balanced_cursor[cohort_id]
        result = (torch.arange(count, device=self.device) + start).remainder(choices)
        self._balanced_cursor[cohort_id] = (start + count) % choices
        return result

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        self.elapsed_s[ids] = 0.0
        self.source_hold_s[ids] = 2.0 + torch.rand(ids.numel(), device=self.device)
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False
        self.condition_index[ids] = -1
        zero = ids[self.cohort[ids] == ZERO_HOLD]
        self.source_speed[zero] = self.target_speed[zero] = 0.0
        steady = ids[self.cohort[ids] == LOW_SPEED_STEADY]
        if steady.numel():
            choice = self._balanced_choice(steady.numel(), len(LOW_SPEEDS), LOW_SPEED_STEADY)
            values = torch.tensor(LOW_SPEEDS, device=self.device)[choice]
            self.condition_index[steady] = choice
            self.source_speed[steady] = self.target_speed[steady] = values
        for cohort_id, choices in (
            (LOW_SPEED_TRANSITION, LOW_PAIRS), (CAPABILITY_ANCHOR, ANCHORS)
        ):
            selected = ids[self.cohort[ids] == cohort_id]
            if selected.numel():
                choice = self._balanced_choice(selected.numel(), len(choices), cohort_id)
                pairs = torch.tensor(choices, device=self.device)
                self.condition_index[selected] = choice
                self.source_speed[selected] = pairs[choice, 0]
                self.target_speed[selected] = pairs[choice, 1]

    def _update_command(self) -> None:
        self.elapsed_s += float(self._env.step_dt)
        command = self.source_speed.clone()
        transition = (self.source_speed != self.target_speed)
        profile = minimum_jerk((self.elapsed_s - self.source_hold_s) / self.ramp_duration_s)
        command[transition] += (
            self.target_speed[transition] - self.source_speed[transition]
        ) * profile[transition]
        self.vel_command_b[:, 0] = command.clamp(0.0, 2.0)
        self.vel_command_b[:, 1:] = 0.0
