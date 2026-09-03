"""Fixed-cohort STAND/WALK/RUN_LOW/bidirectional command curriculum."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand

from .phase_gated_heading import KP, OMEGA_MAX, minimum_jerk, wrapped_heading_error, yaw_from_quat_wxyz

ZERO_HOLD, WALK_STEADY, RUN_HOLD, BIDIRECTIONAL_SEQUENCE = range(4)
COHORT_NAMES = ("ZERO_HOLD", "WALK_STEADY", "RUN_HOLD", "BIDIRECTIONAL_SEQUENCE")
WALK_SPEEDS = (0.6, 0.8, 1.0, 1.2)
RUN_TARGETS = (2.4, 2.6)
SEQUENCE = (
    (0.0, 1.5, "hold"), (0.6, 1.0, "ramp"), (0.6, 1.0, "hold"),
    (1.2, 1.0, "ramp"), (1.2, 1.5, "hold"), (None, 1.5, "ramp"),
    (None, 3.0, "hold"), (1.2, 1.5, "ramp"), (1.2, 1.5, "hold"),
    (0.6, 1.0, "ramp"), (0.6, 1.0, "hold"), (0.0, 1.0, "ramp"),
    (0.0, 2.0, "hold"),
)


class G1BidirectionalVelocityCommand(UniformVelocityCommand):
    """One actor, fixed environment cohorts, and only time-varying commands."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids = torch.arange(self.num_envs, device=self.device)
        bucket = ids.remainder(10)
        self.cohort = torch.where(bucket < 2, ZERO_HOLD, torch.where(
            bucket < 4, WALK_STEADY, torch.where(bucket < 6, RUN_HOLD, BIDIRECTIONAL_SEQUENCE)))
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.target_run = torch.full((self.num_envs,), 2.4, device=self.device)
        self.steady_speed = torch.zeros(self.num_envs, device=self.device)
        self.heading_reference = torch.zeros(self.num_envs, device=self.device)
        self.heading_gate = torch.zeros(self.num_envs, device=self.device)
        self.reference_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.segment_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.acquisition_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_segment = torch.full_like(self.segment_index, -1)
        # Evaluation-only command override. Training never enables it.
        self.external_override_enabled = False
        self.external_override = torch.zeros((self.num_envs, 3), device=self.device)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        self.elapsed[ids] = 0.0
        self.heading_gate[ids] = 0.0
        self.reference_valid[ids] = False
        self.segment_index[ids] = 0
        self._previous_segment[ids] = -1
        self.acquisition_age[ids] = 0
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False
        walk = ids[self.cohort[ids] == WALK_STEADY]
        if walk.numel():
            self.steady_speed[walk] = torch.tensor(WALK_SPEEDS, device=self.device)[
                torch.randint(len(WALK_SPEEDS), (walk.numel(),), device=self.device)]
        run = ids[(self.cohort[ids] == RUN_HOLD) | (self.cohort[ids] == BIDIRECTIONAL_SEQUENCE)]
        if run.numel():
            self.target_run[run] = torch.tensor(RUN_TARGETS, device=self.device)[
                torch.randint(2, (run.numel(),), device=self.device)]

    @staticmethod
    def _ramp(t, start, duration, source, target):
        return source + (target - source) * minimum_jerk((t - start) / duration)

    def _sequence_speed(self, ids):
        t = self.elapsed[ids]
        speed = torch.zeros_like(t)
        seg = torch.zeros_like(t, dtype=torch.long)
        cursor = 0.0
        previous = torch.zeros_like(t)
        run = self.target_run[ids]
        for index, (fixed_target, duration, kind) in enumerate(SEQUENCE):
            target = run if fixed_target is None else torch.full_like(t, fixed_target)
            mask = (t >= cursor) & (t < cursor + duration)
            if kind == "ramp":
                value = previous + (target - previous) * minimum_jerk((t - cursor) / duration)
            else:
                value = target
            speed[mask] = value[mask]
            seg[mask] = index
            previous = target
            cursor += duration
        speed[t >= cursor] = 0.0
        seg[t >= cursor] = len(SEQUENCE) - 1
        self.segment_index[ids] = seg
        return speed

    def _update_heading(self, command):
        yaw = yaw_from_quat_wxyz(self._env.scene["robot"].data.root_quat_w)
        actual = self._env.scene["robot"].data.root_lin_vel_b[:, 0]
        steady = (self.cohort == ZERO_HOLD) | (self.cohort == WALK_STEADY)
        capture = steady & (self.elapsed >= 1.0) & (~self.reference_valid)
        self.heading_reference[capture] = yaw[capture]
        self.reference_valid[capture] = True
        activating = steady & (self.elapsed >= 1.0) & (self.elapsed < 1.5)
        self.heading_gate[activating] = minimum_jerk((self.elapsed[activating] - 1.0) / 0.5)
        self.heading_gate[steady & (self.elapsed >= 1.5)] = 1.0

        scheduled = ~steady
        changed = scheduled & (self.segment_index != self._previous_segment)
        self.heading_gate[changed] = 0.0
        self.acquisition_age[changed] = 0
        self.heading_reference[changed] = yaw[changed]
        self.reference_valid[changed] = True
        ramp_segments = torch.zeros_like(scheduled)
        for idx, (_, _, kind) in enumerate(SEQUENCE):
            if kind == "ramp":
                ramp_segments |= self.segment_index == idx
        hold = scheduled & (~ramp_segments)
        tolerance = torch.where(command <= 0.01, 0.08, torch.where(command <= 1.2, 0.20, 0.25))
        within = hold & ((actual - command).abs() <= tolerance)
        self.acquisition_age = torch.where(within, self.acquisition_age + 1, torch.zeros_like(self.acquisition_age))
        acquired = hold & (self.acquisition_age >= 25)
        self.heading_gate[acquired] = torch.clamp(self.heading_gate[acquired] + float(self._env.step_dt) / 0.5, 0.0, 1.0)
        self._previous_segment.copy_(self.segment_index)
        error = wrapped_heading_error(self.heading_reference, yaw)
        return self.heading_gate * torch.clamp(KP * error, -OMEGA_MAX, OMEGA_MAX)

    def _update_command(self) -> None:
        if self.external_override_enabled:
            self.vel_command_b.copy_(self.external_override)
            return
        self.elapsed += float(self._env.step_dt)
        command = torch.zeros(self.num_envs, device=self.device)
        walk = self.cohort == WALK_STEADY
        command[walk] = self.steady_speed[walk]
        run = self.cohort == RUN_HOLD
        t = self.elapsed
        command[run & (t >= 1.0) & (t < 2.0)] = self._ramp(t[run & (t >= 1.0) & (t < 2.0)], 1.0, 1.0, 0.0, 1.2)
        command[run & (t >= 2.0) & (t < 3.5)] = 1.2
        mask = run & (t >= 3.5) & (t < 5.0)
        command[mask] = 1.2 + (self.target_run[mask] - 1.2) * minimum_jerk((t[mask] - 3.5) / 1.5)
        command[run & (t >= 5.0)] = self.target_run[run & (t >= 5.0)]
        seq = torch.where(self.cohort == BIDIRECTIONAL_SEQUENCE)[0]
        if seq.numel():
            command[seq] = self._sequence_speed(seq)
        self.vel_command_b[:, 0] = command.clamp(0.0, 2.6)
        self.vel_command_b[:, 1] = 0.0
        # Stage 1B frozen Pilot-1 amendment: the policy observation and the
        # unchanged yaw-tracking reward consume this same command.  Keep both
        # semantics exactly at zero during PPO; all external controllers are
        # evaluation-only.
        self.heading_gate.zero_()
        self.vel_command_b[:, 2] = 0.0


class G1PhaseARunAcquisitionCommand(G1BidirectionalVelocityCommand):
    """Phase A: every episode walks through 1.2 m/s before a focused RUN target."""

    focus_probability = 0.7
    full_range = (2.3, 2.6)
    focus_range = (2.4, 2.5)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cohort.fill_(RUN_HOLD)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        self.elapsed[ids] = 0.0
        self.heading_gate[ids] = 0.0
        self.reference_valid[ids] = False
        self.segment_index[ids] = 0
        self._previous_segment[ids] = -1
        self.acquisition_age[ids] = 0
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False
        focused = torch.rand(ids.numel(), device=self.device) < self.focus_probability
        middle = torch.empty(ids.numel(), device=self.device).uniform_(*self.focus_range)
        full = torch.empty(ids.numel(), device=self.device).uniform_(*self.full_range)
        self.target_run[ids] = torch.where(focused, middle, full)

    def _update_command(self) -> None:
        if self.external_override_enabled:
            self.vel_command_b.copy_(self.external_override)
            return
        self.elapsed += float(self._env.step_dt)
        t = self.elapsed
        command = torch.zeros(self.num_envs, device=self.device)
        ramp_walk = (t >= 1.0) & (t < 2.0)
        command[ramp_walk] = 1.2 * minimum_jerk((t[ramp_walk] - 1.0) / 1.0)
        walk_hold = (t >= 2.0) & (t < 3.5)
        command[walk_hold] = 1.2
        ramp_run = (t >= 3.5) & (t < 5.0)
        command[ramp_run] = 1.2 + (self.target_run[ramp_run] - 1.2) * minimum_jerk(
            (t[ramp_run] - 3.5) / 1.5
        )
        run_hold = t >= 5.0
        command[run_hold] = self.target_run[run_hold]
        self.segment_index.zero_()
        self.segment_index[ramp_walk] = 1
        self.segment_index[walk_hold] = 2
        self.segment_index[ramp_run] = 3
        self.segment_index[run_hold] = 4
        self.vel_command_b[:, 0] = command.clamp(0.0, 2.6)
        self.vel_command_b[:, 1:] = 0.0
        self.heading_gate.zero_()


R1_RUN_HOLD, R1_WALK_1P2_HOLD, R1_BIDIRECTIONAL = range(3)
R1_COHORT_NAMES = ("RUN_HOLD", "WALK_1P2_HOLD", "BIDIRECTIONAL_WALK_RUN")


class G1ReversePhaseR1Command(G1BidirectionalVelocityCommand):
    """Reverse continuation: preserve RUN while adding 1.2 m/s WALK and both edges."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        bucket = torch.arange(self.num_envs, device=self.device).remainder(10)
        self.r1_cohort = torch.where(
            bucket < 5,
            R1_RUN_HOLD,
            torch.where(bucket < 7, R1_WALK_1P2_HOLD, R1_BIDIRECTIONAL),
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        self.elapsed[ids] = 0.0
        self.heading_gate[ids] = 0.0
        self.reference_valid[ids] = False
        self.segment_index[ids] = 0
        self._previous_segment[ids] = -1
        self.acquisition_age[ids] = 0
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False
        active = ids[self.r1_cohort[ids] != R1_WALK_1P2_HOLD]
        if active.numel():
            self.target_run[active] = torch.where(
                torch.rand(active.numel(), device=self.device) < 0.5,
                torch.full((active.numel(),), 2.4, device=self.device),
                torch.full((active.numel(),), 2.6, device=self.device),
            )

    def _update_command(self) -> None:
        if self.external_override_enabled:
            self.vel_command_b.copy_(self.external_override)
            return
        self.elapsed += float(self._env.step_dt)
        t = self.elapsed
        command = torch.zeros(self.num_envs, device=self.device)
        walk_ramp = t < 1.0
        command[walk_ramp] = 1.2 * minimum_jerk(t[walk_ramp] / 1.0)

        walk = self.r1_cohort == R1_WALK_1P2_HOLD
        command[walk & (t >= 1.0)] = 1.2
        self.segment_index[walk] = torch.where(
            t[walk] < 1.0, torch.zeros_like(self.segment_index[walk]), 1
        )

        run = self.r1_cohort == R1_RUN_HOLD
        command[run & (t >= 1.0) & (t < 2.5)] = 1.2
        run_ramp = run & (t >= 2.5) & (t < 4.0)
        command[run_ramp] = 1.2 + (self.target_run[run_ramp] - 1.2) * minimum_jerk(
            (t[run_ramp] - 2.5) / 1.5
        )
        command[run & (t >= 4.0)] = self.target_run[run & (t >= 4.0)]
        self.segment_index[run] = torch.where(
            t[run] < 1.0, 0, torch.where(t[run] < 2.5, 1, torch.where(t[run] < 4.0, 2, 3))
        )

        sequence = self.r1_cohort == R1_BIDIRECTIONAL
        command[sequence & (t >= 1.0) & (t < 3.0)] = 1.2
        up = sequence & (t >= 3.0) & (t < 4.5)
        command[up] = 1.2 + (self.target_run[up] - 1.2) * minimum_jerk(
            (t[up] - 3.0) / 1.5
        )
        command[sequence & (t >= 4.5) & (t < 8.5)] = self.target_run[
            sequence & (t >= 4.5) & (t < 8.5)
        ]
        down = sequence & (t >= 8.5) & (t < 10.0)
        command[down] = self.target_run[down] + (1.2 - self.target_run[down]) * minimum_jerk(
            (t[down] - 8.5) / 1.5
        )
        command[sequence & (t >= 10.0)] = 1.2
        self.segment_index[sequence] = torch.where(
            t[sequence] < 1.0,
            0,
            torch.where(
                t[sequence] < 3.0,
                1,
                torch.where(
                    t[sequence] < 4.5,
                    2,
                    torch.where(t[sequence] < 8.5, 3, torch.where(t[sequence] < 10.0, 4, 5)),
                ),
            ),
        )
        self.vel_command_b[:, 0] = command.clamp(0.0, 2.6)
        self.vel_command_b[:, 1:] = 0.0
        self.heading_gate.zero_()
