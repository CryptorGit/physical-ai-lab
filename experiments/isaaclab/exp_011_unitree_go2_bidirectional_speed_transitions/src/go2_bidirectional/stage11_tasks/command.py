"""Stage 7 curriculum with the frozen Stage 10 heading controller."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from go2_bidirectional.stage7_tasks.command import LowSpeedGaitVelocityCommand, minimum_jerk


def wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def yaw_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


class PhaseGatedLowSpeedVelocityCommand(LowSpeedGaitVelocityCommand):
    """Apply the frozen Kp=1, +/-0.10 rad/s Stage 10 command controller."""

    heading_kp = 1.0
    heading_limit = 0.10
    activation_s = 0.5
    acquisition_s = 0.5

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.heading_reference = torch.zeros(self.num_envs, device=self.device)
        self.heading_gate = torch.zeros(self.num_envs, device=self.device)
        self.heading_raw = torch.zeros(self.num_envs, device=self.device)
        self.heading_command = torch.zeros(self.num_envs, device=self.device)
        self.acquisition_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.activation_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.heading_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reference_frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._history_steps = max(1, round(0.5 / float(env.step_dt)))
        self._yaw_history = torch.zeros(
            self._history_steps, self.num_envs, device=self.device
        )
        self._history_index = 0
        self._history_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if not hasattr(self, "heading_gate") or ids.numel() == 0:
            return
        self.heading_reference[ids] = 0.0
        self.heading_gate[ids] = 0.0
        self.heading_raw[ids] = 0.0
        self.heading_command[ids] = 0.0
        self.acquisition_age[ids] = 0
        self.activation_elapsed[ids] = 0.0
        self.heading_active[ids] = False
        self.reference_frozen[ids] = False
        self._yaw_history[:, ids] = 0.0
        self._history_count[ids] = 0

    def _freeze_reference(self, mask: torch.Tensor, current_yaw: torch.Tensor) -> None:
        if not mask.any():
            return
        # Circular median around the current sample, robust to +/-pi wrapping.
        history = self._yaw_history[:, mask]
        relative = wrap_angle(history - current_yaw[mask][None])
        reference = wrap_angle(current_yaw[mask] + relative.median(dim=0).values)
        self.heading_reference[mask] = reference
        self.reference_frozen[mask] = True

    def _update_command(self) -> None:
        super()._update_command()
        robot = self._env.scene["robot"]
        yaw = yaw_xyzw(robot.data.root_quat_w.torch)
        self._yaw_history[self._history_index] = yaw
        self._history_index = (self._history_index + 1) % self._history_steps
        self._history_count += 1
        transition = self.source_speed != self.target_speed
        steady = ~transition

        # Frozen steady schedule: reference at 1.0 s, activate over 0.5 s.
        freeze_steady = steady & ~self.reference_frozen & (self.elapsed_s >= 1.0)
        self._freeze_reference(freeze_steady, yaw)
        steady_elapsed = (self.elapsed_s - 1.0).clamp_min(0.0)
        self.heading_gate[steady] = minimum_jerk(
            steady_elapsed[steady] / self.activation_s
        )

        # Transition schedule: source and ramp off, then require 0.5 s acquisition.
        ramp_end = self.source_hold_s + self.ramp_duration_s
        freeze_transition = (
            transition & ~self.reference_frozen & (self.elapsed_s >= self.source_hold_s)
        )
        self._freeze_reference(freeze_transition, yaw)
        waiting = transition & (self.elapsed_s >= ramp_end) & ~self.heading_active
        actual_speed = robot.data.root_lin_vel_b.torch[:, 0]
        target = self.target_speed
        tolerance = torch.where(
            target <= 0.08,
            torch.full_like(target, 0.08),
            torch.where(
                target <= 0.6,
                torch.full_like(target, 0.15),
                torch.where(target <= 1.2, torch.full_like(target, 0.20), torch.full_like(target, 0.25)),
            ),
        )
        acquired = torch.where(
            target <= 0.08,
            actual_speed.abs() <= tolerance,
            (actual_speed - target).abs() <= tolerance,
        )
        self.acquisition_age[waiting & acquired] += 1
        self.acquisition_age[waiting & ~acquired] = 0
        acquired_duration = self.acquisition_age.float() * float(self._env.step_dt)
        activate = waiting & (acquired_duration >= self.acquisition_s)
        self.heading_active[activate] = True
        self.activation_elapsed[activate] = 0.0
        active = transition & self.heading_active
        self.activation_elapsed[active] += float(self._env.step_dt)
        self.heading_gate[transition] = 0.0
        self.heading_gate[active] = minimum_jerk(
            self.activation_elapsed[active] / self.activation_s
        )

        error = wrap_angle(self.heading_reference - yaw)
        self.heading_raw = (self.heading_kp * error).clamp(
            -self.heading_limit, self.heading_limit
        )
        self.heading_command = self.heading_gate * self.heading_raw
        self.vel_command_b[:, 2] = self.heading_command
