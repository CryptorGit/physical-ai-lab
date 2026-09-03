"""Stateful, safety-bounded fixed feedback for the frozen STOP parent policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class StopFeedbackConfig:
    k_heading: float = 0.0
    k_yaw_rate: float = 0.0
    alpha: float = 1.0
    max_delta_per_step: float = 1.0
    braking_scale: float = 1.0
    hold_scale: float = 1.0
    double_support_scale: float = 1.0
    single_support_scale: float = 1.0
    flight_scale: float = 1.0
    yaw_soft_threshold: float = float("inf")
    yaw_hard_threshold: float = float("inf")
    hard_guard_mode: str = "zero"
    action_limit: float = 0.03
    contact_recovery_zero_steps: int = 0
    contact_recovery_ramp_steps: int = 0
    hard_guard_action_limit: float = 0.03
    hard_guard_disable_torso: bool = False
    ankle_utilization_soft: float = 1.0
    ankle_utilization_hard: float = 1.01
    joint_velocity_soft: float = 1.0
    joint_velocity_hard: float = 1.01
    tilt_soft_rad: float = 10.0
    tilt_hard_rad: float = 11.0
    angular_velocity_soft_rps: float = 100.0
    angular_velocity_hard_rps: float = 101.0
    worsening_yaw_scale: float = 1.0
    flight_hard_zero: bool = False

    def validate(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.max_delta_per_step <= 0.0:
            raise ValueError("max_delta_per_step must be positive")
        disabled_guard = math.isinf(self.yaw_soft_threshold) and math.isinf(self.yaw_hard_threshold)
        if not disabled_guard and (
            self.yaw_soft_threshold < 0.0 or self.yaw_hard_threshold <= self.yaw_soft_threshold
        ):
            raise ValueError("yaw thresholds must satisfy 0 <= soft < hard")
        if self.hard_guard_mode not in {"zero", "damping_only"}:
            raise ValueError(f"unknown hard_guard_mode: {self.hard_guard_mode}")
        for name in (
            "braking_scale", "hold_scale", "double_support_scale", "single_support_scale", "flight_scale"
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 < self.action_limit <= 0.03:
            raise ValueError("action_limit must be in (0, 0.03]")
        if not 0.0 <= self.hard_guard_action_limit <= self.action_limit:
            raise ValueError("hard_guard_action_limit must be in [0, action_limit]")
        if self.contact_recovery_zero_steps < 0 or self.contact_recovery_ramp_steps < 0:
            raise ValueError("contact recovery step counts must be non-negative")
        for soft_name, hard_name in (
            ("ankle_utilization_soft", "ankle_utilization_hard"),
            ("joint_velocity_soft", "joint_velocity_hard"),
            ("tilt_soft_rad", "tilt_hard_rad"),
            ("angular_velocity_soft_rps", "angular_velocity_hard_rps"),
        ):
            if getattr(self, hard_name) <= getattr(self, soft_name):
                raise ValueError(f"{soft_name} must be smaller than {hard_name}")
        if not 0.0 <= self.worsening_yaw_scale <= 1.0:
            raise ValueError("worsening_yaw_scale must be in [0, 1]")

    def as_dict(self) -> dict:
        return asdict(self)


class StopFixedFeedbackController:
    """Apply low-pass, slew, phase/contact gating, and a yaw-spike guard."""

    def __init__(self, num_envs: int, action_dim: int, device: str, config: StopFeedbackConfig) -> None:
        config.validate()
        self.config = config
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        with torch.inference_mode(False):
            self.filtered_signal = torch.zeros(self.num_envs, device=self.device)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
            self.was_stop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.previous_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
            self.support_stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.previous_yaw_abs = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids.to(device=self.device, dtype=torch.long)
        )
        self.filtered_signal[ids] = 0.0
        self.applied_action[ids] = 0.0
        self.was_stop[ids] = False
        self.previous_contacts[ids] = False
        self.support_stable_steps[ids] = 0
        self.previous_yaw_abs[ids] = 0.0

    @staticmethod
    def _soft_safety_scale(value: torch.Tensor, soft: float, hard: float) -> torch.Tensor:
        return ((float(hard) - value) / max(float(hard) - float(soft), 1.0e-6)).clamp(0.0, 1.0)

    def step(
        self,
        policy_observation: torch.Tensor,
        stop_mask: torch.Tensor,
        hold_progress: torch.Tensor,
        support_count: torch.Tensor,
        contacts: torch.Tensor | None = None,
        ankle_utilization: torch.Tensor | None = None,
        joint_velocity_utilization: torch.Tensor | None = None,
        roll_pitch: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.config
        stop = stop_mask.to(dtype=torch.bool)
        entering = stop & ~self.was_stop
        inactive = ~stop
        reliable = support_count > 0
        if contacts is None:
            contacts = torch.stack((support_count > 0, support_count > 1), dim=1)
        contacts = contacts.to(dtype=torch.bool)
        support_changed = stop & reliable & self.was_stop & (contacts != self.previous_contacts).any(dim=1)
        contact_recovered = stop & reliable & ~self.previous_contacts.any(dim=1)
        reset_recovery = entering | support_changed | contact_recovered
        self.support_stable_steps[reset_recovery] = 0
        stable = stop & reliable & ~reset_recovery
        self.support_stable_steps[stable] += 1
        self.filtered_signal[inactive] = 0.0
        self.applied_action[inactive] = 0.0

        heading_error = torch.atan2(policy_observation[:, 135], policy_observation[:, 136])
        actual_yaw_rate = policy_observation[:, 5]
        legacy_yaw_command = policy_observation[:, 11]
        heading_term = cfg.k_heading * heading_error
        damping_term = cfg.k_yaw_rate * (legacy_yaw_command - actual_yaw_rate)
        raw = heading_term + damping_term

        yaw_abs = actual_yaw_rate.abs()
        finite_guard = torch.isfinite(torch.tensor(cfg.yaw_hard_threshold, device=self.device))
        if bool(finite_guard.item()):
            guard_scale = ((cfg.yaw_hard_threshold - yaw_abs) / max(
                cfg.yaw_hard_threshold - cfg.yaw_soft_threshold, 1.0e-6
            )).clamp(0.0, 1.0)
            hard = yaw_abs >= cfg.yaw_hard_threshold
            if cfg.hard_guard_mode == "damping_only":
                # Preserve damping while continuously withdrawing the heading
                # drive as yaw rate approaches the hard threshold.
                guarded = heading_term * guard_scale + damping_term
            else:
                guarded = raw * guard_scale
        else:
            guard_scale = torch.ones_like(raw)
            hard = torch.zeros_like(stop)
            guarded = raw
        spike_active = stop & (guard_scale < 1.0)

        if ankle_utilization is None:
            ankle_utilization = torch.zeros(self.num_envs, 2, device=self.device)
        if joint_velocity_utilization is None:
            joint_velocity_utilization = torch.zeros(self.num_envs, device=self.device)
        if roll_pitch is None:
            roll_pitch = torch.zeros(self.num_envs, 2, device=self.device)
        if angular_velocity is None:
            angular_velocity = policy_observation[:, 3:6]
        supported_ankle = torch.where(contacts, ankle_utilization, torch.zeros_like(ankle_utilization)).amax(dim=1)
        ankle_scale = self._soft_safety_scale(
            supported_ankle, cfg.ankle_utilization_soft, cfg.ankle_utilization_hard
        )
        joint_scale = self._soft_safety_scale(
            joint_velocity_utilization, cfg.joint_velocity_soft, cfg.joint_velocity_hard
        )
        tilt = torch.linalg.vector_norm(roll_pitch, dim=1)
        tilt_scale = self._soft_safety_scale(tilt, cfg.tilt_soft_rad, cfg.tilt_hard_rad)
        angular_speed = torch.linalg.vector_norm(angular_velocity, dim=1)
        angular_scale = self._soft_safety_scale(
            angular_speed, cfg.angular_velocity_soft_rps, cfg.angular_velocity_hard_rps
        )
        safety_scale = torch.minimum(torch.minimum(ankle_scale, joint_scale), torch.minimum(tilt_scale, angular_scale))
        yaw_worsening = stop & (yaw_abs >= cfg.yaw_soft_threshold) & (
            yaw_abs > self.previous_yaw_abs + 1.0e-4
        )
        response_scale = torch.where(
            yaw_worsening, torch.full_like(raw, cfg.worsening_yaw_scale), torch.ones_like(raw)
        )

        phase_scale = torch.where(
            hold_progress > 0.0,
            torch.full_like(raw, cfg.hold_scale),
            torch.full_like(raw, cfg.braking_scale),
        )
        contact_scale = torch.where(
            support_count >= 2,
            torch.full_like(raw, cfg.double_support_scale),
            torch.where(
                support_count == 1,
                torch.full_like(raw, cfg.single_support_scale),
                torch.full_like(raw, cfg.flight_scale),
            ),
        )
        recovery_steps = self.support_stable_steps.to(raw.dtype)
        after_zero = (recovery_steps - float(cfg.contact_recovery_zero_steps)).clamp_min(0.0)
        if cfg.contact_recovery_ramp_steps > 0:
            recovery_scale = (after_zero / float(cfg.contact_recovery_ramp_steps)).clamp(0.0, 1.0)
        else:
            recovery_scale = (recovery_steps >= cfg.contact_recovery_zero_steps).to(raw.dtype)
        recovery_scale = torch.where(reliable, recovery_scale, torch.zeros_like(recovery_scale))
        target = guarded * phase_scale * contact_scale * recovery_scale * safety_scale * response_scale
        filtered = (1.0 - cfg.alpha) * self.filtered_signal + cfg.alpha * target
        self.filtered_signal[stop] = filtered[stop]
        self.filtered_signal[entering] = cfg.alpha * target[entering]

        desired = torch.zeros_like(self.applied_action)
        signal = self.filtered_signal
        desired[:, 2] = -0.60 * signal
        desired[:, 7] = -1.00 * signal
        desired[:, 8] = -0.50 * signal
        desired.clamp_(-cfg.action_limit, cfg.action_limit)
        hard_limit = torch.where(
            hard, torch.full_like(signal, cfg.hard_guard_action_limit), torch.full_like(signal, cfg.action_limit)
        )
        desired = torch.maximum(torch.minimum(desired, hard_limit.unsqueeze(1)), -hard_limit.unsqueeze(1))
        if cfg.hard_guard_disable_torso:
            desired[hard, 2] = 0.0
        desired[inactive] = 0.0
        requested_change = desired - self.applied_action
        limited_change = requested_change.clamp(-cfg.max_delta_per_step, cfg.max_delta_per_step)
        applied = (self.applied_action + limited_change).clamp(-cfg.action_limit, cfg.action_limit)
        # Hard safety caps are downstream of slew limiting.  They may reduce an
        # unsafe action immediately rather than preserving a stale correction.
        applied = torch.maximum(torch.minimum(applied, hard_limit.unsqueeze(1)), -hard_limit.unsqueeze(1))
        if cfg.hard_guard_disable_torso:
            applied[hard, 2] = 0.0
        # Flight/unreliable support is an invariant, not a filtered target.
        # Clear every state so stale feedback cannot reappear at touchdown.
        flight = stop & ~reliable
        if cfg.flight_hard_zero:
            applied[flight] = 0.0
            self.filtered_signal[flight] = 0.0
            self.support_stable_steps[flight] = 0
        applied[inactive] = 0.0
        slew_active = stop & (requested_change.abs().amax(dim=1) > cfg.max_delta_per_step + 1.0e-9)
        self.applied_action.copy_(applied)
        self.was_stop.copy_(stop)
        self.previous_contacts.copy_(torch.where(stop.unsqueeze(1), contacts, torch.zeros_like(contacts)))
        self.previous_yaw_abs.copy_(torch.where(stop, yaw_abs, torch.zeros_like(yaw_abs)))
        diagnostics = {
            "raw_signal": raw,
            "filtered_signal": self.filtered_signal.clone(),
            "feedback_action": applied.clone(),
            "feedback_norm": torch.linalg.vector_norm(applied, dim=1),
            "spike_guard_active": spike_active,
            "hard_guard_active": stop & hard,
            "slew_limiter_active": slew_active,
            "phase_scale": phase_scale,
            "contact_scale": contact_scale,
            "support_count": support_count,
            "support_stable_steps": self.support_stable_steps.clone(),
            "contact_recovery_scale": recovery_scale,
            "ankle_safety_scale": ankle_scale,
            "joint_velocity_safety_scale": joint_scale,
            "tilt_safety_scale": tilt_scale,
            "angular_velocity_safety_scale": angular_scale,
            "combined_safety_scale": safety_scale,
            "yaw_worsening_guard_active": yaw_worsening,
        }
        return applied, diagnostics
