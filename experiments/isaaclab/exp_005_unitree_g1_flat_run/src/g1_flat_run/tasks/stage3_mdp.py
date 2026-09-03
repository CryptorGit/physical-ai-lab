"""Stage 3 command and reward terms for periodic G1 running."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor


class FocusedRunVelocityCommand(UniformVelocityCommand):
    """Sample 70% of forward commands from the middle third of the configured range.

    Stage 3 configures the full range as 2.3--2.6 m/s, so the focused range is
    2.4--2.5 m/s.  Lateral and yaw commands retain the official uniform sampler.
    """

    focus_probability = 0.7

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        count = len(env_ids)
        if count == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        low, high = self.cfg.ranges.lin_vel_x
        third = (high - low) / 3.0
        use_focus = torch.rand(count, device=self.device) < self.focus_probability
        focused = torch.empty(count, device=self.device).uniform_(low + third, high - third)
        full = torch.empty(count, device=self.device).uniform_(low, high)
        self.vel_command_b[env_ids_tensor, 0] = torch.where(use_focus, focused, full)


class SafePeriodicFlightReward(ManagerTermBase):
    """Reward a short flight only after a safe, alternating single-foot landing.

    The sparse reward is emitted on landing, never while airborne.  A first landing
    establishes phase but earns no reward; the next landing must be on the other foot.
    """

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.foot_ids = cfg.params["sensor_cfg"].body_ids
        if len(self.foot_ids) != 2:
            raise ValueError(f"SafePeriodicFlightReward requires exactly two feet, got {len(self.foot_ids)}")

        self._was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._flight_duration = torch.zeros(self.num_envs, device=self.device)
        self._event_precursor_reward = torch.zeros(self.num_envs, device=self.device)
        self._last_landing_foot = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.last_raw_reward = torch.zeros(self.num_envs, device=self.device)
        counter_names = (
            "flight_events",
            "completed_flights",
            "duration_20ms",
            "duration_40ms",
            "duration_60ms",
            "duration_ge_80ms",
            "landing_left",
            "landing_right",
            "same_side_landing",
            "double_foot_landing",
            "reward_fires",
            "raw_reward_value",
            "weighted_reward_value",
            "precursor_reward_steps",
            "precursor_raw_reward_value",
            "completion_reward_fires",
            "completion_raw_reward_value",
            "excess_flight_penalty_steps",
            "reset_phase_initialization",
            "reset_not_single_foot",
            "reset_flight_too_short",
            "reset_flight_too_long",
            "reset_command_too_slow",
            "reset_tracking_error",
            "reset_torso_tilt",
            "reset_vertical_speed",
            "reset_same_side_landing",
        )
        condition_names = (
            "high_command",
            "tracking",
            "torso_tilt",
            "vertical_speed",
            "short_flight",
            "single_foot",
            "alternating",
        )
        self._counters = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in counter_names
        }
        for name in condition_names:
            self._counters[f"pass_{name}"] = torch.zeros(self.num_envs, device=self.device)
            self._counters[f"fail_{name}"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        log = self._env.extras.setdefault("log", {})
        for name, values in self._counters.items():
            log[f"Diagnostics/safe_periodic_flight/{name}"] = torch.mean(values[env_ids]).item()
            values[env_ids] = 0.0
        self._was_in_flight[env_ids] = False
        self._flight_duration[env_ids] = 0.0
        self._event_precursor_reward[env_ids] = 0.0
        self._last_landing_foot[env_ids] = -1
        self.last_raw_reward[env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        min_command_speed: float,
        max_tracking_error: float,
        max_torso_tilt_rad: float,
        max_vertical_speed: float,
        min_flight_time: float,
        max_flight_time: float,
        precursor_reward_per_step: float = 0.0,
        takeoff_precursor_reward_per_step: float = 0.0,
        precursor_event_cap: float = 0.0,
        precursor_min_flight_time: float = 0.0,
        precursor_max_tracking_error: float = 0.30,
        completion_reward: float = 1.0,
        excess_flight_penalty_per_step: float = 0.0,
        use_yaw_frame_tracking: bool = False,
        continuous_tracking_decay: bool = False,
        tracking_forward_scale_mps: float = 0.30,
        tracking_lateral_scale_mps: float = 0.20,
    ) -> torch.Tensor:
        del asset_cfg, sensor_cfg
        forces = self.contact_sensor.data.net_forces_w_history.torch[:, :, self.foot_ids, :]
        contacts = forces.norm(dim=-1).amax(dim=1) > 1.0
        contact_count = contacts.sum(dim=1)
        in_flight = contact_count == 0
        flight_started = in_flight & ~self._was_in_flight
        landing = self._was_in_flight & ~in_flight
        self._counters["flight_events"] += flight_started.float()
        self._event_precursor_reward[flight_started] = 0.0

        self._flight_duration[in_flight] += env.step_dt
        landing_foot = contacts.to(torch.int64).argmax(dim=1)
        single_foot_landing = landing & (contact_count == 1)
        alternating = (
            single_foot_landing
            & (self._last_landing_foot >= 0)
            & (landing_foot != self._last_landing_foot)
        )

        command_xy = env.command_manager.get_command(command_name)[:, :2]
        command = command_xy[:, 0]
        body_frame_speed = self.robot.data.root_lin_vel_b.torch[:, 0]
        yaw_frame_velocity = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w.torch), self.robot.data.root_lin_vel_w.torch[:, :3]
        )
        yaw_frame_speed = yaw_frame_velocity[:, 0]
        actual_speed = yaw_frame_speed if use_yaw_frame_tracking else body_frame_speed
        forward_error = actual_speed - command
        lateral_error = yaw_frame_velocity[:, 1] - command_xy[:, 1]
        tracking_quality = torch.exp(
            -torch.square(forward_error / max(tracking_forward_scale_mps, 1.0e-6))
            -torch.square(lateral_error / max(tracking_lateral_scale_mps, 1.0e-6))
        )
        torso_tilt = torch.acos(torch.clamp(-self.robot.data.projected_gravity_b.torch[:, 2], -1.0, 1.0))
        vertical_speed = self.robot.data.root_lin_vel_b.torch[:, 2].abs()
        short_flight = (self._flight_duration >= min_flight_time - 1.0e-6) & (
            self._flight_duration <= max_flight_time + 1.0e-6
        )

        high_command = command >= min_command_speed
        tracking = forward_error.abs() <= max_tracking_error
        cycle_tracking = torch.ones_like(tracking) if continuous_tracking_decay else tracking
        stable_torso = torso_tilt <= max_torso_tilt_rad
        bounded_vertical_speed = vertical_speed <= max_vertical_speed
        has_previous_landing = self._last_landing_foot >= 0

        common_precursor_safety = (
            in_flight
            & high_command
            & stable_torso
            & bounded_vertical_speed
            & (self._flight_duration <= max_flight_time + 1.0e-6)
        )
        sustained_precursor = (
            common_precursor_safety
            & (
                torch.ones_like(tracking)
                if continuous_tracking_decay
                else (forward_error.abs() <= precursor_max_tracking_error)
            )
            & (self._flight_duration >= precursor_min_flight_time - 1.0e-6)
        )
        safe_takeoff = (
            common_precursor_safety
            & cycle_tracking
            & (self._flight_duration < precursor_min_flight_time - 1.0e-6)
        )
        remaining_cap = torch.clamp(precursor_event_cap - self._event_precursor_reward, min=0.0)
        requested_precursor = torch.where(
            sustained_precursor,
            torch.full_like(remaining_cap, precursor_reward_per_step),
            torch.where(
                safe_takeoff,
                torch.full_like(remaining_cap, takeoff_precursor_reward_per_step),
                torch.zeros_like(remaining_cap),
            ),
        )
        if continuous_tracking_decay:
            requested_precursor *= tracking_quality
        precursor_reward = torch.minimum(
            requested_precursor, remaining_cap
        )
        self._event_precursor_reward += precursor_reward
        excess_flight = in_flight & (self._flight_duration > max_flight_time + 1.0e-6)
        excess_penalty = excess_flight.float() * excess_flight_penalty_per_step
        self._counters["precursor_reward_steps"] += (precursor_reward > 0.0).float()
        self._counters["precursor_raw_reward_value"] += precursor_reward
        self._counters["excess_flight_penalty_steps"] += excess_flight.float()

        if landing.any():
            self._counters["completed_flights"] += landing.float()
            duration_steps = torch.round(self._flight_duration / env.step_dt).to(torch.long)
            self._counters["duration_20ms"] += (landing & (duration_steps == 1)).float()
            self._counters["duration_40ms"] += (landing & (duration_steps == 2)).float()
            self._counters["duration_60ms"] += (landing & (duration_steps == 3)).float()
            self._counters["duration_ge_80ms"] += (landing & (duration_steps >= 4)).float()
            self._counters["landing_left"] += (single_foot_landing & (landing_foot == 0)).float()
            self._counters["landing_right"] += (single_foot_landing & (landing_foot == 1)).float()
            self._counters["same_side_landing"] += (
                single_foot_landing & has_previous_landing & (landing_foot == self._last_landing_foot)
            ).float()
            self._counters["double_foot_landing"] += (landing & (contact_count == 2)).float()

            conditions = {
                "high_command": high_command,
                "tracking": tracking,
                "torso_tilt": stable_torso,
                "vertical_speed": bounded_vertical_speed,
                "short_flight": short_flight,
                "single_foot": single_foot_landing,
                "alternating": alternating,
            }
            for name, condition in conditions.items():
                self._counters[f"pass_{name}"] += (landing & condition).float()
                self._counters[f"fail_{name}"] += (landing & ~condition).float()

        safe_landing = (
            alternating
            & high_command
            & cycle_tracking
            & stable_torso
            & bounded_vertical_speed
            & short_flight
        )

        unresolved_reset = landing & ~safe_landing
        reset_reasons = (
            ("reset_not_single_foot", ~single_foot_landing),
            ("reset_flight_too_short", self._flight_duration < min_flight_time - 1.0e-6),
            ("reset_flight_too_long", self._flight_duration > max_flight_time + 1.0e-6),
            ("reset_command_too_slow", ~high_command),
            (
                "reset_tracking_error",
                torch.zeros_like(tracking) if continuous_tracking_decay else ~tracking,
            ),
            ("reset_torso_tilt", ~stable_torso),
            ("reset_vertical_speed", ~bounded_vertical_speed),
            ("reset_phase_initialization", ~has_previous_landing),
            ("reset_same_side_landing", has_previous_landing & ~alternating),
        )
        for name, reason in reset_reasons:
            selected = unresolved_reset & reason
            self._counters[name] += selected.float()
            unresolved_reset &= ~selected

        completion_value = safe_landing.float() * completion_reward
        if continuous_tracking_decay:
            completion_value *= tracking_quality
        self.last_raw_reward.copy_(precursor_reward + completion_value - excess_penalty)
        self._counters["completion_reward_fires"] += safe_landing.float()
        self._counters["completion_raw_reward_value"] += completion_value
        self._counters["reward_fires"] += (self.last_raw_reward != 0.0).float()
        self._counters["raw_reward_value"] += self.last_raw_reward
        self._counters["weighted_reward_value"] += self.last_raw_reward * self.cfg.weight * env.step_dt

        self._last_landing_foot[single_foot_landing] = landing_foot[single_foot_landing]
        self._flight_duration[landing] = 0.0
        self._was_in_flight.copy_(in_flight)
        return self.last_raw_reward
