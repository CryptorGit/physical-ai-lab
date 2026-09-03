"""Stage 5 command curriculum and non-rewarding high-speed diagnostics."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .stage3_mdp import FocusedRunVelocityCommand

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


class Stage5ProgressiveVelocityCommand(FocusedRunVelocityCommand):
    """Sample 70% of commands from 3.6 m/s to the current curriculum ceiling."""

    focus_low = 3.6

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.target_band_steps = torch.zeros(self.num_envs, device=self.device)
        self.command_steps = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # UniformVelocityCommand supplies lateral/yaw commands and standing masks.
        super(FocusedRunVelocityCommand, self)._resample_command(env_ids)
        count = len(env_ids)
        if count == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        low, high = self.cfg.ranges.lin_vel_x
        focus_low = min(max(self.focus_low, low), high)
        use_focus = torch.rand(count, device=self.device) < self.focus_probability
        focused = torch.empty(count, device=self.device).uniform_(focus_low, high)
        full = torch.empty(count, device=self.device).uniform_(low, high)
        self.vel_command_b[env_ids_tensor, 0] = torch.where(use_focus, focused, full)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        upper = float(self.cfg.ranges.lin_vel_x[1])
        self.target_band_steps += (self.vel_command_b[:, 0] >= upper - 0.05).float()
        self.command_steps += 1.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        extras = super().reset(env_ids)
        self.target_band_steps[env_ids] = 0.0
        self.command_steps[env_ids] = 0.0
        return extras


class ProgressiveSpeedCurriculum(ManagerTermBase):
    """Advance the command ceiling after sustained success near the current ceiling."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.command_term: Stage5ProgressiveVelocityCommand = env.command_manager.get_term(
            cfg.params["command_name"]
        )
        reward_cfg = env.reward_manager.get_term_cfg(cfg.params["safe_reward_name"])
        self.safe_reward = reward_cfg.func
        diagnostic_cfg = env.reward_manager.get_term_cfg(cfg.params["diagnostic_reward_name"])
        self.diagnostics: HighSpeedRunningDiagnostics = diagnostic_cfg.func
        self.stage_upper_bounds = tuple(float(value) for value in cfg.params["stage_upper_bounds"])
        self.stage = int(cfg.params.get("initial_stage", 0))
        if not 0 <= self.stage < len(self.stage_upper_bounds):
            raise ValueError(f"initial_stage must be 0..{len(self.stage_upper_bounds) - 1}")
        self.command_term.cfg.ranges.lin_vel_x = (3.4, self.stage_upper_bounds[self.stage])
        self.outcomes: deque[bool] = deque(maxlen=int(cfg.params["window_size"]))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # CurriculumManager calls reset after every environment reset.  The rolling
        # success window is global training state and must persist across episodes.
        del env_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        safe_reward_name: str,
        diagnostic_reward_name: str,
        stage_upper_bounds: tuple[float, ...],
        success_threshold: float,
        min_samples: int,
        window_size: int,
        target_exposure_fraction: float,
        min_episode_fraction: float,
        min_safe_landings: int,
        max_slip_mps: float,
        max_velocity_limit_fraction: float,
        max_torque_limit_fraction: float,
        max_landing_impact_n: float,
        max_vertical_excursion_m: float,
        max_asymmetry: float,
        initial_stage: int = 0,
    ) -> dict[str, float]:
        del command_name, safe_reward_name, diagnostic_reward_name, stage_upper_bounds, window_size, initial_stage
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() > 0:
            episode_steps = env.episode_length_buf[ids]
            valid = episode_steps > 0
            if valid.any():
                ids = ids[valid]
                exposure = self.command_term.target_band_steps[ids] / torch.clamp(
                    self.command_term.command_steps[ids], min=1.0
                )
                tested = exposure >= target_exposure_fraction
                if tested.any():
                    tested_ids = ids[tested]
                    fell = env.termination_manager.get_term("base_contact")[tested_ids]
                    long_enough = (
                        env.episode_length_buf[tested_ids].float()
                        >= min_episode_fraction * float(env.max_episode_length)
                    )
                    safe_landings = self.safe_reward._counters["completion_reward_fires"][tested_ids]
                    steps = torch.clamp(self.diagnostics.steps[tested_ids], min=1.0)
                    slip = self.diagnostics.slip_sum[tested_ids] / torch.clamp(
                        self.diagnostics.slip_samples[tested_ids], min=1.0
                    )
                    velocity_fraction = (self.diagnostics.velocity_limit_steps[tested_ids] / steps[:, None]).amax(dim=1)
                    torque_fraction = (self.diagnostics.torque_limit_steps[tested_ids] / steps[:, None]).amax(dim=1)
                    impact = self.diagnostics.max_landing_impact[tested_ids].amax(dim=1)
                    vertical_excursion = (
                        self.diagnostics.max_base_height[tested_ids] - self.diagnostics.min_base_height[tested_ids]
                    )
                    stride = self.diagnostics.stride_sum[tested_ids] / torch.clamp(
                        self.diagnostics.stride_count[tested_ids], min=1.0
                    )
                    contact_time = self.diagnostics.contact_steps[tested_ids] / torch.clamp(
                        self.diagnostics.contact_events[tested_ids], min=1.0
                    )
                    stride_asym = (stride[:, 0] - stride[:, 1]).abs() / torch.clamp(stride.mean(dim=1), min=1.0e-6)
                    contact_asym = (contact_time[:, 0] - contact_time[:, 1]).abs() / torch.clamp(
                        contact_time.mean(dim=1), min=1.0e-6
                    )
                    physical_quality = (
                        (slip.amax(dim=1) <= max_slip_mps)
                        & (velocity_fraction <= max_velocity_limit_fraction)
                        & (torque_fraction <= max_torque_limit_fraction)
                        & (impact <= max_landing_impact_n)
                        & (vertical_excursion <= max_vertical_excursion_m)
                        & (stride_asym <= max_asymmetry)
                        & (contact_asym <= max_asymmetry)
                    )
                    success = (~fell) & long_enough & (safe_landings >= min_safe_landings) & physical_quality
                    self.outcomes.extend(bool(value) for value in success.detach().cpu().tolist())

        success_rate = sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0.0
        if (
            self.stage < len(self.stage_upper_bounds) - 1
            and len(self.outcomes) >= min_samples
            and success_rate >= success_threshold
        ):
            self.stage += 1
            self.outcomes.clear()
            success_rate = 0.0
            self.command_term.cfg.ranges.lin_vel_x = (3.4, self.stage_upper_bounds[self.stage])

        return {
            "stage": float(self.stage),
            "upper_speed_mps": self.stage_upper_bounds[self.stage],
            "target_samples": float(len(self.outcomes)),
            "target_success_rate": float(success_rate),
        }


class HighSpeedRunningDiagnostics(ManagerTermBase):
    """Collect saturation, contact, stride and symmetry metrics without adding reward."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.foot_ids = cfg.params["sensor_cfg"].body_ids
        if len(self.foot_ids) != 2:
            raise ValueError("HighSpeedRunningDiagnostics requires exactly two feet")
        self.num_joints = len(self.robot.joint_names)
        shape = (self.num_envs, self.num_joints)
        self.max_joint_speed = torch.zeros(shape, device=self.device)
        self.max_joint_speed_ratio = torch.zeros(shape, device=self.device)
        self.max_joint_torque = torch.zeros(shape, device=self.device)
        self.max_joint_torque_ratio = torch.zeros(shape, device=self.device)
        self.velocity_limit_steps = torch.zeros(shape, device=self.device)
        self.torque_limit_steps = torch.zeros(shape, device=self.device)
        self.steps = torch.zeros(self.num_envs, device=self.device)
        self.previous_contacts = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.previous_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_landing_xy = torch.zeros((self.num_envs, 2, 2), device=self.device)
        self.has_landing = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.stride_sum = torch.zeros((self.num_envs, 2), device=self.device)
        self.stride_count = torch.zeros((self.num_envs, 2), device=self.device)
        self.contact_steps = torch.zeros((self.num_envs, 2), device=self.device)
        self.contact_events = torch.zeros((self.num_envs, 2), device=self.device)
        self.flight_steps = torch.zeros(self.num_envs, device=self.device)
        self.flight_events = torch.zeros(self.num_envs, device=self.device)
        self.force_horizontal_sum = torch.zeros((self.num_envs, 2), device=self.device)
        self.force_vertical_sum = torch.zeros((self.num_envs, 2), device=self.device)
        self.force_samples = torch.zeros((self.num_envs, 2), device=self.device)
        self.max_landing_impact = torch.zeros((self.num_envs, 2), device=self.device)
        self.slip_sum = torch.zeros((self.num_envs, 2), device=self.device)
        self.slip_samples = torch.zeros((self.num_envs, 2), device=self.device)
        self.min_base_height = torch.full((self.num_envs,), torch.inf, device=self.device)
        self.max_base_height = torch.full((self.num_envs,), -torch.inf, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        log = self._env.extras.setdefault("log", {})
        steps = torch.clamp(self.steps[env_ids], min=1.0)
        for joint_id, joint_name in enumerate(self.robot.joint_names):
            prefix = f"Diagnostics/high_speed/joints/{joint_name}"
            log[f"{prefix}/max_speed_rad_s"] = self.max_joint_speed[env_ids, joint_id].mean().item()
            log[f"{prefix}/max_speed_limit_ratio"] = self.max_joint_speed_ratio[env_ids, joint_id].mean().item()
            log[f"{prefix}/velocity_limit_fraction"] = (
                self.velocity_limit_steps[env_ids, joint_id] / steps
            ).mean().item()
            log[f"{prefix}/max_torque_nm"] = self.max_joint_torque[env_ids, joint_id].mean().item()
            log[f"{prefix}/max_torque_limit_ratio"] = self.max_joint_torque_ratio[env_ids, joint_id].mean().item()
            log[f"{prefix}/torque_limit_fraction"] = (
                self.torque_limit_steps[env_ids, joint_id] / steps
            ).mean().item()

        stride_mean = self.stride_sum[env_ids] / torch.clamp(self.stride_count[env_ids], min=1.0)
        contact_time = self.contact_steps[env_ids] * self._env.step_dt / torch.clamp(
            self.contact_events[env_ids], min=1.0
        )
        force_h = self.force_horizontal_sum[env_ids] / torch.clamp(self.force_samples[env_ids], min=1.0)
        force_z = self.force_vertical_sum[env_ids] / torch.clamp(self.force_samples[env_ids], min=1.0)
        slip = self.slip_sum[env_ids] / torch.clamp(self.slip_samples[env_ids], min=1.0)
        for foot_id, foot_name in enumerate(("left", "right")):
            prefix = f"Diagnostics/high_speed/feet/{foot_name}"
            log[f"{prefix}/stride_length_m"] = stride_mean[:, foot_id].mean().item()
            log[f"{prefix}/contact_time_s"] = contact_time[:, foot_id].mean().item()
            log[f"{prefix}/horizontal_force_n"] = force_h[:, foot_id].mean().item()
            log[f"{prefix}/vertical_force_n"] = force_z[:, foot_id].mean().item()
            log[f"{prefix}/max_landing_impact_n"] = self.max_landing_impact[env_ids, foot_id].mean().item()
            log[f"{prefix}/contact_slip_mps"] = slip[:, foot_id].mean().item()
        duration_s = steps * self._env.step_dt
        log["Diagnostics/high_speed/step_frequency_hz"] = (
            self.contact_events[env_ids].sum(dim=1) / duration_s
        ).mean().item()
        log["Diagnostics/high_speed/mean_flight_time_s"] = (
            self.flight_steps[env_ids] * self._env.step_dt / torch.clamp(self.flight_events[env_ids], min=1.0)
        ).mean().item()
        log["Diagnostics/high_speed/base_vertical_excursion_m"] = (
            self.max_base_height[env_ids] - self.min_base_height[env_ids]
        ).nan_to_num().mean().item()
        stride_asym = (stride_mean[:, 0] - stride_mean[:, 1]).abs() / torch.clamp(
            stride_mean.mean(dim=1), min=1.0e-6
        )
        contact_asym = (contact_time[:, 0] - contact_time[:, 1]).abs() / torch.clamp(
            contact_time.mean(dim=1), min=1.0e-6
        )
        log["Diagnostics/high_speed/stride_asymmetry"] = stride_asym.mean().item()
        log["Diagnostics/high_speed/contact_time_asymmetry"] = contact_asym.mean().item()

        tensor_names = (
            "max_joint_speed", "max_joint_speed_ratio", "max_joint_torque", "max_joint_torque_ratio",
            "velocity_limit_steps", "torque_limit_steps", "steps", "previous_contacts", "previous_in_flight",
            "last_landing_xy", "has_landing", "stride_sum", "stride_count", "contact_steps", "contact_events",
            "flight_steps", "flight_events", "force_horizontal_sum", "force_vertical_sum", "force_samples",
            "max_landing_impact", "slip_sum", "slip_samples",
        )
        for name in tensor_names:
            getattr(self, name)[env_ids] = 0
        self.min_base_height[env_ids] = torch.inf
        self.max_base_height[env_ids] = -torch.inf

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        saturation_ratio: float = 0.95,
    ) -> torch.Tensor:
        del asset_cfg, sensor_cfg
        velocity = self.robot.data.joint_vel.torch.abs()
        velocity_limit = torch.clamp(self.robot.data.joint_vel_limits.torch.abs(), min=1.0e-6)
        torque = self.robot.data.applied_torque.torch.abs()
        torque_limit = torch.clamp(self.robot.data.joint_effort_limits.torch.abs(), min=1.0e-6)
        velocity_ratio = velocity / velocity_limit
        torque_ratio = torque / torque_limit
        self.max_joint_speed = torch.maximum(self.max_joint_speed, velocity)
        self.max_joint_speed_ratio = torch.maximum(self.max_joint_speed_ratio, velocity_ratio)
        self.max_joint_torque = torch.maximum(self.max_joint_torque, torque)
        self.max_joint_torque_ratio = torch.maximum(self.max_joint_torque_ratio, torque_ratio)
        self.velocity_limit_steps += (velocity_ratio >= saturation_ratio).float()
        self.torque_limit_steps += (torque_ratio >= saturation_ratio).float()
        self.steps += 1.0

        forces = self.contact_sensor.data.net_forces_w_history.torch[:, :, self.foot_ids, :]
        current_force = forces[:, 0]
        contacts = forces.norm(dim=-1).amax(dim=1) > 1.0
        first_contact = contacts & ~self.previous_contacts
        in_flight = contacts.sum(dim=1) == 0
        self.contact_steps += contacts.float()
        self.contact_events += first_contact.float()
        self.flight_steps += in_flight.float()
        self.flight_events += (in_flight & ~self.previous_in_flight).float()
        horizontal_force = torch.linalg.norm(current_force[:, :, :2], dim=-1)
        vertical_force = current_force[:, :, 2].abs()
        self.force_horizontal_sum += horizontal_force * contacts
        self.force_vertical_sum += vertical_force * contacts
        self.force_samples += contacts.float()
        self.max_landing_impact = torch.maximum(
            self.max_landing_impact, vertical_force * first_contact
        )
        foot_velocity = self.robot.data.body_lin_vel_w.torch[:, self.foot_ids, :2].norm(dim=-1)
        self.slip_sum += foot_velocity * contacts
        self.slip_samples += contacts.float()
        foot_xy = self.robot.data.body_pos_w.torch[:, self.foot_ids, :2]
        for foot_id in range(2):
            landed = first_contact[:, foot_id]
            has_previous = landed & self.has_landing[:, foot_id]
            stride = torch.linalg.norm(foot_xy[:, foot_id] - self.last_landing_xy[:, foot_id], dim=-1)
            self.stride_sum[:, foot_id] += stride * has_previous
            self.stride_count[:, foot_id] += has_previous.float()
            self.last_landing_xy[landed, foot_id] = foot_xy[landed, foot_id]
            self.has_landing[landed, foot_id] = True
        base_height = self.robot.data.root_pos_w.torch[:, 2]
        self.min_base_height = torch.minimum(self.min_base_height, base_height)
        self.max_base_height = torch.maximum(self.max_base_height, base_height)
        self.previous_contacts.copy_(contacts)
        self.previous_in_flight.copy_(in_flight)
        return torch.zeros(self.num_envs, device=self.device)
