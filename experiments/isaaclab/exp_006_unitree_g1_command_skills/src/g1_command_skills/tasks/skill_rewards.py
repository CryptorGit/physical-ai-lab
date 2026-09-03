"""Command-gated rewards for RUN, TURN, STOP and CROUCH."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from g1_flat_run.tasks.stage3_mdp import SafePeriodicFlightReward

from .command_mdp import SkillId


def _term(env, command_name: str):
    return env.command_manager.get_term(command_name)


class GatedSafePeriodicFlightReward(SafePeriodicFlightReward):
    """Preserve Stage-4 periodic-flight state while activating it only for RUN."""

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
        reward = super().__call__(
            env,
            command_name,
            asset_cfg,
            sensor_cfg,
            min_command_speed,
            max_tracking_error,
            max_torso_tilt_rad,
            max_vertical_speed,
            min_flight_time,
            max_flight_time,
            precursor_reward_per_step,
            takeoff_precursor_reward_per_step,
            precursor_event_cap,
            precursor_min_flight_time,
            precursor_max_tracking_error,
            completion_reward,
            excess_flight_penalty_per_step,
            use_yaw_frame_tracking,
            continuous_tracking_decay,
            tracking_forward_scale_mps,
            tracking_lateral_scale_mps,
        )
        return reward * _term(env, command_name).blend_weight((SkillId.RUN,))


def run_tracking(env, command_name: str, speed_std: float, heading_std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    term = _term(env, command_name)
    del asset_cfg
    speed_error = term.path_forward_velocity - term.target_speed
    reward = torch.exp(-speed_error.square() / speed_std**2)
    reward *= torch.exp(-term.heading_error.square() / heading_std**2)
    return reward * term.blend_weight((SkillId.RUN,))


def run_path_lateral_error(
    env, command_name: str, dead_zone_m: float, error_scale_m: float
) -> torch.Tensor:
    """Squared hinge outside a centerline dead zone, allowing natural gait sway."""
    term = _term(env, command_name)
    excess = torch.relu(term.path_lateral_error.abs() - dead_zone_m)
    penalty = (excess / error_scale_m).square()
    return penalty * term.blend_weight((SkillId.RUN,))


def run_path_lateral_velocity(env, command_name: str, velocity_scale_mps: float) -> torch.Tensor:
    term = _term(env, command_name)
    penalty = (term.path_lateral_velocity / velocity_scale_mps).square()
    return penalty * term.blend_weight((SkillId.RUN,))


class RunPathRecoveryProgressReward(ManagerTermBase):
    """Reward reduction of lateral error outside the dead zone, not absolute position."""

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_effective_error = torch.zeros(self.num_envs, device=self.device)
        self._was_run = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_effective_error[env_ids] = 0.0
        self._was_run[env_ids] = False

    def __call__(
        self,
        env,
        command_name: str,
        dead_zone_m: float,
        progress_scale_m_per_step: float,
    ) -> torch.Tensor:
        del env
        term = _term(self._env, command_name)
        is_run = term.skill_id == int(SkillId.RUN)
        effective = torch.relu(term.path_lateral_error.abs() - dead_zone_m)
        progress = (self._previous_effective_error - effective) / progress_scale_m_per_step
        progress = progress.clamp(-1.0, 1.0)
        progress = torch.where(is_run & self._was_run, progress, torch.zeros_like(progress))
        self._previous_effective_error.copy_(effective.detach())
        self._was_run.copy_(is_run)
        return progress * term.blend_weight((SkillId.RUN,))


def turn_tracking(env, command_name: str, speed_std: float, heading_std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    speed_error = robot.data.root_lin_vel_b.torch[:, 0] - term.target_speed
    reward = torch.exp(-speed_error.square() / speed_std**2)
    reward *= torch.exp(-term.heading_error.square() / heading_std**2)
    return reward * term.blend_weight((SkillId.TURN,))


def turn_side_slip(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_b.torch[:, 1].abs() * term.blend_weight((SkillId.TURN,))


def stop_position_velocity(
    env, command_name: str, position_std: float, speed_std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    position_error = torch.linalg.norm(term.target_displacement_b, dim=1)
    speed = torch.linalg.norm(robot.data.root_lin_vel_b.torch[:, :2], dim=1)
    reward = torch.exp(-position_error.square() / position_std**2)
    reward *= torch.exp(-speed.square() / speed_std**2)
    return reward * term.blend_weight((SkillId.STOP,))


class StopProgressReward(ManagerTermBase):
    """Dense reward for reducing positive distance to the fixed STOP goal."""

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_remaining = torch.zeros(self.num_envs, device=self.device)
        self._was_stop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_remaining[env_ids] = 0.0
        self._was_stop[env_ids] = False

    def __call__(self, env, command_name: str, progress_scale_m_per_step: float) -> torch.Tensor:
        del env
        term = _term(self._env, command_name)
        is_stop = term.skill_id == int(SkillId.STOP)
        remaining = torch.relu(term.target_displacement_b[:, 0])
        progress = ((self._previous_remaining - remaining) / progress_scale_m_per_step).clamp(-1.0, 1.0)
        progress = torch.where(is_stop & self._was_stop, progress, torch.zeros_like(progress))
        self._previous_remaining.copy_(remaining.detach())
        self._was_stop.copy_(is_stop)
        return progress * term.blend_weight((SkillId.STOP,))


def stop_braking_speed_tracking(
    env, command_name: str, speed_std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = robot.data.root_lin_vel_b.torch[:, 0] - term.stop_braking_target_speed
    return torch.exp(-error.square() / speed_std**2) * term.blend_weight((SkillId.STOP,))


def stop_overshoot(env, command_name: str, scale_m: float) -> torch.Tensor:
    term = _term(env, command_name)
    return (torch.relu(-term.target_displacement_b[:, 0]) / scale_m).square() * term.blend_weight((SkillId.STOP,))


def stop_early(
    env,
    command_name: str,
    speed_margin_mps: float,
    scale_mps: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    forward_speed = robot.data.root_lin_vel_b.torch[:, 0].clamp_min(0.0)
    deficit = torch.relu(term.stop_braking_target_speed - speed_margin_mps - forward_speed)
    return (deficit / scale_mps).square() * term.blend_weight((SkillId.STOP,))


def stop_heading(env, command_name: str, heading_std: float) -> torch.Tensor:
    term = _term(env, command_name)
    return torch.exp(-term.heading_error.square() / heading_std**2) * term.blend_weight((SkillId.STOP,))


def stop_heading_tail(env, command_name: str, dead_zone_rad: float, scale_rad: float) -> torch.Tensor:
    """Emphasize p95/max heading excursions without replacing dense tracking."""
    term = _term(env, command_name)
    tail = torch.relu(term.heading_error.abs() - dead_zone_rad) / scale_rad
    return tail.square() * term.blend_weight((SkillId.STOP,))


def stop_yaw_rate_tracking(
    env, command_name: str, rate_scale_rps: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize disagreement with the frozen base actor's legacy yaw command."""
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = robot.data.root_ang_vel_b.torch[:, 2] - term.vel_command_b[:, 2]
    return (error / rate_scale_rps).square() * term.blend_weight((SkillId.STOP,))


def stop_attitude_stability(
    env,
    command_name: str,
    tilt_scale: float,
    angular_rate_scale_rps: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[:, :2], dim=1)
    roll_pitch_rate = torch.linalg.vector_norm(robot.data.root_ang_vel_b.torch[:, :2], dim=1)
    penalty = (tilt / tilt_scale).square() + (roll_pitch_rate / angular_rate_scale_rps).square()
    return penalty * term.blend_weight((SkillId.STOP,))


def stop_instability_tail(
    env,
    command_name: str,
    tilt_threshold: float,
    yaw_rate_threshold_rps: float,
    scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Focus learning on the high-tilt/high-yaw-rate precursor to STOP falls."""
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[:, :2], dim=1)
    yaw_rate = robot.data.root_ang_vel_b.torch[:, 2].abs()
    danger = torch.relu(tilt - tilt_threshold) + torch.relu(yaw_rate - yaw_rate_threshold_rps)
    return (danger / scale).square() * term.blend_weight((SkillId.STOP,))


def stop_hold_heading(env, command_name: str, heading_std: float) -> torch.Tensor:
    term = _term(env, command_name)
    reward = torch.exp(-term.heading_error.square() / heading_std**2) * term.stop_hold_progress
    return reward * term.blend_weight((SkillId.STOP,))


def stop_parent_action_deviation(env, command_name: str) -> torch.Tensor:
    """Penalize the squared corrective delta from the frozen model_31 action."""
    term = _term(env, command_name)
    # Imported lazily to keep task discovery independent from the RSL-RL model import.
    from g1_command_skills.models.residual_actor import latest_stop_correction

    correction = latest_stop_correction().get("parent_action_deviation")
    if correction is None or correction.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=term.skill_id.device)
    penalty = correction.to(device=term.skill_id.device).square().sum(dim=-1)
    return penalty * term.blend_weight((SkillId.STOP,))


def stop_hold(env, command_name: str) -> torch.Tensor:
    term = _term(env, command_name)
    return term.stop_hold_progress * term.blend_weight((SkillId.STOP,))


def stop_joint_saturation(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    quantity: str,
    threshold: float,
    scale: float,
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    if quantity == "velocity":
        value, limit = robot.data.joint_vel.torch.abs(), robot.data.joint_vel_limits.torch.abs()
    elif quantity == "torque":
        value, limit = robot.data.applied_torque.torch.abs(), robot.data.joint_effort_limits.torch.abs()
    else:
        raise ValueError(f"Unknown saturation quantity: {quantity}")
    ratio = value[:, asset_cfg.joint_ids] / limit[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    penalty = (torch.relu(ratio - threshold) / scale).square().mean(dim=1)
    return penalty * term.blend_weight((SkillId.STOP,))


def stop_foot_slip(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1) > 1.0
    speed = robot.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    slip = (speed * contacts).sum(dim=1) / contacts.sum(dim=1).clamp_min(1)
    return slip * term.blend_weight((SkillId.STOP,))


def stop_impact(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold_n: float,
    force_scale_n: float,
) -> torch.Tensor:
    term = _term(env, command_name)
    sensor = env.scene.sensors[sensor_cfg.name]
    vertical = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, 2].abs().mean(dim=1)
    penalty = (torch.relu(vertical - force_threshold_n) / force_scale_n).square().mean(dim=1)
    return penalty * term.blend_weight((SkillId.STOP,))


def skill_upright(env, command_name: str, skill_ids: tuple[int, ...], tilt_std: float, asset_cfg: SceneEntityCfg):
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    tilt = torch.linalg.norm(robot.data.projected_gravity_b.torch[:, :2], dim=1)
    return torch.exp(-tilt.square() / tilt_std**2) * term.blend_weight(skill_ids)


def crouch_height_tracking(
    env, command_name: str, height_std_m: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = robot.data.root_pos_w.torch[:, 2] - term.target_pelvis_height
    return torch.exp(-error.square() / height_std_m**2) * term.blend_weight((SkillId.CROUCH,))


def crouch_vertical_velocity_tracking(
    env, command_name: str, velocity_std_mps: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = robot.data.root_lin_vel_w.torch[:, 2] - term.target_vertical_velocity
    return torch.exp(-error.square() / velocity_std_mps**2) * term.blend_weight((SkillId.CROUCH,))


class CrouchDepthProgressReward(ManagerTermBase):
    """Reward actual relative depth progress during DOWN, never raw low height."""

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        with torch.inference_mode(False):
            self._previous_depth = torch.zeros(self.num_envs, device=self.device)
            self._was_down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_depth[env_ids] = 0.0
        self._was_down[env_ids] = False

    def __call__(self, env, command_name: str, progress_scale_m_per_step: float) -> torch.Tensor:
        term = _term(env, command_name)
        robot = env.scene["robot"]
        is_down = (term.skill_id == int(SkillId.CROUCH)) & (term.crouch_phase == 1)
        depth = (term.crouch_entry_height - robot.data.root_pos_w.torch[:, 2]).clamp_min(0.0)
        progress = ((depth - self._previous_depth) / progress_scale_m_per_step).clamp(-1.0, 1.0)
        progress = torch.where(is_down & self._was_down, progress, torch.zeros_like(progress))
        self._previous_depth.copy_(depth.detach())
        self._was_down.copy_(is_down)
        return progress * term.blend_weight((SkillId.CROUCH,))


def crouch_hold_height(env, command_name: str, height_std_m: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = robot.data.root_pos_w.torch[:, 2] - (
        term.crouch_entry_height - term.crouch_commanded_drop
    )
    active = (term.crouch_phase == 2).float()
    return active * torch.exp(-error.square() / height_std_m**2) * term.blend_weight((SkillId.CROUCH,))


def crouch_return_height(env, command_name: str, height_std_m: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = robot.data.root_pos_w.torch[:, 2] - term.crouch_entry_height
    active = (term.crouch_phase == 4).float()
    return active * torch.exp(-error.square() / height_std_m**2) * term.blend_weight((SkillId.CROUCH,))


def crouch_joint_symmetry(
    env, command_name: str, asset_cfg: SceneEntityCfg, difference_scale_rad: float
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    values = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    if values.shape[1] != 6:
        raise ValueError("CROUCH symmetry expects ordered left/right hip, knee and ankle-pitch joints")
    difference = torch.stack((values[:, 0] - values[:, 1], values[:, 2] - values[:, 3], values[:, 4] - values[:, 5]), dim=1)
    return (difference / difference_scale_rad).square().mean(dim=1) * term.blend_weight((SkillId.CROUCH,))


def crouch_foot_contact(
    env, command_name: str, sensor_cfg: SceneEntityCfg, force_threshold_n: float
) -> torch.Tensor:
    term = _term(env, command_name)
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    both_contact = (force > force_threshold_n).all(dim=1).float()
    return both_contact * term.blend_weight((SkillId.CROUCH,))


def crouch_foot_contact_loss(
    env, command_name: str, sensor_cfg: SceneEntityCfg, force_threshold_n: float
) -> torch.Tensor:
    """Penalize true flight, not an ordinary single-support sample.

    Double support is still rewarded separately by ``crouch_foot_contact``.
    Treating its complement as contact loss incorrectly made every single-support
    step a safety violation.
    """
    term = _term(env, command_name)
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    both_airborne = ~(force > force_threshold_n).any(dim=1)
    return both_airborne.float() * term.blend_weight((SkillId.CROUCH,))


def crouch_foot_slip(
    env, command_name: str, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1) > 1.0
    speed = robot.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    slip = (speed * contacts).sum(dim=1) / contacts.sum(dim=1).clamp_min(1)
    return slip * term.blend_weight((SkillId.CROUCH,))


def crouch_action_rate(env, command_name: str) -> torch.Tensor:
    term = _term(env, command_name)
    rate = (env.action_manager.action - env.action_manager.prev_action).square().mean(dim=1)
    return rate * term.blend_weight((SkillId.CROUCH,))


def crouch_joint_saturation(
    env, command_name: str, asset_cfg: SceneEntityCfg, quantity: str, threshold: float, scale: float
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    if quantity == "velocity":
        value, limit = robot.data.joint_vel.torch.abs(), robot.data.joint_vel_limits.torch.abs()
    elif quantity == "torque":
        value, limit = robot.data.applied_torque.torch.abs(), robot.data.joint_effort_limits.torch.abs()
    else:
        raise ValueError(f"Unknown saturation quantity: {quantity}")
    ratio = value[:, asset_cfg.joint_ids] / limit[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    return (torch.relu(ratio - threshold) / scale).square().mean(dim=1) * term.blend_weight((SkillId.CROUCH,))


def crouch_joint_limit_proximity(
    env, command_name: str, asset_cfg: SceneEntityCfg, threshold: float, scale: float
) -> torch.Tensor:
    term = _term(env, command_name)
    robot = env.scene[asset_cfg.name]
    limits = robot.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids]
    position = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    center = 0.5 * (limits[..., 0] + limits[..., 1])
    half_range = 0.5 * (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6)
    proximity = (position - center).abs() / half_range
    return (torch.relu(proximity - threshold) / scale).square().mean(dim=1) * term.blend_weight((SkillId.CROUCH,))


def crouch_completion(env, command_name: str, completion: str) -> torch.Tensor:
    term = _term(env, command_name)
    value = {
        "hold": term.crouch_hold_complete,
        "return": term.crouch_return_complete,
        "stand_hold": term.crouch_stand_hold_complete,
    }[completion]
    return value.float() * term.blend_weight((SkillId.CROUCH,))
